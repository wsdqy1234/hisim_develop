#!/usr/bin/env python3
"""Phase 5E.2a ledger builder: per-request / per-batch / per-I/O timing
evidence with NON-OVERLAPPING TTFT accounting (P0 evidence + P4 stats).

Sources per gold case dir:
  - gold_<case>.log      : Req / HOST-LOAD / HBF-BACKUP / BATCH-TIME lines
  - io_trace.jsonl       : per-I/O arrival/duration/media/ic/queue
  - parse.json           : request-level metrics
  - win_pool_summary.json: per-WIN stats
  - manifest.json        : case id + config hash

Output: ledger.json + ledger_batches.csv + ledger_requests.csv in the case
dir. Semantics (locked by construction):
  batch load = max(per-request L2 loads, per-request HBF loads)  [media-parallel]
  same-medium concurrent loads serialize via the shared resource cursor
  phases serial: load -> infer -> backup (barrier mode)
  per-request TTFT = its batch's response time - its release time
"""
import json
import os
import re
import sys


def build(case_dir, case_id):
    log = None
    for f in os.listdir(case_dir):
        if f.startswith("gold_") and f.endswith(".log"):
            log = os.path.join(case_dir, f)
    txt = open(log, errors="ignore").read()

    # ---- per-batch ledger from BATCH-TIME ----
    batches = []
    for m in re.finditer(
            r"\[BATCH-TIME\] (PREFILL|DECODE) bs=(\d+) "
            r"load=([\d.]+)μs infer=([\d.]+)ms backup=([\d.]+)μs "
            r"l3_ovf=([\d.]+)μs l3_read=([\d.]+)μs clock=([\d.]+)ms", txt):
        g = m.groups()
        batches.append({
            "i": len(batches), "type": g[0], "bs": int(g[1]),
            "load_us": float(g[2]), "infer_ms": float(g[3]),
            "backup_us": float(g[4]), "l3_ovf_us": float(g[5]),
            "clock_ms": float(g[7]),
        })

    # ---- per-I/O resource timeline from io_trace ----
    ios = []
    iot = os.path.join(case_dir, "io_trace.jsonl")
    if os.path.exists(iot):
        for line in open(iot):
            try:
                ios.append(json.loads(line))
            except Exception:
                pass

    # ---- HOST-LOAD / BACKUP events (per-request, ordered) ----
    host_loads = re.findall(
        r"\[HOST-LOAD\] rid=(\w+) host_hit=(\d+) HBF=(\d+)tk DRAM=(\d+)tk "
        r"hbf_bytes=(\d+) dram_bytes=(\d+)", txt)
    hbf_backups = re.findall(r"\[HBF-BACKUP\] (\d+)tk→HBF \(([\d.]+)MB\) "
                             r"dur=([\d.]+)μs", txt)
    events = []
    for rid, hh, hb, dr, hby, dby in host_loads:
        events.append({"kind": "host_load", "rid": rid, "host_hit": int(hh),
                       "hbf_tk": int(hb), "dram_tk": int(dr),
                       "hbf_bytes": int(hby), "dram_bytes": int(dby)})
    for tk, mb, dur in hbf_backups:
        events.append({"kind": "hbf_backup", "tokens": int(tk),
                       "bytes_mb": float(mb), "dur_us": float(dur)})

    # ---- per-request from parse.json + Req lines ----
    reqs = []
    for m in re.finditer(
            r"Req #(\w*): input=(\d+)\(actual=(\d+)\) L1=(\d+) HBF=(\d+) "
            r"DRAM=(\d+) L3\[exists=(\d+) consumed=(\d+)\] Recomputed=(\d+) "
            r"SchedPF=(\d+) \(conservation: (\d+)\+(\d+)\+(\d+)=(\d+)/(\d+)\) "
            r"TTFT=([-\d.]+)ms", txt):
        g = m.groups()
        reqs.append({"rid": g[0], "input": int(g[1]), "actual": int(g[2]),
                     "l1": int(g[3]), "hbf": int(g[4]), "dram": int(g[5]),
                     "l3": int(g[6]), "rc": int(g[8]),
                     "cons_ok": int(g[10]) + int(g[11]) + int(g[12])
                     == int(g[13]),
                     "ttft_ms": float(g[15])})

    # ---- request ledger: join Req <-> host_load <-> io ops by rid ----
    io_by_rid = {}
    for io in ios:
        rid = io.get("rid")
        if rid:
            io_by_rid.setdefault(rid, []).append(io)
    for r in reqs:
        rid = r["rid"]
        r["io_ops"] = [
            {"op": io["op"], "pool": io["pool"], "tokens": io["tokens"],
             "bytes": io["bytes"], "dur_us": round(io["dur_s"] * 1e6, 1),
             "media_us": round(io["media_s"] * 1e6, 1),
             "ic_us": round(io["ic_s"] * 1e6, 1),
             "qw_us": round(io["qw_s"] * 1e6, 1),
             "arrival_s": io["t"]}
            for io in io_by_rid.get(rid, [])]
        # non-overlapping ledger: per-medium media time (parallel across
        # media -> max), own infer share, own backup ops
        med = [o["media_us"] for o in r["io_ops"] if o["op"] == "load"]
        ic = [o["ic_us"] for o in r["io_ops"] if o["op"] == "load"]
        r["load_media_us"] = max(med) if med else 0
        r["load_ic_us"] = max(ic) if ic else 0
        r["load_dur_us"] = max(
            (o["dur_us"] for o in r["io_ops"] if o["op"] == "load"),
            default=0)
        bmed = [o["dur_us"] for o in r["io_ops"] if o["op"] == "backup"]
        r["backup_dur_us"] = max(bmed) if bmed else 0

    # ---- TTFT time-ledger for reuse requests (host-hit ones) ----
    ledgers = []
    for r in reqs:
        if r["hbf"] + r["dram"] == 0 or r["input"] < 64:
            continue
        # critical-path: phases serial (load -> infer); cross-media parallel
        load = r["load_dur_us"]
        inf = r.get("infer_ms", 0) * 1e3
        # infer for reuse ≈ 1-2 token prefill; take from its batch if known
        ledgers.append({
            "rid": r["rid"][:12], "hbf_tk": r["hbf"], "dram_tk": r["dram"],
            "ttft_ms": r["ttft_ms"],
            "ledger": {
                "own_hbf_load_us": next(
                    (o["dur_us"] for o in r["io_ops"]
                     if o["op"] == "load" and o["pool"] == "hbf"), 0),
                "own_dram_load_us": next(
                    (o["dur_us"] for o in r["io_ops"]
                     if o["op"] == "load" and o["pool"] == "dram"), 0),
                "hbf_media_us": next(
                    (o["media_us"] for o in r["io_ops"]
                     if o["op"] == "load" and o["pool"] == "hbf"), 0),
                "hbf_ic_us": next(
                    (o["ic_us"] for o in r["io_ops"]
                     if o["op"] == "load" and o["pool"] == "hbf"), 0),
                "load_media_max_us": load,
                "note": "phases serial (load->infer); cross-media loads "
                        "parallel (max); same-medium concurrent loads "
                        "serialize via shared resource cursor",
            }})

    out = {
        "case_id": case_id,
        "n_batches": len(batches),
        "prefill_batches": [b for b in batches if b["type"] == "PREFILL"],
        "n_io_ops": len(ios),
        "io_ops": ios[:200],
        "events": events,
        "requests": reqs,
        "reuse_ledgers": ledgers,
        "conservation_all": all(r["cons_ok"] for r in reqs),
    }
    with open(os.path.join(case_dir, "ledger.json"), "w") as f:
        json.dump(out, f, indent=1)
    # CSVs
    import csv
    with open(os.path.join(case_dir, "ledger_batches.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(batches[0].keys()))
        w.writeheader()
        w.writerows(batches)
    with open(os.path.join(case_dir, "ledger_requests.csv"), "w",
              newline="") as f:
        cols = ["rid", "input", "actual", "l1", "hbf", "dram", "l3", "rc",
                "ttft_ms", "load_dur_us", "load_media_us", "load_ic_us",
                "backup_dur_us", "cons_ok"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(reqs)
    return out


if __name__ == "__main__":
    case_dir, case_id = sys.argv[1], sys.argv[2]
    o = build(case_dir, case_id)
    print(json.dumps({
        "case": case_id, "batches": o["n_batches"],
        "prefill": len(o["prefill_batches"]), "io_ops": o["n_io_ops"],
        "requests": len(o["requests"]),
        "reuse_ledgers": len(o["reuse_ledgers"]),
        "conservation": o["conservation_all"]}, indent=1))
