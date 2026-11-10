"""Fixed logical Host-transfer replay with varied stripe granularity.

Logical arrivals, slots and operation order are fixed. This isolates mapping
and model timing; it does not emulate a new SGLang eviction policy or free
events. Restore efficiency is per-I/O, not sustained pool efficiency.
"""
import argparse
import csv
import json
from pathlib import Path
import sys


def run(case, output, repo):
    sys.path.insert(0, str(Path(repo) / "src"))
    from hisim.simulation.hbf_win_pool import Hbf8WinConfig, HbfWinPoolModel
    case, output = Path(case), Path(output)
    manifest = json.loads((case / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Use a validated source case")
    entries = [json.loads(line) for line in (case / "io_trace.jsonl").read_text().splitlines() if line]
    entries = [e for e in entries if e["path"] == "real" and e["pool"] == "hbf"]
    rows = []
    for stripe in (32, 43, 64, 128, 256, 512, 2048):
        cfg = Hbf8WinConfig.from_json(str(case / "win_config.json"))
        cfg.stripe_block_tokens = stripe
        pool = HbfWinPoolModel(cfg)
        for e in sorted(entries, key=lambda row: row["arrival_s"]):
            d = e["details"]
            st = pool.submit_object(e["arrival_s"], d["slots"],
                                    d["requested_bytes"] // len(d["slots"]),
                                    is_read=e["op"] == "load")
            if e["op"] != "load":
                continue
            served = st.trace_details["bytes"]
            win_bytes = [sum(c["bytes"] for c in st.trace_details["sub_ios"] if c["win"] == w)
                         for w in range(8)]
            effective = served / st.total if st.total else 0
            rows.append({"source_case": case.name, "source_logical_io_id": d["logical_io_id"],
                         "stripe_block_tokens": stripe, "restore_bytes": served,
                         "restore_duration_ms": st.total * 1000,
                         "per_io_effective_read_gbps": effective / 1e9,
                         "per_io_fixed_seq_efficiency": effective / 1600e9,
                         "strictly_above_80pct": effective / 1600e9 > .8 + 1e-12,
                         "wins_touched": sum(b > 0 for b in win_bytes),
                         "per_win_bytes": json.dumps(win_bytes),
                         "backup_dependency_wait_s": st.trace_details["dependency_wait_s"],
                         "status": st.trace_details["status"]})
    output.mkdir(parents=True, exist_ok=False)
    with (output / "placement_replay.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({"source_case": str(case),
        "evidence": "Layer-A fixed-arrival conditional mapping experiment",
        "hardware_calibrated": False, "varied": "stripe_block_tokens only",
        "limits": ["No free-event or scheduler feedback", "Per-I/O efficiency is not sustained saturation efficiency",
                   "Source service latency, sequential mode and interface assumptions preserved"]}, indent=2))
    print(json.dumps({"rows": len(rows), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    run(args.case, args.output, args.repo)
