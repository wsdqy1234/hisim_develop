"""Fixed-arrival HBF logical-transfer replay for interface sensitivity.

Uses the exact recorded slots and logical arrivals, re-evaluating backup
dependencies. It does not replay cache eviction/free events or SGLang
scheduling; capacity and end-to-end policy conclusions are out of scope.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hisim.simulation.hbf_win_pool import Hbf8WinConfig, HbfWinPoolModel


def replay(case_dir, output):
    case_dir, output = Path(case_dir), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw = [json.loads(line) for line in (case_dir / "io_trace.jsonl").read_text().splitlines() if line.strip()]
    entries = [e for e in raw if e.get("pool") == "hbf" and e.get("path") == "real"]
    entries.sort(key=lambda e: e.get("arrival_s", e.get("t", 0)))
    if not entries:
        raise ValueError("No real HBF transfers to replay")
    if any(not e.get("details", {}).get("slots") for e in entries):
        raise ValueError("Replay requires complete new-schema HBF slots, not legacy rounded traces")
    rows, event_outputs = [], []
    for interface in (0, 112, 896, 1280, 1600):
        cfg = Hbf8WinConfig.from_json(str(case_dir / "win_config.json"))
        cfg.per_win_interface_bw = None
        cfg.package_shared_interface_bw = interface * 1e9 if interface else None
        pool = HbfWinPoolModel(cfg)
        records = []
        for entry in entries:
            detail = entry["details"]
            slots = detail["slots"]
            requested = detail["requested_bytes"]
            if requested % len(slots):
                raise ValueError("Non-integral bytes/token")
            refs = detail.get("object_refs", [])
            key = refs[0]["key"] if entry["op"] == "backup" and len(refs) == 1 else None
            st = pool.submit_object(entry["arrival_s"], slots, requested // len(slots),
                                    is_read=entry["op"] == "load", object_key=key,
                                    metadata=detail.get("metadata"))
            records.append((entry, st))
        start = min(e["arrival_s"] for e in entries)
        end = max(e["arrival_s"] + st.total for e, st in records)
        loads = [st for e, st in records if e["op"] == "load"]
        durations = sorted(st.total for st in loads)
        actual = [e.get("completion_s", e["arrival_s"] + e["dur_s"]) - e["arrival_s"] for e in entries]
        errors = [abs(st.total - baseline) for (_, st), baseline in zip(records, actual)]
        rows.append({"shared_interface_gbps": interface,
                     "unbounded_interface_hypothesis": interface == 0,
                     "logical_ios": len(records), "restore_ios": len(loads),
                     "requested_read_bytes": sum(e["requested_bytes"] for e in entries if e["op"] == "load"),
                     "completed_read_bytes": sum(st.trace_details["bytes"] for st in loads),
                     "completed_write_bytes": sum(st.trace_details["bytes"] for e, st in records if e["op"] == "backup"),
                     "restore_duration_max_ms": max(durations, default=0) * 1000,
                     "restore_duration_mean_ms": sum(durations) / len(durations) * 1000 if durations else 0,
                     "observation_span_s": end - start,
                     "workload_read_average_gbps": sum(st.trace_details["bytes"] for st in loads) / (end - start) / 1e9,
                     "max_absolute_duration_delta_to_source_s": max(errors, default=0),
                     "restore_miss_ios": sum(st.trace_details["status"] != "complete" for st in loads),
                     "evidence": "fixed-arrival transfer model, no cache lifecycle feedback"})
        event_outputs.extend({"shared_interface_gbps": interface, **event} for event in pool.get_io_events())
    with open(output / "interface_replay.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output / "replay_events.jsonl", "w") as handle:
        for event in event_outputs:
            handle.write(json.dumps(event) + "\n")
    (output / "manifest.json").write_text(json.dumps({"schema_version": 1,
        "source_case": str(case_dir), "hardware_calibrated": False,
        "fixed_inputs": "logical arrivals, slots, bytes/token, backup object keys, operation order",
        "varied": "shared interface bandwidth hypothesis only",
        "limits": ["No free/eviction events; no capacity-policy conclusion",
                   "No SGLang clock feedback; restore duration is not request TTFT",
                   "Whole-observation average is not saturated pool efficiency"]}, indent=2))
    print(json.dumps({"output": str(output), "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replay(args.case_dir, args.output)
