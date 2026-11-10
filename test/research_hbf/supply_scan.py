"""Layer-A closed-loop envelope scan, with fixed 8-WIN rated denominators.

This is a conditional model experiment, not a hardware bandwidth prediction.
All submitted payloads are drained; span includes ramp-up and final drain.
QD means outstanding logical requests, not WIN internal NAND parallelism.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import heapq
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hisim.simulation.hbf_win_pool import Hbf8WinConfig, HbfWinPoolModel, WinSpec


RATED = {"sequential": 1600e9, "random": 800e9, "write": 80e9}


def saturation(size, qd, mode, interface=None, placement="balanced", n_ios=512):
    cfg = Hbf8WinConfig(placement=placement, package_shared_interface_bw=interface,
                       win=WinSpec())
    pool = HbfWinPoolModel(cfg)
    pool.begin_batch(0.0)
    outstanding, stats = [], []
    for i in range(n_ios):
        arrival = heapq.heappop(outstanding)[0] if len(outstanding) >= qd else 0.0
        if mode == "mixed":
            # Eight-WIN RR must not alias alternating R/W into disjoint WINs.
            is_read, access = (i // 8) % 2 == 0, "sequential"
        else:
            is_read, access = mode != "write", mode if mode != "write" else None
        event = pool.submit(arrival, size, is_read=is_read,
                            access_mode=access, object_id=f"synthetic-{i}")
        heapq.heappush(outstanding, (arrival + event.total, i))
        stats.append(event)
    span = max(end for end, _ in outstanding)
    per_win = [w.bytes_read_seq + w.bytes_read_rand + w.bytes_written for w in pool.wins]
    rows = []
    for is_read, denominator in ((True, RATED["random" if mode == "random" else "sequential"]),
                                 (False, RATED["write"])):
        selected = [s for s in stats if s.is_read == is_read]
        if not selected:
            continue
        payload = sum(s.bytes for s in selected)
        bw = payload / span
        direction_per_win = [sum(s.bytes for s in selected if s.win == win) for win in range(8)]
        rows.append({"experiment": "closed_loop_synthetic", "layer": "A-envelope",
                     "hardware_calibrated": False, "mode": mode,
                     "direction": "read" if is_read else "write", "size_bytes": size,
                     "logical_qd": qd, "placement": placement,
                     "shared_interface_gbps": interface / 1e9 if interface else 0,
                     "n_ios_total": n_ios, "direction_ios": len(selected),
                     "completed_payload_bytes": payload, "span_s": span,
                     "payload_gbps": bw / 1e9, "fixed_rated_gbps": denominator / 1e9,
                     "fixed_pool_efficiency": bw / denominator,
                     "strictly_above_80pct": bw / denominator > .8 + 1e-12,
                     "direction_wins_touched": sum(n > 0 for n in direction_per_win),
                     "all_wins_touched": sum(n > 0 for n in per_win),
                     "direction_per_win_payload_bytes": json.dumps(direction_per_win),
                     "mixed_schedule": "8 reads then 8 writes repeatedly; every WIN handles both" if mode == "mixed" else "",
                     "load_balance_min_max": min(per_win) / max(per_win) if max(per_win) else 0,
                     "media_service_util_mean": sum(w.media_service_sum for w in pool.wins) / (8 * span),
                     "max_media_queue_wait_s": max(s.queue_wait for s in selected)})
    return rows


def scan(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    sizes = [4096, 65536, 262144, 1048576, 4194304, 16777216, 339 * 114688]
    qds = [1, 2, 4, 8, 16, 32]
    for mode in ("sequential", "random", "write"):
        for size in sizes:
            for qd in qds:
                rows.extend(saturation(size, qd, mode))
    # Interface and placement controls use large payloads / enough QD.
    for mode in ("sequential", "random", "write", "mixed"):
        for interface in (None, 112e9, 896e9, 1280e9, 1600e9):
            for placement in ("balanced", "hotspot"):
                rows.extend(saturation(4 * 1024 * 1024, 16, mode, interface, placement))
    with open(output / "supply_scan.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"schema_version": 1, "evidence": "Layer-A conditional service envelope",
        "hardware_calibrated": False, "denominators_gbps": {k: v / 1e9 for k, v in RATED.items()},
        "sizes_bytes": sizes, "logical_qds": qds, "config_defaults": asdict(Hbf8WinConfig()),
        "window": "first submit to final completion; all submitted payload drained",
        "mixed_schedule": "8 reads then 8 writes repeatedly to avoid parity aliasing with 8-WIN RR",
        "limits": ["No model of WIN-internal QD scaling", "Interface topology/bandwidth are hypotheses",
                   "Mixed reads/writes have separate fixed denominators; no simultaneous 80% requirement",
                   "Capacity is not a stress dimension in this timing microbenchmark"],
        "csv_sha256": hashlib.sha256((output / "supply_scan.csv").read_bytes()).hexdigest()}
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"rows": len(rows), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    scan(args.output)
