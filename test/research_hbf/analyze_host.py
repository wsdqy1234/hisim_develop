"""Offline Host-prefix demand analysis. Uses only the Python standard library.

Logical-prefix LRU curves are reference-stream counterfactuals, not a replay
of SGLang's admission/eviction policy. Actual Host transfers are reported
separately. No hardware efficiency claim is made by this program.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import csv
import json
import math
from pathlib import Path


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(p * len(values)) - 1)]


def distribution(values):
    return {"count": len(values), "min": min(values) if values else None,
            "p50": percentile(values, .5), "p95": percentile(values, .95),
            "max": max(values) if values else None}


def inflight(intervals):
    """Half-open [submit, complete); completion at t precedes arrival at t."""
    changes = defaultdict(int)
    for start, end in intervals:
        if end > start:
            changes[start] += 1
            changes[end] -= 1
    if not changes:
        return {"max": 0, "time_mean": 0.0, "time_p95": 0,
                "observation_span_s": 0.0}
    times = sorted(changes)
    current = peak = 0
    occupancy = defaultdict(float)
    for i, time in enumerate(times[:-1]):
        current += changes[time]
        peak = max(peak, current)
        occupancy[current] += times[i + 1] - time
    span = times[-1] - times[0]
    cumulative = 0.0
    p95 = 0
    for count, duration in sorted(occupancy.items()):
        cumulative += duration
        if cumulative >= span * .95:
            p95 = count
            break
    return {"max": peak, "time_mean": sum(k * v for k, v in occupancy.items()) / span,
            "time_p95": p95, "observation_span_s": span}


class Fenwick:
    def __init__(self, n):
        self.data = [0] * (n + 2)

    def add(self, i, value):
        while i < len(self.data):
            self.data[i] += value
            i += i & -i

    def sum(self, i):
        value = 0
        while i:
            value += self.data[i]
            i -= i & -i
        return value


def logical_profile(accesses, capacities=None):
    accesses = sorted(accesses, key=lambda row: (row["arrival_s"], row["request_id"]))
    references = []
    sizes, visits, first, last = {}, defaultdict(int), {}, {}
    intervals = []
    last_index = {}
    distance_bytes = []
    for row in accesses:
        for block in row["blocks"]:
            key, size = str(block["id"]), int(block["bytes"])
            if key in sizes and sizes[key] != size:
                raise ValueError("Content ID has inconsistent block byte size")
            sizes[key] = size
            if key in last:
                intervals.append(row["arrival_s"] - last[key])
            first.setdefault(key, row["arrival_s"])
            last[key] = row["arrival_s"]
            visits[key] += 1
            references.append((key, size))
    tree = Fenwick(len(references))
    for index, (key, size) in enumerate(references, 1):
        if key in last_index:
            previous = last_index[key]
            distance_bytes.append(tree.sum(index - 1) - tree.sum(previous))
            tree.add(previous, -size)
        tree.add(index, size)
        last_index[key] = index
    unique_bytes = sum(sizes.values())
    if capacities is None:
        capacities = sorted({0, *[int(unique_bytes * fraction)
                                  for fraction in (.05, .1, .25, .5, .75, 1)],
                             800_000_000_000})
    curves = []
    for capacity in sorted(set(capacities)):
        cache, seen = OrderedDict(), set()
        used = hit = total = eligible = eligible_hit = 0
        for key, size in references:
            repeated = key in seen
            seen.add(key)
            total += size
            eligible += size if repeated else 0
            if key in cache:
                hit += size
                eligible_hit += size if repeated else 0
                cache.move_to_end(key)
            elif size <= capacity:
                while used + size > capacity and cache:
                    _, removed = cache.popitem(last=False)
                    used -= removed
                cache[key] = size
                used += size
        curves.append({"capacity_bytes": capacity, "reference_bytes": total,
                       "hit_bytes": hit, "miss_bytes": total - hit,
                       "all_reference_byte_hit_ratio": hit / total if total else None,
                       "repeat_reference_bytes": eligible,
                       "repeat_reference_byte_hit_ratio": eligible_hit / eligible
                       if eligible else None})
    targets = {}
    for target in (.8, .9, .95):
        valid = [row["capacity_bytes"] for row in curves
                 if row["repeat_reference_byte_hit_ratio"] is not None and
                 row["repeat_reference_byte_hit_ratio"] >= target]
        targets[str(target)] = min(valid) if valid else None
    return {"scope": "logical reusable-prefix references; suffix/output excluded",
            "capacity_policy": "byte-capacity LRU, offline all-prefix references",
            "not_actual_host_policy": True,
            "unique_prefix_count": len({row["prefix_id"] for row in accesses}),
            "requests": len(accesses), "unique_blocks": len(sizes),
            "unique_reusable_prefix_bytes": unique_bytes,
            "reference_bytes": sum(size for _, size in references),
            "reference_count_per_block": distribution(list(visits.values())),
            "reuse_interval_s": distribution(intervals),
            "reuse_distance_distinct_bytes": distribution(distance_bytes),
            "needed_reference_lifetime_s": distribution([last[k] - first[k] for k in first]),
            "sampled_capacity_for_repeat_hit_target_bytes": targets,
            "capacity_curve": curves}


def windowed(entries, window_s, time_key):
    """Payload credited at submission or completion; sparse fixed buckets."""
    if not entries:
        return {"windows": 0, "active_windows": 0, "mean_gbps": 0.0,
                "active_p95_gbps": 0.0, "peak_gbps": 0.0}, []
    t0 = min(row["arrival_s"] for row in entries)
    t1 = max(row["completion_s"] for row in entries)
    origin = math.floor(t0 / window_s) * window_s
    counts, sums = defaultdict(int), defaultdict(int)
    for row in entries:
        index = max(0, math.floor((row[time_key] - origin) / window_s + 1e-9))
        counts[index] += 1
        sums[index] += (row.get("requested_bytes", row["bytes"])
                        if time_key == "arrival_s" else row["bytes"])
    n = max(max(sums) + 1, math.floor((t1 - origin) / window_s) + 1)
    bws = [value / window_s / 1e9 for value in sums.values()]
    result = {"windows": n, "active_windows": len(sums),
              "mean_gbps": sum(sums.values()) / (n * window_s) / 1e9,
              "active_p95_gbps": percentile(bws, .95),
              "peak_gbps": max(bws), "origin_s": origin}
    rows = [{"window_start_s": origin + i * window_s, "bytes": sums[i],
             "ios": counts[i], "gbps": sums[i] / window_s / 1e9}
            for i in sorted(sums)]
    return result, rows


def io_profile(raw):
    entries, warnings = [], set()
    for event in raw:
        if event.get("path", "real") != "real":
            continue
        start = event.get("arrival_s", event.get("t", 0.0))
        end = event.get("completion_s")
        if end is None:
            warnings.add("legacy rounded submit/completion timing")
            end = start + event.get("dur_s", 0.0)
        if end < start or event.get("bytes", 0) < 0:
            raise ValueError("Invalid I/O interval or byte count")
        entries.append({**event, "arrival_s": start, "completion_s": end})
    report, window_rows = {}, []
    for pool in ("all", "hbf", "dram"):
        for op in ("all", "load", "backup"):
            selected = [e for e in entries if (pool == "all" or e["pool"] == pool)
                        and (op == "all" or e["op"] == op)]
            if not selected:
                continue
            group = f"{pool}:{op}"
            total = sum(e["bytes"] for e in selected)
            span = max(e["completion_s"] for e in selected) - min(e["arrival_s"] for e in selected)
            data = {"count": len(selected), "bytes": total, "span_s": span,
                    "span_payload_gbps": total / span / 1e9 if span else None,
                    "io_size_bytes": distribution([e["bytes"] for e in selected]),
                    "duration_s": distribution([e["completion_s"] - e["arrival_s"] for e in selected]),
                    "logical_inflight": inflight([(e["arrival_s"], e["completion_s"]) for e in selected]),
                    "windows": {}}
            for window in (.001, .01, .1):
                for label, key in (("arrival", "arrival_s"), ("completion", "completion_s")):
                    metrics, rows = windowed(selected, window, key)
                    data["windows"][f"{label}_{window:g}s"] = metrics
                    window_rows.extend({"group": group, "basis": label,
                                        "window_s": window, **row} for row in rows)
            report[group] = data
    return {"scope": "observed model Host I/O; includes policy and service feedback",
            "warnings": sorted(warnings), "groups": report}, window_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(case_dir, output_dir=None, capacities=None, allow_incomplete=False):
    case_dir = Path(case_dir)
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("status") not in (None, "complete") and not allow_incomplete:
        raise ValueError("Case did not pass runner validation; use --allow-incomplete only for debugging")
    output = Path(output_dir) if output_dir else case_dir / "demand_analysis"
    output.mkdir(parents=True, exist_ok=True)
    accesses = read_jsonl(case_dir / "logical_prefix_accesses.jsonl")
    raw = read_jsonl(case_dir / "io_trace.jsonl")
    result = {"schema_version": 1, "case_dir": str(case_dir),
              "evidence": "simulation / Layer-A assumptions; not hardware measurement",
              "logical_demand": logical_profile(accesses, capacities) if accesses else None}
    io, windows = io_profile(raw)
    result["observed_host_io"] = io
    requests = read_jsonl(case_dir / "request.jsonl")
    negative_intervals = sum(value < -1e-9 for row in requests
                             for value in row.get("gen_token_latencies", []))
    if negative_intervals and not allow_incomplete:
        raise ValueError("Negative virtual token interval; case is not valid research evidence")
    result["requests"] = {"count": len(requests),
        "hbf_load_bytes": sum(row.get("hbf_load_bytes", 0) for row in requests),
        "dram_load_bytes": sum(row.get("dram_load_bytes", 0) for row in requests),
        "hbf_hit_tokens": sum(row.get("hbf_hit_tokens", 0) for row in requests),
        "dram_hit_tokens": sum(row.get("dram_hit_tokens", 0) for row in requests)}
    result["load_byte_conservation"] = {}
    for pool in ("hbf", "dram"):
        trace_bytes = sum(row.get("bytes", 0) for row in raw
                          if row.get("pool") == pool and row.get("op") == "load"
                          and row.get("path") == "real")
        request_bytes = result["requests"][f"{pool}_load_bytes"]
        result["load_byte_conservation"][pool] = {
            "trace_bytes": trace_bytes, "request_bytes": request_bytes,
            "equal": trace_bytes == request_bytes}
    # Service-deadline demand is a conditional requirement, not measured BW.
    restores = [row for row in raw if row.get("op") == "load" and row.get("path") == "real"]
    required = [{"restore_bytes": row["bytes"], "pool": row["pool"],
                 "deadline_ms": deadline, "required_payload_gbps": row["bytes"] / (deadline / 1000) / 1e9,
                 "note": "per-I/O isolated requirement; simultaneous I/Os require aggregate capacity"}
                for row in restores for deadline in (1, 10, 20, 100)]
    result["data_quality"] = {"logical_reference_stream_present": bool(accesses),
        "host_trace_present": bool(raw), "request_stats_present": bool(requests),
        "new_exact_timing": bool(raw) and all("completion_s" in row for row in raw),
        "capacity_is_counterfactual_lru": bool(accesses),
        "runner_status": manifest.get("status", "legacy_not_available"),
        "negative_virtual_token_intervals": negative_intervals,
        "debug_incomplete_override": allow_incomplete,
        "all_load_bytes_conserved": all(row["equal"] for row in result["load_byte_conservation"].values())}
    with open(output / "demand_summary.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    if accesses:
        write_csv(output / "capacity_curve.csv", result["logical_demand"]["capacity_curve"])
    write_csv(output / "host_io_windows.csv", windows)
    write_csv(output / "restore_deadline_requirements.csv", required)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--capacities-gb", help="Comma-separated decimal GB capacities")
    parser.add_argument("--allow-incomplete", action="store_true", help="Debug only; never final research evidence")
    args = parser.parse_args()
    capacities = [int(float(x) * 1e9) for x in args.capacities_gb.split(",")] if args.capacities_gb else None
    result = analyze(args.case_dir, args.output_dir, capacities, args.allow_incomplete)
    print(json.dumps({"output": str(Path(args.output_dir) if args.output_dir else Path(args.case_dir) / "demand_analysis"),
                      "data_quality": result["data_quality"]}))
