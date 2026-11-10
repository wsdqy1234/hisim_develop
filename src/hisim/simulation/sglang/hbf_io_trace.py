"""Versioned logical KV I/O trace.

Arrival/completion windows are whole-I/O event traffic, not link samples.
Interval occupancy includes queue/dependency waits, not hardware admission QD.
Only path=real carries model-sized KV traffic. Missing breakdowns are null.
"""
from __future__ import annotations

import copy
import json
import math
import os
import threading
from collections import defaultdict
from typing import Optional

TRACE_SCHEMA_VERSION = 2
_TRACE_LIST: list = []
_TRACE_LOCK = threading.Lock()
_TRACE_ENABLED: Optional[bool] = None
_CTX_RID: Optional[str] = None


def trace_enabled() -> bool:
    global _TRACE_ENABLED
    if _TRACE_ENABLED is None:
        _TRACE_ENABLED = os.environ.get("HISIM_HBF_IO_TRACE", "0") in ("1", "true", "True")
    return _TRACE_ENABLED


def set_context(rid: Optional[str] = None) -> None:
    global _CTX_RID
    _CTX_RID = rid


def clear_context() -> None:
    global _CTX_RID
    _CTX_RID = None


def normalize_details(details: Optional[dict]) -> Optional[dict]:
    """Copy model details; expose exact second-based times for older models."""
    if details is None:
        return None
    result = copy.deepcopy(details)
    for entry in [result, *result.get("sub_ios", [])]:
        for name in ("arrival", "effective_arrival", "completion", "media_start",
                     "media_completion", "ic_start", "ic_completion"):
            if name in entry and name + "_s" not in entry:
                entry[name + "_s"] = entry[name]
    return result


def record(op: str, pool: str, tokens: int, bytes_: float,
           arrival: float, duration: float, path: str = "real",
           media_time: Optional[float] = None, ic_time: Optional[float] = None,
           queue_wait: Optional[float] = None, node_ids: Optional[list] = None,
           details: Optional[dict] = None) -> None:
    if not trace_enabled():
        return
    global _CTX_RID
    details = normalize_details(details)
    completion = (details or {}).get("completion_s")
    if completion is None:
        completion = float(arrival) + float(duration)
    arrival = float(arrival)
    entry = {
        "schema_version": TRACE_SCHEMA_VERSION, "time_unit": "s",
        "t": arrival, "arrival_s": arrival, "completion_s": float(completion),
        "op": op, "pool": pool, "path": path, "tokens": int(tokens),
        "bytes": int((details or {}).get("bytes", bytes_)),
        "requested_bytes": int((details or {}).get("requested_bytes", bytes_)),
        "status": (details or {}).get(
            "status", "diagnostic_mock" if path == "mock_lbk" else "complete"),
        "authoritative": path == "real",
        "dur_s": float(completion) - arrival,
        "media_s": media_time, "ic_s": ic_time, "qw_s": queue_wait,
        "rid": _CTX_RID,
        "nodes": len(node_ids) if node_ids is not None else 0,
        "node_ids": sorted(int(n) for n in node_ids) if node_ids else [],
        "details": details,
        "availability": {
            "media_timing": media_time is not None,
            "interface_timing": ic_time is not None,
            "queue_wait": queue_wait is not None,
            "object_details": details is not None,
            "node_ids": node_ids is not None,
        },
    }
    with _TRACE_LOCK:
        entry["trace_io_id"] = len(_TRACE_LIST) + 1
        _TRACE_LIST.append(entry)
    _CTX_RID = None


def get_trace() -> list:
    return _TRACE_LIST


def reset_trace() -> None:
    global _TRACE_LIST, _CTX_RID
    with _TRACE_LOCK:
        _TRACE_LIST = []
    _CTX_RID = None


def _arrival(entry):
    return entry.get("arrival_s", entry.get("t", 0.0))


def _completion(entry):
    return entry.get("completion_s", _arrival(entry) + entry.get("dur_s", 0.0))


def _windowed_bw(entries: list, window: float, event: str = "arrival"):
    """Bucket whole-I/O bytes at arrival OR completion, never spread service."""
    if not entries:
        return 0, []
    t0 = min(_arrival(e) for e in entries)
    t_end = max(_completion(e) for e in entries)
    n_win = max(1, int((t_end - t0) / window) + 1)
    sums = [0] * n_win
    for e in entries:
        timestamp = _completion(e) if event == "completion" else _arrival(e)
        k = int((timestamp - t0) / window)
        if 0 <= k < n_win:
            sums[k] += e.get("requested_bytes", e["bytes"]) if event == "arrival" else e["bytes"]
    return n_win, sums


def _pct(values: list, p: float):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100.0 * len(values)))]


def _traffic_windows(entries, event):
    result = {}
    for win_ms in (1, 10):
        window = win_ms / 1000.0
        count, sums = _windowed_bw(entries, window, event)
        rates = [b / window / 1e9 for b in sums]
        active = [b for b in rates if b > 0]
        result[f"{win_ms}ms"] = {
            "n_windows": count, "active_windows": len(active),
            "all_window_mean_gbps": sum(rates) / len(rates) if rates else 0.0,
            "active_window_mean_gbps": sum(active) / len(active) if active else 0.0,
            "active_window_p95_gbps": _pct(active, 95) if active else 0.0,
            "peak_gbps": max(rates) if rates else 0.0, "total_bytes": sum(sums),
        }
    return result


def _interval_occupancy(entries):
    """Time-weighted outstanding count on half-open [arrival,completion)."""
    changes = defaultdict(int)
    count = 0
    for e in entries:
        start, end = _arrival(e), _completion(e)
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        changes[start] += 1
        changes[end] -= 1
        count += 1
    if not changes:
        return {"interval_count": 0, "peak": 0, "time_weighted_mean": 0.0,
                "span_s": 0.0, "io_time_s": 0.0}
    times = sorted(changes)
    active = peak = 0
    area = 0.0
    for i, time in enumerate(times):
        active += changes[time]
        if i + 1 < len(times):
            delta = times[i + 1] - time
            area += active * delta
            if delta > 0:
                peak = max(peak, active)
    span = times[-1] - times[0]
    return {"interval_count": count, "peak": peak,
            "time_weighted_mean": area / span if span > 0 else 0.0,
            "span_s": span, "io_time_s": area}


def _same_arrival_burst(entries):
    by_t = defaultdict(int)
    for e in entries:
        by_t[_arrival(e)] += 1
    return max(by_t.values(), default=0)


def _group_summary(entries):
    total_bytes = sum(e["bytes"] for e in entries)
    span = (max(_completion(e) for e in entries) - min(_arrival(e) for e in entries)) if entries else 0.0
    sizes = [e["bytes"] for e in entries]
    return {
        "count": len(entries), "bytes": total_bytes,
        "requested_bytes": sum(e.get("requested_bytes", e["bytes"]) for e in entries),
        "span_s": span,
        "span_useful_bw_gbps": total_bytes / span / 1e9 if span > 0 else None,
        "io_size_bytes": {"min": min(sizes, default=None), "p50": _pct(sizes, 50),
                          "p95": _pct(sizes, 95), "max": max(sizes, default=None)},
        "arrival_windows": _traffic_windows(entries, "arrival"),
        "completion_windows": _traffic_windows(entries, "completion"),
        "logical_inflight": _interval_occupancy(entries),
        "same_arrival_burst_max": _same_arrival_burst(entries),
    }


def summarize() -> dict:
    with _TRACE_LOCK:
        entries = list(_TRACE_LIST)
    real = [e for e in entries if e["path"] == "real"]
    summary = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "count_total": len(entries), "count_real": len(real),
        "count_mock_lbk": len(entries) - len(real),
        "definitions": {
            "bandwidth_unit": "GB/s = 1e9 bytes/s",
            "windows": "Whole requested bytes at arrival, served bytes at completion: event traffic, not interpolated link utilization.",
            "logical_inflight": "Outstanding logical I/Os on [arrival_s,completion_s), including queue/dependency waits; not hardware admission QD.",
            "working_set_approx": "Radix node IDs change after eviction/recompute. Backup traffic/node counts are not required capacity.",
            "legacy_window_fields": "restore/backup_w* aliases are arrival active-window rates.",
            "legacy_burst": "load_burst_max_concurrent aliases exact same-arrival burst, not interval concurrency.",
            "media_bandwidth": "No aggregate media bandwidth is inferred from overlapping service-time sums.",
        },
    }
    backups = [e for e in real if e["op"] == "backup"]
    loads = [e for e in real if e["op"] == "load"]
    summary["working_set_approx"] = {
        "distinct_recorded_node_ids": len({n for e in backups for n in e.get("node_ids", [])}),
        "backup_records_with_node_ids": sum(bool(e.get("node_ids")) for e in backups),
        "required_capacity_bytes": None,
        "availability": "identity_proxy_only; join workload/request lifetimes for capacity demand",
    }
    summary["distinct_first_nodes"] = len({e["node_ids"][0] for e in backups if e.get("node_ids")})
    for label, selected in (("backup", backups), ("load", loads)):
        summary[f"total_{label}_tokens"] = sum(e["tokens"] for e in selected)
        summary[f"total_{label}_bytes"] = sum(e["bytes"] for e in selected)
    summary["logical_inflight"] = _interval_occupancy(real)
    summary["by_pool_op"] = {}
    for pool in sorted({e["pool"] for e in real}):
        pe = [e for e in real if e["pool"] == pool]
        group = _group_summary(pe)
        summary[f"{pool}_span_s"] = group["span_s"]
        summary[f"{pool}_useful_bw_gbps"] = group["span_useful_bw_gbps"]
        summary[f"{pool}_measured_media_bw_gbps"] = None
        summary[f"{pool}_measured_ic_bw_gbps"] = None
        summary["by_pool_op"][pool] = {
            op: _group_summary([e for e in pe if e["op"] == op])
            for op in sorted({e["op"] for e in pe})}
    summary["by_op"] = {}
    for label, selected in (("restore", loads), ("backup", backups)):
        group = _group_summary(selected)
        summary["by_op"][label] = group
        for ms in (1, 10):
            stats = group["arrival_windows"][f"{ms}ms"]
            prefix = f"{label}_w{ms}ms_"
            summary[prefix + "active_windows"] = stats["active_windows"]
            summary[prefix + "avg_gbps"] = stats["active_window_mean_gbps"]
            summary[prefix + "p95_gbps"] = stats["active_window_p95_gbps"]
            summary[prefix + "peak_gbps"] = stats["peak_gbps"]
    summary["same_arrival_load_burst_max"] = _same_arrival_burst(loads)
    summary["load_burst_max_concurrent"] = summary["same_arrival_load_burst_max"]
    waits = [e["qw_s"] for e in loads if e.get("qw_s") is not None]
    summary["load_queue_wait_max_ms"] = max(waits) * 1e3 if waits else None
    sizes = [e["bytes"] for e in real]
    if sizes:
        summary.update(io_size_min_mb=min(sizes)/1e6, io_size_p50_mb=_pct(sizes, 50)/1e6,
                       io_size_max_mb=max(sizes)/1e6)
    sub_ios = []
    for e in real:
        for sub in (e.get("details") or {}).get("sub_ios", []):
            sub = dict(sub)
            sub["op"] = e["op"]
            sub_ios.append(sub)
    summary["sub_win_io_availability"] = {
        "logical_records_with_details": sum(e.get("details") is not None for e in real),
        "sub_io_count": len(sub_ios),
    }
    summary["by_win_op"] = {
        str(win): {op: _group_summary([s for s in sub_ios if s["win"] == win and s["op"] == op])
                   for op in sorted({s["op"] for s in sub_ios if s["win"] == win})}
        for win in sorted({s["win"] for s in sub_ios})}
    return summary


def dump_trace(output_dir: str, extra: Optional[dict] = None) -> dict:
    if not trace_enabled():
        return
    os.makedirs(output_dir, exist_ok=True)
    summary = summarize()
    if extra:
        summary.update(extra)
    with open(os.path.join(output_dir, "io_trace.jsonl"), "w") as f:
        with _TRACE_LOCK:
            for e in _TRACE_LIST:
                f.write(json.dumps(e) + "\n")
    with open(os.path.join(output_dir, "io_trace_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[HBF-TRACE] dumped {len(_TRACE_LIST)} I/Os (schema {TRACE_SCHEMA_VERSION})", flush=True)
    return summary
