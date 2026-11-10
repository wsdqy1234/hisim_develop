import importlib.util
import json
from pathlib import Path
import pytest


spec = importlib.util.spec_from_file_location("analyze_host", Path(__file__).with_name("analyze_host.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def access(index, time, blocks):
    return {"request_id": str(index), "arrival_s": time, "prefix_id": str(blocks),
            "blocks": [{"id": key, "bytes": size} for key, size in blocks]}


def test_lru_reuse_distance_and_compulsory_misses():
    # A(2), B(3), A(2): A needs five bytes capacity to survive B.
    result = m.logical_profile([access(0, 0, [("A", 2)]), access(1, 1, [("B", 3)]),
                                access(2, 2, [("A", 2)])], [0, 4, 5])
    assert result["unique_reusable_prefix_bytes"] == 5
    assert result["reuse_distance_distinct_bytes"]["p50"] == 3
    curve = result["capacity_curve"]
    assert curve[1]["hit_bytes"] == 0
    assert curve[2]["hit_bytes"] == 2
    assert curve[2]["repeat_reference_byte_hit_ratio"] == 1
    assert curve[2]["all_reference_byte_hit_ratio"] == 2 / 7


def test_interval_qd_not_same_arrival_burst():
    result = m.inflight([(0, 2), (1, 3), (3, 4), (4, 4)])
    assert result["max"] == 2
    assert result["time_mean"] == 1.25
    assert m.inflight([(0, 1), (1, 2)])["max"] == 1


def test_arrival_and_completion_windows_differ_and_conserve_bytes():
    rows = [{"arrival_s": 0, "completion_s": .002, "bytes": 100},
            {"arrival_s": 0, "completion_s": .003, "bytes": 100}]
    arrival, a = m.windowed(rows, .001, "arrival_s")
    completion, c = m.windowed(rows, .001, "completion_s")
    assert sum(row["bytes"] for row in a) == sum(row["bytes"] for row in c) == 200
    assert arrival["peak_gbps"] == 2 * completion["peak_gbps"]


def test_failed_case_cannot_become_research_evidence(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="runner validation"):
        m.analyze(tmp_path)


def test_partial_miss_offered_and_completed_bytes_stay_separate():
    rows = [{"arrival_s": 0, "completion_s": .001, "bytes": 50, "requested_bytes": 100}]
    _, arrival = m.windowed(rows, .001, "arrival_s")
    _, completion = m.windowed(rows, .001, "completion_s")
    assert sum(row["bytes"] for row in arrival) == 100
    assert sum(row["bytes"] for row in completion) == 50
