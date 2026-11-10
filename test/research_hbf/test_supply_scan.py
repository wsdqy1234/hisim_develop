import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("supply_scan", Path(__file__).with_name("supply_scan.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_mixed_workload_really_contends_on_every_win():
    rows = m.saturation(4 * 1024 * 1024, 16, "mixed", n_ios=32)
    assert len(rows) == 2
    for row in rows:
        assert row["direction_wins_touched"] == 8
        assert all(n > 0 for n in json.loads(row["direction_per_win_payload_bytes"]))
    assert rows[0]["completed_payload_bytes"] == rows[1]["completed_payload_bytes"]


def test_fixed_denominator_and_strict_boundary():
    boundary = m.saturation(4 * 1024 * 1024, 16, "sequential", 1280e9, n_ios=32)[0]
    assert abs(boundary["fixed_pool_efficiency"] - .8) < 1e-12
    assert not boundary["strictly_above_80pct"]
    hotspot = m.saturation(4 * 1024 * 1024, 16, "sequential", placement="hotspot", n_ios=32)[0]
    assert hotspot["fixed_rated_gbps"] == 1600
    assert hotspot["fixed_pool_efficiency"] <= .125 + 1e-12
