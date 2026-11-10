"""Trace contract checks: exact time, offered/delivered bytes, and interval QD.

Loads the dependency-free recorder directly; no SGLang/GPU/server needed.
"""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hbf_io_trace_v2_test", ROOT / "src/hisim/simulation/sglang/hbf_io_trace.py")
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)


@pytest.fixture(autouse=True)
def fresh_trace(monkeypatch):
    trace.reset_trace()
    monkeypatch.setattr(trace, "_TRACE_ENABLED", True)
    yield
    trace.reset_trace()


def emit(arrival=0.0, duration=0.002, bytes_=100, **kwargs):
    trace.record("load", "hbf", 1, bytes_, arrival, duration, **kwargs)


def test_exact_times_full_node_ids_and_unknown_breakdown(tmp_path):
    start = 0.123456789123
    duration = 7.8125e-8
    trace.set_context("request-A")
    emit(start, duration, node_ids=list(range(21)))
    trace.dump_trace(str(tmp_path))
    event = json.loads((tmp_path / "io_trace.jsonl").read_text().strip())
    assert event["schema_version"] == 2
    assert event["arrival_s"] == start == event["t"]
    assert event["completion_s"] == start + duration
    assert event["dur_s"] > 0
    assert event["node_ids"] == list(range(21))
    assert event["rid"] == "request-A"
    assert event["media_s"] is None and event["ic_s"] is None and event["qw_s"] is None
    assert not event["availability"]["media_timing"]


def test_arrival_and_completion_windows_conserve_different_traffic():
    details = {"completion_s": 0.0025, "bytes": 60,
               "requested_bytes": 100, "status": "partial_miss"}
    emit(0.0002, 0.0023, details=details)
    event = trace.get_trace()[0]
    assert event["bytes"] == 60 and event["requested_bytes"] == 100
    _, offered = trace._windowed_bw([event], 0.001, "arrival")
    _, delivered = trace._windowed_bw([event], 0.001, "completion")
    assert offered == [100, 0, 0]
    assert delivered == [0, 0, 60]
    summary = trace.summarize()["by_pool_op"]["hbf"]["load"]
    assert summary["arrival_windows"]["1ms"]["total_bytes"] == 100
    assert summary["completion_windows"]["1ms"]["total_bytes"] == 60


def test_interval_inflight_is_not_same_arrival_burst():
    emit(0.0, 0.002)
    emit(0.001, 0.002)
    summary = trace.summarize()
    occupancy = summary["logical_inflight"]
    assert occupancy["peak"] == 2
    assert occupancy["time_weighted_mean"] == pytest.approx(4 / 3)
    assert occupancy["io_time_s"] == pytest.approx(0.004)
    assert summary["same_arrival_load_burst_max"] == 1
    assert summary["load_burst_max_concurrent"] == 1  # documented legacy alias


def test_touching_intervals_do_not_overlap_and_mock_is_excluded():
    emit(0.0, 0.001)
    emit(0.001, 0.001)
    emit(0.0, 1.0, bytes_=1000000, path="mock_lbk")
    summary = trace.summarize()
    assert summary["logical_inflight"]["peak"] == 1
    assert summary["total_load_bytes"] == 200
    assert summary["count_real"] == 2 and summary["count_mock_lbk"] == 1


def test_sub_win_groups_retain_physical_intervals_and_object_generation():
    details = {
        "logical_io_id": 5, "arrival_s": 0.0, "completion_s": 0.003,
        "bytes": 100, "requested_bytes": 100,
        "object_refs": [{"key": 4, "generation": 2}],
        "sub_ios": [
            {"win": 3, "bytes": 60, "arrival_s": 0.0, "completion_s": 0.002},
            {"win": 4, "bytes": 40, "arrival_s": 0.0, "completion_s": 0.003},
        ],
    }
    emit(0.0, 0.003, media_time=0.001, ic_time=0.002,
         queue_wait=0.0005, details=details)
    details["object_refs"][0]["generation"] = 999
    event = trace.get_trace()[0]
    assert event["details"]["object_refs"] == [{"key": 4, "generation": 2}]
    assert event["media_s"] == 0.001 and event["qw_s"] == 0.0005
    summary = trace.summarize()
    assert summary["logical_inflight"]["peak"] == 1
    assert summary["by_win_op"]["3"]["load"]["bytes"] == 60
    assert summary["by_win_op"]["4"]["load"]["logical_inflight"]["peak"] == 1
    assert summary["hbf_measured_media_bw_gbps"] is None
    assert summary["hbf_useful_bw_gbps"] == pytest.approx(100 / 0.003 / 1e9)


def test_working_set_proxy_is_not_required_capacity():
    trace.record("backup", "hbf", 11, 1100, 0.0, 0.001, node_ids=list(range(11)))
    trace.record("backup", "hbf", 1, 100, 0.002, 0.001, node_ids=[10])
    summary = trace.summarize()
    assert summary["working_set_approx"]["distinct_recorded_node_ids"] == 11
    assert summary["working_set_approx"]["required_capacity_bytes"] is None
    assert summary["total_backup_bytes"] == 1200
    assert summary["load_queue_wait_max_ms"] is None


def test_reset_clears_pending_request_context():
    trace.set_context("stale")
    trace.reset_trace()
    emit()
    assert trace.get_trace()[0]["rid"] is None


def test_real_win8_hook_preserves_breakdown_physical_target_and_dependency(monkeypatch):
    """Exercise the actual hook choke points and resource model together."""
    from hisim.simulation.sglang import sglang_hook as hook
    from hisim.simulation.hbf_shared_resource import SharedHbfResourceManager
    from hisim.simulation.hbf_win_pool import Hbf8WinConfig
    from hisim.simulation.manager import ConfigManager, StateManager
    from hisim.simulation.types import HBFConfig, PlatformConfig

    manager = SharedHbfResourceManager()
    manager.configure(media_bw=1.6e12, interconnect_bw=112e9,
                      media_model="win8", win_config=Hbf8WinConfig(
                          placement="striped", stripe_block_tokens=256,
                          package_shared_interface_bw=112e9))
    manager.begin_batch(0.0)
    monkeypatch.setattr(hook, "get_shared_hbf_manager", lambda: manager)
    monkeypatch.setattr(hook, "hbf_io_trace", trace)
    monkeypatch.setattr(ConfigManager, "_platform_config", PlatformConfig(
        device="h100_sxm", enable_hbf=True, hbf=HBFConfig()))
    monkeypatch.setenv("HISIM_HBF_TIMING_MODEL", "channel")
    StateManager.reset()
    try:
        slots, bpt = list(range(768, 784)), 114688
        written = hook._record_hbf_backup_timing(
            len(slots) * bpt, len(slots), virtual_indices=slots,
            node_ids=list(range(12)))
        loaded = hook._record_hbf_load_timing(
            len(slots) * bpt, len(slots), virtual_indices=slots)
        write_event, read_event = trace.get_trace()
        assert write_event["media_s"] > 0 and write_event["ic_s"] > 0
        assert read_event["qw_s"] >= written
        assert loaded > written
        assert write_event["node_ids"] == list(range(12))
        assert write_event["details"]["sub_ios"][0]["win"] == 3
        assert read_event["details"]["sub_ios"][0]["win"] == 3
        assert manager.win_pool.wins[3].io_count == 2
        assert manager.win_pool.wins[0].io_count == 0
        assert read_event["completion_s"] == manager.win_pool.get_io_events()[-1]["completion_s"]
        summary = trace.summarize()
        assert summary["by_win_op"]["3"]["load"]["bytes"] == len(slots) * bpt
        assert summary["load_queue_wait_max_ms"] > 0
    finally:
        StateManager.reset()


def test_mock_load_bookkeeping_does_not_submit_resources_or_accumulate_load(monkeypatch):
    from hisim.simulation.sglang import sglang_hook as hook
    from hisim.simulation.hbf_shared_resource import (
        SharedHbfResourceManager, SharedDramResourceManager)
    from hisim.simulation.hbf_win_pool import Hbf8WinConfig
    from hisim.simulation.manager import ConfigManager, StateManager
    from hisim.simulation.types import HBFConfig, PlatformConfig

    hbf = SharedHbfResourceManager()
    hbf.configure(media_bw=1.6e12, interconnect_bw=112e9,
                  media_model="win8", win_config=Hbf8WinConfig(
                      placement="striped", stripe_block_tokens=256))
    hbf.begin_batch(0.0)
    dram = SharedDramResourceManager()
    dram.configure(dram_bw=13.6e9)
    dram.begin_batch(0.0)
    monkeypatch.setattr(hook, "get_shared_hbf_manager", lambda: hbf)
    monkeypatch.setattr(hook, "get_shared_dram_manager", lambda: dram)
    monkeypatch.setattr(hook, "hbf_io_trace", trace)
    monkeypatch.setattr(ConfigManager, "_platform_config", PlatformConfig(
        device="h100_sxm", enable_hbf=True, hbf=HBFConfig()))
    monkeypatch.setenv("HISIM_HBF_TIMING_MODEL", "channel")
    monkeypatch.setenv("HISIM_SHARED_DRAM", "1")
    StateManager.reset()
    try:
        # Existing in-flight work makes even tiny accidental second loads
        # observable through cursor, count, and nonzero round latency.
        hbf.win_pool.submit_object(0.0, range(256), 114688, is_read=False)
        dram.submit(0.0, 114688 * 256, is_load=False)
        before_wins = [(w.io_count, w.media_cursor()) for w in hbf.win_pool.wins]
        before_dram = (dram.busy_until, len(dram.io_stats))
        before_events = len(hbf.win_pool.get_io_events())
        assert hook._record_hbf_load_timing(38080, 340, path="mock_lbk") == 0.0
        assert hook._record_l2_load_timing(38080, 340, path="mock_lbk") == 0.0
        assert [(w.io_count, w.media_cursor()) for w in hbf.win_pool.wins] == before_wins
        assert (dram.busy_until, len(dram.io_stats)) == before_dram
        assert len(hbf.win_pool.get_io_events()) == before_events
        assert StateManager.pop_hicache_hbf_load_dur() == 0.0
        assert StateManager.pop_hicache_l2_load_dur() == 0.0
        assert StateManager.get_global_clock() == 0.0
        events = trace.get_trace()
        assert len(events) == 2 and {e["pool"] for e in events} == {"hbf", "dram"}
        assert all(e["path"] == "mock_lbk" and e["dur_s"] == 0 for e in events)
        assert all(e["status"] == "diagnostic_mock" and not e["authoritative"] for e in events)
        assert all(e["media_s"] is None and e["details"] is None for e in events)
        assert trace.summarize()["count_real"] == 0
    finally:
        StateManager.reset()
