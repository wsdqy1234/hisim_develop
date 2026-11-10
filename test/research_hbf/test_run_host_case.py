"""Lightweight reproducibility and isolation checks; no SGLang server required."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

SPEC = importlib.util.spec_from_file_location("research_run_host_case", Path(__file__).with_name("run_host_case.py"))
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_logical_references_include_misses_and_keep_output_separate():
    spec = runner.WorkloadSpec("deterministic", prefix_tokens=340, suffix_tokens=8,
                               concurrency=2, reuse=2, churn=1)
    requests, refs = runner.make_workload(spec, 114688)
    assert len(requests) == 4 + 2 * (1 + 2)
    assert sum(r["phase"] == "build" for r in refs) == 4
    assert sum(r["phase"] == "reuse" for r in refs) == 4
    assert all(r["input_tokens"] == r["prefix_tokens"] + r["suffix_tokens"] == 348 for r in refs)
    assert all(sum(b["tokens"] for b in r["blocks"]) == 340 for r in refs)
    assert all(sum(b["bytes"] for b in r["blocks"]) == 340 * 114688 for r in refs)
    assert all(r["requested_output_tokens"] == 16 for r in refs)
    first = refs[0]
    repeated = [r for r in refs if r["prefix_id"] == first["prefix_id"]]
    assert len(repeated) == 3
    assert all(r["blocks"] == first["blocks"] for r in repeated)
    assert len({r["suffix_id"] for r in repeated}) == 3


def test_frozen_workload_and_strict_arrival_order():
    spec = runner.WorkloadSpec("stable", prefix_tokens=16384, concurrency=3, reuse=2)
    assert runner.make_workload(spec, 114688) == runner.make_workload(spec, 114688)
    requests, _ = runner.make_workload(spec, 114688)
    arrivals = [r["arrival_s"] for r in requests]
    assert all(b > a for a, b in zip(arrivals, arrivals[1:]))
    assert len({r["request_id"] for r in requests}) == len(requests)
    assert max(max(r["input_ids"]) for r in requests) < 151936
    assert 1 not in requests[0]["input_ids"]
    paired, _ = runner.make_workload(runner.WorkloadSpec("paired", prefix_tokens=16384, concurrency=3, reuse=2), 114688)
    assert runner.workload_content_digest(requests) == runner.workload_content_digest(paired)


def test_effective_capacity_and_case_isolation(tmp_path):
    args = runner.parse_args(["--case-id", "isolated", "--port", "30231", "--output-root", str(tmp_path),
                              "--scenario", "mixed", "--host-capacity-tokens", "9000"])
    spec = runner.WorkloadSpec(args.case_id)
    plan = runner.build_plan(args, spec, {"bytes_per_token": 114688}, tmp_path / args.case_id)
    cap = plan["capacities"]
    assert cap["hbf_tokens"] + cap["dram_tokens"] == cap["host_total_tokens"] == 9000
    ratio = float(plan["command"][plan["command"].index("--hicache-ratio") + 1])
    assert int(cap["l1_tokens"] * ratio) == cap["dram_tokens"]
    assert plan["server_env"]["HISIM_OUTPUT_DIR"] == str(tmp_path / "isolated")
    assert plan["server_env"]["HISIM_SHARED_DRAM"] == "1"
    assert plan["server_env"]["HISIM_L3_WRITE_MODE"] == "barrier"
    assert "--allow-auto-truncate" not in plan["command"]


def test_prior_experiment_environment_cannot_leak(monkeypatch):
    monkeypatch.setenv("HISIM_HBF_CAPACITY_GB", "99999")
    monkeypatch.setenv("HISIM_L3_WRITE_MODE", "async")
    env = runner.clean_server_env({"HISIM_L3_WRITE_MODE": "barrier"})
    assert "HISIM_HBF_CAPACITY_GB" not in env
    assert env["HISIM_L3_WRITE_MODE"] == "barrier"


@pytest.mark.skipif(os.name != "posix", reason="Linux container process groups")
def test_cleanup_signals_only_the_case_owned_process_group():
    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        result = runner.stop_owned_server(owned)
        assert result["process_group"] == owned.pid
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        if owned.poll() is None:
            owned.kill()
            owned.wait()
        unrelated.terminate()
        unrelated.wait()


def make_export_fixture(path):
    row = {"rid": "case-r00000", "output_length": 2, "gen_token_latencies": [0.01, 0.01],
           "last_event_time": 0.02, "created_time": 0.0, "final_reused_tokens": 0,
           "l2_hit_tokens": 0, "recomputed_tokens": 348, "actual_input_tokens": 348,
           "hbf_hit_tokens": 0, "dram_hit_tokens": 0, "hbf_load_bytes": 0, "dram_load_bytes": 0}
    runner.write_json(path / "metrics.json", {"completed": 1})
    runner.write_jsonl(path / "request.jsonl", [row])
    runner.write_jsonl(path / "iteration.jsonl", [])
    runner.write_jsonl(path / "io_trace.jsonl", [])
    runner.write_json(path / "effective_config.json", {
        "geometry": {"bytes_per_token": 114688}, "server_env": {"HISIM_HBF_ENABLE": "0"}})
    return row, [{"request_id": row["rid"]}]


def test_validation_rejects_negative_token_time_without_clamping(tmp_path):
    row, workload = make_export_fixture(tmp_path)
    row["gen_token_latencies"][1] = -0.0001
    runner.write_jsonl(tmp_path / "request.jsonl", [row])
    with pytest.raises(RuntimeError, match="negative token latency"):
        runner.verify_outputs(tmp_path, workload)
    assert json.loads((tmp_path / "validation.json").read_text())["valid"] is False
    assert json.loads((tmp_path / "request.jsonl").read_text())["gen_token_latencies"][1] == -0.0001


def test_validation_rejects_warmup_request_contamination(tmp_path):
    row, workload = make_export_fixture(tmp_path)
    extra = dict(row, rid="")
    runner.write_jsonl(tmp_path / "request.jsonl", [row, extra])
    runner.write_json(tmp_path / "metrics.json", {"completed": 2})
    with pytest.raises(RuntimeError, match="count differs"):
        runner.verify_outputs(tmp_path, workload)


def test_validation_rejects_child_route_mismatch_even_when_bytes_conserve(tmp_path):
    _, workload = make_export_fixture(tmp_path)
    runner.write_json(tmp_path / "effective_config.json", {
        "geometry": {"bytes_per_token": 114688}, "server_env": {"HISIM_HBF_ENABLE": "1"},
        "win_config": {"stripe_block_tokens": 128}})
    event = {"logical_io_id": 1, "bytes": 114688, "sub_ios": [
        {"bytes": 114688, "slots": [128], "win": 0, "arrival_s": 0, "completion_s": 0.01}]}
    runner.write_jsonl(tmp_path / "win_io_events.jsonl", [event])
    runner.write_jsonl(tmp_path / "io_trace.jsonl", [{"pool": "hbf", "op": "backup", "bytes": 114688, "path": "real",
                                                  "arrival_s": 0, "completion_s": 0.01, "details": event}])
    runner.write_json(tmp_path / "win_pool_summary.json", {"conservation": {"bytes_consistent": True, "slots_consistent": True}})
    with pytest.raises(RuntimeError, match="immutable slot stripe"):
        runner.verify_outputs(tmp_path, workload)


def test_mock_placeholder_diagnostics_do_not_count_as_actual_kv_bytes(tmp_path):
    _, workload = make_export_fixture(tmp_path)
    runner.write_jsonl(tmp_path / "io_trace.jsonl", [{"pool": "hbf", "op": "load", "bytes": 38080,
                                                   "path": "mock_lbk", "arrival_s": 0, "completion_s": 0.01}])
    result = runner.verify_outputs(tmp_path, workload)
    assert result["valid"]
    assert result["hbf_load_bytes"] == 0
    assert result["diagnostics"]["non_real_hbf_bytes"] == 38080
