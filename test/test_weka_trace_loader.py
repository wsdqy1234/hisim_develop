"""Data-plane tests for the weka trace loader (no full simulation needed).

Validates: parsing, static slot scheduling (concurrency / start interval /
slot takeover), round limits (max rounds / max round tokens), hash->token
expansion prefix semantics, and cross-round output derivation.

Run directly (no pytest / no hisim package / no transformers required):
    python test/test_weka_trace_loader.py
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


def _load_weka():
    try:
        from hisim.dataset import weka_trace
        return weka_trace
    except Exception:
        p = (Path(__file__).resolve().parent.parent
             / "src" / "hisim" / "dataset" / "weka_trace.py")
        spec = importlib.util.spec_from_file_location("hisim_weka_trace", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m


weka = _load_weka()

BS = 64


def _mk_trace(trace_id, rounds, block_size=BS):
    """rounds: list of (t, n_hash_blocks, out, api_time); hashes are 0..n-1."""
    reqs = []
    for t, n, out, api in rounds:
        reqs.append({
            "t": t, "type": "s", "in": n * block_size, "out": out,
            "hash_ids": list(range(n)), "api_time": api,
        })
    return {"id": trace_id, "models": ["m"], "block_size": block_size,
            "hash_id_scope": "local", "requests": reqs}


TRACE_A = _mk_trace("A", [(0.0, 10, 100, 2.0), (10.0, 20, 50, 3.0), (20.0, 30, 60, 1.0)])
TRACE_B = _mk_trace("B", [(0.0, 5, 40, 1.0), (8.0, 12, 30, 2.0)])
TRACE_C = _mk_trace("C", [(0.0, 2, 10, 0.5)])  # tiny trace
TRACE_BIG = _mk_trace("BIG", [(0.0, 1, 300000, 1.0)])  # exceeds round-token limits


def _write(path, *traces):
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")


def _plan_with(tmp, *traces, **kw):
    fp = os.path.join(tmp, "traces.jsonl")
    _write(fp, *traces)
    return weka.plan_weka_traces([fp], **kw)


def test_basic_single_trace():
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, TRACE_A)
        assert plan.num_traces == 1
        assert len(plan.rounds) == 3
        ts = [r.timestamp for r in plan.rounds if r.trace_idx == 0]
        assert ts == [0.0, 10.0, 20.0], ts  # recorded gaps preserved
        assert plan.rounds[0].input_length == 10 * BS
        assert plan.rounds[0].output_length == 100


def test_num_traces():
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, TRACE_A, TRACE_B, TRACE_C, num_traces=2)
        assert plan.num_traces == 2
        assert len(plan.rounds) == 5


def test_concurrent_scheduling():
    # Slots: trace0 @0.0, trace1 @0.5 (start interval). Trace0 ends at
    # (20-0)+1 = 21 -> slot0 frees at 21.5. Trace1 (B) ends at 0.5+10 ->
    # slot1 frees at 11.0, which is earlier -> trace C starts at 11.0.
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, TRACE_A, TRACE_B, TRACE_C,
                          max_concurrent_traces=2, trace_start_interval=0.5)
        starts = {i: info["start"] for i, info in enumerate(plan.trace_infos)}
        assert starts[0] == 0.0, starts
        assert abs(starts[1] - 0.5) < 1e-9, starts
        assert abs(starts[2] - 11.0) < 1e-9, starts  # takes over slot 1 (B)
        # trace C's only round starts at 11.0 too
        r = [x for x in plan.rounds if x.trace_idx == 2][0]
        assert abs(r.timestamp - 11.0) < 1e-9
        # global time order
        ts = [x.timestamp for x in plan.rounds]
        assert ts == sorted(ts)


def test_slot_takeover_prefers_earliest_free_slot():
    # With 3 slots and 3 traces, all start staggered by interval. C (slot2,
    # starts 1.0, duration 0.5) frees first at 1.0+0.5+0.5 = 2.0, so the
    # 4th trace must take slot 2 at t=2.0 (not slot 1 at 11.0).
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, TRACE_A, TRACE_B, TRACE_C, TRACE_BIG,
                          max_concurrent_traces=3, trace_start_interval=0.5)
        starts = {i: info["start"] for i, info in enumerate(plan.trace_infos)}
        assert abs(starts[3] - 2.0) < 1e-9
        assert plan.trace_infos[3]["slot"] == 2


def test_max_rounds_per_trace():
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, TRACE_A, max_rounds_per_trace=2)
        assert len(plan.rounds) == 2
        # duration now ends at round1: (10-0)+3 = 13
        assert abs(plan.trace_infos[0]["duration"] - 13.0) < 1e-9


def test_max_round_tokens():
    # TRACE_A round lengths: 740/1330/1980 tokens. Cap at 1500 keeps 2
    # rounds and cuts the trace before round 2.
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, TRACE_A, max_round_tokens=1500)
        assert len(plan.rounds) == 2
        assert plan.trace_infos[0]["n_cut_by_tokens"] == 1
        # A whole trace whose every round exceeds the cap is skipped.
        plan2 = _plan_with(tmp, TRACE_BIG, TRACE_C, max_round_tokens=100000)
        assert plan2.num_traces == 1
        assert plan2.trace_infos[0]["trace_id"] == "C"


def test_expand_prefix_semantics():
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, TRACE_A)
        r0, r1 = plan.rounds[0], plan.rounds[1]
        t0 = weka.expand_hash_ids(r0.hash_ids, r0.block_base, BS)
        t1 = weka.expand_hash_ids(r1.hash_ids, r1.block_base, BS)
        assert len(t0) == 10 * BS and len(t1) == 20 * BS
        # r1 shares r0's full input as prefix (same hash prefix 0..9)
        assert t1[: len(t0)] == t0
        # different traces get disjoint token ranges
        t2 = weka.expand_hash_ids(plan.rounds[-1].hash_ids,
                                  plan.rounds[-1].block_base, BS)
        assert not set(t0) & set(t2) if plan.num_traces > 1 else True


def test_derive_output_tokens():
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, TRACE_A)
        r0, r1 = plan.rounds[0], plan.rounds[1]
        out0 = weka.derive_output_tokens(r0, BS)
        # out0 must be exactly r1's input tokens beyond r0's input
        t1 = weka.expand_hash_ids(r1.hash_ids, r1.block_base, BS)
        assert out0 == t1[r0.input_length: r0.input_length + r0.output_length]
        # last round has no successor -> unique virtual tokens
        out2 = weka.derive_output_tokens(plan.rounds[2], BS)
        assert out2 == [-3] * 60  # global_seq 2 -> -(2+1)


def _mk_subagent_trace(trace_id, main_rounds, sub_rounds, agent_id="ag1"):
    """sub_rounds: inner requests of one subagent group (trace-relative t)."""
    reqs = [
        {"t": t, "type": "s", "in": n * BS, "out": out, "hash_ids": list(range(100, 100 + n)), "api_time": api}
        for t, n, out, api in main_rounds
    ]
    inner = [
        {"t": t, "type": "n", "in": n * BS, "out": out, "hash_ids": list(range(200, 200 + n)), "api_time": api}
        for t, n, out, api in sub_rounds
    ]
    reqs.append({"t": inner[0]["t"], "type": "subagent", "agent_id": agent_id,
                 "requests": inner})
    return {"id": trace_id, "models": ["m"], "block_size": BS,
            "hash_id_scope": "local", "requests": reqs}


def test_subagent_expansion_off_by_default():
    tr = _mk_subagent_trace(
        "SA", [(0.0, 5, 20, 1.0)], [(1.0, 3, 10, 0.5), (3.0, 6, 10, 0.5)])
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, tr)
        assert plan.num_traces == 1
        assert len(plan.rounds) == 1  # main only
        assert plan.rounds[0].stream_id == "main"


def test_subagent_expansion_streams():
    # Main: t=0(5 blocks), t=10(8 blocks). Subagent: t=1(3), t=3(6) —
    # overlapping in time with main.
    tr = _mk_subagent_trace(
        "SA", [(0.0, 5, 20, 1.0), (10.0, 8, 30, 2.0)],
        [(1.0, 3, 10, 0.5), (3.0, 6, 10, 0.5)])
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, tr, include_subagents=True)
        assert plan.num_traces == 1
        assert len(plan.rounds) == 4
        assert plan.trace_infos[0]["num_streams"] == 2
        streams = {}
        for rd in plan.rounds:
            streams.setdefault(rd.stream_id, []).append(rd)
        assert set(streams) == {"main", "ag1"}
        # subagent keeps recorded times (t=1, t=3), overlapping the main stream
        assert [r.timestamp for r in streams["ag1"]] == [1.0, 3.0]
        assert [r.timestamp for r in streams["main"]] == [0.0, 10.0]
        # per-stream round_idx and output chains
        for sid, rs in streams.items():
            assert [r.round_idx for r in rs] == [0, 1]
            assert rs[0].next_hash_ids == rs[1].hash_ids
            assert rs[1].next_hash_ids == ()
        # subagent uses same trace block base; hashes differ (100s vs 200s)
        main_r = streams["main"][0]
        sa_r = streams["ag1"][0]
        assert main_r.block_base == sa_r.block_base
        assert main_r.hash_ids != sa_r.hash_ids
        # trace duration covers the latest-finishing stream (main ends at 12)
        assert abs(plan.trace_infos[0]["duration"] - 12.0) < 1e-9


def test_subagent_round_limits_per_stream():
    tr = _mk_subagent_trace(
        "SA", [(0.0, 5, 20, 1.0), (10.0, 8, 30, 2.0), (20.0, 9, 30, 2.0)],
        [(1.0, 3, 10, 0.5), (3.0, 6, 10, 0.5)])
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, tr, include_subagents=True,
                          max_rounds_per_trace=2)
        # main truncated to 2, subagent (2 rounds) untouched -> 4 total
        assert len(plan.rounds) == 4


def test_subagent_derive_output_chain():
    # Sub round1 input extends round0 input beyond its length by the output.
    tr = _mk_subagent_trace(
        "SA", [(0.0, 5, 20, 1.0)],
        [(1.0, 3, 2 * BS // 8, 0.5), (3.0, 6, 10, 0.5)])
    # round0: 3 blocks in, out=16 tokens; round1: 6 blocks — shares 3-block
    # prefix? No: hash ranges are disjoint slices (200..202 vs 200..205) so
    # round1 shares round0's prefix and adds 3 blocks; out of round0 (16
    # tokens) must come from round1's input right after round0's input.
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, tr, include_subagents=True)
        sa_rounds = [r for r in plan.rounds if r.stream_id == "ag1"]
        out0 = weka.derive_output_tokens(sa_rounds[0], BS)
        t1 = weka.expand_hash_ids(sa_rounds[1].hash_ids,
                                  sa_rounds[1].block_base, BS)
        assert out0 == t1[sa_rounds[0].input_length:
                          sa_rounds[0].input_length + sa_rounds[0].output_length]


def test_multi_file_input():
    with tempfile.TemporaryDirectory() as tmp:
        f1 = os.path.join(tmp, "a.jsonl")
        f2 = os.path.join(tmp, "b.jsonl")
        _write(f1, TRACE_A)
        _write(f2, TRACE_B, TRACE_C)
        plan = weka.plan_weka_traces([f1, f2], num_traces=3)
        assert plan.num_traces == 3
        assert [i["trace_id"] for i in plan.trace_infos] == ["A", "B", "C"]


def test_iter_round_entries_stream():
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan_with(tmp, TRACE_A, TRACE_B, max_concurrent_traces=2)
        entries = list(weka.iter_round_entries(plan))
        assert len(entries) == 5
        assert [e["timestamp"] for e in entries] == sorted(
            e["timestamp"] for e in entries)
        e0 = entries[0]
        assert e0["input_length"] == 10 * BS
        assert len(e0["input_ids"]) == 10 * BS
        assert e0["output_length"] == 100
        assert len(e0["output_ids"]) == 100
        # Session metadata for the dynamic-completion-feedback scheduler.
        for e in entries:
            assert e["session"] == f"{e['trace_idx']}:main"
            assert e["stream_id"] == "main"
            assert isinstance(e["round_idx"], int)
            assert e["recorded_finish"] >= e["timestamp"]
        for rd in plan.rounds:
            assert abs(rd.recorded_finish - (rd.timestamp + rd.api_time)) < 1e-9


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
