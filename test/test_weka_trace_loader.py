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
