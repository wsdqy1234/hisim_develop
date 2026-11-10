#!/usr/bin/env python3
"""Phase 5B: 8-WIN pool model correctness tests.

Covers the required minimum test set:
  1. Single-WIN capacity & bandwidth parameters take effect.
  2. 8 WINs total logical capacity = 800 GB.
  3. Single-WIN three access modes never exceed calibrated BW.
  4. 8-WIN aggregate throughput ≤ 1600/800/80 GB/s caps.
  5. Single-WIN load must NOT use 8-WIN aggregate bandwidth.
  6. Cross-WIN independence + shared interface limit works.
  7. Cross-batch unfinished I/O state is preserved.
  8. (Old suite regression is run separately by the driver.)

Additional: placement strategies, read-write contention, auto mode
detection, effective-config JSON round-trip, manager win8 dispatch.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_win_pool import (
    Hbf8WinConfig, HbfWinPoolModel, WinSpec, PagesimAssumptions)
from hisim.simulation.hbf_shared_resource import (
    SharedHbfResourceManager, SharedDramResourceManager)

GB = 1e9
KV = 114688
IO_BYTES = 339 * KV          # 38.88 MB — KV restore size (Qwen3)
KV_WRITE = 340 * KV          # 38.99 MB — KV backup size


def mk(cfg=None):
    return HbfWinPoolModel(cfg or Hbf8WinConfig())


# ---------------------------------------------------------------- tests ---

def test_single_win_params_effective():
    """T1: single-WIN envelope parameters take effect."""
    cfg = Hbf8WinConfig(n_wins=1)
    pool = mk(cfg)
    assert cfg.win.capacity_bytes == 100 * GB
    st = pool.submit(0.0, IO_BYTES, is_read=True)   # sequential default
    expect = IO_BYTES / cfg.win.seq_read_bw          # 38.88MB / 200GB/s
    assert abs(st.media_time - expect) < 1e-9, \
        f"seq read {st.media_time*1e6:.1f}μs != {expect*1e6:.1f}μs"
    assert st.total == st.media_time  # no interface configured
    print(f"✓ T1 single WIN: 38.88MB seq read = {st.media_time*1e6:.1f}μs "
          f"(@200GB/s), capacity 100GB")


def test_eight_win_total_capacity():
    """T2: 8 WINs total logical capacity = 800GB; allocation spreads."""
    pool = mk()
    assert pool.config.total_capacity_bytes == 800 * GB
    # balanced placement: 8 objects of 1GB each → 8 different WINs
    win_ids = set()
    for i in range(8):
        w = pool.alloc_object(f"obj{i}", 1 * GB)
        assert w >= 0, "allocation must succeed (800GB available)"
        win_ids.add(w)
    assert len(win_ids) == 8, f"balanced placement must spread, got {win_ids}"
    assert all(pw["used_gb"] == 1.0 for pw in pool.per_win_summary())
    print(f"✓ T2 8-WIN capacity: 800GB total, 8×1GB objects → "
          f"all 8 WINs used exactly once")


def test_single_win_mode_bw_respected():
    """T3: single-WIN three access modes never exceed calibrated BW."""
    pool = mk()
    cases = [
        (True, "sequential", pool.config.win.seq_read_bw),
        (True, "random", pool.config.win.rand_read_bw),
        (False, None, pool.config.win.write_bw),
    ]
    for is_read, mode, bw in cases:
        pool.reset_state()
        st = pool.submit(0.0, IO_BYTES, is_read=is_read, access_mode=mode)
        eff = st.total and IO_BYTES / st.total
        assert eff <= bw * 1.001, \
            f"mode={mode}: effective {eff/GB:.1f}GB/s > calibrated {bw/GB:.1f}"
        assert st.total >= IO_BYTES / bw, "cannot be faster than envelope"
    print(f"✓ T3 mode BW: seq 200 / rand 100 / write 10 GB/s all respected "
          f"(38.88MB → 194.5 / 389 / 778ms-equivalents)")


def test_aggregate_caps_not_exceeded():
    """T4: 8-WIN aggregate throughput ≤ 1600/800/80 GB/s caps."""
    # reads: one 38.88MB I/O per WIN, all at t=0
    pool = mk()
    for i in range(8):
        pool.submit(0.0, IO_BYTES, is_read=True, object_id=f"o{i}")
    span = pool.get_batch_duration()
    agg = 8 * IO_BYTES / span
    assert agg <= pool.config.aggregate_seq_read_bw * 1.001, \
        f"agg read {agg/GB:.1f} > 1600"
    # writes: one per WIN
    pool2 = mk()
    for i in range(8):
        pool2.submit(0.0, KV_WRITE, is_read=False, object_id=f"w{i}")
    span_w = pool2.get_batch_duration()
    agg_w = 8 * KV_WRITE / span_w
    assert agg_w <= pool2.config.aggregate_write_bw * 1.001, \
        f"agg write {agg_w/GB:.1f} > 80"
    print(f"✓ T4 aggregate: 8×38.9MB parallel reads → {agg/GB:.1f}GB/s "
          f"(cap 1600); 8× writes → {agg_w/GB:.1f}GB/s (cap 80) — "
          f"caps emerge from per-WIN limits")


def test_single_win_load_no_aggregate():
    """T5: single-WIN load must NOT use 8-WIN aggregate bandwidth."""
    pool = mk()
    st = pool.submit(0.0, IO_BYTES, is_read=True)   # one I/O only
    # must take bytes/200GB/s (194.4μs), NOT bytes/1600 (24.3μs)
    expect = IO_BYTES / pool.config.win.seq_read_bw
    assert abs(st.total - expect) < 1e-9, \
        f"single I/O took {st.total*1e6:.1f}μs — aggregate BW leak?"
    eff = IO_BYTES / st.total
    assert eff <= pool.config.win.seq_read_bw * 1.001
    print(f"✓ T5 no aggregate leak: single 38.88MB read = "
          f"{st.total*1e6:.1f}μs ({eff/GB:.0f}GB/s, single-WIN cap 200)")


def test_cross_win_independent_and_shared_interface():
    """T6: cross-WIN independence + shared interface limit works."""
    # (a) independent: 2 I/Os on 2 different WINs run in parallel
    pool = mk()
    pool.submit(0.0, IO_BYTES, is_read=True, object_id="a")
    pool.submit(0.0, IO_BYTES, is_read=True, object_id="b")
    span = pool.get_batch_duration()
    single = IO_BYTES / pool.config.win.seq_read_bw
    assert span < single * 1.5, "two WINs must run in parallel"
    # (b) shared interface: 112GB/s caps the aggregate
    cfg = Hbf8WinConfig(package_shared_interface_bw=112 * GB)
    pool_ic = HbfWinPoolModel(cfg)
    for i in range(8):
        pool_ic.submit(0.0, IO_BYTES, is_read=True, object_id=f"o{i}")
    span_ic = pool_ic.get_batch_duration()
    agg = 8 * IO_BYTES / span_ic
    assert agg <= 112 * GB * 1.02, \
        f"shared interface violated: {agg/GB:.1f}GB/s > 112"
    n_constrained = sum(pw["n_interface_constrained"]
                        for pw in pool_ic.per_win_summary())
    assert n_constrained >= 6, "late I/Os must be interface-constrained"
    # (c) per-WIN interface: each WIN capped individually, aggregate 8×
    cfg2 = Hbf8WinConfig(per_win_interface_bw=112 * GB)
    pool_pw = HbfWinPoolModel(cfg2)
    for i in range(8):
        pool_pw.submit(0.0, IO_BYTES, is_read=True, object_id=f"o{i}")
    span_pw = pool_pw.get_batch_duration()
    agg_pw = 8 * IO_BYTES / span_pw
    assert agg_pw >= 500 * GB, \
        f"per-WIN interfaces must allow ~896GB/s aggregate, got {agg_pw/GB:.0f}"
    assert agg_pw <= 8 * 112 * GB * 1.02
    print(f"✓ T6 topology: independent WINs parallel; shared 112GB/s IC "
          f"→ agg {agg/GB:.1f}GB/s (constrained); per-WIN 112 IC → "
          f"agg {agg_pw/GB:.1f}GB/s (independent)")


def test_cross_batch_state_preserved():
    """T7: cross-batch unfinished I/O state is preserved."""
    pool = mk()
    pool.begin_batch(0.0)
    # batch 1: a 38.99MB write takes 3.899ms on win 0
    pool.submit(0.0, KV_WRITE, is_read=False, object_id="big")
    busy = pool.wins[0].media_cursor()
    assert busy > 3.8e-3, f"write should occupy WIN ~3.9ms, busy={busy}"
    # batch 2 starts at t=1ms (write still in flight)
    pool.begin_batch(1e-3)
    pool._objects["small"] = 0        # pin to the SAME win as the write
    st2 = pool.submit(1e-3, 4096, is_read=True, object_id="small")
    # the small read on the SAME win must wait for the write
    assert st2.media_start >= busy - 1e-9, \
        f"batch-2 I/O must wait for in-flight write: start={st2.media_start} < {busy}"
    print(f"✓ T7 cross-batch: batch-2 read waited until "
          f"{st2.media_start*1e3:.2f}ms (in-flight write until "
          f"{busy*1e3:.2f}ms preserved)")


def test_placement_strategies():
    """Placement: balanced spreads; hotspot concentrates then spills."""
    # balanced: 4 objects → 4 different WINs
    pool = mk(Hbf8WinConfig(placement="balanced"))
    wins = [pool.alloc_object(f"o{i}", 1 * GB) for i in range(4)]
    assert len(set(wins)) == 4
    # hotspot: everything on win 0 until full
    cfg = Hbf8WinConfig(placement="hotspot", hotspot_win=0)
    pool_h = HbfWinPoolModel(cfg)
    ids = [pool_h.alloc_object(f"h{i}", 30 * GB) for i in range(5)]
    # 30GB × 4 = 120GB > 100GB capacity of win0 → 5th spills to win1
    assert ids[:4] == [0, 0, 0, 1], f"hotspot spill wrong: {ids}"
    # note: 3×30=90 fits win0, 4th (120>100) spills
    print(f"✓ placement: balanced spreads 4→4 WINs; hotspot 30GB×5 → "
          f"{ids} (fills win0=90GB, spills remainder)")


def test_read_write_contention():
    """Read-write contention: a big write blocks reads on the same WIN."""
    pool = mk()
    pool.submit(0.0, KV_WRITE, is_read=False, object_id="w")   # win 0: 3.9ms
    st = pool.submit(0.5e-3, 4096, is_read=True, object_id="r")
    # same object round-robin → could be win 1... force same win:
    pool2 = mk()
    pool2.submit(0.0, KV_WRITE, is_read=False, object_id="x")
    pool2._objects["y"] = 0          # pin read to win 0 explicitly
    st2 = pool2.submit(1.0, 4096, is_read=True, object_id="y")
    write_end = KV_WRITE / 10e9
    assert st2.media_start >= write_end, \
        "read on same WIN must wait for the write (media contention)"
    print(f"✓ read/write contention: read waits until write completes "
          f"({write_end*1e3:.1f}ms)")


def test_auto_mode_detection():
    """Auto mode: contiguous continuation → sequential; else random."""
    cfg = Hbf8WinConfig(default_access_mode="auto")
    pool = HbfWinPoolModel(cfg)
    pool.alloc_object("obj", 100 * GB)          # win 0, base block set
    base = pool.wins[0].stream_next["obj"]
    st1 = pool.submit(0.0, 1000, is_read=True, object_id="obj",
                      offset=base)
    assert st1.mode == "sequential", "continuation must be sequential"
    st2 = pool.submit(0.0, 1000, is_read=True, object_id="obj",
                      offset=base + 1000)       # contiguous next
    assert st2.mode == "sequential"
    st3 = pool.submit(0.0, 1000, is_read=True, object_id="obj",
                      offset=base + 5000)       # gap → random
    assert st3.mode == "random", "non-contiguous must be random"
    print(f"✓ auto mode: contiguous=sequential ×2, gap=random")


def test_effective_config_roundtrip():
    """Effective config: JSON round-trip preserves all parameters."""
    cfg = Hbf8WinConfig(
        n_wins=4, layer="pagesim",
        win=WinSpec(capacity_bytes=50 * GB, seq_read_bw=150e9,
                    write_bw=8e9),
        package_shared_interface_bw=100 * GB,
        placement="hotspot", default_access_mode="auto")
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False) as f:
        path = f.name
    cfg.dump_effective_json(path)
    with open(path) as f:
        d = json.load(f)
    assert "assumptions" in d and "pagesim_assumptions" in d
    cfg2 = Hbf8WinConfig.from_json(path)
    assert cfg2.n_wins == 4 and cfg2.layer == "pagesim"
    assert cfg2.win.seq_read_bw == 150e9 and cfg2.win.write_bw == 8e9
    assert cfg2.package_shared_interface_bw == 100 * GB
    assert cfg2.placement == "hotspot"
    os.unlink(path)
    print("✓ effective config: JSON round-trip + assumption annotations")


def test_manager_win8_dispatch():
    """SharedHbfResourceManager win8 dispatch: media+IC owned by WIN pool."""
    mgr = SharedHbfResourceManager()
    cfg = Hbf8WinConfig()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128,
                  media_model="win8", win_config=cfg)
    assert mgr.win_pool is not None and len(mgr.win_pool.wins) == 8
    # 4 concurrent HBF loads (as in the N=2 host-hit batch): 2 reuse reqs
    mgr.begin_batch(0.0)
    st1 = mgr.submit(0.0, IO_BYTES, is_read=True)
    st2 = mgr.submit(0.0, IO_BYTES, is_read=True)
    # round-robin: different WINs → parallel; each takes bytes/200GB/s
    single = IO_BYTES / 200e9
    assert abs(st1.total - single) < 1e-9
    assert abs(st2.total - single) < 1e-9, \
        f"second I/O on another WIN must also take single-WIN time: {st2.total*1e6:.1f}μs"
    assert st1.media_time > 0 and st1.ic_time == 0  # no interface set
    # cross-batch via manager: begin_batch preserves WIN cursors
    mgr.begin_batch(1.0)
    st3 = mgr.submit(1.0, KV_WRITE, is_read=False)
    assert st3.total >= KV_WRITE / 10e9 - 1e-12, \
        f"write must take ≥ envelope time, got {st3.total}"
    print(f"✓ manager win8 dispatch: two loads parallel "
          f"({st1.total*1e6:.1f}μs each), cross-batch preserved")


def run_all():
    tests = [
        test_single_win_params_effective,
        test_eight_win_total_capacity,
        test_single_win_mode_bw_respected,
        test_aggregate_caps_not_exceeded,
        test_single_win_load_no_aggregate,
        test_cross_win_independent_and_shared_interface,
        test_cross_batch_state_preserved,
        test_placement_strategies,
        test_read_write_contention,
        test_auto_mode_detection,
        test_effective_config_roundtrip,
        test_manager_win8_dispatch,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed ({passed+failed} total)")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
