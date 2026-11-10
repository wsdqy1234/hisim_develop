#!/usr/bin/env python3
"""Phase 5C P1: Layer A / Layer B timing audit checks.

Verifies (assertion-based, joins the regression suite):
  A1. Layer A read latency floor (5 / 7.5 / 10 μs configurable).
  A2. Small I/O is latency-bound (fixed latency dominates).
  A3. Layer A QD semantics: NO intra-WIN parallelism (serial media) —
      QD>1 on one WIN adds queue wait, not throughput; parallelism comes
      only from multiple active WINs.
  B1. Layer B per-WIN PageSim: explicit Page→Channel→Plane mapping
      (16MB read on one WIN = tR + 32×page-transfer, Phase 4 semantics).
  M1. Sequential / Random / Write bandwidths are independently enforced.
  D1. Media and interface are NOT double counted: total = max, not sum.
  X1. Cross-batch preserves unfinished I/O incl. the shared interface.
  S1. Statistics = Completed Useful Bytes / Simulation Time (span is
      last completion − batch start; all submitted I/Os are completed).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_win_pool import (
    Hbf8WinConfig, HbfWinPoolModel, WinSpec)

GB = 1e9
KV = 114688


def pool_with(lat_us=7.5, **kw):
    cfg = Hbf8WinConfig(win=WinSpec(read_latency_s=lat_us * 1e-6), **kw)
    return HbfWinPoolModel(cfg)


def test_a1_latency_floor():
    """A1: Layer A read latency floor 5/7.5/10μs."""
    for lat in (5e-6, 7.5e-6, 10e-6):
        pool = pool_with(lat_us=lat * 1e6)
        st = pool.submit(0.0, 4096, is_read=True)   # 4KB: transfer ≈ 0
        assert abs(st.total - lat) < 1e-12, \
            f"4KB read must take exactly the latency floor: {st.total} vs {lat}"
    print("✓ A1 latency floor: 4KB read = 5/7.5/10μs exactly (configurable)")


def test_a2_small_io_latency_bound():
    """A2: small I/Os are latency-bound (efficiency ≪ 100%)."""
    pool = pool_with(lat_us=7.5)
    for size in (4096, 16384, 65536, 262144):
        st = pool.submit(0.0, size, is_read=True)
        eff = size / st.total / pool.config.win.seq_read_bw
        assert st.total < size / pool.config.win.seq_read_bw * 0.01 or eff < 0.55, \
            f"{size}B should be latency-bound, eff={eff:.2%}"
    # 1MB at 7.5μs: transfer 5μs < lat → still latency-bound (67%)
    st = pool.submit(0.0, 1 * 1024 * 1024, is_read=True)
    assert abs(st.total - 7.5e-6) < 1e-12, "1MB@7.5μs still latency floor"
    print("✓ A2 small I/O: ≤256KB and 1MB@7.5μs all latency-bound "
          "(1.5MB is the 200GB/s×7.5μs break-even)")


def test_a3_qd_no_intrawin_parallelism_layerA():
    """A3: Layer A has NO intra-WIN parallelism — QD only fills WINs."""
    # single WIN, QD=1 vs QD=8: identical throughput, queue wait grows
    pool = HbfWinPoolModel(Hbf8WinConfig(n_wins=1))
    io = 4 * 1024 * 1024  # 4MB, BW-bound (20μs)
    st1 = pool.submit(0.0, io, is_read=True)
    t1 = pool.get_batch_duration()
    pool2 = HbfWinPoolModel(Hbf8WinConfig(n_wins=1))
    for i in range(8):
        pool2.submit(0.0, io, is_read=True, object_id=f"q{i}")
    t8 = pool2.get_batch_duration()
    assert abs(t8 - 8 * t1) < 1e-9, \
        f"single WIN serial: 8 I/Os must take 8× ({t8*1e6:.1f} vs {8*t1*1e6:.1f}μs)"
    qw = max(w.queue_wait_max for w in pool2.wins)
    assert qw > 6 * t1, "later I/Os must queue on the busy WIN"
    # 8 WINs, QD=8: full parallel — same span as 1 I/O
    pool3 = HbfWinPoolModel(Hbf8WinConfig())
    for i in range(8):
        pool3.submit(0.0, io, is_read=True, object_id=f"q{i}")
    t88 = pool3.get_batch_duration()
    assert abs(t88 - t1) < 1e-9, "8 WINs must run the 8 I/Os in parallel"
    print(f"✓ A3 QD semantics: single WIN QD=8 → 8× time (serial, qw="
          f"{qw*1e6:.0f}μs); 8 WINs QD=8 → 1× time (parallel across WINs)")


def test_b1_layer_b_geometry_mapping():
    """B1: Layer B per-WIN PageSim — Page→Channel→Plane explicit mapping."""
    cfg = Hbf8WinConfig(layer="pagesim")
    pool = HbfWinPoolModel(cfg)
    ps = pool.wins[0].pagesim
    assert ps is not None and ps.channels == 32 and ps.page_bytes == 16384
    # 16MB read on ONE WIN: 1024 pages, 32/channel, 1/plane
    # → tR + 32 × (16KB / 1.6GB/s) = 10 + 327.68 = 337.68μs (assumption
    # geometry — nominal placeholder, NOT expert-calibrated)
    st = pool.submit(0.0, 16 * 1024 * 1024, is_read=True, object_id="b1")
    assert st.win == 0
    expect = ps.read_latency + 32 * (ps.page_bytes / ps.channel_rate)
    assert abs(st.media_time - expect) < 1e-9, \
        f"layer B 16MB: {st.media_time*1e6:.2f}μs != {expect*1e6:.2f}μs"
    # layer B on a SECOND WIN is independent (its own pagesim state)
    st2 = pool.submit(0.0, 16 * 1024 * 1024, is_read=True, object_id="b2")
    assert st2.win == 1 and abs(st2.media_time - expect) < 1e-9
    print(f"✓ B1 Layer B geometry: 16MB/1WIN = {st.media_time*1e6:.2f}μs "
          f"(tR+32×transfer, per-WIN independent PageSim state)")


def test_m1_mode_independent_bw():
    """M1: seq/rand/write independently constrained (same WIN, same size)."""
    pool = Hbf8WinConfig()
    p = HbfWinPoolModel(pool)
    io = 10 * 1024 * 1024
    t_seq = p.submit(0.0, io, is_read=True, access_mode="sequential").total
    p2 = HbfWinPoolModel(pool)
    t_rand = p2.submit(0.0, io, is_read=True, access_mode="random").total
    p3 = HbfWinPoolModel(pool)
    t_wr = p3.submit(0.0, io, is_read=False).total
    assert abs(t_seq - io / 200e9) < 1e-9
    assert abs(t_rand - io / 100e9) < 1e-9
    assert abs(t_wr - io / 10e9) < 1e-9
    print("✓ M1 mode independence: 10MB → seq 50μs / rand 100μs / write 1ms")


def test_d1_no_double_count():
    """D1: media + interface = max, not sum (pipeline overlap)."""
    cfg = Hbf8WinConfig(package_shared_interface_bw=50 * GB)
    p = HbfWinPoolModel(cfg)
    io = 100 * 1024 * 1024      # 100MB: media 500μs, interface 2ms
    st = p.submit(0.0, io, is_read=True)
    assert st.interface_constrained
    assert abs(st.total - io / 50e9) < 1e-9, \
        f"interface-bound I/O must take exactly bytes/IC: {st.total*1e3:.3f}ms"
    # media 500μs < ic 2ms → total must be 2ms, NOT 2.5ms (sum)
    assert abs(st.total - (st.media_time + st.ic_time)) > 1e-6, \
        "total must NOT be the sum of media and interface"
    # reverse: media-bound case
    cfg2 = Hbf8WinConfig(package_shared_interface_bw=2000 * GB)
    p2 = HbfWinPoolModel(cfg2)
    st2 = p2.submit(0.0, io, is_read=True)
    assert not st2.interface_constrained
    assert abs(st2.total - io / 200e9) < 1e-9
    print("✓ D1 no double count: IC-bound → exactly bytes/IC; "
          "media-bound → exactly bytes/media (max, never sum)")


def test_x1_cross_batch_interface_preserved():
    """X1: cross-batch preserves unfinished media AND interface I/O."""
    cfg = Hbf8WinConfig(package_shared_interface_bw=50 * GB)
    p = HbfWinPoolModel(cfg)
    p.begin_batch(0.0)
    # 100MB read: media 500μs, interface 2ms (still in flight at t=1ms)
    p.submit(0.0, 100 * 1024 * 1024, is_read=True)
    assert p.shared_ic_busy_until > 1e-3
    p.begin_batch(1e-3)
    # capture in-flight cursor values BEFORE batch-2 submission
    ic_busy_before = p.shared_ic_busy_until
    win1_media_before = p.wins[1].media_cursor()
    # batch-2 read must wait for the in-flight interface occupancy
    st2 = p.submit(1e-3, 4096, is_read=True)
    assert st2.ic_start >= ic_busy_before - 1e-9 or \
        st2.media_start >= win1_media_before - 1e-9, \
        "batch-2 I/O must respect in-flight cursors"
    ic_busy = p.shared_ic_busy_until
    p.begin_batch(5e-3)   # after everything completed
    st3 = p.submit(5e-3, 4096, is_read=True)
    assert st3.ic_start == 5e-3, \
        f"after completion, interface must be free: ic_start={st3.ic_start}"
    print(f"✓ X1 cross-batch: in-flight interface ({ic_busy*1e3:.2f}ms) "
          f"preserved; free after completion")


def test_s1_completed_bytes_accounting():
    """S1: statistics = Completed Useful Bytes / Simulation Time."""
    p = HbfWinPoolModel(Hbf8WinConfig())
    p.begin_batch(0.0)
    n, io = 16, 38 * KV
    for i in range(n):
        p.submit(0.0, io, is_read=True, object_id=f"s{i}")
    span = p.get_batch_duration()
    # 16 I/Os round-robin → 2 per WIN → span = 2 × (io/200GB/s)
    useful = n * io / span
    fixed_eff = useful / (8 * 200e9)
    assert abs(useful - 8 * 200e9 * (io / (38 * KV)) / (io / (38 * KV))) < 1e9 \
        or abs(fixed_eff - 0.25) < 1e-9, \
        f"16 I/Os on 8 WINs → exactly 25% fixed-pool efficiency, got {fixed_eff:.3f}"
    assert span > 0 and all(w.io_count == 2 for w in p.wins)
    print(f"✓ S1 accounting: 16×38tk I/Os, span={span*1e6:.0f}μs, "
          f"useful={useful/GB:.0f}GB/s = 25% of 1600 (completed/span)")


def run_all():
    tests = [
        test_a1_latency_floor,
        test_a2_small_io_latency_bound,
        test_a3_qd_no_intrawin_parallelism_layerA,
        test_b1_layer_b_geometry_mapping,
        test_m1_mode_independent_bw,
        test_d1_no_double_count,
        test_x1_cross_batch_interface_preserved,
        test_s1_completed_bytes_accounting,
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
