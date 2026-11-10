#!/usr/bin/env python3
"""Phase 3E: Cross-batch resource lifecycle and DRAM shared model tests."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_shared_resource import (
    SharedHbfResourceManager,
    SharedDramResourceManager,
    get_shared_hbf_manager,
    get_shared_dram_manager,
)


def test_cross_batch_preserves_busy_until():
    """Batch 2 must see Batch 1's unfinished I/O resource occupation.

    Counterexample: Batch 1 at T=0 starts a 10ms HBF Read.
    Batch 2 at T=2ms starts another Read — it MUST wait for Batch 1's
    media to be free, not start from T=2ms with full bandwidth.
    """
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)

    # Batch 1 at T=0: 233MB read → media takes 4.55ms, ic takes 14.56ms
    mgr.begin_batch(0.0)
    s1 = mgr.submit(arrival=0.0, bytes_=233e6, is_read=True)
    batch1_dur = mgr.get_batch_duration()
    # batch1_dur ≈ 14.56ms (interconnect bottleneck)

    # Simulate clock advancing only 2ms (not full 14.56ms)
    # This can happen if HBF load is not the bottleneck in combined_dur
    clock_after_batch1 = 0.002  # 2ms

    # Batch 2 at T=2ms
    mgr.begin_batch(clock_after_batch1)
    s2 = mgr.submit(arrival=clock_after_batch1, bytes_=233e6, is_read=True)

    # Batch 2's I/O should NOT start at T=2ms with full bandwidth.
    # It should wait for Batch 1's resources to be free.
    # media_busy_until from Batch 1 = 0 + 4.55ms = 4.55ms > 2ms
    # So Batch 2's media_start should be >= 4.55ms
    assert s2.media_start >= 0.004, \
        f"Cross-batch: media_start={s2.media_start:.6f} should be >= 0.004 " \
        f"(Batch 1 media still busy)"

    # ic_busy_until from Batch 1 = 0 + 14.56ms = 14.56ms > 2ms
    # So Batch 2's ic_start should be >= 14.56ms
    assert s2.ic_start >= 0.014, \
        f"Cross-batch: ic_start={s2.ic_start:.6f} should be >= 0.014 " \
        f"(Batch 1 interconnect still busy)"

    print(f"✓ Cross-batch: Batch 2 media_start={s2.media_start*1e3:.3f}ms "
          f"(≥4.55ms), ic_start={s2.ic_start*1e3:.3f}ms (≥14.56ms)")


def test_cross_batch_no_oversell():
    """Aggregate throughput across batches must not exceed resource capacity."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)

    # Batch 1: 1 I/O of 100MB
    mgr.begin_batch(0.0)
    mgr.submit(arrival=0.0, bytes_=100e6, is_read=True)
    b1_dur = mgr.get_batch_duration()

    # Batch 2 at T = b1_dur (clock advanced by full batch duration)
    mgr.begin_batch(b1_dur)
    mgr.submit(arrival=b1_dur, bytes_=100e6, is_read=True)
    b2_dur = mgr.get_batch_duration()

    # Each batch should have similar duration (no cross-batch interference
    # when clock advances by full batch duration)
    ratio = b2_dur / b1_dur if b1_dur > 0 else 0
    assert 0.9 < ratio < 1.1, \
        f"Cross-batch no-oversell: b1={b1_dur*1e3:.3f}ms b2={b2_dur*1e3:.3f}ms ratio={ratio:.2f}"

    print(f"✓ No oversell: b1={b1_dur*1e3:.3f}ms b2={b2_dur*1e3:.3f}ms ratio={ratio:.2f}")


def test_dram_shared_single_io():
    """DRAM shared model: single I/O matches independent model."""
    mgr = SharedDramResourceManager()
    mgr.configure(dram_bw=13.6e9)

    mgr.begin_batch(0.0)
    bytes_ = 38.9e6  # 339 tokens
    stats = mgr.submit(arrival=0.0, bytes_=bytes_, is_load=True)

    expected = 6.67e-6 + bytes_ / 13.6e9
    assert abs(stats["total"] - expected) < 1e-6, \
        f"DRAM single: total={stats['total']:.9f} expected={expected:.9f}"

    print(f"✓ DRAM single: total={stats['total']*1e3:.3f}ms")


def test_dram_shared_concurrent():
    """DRAM shared model: 2 concurrent I/Os share bandwidth."""
    mgr = SharedDramResourceManager()
    mgr.configure(dram_bw=13.6e9)

    mgr.begin_batch(0.0)
    bytes_ = 38.9e6

    s1 = mgr.submit(arrival=0.0, bytes_=bytes_, is_load=True)
    s2 = mgr.submit(arrival=0.0, bytes_=bytes_, is_load=True)

    # Second I/O should be slower (queued on shared bandwidth)
    assert s2["total"] > s1["total"], \
        f"DRAM concurrent: s2={s2['total']*1e3:.3f}ms should be > s1={s1['total']*1e3:.3f}ms"

    # Batch duration = max(s1, s2) ≈ 2 × single
    batch_dur = mgr.get_batch_duration()
    single_dur = 6.67e-6 + bytes_ / 13.6e9
    assert batch_dur > single_dur * 1.5, \
        f"DRAM batch={batch_dur*1e3:.3f}ms should be > 1.5x single={single_dur*1e3:.3f}ms"

    print(f"✓ DRAM concurrent: s1={s1['total']*1e3:.3f}ms s2={s2['total']*1e3:.3f}ms "
          f"batch={batch_dur*1e3:.3f}ms")


def test_dram_cross_batch():
    """DRAM cross-batch: unfinished I/O preserved."""
    mgr = SharedDramResourceManager()
    mgr.configure(dram_bw=13.6e9)

    mgr.begin_batch(0.0)
    mgr.submit(arrival=0.0, bytes_=100e6, is_load=True)
    # busy_until = 6.67μs + 100e6/13.6e9 = 7.35ms

    # Batch 2 at T=1ms (before I/O completes)
    mgr.begin_batch(0.001)
    s2 = mgr.submit(arrival=0.001, bytes_=50e6, is_load=True)

    # Should wait for previous I/O
    assert s2["start"] >= 0.007, \
        f"DRAM cross-batch: start={s2['start']:.6f} should be >= 0.007"

    print(f"✓ DRAM cross-batch: start={s2['start']*1e3:.3f}ms (≥7.35ms)")


def test_hbf_dram_fairness():
    """HBF and DRAM both use shared model → fair comparison."""
    hbf_mgr = SharedHbfResourceManager()
    hbf_mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)

    dram_mgr = SharedDramResourceManager()
    dram_mgr.configure(dram_bw=13.6e9)

    # Single I/O: 38.9MB
    bytes_ = 38.9e6

    hbf_mgr.begin_batch(0.0)
    hbf_s = hbf_mgr.submit(0.0, bytes_, is_read=True)
    hbf_single = hbf_s.total

    dram_mgr.begin_batch(0.0)
    dram_s = dram_mgr.submit(0.0, bytes_, is_load=True)
    dram_single = dram_s["total"]  # DRAM returns dict

    # HBF single should be faster (16 GB/s ic > 13.6 GB/s dram)
    assert hbf_single < dram_single, \
        f"HBF single ({hbf_single*1e3:.3f}ms) should be < DRAM single ({dram_single*1e3:.3f}ms)"

    # 2 concurrent
    hbf_mgr.begin_batch(0.0)
    hbf_mgr.submit(0.0, bytes_, is_read=True)
    hbf_mgr.submit(0.0, bytes_, is_read=True)
    hbf_2 = hbf_mgr.get_batch_duration()

    dram_mgr.begin_batch(0.0)
    dram_mgr.submit(0.0, bytes_, is_load=True)
    dram_mgr.submit(0.0, bytes_, is_load=True)
    dram_2 = dram_mgr.get_batch_duration()

    # HBF 2-concurrent should still be faster
    assert hbf_2 < dram_2, \
        f"HBF 2-conc ({hbf_2*1e3:.3f}ms) should be < DRAM 2-conc ({dram_2*1e3:.3f}ms)"

    speedup = dram_2 / hbf_2
    print(f"✓ Fairness: single HBF={hbf_single*1e3:.3f}ms DRAM={dram_single*1e3:.3f}ms | "
          f"2-conc HBF={hbf_2*1e3:.3f}ms DRAM={dram_2*1e3:.3f}ms | "
          f"HBF speedup={speedup:.2f}x")


def test_n_concurrent_scaling():
    """N=1/2/4/8/16/32 concurrent I/Os: batch duration scales linearly."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)

    bytes_ = 38.9e6  # 339 tokens
    single_dur = None
    results = {}

    for n in [1, 2, 4, 8, 16, 32]:
        # Fresh manager for each N to avoid cross-iteration state
        mgr = SharedHbfResourceManager()
        mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)
        mgr.begin_batch(0.0)
        for i in range(n):
            mgr.submit(0.0, bytes_, is_read=True)
        dur = mgr.get_batch_duration()
        results[n] = dur
        if n == 1:
            single_dur = dur

    # N=1 baseline
    assert single_dur is not None and single_dur > 0

    # N concurrent should be ~N × single (interconnect is bottleneck, serial)
    for n, dur in results.items():
        ratio = dur / single_dur
        # Allow 10% tolerance for overhead
        expected = n
        assert abs(ratio - expected) / expected < 0.15, \
            f"N={n}: ratio={ratio:.2f} expected≈{expected} (±15%)"

    print(f"✓ N-scaling: " + " ".join(
        f"N={n}:{results[n]*1e3:.1f}ms({results[n]/single_dur:.1f}x)"
        for n in sorted(results)
    ))


def test_mixed_io_sizes():
    """Different I/O sizes in same batch: larger I/O gets proportionally longer."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)

    mgr.begin_batch(0.0)
    # Small I/O: 10MB, Large I/O: 100MB
    s_small = mgr.submit(0.0, 10e6, is_read=True)
    s_large = mgr.submit(0.0, 100e6, is_read=True)

    # Large I/O should take longer
    assert s_large.total > s_small.total, \
        f"Large ({s_large.total*1e3:.3f}ms) should be > small ({s_small.total*1e3:.3f}ms)"

    # Small I/O's ic_time should be 10/100 of large's ic_time
    ratio = s_small.ic_time / s_large.ic_time
    assert abs(ratio - 0.1) < 0.02, \
        f"Size ratio: {ratio:.3f} should be ≈0.1"

    print(f"✓ Mixed sizes: small={s_small.total*1e3:.3f}ms large={s_large.total*1e3:.3f}ms")


def test_read_write_share_resources():
    """Reads and writes share the same media and interconnect resources."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)

    mgr.begin_batch(0.0)
    # 2 reads + 2 writes
    for i in range(4):
        mgr.submit(0.0, 50e6, is_read=(i < 2))

    summary = mgr.get_stats_summary()
    assert summary["count"] == 4
    assert summary["batch_duration"] > 0

    # With 4 I/Os sharing, batch should be ~4x single
    single = max(50e6 / 51.2e9, 50e6 / 16e9)  # ic dominates
    assert summary["batch_duration"] > single * 3, \
        f"4 mixed I/Os: batch={summary['batch_duration']*1e3:.3f}ms should be > 3x single={single*1e3:.3f}ms"

    print(f"✓ Read/Write sharing: batch={summary['batch_duration']*1e3:.3f}ms, "
          f"4 I/Os (2R+2W)")


def run_all():
    tests = [
        test_cross_batch_preserves_busy_until,
        test_cross_batch_no_oversell,
        test_dram_shared_single_io,
        test_dram_shared_concurrent,
        test_dram_cross_batch,
        test_hbf_dram_fairness,
        test_n_concurrent_scaling,
        test_mixed_io_sizes,
        test_read_write_share_resources,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed ({passed+failed} total)")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
