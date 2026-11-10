#!/usr/bin/env python3
"""Phase 3D-P1: Unit tests for SharedHbfResourceManager.

Tests invariants:
- Concurrency=1 matches independent model (max(media, ic))
- Aggregate throughput ≤ min(media_bw, ic_bw)
- Queue wait, completion time ≥ 0
- Read/Write share resources
- I/O not consumed before completion
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_shared_resource import (
    SharedHbfResourceManager,
    get_shared_hbf_manager,
    HbfIoStats,
)


def test_single_io_matches_independent_model():
    """Concurrency=1: result should match max(media_time, ic_time)."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)
    mgr.begin_batch(0.0)

    bytes_ = 233e6  # 233MB (4K tokens)
    stats = mgr.submit(arrival=0.0, bytes_=bytes_, is_read=True)

    expected_media = bytes_ / 51.2e9  # 4.557ms
    expected_ic = bytes_ / 16e9       # 14.563ms
    expected_total = max(expected_media, expected_ic)

    assert abs(stats.total - expected_total) < 1e-6, \
        f"Single I/O: total={stats.total:.6f}, expected={expected_total:.6f}"
    assert stats.queue_wait == 0.0, f"Single I/O: queue_wait={stats.queue_wait}"
    print(f"✓ Single I/O: total={stats.total*1e3:.3f}ms "
          f"(media={stats.media_time*1e3:.3f}ms ic={stats.ic_time*1e3:.3f}ms)")


def test_concurrent_ios_share_bandwidth():
    """4 concurrent I/Os: aggregate throughput ≤ min(media_bw, ic_bw)."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)
    mgr.begin_batch(0.0)

    bytes_per_io = 233e6  # 233MB each
    n = 4
    stats_list = []
    for i in range(n):
        s = mgr.submit(arrival=0.0, bytes_=bytes_per_io, is_read=True)
        stats_list.append(s)

    batch_dur = mgr.get_batch_duration()
    total_bytes = n * bytes_per_io
    effective_bw = total_bytes / batch_dur if batch_dur > 0 else 0

    # Aggregate throughput ≤ min(media_bw, ic_bw) = 16 GB/s
    assert effective_bw <= 16e9 * 1.01, \
        f"Aggregate BW {effective_bw/1e9:.1f} GB/s exceeds limit 16 GB/s"

    # Each I/O should have total > 0
    for s in stats_list:
        assert s.total > 0, f"I/O total={s.total}"
        assert s.queue_wait >= 0, f"Queue wait negative: {s.queue_wait}"

    print(f"✓ 4 concurrent: batch={batch_dur*1e3:.3f}ms "
          f"eff_bw={effective_bw/1e9:.1f}GB/s "
          f"max_wait={max(s.queue_wait for s in stats_list)*1e3:.3f}ms")


def test_completion_times_non_negative():
    """All completion times and queue waits must be ≥ 0."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=8e9, queue_depth=4)
    mgr.begin_batch(0.0)

    for i in range(16):
        s = mgr.submit(arrival=0.0, bytes_=27e6, is_read=(i % 2 == 0))
        assert s.total >= 0, f"I/O {i}: total={s.total}"
        assert s.queue_wait >= 0, f"I/O {i}: queue_wait={s.queue_wait}"
        assert s.media_completion >= 0, f"I/O {i}: media_completion={s.media_completion}"
        assert s.ic_completion >= 0, f"I/O {i}: ic_completion={s.ic_completion}"

    print(f"✓ 16 I/Os: all non-negative, peak_outstanding={mgr.peak_outstanding}")


def test_read_write_share_resources():
    """Reads and writes share the same media and interconnect."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)
    mgr.begin_batch(0.0)

    # 2 reads + 2 writes, all concurrent
    for i in range(4):
        is_read = (i < 2)
        mgr.submit(arrival=0.0, bytes_=100e6, is_read=is_read)

    summary = mgr.get_stats_summary()
    assert summary["reads"] == 2, f"Expected 2 reads, got {summary['reads']}"
    assert summary["writes"] == 2, f"Expected 2 writes, got {summary['writes']}"
    assert summary["count"] == 4

    # All 4 I/Os share the same resources → batch_dur > single I/O dur
    single_io_dur = max(100e6 / 51.2e9, 100e6 / 16e9)
    assert summary["batch_duration"] > single_io_dur * 1.5, \
        f"Batch dur {summary['batch_duration']*1e3:.3f}ms should be > 1.5x single {single_io_dur*1e3:.3f}ms"

    print(f"✓ Read/Write sharing: batch={summary['batch_duration']*1e3:.3f}ms "
          f"reads={summary['reads']} writes={summary['writes']}")


def test_queue_depth_limits_outstanding():
    """Queue depth limits concurrent outstanding I/Os."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=2)
    mgr.begin_batch(0.0)

    # Submit 8 I/Os with QD=2
    for i in range(8):
        mgr.submit(arrival=0.0, bytes_=50e6, is_read=True)

    assert mgr.peak_outstanding <= 2, \
        f"Peak outstanding {mgr.peak_outstanding} exceeds QD=2"

    print(f"✓ QD=2: peak_outstanding={mgr.peak_outstanding}, "
          f"batch={mgr.get_batch_duration()*1e3:.3f}ms")


def test_interconnect_bottleneck():
    """When ic_bw < media_bw, interconnect is the bottleneck."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=8e9, queue_depth=128)
    mgr.begin_batch(0.0)

    # Single I/O
    bytes_ = 233e6
    s = mgr.submit(arrival=0.0, bytes_=bytes_, is_read=True)

    # With ic_bw=8, ic_time dominates
    ic_time = bytes_ / 8e9  # 29.1ms
    media_time = bytes_ / 51.2e9  # 4.55ms
    assert s.total >= ic_time * 0.99, \
        f"Ic bottleneck: total={s.total*1e3:.3f}ms should ≈ ic_time={ic_time*1e3:.3f}ms"
    assert s.ic_time > s.media_time, \
        f"Ic should be slower: ic={s.ic_time*1e3:.3f}ms media={s.media_time*1e3:.3f}ms"

    print(f"✓ IC bottleneck: total={s.total*1e3:.3f}ms "
          f"(ic={s.ic_time*1e3:.3f}ms > media={s.media_time*1e3:.3f}ms)")


def test_media_bottleneck():
    """When media_bw < ic_bw, media is the bottleneck."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=64e9, queue_depth=128)
    mgr.begin_batch(0.0)

    bytes_ = 233e6
    s = mgr.submit(arrival=0.0, bytes_=bytes_, is_read=True)

    media_time = bytes_ / 51.2e9  # 4.55ms
    ic_time = bytes_ / 64e9       # 3.64ms
    assert s.total >= media_time * 0.99, \
        f"Media bottleneck: total={s.total*1e3:.3f}ms should ≈ media={media_time*1e3:.3f}ms"

    print(f"✓ Media bottleneck: total={s.total*1e3:.3f}ms "
          f"(media={s.media_time*1e3:.3f}ms > ic={s.ic_time*1e3:.3f}ms)")


def test_later_io_has_longer_duration():
    """With concurrent I/Os, later I/Os should have longer durations (queue wait)."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)
    mgr.begin_batch(0.0)

    bytes_ = 100e6
    stats_list = []
    for i in range(4):
        s = mgr.submit(arrival=0.0, bytes_=bytes_, is_read=True)
        stats_list.append(s)

    # First I/O should have shortest duration (no queue wait)
    # Last I/O should have longest duration (most queue wait)
    assert stats_list[0].total <= stats_list[-1].total, \
        f"First I/O ({stats_list[0].total*1e3:.3f}ms) should be ≤ last ({stats_list[-1].total*1e3:.3f}ms)"
    assert stats_list[0].queue_wait == 0.0, \
        f"First I/O queue_wait should be 0, got {stats_list[0].queue_wait}"

    print(f"✓ Ordering: first={stats_list[0].total*1e3:.3f}ms "
          f"last={stats_list[-1].total*1e3:.3f}ms")


def test_batch_duration_equals_max_individual():
    """Batch duration = max(individual durations) = max_completion - batch_start."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)
    mgr.begin_batch(0.0)

    for i in range(4):
        mgr.submit(arrival=0.0, bytes_=100e6, is_read=True)

    batch_dur = mgr.get_batch_duration()
    max_individual = max(s.total for s in mgr.io_stats)

    assert abs(batch_dur - max_individual) < 1e-9, \
        f"Batch dur {batch_dur:.9f} != max individual {max_individual:.9f}"

    print(f"✓ Batch dur = max individual: {batch_dur*1e3:.3f}ms")


def test_reset_clears_state():
    """Reset clears all state for a new batch."""
    mgr = SharedHbfResourceManager()
    mgr.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)
    mgr.begin_batch(0.0)

    mgr.submit(arrival=0.0, bytes_=100e6, is_read=True)
    assert len(mgr.io_stats) == 1

    mgr.begin_batch(1.0)  # new batch at t=1.0
    assert len(mgr.io_stats) == 0
    assert mgr.outstanding == 0
    assert mgr.media_busy_until == 1.0  # reset to batch_start

    print("✓ Reset clears state")


def run_all():
    tests = [
        test_single_io_matches_independent_model,
        test_concurrent_ios_share_bandwidth,
        test_completion_times_non_negative,
        test_read_write_share_resources,
        test_queue_depth_limits_outstanding,
        test_interconnect_bottleneck,
        test_media_bottleneck,
        test_later_io_has_longer_duration,
        test_batch_duration_equals_max_individual,
        test_reset_clears_state,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: FAIL: {e}")
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed ({passed+failed} total)")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    import pytest
    sys.exit(0 if run_all() else 1)
