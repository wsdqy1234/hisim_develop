#!/usr/bin/env python3
"""Phase 3E final regression: Shared Downstream cross-batch competition.

Verifies that the downstream busy cursor is NOT lost on batch switch,
matching the HBF/DRAM manager cross-batch safety semantics.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_shared_resource import (
    SharedHbfResourceManager,
    SharedDramResourceManager,
    configure_shared_downstream,
    reset_downstream,
    get_downstream_busy_until,
    update_downstream_busy_until,
    is_shared_downstream_enabled,
)


def test_downstream_cross_batch_preserves_busy():
    """Batch 2 must see Batch 1's downstream occupation."""
    configure_shared_downstream(13.6e9)

    # Batch 1 at T=0: HBF read 38.9MB
    hbf = SharedHbfResourceManager()
    hbf.configure(media_bw=51.2e9, interconnect_bw=51.2e9, queue_depth=128)
    hbf.begin_batch(0.0)
    s1 = hbf.submit(0.0, 38.9e6, is_read=True)

    # I/O goes through shared downstream: 38.9e6 / 13.6e9 = 2.864ms
    # downstream busy_until = 2.864ms
    busy = get_downstream_busy_until()
    assert busy > 0.002, f"Downstream busy={busy:.6f} should be > 2ms"

    # Batch 2 at T=1ms (before I/O completes)
    reset_downstream(0.001)  # must NOT reduce busy below 2.864ms
    busy2 = get_downstream_busy_until()
    assert busy2 >= busy, f"Downstream busy lost: {busy2:.6f} < {busy:.6f}"

    # Batch 2's DRAM I/O must wait for Batch 1's downstream
    dram = SharedDramResourceManager()
    dram.configure(dram_bw=13.6e9)
    dram.begin_batch(0.001)
    s2 = dram.submit(0.001, 38.9e6, is_load=True)

    assert s2["start"] >= busy, \
        f"Batch 2 DRAM start={s2['start']:.6f} should be >= downstream busy={busy:.6f}"

    print(f"✓ DS cross-batch: Batch2 start={s2['start']*1e3:.3f}ms >= busy={busy*1e3:.3f}ms")


def test_downstream_reset_at_batch_start():
    """Reset downstream cursor at batch start (when previous I/O completed)."""
    configure_shared_downstream(13.6e9)
    # Clear global downstream state for clean test
    import hisim.simulation.hbf_shared_resource as mod
    mod._downstream_busy_until = 0.0

    # Batch 1 completes at T=2.864ms
    hbf = SharedHbfResourceManager()
    hbf.configure(media_bw=51.2e9, interconnect_bw=51.2e9, queue_depth=128)
    hbf.begin_batch(0.0)
    hbf.submit(0.0, 38.9e6, is_read=True)
    busy1 = get_downstream_busy_until()

    # Clock advances to 5ms (after I/O completes)
    reset_downstream(0.005)
    busy2 = get_downstream_busy_until()
    assert busy2 == 0.005, f"Downstream should reset to 5ms, got {busy2}"

    print(f"✓ DS reset: busy after completed I/O = {busy2*1e3:.1f}ms")


def test_downstream_hbf_dram_compete():
    """HBF and DRAM I/Os in same batch share the downstream."""
    configure_shared_downstream(13.6e9)
    # Clear global downstream state for clean test
    import hisim.simulation.hbf_shared_resource as mod
    mod._downstream_busy_until = 0.0

    hbf = SharedHbfResourceManager()
    hbf.configure(media_bw=51.2e9, interconnect_bw=51.2e9, queue_depth=128)
    hbf.begin_batch(0.0)

    dram = SharedDramResourceManager()
    dram.configure(dram_bw=13.6e9)
    dram.begin_batch(0.0)

    # HBF I/O first: 38.9MB at 13.6 GB/s = 2.864ms
    hbf_s = hbf.submit(0.0, 38.9e6, is_read=True)

    # DRAM I/O second: must wait for HBF's downstream usage
    dram_s = dram.submit(0.0, 38.9e6, is_load=True)

    # DRAM start should be after HBF's ic_completion
    assert dram_s["start"] >= hbf_s.ic_completion, \
        f"DRAM start={dram_s['start']:.6f} should be >= HBF ic_completion={hbf_s.ic_completion:.6f}"

    # Total batch time should be ~2x single I/O (serialized on downstream)
    total = max(hbf_s.ic_completion, dram_s["completion"])
    single = 38.9e6 / 13.6e9
    assert total > single * 1.5, \
        f"Total={total*1e3:.3f}ms should be > 1.5x single={single*1e3:.3f}ms"

    print(f"✓ HBF+DRAM compete: HBF ic={hbf_s.ic_completion*1e3:.3f}ms "
          f"DRAM start={dram_s['start']*1e3:.3f}ms total={total*1e3:.3f}ms")


def test_downstream_disabled_by_default():
    """Default (unset) = Independent topology."""
    # Reset the global state
    import hisim.simulation.hbf_shared_resource as mod
    mod._shared_downstream_bw = 0
    mod._downstream_busy_until = 0

    assert not is_shared_downstream_enabled(), "Default should be disabled"

    # HBF and DRAM should use independent resources
    hbf = SharedHbfResourceManager()
    hbf.configure(media_bw=51.2e9, interconnect_bw=16e9, queue_depth=128)
    hbf.begin_batch(0.0)
    hbf_s = hbf.submit(0.0, 38.9e6, is_read=True)

    dram = SharedDramResourceManager()
    dram.configure(dram_bw=13.6e9)
    dram.begin_batch(0.0)
    dram_s = dram.submit(0.0, 38.9e6, is_load=True)

    # With independent topology, DRAM should NOT wait for HBF
    assert dram_s["start"] == 0.0, \
        f"Independent: DRAM start={dram_s['start']} should be 0 (no wait)"

    print(f"✓ Default independent: DRAM start=0 (no HBF wait)")


def run_all():
    tests = [
        test_downstream_cross_batch_preserves_busy,
        test_downstream_reset_at_batch_start,
        test_downstream_hbf_dram_compete,
        test_downstream_disabled_by_default,
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

    # Cleanup: disable shared downstream after tests
    import hisim.simulation.hbf_shared_resource as mod
    mod._shared_downstream_bw = 0
    mod._downstream_busy_until = 0

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed ({passed+failed} total)")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
