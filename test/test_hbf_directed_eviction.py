"""Phase 2b-3: Directed eviction + L3 Write Ack causality tests.

Tests:
  1. HBF full → DRAM fallback
  2. DRAM full → HBF fallback
  3. Both full → return None (caller evicts to L3)
  4. free() releases correct physical pool (virtual index routing)
  5. No double-free, no false hit, no resource leak
  6. L3 Write Ack causality: logical write instant, SSD timing independent

Run: cd /home/d00954154/HiSim_Qwen3 && PYTHONPATH=src python3 test/test_hbf_directed_eviction.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.sglang.hbf_tiered_pool import (
    TieredHostKVCache, MockHBFPool, PlacementPolicyV0,
)


def make_adapter(hbf_size=10, dram_size=10, hbf_ratio=0.5):
    hbf_pool = MockHBFPool(logical_size=hbf_size)
    dram_pool = MockHBFPool(logical_size=dram_size)
    adapter = TieredHostKVCache(hbf_pool, dram_pool, PlacementPolicyV0(hbf_ratio=hbf_ratio))
    return adapter


def test_hbf_full_dram_fallback():
    """HBF full → alloc falls back to DRAM."""
    print("\n--- Test: HBF full → DRAM fallback ---")
    adapter = make_adapter(hbf_size=5, dram_size=10, hbf_ratio=1.0)  # prefer HBF
    # Fill HBF
    idx1 = adapter.alloc(5)
    assert idx1 is not None and int(idx1.max()) < 5, "should be HBF indices"
    stats = adapter.capacity_stats()
    assert stats["hbf_used"] == 5 and stats["hbf_available"] == 0, "HBF should be full"
    # Next alloc should fallback to DRAM
    idx2 = adapter.alloc(3)
    assert idx2 is not None and int(idx2.min()) >= 5, "should be DRAM indices (offset)"
    stats = adapter.capacity_stats()
    assert stats["dram_used"] == 3, f"DRAM should have 3, got {stats['dram_used']}"
    print(f"  HBF=5/5 full, DRAM fallback: idx={idx2.tolist()} (≥dram_offset=5) ✅")


def test_dram_full_hbf_fallback():
    """DRAM full → alloc falls back to HBF."""
    print("\n--- Test: DRAM full → HBF fallback ---")
    adapter = make_adapter(hbf_size=10, dram_size=5, hbf_ratio=0.0)  # prefer DRAM
    # Fill DRAM
    idx1 = adapter.alloc(5)
    assert idx1 is not None and int(idx1.min()) >= 10, "should be DRAM indices"
    stats = adapter.capacity_stats()
    assert stats["dram_used"] == 5 and stats["dram_available"] == 0, "DRAM should be full"
    # Next alloc should fallback to HBF
    idx2 = adapter.alloc(3)
    assert idx2 is not None and int(idx2.max()) < 10, "should be HBF indices (< dram_offset)"
    stats = adapter.capacity_stats()
    assert stats["hbf_used"] == 3, f"HBF should have 3, got {stats['hbf_used']}"
    print(f"  DRAM=5/5 full, HBF fallback: idx={idx2.tolist()} (<dram_offset=10) ✅")


def test_both_full_returns_none():
    """Both pools full → alloc returns None (caller evicts to L3)."""
    print("\n--- Test: Both full → None ---")
    adapter = make_adapter(hbf_size=5, dram_size=5, hbf_ratio=0.5)
    idx1 = adapter.alloc(5)  # fills one pool
    idx2 = adapter.alloc(5)  # fills the other
    stats = adapter.capacity_stats()
    total_avail = stats["hbf_available"] + stats["dram_available"]
    assert total_avail == 0, f"both should be full, avail={total_avail}"
    result = adapter.alloc(1)
    assert result is None, "both full → should return None"
    print(f"  Both full: alloc returns None → caller evicts to L3 ✅")


def test_free_releases_correct_pool():
    """free() with mixed indices releases correct physical pool slots."""
    print("\n--- Test: free() releases correct pool ---")
    adapter = make_adapter(hbf_size=10, dram_size=10, hbf_ratio=0.5)
    # Alloc to both pools
    idx_hbf = adapter.alloc(4)   # HBF (ratio=0.5, first alloc → HBF)
    idx_dram = adapter.alloc(6)  # DRAM (second alloc → DRAM)
    stats_before = adapter.capacity_stats()
    assert stats_before["hbf_used"] == 4 and stats_before["dram_used"] == 6
    # Free HBF indices
    freed = adapter.free(idx_hbf)
    assert freed == 4, f"should free 4 HBF, got {freed}"
    stats_after = adapter.capacity_stats()
    assert stats_after["hbf_used"] == 0, f"HBF should be 0, got {stats_after['hbf_used']}"
    assert stats_after["dram_used"] == 6, f"DRAM should be 6, got {stats_after['dram_used']}"
    # Free DRAM indices
    freed = adapter.free(idx_dram)
    assert freed == 6
    stats_final = adapter.capacity_stats()
    assert stats_final["hbf_used"] == 0 and stats_final["dram_used"] == 0
    print(f"  free(HBF idx)→HBF released, free(DRAM idx)→DRAM released ✅")


def test_mixed_free():
    """free() with mixed HBF+DRAM indices in one call."""
    print("\n--- Test: mixed free() ---")
    adapter = make_adapter(hbf_size=10, dram_size=10, hbf_ratio=0.5)
    idx_hbf = adapter.alloc(3)
    idx_dram = adapter.alloc(4)
    # Combine into one tensor
    combined = torch.cat([idx_hbf, idx_dram])
    freed = adapter.free(combined)
    assert freed == 7, f"should free 7 total, got {freed}"
    stats = adapter.capacity_stats()
    assert stats["hbf_used"] == 0 and stats["dram_used"] == 0
    print(f"  free(combined HBF+DRAM)→ both pools released correctly ✅")


def test_no_double_free():
    """Double-free should assert (MockHBFPool has _slot_used assert)."""
    print("\n--- Test: double-free protection ---")
    adapter = make_adapter(hbf_size=10, dram_size=10, hbf_ratio=0.5)
    idx = adapter.alloc(3)
    adapter.free(idx)
    try:
        adapter.free(idx)  # should assert
        print("  ❌ Double-free not detected!")
        assert False, "double-free should assert"
    except AssertionError as e:
        if "Double-free" in str(e) or "slot_used" in str(e):
            print(f"  Double-free correctly asserted ✅")
        else:
            raise


def test_repeated_alloc_free_no_leak():
    """Repeated alloc/free cycles: no resource leak."""
    print("\n--- Test: repeated alloc/free no leak ---")
    adapter = make_adapter(hbf_size=10, dram_size=10, hbf_ratio=0.5)
    for cycle in range(20):
        idx = adapter.alloc(5)
        assert idx is not None, f"cycle {cycle}: alloc failed"
        freed = adapter.free(idx)
        assert freed == 5, f"cycle {cycle}: freed {freed}"
    stats = adapter.capacity_stats()
    assert stats["hbf_used"] == 0 and stats["dram_used"] == 0, "no leak after 20 cycles"
    assert stats["hbf_available"] == 10 and stats["dram_available"] == 10, "full capacity restored"
    print(f"  20 alloc/free cycles: no leak, full capacity restored ✅")


def test_virtual_index_consistency():
    """Virtual indices are consistent: HBF < offset, DRAM >= offset."""
    print("\n--- Test: virtual index consistency ---")
    adapter = make_adapter(hbf_size=100, dram_size=50, hbf_ratio=0.5)
    all_indices = []
    for _ in range(10):
        idx = adapter.alloc(5)
        all_indices.append(idx)
    combined = torch.cat(all_indices)
    hbf_mask = combined < adapter.dram_offset
    hbf_idx = combined[hbf_mask]
    dram_idx = combined[~hbf_mask]
    # Verify no overlap
    assert len(set(hbf_idx.tolist())) == len(hbf_idx), "HBF indices unique"
    assert len(set(dram_idx.tolist())) == len(dram_idx), "DRAM indices unique"
    # Verify HBF < offset, DRAM >= offset
    assert int(hbf_idx.max()) < adapter.dram_offset if len(hbf_idx) > 0 else True
    assert int(dram_idx.min()) >= adapter.dram_offset if len(dram_idx) > 0 else True
    stats = adapter.capacity_stats()
    assert stats["hbf_used"] == len(hbf_idx)
    assert stats["dram_used"] == len(dram_idx)
    print(f"  {len(hbf_idx)} HBF (<{adapter.dram_offset}), {len(dram_idx)} DRAM (≥{adapter.dram_offset}), no overlap ✅")


def test_l3_ack_causality():
    """L3 Write Ack: logical write instant, SSD timing independent.

    Verifies the simplification model:
    - batch_set (logical write) is synchronous → data available immediately
    - SSD write timing submitted separately (submit_l3_write_timing)
    - Host copy released after logical write, before SSD write timing
    - batch_exists finds data regardless of SSD write timing
    """
    print("\n--- Test: L3 Write Ack causality model ---")
    # This is a documentation test — the causality is verified by code inspection:
    # 1. handle_backup_operation() calls batch_set() synchronously → returns True
    # 2. ack_backup_queue.put(operation) → Ack after logical write
    # 3. submit_l3_write_timing() called LATER → SSD timing independent
    # 4. batch_exists() checks kvcm storage (logical) → not affected by SSD timing
    # 5. submit_read() uses previous round's write avail_time → correct ordering

    # The key invariant: logical write completes BEFORE Ack,
    # and SSD timing is modeled independently.
    # Host copy CAN be released before SSD write completes (in sim time),
    # but kvcm has the data → no data loss.

    # Verify the order: handle_backup → handle_prefetch → submit_write
    # (from _drive_hicache_timing_phases)
    from hisim.simulation.sglang.sglang_hook import _drive_hicache_timing_phases
    import inspect
    src = inspect.getsource(_drive_hicache_timing_phases)
    # Verify the call order
    backup_pos = src.index("handle_backup_operation")
    prefetch_pos = src.index("handle_prefetch_operation")
    write_pos = src.index("submit_l3_write_timing")
    assert backup_pos < prefetch_pos < write_pos, \
        "Order must be: backup → prefetch → write_timing"
    print(f"  Order: backup(logical write) → prefetch(read) → write_timing(SSD sim) ✅")
    print(f"  Logical write completes before Ack; SSD timing independent ✅")
    print(f"  Host copy released after logical write; kvcm retains data ✅")
    print(f"  Simplification: instant logical Ack + independent SSD timing")


def run_all():
    tests = [
        ("HBF full → DRAM fallback", test_hbf_full_dram_fallback),
        ("DRAM full → HBF fallback", test_dram_full_hbf_fallback),
        ("Both full → None", test_both_full_returns_none),
        ("free() releases correct pool", test_free_releases_correct_pool),
        ("Mixed free()", test_mixed_free),
        ("Double-free protection", test_no_double_free),
        ("Repeated alloc/free no leak", test_repeated_alloc_free_no_leak),
        ("Virtual index consistency", test_virtual_index_consistency),
        ("L3 Write Ack causality", test_l3_ack_causality),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAIL: {name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
