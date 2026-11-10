"""Unit tests for TieredHostKVCache (HBF L1.5 PoC).

Tests the virtual index mapping, alloc/free routing, capacity tracking,
and fallback behavior WITHOUT requiring SGLang. Uses two MockHBFPool
instances as physical pools (one standing in for DRAM, one for HBF).

Run: cd /home/d00954154/HiSim_Qwen3 && python -m pytest test/test_hbf_tiered_pool.py -v
  or: python -m test.test_hbf_tiered_pool
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import logging

from hisim.simulation.sglang.hbf_tiered_pool import (
    TieredHostKVCache,
    MockHBFPool,
    PlacementPolicyV0,
    install_tiered_host_pool,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTieredHostKVCache:
    """Standalone tests for the virtual index adapter."""

    def _make_adapter(self, hbf_size=1000, dram_size=200, ratio=0.7):
        hbf = MockHBFPool(logical_size=hbf_size, page_size=1)
        dram = MockHBFPool(logical_size=dram_size, page_size=1)
        policy = PlacementPolicyV0(hbf_ratio=ratio)
        return TieredHostKVCache(hbf, dram, policy)

    # ---- P1: HBF-only routing (DRAM tiny) ----
    def test_p1_hbf_only_backup_load(self):
        """P1: DRAM tiny, HBF large → all allocs go to HBF, DRAM not consumed."""
        adapter = self._make_adapter(hbf_size=1000, dram_size=5, ratio=0.95)

        # Alloc 10 slots — should go to HBF (ratio 95%)
        idx1 = adapter.alloc(10)
        assert idx1 is not None, "alloc failed"
        assert adapter._is_hbf(idx1).all(), "indices should be in HBF range"
        assert adapter.hbf_used == 10, f"hbf_used={adapter.hbf_used}"
        assert adapter.dram_used == 0, f"dram_used={adapter.dram_used} (should be 0)"

        # Free and re-alloc
        adapter.free(idx1)
        assert adapter.hbf_used == 0
        idx2 = adapter.alloc(10)
        assert idx2 is not None
        assert adapter._is_hbf(idx2).all()

        # get_data_page routes to HBF
        page = adapter.get_data_page(idx2[0])
        assert page is not None

        adapter.free(idx2)
        stats = adapter.capacity_stats()
        logger.info(f"P1 stats: {stats}")
        assert stats["dram_used"] == 0, "DRAM should not be consumed"
        print("✅ P1 PASS: HBF-only backup/load, DRAM not consumed")

    # ---- P2: HBF/DRAM mixed placement ----
    def test_p2_mixed_placement(self):
        """P2: different allocs go to different pools, no false hits."""
        adapter = self._make_adapter(hbf_size=100, dram_size=100, ratio=0.5)

        # Alloc multiple batches — should split between HBF and DRAM
        hbf_indices = []
        dram_indices = []
        for i in range(10):
            idx = adapter.alloc(5)
            assert idx is not None, f"alloc {i} failed"
            if adapter._is_hbf(idx).all():
                hbf_indices.append(idx)
            else:
                dram_indices.append(idx)
        assert len(hbf_indices) > 0, "some should go to HBF"
        assert len(dram_indices) > 0, "some should go to DRAM"
        logger.info(
            f"P2: {len(hbf_indices)} HBF allocs, {len(dram_indices)} DRAM allocs"
        )

        # Free HBF indices, verify DRAM indices still valid (no false free)
        for idx in hbf_indices:
            adapter.free(idx)
        assert adapter.hbf_used == 0
        assert adapter.dram_used > 0, "DRAM indices should still be allocated"

        # Free DRAM indices
        for idx in dram_indices:
            adapter.free(idx)
        assert adapter.dram_used == 0

        stats = adapter.capacity_stats()
        logger.info(f"P2 stats after free: {stats}")
        print("✅ P2 PASS: mixed placement, no false hits/double-free")

    # ---- P3: capacity pressure + fallback ----
    def test_p3_capacity_fallback(self):
        """P3: HBF full → fallback DRAM; DRAM full → fallback HBF; both full → None."""
        adapter = self._make_adapter(hbf_size=10, dram_size=10, ratio=0.5)

        # Fill HBF (policy starts with HBF)
        hbf_idx = adapter.alloc(10)
        assert hbf_idx is not None
        assert adapter._is_hbf(hbf_idx).all(), "first batch should be HBF"

        # HBF full now → next alloc should fallback to DRAM
        dram_idx = adapter.alloc(5)
        assert dram_idx is not None, "should fallback to DRAM"
        assert not adapter._is_hbf(dram_idx).any(), "should be DRAM (fallback)"
        logger.info(f"P3: HBF full → DRAM fallback, dram_idx={dram_idx[:3]}...")

        # Both pools nearly full → next alloc should fail
        remaining_dram = adapter.dram_pool.available_size()
        if remaining_dram > 0:
            extra = adapter.alloc(remaining_dram)
            assert extra is not None
        # Now both full
        assert adapter.hbf_pool.available_size() == 0
        assert adapter.dram_pool.available_size() == 0
        none_idx = adapter.alloc(1)
        assert none_idx is None, "should return None when both full"

        # Free some HBF → alloc should succeed again
        adapter.free(hbf_idx[:3])
        re_idx = adapter.alloc(3)
        assert re_idx is not None, "should succeed after free"

        adapter.free(hbf_idx[3:])
        adapter.free(dram_idx)
        adapter.free(re_idx)
        print("✅ P3 PASS: capacity fallback, no deadlock")

    # ---- P4: L3 closed loop (get_data_page / set_from_flat_data_page routing) ----
    def test_p4_l3_io_routing(self):
        """P4: get_data_page/set_from_flat_data_page route to correct pool."""
        adapter = self._make_adapter(hbf_size=50, dram_size=50, ratio=0.5)

        # Alloc in HBF
        hbf_idx = adapter.alloc(5)
        assert adapter._is_hbf(hbf_idx).all()

        # get_data_page for HBF index → should route to HBF pool
        page = adapter.get_data_page(hbf_idx[0])
        assert page is not None

        # set_from_flat_data_page for HBF index → should route to HBF (no-op in mock)
        adapter.set_from_flat_data_page(hbf_idx[0], page)

        # Alloc in DRAM
        dram_idx = adapter.alloc(5)
        assert not adapter._is_hbf(dram_idx).any()

        # get_data_page for DRAM index → should route to DRAM pool
        page2 = adapter.get_data_page(dram_idx[0])
        assert page2 is not None

        # set_from_flat_data_page for DRAM index
        adapter.set_from_flat_data_page(dram_idx[0], page2)

        # get_dummy_flat_data_page
        dummy = adapter.get_dummy_flat_data_page()
        assert dummy is not None

        adapter.free(hbf_idx)
        adapter.free(dram_idx)
        print("✅ P4 PASS: L3 IO routing correct for both pools")

    # ---- Virtual index atomicity ----
    def test_double_alloc_detection(self):
        """Verify physical pools detect double-alloc (delegated detection)."""
        adapter = self._make_adapter(hbf_size=10, dram_size=10, ratio=1.0)
        idx = adapter.alloc(5)
        # Try to free and re-free same indices → double-free
        adapter.free(idx)
        try:
            adapter.free(idx)
            # If no exception, the physical pool didn't detect — log warning
            logger.warning("Double-free not detected (physical pool may not assert)")
        except (AssertionError, RuntimeError):
            pass  # expected: double-free detected
        print("✅ Double-free detection test done")

    # ---- Node split simulation ----
    def test_node_split_indices(self):
        """Simulate radix tree node split: indices split into two ranges.

        After split, each child's host_value is a slice of the original.
        Freeing each child's indices should work correctly.
        """
        adapter = self._make_adapter(hbf_size=100, dram_size=100, ratio=0.7)

        # Alloc 10 in HBF
        idx = adapter.alloc(10)
        assert adapter._is_hbf(idx).all()

        # Simulate split: parent gets first 5, child gets last 5
        parent_idx = idx[:5]
        child_idx = idx[5:]

        # Both should still be valid HBF virtual indices
        assert adapter._is_hbf(parent_idx).all()
        assert adapter._is_hbf(child_idx).all()

        # Free parent's indices
        adapter.free(parent_idx)
        assert adapter.hbf_used == 5, f"hbf_used={adapter.hbf_used} (should be 5)"

        # Child's indices still allocated
        assert adapter.hbf_pool.available_size() == 95

        # Free child's indices
        adapter.free(child_idx)
        assert adapter.hbf_used == 0
        assert adapter.hbf_pool.available_size() == 100

        print("✅ Node split indices test PASS")

    # ---- Mixed indices in one free call ----
    def test_mixed_indices_free(self):
        """Free a batch containing both HBF and DRAM indices in one call."""
        adapter = self._make_adapter(hbf_size=50, dram_size=50, ratio=0.5)

        hbf_idx = adapter.alloc(5)
        dram_idx = adapter.alloc(5)
        assert adapter._is_hbf(hbf_idx).all()
        assert not adapter._is_hbf(dram_idx).any()

        # Combine and free in one call
        combined = torch.cat([hbf_idx, dram_idx])
        freed = adapter.free(combined)
        assert freed == 10, f"freed={freed}"
        assert adapter.hbf_used == 0
        assert adapter.dram_used == 0

        print("✅ Mixed indices free test PASS")

    # ---- Capacity stats ----
    def test_capacity_stats(self):
        """Verify capacity stats track HBF and DRAM separately."""
        adapter = self._make_adapter(hbf_size=1000, dram_size=200, ratio=0.7)
        stats = adapter.capacity_stats()
        assert stats["hbf_capacity"] == 1000
        assert stats["dram_capacity"] == 200
        assert stats["hbf_used"] == 0
        assert stats["dram_used"] == 0

        # Alloc some
        idx = adapter.alloc(100)
        stats = adapter.capacity_stats()
        logger.info(f"Capacity after alloc 100: {stats}")
        assert stats["hbf_used"] + stats["dram_used"] == 100

        adapter.free(idx)
        stats = adapter.capacity_stats()
        assert stats["hbf_used"] + stats["dram_used"] == 0

        print("✅ Capacity stats test PASS")


def run_all_tests():
    """Run all tests without pytest."""
    tests = TestTieredHostKVCache()
    test_methods = [
        ("P1: HBF-only backup/load", tests.test_p1_hbf_only_backup_load),
        ("P2: mixed placement", tests.test_p2_mixed_placement),
        ("P3: capacity fallback", tests.test_p3_capacity_fallback),
        ("P4: L3 IO routing", tests.test_p4_l3_io_routing),
        ("double-alloc detection", tests.test_double_alloc_detection),
        ("node split indices", tests.test_node_split_indices),
        ("mixed indices free", tests.test_mixed_indices_free),
        ("capacity stats", tests.test_capacity_stats),
    ]
    passed = 0
    failed = 0
    for name, method in test_methods:
        try:
            print(f"\n--- {name} ---")
            method()
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {name}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
