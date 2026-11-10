"""Phase 2d: L3 Read/Write Restore Loop tests.

Tests the Metadata-only Restore Mock and L3 read pipeline:
  1. batch_get returns data for existing keys (Metadata-only Restore Mock)
  2. batch_get returns None for non-existing keys
  3. batch_get_v1 / batch_get_v2 return correct results
  4. L3 read metrics are tracked (exists / issued / completed / restored / consumed)
  5. StateManager L3 load dur tracking (max semantics)

Run: cd /home/d00954154/HiSim_Qwen3 && PYTHONPATH=src python3 test/test_l3_restore_loop.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.sglang.sglang_mock_class import MockHiCacheStorage
from hisim.simulation.types import RequestStats
from hisim.simulation.manager.state import StateManager


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def run(self, name, func):
        try:
            func()
            print(f"  ✓ {name}")
            self.passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            self.failed += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            self.failed += 1

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*55}")
        print(f"Results: {self.passed} passed, {self.failed} failed ({total} total)")
        print(f"{'='*55}")
        return self.failed == 0


def reset_storage():
    """Reset MockHiCacheStorage class-level counters and create fresh instance."""
    MockHiCacheStorage._DEBUG_BATCH_SET_COUNT = 0
    MockHiCacheStorage._DEBUG_BATCH_EXISTS_COUNT = 0
    MockHiCacheStorage._DEBUG_BATCH_EXISTS_HITS = 0
    MockHiCacheStorage._DEBUG_BATCH_GET_COUNT = 0
    MockHiCacheStorage._DEBUG_BATCH_GET_HITS = 0
    StateManager.reset()
    return MockHiCacheStorage.__new__(MockHiCacheStorage)


def test_batch_get_returns_data_for_existing_keys():
    """batch_get should return dummy pages for keys that exist in storage."""
    storage = reset_storage()
    storage.storage = set()
    storage.persist_storage = False
    # Add some keys
    storage.storage.add("key1")
    storage.storage.add("key2")
    storage.storage.add("key3")
    # batch_get with dummy target_locations
    dummy = ["page1", "page2", "page3"]
    result = storage.batch_get(["key1", "key2", "key3"], target_locations=dummy)
    assert result is not None, "batch_get returned None"
    assert len(result) == 3, f"Expected 3 results, got {len(result)}"
    assert result[0] == "page1", f"Expected page1, got {result[0]}"
    assert result[1] == "page2", f"Expected page2, got {result[1]}"
    assert result[2] == "page3", f"Expected page3, got {result[2]}"
    assert MockHiCacheStorage._DEBUG_BATCH_GET_COUNT == 1
    assert MockHiCacheStorage._DEBUG_BATCH_GET_HITS == 3


def test_batch_get_returns_none_for_missing_keys():
    """batch_get should return None for keys that don't exist."""
    storage = reset_storage()
    storage.storage = set()
    storage.persist_storage = False
    storage.storage.add("key1")
    # key2 doesn't exist
    dummy = ["page1", "page2"]
    result = storage.batch_get(["key1", "key2"], target_locations=dummy)
    assert result is not None, "batch_get returned None"
    assert result[0] == "page1", f"Expected page1 for existing key, got {result[0]}"
    assert result[1] is None, f"Expected None for missing key, got {result[1]}"
    assert MockHiCacheStorage._DEBUG_BATCH_GET_HITS == 1


def test_batch_get_without_target_locations():
    """batch_get should work without target_locations (fallback)."""
    storage = reset_storage()
    storage.storage = set()
    storage.persist_storage = False
    storage.storage.add("key1")
    result = storage.batch_get(["key1"])
    assert result is not None, "batch_get returned None"
    assert len(result) == 1
    assert result[0] is not None, "Expected non-None for existing key"


def test_batch_get_v1_returns_bool_list():
    """batch_get_v1 should return True for existing keys, False otherwise."""
    storage = reset_storage()
    storage.storage = set()
    storage.persist_storage = False
    storage.storage.add("key1")
    storage.storage.add("key3")
    import torch
    host_indices = torch.tensor([0, 1, 2])
    result = storage.batch_get_v1(["key1", "key2", "key3"], host_indices)
    assert result == [True, False, True], f"Expected [True, False, True], got {result}"


def test_batch_get_v2_returns_dict():
    """batch_get_v2 should return per-transfer hit lists."""
    storage = reset_storage()
    storage.storage = set()
    storage.persist_storage = False
    storage.storage.add("k1")
    storage.storage.add("k2")

    class FakeTransfer:
        def __init__(self, name, keys):
            self.name = name
            self.keys = keys
            self.host_indices = None

    transfers = [FakeTransfer("KV", ["k1", "k2", "k3"])]
    result = storage.batch_get_v2(transfers)
    assert "KV" in result, f"Expected 'KV' key, got {list(result.keys())}"
    assert result["KV"] == [True, True, False], f"Expected [T,T,F], got {result['KV']}"


def test_get_returns_data_for_existing():
    """get() should return non-None for existing keys."""
    storage = reset_storage()
    storage.storage = set()
    storage.persist_storage = False
    storage.storage.add("key1")
    result = storage.get("key1")
    assert result is not None, "Expected non-None for existing key"
    result_missing = storage.get("key2")
    assert result_missing is None, "Expected None for missing key"


def test_request_stats_l3_fields():
    """RequestStats should have all L3 read metric fields with defaults."""
    rs = RequestStats()
    assert rs.l3_exists_tokens == 0, f"l3_exists_tokens default: {rs.l3_exists_tokens}"
    assert rs.l3_read_issued_tokens == 0
    assert rs.l3_read_completed_tokens == 0
    assert rs.l3_restored_tokens == 0
    assert rs.l3_consumed_tokens == 0
    assert rs.l3_read_bytes == 0
    assert rs.l3_read_dur_ms == 0.0
    # Set and verify
    rs.l3_exists_tokens = 100
    rs.l3_read_issued_tokens = 80
    rs.l3_read_completed_tokens = 80
    rs.l3_restored_tokens = 75
    rs.l3_consumed_tokens = 70
    rs.l3_read_bytes = 1024
    rs.l3_read_dur_ms = 5.5
    assert rs.l3_exists_tokens == 100
    assert rs.l3_restored_tokens == 75
    assert rs.l3_consumed_tokens == 70


def test_state_manager_l3_load_dur():
    """StateManager should track L3 load duration with max semantics."""
    StateManager.reset()
    assert StateManager.pop_hicache_l3_load_dur() == 0.0
    StateManager.inc_hicache_l3_load_dur(0.001)
    assert abs(StateManager.pop_hicache_l3_load_dur() - 0.001) < 1e-9
    # Max semantics: inc with smaller value doesn't override
    StateManager.inc_hicache_l3_load_dur(0.005)
    StateManager.inc_hicache_l3_load_dur(0.002)  # smaller, should NOT override
    dur = StateManager.pop_hicache_l3_load_dur()
    assert abs(dur - 0.005) < 1e-9, f"Expected max(0.005, 0.002)=0.005, got {dur}"
    # Pop resets to 0
    assert StateManager.pop_hicache_l3_load_dur() == 0.0


def test_l3_exists_vs_read_distinction():
    """Verify that exists and read are separate operations.

    In the L3 pipeline:
    - batch_exists = existence check (fast, no data transfer)
    - batch_get = actual read (slow, data transfer)
    - A key can exist (batch_exists=True) but read can fail (batch_get=None)
      if the data is corrupted or unavailable.
    - In the Metadata-only Restore Mock, batch_get succeeds for existing keys.
    """
    storage = reset_storage()
    storage.storage = set()
    storage.persist_storage = False
    storage.storage.add("key1")
    # Exists check
    exists_count = storage.batch_exists(["key1", "key2"], None)
    assert exists_count == 1, f"Expected 1 existing key, got {exists_count}"
    # Read check: should succeed for existing key
    result = storage.batch_get(["key1"], target_locations=["dummy"])
    assert result[0] is not None, "batch_get should succeed for existing key"
    # Verify counters are separate
    assert MockHiCacheStorage._DEBUG_BATCH_EXISTS_COUNT == 1
    assert MockHiCacheStorage._DEBUG_BATCH_GET_COUNT == 1


def test_batch_get_consistency_with_batch_set():
    """batch_get should return data for keys that were batch_set."""
    storage = reset_storage()
    storage.storage = set()
    storage.persist_storage = False
    # Write keys via batch_set
    keys = ["k1", "k2", "k3"]
    storage.batch_set(keys, [None] * len(keys))
    # All keys should exist
    for k in keys:
        assert storage.exists(k), f"Key {k} should exist after batch_set"
    # batch_get should return data for all
    result = storage.batch_get(keys, target_locations=["d"] * len(keys))
    assert all(r is not None for r in result), f"All keys should be readable: {result}"


def main():
    runner = TestRunner()
    print("Phase 2d: L3 Read/Write Restore Loop tests\n")

    # P0-1: batch_get Metadata-only Restore Mock
    runner.run("batch_get returns data for existing keys", test_batch_get_returns_data_for_existing_keys)
    runner.run("batch_get returns None for missing keys", test_batch_get_returns_none_for_missing_keys)
    runner.run("batch_get works without target_locations", test_batch_get_without_target_locations)
    runner.run("batch_get_v1 returns bool list", test_batch_get_v1_returns_bool_list)
    runner.run("batch_get_v2 returns dict", test_batch_get_v2_returns_dict)
    runner.run("get() returns data for existing", test_get_returns_data_for_existing)

    # P0-1: RequestStats L3 fields
    runner.run("RequestStats has L3 read metric fields", test_request_stats_l3_fields)

    # P0-1: StateManager L3 load dur
    runner.run("StateManager L3 load dur (max semantics)", test_state_manager_l3_load_dur)

    # P0-1: Exists vs Read distinction
    runner.run("L3 exists vs read distinction", test_l3_exists_vs_read_distinction)

    # P0-1: Consistency with batch_set
    runner.run("batch_get consistency with batch_set", test_batch_get_consistency_with_batch_set)

    return runner.summary()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
