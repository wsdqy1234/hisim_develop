"""Native SGLang L3 page read -> accepted HBF program -> storage ACK.

The test uses the real generic page getter, MockHiCacheStorage.batch_get,
TieredHostKVCache, native Hybrid KV batch completion and native ACK producer.
Only the model geometry and timing budget are small controlled inputs.
"""
from collections import defaultdict
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from hisim.simulation.manager import ConfigManager, StateManager
from hisim.simulation.types import RequestStats
from hisim.simulation.sglang import sglang_hook as hook
from hisim.simulation.sglang.hbf_tiered_pool import MockHBFPool, TieredHostKVCache
from hisim.simulation.sglang.sglang_mock_class import MockHiCacheStorage
from hisim.simulation.hbf_shared_resource import SharedHbfResourceManager
from hisim.simulation.hbf_win_pool import Hbf8WinConfig
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController, PoolName, PoolTransfer)


BPT = 114688


@pytest.fixture
def setup(monkeypatch):
    class Controller(HybridCacheController):
        pass

    # Installing both hooks must not double-wrap inherited batch transfer.
    hook.C_HiCacheController.hook(Controller)
    wrapped_once = Controller._page_transfer_kv_batch
    hook.C_HybridCacheControllerHook.hook(Controller)
    assert Controller._page_transfer_kv_batch is wrapped_once
    controller = Controller.__new__(Controller)
    adapter = TieredHostKVCache(MockHBFPool(2048, page_size=1),
                               MockHBFPool(32, page_size=1))
    controller._hbf_adapter_instance = adapter
    controller.storage_host_pool = adapter
    controller.storage_backend = MockHiCacheStorage.__new__(MockHiCacheStorage)
    controller.storage_backend.storage = {"a", "b", "c", "d"}
    controller.page_get_func = controller._generic_page_get
    controller.page_size = 1
    controller.enable_storage = True
    controller.prefetch_sync_queue = Queue()
    controller.prefetch_buffer = Queue()
    controller.prefetch_queue = Queue()
    controller.prefetch_hit_queue = Queue()
    manager = SharedHbfResourceManager()
    manager.configure(media_bw=1.6e12, interconnect_bw=112e9, media_model="win8",
                      win_config=Hbf8WinConfig(placement="striped", stripe_block_tokens=256))
    manager.begin_batch(1.0)
    monkeypatch.setattr(hook, "get_shared_hbf_manager", lambda: manager)
    monkeypatch.setattr(ConfigManager, "get_kv_cache_bytes", classmethod(lambda cls: BPT))
    monkeypatch.setattr(ConfigManager, "_platform_config", SimpleNamespace(enable_l3_direct_to_l1=False))
    stats = defaultdict(RequestStats)
    stats["l3-request"]
    monkeypatch.setattr(hook.C_SchedulerHook, "REQUEST_STATS", stats)
    monkeypatch.setattr(hook.C_SchedulerHook, "_L3_PREFETCHED_TOKENS", 0)
    monkeypatch.setattr(hook.hbf_io_trace, "_TRACE_ENABLED", True)
    hook.hbf_io_trace.reset_trace()
    StateManager.reset()
    StateManager.advance_global_clock_to(1.0)
    yield controller, manager, stats
    hook.hbf_io_trace.reset_trace()
    StateManager.reset()


def operation(slots=(768, 769, 770, 771), keys=("a", "b", "c", "d")):
    return SimpleNamespace(request_id="l3-request", host_indices=torch.tensor(slots),
        hash_value=list(keys), prefix_keys=[], pool_transfers=None,
        is_terminated=lambda: False, _hisim_l3_read_issue_s=.99,
        _hisim_l3_read_complete_s=1.0, _hisim_l3_read_service_s=.001,
        _hisim_l3_read_completion_basis="budget_release_boundary")


def test_native_page_read_materializes_before_ack_and_restore_waits(setup):
    controller, manager, stats = setup
    op = operation()
    stats[op.request_id].l3_read_issue_time = .99
    controller.prefetch_buffer.put(op)
    controller.handle_prefetch_io_operation()
    event, = manager.win_pool.get_io_events()
    generic, = hook.hbf_io_trace.get_trace()
    assert event["metadata"]["source"] == "l3_restore"
    assert generic["details"] == event
    assert generic["rid"] == op.request_id
    assert event["bytes"] == 4 * BPT
    assert [c["win"] for c in event["sub_ios"]] == [3]
    assert manager.win_pool.wins[3].io_count == 1
    assert manager.win_pool.wins[0].io_count == 0
    assert StateManager.pop_hicache_hbf_backup_dur() == 0
    assert StateManager.get_global_clock() == 1.0
    assert stats[op.request_id].l3_read_completed_tokens == 4
    acks = [controller.prefetch_sync_queue.get_nowait(), controller.prefetch_sync_queue.get_nowait()]
    assert acks[0].completed_tokens == 4 and acks[-1].completed_req
    read = manager.win_pool.submit_object(1.0, op.host_indices.tolist(), BPT, is_read=True)
    assert read.trace_details["status"] == "complete"
    assert read.trace_details["dependency_wait_s"] > 0
    assert read.trace_details["sub_ios"][0]["media_start_s"] >= event["completion_s"]
    assert read.trace_details["completion_s"] > event["completion_s"]


def test_partial_native_read_page_size_two_materializes_only_accepted_prefix(setup):
    controller, manager, stats = setup
    controller.page_size = 2
    controller.storage_backend.storage = {"a", "c"}  # Missing b stops the prefix.
    op = operation((768, 769, 770, 771, 772, 773), ("a", "b", "c"))
    controller.prefetch_buffer.put(op)
    controller.handle_prefetch_io_operation()
    event, = manager.win_pool.get_io_events()
    assert event["slots"] == [768, 769]
    assert event["bytes"] == 2 * BPT
    assert stats[op.request_id].l3_read_completed_tokens == 2
    assert 770 not in manager.win_pool.registry.slot_owner
    ack = controller.prefetch_sync_queue.get_nowait()
    assert ack.completed_tokens == 2


def test_sidecar_minimum_controls_programmed_prefix_and_repeat_does_not_reprogram(setup):
    controller, manager, stats = setup
    op = operation()
    controller.storage_backend.batch_get_v2 = lambda transfers: {"indexer": [True, False, True, True]}
    derived = [PoolTransfer(name="indexer", indices_from_pool=PoolName.KV)]
    args = (op, op.hash_value, op.host_indices, None, derived)
    assert controller._page_transfer_kv_batch(*args) == 1
    assert controller._page_transfer_kv_batch(*args) == 1
    event, = manager.win_pool.get_io_events()
    assert event["slots"] == [768] and event["bytes"] == BPT
    assert op._hisim_l3_transferred_tokens == 1
    assert len(hook.hbf_io_trace.get_trace()) == 1


def test_failed_native_read_never_creates_hbf_object(setup):
    controller, manager, stats = setup
    controller.storage_backend.storage = set()
    op = operation()
    stats[op.request_id].prefetch_complete_tokens = 4  # Timing-only prediction.
    controller.prefetch_buffer.put(op)
    controller.handle_prefetch_io_operation()
    assert stats[op.request_id].l3_read_completed_tokens == 0
    assert not manager.win_pool.get_io_events()
    assert manager.win_pool.total_ios == 0


def test_chunk_budget_release_uses_paid_clock_not_another_ssd_tail(setup, monkeypatch):
    controller, manager, stats = setup
    op = operation()
    op._hisim_l3_read_service_s = .003
    stats[op.request_id].l3_read_issue_time = .99
    controller.chunked_prefetch_operation = {"operation": op, "storage_hit_count": 4, "progress": 2}
    monkeypatch.setattr(hook.C_HiCacheController, "KV_CACHE_BYTES", BPT)
    monkeypatch.setattr(hook.C_HiCacheController, "KV_CACHE_BYTES_FULL", BPT)
    monkeypatch.setattr(hook.C_HiCacheController, "DISK_READ_BANDWIDTH_BYTES", 8e9)
    monkeypatch.setattr(hook.C_HiCacheController, "_tier_sims", None)
    monkeypatch.setattr(hook.C_HiCacheController, "calc_prefetch_pages", staticmethod(lambda *args: (2, .001)))
    StateManager.set_current_inference_dur(.005)
    StateManager.step_global_clock(.005)
    controller.handle_prefetch_operation()
    assert op._hisim_l3_read_complete_s == 1.005
    assert stats[op.request_id].l3_read_complete_time == 1.005
    assert stats[op.request_id].l3_read_dur_ms == 4
    controller.prefetch_buffer.put(controller.prefetch_hit_queue.get_nowait())
    controller.handle_prefetch_io_operation()
    event, = manager.win_pool.get_io_events()
    assert event["arrival_s"] == 1.005
    assert event["metadata"]["l3_read_complete_s"] == 1.005
    assert StateManager.get_global_clock() == 1.005


def test_nonoverlap_consume_stat_follows_host_read_completion(setup, monkeypatch):
    _, _, stats = setup
    stats["consumer"].l3_consumed_tokens = 4
    stats["consumer"].l3_consume_time = 1.0  # Scheduling observation.
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid="consumer")],
                            forward_mode=SimpleNamespace(is_extend=lambda: True))
    monkeypatch.setattr(hook.C_SchedulerHook, "OVERLAP_SCHEDULE", False)
    hook._record_l3_consumption_after_host_load(batch, 1.0, .0002)
    assert stats["consumer"].l3_consume_time == 1.0002


def test_future_read_availability_is_not_booked_as_a_future_win_write(setup):
    controller, manager, _ = setup
    op = operation()
    op._hisim_l3_read_complete_s = 1.1
    with pytest.raises(ValueError, match="availability boundary"):
        controller._page_transfer_kv_batch(op, op.hash_value, op.host_indices, None, [])
    assert manager.win_pool.total_ios == 0
    assert manager.win_pool.get_io_events() == []
    assert StateManager.get_global_clock() == 1.0


def test_dram_endpoint_does_not_invent_an_hbm_backup(setup):
    controller, manager, stats = setup
    op = operation((2048, 2049, 2050, 2051))
    controller.prefetch_buffer.put(op)
    controller.handle_prefetch_io_operation()
    assert stats[op.request_id].l3_read_completed_tokens == 4
    assert controller._hbf_adapter_instance._dram_set_data_page_count == 4
    assert manager.win_pool.get_io_events() == []
    assert StateManager.pop_hicache_l2_backup_dur() == 0
