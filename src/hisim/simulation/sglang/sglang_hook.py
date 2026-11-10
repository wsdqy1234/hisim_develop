import torch
import time
from dataclasses import asdict
from collections import defaultdict
import json
import os
import heapq
import importlib
import threading
from queue import Empty

from hisim.simulation.hbf_shared_resource import (
    get_shared_hbf_manager,
    get_shared_dram_manager,
    configure_shared_downstream,
    reset_downstream,
    is_shared_downstream_enabled,
)
from hisim.simulation.sglang import hbf_io_trace

from hisim.utils import get_logger

from hisim.hook import BaseHook
from hisim.simulation.types import (
    MockSimulationMode,
    RequestStats,
)
from hisim.hook.utils import get_obj_from_args
from hisim.utils.json import CustomJsonEncoder
from hisim.simulation.manager import StateManager, ConfigManager, Envs
from hisim.time_predictor import (
    InferTimePredictor,
    FakeRequest,
    ScheduleBatch as HisimScheduleBatch,
)
from hisim.simulation.sglang.sglang_mock_class import (
    MockAttentionBackend,
    MockHiCacheStorage,
    MockKVIndexTranslator,
    MockNgramEmbeddingManager,
    MockPagedTokenToKVPoolAllocator,
    MockReqToTokenPool,
    MockTokenToKVPool,
    MockTokenToKVPoolAllocator,
    MockTokenToKVPoolHost,
    MockWeightLoader,
)
from hisim.simulation.utils import (
    calc_metrics,
    estimate_kv_cache_pool_capacity,
)
from hisim.simulation.sglang.version import VersionDispatcher


logger = get_logger("hisim")


def _is_legacy_sglang() -> bool:
    """True for sglang <= 0.5.9 (async hicache transfer threads + CUDA events
    that never complete on CPU); False for >= 0.5.10 (synchronous transfers,
    event-driven acks)."""
    try:
        from sglang.version import __version__ as _v
    except Exception:
        return True
    try:
        from packaging.version import InvalidVersion, Version

        return Version(_v) < Version("0.5.10")
    except InvalidVersion:
        return True


_hbf_adapter_installed = False
# Phase 4-Closure P3: adapter ref for end-of-run occupancy stats in trace dump
_HBF_ADAPTER_REF = None


def _install_hbf_adapter_if_enabled(cache_obj) -> None:
    """One-time installation of TieredHostKVCache adapter (HBF L1.5 PoC).

    Triggered by HISIM_HBF_ENABLE=1 env var. Replaces the anchor host pool
    with a TieredHostKVCache that routes to HBF + DRAM. Must run AFTER pool
    stack assembly (first check_hicache_events) and BEFORE any backup.
    """
    global _hbf_adapter_installed
    if _hbf_adapter_installed:
        return
    import os

    if not os.environ.get("HISIM_HBF_ENABLE", "0") in ("1", "true", "True"):
        return
    cc = getattr(cache_obj, "cache_controller", None)
    if cc is None or not getattr(cc, "mem_pool_host", None):
        print(f"[HBF-DEBUG] skip: cc={cc}, mem_pool_host={getattr(cc, 'mem_pool_host', None) if cc else 'N/A'}", flush=True)
        return
    # The adapter replaces the L2 anchor host pool — this exists whenever
    # hierarchical cache is enabled, regardless of L3 storage backend.
    # Don't gate on enable_storage (that's L3-specific).
    print(f"[HBF-DEBUG] attempting install: cc={type(cc).__name__}, mem_pool_host={type(cc.mem_pool_host).__name__}", flush=True)
    try:
        from hisim.simulation.sglang.hbf_tiered_pool import install_tiered_host_pool
        from hisim.simulation.manager import ConfigManager

        hbf_size = int(os.environ.get("HISIM_HBF_LOGICAL_SIZE", "100000"))
        hbf_ratio = float(os.environ.get("HISIM_HBF_RATIO", "0.7"))
        # Phase 5B: capacity entry in GB (converted to tokens via real KV
        # bytes/token; e.g. 800GB @ 114,688 B/token = 6,975,660 tokens —
        # physical memory stays the 4 B/token mock placeholder).
        cap_gb = os.environ.get("HISIM_HBF_CAPACITY_GB")
        if cap_gb:
            try:
                kv_bpt = ConfigManager.get_kv_cache_bytes() or 114688
            except Exception:
                kv_bpt = 114688
            hbf_size = max(1, int(float(cap_gb) * 1e9 / kv_bpt))
            print(f"[HBF-DEBUG] HBF capacity {float(cap_gb)}GB → "
                  f"{hbf_size} tokens ({kv_bpt} B/token, mock mem only)",
                  flush=True)
        adapter = install_tiered_host_pool(cc, hbf_size, hbf_ratio)
        if adapter is not None:
            _hbf_adapter_installed = True
            # Phase 4-Closure P3: module-level ref for trace-time occupancy
            global _HBF_ADAPTER_REF
            _HBF_ADAPTER_REF = adapter
            # Store on cache object for periodic stats logging
            cache_obj._hbf_adapter_instance = adapter
            cc._hbf_adapter_instance = adapter  # also on controller for mock_start_writing/loading
            # V1a-2: Set eviction callback for bounded retry in alloc()
            # cache_obj is the UnifiedRadixCache which has evict_host()
            adapter._evict_callback = lambda n: cache_obj.evict_host(n)
            # Also set HBFConfig in ConfigManager so _hbf_transfer_duration works
            from hisim.simulation.types import HBFConfig
            from hisim.simulation.manager import ConfigManager
            read_lat = float(os.environ.get("HISIM_HBF_READ_LATENCY_US", "10"))
            write_lat = float(os.environ.get("HISIM_HBF_WRITE_LATENCY_US", "100"))
            read_bw = float(os.environ.get("HISIM_HBF_READ_BW_GB", "512"))
            write_bw = os.environ.get("HISIM_HBF_WRITE_BW_GB")
            hbf_cfg = HBFConfig(
                capacity_gb=hbf_size * adapter.size_per_token / 1e9,
                read_bandwidth_gb=read_bw,
                write_bandwidth_gb=float(write_bw) if write_bw else None,
                read_latency_us=read_lat,
                write_latency_us=write_lat,
                placement_ratio=hbf_ratio,
                # Phase 3B: channel geometry from env (defaults match nominal.json)
                channels=int(os.environ.get("HISIM_HBF_CHANNELS", "32")),
                channel_width_bits=int(os.environ.get("HISIM_HBF_CHANNEL_WIDTH_BITS", "8")),
                channel_transfer_rate_mtps=int(os.environ.get("HISIM_HBF_CHANNEL_RATE_MTPS", "1600")),
                page_bytes=int(os.environ.get("HISIM_HBF_PAGE_BYTES", "16384")),
                dies_per_channel=int(os.environ.get("HISIM_HBF_DIES_PER_CHANNEL", "8")),
                planes_per_die=int(os.environ.get("HISIM_HBF_PLANES_PER_DIE", "4")),
            )
            pc = ConfigManager.get_platform_config()
            if pc is not None:
                pc.hbf = hbf_cfg
                pc.enable_hbf = True
                print(f"[HBF-DEBUG] ConfigManager.hbf set: read_lat={read_lat}μs "
                      f"write_lat={write_lat}μs read_bw={read_bw}GB/s", flush=True)
            # Phase 3D-P1: Configure shared HBF resource manager
            ic_bw_gb = float(os.environ.get("HISIM_HBF_INTERCONNECT_BW_GB",
                                            str(hbf_cfg.aggregate_channel_bw_bytes_per_s / 1e9)))
            # Phase 4-Closure P0: media BW override (efficiency experiments)
            #   e.g. HISIM_HBF_MEDIA_BW_GB=40.96 models 80% media efficiency
            #   vs the 51.2 GB/s nominal aggregate (simulated reference).
            media_bw_override = os.environ.get("HISIM_HBF_MEDIA_BW_GB")
            media_bw = (float(media_bw_override) * 1e9
                        if media_bw_override
                        else hbf_cfg.aggregate_channel_bw_bytes_per_s)
            # Phase 4-Closure P1: media model selection
            #   "formula" (default) — closed-form, Theory Bound
            #   "pagesim" — discrete-event page/plane/channel, Model Behavior
            #   "win8" (Phase 5B) — 8-WIN pool: all params from
            #   Hbf8WinConfig (JSON via HISIM_HBF_WIN_CONFIG); the WIN pool
            #   owns media AND interface, so media_bw/interconnect_bw
            #   overrides are IGNORED in this mode.
            media_model = os.environ.get("HISIM_HBF_MEDIA_MODEL", "formula")
            win_config_obj = None
            if media_model == "win8":
                from hisim.simulation.hbf_win_pool import Hbf8WinConfig
                win_cfg_path = os.environ.get("HISIM_HBF_WIN_CONFIG")
                if win_cfg_path and os.path.exists(win_cfg_path):
                    win_config_obj = Hbf8WinConfig.from_json(win_cfg_path)
                    print(f"[HBF-DEBUG] WIN config loaded from {win_cfg_path}",
                          flush=True)
                else:
                    win_config_obj = Hbf8WinConfig()
                    if win_cfg_path:
                        print(f"[HBF-DEBUG] WIN config path not found "
                              f"({win_cfg_path}) — using expert defaults",
                              flush=True)
                # Effective config snapshot (assumptions annotated)
                try:
                    from hisim.simulation.manager import Envs
                    eff_dir = Envs.output_dir()
                    os.makedirs(eff_dir, exist_ok=True)
                    eff_path = os.path.join(eff_dir, "hbf_win_effective_config.json")
                    win_config_obj.dump_effective_json(eff_path)
                    print(f"[HBF-DEBUG] WIN effective config → {eff_path}",
                          flush=True)
                except Exception as e:
                    print(f"[HBF-DEBUG] WIN config dump failed: {e}", flush=True)
            mgr = get_shared_hbf_manager()
            mgr.configure(
                media_bw=media_bw,
                interconnect_bw=ic_bw_gb * 1e9,
                queue_depth=hbf_cfg.queue_depth,
                media_model=media_model,
                win_config=win_config_obj,
            )
            print(f"[HBF-DEBUG] SharedHbfResourceManager: media={media_bw/1e9:.1f}GB/s "
                  f"(model={media_model}) ic={ic_bw_gb:.1f}GB/s QD={hbf_cfg.queue_depth}",
                  flush=True)
            if media_model == "win8":
                wp = mgr.win_pool
                print(f"[HBF-DEBUG] 8-WIN pool: {len(wp.wins)} WINs × "
                      f"{wp.config.win.capacity_bytes/1e9:.0f}GB "
                      f"(seq={wp.config.aggregate_seq_read_bw/1e9:.0f} "
                      f"rand={wp.config.aggregate_rand_read_bw/1e9:.0f} "
                      f"write={wp.config.aggregate_write_bw/1e9:.0f} GB/s agg) "
                      f"layer={wp.config.layer} placement={wp.config.placement} "
                      f"shared_ic="
                      f"{wp.config.package_shared_interface_bw/1e9 if wp.config.package_shared_interface_bw else 'off'}GB/s",
                      flush=True)
            # Phase 3E: Configure shared DRAM resource manager
            dram_mgr = get_shared_dram_manager()
            dram_read_bw = pc.memory_read_bandwidth if pc else 16e9
            dram_mgr.configure(dram_bw=dram_read_bw * 0.85)
            print(f"[HBF-DEBUG] SharedDramResourceManager: dram_bw={dram_read_bw*0.85/1e9:.1f}GB/s", flush=True)
            # Phase 3E-Closure: Configure shared downstream if requested
            ds_bw_gb = os.environ.get("HISIM_SHARED_DOWNSTREAM_BW_GB")
            if ds_bw_gb:
                if media_model == "win8":
                    # Phase 5D P5: correct composition — bridge the WIN
                    # pool's package interface onto the SHARED downstream
                    # cursor so HBF and DRAM I/Os contend on one resource
                    # (replaces the previous WARNING-only behavior).
                    try:
                        from hisim.simulation.hbf_shared_resource import (
                            get_downstream_busy_until,
                            update_downstream_busy_until,
                        )
                        mgr.win_pool.attach_external_interface(
                            get_downstream_busy_until,
                            update_downstream_busy_until,
                            float(ds_bw_gb) * 1e9)
                        print(f"[HBF-DEBUG] win8 + SharedDownstream composed: "
                              f"WIN interface bridged onto shared downstream "
                              f"{float(ds_bw_gb)}GB/s cursor", flush=True)
                    except Exception as e:
                        print(f"[HBF-DEBUG] win8+DS bridge failed: {e}",
                              flush=True)
                else:
                    configure_shared_downstream(float(ds_bw_gb) * 1e9)
                print(f"[HBF-DEBUG] SharedDownstream: bw={float(ds_bw_gb):.1f}GB/s "
                      f"(HBF+DRAM share Host→HBM path)", flush=True)
            # Phase 5D P0 (G3): wire adapter frees → WIN registry invalidation
            if media_model == "win8" and mgr.win_pool is not None:
                adapter._on_hbf_free = mgr.win_pool.notify_slots_freed
            print(f"[HBF-DEBUG] installed OK: {adapter.capacity_stats()}", flush=True)
    except Exception as e:
        print(f"[HBF-DEBUG] install failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        _hbf_adapter_installed = True  # don't retry


def _drive_hicache_timing_phases(cache_obj) -> None:
    """Run the hisim storage data/timing phases on a tree cache:

    1. Data phase: batch_set writes the previous round's KV to the storage.
    2. Data + read timing: batch_exists queries the storage (+ SSD read
       timing on simulators).
    3. Write timing: submit_write advances the SSD simulation state.

    Called from the hooked check_hicache_events before the stock event
    processing. Shared by the HiRadixCache (<= 0.5.9) and UnifiedRadixCache
    (>= 0.5.10) hooks."""
    cc = getattr(cache_obj, "cache_controller", None)
    if cc is None or not getattr(cc, "enable_storage", False):
        return
    cc.handle_backup_operation()
    cc.handle_prefetch_operation()
    cc.submit_l3_write_timing()


def _drive_prefetch_io_phase(cache_obj) -> None:
    """Run the synchronous L3 data-loading phase AFTER the stock
    check_hicache_events.

    The stock ``_drain_and_alloc_storage_hit`` (inside
    ``original_check_hicache_events``) allocates host memory for
    prefetch-hit operations and puts them in ``prefetch_buffer``. This
    function drains ``prefetch_buffer`` synchronously, calling
    ``_page_transfer`` → ``batch_get`` (Metadata-only Restore Mock) to
    logically restore KV pages, then emits ``PrefetchAck`` into
    ``prefetch_sync_queue`` for the NEXT round's ``_drain_ack_prefetch``
    to process (→ ``_handle_prefetch_result`` → radix-tree insertion).

    This one-round pipeline (query → alloc → load → ack → insert → consume)
    mirrors the real async prefetch while keeping timing deterministic.
    """
    cc = getattr(cache_obj, "cache_controller", None)
    if cc is None or not getattr(cc, "enable_storage", False):
        return

    # Track L3 restored tokens: _drain_ack_prefetch (inside the preceding
    # original_check_hicache_events) processed acks from the PREVIOUS round's
    # handle_prefetch_io_operation, calling _handle_prefetch_result which
    # updates prefetch_loaded_tokens_by_reqid with loaded_from_storage.
    loaded_map = getattr(cache_obj, "prefetch_loaded_tokens_by_reqid", None)
    if loaded_map:
        for rid, loaded in list(loaded_map.items()):
            try:
                rs = C_SchedulerHook.REQUEST_STATS.get(rid)
                if rs is not None:
                    rs.l3_restored_tokens = loaded
                    # Read-completed tokens come from accepted native page
                    # transfers, not Exists or timing progress. A failed data
                    # read must remain zero even if timing predicted a hit.
                    # Phase 2d-Closure: restore commit timestamp
                    rs.l3_restore_commit_time = StateManager.get_global_clock()
            except Exception:
                pass
        loaded_map.clear()

    if hasattr(cc, "handle_prefetch_io_operation"):
        cc.handle_prefetch_io_operation()


def _mark_l3_read_timing_complete(operation) -> None:
    """The existing prefetch budget has reached its completion boundary.

    Progress consumed completed inference/idle-poll budgets. Using the
    release clock is conservative and avoids charging the final budget tail
    a second time or booking a future HBF write ahead of earlier arrivals.
    This is a budget-model availability boundary, not a page-level SSD trace.
    """
    now = StateManager.get_global_clock()
    issue = getattr(operation, "_hisim_l3_read_issue_s", now)
    if now < issue:
        raise ValueError("L3 read release precedes its issue time")
    operation._hisim_l3_read_complete_s = now
    operation._hisim_l3_read_completion_basis = "budget_release_boundary"
    rs = C_SchedulerHook.REQUEST_STATS.get(operation.request_id)
    if rs is not None:
        rs.l3_read_complete_time = now
        rs.l3_read_dur_ms = getattr(operation, "_hisim_l3_read_service_s", 0.0) * 1000


def _materialize_l3_hbf_prefix(controller, operation, host_indices,
                               completed_pages: int) -> None:
    """Program only the native-accepted HBF prefix, before its storage ACK.

    Ordinary DRAM is the endpoint of the existing L3 read. HBF additionally
    requires media programming and registry ownership. This does not invent
    an HBM->Host backup or charge its round accumulator; a subsequent HBF
    restore waits on the registry's write-completion gate.
    """
    count = int(completed_pages) * int(controller.page_size)
    if count <= 0:
        return
    values = host_indices.tolist() if hasattr(host_indices, "tolist") else list(host_indices)
    if count > len(values):
        raise ValueError("Native L3 accepted more tokens than the transfer batch")
    accepted = [int(s) for s in values[:count]]
    seen = getattr(operation, "_hisim_l3_accepted_slots", set())
    new_accepted = [s for s in accepted if s not in seen]
    seen.update(accepted)
    operation._hisim_l3_accepted_slots = seen
    operation._hisim_l3_transferred_tokens = (
        getattr(operation, "_hisim_l3_transferred_tokens", 0) + len(new_accepted))
    adapter = getattr(controller, "_hbf_adapter_instance", None)
    if adapter is None:
        return
    hbf_slots = [s for s in accepted if s < adapter.dram_offset]
    previous = getattr(operation, "_hisim_l3_materialized_hbf_slots", set())
    hbf_slots = [s for s in hbf_slots if s not in previous]
    if not hbf_slots:
        return
    manager = get_shared_hbf_manager()
    if getattr(manager, "media_model", "") != "win8" or manager.win_pool is None:
        return
    now = StateManager.get_global_clock()
    read_complete = getattr(operation, "_hisim_l3_read_complete_s", now)
    if read_complete > now:
        raise ValueError("L3 Host materialization before its read-availability boundary")
    from hisim.simulation.manager import ConfigManager
    bpt = ConfigManager.get_kv_cache_bytes()
    metadata = {"source": "l3_restore", "request_id": operation.request_id,
                "submission_clock_s": now,
                "l3_read_issue_s": getattr(operation, "_hisim_l3_read_issue_s", now),
                "l3_read_complete_s": read_complete,
                "l3_read_completion_basis": getattr(operation,
                    "_hisim_l3_read_completion_basis", "native_transfer_boundary"),
                "l3_read_service_s": getattr(operation, "_hisim_l3_read_service_s", None),
                "dram_endpoint": "covered_by_existing_l3_read; additional DRAM bus contention unmodeled"}
    stats = manager.win_pool.submit_object(now, hbf_slots, bpt,
                                           is_read=False, metadata=metadata)
    previous.update(hbf_slots)
    operation._hisim_l3_materialized_hbf_slots = previous
    hbf_io_trace.set_context(operation.request_id)
    hbf_io_trace.record("backup", "hbf", len(hbf_slots), len(hbf_slots) * bpt,
        now, stats.total, media_time=stats.media_service, ic_time=stats.ic_time,
        queue_wait=stats.queue_wait, details=stats.trace_details)


def _wrap_l3_hbf_materialization(target) -> None:
    original = getattr(target, "_page_transfer_kv_batch", None)
    if original is None or getattr(original, "_hisim_l3_materialization", False):
        return

    def wrapped(self, operation, batch_hashes, batch_host_indices,
                extra_info, kv_derived_transfers):
        completed = original(self, operation, batch_hashes, batch_host_indices,
                             extra_info, kv_derived_transfers)
        # The native return is already clamped to KV-derived sidecar hits.
        # Its parent emits the token ACK only after this method returns.
        _materialize_l3_hbf_prefix(self, operation, batch_host_indices, completed)
        return completed

    wrapped._hisim_l3_materialization = True
    target._page_transfer_kv_batch = wrapped


def _record_l3_consumption_after_host_load(batch, batch_start, load_duration):
    """Non-overlap inference consumes restored KV only after Host load."""
    if C_SchedulerHook.OVERLAP_SCHEDULE or not batch.forward_mode.is_extend():
        return
    for req in batch.reqs:
        stats = C_SchedulerHook.REQUEST_STATS.get(req.rid)
        if stats is not None and stats.l3_consumed_tokens > 0:
            stats.l3_consume_time = batch_start + load_duration


# ---- Phase 4-Closure P3: I/O trace state ----
# Last stats returned by the shared resource managers, stashed by the
# transfer-duration functions so the _record_*_timing choke points can
# attach the media/ic/queue-wait breakdown to each trace entry.
_LAST_HBF_IO_STATS = None      # HbfIoStats / object-aware WinIoStats / None
_LAST_DRAM_IO_STATS = None     # dict or None


def _sim_clock() -> float:
    """Current simulation clock (0.0 on failure)."""
    try:
        return StateManager.get_global_clock()
    except Exception:
        return 0.0


def _request_is_unfinished(req) -> bool:
    finished = getattr(req, "finished", None)
    if callable(finished):
        return not finished()
    # Unknown request types are conservatively active until detached.
    return not finished if finished is not None else True


def _scheduler_has_pending_batch(scheduler, prefill_result=None,
                                 supplied_batches=()) -> bool:
    """Include locally-owned batches while SGLang is constructing its plan.

    Newer schedulers pass a local running_batch into get_new_batch_prefill
    and assign it back to self.running_batch only after this hook returns.
    An empty instance field therefore cannot authorize an idle clock jump.
    Finished requests in stale debug/last-batch views do not block idleness.
    """
    batches = [prefill_result,
               getattr(prefill_result, "running_batch", None),
               getattr(prefill_result, "batch_to_run", None),
               *supplied_batches]
    batches.extend(getattr(scheduler, name, None) for name in (
        "running_batch", "cur_batch", "cur_batch_for_debug", "last_batch"))
    for batch in batches:
        if batch is not None and any(
                _request_is_unfinished(req) for req in getattr(batch, "reqs", ())):
            return True
    chunked = getattr(scheduler, "chunked_req", None)
    return ((chunked is not None and _request_is_unfinished(chunked))
            or bool(getattr(scheduler, "result_queue", ())))


def _l2_transfer_duration(direction: str, num_bytes: float) -> float:
    """L2 (device <-> host) transfer duration.

    Two modes:
    - Default (no shared manager): independent model
      duration = t0 + bytes / (raw_bandwidth * 0.85)
    - Shared mode (HISIM_SHARED_DRAM=1): submits to SharedDramResourceManager
      for concurrent I/O bandwidth sharing, matching HBF's shared model.

    direction="H2D": host->device load, t0=6.67us
    direction="D2H": device->host backup, t0=4us
    """
    from hisim.simulation.manager import ConfigManager

    pc = ConfigManager.get_platform_config()
    if direction == "H2D":
        raw_bw = pc.memory_read_bandwidth
        t0 = 6.67e-6
    else:
        raw_bw = pc.memory_write_bandwidth
        t0 = 4e-6
    if not raw_bw:
        return 0.0

    global _LAST_DRAM_IO_STATS
    _LAST_DRAM_IO_STATS = None

    # Phase 3E: Use shared DRAM resource manager when enabled
    if os.environ.get("HISIM_SHARED_DRAM", "0") in ("1", "true", "True"):
        mgr = get_shared_dram_manager()
        if mgr.dram_bw > 0:
            try:
                from hisim.simulation.manager import StateManager
                arrival = StateManager.get_global_clock()
            except Exception:
                arrival = mgr.batch_start
            is_load = (direction == "H2D")
            stats = mgr.submit(arrival, int(num_bytes), is_load=is_load)
            _LAST_DRAM_IO_STATS = stats
            return stats["total"]

    # Default: independent model (no sharing)
    return t0 + num_bytes / (raw_bw * 0.85)


def _record_l2_backup_timing(num_bytes: float, num_tokens: int = 0) -> float:
    """D2H (write-through backup) timing for sglang >= 0.5.10, where the
    mocked synchronous start_writing replaces the MockTokenToKVPoolHost
    backup path that used to record this on <= 0.5.9.

    Phase 4-Closure P3: I/O trace entry (real KV bytes).
    """
    dur = _l2_transfer_duration("D2H", num_bytes)
    StateManager.inc_hicache_l2_backup_dur(dur)
    s = _LAST_DRAM_IO_STATS
    hbf_io_trace.record(
        op="backup", pool="dram", tokens=num_tokens, bytes_=num_bytes,
        arrival=_sim_clock(), duration=dur, path="real",
        media_time=s["service_time"] if s else None,
        queue_wait=s["queue_wait"] if s else None,
    )
    return dur


def _record_l2_load_timing(num_bytes: float, num_tokens: int,
                           path: str = "real") -> float:
    """H2D (load back) timing for sglang >= 0.5.10, with the L3
    direct-to-L1 split of the <= 0.5.9 MockTokenToKVPoolHost model: tokens
    prefetched straight from L3 are charged to the direct bandwidth while the
    remaining L2-resident tokens are charged to the memory bandwidth (both
    transfers overlap; the consumer takes the max).

    Phase 4-Closure P3: I/O trace entry. path="mock_lbk" marks the residual
    load_back record with mock buffer bytes.
    """
    if path == "mock_lbk":
        # Placeholder buffer bookkeeping is already represented by the
        # model-sized, per-request real load. Keep it observable without
        # submitting a second I/O or changing authoritative round timing.
        hbf_io_trace.record(
            op="load", pool="dram", tokens=num_tokens, bytes_=num_bytes,
            arrival=_sim_clock(), duration=0.0, path=path)
        return 0.0

    from hisim.simulation.manager import ConfigManager

    pc = ConfigManager.get_platform_config()
    direct_bw = None
    try:
        if pc.enable_l3_direct_to_l1:
            direct_bw = pc.l3_to_l1_bandwidth
    except Exception:
        direct_bw = None

    s = _LAST_DRAM_IO_STATS
    if direct_bw is not None and num_tokens > 0:
        l3_count = min(StateManager.get_l3_direct_load_count(), num_tokens)
        l2_count = num_tokens - l3_count
        bytes_per_token = num_bytes / num_tokens
        if l2_count > 0:
            StateManager.inc_hicache_l2_load_dur(
                _l2_transfer_duration("H2D", l2_count * bytes_per_token)
            )
        if l3_count > 0:
            StateManager.inc_hicache_l3_direct_load_dur(
                (l3_count * bytes_per_token) / direct_bw
            )
        hbf_io_trace.record(
            op="load", pool="dram", tokens=num_tokens, bytes_=num_bytes,
            arrival=_sim_clock(),
            duration=(l2_count + l3_count) * bytes_per_token
                     / (pc.memory_read_bandwidth * 0.85),
            path=path, queue_wait=None,
        )
        return

    dur = _l2_transfer_duration("H2D", num_bytes)
    StateManager.inc_hicache_l2_load_dur(dur)
    s = _LAST_DRAM_IO_STATS
    hbf_io_trace.record(
        op="load", pool="dram", tokens=num_tokens, bytes_=num_bytes,
        arrival=_sim_clock(), duration=dur, path=path,
        media_time=s["service_time"] if s else None,
        queue_wait=s["queue_wait"] if s else None,
    )
    return dur


def _hbf_transfer_duration(direction: str, num_bytes: float,
                           virtual_indices=None,
                           num_tokens: int = 0) -> float:
    """HBF L1.5 transfer timing.

    Two models selectable via HISIM_HBF_TIMING_MODEL env var:
      - "fast" (default): max(base_latency, bytes / aggregate_bandwidth)
      - "channel": per-channel bandwidth + page-level parallelism

    direction="H2D": HBF→L1 load (read), base = read_latency
    direction="D2H": L1→HBF backup (write), base = write_latency

    Phase 5D P0: virtual_indices (HBF slot ids) flow through to the
    win8 object-aware path (submit_object); ignored by fast/formula/
    pagesim models.
    """
    from hisim.simulation.manager import ConfigManager

    pc = ConfigManager.get_platform_config()
    if not pc.hbf:
        return 0.0

    global _LAST_HBF_IO_STATS
    _LAST_HBF_IO_STATS = None

    timing_model = os.environ.get("HISIM_HBF_TIMING_MODEL", "fast")
    if timing_model == "channel":
        return _hbf_channel_transfer_duration(
            direction, num_bytes, pc.hbf,
            virtual_indices=virtual_indices, num_tokens=num_tokens)

    # Fast model: max(base_latency, bytes / aggregate_bandwidth)
    if direction == "H2D":
        base_latency = pc.hbf.read_latency_us * 1e-6
        bw = pc.hbf_read_bandwidth
    else:
        base_latency = pc.hbf.write_latency_us * 1e-6
        bw = pc.hbf_write_bandwidth
    if not bw:
        return 0.0
    transfer = num_bytes / bw
    return max(base_latency, transfer)


def _hbf_channel_transfer_duration(direction: str, num_bytes: float, hbf_cfg,
                                   virtual_indices=None,
                                   num_tokens: int = 0) -> float:
    """Phase 3D: Channel/queue model latency for a single HBF I/O.

    Uses SharedHbfResourceManager for concurrent I/O sharing:
    - Media (channel aggregate BW) and interconnect are shared resources
    - Each I/O queues on both resources (serial within, parallel across)
    - For concurrency=1: matches Phase 3D-P0 (max(media, ic))
    - For concurrency=N: resources shared → longer durations

    Per-channel aggregate BW = 51.2 GB/s (vs 512 GB/s envelope → 10x lower).

    Phase 5D P0: virtual_indices + win8 mode → object-aware submission
    (registry WIN mapping / block striping with REAL per-WIN queues).
    """
    if num_bytes <= 0:
        return 0.0

    is_read = (direction == "H2D")

    # Get shared resource manager
    mgr = get_shared_hbf_manager()

    # If manager is not configured (no batch started), fall back to independent model
    if mgr.media_bw <= 0 or mgr.batch_start < 0:
        # Fallback: independent model (no sharing)
        page_latency = (hbf_cfg.read_latency_us if is_read else hbf_cfg.write_latency_us) * 1e-6
        channels = hbf_cfg.channels
        per_channel_bw = hbf_cfg.per_channel_bw_bytes_per_s
        page_bytes = hbf_cfg.page_bytes
        planes_per_channel = hbf_cfg.planes_per_channel
        per_channel_bytes = num_bytes / channels
        channel_transfer = per_channel_bytes / per_channel_bw
        per_channel_pages = per_channel_bytes / page_bytes
        pages_per_plane = max(1.0, per_channel_pages / planes_per_channel)
        page_latency_total = pages_per_plane * page_latency
        channel_completion = max(channel_transfer, page_latency_total)
        aggregate_bw = (hbf_cfg.read_bandwidth_gb * 1e9 if is_read
                        else hbf_cfg.write_bandwidth * 1e9)
        aggregate_bound = num_bytes / aggregate_bw
        media_service = max(channel_completion, aggregate_bound)
        interconnect_bw_gb = float(os.environ.get("HISIM_HBF_INTERCONNECT_BW_GB", "51.2"))
        interconnect_time = num_bytes / (interconnect_bw_gb * 1e9)
        return max(media_service, interconnect_time)

    # Submit to shared resource manager
    # Arrival = current simulation clock
    try:
        from hisim.simulation.manager import StateManager
        arrival = StateManager.get_global_clock()
    except Exception:
        arrival = mgr.batch_start

    global _LAST_HBF_IO_STATS
    # Phase 5D P0: object-aware path when virtual slots are available
    if (getattr(mgr, "media_model", "") == "win8" and mgr.win_pool is not None
            and virtual_indices is not None and len(virtual_indices) > 0):
        bpt = int(num_bytes) // max(1, len(virtual_indices)) \
            if num_bytes and len(virtual_indices) else 0
        wst = mgr.win_pool.submit_object(
            arrival, virtual_indices, bpt, is_read=(direction == "H2D"))
        stats = wst
        _LAST_HBF_IO_STATS = wst   # preserve real sub-WIN timing and identity
        return wst.total
    stats = mgr.submit(arrival, int(num_bytes), is_read=is_read)
    _LAST_HBF_IO_STATS = stats
    return stats.total


def _record_hbf_backup_timing(num_bytes: float, num_tokens: int = 0,
                              node_ids=None,
                              virtual_indices=None) -> float:
    """Record HBF backup (L1→HBF write) timing into StateManager.

    Phase 4-Closure P3: also records an I/O trace entry (real KV bytes).
    Phase 5D P0: virtual_indices (HBF slot ids) enable object-aware WIN
    submission (registry mapping, per-WIN striping).
    """
    dur = _hbf_transfer_duration("D2H", num_bytes,
                                 virtual_indices=virtual_indices,
                                 num_tokens=num_tokens)
    StateManager.inc_hicache_hbf_backup_dur(dur)
    s = _LAST_HBF_IO_STATS
    hbf_io_trace.record(
        op="backup", pool="hbf", tokens=num_tokens, bytes_=num_bytes,
        arrival=_sim_clock(), duration=dur, path="real",
        media_time=getattr(s, "media_service", s.media_time) if s else None,
        ic_time=s.ic_time if s else None,
        queue_wait=s.queue_wait if s else None,
        node_ids=list(node_ids) if node_ids is not None else None,
        details=getattr(s, "trace_details", None) if s else None,
    )
    return dur


def _record_hbf_load_timing(num_bytes: float, num_tokens: int = 0,
                            path: str = "real",
                            virtual_indices=None) -> float:
    """Record HBF load (HBF→L1 read) timing into StateManager.

    Phase 4-Closure P3: I/O trace entry. path="mock_lbk" marks the
    residual load_back record with mock buffer bytes (~56 B/token).
    Phase 5D P0: virtual_indices → object-aware WIN restore (same WINs
    as the backup — registry/static block mapping).
    """
    if path == "mock_lbk":
        hbf_io_trace.record(
            op="load", pool="hbf", tokens=num_tokens, bytes_=num_bytes,
            arrival=_sim_clock(), duration=0.0, path=path)
        return 0.0

    dur = _hbf_transfer_duration("H2D", num_bytes,
                                 virtual_indices=virtual_indices,
                                 num_tokens=num_tokens)
    StateManager.inc_hicache_hbf_load_dur(dur)
    s = _LAST_HBF_IO_STATS
    hbf_io_trace.record(
        op="load", pool="hbf", tokens=num_tokens, bytes_=num_bytes,
        arrival=_sim_clock(), duration=dur, path=path,
        media_time=getattr(s, "media_service", s.media_time) if s else None,
        ic_time=s.ic_time if s else None,
        queue_wait=s.queue_wait if s else None,
        details=getattr(s, "trace_details", None) if s else None,
    )
    return dur


# ---- Host Load source attribution (真实 host_value 归因) ----
# Per-request cache of the HBF/DRAM token split from the actual host_value
# virtual indices collected during match_prefix. Keyed by req.rid.
# Populated by C_UnifiedRadixCacheHook.wrapped_match_prefix, consumed by
# C_SchedulerHook.wrapped_get_new_batch_prefill.
_HOST_LOAD_SPLIT: dict = {}  # rid -> {"hbf_tokens": int, "dram_tokens": int}
_PREFIX_CAP_WARNED = [False]   # Phase 5E.1 P2: one-shot prefix-capacity guard


def _lpf_tree_dump(tree_core, tag: str, max_nodes: int = 60) -> None:
    """Phase 5E P1: dump the radix tree state (device/host coverage per node).

    Only active when HISIM_LPF_DEBUG=1. Settles the A/B/C discrimination:
    A = host_value never written fully; B = stored but match/restore partial;
    C = stored+restored but consumption miscounted.
    """
    import os as _os
    if _os.environ.get("HISIM_LPF_DEBUG", "0") != "1":
        return
    try:
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        ct = ComponentType.FULL
    except Exception:
        ct = 0
    try:
        root = tree_core.root_node
        lines = []
        q = [(root, 0)]
        seen = 0
        while q and seen < max_nodes:
            node, depth = q.pop(0)
            seen += 1
            cd = node.component_data[ct] if hasattr(node, "component_data") \
                else None
            has_dev = cd is not None and getattr(cd, "value", None) is not None
            hv = getattr(cd, "host_value", None) if cd is not None else None
            hv_len = len(hv) if hv is not None else -1
            hv_head = (int(hv[0]), int(hv[-1])) if hv is not None and len(hv) \
                else None
            key = getattr(node, "key", None)
            try:
                key_len = len(key) if key is not None else 0
            except Exception:
                key_len = -1
            key_head = None
            try:
                _kv = list(key)[:2] if key is not None and len(key) else []
                key_head = tuple(int(v) for v in _kv) if _kv else None
            except Exception:
                key_head = repr(key)[:24]
            nid = getattr(node, "id", -1)
            pid = getattr(getattr(node, "parent", None), "id", -1)
            lines.append(
                f"    N{nid} p={pid} d={depth} key_len={key_len} key={key_head} "
                f"dev={int(has_dev)} host_len={hv_len} host={hv_head} "
                f"evicted={int(getattr(node, 'evicted', False))} "
                f"backuped={int(getattr(node, 'backuped', False))} "
                f"hits={getattr(node, 'hit_count', '?')}")
            for c in (getattr(node, "children", None) or {}).values():
                q.append((c, depth + 1))
        print(f"[LPF-TREE {tag}] nodes={seen}: ", flush=True)
        for ln in lines:
            print(ln, flush=True)
    except Exception as e:
        print(f"[LPF-TREE {tag}] dump failed: {e}", flush=True)


def _lpf_match_trace(tree_core, result, rid) -> None:
    """Phase 5E P1: per-match trace of the host-hit walk (finalize path)."""
    import os as _os
    if _os.environ.get("HISIM_LPF_DEBUG", "0") != "1":
        return
    try:
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        ct = ComponentType.FULL
        root = tree_core.root_node
        bm = result.best_match_node
        if isinstance(bm, int):
            bm = tree_core.node_by_id(bm)
        ld = result.last_device_node
        if isinstance(ld, int):
            ld = tree_core.node_by_id(ld)
        node = bm
        walk = []
        hops = 0
        while node is not ld and node is not root and hops < 64:
            cd = node.component_data[ct]
            hv = getattr(cd, "host_value", None)
            walk.append(
                f"N{getattr(node, 'id', -1)}"
                f"(key={len(getattr(node, 'key', []) or [])},"
                f"host={len(hv) if hv is not None else -1},"
                f"evicted={int(getattr(node, 'evicted', False))})")
            node = node.parent
            hops += 1
        try:
            dev_hit = len(result.device_indices)
        except Exception:
            dev_hit = -1
        print(f"[LPF-MATCH rid={rid}] dev_hit={dev_hit} "
              f"host_hit={result.host_hit_length} "
              f"full_kv={result.full_kv_hit_length} "
              f"best={getattr(bm, 'id', -1)} "
              f"last_dev={getattr(ld, 'id', -1)} "
              f"walk={' <- '.join(walk) if walk else '(empty: best==last_dev/root)'}",
              flush=True)
    except Exception as e:
        print(f"[LPF-MATCH rid={rid}] trace failed: {e}", flush=True)


def _collect_host_load_split(tree_core, result, dram_offset):
    """Walk from best_match_node up to last_device_node, collecting the actual
    host_value virtual indices from each evicted node's FULL component.

    Splits the collected indices by dram_offset to determine how many tokens
    are in HBF ([0, dram_offset)) vs DRAM ([dram_offset, total)).

    This mirrors SGLang's FullTreeComponent.finalize_match_result_in_cache
    (host_hit_length computation) and build_hicache_transfers(LOAD_BACK)
    (host_indices collection), but additionally splits by physical medium.

    Phase 5D P0: also returns the actual HBF virtual slot indices (the
    stable object addresses for WIN mapping / restore decomposition).

    Returns (hbf_tokens, dram_tokens, hbf_slots).
    """
    hbf_tokens = 0
    dram_tokens = 0
    hbf_slots = []
    try:
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        ct = ComponentType.FULL
    except Exception:
        ct = 0  # ComponentType.FULL == 0

    try:
        node = tree_core.node_by_id(result.best_match_node)
        last_device = tree_core.node_by_id(result.last_device_node)
        root = tree_core.root_node
        while node is not last_device and node is not root:
            cd = node.component_data.get(ct) if hasattr(node.component_data, 'get') else node.component_data[ct]
            if cd is not None and cd.host_value is not None:
                hv = cd.host_value
                hbf_tokens += int((hv < dram_offset).sum().item())
                dram_tokens += int((hv >= dram_offset).sum().item())
                # Phase 5D P0: collect actual HBF slot indices (stable
                # object addresses — same slots the backup wrote)
                hbf_slots.extend(int(v) for v in hv[hv < dram_offset].tolist())
            node = node.parent
    except Exception:
        pass
    return hbf_tokens, dram_tokens, hbf_slots


class C_EngineHook(BaseHook):
    HOOK_CLASS_NAME = "Engine"
    HOOK_MODULE_NAME = "sglang.srt.entrypoints.engine"

    @classmethod
    def hook(cls, target):
        def hook_clear_hicache_storage(self):
            return self.loop.run_until_complete(
                self.tokenizer_manager.clear_hicache_storage()
            )

        target.clear_hicache_storage = hook_clear_hicache_storage


class C_TokenizerManagerHook(BaseHook):
    HOOK_CLASS_NAME = "TokenizerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.tokenizer_manager"

    @classmethod
    def hook(cls, target):
        original_send_one_request = target._send_one_request

        # When running with blocking mode, send the created time to schedule.
        def wrapped_send_one_request(self, *args):
            import time as _time

            if len(args) >= 1 and args[0].__class__.__name__ == "GenerateReqInput":
                obj = args[0]
                tokenized_obj = args[1] if len(args) >= 2 else None
                if tokenized_obj is not None:
                    custom_params = getattr(
                        tokenized_obj.sampling_params, "custom_params", None
                    )
                    if custom_params and "simulation" in custom_params:
                        custom_params["simulation"]["server_created_time"] = (
                            _time.time()
                        )
            return original_send_one_request(self, *args)

        target._send_one_request = wrapped_send_one_request


class C_ModelRunnerHook(BaseHook):
    HOOK_CLASS_NAME = "ModelRunner"
    HOOK_MODULE_NAME = "sglang.srt.model_executor.model_runner"

    @classmethod
    def hook(cls, target):
        _version_dispatcher = VersionDispatcher()

        def override_initialize(self, *args, **kwargs):
            class MockModel:
                def forward(self):
                    pass

            self.model = MockModel()

            # Attributes that may not be set by the hook but are expected by sglang
            self.use_ngram_embedding = False
            # None matches the non-SWA semantics of sglang >= 0.5.6, where
            # `is_hybrid_swa is not None` is used as the hybrid-SWA check.
            self.is_hybrid_swa = None
            self.is_draft_worker = False
            self.sliding_window_size = None

            self.dtype = self.model_config.dtype
            self.kv_cache_dtype = (
                self.dtype
            )  # FIXME: get kv cache dtype from server args

            model = ConfigManager.get_model_info(self.model_config.hf_config.__dict__)
            hw = ConfigManager.get_accelerator_info()
            config = ConfigManager.get_scheduler_config(
                self.server_args.__dict__,
                "sglang",
                self.model_config.hf_config.__dict__,
            )

            assert model is not None and hw is not None and config is not None

            if self.server_args.max_total_tokens is not None:
                self.max_total_num_tokens = self.server_args.max_total_tokens
            else:
                self.max_total_num_tokens = estimate_kv_cache_pool_capacity(
                    model, hw, config
                )

            if hasattr(self, "page_size") and self.page_size > 1:
                self.max_total_num_tokens = (
                    self.max_total_num_tokens // self.page_size * self.page_size
                )
            self.full_max_total_num_tokens = self.max_total_num_tokens
            self.swa_max_total_num_tokens = self.max_total_num_tokens

            max_num_reqs = min(
                max(
                    int(self.max_total_num_tokens / model.max_seq_len * 512),
                    2048,
                ),
                4096,
            )
            logger.info(
                f"Model runner initialized with {self.max_total_num_tokens} tokens. Maximum number of requests: {max_num_reqs}"
            )

            model_has_mtp_layers = (
                self.model_config.num_nextn_predict_layers is not None
            )
            model_num_layers = (
                self.model_config.num_nextn_predict_layers
                if self.is_draft_worker and model_has_mtp_layers
                else max(
                    self.model_config.num_hidden_layers,
                    self.model_config.num_attention_layers,
                )
            )
            self.start_layer = getattr(self.model, "start_layer", 0)
            self.end_layer = getattr(self.model, "end_layer", model_num_layers)
            self.num_effective_layers = self.end_layer - self.start_layer

            self.req_to_token_pool = MockReqToTokenPool(
                size=max_num_reqs,
                max_context_len=model.max_seq_len,
                device=self.device,
                enable_memory_saver=False,
            )

            self.token_to_kv_pool = MockTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(
                    1  # get_attention_tp_size()
                ),
                head_dim=self.model_config.head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )

            if self.page_size == 1:
                self.token_to_kv_pool_allocator = MockTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=1,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )
            else:
                self.token_to_kv_pool_allocator = MockPagedTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )

            # self.init_memory_pool(50)
            # sglang >= 0.5.10 dereferences attn_backend and several other
            # attributes in ForwardBatch.init_new / downstream code paths even
            # though the forward pass itself is intercepted, so provide stubs.
            self.attn_backend = MockAttentionBackend()
            self.graph_mem_usage = 0
            self.weight_load_mem_usage = 10
            self.lora_manager = None
            self.loader = MockWeightLoader()
            self.canary_manager = None
            self.prefill_aware_swa = False
            self.ngram_embedding_manager = MockNgramEmbeddingManager()
            self.prefill_attention_backend_str = "torch_native"
            self.decode_attention_backend_str = "torch_native"
            self.kv_index_translator = MockKVIndexTranslator()
            try:
                from sglang.srt.layers.sampler import create_sampler

                self.sampler = create_sampler()
            except Exception:
                self.sampler = None
            try:
                from sglang.srt.eplb.expert_location_updater import (
                    ExpertLocationUpdater,
                )

                self.expert_location_updater = ExpertLocationUpdater()
            except Exception:
                self.expert_location_updater = None
            # sglang >= 0.5.10 allocates the memory pools from
            # Scheduler.init_memory_pools via ModelRunner.alloc_memory_pool,
            # which would overwrite the mock pools. A non-None
            # memory_pool_config makes init_target_memory_pool() skip it.
            try:
                from sglang.srt.model_executor.pool_configurator import (
                    MemoryPoolConfig,
                )

                self.memory_pool_config = MemoryPoolConfig(
                    max_total_num_tokens=self.max_total_num_tokens
                )
            except Exception:
                import types as _types

                self.memory_pool_config = _types.SimpleNamespace(
                    max_total_num_tokens=self.max_total_num_tokens
                )

            self.max_running_requests = min(
                (
                    self.max_total_num_tokens // 2
                    if self.server_args.max_running_requests is None
                    else self.server_args.max_running_requests
                    // (
                        self.server_args.dp_size
                        if self.server_args.enable_dp_attention
                        else 1
                    )
                ),
                self.req_to_token_pool.size,
            )

            return

        def wrapped_forward_v1(self, *args, **kwargs):
            batch = args[0]
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            output = LogitsProcessorOutput(
                next_token_logits=torch.empty(
                    size=(batch.batch_size, self.model_config.vocab_size),
                    device=self.device,
                )
            )

            return output, False

        _version_dispatcher.register_method(
            "forward", ["0.5.6", "0.5.6.post1", "0.5.6.post2"], wrapped_forward_v1
        )

        def wrapped_forward_v2(self, *args, **kwargs):
            from sglang.srt.model_executor.model_runner import ModelRunnerOutput

            output, _ = wrapped_forward_v1(self, *args, **kwargs)
            return ModelRunnerOutput(
                logits_output=output,
                can_run_graph=False,
                expert_distribution_metrics=None,
            )

        _version_dispatcher.register_method(
            "forward",
            ["0.5.7", "0.5.8", "0.5.8.post1", "0.5.9", "0.5.19"],
            wrapped_forward_v2,
        )

        def wrapped_sample(self, *args, **kwargs):
            logits = args[0]
            ids = torch.ones(
                size=(logits.next_token_logits.shape[0],),
                device=self.device,
                dtype=torch.int64,
            )
            return ids

        def wrapped_compute_logprobs_only(*args, **kwargs):
            return None

        target.initialize = override_initialize
        target.forward = _version_dispatcher.get_compat_method("forward")
        target.sample = wrapped_sample
        target.compute_logprobs_only = wrapped_compute_logprobs_only

        # sglang >= 0.5.10 runs torch distributed init inside
        # ModelRunner.__init__ (before `initialize`), and its CPU bootstrap
        # path calls torch.ops.sgl_kernel.* ops that are unavailable in
        # CPU-only builds without the sgl-kernel extension. Neutralize
        # `_init_cpu_threads_env` for the duration of the call.
        if hasattr(target, "init_torch_distributed"):
            _original_init_torch_distributed = target.init_torch_distributed

            def wrapped_init_torch_distributed(self, *args, **kwargs):
                try:
                    from sglang.srt.distributed import bootstrap as _bootstrap
                except ImportError:
                    return _original_init_torch_distributed(self, *args, **kwargs)
                if not hasattr(_bootstrap, "_init_cpu_threads_env"):
                    return _original_init_torch_distributed(self, *args, **kwargs)
                _original = _bootstrap._init_cpu_threads_env
                _bootstrap._init_cpu_threads_env = lambda *a, **k: None
                try:
                    return _original_init_torch_distributed(self, *args, **kwargs)
                finally:
                    _bootstrap._init_cpu_threads_env = _original

            target.init_torch_distributed = wrapped_init_torch_distributed

        # sglang >= 0.5.10 builds real attention backends from
        # Scheduler.init_all_attention_backends; the mock forward pass only
        # needs the stub backend installed above.
        if hasattr(target, "init_attention_backends"):
            target.init_attention_backends = lambda self, *args, **kwargs: None
        # sglang >= 0.5.10 triggers cuda graph capture from
        # Scheduler.init_all_cuda_graphs; graphs are irrelevant for the
        # simulation (the forward pass is intercepted).
        if hasattr(target, "init_cuda_graphs"):
            target.init_cuda_graphs = lambda self, *args, **kwargs: None

        # sglang >= 0.5.10 host pools allocate their buffers with
        # cudaHostRegister by default, which is unavailable on CPU-only
        # builds; route CPU allocations to plain (non-pinned) host tensors.
        try:
            import torch as _torch
            from sglang.srt.mem_cache.pool_host import common as _host_common

            def _cpu_host_alloc(
                dims,
                dtype,
                device,
                pin_memory,
                allocator=None,
                registration_granularity_bytes=None,
            ):
                return _torch.empty(dims, dtype=dtype, device=device, pin_memory=False)

            _host_common.ALLOC_MEMORY_FUNCS.setdefault("cpu", _cpu_host_alloc)
        except Exception:  # pragma: no cover - older sglang has no pool_host pkg
            pass


class C_HiCacheController(BaseHook):
    HOOK_CLASS_NAME = "HiCacheController"
    HOOK_MODULE_NAME = "sglang.srt.managers.cache_controller"

    KV_CACHE_BYTES: int = None
    KV_CACHE_BYTES_FULL: int = None
    DISK_READ_BANDWIDTH_BYTES: float = None
    DISK_WRITE_BANDWIDTH_BYTES: float = None

    _ssd_simulator = None
    _ssd_simulator_initialized = False
    _tier_sims = None
    _tier_bandwidths = None
    _tier_write_bandwidths = None
    _tier_read_mode = "serial"
    _tier_write_mode = "parallel"

    @classmethod
    def _get_ssd_simulator(cls):
        if not cls._ssd_simulator_initialized:
            cls._init_ssd_simulator()
        return cls._ssd_simulator

    @classmethod
    def _init_ssd_simulator(cls):
        cls._ssd_simulator_initialized = True
        env = os.environ.get("HISIM_SSD_SIM", "")

        # Check for multi-tier config
        try:
            from hisim.simulation.manager import ConfigManager

            platform_cfg = ConfigManager.get_platform_config()
            tiers = platform_cfg.storage_tiers
            cls._tier_read_mode = platform_cfg.tier_read_mode
            cls._tier_write_mode = platform_cfg.tier_write_mode
        except Exception:
            tiers = []

        if tiers:
            cls._tier_sims = {}
            cls._tier_bandwidths = {}
            cls._tier_write_bandwidths = {}
            for tier in tiers:
                cls._tier_bandwidths[tier.name] = tier.read_bandwidth_gb * 1e9
                wbg = tier.write_bandwidth_gb or tier.read_bandwidth_gb
                cls._tier_write_bandwidths[tier.name] = wbg * 1e9
                if env in ("1", "default", "true", "True", "simpyssd", "simpyssd_multi"):
                    try:
                        from simpyssd import SSDTimingSimulator

                        cls._tier_sims[tier.name] = SSDTimingSimulator(tier.ssd_config)
                        get_logger().info(
                            f"SSD simulator enabled for tier '{tier.name}' with config: {tier.ssd_config}"
                        )
                    except Exception as e:
                        get_logger().warning(
                            f"Failed to init SSD simulator for tier '{tier.name}': {e}"
                        )
            return

        # Single-tier fallback (existing behavior)
        if not env:
            return
        try:
            if env in ("1", "default", "true", "True", "simpyssd"):
                from simpyssd import SSDTimingSimulator
                import simpyssd as _pkg

                cfg_path = os.path.join(
                    os.path.dirname(_pkg.__file__), "configs", "es3500p_pcie5.cfg"
                )
                cls._ssd_simulator = SSDTimingSimulator(cfg_path)
                get_logger().info(
                    f"SSD simulator enabled (SimpySSD) with config: {cfg_path}"
                )
            else:
                from simpyssd import SSDTimingSimulator

                cls._ssd_simulator = SSDTimingSimulator(env)
                get_logger().info(
                    f"SSD simulator enabled (SimpySSD) with config: {env}"
                )
        except Exception as e:
            get_logger().warning(f"Failed to init SSD simulator: {e}")

    @staticmethod
    def _calc_prefetch_pages_ssd(
        sim, required_pages: int, page_size_byte: int, max_dur: float
    ) -> tuple[float, float]:
        # KV_CACHE_BYTES_FULL = total across all TP workers
        full_bytes = C_HiCacheController.KV_CACHE_BYTES_FULL or page_size_byte
        # Try timing-aware API first
        if hasattr(sim, 'submit_read'):
            total_bytes = required_pages * full_bytes
            # read_arrival = T_{N-1} + l2_load_{N-1}
            #              = T_N - inference_{N-1} - l2_backup_{N-1} - l3_overflow_{N-1}
            read_arrival = (
                StateManager.get_global_clock()
                - StateManager.get_last_inference_dur()
                - StateManager.get_last_l2_backup_dur()
                - StateManager.get_last_l3_write_overflow()
            )
            avail_snapshot = sim.get_all_avail_time()
            comp, lat, wait, exec_t = sim.submit_read(
                read_arrival, total_bytes, page_size_byte, is_sequential=True
            )
            # Restore all chips' avail_time to undo read's effect (sglang aborts prefetch)
            sim.set_all_avail_time(avail_snapshot)
            # Separate wait and exec for accurate partial completion
            remaining_budget = max_dur - wait
            if remaining_budget <= 0:
                return 0.0, max_dur
            if exec_t <= remaining_budget:
                return float(required_pages), lat
            ratio = remaining_budget / exec_t
            completed = max(1.0, required_pages * ratio)
            return completed, max_dur
        # Fallback: original non-timing API
        total_bytes = required_pages * full_bytes
        read_time = sim.predict_read_time(
            total_bytes, page_size_byte, is_sequential=True
        )
        if read_time <= max_dur:
            return float(required_pages), read_time
        ratio = max_dur / read_time
        completed = max(1, int(required_pages * ratio))
        return float(completed), max_dur

    @staticmethod
    def _calc_prefetch_multi_tier(
        per_tier_hits: dict, page_size_byte: int, max_dur: float
    ) -> tuple[float, float]:
        # read_arrival = T_{N-1} + l2_load_{N-1}
        #              = T_N - inference_{N-1} - l2_backup_{N-1} - l3_overflow_{N-1}
        read_arrival = (
            StateManager.get_global_clock()
            - StateManager.get_last_inference_dur()
            - StateManager.get_last_l2_backup_dur()
            - StateManager.get_last_l3_write_overflow()
        )

        def _read_tier(tier_name, hit_count, arrival):
            """Submit a read to one tier, return (wait, exec_t, lat)."""
            full_bytes = C_HiCacheController.KV_CACHE_BYTES_FULL or page_size_byte
            total_bytes = hit_count * full_bytes
            sim = (
                C_HiCacheController._tier_sims.get(tier_name)
                if C_HiCacheController._tier_sims
                else None
            )
            if sim and hasattr(sim, 'submit_read'):
                avail_snapshot = sim.get_all_avail_time()
                _, lat, wait, exec_t = sim.submit_read(
                    arrival, total_bytes, page_size_byte, is_sequential=True
                )
                sim.set_all_avail_time(avail_snapshot)
                return wait, exec_t, lat
            elif sim:
                t = sim.predict_read_time(
                    total_bytes, page_size_byte, is_sequential=True
                )
                return 0.0, t, t
            else:
                bw = (
                    C_HiCacheController._tier_bandwidths.get(tier_name, 8e9)
                    if C_HiCacheController._tier_bandwidths
                    else 8e9
                )
                t = total_bytes / bw
                return 0.0, t, t

        active = {n: c for n, c in per_tier_hits.items() if c > 0}
        if not active:
            return 0.0, 0.0

        if C_HiCacheController._tier_read_mode == "serial":
            # Serial: tiers read in priority order, each arrival chains
            # off the previous tier's completion.
            accumulated = 0.0
            completed = 0.0
            for tier_name, hit_count in active.items():
                arrival = read_arrival + accumulated
                wait, exec_t, lat = _read_tier(tier_name, hit_count, arrival)
                budget = max_dur - accumulated
                if budget <= wait:
                    # Can't even finish waiting — this and all later tiers get 0
                    break
                if budget >= lat:
                    # Full completion
                    completed += hit_count
                    accumulated += lat
                else:
                    # Partial: only (budget - wait) time for exec
                    if exec_t > 0:
                        partial = hit_count * (budget - wait) / exec_t
                        completed += max(1.0, partial) if hit_count > 0 else 0
                    return completed, max_dur
            return completed, accumulated
        else:
            # Parallel: all tiers start at the same read_arrival
            tier_results = {}
            for tier_name, hit_count in active.items():
                wait, exec_t, lat = _read_tier(tier_name, hit_count, read_arrival)
                tier_results[tier_name] = (hit_count, wait, exec_t, lat)

            total_completed = sum(c for c, _, _, _ in tier_results.values())
            total_dur = max(lat for _, _, _, lat in tier_results.values())
            if total_dur <= max_dur:
                return float(total_completed), total_dur
            # Partial: each tier independently checks budget after its own wait
            completed = 0.0
            for _, (hit_count, wait, exec_t, lat) in tier_results.items():
                remaining = max_dur - wait
                if remaining <= 0:
                    continue
                if exec_t <= remaining:
                    completed += hit_count
                elif exec_t > 0:
                    completed += max(1.0, hit_count * (remaining / exec_t))
            return completed, max_dur

    @staticmethod
    def calc_prefetch_pages(
        required_pages, page_size_byte: int, max_dur: float, bandwidth: float
    ) -> tuple[float, float]:
        # Multi-tier mode: required_pages is dict[str, int]
        if isinstance(required_pages, dict):
            return C_HiCacheController._calc_prefetch_multi_tier(
                required_pages, page_size_byte, max_dur
            )
        # Single-tier SSD sim
        sim = C_HiCacheController._get_ssd_simulator()
        if sim is not None:
            return C_HiCacheController._calc_prefetch_pages_ssd(
                sim, required_pages, page_size_byte, max_dur
            )
        # Original bandwidth model
        full_bytes = C_HiCacheController.KV_CACHE_BYTES_FULL or page_size_byte
        _prefetch_dur = required_pages * full_bytes / bandwidth
        if _prefetch_dur > max_dur:
            _completed_pages = max(max_dur * bandwidth / full_bytes, 1)
            return _completed_pages, max_dur
        else:
            return required_pages, _prefetch_dur

    @classmethod
    def hook(cls, target):
        def override_backup_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_backup_operation`
            pass

        def override_prefetch_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_prefetch_operation`
            pass

        def handle_backup_operation(self):
            """Data phase only: dequeue, batch_set to kvcm, accumulate write info."""
            if not self.enable_storage:
                return
            total_write_tokens = 0
            per_tier_write_accum = {}
            while True:
                try:
                    operation = self.backup_queue.get(block=False)
                    if operation is None:
                        break

                    if not self.backup_skip:
                        self._page_backup(operation)
                    if hasattr(operation, "host_indices"):
                        total_write_tokens += len(operation.host_indices)
                    if C_HiCacheController._tier_sims:
                        try:
                            per_tier = self.storage_backend.get_last_write_per_tier()
                            for tn, cnt in per_tier.items():
                                per_tier_write_accum[tn] = per_tier_write_accum.get(tn, 0) + cnt
                        except Exception:
                            pass
                    self.ack_backup_queue.put(operation)

                except Empty:
                    break

            # Stash for submit_l3_write_timing() to consume after prefetch
            self._pending_write_data = (per_tier_write_accum, total_write_tokens)

        def submit_l3_write_timing(self):
            """Timing phase: submit_write to SSD sims + update state.
            Called AFTER handle_prefetch_operation so that submit_read
            sees the previous round's write avail_time, not this round's."""
            per_tier_write_accum, total_write_tokens = getattr(
                self, "_pending_write_data", ({}, 0)
            )
            if total_write_tokens <= 0:
                return

            if C_HiCacheController.KV_CACHE_BYTES is None:
                C_HiCacheController.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()
            if C_HiCacheController.KV_CACHE_BYTES_FULL is None:
                C_HiCacheController.KV_CACHE_BYTES_FULL = ConfigManager.get_kv_cache_bytes_full()

            # L3 write starts at T_N (data's L2 backup was 2 rounds ago, already done)
            write_arrival = StateManager.get_global_clock()

            # Multi-tier: per-tier SSD timing with independent engines
            if C_HiCacheController._tier_sims and per_tier_write_accum:
                tier_lats = {}
                accumulated = 0.0
                for tier_name, write_blocks in per_tier_write_accum.items():
                    if write_blocks <= 0:
                        continue
                    write_bytes = write_blocks * C_HiCacheController.KV_CACHE_BYTES_FULL
                    sim = C_HiCacheController._tier_sims.get(tier_name)
                    if C_HiCacheController._tier_write_mode == "serial":
                        arrival = write_arrival + accumulated
                    else:
                        arrival = write_arrival
                    if sim is not None and hasattr(sim, 'submit_write'):
                        _, lat, _, _ = sim.submit_write(
                            arrival, write_bytes,
                            C_HiCacheController.KV_CACHE_BYTES,
                            is_sequential=True
                        )
                        tier_lats[tier_name] = lat
                    elif sim is not None:
                        lat = sim.predict_write_time(
                            write_bytes,
                            C_HiCacheController.KV_CACHE_BYTES,
                            is_sequential=True,
                        )
                        tier_lats[tier_name] = lat
                    else:
                        bw = (
                            C_HiCacheController._tier_write_bandwidths.get(tier_name, 8e9)
                            if C_HiCacheController._tier_write_bandwidths
                            else 8e9
                        )
                        lat = write_bytes / bw
                        tier_lats[tier_name] = lat
                    accumulated += tier_lats[tier_name]

                if C_HiCacheController._tier_write_mode == "serial":
                    write_time = sum(tier_lats.values())
                else:
                    write_time = max(tier_lats.values()) if tier_lats else 0.0
            else:
                # Single-tier fallback
                total_write_bytes = total_write_tokens * C_HiCacheController.KV_CACHE_BYTES_FULL
                sim = C_HiCacheController._get_ssd_simulator()
                if sim is not None and hasattr(sim, 'submit_write'):
                    _, lat, _, _ = sim.submit_write(
                        write_arrival, total_write_bytes,
                        C_HiCacheController.KV_CACHE_BYTES,
                        is_sequential=True
                    )
                    write_time = lat
                elif sim is not None:
                    write_time = sim.predict_write_time(
                        total_write_bytes,
                        C_HiCacheController.KV_CACHE_BYTES,
                        is_sequential=True,
                    )
                else:
                    if C_HiCacheController.DISK_WRITE_BANDWIDTH_BYTES is None:
                        C_HiCacheController.DISK_WRITE_BANDWIDTH_BYTES = (
                            ConfigManager.get_platform_config().disk_write_bandwidth
                        )
                    bw = C_HiCacheController.DISK_WRITE_BANDWIDTH_BYTES or 8e9
                    write_time = total_write_bytes / bw
            StateManager.inc_hicache_l3_backup_dur(write_time)
            logger.debug(
                f"L3 write timing: tokens={total_write_tokens}, "
                f"write_time={write_time:.3e}s"
            )

        def handle_prefetch_operation(self):
            if not self.enable_storage:
                return

            if C_HiCacheController.KV_CACHE_BYTES is None:
                C_HiCacheController.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()
            if C_HiCacheController.KV_CACHE_BYTES_FULL is None:
                C_HiCacheController.KV_CACHE_BYTES_FULL = ConfigManager.get_kv_cache_bytes_full()
            if C_HiCacheController.DISK_READ_BANDWIDTH_BYTES is None:
                C_HiCacheController.DISK_READ_BANDWIDTH_BYTES = (
                    ConfigManager.get_platform_config().disk_read_bandwidth
                )

            direct_mode = False
            try:
                direct_mode = ConfigManager.get_platform_config().enable_l3_direct_to_l1
            except Exception:
                pass

            if not _is_legacy_sglang():
                # sglang >= 0.5.10: the storage query result is released to the
                # stock drain path (prefetch_hit_queue -> host allocation ->
                # io_aux ack -> tree insert) only after the simulated storage
                # read completes, advancing across rounds within the current
                # inference budget (chunked, mirroring the <= 0.5.9 semantics).
                remain_dur = StateManager.get_current_inference_dur()

                # 1. Continue a partially "read" operation from the previous
                #    round. NOTE: the simulated read progress is tracked in
                #    the chunked stash only - `operation.completed_tokens`
                #    belongs to the stock io_aux ack path (real pages read)
                #    and is asserted to stay <= the acked count.
                chunked = getattr(self, "chunked_prefetch_operation", None)
                if chunked is not None:
                    operation = chunked["operation"]
                    storage_hit_count = chunked["storage_hit_count"]
                    per_tier_hits = chunked.get("per_tier_hits")
                    progress = chunked.get("progress", 0)
                    remaining = storage_hit_count - progress
                    if (
                        per_tier_hits
                        and storage_hit_count > 0
                        and C_HiCacheController._tier_sims
                    ):
                        ratio = remaining / storage_hit_count
                        calc_arg = {
                            k: max(0, int(v * ratio))
                            for k, v in per_tier_hits.items()
                        }
                    else:
                        calc_arg = remaining
                    completed_tokens, _prefetch_dur = (
                        C_HiCacheController.calc_prefetch_pages(
                            calc_arg,
                            C_HiCacheController.KV_CACHE_BYTES,
                            remain_dur,
                            C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                        )
                    )
                    operation._hisim_l3_read_service_s = (
                        getattr(operation, "_hisim_l3_read_service_s", 0.0) + _prefetch_dur)
                    # Record L3 read timing for chunked continuation (Phase 2d)
                    if remaining > 0:
                        StateManager.inc_hicache_l3_load_dur(_prefetch_dur)
                    if completed_tokens < remaining:
                        # The read cannot finish within this round's budget;
                        # keep holding the operation and retry next round.
                        chunked["progress"] = progress + int(completed_tokens)
                        C_SchedulerHook._L3_PREFETCHED_TOKENS += int(
                            completed_tokens
                        )
                        return
                    C_SchedulerHook._L3_PREFETCHED_TOKENS += int(remaining)
                    try:
                        req_stats = C_SchedulerHook.REQUEST_STATS[
                            operation.request_id
                        ]
                        req_stats.prefetch_complete_tokens = storage_hit_count
                    except Exception:
                        pass
                    if direct_mode and storage_hit_count > 0:
                        StateManager.inc_l3_direct_load_count(storage_hit_count)
                    _mark_l3_read_timing_complete(operation)
                    self.prefetch_hit_queue.put(operation)
                    setattr(self, "chunked_prefetch_operation", None)

                # 2. Query newly enqueued operations.
                while True:
                    try:
                        operation = self.prefetch_queue.get(block=False)
                        if operation is None:
                            return
                        hash_value, storage_hit_count = self._storage_hit_query(
                            operation
                        )
                        try:
                            req_stats = C_SchedulerHook.REQUEST_STATS[
                                operation.request_id
                            ]
                            req_stats.l3_hit_tokens = storage_hit_count
                            req_stats.l3_exists_tokens = storage_hit_count
                        except Exception:
                            pass
                        C_SchedulerHook._L3_QUERY_COUNT += 1
                        C_SchedulerHook._L3_TOTAL_HITS += storage_hit_count
                        per_tier_hits = None
                        if C_HiCacheController._tier_sims:
                            try:
                                per_tier_hits = (
                                    self.storage_backend.batch_exists_per_tier()
                                )
                                for tn, cnt in per_tier_hits.items():
                                    C_SchedulerHook._L3_HIT_STATS[tn] = (
                                        C_SchedulerHook._L3_HIT_STATS.get(tn, 0)
                                        + cnt
                                    )
                            except Exception:
                                per_tier_hits = None

                        if (
                            self.prefetch_threshold is not None
                            and storage_hit_count < self.prefetch_threshold
                        ):
                            # Not enough benefit: release as terminated; the
                            # stock drain revokes it and frees host memory.
                            operation.mark_terminate()
                            operation.storage_hit_count = 0
                            self.prefetch_hit_queue.put(operation)
                            continue

                        operation.hash_value = hash_value[
                            : (storage_hit_count // self.page_size)
                        ]
                        storage_hit_count = (
                            storage_hit_count // self.page_size * self.page_size
                        )
                        operation.storage_hit_count = storage_hit_count

                        calc_arg = (
                            per_tier_hits if per_tier_hits else storage_hit_count
                        )
                        completed_tokens, _prefetch_dur = (
                            C_HiCacheController.calc_prefetch_pages(
                                calc_arg,
                                C_HiCacheController.KV_CACHE_BYTES,
                                remain_dur,
                                C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                            )
                        )
                        operation._hisim_l3_read_issue_s = StateManager.get_global_clock()
                        operation._hisim_l3_read_service_s = _prefetch_dur
                        # Record L3 read timing metrics (Phase 2d)
                        if storage_hit_count > 0:
                            read_bytes = storage_hit_count * (
                                C_HiCacheController.KV_CACHE_BYTES_FULL
                                or C_HiCacheController.KV_CACHE_BYTES
                            )
                            StateManager.inc_hicache_l3_load_dur(_prefetch_dur)
                            try:
                                rs = C_SchedulerHook.REQUEST_STATS[
                                    operation.request_id
                                ]
                                rs.l3_read_issued_tokens = storage_hit_count
                                rs.l3_read_bytes = read_bytes
                                rs.l3_read_dur_ms = _prefetch_dur * 1000.0
                                # Phase 2d-Closure: event timestamps for causality
                                rs.l3_read_issue_time = StateManager.get_global_clock()
                                # Availability is set at the final budget
                                # release, including chunked continuation.
                                rs.l3_read_complete_time = 0.0
                            except Exception:
                                pass
                        if completed_tokens < storage_hit_count:
                            # Hold the operation and continue the read next
                            # round (chunked prefetch). The progress lives in
                            # the stash; operation.completed_tokens is owned
                            # by the stock io_aux ack path.
                            C_SchedulerHook._L3_PREFETCHED_TOKENS += int(
                                completed_tokens
                            )
                            setattr(
                                self,
                                "chunked_prefetch_operation",
                                {
                                    "operation": operation,
                                    "storage_hit_count": storage_hit_count,
                                    "per_tier_hits": per_tier_hits,
                                    "progress": int(completed_tokens),
                                },
                            )
                            return  # This round's budget is exhausted.

                        C_SchedulerHook._L3_PREFETCHED_TOKENS += storage_hit_count
                        _mark_l3_read_timing_complete(operation)
                        try:
                            req_stats = C_SchedulerHook.REQUEST_STATS[
                                operation.request_id
                            ]
                            req_stats.prefetch_complete_tokens = storage_hit_count
                        except Exception:
                            pass
                        if direct_mode and storage_hit_count > 0:
                            StateManager.inc_l3_direct_load_count(storage_hit_count)
                        self.prefetch_hit_queue.put(operation)
                    except Empty:
                        return
                return

            if direct_mode:
                chunked = getattr(self, "chunked_prefetch_operation", None)
                if chunked is not None:
                    operation = chunked["operation"]
                    storage_hit_count = chunked["storage_hit_count"]
                    operation.completed_tokens = int(storage_hit_count)
                    operation.mark_terminate()
                    setattr(self, "chunked_prefetch_operation", None)
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                    req_stats.prefetch_complete_tokens = operation.completed_tokens
                    StateManager.inc_l3_direct_load_count(storage_hit_count)

                while True:
                    try:
                        operation = self.prefetch_queue.get(block=False)
                        if operation is None:
                            return
                        hash_value, storage_hit_count = self._storage_hit_query(operation)
                        if storage_hit_count > 0:
                            try:
                                req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                                req_stats.l3_hit_tokens = storage_hit_count
                            except Exception:
                                pass
                        if (
                            self.prefetch_threshold is not None
                            and storage_hit_count < self.prefetch_threshold
                        ):
                            operation.mark_terminate()
                            self.append_host_mem_release(operation.host_indices)
                            continue
                        operation.hash_value = hash_value[
                            : (storage_hit_count // self.page_size)
                        ]
                        storage_hit_count = (
                            storage_hit_count // self.page_size * self.page_size
                        )
                        if C_HiCacheController._tier_sims:
                            try:
                                per_tier_hits = self.storage_backend.batch_exists_per_tier()
                                C_SchedulerHook._L3_QUERY_COUNT += 1
                                for tn, cnt in per_tier_hits.items():
                                    C_SchedulerHook._L3_HIT_STATS[tn] = C_SchedulerHook._L3_HIT_STATS.get(tn, 0) + cnt
                                    C_SchedulerHook._L3_TOTAL_HITS += cnt
                            except Exception:
                                pass
                        operation.completed_tokens = int(storage_hit_count)
                        operation.mark_terminate()
                        C_SchedulerHook._L3_PREFETCHED_TOKENS += int(storage_hit_count)
                        req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                        req_stats.prefetch_complete_tokens = operation.completed_tokens
                        self.append_host_mem_release(
                            operation.host_indices[storage_hit_count:]
                        )
                        StateManager.inc_l3_direct_load_count(storage_hit_count)
                    except Empty:
                        return
                return

            # TODO: Overlap schedule
            remain_dur = StateManager.get_current_inference_dur()

            chunked_prefetch_operation = getattr(
                self, "chunked_prefetch_operation", None
            )
            if chunked_prefetch_operation is not None:
                operation = chunked_prefetch_operation["operation"]
                storage_hit_count = chunked_prefetch_operation["storage_hit_count"]
                per_tier_hits = chunked_prefetch_operation.get("per_tier_hits")
                remaining = storage_hit_count - operation.completed_tokens
                if per_tier_hits and storage_hit_count > 0 and C_HiCacheController._tier_sims:
                    ratio = remaining / storage_hit_count
                    calc_arg = {k: max(0, int(v * ratio)) for k, v in per_tier_hits.items()}
                else:
                    calc_arg = remaining
                completed_tokens, prefetch_dur = (
                    C_HiCacheController.calc_prefetch_pages(
                        calc_arg,
                        C_HiCacheController.KV_CACHE_BYTES,
                        remain_dur,
                        C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                    )
                )
                if completed_tokens < storage_hit_count - operation.completed_tokens:
                    operation.completed_tokens += completed_tokens
                    remain_dur = 0
                else:
                    operation.completed_tokens = int(storage_hit_count)
                    operation.mark_terminate()
                    remain_dur -= prefetch_dur
                    setattr(self, "chunked_prefetch_operation", None)
                    # Release host memory after current operation is finished
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                # update request states
                req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                req_stats.prefetch_complete_tokens = operation.completed_tokens

            while remain_dur > 0:
                try:
                    operation = self.prefetch_queue.get(block=False)
                    if operation is None:
                        return

                    hash_value, storage_hit_count = self._storage_hit_query(operation)
                    # Track L3 hit tokens per request (page_size=1, so count=tokens)
                    if storage_hit_count > 0:
                        try:
                            req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                            req_stats.l3_hit_tokens = storage_hit_count
                        except Exception:
                            pass
                    # not to prefetch if not enough benefits
                    if (
                        self.prefetch_threshold is not None
                        and storage_hit_count < self.prefetch_threshold
                    ):
                        operation.mark_terminate()
                        self.append_host_mem_release(operation.host_indices)
                        continue

                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    storage_hit_count = (
                        storage_hit_count // self.page_size * self.page_size
                    )

                    # Get per-tier hits if multi-tier is configured
                    per_tier_hits = None
                    if C_HiCacheController._tier_sims:
                        try:
                            per_tier_hits = self.storage_backend.batch_exists_per_tier()
                            C_SchedulerHook._L3_QUERY_COUNT += 1
                            for tn, cnt in per_tier_hits.items():
                                C_SchedulerHook._L3_HIT_STATS[tn] = C_SchedulerHook._L3_HIT_STATS.get(tn, 0) + cnt
                                C_SchedulerHook._L3_TOTAL_HITS += cnt
                        except Exception:
                            pass
                    calc_arg = per_tier_hits if per_tier_hits else storage_hit_count

                    completed_tokens, prefetch_dur = (
                        C_HiCacheController.calc_prefetch_pages(
                            calc_arg,
                            C_HiCacheController.KV_CACHE_BYTES,
                            remain_dur,
                            C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                        )
                    )
                    if completed_tokens < storage_hit_count:
                        # Continue to prefetch data next time.
                        operation.completed_tokens = int(completed_tokens)
                        C_SchedulerHook._L3_PREFETCHED_TOKENS += int(completed_tokens)
                        setattr(
                            self,
                            "chunked_prefetch_operation",
                            {
                                "operation": operation,
                                "storage_hit_count": storage_hit_count,
                                "per_tier_hits": per_tier_hits,
                            },
                        )
                        remain_dur = 0
                    else:
                        operation.completed_tokens = int(
                            storage_hit_count // self.page_size * self.page_size
                        )
                        operation.mark_terminate()
                        remain_dur -= prefetch_dur
                        C_SchedulerHook._L3_PREFETCHED_TOKENS += int(
                            storage_hit_count // self.page_size * self.page_size
                        )
                    # update request states
                    req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                    req_stats.prefetch_complete_tokens = operation.completed_tokens
                    # Release host memory after current operation is finished
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )

                except Empty:
                    return

        def override_generic_page_set(
            self, hash_values, host_indices, extra_info=None
        ) -> bool:
            # Always pass extra_info to storage_backend.
            host_pool = self.mem_pool_host
            # sglang >= 0.5.10 wraps host pools in a HostPoolGroup; unwrap the
            # primary (KV) pool for page-level access.
            get_pool = getattr(host_pool, "get_pool", None)
            if get_pool is not None:
                try:
                    host_pool = get_pool(None)
                except Exception:
                    try:
                        host_pool = host_pool.entries[0].host_pool
                    except Exception:
                        pass
            data = [
                host_pool.get_data_page(host_indices[i * self.page_size])
                for i in range(len(hash_values))
            ]
            return self.storage_backend.batch_set(hash_values, data, extra_info)

        def handle_prefetch_io_operation(self):
            """Synchronous L3 data-loading phase (Phase 2d).

            Replaces the async ``prefetch_io_aux_func`` thread. Drains
            ``prefetch_buffer`` (populated by sglang's
            ``_drain_and_alloc_storage_hit`` during the preceding
            ``check_hicache_events``) and calls ``_page_transfer`` which
            invokes ``_generic_page_get`` → ``batch_get`` (Metadata-only
            Restore Mock returns dummy pages for existing keys).

            The resulting ``PrefetchAck`` is placed in ``prefetch_sync_queue``
            so that the NEXT ``check_hicache_events`` call's
            ``_drain_ack_prefetch`` processes it and calls
            ``_handle_prefetch_result`` to insert the restored host_indices
            into the radix tree.

            This one-round delay (query → load → ack → insert) mirrors the
            real async prefetch pipeline while keeping timing deterministic.
            """
            if not getattr(self, "enable_storage", False):
                return
            from queue import Empty as _Empty

            loaded_total = 0
            while True:
                try:
                    operation = self.prefetch_buffer.get(block=False)
                    if operation is None:
                        continue
                    # _page_transfer calls _generic_page_get → batch_get.
                    # With the Metadata-only Restore Mock, batch_get returns
                    # dummy pages for existing keys, so completed_pages > 0.
                    # NOTE: HybridCacheController._page_transfer returns None
                    # (only base CacheController returns int). The actual
                    # completed_tokens are tracked via PrefetchAck in
                    # prefetch_sync_queue, processed by _drain_ack_prefetch.
                    returned_pages = self._page_transfer(operation)
                    completed_tokens = getattr(operation, "_hisim_l3_transferred_tokens", None)
                    if completed_tokens is None:
                        completed_tokens = int(returned_pages or 0) * self.page_size
                    completed_pages = completed_tokens // self.page_size

                    from sglang.srt.managers.cache_controller import PrefetchAck
                    self.prefetch_sync_queue.put(
                        PrefetchAck(
                            rid=operation.request_id,
                            completed_req=True,
                            operation=operation,
                        )
                    )
                    # Track L3 read-completed tokens for metrics
                    try:
                        req_stats = C_SchedulerHook.REQUEST_STATS.get(
                            operation.request_id
                        )
                        if req_stats is not None:
                            req_stats.l3_read_completed_tokens = (
                                completed_pages * self.page_size
                            )
                    except Exception:
                        pass
                    loaded_total += completed_pages * self.page_size
                except _Empty:
                    break
            if loaded_total > 0:
                logger.debug(
                    f"L3 read-completed: {loaded_total} tokens loaded "
                    f"(prefetch_buffer drained synchronously)"
                )

        def override_prefetch_io_aux_func(self, *args, **kwargs):
            # Async IO thread: no-op. The data loading is driven by
            # handle_prefetch_io_operation from the hooked
            # check_hicache_events (scheduler-thread lockstep, deterministic).
            pass

        # The async backup/prefetch threads must not run: the data/timing
        # phases are driven by handle_backup_operation /
        # handle_prefetch_operation from the hooked check_hicache_events
        # instead (scheduler-thread lockstep, deterministic timing). On
        # sglang >= 0.5.10 HybridCacheController overrides backup_thread_func
        # (shadowing this no-op); C_HybridCacheControllerHook handles that.
        target.prefetch_thread_func = override_prefetch_thread_func
        target.backup_thread_func = override_backup_thread_func
        target.prefetch_io_aux_func = override_prefetch_io_aux_func
        target.handle_backup_operation = handle_backup_operation
        target.handle_prefetch_operation = handle_prefetch_operation
        target.handle_prefetch_io_operation = handle_prefetch_io_operation
        target.submit_l3_write_timing = submit_l3_write_timing
        target._generic_page_set = override_generic_page_set
        _wrap_l3_hbf_materialization(target)

        # sglang >= 0.5.10 performs L2 transfers synchronously via
        # start_writing/start_loading (CUDA kernels + events). The mock host
        # pool holds no real KV data, so the "transfer" is instantaneous:
        # keep the stock bookkeeping but complete the ack events right away.
        if not _is_legacy_sglang() and hasattr(target, "start_writing"):
            def mock_start_writing(self):
                from sglang.srt.managers.cache_controller import (
                    CacheOperation,
                    HiCacheAck,
                )
                from sglang.srt.utils import get_device_module

                if len(self.write_queue) == 0:
                    return
                op = CacheOperation.merge_ops(self.write_queue)
                host_indices, device_indices, pool_transfers = (
                    self._move_write_operation(op)
                )
                self.write_queue.clear()
                device_module = get_device_module()
                start_event = device_module.Event(enable_timing=False)
                finish_event = device_module.Event(enable_timing=False)
                start_event.record()
                finish_event.record()
                write_bytes = self._transfer_num_bytes(op)
                self.ack_write_queue.append(
                    HiCacheAck(
                        start_event=start_event,
                        finish_event=finish_event,
                        node_ids=op.node_ids,
                        num_tokens=len(op.device_indices),
                        timing_enabled=False,
                        num_tokens_by_pool=self._num_tokens_by_pool(op),
                        num_bytes=write_bytes,
                    )
                )
                # L2 write-through (D2H) timing — split HBF vs DRAM if adapter active
                # Phase 3D fix: use REAL KV bytes per token (ConfigManager.get_kv_cache_bytes),
                # NOT mock buffer bytes from _transfer_num_bytes (which returns ~56 bytes/token
                # because the mock host pool uses tiny uint8 placeholder tensors).
                from hisim.simulation.manager import ConfigManager
                real_bytes_per_token = ConfigManager.get_kv_cache_bytes() or 32768
                hbf_adapter = getattr(self, "_hbf_adapter_instance", None)
                if hbf_adapter is not None and hasattr(hbf_adapter, "_is_hbf"):
                    # Split host_indices by pool, record timing separately
                    h_indices = op.host_indices
                    hbf_mask = h_indices < hbf_adapter.dram_offset
                    hbf_count = int(hbf_mask.sum().item())
                    dram_count = len(h_indices) - hbf_count
                    if len(h_indices) > 0:
                        hbf_bytes = hbf_count * real_bytes_per_token
                        dram_bytes = dram_count * real_bytes_per_token
                        if hbf_count > 0:
                            # Phase 5D P0: actual HBF virtual slots →
                            # object-aware WIN submission
                            hbf_indices = h_indices[hbf_mask]
                            hbf_dur = _record_hbf_backup_timing(
                                hbf_bytes, num_tokens=hbf_count,
                                node_ids=op.node_ids,
                                virtual_indices=hbf_indices.tolist()
                                if hasattr(hbf_indices, "tolist")
                                else list(hbf_indices))
                            print(f"[HBF-BACKUP] {hbf_count}tk→HBF "
                                  f"({hbf_bytes/1e6:.2f}MB) dur={hbf_dur*1e6:.1f}μs",
                                  flush=True)
                            # Phase 5E.1 P2 guardrail: prefix-capacity rule.
                            # An object >50% of the host pool leaves <2 slots
                            # per object → concurrent-reuse waves evict LIVE
                            # objects (retraction cascade, 5E.1 root cause).
                            if not _PREFIX_CAP_WARNED[0] and hbf_adapter is not None:
                                _PREFIX_CAP_WARNED[0] = True
                                host_total = getattr(hbf_adapter, "logical_size",
                                                     0) or 0
                                if host_total and \
                                        hbf_count * 2 > host_total:
                                    print(
                                        f"[HBF-WARN] prefix-capacity rule: "
                                        f"object {hbf_count}tk > 50% of host "
                                        f"pool ({host_total}tk) — host holds "
                                        f"<2 such objects; concurrent reuse "
                                        f"will evict live objects and fall "
                                        f"back to partial L3 hits. Size host "
                                        f">= (live+reinsert) x prefix and "
                                        f"max_total >= n_reuse x prefix.",
                                        flush=True)
                        if dram_count > 0:
                            dram_dur = _record_l2_backup_timing(
                                dram_bytes, num_tokens=dram_count)
                            print(f"[DRAM-BACKUP] {dram_count}tk→DRAM "
                                  f"({dram_bytes/1e6:.2f}MB) dur={dram_dur*1e6:.1f}μs",
                                  flush=True)
                else:
                    # No HBF adapter: use real bytes for DRAM backup too
                    real_total = len(op.host_indices) * real_bytes_per_token if hasattr(op, 'host_indices') else write_bytes
                    _record_l2_backup_timing(real_total)

            def mock_start_loading(self):
                from sglang.srt.managers.cache_controller import (
                    CacheOperation,
                    HiCacheAck,
                )
                from sglang.srt.utils import get_device_module

                if len(self.load_queue) == 0:
                    return -1
                producer_id = self.layer_done_counter.update_producer()
                op = CacheOperation.merge_ops(self.load_queue)
                host_indices, device_indices, pool_transfers = self._move_op_indices(
                    op
                )
                self.load_queue.clear()
                producer_event = self.layer_done_counter.events[producer_id]
                producer_event.start_event.record()
                # Simulate the per-layer completion of the host->device copy.
                for layer_index in range(self.layer_num):
                    producer_event.complete(layer_index)
                device_module = get_device_module()
                start_event = device_module.Event(enable_timing=False)
                finish_event = device_module.Event(enable_timing=False)
                start_event.record()
                finish_event.record()
                load_bytes = self._transfer_num_bytes(op)
                self.ack_load_queue.append(
                    HiCacheAck(
                        start_event=start_event,
                        finish_event=finish_event,
                        node_ids=op.node_ids,
                        num_tokens=len(op.device_indices),
                        timing_enabled=False,
                        num_tokens_by_pool=self._num_tokens_by_pool(op),
                        num_bytes=load_bytes,
                    )
                )
                # L2 load-back (H2D) timing — split HBF vs DRAM if adapter active
                hbf_adapter = getattr(self, "_hbf_adapter_instance", None)
                if hbf_adapter is not None and hasattr(hbf_adapter, "_is_hbf"):
                    h_indices = op.host_indices
                    hbf_mask = h_indices < hbf_adapter.dram_offset
                    hbf_count = int(hbf_mask.sum().item())
                    dram_count = len(h_indices) - hbf_count
                    if len(h_indices) > 0:
                        bytes_per_token = load_bytes / len(h_indices)
                        hbf_bytes = hbf_count * bytes_per_token
                        dram_bytes = dram_count * bytes_per_token
                        if hbf_count > 0:
                            # mock buffer bytes (~56 B/token) — residual path
                            hbf_dur = _record_hbf_load_timing(
                                hbf_bytes, num_tokens=hbf_count,
                                path="mock_lbk")
                            print(f"[HBF-LOAD] {hbf_count}tk←HBF "
                                  f"({hbf_bytes/1e6:.2f}MB) dur={hbf_dur*1e6:.1f}μs",
                                  flush=True)
                        if dram_count > 0:
                            dram_dur = _record_l2_load_timing(
                                dram_bytes, dram_count, path="mock_lbk")
                            print(f"[DRAM-LOAD] {dram_count}tk←DRAM "
                                  f"({dram_bytes/1e6:.2f}MB) dur={dram_dur*1e6:.1f}μs",
                                  flush=True)
                else:
                    _record_l2_load_timing(
                        load_bytes, len(op.device_indices), path="mock_lbk")
                return producer_id

            target.start_writing = mock_start_writing
            target.start_loading = mock_start_loading


class C_HiRadixCacheHook(BaseHook):
    HOOK_CLASS_NAME = "HiRadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.hiradix_cache"

    @classmethod
    def hook(cls, target):
        original_check_hicache_events = target.check_hicache_events
        original_reset = target.reset

        def wrapped_reset(self):
            if hasattr(self, "cache_controller"):
                self.cache_controller.handle_backup_operation()
                self.cache_controller.submit_l3_write_timing()
            original_reset(self)

        def override_init(self, params, server_args):
            if server_args.hicache_io_backend == "direct":
                # FIXME: move this logic into server_args parsing
                if server_args.hicache_mem_layout == "page_first":
                    server_args.hicache_mem_layout = "page_first_direct"
                    logger.warning(
                        "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                    )

            self.page_size = params.page_size
            self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
            # Replace the host pool
            self.token_to_kv_pool_host = MockTokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                pin_memory=False,
                device="cpu",
            )

            self.tp_group = params.tp_cache_group
            self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
            self.enable_storage = server_args.hicache_storage_backend is not None
            self.enable_storage_metrics = self.enable_storage and params.enable_metrics

            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                server_args.hicache_storage_backend_extra_config
            )
            self.prefetch_threshold = prefetch_threshold
            self.prefetch_timeout_base = prefetch_timeout_base
            self.prefetch_timeout_per_page = (
                self.page_size / 1024 * prefetch_timeout_per_ki_token
            )
            self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
            # TODO: support more timeout check functions
            self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
            self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

            HiCacheController = getattr(
                importlib.import_module("sglang.srt.managers.cache_controller"),
                "HiCacheController",
            )
            try:
                StorageMetricsCollector = getattr(
                    importlib.import_module("sglang.srt.metrics.collector"),
                    "StorageMetricsCollector",
                )
            except (ImportError, ModuleNotFoundError):
                StorageMetricsCollector = None

            self.load_cache_event = threading.Event()
            controller_kwargs = dict(
                load_cache_event=self.load_cache_event,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=self.prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
            )
            # sglang >= 0.5.10 introduced extra constructor parameters.
            import inspect as _inspect

            _ctor_params = _inspect.signature(HiCacheController.__init__).parameters
            if "enable_storage_metrics" in _ctor_params:
                controller_kwargs["enable_storage_metrics"] = (
                    self.enable_storage_metrics
                )
            if "host_memory_mode" in _ctor_params:
                controller_kwargs["host_memory_mode"] = getattr(
                    server_args, "hicache_host_memory_mode", "cache"
                )
            self.cache_controller = HiCacheController(
                params.token_to_kv_pool_allocator,
                self.token_to_kv_pool_host,
                self.page_size,
                self.tp_group,
                **controller_kwargs,
            )
            if self.enable_storage_metrics and StorageMetricsCollector is not None:
                # TODO: support pp
                labels = {
                    "storage_backend": server_args.hicache_storage_backend,
                    "tp_rank": self.cache_controller.tp_rank,
                    "dp_rank": self.cache_controller.dp_rank,
                }
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)

            # Record the nodes with ongoing write-through
            self.ongoing_write_through = {}
            # Record the node segments with ongoing load-back
            self.ongoing_load_back = {}
            # Record the ongoing prefetch requests
            self.ongoing_prefetch = {}
            self.ongoing_backup = {}
            # TODO: Dynamically adjust the threshold
            self.write_through_threshold = (
                1 if server_args.hicache_write_policy == "write_through" else 2
            )
            self.load_back_threshold = 10
            # Version: 0.5.9
            self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
            self.evictable_host_leaves = set()
            # super().__init__(params=params)
            target.__mro__[1].__init__(self, params=params)

        def wrapped_check_hicache_events(self, *args, **kwargs):
            # HBF L1.5 adapter installation (one-time, env-gated)
            _install_hbf_adapter_if_enabled(self)
            # Phase 3D-P1: Reset shared HBF resource manager for new batch
            mgr = get_shared_hbf_manager()
            if mgr.media_bw > 0:
                try:
                    mgr.begin_batch(StateManager.get_global_clock())
                except Exception:
                    pass
            # Phase 3E: Reset shared DRAM resource manager for new batch
            dram_mgr = get_shared_dram_manager()
            if dram_mgr.dram_bw > 0:
                try:
                    dram_mgr.begin_batch(StateManager.get_global_clock())
                except Exception:
                    pass
            # Phase 3E-Closure: Reset shared downstream if enabled
            if is_shared_downstream_enabled():
                try:
                    reset_downstream(StateManager.get_global_clock())
                except Exception:
                    pass
            # 1-3. Data + timing phases (storage backend only); see
            # _drive_hicache_timing_phases for the phase ordering rationale.
            _drive_hicache_timing_phases(self)
            result = original_check_hicache_events(self, *args, **kwargs)
            # 3b. L3 data-loading phase: drain prefetch_buffer synchronously
            # AFTER stock _drain_and_alloc_storage_hit has allocated host
            # memory and BEFORE the next round's _drain_ack_prefetch.
            _drive_prefetch_io_phase(self)
            # 4. Fix (legacy sglang <= 0.5.9 only): In CPU simulation, CUDA
            # events in ack_write_queue and ack_load_queue never complete, so
            # writing_check() and loading_check() never process their queues.
            # We process them here as a fallback, skipping the CUDA event
            # check. On sglang >= 0.5.10 the transfers are synchronous with
            # instantly-completed mock events, and the stock checks above
            # already reap the acks.
            if (
                _is_legacy_sglang()
                and hasattr(self, 'cache_controller')
                and self.enable_storage
            ):
                cc = self.cache_controller
                # ack_write_queue fallback: mirrors sglang's writing_check()
                # normal path (hiradix_cache.py:698-706). Must call
                # dec_lock_ref to unlock the node so it can be evicted later.
                # Without dec_lock_ref, backed-up nodes stay locked in L1
                # forever, host_hit_length is always 0, and L2 load_back is
                # never triggered.
                n_write_acks = 0
                for ack in cc.ack_write_queue:
                    for ack_id in ack.node_ids:
                        node = self.ongoing_write_through.pop(ack_id, None)
                        if node is not None:
                            n_write_acks += 1
                            self.dec_lock_ref(node)
                            self.write_backup_storage(node)
                cc.ack_write_queue.clear()
                # ack_load_queue fallback: mirrors sglang's loading_check()
                # (hiradix_cache.py:708-721). Without this, nodes in
                # ongoing_load_back stay locked forever, eventually
                # exhausting evictable L1 space and breaking L2 load_back.
                n_load_acks = 0
                for ack in cc.ack_load_queue:
                    for ack_id in ack.node_ids:
                        end_node = self.ongoing_load_back.pop(ack_id, None)
                        if end_node is not None:
                            n_load_acks += 1
                            self.dec_lock_ref(end_node)
                cc.ack_load_queue.clear()
                if n_write_acks or n_load_acks:
                    logger.debug(
                        f"check_hicache_events fallback: write_acks={n_write_acks}, "
                        f"load_acks={n_load_acks}, "
                        f"ongoing_write_through={len(self.ongoing_write_through)}, "
                        f"ongoing_load_back={len(self.ongoing_load_back)}, "
                        f"backup_queue={cc.backup_queue.qsize() if hasattr(cc.backup_queue, 'qsize') else 'N/A'}, "
                        f"prefetch_queue={cc.prefetch_queue.qsize() if hasattr(cc.prefetch_queue, 'qsize') else 'N/A'}"
                    )
            return result

        target.__init__ = override_init
        target.check_hicache_events = wrapped_check_hicache_events
        target.reset = wrapped_reset

        # Node lifecycle tracking (enabled via HISIM_NODE_TRACK env var)
        if os.environ.get("HISIM_NODE_TRACK", ""):
            _orig_write_backup = target.write_backup
            _orig_evict_backuped = target._evict_backuped
            _orig_evict_regular = target._evict_regular
            _orig_evict_host = target.evict_host
            _orig_load_back = target.load_back

            def _tracked_write_backup(self, node, write_back=False):
                r = _orig_write_backup(self, node, write_back)
                if r > 0:
                    logger.info(f"NODE_BACKUP id={node.id} key_len={len(node.key)} backuped={node.backuped}")
                return r

            def _tracked_evict_backuped(self, node):
                r = _orig_evict_backuped(self, node)
                logger.info(f"NODE_EVICT_L1 id={node.id} key_len={len(node.key)} backuped={node.backuped} evicted={node.evicted}")
                return r

            def _tracked_evict_regular(self, node):
                logger.info(f"NODE_DELETE id={node.id} key_len={len(node.key)} backuped={node.backuped}")
                return _orig_evict_regular(self, node)

            def _tracked_evict_host(self, num_tokens):
                # Log before eviction (nodes still in tree)
                leaves = list(self.evictable_host_leaves)
                r = _orig_evict_host(self, num_tokens)
                # Log what changed
                after = list(self.evictable_host_leaves)
                removed = [n for n in leaves if n not in after]
                for n in removed:
                    logger.info(f"NODE_EVICT_L2 id={n.id} key_len={len(n.key)} backuped={n.backuped} evicted={n.evicted}")
                return r

            def _tracked_load_back(self, node, mem_quota=None):
                logger.info(f"NODE_LOAD_BACK_START id={node.id} key_len={len(node.key)} evicted={node.evicted} backuped={node.backuped}")
                r = _orig_load_back(self, node, mem_quota)
                if r is not None:
                    logger.info(f"NODE_LOAD_BACK_OK id={node.id} tokens={len(r)}")
                else:
                    logger.info(f"NODE_LOAD_BACK_FAIL id={node.id}")
                return r

            target.write_backup = _tracked_write_backup
            target._evict_backuped = _tracked_evict_backuped
            target._evict_regular = _tracked_evict_regular
            target.evict_host = _tracked_evict_host
            target.load_back = _tracked_load_back
            logger.info("Node lifecycle tracking enabled (HISIM_NODE_TRACK)")


class C_HybridCacheControllerHook(BaseHook):
    """sglang >= 0.5.10: HybridCacheController overrides backup_thread_func
    (sidecar pools must be backed up on every TP rank), shadowing the no-op
    installed on HiCacheController. The storage data phase is driven by
    handle_backup_operation from the hooked check_hicache_events instead, so
    the thread must stay idle to avoid double-consuming backup_queue."""

    HOOK_CLASS_NAME = "HybridCacheController"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller"

    @classmethod
    def hook(cls, target):
        def override_backup_thread_func(self, *args, **kwargs):
            # Async thread: perform no action.
            # The action will be performed by `handle_backup_operation`.
            pass

        target.backup_thread_func = override_backup_thread_func
        _wrap_l3_hbf_materialization(target)


class C_UnifiedRadixCacheHook(BaseHook):
    """sglang >= 0.5.10 builds the hierarchical tree cache as
    UnifiedRadixCache (registry.create_tree_cache) instead of HiRadixCache,
    so C_HiRadixCacheHook never fires. Drive the same storage data/timing
    phases from UnifiedRadixCache.check_hicache_events / reset."""

    HOOK_CLASS_NAME = "UnifiedRadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.unified_radix_cache"

    @classmethod
    def hook(cls, target):
        original_check_hicache_events = target.check_hicache_events
        original_reset = target.reset

        def wrapped_check_hicache_events(self, *args, **kwargs):
            _install_hbf_adapter_if_enabled(self)
            _drive_hicache_timing_phases(self)
            # Log HBF adapter stats only when HISIM_HBF_STATS=1 (env-gated
            # to avoid flooding during warmup/decode).
            _hbf_adapter = getattr(self, "_hbf_adapter_instance", None)
            if _hbf_adapter is not None and os.environ.get("HISIM_HBF_STATS", "0") in ("1", "true", "True"):
                _hbf_adapter.log_stats()
            result = original_check_hicache_events(self, *args, **kwargs)
            # L3 data-loading phase: drain prefetch_buffer synchronously
            _drive_prefetch_io_phase(self)
            return result

        def wrapped_reset(self):
            # Flush the pending storage writes and their write timing before
            # the tree state is destroyed.
            cc = getattr(self, "cache_controller", None)
            if cc is not None and getattr(cc, "enable_storage", False):
                cc.handle_backup_operation()
                cc.submit_l3_write_timing()
            _HOST_LOAD_SPLIT.clear()
            original_reset(self)

        # Hook match_prefix to capture the actual host_value virtual indices
        # of L2-only (evicted-from-L1) tokens, enabling precise HBF/DRAM load
        # source attribution instead of proportional estimation.
        original_match_prefix = target.match_prefix

        def wrapped_match_prefix(self, params, *args, **kwargs):
            result = original_match_prefix(self, params, *args, **kwargs)
            _rid_dbg = getattr(params, "req", None)
            _rid_dbg = getattr(_rid_dbg, "rid", None) if _rid_dbg \
                is not None else "?"
            _lpf_match_trace(getattr(self, "tree_core", None), result,
                             _rid_dbg)
            if result.host_hit_length > 0:
                _lpf_tree_dump(getattr(self, "tree_core", None),
                               f"after-match-rid={_rid_dbg}")
            if result.host_hit_length > 0:
                # Get the HBF adapter to access dram_offset
                adapter = getattr(self, "_hbf_adapter_instance", None)
                if adapter is None:
                    cc = getattr(self, "cache_controller", None)
                    if cc is not None:
                        adapter = getattr(cc, "_hbf_adapter_instance", None)
                if adapter is not None:
                    tree_core = getattr(self, "tree_core", None)
                    if tree_core is not None:
                        hbf_tok, dram_tok, hbf_slots = _collect_host_load_split(
                            tree_core, result, adapter.dram_offset
                        )
                        rid = getattr(params, "req", None)
                        rid = getattr(rid, "rid", None) if rid is not None else None
                        if rid is not None:
                            _HOST_LOAD_SPLIT[rid] = {
                                "hbf_tokens": hbf_tok,
                                "dram_tokens": dram_tok,
                                "hbf_slots": hbf_slots,
                            }
            return result

        target.check_hicache_events = wrapped_check_hicache_events
        target.reset = wrapped_reset
        target.match_prefix = wrapped_match_prefix

        # Wrap evict_host to trace capacity before/after and victim medium
        original_evict_host = target.evict_host

        def wrapped_evict_host(self, num_tokens, component_type=0, *args, **kwargs):
            adapter = getattr(self, "_hbf_adapter_instance", None)
            if adapter is None:
                cc = getattr(self, "cache_controller", None)
                if cc is not None:
                    adapter = getattr(cc, "_hbf_adapter_instance", None)
            if adapter is not None:
                cs_before = adapter.capacity_stats()
                hbf_free_before = cs_before["hbf_available"]
                dram_free_before = cs_before["dram_available"]
                freed_before = adapter.conservation_stats()
                hbf_freed_before = freed_before["hbf_free_tokens"]
                dram_freed_before = freed_before["dram_free_tokens"]

                result = original_evict_host(self, num_tokens, component_type, *args, **kwargs)

                cs_after = adapter.capacity_stats()
                freed_after = adapter.conservation_stats()
                hbf_reclaimed = freed_after["hbf_free_tokens"] - hbf_freed_before
                dram_reclaimed = freed_after["dram_free_tokens"] - dram_freed_before

                logger.info(
                    f"[EVICT-HOST] requested={num_tokens} freed={result} "
                    f"HBF: {hbf_free_before}→{cs_after['hbf_available']} "
                    f"(reclaimed={hbf_reclaimed}) "
                    f"DRAM: {dram_free_before}→{cs_after['dram_available']} "
                    f"(reclaimed={dram_reclaimed})"
                )
                return result
            return original_evict_host(self, num_tokens, component_type, *args, **kwargs)

        target.evict_host = wrapped_evict_host


class C_StorageBackendFactory(BaseHook):
    HOOK_CLASS_NAME = "StorageBackendFactory"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.storage.backend_factory"

    @classmethod
    def hook(cls, target):
        def override_create_backend(cls, *args, **kwargs):
            logger.info("Creating hijacked cache storage backend.")
            return MockHiCacheStorage()

        target.create_backend = override_create_backend


class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    INFERENCE_PREDICTOR: InferTimePredictor = None

    REQUEST_STATS: dict[str, RequestStats] = defaultdict(RequestStats)
    ITERATION_STATS: list[dict] = []
    LAST_CPU_TS: float = 0
    LAST_FLUSH_TS: float = 0
    HISIM_BATCH: HisimScheduleBatch = None

    OVERLAP_SCHEDULE: bool = False

    SIM_MODE = MockSimulationMode(Envs.simulation_mode())
    OFFLINE_RECV_ALL_REQUEST: bool = False
    FUTURE_QUEUE: list[
        tuple[float, int, RequestStats]
    ] = []  # tuple(created time, salt, request)

    # --- Dynamic completion feedback (session-aware replay) ---
    # Active only for requests carrying `session` metadata in their simulation
    # params (e.g. weka_trace rounds). The next round of a session is released
    # at max(static time, previous round's actual finish + recorded gap).
    # session -> {"round_idx", "recorded_finish", "finished", "finish_vtime"}
    SESSION_STATE: dict = {}
    # session -> [(enqueue_time, salt, req)] rounds whose static time arrived
    # but whose session predecessor has not finished yet
    PARKED: dict = {}
    # rid -> session, populated when a session-aware request is released
    REQ_SESSION: dict = {}

    # L2/L3 cache hit statistics
    _L3_HIT_STATS: dict = {}  # tier_name -> total hit blocks
    _L3_QUERY_COUNT: int = 0  # total batch_exists calls
    _L3_TOTAL_HITS: int = 0  # total hit blocks across all tiers
    _L3_PREFETCHED_TOKENS: int = 0  # total tokens actually prefetched from L3
    _L2_HIT_TOKENS: int = 0  # total tokens matched in L2 (host_hit_length sum)
    _L2_HIT_COUNT: int = 0  # number of requests with host_hit_length > 0

    SCHEDULE_REQ_STATS = []

    # ------------------------------------------------------------------
    # Helpers shared by the legacy (<= 0.5.9) and new (>= 0.5.10) request
    # paths. `sampling_params` is a `SamplingParams` object on old versions
    # and may arrive as a plain dict on newer msgspec-based IO structs.
    # ------------------------------------------------------------------

    @staticmethod
    def get_simulation_params(req):
        sp = getattr(req, "sampling_params", None)
        if sp is None:
            return None
        if isinstance(sp, dict):
            custom_params = sp.get("custom_params")
        else:
            custom_params = getattr(sp, "custom_params", None)
        if not isinstance(custom_params, dict):
            return None
        return custom_params.get("simulation")

    @staticmethod
    def get_max_new_tokens(req):
        sp = getattr(req, "sampling_params", None)
        if isinstance(sp, dict):
            return sp.get("max_new_tokens")
        return getattr(sp, "max_new_tokens", None)

    @staticmethod
    def get_extend_input_len(req):
        """Number of new tokens processed by the current extend batch."""
        extend_range = getattr(req, "extend_range", None)  # sglang >= 0.5.10
        if extend_range is not None:
            return extend_range.length
        extend_input_len = getattr(req, "extend_input_len", None)  # sglang <= 0.5.9
        if extend_input_len is not None:
            return extend_input_len
        return len(req.origin_input_ids) - len(req.prefix_indices)

    @staticmethod
    def is_chunked_req(scheduler, req):
        is_chunked = getattr(req, "is_chunked", None)  # sglang <= 0.5.9
        if is_chunked is not None:
            return is_chunked != 0
        return req is scheduler.chunked_req  # sglang >= 0.5.10

    @classmethod
    def record_request_stats(cls, req, now):
        req_stats = cls.REQUEST_STATS[req.rid]
        req_stats.rid = req.rid
        req_stats.input_length = len(req.input_ids)
        req_stats.output_length = cls.get_max_new_tokens(req)
        simulation_args = cls.get_simulation_params(req)
        if cls.SIM_MODE == MockSimulationMode.BLOCKING:
            if simulation_args is None:
                # Health-check / warm-up request without simulation parameters.
                logger.warning(
                    "The request's simulation parameters are missing, using the "
                    "current time as the creation time."
                )
                simulation_args = {}
            req_stats.created_time = simulation_args.get("server_created_time", now)
            req_stats.last_event_time = req_stats.created_time
            req_stats.queue_start = now
        elif cls.SIM_MODE == MockSimulationMode.OFFLINE:
            if simulation_args is None:
                # Warm-up requests are excluded from the statistics.
                cls.REQUEST_STATS.pop(req.rid, None)
                return
            req_stats.created_time = simulation_args["created_time"]
            req_stats.last_event_time = req_stats.created_time
            # Align with the real queue start timestamp if queue_start is not None. For debugging only.
            queue_start = simulation_args.get("queue_start")
            if queue_start is not None:
                StateManager.advance_global_clock_to(queue_start)
            req_stats.queue_start = StateManager.get_global_clock()

    @classmethod
    def _offline_collect_and_release(cls, scheduler, recv_reqs, dispatch):
        """OFFLINE-mode buffering for sglang >= 0.5.10.

        Generate requests are collected into FUTURE_QUEUE until all of them
        (`total_request`) have arrived; afterwards the requests whose simulated
        arrival time is due are released to the dispatcher. Returns the
        requests that should be dispatched in this call."""
        if not cls.OFFLINE_RECV_ALL_REQUEST:
            gen_requests = []
            extra_requests = []
            time.sleep(0.05)  # waiting requests

            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    gen_requests.append(req)
                else:
                    # Such as: /profile_start, /flush_cache, etc.
                    extra_requests.append(req)

            for req in gen_requests:
                sim_params = cls.get_simulation_params(req)
                if sim_params is None:
                    # There are some warm-up requests when starting the server without --skip-server-warmup.
                    extra_requests.append(req)
                    logger.warning(
                        "Failed to extract the simulation parameters required for simulation from the request. Ignore this warning if the request is a warm-up request."
                    )
                    continue
                if sim_params.get("queue_start"):
                    logger.debug(
                        "Add request to waiting queue with custom queue start timestamp."
                    )
                cls.FUTURE_QUEUE.append(
                    (
                        sim_params.get("queue_start") or sim_params["created_time"],
                        time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                        req,
                    )
                )

            if len(cls.FUTURE_QUEUE) != 0:
                _, _, gen_req = cls.FUTURE_QUEUE[-1]
                total_request = cls.get_simulation_params(gen_req)["total_request"]

                if len(cls.FUTURE_QUEUE) == total_request:
                    cls.OFFLINE_RECV_ALL_REQUEST = True
                    heapq.heapify(cls.FUTURE_QUEUE)
                    logger.info("All requests received. Starting simulation now.")
                else:
                    logger.info(
                        f"Offline simulation mode enabled. {total_request} requests expected in total. "
                        f"Received {len(cls.FUTURE_QUEUE)} requests so far."
                    )

            if extra_requests:
                # Schedule the extra requests immediately.
                dispatch(scheduler, extra_requests)
            return []

        # All generate requests have been buffered. Pass through newly arrived
        # requests (flush, abort, health check, etc.) and release the buffered
        # generate requests that are due.
        released = list(recv_reqs)
        current_timestamp = StateManager.get_global_clock()
        while len(cls.FUTURE_QUEUE) > 0:
            enqueue_time, _, req = cls.FUTURE_QUEUE[0]
            if enqueue_time > current_timestamp:
                break
            released.append(req)
            heapq.heappop(cls.FUTURE_QUEUE)
        return released

    @classmethod
    def dump_profile_results(cls):
        """Dump the simulation results (metrics/iteration/request files) and
        reset the state. Shared by the legacy `Scheduler.profile` hook and the
        sglang >= 0.5.10 `SchedulerProfilerManager` hook."""
        stats: list[RequestStats] = []
        for item in cls.REQUEST_STATS.values():
            if item.rid is not None and item.input_length > 0:
                stats.append(item)

        stats = sorted(stats, key=lambda req: req.created_time)

        output_dir = Envs.output_dir()
        os.makedirs(output_dir, exist_ok=True)

        if len(stats) > 0:
            # Remove warmup requests.
            if len(stats) > Envs.num_warmup():
                metrics_stats = stats[Envs.num_warmup() :]
            else:
                metrics_stats = stats

            min_created_time = metrics_stats[0].created_time
            # Align timestamps
            for item in stats:
                item.created_time -= min_created_time
                item.queue_start -= min_created_time
                item.queue_end -= min_created_time
                item.last_event_time -= min_created_time

            metrics = calc_metrics(metrics_stats)
            metrics["time_cost"] = time.time() - cls.LAST_FLUSH_TS

            try:
                with open(f"{output_dir}/metrics.json", "w") as f:
                    f.write(json.dumps(metrics, cls=CustomJsonEncoder) + "\n")

                with open(f"{output_dir}/iteration.jsonl", "w") as f:
                    for item in cls.ITERATION_STATS:
                        f.write(json.dumps(item) + "\n")

                with open(f"{output_dir}/request.jsonl", "w") as f:
                    for item in stats:
                        f.write(json.dumps(asdict(item)) + "\n")

                # Phase 4-Closure P3: dump HBF/DRAM KV I/O trace
                try:
                    extra = {}
                    adapter = _HBF_ADAPTER_REF
                    if adapter is not None:
                        extra["host_pool_occupancy"] = adapter.capacity_stats()
                    hbf_io_trace.dump_trace(output_dir, extra=extra)
                except Exception as e:
                    logger.debug(f"io trace dump failed: {e}")

                # Phase 5D: dump 8-WIN pool summary (per-WIN stats + object
                # registry + conservation) when win8 mode is active
                try:
                    from hisim.simulation.hbf_shared_resource import (
                        get_shared_hbf_manager)
                    _mgr = get_shared_hbf_manager()
                    if getattr(_mgr, "media_model", "") == "win8" \
                            and _mgr.win_pool is not None:
                        wp = _mgr.win_pool
                        wp_summary = wp.summary()
                        try:
                            wp_summary["objects"] = wp.object_summary()
                            wp_summary["conservation"] = \
                                wp.registry.verify_conservation()
                        except Exception:
                            pass
                        with open(os.path.join(
                                output_dir, "win_pool_summary.json"), "w") as f:
                            json.dump(wp_summary, f, indent=1)
                        # Preserve complete logical events and physical WIN
                        # sub-I/Os for reconciliation/replay, including evicted
                        # objects whose lifetime counters leave the registry.
                        if hasattr(wp, "get_io_events"):
                            with open(os.path.join(
                                    output_dir, "win_io_events.jsonl"), "w") as f:
                                for event in wp.get_io_events():
                                    f.write(json.dumps(
                                        hbf_io_trace.normalize_details(event)) + "\n")
                        wp.config.dump_effective_json(os.path.join(
                            output_dir, "hbf_win_effective_config.json"))
                        logger.info(
                            f"[WIN-POOL] objects={wp_summary.get('total_ios')} "
                            f"span={wp_summary.get('batch_span_s')}s "
                            f"read={wp_summary.get('read_throughput_gbps')} "
                            f"write={wp_summary.get('write_throughput_gbps')} "
                            f"GB/s — win_pool_summary.json dumped")
                except Exception as e:
                    logger.debug(f"win pool summary dump failed: {e}")

                logger.info(f"Simulation results saved to {output_dir}.")

                # Log L2/L3 cache hit statistics
                from hisim.simulation.sglang.sglang_mock_class import MockHiCacheStorage
                logger.info(
                    f"Storage stats: batch_set={MockHiCacheStorage._DEBUG_BATCH_SET_COUNT}, "
                    f"batch_exists={MockHiCacheStorage._DEBUG_BATCH_EXISTS_COUNT}, "
                    f"hits={MockHiCacheStorage._DEBUG_BATCH_EXISTS_HITS}"
                )
                if cls._L3_QUERY_COUNT > 0:
                    logger.info(
                        f"L3 cache stats: queries={cls._L3_QUERY_COUNT}, "
                        f"total_hits={cls._L3_TOTAL_HITS} blocks, "
                        f"prefetched={cls._L3_PREFETCHED_TOKENS} tokens, "
                        f"per_tier={cls._L3_HIT_STATS}"
                    )
                logger.info(
                    f"L2 cache stats: hit_requests={cls._L2_HIT_COUNT}, "
                    f"hit_tokens={cls._L2_HIT_TOKENS}"
                )
                # Log per-request L1/L2/L3 hit breakdown + TTFT/TPOT
                for item in metrics_stats:
                    # Phase 2d-Closure: Token conservation fix
                    # - final_reused_tokens (cached_tokens) = device-resident prefix match ONLY
                    #   (does NOT include host_hit_length in sglang 0.5.19)
                    # - l2_hit_tokens (host_hit_length) = host-resident prefix match
                    #   (includes L3-restored tokens that were loaded back to device)
                    # - l3_hit_tokens = existence check count (NOT consumed, NOT in prefill)
                    # - recomputed_tokens = actual prefill tokens from scheduler
                    # Mutual exclusivity: L1_device + L2_host + Recomputed = actual_input
                    # Phase 3A: Fix L1_device computation.
                    # cached_tokens (final_reused_tokens) in sglang 0.5.19 may or
                    # may NOT include host_hit_length, depending on timing:
                    # - L2 host hit (loaded via _prefetch_kvcache BEFORE cached_tokens
                    #   is set): cached_tokens INCLUDES host_hit → subtract
                    # - L3 restore (loaded AFTER cached_tokens is set):
                    #   cached_tokens does NOT include host_hit → don't subtract
                    # Heuristic: if cached_tokens >= l2_hit, subtract (L2 case);
                    # otherwise don't (L3 case).
                    cached_tokens = item.final_reused_tokens
                    l2_host = item.l2_hit_tokens          # host-resident (incl L3-restored)
                    if cached_tokens >= l2_host:
                        l1_device = cached_tokens - l2_host  # L2 host hit included in cached
                    else:
                        l1_device = cached_tokens            # L3 restore, not included
                    l3_exists = getattr(item, 'l3_exists_tokens', 0)
                    l3_consumed = getattr(item, 'l3_consumed_tokens', 0)
                    recomputed = getattr(item, 'recomputed_tokens', 0)
                    actual_input = getattr(item, 'actual_input_tokens', item.input_length)
                    # Verify conservation: L1 + L2 + Recomputed ≈ actual_input
                    conservation_sum = l1_device + l2_host + recomputed
                    lats = item.gen_token_latencies
                    ttft_ms = lats[0] * 1000 if lats else 0
                    tpot_ms = lats[1] * 1000 if len(lats) > 1 else 0
                    hbf_hit = getattr(item, 'hbf_hit_tokens', 0)
                    dram_hit = getattr(item, 'dram_hit_tokens', 0)
                    l3_issued = getattr(item, 'l3_read_issued_tokens', 0)
                    l3_completed = getattr(item, 'l3_read_completed_tokens', 0)
                    l3_restored = getattr(item, 'l3_restored_tokens', 0)
                    l3_read_ms = getattr(item, 'l3_read_dur_ms', 0.0)
                    sched_pf = getattr(item, 'scheduler_prefill_tokens', 0)
                    hbf_load_b = getattr(item, 'hbf_load_bytes', 0)
                    dram_load_b = getattr(item, 'dram_load_bytes', 0)
                    logger.info(
                        f"Req #{item.rid}: input={item.input_length}(actual={actual_input}) "
                        f"L1={l1_device} HBF={hbf_hit} DRAM={dram_hit} "
                        f"L3[exists={l3_exists} consumed={l3_consumed}] "
                        f"Recomputed={recomputed} SchedPF={sched_pf} "
                        f"(conservation: {l1_device}+{l2_host}+{recomputed}={conservation_sum}/{actual_input}) "
                        f"TTFT={ttft_ms:.1f}ms TPOT={tpot_ms:.1f}ms "
                        f"pf_complete={item.prefetch_complete_tokens} "
                        f"L3[issued={l3_issued} completed={l3_completed} "
                        f"restored={l3_restored} read_ms={l3_read_ms:.2f}] "
                        f"LoadBytes[HBF={hbf_load_b} DRAM={dram_load_b}]"
                    )
                    # Phase 2d-Closure: verify L3 read causality
                    ri = getattr(item, 'l3_read_issue_time', 0)
                    rc = getattr(item, 'l3_read_complete_time', 0)
                    rst = getattr(item, 'l3_restore_commit_time', 0)
                    cn = getattr(item, 'l3_consume_time', 0)
                    if rc > 0:
                        causal_ok = cn >= rc if cn > 0 else True
                        logger.info(
                            f"  L3-CAUSALITY rid={item.rid}: "
                            f"issue={ri*1e3:.3f}ms complete={rc*1e3:.3f}ms "
                            f"restore={rst*1e3:.3f}ms consume={cn*1e3:.3f}ms "
                            f"consume>=complete={'OK' if causal_ok else 'VIOLATION'}"
                        )

            except Exception as e:
                logger.error(f"Failed to dump results. Error: {e}")
        else:
            logger.warning("No request statistics available.")

        StateManager.reset()
        cls.REQUEST_STATS.clear()
        cls.ITERATION_STATS.clear()
        cls.LAST_CPU_TS = 0
        cls.LAST_FLUSH_TS = time.time()
        cls.OFFLINE_RECV_ALL_REQUEST = False
        cls._L3_HIT_STATS = {}
        cls._L3_QUERY_COUNT = 0
        cls._L3_TOTAL_HITS = 0
        cls._L3_PREFETCHED_TOKENS = 0
        cls._L2_HIT_TOKENS = 0
        cls._L2_HIT_COUNT = 0
        cls.SESSION_STATE.clear()
        cls.PARKED.clear()
        cls.REQ_SESSION.clear()

        ProfileReqOutput = getattr(
            importlib.import_module("sglang.srt.managers.io_struct"),
            "ProfileReqOutput",
        )
        result = {
            "total_request": len(stats),
            "output_directory": output_dir,
        }

        # sglang >= 0.5.10 defines ProfileReqOutput as a kw-only msgspec
        # struct, so keyword arguments are required.
        return ProfileReqOutput(success=True, message=json.dumps(result))

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_get_new_batch_prefill = target.get_new_batch_prefill
        original_run_batch = target.run_batch
        original_process_batch_result = target.process_batch_result
        original_event_loop_normal = target.event_loop_normal

        # sglang >= 0.5.10 removed `Scheduler.recv_requests`: requests are now
        # received by a separate `SchedulerRequestReceiver` and dispatched via
        # `Scheduler.process_input_requests`.
        legacy_request_path = hasattr(target, "recv_requests")
        if legacy_request_path:
            original_recv_requests = target.recv_requests
        else:
            original_process_input_requests = target.process_input_requests

        def override_event_loop_overlap(self, *args, **kwargs):
            # To reduce the complexity of the simulation, the overlapping schedule is not needed.
            return original_event_loop_normal(self, *args, **kwargs)

        def wrapped_init(self, *args, **kwargs):
            # Disable overlap schedule
            server_args = get_obj_from_args(
                "sglang.srt.server_args.ServerArgs", *args, **kwargs
            )
            C_SchedulerHook.OVERLAP_SCHEDULE = not getattr(
                server_args, "disable_overlap_schedule", False
            )
            # sglang >= 0.5.10 resolves ServerArgs into a read-only struct, so
            # the mutation may fail there; `launch_server` disables overlap
            # before resolution and the instance attribute is overridden below.
            try:
                setattr(server_args, "disable_overlap_schedule", True)
            except (AttributeError, TypeError):
                pass
            logger.debug(
                f"Overlap schedule simulation mode: {C_SchedulerHook.OVERLAP_SCHEDULE}."
            )

            original_init(self, *args, **kwargs)

            # sglang >= 0.5.10: ServerArgs may be read-only, so disable the
            # overlap schedule on the scheduler instance instead. This drives
            # both dispatch_event_loop() and the run_batch() branch selection.
            if getattr(self, "enable_overlap", False):
                self.enable_overlap = False
                self.enable_overlap_mlx = False

            try:
                model = ConfigManager.get_model_info(
                    self.model_config.hf_config.__dict__
                )
                hw = ConfigManager.get_accelerator_info()
                sched_config = ConfigManager.get_scheduler_config(
                    self.server_args.__dict__,
                    "sglang",
                    self.model_config.hf_config.__dict__,
                )
                ConfigManager.set_scheduler_config(sched_config)
                ConfigManager.set_model_info(model)

                C_SchedulerHook.INFERENCE_PREDICTOR = (
                    ConfigManager.get_inference_time_predictor(model, hw, sched_config)
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize inference time predictor. Error: {e}"
                )
                raise e

        def wrapped_recv_requests(self, *args, **kwargs) -> list:
            recv_reqs = []

            if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                recv_reqs.extend(original_recv_requests(self, *args, **kwargs))
            elif C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                # Initializing
                if not C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST:
                    gen_requests = []
                    extra_requests = []
                    time.sleep(0.05)  # waiting requests

                    reqs = original_recv_requests(self, *args, **kwargs)

                    for req in reqs:
                        if req.__class__.__name__ == "TokenizedGenerateReqInput":
                            gen_requests.append(req)
                        else:
                            # Such as: /profile_start, /flush_cache, etc.
                            extra_requests.append(req)

                    # Add requests to future queue
                    for req in gen_requests:
                        sim_params = None
                        if req.sampling_params.custom_params is not None:
                            sim_params = req.sampling_params.custom_params.get(
                                "simulation"
                            )
                        if sim_params is None:
                            # There are some warm-up requests when starting the server without --skip-server-warmup.
                            extra_requests.append(req)
                            logger.warning(
                                "Failed to extract the simulation parameters required for simulation from the request. Ignore this warning if the request is a warm-up request."
                            )
                            continue
                        if sim_params.get("queue_start"):
                            logger.debug(
                                "Add request to waiting queue with custom queue start timestamp."
                            )

                        C_SchedulerHook.FUTURE_QUEUE.append(
                            (
                                sim_params.get("queue_start")
                                or sim_params["created_time"],
                                time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                                req,
                            )
                        )

                    if len(C_SchedulerHook.FUTURE_QUEUE) != 0:
                        _, _, gen_req = C_SchedulerHook.FUTURE_QUEUE[-1]
                        total_request = gen_req.sampling_params.custom_params[
                            "simulation"
                        ]["total_request"]

                        if len(C_SchedulerHook.FUTURE_QUEUE) == total_request:
                            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = True
                            heapq.heapify(C_SchedulerHook.FUTURE_QUEUE)
                            logger.info(
                                "All requests received. Starting simulation now."
                            )
                        else:
                            logger.info(
                                f"Offline simulation mode enabled. {total_request} requests expected in total. Received {len(C_SchedulerHook.FUTURE_QUEUE)} requests so far."
                            )

                    if len(extra_requests) != 0:
                        # Schedule the extra requests immediately.
                        return extra_requests
                else:
                    # Extra requests include: flush request, abort request, etc.
                    recv_reqs.extend(original_recv_requests(self, *args, **kwargs))

                # Process the arrived requests only after all requests have been added to the future queue
                current_timestamp = StateManager.get_global_clock()
                while (
                    C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST
                    and len(C_SchedulerHook.FUTURE_QUEUE) > 0
                ):
                    enqueue_time, salt, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    if enqueue_time > current_timestamp:
                        break
                    heapq.heappop(C_SchedulerHook.FUTURE_QUEUE)

                    # Dynamic completion feedback: a round whose session
                    # predecessor is still in flight is parked (not blocking
                    # other sessions); a finished predecessor may shift the
                    # release time by the accumulated virtual delay.
                    sim_args = (
                        req.sampling_params.custom_params.get("simulation")
                        if req.sampling_params.custom_params is not None
                        else None
                    )
                    session = sim_args.get("session") if sim_args else None
                    if session is not None:
                        st = C_SchedulerHook.SESSION_STATE.get(session)
                        if st is not None and not st["finished"]:
                            # Previous round of this session still in flight.
                            C_SchedulerHook.PARKED.setdefault(
                                session, []
                            ).append((enqueue_time, salt, req))
                            continue
                        if st is not None and st["finished"]:
                            # Shift by the virtual delay of the predecessor:
                            # release = static + max(0, actual - recorded).
                            shift = max(
                                0.0,
                                st["finish_vtime"] - st["recorded_finish"],
                            )
                            if shift > 0 and enqueue_time + shift > current_timestamp:
                                heapq.heappush(
                                    C_SchedulerHook.FUTURE_QUEUE,
                                    (enqueue_time + shift, salt, req),
                                )
                                continue
                        C_SchedulerHook.SESSION_STATE[session] = {
                            "round_idx": sim_args.get("round_idx"),
                            "recorded_finish": sim_args.get(
                                "recorded_finish", enqueue_time
                            ),
                            "finished": False,
                            "finish_vtime": 0.0,
                        }
                    recv_reqs.append(req)

            now = time.time()
            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.rid = req.rid
                    req_stats.input_length = len(req.input_ids)
                    req_stats.output_length = req.sampling_params.max_new_tokens
                    simulation_args = req.sampling_params.custom_params["simulation"]
                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        if "server_created_time" not in simulation_args:
                            logger.warning(
                                "The request's creation time is missing, which may cause the TTFT to be inaccurate."
                            )
                        req_stats.created_time = simulation_args.get(
                            "server_created_time", now
                        )
                        req_stats.last_event_time = req_stats.created_time
                        req_stats.queue_start = now
                    elif C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                        req_stats.created_time = simulation_args["created_time"]
                        req_stats.last_event_time = req_stats.created_time
                        # Align with the real queue start timestamp if queue_start is not None. For debugging only.
                        queue_start = simulation_args.get("queue_start")
                        if queue_start is not None:
                            StateManager.advance_global_clock_to(queue_start)
                        req_stats.queue_start = StateManager.get_global_clock()
                        # Register the session for the completion-event hook
                        # (dynamic completion feedback).
                        if simulation_args.get("session") is not None:
                            C_SchedulerHook.REQ_SESSION[req.rid] = (
                                simulation_args["session"]
                            )

            if recv_reqs and C_SchedulerHook.LAST_CPU_TS == 0:
                C_SchedulerHook.LAST_CPU_TS = time.time()

            return recv_reqs

        def wrapped_process_input_requests(self, recv_reqs, *args, **kwargs):
            """sglang >= 0.5.10 request path.

            Replaces the legacy `recv_requests` interception: requests are
            dispatched through `process_input_requests`, so the OFFLINE
            buffering (collect all generate requests first, release them by
            simulation timestamp) and the per-request statistics recording are
            implemented around the original dispatcher."""
            recv_reqs = list(recv_reqs) if recv_reqs else []

            if C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                recv_reqs = C_SchedulerHook._offline_collect_and_release(
                    self, recv_reqs, original_process_input_requests
                )

            now = time.time()
            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    C_SchedulerHook.record_request_stats(req, now)

            if recv_reqs and C_SchedulerHook.LAST_CPU_TS == 0:
                C_SchedulerHook.LAST_CPU_TS = time.time()

            if recv_reqs:
                original_process_input_requests(self, recv_reqs, *args, **kwargs)

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            result = original_get_new_batch_prefill(self, *args, **kwargs)
            # sglang >= 0.5.10 returns NextBatchPlan(batch_to_run, running_batch)
            # instead of the ScheduleBatch itself.
            new_batch = getattr(result, "batch_to_run", result)
            pending_batch = _scheduler_has_pending_batch(
                self, result, [*args, kwargs.get("running_batch")])
            now = time.time()
            if new_batch is not None:
                for req in new_batch.reqs:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.final_reused_tokens = req.cached_tokens
                    # Phase 2d-Closure: track actual input and recomputed tokens
                    req_stats.actual_input_tokens = len(req.origin_input_ids) if hasattr(req, 'origin_input_ids') else len(req.input_ids)
                    # Phase 3A: scheduler_prefill_tokens = independent measurement
                    # from the scheduler's extend_range/extend_input_len.
                    # Accumulate across chunks for chunked prefill.
                    sched_pf = C_SchedulerHook.get_extend_input_len(req)
                    if C_SchedulerHook.is_chunked_req(self, req):
                        req_stats.scheduler_prefill_tokens += sched_pf
                    else:
                        req_stats.scheduler_prefill_tokens = sched_pf
                    req_stats.recomputed_tokens = sched_pf  # overwritten by conservation at finalize
                    if req_stats.queue_end == -1:
                        if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                            req_stats.queue_end = now
                        else:
                            req_stats.queue_end = StateManager.get_global_clock()
                    else:
                        # Chunked request
                        pass
                    # Track L2 host_hit_length (set by _prefetch_kvcache -> init_next_round_input)
                    host_hit = getattr(req, 'host_hit_length', 0)
                    if host_hit > 0:
                        req_stats.l2_hit_tokens = host_hit
                        C_SchedulerHook._L2_HIT_TOKENS += host_hit
                        C_SchedulerHook._L2_HIT_COUNT += 1
                        # Phase 2d-Closure: consume timestamp
                        req_stats.l3_consume_time = StateManager.get_global_clock()
                        # L3 consumed tokens: the portion of host_hit_length that
                        # came from L3-restored data (approximation: L3-restored
                        # tokens are consumed first, before L2-backed-up tokens).
                        l3_restored = getattr(req_stats, 'l3_restored_tokens', 0)
                        if l3_restored > 0:
                            req_stats.l3_consumed_tokens = min(l3_restored, host_hit)
                        # Compute HBF/DRAM load timing from ACTUAL host_value
                        # virtual indices (collected by wrapped_match_prefix).
                        # In simulation mode, mock_start_loading isn't called
                        # (load_queue=0), so we inject load timing here based
                        # on the real prefix match host_hit_length + the actual
                        # HBF/DRAM split from host_value.
                        try:
                            from hisim.simulation.manager import ConfigManager
                            pc = ConfigManager.get_platform_config()
                            bytes_per_token = ConfigManager.get_kv_cache_bytes() or 32768
                            # Look up the actual split captured during match_prefix
                            split = _HOST_LOAD_SPLIT.pop(getattr(req, 'rid', None), None)
                            if split is not None:
                                hbf_hit = split["hbf_tokens"]
                                dram_hit = split["dram_tokens"]
                            elif pc.hbf is not None:
                                # match_prefix didn't capture (e.g. buffer mode
                                # staged splice path) — log and fall back to
                                # all-DRAM to avoid false HBF attribution.
                                hbf_hit = 0
                                dram_hit = host_hit
                                logger.debug(
                                    f"Host load split missing for rid={req.rid}, "
                                    f"host_hit={host_hit} — attributing all to DRAM"
                                )
                            else:
                                hbf_hit = 0
                                dram_hit = host_hit
                            # Record per-medium load timing (parallel: max)
                            hbf_slots = split.get("hbf_slots") if split else None
                            if hbf_hit > 0:
                                hbf_io_trace.set_context(
                                    getattr(req, "rid", None))
                                _record_hbf_load_timing(
                                    hbf_hit * bytes_per_token,
                                    num_tokens=hbf_hit,
                                    virtual_indices=hbf_slots)
                            if dram_hit > 0:
                                hbf_io_trace.set_context(
                                    getattr(req, "rid", None))
                                _record_l2_load_timing(
                                    dram_hit * bytes_per_token, dram_hit)
                            # Populate per-request HBF/DRAM split for metrics
                            req_stats.hbf_hit_tokens = hbf_hit
                            req_stats.dram_hit_tokens = dram_hit
                            req_stats.hbf_load_bytes = hbf_hit * bytes_per_token
                            req_stats.dram_load_bytes = dram_hit * bytes_per_token
                            logger.info(
                                f"[HOST-LOAD] rid={getattr(req,'rid',None)} "
                                f"host_hit={host_hit} HBF={hbf_hit}tk DRAM={dram_hit}tk "
                                f"hbf_bytes={hbf_hit*bytes_per_token} dram_bytes={dram_hit*bytes_per_token}"
                            )
                        except Exception:
                            pass
                    # Log cache state per prefill batch (L1/L2 pool sizes)
                    tc = getattr(self, 'tree_cache', None)
                    l1_evict = l1_total = l2_avail = owt = olb = -1
                    n_host_leaves = 0
                    if tc is not None and hasattr(tc, 'evictable_size_'):
                        cc = getattr(tc, 'cache_controller', None)
                        l2_avail = cc.mem_pool_host.available_size() if cc and hasattr(cc, 'mem_pool_host') else -1
                        l1_evict = tc.evictable_size_
                        l1_total = tc.token_to_kv_pool_allocator.available_size() + l1_evict if hasattr(tc, 'token_to_kv_pool_allocator') else -1
                        owt = len(getattr(tc, 'ongoing_write_through', {}))
                        olb = len(getattr(tc, 'ongoing_load_back', {}))
                        # HIRadixCache only (absent on a plain RadixCache)
                        n_host_leaves = len(getattr(tc, 'evictable_host_leaves', {}))
                    logger.info(
                        f"Cache state: L1_evict={l1_evict} L1_total={l1_total} "
                        f"L2_avail={l2_avail} L2_host_leaves={n_host_leaves} "
                        f"owt={owt} olb={olb}"
                    )
            elif not pending_batch and len(self.waiting_queue) > 0:
                # Prefetching
                StateManager.step_global_clock(0.005)
                StateManager.set_current_inference_dur(0.005)
            else:
                if C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE and (
                    len(C_SchedulerHook.FUTURE_QUEUE) != 0
                    and not pending_batch
                    and len(self.waiting_queue) == 0
                ):
                    next_created_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    StateManager.advance_global_clock_to(next_created_time)
                elif (
                    C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE
                    and len(C_SchedulerHook.FUTURE_QUEUE) == 0
                    and len(C_SchedulerHook.PARKED) != 0
                    and not pending_batch
                    and len(self.waiting_queue) == 0
                ):
                    # Defensive: everything left is parked but nothing is
                    # running (e.g. a finish event was missed). Force-release
                    # the earliest parked round to avoid a deadlock.
                    session = min(
                        C_SchedulerHook.PARKED,
                        key=lambda s: C_SchedulerHook.PARKED[s][0][0],
                    )
                    enqueue_time, salt, req = C_SchedulerHook.PARKED.pop(session)[0]
                    logger.warning(
                        f"Dynamic feedback: force-releasing parked round of "
                        f"session {session} (no running request)."
                    )
                    StateManager.advance_global_clock_to(enqueue_time)
                    heapq.heappush(
                        C_SchedulerHook.FUTURE_QUEUE, (enqueue_time, salt, req)
                    )
            logger.debug(
                f"Get new batch prefill: global iteration={StateManager.get_iteration()}, "
                f"new batch={new_batch.batch_size() if new_batch is not None else 0}, "
                f"waiting queue={len(self.waiting_queue)}"
            )

            return result

        def wrapped_run_batch(self, *args, **kwargs):
            ret = original_run_batch(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )

            if ret.__class__.__name__ == "GenerationBatchResult":
                hisim_batch = HisimScheduleBatch(reqs=[])
                if batch.forward_mode.is_extend():
                    for req in batch.reqs:
                        hisim_batch.reqs.append(
                            FakeRequest(
                                input_length=C_SchedulerHook.get_extend_input_len(req),
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )
                elif batch.forward_mode.is_decode():
                    for req in batch.reqs:
                        hisim_batch.reqs.append(
                            FakeRequest(
                                input_length=1,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )

                if not hisim_batch.is_empty():
                    StateManager.inc_iteration()
                    predicted_latency = (
                        C_SchedulerHook.INFERENCE_PREDICTOR.predict_infer_time(
                            hisim_batch
                        )
                    )
                    predicted_latency = float(predicted_latency)

                    forward_latency = 0
                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        now = time.time()
                        time.sleep(abs(predicted_latency))
                        now = time.time()
                        forward_latency = now - C_SchedulerHook.LAST_CPU_TS
                        C_SchedulerHook.LAST_CPU_TS = now
                    else:
                        now = time.time()
                        forward_latency = predicted_latency

                    StateManager.set_current_inference_dur(forward_latency)

                C_SchedulerHook.HISIM_BATCH = hisim_batch

            return ret

        def wrapped_process_batch_result(self, *args, **kwargs):
            ret = original_process_batch_result(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            if batch is not None:
                if len(batch.reqs) == 0:
                    return ret

                batch_start = StateManager.get_global_clock()
                hicache_l2_load_dur = StateManager.pop_hicache_l2_load_dur()
                hicache_l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()
                hicache_l3_backup_dur = StateManager.pop_hicache_l3_backup_dur()
                hicache_hbf_load_dur = StateManager.pop_hicache_hbf_load_dur()
                hicache_hbf_backup_dur = StateManager.pop_hicache_hbf_backup_dur()
                hicache_l3_load_dur = StateManager.pop_hicache_l3_load_dur()
                current_inference_dur = StateManager.get_current_inference_dur()

                try:
                    if ConfigManager.get_platform_config().enable_l3_direct_to_l1:
                        l3_direct_dur = StateManager.pop_hicache_l3_direct_load_dur()
                        hicache_l2_load_dur = max(hicache_l2_load_dur, l3_direct_dur)
                except Exception:
                    pass
                StateManager.pop_l3_direct_load_count()

                # HBF and DRAM transfers happen in parallel (different tokens
                # in the same batch go to different pools). The effective
                # backup/load duration is the max of the two.
                combined_backup_dur = max(hicache_l2_backup_dur, hicache_hbf_backup_dur)
                combined_load_dur = max(hicache_l2_load_dur, hicache_hbf_load_dur)
                _record_l3_consumption_after_host_load(
                    batch, batch_start, combined_load_dur)

                # Save last values for next iteration's timing-aware SSD simulation
                StateManager.set_last_l2_load_dur(combined_load_dur)
                StateManager.set_last_l2_backup_dur(combined_backup_dur)

                # L3 write starts at T_N (round start), overlaps with entire round:
                # combined_load + inference + combined_backup. Only overflow delays.
                # (combined_* = max of L2/HBF — parallel transfers, max semantics)
                #
                # Phase 2d-Closure: Write Barrier vs Async Write mode
                # - Barrier (default): overflow delays next round (current behavior)
                # - Async (HISIM_L3_WRITE_MODE=async): writes don't block; reads
                #   of incomplete writes fail (data not yet persisted)
                _l3_write_mode = os.environ.get("HISIM_L3_WRITE_MODE", "barrier")
                if _l3_write_mode == "async":
                    # Async Write: SSD writes proceed in background, don't block
                    # next round. Reading incomplete writes should fail (the kvcm
                    # storage_manager handles visibility).
                    l3_write_overflow = 0.0
                else:
                    # Write Barrier: overflow delays next round
                    l3_write_overflow = max(
                        0,
                        hicache_l3_backup_dur
                        - (combined_load_dur + current_inference_dur + combined_backup_dur),
                    )
                StateManager.set_last_l3_write_overflow(l3_write_overflow)

                if C_SchedulerHook.OVERLAP_SCHEDULE:
                    # Load overflow: combined load (HBF+DRAM, max) that exceeds
                    # the previous round's inference delays the next round.
                    StateManager.step_global_clock(
                        max(
                            combined_load_dur - StateManager.get_last_inference_dur(),
                            0,
                        )
                    )
                    StateManager.step_global_clock(current_inference_dur)
                    request_response_time = (
                        StateManager.get_global_clock()
                        + combined_backup_dur
                        + l3_write_overflow
                    )
                else:
                    StateManager.step_global_clock(
                        combined_load_dur
                        + current_inference_dur
                        + combined_backup_dur
                        + l3_write_overflow
                    )
                    request_response_time = StateManager.get_global_clock()
                # Per-batch timing breakdown for TPOT/TTFT diagnosis
                if batch.forward_mode.is_extend():
                    btype = "PREFILL"
                elif batch.forward_mode.is_decode():
                    btype = "DECODE"
                else:
                    btype = str(batch.forward_mode)
                logger.info(
                    f"[BATCH-TIME] {btype} bs={batch.batch_size()} "
                    f"load={combined_load_dur*1e6:.1f}μs "
                    f"infer={current_inference_dur*1e3:.3f}ms "
                    f"backup={combined_backup_dur*1e6:.1f}μs "
                    f"l3_ovf={l3_write_overflow*1e6:.1f}μs "
                    f"l3_read={hicache_l3_load_dur*1e6:.1f}μs"
                    f" clock={StateManager.get_global_clock()*1e3:.3f}ms"
                )
                # Request statistics
                for req in batch.reqs:
                    if not C_SchedulerHook.is_chunked_req(self, req):
                        req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                        # Phase 3A: Fix recomputed_tokens computation.
                        # cached_tokens may or may not include host_hit_length
                        # (see L1_device fix in dump_profile_results above).
                        # Use the same heuristic: if cached >= host_hit,
                        # cached includes host_hit → subtract both;
                        # otherwise cached doesn't include host_hit.
                        if batch.forward_mode.is_extend():
                            actual_input = len(req.origin_input_ids) if hasattr(req, 'origin_input_ids') else len(req.input_ids)
                            cached = req.cached_tokens
                            l2_host = getattr(req, 'host_hit_length', 0) or 0
                            if cached >= l2_host:
                                # cached includes host_hit (L2 host hit case)
                                req_stats.recomputed_tokens = max(
                                    0, actual_input - cached
                                )
                            else:
                                # cached doesn't include host_hit (L3 restore case)
                                req_stats.recomputed_tokens = max(
                                    0, actual_input - cached - l2_host
                                )
                        req_stats.gen_token_latencies.append(
                            request_response_time
                            - req_stats.last_event_time  # queue duration
                        )
                        req_stats.last_event_time = request_response_time
                    else:
                        # Chunked request: nothing to do
                        pass
                # Dynamic completion feedback: when a session round finishes,
                # release its parked successor rounds with the accumulated
                # virtual delay (actual finish - recorded finish) applied.
                if C_SchedulerHook.SESSION_STATE:
                    for req in batch.reqs:
                        finished = getattr(req, "finished", False) or getattr(
                            req, "finished_status", None
                        )
                        if not finished:
                            continue
                        session = C_SchedulerHook.REQ_SESSION.get(req.rid)
                        if session is None:
                            continue
                        st = C_SchedulerHook.SESSION_STATE.get(session)
                        if st is None or st["finished"]:
                            continue
                        st["finished"] = True
                        st["finish_vtime"] = request_response_time
                        parked = C_SchedulerHook.PARKED.pop(session, [])
                        if parked:
                            shift = max(
                                0.0,
                                st["finish_vtime"] - st["recorded_finish"],
                            )
                            for enqueue_time, salt, preq in parked:
                                heapq.heappush(
                                    C_SchedulerHook.FUTURE_QUEUE,
                                    (enqueue_time + shift, salt, preq),
                                )
                # Iteration statistics (with HBF per-medium breakdown)
                iter_stat = {
                    "requests": C_SchedulerHook.HISIM_BATCH.request_info(),
                    "forward_latency": current_inference_dur,
                    "l2_load_latency": hicache_l2_load_dur,
                    "l2_backup_latency": hicache_l2_backup_dur,
                    "l3_backup_latency": hicache_l3_backup_dur,
                    "l3_read_latency": hicache_l3_load_dur,
                    "hbf_load_latency": hicache_hbf_load_dur,
                    "hbf_backup_latency": hicache_hbf_backup_dur,
                    "combined_load_latency": combined_load_dur,
                    "combined_backup_latency": combined_backup_dur,
                    "l3_write_overflow": l3_write_overflow,
                }
                # Add HBF/DRAM capacity + conservation stats if adapter is available
                tc = getattr(self, 'tree_cache', None)
                if tc is not None and hasattr(tc, 'cache_controller'):
                    adapter = getattr(tc.cache_controller, '_hbf_adapter_instance', None)
                    if adapter is not None:
                        cs = adapter.capacity_stats()
                        es = adapter.event_stats()
                        cons = adapter.conservation_stats()
                        kv_bytes = ConfigManager.get_kv_cache_bytes() or 32768
                        iter_stat["hbf_used_tokens"] = cs["hbf_used"]
                        iter_stat["dram_used_tokens"] = cs["dram_used"]
                        iter_stat["hbf_used_bytes"] = cs["hbf_used"] * kv_bytes
                        iter_stat["dram_used_bytes"] = cs["dram_used"] * kv_bytes
                        iter_stat["hbf_capacity_tokens"] = cs["hbf_capacity"]
                        iter_stat["dram_capacity_tokens"] = cs["dram_capacity"]
                        iter_stat["hbf_available"] = cs["hbf_available"]
                        iter_stat["dram_available"] = cs["dram_available"]
                        iter_stat["alloc_attempts"] = cons["alloc_attempts"]
                        iter_stat["alloc_attempt_tokens"] = cons["alloc_attempt_tokens"]
                        iter_stat["alloc_success_tokens"] = cons["alloc_success_tokens"]
                        iter_stat["alloc_failed_tokens"] = cons["alloc_failed_tokens"]
                        iter_stat["alloc_fail_rate"] = cons["alloc_fail_rate"]
                        iter_stat["hbf_full_fallback_to_dram"] = cons["hbf_full_fallback_to_dram"]
                        iter_stat["dram_full_fallback_to_hbf"] = cons["dram_full_fallback_to_hbf"]
                        iter_stat["both_full_failures"] = cons["both_full_failures"]
                        iter_stat["hbf_peak_used"] = cons["hbf_peak_used"]
                        iter_stat["dram_peak_used"] = cons["dram_peak_used"]
                        iter_stat["hbf_free_tokens"] = cons["hbf_free_tokens"]
                        iter_stat["dram_free_tokens"] = cons["dram_free_tokens"]
                        iter_stat["retry_attempts"] = cons["retry_attempts"]
                        iter_stat["retry_successes"] = cons["retry_successes"]
                        iter_stat["retry_failures"] = cons["retry_failures"]
                        iter_stat["retry_success_rate"] = cons["retry_success_rate"]
                C_SchedulerHook.ITERATION_STATS.append(iter_stat)
            C_SchedulerHook.LAST_CPU_TS = time.time()
            return ret

        def wrapped_profile(self, req, *args, **kwargs):
            return C_SchedulerHook.dump_profile_results()

        target.event_loop_overlap = override_event_loop_overlap
        target.__init__ = wrapped_init
        target.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_batch_result
        if legacy_request_path:
            # sglang <= 0.5.9: requests are received by Scheduler.recv_requests
            # and profile requests are handled by Scheduler.profile.
            target.recv_requests = wrapped_recv_requests
            target.profile = wrapped_profile
        else:
            # sglang >= 0.5.10: requests are dispatched via
            # Scheduler.process_input_requests; profile requests are handled by
            # SchedulerProfilerManager (hooked by C_SchedulerProfilerManagerHook).
            target.process_input_requests = wrapped_process_input_requests


class C_SchedulerProfilerManagerHook(BaseHook):
    """sglang >= 0.5.10 moved profile handling from `Scheduler.profile` to
    `SchedulerProfilerManager._profile`; hook it to dump the simulation
    results (the legacy path keeps hooking `Scheduler.profile`)."""

    HOOK_CLASS_NAME = "SchedulerProfilerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler_components.profiler_manager"

    @classmethod
    def hook(cls, target):
        def wrapped_profile(self, recv_req, *args, **kwargs):
            return C_SchedulerHook.dump_profile_results()

        target._profile = wrapped_profile
