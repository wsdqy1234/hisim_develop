"""HBF Tiered Host Pool Adapter — PoC for L1.5 HBF parallel to L2 DRAM.

This module implements the Virtual/Tiered Host Pool Adapter (方案 A) that
wraps two physical host pools (HBF + DRAM) behind a unified virtual index
space, enabling HBF to replace part of DRAM capacity.

Key design:
  - Virtual index space: HBF [0, hbf_size) + DRAM [hbf_size, total)
  - dram_offset = hbf.logical_size (fixed at init)
  - alloc() routes to HBF or DRAM per placement policy; fallback if one full
  - free()/get_data_page()/set_from_flat_data_page() route by virtual index
  - available_size() reports combined capacity

In HiSim's mock path (mock_start_writing/mock_start_loading), the L2 transfer
engine is bypassed, so backup_from_device_all_layer/load_to_device_per_layer
are NOT called. The adapter only needs: alloc/free/available_size/
get_data_page/set_from_flat_data_page/get_dummy_flat_data_page.

Phase 0.5 PoC — minimal, no MQSim, no thermal, V0 proportional placement.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class PlacementPolicyV0:
    """V0: proportional split placement policy.

    Routes each alloc to HBF or DRAM based on a target ratio. Does NOT
    distinguish hot/cold or object characteristics (that's V1/V2). This is a
    functional baseline to verify the parallel pool data path.
    """

    hbf_ratio: float = 0.7  # target 70% to HBF, 30% to DRAM
    _hbf_alloc_count: int = 0
    _dram_alloc_count: int = 0

    def decide(self, need_size: int, hbf_avail: int, dram_avail: int) -> str:
        """Decide 'hbf' or 'dram' for this allocation.

        V0 logic: maintain the target ratio based on cumulative allocation
        counts. Fallback to the other pool if the preferred one is full.
        """
        total = self._hbf_alloc_count + self._dram_alloc_count
        if total == 0:
            target = "hbf" if self.hbf_ratio > 0 else "dram"
        else:
            current_hbf_ratio = self._hbf_alloc_count / total
            target = "hbf" if current_hbf_ratio < self.hbf_ratio else "dram"

        # Try preferred, fallback if full
        if target == "hbf" and hbf_avail >= need_size:
            self._hbf_alloc_count += need_size
            return "hbf"
        if target == "dram" and dram_avail >= need_size:
            self._dram_alloc_count += need_size
            return "dram"
        # Fallback: try the other pool
        if target == "hbf" and dram_avail >= need_size:
            self._dram_alloc_count += need_size
            return "dram"
        if target == "dram" and hbf_avail >= need_size:
            self._hbf_alloc_count += need_size
            return "hbf"
        # Both insufficient — caller will handle (evict or fail)
        if hbf_avail >= need_size:
            self._hbf_alloc_count += need_size
            return "hbf"
        if dram_avail >= need_size:
            self._dram_alloc_count += need_size
            return "dram"
        return "hbf"  # let the physical pool's alloc return None

    def reset(self):
        self._hbf_alloc_count = 0
        self._dram_alloc_count = 0


class MockHBFPool:
    """Minimal HBF mock host pool for PoC.

    Implements the essential HostKVCache interface subset needed by
    TieredHostKVCache: alloc/free/available_size/get_data_page/
    set_from_flat_data_page/get_dummy_flat_data_page.

    Uses a simple free-list (like HostKVCache base). No real memory
    allocation — mock 1-element pages. Logical capacity from HBFConfig.
    """

    def __init__(
        self,
        logical_size: int,
        page_size: int = 1,
        size_per_token: int = 1,
        dtype: torch.dtype = torch.float16,
    ):
        self.logical_size = logical_size
        self.size = logical_size  # for compatibility
        self.page_size = page_size
        self.size_per_token = size_per_token
        self.dtype = dtype
        self.layout = "layer_first"
        self.device = "cpu"
        self.layer_num = 1
        self.start_layer = 0
        self.end_layer = 0
        self.dcp_size = 1
        self.dcp_rank = 0
        self.head_num = 1
        self.head_dim = 1

        self._lock = threading.RLock()
        self.clear()

        # Mock data buffer: tiny (1 element per slot) — no real memory
        self._mock_buffer = torch.zeros(
            (2, self.layer_num, max(logical_size, 1), 1, 1),
            dtype=dtype,
        )
        logger.info(
            f"MockHBFPool: logical_size={logical_size}, page_size={page_size}, "
            f"mock_buffer={self._mock_buffer.numel()} elements (tiny)"
        )

    def clear(self):
        with self._lock:
            self._free_slots = torch.arange(self.logical_size, dtype=torch.int64)
            self._release_slots = []
            self._num_release = 0
            self._slot_used = torch.zeros(self.logical_size, dtype=torch.bool)

    def available_size(self):
        return len(self._free_slots) + self._num_release

    def _merge_release(self):
        if self._num_release == 0:
            return
        if len(self._free_slots) == 0 and len(self._release_slots) == 1:
            self._free_slots = self._release_slots[0]
        else:
            self._free_slots = torch.cat([self._free_slots, *self._release_slots])
        self._release_slots = []
        self._num_release = 0

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with self._lock:
            if need_size > self.available_size():
                return None
            if need_size > len(self._free_slots):
                self._merge_release()
            if need_size > len(self._free_slots):
                return None
            select = self._free_slots[:need_size]
            self._free_slots = self._free_slots[need_size:]
            assert not self._slot_used[select].any(), "Double-alloc in HBF pool"
            self._slot_used[select] = True
            return select

    def free(self, indices: torch.Tensor) -> int:
        indices = indices.cpu()
        if indices.numel() == 0:
            return 0
        with self._lock:
            assert self._slot_used[indices].all(), "Double-free in HBF pool"
            self._slot_used[indices] = False
            self._release_slots.append(indices)
            self._num_release += indices.numel()
        return indices.numel()

    def get_data_page(self, index, flat: bool = True):
        """Return mock data page for L3 IO routing."""
        idx = int(index) if not isinstance(index, torch.Tensor) else int(index.item())
        return self._mock_buffer[:, :, idx % max(self.logical_size, 1), :, :]

    def get_dummy_flat_data_page(self):
        return torch.zeros(self.page_size * self.size_per_token, dtype=self.dtype)

    def set_from_flat_data_page(self, index, data_page):
        # No-op: mock pool doesn't store real data
        pass

    # Stubs for methods not called in HiSim mock path
    def backup_from_device_all_layer(self, *a, **kw):
        pass

    def load_to_device_per_layer(self, *a, **kw):
        pass

    def init_kv_buffer(self):
        pass


class TieredHostKVCache:
    """Virtual/Tiered Host Pool Adapter (方案 A).

    Wraps two physical HostKVCache pools (HBF + DRAM) behind a unified
    virtual index space. Installed as the anchor pool's host_pool, replacing
    all direct references (mem_pool_host.anchor_entry.host_pool,
    storage_host_pool, storage_backend.mem_pool_host).

    Virtual index encoding:
      HBF:  [0, dram_offset)           — indices from hbf_pool.alloc (direct)
      DRAM: [dram_offset, total)       — indices from dram_pool.alloc + offset
      dram_offset = hbf_pool.logical_size (fixed at init)
    """

    def __init__(
        self,
        hbf_pool,
        dram_pool,
        placement_policy=None,
    ):
        self.hbf_pool = hbf_pool
        self.dram_pool = dram_pool
        self.policy = placement_policy or PlacementPolicyV0(hbf_ratio=0.7)

        # Virtual index offset: DRAM indices are offset by HBF's logical size
        self.dram_offset = hbf_pool.logical_size
        self.logical_size = hbf_pool.logical_size + dram_pool.logical_size
        self.size = self.logical_size

        # Delegate properties from DRAM pool (same model params)
        self.page_size = dram_pool.page_size
        self.size_per_token = dram_pool.size_per_token
        self.dtype = dram_pool.dtype
        self.layout = dram_pool.layout
        self.device = dram_pool.device
        self.layer_num = dram_pool.layer_num
        self.start_layer = dram_pool.start_layer
        self.end_layer = dram_pool.end_layer
        self.dcp_size = getattr(dram_pool, "dcp_size", 1)
        self.dcp_rank = getattr(dram_pool, "dcp_rank", 0)

        # Capacity tracking (for metrics)
        self._hbf_used = 0
        self._dram_used = 0
        self._lock = threading.RLock()

        # Event tracking (for Phase 2a verification)
        self._hbf_alloc_count = 0
        self._dram_alloc_count = 0
        self._hbf_free_count = 0
        self._dram_free_count = 0
        self._hbf_get_data_page_count = 0
        self._dram_get_data_page_count = 0
        self._hbf_set_data_page_count = 0
        self._dram_set_data_page_count = 0
        self._hbf_alloc_tokens = 0
        self._dram_alloc_tokens = 0

        # Phase 2b-5: Cache conservation metrics
        self._alloc_attempts = 0
        self._alloc_attempt_tokens = 0
        self._alloc_success_tokens = 0
        self._alloc_failed_tokens = 0
        self._hbf_full_fallback_to_dram = 0
        self._dram_full_fallback_to_hbf = 0
        self._both_full_failures = 0
        self._hbf_free_tokens = 0
        self._dram_free_tokens = 0
        self._hbf_peak_used = 0
        self._dram_peak_used = 0

        # V1a-2/V1a-3: Eviction callback + bounded retry
        self._evict_callback = None  # set by install function
        self._retry_attempts = 0
        self._retry_successes = 0
        self._retry_failures = 0
        self._max_retries = 3

        logger.info(
            f"TieredHostKVCache: HBF[0,{self.dram_offset}) + "
            f"DRAM[{self.dram_offset},{self.logical_size}), "
            f"total_logical={self.logical_size}"
        )

    # ---- Routing helpers ----
    def _is_hbf(self, virtual_idx):
        return virtual_idx < self.dram_offset

    def _to_physical(self, virtual_indices):
        """Split virtual indices into (hbf_indices, dram_physical_indices)."""
        v = virtual_indices if isinstance(virtual_indices, torch.Tensor) else torch.tensor(virtual_indices)
        hbf_mask = v < self.dram_offset
        hbf_v = v[hbf_mask]
        dram_v = v[~hbf_mask] - self.dram_offset
        return hbf_v, dram_v, hbf_mask

    # ---- Core HostKVCache interface ----
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with self._lock:
            self._alloc_attempts += 1
            self._alloc_attempt_tokens += need_size
            target = self.policy.decide(
                need_size,
                self.hbf_pool.available_size(),
                self.dram_pool.available_size(),
            )
            if target == "hbf":
                idx = self.hbf_pool.alloc(need_size)
                if idx is not None:
                    self._hbf_used += need_size
                    self._hbf_alloc_count += 1
                    self._hbf_alloc_tokens += need_size
                    self._alloc_success_tokens += need_size
                    self._hbf_peak_used = max(self._hbf_peak_used, self._hbf_used)
                    return idx  # HBF indices are [0, dram_offset), no offset
                # Fallback to DRAM
                self._hbf_full_fallback_to_dram += 1
                idx = self.dram_pool.alloc(need_size)
                if idx is not None:
                    self._dram_used += need_size
                    self._dram_alloc_count += 1
                    self._dram_alloc_tokens += need_size
                    self._alloc_success_tokens += need_size
                    self._dram_peak_used = max(self._dram_peak_used, self._dram_used)
                    return idx + self.dram_offset
            else:
                idx = self.dram_pool.alloc(need_size)
                if idx is not None:
                    self._dram_used += need_size
                    self._dram_alloc_count += 1
                    self._dram_alloc_tokens += need_size
                    self._alloc_success_tokens += need_size
                    self._dram_peak_used = max(self._dram_peak_used, self._dram_used)
                    return idx + self.dram_offset
                # Fallback to HBF
                self._dram_full_fallback_to_hbf += 1
                idx = self.hbf_pool.alloc(need_size)
                if idx is not None:
                    self._hbf_used += need_size
                    self._hbf_alloc_count += 1
                    self._hbf_alloc_tokens += need_size
                    self._alloc_success_tokens += need_size
                    self._hbf_peak_used = max(self._hbf_peak_used, self._hbf_used)
                    return idx
            # Both pools full — try V1a-2 bounded retry with eviction
            if self._evict_callback is not None and self._max_retries > 0:
                for retry in range(self._max_retries):
                    self._retry_attempts += 1
                    hbf_before = self.hbf_pool.available_size()
                    dram_before = self.dram_pool.available_size()
                    deficit = need_size - max(hbf_before, dram_before)
                    if deficit <= 0:
                        break  # space available now (race resolved)
                    try:
                        freed = self._evict_callback(deficit)
                    except Exception as e:
                        logger.info(f"[RETRY] retry={retry} deficit={deficit} evict_exception={e}")
                        freed = 0
                    hbf_after = self.hbf_pool.available_size()
                    dram_after = self.dram_pool.available_size()
                    logger.info(
                        f"[RETRY] retry={retry} need={need_size} deficit={deficit} "
                        f"freed={freed} "
                        f"HBF: {hbf_before}→{hbf_after} "
                        f"DRAM: {dram_before}→{dram_after}"
                    )
                    if freed <= 0:
                        logger.info(f"[RETRY] no_progress_exit: evict returned 0")
                        break  # no evictable victims
                    # Retry alloc after eviction
                    if hbf_after >= need_size:
                        idx = self.hbf_pool.alloc(need_size)
                        if idx is not None:
                            self._hbf_used += need_size
                            self._hbf_alloc_count += 1
                            self._hbf_alloc_tokens += need_size
                            self._alloc_success_tokens += need_size
                            self._hbf_peak_used = max(self._hbf_peak_used, self._hbf_used)
                            self._retry_successes += 1
                            logger.info(f"[RETRY] success: retry={retry} pool=HBF")
                            return idx
                        else:
                            logger.info(f"[RETRY] alloc_fail_after_evict: HBF avail={hbf_after} but alloc({need_size}) returned None")
                    if dram_after >= need_size:
                        idx = self.dram_pool.alloc(need_size)
                        if idx is not None:
                            self._dram_used += need_size
                            self._dram_alloc_count += 1
                            self._dram_alloc_tokens += need_size
                            self._alloc_success_tokens += need_size
                            self._dram_peak_used = max(self._dram_peak_used, self._dram_used)
                            self._retry_successes += 1
                            logger.info(f"[RETRY] success: retry={retry} pool=DRAM")
                            return idx + self.dram_offset
                        else:
                            logger.info(f"[RETRY] alloc_fail_after_evict: DRAM avail={dram_after} but alloc({need_size}) returned None")
                self._retry_failures += 1
                logger.info(f"[RETRY] exhausted: retries={self._max_retries} final HBF={self.hbf_pool.available_size()} DRAM={self.dram_pool.available_size()}")
            # All retries exhausted — alloc failed
            self._both_full_failures += 1
            self._alloc_failed_tokens += need_size
            logger.info(
                f"[ALLOC-FAIL] need={need_size} "
                f"HBF_avail={self.hbf_pool.available_size()}/{self.hbf_pool.logical_size} "
                f"DRAM_avail={self.dram_pool.available_size()}/{self.dram_pool.logical_size} "
                f"target={target} attempts={self._alloc_attempts} "
                f"retry_failed={self._retry_failures}"
            )
            return None

    def free(self, indices: torch.Tensor) -> int:
        indices = indices.cpu()
        if indices.numel() == 0:
            return 0
        with self._lock:
            hbf_v, dram_v, hbf_mask = self._to_physical(indices)
            freed = 0
            if hbf_v.numel() > 0:
                freed += self.hbf_pool.free(hbf_v)
                self._hbf_used -= hbf_v.numel()
                self._hbf_free_tokens += hbf_v.numel()
                # Phase 5D P0 (G3): notify the WIN object registry so freed
                # slots leave no stale Object→WIN mappings (eviction/CoW).
                cb = getattr(self, "_on_hbf_free", None)
                if cb is not None:
                    try:
                        cb(hbf_v.tolist())
                    except Exception:
                        pass
            if dram_v.numel() > 0:
                freed += self.dram_pool.free(dram_v)
                self._dram_used -= dram_v.numel()
                self._dram_free_tokens += dram_v.numel()
        return freed

    def available_size(self):
        """Max single-pool allocatable capacity.

        Returns max(hbf_avail, dram_avail), NOT the sum. This is critical
        because alloc(n) requires ONE physical pool to hold all n slots.
        Returning the sum would let the pre-check in _execute_kv_backup
        pass (sum >= n) even when neither pool can hold n, causing alloc
        to fail without triggering eviction — the V0 vicious cycle.

        For total capacity metrics, use total_available_size().
        """
        return max(self.hbf_pool.available_size(), self.dram_pool.available_size())

    def total_available_size(self):
        """Sum of both pools' free capacity — for metrics only, not alloc."""
        return self.hbf_pool.available_size() + self.dram_pool.available_size()

    def _idx_to_int(self, index) -> int:
        """Convert int or tensor (0-dim or 1-dim) to Python int."""
        if isinstance(index, torch.Tensor):
            return int(index.item()) if index.dim() == 0 else int(index[0].item())
        return int(index)

    def get_data_page(self, index, flat: bool = True):
        idx_val = self._idx_to_int(index)
        if idx_val < self.dram_offset:
            self._hbf_get_data_page_count += 1
            return self.hbf_pool.get_data_page(idx_val, flat)
        else:
            self._dram_get_data_page_count += 1
            return self.dram_pool.get_data_page(idx_val - self.dram_offset, flat)

    def get_dummy_flat_data_page(self):
        return self.dram_pool.get_dummy_flat_data_page()

    def set_from_flat_data_page(self, index, data_page):
        idx_val = self._idx_to_int(index)
        if idx_val < self.dram_offset:
            self._hbf_set_data_page_count += 1
            self.hbf_pool.set_from_flat_data_page(idx_val, data_page)
        else:
            self._dram_set_data_page_count += 1
            self.dram_pool.set_from_flat_data_page(idx_val - self.dram_offset, data_page)

    # ---- Stubs for methods not called in HiSim mock path ----
    def backup_from_device_all_layer(self, *a, **kw):
        # In mock path, transfer engine is bypassed. If called, delegate to DRAM.
        return self.dram_pool.backup_from_device_all_layer(*a, **kw)

    def load_to_device_per_layer(self, *a, **kw):
        return self.dram_pool.load_to_device_per_layer(*a, **kw)

    def init_kv_buffer(self):
        pass  # physical pools already initialized

    def clear(self):
        self.hbf_pool.clear()
        self.dram_pool.clear()
        self._hbf_used = 0
        self._dram_used = 0
        self.policy.reset()

    # ---- Capacity metrics ----
    @property
    def hbf_used(self):
        return self._hbf_used

    @property
    def dram_used(self):
        return self._dram_used

    @property
    def hbf_capacity(self):
        return self.hbf_pool.logical_size

    @property
    def dram_capacity(self):
        return self.dram_pool.logical_size

    def capacity_stats(self):
        return {
            "hbf_used": self._hbf_used,
            "hbf_capacity": self.hbf_pool.logical_size,
            "hbf_available": self.hbf_pool.available_size(),
            "dram_used": self._dram_used,
            "dram_capacity": self.dram_pool.logical_size,
            "dram_available": self.dram_pool.available_size(),
            "max_allocatable": self.available_size(),
            "total_available": self.total_available_size(),
        }

    def conservation_stats(self):
        """Cache object conservation metrics for Phase 2b-5."""
        return {
            "alloc_attempts": self._alloc_attempts,
            "alloc_attempt_tokens": self._alloc_attempt_tokens,
            "alloc_success_tokens": self._alloc_success_tokens,
            "alloc_failed_tokens": self._alloc_failed_tokens,
            "alloc_fail_rate": (self._alloc_failed_tokens / self._alloc_attempt_tokens
                                if self._alloc_attempt_tokens > 0 else 0.0),
            "hbf_full_fallback_to_dram": self._hbf_full_fallback_to_dram,
            "dram_full_fallback_to_hbf": self._dram_full_fallback_to_hbf,
            "both_full_failures": self._both_full_failures,
            "hbf_free_tokens": self._hbf_free_tokens,
            "dram_free_tokens": self._dram_free_tokens,
            "hbf_peak_used": self._hbf_peak_used,
            "dram_peak_used": self._dram_peak_used,
            "retry_attempts": self._retry_attempts,
            "retry_successes": self._retry_successes,
            "retry_failures": self._retry_failures,
            "retry_success_rate": (self._retry_successes / self._retry_attempts
                                   if self._retry_attempts > 0 else 0.0),
        }

    def event_stats(self):
        """Event counts for Phase 2a verification."""
        return {
            "hbf_alloc_count": self._hbf_alloc_count,
            "dram_alloc_count": self._dram_alloc_count,
            "hbf_alloc_tokens": self._hbf_alloc_tokens,
            "dram_alloc_tokens": self._dram_alloc_tokens,
            "hbf_free_count": self._hbf_free_count,
            "dram_free_count": self._dram_free_count,
            "hbf_get_data_page": self._hbf_get_data_page_count,
            "dram_get_data_page": self._dram_get_data_page_count,
            "hbf_set_data_page": self._hbf_set_data_page_count,
            "dram_set_data_page": self._dram_set_data_page_count,
        }

    def log_stats(self):
        """Log capacity + event + conservation stats for Phase 2b-5 verification."""
        cs = self.capacity_stats()
        es = self.event_stats()
        cons = self.conservation_stats()
        print(
            f"[HBF-STATS] HBF: used={cs['hbf_used']}/{cs['hbf_capacity']} "
            f"allocs={es['hbf_alloc_count']}({es['hbf_alloc_tokens']}tk) "
            f"getData={es['hbf_get_data_page']} setData={es['hbf_set_data_page']} | "
            f"DRAM: used={cs['dram_used']}/{cs['dram_capacity']} "
            f"allocs={es['dram_alloc_count']}({es['dram_alloc_tokens']}tk) "
            f"getData={es['dram_get_data_page']} setData={es['dram_set_data_page']} | "
            f"max_alloc={cs['max_allocatable']} total_avail={cs['total_available']}",
            flush=True,
        )
        print(
            f"[HBF-CONSERV] attempts={cons['alloc_attempts']}({cons['alloc_attempt_tokens']}tk) "
            f"success={cons['alloc_success_tokens']}tk failed={cons['alloc_failed_tokens']}tk "
            f"fail_rate={cons['alloc_fail_rate']:.1%} | "
            f"hbf→dram_fb={cons['hbf_full_fallback_to_dram']} "
            f"dram→hbf_fb={cons['dram_full_fallback_to_hbf']} "
            f"both_full={cons['both_full_failures']} | "
            f"hbf_peak={cons['hbf_peak_used']} dram_peak={cons['dram_peak_used']} "
            f"hbf_freed={cons['hbf_free_tokens']} dram_freed={cons['dram_free_tokens']} | "
            f"retry: attempts={cons['retry_attempts']} "
            f"success={cons['retry_successes']} fail={cons['retry_failures']} "
            f"rate={cons['retry_success_rate']:.1%}",
            flush=True,
        )


def install_tiered_host_pool(
    cache_controller,
    hbf_logical_size: int,
    hbf_ratio: float = 0.7,
):
    """Install TieredHostKVCache as the anchor host pool.

    Replaces all direct host pool references on the controller:
      1. mem_pool_host.anchor_entry.host_pool → adapter
      2. storage_host_pool → adapter
      3. storage_backend.mem_pool_host → adapter (re-register)

    Must be called AFTER pool stack assembly (e.g., on first
    check_hicache_events) and BEFORE any backup operation.

    Returns the adapter instance, or None if installation failed.
    """
    try:
        mem_pool_host = cache_controller.mem_pool_host
        # Get the anchor (KV DRAM) host pool
        anchor_entry = mem_pool_host.anchor_entry
        dram_pool = anchor_entry.host_pool

        # Create HBF mock pool with same params
        hbf_pool = MockHBFPool(
            logical_size=hbf_logical_size,
            page_size=dram_pool.page_size,
            size_per_token=dram_pool.size_per_token,
            dtype=dram_pool.dtype,
        )

        policy = PlacementPolicyV0(hbf_ratio=hbf_ratio)
        adapter = TieredHostKVCache(hbf_pool, dram_pool, policy)

        # Replace all references
        anchor_entry.host_pool = adapter
        cache_controller.storage_host_pool = adapter
        if hasattr(cache_controller, "storage_backend") and cache_controller.storage_backend:
            cache_controller.storage_backend.register_mem_pool_host(adapter)
            logger.info("Re-registered adapter with storage_backend")

        logger.info(
            f"TieredHostKVCache installed: HBF={hbf_logical_size} slots, "
            f"DRAM={dram_pool.logical_size} slots, ratio={hbf_ratio}"
        )
        return adapter

    except Exception as e:
        logger.error(f"Failed to install TieredHostKVCache: {e}")
        import traceback

        traceback.print_exc()
        return None
