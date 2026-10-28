from typing import List, Union, Optional, Any
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
import psutil
import os
import threading
from functools import wraps

from hisim.utils.logger import get_logger
from hisim.simulation.manager import StateManager, ConfigManager, Envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
)
from hisim.simulation.sglang.version import VersionDispatcher

try:
    import kvcm_py_optimizer
except ImportError:
    kvcm_py_optimizer = None

# from sglang.srt.managers.schedule_batch import Req

logger = get_logger("hisim")
_CURRENT_DIR = Path(__file__).parent.resolve()
_KVCM_BLOCK_SIZE = 16


def _mock_mamba_pool_size() -> int:
    """Mamba slot capacity for the mock allocator/pool.

    Default matches max_mamba_cache_size (80). Set HISIM_MAMBA_POOL_SIZE
    to override for stress testing.
    """
    try:
        return max(1, int(os.environ.get("HISIM_MAMBA_POOL_SIZE", "80")))
    except ValueError:
        return 80


def alloc_extend_cpu(
    pre_lens_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,  # Pre-allocated output tensor (consistent with Triton kernel)
    bs_upper: int,  # CPU doesn't need this, but kept for interface consistency (can be ignored)
    page_size: int,
    max_num_extend_tokens: int,  # Also kept but not used
):
    # Convert to Python list or scalar for processing
    pre_lens = pre_lens_ptr.cpu().tolist()
    seq_lens = seq_lens_ptr.cpu().tolist()
    last_loc = last_loc_ptr.cpu().tolist()
    free_pages = free_page_ptr.cpu().tolist()

    batch_size = len(pre_lens)

    def ceil_div(a, b):
        return (a + b - 1) // b

    extend_lens = [seq_lens[i] - pre_lens[i] for i in range(batch_size)]
    num_new_pages_per_seq = []
    for i in range(batch_size):
        pages_before = ceil_div(pre_lens[i], page_size)
        pages_after = ceil_div(seq_lens[i], page_size)
        num_new_pages_per_seq.append(pages_after - pages_before)

    # Initialize offsets
    output_offset = 0
    free_page_offset = 0

    # Process each sequence
    for pid in range(batch_size):
        pre_len = pre_lens[pid]
        seq_len = seq_lens[pid]
        extend_len = extend_lens[pid]
        last_token_pos = last_loc[pid]

        if extend_len <= 0:
            # No extension, skip (but still need to advance free_page_offset)
            free_page_offset += num_new_pages_per_seq[pid]
            continue

        # === Part 1: Fill the remaining space of the current incomplete page ===
        current_page_end = ceil_div(pre_len, page_size) * page_size
        part1_end = min(seq_len, current_page_end)
        num_part1 = part1_end - pre_len

        if num_part1 > 0:
            # out_indices[output_offset : output_offset + num_part1] = last_token_pos + 1 + i
            for i in range(num_part1):
                out_indices[output_offset + i] = last_token_pos + 1 + i
            output_offset += num_part1

        if pre_len + num_part1 == seq_len:
            free_page_offset += num_new_pages_per_seq[pid]
            continue

        # === Part 2: Fill complete new pages ===
        full_pages_start = current_page_end
        full_pages_end = (seq_len // page_size) * page_size
        num_part2 = full_pages_end - full_pages_start

        if num_part2 > 0:
            for i in range(num_part2):
                page_idx_in_free = free_page_offset + (i // page_size)
                page_id = free_pages[page_idx_in_free]
                token_in_page = i % page_size
                out_indices[output_offset + i] = page_id * page_size + token_in_page
            output_offset += num_part2

        if pre_len + num_part1 + num_part2 == seq_len:
            free_page_offset += num_new_pages_per_seq[pid]
            continue

        # === Part 3: Fill the last incomplete new page ===
        num_part3 = seq_len - full_pages_end
        if num_part3 > 0:
            last_page_idx = free_page_offset + num_new_pages_per_seq[pid] - 1
            last_page_id = free_pages[last_page_idx]
            for i in range(num_part3):
                out_indices[output_offset + i] = last_page_id * page_size + i
            output_offset += num_part3

        # Push forward free_page_offset
        free_page_offset += num_new_pages_per_seq[pid]


def alloc_decode_cpu(
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,  # Reserved parameter (not used in CPU)
    page_size: int,
):
    seq_lens = seq_lens_ptr.cpu().tolist()
    last_loc = last_loc_ptr.cpu().tolist()
    free_pages = free_page_ptr.cpu().tolist()

    batch_size = len(seq_lens)

    def ceil_div(a, b):
        return (a + b - 1) // b

    # Calculate the number of new pages needed for each sequence (used to determine free_page offset)
    num_new_pages_per_seq = []
    for i in range(batch_size):
        pre_len_i = seq_lens[i] - 1
        pages_before = ceil_div(pre_len_i, page_size)
        pages_after = ceil_div(seq_lens[i], page_size)
        num_new_pages_per_seq.append(pages_after - pages_before)

    # Calculate prefix sum to determine the starting position in free_page_ptr for each sequence
    prefix_sum = 0
    for pid in range(batch_size):
        num_new_pages_self = num_new_pages_per_seq[pid]
        new_page_start_loc = (
            prefix_sum  # Starting index in free_pages for current sequence
        )
        prefix_sum += num_new_pages_self

        seq_len = seq_lens[pid]
        pre_len = seq_len - 1

        num_page_start_loc_self = ceil_div(seq_len, page_size) - ceil_div(
            pre_len, page_size
        )

        if num_page_start_loc_self == 0:
            # Reuse current page, directly write last_loc + 1
            out_indices[pid] = last_loc[pid] + 1
        else:
            # Allocate new page, take the first new page ID
            page_id = free_pages[new_page_start_loc]
            out_indices[pid] = page_id * page_size


def get_num_new_pages(
    seq_lens: torch.Tensor,
    page_size: int,
    prefix_lens: Optional[torch.Tensor] = None,
    decode: bool = False,
) -> torch.Tensor:
    """
    Get the number of new pages for the given prefix and sequence lengths.
    We use cpu tensors to avoid blocking kernel launch.
    """
    cpu_device = torch.device("cpu")
    assert seq_lens.device == cpu_device

    if prefix_lens is None or decode:
        # NOTE: Special case for handling decode, which prefix lens is `seq_lens - 1`.
        assert decode
        return (seq_lens % page_size == 1).int().sum().item()

    assert prefix_lens.device == cpu_device
    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (prefix_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before
    sum_num_new_pages = torch.sum(num_new_pages).to(torch.int64)
    return sum_num_new_pages.item()


class MockKVIndexTranslator:
    """KV index translator stub (sglang >= 0.5.10 calls ``rebind_write_loc`` in
    ``ForwardBatch.init_new``; real translation only matters for attention
    kernels, which are intercepted)."""

    is_translating = False

    def rebind_write_loc(self, forward_batch) -> None:
        pass

    def build_index_table(self, *args, **kwargs):
        return None

    def index_table_for_batch(self, forward_batch):
        return None

    def sliding_window_write_loc_for(self, out_cache_loc):
        return out_cache_loc

    def make_capture_tables(self, *args, **kwargs):
        return None


class _MockMambaPool:
    """Mock Mamba (KDA) state pool for _MambaStrategy.build."""
    def __init__(self):
        self.size = _mock_mamba_pool_size()
        self.num_mamba_layers = 18  # K3-23L has 18 KDA layers
        self.layer_transfer_counter = None

        # Mock mamba_cache with conv/temporal state buffers that
        # MambaPoolHost.__init__ reads.
        #   conv: list with ONE tensor of shape (num_layers, 1, conv_dim, 4);
        #     iterating gives conv_state whose [i] indexes per-layer (line 123)
        #     and .shape[2:] is the state dim (line 62), conv[0].dtype (line 67).
        #   temporal: single tensor (num_layers, 1, temporal_dim, 4) so
        #     temporal.shape[2:] (line 69) and temporal[i] (line 113) both work.
        _n = self.num_mamba_layers
        _conv_dim = 7168   # K3 hidden_size
        _temporal_dim = 512  # K3 kv_lora_rank (state dim)
        import torch as _torch
        class _MockMambaCache:
            conv = [_torch.zeros(_n, 1, _conv_dim, 4, dtype=_torch.float32)]
            temporal = _torch.zeros(_n, 1, _temporal_dim, 4, dtype=_torch.float32)
        self.mamba_cache = _MockMambaCache()

    def get_kv_size_bytes(self):
        return 14653440  # 18 layers × 814,080 B/layer — matches sglang MambaSubPoolSpec.entry_bytes()

    def copy_from(self, src_indices, dst_indices):
        """Device-to-device slot copy (no_buffer strategy path).

        HiSim tracks KDA state symbolically, so there is no real state to
        move. Without this method the catch-all __getattr__ returns None and
        sglang 0.5.19's MambaComponent.prepare_for_caching_req raises
        "'NoneType' object is not callable" on every chunked prefill.
        """
        return None

    def __getattr__(self, name):
        return None


class MockReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    # from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        self.size = size
        self.max_context_len = 1  # overwrite
        self.device = device
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )

        self.free_slots = list(range(size))

        # K3 hybrid architecture: mamba_map maps KDA layer IDs to mamba pool indices.
        # K3-23L has 5 MLA layers (0-4) + 18 KDA layers (5-22).
        self.mamba_map = {5 + i: i for i in range(18)}
        # mamba_pool: mock for _MambaStrategy.build. Has __getattr__ catch-all.
        self.mamba_pool = _MockMambaPool()

        # layer_transfer_counter for HybridCacheController
        self.layer_transfer_counter = None

        # Mirrors HybridReqToTokenPool: enable_mamba_extra_buffer controls
        # whether ping-pong tracking buffers are used. K3 uses extra_buffer
        # strategy by default (mamba_radix_cache_strategy='extra_buffer').
        self.enable_mamba_extra_buffer = True
        self.enable_mamba_extra_buffer_lazy = False

        # req_index_to_mamba_index_mapping: tensor[req_pool_idx] → mamba_pool_idx.
        # Set during alloc, used by scheduler/attention backends and free_mamba_cache.
        self.req_index_to_mamba_index_mapping = torch.zeros(
            (size,), dtype=torch.int32, device=device
        )
        # req_index_to_mamba_ping_pong_track_buffer_mapping: tensor[req_pool_idx] → pp_buf[2].
        self._mock_pp_mapping = torch.full(
            (size, 2), -1, dtype=torch.int64, device=device
        )

        # Version dispatcher for alloc/free — must be set in __init__ so that
        # Two Pools Plain mode (no init_hicache) also works.
        _version_dispatcher = VersionDispatcher()
        _version_dispatcher.register_method(
            "alloc",
            ["0.5.6", "0.5.6.post1", "0.5.6.post2", "0.5.7", "0.5.8", "0.5.8.post1"],
            self._alloc_v1,
        )
        _version_dispatcher.register_method("alloc", ["0.5.9"], self._alloc_v2)
        _version_dispatcher.register_method(
            "alloc", ["0.5.18", "0.5.19"], self._alloc_v3
        )

        _version_dispatcher.register_method(
            "free",
            ["0.5.6", "0.5.6.post1", "0.5.6.post2", "0.5.7", "0.5.8", "0.5.8.post1"],
            self._free_v1,
        )
        _version_dispatcher.register_method("free", ["0.5.9"], self._free_v2)
        _version_dispatcher.register_method(
            "free", ["0.5.18", "0.5.19"], self._free_v3
        )

        self._alloc_func = _version_dispatcher.get_compat_method("alloc")
        self._free_func = _version_dispatcher.get_compat_method("free")

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def schedulable_token_capacity(self, physical_capacity: int) -> int:
        # sglang >= 0.5.18: ModelRunner.effective_max_total_num_tokens /
        # tp_worker / invariant_checker call this. No aux cache in HiSim.
        return physical_capacity

    # ---- sglang >= 0.5.19 HybridReqToTokenPool mamba APIs ----

    def translate_mamba_indices(self, mamba_indices):
        # Identity for a static pool (slots are physical).
        return mamba_indices

    def mamba2_layer_index(self, layer_id: int) -> int:
        assert layer_id in self.mamba_map
        return self.mamba_map[layer_id]

    def mamba2_layer_cache(self, layer_id: int):
        return None

    def get_mamba_indices(self, reqs):
        return torch.tensor(
            [int(r.kv.mamba_pool_idx) for r in reqs],
            dtype=torch.int64,
        )

    def get_speculative_mamba2_params_all_layers(self):
        return None

    def get_state_buf_infos(self):
        return None

    def get_state_dim_per_tensor(self):
        return None

    def get_state_slice_outer_counts(self):
        return None

    def get_state_conv_shard_groups(self):
        return None

    def clone_with_new_mamba(self, *args, **kwargs):
        return self

    def _init_mamba_pool(self, *args, **kwargs):
        pass

    def _alloc_ping_pong_buffer(self, req):
        """Allocate the ping-pong track buffer for a new request.

        Mirrors HybridReqToTokenPool._alloc_ping_pong_buffer. Returns the
        allocated slots tensor (so the caller can roll them back on failure),
        or None if the mamba pool is exhausted.
        """
        n = (
            1
            if self.enable_mamba_extra_buffer_lazy
            else self.mamba_ping_pong_track_buffer_size
        )
        slots = self.mamba_allocator.alloc(n)
        if slots is None:
            return None
        buf = torch.full(
            (self.mamba_ping_pong_track_buffer_size,),
            -1,
            dtype=slots.dtype,
            device=slots.device,
        )
        buf[:n] = slots
        req.kv.mamba_ping_pong_track_buffer = buf
        req.kv.mamba_next_track_idx = 0
        req.kv.mamba_last_track_idx = (
            0
            if self.enable_mamba_extra_buffer_lazy
            else self.get_mamba_ping_pong_other_idx(0)
        )
        return slots

    # ---- sglang >= 0.5.19 aux-cache/row APIs (no-op without an aux cache) ----

    def alloc_aux_to_lengths(self, *, req_pool_indices_cpu, target_seq_lens_cpu):
        pass

    def attach_aux_cache(self, aux_cache):
        pass

    def reset_aux_cache_allocator(self):
        pass

    def alloc_rows(self, need_size: int):
        # Take need_size rows with no Req bound (beam members etc.).
        if need_size > len(self.free_slots):
            return None
        if need_size == 0:
            return []
        select_index = self.free_slots[-need_size:]
        del self.free_slots[-need_size:]
        return select_index

    def free_rows(self, indices):
        self.free_slots.extend(indices)

    # ---- sglang 0.5.18 K3 Mamba/KDA compatibility ----
    # These methods are called by schedule_batch.py's Mamba track/prealloc code.
    # HiSim simulates KDA state separately (MockMambaStatePool), so these are
    # minimal stubs that let the scheduler's K3 code path execute without crash.

    @property
    def mamba_ping_pong_track_buffer_size(self):
        return 2

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        return 1 - mamba_next_track_idx

    def get_mamba_ping_pong_keep_idx(self, req) -> int:
        return getattr(req.kv, "mamba_last_track_idx", 0) or 0

    def set_mamba_ping_pong_slot(self, req, idx: int, value):
        """Update a ping-pong slot value and sync the device-side mapping.

        The req holds the authoritative buffer; this keeps the
        req_index_to_mamba_ping_pong_track_buffer_mapping in sync so that
        set_mamba_track_indices_from_reqs reads correct slot indices.
        """
        buf = getattr(req.kv, "mamba_ping_pong_track_buffer", None)
        if buf is not None:
            buf[idx] = value
            self._mock_pp_mapping[req.kv.req_pool_idx] = buf

    def donate_mamba_ping_pong_slot(self, req, new_slot):
        # Native: donate the keep_idx pp_buf slot to radix cache, replace with new_slot.
        # Returns the donated slot value (shape [1]) for cache insertion.
        donate_idx = self.get_mamba_ping_pong_keep_idx(req)
        mamba_value_donated = (
            req.kv.mamba_ping_pong_track_buffer[donate_idx].unsqueeze(-1).clone()
        )
        self.set_mamba_ping_pong_slot(req, donate_idx, new_slot[0])
        return mamba_value_donated

    def translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:
        # Identity for non-unified pool (Two Pools mode).
        # UnifiedHybridReqToTokenPool overrides for virtual→physical translation.
        return mamba_indices

    def free_mamba_cache(
        self,
        req=None,
        mamba_ping_pong_track_buffer_to_keep=None,
        *args,
        **kwargs,
    ):
        """Free mamba state cache slots when a request finishes.

        [P2A.2] Fix: check if slot was already freed by eviction before
        calling mamba_allocator.free(). If slot is in free_slots, it was
        already freed (stale reference from eviction) — clear the ref
        without calling free() again. The free() method remains strict.
        """
        if req is None:
            if args and hasattr(getattr(args[0], "kv", args[0]), "mamba_pool_idx"):
                req = args[0]
            else:
                return
        import torch as _t
        import logging as _ma_l_log
        _ma_l = _ma_l_log.getLogger("hisim.mock.mamba_allocator")

        _kv = req.kv
        _alloc = self._mock_mamba_allocator if hasattr(self, "_mock_mamba_allocator") else None

        def _safe_free(val, label=""):
            """Free slot only if still allocated (not already freed by eviction).
            
            [P2B] Safety: check val in free_slots to detect stale references
            from eviction. Do NOT use was_reallocated() — it returns True
            forever after one realloc, causing legitimate frees to be skipped
            and the MAMBA pool to leak.
            """
            if val == -1 or val is None:
                return
            if _alloc is not None and val in _alloc.free_slots:
                _ma_l.warning("[MA-STALE] %s slot %d already freed (eviction?), "
                              "clearing stale ref (alloc_count=%d free_count=%d)",
                              label, val, _alloc._alloc_count, _alloc._free_count)
                return  # Don't call free() — slot already freed
            self.mamba_allocator.free(_t.tensor([val], dtype=_t.int64))

        # 1. Always free the active mamba slot (if any).
        _mpi = getattr(_kv, "mamba_pool_idx", None)
        if _mpi is not None:
            _mpi_val = _mpi.item() if hasattr(_mpi, "item") else int(_mpi)
            if _mpi_val != -1:
                _safe_free(_mpi_val, "mamba_pool_idx")
            _kv.mamba_pool_idx = None

        # 2. Free the non-keep ping-pong slot(s).
        _pp = getattr(_kv, "mamba_ping_pong_track_buffer", None)
        if _pp is not None and self.enable_mamba_extra_buffer:
            if mamba_ping_pong_track_buffer_to_keep is not None:
                other_idx = self.get_mamba_ping_pong_other_idx(
                    mamba_ping_pong_track_buffer_to_keep
                )
                _other = _pp[other_idx]
                _other_val = _other.item() if hasattr(_other, "item") else int(_other)
                if _other_val != -1:
                    _safe_free(_other_val, "ping_pong_keep")
            else:
                _vals = []
                for _i in range(len(_pp)):
                    _v = _pp[_i].item() if hasattr(_pp[_i], "item") else int(_pp[_i])
                    if _v != -1:
                        _vals.append(_v)
                if _vals:
                    # Check each slot before batch free
                    _to_free = []
                    for _v in _vals:
                        if _alloc is not None and _v in _alloc.free_slots:
                            _ma_l.warning("[MA-STALE] ping_pong slot %d already freed, clearing", _v)
                        else:
                            _to_free.append(_v)
                    if _to_free:
                        self.mamba_allocator.free(_t.tensor(_to_free, dtype=_t.int64))

        # 3. Clear the req.kv tracking attributes.
        _kv.mamba_ping_pong_track_buffer = None
        _kv.mamba_next_track_idx = None
        _kv.mamba_last_track_idx = None

    @property
    def mamba_allocator(self):
        if not hasattr(self, "_mock_mamba_allocator"):
            import torch as _torch
            _pool_size = _mock_mamba_pool_size()
            # Real alloc/free with fixed capacity — matches One Pool's
            # max_mamba_cache_size constraint. When full, alloc returns
            # None, triggering mamba eviction in the scheduler.
            import logging as _ma_logging
            _ma_l = _ma_logging.getLogger("hisim.mock.mamba_allocator")
            class _StrictMambaAllocator:
                def __init__(self, capacity, device):
                    self.device = device
                    self.capacity = capacity
                    self.free_slots = list(range(1, capacity + 1))
                    self._alloc_count = 0
                    self._free_count = 0
                    # [P2A.2] Slot ledger: {slot_id: [(event, gen, ts, stack)]}
                    self._ledger = {}
                    # [P2B] Generation per slot: incremented on alloc and free
                    # If a stale reference tries to free a slot that was freed
                    # and reallocated, the generation will not match.
                    self._slot_gen = {s: 0 for s in range(1, capacity + 1)}
                    _ma_l.info("[MA-INIT] capacity=%d device=%s free_slots=%s", capacity, device, self.free_slots)
                def available_size(self):
                    return len(self.free_slots)
                def schedulable_available_size(self):
                    return len(self.free_slots)
                def alloc_group_begin(self, n):
                    pass
                def alloc_group_end(self):
                    pass
                def alloc(self, n=1):
                    if n <= 0 or len(self.free_slots) < n:
                        _ma_l.debug("[MA-ALLOC-FAIL] n=%d available=%d", n, len(self.free_slots))
                        return None
                    slots = self.free_slots[:n]
                    self.free_slots = self.free_slots[n:]
                    self._alloc_count += n
                    for s in slots:
                        self._slot_gen[s] += 1
                    _ma_l.debug("[MA-ALLOC] n=%d slots=%s remaining=%d alloc_count=%d",
                                n, slots, len(self.free_slots), self._alloc_count)
                    return _torch.tensor(slots, dtype=_torch.int64, device=self.device)
                def free(self, idx):
                    if idx is None:
                        return
                    _vals = idx.tolist() if hasattr(idx, 'tolist') else [idx]
                    for v in _vals:
                        if v in self.free_slots:
                            _hist = self._ledger.get(v, [])
                            _ma_l.error("[MA-DOUBLE-FREE] slot %d (alloc_count=%d free_count=%d)",
                                        v, self._alloc_count, self._free_count)
                            for _evt, _gen, _cnt, _s in _hist[-3:]:
                                _ma_l.error("  %s gen=%d cnt=%d", _evt, _gen, _cnt)
                            raise AssertionError(
                                f"double free of mamba slot {v} "
                                f"(alloc_count={self._alloc_count}, "
                                f"free_count={self._free_count}, capacity={self.capacity})")
                        self._slot_gen[v] = self._slot_gen.get(v, 0) + 1
                        self.free_slots.append(v)
                        # Only log to ledger on error (not every alloc/free)
                    self._free_count += len(_vals)
                    _ma_l.debug("[MA-FREE] slots=%s remaining=%d alloc_count=%d free_count=%d",
                                _vals, len(self.free_slots), self._alloc_count, self._free_count)
                def get_slot_gen(self, slot_id):
                    """Get current generation of a slot for owner verification."""
                    return self._slot_gen.get(slot_id, -1)
                def was_reallocated(self, slot_id):
                    """Check if slot was freed and reallocated (generation mismatch risk)."""
                    hist = self._ledger.get(slot_id, [])
                    if len(hist) < 2:
                        return False
                    # Check if there's a free followed by an alloc
                    for i in range(len(hist) - 1):
                        if hist[i][0] == "free" and hist[i+1][0] == "alloc":
                            return True
                    return False
                def get_stats(self):
                    return {
                        "capacity": self.capacity,
                        "available": len(self.free_slots),
                        "alloc_count": self._alloc_count,
                        "free_count": self._free_count,
                        "free_slots": list(self.free_slots),
                    }
                def get_ledger(self):
                    """Return slot ownership ledger for diagnostics."""
                    allocated = [s for s, v in self._ledger.items() if v["state"] == "allocated"]
                    freed = [s for s, v in self._ledger.items() if v["state"] == "freed"]
                    return {
                        "allocated_slots": sorted(allocated),
                        "freed_slots": sorted(freed),
                        "total_seen": len(self._ledger),
                        "currently_allocated": len(allocated),
                        "currently_free": len(self.free_slots),
                        "capacity": self.capacity,
                        "alloc_count": self._alloc_count,
                        "free_count": self._free_count,
                        "invariant": len(allocated) + len(self.free_slots) == self.capacity,
                    }
                def clear(self):
                    self.free_slots = list(range(1, self.capacity + 1))
                    self._alloc_count = 0
                    self._free_count = 0
            self._mock_mamba_allocator = _StrictMambaAllocator(_pool_size, self.device)
        return self._mock_mamba_allocator

    @property
    def req_index_to_mamba_ping_pong_track_buffer_mapping(self):
        # Return a tensor that supports indexing by req_pool_indices
        if not hasattr(self, "_mock_pp_mapping"):
            self._mock_pp_mapping = torch.full(
                (self.size, 2), -1, dtype=torch.int64, device=self.device
            )
        return self._mock_pp_mapping

    def alloc(self, *args, **kwargs):
        return self._alloc_func(*args, **kwargs)

    def _alloc_v1(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def _alloc_v2(self, reqs: list) -> Optional[List[int]]:
        # sglang 0.5.9: req_pool_idx is a bare Req attribute. This version is
        # never dispatched on the authoritative 0.5.19 target; mamba allocation
        # lives in _alloc_v3 (mirroring HybridReqToTokenPool.alloc).
        chunked = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        if not any(r.is_dllm() for r in reqs):
            assert len(chunked) <= 1, (
                "only one chunked request may reuse req_pool_idx in a batch"
            )
        assert all(
            getattr(reqs[i], "is_chunked", 0) > 0 or reqs[i].kv_committed_len > 0 for i in chunked
        ), "request has req_pool_idx but is not chunked"

        need_size = len(reqs) - len(chunked)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def _alloc_v3(self, reqs: list) -> Optional[List[int]]:
        # sglang >= 0.5.18: req_pool_idx moved into the Req.kv state object
        # (r.kv.holds_kv / r.kv.req_pool_idx), mirroring the stock
        # ReqToTokenPool.alloc semantics. This version also mirrors
        # HybridReqToTokenPool.alloc's mamba_pool_idx + ping-pong buffer
        # allocation: both live on req.kv for 0.5.18+.
        import logging as _alloc_l
        _ma_l = _alloc_l.getLogger("hisim.mock.mamba_allocator")

        need_size = sum(1 for r in reqs if not r.kv.holds_kv)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        # Track ONLY what this call allocates, so rollback never touches slots
        # owned by a continuing (chunked prefill / COW) request.
        _new_kv_reqs: list = []
        _new_mamba_pairs: list = []  # (req, slot_int)
        _new_pp_pairs: list = []  # (req, slots_tensor)

        def _rollback():
            if _new_pp_pairs:
                import torch as _t
                self.mamba_allocator.free(
                    _t.cat([t for _, t in _new_pp_pairs])
                )
                for _r, _ in _new_pp_pairs:
                    _r.kv.mamba_ping_pong_track_buffer = None
                    _r.kv.mamba_next_track_idx = None
                    _r.kv.mamba_last_track_idx = None
            if _new_mamba_pairs:
                import torch as _t
                self.mamba_allocator.free(
                    _t.tensor([s for _, s in _new_mamba_pairs], dtype=_t.int64)
                )
                for _r, _ in _new_mamba_pairs:
                    _r.kv.mamba_pool_idx = None
            for _r in _new_kv_reqs:
                self.free_slots.append(_r.kv.req_pool_idx)
                _r.kv.req_pool_idx = None

        offset = 0
        for r in reqs:
            if not r.kv.holds_kv:
                r.kv.req_pool_idx = select_index[offset]
                _new_kv_reqs.append(r)
                offset += 1

        mamba_indices: list = []
        for r in reqs:
            if r.kv.holds_mamba:
                # Continuing chunked prefill / radix-cache reuse: slot retained.
                pass
            else:
                _mid = self.mamba_allocator.alloc(1)
                if _mid is None:
                    _ma_l.warning(
                        "[MA-ALLOC-FAIL] rid=%s mamba_pool exhausted "
                        "(available=0, batch=%d), deferring request",
                        getattr(r, "rid", "?"), len(reqs),
                    )
                    _rollback()
                    return None
                r.kv.mamba_pool_idx = _mid[0]
                r.kv.mamba_needs_clear = True
                _new_mamba_pairs.append((r, int(_mid[0])))
            mamba_indices.append(r.kv.mamba_pool_idx)

            if self.enable_mamba_extra_buffer:
                if r.kv.mamba_ping_pong_track_buffer is None:
                    _pp = self._alloc_ping_pong_buffer(r)
                    if _pp is None:
                        _ma_l.warning(
                            "[MA-ALLOC-FAIL] rid=%s ping-pong buffer exhausted, "
                            "deferring request",
                            getattr(r, "rid", "?"),
                        )
                        _rollback()
                        return None
                    _new_pp_pairs.append((r, _pp))

        # Sync mappings (mirrors native alloc).
        for r in reqs:
            _mpi = r.kv.mamba_pool_idx
            _mpi_val = int(_mpi.item()) if hasattr(_mpi, "item") else int(_mpi)
            self.req_index_to_mamba_index_mapping[r.kv.req_pool_idx] = _mpi_val
            if r.kv.mamba_ping_pong_track_buffer is not None:
                self._mock_pp_mapping[r.kv.req_pool_idx] = (
                    r.kv.mamba_ping_pong_track_buffer
                )

        return [r.kv.req_pool_idx for r in reqs]

    def _free_v3(self, req):
        assert req.kv.holds_kv, "request must have req_pool_idx"
        self.free_slots.append(req.kv.req_pool_idx)
        req.kv.req_pool_idx = None

    def free(self, *args, **kwargs):
        return self._free_func(*args, **kwargs)

    def _free_v1(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def _free_v2(self, req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(self.size))


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class MockTokenToKVPool:
    # from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # sglang 0.5.18 compatibility attributes
        self.post_capture_active = False

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None

        # K3 hybrid architecture: FULL (MLA) + MAMBA (KDA) layers.
        # _MambaStrategy.build accesses these to build the hybrid host pool stack.
        self.use_mla = True
        self.full_attention_layer_id_mapping = {i: i for i in range(5)}  # 5 MLA layers
        self.full_kv_pool = self  # the FULL KV pool is this mock itself
        self.kv_lora_rank = 512  # K3 MLA kv_lora_rank
        self.qk_nope_head_dim = 128  # K3 MLA qk_nope_head_dim
        self.qk_rope_head_dim = 64  # K3 MLA qk_rope_head_dim

        # self.head_num = head_num
        # self.head_dim = head_dim
        # Save real values for capacity calculations (MockTokenToKVPoolHost)
        self._real_head_num = head_num
        self._real_head_dim = head_dim
        # Overwrite to 1x1 to save memory (actual buffers stay tiny)
        self.head_num = 1
        self.head_dim = 1
        # sglang 0.5.18 compatibility
        self.v_head_dim = 1  # same as head_dim for MLA models
        self.swa_head_num = 0
        self.swa_head_dim = 0
        self.swa_v_head_dim = 0
        self.use_fia = False
        self.layer_shard_enabled = False

        self.custom_mem_pool = None

        self._create_buffers()
        _is_cuda = "cuda" in str(self.device)

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None
        # self._finalize_allocation_log(size)

    def __getattr__(self, name):
        # Catch-all for attributes accessed by sglang 0.5.18's
        # build_hybrid_mamba_stack and HybridCacheController during
        # two-pool hierarchical cache setup. Returns None for unknown
        # attributes — safe for feature-detection checks (falsy/disabled).
        # This is safe because _MambaStrategy.matches is patched to accept
        # MockTokenToKVPool by class name, not by hasattr checks.
        return None

    def _create_buffers(self):
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.k_buffer = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def register_layer_transfer_counter(self, layer_transfer_counter):
        # required by the hicache backend.
        self.layer_transfer_counter = layer_transfer_counter

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_cpu_copy(self, indices):
        torch.get_device_module(self.device).synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.get_device_module(self.device).synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.get_device_module(self.device).synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.get_device_module(self.device).synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer,  # "RadixAttention"
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        return None


class MockBaseTokenToKVPoolAllocator:
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: MockTokenToKVPool,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    def debug_print(self) -> str:
        return ""

    def verify_byte_accounting(self):
        # sglang >= 0.5.19 scheduler self-check (static pools return []).
        return []

    # -- sglang >= 0.5.19 scheduler-facing capacity hooks (called
    #    unconditionally by ScheduleBatch.check_decode_mem) --

    def evict_to_free_tokens(self, tree_cache, num_tokens: int) -> None:
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        evict_from_tree_cache(tree_cache, num_tokens)

    def check_decode_capacity(self, *, num_tokens: int, tree_cache) -> bool:
        self.evict_to_free_tokens(tree_cache, num_tokens)
        return self.available_size() >= num_tokens

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def free_segment(self, *args, **kwargs):
        pass

    def free_segments(self, *args, **kwargs):
        pass

    def get_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def clear(self):
        raise NotImplementedError()

    def alloc(self, need_size: int):
        raise NotImplementedError()

    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()


class MockTokenToKVPoolAllocator(MockBaseTokenToKVPoolAllocator):
    # from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: MockTokenToKVPool,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)

        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.is_not_in_free_group = True
        self.free_group = []

    def clear(self):
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )

    def alloc(self, need_size: int):
        # It is not necessary to sort the slots during simulation.
        # if self.need_sort and need_size > len(self.free_pages):
        #     self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        self.free_pages = torch.cat((self.free_pages, free_index))


class MockPagedTokenToKVPoolAllocator(MockBaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: MockTokenToKVPool,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)

        self.num_pages = size // page_size
        # self.seen_max_num_extend_tokens_next_power_of_2 = 1
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        # self.seen_max_num_extend_tokens_next_power_of_2 = max(
        #     self.seen_max_num_extend_tokens_next_power_of_2,
        #     next_power_of_2(extend_num_tokens),
        # )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        alloc_extend_cpu(
            pre_lens_ptr=prefix_lens,
            seq_lens_ptr=seq_lens,
            last_loc_ptr=last_loc,
            free_page_ptr=self.free_pages,
            out_indices=out_indices,
            bs_upper=None,
            page_size=self.page_size,
            max_num_extend_tokens=None,
        )

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_cpu(
            seq_lens_ptr=seq_lens,
            last_loc_ptr=last_loc,
            free_page_ptr=self.free_pages,
            out_indices=out_indices,
            bs_upper=None,  # Reserved parameter (not used in CPU)
            page_size=self.page_size,
        )

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


def calc_kv_host_size_per_token(device_pool) -> int:
    """DRAM bytes per MLA token used by the L2 host pool.

    [D1.4-FIX] Use HiSim Kimi-K3 MLA entry_bytes (5760 B/token) as the
    authoritative source. The previous implementation read native device_pool
    attrs (layer_num=61, real_head_num=128, real_head_dim=128, itemsize=4)
    which returns ~10MB/token — 1785× larger than HiSim's simplified MLA
    capacity mode (head_num=1, head_dim=288, entry_bytes=5760).

    This matches MockTokenToKVPoolHost.get_size_per_token() and the HiSim
    launch_server MLA capacity patch (head_dim=288 → entry_bytes=5760).
    """
    # HiSim Kimi-K3 MLA: head_num=1, head_dim=288
    # entry_bytes = 5 × head_num × head_dim × 2(K+V) × 2(bf16) = 5760
    HISIM_MLA_ENTRY_BYTES = 5760

    # Configuration assertion: compare with native device_pool computation
    try:
        layer_num = getattr(device_pool, 'layer_num', 23)
        real_head_num = getattr(device_pool, '_real_head_num', 1)
        real_head_dim = getattr(device_pool, '_real_head_dim', 1)
        dtype = getattr(device_pool, 'store_dtype', torch.float32)
        itemsize = getattr(dtype, 'itemsize', 4)
        native_bytes = int(real_head_dim * real_head_num * layer_num * itemsize * 2)
        if native_bytes != HISIM_MLA_ENTRY_BYTES:
            logger.info(
                f"[D1.4] calc_kv_host_size_per_token: native device_pool gives "
                f"{native_bytes} B/token (layer={layer_num}, head={real_head_num}, "
                f"dim={real_head_dim}, itemsize={itemsize}), using HiSim MLA "
                f"entry_bytes={HISIM_MLA_ENTRY_BYTES} instead"
            )
    except Exception:
        pass

    return HISIM_MLA_ENTRY_BYTES


class MockTokenToKVPoolHost:
    KV_CACHE_BYTES: int = None
    KV_CACHE_BYTES_PER_LAYER: int = None

    MEMORY_READ_BANDWIDTH_BYTES: float = None
    MEMORY_WRITE_BANDWIDTH_BYTES: float = None

    # from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = False
        self.device = device

        self.dtype = getattr(device_pool, 'store_dtype', torch.float32)
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(getattr(device_pool, 'size', 200000) * host_to_device_ratio)
        # Align up the host memory pool size to the page size
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.logical_size = self.size  # used by HybridCacheController
        self.entry_map = {}  # used by append_host_mem_release
        self.start_layer = getattr(device_pool, 'start_layer', 0)
        self.end_layer = getattr(device_pool, 'end_layer', 23)
        self.layer_transfer_counter = None

        _dp_size = getattr(device_pool, 'size', 200000)
        if self.size <= _dp_size:
            logger.warning(
                f"Host memory pool ({self.size} tokens) is not larger than device pool "
                f"({_dp_size} tokens). L2 may evict before L1 does."
            )

        # Verify there is enough available host memory.
        # Mock buffer uses 1x1 per token (not real head_dim), so the actual
        # memory per token matches size_per_token (not the virtual capacity).
        mock_per_token = self.size_per_token
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * mock_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache "
                f"(virtual capacity: {self.size} tokens × {self.size_per_token / 1024:.0f} KB/token "
                f"= {self.size * self.size_per_token / 1e9:.1f} GB virtual)."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

        # Optional unified L2 DRAM pool (KDA/MLA shared budget).  See
        # hisim/simulation/l2_dram_pool.py.  None = legacy flat allocation.
        self._unified = None

    # ---- unified L2 DRAM pool integration --------------------------------
    def attach_unified(self, pool) -> None:
        self._unified = pool

    def get_unified(self):
        return self.__dict__.get('_unified')


    def __getattr__(self, name):
        # Catch-all for attributes accessed by HybridCacheController during
        # two-pool hierarchical cache setup. Returns None for unknown attrs.
        return None

    def get_size_per_token(self):
        self.layer_num = getattr(self.device_pool, 'layer_num', 23)
        # Use real model params for capacity calculation (not mock 1x1)
        real_head_num = getattr(self.device_pool, '_real_head_num', 1)
        real_head_dim = getattr(self.device_pool, '_real_head_dim', 1)
        # Keep mock 1x1 for actual buffer allocation (saves memory)
        self.head_num = getattr(self.device_pool, 'head_num', 1)
        self.head_dim = getattr(self.device_pool, 'head_dim', 576)
        return real_head_dim * real_head_num * self.layer_num * self.dtype.itemsize * 2

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        elif self.layout == "page_head":
            dims = (
                2,
                self.page_num,
                self.head_num,
                self.page_size,
                self.layer_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        buffer = torch.empty(
            dims,
            dtype=self.dtype,
            device=self.device,
        )
        if self.pin_memory and "cuda" in str(self.device):
            torch.cuda.cudart().cudaHostRegister(
                buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0
            )
        return buffer

    def est_bandwidth_batch(self, size_bytes_arr: np.ndarray, cat: str):
        if MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES is None:
            MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES = (
                ConfigManager.get_platform_config().memory_read_bandwidth
            )
        if MockTokenToKVPoolHost.MEMORY_WRITE_BANDWIDTH_BYTES is None:
            MockTokenToKVPoolHost.MEMORY_WRITE_BANDWIDTH_BYTES = (
                ConfigManager.get_platform_config().memory_write_bandwidth
            )
        x = size_bytes_arr.astype(np.float64)
        if cat == "H2D":
            eff = 0.85
            t0 = 6.67e-6
            bw = MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES * eff
        else:
            eff = 0.85
            t0 = 4e-6
            bw = MockTokenToKVPoolHost.MEMORY_WRITE_BANDWIDTH_BYTES * eff
        return x * bw / (t0 * bw + x)

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        # update global clock
        # Merge cache indices
        # https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/kvcacheio/transfer.cu#L713
        assert len(host_indices) == len(device_indices)
        num_indices = len(host_indices)

        host = np.asarray(host_indices, dtype=np.int64)
        dev = np.asarray(device_indices, dtype=np.int64)
        cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
        cut = np.flatnonzero(~cont) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, num_indices]
        seg_len = (ends - starts).astype(np.float64)

        if MockTokenToKVPoolHost.KV_CACHE_BYTES_PER_LAYER is None:
            MockTokenToKVPoolHost.KV_CACHE_BYTES_PER_LAYER = (
                ConfigManager.get_kv_cache_bytes_per_layer()
            )

        bpl = float(MockTokenToKVPoolHost.KV_CACHE_BYTES_PER_LAYER)

        direct_bw = None
        try:
            pcfg = ConfigManager.get_platform_config()
            if pcfg.enable_l3_direct_to_l1:
                direct_bw = pcfg.l3_to_l1_bandwidth
        except Exception:
            pass

        if direct_bw is not None:
            total = num_indices
            l3_count = min(StateManager.get_l3_direct_load_count(), total)
            l2_count = total - l3_count

            if l2_count > 0:
                l2_bytes = l2_count * bpl
                if MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES is None:
                    MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES = (
                        ConfigManager.get_platform_config().memory_read_bandwidth
                    )
                eff_bw = MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES * 0.85
                t0 = 6.67e-6
                h2d_dur = t0 + l2_bytes / eff_bw
                StateManager.inc_hicache_l2_load_dur(h2d_dur)

            if l3_count > 0:
                l3_bytes = l3_count * bpl
                direct_dur = l3_bytes / direct_bw
                StateManager.inc_hicache_l3_direct_load_dur(direct_dur)
            return

        size_bytes_arr = seg_len * bpl
        bandwidth_arr = self.est_bandwidth_batch(size_bytes_arr, cat="H2D")
        total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
        # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead
        StateManager.inc_hicache_l2_load_dur(total_time_cost)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        # update global clock
        num_indices = len(host_indices)

        host = np.asarray(host_indices, dtype=np.int64)
        dev = np.asarray(device_indices, dtype=np.int64)
        cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
        cut = np.flatnonzero(~cont) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, num_indices]
        seg_len = (ends - starts).astype(np.float64)

        if MockTokenToKVPoolHost.KV_CACHE_BYTES is None:
            MockTokenToKVPoolHost.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()

        size_bytes_arr = seg_len * float(MockTokenToKVPoolHost.KV_CACHE_BYTES)
        bandwidth_arr = self.est_bandwidth_batch(size_bytes_arr, cat="D2H")
        total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
        # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead

        StateManager.inc_hicache_l2_backup_dur(total_time_cost)

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        return torch.ones(size=(1, 1)) * index

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        pass

    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        _up = self.__dict__.get('_unified')
        if _up is not None:
            # Largest contiguous MLA allocation the shared pool can serve.
            return _up.mla_available_tokens()
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        _up = self.__dict__.get('_unified')
        if _up is not None:
            # Unified DRAM pool: page/extent allocation, budget shared with KDA.
            return _up.alloc_mla_tokens(need_size)
        assert need_size % self.page_size == 0, (
            "The requested size should be a multiple of the page size."
        )
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        _up = self.__dict__.get('_unified')
        if _up is not None:
            return _up.free_mla_tokens(indices)
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)


def _pass_str_to_block_ids(hash_key):
    int_hash = int(hash_key, 16)
    MAX_18_DIGITS = 10**18
    return int_hash % MAX_18_DIGITS


class MockMambaStatePool:
    """Simulates sglang MambaPool: slot-level KDA state (conv+ssm) management in HBM.
    Unlike MockTokenToKVPool (page/token-level MLA KV), this pool manages fixed-size
    recurrent state slots (14.45 MB each for K3-23L TP=8).
    """

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        slot_bytes: int,
    ):
        self.size = size  # number of slots (e.g. 70 or 165)
        self.dtype = dtype
        self.layer_num = layer_num  # num_linear_attention (18 for K3-23L)
        self.device = device
        self.slot_bytes = slot_bytes  # bytes per slot (14.45MB for K3-23L)
        # Slot allocator: track which slots are in use
        self._free_slots = list(range(size))
        self._used_slots = set()
        # LRU: ordered dict {slot_idx: last_access_time}
        from collections import OrderedDict
        self._lru = OrderedDict()
        # B5: Per-request slot tracking: {req_id: [(slot_idx, token_offset), ...]}
        self._req_slots = {}  # req_id -> list of (slot_idx, token_offset)
        # Statistics
        self._alloc_count = 0
        self._evict_count = 0
        self._cow_count = 0

    def alloc_slot(self, req_id: str = None, token_offset: int = 0) -> int:
        """Allocate a slot. If pool full, LRU evict and return evicted slot.
        If req_id is provided, tracks the slot ownership for per-request management."""
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            slot = self._evict_lru()
        self._used_slots.add(slot)
        self._lru[slot] = self._alloc_count
        self._alloc_count += 1
        if req_id is not None:
            if req_id not in self._req_slots:
                self._req_slots[req_id] = []
            self._req_slots[req_id].append((slot, token_offset))
        return slot

    def free_slot(self, slot_idx: int):
        """Free a slot (return to pool)."""
        if slot_idx in self._used_slots:
            self._used_slots.discard(slot_idx)
            self._free_slots.append(slot_idx)
            self._lru.pop(slot_idx, None)

    def free_slots_for_request(self, req_id: str) -> list:
        """B5: Free all slots owned by a request. Returns list of freed slot indices."""
        freed = []
        if req_id in self._req_slots:
            for slot_idx, _ in self._req_slots[req_id]:
                self.free_slot(slot_idx)
                freed.append(slot_idx)
            del self._req_slots[req_id]
        return freed

    def get_slots_for_request(self, req_id: str) -> list:
        """B5: Get list of (slot_idx, token_offset) for a request."""
        return list(self._req_slots.get(req_id, []))

    def _evict_lru(self) -> int:
        """Evict the least recently used slot. Returns the evicted slot index.
        The caller (hook) is responsible for backing up the evicted slot to L2."""
        if not self._lru:
            raise RuntimeError("MambaStatePool: no slots to evict")
        slot, _ = self._lru.popitem(last=False)
        self._used_slots.discard(slot)
        self._evict_count += 1
        return slot

    def touch(self, slot_idx: int):
        """Update LRU order (mark slot as recently used)."""
        if slot_idx in self._lru:
            self._lru.move_to_end(slot_idx)

    def cow_copy(self, src_slot: int) -> int:
        """COW: allocate a new slot, mark as copy of src. Returns new slot idx.
        In simulation, no actual data copy — just slot allocation."""
        new_slot = self.alloc_slot()
        self._cow_count += 1
        return new_slot

    def get_stats(self) -> dict:
        return {
            "alloc": self._alloc_count,
            "evict": self._evict_count,
            "cow": self._cow_count,
            "free": len(self._free_slots),
            "used": len(self._used_slots),
            "tracked_reqs": len(self._req_slots),
        }


class MockMambaStatePoolHost:
    """Simulates KDA state DRAM backup pool (L2).
    Handles D2H (backup) and H2D (load_back/COW) timing for 14.45MB/slot transfers.
    Uses the same bandwidth model as MockTokenToKVPoolHost but with slot granularity.

    [P2C.3] Tier-Aware Placement: supports pluggable L2 eviction policy.
    Policy is installed via HISIM_L2_POLICY env var. The policy decides
    which L2 slots to evict to L3 when L2 is full. L1 eviction is NOT
    modified — L1 stays native LRU.
    """

    MEMORY_READ_BANDWIDTH_BYTES = None
    MEMORY_WRITE_BANDWIDTH_BYTES = None

    def __init__(self, size: int, slot_bytes: int):
        self.size = size  # number of slots L2 can hold
        self.slot_bytes = slot_bytes  # 14.45 MB per slot
        self.page_size = 1
        self._slots = {}  # {slot_hash: placeholder}
        self.free_slots = list(range(size))
        self.release_slots = []
        self._last_evicted_hashes = []  # hashes evicted during last backup (for L3 promotion)
        from collections import OrderedDict
        self._lru = OrderedDict()  # {slot_hash: access_time}
        self.layer_transfer_counter = None
        # [P2C.3] Tier-Aware L2 placement policy
        self._policy = None  # set lazily via get_l2_policy()
        # Unified L2 DRAM pool integration (None = legacy capacity semantics)
        self._unified = None
        self._evict_hook = None
        self._allocated_idx = set()
        # Round-robin cursor for logical host indices (unified mode only).
        self._rr_cursor = 0

    # ---- unified L2 DRAM pool integration --------------------------------
    def attach_unified(self, pool) -> None:
        self._unified = pool
        self._allocated_idx = set()
        self._rr_cursor = 0

    def get_unified(self):
        return self.__dict__.get('_unified')

    def set_evict_hook(self, fn) -> None:
        """Callback receiving hashes evicted outside the normal backup path
        (e.g. cross-view reclaim).  Default consumer promotes them to L3."""
        self._evict_hook = fn

    def notify_evicted(self, slot_hash: str) -> None:
        """Route an L2 eviction to L3 write-through (existing mechanism)."""
        _hook = self.__dict__.get('_evict_hook')
        if _hook is not None:
            try:
                _hook(slot_hash)
                return
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"KDA evict hook failed: {exc}")
        self._last_evicted_hashes.append(slot_hash)

    def _get_policy(self):
        """Lazily get L2 placement policy (avoids import during __init__)."""
        if self._policy is None:
            try:
                from hisim.simulation.l2_placement_policy import get_l2_policy
                self._policy = get_l2_policy()
            except Exception:
                self._policy = None  # Fallback to native LRU
        return self._policy

    def available_size(self):
        _up = self.__dict__.get('_unified')
        if _up is not None:
            # Shared budget is the capacity authority; the native free list is
            # not maintained in unified mode (see alloc()).
            return _up.kda_available_slots()
        return len(self.free_slots)

    def get_stats(self) -> dict:
        """Return L2 KDA cache pool statistics (instantaneous snapshot)."""
        _up = self.__dict__.get('_unified')
        return {
            "cached_slots": len(self._slots),  # slots currently cached in L2
            "capacity": self.size,             # total L2 capacity
            "free_slots": self.available_size(),
            "lru_entries": len(self._lru),     # LRU tracked entries
        }

    def alloc(self, need_size):
        _up = self.__dict__.get('_unified')
        if _up is not None:
            # Host indices are opaque logical ids in the mock: nothing reads
            # them back, and real capacity is owned by the shared budget
            # (admit_kda).  The native ack path that would return indices to
            # this free list is stubbed out by the synchronous mock write, so
            # a finite list would leak one entry per KDA checkpoint and
            # eventually stop host backups altogether.  Hand out ids
            # round-robin instead; duplicates are harmless because no data
            # movement is keyed by them.
            n = max(int(self.size), 1)
            start = self._rr_cursor
            self._rr_cursor = (start + int(need_size)) % n
            idx = [(start + i) % n for i in range(int(need_size))]
            self._allocated_idx.update(idx)
            return torch.tensor(idx, dtype=torch.int64)
        if need_size > len(self.free_slots):
            return None
        result = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        # Logical host index only: physical KDA bytes are charged at checkpoint
        # admission (backup_from_device), which is the semantic L2 entry point.
        self._allocated_idx.update(result)
        return torch.tensor(result, dtype=torch.int64)

    def free(self, indices):
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        # Idempotent: only recycle indices that are still logically allocated.
        released = [i for i in indices if i in self._allocated_idx]
        for i in released:
            self._allocated_idx.discard(i)
        self.free_slots.extend(released)
        return len(released)

    def __getattr__(self, name):
        # Catch-all for attributes accessed by HybridCacheController
        # and MambaComponent during hierarchical cache operations.
        return None

    def _est_bandwidth(self, size_bytes: float, cat: str) -> float:
        """Saturating bandwidth model: x*bw/(t0*bw + x), same as MockTokenToKVPoolHost."""
        if self.MEMORY_READ_BANDWIDTH_BYTES is None:
            try:
                from hisim.simulation.manager import ConfigManager
                pcfg = ConfigManager.get_platform_config()
                self.MEMORY_READ_BANDWIDTH_BYTES = pcfg.memory_read_bandwidth
                self.MEMORY_WRITE_BANDWIDTH_BYTES = pcfg.memory_write_bandwidth
            except Exception:
                self.MEMORY_READ_BANDWIDTH_BYTES = 64e9  # 64 GB/s default
                self.MEMORY_WRITE_BANDWIDTH_BYTES = 64e9

        if cat == "H2D":
            eff = 0.85
            t0 = 6.67e-6
            bw = self.MEMORY_READ_BANDWIDTH_BYTES * eff
        else:
            eff = 0.85
            t0 = 4e-6
            bw = self.MEMORY_WRITE_BANDWIDTH_BYTES * eff
        return size_bytes * bw / (t0 * bw + size_bytes)

    def backup_from_device(self, slot_idx: int, slot_hash: str, from_l3: bool = False,
                           depth: int = 0):
        """D2H: Backup one slot (MAMBA_SLOT_BYTES) from HBM to DRAM. Accumulates timing.

        If from_l3=True, this is an L3→L2 restore (data already in DRAM from SSD read).
        No additional D2H timing — SSD read timing is handled by the caller.

        [P2C.3] depth parameter: checkpoint depth (token count) for CostAwareL2Policy.
        """
        from hisim.simulation.manager.state import StateManager

        if not from_l3:
            # Normal HBM → DRAM backup (eviction path): count D2H timing
            bw = self._est_bandwidth(self.slot_bytes, cat="D2H")
            dur = self.slot_bytes / bw
            StateManager.inc_hicache_kda_l2_backup_dur(dur)
        # from_l3=True: data already in DRAM (SSD read already timed by caller)
        # No additional timing — just store in L2

        _up = self.__dict__.get('_unified')
        if _up is not None and slot_hash not in self._slots:
            # Charge one KDA slot extent from the shared DRAM budget.  When the
            # budget is exhausted the checkpoint stays in L3 only (the caller
            # still writes it through), so no fake L2 hit can be created.
            if not _up.admit_kda(slot_hash):
                return False

        # Store in L2 (even for L3 load_back: data now in DRAM)
        self._slots[slot_hash] = slot_idx
        self._lru[slot_hash] = len(self._lru)

        # [P2C.3] Record admission for L2 policy
        _pol = self._get_policy()
        if _pol is not None:
            _pol.record_admission(slot_hash, slot_idx, depth)

        # Evict if L2 full — use L2 placement policy if available
        _evicted = []
        while len(self._slots) > self.size:
            if _pol is not None:
                evicted_hash = _pol.select_victim(self)
            else:
                # Fallback: native LRU
                evicted_hash = None
                if self._lru:
                    evicted_hash, _ = self._lru.popitem(last=False)
            if evicted_hash is None:
                break
            self._slots.pop(evicted_hash, None)
            self._lru.pop(evicted_hash, None)
            if _up is not None:
                _up.release_kda(evicted_hash)
            if _pol is not None:
                _pol.record_eviction(evicted_hash)
            _evicted.append(evicted_hash)
        if _evicted:
            # Accumulate: a single write batch may trigger several evictions,
            # and the consumer clears this list after promoting to L3.
            self._last_evicted_hashes.extend(_evicted)

    def load_to_device(self, slot_hash: str) -> int:
        """H2D: Load one slot (14.45MB) from DRAM to HBM (COW hit). Returns slot idx.
        Accumulates timing. Returns -1 if not in L2."""
        from hisim.simulation.manager.state import StateManager

        if slot_hash not in self._slots:
            return -1

        bw = self._est_bandwidth(self.slot_bytes, cat="H2D")
        dur = self.slot_bytes / bw
        StateManager.inc_hicache_kda_l2_load_dur(dur)

        # Update LRU
        self._lru.move_to_end(slot_hash)

        # [P2C.3.2] record_access moved to _mock_start_loading (L2 lookup)
        # to capture both hits AND misses. Keeping it here would double-count.
        # _pol = self._get_policy()
        # if _pol is not None:
        #     _pol.record_access(slot_hash, StateManager.get_global_clock())
        return self._slots[slot_hash]

    def evict(self) -> str:
        """Evict one slot from L2 (for L3 backup). Returns slot_hash or None.

        [P2C.3] Uses L2 placement policy if available, otherwise native LRU.
        """
        _pol = self._get_policy()
        if _pol is not None:
            victim = _pol.select_victim(self)
            if victim is None:
                return None
            self._slots.pop(victim, None)
            self._lru.pop(victim, None)
            _up = self.__dict__.get('_unified')
            if _up is not None:
                _up.release_kda(victim)
            _pol.record_eviction(victim)
            return victim
        # Fallback: native LRU
        if not self._lru:
            return None
        slot_hash, _ = self._lru.popitem(last=False)
        self._slots.pop(slot_hash, None)
        _up = self.__dict__.get('_unified')
        if _up is not None:
            _up.release_kda(slot_hash)
        return slot_hash

    def has(self, slot_hash: str) -> bool:
        return slot_hash in self._slots


class MockHiCacheStorage:
    _DEBUG_BATCH_SET_COUNT = 0
    _DEBUG_BATCH_EXISTS_COUNT = 0
    _DEBUG_BATCH_EXISTS_HITS = 0

    def __init__(self, *args, **kwargs):
        self._last_per_tier_hits = None
        self._last_per_tier_write = None
        if kvcm_py_optimizer is not None:
            logger.info("Using KVCM HiCache storage")
            self.init_kvcm()
        else:
            logger.info("Using common set HiCache storage")
            self.storage: set = set()
            self.storage_file_path: str = "/tmp/hisim/hicache/storage_keys.txt"
            os.makedirs(os.path.dirname(self.storage_file_path), exist_ok=True)

            if os.path.exists(self.storage_file_path):
                with open(self.storage_file_path) as f:
                    line = f.readline()
                    while line:
                        self.storage.add(line.strip())
                        line = f.readline()

        if Envs.reset_hicache_storage():
            logger.info(
                "Cleared KV cache saved in the storage backend because the system environment variable (`HISIM_RESET_HICACHE_STORAGE`) is set."
            )
            if hasattr(self, "storage_manager"):
                self.storage_manager.ClearAllCaches()
            else:
                with open(self.storage_file_path, "w") as f:
                    pass

    def init_kvcm(self):
        # Initialize kvcm based on reference examples
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory: {self.temp_dir}")

        config_path = os.environ.get("HISIM_KVCM_CONFIG_PATH") or kvcm_py_optimizer.get_config_path()

        # If HiSim config defines storage_tiers, override kvcm's storages[] section
        try:
            from hisim.simulation.manager import ConfigManager

            platform_cfg = ConfigManager.get_platform_config()
            hisim_tiers = platform_cfg.storage_tiers
        except Exception:
            hisim_tiers = []

        if hisim_tiers:
            import json as _json

            with open(config_path) as f:
                kvcm_config = _json.load(f)
            storages = []
            for tier in hisim_tiers:
                storages.append({
                    "unique_name": tier.name,
                    "storage_type": tier.kvcm_storage_type,
                    "band_width_mbps": tier.kvcm_bandwidth_mbps,
                    "capacity": int(tier.kvcm_capacity_gb * 1e9),
                    "priority": tier.kvcm_priority,
                })
            kvcm_config["instance_groups"][0]["storages"] = storages
            tmp_config = os.path.join(self.temp_dir, "kvcm_config.json")
            with open(tmp_config, "w") as f:
                _json.dump(kvcm_config, f)
            config_path = tmp_config
            logger.info(f"Overrode kvcm storages with {len(hisim_tiers)} HiSim tiers")

        self.config_loader = kvcm_py_optimizer.OptimizerConfigLoader()

        if not self.config_loader.load(str(config_path)):
            raise RuntimeError(f"Failed to load optimizer config from {config_path}")
        self.config = self.config_loader.config()
        self.storage_manager = kvcm_py_optimizer.OptimizerManager(self.config)
        self.storage_manager.Init()

        # Multi-instance not supported yet; using a single shared instance_id.
        self.instance_id = "3780643326877293460"

    def tearDown(self):
        if hasattr(self, "temp_dir"):
            import shutil

            logger.info("Cleaning up temporary directory")
            shutil.rmtree(self.temp_dir)

    def register_mem_pool_host(self, mem_pool_host):
        pass

    # sglang 0.5.18 compatibility: new v2 registration method
    def register_mem_host_pool_v2(self, *args, **kwargs):
        pass

    # sglang 0.5.18 compatibility: batch host pool registration
    def register_mem_host_pool_batch(self, *args, **kwargs):
        pass

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            return True
        if hasattr(self, "storage_manager"):
            # KVCM backend: write through the kvcm interface. The common-set
            # path below only exists for the non-kvcm fallback; without this
            # branch `self.storage` is undefined and every L3 write raised
            # AttributeError (silently caught by the caller).
            try:
                int_hash_keys = [_pass_str_to_block_ids(str(key))]
                self.storage_manager.WriteCache(
                    self.instance_id, "1", int(time.time() * 1000), int_hash_keys
                )
                return True
            except Exception as e:
                logger.debug(f"KVCM set failed for {key}: {e}")
                return False
        self.storage.add(key)
        with open(self.storage_file_path, "a+") as f:
            f.write(key + "\n")
        return True

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        extra_info: HiCacheStorageExtraInfo = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if hasattr(self, "storage_manager"):
            if extra_info and extra_info.prefix_keys is not None:
                complete_prefix_hashs = extra_info.prefix_keys + keys
            else:
                complete_prefix_hashs = keys
            int_hash_keys = [
                _pass_str_to_block_ids(key) for key in complete_prefix_hashs
            ]
            # insert to kvcm
            trace_id = "1"
            write_timestamp = int(time.time() * 1000)
            res = self.storage_manager.WriteCache(
                self.instance_id,
                trace_id,
                write_timestamp,
                int_hash_keys,
            )
            MockHiCacheStorage._DEBUG_BATCH_SET_COUNT += 1
            # Cache per-tier write block counts for get_last_write_per_tier()
            if (hasattr(res, "tier_names") and hasattr(res, "per_tier_write_block_num")
                    and res.tier_names):
                self._last_per_tier_write = dict(
                    zip(res.tier_names, res.per_tier_write_block_num)
                )
            else:
                self._last_per_tier_write = {"default": res.kvcm_write_length}
            return True
        else:
            for key, value in zip(keys, values):
                if not self.set(key, value):
                    return False
            return True

    def exists(self, key: str) -> bool:
        if hasattr(self, "storage_manager"):
            # KVCM backend: query the kvcm interface. `self.storage` is only
            # populated by the non-kvcm fallback, so the old `key in
            # self.storage` form raised AttributeError under KVCM.
            try:
                int_hash_keys = [_pass_str_to_block_ids(str(key))]
                res = self.storage_manager.GetCacheLocation(
                    self.instance_id,
                    "2",
                    int(time.time() * 1000),
                    int_hash_keys,
                    0,
                    len(int_hash_keys) * _KVCM_BLOCK_SIZE,
                )
                return res.kvcm_hit_length > 0
            except Exception as e:
                logger.debug(f"KVCM exists failed for {key}: {e}")
                return False
        return key in self.storage

    def batch_exists(self, keys: List[str], extra_info) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        if hasattr(self, "storage_manager"):
            int_hash_keys = [_pass_str_to_block_ids(key) for key in keys]
            # Call the kvcm interface to match L3 prefix
            trace_id = "2"
            read_timestamp = int(time.time() * 1000)
            block_mask = 0
            input_len = len(int_hash_keys) * _KVCM_BLOCK_SIZE
            res = self.storage_manager.GetCacheLocation(
                self.instance_id,
                trace_id,
                read_timestamp,
                int_hash_keys,
                block_mask,
                input_len,
            )
            MockHiCacheStorage._DEBUG_BATCH_EXISTS_COUNT += 1
            MockHiCacheStorage._DEBUG_BATCH_EXISTS_HITS += res.kvcm_hit_length
            logger.debug(f"{res.kvcm_hit_length=}")
            # Cache per-tier hits for batch_exists_per_tier()
            if hasattr(res, "tier_names") and hasattr(res, "per_tier_hit_block_num") and res.tier_names:
                self._last_per_tier_hits = dict(zip(res.tier_names, res.per_tier_hit_block_num))
            else:
                self._last_per_tier_hits = {"default": res.kvcm_hit_length}
            return res.kvcm_hit_length
        else:
            for i in range(len(keys)):
                if not self.exists(keys[i]):
                    return i
            return len(keys)

    def batch_exists_per_tier(self) -> dict:
        """Return per-tier hit counts from the last batch_exists() call. No extra kvcm call."""
        if self._last_per_tier_hits is not None:
            return self._last_per_tier_hits
        return {"default": 0}

    def get_last_write_per_tier(self) -> dict:
        """Return per-tier newly-inserted write block counts from the last batch_set() call.
        No extra kvcm call. Mirrors batch_exists_per_tier() pattern."""
        if self._last_per_tier_write is not None:
            return self._last_per_tier_write
        return {"default": 0}

    def clear(self) -> bool:
        if hasattr(self, "storage_manager"):
            logger.info("Clear all storage cache in kvcm.")
            self.storage_manager.ClearAllCaches()
            return True
        else:
            self.storage.clear()
            with open(self.storage_file_path, "w"):
                pass
            return True

    def batch_set_mamba(self, slot_hashes: List[str]) -> bool:
        """L3 write: KDA state slots (slot-level, not token-level).
        Uses 'mamba_' prefix to distinguish from MLA KV in kvcm."""
        if hasattr(self, "storage_manager"):
            prefixed_keys = [f"mamba_{h}" for h in slot_hashes]
            # [P0-1 FIX] Convert descriptive hash keys to hex via MD5
            # so _pass_str_to_block_ids can parse them as int(,16)
            import hashlib
            int_hash_keys = [_pass_str_to_block_ids(hashlib.md5(k.encode()).hexdigest()) for k in prefixed_keys]
            trace_id = "1"
            write_timestamp = int(time.time() * 1000)
            res = self.storage_manager.WriteCache(
                self.instance_id, trace_id, write_timestamp, int_hash_keys,
            )
            MockHiCacheStorage._DEBUG_BATCH_SET_COUNT += 1
            return True
        else:
            for h in slot_hashes:
                self.storage.add(f"mamba_{h}")
                with open(self.storage_file_path, "a+") as f:
                    f.write(f"mamba_{h}\n")
            return True

    def batch_exists_mamba(self, slot_hashes: List[str]) -> dict:
        """L3 query: KDA state slots. Returns {slot_hash: hit(1/0)}.
        Unlike MLA's batch_exists (returns consecutive hit length),
        KDA slots are independent — each slot is a hit or miss."""
        if hasattr(self, "storage_manager"):
            prefixed_keys = [f"mamba_{h}" for h in slot_hashes]
            # [P0-1 FIX] Convert descriptive hash keys to hex via MD5
            import hashlib
            int_hash_keys = [_pass_str_to_block_ids(hashlib.md5(k.encode()).hexdigest()) for k in prefixed_keys]
            trace_id = "2"
            read_timestamp = int(time.time() * 1000)
            block_mask = 0
            input_len = len(int_hash_keys) * _KVCM_BLOCK_SIZE
            res = self.storage_manager.GetCacheLocation(
                self.instance_id, trace_id, read_timestamp,
                int_hash_keys, block_mask, input_len,
            )
            MockHiCacheStorage._DEBUG_BATCH_EXISTS_COUNT += 1
            # KDA: kvcm_hit_length gives consecutive prefix hits
            # For slot-level, we interpret >0 as hit for that slot
            hits = {}
            for i, h in enumerate(slot_hashes):
                hits[h] = 1 if i < res.kvcm_hit_length else 0
            MockHiCacheStorage._DEBUG_BATCH_EXISTS_HITS += sum(hits.values())
            return hits
        else:
            return {h: (1 if f"mamba_{h}" in self.storage else 0) for h in slot_hashes}
