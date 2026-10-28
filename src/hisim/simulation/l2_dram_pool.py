"""Minimal KDA/MLA Unified DRAM Pool (L2 host tier).

Goal
----
Give KDA (mamba checkpoints) and MLA (full-attention KV tokens) **one DRAM
byte budget** while keeping their logical objects, metadata and restore paths
separate:

    +---------------------------------------------------------------+
    |  L2 DRAM budget (HISIM_L2_TOTAL_GB / --hicache-size)           |
    |                                                               |
    |   Physical layer: page -> extent allocator                     |
    |        page_bytes = HISIM_L2_PAGE_TOKENS * mla_token_bytes     |
    |        MLA from the low end, KDA from the high end            |
    |                                                               |
    |   Logical MLA view                 Logical KDA view            |
    |   - object: token / block          - object: checkpoint slot   |
    |   - index: token id (int64)        - key:    slot hash (str)   |
    |   - charged on alloc / free        - charged on admit / evict  |
    +---------------------------------------------------------------+

Modes
-----
``two_pools``  Two fixed regions (MLA region + KDA region) carved from the
               *same* total budget.  No borrowing.  This is the fair static
               baseline for fragmentation / occupancy comparisons.
``one_pool``   One region shared dynamically.  When one view runs out of
               contiguous pages it reclaims from the other view:
                 * MLA short  -> evict KDA slots (existing L2 -> L3 path,
                   policy driven by ``l2_placement_policy``);
                 * KDA short  -> same-view LRU replacement first, then evict
                   MLA entries through the native ``cache.evict_host(n, FULL)``
                   path (only when L3 write-through is enabled); otherwise
                   admission fails and the checkpoint stays in L3 only.
               ``HISIM_L2_BORROW_POLICY=self_first`` (default) does
               replace-own-first then borrow; ``other_first`` borrows first
               (demand driven, can starve the other view).

Fragmentation model
-------------------
``internal``  pages reserved for an extent minus the bytes the logical object
              asked for (tail waste).  e.g. a 512-token MLA block on a
              16-token page wastes < 15 tokens.
``external``  free pages that are *not* part of the largest free run, i.e.
              memory that exists but cannot satisfy a large contiguous
              request.  With the ``pow2`` size-class policy internal
              fragmentation grows (round up to a power of two pages) while
              external fragmentation shrinks; ``exact`` is the opposite.

Nothing here touches L1 eviction, L1 locks, the ack lifecycle or the L3
write-through / restore paths.  L2 eviction order is still decided by the
existing policy layer.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_KDA_SLOT_BYTES = 14_653_440  # K3 conv+ssm state per slot
DEFAULT_PAGE_TOKENS = 16
VALID_POLICIES = ("exact", "pow2")


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# physical layer: page -> extent
# ---------------------------------------------------------------------------
@dataclass
class Extent:
    """A physically reserved contiguous page range."""

    owner: str  # "mla" | "kda"
    start_page: int
    pages: int
    req_bytes: int
    eid: int = 0
    live: int = 0  # live logical units (tokens for MLA)


class PageExtentAllocator:
    """Address-ordered first-fit page-extent allocator with coalescing.

    ``policy="exact"`` serves exactly ``ceil(bytes/page_bytes)`` pages.
    ``policy="pow2"`` rounds the request up to a power-of-two page count
    (classic size-class allocation) which reduces external fragmentation at
    the cost of internal fragmentation.
    """

    def __init__(self, total_pages: int, page_bytes: int, policy: str = "exact"):
        if total_pages <= 0 or page_bytes <= 0:
            raise ValueError("total_pages and page_bytes must be positive")
        if policy not in VALID_POLICIES:
            raise ValueError(f"policy must be one of {VALID_POLICIES}, got {policy!r}")
        self.total_pages = int(total_pages)
        self.page_bytes = int(page_bytes)
        self.policy = policy
        # Free runs, address sorted, coalesced. [[start_page, pages], ...]
        self._runs: List[List[int]] = [[0, self.total_pages]]
        self._free_pages = self.total_pages
        self._used_pages = 0
        self._internal_bytes = 0
        self._peak_used_pages = 0
        self._alloc_ok = 0
        self._alloc_fail = 0
        self._free_calls = 0
        self._lock = threading.RLock()

    # -- size classes -------------------------------------------------------
    def class_pages(self, pages: int) -> int:
        n = max(1, int(pages))
        if self.policy == "exact":
            return n
        p = 1
        while p < n:
            p <<= 1
        return min(p, self.total_pages)

    # -- allocation ---------------------------------------------------------
    def alloc(
        self,
        pages: int,
        req_bytes: int,
        owner: str,
        eid: int = 0,
        from_high: bool = False,
    ) -> Optional[Extent]:
        """Reserve an extent of ``class_pages(pages)`` pages.

        ``from_high`` takes the extent from the *tail* of the best-fitting run
        instead of the head.  The MLA view allocates from the low end and the
        KDA view from the high end, so the two object shapes grow towards each
        other and only fragment where they meet (two-ended / stack-heap
        split).  Both ends still share one address space and one budget.
        """
        need = self.class_pages(pages)
        need = min(need, self.total_pages)
        with self._lock:
            runs = reversed(self._runs) if from_high else iter(self._runs)
            for run in runs:
                if run[1] >= need:
                    if from_high:
                        start = run[0] + run[1] - need
                    else:
                        start = run[0]
                    run[0] = run[0] if from_high else start + need
                    run[1] -= need
                    if run[1] == 0:
                        self._runs.remove(run)
                    self._free_pages -= need
                    self._used_pages += need
                    if self._used_pages > self._peak_used_pages:
                        self._peak_used_pages = self._used_pages
                    self._internal_bytes += need * self.page_bytes - int(req_bytes)
                    self._alloc_ok += 1
                    return Extent(
                        owner=owner,
                        start_page=start,
                        pages=need,
                        req_bytes=int(req_bytes),
                        eid=eid,
                    )
            self._alloc_fail += 1
            return None

    def free(self, extent: Extent) -> None:
        with self._lock:
            start, pages = extent.start_page, extent.pages
            # Insert keeping address order, then coalesce with neighbours.
            idx = 0
            while idx < len(self._runs) and self._runs[idx][0] < start:
                idx += 1
            self._runs.insert(idx, [start, pages])
            merged: List[List[int]] = []
            for run in self._runs:
                if merged and merged[-1][0] + merged[-1][1] == run[0]:
                    merged[-1][1] += run[1]
                else:
                    merged.append(run)
            self._runs = merged
            self._free_pages += pages
            self._used_pages -= pages
            self._internal_bytes -= pages * self.page_bytes - extent.req_bytes
            self._free_calls += 1

    # -- introspection ------------------------------------------------------
    def largest_free_run(self) -> int:
        with self._lock:
            return max((r[1] for r in self._runs), default=0)

    @property
    def free_pages(self) -> int:
        return self._free_pages

    @property
    def used_pages(self) -> int:
        return self._used_pages

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            free_pages = self._free_pages
            largest = max((r[1] for r in self._runs), default=0)
            internal = max(0, self._internal_bytes)
            external_pages = max(0, free_pages - largest)
            return {
                "total_pages": self.total_pages,
                "used_pages": self._used_pages,
                "free_pages": free_pages,
                "peak_used_pages": self._peak_used_pages,
                "free_run_count": len(self._runs),
                "largest_free_run_pages": largest,
                "internal_frag_bytes": internal,
                "external_frag_bytes": external_pages * self.page_bytes,
                "internal_frag_ratio": (internal / (self._used_pages * self.page_bytes))
                if self._used_pages
                else 0.0,
                "external_frag_ratio": (external_pages / free_pages) if free_pages else 0.0,
                "alloc_ok": self._alloc_ok,
                "alloc_fail": self._alloc_fail,
                "free_calls": self._free_calls,
                "policy": self.policy,
            }


# ---------------------------------------------------------------------------
# logical views
# ---------------------------------------------------------------------------
class MLAView:
    """MLA token/block view.  Identity = token index (int64), unchanged.

    Allocation is page aligned: a request of ``n`` tokens reserves
    ``ceil(n / page_tokens)`` pages and returns the first ``n`` token ids of
    that extent.  Token ids stay dense inside the extent, so existing
    ``host_value`` / ``get_data_page(index)`` semantics are untouched.
    """

    def __init__(
        self,
        allocator: PageExtentAllocator,
        token_bytes: int,
        page_tokens: int,
        cap_pages: Optional[int] = None,
    ):
        self.alloc = allocator
        self.token_bytes = int(token_bytes)
        self.page_tokens = int(page_tokens)
        self.cap_pages = cap_pages
        self.extents: Dict[int, Extent] = {}
        self._next_eid = 1
        self._used_pages = 0
        max_tokens = allocator.total_pages * self.page_tokens
        # token -> extent id (0 = not allocated)
        self.owner = np.zeros(max_tokens, dtype=np.int32)

    # -- helpers ------------------------------------------------------------
    def pages_for(self, n_tokens: int) -> int:
        return max(1, math.ceil(int(n_tokens) / self.page_tokens))

    def available_pages(self) -> int:
        pages = self.alloc.largest_free_run()
        if self.cap_pages is not None:
            pages = min(pages, max(0, self.cap_pages - self._used_pages))
        return pages

    @property
    def used_pages(self) -> int:
        return self._used_pages

    def available_tokens(self) -> int:
        return self.available_pages() * self.page_tokens

    # -- alloc / free -------------------------------------------------------
    def alloc_tokens(self, n_tokens: int):
        import torch

        n = int(n_tokens)
        if n <= 0:
            return torch.empty(0, dtype=torch.int64)

        pages = self.pages_for(n)
        if self.cap_pages is not None and self._used_pages + pages > self.cap_pages:
            return None
        eid = self._next_eid
        ext = self.alloc.alloc(pages, n * self.token_bytes, owner="mla", eid=eid)
        if ext is None:
            return None

        self._next_eid += 1
        ext.live = n
        self.extents[eid] = ext
        self._used_pages += ext.pages
        start_tok = ext.start_page * self.page_tokens
        end_tok = start_tok + ext.pages * self.page_tokens
        self.owner[start_tok:end_tok] = eid
        idx = np.arange(start_tok, start_tok + n, dtype=np.int64)
        return torch.from_numpy(idx)

    def free_tokens(self, indices) -> int:
        if indices is None:
            return 0
        if hasattr(indices, "detach"):
            idx = indices.detach().to("cpu").numpy().astype(np.int64, copy=False)
        else:
            idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return 0

        ids = self.owner[idx]
        uniques, counts = np.unique(ids[ids != 0], return_counts=True)
        for eid, cnt in zip(uniques.tolist(), counts.tolist()):
            ext = self.extents.get(int(eid))
            if ext is None:
                continue
            ext.live -= int(cnt)
            if ext.live <= 0:
                start_tok = ext.start_page * self.page_tokens
                end_tok = start_tok + ext.pages * self.page_tokens
                self.owner[start_tok:end_tok] = 0
                self._used_pages -= ext.pages
                self.alloc.free(ext)
                self.extents.pop(int(eid), None)
        return int(idx.size)

    # -- stats --------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "extents": len(self.extents),
            "used_pages": self._used_pages,
            "live_tokens": sum(e.live for e in self.extents.values()),
            "available_tokens": self.available_tokens(),
        }


class KDAView:
    """KDA checkpoint view.  Identity = slot hash (content addressed).

    A checkpoint is charged one slot extent at admission
    (``backup_from_device``) and released on eviction.  The logical slot hash
    metadata stays in ``MockMambaStatePoolHost._slots``.
    """

    def __init__(self, allocator: PageExtentAllocator, slot_bytes: int):
        self.alloc = allocator
        self.slot_bytes = int(slot_bytes)
        self.slot_pages = max(1, math.ceil(self.slot_bytes / allocator.page_bytes))
        self.slots: Dict[str, Extent] = {}
        self._used_pages = 0

    def has(self, key: str) -> bool:
        return key in self.slots

    def available_slots(self) -> int:
        return self.alloc.largest_free_run() // self.slot_pages

    @property
    def used_pages(self) -> int:
        return self._used_pages

    def admit(self, key: str) -> bool:
        if key in self.slots:
            return True
        ext = self.alloc.alloc(self.slot_pages, self.slot_bytes, owner="kda",
                               from_high=True)
        if ext is None:
            return False
        self.slots[key] = ext
        self._used_pages += ext.pages
        return True

    def release(self, key: str) -> bool:
        ext = self.slots.pop(key, None)
        if ext is None:
            return False
        self._used_pages -= ext.pages
        self.alloc.free(ext)
        return True

    def stats(self) -> Dict[str, Any]:
        return {
            "cached_slots": len(self.slots),
            "available_slots": self.available_slots(),
            "slot_pages": self.slot_pages,
            "used_pages": self._used_pages,
            "logical_bytes": len(self.slots) * self.slot_bytes,
        }


# ---------------------------------------------------------------------------
# unified pool
# ---------------------------------------------------------------------------
class UnifiedDRAMPool:
    """One DRAM budget, one physical allocator, two logical views."""

    def __init__(
        self,
        total_bytes: int,
        mode: str = "two_pools",
        token_bytes: int = 5760,
        kda_slot_bytes: int = DEFAULT_KDA_SLOT_BYTES,
        page_tokens: int = DEFAULT_PAGE_TOKENS,
        policy: str = "exact",
        kda_region_bytes: Optional[int] = None,
        kda_floor_slots: int = 0,
        kv_evict: bool = True,
        borrow_policy: str = "self_first",
    ):
        if mode not in ("two_pools", "one_pool"):
            raise ValueError(f"bad mode {mode!r}")
        if page_tokens < 1:
            raise ValueError("page_tokens must be >= 1")

        self.mode = mode
        self.policy = policy
        self.token_bytes = int(token_bytes)
        self.kda_slot_bytes = int(kda_slot_bytes)
        self.page_tokens = int(page_tokens)
        self.page_bytes = self.page_tokens * self.token_bytes
        budget_pages = int(total_bytes) // self.page_bytes
        if budget_pages <= 0:
            raise ValueError("DRAM budget smaller than one page")
        self.budget_pages = budget_pages
        self.total_bytes = budget_pages * self.page_bytes
        self.kv_evict = bool(kv_evict)
        self.borrow_policy = (
            borrow_policy if borrow_policy in ("other_first", "self_first") else "self_first"
        )

        # runtime wiring
        self._kda_pools: List[Any] = []
        self._mla_pool: Any = None
        self._mla_evict_fn: Optional[Callable[[int], Any]] = None
        self._lock = threading.RLock()

        # counters
        self.alloc_fail_mla = 0
        self.alloc_fail_kda = 0
        self.cross_evict_kda_slots = 0
        self.cross_evict_kda_bytes = 0
        self.cross_evict_mla_tokens = 0
        self.cross_reclaim_calls = 0
        self.admission_reject_kda = 0
        self.kda_floor_bytes = 0

        if mode == "two_pools":
            kda_bytes = int(kda_region_bytes or 0)
            kda_pages = min(budget_pages, max(0, kda_bytes // self.page_bytes))
            mla_pages = budget_pages - kda_pages
            self.mla_alloc = PageExtentAllocator(mla_pages, self.page_bytes, policy)
            self.kda_alloc = PageExtentAllocator(kda_pages, self.page_bytes, policy)
            self.mla = MLAView(self.mla_alloc, token_bytes, page_tokens)
            self.kda = KDAView(self.kda_alloc, kda_slot_bytes)
            self.mla_capacity_bytes = mla_pages * self.page_bytes
            self.kda_capacity_bytes = kda_pages * self.page_bytes
            self._kda_pages = kda_pages
        else:
            floor_pages = 0
            if kda_floor_slots > 0:
                floor_pages = min(
                    budget_pages,
                    math.ceil(kda_floor_slots * self.kda_slot_bytes / self.page_bytes),
                )
            self.kda_floor_bytes = floor_pages * self.page_bytes
            self.shared_alloc = PageExtentAllocator(budget_pages, self.page_bytes, policy)
            self.mla = MLAView(
                self.shared_alloc,
                token_bytes,
                page_tokens,
                cap_pages=budget_pages - floor_pages,
            )
            self.kda = KDAView(self.shared_alloc, kda_slot_bytes)
            self.mla_capacity_bytes = (budget_pages - floor_pages) * self.page_bytes
            self.kda_capacity_bytes = budget_pages * self.page_bytes
            self._kda_pages = budget_pages

        self.kda_floor_slots = int(kda_floor_slots)
        logger.info(
            "L2 unified DRAM pool: mode=%s total=%.2fGB page=%dB (%d tok) "
            "policy=%s mla=%.2fGB kda=%.2fGB (%d slots) kv_evict=%s",
            mode,
            self.total_bytes / 1e9,
            self.page_bytes,
            self.page_tokens,
            policy,
            self.mla_capacity_bytes / 1e9,
            self.kda_capacity_bytes / 1e9,
            self.kda_capacity_slots,
            kv_evict,
        )

    # -- capacity hints used by the hook -----------------------------------
    @property
    def mla_capacity_tokens(self) -> int:
        return (self.mla_capacity_bytes // self.page_bytes) * self.page_tokens

    @property
    def kda_capacity_slots(self) -> int:
        # Achievable slots: one slot spans ``slot_pages`` whole pages, so page
        # rounding (internal fragmentation) can cost a slot at the boundary.
        return max(0, self._kda_pages // max(1, self.kda.slot_pages))

    # -- wiring -------------------------------------------------------------
    def bind_mla_pool(self, pool: Any) -> None:
        self._mla_pool = pool

    def register_kda_pool(self, pool: Any) -> None:
        if pool is not None and all(pool is not p for p in self._kda_pools):
            self._kda_pools.append(pool)

    def set_mla_evictor(self, fn: Optional[Callable[[int], Any]]) -> None:
        """``fn(bytes_to_free) -> freed_bytes`` implemented by the native
        ``cache.evict_host`` path.  ``None`` disables MLA reclaim."""
        self._mla_evict_fn = fn

    @property
    def mla_reclaim_enabled(self) -> bool:
        return self.mode == "one_pool" and self.kv_evict and self._mla_evict_fn is not None

    # -- MLA API ------------------------------------------------------------
    def alloc_mla_tokens(self, n_tokens: int):
        idx = self.mla.alloc_tokens(n_tokens)
        if idx is not None:
            return idx
        if self.mode == "one_pool":
            pages = self.mla.pages_for(n_tokens)
            need_pages = self.mla.alloc.class_pages(pages)
            if self.mla.cap_pages is not None and \
                    self.mla.used_pages + pages > self.mla.cap_pages:
                # Reserved KDA floor: reclaiming KDA cannot help.
                self.alloc_fail_mla += 1
                return None
            for _ in range(4):
                avail = self.mla.available_pages()
                if avail >= need_pages:
                    break
                deficit = (need_pages - avail) * self.page_bytes
                if self._reclaim_from_kda(deficit) <= 0:
                    break
                idx = self.mla.alloc_tokens(n_tokens)
                if idx is not None:
                    return idx
        self.alloc_fail_mla += 1
        return None

    def free_mla_tokens(self, indices) -> int:
        return self.mla.free_tokens(indices)

    def mla_available_tokens(self) -> int:
        return self.mla.available_tokens()

    # -- KDA API ------------------------------------------------------------
    def _kda_replace_self(self, key: str, need_pages: int) -> bool:
        """Same-view replacement: evict KDA checkpoints (LRU / placement
        policy) until the slot fits.  Keeps baseline L2 semantics (a full L2
        replaces entries instead of refusing new ones)."""
        for _ in range(8):
            missing = need_pages - self.kda.alloc.largest_free_run()
            if missing <= 0:
                break
            if self._reclaim_from_kda(missing * self.page_bytes) <= 0:
                break
            if self.kda.admit(key):
                return True
        return False

    def _kda_borrow_other(self, key: str, need_pages: int) -> bool:
        """Cross-view borrow: reclaim MLA pages through the native host
        eviction path (only available with L3 write-through)."""
        if self.mode != "one_pool":
            return False
        for _ in range(4):
            missing = need_pages - self.kda.alloc.largest_free_run()
            if missing <= 0:
                break
            if self._reclaim_from_mla(missing * self.page_bytes) <= 0:
                break
            if self.kda.admit(key):
                return True
        return False

    def admit_kda(self, key: str) -> bool:
        if self.kda.admit(key):
            return True
        need_pages = self.kda.alloc.class_pages(self.kda.slot_pages)
        if self.borrow_policy == "self_first":
            if self._kda_replace_self(key, need_pages):
                return True
            if self._kda_borrow_other(key, need_pages):
                return True
        else:
            if self._kda_borrow_other(key, need_pages):
                return True
            if self._kda_replace_self(key, need_pages):
                return True
        self.alloc_fail_kda += 1
        self.admission_reject_kda += 1
        return False

    def release_kda(self, key: str) -> bool:
        return self.kda.release(key)

    def kda_available_slots(self) -> int:
        return self.kda.available_slots()

    # -- cross-view reclaim -------------------------------------------------
    def _reclaim_from_kda(self, need_bytes: int) -> int:
        """Evict KDA checkpoints (existing L2->L3 path) to free ``need_bytes``."""
        if not self._kda_pools:
            return 0
        self.cross_reclaim_calls += 1
        freed = 0
        evictions = 0
        with self._lock:
            # Bound the loop by the number of evictions, not by a byte estimate,
            # so large contiguous requests still get served in one pass.
            while freed < need_bytes and evictions < 512:
                progress = False
                for pool in list(self._kda_pools):
                    pages_before = self.kda.used_pages
                    try:
                        victim = pool.evict()
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug("KDA reclaim evict failed: %s", exc)
                        victim = None
                    if victim is None:
                        continue
                    # Idempotent: the owning pool may already have released it.
                    self.kda.release(victim)
                    delta_pages = pages_before - self.kda.used_pages
                    if delta_pages <= 0:
                        continue
                    progress = True
                    evictions += 1
                    freed += delta_pages * self.page_bytes
                    self.cross_evict_kda_slots += 1
                    self.cross_evict_kda_bytes += delta_pages * self.page_bytes
                    self._notify_kda_evicted(pool, victim)
                    if freed >= need_bytes:
                        break
                if not progress:
                    break
        return freed

    @staticmethod
    def _notify_kda_evicted(pool: Any, victim: str) -> None:
        """Keep the existing L3 promotion (write-through) in the loop."""
        notify = getattr(pool, "notify_evicted", None)
        try:
            if callable(notify):
                notify(victim)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("KDA eviction notification failed: %s", exc)

    def _reclaim_from_mla(self, need_bytes: int) -> int:
        """Evict MLA entries through the native host-eviction path."""
        if not self.mla_reclaim_enabled:
            return 0
        self.cross_reclaim_calls += 1
        need_tokens = max(1, math.ceil(need_bytes / max(1, self.token_bytes)))
        try:
            freed_tokens = int(self._mla_evict_fn(need_tokens) or 0)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("MLA reclaim via evict_host failed: %s", exc)
            return 0
        if freed_tokens > 0:
            self.cross_evict_mla_tokens += freed_tokens
        return freed_tokens * self.token_bytes

    # -- metrics ------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        alloc = self.shared_alloc if self.mode == "one_pool" else None
        mla_s = self.mla.stats()
        kda_s = self.kda.stats()
        if alloc is not None:
            a = alloc.stats()
            free_bytes = a["free_pages"] * alloc.page_bytes
            largest = a["largest_free_run_pages"] * alloc.page_bytes
            internal = a["internal_frag_bytes"]
            external = a["external_frag_bytes"]
            used_pages = a["used_pages"]
            total_pages = a["total_pages"]
            run_count = a["free_run_count"]
            policy = a["policy"]
        else:
            a_mla = self.mla_alloc.stats()
            a_kda = self.kda_alloc.stats()
            free_bytes = (a_mla["free_pages"] + a_kda["free_pages"]) * self.page_bytes
            largest = max(a_mla["largest_free_run_pages"], a_kda["largest_free_run_pages"]) * self.page_bytes
            internal = a_mla["internal_frag_bytes"] + a_kda["internal_frag_bytes"]
            external = a_mla["external_frag_bytes"] + a_kda["external_frag_bytes"]
            used_pages = a_mla["used_pages"] + a_kda["used_pages"]
            total_pages = self.budget_pages
            run_count = a_mla["free_run_count"] + a_kda["free_run_count"]
            policy = self.policy

        used_bytes = used_pages * self.page_bytes
        mla_bytes = mla_s["used_pages"] * self.page_bytes
        kda_bytes = kda_s["used_pages"] * self.page_bytes
        return {
            "mode": self.mode,
            "total_bytes": self.total_bytes,
            "page_bytes": self.page_bytes,
            "page_tokens": self.page_tokens,
            "alloc_policy": policy,
            "total_pages": total_pages,
            "used_pages": used_pages,
            "used_bytes": used_bytes,
            "kv_bytes": mla_bytes,
            "kda_bytes": kda_bytes,
            "free_bytes": free_bytes,
            "largest_free_run_bytes": largest,
            "free_run_count": run_count,
            "internal_frag_bytes": internal,
            "external_frag_bytes": external,
            "internal_frag_ratio": (internal / used_bytes) if used_bytes else 0.0,
            "external_frag_ratio": (external / free_bytes) if free_bytes else 0.0,
            "mla_extents": mla_s["extents"],
            "mla_live_tokens": mla_s["live_tokens"],
            "mla_available_tokens": mla_s["available_tokens"],
            "kda_cached_slots": kda_s["cached_slots"],
            "kda_logical_bytes": kda_s["logical_bytes"],
            "kda_available_slots": kda_s["available_slots"],
            "kda_slot_pages": kda_s["slot_pages"],
            "kda_floor_bytes": self.kda_floor_bytes,
            "mla_capacity_bytes": self.mla_capacity_bytes,
            "kda_capacity_bytes": self.kda_capacity_bytes,
            "alloc_fail_mla": self.alloc_fail_mla,
            "alloc_fail_kda": self.alloc_fail_kda,
            "admission_reject_kda": self.admission_reject_kda,
            "cross_reclaim_calls": self.cross_reclaim_calls,
            "cross_evict_kda_slots": self.cross_evict_kda_slots,
            "cross_evict_kda_bytes": self.cross_evict_kda_bytes,
            "cross_evict_mla_tokens": self.cross_evict_mla_tokens,
            "kv_evict_enabled": bool(self.kv_evict and self._mla_evict_fn is not None),
            "borrow_policy": self.borrow_policy,
        }


# ---------------------------------------------------------------------------
# process singleton
# ---------------------------------------------------------------------------
_POOL: Optional[UnifiedDRAMPool] = None
_POOL_LOCK = threading.Lock()


def get_l2_dram_pool() -> Optional[UnifiedDRAMPool]:
    return _POOL


def reset_l2_dram_pool() -> None:
    global _POOL
    with _POOL_LOCK:
        _POOL = None


def configure_l2_dram_pool(
    hicache_gb: float,
    token_bytes: int,
    kda_slot_bytes: int = DEFAULT_KDA_SLOT_BYTES,
) -> Optional[UnifiedDRAMPool]:
    """Create the process-wide pool from env.  Returns ``None`` when disabled.

    Enabled when ``HISIM_L2_ONE_POOL=1`` or ``HISIM_L2_TOTAL_GB`` is set, so
    Two Pools and One Pool can be run against the same byte budget.
    """
    global _POOL
    one_pool = _env_bool("HISIM_L2_ONE_POOL", False)
    total_gb_env = os.environ.get("HISIM_L2_TOTAL_GB")
    explicit_total = total_gb_env is not None and str(total_gb_env).strip() != ""
    if not one_pool and not explicit_total:
        return None

    total_gb = _env_float("HISIM_L2_TOTAL_GB", float(hicache_gb or 0.0))
    if total_gb <= 0:
        logger.warning("L2 unified pool requested but budget is 0; disabled")
        return None

    page_tokens = max(1, _env_int("HISIM_L2_PAGE_TOKENS", DEFAULT_PAGE_TOKENS))
    policy = os.environ.get("HISIM_L2_ALLOC_POLICY", "exact").strip().lower()
    if policy not in VALID_POLICIES:
        logger.warning("Unknown HISIM_L2_ALLOC_POLICY=%r, using exact", policy)
        policy = "exact"

    mode = "one_pool" if one_pool else "two_pools"

    mamba_slots = max(1, _env_int("HISIM_L2_MAMBA_SIZE", 330))
    split_kda_gb = _env_float("HISIM_L2_SPLIT_KDA_GB", 0.0)
    if split_kda_gb > 0:
        kda_region_bytes = int(split_kda_gb * 1e9)
    else:
        kda_region_bytes = mamba_slots * int(kda_slot_bytes)

    floor_slots = max(0, _env_int("HISIM_L2_MAMBA_MIN_SLOTS", 0))
    kv_evict = _env_bool("HISIM_L2_KV_EVICT", True)
    borrow_policy = os.environ.get("HISIM_L2_BORROW_POLICY", "self_first").strip().lower()
    if borrow_policy not in ("other_first", "self_first"):
        logger.warning("Unknown HISIM_L2_BORROW_POLICY=%r, using self_first", borrow_policy)
        borrow_policy = "self_first"

    with _POOL_LOCK:
        if _POOL is not None:
            logger.warning("L2 unified DRAM pool already configured; reusing")
            return _POOL
        page_bytes = page_tokens * int(token_bytes)
        kda_bytes = min(int(kda_region_bytes), int(total_gb * 1e9))
        kda_bytes = (kda_bytes // page_bytes) * page_bytes if mode != "one_pool" else kda_bytes
        _POOL = UnifiedDRAMPool(
            total_bytes=int(total_gb * 1e9),
            mode=mode,
            token_bytes=int(token_bytes),
            kda_slot_bytes=int(kda_slot_bytes),
            page_tokens=page_tokens,
            policy=policy,
            kda_region_bytes=kda_bytes,
            kda_floor_slots=floor_slots,
            kv_evict=kv_evict,
            borrow_policy=borrow_policy,
        )
        return _POOL
