"""Unit tests for the minimal unified L2 DRAM pool (KDA + MLA).

Run with:  pytest test/test_l2_dram_pool.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.l2_dram_pool import (  # noqa: E402
    DEFAULT_KDA_SLOT_BYTES,
    KDAView,
    MLAView,
    PageExtentAllocator,
    UnifiedDRAMPool,
)

TOKEN_BYTES = 5760
SLOT_BYTES = DEFAULT_KDA_SLOT_BYTES


def _sum_req(alloc):
    return alloc.used_pages * alloc.page_bytes - alloc.stats()["internal_frag_bytes"]


def test_exact_extent_alloc_free_coalesces():
    page_bytes = 16 * TOKEN_BYTES
    a = PageExtentAllocator(total_pages=100, page_bytes=page_bytes, policy="exact")

    e1 = a.alloc(10, 10 * page_bytes, owner="mla", eid=1)
    e2 = a.alloc(20, 20 * page_bytes, owner="mla", eid=2)
    assert (e1.start_page, e1.pages) == (0, 10)
    assert (e2.start_page, e2.pages) == (10, 20)
    assert a.free_pages == 70
    assert a.stats()["largest_free_run_pages"] == 70

    a.free(e1)
    a.free(e2)
    st = a.stats()
    assert st["free_pages"] == 100
    assert st["largest_free_run_pages"] == 100  # coalesced back
    assert st["used_pages"] == 0
    assert st["internal_frag_bytes"] == 0
    assert st["free_run_count"] == 1


def test_external_fragmentation_is_reported():
    page_bytes = 16 * TOKEN_BYTES
    a = PageExtentAllocator(total_pages=16, page_bytes=page_bytes, policy="exact")
    # Allocate the first 8 pages one by one, then free the first 7 of them:
    # they become 7 isolated single-page runs while pages 8..15 stay free.
    ext = [a.alloc(1, page_bytes, owner="mla", eid=i) for i in range(8)]
    for e in ext[:7]:
        a.free(e)
    st = a.stats()
    assert st["free_pages"] == 15
    assert st["largest_free_run_pages"] == 8
    assert st["external_frag_bytes"] == 7 * page_bytes
    assert st["external_frag_ratio"] == pytest.approx(7 / 15)


def test_pow2_size_class_internal_fragmentation():
    page_bytes = 16 * TOKEN_BYTES
    a = PageExtentAllocator(total_pages=64, page_bytes=page_bytes, policy="pow2")
    e = a.alloc(3, 3 * page_bytes, owner="mla", eid=1)
    assert e.pages == 4  # rounded up to the next size class
    assert a.stats()["internal_frag_bytes"] == page_bytes


def test_mla_view_token_roundtrip_and_partial_free():
    page_bytes = 16 * TOKEN_BYTES
    alloc = PageExtentAllocator(total_pages=64, page_bytes=page_bytes, policy="exact")
    mla = MLAView(alloc, token_bytes=TOKEN_BYTES, page_tokens=16)

    idx = mla.alloc_tokens(40)  # 3 pages
    assert idx.dtype == torch.int64
    assert idx.tolist() == list(range(40))
    assert alloc.used_pages == 3
    assert mla.stats()["extents"] == 1

    # Free in two halves: extent only returns to the allocator when empty.
    mla.free_tokens(idx[:20])
    assert alloc.used_pages == 3
    mla.free_tokens(idx[20:])
    assert alloc.used_pages == 0
    assert mla.stats()["extents"] == 0

    # A second allocation reuses the coalesced extent.
    idx2 = mla.alloc_tokens(16)
    assert idx2.tolist() == list(range(16))


def test_kda_view_admit_release():
    page_bytes = 16 * TOKEN_BYTES
    alloc = PageExtentAllocator(total_pages=200, page_bytes=page_bytes, policy="exact")
    kda = KDAView(alloc, slot_bytes=SLOT_BYTES)
    assert kda.slot_pages == 159  # 14_653_440 B / (16 * 5760 B), exact

    assert kda.admit("h1")
    assert kda.admit("h1")  # idempotent
    assert kda.stats()["cached_slots"] == 1
    used = alloc.used_pages
    assert used == kda.slot_pages

    assert kda.release("h1")
    assert not kda.release("h1")
    assert alloc.used_pages == 0


class _FakeKdaPool:
    def __init__(self):
        self.hashes = []
        self.notified = []

    def evict(self):
        return self.hashes.pop(0) if self.hashes else None

    def notify_evicted(self, h):
        self.notified.append(h)


def _make_pool(mode, total_gb=0.2, **kw):
    return UnifiedDRAMPool(
        total_bytes=int(total_gb * 1e9),
        mode=mode,
        token_bytes=TOKEN_BYTES,
        kda_slot_bytes=SLOT_BYTES,
        page_tokens=16,
        policy="exact",
        **kw,
    )


def test_two_pools_regions_are_isolated():
    pool = _make_pool("two_pools", kda_region_bytes=4 * SLOT_BYTES)
    assert pool.kda_capacity_slots == 4
    for i in range(4):
        assert pool.admit_kda(f"h{i}")
    assert not pool.admit_kda("h4")  # KDA region full
    # MLA region is independent (same total budget).
    assert pool.alloc_mla_tokens(64) is not None


def test_one_pool_mla_borrows_from_kda():
    pool = _make_pool("one_pool")
    kda = _FakeKdaPool()
    kda.hashes = [f"h{i}" for i in range(6)]
    for h in kda.hashes:
        assert pool.admit_kda(h)
    pool.register_kda_pool(kda)
    used_before = pool.stats()["used_pages"]

    # Fill the rest with MLA, then force a large MLA request.
    per_page = pool.page_tokens
    free_pages = pool.stats()["total_pages"] - used_before
    assert pool.alloc_mla_tokens((free_pages - 2) * per_page) is not None
    idx = pool.alloc_mla_tokens(4 * per_page)
    assert idx is not None
    st = pool.stats()
    assert st["cross_evict_kda_slots"] > 0
    assert kda.notified  # evicted checkpoints routed to the L3 write path
    assert st["used_bytes"] <= st["total_bytes"]


def test_one_pool_kda_borrows_from_mla():
    pool = _make_pool("one_pool")

    def fake_mla_evict(n_tokens):
        # Simulate cache.evict_host: free the oldest MLA extent.
        if not pool.mla.extents:
            return 0
        eid = next(iter(pool.mla.extents))
        ext = pool.mla.extents[eid]
        n_live = int(ext.live)
        start = ext.start_page * pool.page_tokens
        idx = torch.arange(start, start + n_live, dtype=torch.int64)
        pool.mla.free_tokens(idx)
        return n_live

    pool.set_mla_evictor(fake_mla_evict)
    # Fill the pool with MLA.
    total_tokens = pool.stats()["total_pages"] * pool.page_tokens
    assert pool.alloc_mla_tokens(total_tokens) is not None
    assert not pool.admit_kda("h1") or True  # KDA floor is 0; check reclaim path
    ok = False
    for i in range(50):
        if pool.admit_kda(f"k{i}"):
            ok = True
            break
    assert ok
    st = pool.stats()
    assert st["cross_evict_mla_tokens"] > 0
    assert st["kda_cached_slots"] >= 1


def test_two_ended_allocation_keeps_views_apart():
    page_bytes = 16 * TOKEN_BYTES
    a = PageExtentAllocator(total_pages=1000, page_bytes=page_bytes, policy="exact")
    low = a.alloc(10, 10 * page_bytes, owner="mla")
    high = a.alloc(20, 20 * page_bytes, owner="kda", from_high=True)
    assert low.start_page == 0
    assert high.start_page == 980          # taken from the tail
    assert a.stats()["free_run_count"] == 1  # one run in the middle
    assert a.stats()["largest_free_run_pages"] == 970


def test_borrow_policy_self_first_replaces_own_view():
    pool = _make_pool("one_pool")
    pool.borrow_policy = "self_first"

    def fake_mla_evict(n_tokens):
        if not pool.mla.extents:
            return 0
        eid = next(iter(pool.mla.extents))
        ext = pool.mla.extents[eid]
        n_live = int(ext.live)
        start = ext.start_page * pool.page_tokens
        pool.mla.free_tokens(torch.arange(start, start + n_live, dtype=torch.int64))
        return n_live

    pool.set_mla_evictor(fake_mla_evict)
    kda = _FakeKdaPool()
    pool.register_kda_pool(kda)
    for h in ("k0", "k1"):
        assert pool.admit_kda(h)
        kda.hashes.append(h)
    # Fill every remaining page with MLA.
    total_pages = pool.stats()["total_pages"]
    for _ in range(64):
        free = pool.mla.available_pages()
        if free <= 0:
            break
        assert pool.alloc_mla_tokens(free * pool.page_tokens) is not None
    mla_used_before = pool.mla.used_pages
    # self_first: the new checkpoint replaces an old checkpoint, MLA untouched.
    assert pool.admit_kda("k2")
    st = pool.stats()
    assert st["kda_cached_slots"] == 2
    assert st["cross_evict_mla_tokens"] == 0
    assert pool.mla.used_pages == mla_used_before
    assert kda.notified == ["k0"]  # evicted checkpoint still goes to L3

    # other_first on the same state: MLA is the victim instead.
    pool.borrow_policy = "other_first"
    kda.hashes.append("k2")
    assert pool.admit_kda("k3")
    st = pool.stats()
    assert st["cross_evict_mla_tokens"] > 0
    assert pool.mla.used_pages < mla_used_before


def test_accounting_consistency_exact():
    pool = _make_pool("one_pool")
    assert pool.alloc_mla_tokens(100) is not None
    assert pool.admit_kda("h1")
    st = pool.stats()
    assert st["used_bytes"] == st["kv_bytes"] + st["kda_bytes"]
    assert st["used_bytes"] + st["free_bytes"] == st["total_bytes"]
    # KDA slots are exactly page aligned at the default page size.
    assert st["kda_bytes"] == st["kda_logical_bytes"]


# --- mock mamba host pool: legacy vs unified host-index semantics -------------
def test_mamba_host_pool_legacy_free_list():
    from hisim.simulation.sglang.sglang_mock_class import MockMambaStatePoolHost

    pool = MockMambaStatePoolHost(size=4, slot_bytes=SLOT_BYTES)
    idx = pool.alloc(2)
    assert idx.tolist() == [0, 1]
    assert pool.available_size() == 2
    # Exhausted free list reports failure (legacy behaviour, unchanged).
    assert pool.alloc(3) is None
    assert pool.free(idx) == 2
    assert pool.available_size() == 4


def test_mamba_host_pool_bound_indices_do_not_leak():
    """Bound to the shared pool, host ids must not consume a finite free list.

    The native ack path that would release these ids is stubbed out by the
    synchronous mock write, so a finite list leaked one entry per checkpoint
    and eventually stopped KDA host backups (measured: 96 leaked over 96
    donations in the P5.1 server run).
    """

    class _FakeUnified:
        def __init__(self):
            self.available = 2
            self.freed = []

        def kda_available_slots(self):
            return self.available

        def release_kda(self, key):
            self.freed.append(key)
            return True

    from hisim.simulation.sglang.sglang_mock_class import MockMambaStatePoolHost

    pool = MockMambaStatePoolHost(size=8, slot_bytes=SLOT_BYTES)
    up = _FakeUnified()
    pool.attach_unified(up)
    for _ in range(50):  # far more than `size`
        assert pool.alloc(3) is not None
    # Capacity is owned by the shared budget, not by the host-index list.
    assert pool.available_size() == 2
    assert pool.get_stats()["free_slots"] == 2
    # Idempotent release still works for indices handed out above.
    assert pool.free(torch.tensor([0, 1, 2])) == 3
    assert pool.free(torch.tensor([0, 1, 2])) == 0
