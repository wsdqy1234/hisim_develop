#!/usr/bin/env python3
"""Multi-class (N>2) DRAM allocation study for the unified L2 pool.

Question
--------
The shipped allocator splits the one byte budget between TWO classes: MLA grows
up from the low address end, KDA/Mamba grows down from the high end.  If a
model needs THREE block shapes in the same budget (the sglang "tri-pool" shape:
full-attention KV + sliding-window KV + Mamba/SSM state), two "ends" are not
enough: only two regions can be anchored at the extremes, and the third class
has no natural home.

This script compares the structural options on the same open-loop trace:

    ends2_ff     two ends, third class simply joins the low end (naive
                 "add a view" extension)
    ends2_spill  two ends + third class spills anywhere (best-fit); no region
                 and no data movement
    chain3       3 ordered spans, boundaries move only into the neighbour's
                 FREE frontier page (no data movement)
    chain3_move  same, but boundary pages may be relocated into the donor's
                 own holes (page-level compaction, no cascade)
    chain3_cascade  same, and the donor itself is pushed outward first -- the
                 full sglang `make_room` behaviour (moves cascade the chain)
    static3      three fixed spans sized from the expected byte ratio
    global_bf    one space, best-fit for every class (MemCache today)
    ideal        zero external fragmentation (upper bound)

Only the physical allocator differs; the trace, lifetimes and byte shares are
identical, so `fail%` per class is the capacity the policy wastes.

Usage:
    python benchmark/collection/l2_pool_multiclass_study.py
    python benchmark/collection/l2_pool_multiclass_study.py --seeds 1,7,11 --swa small,mid
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import os
import random
import statistics
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

TOKEN_BYTES = 5760
SLOT_BYTES = 14_653_440
PAGE_TOKENS = 16

GEOMETRIES = {
    "real_5760B": dict(token_bytes=TOKEN_BYTES, page_tokens=PAGE_TOKENS,
                       slot_bytes=SLOT_BYTES, budget_gb=8.0),
    "server_1656KB": dict(token_bytes=1_695_744, page_tokens=PAGE_TOKENS,
                          slot_bytes=SLOT_BYTES, budget_gb=85.0),
}

MLA_SIZE_CHOICES = [16, 32, 64, 128, 256, 512]
MLA_SIZE_WEIGHTS = [15, 20, 25, 20, 12, 8]
MEAN_MLA_PAGES = sum(p * w for p, w in zip(MLA_SIZE_CHOICES, MLA_SIZE_WEIGHTS)) \
    / sum(MLA_SIZE_WEIGHTS)

# Third class ("swa": window-capped KV). Its object size decides whether it
# competes with MLA's small runs or behaves like a second "large" class.
SWA_SCENARIOS = {
    "small": ([4, 8, 16], [30, 40, 30]),
    "mid": ([32, 64, 128], [30, 40, 30]),
    "big": ([128, 256, 512], [30, 40, 30]),
}

T_MLA = 300
T_STATE = 3000
T_SWA = 800
RHO = 0.95
SHARES = dict(mla=0.60, swa=0.15, state=0.25)   # offered bytes


class Obj:
    __slots__ = ("cls", "pages", "start")

    def __init__(self, cls: str, pages: int, start: int):
        self.cls, self.pages, self.start = cls, pages, start


# ---------------------------------------------------------------------------
# generic run-list helpers
# ---------------------------------------------------------------------------
def run_insert(starts: List[int], sizes: List[int], start: int, pages: int) -> None:
    pos = bisect_left(starts, start)
    if pos > 0 and starts[pos - 1] + sizes[pos - 1] == start:
        sizes[pos - 1] += pages
        m = pos - 1
    else:
        starts.insert(pos, start)
        sizes.insert(pos, pages)
        m = pos
    if m + 1 < len(starts) and starts[m] + sizes[m] == starts[m + 1]:
        sizes[m] += sizes[m + 1]
        del starts[m + 1]
        del sizes[m + 1]


def run_take(starts: List[int], sizes: List[int], i: int, pages: int, side: str) -> int:
    if side == "low":
        start = starts[i]
        starts[i] += pages
    else:
        start = starts[i] + sizes[i] - pages
    sizes[i] -= pages
    if sizes[i] == 0:
        del starts[i]
        del sizes[i]
    return start


# ---------------------------------------------------------------------------
# one global run list, class -> preferred end (two-ended + spill)
# ---------------------------------------------------------------------------
class GlobalAlloc:
    def __init__(self, total: int, ends: Dict[str, str], picks: Dict[str, str]):
        self.starts = [0]
        self.sizes = [total]
        self.total = total
        self.free = total
        self.used = 0
        self.ends = ends
        self.picks = picks
        self.attempts = {c: 0 for c in ends}
        self.fails = {c: 0 for c in ends}
        self.frag = {c: 0 for c in ends}
        self.moves = 0

    def largest(self) -> int:
        return max(self.sizes, default=0)

    def fit(self, pages: int, cls: str) -> Optional[int]:
        end = self.ends[cls]
        pick = self.picks.get(cls, "best")
        sizes = self.sizes
        if pick == "best":
            best, key = -1, None
            for i, n in enumerate(sizes):
                if n < pages:
                    continue
                k = (n, self.starts[i])
                if key is None or k < key:
                    key, best = k, i
            return best if best >= 0 else None
        if end == "high":
            for i in range(len(sizes) - 1, -1, -1):
                if sizes[i] >= pages:
                    return i
            return None
        for i, n in enumerate(sizes):
            if n >= pages:
                return i
        return None

    def _shrink(self, j: int, side: str, want: int, move: bool) -> Tuple[int, int]:
        """Take `want` pages off region `j` at `side`; return (taken, moved).

        Free frontier pages go first (no cost).  With `move`, the whole extent
        sitting on the boundary is relocated into a hole inside the same
        region -- that is the only relocation a cache pool can do without
        breaking extent contiguity.
        """
        r = self.r[j]
        taken = r.trim(side, want)          # bulk: the free frontier run
        moved = 0
        if not move:
            return taken, moved
        while taken < want:
            p = r.lo if side == "lo" else r.hi - 1
            o = self.live[j].get(p)
            if o is None:                   # nothing owns it and it is not free
                break
            dest = r.alloc(o.pages)         # a hole inside the same region
            if dest is None:
                break
            old = o.start
            for q in range(old, old + o.pages):
                del self.live[j][q]
            o.start = dest
            for q in range(dest, dest + o.pages):
                self.live[j][q] = o
            moved += o.pages
            run_insert(r.starts, r.sizes, old, o.pages)   # vacated boundary range
            r.free += o.pages
            taken += r.trim(side, want - taken)
        return taken, moved

    def alloc(self, pages: int, cls: str) -> Optional[Obj]:
        self.attempts[cls] += 1
        i = self.fit(pages, cls)
        if i is None:
            self.fails[cls] += 1
            if self.free >= pages:
                self.frag[cls] += 1
            return None
        end = "high" if (self.picks.get(cls) != "best"
                         and self.ends[cls] == "high") else "low"
        self.free -= pages
        self.used += pages
        return Obj(cls, pages, run_take(self.starts, self.sizes, i, pages, end))

    def free_obj(self, o: Obj) -> None:
        run_insert(self.starts, self.sizes, o.start, o.pages)
        self.free += o.pages
        self.used -= o.pages

    def clone(self):
        c = GlobalAlloc.__new__(GlobalAlloc)
        c.__dict__.update(self.__dict__)
        c.starts = list(self.starts)
        c.sizes = list(self.sizes)
        return c


# ---------------------------------------------------------------------------
# chained spans: [region0(up) | region1(both) | region2(down)]
# ---------------------------------------------------------------------------
class Region:
    __slots__ = ("lo", "hi", "grow", "starts", "sizes", "free")

    def __init__(self, lo: int, hi: int, grow: str):
        self.lo, self.hi, self.grow = lo, hi, grow
        self.starts: List[int] = []
        self.sizes: List[int] = []
        self.free = 0
        if hi > lo:
            self.starts = [lo]
            self.sizes = [hi - lo]
            self.free = hi - lo

    def largest(self) -> int:
        return max(self.sizes, default=0)

    def fit(self, pages: int) -> Optional[int]:
        best, key = -1, None
        for i, n in enumerate(self.sizes):
            if n < pages:
                continue
            k = (n, self.starts[i])
            if key is None or k < key:
                key, best = k, i
        return best if best >= 0 else None

    def alloc(self, pages: int) -> Optional[int]:
        if self.grow == "up":
            for i, n in enumerate(self.sizes):
                if n >= pages:
                    self.free -= pages
                    return run_take(self.starts, self.sizes, i, pages, "low")
            return None
        if self.grow == "down":
            for i in range(len(self.sizes) - 1, -1, -1):
                if self.sizes[i] >= pages:
                    self.free -= pages
                    return run_take(self.starts, self.sizes, i, pages, "high")
            return None
        # middle span: reuse interior holes first (keeps slack on BOTH movable
        # boundaries), then eat the larger of the two edge gaps
        best, key = -1, None
        low_edge = high_edge = -1
        for i, n in enumerate(self.sizes):
            s = self.starts[i]
            if s == self.lo:
                low_edge = i
                continue
            if s + n == self.hi:
                high_edge = i
                continue
            if n < pages:
                continue
            k = (n, s)
            if key is None or k < key:
                key, best = k, i
        if best >= 0:
            self.free -= pages
            return run_take(self.starts, self.sizes, best, pages, "low")
        cands = []
        if low_edge >= 0 and self.sizes[low_edge] >= pages:
            cands.append((self.sizes[low_edge], low_edge, "low"))
        if high_edge >= 0 and self.sizes[high_edge] >= pages:
            cands.append((self.sizes[high_edge], high_edge, "high"))
        if not cands:
            return None
        _, i, side = max(cands)
        self.free -= pages
        return run_take(self.starts, self.sizes, i, pages, side)

    def free_obj(self, start: int, pages: int) -> None:
        run_insert(self.starts, self.sizes, start, pages)
        self.free += pages

    def boundary_free(self, side: str) -> int:
        """Length of the free run touching `side` ('hi' or 'lo')."""
        if side == "hi":
            for i, n in enumerate(self.sizes):
                if self.starts[i] + n == self.hi:
                    return n
            return 0
        for i, n in enumerate(self.sizes):
            if self.starts[i] == self.lo:
                return n
        return 0

    def is_free(self, p: int) -> bool:
        i = bisect_left(self.starts, p + 1) - 1
        return i >= 0 and self.starts[i] <= p < self.starts[i] + self.sizes[i]

    def trim(self, side: str, k: int) -> int:
        """Shrink the free run touching `side` by `k` pages (bulk)."""
        k = min(k, self.boundary_free(side))
        if k <= 0:
            return 0
        if side == "hi":
            self.sizes[-1] -= k
            if self.sizes[-1] == 0:
                del self.starts[-1]
                del self.sizes[-1]
            self.hi -= k
        else:
            self.starts[0] += k
            self.sizes[0] -= k
            if self.sizes[0] == 0:
                del self.starts[0]
                del self.sizes[0]
            self.lo += k
        self.free -= k
        return k

    def take(self, side: str, k: int) -> None:
        if k <= 0:
            return
        if side == "hi":
            old = self.hi
            self.hi += k
            run_insert(self.starts, self.sizes, old, k)
        else:
            self.lo -= k
            run_insert(self.starts, self.sizes, self.lo, k)
        self.free += k


class ChainAlloc:
    def __init__(self, total: int, move: bool, cascade: bool = False,
                 classes=("mla", "swa", "state")):
        mid = total // 2
        self.r = [Region(0, mid, "up"), Region(mid, mid, "both"),
                  Region(mid, total, "down")]
        self.index = {c: i for i, c in enumerate(classes)}
        self.move = move
        self.cascade = cascade
        self.total = total
        self.attempts = {c: 0 for c in classes}
        self.fails = {c: 0 for c in classes}
        self.frag = {c: 0 for c in classes}
        self.moves = 0
        self.used = 0
        # page -> live Obj, per region: relocation must move WHOLE extents
        # (an extent is contiguous by contract), so the allocator needs to
        # know which object owns the boundary page.
        self.live: List[Dict[int, Obj]] = [dict(), dict(), dict()]

    def free_total(self) -> int:
        return sum(r.free for r in self.r)

    @property
    def free(self) -> int:
        return self.free_total()

    def _grow(self, i: int, side: str, want: int, depth: int = 0) -> int:
        """Take `want` free pages for region `i` across its `side` boundary.

        Empty spans are transparent (they own no pages yet).  With `move` the
        donor may also be pushed outward first, which is what a real float
        allocator does: relocate the donor's boundary pages, cascading through
        the chain, before handing the space over.
        """
        step = 1 if side == "hi" else -1
        j = i + step
        while 0 <= j < 3 and self.r[j].hi == self.r[j].lo:
            j += step
        if not (0 <= j < 3):
            return 0
        other = "lo" if side == "hi" else "hi"
        d = self.r[j]
        r = self.r[i]
        if self.move and self.cascade and depth < 2:
            avail = d.free
            if avail < want:
                # push the donor's far edge outward to open room inside it
                self._grow(j, "hi" if other == "lo" else "lo", want - avail,
                           depth + 1)
        k, moved = self._shrink(j, other, want, self.move)
        if not k:
            return 0
        self.moves += moved
        r.take(side, k)
        # re-seat the empty spans we skipped over onto the new boundary
        for m in range(i + step, j, step):
            self.r[m].lo = self.r[m].hi = r.hi if side == "hi" else r.lo
        return k

    def _shrink(self, j: int, side: str, want: int, move: bool) -> Tuple[int, int]:
        """Take `want` pages off region `j` at `side`; return (taken, moved).

        Free frontier pages go first (no cost).  With `move`, the whole extent
        sitting on the boundary is relocated into a hole inside the same
        region -- that is the only relocation a cache pool can do without
        breaking extent contiguity.
        """
        r = self.r[j]
        taken = r.trim(side, want)          # bulk: the free frontier run
        moved = 0
        if not move:
            return taken, moved
        while taken < want:
            p = r.lo if side == "lo" else r.hi - 1
            o = self.live[j].get(p)
            if o is None:                   # nothing owns it and it is not free
                break
            dest = r.alloc(o.pages)         # a hole inside the same region
            if dest is None:
                break
            old = o.start
            for q in range(old, old + o.pages):
                del self.live[j][q]
            o.start = dest
            for q in range(dest, dest + o.pages):
                self.live[j][q] = o
            moved += o.pages
            run_insert(r.starts, r.sizes, old, o.pages)   # vacated boundary range
            r.free += o.pages
            taken += r.trim(side, want - taken)
        return taken, moved

    def alloc(self, pages: int, cls: str) -> Optional[Obj]:
        self.attempts[cls] += 1
        i = self.index[cls]
        r = self.r[i]
        start = r.alloc(pages)
        if start is None:
            sides = {"up": ("hi",), "down": ("lo",),
                     "both": ("hi", "lo")}[r.grow]
            cands = []
            for side in sides:
                step = 1 if side == "hi" else -1
                j = i + step
                while 0 <= j < 3 and self.r[j].hi == self.r[j].lo:
                    j += step
                if not (0 <= j < 3):
                    continue
                d = self.r[j]
                other = "lo" if side == "hi" else "hi"
                cands.append((d.free if self.move else d.boundary_free(other),
                              side))
            cands.sort(reverse=True)
            for a, side in cands:
                if a <= 0:
                    continue
                need = pages - r.boundary_free(side)
                if need <= 0:
                    break
                if self._grow(i, side, need):
                    start = r.alloc(pages)
                    if start is not None:
                        break
        if start is None:
            self.fails[cls] += 1
            if self.free_total() >= pages:
                self.frag[cls] += 1
            return None
        self.used += pages
        o = Obj(cls, pages, start)
        lv = self.live[i]
        for q in range(start, start + pages):
            lv[q] = o
        return o

    def free_obj(self, o: Obj) -> None:
        i = self.index[o.cls]
        lv = self.live[i]
        for q in range(o.start, o.start + o.pages):
            del lv[q]
        self.r[i].free_obj(o.start, o.pages)
        self.used -= o.pages

    def clone(self):
        c = ChainAlloc.__new__(ChainAlloc)
        c.__dict__.update(self.__dict__)
        c.r = []
        for r in self.r:
            n = Region(r.lo, r.hi, r.grow)
            n.starts = list(r.starts)
            n.sizes = list(r.sizes)
            n.free = r.free
            c.r.append(n)
        c.live = [dict(x) for x in self.live]
        return c


class StaticAlloc(GlobalAlloc):
    """Fixed 3-way partition; falls back to nothing (a hard segment)."""

    def __init__(self, total: int, fractions: Dict[str, float]):
        super().__init__(total, {c: "low" for c in fractions}, {c: "best" for c in fractions})
        self.starts, self.sizes = [], []
        self.spans = {}
        acc = 0
        for c, f in fractions.items():
            k = int(total * f)
            self.spans[c] = (acc, acc + k)
            acc += k
        self.spans[list(fractions)[-1]] = (self.spans[list(fractions)[-1]][0], total)
        for c, (lo, hi) in self.spans.items():
            run_insert(self.starts, self.sizes, lo, hi - lo)

    def fit(self, pages: int, cls: str) -> Optional[int]:
        lo, hi = self.spans[cls]
        best, key = -1, None
        for i, n in enumerate(self.sizes):
            s = self.starts[i]
            a, b = max(s, lo), min(s + n, hi)
            if b - a < pages:
                continue
            k = (n, s)
            if key is None or k < key:
                key, best = k, i
        return best if best >= 0 else None

    def _shrink(self, j: int, side: str, want: int, move: bool) -> Tuple[int, int]:
        """Take `want` pages off region `j` at `side`; return (taken, moved).

        Free frontier pages go first (no cost).  With `move`, the whole extent
        sitting on the boundary is relocated into a hole inside the same
        region -- that is the only relocation a cache pool can do without
        breaking extent contiguity.
        """
        r = self.r[j]
        taken = r.trim(side, want)          # bulk: the free frontier run
        moved = 0
        if not move:
            return taken, moved
        while taken < want:
            p = r.lo if side == "lo" else r.hi - 1
            o = self.live[j].get(p)
            if o is None:                   # nothing owns it and it is not free
                break
            dest = r.alloc(o.pages)         # a hole inside the same region
            if dest is None:
                break
            old = o.start
            for q in range(old, old + o.pages):
                del self.live[j][q]
            o.start = dest
            for q in range(dest, dest + o.pages):
                self.live[j][q] = o
            moved += o.pages
            run_insert(r.starts, r.sizes, old, o.pages)   # vacated boundary range
            r.free += o.pages
            taken += r.trim(side, want - taken)
        return taken, moved

    def alloc(self, pages: int, cls: str) -> Optional[Obj]:
        self.attempts[cls] += 1
        i = self.fit(pages, cls)
        if i is None:
            self.fails[cls] += 1
            if self.free >= pages:
                self.frag[cls] += 1
            return None
        lo, hi = self.spans[cls]
        s, n = self.starts[i], self.sizes[i]
        start = max(s, lo)
        del self.starts[i]
        del self.sizes[i]
        if start > s:                      # keep the part left of the span
            run_insert(self.starts, self.sizes, s, start - s)
        if start + pages < s + n:          # keep the part right of the request
            run_insert(self.starts, self.sizes, start + pages, s + n - start - pages)
        self.free -= pages
        self.used += pages
        return Obj(cls, pages, start)


class IdealAlloc:
    def __init__(self, total: int, classes=("mla", "swa", "state")):
        self.free = self.total = total
        self.used = 0
        self.attempts = {c: 0 for c in classes}
        self.fails = {c: 0 for c in classes}
        self.frag = {c: 0 for c in classes}
        self.moves = 0

    def _shrink(self, j: int, side: str, want: int, move: bool) -> Tuple[int, int]:
        """Take `want` pages off region `j` at `side`; return (taken, moved).

        Free frontier pages go first (no cost).  With `move`, the whole extent
        sitting on the boundary is relocated into a hole inside the same
        region -- that is the only relocation a cache pool can do without
        breaking extent contiguity.
        """
        r = self.r[j]
        taken = r.trim(side, want)          # bulk: the free frontier run
        moved = 0
        if not move:
            return taken, moved
        while taken < want:
            p = r.lo if side == "lo" else r.hi - 1
            o = self.live[j].get(p)
            if o is None:                   # nothing owns it and it is not free
                break
            dest = r.alloc(o.pages)         # a hole inside the same region
            if dest is None:
                break
            old = o.start
            for q in range(old, old + o.pages):
                del self.live[j][q]
            o.start = dest
            for q in range(dest, dest + o.pages):
                self.live[j][q] = o
            moved += o.pages
            run_insert(r.starts, r.sizes, old, o.pages)   # vacated boundary range
            r.free += o.pages
            taken += r.trim(side, want - taken)
        return taken, moved

    def alloc(self, pages: int, cls: str) -> Optional[Obj]:
        self.attempts[cls] += 1
        if self.free < pages:
            self.fails[cls] += 1
            return None
        self.free -= pages
        self.used += pages
        return Obj(cls, pages, -1)

    def free_obj(self, o: Obj) -> None:
        self.free += o.pages
        self.used -= o.pages

    def clone(self):
        c = IdealAlloc.__new__(IdealAlloc)
        c.__dict__.update(self.__dict__)
        return c

    def runs(self) -> int:
        return 0


def make_policy(name: str, total: int):
    if name == "ends2_ff":
        # today's code path: every class takes the first run that fits, from
        # its preferred end (the third class is simply added to the low end)
        return GlobalAlloc(total, {"mla": "low", "swa": "low", "state": "high"},
                           {"mla": "first", "swa": "first", "state": "first"})
    if name == "ends2_spill":
        # two ends keep their fit-from-the-end rule; the third class is not
        # given an end at all and spills into the tightest hole anywhere
        return GlobalAlloc(total, {"mla": "low", "swa": "any", "state": "high"},
                           {"mla": "first", "swa": "best", "state": "first"})
    if name == "global_bf":
        return GlobalAlloc(total, {c: "low" for c in ("mla", "swa", "state")},
                           {c: "best" for c in ("mla", "swa", "state")})
    if name == "chain3":
        return ChainAlloc(total, move=False)
    if name == "chain3_move":
        return ChainAlloc(total, move=True)
    if name == "chain3_cascade":
        return ChainAlloc(total, move=True, cascade=True)
    if name == "static3":
        return StaticAlloc(total, {"mla": SHARES["mla"], "swa": SHARES["swa"],
                                   "state": SHARES["state"]})
    if name == "ideal":
        return IdealAlloc(total)
    raise ValueError(name)


POLICIES = ["ends2_ff", "ends2_spill", "global_bf", "chain3", "chain3_move",
            "chain3_cascade", "static3", "ideal"]


def run_case(geom: str, swa: str, policy: str, steps: int, seed: int) -> dict:
    g = GEOMETRIES[geom]
    page_bytes = g["token_bytes"] * g["page_tokens"]
    total_pages = int(g["budget_gb"] * 1e9) // page_bytes
    state_pages = max(1, -(-g["slot_bytes"] // page_bytes))
    swa_choices, swa_weights = SWA_SCENARIOS[swa]
    mean_swa = sum(p * w for p, w in zip(swa_choices, swa_weights)) / sum(swa_weights)

    rng = random.Random(seed)
    alloc = make_policy(policy, total_pages)

    lam = {
        "mla": RHO * SHARES["mla"] * total_pages / (MEAN_MLA_PAGES * T_MLA),
        "swa": RHO * SHARES["swa"] * total_pages / (mean_swa * T_SWA),
        "state": RHO * SHARES["state"] * total_pages / (state_pages * T_STATE),
    }
    nxt = {c: rng.expovariate(lam[c]) for c in lam}
    frees: List[Tuple[int, int, Obj]] = []
    seq = itertools.count()
    live = {c: 0 for c in lam}
    satt = {}
    sfail = {}
    samples = []
    warmup = min(steps // 2, 3 * T_STATE)

    t = 0.0
    step = 0
    while step < steps:
        step = int(t)
        while frees and frees[0][0] <= step:
            _, _, o = heapq.heappop(frees)
            alloc.free_obj(o)
            live[o.cls] -= o.pages

        for cls in ("mla", "swa", "state"):
            while step < steps and nxt[cls] < t + 1.0:
                if cls == "mla":
                    pages = rng.choices(MLA_SIZE_CHOICES, weights=MLA_SIZE_WEIGHTS, k=1)[0]
                    life = T_MLA
                elif cls == "swa":
                    pages = rng.choices(swa_choices, weights=swa_weights, k=1)[0]
                    life = T_SWA
                else:
                    pages, life = state_pages, T_STATE
                # draw the lifetime unconditionally: the random stream then
                # stays identical across policies, so every policy sees the
                # exact same arrival/size/lifetime trace
                life_draw = rng.expovariate(1.0 / life)
                o = alloc.alloc(pages, cls)
                key = (cls, pages)
                satt[key] = satt.get(key, 0) + 1
                if o is not None:
                    live[cls] += o.pages
                    heapq.heappush(frees, (step + int(life_draw) + 1,
                                           next(seq), o))
                else:
                    sfail[key] = sfail.get(key, 0) + 1
                nxt[cls] += rng.expovariate(lam[cls])

        if step >= warmup and step % 10 == 0:
            samples.append(dict(used=alloc.used, free=alloc.free,
                                runs=(len(alloc.r[0].sizes) + len(alloc.r[1].sizes)
                                      + len(alloc.r[2].sizes)) if isinstance(alloc, ChainAlloc)
                                else (0 if isinstance(alloc, IdealAlloc)
                                      else len(alloc.sizes))))
        t += 1.0

    n = max(len(samples), 1)
    row = dict(geometry=geom, swa=swa, policy=policy, seed=seed,
               total_pages=total_pages, state_pages=state_pages,
               used_pct=100.0 * sum(s["used"] for s in samples) / n / total_pages,
               avg_runs=sum(s["runs"] for s in samples) / n,
               moved_pages=alloc.moves,
               moved_per_alloc=alloc.moves / max(sum(alloc.attempts.values()), 1))
    for cls in ("mla", "swa", "state"):
        a = max(alloc.attempts[cls], 1)
        row[f"{cls}_attempts"] = alloc.attempts[cls]
        row[f"{cls}_fail_pct"] = 100.0 * alloc.fails[cls] / a
        row[f"{cls}_frag_pct"] = 100.0 * alloc.frag[cls] / a
    for cls, pages in (("mla", 512), ("mla", 16), ("swa", max(swa_choices))):
        k_att = max(satt.get((cls, pages), 0), 1)
        row[f"size_{cls}_{pages}_fail_pct"] = 100.0 * sfail.get((cls, pages), 0) / k_att
    row["state_resident"] = live["state"] // max(state_pages, 1)
    return row


def aggregate(rows: List[dict]) -> List[dict]:
    keys = ("mla_fail_pct", "swa_fail_pct", "state_fail_pct",
            "mla_frag_pct", "swa_frag_pct", "state_frag_pct",
            "size_mla_512_fail_pct", "size_mla_16_fail_pct",
            "used_pct", "avg_runs", "moved_pages")
    groups = {}
    for r in rows:
        groups.setdefault((r["geometry"], r["swa"], r["policy"]), []).append(r)
    out = []
    for (geom, swa, pol), rs in groups.items():
        a = dict(geometry=geom, swa=swa, policy=pol, seeds=len(rs))
        for k in keys:
            a[k] = sum(r[k] for r in rs) / len(rs)
        out.append(a)
    out.sort(key=lambda r: (r["geometry"], r["swa"], r["policy"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seeds", default="1,7,11,23,42")
    ap.add_argument("--geometries", default="real_5760B")
    ap.add_argument("--swa", default="small,mid,big")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "l2_pool_multiclass_results.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    geoms = args.geometries.split(",")
    swas = args.swa.split(",")

    raw = []
    for geom in geoms:
        for swa in swas:
            for pol in POLICIES:
                for seed in seeds:
                    raw.append(run_case(geom, swa, pol, args.steps, seed))
    rows = aggregate(raw)

    lines = []
    for geom in geoms:
        for swa in swas:
            lines.append(f"\n### {geom} / third class = swa_{swa}\n")
            lines.append("| policy | used% | fail MLA% | 512p MLA% | 16p MLA% "
                         "| fail SWA% | fail STATE% | frag MLA% | runs "
                         "| moved/alloc |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for r in [x for x in rows
                      if x["geometry"] == geom and x["swa"] == swa]:
                lines.append(
                    f"| `{r['policy']}` | {r['used_pct']:.1f} "
                    f"| {r['mla_fail_pct']:.2f} "
                    f"| {r['size_mla_512_fail_pct']:.2f} "
                    f"| {r['size_mla_16_fail_pct']:.2f} "
                    f"| {r['swa_fail_pct']:.2f} "
                    f"| {r['state_fail_pct']:.2f} | {r['mla_frag_pct']:.2f} "
                    f"| {r['avg_runs']:.1f} | {r['moved_pages']:.0f} |")
    print("\n".join(lines))

    with open(args.out, "w") as f:
        json.dump(dict(steps=args.steps, seeds=seeds, rows=rows, raw=raw), f, indent=2)
    print(f"\n[multiclass] rows={len(rows)} seeds={len(seeds)} "
          f"steps={args.steps} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
