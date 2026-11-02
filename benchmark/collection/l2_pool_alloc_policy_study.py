#!/usr/bin/env python3
"""Allocation-policy study: two-ended first-fit vs MemCache-style best-fit.

Question
--------
In the unified L2 DRAM pool two object shapes share one byte budget:

  * MLA: contiguous token runs (16..512 pages), short lived, churns with the
    prefix tree / LRU;
  * KDA: one checkpoint slot per object, long lived.

``PageExtentAllocator`` serves MLA from the low address end and KDA from the
high address end with **first-fit by address** (two-ended / stack-heap split).
MemCache's ``MmcBlobAllocator`` instead keeps an address tree plus a size tree
and serves every request with **best-fit** (smallest free run that fits),
splitting from that run's low end.  MemCache also has a *design* for dual-ended
allocation (small -> head, large -> tail, best-fit inside each end) that is not
merged into the C++ source yet.

Method
------
This is an **open-loop** allocator benchmark: the arrival/free trace is
generated independently of the allocator (exponential inter-arrival per class,
exponential lifetimes), so every policy sees exactly the same offered
workload.  Nothing is evicted to make room, which is what makes the comparison
fair: a request that the allocator cannot place is simply lost, and the loss is
the fragmentation cost we want to measure.

Policies
--------
    two_ended_ff    MLA low / KDA high, first-fit inside the end   (current)
    two_ended_bf    MLA low / KDA high, best-fit inside the end    (option D)
    best_fit_low    one space, best-fit, split at run head         (MemCache)
    best_fit_high   one space, best-fit, split at run tail
    dual_ended_bf@T size <= T pages -> low, else high, best-fit    (MemCache
                    dual-ended design, threshold sweep)
    ideal_compact   zero external fragmentation (compaction upper bound)

Metrics
-------
    frag MLA%       share of MLA requests rejected although free_pages >=
                    request size (pure external fragmentation) -- lower better
    frag KDA%       same for KDA
    drop%           offered pages lost to any reject (cumulative)
    used_pct        average occupancy in the sampling window
    contig_pct      avg largest_free_run / free_pages
    big-blocked%    share of samples where a full 512-page write batch would
                    have been rejected even though 512 pages were free
    runs            average number of free runs (allocator bookkeeping cost)

Usage:
    python benchmark/collection/l2_pool_alloc_policy_study.py
    python benchmark/collection/l2_pool_alloc_policy_study.py --steps 40000 --seed 11
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import os
import random
from bisect import bisect_left
from dataclasses import dataclass
from typing import List, Optional

TOKEN_BYTES = 5760            # real K3 MLA host bytes per token
SLOT_BYTES = 14_653_440       # K3-23L TP=8 KDA state slot
PAGE_TOKENS = 16

GEOMETRIES = {
    # Real entry accounting: page = 92,160 B, a KDA slot spans 159 pages, so the
    # two classes overlap in size.  8 GB budget.
    "real_5760B": dict(token_bytes=TOKEN_BYTES, page_tokens=PAGE_TOKENS,
                       slot_bytes=SLOT_BYTES, budget_gb=8.0),
    # What the server runs today: mock pool size_per_token is ~64x the real
    # entry, page = 27.1 MB, KDA slot fits in 1 page.  85 GB budget.
    "server_1656KB": dict(token_bytes=1_695_744, page_tokens=PAGE_TOKENS,
                          slot_bytes=SLOT_BYTES, budget_gb=85.0),
}

MLA_SIZE_CHOICES = [16, 32, 64, 128, 256, 512]   # pages per MLA run
MLA_SIZE_WEIGHTS = [15, 20, 25, 20, 12, 8]
MEAN_MLA_PAGES = sum(p * w for p, w in zip(MLA_SIZE_CHOICES, MLA_SIZE_WEIGHTS)) \
    / sum(MLA_SIZE_WEIGHTS)
BIG_REQUEST_PAGES = 512                          # one full write batch


@dataclass(frozen=True)
class Policy:
    name: str
    pick: str                 # first_fit | best_fit | ideal
    mode: str                 # class_end | size_end | single_low | single_high
    threshold: int = 0        # pages, for size_end


POLICIES = [
    Policy("two_ended_ff", "first_fit", "class_end"),
    Policy("two_ended_bf", "best_fit", "class_end"),
    Policy("best_fit_low", "best_fit", "single_low"),
    Policy("best_fit_high", "best_fit", "single_high"),
    Policy("dual_ended_bf@16", "best_fit", "size_end", 16),
    Policy("dual_ended_bf@64", "best_fit", "size_end", 64),
    Policy("dual_ended_bf@159", "best_fit", "size_end", 159),
    Policy("ideal_compact", "ideal", "class_end"),
]


@dataclass
class Obj:
    cls: str
    pages: int
    start: int


class PageAllocator:
    """Page-range allocator with pluggable end selection and fit policy.

    Free space is kept as address ordered ``(starts, sizes)`` arrays; ``free``
    coalesces both neighbours, so the structure is exactly what
    ``PageExtentAllocator`` keeps (one address-ordered run list).
    """

    def __init__(self, total_pages: int, policy: Policy):
        self.total = int(total_pages)
        self.policy = policy
        self.starts: List[int] = [0]
        self.sizes: List[int] = [self.total]
        self.free_pages = self.total
        self.used_pages = 0

        self.offered_pages = 0
        self.frag_fail = {"mla": 0, "kda": 0}
        self.overload_fail = {"mla": 0, "kda": 0}
        self.attempts = {"mla": 0, "kda": 0}
        self.drop_pages = 0

    # -- helpers ------------------------------------------------------------
    def _end(self, cls: str, pages: int) -> str:
        mode = self.policy.mode
        if mode == "single_low":
            return "low"
        if mode == "single_high":
            return "high"
        if mode == "size_end":
            return "low" if pages <= self.policy.threshold else "high"
        return "low" if cls == "mla" else "high"   # class_end

    def largest_free_run(self) -> int:
        return max(self.sizes, default=0)

    def _take(self, idx: int, need: int, end: str) -> int:
        if end == "low":
            start = self.starts[idx]
            self.starts[idx] += need
        else:
            start = self.starts[idx] + self.sizes[idx] - need
        self.sizes[idx] -= need
        if self.sizes[idx] == 0:
            del self.starts[idx]
            del self.sizes[idx]
        self.free_pages -= need
        self.used_pages += need
        return start

    # -- allocation ---------------------------------------------------------
    def alloc(self, pages: int, cls: str) -> Optional[Obj]:
        need = int(pages)
        if need <= 0:
            return Obj(cls, 0, -1)
        self.attempts[cls] += 1
        self.offered_pages += need

        if self.policy.pick == "ideal":
            if self.free_pages < need:
                self.overload_fail[cls] += 1
                self.drop_pages += need
                return None
            self.free_pages -= need
            self.used_pages += need
            return Obj(cls, need, -1)

        end = self._end(cls, need)
        sizes = self.sizes
        if self.policy.pick == "best_fit":
            best = -1
            best_key = None
            sign = 1 if end == "low" else -1
            for i, n in enumerate(sizes):
                if n < need:
                    continue
                key = (n, sign * self.starts[i])
                if best_key is None or key < best_key:
                    best_key = key
                    best = i
            idx = best
        elif end == "low":
            idx = -1
            for i, n in enumerate(sizes):
                if n >= need:
                    idx = i
                    break
        else:
            idx = -1
            for i in range(len(sizes) - 1, -1, -1):
                if sizes[i] >= need:
                    idx = i
                    break

        if idx < 0:
            if self.free_pages >= need:
                self.frag_fail[cls] += 1
            else:
                self.overload_fail[cls] += 1
            self.drop_pages += need
            return None
        return Obj(cls, need, self._take(idx, need, end))

    def free(self, obj: Optional[Obj]) -> None:
        if obj is None or obj.pages <= 0:
            return
        if self.policy.pick == "ideal":
            self.free_pages += obj.pages
            self.used_pages -= obj.pages
            return

        start, pages = obj.start, obj.pages
        starts, sizes = self.starts, self.sizes
        pos = bisect_left(starts, start)
        # coalesce with the left neighbour
        if pos > 0 and starts[pos - 1] + sizes[pos - 1] == start:
            sizes[pos - 1] += pages
            m = pos - 1
        else:
            starts.insert(pos, start)
            sizes.insert(pos, pages)
            m = pos
        # coalesce with the right neighbour
        if m + 1 < len(starts) and starts[m] + sizes[m] == starts[m + 1]:
            sizes[m] += sizes[m + 1]
            del starts[m + 1]
            del sizes[m + 1]
        self.free_pages += pages
        self.used_pages -= pages

    def snapshot(self) -> dict:
        free = self.free_pages
        if self.policy.pick == "ideal":
            # Perfect compaction: any free page is usable by any request.
            return dict(used=self.used_pages, free=free, runs=0,
                        largest=free, contig=1.0 if free else 1.0)
        largest = self.largest_free_run()
        return dict(used=self.used_pages, free=free, runs=len(self.sizes),
                    largest=largest,
                    contig=(largest / free) if free else 1.0)


@dataclass(frozen=True)
class Workload:
    name: str
    mla_share: float          # share of offered bytes that is MLA
    rho: float                # offered bytes / pool bytes


WORKLOADS = [
    Workload("mla_heavy", 0.70, 0.95),
    Workload("balanced", 0.50, 0.95),
    Workload("kda_heavy", 0.30, 0.95),
    Workload("mla_heavy@0.99", 0.70, 0.99),
    Workload("balanced@0.99", 0.50, 0.99),
    Workload("kda_heavy@0.99", 0.30, 0.99),
]

T_MLA_STEPS = 300             # mean MLA extent lifetime
T_KDA_STEPS = 3000            # KDA slots live ~10x longer


def run_case(geom: str, policy: Policy, workload: Workload,
             steps: int, seed: int) -> dict:
    g = GEOMETRIES[geom]
    page_bytes = g["token_bytes"] * g["page_tokens"]
    total_pages = int(g["budget_gb"] * 1e9) // page_bytes
    kda_pages = max(1, -(-g["slot_bytes"] // page_bytes))  # ceil

    rng = random.Random(seed)
    alloc = PageAllocator(total_pages, policy)

    # Arrival rates chosen so offered bytes = rho * pool bytes.
    lam_mla = workload.rho * workload.mla_share * total_pages / (MEAN_MLA_PAGES * T_MLA_STEPS)
    lam_kda = workload.rho * (1.0 - workload.mla_share) * total_pages / (kda_pages * T_KDA_STEPS)

    # Exponential inter-arrival times, exact for any lambda.
    next_mla = rng.expovariate(lam_mla) if lam_mla > 0 else float("inf")
    next_kda = rng.expovariate(lam_kda) if lam_kda > 0 else float("inf")

    frees = []      # heap of (deadline, seq, obj)
    seq = itertools.count()
    live_pages = {"mla": 0, "kda": 0}
    live_count = {"mla": 0, "kda": 0}
    samples = []
    warmup = max(3 * T_KDA_STEPS, 10 * T_MLA_STEPS)
    warmup = min(warmup, steps // 2)

    def free_due(now: int) -> None:
        while frees and frees[0][0] <= now:
            _, _, obj = heapq.heappop(frees)
            alloc.free(obj)
            live_pages[obj.cls] -= obj.pages
            live_count[obj.cls] -= 1

    t = 0.0
    step = 0
    while step < steps:
        step = int(t)
        free_due(step)

        while step < steps and next_mla < t + 1.0:
            pages = rng.choices(MLA_SIZE_CHOICES, weights=MLA_SIZE_WEIGHTS, k=1)[0]
            obj = alloc.alloc(pages, "mla")
            if obj is not None:
                live_pages["mla"] += obj.pages
                live_count["mla"] += 1
                heapq.heappush(frees, (step + int(rng.expovariate(1.0 / T_MLA_STEPS)) + 1,
                                       next(seq), obj))
            next_mla += rng.expovariate(lam_mla)

        while step < steps and next_kda < t + 1.0:
            obj = alloc.alloc(kda_pages, "kda")
            if obj is not None:
                live_pages["kda"] += obj.pages
                live_count["kda"] += 1
                heapq.heappush(frees, (step + int(rng.expovariate(1.0 / T_KDA_STEPS)) + 1,
                                       next(seq), obj))
            next_kda += rng.expovariate(lam_kda)

        if step >= warmup and step % 10 == 0:
            samples.append(alloc.snapshot())
        t += 1.0

    n = max(len(samples), 1)
    avg_used = sum(s["used"] for s in samples) / n
    avg_contig = sum(s["contig"] for s in samples) / n
    avg_runs = sum(s["runs"] for s in samples) / n
    big_blocked = sum(1 for s in samples
                      if s["free"] >= BIG_REQUEST_PAGES and s["largest"] < BIG_REQUEST_PAGES)
    min_largest = min((s["largest"] for s in samples), default=0)
    att_mla, att_kda = alloc.attempts["mla"], alloc.attempts["kda"]
    return dict(
        geometry=geom,
        policy=policy.name,
        workload=workload.name,
        seed=seed,
        total_pages=total_pages,
        kda_pages=kda_pages,
        offered_pages=alloc.offered_pages,
        attempts_mla=att_mla,
        attempts_kda=att_kda,
        frag_fail_mla=alloc.frag_fail["mla"],
        frag_fail_kda=alloc.frag_fail["kda"],
        frag_rate_mla=100.0 * alloc.frag_fail["mla"] / max(att_mla, 1),
        frag_rate_kda=100.0 * alloc.frag_fail["kda"] / max(att_kda, 1),
        over_fail_mla=alloc.overload_fail["mla"],
        over_fail_kda=alloc.overload_fail["kda"],
        drop_pages=alloc.drop_pages,
        drop_pct=100.0 * alloc.drop_pages / alloc.offered_pages
        if alloc.offered_pages else 0.0,
        used_pct=100.0 * avg_used / total_pages,
        contig_pct=100.0 * avg_contig,
        avg_runs=avg_runs,
        min_largest_pages=min_largest,
        big_blocked_pct=100.0 * big_blocked / n,
        mla_resident=live_pages["mla"],
        kda_resident=live_count["kda"],
        kda_pages_resident=live_pages["kda"],
    )


MEAN_FIELDS = ["used_pct", "frag_rate_mla", "frag_rate_kda", "drop_pct",
               "contig_pct", "big_blocked_pct", "avg_runs",
               "frag_fail_mla", "frag_fail_kda"]


def aggregate(rows: List[dict]) -> List[dict]:
    """Mean over seeds for every (geometry, workload, policy)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["geometry"], r["workload"], r["policy"]), []).append(r)
    out = []
    for (geom, wl, pol), rs in groups.items():
        agg = dict(geometry=geom, workload=wl, policy=pol, seeds=len(rs))
        for f in MEAN_FIELDS:
            agg[f] = sum(r[f] for r in rs) / len(rs)
        out.append(agg)
    out.sort(key=lambda r: (r["geometry"], r["workload"], r["policy"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seeds", default="1,7,11,23,42")
    ap.add_argument("--geometries", default="real_5760B,server_1656KB")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "l2_pool_alloc_policy_results.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    raw = []
    for geom in args.geometries.split(","):
        for wl in WORKLOADS:
            for pol in POLICIES:
                for seed in seeds:
                    raw.append(run_case(geom, pol, wl, args.steps, seed))
    rows = aggregate(raw)

    # ---- markdown ---------------------------------------------------------
    lines = []
    for geom in args.geometries.split(","):
        g = GEOMETRIES[geom]
        page_bytes = g["token_bytes"] * g["page_tokens"]
        total_pages = int(g["budget_gb"] * 1e9) // page_bytes
        lines.append(
            f"\n### geometry `{geom}` - budget {g['budget_gb']:.0f} GB, "
            f"page {page_bytes/1e3:.0f} KB, {total_pages} pages, "
            f"KDA slot = {max(1, -(-g['slot_bytes'] // page_bytes))} pages\n")
        lines.append("| workload | policy | used% | frag MLA% | frag KDA% "
                     "| drop% | contig% | big-blocked% | runs |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for wl in WORKLOADS:
            for r in [x for x in rows
                      if x["geometry"] == geom and x["workload"] == wl.name]:
                lines.append(
                    f"| {wl.name} | `{r['policy']}` | {r['used_pct']:.1f} "
                    f"| {r['frag_rate_mla']:.2f} | {r['frag_rate_kda']:.2f} "
                    f"| {r['drop_pct']:.2f} | {r['contig_pct']:.1f} "
                    f"| {r['big_blocked_pct']:.1f} | {r['avg_runs']:.1f} |")
    md = "\n".join(lines)
    print(md)

    with open(args.out, "w") as f:
        json.dump(dict(steps=args.steps, seeds=seeds,
                       rows=rows, raw=raw), f, indent=2)
    print(f"\n[alloc-policy] rows={len(rows)} seeds={len(seeds)} "
          f"steps={args.steps} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
