#!/usr/bin/env python3
"""Fragmentation study for the unified L2 DRAM pool (KDA + MLA).

Synthetic but representative workload over one fixed byte budget:

  * KDA objects: one checkpoint slot (14.65 MB), identity = slot hash.
  * MLA objects: a contiguous token run (256..8192 tokens), identity = token
    index.  Size distribution follows the K3 request/node write sizes.
  * Eviction on pressure: LRU inside the same view for ``two_pools``; for
    ``one_pool`` the default ``self_first`` borrow policy replaces inside the
    view first and only steals from the other view when that cannot satisfy
    the request.  ``other_first`` is reported separately (demand-driven).

Reported numbers:
  * internal fragmentation = reserved pages - requested bytes (tail waste)
  * external fragmentation = free bytes - largest free run
  * achievable occupancy at steady state for a fixed budget

Usage:
    python benchmark/collection/l2_pool_frag_study.py [--budget-gb 8]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hisim.simulation.l2_dram_pool import (  # noqa: E402
    DEFAULT_KDA_SLOT_BYTES,
    UnifiedDRAMPool,
)

TOKEN_BYTES = 5760  # K3 MLA host bytes per token (5 full-attention layers)
SLOT_BYTES = DEFAULT_KDA_SLOT_BYTES

MLA_SIZE_CHOICES = [256, 512, 1024, 2048, 4096, 8192]
MLA_SIZE_WEIGHTS = [20, 25, 20, 15, 12, 8]


@dataclass
class Workload:
    name: str
    kda_frac: float  # probability that the next object is a KDA checkpoint


WORKLOADS = [
    Workload("kda_heavy", 0.70),
    Workload("balanced", 0.40),
    Workload("mla_heavy", 0.10),
]


def run_case(
    mode: str,
    page_tokens: int,
    policy: str,
    workload: Workload,
    budget_gb: float,
    kda_slots: int,
    floor_slots: int = 0,
    borrow_policy: str = "self_first",
    ops: int = 6000,
    seed: int = 7,
) -> dict:
    rng = random.Random(seed)
    kda_region = min(int(budget_gb * 1e9), kda_slots * SLOT_BYTES)
    pool = UnifiedDRAMPool(
        total_bytes=int(budget_gb * 1e9),
        mode=mode,
        token_bytes=TOKEN_BYTES,
        kda_slot_bytes=SLOT_BYTES,
        page_tokens=page_tokens,
        policy=policy,
        kda_region_bytes=kda_region if mode == "two_pools" else None,
        kda_floor_slots=floor_slots if mode == "one_pool" else 0,
        borrow_policy=borrow_policy,
    )

    # Logical LRU bookkeeping (oldest first).
    mla_lru: List[int] = []  # extent ids
    kda_lru: List[str] = []
    next_kda = 0
    failures_mla = 0
    failures_kda = 0
    evictions_mla = 0
    evictions_kda = 0

    def drop_oldest_mla() -> bool:
        nonlocal evictions_mla
        while mla_lru:
            eid = mla_lru.pop(0)
            ext = pool.mla.extents.get(eid)
            if ext is None:
                continue
            start = ext.start_page * pool.page_tokens
            idx = torch.arange(start, start + ext.live, dtype=torch.int64)
            pool.mla.free_tokens(idx)
            evictions_mla += 1
            return True
        return False

    def drop_oldest_kda() -> bool:
        nonlocal evictions_kda
        while kda_lru:
            h = kda_lru.pop(0)
            if pool.kda.release(h):
                evictions_kda += 1
                return True
        return False

    def alloc_mla(n_tokens: int) -> bool:
        nonlocal failures_mla
        idx = pool.alloc_mla_tokens(n_tokens)
        pages = pool.mla.pages_for(n_tokens)
        cap_blocked = (
            pool.mla.cap_pages is not None
            and pool.mla.used_pages + pages > pool.mla.cap_pages
        )
        tries = 0
        while idx is None and tries < 256:
            tries += 1
            # Same as the implementation: replace inside the view first (the
            # tree already evicts MLA before backing up), then borrow.
            if drop_oldest_mla():
                idx = pool.alloc_mla_tokens(n_tokens)
                continue
            if mode == "one_pool" and not cap_blocked and pool.kda.slots and \
                    pool.mla.available_pages() < pages:
                if drop_oldest_kda():
                    idx = pool.alloc_mla_tokens(n_tokens)
                    continue
            break
        if idx is None:
            failures_mla += 1
            return False
        mla_lru.append(max(pool.mla.extents))
        return True

    def admit_kda() -> bool:
        nonlocal failures_kda, next_kda
        key = f"k{next_kda}"
        ok = pool.admit_kda(key)
        # Mirror UnifiedDRAMPool.admit_kda's borrow policy.
        def _self():
            if drop_oldest_kda():
                return pool.admit_kda(key)
            return False

        def _other():
            if mode == "one_pool" and drop_oldest_mla():
                return pool.admit_kda(key)
            return False

        steps = (_self, _other) if borrow_policy == "self_first" else (_other, _self)
        ok = pool.admit_kda(key)
        for step in steps:
            if ok:
                break
            ok = step()
        tries = 0
        while not ok and tries < 256:
            tries += 1
            ok = _self() or _other()
            if not ok:
                break
        if not ok:
            failures_kda += 1
            return False
        next_kda += 1
        kda_lru.append(key)
        return True

    for _ in range(ops):
        if rng.random() < workload.kda_frac:
            admit_kda()
        else:
            n = rng.choices(MLA_SIZE_CHOICES, weights=MLA_SIZE_WEIGHTS, k=1)[0]
            alloc_mla(n)

    st = pool.stats()
    mode_label = mode
    if mode == "one_pool":
        mode_label += f"+{borrow_policy.replace('_first', '')}"
        if floor_slots:
            mode_label += f"+floor{floor_slots}"
    return {
        "mode": mode_label,
        "workload": workload.name,
        "page_tokens": page_tokens,
        "policy": policy,
        "borrow_policy": borrow_policy,
        "budget_gb": budget_gb,
        "used_gb": st["used_bytes"] / 1e9,
        "used_pct": 100.0 * st["used_bytes"] / st["total_bytes"],
        "kv_gb": st["kv_bytes"] / 1e9,
        "kda_gb": st["kda_bytes"] / 1e9,
        "mla_tokens": st["mla_live_tokens"],
        "kda_slots": st["kda_cached_slots"],
        "internal_frag_pct": 100.0 * st["internal_frag_ratio"],
        "external_frag_pct": 100.0 * st["external_frag_ratio"],
        "largest_free_mb": st["largest_free_run_bytes"] / 1e6,
        "free_runs": st["free_run_count"],
        "fail_mla": failures_mla,
        "fail_kda": failures_kda,
        "evict_mla": evictions_mla,
        "evict_kda": evictions_kda,
        "mla_extents": st["mla_extents"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=8.0)
    ap.add_argument("--kda-slots", type=int, default=330,
                    help="KDA region size for two_pools (default 330 slots)")
    ap.add_argument("--floor-slots", type=int, default=64,
                    help="KDA floor for the extra one_pool rows (default 64)")
    ap.add_argument("--ops", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "l2_pool_frag_study_results.json"))
    args = ap.parse_args()

    rows = []
    for workload in WORKLOADS:
        for mode in ("two_pools", "one_pool"):
            for page_tokens in (1, 16, 64, 256):
                for policy in ("exact", "pow2"):
                    rows.append(run_case(
                        mode, page_tokens, policy, workload,
                        args.budget_gb, args.kda_slots, ops=args.ops,
                    ))
        # One Pool borrow-order comparison + a reserved KDA floor (protects
        # checkpoints from a KV-dominated workload).
        for page_tokens in (16, 64):
            rows.append(run_case(
                "one_pool", page_tokens, "exact", workload,
                args.budget_gb, args.kda_slots,
                borrow_policy="other_first", ops=args.ops,
            ))
            rows.append(run_case(
                "one_pool", page_tokens, "exact", workload,
                args.budget_gb, args.kda_slots,
                floor_slots=args.floor_slots, ops=args.ops,
            ))

    header = (
        "| workload | mode | page | alloc | used% | KV GB | KDA slots | "
        "int frag% | ext frag% | largest free MB | fail(MLA/KDA) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    print(f"### L2 DRAM pool fragmentation study (budget={args.budget_gb} GB, "
          f"KDA region={args.kda_slots} slots, ops={args.ops})\n")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['workload']} | {r['mode']} | {r['page_tokens']} | {r['policy']} | "
            f"{r['used_pct']:.1f} | {r['kv_gb']:.2f} | {r['kda_slots']} | "
            f"{r['internal_frag_pct']:.2f} | {r['external_frag_pct']:.2f} | "
            f"{r['largest_free_mb']:.1f} | {r['fail_mla']}/{r['fail_kda']} |"
        )

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
