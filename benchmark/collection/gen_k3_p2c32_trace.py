#!/usr/bin/env python3
"""Phase 2C.3.2 trace: L2 Placement Policy Differentiation (v2).

Key fix: enough build requests to fill L1 (16 slots) and trigger eviction
BEFORE re-access. With 1 checkpoint/request and L1=16, need 17+ build
requests before any L1 eviction happens.

Structure (l2diff_hot):
  Phase 1: 20 build A (fills L1, starts eviction at req 17) → L2 fills
  Phase 2: 4 re-access A0-A3 (L2 lookup → freq>0 for hot set)
  Phase 3: 10 build B (L2 pressure → evictions with freq available)
  Phase 4: 4 re-access A0-A3 (positive: HotnessL2 should hit)
  Phase 5: 4 re-access B0-B3 (negative: HotnessL2 may miss)
  Phase 6: 4 final re-access A0-A3 (adaptation test)

Total: 46 requests × 8K tokens = 368K trace tokens.
KV cache: max 1 × 8K = 8K active (sequential timestamps).

Variants:
  l2diff_hot — Phase 4 re-accesses A (HotnessL2 should preserve A)
  l2diff_mig — Phase 4 re-accesses B (hotspot migration, HotnessL2 degrades)
"""
import json, os, sys


def tokens(base, n):
    return [base + i for i in range(n + 10)]


def row(ids, length, ts, label=""):
    return {"input_ids": ids[:length], "input_length": length,
            "output_length": 1, "timestamp": ts, "session_label": label}


def gen_l2diff(variant, outdir):
    TOKENS = 8192  # 8K tokens, 1 checkpoint per request
    N_BUILD_A = 20  # enough to fill L1=16 and trigger eviction
    N_BUILD_B = 10
    N_HOT = 4       # re-access first 4 of each set
    rows = []
    ts = 0.0

    # Generate distinct token bases
    base_A = [100000 * (i + 1) for i in range(N_BUILD_A)]
    base_B = [500000 * (i + 1) for i in range(N_BUILD_B)]

    A = [tokens(base_A[i], TOKENS) for i in range(N_BUILD_A)]
    B = [tokens(base_B[i], TOKENS) for i in range(N_BUILD_B)]

    # Phase 1: Build A (20 requests, sequential → fills L1, triggers eviction)
    for i in range(N_BUILD_A):
        rows.append(row(A[i], TOKENS, ts, f"P1_build_A{i}"))
        ts += 10.0

    # Phase 2: Re-access A0-A3 (4 requests → L2 lookup for evicted checkpoints)
    for i in range(N_HOT):
        rows.append(row(A[i], TOKENS + 1, ts, f"P2_reaccess_A{i}"))
        ts += 10.0

    # Phase 3: Build B (10 requests → L2 fills → evictions with freq available)
    for i in range(N_BUILD_B):
        rows.append(row(B[i], TOKENS, ts, f"P3_build_B{i}"))
        ts += 10.0

    if variant == "hot":
        # POSITIVE: Phase 4 re-accesses A (HotnessL2 should preserve A)
        for i in range(N_HOT):
            rows.append(row(A[i], TOKENS + 2, ts, f"P4_reaccess_A{i}_hot"))
            ts += 10.0
        # Phase 5: Re-access B (HotnessL2 may miss B)
        for i in range(N_HOT):
            rows.append(row(B[i], TOKENS + 2, ts, f"P5_reaccess_B{i}_cold"))
            ts += 10.0
    else:
        # NEGATIVE (migration): Phase 4 re-accesses B (not A)
        for i in range(N_HOT):
            rows.append(row(B[i], TOKENS + 2, ts, f"P4_reaccess_B{i}_new_hot"))
            ts += 10.0
        # Phase 5: Re-access A (old hot)
        for i in range(N_HOT):
            rows.append(row(A[i], TOKENS + 2, ts, f"P5_reaccess_A{i}_old_hot"))
            ts += 10.0

    # Phase 6: Final re-access A0-A3
    for i in range(N_HOT):
        rows.append(row(A[i], TOKENS + 3, ts, f"P6_final_A{i}"))
        ts += 10.0

    # Pad to 48 (multiples of 4)
    while len(rows) < 48:
        rows.append(row(A[0], TOKENS + 4, ts, f"pad_{len(rows)}"))
        ts += 10.0

    name = f"k3_p2c32_l2diff_{variant}.jsonl"
    path = os.path.join(outdir, name)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Generated {name}: {len(rows)} requests, {TOKENS} tokens each")
    return path


if __name__ == "__main__":
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode"
    variant = sys.argv[1] if len(sys.argv) > 1 else "hot"
    gen_l2diff(variant, outdir)
