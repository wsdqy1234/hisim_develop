#!/usr/bin/env python3
"""Phase 2C.3.2 trace: L2 Placement Policy Differentiation.

Designs workloads where NativeL2, HotnessL2, DepthAwareL2, TierCostAwareL2
select DIFFERENT victims by varying:
  - Access frequency (hot vs cold)
  - Recency (time-decayed热度)
  - Checkpoint depth (shallow vs deep)
  - Hotspot migration (old hot → new hot)

Structure:
  Phase 1: Build A (4 req × 4K) — fills L2, triggers eviction
  Phase 2: Re-access A0,A1 (2 req) — creates freq>0 for hot set
  Phase 3: Build B (4 req × 4K) — L2 pressure, evictions use freq
  Phase 4: Re-access A0,A1 (2 req) — positive: HotnessL2 should hit
  Phase 5: Re-access B0,B1 (2 req) — negative: HotnessL2 may miss
  Phase 6: Churn (8 req × 4K) — capacity pressure
  Phase 7: Re-access A0,A1 (2 req) — final: adaptation test
  Phase 8: Re-access B0,B1 (2 req) — final: migration test

Total: 26 requests. Short tokens (4K) for fast iteration.

Two variants:
  l2diff_hot  — Phase 4 re-accesses A (hot set preserved by HotnessL2)
  l2diff_mig  — Phase 4 re-accesses B (hotspot migration, HotnessL2 degrades)
"""
import json, os, sys


def tokens(base, n):
    return [base + i for i in range(n + 10)]


def row(ids, length, ts, label=""):
    return {"input_ids": ids[:length], "input_length": length,
            "output_length": 1, "timestamp": ts, "session_label": label}


def gen_l2diff(variant, outdir):
    """Generate L2 differentiation trace.

    variant: "hot" (positive test) or "mig" (hotspot migration / negative test)
    """
    TOKENS = 4096  # 4K tokens per request → ~4 checkpoints (track_interval=1024)
    rows = []
    ts = 0.0

    # Distinct token bases for each session (ensures different radix tree paths)
    base_A = [100000 * (i + 1) for i in range(4)]  # A0-A3
    base_B = [500000 * (i + 1) for i in range(4)]  # B0-B3
    base_churn = [900000 * (i + 1) for i in range(8)]  # C0-C7

    A = [tokens(base_A[i], TOKENS) for i in range(4)]
    B = [tokens(base_B[i], TOKENS) for i in range(4)]
    C = [tokens(base_churn[i], TOKENS) for i in range(8)]

    # Phase 1: Build A (4 requests, same timestamp → batched, fills L1/L2)
    for i in range(4):
        rows.append(row(A[i], TOKENS, ts, f"P1_build_A{i}"))
    ts += 10.0

    # Phase 2: Re-access A0, A1 (2 requests → L2 lookup → freq>0)
    for i in range(2):
        rows.append(row(A[i], TOKENS + 1, ts, f"P2_reaccess_A{i}"))
    ts += 10.0

    # Phase 3: Build B (4 requests → L2 fills → evictions with freq available)
    for i in range(4):
        rows.append(row(B[i], TOKENS, ts, f"P3_build_B{i}"))
    ts += 10.0

    if variant == "hot":
        # POSITIVE TEST: Phase 4 re-accesses A (HotnessL2 should preserve A)
        for i in range(2):
            rows.append(row(A[i], TOKENS + 2, ts, f"P4_reaccess_A{i}_hot"))
        ts += 10.0
        # Phase 5: Re-access B (HotnessL2 may miss B if B was evicted)
        for i in range(2):
            rows.append(row(B[i], TOKENS + 2, ts, f"P5_reaccess_B{i}_cold"))
        ts += 10.0
    else:
        # NEGATIVE TEST (migration): Phase 4 re-accesses B (not A)
        # HotnessL2 keeps old hot A (freq>0) but A is no longer needed → waste
        # NativeL2 keeps recent B → B hits
        for i in range(2):
            rows.append(row(B[i], TOKENS + 2, ts, f"P4_reaccess_B{i}_new_hot"))
        ts += 10.0
        # Phase 5: Re-access A (old hot, HotnessL2 might still have it)
        for i in range(2):
            rows.append(row(A[i], TOKENS + 2, ts, f"P5_reaccess_A{i}_old_hot"))
        ts += 10.0

    # Phase 6: Churn (8 requests → heavy L2 pressure)
    for i in range(8):
        rows.append(row(C[i], TOKENS, ts, f"P6_churn_C{i}"))
    ts += 10.0

    # Phase 7: Final re-access A0, A1
    for i in range(2):
        rows.append(row(A[i], TOKENS + 3, ts, f"P7_final_A{i}"))
    ts += 10.0

    # Phase 8: Final re-access B0, B1
    for i in range(2):
        rows.append(row(B[i], TOKENS + 3, ts, f"P8_final_B{i}"))
    ts += 10.0

    # Pad to 28 requests (multiples of 4 for batching)
    while len(rows) < 28:
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
