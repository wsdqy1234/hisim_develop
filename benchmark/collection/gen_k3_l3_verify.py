#!/usr/bin/env python3
"""Generate KDA L3 reusability verification trace.

Design (审查意见 5):
  Build → Churn (evict L1→L2→L3) → Reuse (find in L3)

  - 20 build sessions with 1024-token unique prefixes (C=1024)
  - Reuse: re-send sessions 0,5,10,15 with +1 token (C+1=1025)
  - Run with small L1 (KDA=10) and L2 (HISIM_L2_MAMBA_SIZE=3) to force L3

  Expected:
  - Build sessions 0-2 stay in L1 (3 slots × 3 slots/req ≈ 9 < 10)
  - Sessions 3+ evict older to L2, L2 fills, overflow to L3
  - Reuse sessions 0,5 should find checkpoints in L3
  - Reuse sessions 10,15 should find in L2

Usage: gen_k3_l3_verify.py [outdir]  (default /tmp/opencode)
"""
import json
import os
import sys

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode"

C = 1024          # track interval / build length
OUT_LEN = 1       # minimal output
SESSIONS = 20
REUSE_SESSIONS = [0, 5, 10, 15]


def session_tokens(s):
    """Disjoint token ranges per session (no cross-session prefix collision)."""
    return [s * 200_000 + i for i in range(C + 10)]  # extra for reuse


def row(ids, length, ts):
    return {
        "input_ids": ids[:length],
        "input_length": length,
        "output_length": OUT_LEN,
        "timestamp": ts,
    }


def gen_trace():
    rows = []
    # Build phase: 20 sessions, each with unique 1024-token prefix
    for s in range(SESSIONS):
        rows.append(row(session_tokens(s), C, s * 1.0))
    # Reuse phase: re-send 4 sessions with +1 token (C+1=1025)
    for i, s in enumerate(REUSE_SESSIONS):
        rows.append(row(session_tokens(s), C + 1, 100.0 + i))
    return rows


if __name__ == "__main__":
    rows = gen_trace()
    path = os.path.join(OUTDIR, "k3_l3_verify.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = sum(r["input_length"] for r in rows)
    print(f"[gen_l3_verify] {path}: {len(rows)} reqs, total input {total} tokens")
    print(f"  Build: {SESSIONS} sessions × {C} tokens")
    print(f"  Reuse: {len(REUSE_SESSIONS)} sessions × {C+1} tokens")
