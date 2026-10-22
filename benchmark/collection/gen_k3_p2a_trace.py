#!/usr/bin/env python3
"""Phase 2A trace v4: Fork test + no token collisions.

1. Build hot S0-S7 (base=100001..800001)
2. Re-access S0-S7 (frequency becomes 1)
3. Build cold S8-S15 (base=1000001..1800001)
4. Build fork F0-F1 (shares first token with S0/S1, different suffix)
5. Churn C0-C11 (12, capacity pressure → eviction)
6. Reuse S0-S7 (should hit with Hotness/Oracle, miss with Native)
7. Reuse S8-S15 (should miss with Native, partial with Oracle)
8. Reuse F0-F1 (should NOT be merged with S0/S1)

Fork test: F0 shares first token 100001 with S0, but F0's second token
is 999001 (different from S0's 100002). This proves:
  - _node_first_token would merge S0 and F0 (same first token)
  - _node_prefix_identity correctly distinguishes them (different full prefix)
"""
import json, os, sys

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode"
C = 1024

def tokens(base, n=C):
    return [base + i for i in range(n + 10)]

def fork_tokens(shared_first, fork_second, n=C):
    """Create tokens that share first token but diverge after."""
    ids = [shared_first]
    ids.extend([fork_second + i for i in range(n + 9)])
    return ids

def row(ids, length, ts, label=""):
    return {"input_ids": ids[:length], "input_length": length,
            "output_length": 1, "timestamp": ts, "session_label": label}

def gen():
    rows = []
    ts = 0.0
    hot = [tokens(100000 * (i+1)) for i in range(8)]        # 100001-800001
    cold = [tokens(100000 * (i+10)) for i in range(8)]      # 1000001-1800001
    churn = [tokens(100000 * (i+20)) for i in range(12)]    # 2000001-3100001 (12 only)
    # Fork: F0 shares first token (100000) with S0, F1 shares with S1 (200000)
    fork0 = fork_tokens(100000, 999000)   # [100000, 999000, 999001, ...]
    fork1 = fork_tokens(200000, 888000)   # [200000, 888000, 888001, ...]

    # 1. Build hot (8)
    for i, t in enumerate(hot):
        rows.append(row(t, C, ts, f"build_hot_S{i}")); ts += 10.0
    # 2. Re-access hot (frequency becomes 1)
    for i, t in enumerate(hot):
        rows.append(row(t, C+1, ts, f"reaccess_hot_S{i}")); ts += 10.0
    # 3. Build cold (8)
    for i, t in enumerate(cold):
        rows.append(row(t, C, ts, f"build_cold_S{i}")); ts += 10.0
    # 4. Build fork (2) — shares first token with S0/S1
    rows.append(row(fork0, C, ts, "build_fork_F0")); ts += 10.0
    rows.append(row(fork1, C, ts, "build_fork_F1")); ts += 10.0
    # 5. Churn (12) — capacity pressure
    for i, t in enumerate(churn):
        rows.append(row(t, C, ts, f"churn_C{i}")); ts += 10.0
    # 6. Reuse hot (should hit with Hotness/Oracle)
    for i, t in enumerate(hot):
        rows.append(row(t, C+1, ts, f"reuse_hot_S{i}")); ts += 10.0
    # 7. Reuse cold (should miss with Native, partial with Oracle)
    for i, t in enumerate(cold):
        rows.append(row(t, C+1, ts, f"reuse_cold_S{i}")); ts += 10.0
    # 8. Reuse fork (should NOT be merged with S0/S1)
    rows.append(row(fork0, C+1, ts, "reuse_fork_F0")); ts += 10.0
    rows.append(row(fork1, C+1, ts, "reuse_fork_F1")); ts += 10.0
    return rows

if __name__ == "__main__":
    rows = gen()
    path = os.path.join(OUTDIR, "k3_p2a_trace.jsonl")
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    total = sum(r["input_length"] for r in rows)
    # Verify first tokens: S0 and F0 share 100001
    s0_ft = rows[0]["input_ids"][0]
    f0_idx = 8 + 8 + 8  # build_fork_F0 is after 8 hot + 8 reaccess + 8 cold = 24th (0-indexed)
    f0_ft = rows[f0_idx]["input_ids"][0]
    s0_st = rows[0]["input_ids"][1]
    f0_st = rows[f0_idx]["input_ids"][1]
    print(f"[P2A-trace-v4] {path}: {len(rows)} reqs, {total} tokens")
    print(f"  S0: first={s0_ft} second={s0_st}")
    print(f"  F0: first={f0_ft} second={f0_st}")
    print(f"  Same first token: {s0_ft == f0_ft} (expected True)")
    print(f"  Different second: {s0_st != f0_st} (expected True)")
    assert s0_ft == f0_ft, "Fork test: S0 and F0 must share first token"
    assert s0_st != f0_st, "Fork test: S0 and F0 must have different second token"
    # Count unique first 1024-token hashes
    import hashlib
    identities = set()
    for r in rows:
        ids = r["input_ids"][:1024]
        ident = hashlib.md5(str(ids).encode()).hexdigest()[:16]
        identities.add(ident)
    # Expected: 8 hot + 8 cold + 2 fork + 12 churn = 30 unique sessions
    # (build/re-access/reuse of same session share the same prefix identity)
    print(f"  Unique prefix identities: {len(identities)} (expected 30)")
    assert len(identities) == 30, f"Identity count mismatch! {len(identities)} != 30"
