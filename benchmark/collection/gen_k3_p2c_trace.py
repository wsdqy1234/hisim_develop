#!/usr/bin/env python3
"""Phase 2C mixed trace: 8K/32K/64K sessions + Fork + hotspot migration.

Structure:
1. Build shared 8K prefix (1)
2. Build 4 short branches 8K+4K=12K (4)
3. Build 2 long branches 8K+8K=16K (2)
4. Re-access shared + short[0-1] + long[0] (4) — makes them hot
5. Build 4 cold 4K + 4 cold 8K sessions (8)
6. Churn 4K (8) — capacity pressure
7. Reuse shared + short + long + cold (15)
8. Hotspot migration: re-access shared (1), switch to short[2] (1)
9. Extra re-access shared (3), long[0] (1)
10. Churn reuse (8)

Total: 1+4+2+4+8+8+15+1+1+3+1+8 = 56

Tests:
- Mixed checkpoint depths (4K, 8K, 12K, 16K)
- Fork structure (shared prefix + branches)
- Different replay costs (marginal vs full)
- Hotspot migration (shared→short[2])
"""
import json, os, sys, hashlib

def tokens(base, n):
    return [base + i for i in range(n + 10)]

def row(ids, length, ts, label=""):
    return {"input_ids": ids[:length], "input_length": length,
            "output_length": 1, "timestamp": ts, "session_label": label}

def gen(outdir):
    rows = []
    ts = 0.0
    base = 100000

    SHARED = 8192   # 8K shared prefix
    SHORT_SUF = 4096  # 4K branch suffix → 12K total
    LONG_SUF = 8192   # 8K branch suffix → 16K total

    shared = tokens(base, SHARED)

    # Short branches: shared + 4K unique suffix
    shorts = []
    for i in range(4):
        suf = [base + 1000000*(i+1) + j for j in range(SHORT_SUF + 10)]
        shorts.append(shared[:SHARED] + suf[:SHORT_SUF])

    # Long branches: shared + 8K unique suffix
    longs = []
    for i in range(2):
        suf = [base + 2000000*(i+1) + j for j in range(LONG_SUF + 10)]
        longs.append(shared[:SHARED] + suf[:LONG_SUF])

    # Cold sessions: 4K and 8K
    cold_4k = [tokens(base + 3000000*(i+1), 4096) for i in range(4)]
    cold_8k = [tokens(base + 4000000*(i+1), 8192) for i in range(4)]

    # Churn: 4K sessions
    churn = [tokens(base + 5000000*(i+1), 4096) for i in range(8)]

    # 1. Build shared (1)
    rows.append(row(shared, SHARED, ts, "build_shared")); ts += 10.0
    # 2. Build short branches (4)
    for i, b in enumerate(shorts):
        rows.append(row(b, SHARED+SHORT_SUF, ts, f"build_short_{i}")); ts += 10.0
    # 3. Build long branches (2)
    for i, b in enumerate(longs):
        rows.append(row(b, SHARED+LONG_SUF, ts, f"build_long_{i}")); ts += 10.0
    # 4. Re-access hot (4): shared + short[0-1] + long[0]
    rows.append(row(shared, SHARED+1, ts, "reaccess_shared")); ts += 10.0
    for i in range(2):
        rows.append(row(shorts[i], SHARED+SHORT_SUF+1, ts, f"reaccess_short_{i}")); ts += 10.0
    rows.append(row(longs[0], SHARED+LONG_SUF+1, ts, "reaccess_long_0")); ts += 10.0
    # 5. Build cold (8): 4×4K + 4×8K
    for i, t in enumerate(cold_4k):
        rows.append(row(t, 4096, ts, f"build_cold4k_{i}")); ts += 10.0
    for i, t in enumerate(cold_8k):
        rows.append(row(t, 8192, ts, f"build_cold8k_{i}")); ts += 10.0
    # 6. Churn (8)
    for i, t in enumerate(churn):
        rows.append(row(t, 4096, ts, f"churn_{i}")); ts += 10.0
    # 7. Reuse (15): shared + 4 short + 2 long + 4 cold4k + 4 cold8k
    rows.append(row(shared, SHARED+1, ts, "reuse_shared")); ts += 10.0
    for i, b in enumerate(shorts):
        rows.append(row(b, SHARED+SHORT_SUF+1, ts, f"reuse_short_{i}")); ts += 10.0
    for i, b in enumerate(longs):
        rows.append(row(b, SHARED+LONG_SUF+1, ts, f"reuse_long_{i}")); ts += 10.0
    for i, t in enumerate(cold_4k):
        rows.append(row(t, 4097, ts, f"reuse_cold4k_{i}")); ts += 10.0
    for i, t in enumerate(cold_8k):
        rows.append(row(t, 8193, ts, f"reuse_cold8k_{i}")); ts += 10.0
    # 8. Hotspot migration (2)
    rows.append(row(shared, SHARED+1, ts, "mig_reaccess_shared")); ts += 10.0
    rows.append(row(shorts[2], SHARED+SHORT_SUF+1, ts, "mig_reaccess_short2")); ts += 10.0
    # 9. Extra re-access (4)
    for i in range(3):
        rows.append(row(shared, SHARED+1, ts, f"extra_shared_{i}")); ts += 10.0
    rows.append(row(longs[0], SHARED+LONG_SUF+1, ts, "extra_long0")); ts += 10.0
    # 10. Churn reuse (8)
    for i, t in enumerate(churn):
        rows.append(row(t, 4097, ts, f"reuse_churn_{i}")); ts += 10.0

    # Trim/pad to 56
    while len(rows) > 56: rows.pop()
    while len(rows) < 56:
        rows.append(row(shared, SHARED+1, ts, f"pad_{len(rows)}")); ts += 10.0

    path = os.path.join(outdir, "k3_p2c_mixed.jsonl")
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    # Manifest
    labels = [r["session_label"] for r in rows]
    depths = {}
    for r in rows:
        l = r["session_label"]
        if "shared" in l: d = r["input_length"]
        elif "short" in l: d = r["input_length"]
        elif "long" in l: d = r["input_length"]
        elif "cold4k" in l: d = 4096
        elif "cold8k" in l: d = 8192
        elif "churn" in l: d = 4096
        else: d = r["input_length"]
        depths[d] = depths.get(d, 0) + 1
    identities = set()
    for r in rows:
        ids = r["input_ids"][:r["input_length"]]
        identities.add(hashlib.md5(str(ids).encode()).hexdigest()[:16])
    print(f"[P2C-Mixed] {path}: {len(rows)} reqs")
    print(f"  Depth distribution: {dict(sorted(depths.items()))}")
    print(f"  Unique identities: {len(identities)}")
    return path

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode"
    gen(outdir)
