#!/usr/bin/env python3
"""Phase 2B trace generators: Long context, Fork prefix, Hotspot migration.

Usage:
  python gen_k3_p2b_trace.py <type> <outdir> [options]

Types:
  b1-8k    — Long context 8K tokens per session
  b1-32k   — Long context 32K tokens per session
  b1-128k  — Long context 128K tokens per session
  b2-fork  — Fork prefix with shared 32K history
  b3-mig   — Hotspot migration A→B→C
"""
import json, os, sys, hashlib

def tokens(base, n):
    return [base + i for i in range(n + 10)]

def row(ids, length, ts, label=""):
    return {"input_ids": ids[:length], "input_length": length,
            "output_length": 1, "timestamp": ts, "session_label": label}

def gen_b1_long_context(token_count, outdir):
    """B1: Long context with hot/cold/churn/reuse pattern.
    
    Structure: 8 hot (build) + 8 hot (re-access) + 8 cold (build) + 16 churn
             + 8 reuse hot + 8 reuse cold = 56 requests
    Each session has `token_count` tokens. Replay cost per miss = token_count.
    """
    C = token_count
    rows = []
    ts = 0.0
    base_hot = 100000
    base_cold = 1000000
    base_churn = 2000000
    hot = [tokens(base_hot * (i+1), C) for i in range(8)]
    cold = [tokens(base_cold * (i+1), C) for i in range(8)]
    churn = [tokens(base_churn * (i+1), C) for i in range(16)]
    
    # 1. Build hot (8)
    for i, t in enumerate(hot):
        rows.append(row(t, C, ts, f"build_hot_S{i}")); ts += 10.0
    # 2. Re-access hot (frequency=1)
    for i, t in enumerate(hot):
        rows.append(row(t, C+1, ts, f"reaccess_hot_S{i}")); ts += 10.0
    # 3. Build cold (8)
    for i, t in enumerate(cold):
        rows.append(row(t, C, ts, f"build_cold_S{i}")); ts += 10.0
    # 4. Churn (16) — capacity pressure
    for i, t in enumerate(churn):
        rows.append(row(t, C, ts, f"churn_C{i}")); ts += 10.0
    # 5. Reuse hot (should hit with Hotness/Oracle)
    for i, t in enumerate(hot):
        rows.append(row(t, C+1, ts, f"reuse_hot_S{i}")); ts += 10.0
    # 6. Reuse cold (should miss with Native, partial with Oracle)
    for i, t in enumerate(cold):
        rows.append(row(t, C+1, ts, f"reuse_cold_S{i}")); ts += 10.0
    
    path = os.path.join(outdir, f"k3_p2b_b1_{C}.jsonl")
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    _verify_manifest(rows, path, f"B1-{C//1024}K")
    return path

def gen_b2_fork(outdir):
    """B2: Fork prefix with shared 32K history, multi-level fork.
    
    Structure:
    - Shared root: 32K tokens (common prefix for all branches)
    - L1 branches: 4 branches diverging at position 32K (each +8K unique)
    - L2 sub-branches: 2 branches diverging from L1[0] at 36K (each +4K)
    - Re-access makes shared root and L1[0-1] "hot"
    - Churn creates capacity pressure
    - Reuse tests preservation of shared vs branch-specific checkpoints
    
    Total: 56 requests
    """
    SHARED = 32768
    L1_SUFFIX = 8192
    L2_FORK_POINT = 36864  # 32K + 4K into L1[0]
    L2_SUFFIX = 4096
    C_L1 = SHARED + L1_SUFFIX  # 40K
    
    rows = []
    ts = 0.0
    base = 100000
    
    shared_tokens = [base + i for i in range(SHARED + 10)]
    
    # L1 branches: shared prefix + unique suffix
    l1_branches = []
    for i in range(4):
        suffix = [base + 1000000 * (i+1) + j for j in range(L1_SUFFIX + 10)]
        b = shared_tokens[:SHARED] + suffix[:L1_SUFFIX]
        l1_branches.append(b)
    
    # L2 sub-branches: L1[0][:36K] + unique suffix
    l2_branches = []
    for i in range(2):
        suffix2 = [base + 2000000 * (i+1) + j for j in range(L2_SUFFIX + 10)]
        sb = l1_branches[0][:L2_FORK_POINT] + suffix2[:L2_SUFFIX]
        l2_branches.append(sb)
    
    churn = [[base + 3000000 * (i+1) + j for j in range(C_L1 + 10)] for i in range(16)]
    
    # 1. Build shared root (1)
    rows.append(row(shared_tokens, SHARED, ts, "build_shared")); ts += 10.0
    # 2. Build L1 branches (4)
    for i, b in enumerate(l1_branches):
        rows.append(row(b, C_L1, ts, f"build_L1_{i}")); ts += 10.0
    # 3. Re-access shared + L1[0-1] (3) — makes them hot
    rows.append(row(shared_tokens, SHARED+1, ts, "reaccess_shared")); ts += 10.0
    for i in range(2):
        rows.append(row(l1_branches[i], C_L1+1, ts, f"reaccess_L1_{i}")); ts += 10.0
    # 4. Build L2 sub-branches (2)
    for i, sb in enumerate(l2_branches):
        rows.append(row(sb, L2_FORK_POINT + L2_SUFFIX, ts, f"build_L2_{i}")); ts += 10.0
    # 5. Churn (16) — capacity pressure
    for i, t in enumerate(churn):
        rows.append(row(t, C_L1, ts, f"churn_C{i}")); ts += 10.0
    # 6. Reuse shared root (1) — should be preserved by Hotness/Oracle
    rows.append(row(shared_tokens, SHARED+1, ts, "reuse_shared")); ts += 10.0
    # 7. Reuse L1 branches (4) — hot[0-1] hit, cold[2-3] miss
    for i, b in enumerate(l1_branches):
        rows.append(row(b, C_L1+1, ts, f"reuse_L1_{i}")); ts += 10.0
    # 8. Reuse L2 sub-branches (2) — should miss (evicted)
    for i, sb in enumerate(l2_branches):
        rows.append(row(sb, L2_FORK_POINT+L2_SUFFIX+1, ts, f"reuse_L2_{i}")); ts += 10.0
    # 9. Re-access shared root again (1) — test adaptation
    rows.append(row(shared_tokens, SHARED+1, ts, "reaccess_shared_2")); ts += 10.0
    # 10. Re-access hot L1[0-1] (2) — test adaptation
    for i in range(2):
        rows.append(row(l1_branches[i], C_L1+1, ts, f"reaccess_L1_{i}_2")); ts += 10.0
    # 11. Re-access L2[0] (1) — test if fork checkpoint preserved
    rows.append(row(l2_branches[0], L2_FORK_POINT+L2_SUFFIX+1, ts, "reaccess_L2_0")); ts += 10.0
    # 12. Churn reuse (16) — reuse all churn sessions
    for i in range(16):
        rows.append(row(churn[i], C_L1+1, ts, f"reuse_churn_{i}")); ts += 10.0
    # 13. Extra re-access shared (3) — makes shared very hot
    for i in range(3):
        rows.append(row(shared_tokens, SHARED+1, ts, f"reaccess_shared_r{i}")); ts += 10.0
    
    # Trim or pad to exactly 56
    while len(rows) > 56:
        rows.pop()
    while len(rows) < 56:
        rows.append(row(shared_tokens, SHARED+1, ts, f"pad_{len(rows)}")); ts += 10.0
    
    path = os.path.join(outdir, "k3_p2b_b2_fork.jsonl")
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    _verify_manifest(rows, path, "B2-Fork")
    return path

def gen_b3_migration(outdir):
    """B3: Hotspot migration A→B→C.
    
    Phase A: Session A0-A7 built, A0-A3 re-accessed 2x (hot)
    Phase B: Session B0-B7 built, B0-B3 re-accessed 2x (hot)
             (A sessions become "cold" — still in L1 but not recently accessed)
    Phase C: Re-access A0-A7 and B0-B7 (test if Hotness adapted)
    
    Total: 56 requests
    A build=8 + A re-access=8 + B build=8 + B re-access=8 + churn=8 + C reuse A=8 + C reuse B=8 = 56
    """
    C = 1024
    rows = []
    ts = 0.0
    A = [tokens(100000 * (i+1), C) for i in range(8)]
    B = [tokens(200000 * (i+1), C) for i in range(8)]
    churn = [tokens(300000 * (i+1), C) for i in range(8)]
    
    # Phase A: Build A0-A7 (8)
    for i, t in enumerate(A):
        rows.append(row(t, C, ts, f"phaseA_build_A{i}")); ts += 10.0
    # Phase A: Re-access A0-A3 × 2 (8) — makes A0-A3 "hot"
    for round_i in range(2):
        for i in range(4):
            rows.append(row(A[i], C+1, ts, f"phaseA_reaccess_A{i}_r{round_i}")); ts += 10.0
    
    # Phase B: Build B0-B7 (8)
    for i, t in enumerate(B):
        rows.append(row(t, C, ts, f"phaseB_build_B{i}")); ts += 10.0
    # Phase B: Re-access B0-B3 × 2 (8) — makes B0-B3 "hot", A sessions "cold"
    for round_i in range(2):
        for i in range(4):
            rows.append(row(B[i], C+1, ts, f"phaseB_reaccess_B{i}_r{round_i}")); ts += 10.0
    
    # Churn (8) — capacity pressure
    for i in range(8):
        rows.append(row(churn[i], C, ts, f"churn_C{i}")); ts += 10.0
    
    # Phase C: Re-access A0-A7 (8) — test if Hotness preserved A0-A3
    for i, t in enumerate(A):
        rows.append(row(t, C+1, ts, f"phaseC_reuse_A{i}")); ts += 10.0
    # Phase C: Re-access B0-B7 (8) — test if Hotness preserved B0-B3
    for i, t in enumerate(B):
        rows.append(row(t, C+1, ts, f"phaseC_reuse_B{i}")); ts += 10.0
    
    path = os.path.join(outdir, "k3_p2b_b3_migration.jsonl")
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    _verify_manifest(rows, path, "B3-Migration")
    return path

def _verify_manifest(rows, path, label):
    """Verify trace manifest: unique requests, phases, access counts."""
    labels = [r["session_label"] for r in rows]
    unique_labels = set(labels)
    # Count by phase
    phases = {}
    for l in labels:
        phase = l.split("_")[0]  # e.g., "build", "reaccess", "reuse", "churn"
        phases[phase] = phases.get(phase, 0) + 1
    # Count unique sessions (by first token)
    sessions = set()
    for r in rows:
        ft = r["input_ids"][0]
        sessions.add(ft)
    total_tokens = sum(r["input_length"] for r in rows)
    print(f"[P2B-{label}] {path}")
    print(f"  Requests: {len(rows)} (unique labels: {len(unique_labels)})")
    print(f"  Unique sessions (by first token): {len(sessions)}")
    print(f"  Phases: {dict(sorted(phases.items()))}")
    print(f"  Total tokens: {total_tokens}")
    # Verify prefix identities
    identities = set()
    for r in rows:
        ids = r["input_ids"][:1024]
        ident = hashlib.md5(str(ids).encode()).hexdigest()[:16]
        identities.add(ident)
    print(f"  Unique prefix identities (1024): {len(identities)}")

if __name__ == "__main__":
    trace_type = sys.argv[1] if len(sys.argv) > 1 else "b1-8k"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode"
    
    if trace_type == "b1-8k":
        gen_b1_long_context(8192, outdir)
    elif trace_type == "b1-32k":
        gen_b1_long_context(32768, outdir)
    elif trace_type == "b1-128k":
        gen_b1_long_context(131072, outdir)
    elif trace_type == "b2-fork":
        gen_b2_fork(outdir)
    elif trace_type == "b3-mig":
        gen_b3_migration(outdir)
    elif trace_type == "all":
        gen_b1_long_context(8192, outdir)
        gen_b1_long_context(32768, outdir)
        gen_b2_fork(outdir)
        gen_b3_migration(outdir)
        # Skip 128K for now (very large, slow to simulate)
        print("  (skipping 128K — too slow for pilot)")
    else:
        print(f"Unknown type: {trace_type}")
        sys.exit(1)
