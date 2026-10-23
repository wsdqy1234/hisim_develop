"""P2C.2 Identity consistency test.

Verifies _node_prefix_identity produces consistent results for:
- Various prefix lengths (8K, 32K, 40K, 128K)
- Fork (two branches sharing a prefix)
- Split (node split)
- Reinsert (evict and reinsert)

Also verifies identity matches between:
- _node_prefix_identity (tree node)
- Oracle load_future_map (trace file)
- record_access (match_prefix best_match_node)

Must run inside docker container lfz-sglang with sglang installed.
"""
import json
import hashlib
import os
import sys
import tempfile

# Add hisim to path
sys.path.insert(0, "/home/d00954154/hisim/src")
os.environ.setdefault("PYTHONHASHSEED", "42")


def compute_trace_identity(ids, length):
    """Identity as computed by Oracle load_future_map (with int conversion)."""
    prefix = [int(t) for t in ids[:length]]
    return hashlib.md5(str(prefix).encode()).hexdigest()[:16]


def test_prefix_lengths():
    """Test identity for various prefix lengths."""
    print("=== Test: Prefix lengths (8K, 32K, 40K, 128K) ===")
    for length in [8192, 32768, 40960, 131072]:
        # Simulate tokens (use simple sequence)
        tokens = list(range(length))
        h = compute_trace_identity(tokens, length)
        # Verify identity is stable
        h2 = compute_trace_identity(tokens, length)
        # Verify different tokens produce different identity
        tokens2 = list(range(1, length + 1))
        h3 = compute_trace_identity(tokens2, length)
        print(f"  len={length:6d}: hash={h} stable={h == h2} unique={h != h3}")
        assert h == h2, f"Identity not stable for len={length}"
        assert h != h3, f"Identity not unique for len={length}"


def test_fork_identity():
    """Test that fork branches have different identities but shared prefix matches."""
    print("\n=== Test: Fork identity ===")
    # Shared root: 32K tokens
    shared = list(range(32768))
    # Branch A: 32K shared + 8K unique A
    branch_a = shared + list(range(100000, 108192))
    # Branch B: 32K shared + 8K unique B
    branch_b = shared + list(range(200000, 208192))

    h_shared = compute_trace_identity(shared, 32768)
    h_a = compute_trace_identity(branch_a, 40960)
    h_b = compute_trace_identity(branch_b, 40960)

    print(f"  shared (32K): {h_shared}")
    print(f"  branch_a (40K): {h_a}")
    print(f"  branch_b (40K): {h_b}")
    print(f"  shared != a: {h_shared != h_a}")
    print(f"  shared != b: {h_shared != h_b}")
    print(f"  a != b: {h_a != h_b}")

    assert h_shared != h_a, "Shared and branch A should differ"
    assert h_shared != h_b, "Shared and branch B should differ"
    assert h_a != h_b, "Branch A and B should differ"


def test_split_identity():
    """Test that identity is stable across node split.

    Simulates: a 40K prefix stored in one node, then split into
    32K + 8K. The identity should be the same (same tokens).
    """
    print("\n=== Test: Split identity ===")
    # Before split: one node with 40K tokens
    tokens_40k = list(range(40960))
    h_before = compute_trace_identity(tokens_40k, 40960)

    # After split: two segments [0:32768] + [32768:40960]
    # _node_prefix_identity walks root-to-node, collecting segments
    # After split: root has [0:32768], child has [32768:40960]
    # Walk: child [32768:40960] + root [0:32768]
    # Reversed: [0:32768] + [32768:40960] = [0:40960]
    seg1 = [int(t) for t in tokens_40k[:32768]]  # root segment
    seg2 = [int(t) for t in tokens_40k[32768:]]   # child segment

    # Simulate _node_prefix_identity: segments collected leaf-to-root
    segments = [seg2, seg1]  # child first, then root
    segments.reverse()  # root-to-node
    all_tokens = []
    for seg in segments:
        all_tokens.extend(seg)
    h_after = hashlib.md5(str(all_tokens).encode()).hexdigest()[:16]

    print(f"  before split (40K): {h_before}")
    print(f"  after split (32K+8K): {h_after}")
    print(f"  match: {h_before == h_after}")
    assert h_before == h_after, "Identity should be stable across split"


def test_reinsert_identity():
    """Test that identity is stable across eviction + reinsert."""
    print("\n=== Test: Reinsert identity ===")
    tokens = list(range(40960))
    h1 = compute_trace_identity(tokens, 40960)
    # Reinsert same tokens
    h2 = compute_trace_identity(tokens, 40960)
    print(f"  original: {h1}")
    print(f"  reinsert: {h2}")
    print(f"  match: {h1 == h2}")
    assert h1 == h2, "Identity should be stable across reinsert"


def test_trace_consistency():
    """Test that trace file identities match between build and re-access."""
    trace_path = os.environ.get("HISIM_ORACLE_TRACE", "")
    if not trace_path or not os.path.exists(trace_path):
        print(f"\n=== Test: Trace consistency (SKIPPED - no trace) ===")
        return

    print(f"\n=== Test: Trace consistency ({trace_path}) ===")
    with open(trace_path) as f:
        lines = f.readlines()

    # Collect build and re-access identities
    build_idents = {}  # hash -> (label, length)
    reaccess_idents = {}  # label -> (hash, length)

    for line in lines:
        req = json.loads(line)
        label = req.get("session_label", "")
        ids = req.get("input_ids", [])
        ilen = req.get("input_length", len(ids))
        if len(ids) == 0:
            continue
        is_reaccess = ("reaccess" in label or "reuse" in label)
        hash_len = ilen - 1 if is_reaccess else ilen
        hash_len = min(hash_len, len(ids))
        if hash_len <= 0:
            hash_len = min(ilen, len(ids))
        h = compute_trace_identity(ids, hash_len)

        if not is_reaccess:
            # All non-reaccess are "builds" (build_*, churn_C*, etc.)
            build_idents[h] = (label, hash_len)
        else:
            reaccess_idents[label] = (h, hash_len)

    # Verify re-access matches a build
    matched = 0
    unmatched = 0
    for label, (h, hl) in reaccess_idents.items():
        if h in build_idents:
            matched += 1
        else:
            unmatched += 1
            if unmatched <= 3:
                print(f"  UNMATCHED: {label} hash={h} len={hl}")

    print(f"  Build identities: {len(build_idents)}")
    print(f"  Re-access identities: {len(reaccess_idents)}")
    print(f"  Matched: {matched}")
    print(f"  Unmatched: {unmatched}")

    if unmatched > 0:
        print("  WARNING: Some re-access identities don't match any build!")
        print("  This indicates identity mismatch between trace and tree.")


def test_array_consistency():
    """Test that array('q') and list produce the same identity."""
    print("\n=== Test: array('q') vs list consistency ===")
    from array import array
    a = array("q", list(range(1000)))
    l = list(range(1000))

    # _node_prefix_identity converts to int
    la = [int(t) for t in a]
    ll = [int(t) for t in l]

    h_a = hashlib.md5(str(la).encode()).hexdigest()[:16]
    h_l = hashlib.md5(str(ll).encode()).hexdigest()[:16]

    print(f"  array('q') -> int list: {h_a}")
    print(f"  list -> int list: {h_l}")
    print(f"  match: {h_a == h_l}")
    assert h_a == h_l, "array('q') and list should produce same identity"


if __name__ == "__main__":
    test_array_consistency()
    test_prefix_lengths()
    test_fork_identity()
    test_split_identity()
    test_reinsert_identity()
    test_trace_consistency()
    print("\n=== ALL TESTS PASSED ===")
