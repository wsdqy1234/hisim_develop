"""Diagnostic: verify checkpoint identity consistency between
record_access (match_prefix) and select_victims (eviction).

Run inside docker container lfz-sglang with hisim installed.
Usage:
    PYTHONPATH=/home/d00954154/hisim/src python3 -m benchmark.diag_identity
"""
import json
import hashlib
import sys
import os


def trace_identity(ids, length):
    """Identity as computed by Oracle load_future_map."""
    prefix = [int(t) for t in ids[:length]]
    return hashlib.md5(str(prefix).encode()).hexdigest()[:16]


def test_array_vs_list():
    """Test that array('q') and list produce the same str() for MD5."""
    from array import array
    a = array("q", [1, 2, 3, 4, 5])
    l = [1, 2, 3, 4, 5]
    # list(array) -> list of Python ints
    la = list(a)
    print(f"array type: {type(a)}")
    print(f"list(array) type: {type(la)}, elem type: {type(la[0])}")
    print(f"str(list): {str(l)}")
    print(f"str(list(array)): {str(la)}")
    print(f"Match: {str(l) == str(la)}")
    # MD5 comparison
    h1 = hashlib.md5(str(l).encode()).hexdigest()[:16]
    h2 = hashlib.md5(str(la).encode()).hexdigest()[:16]
    print(f"MD5 list: {h1}")
    print(f"MD5 list(array): {h2}")
    print(f"MD5 match: {h1 == h2}")
    # What if we DON'T convert to list? (old behavior)
    h3 = hashlib.md5(str(a).encode()).hexdigest()[:16]
    print(f"MD5 str(array): {h3}")
    print(f"MD5 str(array) == MD5 str(list): {h3 == h1}")


def test_radixkey_identity():
    """Test that RadixKey.raw_token_ids() produces consistent identity."""
    try:
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from array import array
        # Non-bigram key
        k1 = RadixKey(array("q", [1, 2, 3, 4, 5]))
        raw1 = k1.raw_token_ids()
        print(f"\nRadixKey non-bigram:")
        print(f"  token_ids type: {type(k1.token_ids)}")
        print(f"  raw_token_ids type: {type(raw1)}")
        print(f"  len(key)={len(k1)}, len(token_ids)={len(k1.token_ids)}, len(raw)={len(raw1)}")
        # Identity from raw_token_ids
        ids_raw = [int(t) for t in raw1]
        h_raw = hashlib.md5(str(ids_raw).encode()).hexdigest()[:16]
        # Identity from token_ids directly
        ids_direct = list(k1.token_ids)
        h_direct = hashlib.md5(str(ids_direct).encode()).hexdigest()[:16]
        print(f"  MD5(raw_token_ids): {h_raw}")
        print(f"  MD5(token_ids): {h_direct}")
        print(f"  Match: {h_raw == h_direct}")

        # Key with limit
        k2 = RadixKey(array("q", [1, 2, 3, 4, 5]), limit=3)
        raw2 = k2.raw_token_ids()
        print(f"\nRadixKey with limit=3:")
        print(f"  len(key)={len(k2)}, len(token_ids)={len(k2.token_ids)}, len(raw)={len(raw2)}")
        ids_raw2 = [int(t) for t in raw2]
        ids_direct2 = list(k2.token_ids)
        print(f"  raw tokens: {ids_raw2}")
        print(f"  direct tokens: {ids_direct2}")
        h_raw2 = hashlib.md5(str(ids_raw2).encode()).hexdigest()[:16]
        h_direct2 = hashlib.md5(str(ids_direct2).encode()).hexdigest()[:16]
        print(f"  MD5(raw): {h_raw2}, MD5(direct): {h_direct2}")
        print(f"  Match: {h_raw2 == h_direct2}")

        # Bigram key
        k3 = RadixKey(array("q", [1, 2, 3, 4, 5]), is_bigram=True)
        raw3 = k3.raw_token_ids()
        print(f"\nRadixKey bigram:")
        print(f"  len(key)={len(k3)}, len(token_ids)={len(k3.token_ids)}, len(raw)={len(raw3)}")
        print(f"  raw tokens: {[int(t) for t in raw3]}")
        print(f"  iter yields: {list(k3)[:3]}...")

    except ImportError as e:
        print(f"Cannot import RadixKey: {e}")


def test_trace_identity_match():
    """Test that trace file identity matches what _node_prefix_identity would produce."""
    trace_path = os.environ.get("HISIM_ORACLE_TRACE", "")
    if not trace_path or not os.path.exists(trace_path):
        print(f"\nNo trace file found at {trace_path}")
        return
    print(f"\n=== Trace identity test: {trace_path} ===")
    with open(trace_path) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            req = json.loads(line)
            ids = req.get("input_ids", [])
            ilen = req.get("input_length", len(ids))
            label = req.get("session_label", "")
            hash_len = ilen - 1 if ("reaccess" in label or "reuse" in label) else ilen
            hash_len = min(hash_len, len(ids))
            # Current Oracle approach
            prefix_raw = ids[:hash_len]
            h_current = hashlib.md5(str(prefix_raw).encode()).hexdigest()[:16]
            # Fixed approach: convert to int
            prefix_int = [int(t) for t in ids[:hash_len]]
            h_fixed = hashlib.md5(str(prefix_int).encode()).hexdigest()[:16]
            print(f"  req {i}: label={label} ilen={ilen} hash_len={hash_len} "
                  f"n_ids={len(ids)} h_current={h_current} h_fixed={h_fixed} "
                  f"match={h_current == h_fixed}")
            if i == 0:
                print(f"    ids[:5]={ids[:5]} type={type(ids[0])}")


if __name__ == "__main__":
    print("=== Test 1: array vs list str() consistency ===")
    test_array_vs_list()
    print("\n=== Test 2: RadixKey identity ===")
    test_radixkey_identity()
    print("\n=== Test 3: Trace file identity ===")
    test_trace_identity_match()
