#!/usr/bin/env python3
"""HiSim 3-tier cache analyzer v3 — HiSim-aware radix tree model.

V3 extends v2 (sglang radix tree) with HiSim-specific KV reuse logic:
- write_through backup: backup on first hit + lock until ACK
- load_back: re-materialize evicted chain from L2 to L1
- Contiguous prefix invariant: parent must be backed up before child
- 16384-token chunked insertion (HiSim chunked_prefill_size)
- Phase 3 re-materialization (churn cycle: evict → match → insert re-materialize)
- Shallow-first walk in Phase 3 (parent processed before child)
- Output leaf flow: insert output tokens (all 1s, like HiSim mock)
- L3 conditional on backup success (hashes written during ACK)
- L3 only queried through prefetch_from_storage (not direct hash lookup)
- Prefetch buffer: L2 allocated for the full remaining input at receipt
  (evicting host LRU leaves if short — the evictions stick), but released
  when the request is scheduled — BEFORE its own cache_finished_req backups,
  so it never overlaps Phase 3 backup pressure (verified against the hisim
  sglang simulation: per-request L1/L2 match exactly)
- Delayed L3 hash write: written during ACK (write_backup_storage)
"""

import json, time, argparse
from typing import List, Optional, Tuple

class TreeNode:
    __slots__ = ['key', 'value', 'host_value', 'children', 'parent',
                 'lock_ref', 'last_access', 'hit_count', 'hash_vals']
    def __init__(self, key=None, parent=None):
        self.key = list(key) if key else []
        self.value = None       # True=in L1, None=evicted
        self.host_value = None  # int=tokens in L2, None=not backed up
        self.children = {}
        self.parent = parent
        self.lock_ref = 0
        self.last_access = 0.0
        self.hit_count = 0
        self.hash_vals = []     # cumulative page hashes for L3
    @property
    def evicted(self): return self.value is None
    @property
    def backuped(self): return self.host_value is not None
    @property
    def key_len(self): return len(self.key)
    def is_evictable_leaf(self):
        # HiSim _update_leaf_status: not evicted, no lock, all children evicted
        if self.evicted or self.lock_ref > 0: return False
        return all(c.evicted for c in self.children.values())
    def is_evictable_host_leaf(self):
        # HiSim _update_host_leaf_status: evicted, backuped, no lock,
        # and NO child is evicted (deepest evicted node in branch)
        if not self.evicted or not self.backuped or self.lock_ref > 0: return False
        return not any(c.evicted for c in self.children.values())

class RadixTreeCache:
    def __init__(self, l1_cap, l2_cap, page_size=1):
        self.l1_cap = l1_cap
        self.l2_cap = l2_cap
        self.page_size = page_size
        self.root = TreeNode(parent=None)
        self.l1_used = 0
        self.l2_used = 0
        self.l3 = set()
        self._tc = 0
        self.pending_acks = []   # nodes backed up, waiting ACK (write_through lock)
        self.pending_l3 = {}    # node_id → (prev_hash, hash_vals) for delayed L3 write
    def _t(self):
        self._tc += 1; return float(self._tc)
    def _hashes(self, tokens, start=None):
        h, out = start, []
        for i in range(0, len(tokens), self.page_size):
            block = tuple(tokens[i:i+self.page_size])
            h = str(hash((h, block)))
            out.append(h)
        return out
    def _match(self, tokens):
        """Walk tree. Returns (l1_tokens, last_node, host_hit, remaining_tokens)."""
        node = self.root
        node.last_access = self._t()
        l1 = 0; hh = 0
        rem = list(tokens)
        while rem:
            fk = rem[0]
            if fk not in node.children: break
            child = node.children[fk]
            child.last_access = self._t()
            kl = child.key_len
            if rem[:kl] == child.key:
                if not child.evicted: l1 += kl
                node = child; rem = rem[kl:]
            else:
                ml = 0
                for a, b in zip(child.key, rem):
                    if a == b: ml += 1
                    else: break
                if ml == 0: break
                np_ = self._split(child, ml)
                np_.last_access = self._t()
                if not np_.evicted: l1 += ml
                node = np_; rem = rem[ml:]
                break
        # host_hit: walk up while evicted (contiguous evicted suffix)
        n = node
        while n != self.root and n.evicted:
            hh += n.key_len; n = n.parent
        return l1, node, hh, rem
    def _split(self, child, sl):
        orig_first = child.key[0]
        np_ = TreeNode(key=child.key[:sl], parent=child.parent)
        child.key = child.key[sl:]
        child.parent = np_
        np_.children = {child.key[0]: child}
        np_.parent.children[orig_first] = np_
        np_.lock_ref = child.lock_ref
        np_.hit_count = child.hit_count
        np_.last_access = child.last_access
        if child.evicted: np_.value = None
        else: np_.value = True; child.value = True
        if child.backuped:
            np_.host_value = sl; child.host_value = child.key_len
        np_.hash_vals = child.hash_vals[:sl] if child.hash_vals else []
        if child.hash_vals: child.hash_vals = child.hash_vals[sl:]
        return np_
    def _write_backup(self, node):
        """Backup node to L2 (HiSim write_backup with write_through).
        - Contiguous prefix invariant: skip if parent not backuped (L614-617)
        - Locks node (inc_lock_ref) until ACK.
        - L3 hashes written during ACK (write_backup_storage), NOT here."""
        if node.parent != self.root and not node.parent.backuped:
            return False
        if not self._evict_l2(node.key_len):
            return False
        node.host_value = node.key_len
        self.l2_used += node.key_len
        node.lock_ref += 1  # write_through: lock until ACK
        self.pending_acks.append(node)
        # Save hash info for delayed L3 write during ACK
        prev = node.parent.hash_vals[-1] if (node.parent and node.parent.hash_vals) else None
        self.pending_l3[id(node)] = (prev, list(node.hash_vals))
        return True
    def _evict_l1(self, need):
        """L1 eviction: trigger when l1_used + need > cap (includes locked nodes).
        - backuped nodes: value=None (evicted, stay in tree) 
        - non-backuped nodes: _del (deleted from tree, like _evict_regular)
        Skips locked nodes (lock_ref > 0) — they can't be evicted."""
        while self.l1_used + need > self.l1_cap:
            leaves = []; self._collect_leaves(self.root, leaves)
            if not leaves: return False
            leaves.sort(key=lambda n: n.last_access)
            evicted = False
            for v in leaves:
                if self.l1_used + need <= self.l1_cap:
                    return True
                if v.lock_ref > 0 or v.evicted: continue
                self.l1_used -= v.key_len
                if v.backuped:
                    v.value = None
                else:
                    self._del(v)
                evicted = True
            if not evicted: return False
        return True
    def _collect_leaves(self, node, leaves):
        if node is not self.root:
            if node.is_evictable_leaf(): leaves.append(node); return
        for c in node.children.values(): self._collect_leaves(c, leaves)
    def _evict_l2(self, need):
        """L2 eviction: remove host_value + remove from parent's children.
        Only evicts host leaves (deepest evicted nodes with no evicted children)."""
        while self.l2_used + need > self.l2_cap:
            leaves = []; self._collect_host_leaves(self.root, leaves)
            if not leaves: return False
            leaves.sort(key=lambda x: x.last_access)
            v = leaves[0]
            self.l2_used -= v.host_value; v.host_value = None
            for child in list(v.children.values()):
                self._free_l1(child)
            if v.parent:
                k = v.key[0] if v.key else None
                v.parent.children.pop(k, None)
        return True
    def _free_l1(self, node):
        """Free L1 memory of orphaned subtree (children of L2-evicted node)."""
        if node.value is not None:
            self.l1_used -= node.key_len
            node.value = None
        for child in list(node.children.values()):
            self._free_l1(child)
    def _del(self, node):
        """Delete node from tree (HiSim _evict_regular for non-backuped nodes)."""
        for child in list(node.children.values()):
            self._free_l1(child)
        if node.parent:
            k = node.key[0] if node.key else None
            node.parent.children.pop(k, None)
    def _collect_host_leaves(self, node, leaves):
        if node is not self.root and node.is_evictable_host_leaf():
            leaves.append(node); return
        for c in node.children.values(): self._collect_host_leaves(c, leaves)

    def _insert(self, tokens, start_node, chunk_size=16384):
        """Insert tokens in 16384-token chunks. Like HiSim's cache_unfinished_req
        with chunked=True: NO backup during insert (skip _inc_hit_count).
        Backup happens later in Phase 3 (cache_finished_req)."""
        parent = start_node
        remaining = list(tokens)
        while remaining:
            chunk = remaining[:chunk_size]
            remaining = remaining[chunk_size:]
            key = list(chunk)
            # Try to match existing prefix
            while key:
                fk = key[0]
                if fk in parent.children:
                    child = parent.children[fk]
                    ml = 0
                    for a, b in zip(child.key, key):
                        if a == b: ml += 1
                        else: break
                    if ml > 0:
                        child.last_access = self._t()
                        if ml < len(child.key):
                            np_ = self._split(child, ml)
                            np_.last_access = self._t()
                            parent = np_
                        else:
                            parent = child
                        key = key[ml:]
                        if not key:
                            break
                        continue
                # Create new node for remaining key in this chunk
                self._evict_l1(len(key))
                nn = TreeNode(key=key, parent=parent)
                nn.value = True; nn.last_access = self._t()
                nn.lock_ref = 1  # locked (insertion)
                pstart = parent.hash_vals[-1] if parent.hash_vals else None
                nn.hash_vals = self._hashes(key, start=pstart)
                parent.children[key[0]] = nn
                self.l1_used += len(key)
                parent = nn
                break
        return parent

    def _inc_hit_count(self, node):
        """HiSim _inc_hit_count with write_through: backup on first hit (threshold=1).
        Backs up to L2 and locks the node (inc_lock_ref) until ACK."""
        if node.evicted:
            return
        if node.backuped:
            return
        node.hit_count += 1
        if node.hit_count >= 1:
            self._write_backup(node)

    def _load_back(self, last):
        """Re-materialize evicted chain from L2 to L1 (HiSim load_back).
        Loads from top (closest to non-evicted ancestor) down.
        Maintains invariant: path stays "non-evicted prefix + evicted suffix".
        This is the mechanism for USING L2 hits (H2D transfer). Does NOT
        change hit accounting — L2 hits are determined at match time."""
        chain = []
        n = last
        while n != self.root and n.evicted:
            chain.insert(0, n)  # root-to-leaf order
            n = n.parent
        for node in chain:  # top-down loading
            if self._evict_l1(node.key_len):
                node.value = True
                self.l1_used += node.key_len
            else:
                break  # can't make room, rest stays evicted (rare)

    def _process_acks(self):
        """Process ACKs: unlock backup locks AND write L3 hashes.
        In HiSim, ACK = check_hicache_events → writing_check → dec_lock_ref
        + write_backup_storage. L3 hashes are written here, not during backup."""
        for node in self.pending_acks:
            node.lock_ref = max(0, node.lock_ref - 1)
            nid = id(node)
            if nid in self.pending_l3:
                _, hashes = self.pending_l3.pop(nid)
                for h in hashes:
                    self.l3.add(h)
        self.pending_acks = []

    def analyze(self, tokens, rid, output_tokens=None, output_ids=None):
        """Models HiSim's full lifecycle:
        1. match_prefix (L1 hit = non-evicted, L2 hit = evicted suffix)
        2. load_back (re-materialize L2 hits to L1 for use — doesn't change accounting)
        3. L3 hit check (for tokens not in tree, via prefetch_from_storage)
        4. cache_unfinished_req (insert remaining tokens, chunked=True, no backup)
        5. cache_finished_req (insert output, backup path nodes, unlock)
        """
        # Process ACKs from previous request
        self._process_acks()

        l1, last, hh, rem = self._match(tokens)
        # Lock matched path before load_back (prevents eviction of matched nodes)
        ln = last
        while ln != self.root:
            ln.lock_ref += 1; ln = ln.parent
        # load_back: re-materialize evicted nodes from L2 (H2D transfer)
        # L2 hits are already counted in hh at match time. load_back just
        # brings data to L1 for use. Does NOT change l1 or hh.
        if hh > 0:
            self._load_back(last)
        # L3 hit check + Prefetch buffer (prefetch_from_storage)
        # HiSim: L3 query only happens inside prefetch_from_storage. If prefetch
        # is disabled (threshold/rate-limit), no L3 query → L3=0.
        l3 = 0
        pf_cap = int(0.8 * max(0, self.l2_cap - self.l1_cap))
        pf_len = len(rem) if rem else 0
        prefetch_enabled = (rem and len(rem) >= 256 and pf_len <= pf_cap)
        if prefetch_enabled:
            # L3 query (only when prefetch would run)
            start_node = last
            while start_node != self.root and not start_node.backuped:
                start_node = start_node.parent
            prev_str = start_node.hash_vals[-1] if (start_node != self.root and start_node.hash_vals) else None
            edge_prev = prev_str
            for i in range(0, len(rem), self.page_size):
                block = tuple(rem[i:i+self.page_size])
                h = str(hash((edge_prev, block)))
                if h in self.l3:
                    l3 += 1
                    edge_prev = h
                else:
                    break
            # Prefetch buffer: sglang allocates L2 for the full remaining
            # input at request receipt, EVICTING host LRU leaves if the pool
            # is short (evictions are permanent), but the buffer itself is
            # released when the request is scheduled/finishes — BEFORE the
            # request's own cache_finished_req backups. Model exactly that:
            # evict-if-needed for the buffer now, then free it immediately so
            # it never overlaps with Phase 3's backup pressure.
            self._evict_l2(pf_len)
        # Phase 1: Prefill — insert remaining input tokens (cache_unfinished_req)
        parent = last
        if rem:
            parent = self._insert(rem, last)
        # Phase 2: Insert output tokens (cache_finished_req: insert input+output)
        # HiSim's cache_finished_req inserts input+output. Output tokens are
        # mock-generated (all 1s in simulation). They create garbage leaves
        # that get backed up and evicted — contributing L2 capacity tax and
        # host leaf fuel. ~1K output/request × 500 = 512K tokens of garbage.
        if output_tokens and output_tokens > 0:
            out_key = [1] * output_tokens  # mock output = all 1s (like HiSim)
            parent = self._insert(out_key, parent)
        # Phase 3: cache_finished_req — re-materialize evicted ancestors,
        # backup all path nodes (shallow→deep), unlock.
        # Walk shallow→deep (root→parent) so parent is backed up BEFORE child.
        # This ensures contiguous prefix invariant: child checks parent.backuped
        # AFTER parent has been processed (not before).
        path = []
        n = parent if rem else last
        while n != self.root:
            path.append(n)
            n = n.parent
        path.reverse()  # shallow (root-side) first
        for n in path:
            # Re-materialize evicted ancestors (churn cycle, HiSim L1772-1777)
            if n.evicted:
                if self._evict_l1(n.key_len):
                    n.value = True
                    self.l1_used += n.key_len
            # Backup non-backuped nodes (write_through)
            # _write_backup checks parent.backuped — now parent is already
            # processed (shallow first) → correct cascade behavior
            if not n.backuped and not n.evicted:
                self._inc_hit_count(n)
            n.lock_ref = max(0, n.lock_ref - 1)
        return {'l1': l1, 'l2': hh, 'l3': l3, 'new': len(tokens)-l1-hh-l3}

def _load_weka_module():
    """Load the weka_trace loader without requiring the full hisim package."""
    try:
        from hisim.dataset import weka_trace
        return weka_trace
    except Exception:
        import importlib.util
        from pathlib import Path
        p = (Path(__file__).resolve().parent.parent
             / "src" / "hisim" / "dataset" / "weka_trace.py")
        spec = importlib.util.spec_from_file_location("hisim_weka_trace", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default=None,
                   help="Input JSONL (hisim_collection format); optional if --weka-traces is given")
    p.add_argument("--l1-tokens", type=int, default=200000)
    p.add_argument("--l2-gb", type=float, default=80)
    p.add_argument("--kv-bytes-per-token", type=int, default=262144)
    p.add_argument("--page-size", type=int, default=1)
    p.add_argument("--max-requests", type=int, default=0)
    weka_mod = _load_weka_module()
    weka_mod.add_weka_arguments(p)
    args = p.parse_args()
    l2t = int(args.l2_gb * 1e9 / args.kv_bytes_per_token) if args.l2_gb > 0 else 0
    print(f"Radix Tree Cache v3: L1={args.l1_tokens} L2={l2t} page={args.page_size}\n")
    cache = RadixTreeCache(args.l1_tokens, l2t, args.page_size)
    tl=tl2=tl3=tn=ti=0; l2r=0
    if args.weka_traces:
        plan = weka_mod.weka_plan_from_args(args)
        print(f"Weka plan: {plan.num_traces} traces, {len(plan.rounds)} rounds, "
              f"block_size={plan.block_size}\n")
        entries = weka_mod.iter_round_entries(plan)  # time-sorted generator
        total = len(plan.rounds)
    else:
        if not args.input:
            p.error("--input is required unless --weka-traces is given")
        entries = [json.loads(l) for l in open(args.input)]
        entries.sort(key=lambda d: d.get('timestamp', 0))
        total = len(entries)
    mx = args.max_requests or total
    t0=time.time()
    for i, d in enumerate(entries):
        if i >= mx: break
        toks = d.get('input_ids', [])
        out_ids = d.get('output_ids')
        out_len = d.get('output_length', 0)
        ti += d.get("input_length", len(toks))
        r = cache.analyze(toks, i,
                          output_tokens=(None if out_ids else out_len),
                          output_ids=out_ids)
        tl+=r['l1']; tl2+=r['l2']; tl3+=r['l3']; tn+=r['new']
        if r['l2']>0: l2r+=1
        if r['l2']>0 or r['l3']>0:
            tag = (f" tr{d['trace_idx']}.{d.get('stream_id','m')}r{d['round_idx']}"
                   if 'trace_idx' in d else "")
            print(f"  #{i+1}{tag}: inp={len(toks)} L1={r['l1']} L2={r['l2']} L3={r['l3']} New={r['new']}")
        if (i+1)%100==0: print(f"  {i+1} reqs {time.time()-t0:.0f}s", flush=True)
    print(f"\n{'='*60}")
    print(f"Total input: {ti:,}")
    print(f"L1: {tl:,} ({tl/ti*100:.2f}%)")
    print(f"L2: {tl2:,} ({tl2/ti*100:.2f}%)  [{l2r} reqs]")
    print(f"L3: {tl3:,} ({tl3/ti*100:.2f}%)")
    print(f"New:{tn:,} ({tn/ti*100:.2f}%)")

if __name__ == "__main__":
    main()
