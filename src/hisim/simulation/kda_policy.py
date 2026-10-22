"""KDA Eviction Policy interface for Phase 2 strategy experiments.

Hook point: MambaComponent._evict_device_start / _evict_device_next_node
  (inside wrapped_init_hicache, via tree_core.components_by_type[MAMBA])

Design:
  - _evict_device_start: after native cursor setup, walk LRU list to collect
    ALL evictable MAMBA leaves, sort by policy, store as _policy_evict_queue.
  - _evict_device_next_node: pop from pre-sorted queue (NO cursor advancement).
    When queue exhausted, fall back to native walk (for tombstone cleanup).
  - No side effects before policy decision: cursor NOT advanced for policy-
    selected nodes. Device/host frees and tracker updates only happen for
    nodes that are actually evicted (via _evict_device_leaf, native, unmodified).

Safety:
  - Same evictable candidate set (from evictable_device_leaves)
  - Same capacity trigger (evict_for_alloc → _evict_components)
  - Same eviction execution (_evict_device_leaf, lock/ACK/allocator/cascade)
  - Does NOT re-enable HiSim Proactive Eviction
  - Does NOT call allocator.free directly
  - Does NOT modify Lock/ACK/Cascade semantics

Checkpoint Identity:
  - Uses (prefix_hash, checkpoint_depth) instead of node_id
  - prefix_hash from node.hash_value (stable across node split/reinsert)
  - checkpoint_depth from len(node.key) (token count at this node)
"""
import os
import time
import hashlib
import logging
from collections import defaultdict, deque
from typing import Optional, List

logger = logging.getLogger(__name__)


def _node_prefix_identity(node) -> str:
    """Stable identity from full root-to-node prefix token sequence.

    Walks root-to-node, collects all key.token_ids segments,
    concatenates, and hashes with MD5. This is:
    - Stable across node split (same prefix → same tokens → same hash)
    - Stable across eviction + reinsert (same prefix → same hash)
    - Different for different branches (even with same first token)
    - Independent of slot generation (logical prefix, not physical slot)
    """
    if node is None:
        return ""
    import hashlib
    segments = []
    cur = node
    while cur is not None:
        key = getattr(cur, 'key', None)
        if key is not None:
            token_ids = getattr(key, 'token_ids', None)
            if token_ids is not None and len(token_ids) > 0:
                segments.insert(0, list(token_ids))
        cur = getattr(cur, 'parent', None)
    if not segments:
        return ""
    all_tokens = []
    for seg in segments:
        all_tokens.extend(seg)
    return hashlib.md5(str(all_tokens).encode()).hexdigest()[:16]


def _node_first_token(node):
    """Get the first token of the full prefix (root-to-node path).

    Walks from node to root, collecting key segments.
    Returns the first token of the first non-empty segment.
    Handles node split (root key may be empty after split).
    Handles RadixKey (has .token_ids attribute, not directly subscriptable).
    Kept for backward compatibility; prefer _node_prefix_identity.
    """
    if node is None:
        return None
    segments = []
    cur = node
    while cur is not None:
        key = getattr(cur, 'key', None)
        if key is not None:
            token_ids = getattr(key, 'token_ids', None)
            if token_ids is not None and len(token_ids) > 0:
                segments.insert(0, token_ids)
            elif hasattr(key, '__len__') and len(key) > 0:
                segments.insert(0, key)
        cur = getattr(cur, 'parent', None)
    if segments:
        try:
            return int(segments[0][0])
        except (ValueError, TypeError, IndexError):
            return None
    return None


def _node_depth(node) -> int:
    """Get checkpoint depth (token count) from node."""
    if node is None:
        return 0
    key = getattr(node, 'key', None)
    if key is not None:
        return len(key)
    return 0


class KDAEvictionPolicy:
    """Base class: delegates to native LRU (no reordering)."""

    def __init__(self):
        self._name = "native"

    @property
    def name(self) -> str:
        return self._name

    def record_access(self, node_id: int, prefix_hash: str, virtual_time: float):
        """Called when a prefix match accesses a checkpoint."""
        pass

    def select_victims(self, candidates: list, lru_list, tree_core) -> list:
        """Sort/reorder evictable candidates. Return ordered list.

        For NativeLRU: return as-is (already in LRU order from walk).
        For HotnessAware/Oracle: reorder by policy.

        Args:
            candidates: list of (node, prefix_hash) in LRU order
            lru_list: the LRU list (for tie-breaking)
            tree_core: UnifiedTreeCore (for evictable_device_leaves, etc.)
        Returns:
            list of (node, prefix_hash) in eviction priority order
        """
        return candidates  # Native: no reordering

    def record_eviction(self, node_id: int, prefix_hash: str = ""):
        """Called after a node is evicted."""
        pass

    def get_priority(self, prefix_hash: str) -> float:
        """Lower = evict sooner. Used for logging/debugging."""
        return 0.0


class HotnessAwarePolicy(KDAEvictionPolicy):
    """Evict least-recently-accessed (cold) nodes first.

    Uses sliding window frequency + recency.
    Tie-break: native LRU order (preserved from candidate list).
    """

    def __init__(self, window_size: int = 20, cold_threshold: int = 0):
        super().__init__()
        self._name = "hotness"
        self._window_size = window_size
        self._cold_threshold = cold_threshold
        # {prefix_hash: deque([access_times])} — sliding window
        self._access_history = defaultdict(deque)
        # {prefix_hash: last_access_time}
        self._last_access = {}
        # {node_id: prefix_hash} for reverse lookup
        self._node_to_hash = {}

    def record_access(self, node_id, prefix_hash, virtual_time):
        if not prefix_hash:
            return
        hist = self._access_history[prefix_hash]
        hist.append(virtual_time)
        if len(hist) > self._window_size:
            hist.popleft()
        self._last_access[prefix_hash] = virtual_time
        self._node_to_hash[node_id] = prefix_hash

    def select_victims(self, candidates, lru_list, tree_core):
        """Sort: cold (fewer accesses) first, hot (more accesses) last.
        Tie-break: native LRU order (stable sort preserves original order).
        """
        def hotness_key(item):
            node, prefix_hash = item
            count = len(self._access_history.get(prefix_hash, []))
            return count  # Lower count = colder = evict first

        # Stable sort: preserves LRU order for ties
        return sorted(candidates, key=hotness_key)

    def get_priority(self, prefix_hash):
        return float(len(self._access_history.get(prefix_hash, [])))

    def record_eviction(self, node_id, prefix_hash=""):
        self._access_history.pop(prefix_hash, None)
        self._last_access.pop(prefix_hash, None)
        self._node_to_hash.pop(node_id, None)


class OraclePolicy(KDAEvictionPolicy):
    """Ideal policy with future access knowledge (Belady's algorithm).

    Evicts nodes that will be re-accessed FARTHEST in the future,
    or never re-accessed. This is a hit-rate reference, NOT a
    cost-aware TTFT optimizer.

    NOTE: Farthest-next-use Oracle maximizes hit rate but does NOT
    guarantee TTFT optimality — a cost-aware oracle would also consider
    restore latency and replay cost.
    """

    def __init__(self, future_access_map: dict = None):
        super().__init__()
        self._name = "oracle"
        # {prefix_hash: [future_access_times]}
        self._future = future_access_map or {}
        self._current_time = 0.0
        self._future_idx = defaultdict(int)
        self._evicted = set()

    def set_current_time(self, t: float):
        self._current_time = t

    def load_future_map(self, trace_path: str):
        """Pre-compute future access map from trace file.

        Uses full prefix token hash as identity. For build requests,
        hash input_ids[:input_length]. For re-access/reuse requests
        (input_length = build_length + 1), hash input_ids[:input_length-1]
        so the hash matches the build's prefix.
        """
        import json, hashlib
        prefix_access = defaultdict(list)
        with open(trace_path) as f:
            for line in f:
                req = json.loads(line)
                ts = req.get("timestamp", 0.0)
                ids = req.get("input_ids", [])
                ilen = req.get("input_length", len(ids))
                label = req.get("session_label", "")
                if len(ids) == 0:
                    continue
                # For re-access/reuse (length = build+1), use build length
                hash_len = ilen - 1 if ("reaccess" in label or "reuse" in label or "reaccess" in label) else ilen
                hash_len = min(hash_len, len(ids))
                if hash_len <= 0:
                    hash_len = min(ilen, len(ids))
                prefix = ids[:hash_len]
                identity = hashlib.md5(str(prefix).encode()).hexdigest()[:16]
                prefix_access[identity].append(ts)
        self._future = dict(prefix_access)
        logger.info(f"Oracle: loaded {len(self._future)} prefix entries from {trace_path}")

    def _next_access_time(self, prefix_key: str) -> float:
        """Get next future access time for this prefix identity."""
        times = self._future.get(prefix_key, [])
        idx = self._future_idx.get(prefix_key, 0)
        while idx < len(times) and times[idx] <= self._current_time:
            idx += 1
        self._future_idx[prefix_key] = idx
        if idx < len(times):
            return times[idx]
        return float('inf')  # Never accessed again

    def select_victims(self, candidates, lru_list, tree_core):
        """Sort: farthest-next-use first (evict nodes not needed soon).
        Tie-break: native LRU order (stable sort).

        Uses _node_prefix_identity to match nodes to future access map.
        This is a Belady's-style oracle (hit-rate reference), NOT a
        cost-aware TTFT optimizer.
        """
        from hisim.simulation.kda_policy import _node_prefix_identity
        def oracle_key(item):
            node, prefix_identity = item
            identity = prefix_identity
            if not identity:
                identity = _node_prefix_identity(node)
            next_t = self._next_access_time(identity)
            if next_t == float('inf'):
                return float('inf')  # Never re-accessed, evict first
            return next_t - self._current_time  # Larger distance = evict first

        return sorted(candidates, key=oracle_key, reverse=True)

    def get_priority(self, prefix_hash):
        next_t = self._next_access_time(prefix_hash)
        if next_t == float('inf'):
            return float('inf')
        return next_t - self._current_time

    def record_eviction(self, node_id, prefix_hash=""):
        self._evicted.add(prefix_hash)


def create_policy() -> KDAEvictionPolicy:
    """Factory: select policy via HISIM_KDA_POLICY env var."""
    name = os.environ.get("HISIM_KDA_POLICY", "native").lower()
    if name == "hotness":
        window = int(os.environ.get("HISIM_HOTNESS_WINDOW", "20"))
        threshold = int(os.environ.get("HISIM_HOTNESS_THRESHOLD", "0"))
        p = HotnessAwarePolicy(window_size=window, cold_threshold=threshold)
        logger.info(f"KDA Policy: hotness (window={window}, threshold={threshold})")
        return p
    elif name == "oracle":
        trace = os.environ.get("HISIM_ORACLE_TRACE", "")
        p = OraclePolicy()
        if trace:
            p.load_future_map(trace)
        logger.info(f"KDA Policy: oracle (trace={trace})")
        return p
    else:
        p = KDAEvictionPolicy()
        logger.info("KDA Policy: native (no filtering)")
        return p


# Global policy instance
_POLICY: Optional[KDAEvictionPolicy] = None


def get_policy() -> KDAEvictionPolicy:
    global _POLICY
    if _POLICY is None:
        _POLICY = create_policy()
    return _POLICY


def reset_policy():
    """Reset policy for new experiment."""
    global _POLICY
    _POLICY = None
