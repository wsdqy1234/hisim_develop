"""KDA Eviction Policy interface for Phase 2 strategy experiments.

Hook point: tree_core.evict_device_next_node (inside wrapped_init_hicache)
- Native LRU cursor provides evictable candidates (same set for all policies)
- Policy only filters WHICH candidate to evict first
- Does NOT change: Lock, ACK, Allocator, Cascade, eviction execution
- Does NOT re-enable HiSim Proactive Eviction
"""
import os
import time
import logging
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)


class KDAEvictionPolicy:
    """Base class: delegates to native LRU (no filtering)."""

    def __init__(self):
        self._name = "native"

    @property
    def name(self) -> str:
        return self._name

    def record_access(self, node_id: int, prefix_hash: str, virtual_time: float):
        """Called when a prefix match accesses a checkpoint."""
        pass

    def should_evict(self, node_id: int, prefix_hash: str = "") -> bool:
        """Return True if this node should be evicted (native: always True)."""
        return True

    def record_eviction(self, node_id: int, prefix_hash: str = ""):
        """Called after a node is evicted."""
        pass


class HotnessAwarePolicy(KDAEvictionPolicy):
    """Evict least-recently-accessed (cold) nodes first.

    Uses sliding window frequency + recency.
    Tie-break: native LRU order.
    """

    def __init__(self, window_size: int = 20, cold_threshold: int = 1):
        super().__init__()
        self._name = "hotness"
        self._window_size = window_size
        self._cold_threshold = cold_threshold
        # {node_id: deque([access_times])} — sliding window
        self._access_history = defaultdict(deque)
        # {node_id: last_access_time}
        self._last_access = {}

    def record_access(self, node_id, prefix_hash, virtual_time):
        hist = self._access_history[node_id]
        hist.append(virtual_time)
        if len(hist) > self._window_size:
            hist.popleft()
        self._last_access[node_id] = virtual_time

    def should_evict(self, node_id, prefix_hash=""):
        count = len(self._access_history.get(node_id, []))
        # Cold: fewer than threshold accesses in window → evict
        # Hot: more accesses → preserve
        if count <= self._cold_threshold:
            return True  # Cold, evict
        return False  # Hot, skip (let native LRU find another)

    def record_eviction(self, node_id, prefix_hash=""):
        self._access_history.pop(node_id, None)
        self._last_access.pop(node_id, None)


class OraclePolicy(KDAEvictionPolicy):
    """Ideal policy with future access knowledge.

    Evicts nodes that will be re-accessed FARTHEST in the future,
    or never re-accessed. This is a hit-rate reference (Belady's algorithm),
    NOT a cost-aware TTFT optimizer.
    """

    def __init__(self, future_access_map: dict):
        """future_access_map: {prefix_hash: [future_access_times]}"""
        super().__init__()
        self._name = "oracle"
        self._future = future_access_map
        self._current_time = 0.0
        self._future_idx = defaultdict(int)  # {prefix_hash: next_access_index}
        self._evicted = set()

    def set_current_time(self, t: float):
        self._current_time = t

    def _next_access_time(self, prefix_hash: str) -> float:
        times = self._future.get(prefix_hash, [])
        idx = self._future_idx.get(prefix_hash, 0)
        # Advance past accesses that already happened
        while idx < len(times) and times[idx] <= self._current_time:
            idx += 1
        self._future_idx[prefix_hash] = idx
        if idx < len(times):
            return times[idx]
        return float('inf')  # Never accessed again

    def should_evict(self, node_id, prefix_hash=""):
        if not prefix_hash:
            return True  # No identity info, fall back to native
        next_t = self._next_access_time(prefix_hash)
        if next_t == float('inf'):
            return True  # Never re-accessed, evict first
        # Threshold: if next access is "far" (> 100s), evict
        # This is a simplification of Belady's algorithm
        if next_t - self._current_time > 100.0:
            return True
        return False  # Will be re-accessed soon, preserve

    def record_eviction(self, node_id, prefix_hash=""):
        self._evicted.add(prefix_hash)


def create_policy() -> KDAEvictionPolicy:
    """Factory: select policy via HISIM_KDA_POLICY env var."""
    name = os.environ.get("HISIM_KDA_POLICY", "native").lower()
    if name == "hotness":
        window = int(os.environ.get("HISIM_HOTNESS_WINDOW", "20"))
        threshold = int(os.environ.get("HISIM_HOTNESS_THRESHOLD", "1"))
        p = HotnessAwarePolicy(window_size=window, cold_threshold=threshold)
        logger.info(f"KDA Policy: hotness (window={window}, threshold={threshold})")
        return p
    elif name == "oracle":
        # Oracle needs future access map, loaded separately
        p = OraclePolicy({})
        logger.info("KDA Policy: oracle (future_access_map empty — load trace first)")
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
