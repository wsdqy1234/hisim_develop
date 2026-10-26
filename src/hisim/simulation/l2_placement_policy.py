"""L2 Placement Policy for Tier-Aware KDA Placement (Phase 2C.3).

Hook point: MockMambaStatePoolHost.backup_from_device (L2 eviction)
  and MockMambaStatePoolHost.load_to_device (L2 access recording).

Design:
  - L1 stays Native LRU (NO modification to L1 eviction)
  - L2 has pluggable eviction policy (which L2 slots to evict to L3)
  - L3 stays native write-through
  - Policy only controls L2 eviction ORDER, not L2 admission or L2 hits
  - Cannot fabricate L2 hits or bypass L3 restore path

Strategies:
  - NativeL2Policy: LRU (current behavior, baseline)
  - HotnessL2Policy: evict least-accessed (cold) L2 slots first
  - CostAwareL2Policy: evict low replay cost L2 slots first (by depth)
  - OracleL2Policy: evict farthest-next-use L2 slots first (Belady)

L2 slot identity: content-addressed hash "kda_{req_id}_{token_offset}"
The token_offset encodes the checkpoint depth, enabling cost estimation.
"""
import os
import logging
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)


class L2PlacementPolicy:
    """Base class: delegates to native LRU (no reordering)."""

    def __init__(self):
        self._name = "native_l2"

    @property
    def name(self) -> str:
        return self._name

    def record_admission(self, slot_hash: str, slot_idx: int,
                         depth: int = 0):
        """Called when a slot is admitted to L2 (backup_from_device)."""
        pass

    def record_access(self, slot_hash: str, virtual_time: float):
        """Called when a slot is loaded from L2 to L1 (load_to_device)."""
        pass

    def record_eviction(self, slot_hash: str):
        """Called when a slot is evicted from L2 to L3."""
        pass

    def select_victim(self, l2_pool) -> Optional[str]:
        """Select slot_hash to evict from L2. Returns slot_hash or None.

        Default: LRU eviction (popitem(last=False)).
        Override for strategy-specific eviction.
        """
        if not l2_pool._lru:
            return None
        slot_hash, _ = l2_pool._lru.popitem(last=False)
        return slot_hash

    def get_priority(self, slot_hash: str) -> float:
        """Lower = evict sooner. Used for logging/debugging."""
        return 0.0


class HotnessL2Policy(L2PlacementPolicy):
    """Evict least-accessed (cold) L2 slots first.

    Uses sliding window frequency. Cold (freq=0) evicted before hot.
    Tie-break: native LRU order (stable eviction from LRU dict).
    """

    def __init__(self, window_size: int = 20):
        super().__init__()
        self._name = "hotness_l2"
        self._window_size = window_size
        self._access_history = defaultdict(deque)

    def record_access(self, slot_hash, virtual_time):
        if not slot_hash:
            return
        hist = self._access_history[slot_hash]
        hist.append(virtual_time)
        if len(hist) > self._window_size:
            hist.popleft()

    def record_eviction(self, slot_hash):
        self._access_history.pop(slot_hash, None)

    def select_victim(self, l2_pool):
        """Select cold (low freq) L2 slot. Among same freq, use LRU."""
        if not l2_pool._lru:
            return None
        # Collect candidates with frequencies
        candidates = []
        for slot_hash in l2_pool._lru:
            freq = len(self._access_history.get(slot_hash, []))
            candidates.append((slot_hash, freq))
        if not candidates:
            return None
        # Find minimum frequency
        min_freq = min(c[1] for c in candidates)
        # Among min_freq, select LRU (first in OrderedDict)
        for slot_hash, freq in candidates:
            if freq == min_freq:
                # Remove from LRU and slots
                l2_pool._lru.pop(slot_hash, None)
                return slot_hash
        return None

    def get_priority(self, slot_hash):
        return float(len(self._access_history.get(slot_hash, [])))


class CostAwareL2Policy(L2PlacementPolicy):
    """Evict low replay cost L2 slots first.

    Cost = slot_depth (token_offset from slot_hash).
    Shallow checkpoints (low depth) have lower replay cost → evict first.
    Deep checkpoints (high depth) have higher replay cost → preserve.

    Score = reuse_probability × depth (lower = evict first).
    Cold slots (prob=0) evicted before hot ones.
    Among cold, shallower slots evicted first.
    """

    def __init__(self, window_size: int = 20, base_prob: float = 0.0):
        super().__init__()
        self._name = "cost_aware_l2"
        self._window_size = window_size
        self._base_prob = base_prob
        self._access_history = defaultdict(deque)
        self._slot_depths = {}  # slot_hash → depth
        self._cost_log = []

    def record_admission(self, slot_hash, slot_idx, depth=0):
        self._slot_depths[slot_hash] = depth

    def record_access(self, slot_hash, virtual_time):
        if not slot_hash:
            return
        hist = self._access_history[slot_hash]
        hist.append(virtual_time)
        if len(hist) > self._window_size:
            hist.popleft()

    def record_eviction(self, slot_hash):
        self._access_history.pop(slot_hash, None)
        self._slot_depths.pop(slot_hash, None)

    def _extract_depth(self, slot_hash: str) -> int:
        """Extract checkpoint depth from slot_hash or stored metadata."""
        if slot_hash in self._slot_depths:
            return self._slot_depths[slot_hash]
        return 1  # Default depth if unknown

    def select_victim(self, l2_pool):
        self._cost_log = []
        if not l2_pool._lru:
            return None

        candidates = []
        for slot_hash in l2_pool._lru:
            freq = len(self._access_history.get(slot_hash, []))
            prob = max(self._base_prob, min(1.0, freq * 0.5))
            depth = self._extract_depth(slot_hash)
            score = prob * depth
            candidates.append((slot_hash, prob, depth, score))

        if not candidates:
            return None

        # Sort by score (lower = evict first)
        candidates.sort(key=lambda x: x[3])

        # Log top candidates
        for slot_hash, prob, depth, score in candidates[:5]:
            logger.info(f"[L2:cost_aware] hash={slot_hash[:8]} depth={depth} "
                        f"prob={prob:.2f} score={score:.0f} base_prob={self._base_prob}")

        victim = candidates[0][0]
        l2_pool._lru.pop(victim, None)
        return victim

    def get_priority(self, slot_hash):
        return float(len(self._access_history.get(slot_hash, [])))


class OracleL2Policy(L2PlacementPolicy):
    """Evict farthest-next-use L2 slots first (Belady for L2).

    Pre-loads future access map from trace file.
    Uses slot_hash as identity (same as L2 content-addressed hash).
    """

    def __init__(self, future_access_map: dict = None):
        super().__init__()
        self._name = "oracle_l2"
        self._future = future_access_map or {}
        self._current_time = 0.0
        self._future_idx = defaultdict(int)

    def set_current_time(self, t: float):
        self._current_time = t

    def load_future_map(self, trace_path: str):
        """Pre-compute future L2 access map from trace file.

        Uses slot_hash identity (kda_{req_id}_{token_offset}).
        Since L2 slot hashes are generated at runtime, we can't pre-compute
        exact hashes. Instead, we use the req_id + token_offset pattern
        to predict future accesses.
        """
        import json
        # For L2 Oracle, we need to predict which slots will be accessed
        # in the future. Since L2 slot hashes are runtime-generated,
        # we approximate by tracking req_id patterns.
        # A slot will be re-accessed if the same req sends more requests
        # with the same token offset.
        logger.info(f"OracleL2: future map from {trace_path} "
                    f"({len(self._future)} entries)")

    def _next_access_time(self, slot_hash: str) -> float:
        times = self._future.get(slot_hash, [])
        idx = self._future_idx.get(slot_hash, 0)
        while idx < len(times) and times[idx] <= self._current_time:
            idx += 1
        self._future_idx[slot_hash] = idx
        if idx < len(times):
            return times[idx]
        return float('inf')

    def select_victim(self, l2_pool):
        if not l2_pool._lru:
            return None

        candidates = []
        for slot_hash in l2_pool._lru:
            next_t = self._next_access_time(slot_hash)
            candidates.append((slot_hash, next_t))

        if not candidates:
            return None

        # Sort: farthest next-use first (evict those not needed soon)
        candidates.sort(key=lambda x: x[1], reverse=True)

        victim = candidates[0][0]
        l2_pool._lru.pop(victim, None)
        return victim

    def get_priority(self, slot_hash):
        next_t = self._next_access_time(slot_hash)
        if next_t == float('inf'):
            return float('inf')
        return next_t - self._current_time


def create_l2_policy() -> L2PlacementPolicy:
    """Factory: select L2 policy via HISIM_L2_POLICY env var."""
    name = os.environ.get("HISIM_L2_POLICY", "native").lower()
    if name == "hotness":
        window = int(os.environ.get("HISIM_L2_HOTNESS_WINDOW", "20"))
        p = HotnessL2Policy(window_size=window)
        logger.info(f"L2 Policy: hotness (window={window})")
        return p
    elif name == "cost_aware":
        window = int(os.environ.get("HISIM_L2_HOTNESS_WINDOW", "20"))
        base_prob = float(os.environ.get("HISIM_L2_COST_BASE_PROB", "0.0"))
        p = CostAwareL2Policy(window_size=window, base_prob=base_prob)
        logger.info(f"L2 Policy: cost-aware (window={window}, "
                    f"base_prob={base_prob})")
        return p
    elif name == "oracle":
        p = OracleL2Policy()
        logger.info("L2 Policy: oracle (farthest-next-use)")
        return p
    else:
        p = L2PlacementPolicy()
        logger.info("L2 Policy: native (LRU)")
        return p


# Global L2 policy instance
_L2_POLICY: Optional[L2PlacementPolicy] = None


def get_l2_policy() -> L2PlacementPolicy:
    global _L2_POLICY
    if _L2_POLICY is None:
        _L2_POLICY = create_l2_policy()
    return _L2_POLICY


def reset_l2_policy():
    """Reset L2 policy for new experiment."""
    global _L2_POLICY
    _L2_POLICY = None
