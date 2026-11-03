"""HiSparse (hierarchical sparse KV) simulation for hisim.

This module models the *decode-side* behaviour of sglang's HiSparse path
(``--enable-hisparse``) on top of the hisim replay loop.  sglang's real
implementation cannot run under hisim on a CPU-only host: it needs the DSA
indexer scores, CUDA streams/events and ``sgl_kernel.kvcacheio``.  What we can
reproduce faithfully -- and what actually drives the performance trade-off --
is the memory hierarchy and the data movement it causes:

* The scheduler-visible KV pool is the *logical* pool.  Its capacity is
  ``physical_pool_tokens * host_to_device_ratio`` because the full sequence
  lives in pinned host DRAM.
* HBM only holds a fixed *hot* pool.  Every running request reserves
  ``padded_buffer_size = device_buffer_size + page_size`` slots, so the number
  of concurrently resident requests is bounded by
  ``physical_pool_tokens // padded_buffer_size``.
* At every decode step the DSA-style indexer selects ``top_k`` token positions.
  Positions that are not in the request's hot buffer are copied host->device on
  the critical path (``kvcacheio`` swap-in).  A page-granular LRU keeps the
  hottest positions resident.
* Prefill KV is staged device->host once.  Each newly decoded token is eagerly
  backed up device->host on a side stream (hidden from the critical path by
  default, but the bytes are accounted).

The real indexer scores are not observable in a trace replay, so the top-k
access stream is a configurable abstraction (see :class:`HiSparseSimConfig`).
``sticky`` (the default) keeps a working set whose non-recent part churns by
``churn`` per step: that is the knob to calibrate against counters from a real
sglang HiSparse deployment (hot-buffer hits/misses, swap-in bytes).

The per-layer working sets are modelled as i.i.d.: one canonical stream is
sampled per request and the resulting bytes/hits are scaled by the number of KV
layers.  When ``shared_index_freq > 1`` the IO of the non-anchor layers is
prefetched and therefore removed from the critical path.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from hisim.utils import get_logger

logger = get_logger("hisim")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HiSparseSimConfig:
    """Knobs for the HiSparse simulation.

    Defaults mirror sglang's HiSparse defaults where they exist:
    ``top_k=2048``, ``device_buffer_size=2*top_k`` and
    ``host_to_device_ratio=2``.  The access-model knobs have no sglang
    counterpart (the real values come from the indexer) and are documented as
    calibration parameters.
    """

    enable: bool = False
    # Number of KV positions selected by the DSA indexer per decode step.
    top_k: int = 2048
    # Per-request HBM buffer, in tokens.  0 means 2 * top_k.
    device_buffer_size: int = 0
    # host pool tokens = physical (HBM) pool tokens * this ratio.
    host_to_device_ratio: int = 2
    # Page granularity of the device/host pools.  padded_buffer_size is
    # device_buffer_size + page_size, matching hisparse_coordinator.py.
    page_size: int = 1
    # Access model: sticky | recency | uniform.
    access_model: str = "sticky"
    # Fraction of the non-recent working set replaced every decode step.
    churn: float = 0.05
    # Fraction of top_k that is always the most recent positions.
    recent_ratio: float = 0.5
    # 1 = every layer does its own indexer/swap-in; N > 1 = one anchor layer
    # per group of N, the followers are prefetched (off the critical path).
    shared_index_freq: int = 1
    # Eager device->host backup of the newest token.  sglang runs it on a side
    # stream, so by default it is accounted but not added to the step latency.
    backup_on_critical: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.device_buffer_size <= 0:
            self.device_buffer_size = 2 * self.top_k
        self.top_k = max(1, int(self.top_k))
        self.device_buffer_size = max(self.top_k, int(self.device_buffer_size))
        self.host_to_device_ratio = max(1, int(self.host_to_device_ratio))
        self.page_size = max(1, int(self.page_size))
        self.churn = min(max(float(self.churn), 0.0), 1.0)
        self.recent_ratio = min(max(float(self.recent_ratio), 0.0), 1.0)
        self.shared_index_freq = max(1, int(self.shared_index_freq))
        if self.access_model not in ("sticky", "recency", "uniform"):
            logger.warning(
                "Unknown HiSparse access model %r, falling back to 'sticky'.",
                self.access_model,
            )
            self.access_model = "sticky"

    @property
    def padded_buffer_size(self) -> int:
        # hisparse_coordinator.py: an extra page for the newest token.
        return self.device_buffer_size + self.page_size

    @property
    def critical_fraction(self) -> float:
        """Fraction of per-layer swap-in bytes that is on the critical path."""
        return 1.0 / self.shared_index_freq

    @classmethod
    def from_mapping(cls, raw: Optional[dict], env: Optional[dict] = None) -> "HiSparseSimConfig":
        raw = dict(raw or {})
        env = dict(env or {})
        aliases = {
            "enable": "enable",
            "top_k": "top_k",
            "device_buffer_size": "device_buffer_size",
            "host_to_device_ratio": "host_to_device_ratio",
            "page_size": "page_size",
            "access_model": "access_model",
            "churn": "churn",
            "recent_ratio": "recent_ratio",
            "shared_index_freq": "shared_index_freq",
            "backup_on_critical": "backup_on_critical",
            "seed": "seed",
        }
        kwargs = {}
        for key, attr in aliases.items():
            if key in raw and raw[key] is not None:
                kwargs[attr] = raw[key]
        for key, value in env.items():
            if value is not None:
                kwargs[aliases[key]] = value
        # Coerce values coming from JSON / environment strings.
        for key in ("top_k", "device_buffer_size", "host_to_device_ratio", "page_size",
                    "shared_index_freq", "seed"):
            if key in kwargs:
                kwargs[key] = int(kwargs[key])
        for key in ("churn", "recent_ratio"):
            if key in kwargs:
                kwargs[key] = float(kwargs[key])
        if "enable" in kwargs and not isinstance(kwargs["enable"], bool):
            kwargs["enable"] = str(kwargs["enable"]).strip().lower() in (
                "1", "true", "yes", "on",
            )
        if "backup_on_critical" in kwargs and not isinstance(kwargs["backup_on_critical"], bool):
            kwargs["backup_on_critical"] = str(kwargs["backup_on_critical"]).strip().lower() in (
                "1", "true", "yes", "on",
            )
        return cls(**kwargs)


def load_hisparse_config() -> HiSparseSimConfig:
    """Read the ``hisparse`` section of the hisim config, then apply env
    overrides.  Environment variables win so a single asset can be reused for
    baseline vs. sparse runs."""
    raw = {}
    try:
        from hisim.simulation.manager import Envs

        with open(Envs.config_path()) as f:
            config = json.load(f)
        raw = config.get("hisparse", {}) or {}
    except Exception:
        raw = {}

    env = {
        "enable": os.getenv("HISIM_HISPARSE"),
        "top_k": os.getenv("HISIM_HISPARSE_TOP_K"),
        "device_buffer_size": os.getenv("HISIM_HISPARSE_DEVICE_BUFFER_SIZE"),
        "host_to_device_ratio": os.getenv("HISIM_HISPARSE_HOST_TO_DEVICE_RATIO"),
        "page_size": os.getenv("HISIM_HISPARSE_PAGE_SIZE"),
        "access_model": os.getenv("HISIM_HISPARSE_ACCESS_MODEL"),
        "churn": os.getenv("HISIM_HISPARSE_CHURN"),
        "recent_ratio": os.getenv("HISIM_HISPARSE_RECENT_RATIO"),
        "shared_index_freq": os.getenv("HISIM_HISPARSE_SHARED_INDEX_FREQ"),
        "backup_on_critical": os.getenv("HISIM_HISPARSE_BACKUP_ON_CRITICAL"),
        "seed": os.getenv("HISIM_HISPARSE_SEED"),
    }
    return HiSparseSimConfig.from_mapping(raw, env)


# ---------------------------------------------------------------------------
# Transfer timing
# ---------------------------------------------------------------------------


def transfer_duration(
    direction: str,
    num_bytes: float,
    read_bw: Optional[float] = None,
    write_bw: Optional[float] = None,
) -> float:
    """Same bandwidth model hisim uses for L2 host transfers:

        duration = t0 + bytes / (raw_bandwidth * 0.85)

    with t0 = 6.67us host->device and 4us device->host.
    """
    if num_bytes <= 0:
        return 0.0
    raw_bw = read_bw if direction == "H2D" else write_bw
    if raw_bw is None:
        try:
            from hisim.simulation.manager import ConfigManager

            pc = ConfigManager.get_platform_config()
            raw_bw = (
                pc.memory_read_bandwidth
                if direction == "H2D"
                else pc.memory_write_bandwidth
            )
        except Exception:
            raw_bw = None
    t0 = 6.67e-6 if direction == "H2D" else 4e-6
    if not raw_bw:
        return 0.0
    return t0 + num_bytes / (raw_bw * 0.85)


# ---------------------------------------------------------------------------
# Top-k access model
# ---------------------------------------------------------------------------


class _WorkingSet:
    """Canonical per-request top-k access stream.

    The selected set is represented as a rolling *recent* window (implicit:
    the last ``n_recent`` positions) plus a *sticky* set of older positions.
    Only the changes are materialised, so a step costs
    ``O(churn * top_k + new_tokens)`` instead of ``O(top_k)``.  The hot buffer
    is an LRU set of token positions with capacity ``padded_buffer_size``;
    positions no longer selected are evicted lazily, which is what gives the
    buffer its slack over a bare working set.
    """

    def __init__(
        self,
        top_k: int,
        capacity: int,
        access_model: str,
        churn: float,
        recent_ratio: float,
        rng: random.Random,
    ) -> None:
        self.top_k = top_k
        self.capacity = capacity
        self.access_model = access_model
        self.churn = churn
        self.recent_ratio = recent_ratio
        self.rng = rng
        self.sticky: Optional[set] = None
        self.hot: "OrderedDict[int, None]" = OrderedDict()
        self.prev_seq_len = 0
        self.prev_recent_start = 0
        self.prev_n = 0

    def _recent_count(self, n: int) -> int:
        return min(n, max(1, int(round(n * self.recent_ratio)))) if n else 0

    def _window(self, seq_len: int) -> Tuple[int, int, int]:
        n = min(self.top_k, seq_len)
        n_recent = self._recent_count(n)
        recent_start = seq_len - n_recent
        return n, n_recent, recent_start

    def _sample(self, size: int, limit: int, exclude: set) -> set:
        """Sample ``size`` positions from ``[0, limit)`` not in ``exclude``."""
        if size <= 0 or limit <= 0:
            return set()
        chosen: set = set()
        attempts = 0
        max_attempts = max(size * 20, 64)
        while len(chosen) < size and attempts < max_attempts:
            pos = self.rng.randrange(0, limit)
            if pos not in exclude and pos not in chosen:
                chosen.add(pos)
            attempts += 1
        if len(chosen) < size:
            for pos in range(limit):
                if pos not in exclude and pos not in chosen:
                    chosen.add(pos)
                    if len(chosen) >= size:
                        break
        return chosen

    def _seed_hot_buffer(self, seq_len: int) -> None:
        # Prefill fills the device buffer with the newest tokens.
        start = max(0, seq_len - self.capacity)
        self.hot = OrderedDict((pos, None) for pos in range(start, seq_len))

    def first_step(self, seq_len: int) -> Tuple[int, int]:
        n, n_recent, recent_start = self._window(seq_len)
        if self.access_model == "uniform":
            selected = set(self.rng.sample(range(seq_len), n))
            self.sticky = selected
        else:
            target_sticky = max(0, n - n_recent)
            if self.access_model == "recency":
                self.sticky = set()
            else:
                self.sticky = self._sample(target_sticky, recent_start, set())
        self._seed_hot_buffer(seq_len)
        misses = len(self.sticky - set(self.hot))
        for pos in self.sticky:
            self.hot.setdefault(pos, None)
        self.prev_seq_len = seq_len
        self.prev_recent_start = recent_start
        self.prev_n = n
        self._trim(recent_start)
        return n - misses, misses

    def _trim(self, recent_start: int) -> None:
        # Evict LRU positions that are neither in the recent window nor in the
        # sticky set.  ``capacity >= top_k`` guarantees there are enough.
        while len(self.hot) > self.capacity:
            pos, _ = self.hot.popitem(last=False)
            if pos >= recent_start or pos in (self.sticky or ()):
                self.hot[pos] = None
                continue

    def step(self, seq_len: int) -> Tuple[int, int]:
        if self.sticky is None:
            return self.first_step(seq_len)
        if seq_len < self.prev_seq_len:
            seq_len = self.prev_seq_len
        n, n_recent, recent_start = self._window(seq_len)

        if self.access_model == "uniform":
            selected = set(self.rng.sample(range(seq_len), n))
            misses = len(selected - set(self.hot))
            for pos in selected:
                self.hot[pos] = None
            self.sticky = selected
            self.prev_seq_len = seq_len
            self.prev_recent_start = recent_start
            self.prev_n = n
            self._trim(recent_start)
            return n - misses, misses

        # 1. Positions that entered the recent window roll out of the sticky
        #    set (at most ``seq_len - prev_seq_len`` of them in steady state).
        if self.sticky:
            if recent_start >= self.prev_recent_start:
                for pos in range(self.prev_recent_start, recent_start):
                    self.sticky.discard(pos)
            else:  # window shrank (early steps); resync completely
                self.sticky = {p for p in self.sticky if p < recent_start}
        elif self.access_model == "recency":
            self.sticky = set()

        # 2. Keep the sticky set at its target size (it grows while the
        #    sequence is still shorter than top_k).
        target_sticky = max(0, n - n_recent) if self.access_model != "recency" else 0
        if len(self.sticky) < target_sticky:
            self.sticky |= self._sample(
                target_sticky - len(self.sticky), recent_start, self.sticky
            )
        elif len(self.sticky) > target_sticky:
            drop = len(self.sticky) - target_sticky
            self.sticky -= set(self.rng.sample(sorted(self.sticky), drop))

        # 3. Churn a fraction of the sticky set: replace it with positions from
        #    the history.  Dropped positions stay in the hot buffer until LRU
        #    eviction, so a returning position can still hit.
        if self.access_model == "sticky" and target_sticky > 0:
            k = min(len(self.sticky), int(round(self.churn * target_sticky)))
            if k > 0:
                dropped = set(self.rng.sample(sorted(self.sticky), k))
                self.sticky -= dropped
                self.sticky |= self._sample(k, recent_start, self.sticky)

        # 4. Resolve the newly selected positions against the hot buffer.
        new_items = [p for p in self.sticky if p not in self.hot]
        misses = len(new_items)
        for pos in new_items:
            self.hot[pos] = None
        # Newly decoded tokens enter the recent window and are already in HBM
        # (the newest-token page), so they are hits.
        if seq_len > self.prev_seq_len:
            for pos in range(self.prev_seq_len, seq_len):
                self.hot.setdefault(pos, None)

        self.prev_seq_len = seq_len
        self.prev_recent_start = recent_start
        self.prev_n = n
        self._trim(recent_start)
        return n - misses, misses


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


@dataclass
class StepCost:
    hits: int = 0
    misses: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    load_dur: float = 0.0
    backup_dur: float = 0.0


@dataclass
class _RequestState:
    rid: str
    host_tokens: int
    working: _WorkingSet
    steps: int = 0
    hits: int = 0
    misses: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    staged_bytes: int = 0
    finished: bool = False


class HiSparseSimulator:
    """Global HiSparse accounting/timing model.

    One instance per simulation process (the hisim scheduler is single
    process).  ``configure`` is called from the model-runner hook once the
    physical KV pool capacity and the model geometry are known.
    """

    _instance: Optional["HiSparseSimulator"] = None

    def __init__(
        self,
        config: HiSparseSimConfig,
        *,
        num_layers: int,
        bytes_per_token_per_layer: float,
        physical_pool_tokens: int,
        page_size: int = 1,
        max_num_reqs: Optional[int] = None,
        read_bandwidth: Optional[float] = None,
        write_bandwidth: Optional[float] = None,
    ) -> None:
        self.config = config
        self.num_layers = max(1, int(num_layers))
        self.bytes_per_token_per_layer = float(bytes_per_token_per_layer)
        self.physical_pool_tokens = max(1, int(physical_pool_tokens))
        self.page_size = max(1, int(page_size))
        self.padded_buffer_size = config.padded_buffer_size
        self.read_bandwidth = read_bandwidth
        self.write_bandwidth = write_bandwidth

        self.logical_pool_tokens = int(
            self.physical_pool_tokens * config.host_to_device_ratio
        )
        # Host pool is sized from the device pool exactly like sglang.
        self.host_pool_tokens = self.logical_pool_tokens
        self.max_reqs_by_hot_pool = max(
            1, self.physical_pool_tokens // self.padded_buffer_size
        )
        self.max_num_reqs = (
            min(max_num_reqs, self.max_reqs_by_hot_pool)
            if max_num_reqs
            else self.max_reqs_by_hot_pool
        )

        self.requests: Dict[str, _RequestState] = {}
        self.finished_requests = 0
        self.decode_steps = 0
        self.decode_batches = 0
        self.hits = 0
        self.misses = 0
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.staging_bytes = 0
        self.load_dur = 0.0
        self.backup_dur = 0.0
        # Eager-backup latency that is hidden behind the next forward when
        # backup_on_critical is off.
        self.backup_hidden_dur = 0.0
        self.peak_host_tokens = 0
        self.host_overcommit_tokens = 0
        self.host_tokens = 0
        self.device_alloc_fail = 0
        self.cur_host_tokens = 0
        self._host_warned = False
        self._rng = random.Random(config.seed)

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def configure(
        cls,
        config: HiSparseSimConfig,
        *,
        num_layers: int,
        bytes_per_token_per_layer: float,
        physical_pool_tokens: int,
        page_size: int = 1,
        max_num_reqs: Optional[int] = None,
    ) -> "HiSparseSimulator":
        cls._instance = cls(
            config,
            num_layers=num_layers,
            bytes_per_token_per_layer=bytes_per_token_per_layer,
            physical_pool_tokens=physical_pool_tokens,
            page_size=page_size,
            max_num_reqs=max_num_reqs,
        )
        logger.info(
            "HiSparse simulation enabled: top_k=%d device_buffer=%d "
            "padded_buffer=%d host_to_device_ratio=%d "
            "physical_tokens=%d logical_tokens=%d max_reqs_hot_pool=%d "
            "hbm_kv=%.2fGB host_kv=%.2fGB",
            config.top_k,
            config.device_buffer_size,
            config.padded_buffer_size,
            config.host_to_device_ratio,
            cls._instance.physical_pool_tokens,
            cls._instance.logical_pool_tokens,
            cls._instance.max_reqs_by_hot_pool,
            cls._instance.hbm_kv_bytes / 1e9,
            cls._instance.host_kv_bytes / 1e9,
        )
        return cls._instance

    @classmethod
    def instance(cls) -> Optional["HiSparseSimulator"]:
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    @property
    def hbm_kv_bytes(self) -> float:
        return (
            self.physical_pool_tokens
            * self.num_layers
            * self.bytes_per_token_per_layer
        )

    @property
    def host_kv_bytes(self) -> float:
        return (
            self.logical_pool_tokens
            * self.num_layers
            * self.bytes_per_token_per_layer
        )

    def on_request_start(self, rid: str, prompt_len: int) -> None:
        prompt_len = max(0, int(prompt_len))
        if rid in self.requests:
            return
        working = _WorkingSet(
            top_k=self.config.top_k,
            capacity=self.padded_buffer_size,
            access_model=self.config.access_model,
            churn=self.config.churn,
            recent_ratio=self.config.recent_ratio,
            rng=random.Random(self._rng.random()),
        )
        staged = prompt_len * self.num_layers * self.bytes_per_token_per_layer
        state = _RequestState(
            rid=rid,
            host_tokens=prompt_len,
            working=working,
            staged_bytes=int(staged),
        )
        self.requests[rid] = state
        self.staging_bytes += int(staged)
        self.cur_host_tokens += prompt_len
        if self.cur_host_tokens > self.host_pool_tokens:
            self.host_overcommit_tokens = max(
                self.host_overcommit_tokens, self.cur_host_tokens - self.host_pool_tokens
            )
            if not self._host_warned:
                logger.warning(
                    "HiSparse host pool overcommitted: %d tokens in use > %d "
                    "pool tokens.  Increase HISIM_HISPARSE_HOST_TO_DEVICE_RATIO "
                    "or reduce concurrency.",
                    self.cur_host_tokens,
                    self.host_pool_tokens,
                )
                self._host_warned = True
        self.peak_host_tokens = max(self.peak_host_tokens, self.cur_host_tokens)

    def _decode_one(self, rid: str, past_len: int) -> StepCost:
        """Update one request's access stream and return its byte deltas.

        Transfer timing is *not* computed here: a decode batch is served by
        one swap-in kernel (one fixed launch overhead) whose copies share the
        host<->device bandwidth, so timing is applied once per batch in
        :meth:`on_decode_batch`.
        """
        state = self.requests.get(rid)
        if state is None:
            return StepCost()
        past_len = max(1, int(past_len))
        hits, misses = state.working.step(past_len)

        n_layers = self.num_layers
        h2d = int(misses * n_layers * self.bytes_per_token_per_layer)
        # The freshly decoded token is written to HBM and eagerly backed up
        # to host; the first decode step is skipped (nothing to back up yet).
        backup_tokens = 1 if state.steps > 0 else 0
        d2h = int(backup_tokens * n_layers * self.bytes_per_token_per_layer)

        state.steps += 1
        state.hits += hits * n_layers
        state.misses += misses * n_layers
        state.h2d_bytes += h2d
        state.d2h_bytes += d2h
        # The decoded token extends the host-resident sequence.
        state.host_tokens += 1
        self.cur_host_tokens += 1

        self.decode_steps += 1
        self.hits += hits * n_layers
        self.misses += misses * n_layers
        self.h2d_bytes += h2d
        self.d2h_bytes += d2h
        self.peak_host_tokens = max(self.peak_host_tokens, self.cur_host_tokens)
        if self.cur_host_tokens > self.host_pool_tokens:
            self.host_overcommit_tokens = max(
                self.host_overcommit_tokens, self.cur_host_tokens - self.host_pool_tokens
            )

        return StepCost(
            hits=hits * n_layers,
            misses=misses * n_layers,
            h2d_bytes=h2d,
            d2h_bytes=d2h,
        )

    def _account_transfer(self, cost: StepCost) -> None:
        cost.load_dur = transfer_duration(
            "H2D",
            cost.h2d_bytes * self.config.critical_fraction,
            self.read_bandwidth,
            self.write_bandwidth,
        )
        cost.backup_dur = transfer_duration(
            "D2H", cost.d2h_bytes, self.read_bandwidth, self.write_bandwidth
        )
        self.decode_batches += 1
        self.load_dur += cost.load_dur
        self.backup_dur += cost.backup_dur

    def on_decode_step(self, rid: str, past_len: int) -> StepCost:
        """Single-request decode step (batch of one)."""
        cost = self._decode_one(rid, past_len)
        self._account_transfer(cost)
        return cost

    def on_decode_batch(self, requests: Sequence[Tuple[str, int]]) -> StepCost:
        total = StepCost()
        for rid, past_len in requests:
            cost = self._decode_one(rid, past_len)
            total.hits += cost.hits
            total.misses += cost.misses
            total.h2d_bytes += cost.h2d_bytes
            total.d2h_bytes += cost.d2h_bytes
        self._account_transfer(total)
        return total

    def on_request_finish(self, rid: str) -> None:
        state = self.requests.pop(rid, None)
        if state is None:
            return
        if state.finished:
            return
        state.finished = True
        self.finished_requests += 1
        self.cur_host_tokens -= state.host_tokens
        if self.cur_host_tokens < 0:
            self.cur_host_tokens = 0

    # -- reporting ---------------------------------------------------------

    def metrics(self) -> dict:
        touched = self.hits + self.misses
        hbm = self.hbm_kv_bytes
        host = self.host_kv_bytes
        baseline_hbm = self.logical_pool_tokens * self.num_layers * self.bytes_per_token_per_layer
        return {
            "hisparse_enabled": 1,
            "hisparse_top_k": self.config.top_k,
            "hisparse_device_buffer_size": self.config.device_buffer_size,
            "hisparse_padded_buffer_size": self.padded_buffer_size,
            "hisparse_host_to_device_ratio": self.config.host_to_device_ratio,
            "hisparse_access_model": self.config.access_model,
            "hisparse_churn": self.config.churn,
            "hisparse_recent_ratio": self.config.recent_ratio,
            "hisparse_shared_index_freq": self.config.shared_index_freq,
            "hisparse_backup_on_critical": int(self.config.backup_on_critical),
            "hisparse_physical_pool_tokens": self.physical_pool_tokens,
            "hisparse_logical_pool_tokens": self.logical_pool_tokens,
            "hisparse_host_pool_tokens": self.host_pool_tokens,
            "hisparse_max_reqs_hot_pool": self.max_reqs_by_hot_pool,
            "hisparse_num_layers": self.num_layers,
            "hisparse_bytes_per_token_per_layer": self.bytes_per_token_per_layer,
            "hisparse_hbm_kv_bytes": int(hbm),
            "hisparse_host_kv_bytes": int(host),
            "hisparse_baseline_hbm_kv_bytes": int(baseline_hbm),
            "hisparse_hbm_saving_bytes": int(max(0, baseline_hbm - hbm)),
            "hisparse_requests": len(self.requests) + self.finished_requests,
            "hisparse_finished_requests": self.finished_requests,
            "hisparse_decode_steps": self.decode_steps,
            "hisparse_decode_batches": self.decode_batches,
            "hisparse_hits": self.hits,
            "hisparse_misses": self.misses,
            "hisparse_hit_rate": (self.hits / touched) if touched else 0.0,
            "hisparse_h2d_bytes": int(self.h2d_bytes),
            "hisparse_d2h_bytes": int(self.d2h_bytes),
            "hisparse_staging_bytes": int(self.staging_bytes),
            "hisparse_total_io_bytes": int(self.h2d_bytes + self.d2h_bytes + self.staging_bytes),
            "hisparse_load_latency_s": self.load_dur,
            "hisparse_backup_latency_s": self.backup_dur,
            "hisparse_backup_hidden_latency_s": self.backup_hidden_dur,
            "hisparse_mean_load_us": (
                self.load_dur / self.decode_batches * 1e6
                if self.decode_batches
                else 0.0
            ),
            "hisparse_mean_backup_us": (
                self.backup_dur / self.decode_batches * 1e6
                if self.decode_batches
                else 0.0
            ),
            "hisparse_peak_host_tokens": self.peak_host_tokens,
            "hisparse_host_overcommit_tokens": self.host_overcommit_tokens,
            "hisparse_device_alloc_fail": self.device_alloc_fail,
        }


# ---------------------------------------------------------------------------
# Helpers used by the hooks
# ---------------------------------------------------------------------------


def model_kv_geometry(model_config) -> Tuple[int, float]:
    """Return ``(num_layers, bytes_per_token_per_layer)`` for the HF config.

    Mirrors sglang/hisim: MLA models store ``kv_lora_rank + qk_rope_head_dim``
    elements per token per layer; MHA/GQA models store ``2 * kv_heads * head_dim``.
    """
    num_layers = int(
        getattr(model_config, "num_hidden_layers", 0)
        or getattr(model_config, "num_layers", 0)
        or 1
    )
    dtype = getattr(model_config, "dtype", None)
    try:
        import torch

        itemsize = torch._utils._element_size(dtype) if dtype is not None else 2
    except Exception:
        itemsize = 2

    kv_lora_rank = int(getattr(model_config, "kv_lora_rank", 0) or 0)
    if kv_lora_rank > 0:
        qk_rope = int(getattr(model_config, "qk_rope_head_dim", 0) or 0)
        per_layer = (kv_lora_rank + qk_rope) * itemsize
    else:
        heads = int(
            getattr(model_config, "num_key_value_heads", 0)
            or getattr(model_config, "num_attention_heads", 0)
            or 1
        )
        head_dim = int(
            getattr(model_config, "head_dim", 0)
            or (
                int(getattr(model_config, "hidden_size", 0) or 0)
                // max(1, int(getattr(model_config, "num_attention_heads", 0) or 1))
            )
            or 1
        )
        per_layer = 2 * heads * head_dim * itemsize
    return num_layers, float(per_layer)


def model_info_kv_geometry(
    model_info, tp_size: int = 1, itemsize: Optional[int] = None
) -> Tuple[int, float]:
    """Same as :func:`model_kv_geometry`, but for hisim's ``ModelInfo`` and an
    explicit KV-cache element size (the cache dtype can differ from the model
    dtype, e.g. FP8 KV for DeepSeek)."""
    num_layers = max(1, int(getattr(model_info, "num_hidden_layers", 1) or 1))
    if itemsize is None:
        itemsize = 2
    kv_lora_rank = int(getattr(model_info, "kv_lora_rank", 0) or 0)
    if kv_lora_rank > 0:
        per_layer = (kv_lora_rank + int(getattr(model_info, "qk_rope_head_dim", 0) or 0)) * itemsize
    else:
        heads = max(
            1,
            int(getattr(model_info, "num_key_value_heads", 0) or 1) // max(1, tp_size),
        )
        head_dim = int(getattr(model_info, "head_dim", 0) or 1)
        per_layer = 2 * heads * head_dim * itemsize
    return num_layers, float(per_layer)
