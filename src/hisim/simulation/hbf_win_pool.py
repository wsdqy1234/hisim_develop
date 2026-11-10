"""Phase 5B: 8-WIN HBF Pool Model (WIN-aware mode).

Models the expert-specified 8-WIN HBF as a pool of 8 stacked WIN resources:

  Single WIN (expert-calibrated Service Envelope):
    - capacity 100 GB, seq read 200 GB/s, rand read 100 GB/s,
      write 10 GB/s, read latency 5-10 μs (midpoint default).
  8-WIN aggregate targets (emerge naturally, not enforced separately):
    800 GB / 1600 / 800 / 80 GB/s.

Two modeling layers:
  A. "envelope"  — single-WIN calibrated performance as Service Envelope
                   (expert numbers; reference upper bound).
  B. "pagesim"   — per-WIN PageLevelMediaSim for Page/Channel/Plane loss
                   decomposition. Geometry (channels/plane/page/rates) is
                   UNKNOWN for the new HBF: all geometry params are explicit
                   ASSUMPTIONS (default = HBFSim nominal placeholder) and
                   MUST NOT be tuned to reverse-engineer 200 GB/s or any
                   efficiency target.

Structure (per WIN): independent I/O queue count, media busy cursor
(reads and writes CONTEND on the same WIN media), stream cursor for
sequential/random auto-detection, throughput statistics.

Optional interfaces (both are assumptions pending hardware clarification
of "UB×4@224G" — see Phase 5A missing-parameter list):
  - per_win_interface_bw:    each WIN has its own link.
  - package_shared_interface_bw: one package-level link shared by all WINs.
  Both default None (no interface constraint).
  interface_full_duplex=False (half-duplex, conservative assumption).

Cross-batch safety: begin_batch() records batch start but preserves busy
cursors (in-flight I/O keeps WINs/interface busy), matching
SharedHbfResourceManager semantics.

All parameters live in Hbf8WinConfig (single config object, JSON-loadable).
The effective config is dumped as JSON (with assumption annotations) —
no scattered hardcoded env vars.
"""
from __future__ import annotations

import json
import copy
import os
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional, List

GB = 1e9


# ---------------------------------------------------------------------------
# Configuration objects
# ---------------------------------------------------------------------------

@dataclass
class WinSpec:
    """Single-WIN calibrated Service Envelope (expert spec 2026-11).

    seq_read_bw / rand_read_bw / write_bw / capacity are expert-calibrated.
    read_latency_s: expert range 5-10 μs, default midpoint (assumption
    within the given range — exact value unresolved).
    write_latency_s: NOT expert-given — assumption (NAND program ballpark).
    """
    capacity_bytes: int = 100 * GB
    seq_read_bw: float = 200e9
    rand_read_bw: float = 100e9
    write_bw: float = 10e9
    read_latency_s: float = 7.5e-6
    write_latency_s: float = 100e-6
    source: str = "expert-2026-11"


@dataclass
class PagesimAssumptions:
    """Layer-B geometry per WIN — ALL assumptions, NOT expert-calibrated.

    Defaults are HBFSim nominal placeholders (32ch × 1.6 GB/s = 51.2 GB/s
    per WIN). They deliberately do NOT reproduce the expert's 200 GB/s:
    geometry is unknown and must not be reverse-tuned.
    """
    channels: int = 32
    planes_per_channel: int = 32
    page_bytes: int = 16384
    read_latency_s: float = 10e-6
    write_latency_s: float = 100e-6
    channel_rate: float = 1.6e9
    note: str = ("ASSUMPTION: WIN geometry unknown; HBFSim nominal "
                 "placeholder; NOT expert-calibrated; do not tune to "
                 "match 200 GB/s")


@dataclass
class Hbf8WinConfig:
    """All 8-WIN model parameters in one config object."""
    n_wins: int = 8
    win: WinSpec = field(default_factory=WinSpec)
    # "envelope" (layer A: expert Service Envelope)
    # "pagesim"   (layer B: per-WIN PageSim with ASSUMED geometry)
    layer: str = "envelope"
    # default access mode per I/O when not specified / not auto-detectable:
    # "sequential" | "random" | "auto"
    default_access_mode: str = "sequential"
    # object placement across WINs:
    #   "balanced" — Phase 5B object-level: one object → one WIN (registry)
    #   "hotspot"  — negative control: concentrate on hotspot_win
    #   "striped"  — Phase 5D P1-B: block-level striping across WINs
    placement: str = "balanced"
    hotspot_win: int = 0          # hotspot concentrates here first
    # Phase 5D P1-B: stripe block size in TOKENS (virtual slots per block;
    # block b → WIN b % n_wins). Must divide objects across the 8 WINs'
    # independent resource queues — NOT a time-division shortcut.
    stripe_block_tokens: int = 256
    # Interface assumptions (UB×4@224G unresolved — see audit):
    per_win_interface_bw: Optional[float] = None
    package_shared_interface_bw: Optional[float] = None
    interface_full_duplex: bool = False   # half-duplex is the default assumption
    queue_depth: int = 128
    pagesim_assumptions: PagesimAssumptions = field(
        default_factory=PagesimAssumptions)
    spec_source: str = "expert-2026-11"

    # ---- validation ----

    def validate(self) -> None:
        assert self.n_wins >= 1, "n_wins must be >= 1"
        assert self.layer in ("envelope", "pagesim"), \
            f"unknown layer {self.layer}"
        assert self.default_access_mode in ("sequential", "random", "auto"), \
            f"unknown access mode {self.default_access_mode}"
        assert self.placement in ("balanced", "hotspot", "striped"), \
            f"unknown placement {self.placement}"
        assert 0 <= self.hotspot_win < self.n_wins, "hotspot_win out of range"
        assert self.stripe_block_tokens >= 1, "stripe_block_tokens must be >= 1"
        w = self.win
        assert w.seq_read_bw > 0 and w.rand_read_bw > 0 and w.write_bw > 0
        assert w.capacity_bytes > 0

    # ---- serialization ----

    def to_effective_dict(self) -> dict:
        """Effective config as a dict, with assumption annotations."""
        d = asdict(self)
        d["assumptions"] = {
            "win.write_latency_s": "NOT expert-given (NAND ballpark)",
            "win.read_latency_s": "expert range 5-10μs, midpoint chosen",
            "pagesim_assumptions.*": self.pagesim_assumptions.note,
            "per_win_interface_bw / package_shared_interface_bw":
                ("UB×4@224G unresolved: G unit, lane count, per-WIN vs "
                 "package-shared, duplex, encoding overhead all unknown; "
                 "default None = unconstrained"),
            "interface_full_duplex": "half-duplex modeled by default",
            "aggregate_caps": ("1600/800/80 GB/s emerge from per-WIN "
                               "limits; not separately enforced"),
        }
        return d

    def dump_effective_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_effective_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Hbf8WinConfig":
        d = dict(d)
        d.pop("assumptions", None)   # annotation-only, not a constructor param
        win = d.pop("win", {})
        ps = d.pop("pagesim_assumptions", {})
        cfg = cls(**d)
        if win:
            cfg.win = WinSpec(**win)
        if ps:
            cfg.pagesim_assumptions = PagesimAssumptions(**ps)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "Hbf8WinConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ---- aggregate envelope properties (informational) ----

    @property
    def aggregate_seq_read_bw(self) -> float:
        return self.n_wins * self.win.seq_read_bw

    @property
    def aggregate_rand_read_bw(self) -> float:
        return self.n_wins * self.win.rand_read_bw

    @property
    def aggregate_write_bw(self) -> float:
        return self.n_wins * self.win.write_bw

    @property
    def total_capacity_bytes(self) -> int:
        return self.n_wins * self.win.capacity_bytes


# ---------------------------------------------------------------------------
# Per-WIN resource
# ---------------------------------------------------------------------------

@dataclass
class WinIoStats:
    """Per-I/O statistics from the WIN pool."""
    win: int = 0
    bytes: int = 0
    is_read: bool = True
    mode: str = "sequential"        # sequential | random
    arrival: float = 0.0
    media_start: float = 0.0
    media_time: float = 0.0
    media_completion: float = 0.0
    ic_start: float = 0.0
    ic_time: float = 0.0
    ic_completion: float = 0.0
    queue_wait: float = 0.0
    total: float = 0.0
    interface_constrained: bool = False
    # actual media service duration (excludes queue wait; Layer B =
    # pagesim service incl. internal channel queueing)
    media_service: float = 0.0
    # per-WIN breakdown for object-aware submissions (win -> bytes/total)
    per_win: dict = field(default_factory=dict)
    # Object-path schema hisim.hbf_io.v1. Times are unrounded simulation
    # seconds; child WIN IDs refer to the actual resource that served I/O.
    trace_details: dict = field(default_factory=dict)


class WinResource:
    """One WIN: independent media busy cursor + queue + stats.

    Media within a WIN is SERIAL (reads and writes contend — a large
    write at 10 GB/s blocks subsequent reads on the same WIN).
    """

    def __init__(self, win_id: int, spec: WinSpec,
                 layer: str, pagesim_cfg: Optional[PagesimAssumptions]):
        self.win_id = win_id
        self.spec = spec
        self.layer = layer
        self.media_busy_until = 0.0
        # Object registry entries: object_id -> (base_block, next_block)
        self.stream_next: dict = {}
        self.used_bytes = 0
        self.io_count = 0
        self.n_queue = 0
        # Stats
        self.bytes_read_seq = 0
        self.bytes_read_rand = 0
        self.bytes_written = 0
        self.media_service_sum = 0.0
        self.queue_wait_sum = 0.0
        self.queue_wait_max = 0.0
        self.n_interface_constrained = 0
        # Layer B per-WIN PageSim (assumption geometry)
        self.pagesim = None
        if layer == "pagesim":
            from hisim.simulation.hbf_page_sim import PageLevelMediaSim
            self.pagesim = PageLevelMediaSim(
                channels=pagesim_cfg.channels,
                planes_per_channel=pagesim_cfg.planes_per_channel,
                page_bytes=pagesim_cfg.page_bytes,
                read_latency=pagesim_cfg.read_latency_s,
                write_latency=pagesim_cfg.write_latency_s,
                channel_rate=pagesim_cfg.channel_rate,
            )

    # -- envelope service time (layer A) --

    def envelope_service_time(self, bytes_: int, is_read: bool,
                              mode: str) -> float:
        if is_read:
            bw = self.spec.seq_read_bw if mode == "sequential" \
                else self.spec.rand_read_bw
            lat = self.spec.read_latency_s
        else:
            bw = self.spec.write_bw
            lat = self.spec.write_latency_s
        transfer = bytes_ / bw if bw > 0 else 0.0
        return max(lat, transfer)

    def service_time(self, bytes_: int, is_read: bool, mode: str,
                     start: float) -> float:
        if self.layer == "pagesim" and self.pagesim is not None:
            # Layer B: geometry-based (mode-agnostic — seq/rand distinction
            # is an envelope-level concept; layer B is for loss analysis).
            return self.pagesim.submit(start, bytes_, is_read=is_read).media_time
        return self.envelope_service_time(bytes_, is_read, mode)

    def media_cursor(self) -> float:
        if self.layer == "pagesim" and self.pagesim is not None:
            return self.pagesim.media_busy_cursor()
        return self.media_busy_until

    def record_stats(self, st: WinIoStats) -> None:
        self.io_count += 1
        if st.is_read:
            if st.mode == "sequential":
                self.bytes_read_seq += st.bytes
            else:
                self.bytes_read_rand += st.bytes
        else:
            self.bytes_written += st.bytes
        self.media_service_sum += st.media_service
        self.queue_wait_sum += st.queue_wait
        self.queue_wait_max = max(self.queue_wait_max, st.queue_wait)
        if st.interface_constrained:
            self.n_interface_constrained += 1


# ---------------------------------------------------------------------------
# The 8-WIN pool model
# ---------------------------------------------------------------------------

class HbfWinPoolModel:
    """WIN-aware HBF pool: 8 independent WINs + optional shared interface.

    submit() routes each I/O to a WIN (object registry / placement policy),
    serves it on that WIN's media (serial within WIN, parallel across WINs),
    optionally traverses the interface (per-WIN or package-shared), and
    returns per-I/O timing breakdown.

    Cross-batch safety: begin_batch(clock) records batch start; busy
    cursors (per WIN + shared interface) are PRESERVED if in-flight.
    """

    def __init__(self, config: Optional[Hbf8WinConfig] = None):
        self.config = config or Hbf8WinConfig()
        self.config.validate()
        n = self.config.n_wins
        self.wins: List[WinResource] = [
            WinResource(i, self.config.win, self.config.layer,
                        self.config.pagesim_assumptions)
            for i in range(n)
        ]
        # Package-level shared interface cursor(s)
        self.shared_ic_busy_until = 0.0
        self.shared_ic_read_busy_until = 0.0
        self.shared_ic_write_busy_until = 0.0
        # Per-WIN interface cursors (when per_win_interface_bw set)
        self.win_ic_busy_until = [0.0] * n
        # Object registry: object_id -> win_id
        self._objects: dict = {}
        self._object_sizes: dict = {}  # legacy submit/alloc capacity only
        self._io_events: list = []     # complete object I/O history, not live set
        self._next_logical_io_id = 1
        # Round-robin cursor for balanced placement
        self._rr = 0
        # Phase 5C P3: active-WIN limit for routing (capacity pool stays 8;
        # set_active_wins(k) restricts I/O routing to the first k WINs)
        self._active_limit = len(self.wins)
        self.batch_start = 0.0
        self.max_completion = 0.0
        self.total_ios = 0
        self._lock = threading.RLock()

    # ---- lifecycle ----

    def set_active_wins(self, k: int) -> None:
        """Restrict I/O routing to the first k WINs (default: all).

        The physical pool (capacity, aggregate caps) is unchanged — only
        routing/placement narrows. Used by Phase 5C P3 to study active-WIN
        efficiency with a fixed 8-WIN pool denominator.
        """
        with self._lock:
            self._active_limit = max(1, min(k, len(self.wins)))

    # ------------------------------------------------------------------
    # Phase 5D P5: external interface bridge (win8 + SharedDownstream)
    # ------------------------------------------------------------------

    def attach_external_interface(self, get_busy, update_busy, bw: float):
        """Bridge the WIN pool's package interface onto an EXTERNAL shared
        cursor (the SharedDownstream resource). All WIN I/Os then contend
        with DRAM transfers on the same downstream cursor (correct
        composition — a single shared resource, no double modeling).
        Overrides package_shared_interface_bw."""
        self._ext_interface = (get_busy, update_busy, bw)

    @property
    def active_win_count(self) -> int:
        """Number of WINs that actually served ≥1 I/O this batch."""
        return sum(1 for w in self.wins if w.io_count > 0)

    def begin_batch(self, clock: float) -> None:
        """Record batch start. Busy cursors are NOT reset (in-flight I/O
        keeps WINs / interface occupied — cross-batch continuity)."""
        self.batch_start = clock

    def reset_state(self) -> None:
        """Full reset (tests only)."""
        for w in self.wins:
            w.media_busy_until = 0.0
            w.stream_next.clear()
            w.used_bytes = 0
            if w.pagesim is not None:
                w.pagesim.reset_state()
        self.shared_ic_busy_until = 0.0
        self.shared_ic_read_busy_until = 0.0
        self.shared_ic_write_busy_until = 0.0
        self.win_ic_busy_until = [0.0] * len(self.wins)
        self._objects.clear()
        self._object_sizes.clear()
        self._io_events.clear()
        self._next_logical_io_id = 1
        if hasattr(self, "registry"):
            self.registry = WinObjectRegistry(len(self.wins))
        self._rr = 0
        self.batch_start = 0.0
        self.max_completion = 0.0
        self.total_ios = 0

    # ---- capacity / placement ----

    def _pick_win_balanced(self, size_bytes: int) -> int:
        """Round-robin over active WINs, skipping full ones."""
        n = self._active_limit
        for _ in range(n):
            w = self.wins[self._rr % n]
            self._rr += 1
            if w.used_bytes + size_bytes <= w.spec.capacity_bytes:
                return w.win_id
        return -1  # pool full

    def _pick_win_hotspot(self, size_bytes: int) -> int:
        """Concentrate on hotspot_win until full, then spill to next
        (within the active-WIN limit)."""
        n = self._active_limit
        start = self.config.hotspot_win % n
        for k in range(n):
            w = self.wins[(start + k) % n]
            if w.used_bytes + size_bytes <= w.spec.capacity_bytes:
                return w.win_id
        return -1

    def alloc_object(self, object_id, size_bytes: int,
                     win_id: Optional[int] = None) -> int:
        """Allocate an object's capacity on a WIN. Returns win_id or -1."""
        with self._lock:
            if hasattr(self, "registry"):
                self._sync_registry_capacity()
            if object_id in self._objects:
                return self._objects[object_id]
            if win_id is not None:
                w = self.wins[win_id]
                if w.used_bytes + size_bytes > w.spec.capacity_bytes:
                    return -1
            else:
                if self.config.placement == "hotspot":
                    win_id = self._pick_win_hotspot(size_bytes)
                else:
                    win_id = self._pick_win_balanced(size_bytes)
                if win_id < 0:
                    return -1
            w = self.wins[win_id]
            w.used_bytes += size_bytes
            self._object_sizes[object_id] = size_bytes
            # stream cursor for auto sequential/random detection
            base = sum(x.spec.capacity_bytes for x in self.wins[:win_id]) \
                + w.used_bytes
            w.stream_next[object_id] = base
            self._objects[object_id] = win_id
            return win_id

    def free_object(self, object_id) -> bool:
        with self._lock:
            if hasattr(self, "registry"):
                self._sync_registry_capacity()
            win_id = self._objects.pop(object_id, None)
            if win_id is None:
                return False
            w = self.wins[win_id]
            w.stream_next.pop(object_id, None)
            w.used_bytes -= self._object_sizes.pop(object_id, 0)
            return True

    def win_of(self, object_id) -> Optional[int]:
        return self._objects.get(object_id)

    # ---- access-mode resolution ----

    def _resolve_mode(self, object_id, offset: Optional[int],
                      bytes_: int, explicit: Optional[str]) -> str:
        mode = explicit or self.config.default_access_mode
        if mode != "auto":
            return mode
        # auto: contiguous continuation of the object's stream → sequential
        w = self.wins[self._objects.get(object_id, -1)] \
            if object_id in self._objects else None
        if w is not None and offset is not None \
                and object_id in w.stream_next:
            if offset == w.stream_next[object_id]:
                w.stream_next[object_id] = offset + bytes_
                return "sequential"
        return "random"

    # ---- core submit ----

    def submit(self, arrival: float, bytes_: int, is_read: bool,
               access_mode: Optional[str] = None,
               object_id=None, offset: Optional[int] = None,
               win_id: Optional[int] = None) -> WinIoStats:
        """Submit one I/O. Placement auto-registers unknown objects
        (write allocates capacity; read of unknown object is routed by
        placement policy without capacity charge). Explicit win_id routes
        to that real resource without allocating a legacy object. It is
        used by the object registry, which owns its slot capacity ledger.
        Legacy callers omitting win_id retain their placement behavior.
        """
        with self._lock:
            # 1. route to WIN
            if win_id is not None:
                if not 0 <= win_id < len(self.wins):
                    raise ValueError(f"invalid target WIN {win_id}")
            elif object_id is not None and object_id in self._objects:
                win_id = self._objects[object_id]
            elif object_id is not None:
                win_id = self.alloc_object(
                    object_id, bytes_ if not is_read else 0)
                if win_id < 0:
                    win_id = self._pick_win_balanced(0)
                    if win_id < 0:
                        win_id = 0
                    self._objects[object_id] = win_id
            else:
                # no object identity: round-robin within active WINs
                win_id = self._rr % self._active_limit
                self._rr += 1
            win = self.wins[win_id]

            # 2. mode
            mode = self._resolve_mode(object_id, offset, bytes_, access_mode)

            # 3. media service
            # Layer A: WIN media is serial → media_start gates on the WIN's
            #   busy cursor (reads/writes contend within the WIN).
            # Layer B: per-WIN PageSim owns page/channel/plane cursors and
            #   handles cross-I/O contention INTERNALLY — passing the raw
            #   arrival (not a max-cursor gate) preserves intra-WIN channel
            #   parallelism across concurrent I/Os (P1-B1 semantics).
            if win.pagesim is not None:
                ps = win.pagesim.submit(arrival, bytes_, is_read=is_read)
                media_start = arrival
                media_completion = ps.media_completion
                media_time = ps.media_time
                media_service = ps.media_time
            else:
                media_start = max(arrival, win.media_cursor())
                svc = win.service_time(bytes_, is_read, mode, media_start)
                media_completion = media_start + svc
                media_time = media_completion - arrival
                media_service = svc
                win.media_busy_until = media_completion

            # 4. interface (optional, parallel with media — max semantics)
            # Phase 5D P5: external bridge (win8 + SharedDownstream) takes
            # precedence — a single shared resource with the DRAM path.
            ic_time = 0.0
            ic_completion = 0.0
            ic_bw = None
            ext = getattr(self, "_ext_interface", None)
            if ext is not None:
                get_busy, update_busy, ic_bw = ext
                ic_start = max(arrival, get_busy())
                ic_time = bytes_ / ic_bw
                ic_completion = ic_start + ic_time
                update_busy(ic_completion)
            elif self.config.package_shared_interface_bw is not None:
                shared = self.config.package_shared_interface_bw
                ic_bw = shared
                if self.config.interface_full_duplex:
                    cur = self.shared_ic_read_busy_until if is_read \
                        else self.shared_ic_write_busy_until
                else:
                    cur = self.shared_ic_busy_until
                ic_start = max(arrival, cur)
                ic_time = bytes_ / ic_bw
                ic_completion = ic_start + ic_time
                if self.config.interface_full_duplex:
                    if is_read:
                        self.shared_ic_read_busy_until = ic_completion
                    else:
                        self.shared_ic_write_busy_until = ic_completion
                else:
                    self.shared_ic_busy_until = ic_completion
            elif self.config.per_win_interface_bw is not None:
                per_win = self.config.per_win_interface_bw
                ic_bw = per_win
                ic_start = max(arrival, self.win_ic_busy_until[win_id])
                ic_time = bytes_ / ic_bw
                ic_completion = ic_start + ic_time
                self.win_ic_busy_until[win_id] = ic_completion

            # 5. totals
            completion = max(media_completion, ic_completion)
            st = WinIoStats(
                win=win_id, bytes=bytes_, is_read=is_read, mode=mode,
                arrival=arrival, media_start=media_start,
                media_time=media_completion - arrival,
                media_completion=media_completion,
                ic_start=ic_start if ic_time > 0 else 0.0,
                ic_time=ic_time, ic_completion=ic_completion,
                queue_wait=media_start - arrival,
                total=completion - arrival,
                interface_constrained=(ic_completion > media_completion),
                media_service=media_service,
            )
            win.record_stats(st)
            self.max_completion = max(self.max_completion, completion)
            self.total_ios += 1
            win.n_queue += 1
            return st

    # ---- reporting ----

    def get_batch_duration(self) -> float:
        return max(0.0, self.max_completion - self.batch_start)

    # ------------------------------------------------------------------
    # Phase 5D P0/P1: object-aware submission (virtual slot mapping)
    # ------------------------------------------------------------------

    def _init_registry(self):
        if not hasattr(self, "registry"):
            self.registry = WinObjectRegistry(len(self.wins))

    def _sync_registry_capacity(self):
        """Live registry slots and legacy allocations have separate ledgers.

        Resource used_bytes is their sum; reading fragments never allocates
        capacity, and freeing slots releases capacity immediately. Resource
        busy cursors deliberately persist while in-flight I/O completes.
        """
        self._init_registry()
        legacy = [0] * len(self.wins)
        for key, size in self._object_sizes.items():
            legacy[self._objects[key]] += size
        for w in self.wins:
            w.used_bytes = legacy[w.win_id] + self.registry.win_bytes[w.win_id]

    def _check_backup_capacity(self, slots, bytes_per_token, win, key=None):
        """Preflight replacement/overlap without mutating the live ledger."""
        projected = list(self.registry.win_bytes)
        replaced = self.registry.objects.get(key)
        if replaced is not None:
            for w, size in replaced.win_bytes.items():
                projected[w] -= size
        for slot in slots:
            owner = self.registry.slot_owner.get(slot)
            if owner is None or (replaced is not None and owner == key):
                continue
            old = self.registry.objects[owner]
            w = old.win if old.win >= 0 else self.registry._obj_win_of(slot, old)
            projected[w] -= old.bytes_per_token
        for slot in slots:
            w = win if win >= 0 else self.registry.win_of_slot(
                slot, self.config.stripe_block_tokens)
            projected[w] += bytes_per_token
        for w in self.wins:
            legacy = sum(size for oid, size in self._object_sizes.items()
                         if self._objects[oid] == w.win_id)
            if projected[w.win_id] + legacy > w.spec.capacity_bytes:
                raise ValueError(f"WIN {w.win_id} capacity exceeded")

    def get_io_events(self) -> list:
        """Return a detached full object-I/O history (hisim.hbf_io.v1).

        One event is one logical backup/restore, containing its real WIN
        sub-I/Os. All *_s values are unrounded simulation seconds. Extents
        are half-open virtual token-slot ranges, not NAND physical LBAs.
        Media/interface max overlap remains a modeling assumption; no
        controller/FTL/GC calibration is implied by these observations.
        """
        with self._lock:
            return copy.deepcopy(self._io_events)

    def submit_object(self, arrival: float, slots, bytes_per_token: int,
                      is_read: bool, object_key=None,
                      metadata: Optional[dict] = None) -> WinIoStats:
        """Object-aware I/O: decompose `slots` per WIN by placement mode and
        submit REAL per-WIN I/Os to the independent resource queues.

        is_read=False (backup): registers the object (BACKING_UP → READY at
        completion). is_read=True (restore): verifies readiness (G4) and
        decomposes with the SAME mapping as the backup (G1).

        Returns an AGGREGATED WinIoStats (completion = max over per-WIN
        I/Os; bytes = total; per-WIN stats recorded on each WIN).
        """
        with self._lock:
            return self._submit_object(arrival, slots, bytes_per_token,
                                       is_read, object_key, metadata)

    def _submit_object(self, arrival, slots, bytes_per_token, is_read,
                       object_key, metadata=None):
        self._init_registry()
        self._sync_registry_capacity()
        slots = sorted(set(int(s) for s in slots))
        total_bytes = len(slots) * bytes_per_token
        mode = self.config.placement
        orig_arrival = arrival
        logical_io_id = self._next_logical_io_id
        self._next_logical_io_id += 1

        # ---- placement decision & registry update (backup only) ----
        if not is_read:
            if mode == "striped":
                # static mapping: register for state/conservation accounting
                # with per-WIN decomposition (obj.win = -1 = multi-WIN)
                key = object_key
                self._check_backup_capacity(slots, bytes_per_token, -1, key)
                obj = self.registry.register_backup(
                    slots, bytes_per_token, win=-1, key=key,
                    striped_block_tokens=self.config.stripe_block_tokens)
            else:
                previous = self.registry.objects.get(object_key)
                win = (previous.win if previous is not None and previous.win >= 0
                       else self.config.hotspot_win if mode == "hotspot"
                       else self.registry.pick_win_balanced_bytes())
                self._check_backup_capacity(slots, bytes_per_token,
                                            win, object_key)
                obj = self.registry.register_backup(
                    slots, bytes_per_token, win=win, key=object_key)
            keys = {obj.key}
            mapped_slots = slots
            self._sync_registry_capacity()
        else:
            # restore: readiness gate (P0 G4 — TIME-based, since the
            # synchronous model computes future completions at submit).
            # Phase 5E.0 audit #6 fix: WAIT-then-read semantics — a restore
            # arriving before backup completion queues behind the in-flight
            # backup and the wait is INCLUDED in the returned duration.
            # Never a zero-duration false fast restore.
            mapped_slots = [s for s in slots if s in self.registry.slot_owner]
            ok, keys, ready_time = self.registry.begin_restore(
                mapped_slots, arrival)
            if not ok:
                if ready_time and ready_time > arrival:
                    eff_arrival = ready_time
                else:
                    # BACKING_UP without a computed completion yet: bound
                    # by the pool's last known completion (conservative)
                    eff_arrival = max(arrival, self.max_completion)
                arrival = eff_arrival

        # ---- decompose per WIN (G1: same mapping for backup & restore) ----
        chunks = self.registry.decompose(mapped_slots, mode,
                                         self.config.stripe_block_tokens)
        object_refs = [{"key": k, "generation": self.registry.objects[k].gen}
                       for k in sorted(keys, key=str)
                       if k in self.registry.objects]

        # ---- submit REAL per-WIN I/Os ----
        per_win_stats = []
        sub_ios = []
        base_mode = self.config.default_access_mode
        for win, runs in sorted(chunks.items()):
            run_tokens = sum(e - s for s, e in runs)
            if run_tokens <= 0:
                continue
            chunk_bytes = run_tokens * bytes_per_token
            # Phase 5D P6 / 5E.1: access mode by ACTUAL physical contiguity
            # (per-WIN contiguous runs) when mode is "auto"; an explicit
            # config mode (sequential/random) overrides — the boundary
            # study restores the SAME object under both envelopes.
            if base_mode == "auto":
                chunk_mode = "sequential" if len(runs) == 1 else "random"
            else:
                chunk_mode = base_mode
            st = self.submit(arrival, chunk_bytes, is_read=is_read,
                             access_mode=chunk_mode,
                             win_id=win)
            per_win_stats.append((win, st))
            sub_ios.append({
                "sub_io_index": len(sub_ios), "win": st.win,
                "bytes": st.bytes, "is_read": is_read,
                "object_refs": copy.deepcopy(object_refs),
                "access_mode": st.mode,
                "arrival_s": st.arrival, "media_start_s": st.media_start,
                "media_completion_s": st.media_completion,
                "ic_start_s": st.ic_start,
                "ic_completion_s": st.ic_completion,
                "completion_s": max(st.media_completion, st.ic_completion),
                "queue_wait_s": st.queue_wait,
                "media_service_s": st.media_service,
                "interface_transfer_s": st.ic_time,
                "interface_constrained": st.interface_constrained,
                "slots": [s for s in mapped_slots
                          if any(a <= s < b for a, b in runs)],
                "extents": [[a, b] for a, b in runs],
            })

        # ---- aggregate (total measured from the ORIGINAL arrival: any
        # G4 wait for an in-flight backup is included — honest latency) ----
        if not per_win_stats:
            media_completion = ic_completion = completion = orig_arrival
            agg = WinIoStats(win=-1, bytes=total_bytes, is_read=is_read,
                             mode="empty", arrival=orig_arrival, total=0.0)
        else:
            agg = self._aggregate_object_stats(
                per_win_stats, total_bytes, is_read, orig_arrival, arrival)
            completion = orig_arrival + agg.total

        # Complete history survives eviction and slot reuse. bytes is useful
        # served bytes; requested_bytes preserves misses/partial coverage.
        useful_bytes = sum(s[1].bytes for s in per_win_stats)
        agg.trace_details = {
            "schema_version": "hisim.hbf_io.v1",
            "logical_io_id": logical_io_id,
            "object_refs": object_refs, "is_read": is_read,
            "access_mode": self.config.default_access_mode,
            "bytes": useful_bytes, "requested_bytes": total_bytes,
            "status": ("complete" if useful_bytes == total_bytes else
                       "partial_miss" if useful_bytes else "miss"),
            "arrival_s": orig_arrival, "effective_arrival_s": arrival,
            "completion_s": completion,
            "dependency_wait_s": max(0.0, arrival - orig_arrival),
            "slots": slots,
            "extents": {str(w): [[a, b] for a, b in runs]
                        for w, runs in chunks.items()},
            "sub_ios": sub_ios,
            "assumptions": {
                "time_unit": "simulation seconds, unrounded",
                "address_unit": "virtual token slots; extents are [start,end)",
                "layer": self.config.layer,
                "media_interface_overlap": "max(media,interface)",
                "queue_depth": "no internal closed-loop admission; external workload supplies demand",
            },
        }
        if metadata is not None:
            # Provenance is nested so callers cannot overwrite timing/status
            # or payload accounting. Store before the detached history copy.
            agg.trace_details["metadata"] = copy.deepcopy(metadata)
        self._io_events.append(copy.deepcopy(agg.trace_details))

        if not is_read:
            self.registry.mark_ready(obj.key, completion)
            obj.n_writes += 1
        else:
            self.registry.end_restore(keys)
        self.max_completion = max(self.max_completion, completion)
        return agg

    def _aggregate_object_stats(self, per_win_stats, total_bytes, is_read,
                                orig_arrival, arrival):
        media_completion = max(s.media_completion for _, s in per_win_stats)
        ic_completion = max((s.ic_completion for _, s in per_win_stats),
                            default=0.0)
        completion = max(media_completion, ic_completion)
        g4_wait = max(0.0, arrival - orig_arrival)
        agg = WinIoStats(
            win=per_win_stats[0][0], bytes=total_bytes, is_read=is_read,
            mode=per_win_stats[0][1].mode, arrival=orig_arrival,
            media_start=min(s.media_start for _, s in per_win_stats),
            media_time=media_completion - arrival,
            media_completion=media_completion,
            ic_time=max((s.ic_time for _, s in per_win_stats), default=0.0),
            ic_completion=ic_completion,
            queue_wait=max((s.queue_wait for _, s in per_win_stats),
                           default=0.0) + g4_wait,
            total=completion - orig_arrival,
            interface_constrained=any(s.interface_constrained
                                      for _, s in per_win_stats),
            media_service=max((s.media_service for _, s in per_win_stats),
                              default=0.0),
        )
        agg.per_win = {w: {"bytes": s.bytes, "total": s.total}
                       for w, s in per_win_stats}

        return agg

    def notify_slots_freed(self, slots) -> list:
        """Adapter eviction hook → registry invalidation (P0 G3)."""
        self._init_registry()
        with self._lock:
            evicted = self.registry.notify_freed(slots)
            self._sync_registry_capacity()
            return evicted

    def object_summary(self) -> dict:
        self._init_registry()
        return self.registry.summary()

    def media_cursor_max(self) -> float:
        return max(w.media_cursor() for w in self.wins) if self.wins else 0.0

    def per_win_summary(self) -> list:
        if hasattr(self, "registry"):
            self._sync_registry_capacity()
        out = []
        for w in self.wins:
            out.append({
                "win": w.win_id,
                "used_gb": round(w.used_bytes / GB, 3),
                "io_count": w.io_count,
                "bytes_read_seq_gb": round(w.bytes_read_seq / GB, 3),
                "bytes_read_rand_gb": round(w.bytes_read_rand / GB, 3),
                "bytes_written_gb": round(w.bytes_written / GB, 3),
                "media_busy_s": round(
                    (w.media_cursor() - self.batch_start)
                    if w.media_cursor() > 0 else 0.0, 6),
                # Compatibility media_busy_s is a cursor span, including
                # idle time. Actual served time is cumulative service sum.
                "media_cursor_span_s": max(0.0, w.media_cursor() - self.batch_start),
                "media_service_time_s": w.media_service_sum,
                "media_service_sum_s": round(w.media_service_sum, 6),
                "queue_wait_max_us": round(w.queue_wait_max * 1e6, 1),
                "n_interface_constrained": w.n_interface_constrained,
            })
        return out

    def summary(self) -> dict:
        total_read = sum(w.bytes_read_seq + w.bytes_read_rand
                         for w in self.wins)
        total_write = sum(w.bytes_written for w in self.wins)
        span = self.get_batch_duration()
        return {
            "schema_version": "hisim.hbf_pool_summary.v2",
            "metric_definitions": {
                "media_busy_s": "legacy alias of cursor span; includes idle time; not utilization",
                "media_cursor_span_s": "max(0, media completion cursor - batch_start)",
                "media_service_time_s": "cumulative modeled service seconds; excludes resource queue wait",
                "read_throughput_gbps": "cumulative useful read bytes / cursor-span observation window; not saturation efficiency",
                "layer_b_service_time": "PageSim completion-arrival includes internal channel/plane waits; not physical busy time",
            },
            "n_wins": len(self.wins),
            "layer": self.config.layer,
            "placement": self.config.placement,
            "total_ios": self.total_ios,
            "batch_span_s": round(span, 6),
            "total_read_gb": round(total_read / GB, 3),
            "total_write_gb": round(total_write / GB, 3),
            "read_throughput_gbps": round(total_read / span / GB, 2)
            if span > 0 else 0.0,
            "write_throughput_gbps": round(total_write / span / GB, 2)
            if span > 0 else 0.0,
            "aggregate_cap_read_gbps": round(
                self.config.aggregate_seq_read_bw / GB, 1),
            "aggregate_cap_write_gbps": round(
                self.config.aggregate_write_bw / GB, 1),
            "shared_interface_bw_gbps":
                round(self.config.package_shared_interface_bw / GB, 1)
                if self.config.package_shared_interface_bw else None,
            "per_win": self.per_win_summary(),
        }


# ===========================================================================
# Phase 5D P0: KV Object Registry — stable Object ID → WIN mapping
#
# The stable object address is the HBF VIRTUAL SLOT index space
# [0, dram_offset) managed by TieredHostKVCache:
#   - Backup (mock_start_writing) carries op.host_indices → the allocated
#     virtual slots of the object being written.
#   - Restore (HOST-LOAD / _collect_host_load_split) reads the SAME slots
#     from the radix tree nodes' host_value.
#   - Eviction (TieredHostKVCache.free) frees slots → registry invalidation
#     via the adapter's _on_hbf_free callback.
#
# Placement modes (Phase 5D P1):
#   "balanced"  — object-level: whole object → ONE WIN (registry-assigned,
#                 round-robin by cumulative bytes). Backup/Restore hit the
#                 same WIN by registry lookup.
#   "striped"   — block-level: static win(slot) = (slot // block) % n_wins.
#                 Any slot range decomposes into per-WIN chunks submitted to
#                 the 8 INDEPENDENT resource queues (real submissions, not
#                 time-division). Restore decomposes identically → same WINs.
#   "hotspot"   — negative control: everything on hotspot_win.
#
# Guarantees (Phase 5D P0 requirements, each locked by unit tests):
#   G1. Same object Backup/Restore → same WIN(s).
#   G2. WIN migration only via explicit re-register (migrate_object).
#   G3. Free/eviction/CoW removes mappings (no stale entries).
#   G4. Restore cannot hit an object whose backup has not completed
#       (state machine: BACKING_UP → READY at ready_time; restore asserts
#       arrival ≥ ready_time).
#   G5. Cross-batch resource cursors preserved (existing semantics).
#   G6. Token / object-bytes / capacity conservation (verify_conservation).
# ===========================================================================

OBJ_STATES = ("ALLOCATED", "BACKING_UP", "READY", "RESTORING", "EVICTED")


@dataclass
class WinObject:
    """One KV object hosted on the WIN pool."""
    key: int                          # stable object id (generation counter)
    gen: int                          # generation (bumped on re-register)
    slots: tuple                      # sorted virtual slot indices
    size_tokens: int
    bytes_per_token: int
    win: int = -1                     # object-level WIN (-1 = striped/multi)
    win_bytes: dict = field(default_factory=dict)  # per-WIN hosted bytes
    extents: dict = field(default_factory=dict)   # win -> [(s,e), ...] runs
    state: str = "ALLOCATED"
    backup_ready_time: float = 0.0    # completion of the backup I/O
    n_reads: int = 0
    n_writes: int = 0
    fragmented_runs: int = 1


def _contiguous_runs(sorted_slots) -> list:
    """Split sorted slot indices into contiguous [start, end) runs."""
    runs = []
    if not len(sorted_slots):
        return runs
    start = prev = int(sorted_slots[0])
    for s in sorted_slots[1:]:
        s = int(s)
        if s == prev + 1:
            prev = s
            continue
        runs.append((start, prev + 1))
        start = prev = s
    runs.append((start, prev + 1))
    return runs


class WinObjectRegistry:
    """Object ID → WIN mapping with state machine and conservation."""

    def __init__(self, n_wins: int):
        self.n_wins = n_wins
        self.objects: dict = {}          # key -> WinObject
        self.slot_owner: dict = {}       # slot -> key (object-level modes)
        self._next_key = 1
        self._generations = {}          # survives slot release/reuse
        self.n_migrations = 0
        self.n_evictions = 0
        self.n_stale_blocked_restores = 0        # restores blocked by not-ready
        # external invalidation hook set by the adapter (P0 G3)
        # bytes per WIN accounting (object-bytes, from the registry view)
        self.win_bytes = [0] * n_wins

    # ---- lifecycle ----

    def register_backup(self, slots, bytes_per_token: int,
                        win: int, key=None,
                        striped_block_tokens: int = None) -> WinObject:
        """Register an object entering BACKING_UP.

        win >= 0: object-level placement (single WIN).
        win < 0 with striped_block_tokens: striped placement — per-WIN
        byte accounting is decomposed via the static block map and
        obj.win stays -1 (multi-WIN object)."""
        slots = tuple(sorted(set(int(s) for s in slots)))
        if key is None:
            while self._next_key in self._generations:
                self._next_key += 1
            key = self._next_key
            self._next_key += 1
        gen = self._generations.get(key, 0) + 1
        # Replacement retires its previous extents. Incoming slots can also
        # overlap older objects after host slot reuse; shrink those owners
        # before charging the new generation, never count the bytes twice.
        previous = self.objects.get(key)
        if previous is not None:
            self.notify_freed(previous.slots, count_evictions=False)
        self.notify_freed(slots, count_evictions=False)
        self._generations[key] = gen
        obj = WinObject(key=key, gen=gen, slots=slots,
                        size_tokens=len(slots),
                        bytes_per_token=bytes_per_token, win=win,
                        state="BACKING_UP")
        if win >= 0:
            obj.win_bytes = {win: len(slots) * bytes_per_token}
            obj.extents = {win: _contiguous_runs(slots)}
            obj.fragmented_runs = len(obj.extents[win])
        else:
            # striped: per-WIN decomposition for accounting/extents
            by_win = {}
            for s in slots:
                w = self.win_of_slot(s, striped_block_tokens)
                by_win.setdefault(w, []).append(s)
            obj.win_bytes = {w: sum(e - s for s, e in _contiguous_runs(ss))
                             * bytes_per_token
                             for w, ss in by_win.items()}
            obj.extents = {w: _contiguous_runs(ss)
                           for w, ss in by_win.items()}
            obj.fragmented_runs = sum(len(r) for r in obj.extents.values())
        self.objects[key] = obj
        for s in slots:
            self.slot_owner[s] = key
        for w, b in obj.win_bytes.items():
            self.win_bytes[w] += b
        return obj

    def mark_ready(self, key: int, ready_time: float) -> None:
        obj = self.objects.get(key)
        if obj is not None and obj.state == "BACKING_UP":
            obj.state = "READY"
            obj.backup_ready_time = ready_time

    def begin_restore(self, slots, arrival: float = None) -> tuple:
        """Readiness gate (P0 G4): every object covering `slots` must be
        READY and its backup_ready_time ≤ arrival. In the synchronous
        model a backup's completion can be in the future even though the
        submit already returned — the TIME check is what matters.

        A not-ready restore returns the latest dependency's
        ready_time so the caller can WAIT (wait-then-read semantics —
        never a zero-duration false fast restore).

        Returns (ok, covered_keys, latest_ready_time). All dependencies
        must finish before the logical restore is admitted.
        """
        keys = {self.slot_owner.get(int(s)) for s in slots}
        keys.discard(None)
        latest_ready = None
        for k in keys:
            obj = self.objects.get(k)
            if obj is None:
                continue
            not_ready = (obj.state == "BACKING_UP"
                         or (arrival is not None
                             and obj.state == "READY"
                             and obj.backup_ready_time > arrival))
            if not_ready:
                # READY-but-future → its ready_time; BACKING_UP with
                # ready_time unset → 0.0 (caller bounds by pool completion)
                rt = obj.backup_ready_time if obj.state == "READY" else 0.0
                latest_ready = rt if latest_ready is None else max(latest_ready, rt)
        if latest_ready is not None:
            self.n_stale_blocked_restores += 1
            return False, keys, latest_ready
        return True, keys, None

    def end_restore(self, keys) -> None:
        for k in keys:
            obj = self.objects.get(k)
            if obj is not None:
                obj.n_reads += 1

    def notify_freed(self, slots, count_evictions=True) -> list:
        """Slots freed by the host pool (eviction/invalidation/CoW).
        Removes fully-freed objects; partial frees shrink the object.
        Returns evicted object keys (P0 G3)."""
        evicted = []
        touched = {}
        for s in slots:
            k = self.slot_owner.pop(int(s), None)
            if k is not None:
                touched[k] = touched.get(k, 0) + 1
        for k, n_freed in touched.items():
            obj = self.objects.get(k)
            if obj is None:
                continue
            if n_freed >= obj.size_tokens:
                # whole object gone: subtract per-WIN bytes
                for w, b in obj.win_bytes.items():
                    self.win_bytes[w] -= b
                obj.state = "EVICTED"
                evicted.append(k)
                del self.objects[k]
                if count_evictions:
                    self.n_evictions += 1
            else:
                # partial free (CoW split): shrink the object in place
                remaining = [s for s in obj.slots
                             if self.slot_owner.get(s) == k]
                obj.slots = tuple(remaining)
                obj.size_tokens = len(remaining)
                # recompute extents per WIN from remaining slots
                by_win = {}
                for s in remaining:
                    w = obj.win if obj.win >= 0 else self._obj_win_of(s, obj)
                    by_win.setdefault(w, []).append(s)
                obj.extents = {w: _contiguous_runs(ss)
                               for w, ss in by_win.items()}
                new_bytes = {w: sum(e - s for s, e in runs)
                             * obj.bytes_per_token
                             for w, runs in obj.extents.items()}
                for w, b in obj.win_bytes.items():
                    self.win_bytes[w] -= b
                for w, b in new_bytes.items():
                    self.win_bytes[w] += b
                obj.win_bytes = new_bytes
                obj.fragmented_runs = max(1, sum(
                    len(r) for r in obj.extents.values()))
        return evicted

    def _obj_win_of(self, slot, obj):
        """WIN of a slot for a striped object (block map from extents)."""
        for w, runs in obj.extents.items():
            if any(s <= slot < e for s, e in runs):
                return w
        return 0

    def migrate_object(self, key: int, new_win: int) -> bool:
        """Explicit WIN migration (P0 G2). Only registry accounting moves;
        data movement timing is the caller's responsibility."""
        obj = self.objects.get(key)
        if obj is None or new_win < 0 or new_win >= self.n_wins:
            return False
        if obj.win < 0:
            return False   # striped objects migrate per-slot, not whole
        for w, b in obj.win_bytes.items():
            self.win_bytes[w] -= b
        obj.win = new_win
        obj.win_bytes = {new_win: obj.size_tokens * obj.bytes_per_token}
        obj.extents = {new_win: _contiguous_runs(obj.slots)}
        self.win_bytes[new_win] += obj.size_tokens * obj.bytes_per_token
        self.n_migrations += 1
        return True

    # ---- mapping queries ----

    def win_of_slot(self, slot: int, block_tokens: int) -> int:
        """Static striped mapping: block b → WIN b % n_wins."""
        return (int(slot) // block_tokens) % self.n_wins

    def decompose(self, slots, mode: str, block_tokens: int) -> dict:
        """Split slots into per-WIN chunks.

        mode "striped": static block mapping (no registry lookup).
        mode "balanced"/"hotspot": registry lookup (object's WIN).
        Returns {win: [(run_start, run_end), ...]} with per-run token counts
        derived from run lengths.
        """
        out = {}
        if mode == "striped":
            by_win = {}
            for s in sorted(int(x) for x in slots):
                by_win.setdefault(
                    self.win_of_slot(s, block_tokens), []).append(s)
            for w, ws in by_win.items():
                out[w] = _contiguous_runs(ws)
        else:
            keys = {self.slot_owner.get(int(s)) for s in slots}
            keys.discard(None)
            for k in keys:
                obj = self.objects.get(k)
                if obj is None:
                    continue
                out.setdefault(obj.win, []).extend(
                    (a, b) for a, b in _contiguous_runs(
                        [s for s in slots
                         if self.slot_owner.get(int(s)) == k]))
        return out

    def verify_conservation(self) -> dict:
        """P0 G6: registry bytes per WIN vs slot accounting."""
        total_slots = sum(o.size_tokens for o in self.objects.values())
        total_bytes = sum(self.win_bytes)
        accounted = sum(sum(o.win_bytes.values())
                        for o in self.objects.values())
        return {
            "n_objects": len(self.objects),
            "total_slots": total_slots,
            "slot_index_entries": len(self.slot_owner),
            "win_bytes": list(self.win_bytes),
            "total_bytes": total_bytes,
            "accounted_bytes": accounted,
            "bytes_consistent": total_bytes == accounted,
            "slots_consistent": total_slots == len(self.slot_owner),
            "states": {s: sum(1 for o in self.objects.values()
                              if o.state == s) for s in OBJ_STATES},
        }

    def pick_win_balanced_bytes(self) -> int:
        """Least-bytes WIN (object-level balanced placement)."""
        return min(range(self.n_wins), key=lambda w: self.win_bytes[w])

    def summary(self) -> dict:
        return {
            "n_objects": len(self.objects),
            "n_evictions": self.n_evictions,
            "n_migrations": self.n_migrations,
            "n_stale_blocked_restores": self.n_stale_blocked_restores,
            "win_bytes": list(self.win_bytes),
            "fragmented_objects": sum(
                1 for o in self.objects.values() if o.fragmented_runs > 1),
            "reads_total": sum(o.n_reads for o in self.objects.values()),
        }
