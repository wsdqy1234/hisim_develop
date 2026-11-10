"""Phase 3D-P1: Shared HBF Resource Manager.

Models shared HBF Media and HBF→HBM Interconnect bandwidth across
concurrent I/O requests within the same simulation batch.

Two shared resources (both serial within resource, parallel across):
  1. Media: channel aggregate BW (51.2 GB/s) — NAND read/program
  2. Interconnect: HBF→HBM link BW (configurable) — data transfer

Pipeline model: media and interconnect process different I/Os concurrently
(while interconnect transfers I/O[i], media serves I/O[i+1]).
For a single I/O: duration = max(media_time, ic_time) — matches Phase 3D-P0.
For N concurrent I/Os: resources are shared → each I/O queues on both.

Invariants:
  - Concurrency=1: result matches Phase 3D-P0 baseline
  - Aggregate throughput ≤ min(media_bw, ic_bw)
  - Queue wait, completion time ≥ 0
  - Reads and writes share the same resources
"""
import os
import threading
from dataclasses import dataclass, field
from typing import Optional


# ---- Phase 3E-Closure: Shared Downstream Interconnect (topology option) ----
# When HISIM_SHARED_DOWNSTREAM_BW_GB is set, HBF interconnect and DRAM
# transfers share a common downstream Host→HBM resource. This models the
# case where both media feed into a shared DMA/PCIe path to HBM.
# Default (unset): HBF and DRAM use INDEPENDENT Host→HBM resources.
_downstream_busy_until: float = 0.0
_shared_downstream_bw: float = 0.0  # 0 = disabled (independent topology)


def configure_shared_downstream(bw_bytes_per_s: float) -> None:
    """Enable shared downstream interconnect with given bandwidth."""
    global _shared_downstream_bw
    _shared_downstream_bw = bw_bytes_per_s


def get_downstream_busy_until() -> float:
    return _downstream_busy_until


def update_downstream_busy_until(t: float) -> None:
    global _downstream_busy_until
    _downstream_busy_until = max(_downstream_busy_until, t)


def reset_downstream(clock: float) -> None:
    global _downstream_busy_until
    _downstream_busy_until = max(_downstream_busy_until, clock)


def is_shared_downstream_enabled() -> bool:
    return _shared_downstream_bw > 0


@dataclass
class HbfIoStats:
    """Per-I/O statistics."""
    arrival: float = 0.0          # when I/O was submitted (sim clock, seconds)
    bytes: int = 0                # I/O size in bytes
    is_read: bool = True
    media_start: float = 0.0      # when media started serving
    media_time: float = 0.0       # media service duration
    media_completion: float = 0.0 # when media finished
    ic_start: float = 0.0         # when interconnect started
    ic_time: float = 0.0          # interconnect transfer duration
    ic_completion: float = 0.0    # when interconnect finished
    queue_wait: float = 0.0       # admission queue wait
    total: float = 0.0            # ic_completion - arrival


class SharedHbfResourceManager:
    """Shared bandwidth model for HBF Media + Interconnect.

    Tracks busy-until cursors for media and interconnect resources.
    Each I/O queues on both resources (serial within, parallel across).

    Phase 4B: Includes page-level I/O granularity modeling:
    - Small I/O (< 1 page): latency-bound by page read/program latency
    - Large I/O (≥ 512KB): bandwidth-bound by channel aggregate BW
    - Page splitting across channels and planes
    """

    _instance: Optional["SharedHbfResourceManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SharedHbfResourceManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton (for testing or batch reset)."""
        if cls._instance is not None:
            cls._instance.reset()

    def __init__(self):
        self.media_bw: float = 51.2e9   # 51.2 GB/s default (channel aggregate)
        self.interconnect_bw: float = 51.2e9  # default = no constraint
        self.queue_depth: int = 128
        # Phase 4B: page-level geometry for I/O size sensitivity
        self.page_bytes: int = 16384         # 16KB NAND page
        self.channels: int = 32
        self.planes_per_channel: int = 32    # 8 dies × 4 planes
        self.read_latency: float = 10e-6     # 10μs page read
        self.write_latency: float = 100e-6   # 100μs page program
        # Phase 4-Closure P1: media model selection
        #   "formula" — closed-form max(transfer, page_latency) [Theory Bound]
        #   "pagesim" — discrete-event page/plane/channel simulator [Model Behavior]
        # Phase 5B: "win8" — 8-WIN pool model (per-WIN envelope/pagesim,
        #   optional per-WIN or package-shared interface; the WIN pool owns
        #   BOTH media and interface — the manager's IC path is bypassed).
        self.media_model: str = "formula"
        self.page_sim: Optional["PageLevelMediaSim"] = None
        self.win_pool: Optional["HbfWinPoolModel"] = None
        self.media_busy_until: float = 0.0
        self.ic_busy_until: float = 0.0
        self.outstanding: int = 0
        self.admission_queue: list = []
        self.batch_start: float = 0.0
        # Stats
        self.io_stats: list[HbfIoStats] = []
        self.peak_outstanding: int = 0
        self.total_bytes_read: int = 0
        self.total_bytes_written: int = 0
        self.max_completion: float = 0.0

    def configure(self, media_bw: float, interconnect_bw: float,
                  queue_depth: int = 128, page_bytes: int = 16384,
                  channels: int = 32, planes_per_channel: int = 32,
                  read_latency: float = 10e-6, write_latency: float = 100e-6,
                  media_model: str = "formula",
                  win_config: Optional["Hbf8WinConfig"] = None):
        """Configure bandwidth and geometry parameters.

        media_model:
          - "formula": closed-form _media_service_time (default; backward
            compatible — labeled Theory Bound in Phase 4-Closure reports).
          - "pagesim": discrete-event PageLevelMediaSim (per-channel data
            path + per-plane busy cursors; labeled Model Behavior).
          - "win8": Phase 5B 8-WIN pool model. The WIN pool owns media AND
            interface timing; interconnect_bw is ignored in this mode.
        """
        self.media_bw = media_bw
        self.interconnect_bw = interconnect_bw
        self.queue_depth = queue_depth
        self.page_bytes = page_bytes
        self.channels = channels
        self.planes_per_channel = planes_per_channel
        self.read_latency = read_latency
        self.write_latency = write_latency
        self.media_model = media_model
        if media_model == "pagesim":
            from hisim.simulation.hbf_page_sim import PageLevelMediaSim
            # Per-channel data rate derived from aggregate media BW
            per_channel_rate = media_bw / max(1, channels)
            self.page_sim = PageLevelMediaSim(
                channels=channels,
                planes_per_channel=planes_per_channel,
                page_bytes=page_bytes,
                read_latency=read_latency,
                write_latency=write_latency,
                channel_rate=per_channel_rate,
            )
        else:
            self.page_sim = None
        if media_model == "win8":
            from hisim.simulation.hbf_win_pool import (
                Hbf8WinConfig, HbfWinPoolModel)
            self.win_pool = HbfWinPoolModel(win_config or Hbf8WinConfig())
        else:
            self.win_pool = None

    def _media_service_time(self, bytes_: int, is_read: bool) -> float:
        """Phase 4B: Page-level media service time.

        Small I/O (< 1 page per channel): latency-bound by page latency.
        Large I/O: bandwidth-bound by channel aggregate BW.

        Model:
        - pages = ceil(bytes / page_bytes)
        - pages_per_channel = ceil(pages / channels)
        - pages_per_plane = ceil(pages_per_channel / planes_per_channel)
        - page_latency_total = pages_per_plane × (read|write)_latency
          (sequential within plane, parallel across planes AND channels)
        - channel_transfer = (bytes/channels) / per_channel_bw
        - media_time = max(channel_transfer, page_latency_total, 1_page_latency)
        """
        if bytes_ <= 0 or self.media_bw <= 0:
            return 0.0

        page_latency = self.read_latency if is_read else self.write_latency

        # Number of pages
        pages = max(1, (bytes_ + self.page_bytes - 1) // self.page_bytes)
        # Pages distributed across channels
        pages_per_channel = max(1, (pages + self.channels - 1) // self.channels)
        # Pages distributed across planes within each channel (parallel)
        pages_per_plane = max(1, (pages_per_channel + self.planes_per_channel - 1)
                              // self.planes_per_channel)
        # Sequential page latency within a plane
        # (parallel across planes within channel, and across channels)
        page_latency_total = pages_per_plane * page_latency

        # Channel transfer time (bytes striped across channels)
        per_channel_bw = self.media_bw / self.channels
        per_channel_bytes = bytes_ / self.channels
        channel_transfer = per_channel_bytes / per_channel_bw if per_channel_bw > 0 else 0

        # Media time = max(transfer, page latency, single page latency floor)
        return max(channel_transfer, page_latency_total, page_latency)

    def shared_win_ic_cursor(self) -> float:
        """WIN-pool shared interface busy cursor (win8 mode bookkeeping)."""
        wp = self.win_pool
        if wp is None:
            return 0.0
        return max(wp.shared_ic_busy_until,
                   wp.shared_ic_read_busy_until,
                   wp.shared_ic_write_busy_until,
                   max(wp.win_ic_busy_until) if wp.win_ic_busy_until else 0.0)

    def reset(self):
        """Reset per-batch stats, preserving cross-batch resource cursors.

        Key safety: media_busy_until and ic_busy_until are set to
        max(previous_value, batch_start) — if a previous batch's I/O
        is still in flight (busy_until > batch_start), the cursor is
        preserved so the next batch's I/O correctly waits.
        """
        # Cross-batch safety: preserve cursors if I/O still in flight
        if self.media_model == "win8" and self.win_pool is not None:
            # WIN pool cursors persist on their own; mirror max cursor.
            self.media_busy_until = max(self.win_pool.media_cursor_max(),
                                        self.batch_start)
            self.ic_busy_until = max(self.shared_win_ic_cursor(),
                                     self.batch_start)
        elif self.media_model == "pagesim" and self.page_sim is not None:
            # Simulator cursors persist; media_busy_until mirrors max cursor
            self.media_busy_until = max(self.page_sim.media_busy_cursor(),
                                        self.batch_start)
        else:
            self.media_busy_until = max(self.media_busy_until, self.batch_start)
        self.ic_busy_until = max(self.ic_busy_until, self.batch_start)
        # Reset per-batch stats only
        self.outstanding = 0
        self.admission_queue.clear()
        self.io_stats.clear()
        self.peak_outstanding = 0
        self.total_bytes_read = 0
        self.total_bytes_written = 0
        self.max_completion = self.batch_start

    def begin_batch(self, clock: float):
        """Mark the start of a new batch at the given simulation clock."""
        self.batch_start = clock
        self.reset()
        if self.page_sim is not None:
            self.page_sim.begin_batch(clock)
        if self.win_pool is not None:
            self.win_pool.begin_batch(clock)

    def submit(self, arrival: float, bytes_: int, is_read: bool = True) -> HbfIoStats:
        """Submit an I/O request to the shared resource model.

        Args:
            arrival: simulation clock when I/O arrives (seconds)
            bytes_: I/O size in bytes
            is_read: True for read (load), False for write (backup)

        Returns:
            HbfIoStats with timing breakdown
        """
        stats = HbfIoStats(arrival=arrival, bytes=bytes_, is_read=is_read)

        # Track outstanding: all previous I/Os submitted at same arrival time
        # are "in flight" (their completion > arrival). In synchronous model,
        # peak_outstanding = min(total_submitted, queue_depth).
        n_submitted = len(self.io_stats)
        in_flight = min(n_submitted + 1, self.queue_depth)
        self.peak_outstanding = max(self.peak_outstanding, in_flight)

        # Admission control: if at QD limit, request waits for earliest completion
        admit_time = arrival
        if n_submitted >= self.queue_depth:
            # Wait until the earliest I/O completes (frees a slot)
            earliest_completion = min(s.ic_completion for s in self.io_stats)
            admit_time = max(arrival, earliest_completion)
            stats.queue_wait = admit_time - arrival

        if is_read:
            self.total_bytes_read += bytes_
        else:
            self.total_bytes_written += bytes_

        # Phase 5B: 8-WIN pool mode — the WIN pool owns media AND interface.
        # The manager's separate IC path is bypassed (would double-count).
        if self.media_model == "win8" and self.win_pool is not None:
            wst = self.win_pool.submit(admit_time, bytes_, is_read=is_read)
            stats.media_start = wst.media_start
            stats.media_time = wst.media_time
            stats.media_completion = wst.media_completion
            stats.ic_start = wst.ic_completion - wst.ic_time \
                if wst.ic_time > 0 else admit_time
            stats.ic_time = wst.ic_time
            stats.ic_completion = wst.ic_completion
            if wst.queue_wait > stats.queue_wait:
                stats.queue_wait = wst.queue_wait
            stats.total = max(wst.media_completion, wst.ic_completion) - arrival
            self.media_busy_until = self.win_pool.media_cursor_max()
            self.ic_busy_until = max(self.shared_win_ic_cursor(),
                                     self.batch_start)
            self.max_completion = max(self.max_completion,
                                      max(wst.media_completion,
                                          wst.ic_completion))
            self.io_stats.append(stats)
            return stats

        # Media service (shared resource, serial within)
        # Phase 4B: page-level media time (I/O size sensitive)
        # Phase 4-Closure P1: dispatch to discrete-event simulator when
        # media_model == "pagesim" (Model Behavior) vs closed-form formula
        # (Theory Bound).
        if self.media_model == "pagesim" and self.page_sim is not None:
            # The simulator owns per-channel/per-plane cursors and handles
            # cross-I/O contention internally (FIFO per submission order).
            media_start = admit_time
            ps = self.page_sim.submit(admit_time, bytes_, is_read=is_read)
            media_completion = ps.media_completion
            media_time = ps.media_time
            self.media_busy_until = self.page_sim.media_busy_cursor()
        else:
            media_start = max(admit_time, self.media_busy_until)
            media_time = self._media_service_time(bytes_, is_read)
            media_completion = media_start + media_time
            self.media_busy_until = media_completion

        # Interconnect transfer (shared resource, serial within)
        # Parallel pipeline: ic can start independently of media
        # (while media serves this I/O, ic may be transferring previous I/O)
        # For single I/O: ic_start = admit_time → total = max(media, ic)
        if is_shared_downstream_enabled():
            # Shared downstream topology: HBF and DRAM share Host→HBM path
            ic_start = max(admit_time, self.ic_busy_until, get_downstream_busy_until())
            ic_time = bytes_ / _shared_downstream_bw
            ic_completion = ic_start + ic_time
            update_downstream_busy_until(ic_completion)
        else:
            # Independent topology: HBF has its own interconnect
            ic_start = max(admit_time, self.ic_busy_until)
            ic_time = bytes_ / self.interconnect_bw if self.interconnect_bw > 0 else 0
            ic_completion = ic_start + ic_time
        self.ic_busy_until = ic_completion

        stats.media_start = media_start
        stats.media_time = media_time
        stats.media_completion = media_completion
        stats.ic_start = ic_start
        stats.ic_time = ic_time
        stats.ic_completion = ic_completion
        stats.total = max(media_completion, ic_completion) - arrival

        self.max_completion = max(self.max_completion,
                                   max(media_completion, ic_completion))

        self.io_stats.append(stats)
        return stats

    def get_batch_duration(self) -> float:
        """Get the total batch duration (max completion - batch start)."""
        if not self.io_stats:
            return 0.0
        return max(0.0, self.max_completion - self.batch_start)

    def get_stats_summary(self) -> dict:
        """Get summary statistics for the batch."""
        if not self.io_stats:
            return {"count": 0}
        reads = [s for s in self.io_stats if s.is_read]
        writes = [s for s in self.io_stats if not s.is_read]
        return {
            "count": len(self.io_stats),
            "reads": len(reads),
            "writes": len(writes),
            "peak_outstanding": self.peak_outstanding,
            "total_bytes_read": self.total_bytes_read,
            "total_bytes_written": self.total_bytes_written,
            "batch_duration": self.get_batch_duration(),
            "max_queue_wait": max(s.queue_wait for s in self.io_stats),
            "max_media_time": max(s.media_time for s in self.io_stats),
            "max_ic_time": max(s.ic_time for s in self.io_stats),
            "media_busy_until": self.media_busy_until,
            "ic_busy_until": self.ic_busy_until,
        }


def get_shared_hbf_manager() -> SharedHbfResourceManager:
    """Get the global SharedHbfResourceManager singleton."""
    return SharedHbfResourceManager.get_instance()


class SharedDramResourceManager:
    """Shared bandwidth model for DRAM Host→HBM transfers.

    Single shared resource: effective DRAM bandwidth (13.6 GB/s).
    Uses same event-driven model as HBF but with a single resource
    (media = interconnect = dram_bw).
    """

    _instance: Optional["SharedDramResourceManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SharedDramResourceManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.dram_bw: float = 13.6e9  # 16 GB/s * 0.85
        self.t0_load: float = 6.67e-6   # H2D base latency
        self.t0_backup: float = 4e-6    # D2H base latency
        self.busy_until: float = 0.0
        self.batch_start: float = 0.0
        self.io_stats: list = []
        self.peak_outstanding: int = 0
        self.max_completion: float = 0.0

    def configure(self, dram_bw: float, t0_load: float = 6.67e-6, t0_backup: float = 4e-6):
        self.dram_bw = dram_bw
        self.t0_load = t0_load
        self.t0_backup = t0_backup

    def begin_batch(self, clock: float):
        self.batch_start = clock
        # Cross-batch safety: preserve cursor if I/O still in flight
        self.busy_until = max(self.busy_until, clock)
        self.io_stats.clear()
        self.peak_outstanding = 0
        self.max_completion = clock

    def submit(self, arrival: float, bytes_: int, is_load: bool = True) -> dict:
        """Submit a DRAM I/O. Returns dict with timing breakdown."""
        if bytes_ <= 0:
            return {"total": 0.0, "queue_wait": 0.0, "service_time": 0.0,
                    "completion": arrival}

        t0 = self.t0_load if is_load else self.t0_backup

        if is_shared_downstream_enabled():
            # Shared downstream topology: DRAM shares Host→HBM path with HBF
            service_time = t0 + bytes_ / _shared_downstream_bw
            start = max(arrival, self.busy_until, get_downstream_busy_until())
            completion = start + service_time
            update_downstream_busy_until(completion)
        else:
            # Independent topology: DRAM has its own bandwidth
            service_time = t0 + bytes_ / self.dram_bw
            start = max(arrival, self.busy_until)
            completion = start + service_time
        self.busy_until = completion

        n_in_flight = len(self.io_stats) + 1
        self.peak_outstanding = max(self.peak_outstanding, n_in_flight)
        self.max_completion = max(self.max_completion, completion)

        stats = {
            "arrival": arrival,
            "bytes": bytes_,
            "is_load": is_load,
            "service_time": service_time,
            "start": start,
            "completion": completion,
            "queue_wait": max(0.0, start - arrival),
            "total": completion - arrival,
        }
        self.io_stats.append(stats)
        return stats

    def get_batch_duration(self) -> float:
        if not self.io_stats:
            return 0.0
        return max(0.0, self.max_completion - self.batch_start)

    def get_stats_summary(self) -> dict:
        if not self.io_stats:
            return {"count": 0}
        return {
            "count": len(self.io_stats),
            "peak_outstanding": self.peak_outstanding,
            "batch_duration": self.get_batch_duration(),
            "max_queue_wait": max(s["queue_wait"] for s in self.io_stats),
            "busy_until": self.busy_until,
        }


def get_shared_dram_manager() -> SharedDramResourceManager:
    """Get the global SharedDramResourceManager singleton."""
    return SharedDramResourceManager.get_instance()
