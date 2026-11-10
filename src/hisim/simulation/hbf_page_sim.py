"""Phase 4-Closure (P1): Page-level discrete-event NAND media simulator.

Replaces the closed-form ``_media_service_time()`` formula (Theory Bound)
with an explicit event-driven model of page-level NAND access:

  - A logical I/O is split into NAND pages (page_bytes each).
  - Pages are striped round-robin across channels (sequential logical
    addresses → global page sequence counter).
  - Within a channel, each page is placed on the least-busy plane.
  - READ:  plane array-read (tR) → channel data-out (page/rate).
          Array reads on different planes overlap with channel transfer
          of earlier pages (pipelining).
  - WRITE: channel data-in (page/rate) → plane program (tW).
          Channel is free as soon as data lands in the plane register.

Per-channel data-path cursors and per-plane busy cursors persist across
batches (cross-batch safety, matching SharedHbfResourceManager semantics).

Mechanism switches (for loss attribution, Phase 4-Closure P2):
  - enable_page_latency:      tR/tW = 0 when False
  - enable_plane_contention:  planes never block when False (fresh plane
                              per page, no wait)
  - enable_channel_transfer:  channel data rate = ∞ when False

NOT modeled (reported as Unknown in loss attribution):
  - FTL mapping overhead, ECC, garbage collection, read disturb,
    write amplification, die (LUN) enable bus sharing beyond the
    channel data-path serialization approximation.
  - Out-of-order (fair-share) interleaving of concurrent I/Os: pages are
    admitted FIFO per I/O submission order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PageSimStats:
    """Per-I/O statistics from the page simulator."""
    pages: int = 0
    channels_used: int = 0
    planes_used: int = 0
    media_completion: float = 0.0   # absolute completion time
    media_time: float = 0.0         # completion - arrival
    channel_busy_sum: float = 0.0   # Σ per-channel busy time added by this I/O


class PageLevelMediaSim:
    """Discrete-event page-level NAND media simulator.

    State: per-channel data-path cursor + per-plane busy cursor.
    submit() processes all pages of one logical I/O in order and returns
    the I/O's media completion time.
    """

    def __init__(self, channels: int, planes_per_channel: int,
                 page_bytes: int, read_latency: float, write_latency: float,
                 channel_rate: float):
        self.channels = max(1, channels)
        self.planes_per_channel = max(1, planes_per_channel)
        self.page_bytes = max(1, page_bytes)
        self.read_latency = read_latency
        self.write_latency = write_latency
        self.channel_rate = channel_rate  # bytes/s per channel data path

        # Mechanism switches (default: all enabled = full model)
        self.enable_page_latency = True
        self.enable_plane_contention = True
        self.enable_channel_transfer = True

        # State: busy cursors (persist across batches)
        self.plane_busy: List[List[float]] = [
            [0.0] * self.planes_per_channel for _ in range(self.channels)
        ]
        self.channel_busy: List[float] = [0.0] * self.channels
        # Global logical page sequence → round-robin channel striping
        # (sequential logical addresses across I/Os)
        self._page_seq: int = 0
        self.batch_start: float = 0.0
        # Accounting
        self.total_pages_served: int = 0
        self.total_ios: int = 0

    # ---- lifecycle ----

    def begin_batch(self, clock: float) -> None:
        """Mark batch start. Cursors are NOT reset (cross-batch safety:
        I/Os still in flight keep the plane/channel cursors advanced)."""
        self.batch_start = clock

    def reset_state(self) -> None:
        """Full reset (test only)."""
        self.plane_busy = [[0.0] * self.planes_per_channel
                           for _ in range(self.channels)]
        self.channel_busy = [0.0] * self.channels
        self._page_seq = 0
        self.total_pages_served = 0
        self.total_ios = 0

    # ---- core ----

    def _page_transfer_time(self) -> float:
        """Data-path occupancy per page on one channel."""
        if not self.enable_channel_transfer or self.channel_rate <= 0:
            return 0.0
        return self.page_bytes / self.channel_rate

    def submit(self, arrival: float, bytes_: int, is_read: bool) -> PageSimStats:
        """Submit one logical I/O; process its pages in order.

        Returns PageSimStats with the media completion time.
        """
        stats = PageSimStats()
        if bytes_ <= 0:
            stats.media_completion = arrival
            return stats

        pages = max(1, (bytes_ + self.page_bytes - 1) // self.page_bytes)
        t_lat = (self.read_latency if is_read else self.write_latency)
        if not self.enable_page_latency:
            t_lat = 0.0
        t_transfer = self._page_transfer_time()

        completion = arrival
        channels_used = set()
        planes_used = set()
        channel_busy_delta = [0.0] * self.channels

        for p in range(pages):
            ch = self._page_seq % self.channels
            self._page_seq += 1
            channels_used.add(ch)

            if self.enable_plane_contention:
                # Least-busy plane on this channel
                plane = min(range(self.planes_per_channel),
                            key=lambda i: self.plane_busy[ch][i])
                plane_free = self.plane_busy[ch][plane]
            else:
                # No plane contention: a fresh plane per page, never waits
                plane = p % self.planes_per_channel
                plane_free = arrival
            planes_used.add((ch, plane))

            if is_read:
                # Array read: plane busy for t_lat, data ready after
                plane_start = max(arrival, plane_free)
                data_ready = plane_start + t_lat
                self.plane_busy[ch][plane] = data_ready
                # Channel data-out: after data ready, serialized per channel
                ch_start = max(data_ready, self.channel_busy[ch])
                ch_end = ch_start + t_transfer
                self.channel_busy[ch] = ch_end
                channel_busy_delta[ch] += t_transfer
                completion = max(completion, ch_end)
            else:
                # Channel data-in first, then plane program
                ch_start = max(arrival, self.channel_busy[ch])
                ch_end = ch_start + t_transfer
                self.channel_busy[ch] = ch_end
                channel_busy_delta[ch] += t_transfer
                prog_start = max(ch_end, plane_free)
                prog_end = prog_start + t_lat
                self.plane_busy[ch][plane] = prog_end
                completion = max(completion, prog_end)

        stats.pages = pages
        stats.channels_used = len(channels_used)
        stats.planes_used = len(planes_used)
        stats.media_completion = completion
        stats.media_time = completion - arrival
        stats.channel_busy_sum = sum(channel_busy_delta)
        self.total_pages_served += pages
        self.total_ios += 1
        return stats

    def media_busy_cursor(self) -> float:
        """Max busy cursor across channels (for cross-batch bookkeeping)."""
        return max(self.channel_busy) if self.channel_busy else 0.0

    def device_utilization(self, window_start: float, window_end: float) -> float:
        """Fraction of channel-data-path capacity that was busy.

        Utilization = Σ per-channel busy time / (channels × window).
        Note: this is DATA-PATH utilization only (program latency on
        planes is not counted — it is not a data-path resource).
        """
        if window_end <= window_start:
            return 0.0
        busy = 0.0
        for ch in range(self.channels):
            busy += max(0.0, self.channel_busy[ch] - window_start)
        return busy / (self.channels * (window_end - window_start))
