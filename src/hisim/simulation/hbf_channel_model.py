#!/usr/bin/env python3
"""Phase 3B: Simplified HBF Channel/Queue timing model + microbenchmarks.

This is NOT the real MQSim (which requires C++ build from HBFSim's patched
MQSim submodule — unavailable due to no internet access in this container).

Instead, this implements the KEY timing insight that the fast model misses:
  - Fast model:  latency = max(base_latency, bytes / aggregate_bandwidth)
  - Channel model: latency = max(per_channel_completion, aggregate_bound)

The per-channel completion accounts for:
  1. Channel transfer time: bytes / (channels * channel_width * transfer_rate)
  2. Page-level parallelism: pages striped across planes within each channel
  3. Queue depth: admission control limiting concurrent requests

Parameters from HBFSim nominal.json:
  channels=32, dies_per_channel=8, planes_per_die=4, page_bytes=16384
  channel_width_bits=8, channel_transfer_rate_mtps=1600
  read_latency_ns=10000, program_latency_ns=100000
  queue_depth=128, aggregate_bandwidth_bytes_per_s=512e9

Key finding: per-channel aggregate BW = 32 * (1 byte * 1600 MT/s) = 51.2 GB/s
  vs aggregate_bandwidth = 512 GB/s (10x higher)
  → Fast model OVERESTIMATES HBF performance by ~10x for large I/O
"""
import json
import math
import statistics
from dataclasses import dataclass


@dataclass
class HBFProfile:
    """HBFSim profile parameters (from nominal.json)."""
    channels: int = 32
    dies_per_channel: int = 8
    planes_per_die: int = 4
    page_bytes: int = 16384
    channel_width_bits: int = 8
    channel_transfer_rate_mtps: int = 1600
    read_latency_ns: int = 10000       # 10μs
    program_latency_ns: int = 100000   # 100μs
    queue_depth: int = 128
    aggregate_bandwidth_bytes_per_s: int = 512_000_000_000  # 512 GB/s

    @property
    def channel_width_bytes(self) -> float:
        return self.channel_width_bits / 8

    @property
    def per_channel_bw_bytes_per_s(self) -> float:
        """Per-channel bandwidth in bytes/s."""
        return self.channel_width_bytes * self.channel_transfer_rate_mtps * 1e6

    @property
    def aggregate_channel_bw_bytes_per_s(self) -> float:
        """Aggregate channel bandwidth (sum of all channels)."""
        return self.channels * self.per_channel_bw_bytes_per_s

    @property
    def planes_per_channel(self) -> int:
        return self.dies_per_channel * self.planes_per_die


def fast_model_latency(bytes_: int, profile: HBFProfile, is_read: bool = True) -> float:
    """Fast timing model: max(base_latency, bytes / aggregate_bandwidth).

    This is the current HiSim HBF timing model.
    Read uses aggregate_bandwidth (512 GB/s).
    Write uses 0.2 × aggregate_bandwidth (102.4 GB/s) per HBFConfig default.
    """
    base_latency = (profile.read_latency_ns if is_read else profile.program_latency_ns) * 1e-9
    bw = profile.aggregate_bandwidth_bytes_per_s if is_read else profile.aggregate_bandwidth_bytes_per_s * 0.2
    transfer = bytes_ / bw
    return max(base_latency, transfer)


def channel_model_latency_single(
    bytes_: int, profile: HBFProfile, is_read: bool = True
) -> float:
    """Channel/queue model latency for a SINGLE request (no queue contention).

    Models:
    1. Request striped across all channels (round-robin by page)
    2. Per-channel transfer time
    3. Page-level parallelism across planes within channel
    4. Aggregate bandwidth lower bound
    """
    if bytes_ == 0:
        return 0.0

    page_latency = (profile.read_latency_ns if is_read else profile.program_latency_ns) * 1e-9

    # Per-channel bytes (request striped across channels)
    per_channel_bytes = bytes_ / profile.channels

    # Per-channel transfer time
    channel_transfer = per_channel_bytes / profile.per_channel_bw_bytes_per_s

    # Page-level parallelism: pages within a channel are distributed across planes
    # Pages in different planes can be served in parallel
    # Pages in the same plane are served sequentially
    per_channel_pages = per_channel_bytes / profile.page_bytes
    pages_per_plane = max(1.0, per_channel_pages / profile.planes_per_channel)
    page_latency_total = pages_per_plane * page_latency

    # Per-channel completion (transfer and page latency overlap as max)
    channel_completion = max(channel_transfer, page_latency_total)

    # Aggregate bandwidth lower bound (shared cursor)
    aggregate_bound = bytes_ / profile.aggregate_bandwidth_bytes_per_s

    # Final completion = max(channel_completion, aggregate_bound)
    return max(channel_completion, aggregate_bound)


def channel_model_latency_batch(
    requests: list[tuple[float, int]], profile: HBFProfile, is_read: bool = True
) -> list[float]:
    """Channel/queue model latency for a BATCH of requests with queue depth.

    Args:
        requests: list of (arrival_ns, bytes) tuples
        profile: HBF profile
        is_read: True for read, False for write

    Returns:
        list of completion times (seconds from arrival)
    """
    if not requests:
        return []

    page_latency = (profile.read_latency_ns if is_read else profile.program_latency_ns) * 1e-9
    qd = profile.queue_depth

    # Track per-channel busy-until time
    channel_busy_until = [0.0] * profile.channels
    # Track admission queue (FIFO)
    admission_queue = []
    in_device = 0

    completions = []
    # Aggregate bandwidth shared cursor
    bw_cursor = 0.0

    for arrival_ns, bytes_ in requests:
        arrival = arrival_ns * 1e-9  # convert ns to seconds

        # Admission control: if in_device >= qd, request waits
        if in_device >= qd:
            admission_queue.append((arrival, bytes_))
            continue

        in_device += 1

        # Route to least-loaded channel (simple load balancing)
        ch = min(range(profile.channels), key=lambda i: channel_busy_until[i])

        # Per-channel bytes
        per_channel_bytes = bytes_ / profile.channels

        # Channel transfer time
        channel_transfer = per_channel_bytes / profile.per_channel_bw_bytes_per_s

        # Page-level parallelism
        per_channel_pages = per_channel_bytes / profile.page_bytes
        pages_per_plane = max(1.0, per_channel_pages / profile.planes_per_channel)
        page_latency_total = pages_per_plane * page_latency

        # Channel completion (wait for channel to be free, then transfer)
        ch_start = max(arrival, channel_busy_until[ch])
        ch_completion = ch_start + max(channel_transfer, page_latency_total)

        # Aggregate bandwidth bound
        bw_start = max(arrival, bw_cursor)
        bw_transfer = bytes_ / profile.aggregate_bandwidth_bytes_per_s
        bw_completion = bw_start + bw_transfer

        # Final completion
        completion = max(ch_completion, bw_completion)

        # Update channel busy time
        channel_busy_until[ch] = ch_completion
        # Update bandwidth cursor
        bw_cursor = max(bw_completion, bw_cursor)

        completions.append(completion - arrival)

        # Release admission slot and admit waiting request
        in_device -= 1
        if admission_queue:
            wait_arrival, wait_bytes = admission_queue.pop(0)
            in_device += 1
            # Re-submit at current time
            ch2 = min(range(profile.channels), key=lambda i: channel_busy_until[i])
            per_ch = wait_bytes / profile.channels
            ch_xfer = per_ch / profile.per_channel_bw_bytes_per_s
            ch_pages = per_ch / profile.page_bytes
            ch_pp = max(1.0, ch_pages / profile.planes_per_channel)
            ch_lat = ch_pp * page_latency
            ch_start2 = max(completion, channel_busy_until[ch2])
            ch_comp2 = ch_start2 + max(ch_xfer, ch_lat)
            bw_start2 = max(completion, bw_cursor)
            bw_comp2 = bw_start2 + wait_bytes / profile.aggregate_bandwidth_bytes_per_s
            comp2 = max(ch_comp2, bw_comp2)
            channel_busy_until[ch2] = ch_comp2
            bw_cursor = max(bw_comp2, bw_cursor)
            completions.append(comp2 - wait_arrival * 1e-9)
            in_device -= 1

    return completions


def run_microbenchmark():
    """Run microbenchmarks comparing fast vs channel model."""
    profile = HBFProfile()

    # I/O sizes to test
    io_sizes = [
        ("16KB", 16 * 1024),
        ("64KB", 64 * 1024),
        ("1MB", 1024 * 1024),
        ("4MB", 4 * 1024 * 1024),
        ("16MB", 16 * 1024 * 1024),
        ("64MB", 64 * 1024 * 1024),
        ("256MB", 256 * 1024 * 1024),
    ]

    # Queue depths to test
    queue_depths = [1, 4, 16, 64]

    # bytes_per_token for Qwen3-0.6B
    bytes_per_token = 114688  # 28 * 8 * 128 * 2 * 2

    print("=" * 100)
    print("Phase 3B: HBF Timing Model Comparison (Fast vs Channel/Queue)")
    print("=" * 100)
    print()
    print(f"Profile: channels={profile.channels}, per_channel_BW={profile.per_channel_bw_bytes_per_s/1e9:.1f} GB/s")
    print(f"         aggregate_channel_BW={profile.aggregate_channel_bw_bytes_per_s/1e9:.1f} GB/s")
    print(f"         aggregate_envelope_BW={profile.aggregate_bandwidth_bytes_per_s/1e9:.1f} GB/s")
    print(f"         planes_per_channel={profile.planes_per_channel}, page_bytes={profile.page_bytes}")
    print(f"         read_latency={profile.read_latency_ns/1e3:.0f}μs, write_latency={profile.program_latency_ns/1e3:.0f}μs")
    print(f"         queue_depth={profile.queue_depth}")
    print(f"         bytes_per_token (Qwen3-0.6B)={bytes_per_token}")
    print()

    # Single I/O comparison (no queue)
    print("-" * 100)
    print("Single I/O Latency (no queue contention):")
    print("-" * 100)
    print(f"{'I/O Size':<10} {'Fast Read':>12} {'Channel Read':>14} {'Ratio':>8} {'Fast Write':>12} {'Channel Write':>14} {'Ratio':>8}")
    print(f"{'':10} {'(μs)':>12} {'(μs)':>14} {'':>8} {'(μs)':>12} {'(μs)':>14} {'':>8}")
    print("-" * 100)

    for label, size in io_sizes:
        fast_r = fast_model_latency(size, profile, is_read=True) * 1e6
        ch_r = channel_model_latency_single(size, profile, is_read=True) * 1e6
        fast_w = fast_model_latency(size, profile, is_read=False) * 1e6
        ch_w = channel_model_latency_single(size, profile, is_read=False) * 1e6
        ratio_r = ch_r / fast_r if fast_r > 0 else 0
        ratio_w = ch_w / fast_w if fast_w > 0 else 0
        print(f"{label:<10} {fast_r:>12.1f} {ch_r:>14.1f} {ratio_r:>7.1f}x {fast_w:>12.1f} {ch_w:>14.1f} {ratio_w:>7.1f}x")

    print()

    # Token-level comparison
    print("-" * 100)
    print("Token-level Latency (HiSim HBF Backup/Load):")
    print("-" * 100)
    print(f"{'Tokens':<10} {'I/O Size':<10} {'Fast Load':>12} {'Channel Load':>14} {'Ratio':>8} {'Fast Backup':>12} {'Channel Backup':>14} {'Ratio':>8}")
    print(f"{'':10} {'':10} {'(μs)':>12} {'(μs)':>14} {'':>8} {'(μs)':>12} {'(μs)':>14} {'':>8}")
    print("-" * 100)

    token_counts = [240, 480, 2040, 4080, 8160, 16320]
    for tokens in token_counts:
        size = tokens * bytes_per_token
        fast_r = fast_model_latency(size, profile, is_read=True) * 1e6
        ch_r = channel_model_latency_single(size, profile, is_read=True) * 1e6
        fast_w = fast_model_latency(size, profile, is_read=False) * 1e6
        ch_w = channel_model_latency_single(size, profile, is_read=False) * 1e6
        ratio_r = ch_r / fast_r if fast_r > 0 else 0
        ratio_w = ch_w / fast_w if fast_w > 0 else 0
        size_mb = size / (1024 * 1024)
        print(f"{tokens:<10} {size_mb:.1f}MB{'':<5} {fast_r:>12.1f} {ch_r:>14.1f} {ratio_r:>7.1f}x {fast_w:>12.1f} {ch_w:>14.1f} {ratio_w:>7.1f}x")

    print()

    # Queue depth effect (batch of 16 requests, 1MB each)
    print("-" * 100)
    print("Queue Depth Effect (16 requests × 1MB Read):")
    print("-" * 100)
    print(f"{'QD':<6} {'Fast P50':>10} {'Channel P50':>12} {'Fast P95':>10} {'Channel P95':>12} {'Fast Total':>12} {'Channel Total':>14}")
    print(f"{'':6} {'(μs)':>10} {'(μs)':>12} {'(μs)':>10} {'(μs)':>12} {'(ms)':>12} {'(ms)':>14}")
    print("-" * 100)

    batch_size = 16
    io_size = 1024 * 1024  # 1MB
    for qd in queue_depths:
        test_profile = HBFProfile(queue_depth=qd)
        requests = [(i * 100, io_size) for i in range(batch_size)]  # 100ns apart

        # Fast model: each request independent (no queue)
        fast_lats = [fast_model_latency(io_size, test_profile, is_read=True) for _ in range(batch_size)]

        # Channel model with queue
        ch_lats = channel_model_latency_batch(requests, test_profile, is_read=True)

        fast_p50 = statistics.median(fast_lats) * 1e6
        ch_p50 = statistics.median(ch_lats) * 1e6
        fast_p95 = sorted(fast_lats)[int(0.95 * len(fast_lats))] * 1e6
        ch_p95 = sorted(ch_lats)[int(0.95 * len(ch_lats))] * 1e6
        fast_total = sum(fast_lats) * 1e3
        ch_total = sum(ch_lats) * 1e3

        print(f"{qd:<6} {fast_p50:>10.1f} {ch_p50:>12.1f} {fast_p95:>10.1f} {ch_p95:>12.1f} {fast_total:>12.3f} {ch_total:>14.3f}")

    print()

    # Effective bandwidth comparison
    print("-" * 100)
    print("Effective Bandwidth (single large I/O, Read):")
    print("-" * 100)
    print(f"{'I/O Size':<10} {'Fast BW':>12} {'Channel BW':>12} {'Bottleneck':>20}")
    print(f"{'':10} {'(GB/s)':>12} {'(GB/s)':>12} {'':>20}")
    print("-" * 100)

    for label, size in io_sizes:
        fast_lat = fast_model_latency(size, profile, is_read=True)
        ch_lat = channel_model_latency_single(size, profile, is_read=True)
        fast_bw = size / fast_lat / 1e9 if fast_lat > 0 else 0
        ch_bw = size / ch_lat / 1e9 if ch_lat > 0 else 0

        # Determine bottleneck
        per_ch_bytes = size / profile.channels
        ch_transfer = per_ch_bytes / profile.per_channel_bw_bytes_per_s
        per_ch_pages = per_ch_bytes / profile.page_bytes
        pages_per_plane = max(1.0, per_ch_pages / profile.planes_per_channel)
        page_lat = pages_per_plane * profile.read_latency_ns * 1e-9
        agg_bound = size / profile.aggregate_bandwidth_bytes_per_s

        if ch_transfer >= max(page_lat, agg_bound):
            bottleneck = "channel_transfer"
        elif page_lat >= max(ch_transfer, agg_bound):
            bottleneck = "page_latency"
        else:
            bottleneck = "aggregate_bound"

        print(f"{label:<10} {fast_bw:>12.1f} {ch_bw:>12.1f} {bottleneck:>20}")

    print()
    print("=" * 100)
    print("KEY FINDINGS:")
    print("=" * 100)
    print(f"1. Aggregate channel BW = {profile.aggregate_channel_bw_bytes_per_s/1e9:.1f} GB/s")
    print(f"   vs Aggregate envelope = {profile.aggregate_bandwidth_bytes_per_s/1e9:.1f} GB/s")
    print(f"   → Fast model overestimates by ~{profile.aggregate_bandwidth_bytes_per_s/profile.aggregate_channel_bw_bytes_per_s:.0f}x")
    print(f"2. For 240-token load (27.4MB): Fast={fast_model_latency(240*bytes_per_token, profile)*1e6:.1f}μs, Channel={channel_model_latency_single(240*bytes_per_token, profile)*1e6:.1f}μs")
    print(f"3. For 2039-token load (223MB): Fast={fast_model_latency(2039*bytes_per_token, profile)*1e6:.1f}μs, Channel={channel_model_latency_single(2039*bytes_per_token, profile)*1e6:.1f}μs")
    print(f"4. Channel model shows DRAM (16GB/s) vs HBF channel ({profile.aggregate_channel_bw_bytes_per_s/1e9:.1f}GB/s) → HBF still ~3.2x faster than DRAM")
    print(f"5. Queue depth effect minimal for single large I/O; matters for concurrent small I/Os")


if __name__ == "__main__":
    run_microbenchmark()
