from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
from hisim.spec import ModelInfo, DataType, AcceleratorInfo


@dataclass
class BenchmarkConfig:
    request_rate: float = float("inf")
    max_concurrency: Optional[int] = None
    with_queue_start: bool = (
        False  # For hisim hooks: include queue start time in replay schedule.
    )
    ignore_request_timestamp: bool = False
    # For hisim hooks: session-aware replay where the next round of a session
    # is released at max(static time, previous round's actual finish + gap).
    # Requires per-request session metadata (e.g. weka_trace datasets).
    dynamic_completion_feedback: bool = False


@dataclass
class SchedulerConfig:
    model: Union[ModelInfo, str]
    # For the default value, please refer to "https://docs.sglang.ai/backend/server_arguments.html".
    max_prefill_tokens: int = 16384
    chunked_prefill_size: Optional[int] = None
    data_type: Optional[DataType] = (
        None  # Data type for model weights and activations. If none is set, it will be automatically detected.
    )
    kv_cache_data_type: Optional[DataType] = None
    mem_fraction_static: Optional[float] = None
    hicache_storage_backend: Optional[str] = None
    hicache_storage_prefetch_policy: str = (
        "best_effort"  # choices: best_effort, wait_complete, timeout
    )
    schedule_policy: str = "fcfs"
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    max_running_requests: int = (1 << 31) - 1
    page_size: Optional[int] = None

    # framework backend
    backend_name: str = "sglang"
    backend_version: Optional[str] = None


class MockSimulationMode(Enum):
    BLOCKING = "BLOCKING"
    OFFLINE = "OFFLINE"


@dataclass
class RequestStats:
    rid: str = ""
    last_event_time: float = 1.0
    input_length: int = 1
    output_length: int = 1
    final_reused_tokens: int = 0
    l2_hit_tokens: int = 0
    l3_hit_tokens: int = 0
    prefetch_complete_tokens: int = 0
    queue_start: float = -1
    queue_end: float = -1
    created_time: float = -1
    gen_token_latencies: list[float] = field(default_factory=list)
    # HBF L1.5 per-medium hit split (populated from _HOST_LOAD_SPLIT)
    hbf_hit_tokens: int = 0
    dram_hit_tokens: int = 0
    # Per-medium load bytes (from host_value virtual indices)
    hbf_load_bytes: int = 0
    dram_load_bytes: int = 0
    # L3 prefetch stats
    l3_prefetch_wait_ms: float = 0.0
    # L3 read pipeline metrics (Phase 2d): distinguish Exists / Read Issued /
    # Read Completed / Restored / Consumed so that L3 "hits" (existence checks)
    # are never conflated with actual KV restoration.
    l3_exists_tokens: int = 0       # batch_exists hit count (existence check)
    l3_read_issued_tokens: int = 0  # tokens for which SSD read was issued
    l3_read_completed_tokens: int = 0  # tokens actually read (batch_get success)
    l3_restored_tokens: int = 0     # tokens inserted into radix tree (prefetch result)
    l3_consumed_tokens: int = 0     # tokens consumed by request (host_hit_length from L3)
    l3_read_bytes: int = 0          # SSD read bytes (for timing verification)
    l3_read_dur_ms: float = 0.0     # SSD read duration (from calc_prefetch_pages)
    # Phase 2d-Closure: Token conservation fields
    actual_input_tokens: int = 0    # truncated input length (after auto-truncate)
    recomputed_tokens: int = 0      # prefill tokens from conservation formula
    scheduler_prefill_tokens: int = 0  # prefill tokens from scheduler (independent)
    # L3 read timing causality (Phase 2d-Closure)
    l3_read_issue_time: float = 0.0  # sim clock when SSD read was issued
    l3_read_complete_time: float = 0.0  # sim clock when SSD read completed
    l3_restore_commit_time: float = 0.0  # sim clock when KV inserted into tree
    l3_consume_time: float = 0.0    # sim clock when request consumed restored KV

    def is_complete(self) -> bool:
        return True


@dataclass
class StorageTierConfig:
    name: str
    ssd_config: Optional[str] = None
    read_bandwidth_gb: float = 4.0
    write_bandwidth_gb: Optional[float] = None
    kvcm_storage_type: str = "pace"
    kvcm_bandwidth_mbps: int = 4000
    kvcm_capacity_gb: float = 10.0
    kvcm_priority: int = 0


@dataclass
class HBFConfig:
    """HBF (High-Bandwidth Flash) L1.5 tier configuration.

    Parameters map to HBFSim's nominal.json profile (assumed config, NOT
    measured hardware data). The fast timing formula is:
        latency = max(base_latency, bytes / aggregate_bandwidth)
    (overlap-aware: max, not sum — confirmed from HBFSim calibrator.cpp:46-64).

    Phase 3B: channel model adds per-channel bandwidth constraint:
        per_channel_BW = channel_width_bytes * transfer_rate_mtps
        aggregate_channel_BW = channels * per_channel_BW
    This is ~10x lower than the aggregate envelope (51.2 vs 512 GB/s).
    """
    capacity_gb: float = 1024.0          # nominal: 1 TiB
    read_bandwidth_gb: float = 512.0    # aggregate read BW envelope (512 GB/s)
    write_bandwidth_gb: Optional[float] = None  # default = read * 0.2
    read_latency_us: float = 10.0        # read_latency_ns=10000
    write_latency_us: float = 100.0      # program_latency_ns=100000
    channels: int = 32
    queue_depth: int = 128               # only used by MQSim reference (Phase 3)
    hbm_cache_gb: float = 8.0            # HBF internal HBM cache (Capacity-mode only)
    placement_ratio: float = 0.7         # V0: target 70% to HBF
    # Phase 3B: channel-level geometry (from nominal.json)
    channel_width_bits: int = 8          # ONFI channel width (8 bits = 1 byte)
    channel_transfer_rate_mtps: int = 1600  # ONFI transfer rate (1600 MT/s)
    page_bytes: int = 16384              # NAND page size (16 KB)
    dies_per_channel: int = 8            # dies (chips) per channel
    planes_per_die: int = 4              # planes per die

    @property
    def write_bandwidth(self):
        return (self.write_bandwidth_gb or self.read_bandwidth_gb * 0.2)

    @property
    def per_channel_bw_bytes_per_s(self) -> float:
        """Per-channel bandwidth in bytes/s."""
        return (self.channel_width_bits / 8) * self.channel_transfer_rate_mtps * 1e6

    @property
    def aggregate_channel_bw_bytes_per_s(self) -> float:
        """Aggregate channel bandwidth (sum of all channels)."""
        return self.channels * self.per_channel_bw_bytes_per_s

    @property
    def planes_per_channel(self) -> int:
        return self.dies_per_channel * self.planes_per_die


@dataclass
class PlatformConfig:
    device: Union[AcceleratorInfo, str]
    # Storage configuration for hierarchical cache management.
    disk_capacity_gb: Optional[float] = None
    disk_read_bandwidth_gb: Optional[float] = None
    disk_write_bandwidth_gb: Optional[float] = None
    memory_capacity_gb: Optional[float] = None
    memory_read_bandwidth_gb: Optional[float] = None
    memory_write_bandwidth_gb: Optional[float] = None
    num_device_per_node: int = 8
    storage_tiers: list[StorageTierConfig] = field(default_factory=list)
    tier_read_mode: str = "serial"
    tier_write_mode: str = "parallel"
    enable_l3_direct_to_l1: bool = False
    l3_to_l1_bandwidth_gb: Optional[float] = None
    # HBF L1.5 tier (parallel to L2 DRAM)
    enable_hbf: bool = False
    hbf: Optional[HBFConfig] = None

    @property
    def l3_to_l1_bandwidth(self):
        return (
            self.l3_to_l1_bandwidth_gb * 1e9
            if self.l3_to_l1_bandwidth_gb
            else None
        )

    @property
    def disk_read_bandwidth(self):
        return (
            self.disk_read_bandwidth_gb * 1e9 if self.disk_read_bandwidth_gb else None
        )

    @property
    def disk_write_bandwidth(self):
        return (
            self.disk_write_bandwidth_gb * 1e9 if self.disk_write_bandwidth_gb else None
        )

    @property
    def memory_read_bandwidth(self):
        return (
            self.memory_read_bandwidth_gb * 1e9
            if self.memory_read_bandwidth_gb
            else None
        )

    @property
    def memory_write_bandwidth(self):
        return (
            self.memory_write_bandwidth_gb * 1e9
            if self.memory_write_bandwidth_gb
            else None
        )

    @property
    def hbf_read_bandwidth(self):
        return self.hbf.read_bandwidth_gb * 1e9 if self.hbf else None

    @property
    def hbf_write_bandwidth(self):
        return self.hbf.write_bandwidth * 1e9 if self.hbf else None
