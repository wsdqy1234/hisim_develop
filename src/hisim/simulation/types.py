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

    # KDA/Mamba state pool (for hybrid MLA+KDA models)
    max_mamba_cache_size: Optional[int] = None
    mamba_slots_per_request: int = 5

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

    # [v2 Phase 1] KDA/MLA per-request diagnostic metrics (revised per审查意见3).
    # All default to 0/False so existing baseline behavior is unchanged.
    # Populated only when HISIM_KDA_OBS=1 is set.
    # Recorded at different stages: match前, match后, load-back后, COW后, scheduler最终决策.

    # --- MLA dimension (from match_prefix result) ---
    mla_potential_hit_tokens: int = 0       # full_kv_hit_length: MLA available potential (含 device+host+evicted-backed)

    # --- KDA dimension: recoverable position at each layer ---
    kda_l1_recoverable_position: int = 0    # token position at best_match if MAMBA.value is not None (COW available)
    kda_l2_recoverable_position: int = 0    # token position at best_match if MAMBA.host_value is not None (L2 hit)
    kda_l3_key_hit: bool = False            # L3 batch_exists_mamba returned hit (set in _mock_start_loading)
    kda_restored_position: int = 0          # position after load-back completes (set in _mock_start_loading)

    # --- COW / execution ---
    kda_cow_src_slot: int = -1             # COW source slot id (from mamba_cow_src_index), -1 if no COW

    # --- Derived: effective prefix and gaps ---
    effective_prefix_hit_tokens: int = 0   # min(mla_potential, max(kda_l1, kda_l2, kda_l3)) — actual usable prefix
    retreat_gap_tokens: int = 0           # max(0, mla_potential - max(kda_l1, kda_l2, kda_l3)) — structural KDA gap

    # --- Scheduler / predictor ---
    scheduled_extend_tokens: int = 0       # extend_range.length as set by scheduler (what predictor receives)
    unavoidable_new_tokens: int = 0        # total_input - mla_potential (actual new input beyond MLA)
    actual_extra_recompute_tokens: int = 0 # KDA tokens needing recompute (0 if COW, prefix_len if non-COW with prefix, in predictor if true retreat)
    predictor_input_tokens: int = 0        # FakeRequest.input_length passed to K3ProfilePredictor

    # [A2] New metrics: clear statistical semantics
    scheduler_effective_hit_tokens: int = 0  # req.cached_tokens (post-load-back, includes restored)
    verified_logical_reuse: bool = False      # shadow_verify identity_ok AND ready AND consumed
    historical_tokens_actually_scheduled: int = 0  # actual replay path tokens (extend - new_tokens, 0 if restored)
    kda_restore_tier: str = ""                # "L1" / "L2" / "L3" / "miss" / "unknown"
    kda_restore_latency_ms: float = 0.0       # KDA L2/L3 load duration (from StateManager, per-req delta)
    mla_restore_latency_ms: float = 0.0       # MLA L2/L3 load duration (from StateManager, per-req delta)

    # --- Legacy fields (kept for backward compat, will be removed) ---
    kda_replay_tokens: int = 0            # alias for actual_extra_recompute_tokens
    kda_new_tokens: int = 0              # alias for unavoidable_new_tokens
    kda_l1_hit: bool = False             # alias for kda_cow_src_slot >= 0
    kda_l2_hit: bool = False             # alias for kda_l2_recoverable_position > 0
    kda_l3_hit: bool = False             # alias for kda_l3_key_hit
    kda_recoverable_position: int = 0     # alias for max(kda_l1, kda_l2, kda_l3)
    kda_checkpoint_pos: int = 0           # track_boundary of matched checkpoint
    kda_loadback_bytes: int = 0           # KDA bytes loaded back from L2/L3
    mla_l2_hit_tokens: int = 0            # MLA L2 hit tokens (host_hit_length)
    mla_loadback_bytes: int = 0           # MLA bytes loaded back from L2/L3

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
