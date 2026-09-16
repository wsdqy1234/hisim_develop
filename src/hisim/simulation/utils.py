from hisim.spec.model import ModelInfo
from hisim.spec.accelerator import AcceleratorInfo
from hisim.simulation.types import SchedulerConfig, RequestStats
import numpy as np

try:
    from hisim.time_predictor.aiconfigurator import get_perf_model
    _HAS_AIC = True
except ImportError:
    get_perf_model = None
    _HAS_AIC = False


def calc_kv_cache_cell_elems(model_info: ModelInfo, tp_size: int, pp_size: int) -> int:
    """Per-worker KV cache elements per token (for L1 MLA KV pool sizing).
    For hybrid models (MLA+KDA), only counts full_attention layers —
    linear/Mamba layers produce recurrent state, not token-indexed KV."""
    if model_info.kv_lora_rank != 0:
        num_mla_layers = model_info.num_full_attention // pp_size
        return (model_info.kv_lora_rank + model_info.qk_rope_head_dim) * num_mla_layers
    else:
        num_layers = model_info.num_hidden_layers // pp_size
        num_kv_heads = max(model_info.num_key_value_heads // tp_size, 1)
        return num_kv_heads * model_info.head_dim * num_layers * 2


def calc_kv_cache_cell_elems_full(model_info: ModelInfo, pp_size: int) -> int:
    """Full KV cache size across all TP workers (not divided by tp_size).
    MLA: same as per-worker (MLA doesn't shard, only rank 0 writes L3).
    MHA: uses full num_kv_heads (not divided by tp_size), = per-worker × tp_size."""
    if model_info.kv_lora_rank != 0:
        num_mla_layers = model_info.num_full_attention // pp_size
        return (model_info.kv_lora_rank + model_info.qk_rope_head_dim) * num_mla_layers
    else:
        num_layers = model_info.num_hidden_layers // pp_size
        return model_info.num_key_value_heads * model_info.head_dim * num_layers * 2


def calc_kv_cache_per_layer_elems(
    model_info: ModelInfo, tp_size: int, pp_size: int
) -> int:
    if model_info.kv_lora_rank != 0:
        return model_info.kv_lora_rank + model_info.qk_rope_head_dim
    else:
        num_kv_heads = max(model_info.num_key_value_heads // tp_size, 1)
        return num_kv_heads * model_info.head_dim * 2


def calc_mamba_state_bytes(model_info: ModelInfo, tp_size: int, pp_size: int) -> int:
    """Per-slot KDA/Mamba state size in bytes (conv_state + ssm_state, all linear layers).
    ssm_state = (num_heads / tp) × head_dim² × dtype_size  per layer
    conv_state = measured from deployment (55296 bytes/layer for K3 TP=8)
    Total = (ssm + conv) × num_linear_attention layers
    Returns total bytes per slot."""
    num_linear = model_info.num_linear_attention // pp_size
    if num_linear == 0:
        return 0

    dtype_size = 4  # float32 (KDA conv/ssm state always f32 on NPU)

    # ssm_state per layer: (num_heads / tp) × head_dim² × dtype
    if model_info.attn_linear_num_key_heads and model_info.attn_linear_key_head_dim:
        num_heads = max(model_info.attn_linear_num_key_heads // tp_size, 1)
        head_dim = model_info.attn_linear_key_head_dim
        ssm_per_layer = num_heads * head_dim * head_dim * dtype_size
    else:
        ssm_per_layer = 0

    # conv_state per layer: conv_kernel × num_heads_per_tp × head_dim × dtype
    # Note: K3's SiTU activation adds extra state beyond standard Mamba conv.
    # Formula is approximate; K3 measured = 55296 B/layer at TP=8.
    # Use formula as estimate, will be overridden by measured value if available.
    if model_info.attn_linear_conv_kernel_dim:
        if model_info.attn_linear_num_key_heads and model_info.attn_linear_key_head_dim:
            num_heads = max(model_info.attn_linear_num_key_heads // tp_size, 1)
            head_dim = model_info.attn_linear_key_head_dim
            conv_per_layer = model_info.attn_linear_conv_kernel_dim * num_heads * head_dim * dtype_size
            # K3 correction: SiTU activation inflates conv_state by ~2.25x
            # Measured: 55296 B vs formula 24576 B for K3 TP=8
            # TODO: derive exact formula from sglang source code
            if model_info.model_type == "kimi_k3":
                conv_per_layer = 55296  # measured per-layer at TP=8
        else:
            conv_per_layer = model_info.attn_linear_conv_kernel_dim * model_info.hidden_size * dtype_size
    else:
        conv_per_layer = 0

    return (ssm_per_layer + conv_per_layer) * num_linear


def estimate_kv_cache_pool_capacity(
    model: ModelInfo, device: AcceleratorInfo, scheduler_config: SchedulerConfig
) -> int:
    """Estimate L1 KV cache token capacity (MLA KV pool only).
    For hybrid models, KDA state pool is sized separately by max_mamba_cache_size."""
    if _HAS_AIC:
        perf_model = get_perf_model(scheduler_config, model)
        weights = 0
        for op in perf_model.context_ops:
            weights += op.get_weights()
        weights /= perf_model.config.pp_size
    else:
        weights = 0
    framework_reserved_mem_gb = 1.4
    rest_memory = (
        scheduler_config.mem_fraction_static * device.hbm_capacity_gb
        - framework_reserved_mem_gb
    ) * (1 << 30) - weights
    kv_cache_space_per_token = (
        calc_kv_cache_cell_elems(
            model, scheduler_config.tp_size, scheduler_config.pp_size
        )
        * scheduler_config.kv_cache_data_type.bytes
    )

    # For hybrid models (MLA+KDA), subtract KDA state pool from available memory
    mamba_state_bytes = calc_mamba_state_bytes(
        model, scheduler_config.tp_size, scheduler_config.pp_size
    )
    if mamba_state_bytes > 0 and hasattr(scheduler_config, 'max_mamba_cache_size'):
        mamba_pool_bytes = mamba_state_bytes * scheduler_config.max_mamba_cache_size
        rest_memory -= mamba_pool_bytes

    return int(rest_memory / kv_cache_space_per_token) if kv_cache_space_per_token > 0 else 0


def calc_metrics(requests: list[RequestStats]) -> dict:
    ttfts = []
    tpots = []
    itls = []
    e2e_latencies = []
    total_dur_s = 1e-9
    total_input = 0
    total_output = 0
    completed = 0
    total_reused_tokens = 0
    total_disk_hit_tokens = 0
    queue_durs = []
    for req in requests:
        if not req.is_complete():
            continue
        completed += 1
        ttfts.append(req.gen_token_latencies[0])
        queue_durs.append(req.queue_end - req.queue_start)
        if len(req.gen_token_latencies) > 1:
            # output length > 1
            tpots.append(np.mean(req.gen_token_latencies[1:]))
        itls.extend(req.gen_token_latencies[1:])
        e2e_latencies.append(sum(req.gen_token_latencies))
        total_dur_s = max(total_dur_s, req.last_event_time)
        total_input += req.input_length
        total_output += req.output_length
        total_reused_tokens += req.final_reused_tokens
        total_disk_hit_tokens += req.prefetch_complete_tokens
    return {
        "num_requests": len(requests),
        "completed": completed,
        "total_input": total_input,
        "total_output": total_output,
        "duration": total_dur_s,
        "request_throughput": len(requests) / total_dur_s,
        "input_throughput": total_input / total_dur_s,
        "output_throughput": total_output / total_dur_s,
        "total_throughput": (total_input + total_output) / total_dur_s,
        "prefix_cache_reused_ratio": 0
        if total_input == 0
        else total_reused_tokens / total_input,
        "disk_prefetch_ratio": 0
        if total_input == 0
        else total_disk_hit_tokens / total_input,
        "mean_ttft_ms": np.mean(ttfts or 0) * 1000,
        "median_ttft_ms": np.median(ttfts or 0) * 1000,
        "std_ttft_ms": np.std(ttfts or 0) * 1000,
        "p90_ttft_ms": np.percentile(ttfts or 0, 90) * 1000,
        "p95_ttft_ms": np.percentile(ttfts or 0, 95) * 1000,
        "p99_ttft_ms": np.percentile(ttfts or 0, 99) * 1000,
        "mean_queue_ms": np.mean(queue_durs or 0) * 1000,
        "mean_tpot_ms": np.mean(tpots or 0) * 1000,
        "median_tpot_ms": np.median(tpots or 0) * 1000,
        "std_tpot_ms": np.std(tpots or 0) * 1000,
        "p90_tpot_ms": np.percentile(tpots or 0, 90) * 1000,
        "p95_tpot_ms": np.percentile(tpots or 0, 95) * 1000,
        "p99_tpot_ms": np.percentile(tpots or 0, 99) * 1000,
        "mean_itl_ms": np.mean(itls or 0) * 1000,
        "median_itl_ms": np.median(itls or 0) * 1000,
        "std_itl_ms": np.std(itls or 0) * 1000,
        "p90_itl_ms": np.percentile(itls or 0, 90) * 1000,
        "p95_itl_ms": np.percentile(itls or 0, 95) * 1000,
        "p99_itl_ms": np.percentile(itls or 0, 99) * 1000,
        "max_itl_ms": np.max(itls or 0) * 1000,
        "mean_e2e_latency_ms": np.mean(e2e_latencies) * 1000,
        "median_e2e_latency_ms": np.median(e2e_latencies) * 1000,
        "std_e2e_latency_ms": np.std(e2e_latencies) * 1000,
        "p90_e2e_latency_ms": np.percentile(e2e_latencies or 0, 90) * 1000,
        "p95_e2e_latency_ms": np.percentile(e2e_latencies or 0, 95) * 1000,
        "p99_e2e_latency_ms": np.percentile(e2e_latencies or 0, 99) * 1000,
        "time_cost": -1,  # Updated by external benchmark caller
    }
