"""HiSparse validation: end-to-end replay and dependency-free sweep.

Modes
-----
``bench`` (default)
    Run one hisim replay through the real SGLang scheduler/cache pipeline with
    the HiSparse simulation either enabled (``--enable``) or disabled.  The
    metrics are copied to ``--output``.  Run it twice (baseline / sparse) to
    compare, or use ``run_validation.sh`` which does both.

``sweep``
    Drive :class:`HiSparseSimulator` directly with a DeepSeek-V3.2-sized
    geometry and a synthetic long-context decode workload.  This isolates the
    sparse-KV memory/IO model from the (constant) time predictor and reports
    the sensitivity of the hit rate / swap-in traffic to the top-k churn knob.

Examples
--------
    PYTHONPATH=src BENCHMARK_TEST_MODEL_PATH=/path/to/Qwen3-0.6B \
        python benchmark/hisparse/hisparse_bench.py bench --enable \
        --output /tmp/hisparse_sparse.json

    PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The replay mode imports sglang/hisim at module import time: SGLang spawns the
# scheduler in a child process which re-imports this module, and the hisim
# hooks must be installed there too.
os.environ.setdefault(
    "HISIM_CONFIG_PATH", str(ROOT / "test/assets/mock/config_hisparse.json")
)
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("HISIM_SIMULATION_MODE", "OFFLINE")

from sglang.srt.server_args import ServerArgs
from hisim.dataset import DatasetArgs
from hisim.simulation.types import BenchmarkConfig
from hisim.simulation.sglang.sglang_bench import (
    HISIM_METRICS_PATH,
    SGLangBenchmarkRunner,
)
def _run_bench(args) -> dict:
    if args.enable:
        os.environ["HISIM_HISPARSE"] = "1"
    else:
        os.environ.pop("HISIM_HISPARSE", None)

    if args.config:
        os.environ["HISIM_CONFIG_PATH"] = str(Path(args.config).resolve())

    model_path = args.model_path or os.getenv("BENCHMARK_TEST_MODEL_PATH")
    if not model_path:
        model_path = "Qwen/Qwen3-0.6B"

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=model_path,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=False,
            max_total_tokens=args.max_total_tokens,
            log_level="warning",
        )
    )
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=args.num_prompts,
        min_input_len=args.input_len,
        max_input_len=args.input_len,
        min_output_len=args.output_len,
        max_output_len=args.output_len,
    )
    metrics = runner.benchmark(
        BenchmarkConfig(request_rate=args.request_rate), dataset_args=dataset_args
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    if os.path.exists(HISIM_METRICS_PATH):
        shutil.copyfile(HISIM_METRICS_PATH, str(output) + ".metrics.json")
    print(
        json.dumps(
            {
                "completed": metrics.get("completed"),
                "mean_ttft_ms": metrics.get("mean_ttft_ms"),
                "mean_tpot_ms": metrics.get("mean_tpot_ms"),
                "request_throughput": metrics.get("request_throughput"),
                **{
                    k: v
                    for k, v in metrics.items()
                    if k.startswith("hisparse")
                },
            },
            indent=2,
        )
    )
    return metrics


def _run_sweep(args) -> dict:
    """Analytic decode sweep; no sglang/model needed.

    ``legacy`` uses the DeepSeek-V3.2 MLA geometry, ``dsv41`` uses the
    DeepSeek-V4.1 L0/L1 profile (890 B/token, top_k=512, compressed groups),
    ``dsv41_fp4_csa2`` adds the known FP4/CSA2 L2 topology + optional indexer,
    and ``dsv41_vllm_bf16`` uses the .22 vLLM-Ascend per-rank BF16 layout.
    """
    from hisim.simulation.sglang.hisparse_sim import (
        DSV41_FP4_PROFILE,
        DSV41_VLLM_BF16_PROFILE,
        HiSparseSimConfig,
        HiSparseSimulator,
    )

    profile = getattr(args, "kv_profile", "legacy")
    max_num_reqs = None
    if profile == "dsv41":
        num_layers = 40
        bytes_per_token_per_layer = 0.0
        # 80 GB HBM budget / 890 B/token ~= 89.9M hot tokens.
        physical_tokens = int(80e9 / 890.0)
        top_k = 512
        device_buffer_size = 1024
        host_ratio = 8
        kv_profile = "dsv41"
    elif profile in ("dsv41_fp4_csa2", "dsv41_vllm_bf16"):
        num_layers = 40
        bytes_per_token_per_layer = 0.0
        top_k = 512
        device_buffer_size = 1024
        host_ratio = 8
        max_num_reqs = 8
        if profile == "dsv41_vllm_bf16":
            p = DSV41_VLLM_BF16_PROFILE
            kv_profile = p.name
            # Per-rank bytes; subtract steady SWA/compressor from the 80 GB
            # budget before sizing the long-KV hot pool.
            swa_bytes = max_num_reqs * p.swa_window_tokens * p.swa_bytes_per_token
            compressor_bytes = max_num_reqs * p.compressor_bytes_per_request
            physical_tokens = int(
                (80e9 - swa_bytes - compressor_bytes)
                / (p.offload_bytes_per_token + p.indexer_bytes_per_token)
            )
        else:
            p = DSV41_FP4_PROFILE
            kv_profile = p.name
            physical_tokens = int(
                80e9 / (p.offload_bytes_per_token + p.indexer_bytes_per_token)
            )
    else:
        num_layers = 61
        # DeepSeek-V3.2 MLA, FP8 KV: (512 + 64) bytes per token per layer.
        bytes_per_token_per_layer = 576.0
        # 80 GB HBM budget / (576 B * 61 layers) ~= 2.27M hot tokens.
        physical_tokens = 2_270_000
        top_k = 2048
        device_buffer_size = 4096
        host_ratio = 2
        kv_profile = None
    results = []
    for churn in (0.05, 0.2, 0.5):
        cfg = HiSparseSimConfig(
            enable=True,
            kv_profile=kv_profile,
            top_k=top_k,
            device_buffer_size=device_buffer_size,
            host_to_device_ratio=host_ratio,
            churn=churn,
            recent_ratio=0.5,
            access_model="sticky",
            shared_index_freq=1,
            seed=0,
        )
        sim = HiSparseSimulator(
            cfg,
            num_layers=num_layers,
            bytes_per_token_per_layer=bytes_per_token_per_layer,
            physical_pool_tokens=physical_tokens,
            max_num_reqs=max_num_reqs,
        )
        for i in range(args.num_prompts):
            sim.on_request_start(f"r{i}", args.input_len)
            for step in range(args.output_len):
                sim.on_decode_step(f"r{i}", args.input_len + step)
        m = sim.metrics()
        results.append(
            {
                "churn": churn,
                "hit_rate": m["hisparse_hit_rate"],
                "hit_rate_bytes": m["hisparse_hit_rate_bytes"],
                "h2d_gb": m["hisparse_h2d_bytes"] / 1e9,
                "d2h_gb": m["hisparse_d2h_bytes"] / 1e9,
                "staging_gb": m["hisparse_staging_bytes"] / 1e9,
                "mean_load_us": m["hisparse_mean_load_us"],
                "hbm_kv_gb": m["hisparse_hbm_kv_bytes"] / 1e9,
                "hbm_swa_gb": m["hisparse_hbm_swa_bytes"] / 1e9,
                "hbm_compressor_gb": m["hisparse_hbm_compressor_bytes"] / 1e9,
                "host_kv_gb": m["hisparse_host_kv_bytes"] / 1e9,
                "max_reqs_hot_pool": m["hisparse_max_reqs_hot_pool"],
                "admission_1m_blocks": m["hisparse_admission_1m_blocks"],
                "admission_1m_gb": m["hisparse_admission_1m_bytes"] / 1e9,
                "admission_4223_overhead_gb": (
                    m["hisparse_admission_4223_overhead_bytes"] / 1e9
                ),
                "new_compressed_entries": m["hisparse_new_compressed_entries"],
                "h2d_bytes_by_ratio": m["hisparse_h2d_bytes_by_ratio"],
            }
        )
    payload = {
        "profile": profile,
        "geometry": {
            "num_layers": num_layers,
            "bytes_per_token_per_layer": bytes_per_token_per_layer,
            "physical_tokens": physical_tokens,
            "max_num_reqs": max_num_reqs,
            "top_k": top_k,
            "device_buffer_size": device_buffer_size,
            "host_to_device_ratio": host_ratio,
            "num_prompts": args.num_prompts,
            "input_len": args.input_len,
            "output_len": args.output_len,
        },
        "sweep": results,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode")

    bench = sub.add_parser("bench", help="end-to-end hisim replay")
    bench.add_argument("--enable", action="store_true", help="enable HiSparse")
    bench.add_argument("--output", required=True)
    bench.add_argument("--config", default=None)
    bench.add_argument("--model-path", default=None)
    bench.add_argument("--max-total-tokens", type=int, default=20000,
                       help="HBM KV pool size (physical hot pool)")
    bench.add_argument("--num-prompts", type=int, default=20)
    bench.add_argument("--input-len", type=int, default=1024)
    bench.add_argument("--output-len", type=int, default=64)
    bench.add_argument("--request-rate", type=float, default=100.0)
    bench.set_defaults(func=_run_bench)

    sweep = sub.add_parser("sweep", help="analytic decode sweep")
    sweep.add_argument("--output", default=None)
    sweep.add_argument(
        "--kv-profile",
        choices=(
            "legacy",
            "dsv41",
            "dsv41_fp4_csa2",
            "dsv41_vllm_bf16",
        ),
        default="legacy",
        help=(
            "KV geometry: DeepSeek-V3.2 legacy, DSV4.1 L0/L1, FP4/CSA2 L2 "
            "topology, or .22 vLLM-Ascend BF16 L2"
        ),
    )
    sweep.add_argument("--num-prompts", type=int, default=64)
    sweep.add_argument("--input-len", type=int, default=32768)
    sweep.add_argument("--output-len", type=int, default=512)
    sweep.set_defaults(func=_run_sweep)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
