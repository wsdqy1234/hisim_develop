# HiSparse 稀疏 KV 验证

对应实现：`src/hisim/simulation/sglang/hisparse_sim.py`，设计/假设见
[`docs/hisparse_simulation.md`](../../docs/hisparse_simulation.md)。

## 一键验证

```bash
PYTHONPATH=src BENCHMARK_TEST_MODEL_PATH=/path/to/Qwen3-0.6B \
    bash benchmark/hisparse/run_validation.sh
```

脚本会：

1. 用 DeepSeek-V3.2 几何跑解析 sweep（不依赖模型/GPU）；
2. 用 `test/assets/mock/config_hisparse.json`（constant predictor、`top_k=256`、
   `device_buffer=512`、`ratio=2`）跑 20 请求 × 1024 prompt × 64 output 的 replay：
   - `baseline_capacity` / `sparse_capacity`：HBM 物理池 20000 tokens（容量压力场景）；
   - `baseline_roomy` / `sparse_roomy`：HBM 物理池 40000 tokens（无容量压力，隔离时延开销）；
   - `sparse_roomy_freq4`：同上，但 `HISIM_HISPARSE_SHARED_INDEX_FREQ=4`，模拟 shared-index
     预取（只有 1/4 的 swap-in 字节在关键路径）；
3. 汇总到 `results/summary.json`。

## 单跑

```bash
# 解析 sweep（不需要模型/GPU，但脚本顶层会导入 sglang）
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep

# 单次 replay，--enable 打开 HISIM_HISPARSE
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py bench --enable \
    --output /tmp/sparse.json --max-total-tokens 20000
```

`bench` 需要本地模型目录（`BENCHMARK_TEST_MODEL_PATH`）以及 sglang 源码环境；
`--max-total-tokens` 是 **HBM 物理池**，逻辑池 = 物理池 × `host_to_device_ratio`。
注意 `bench` 在**不带** `--enable` 时会主动清掉 `HISIM_HISPARSE`，以免基线进程
继承到稀疏配置——所以稀疏运行必须显式加 `--enable`（不要只 export 环境变量）。

## DeepSeek-V4.1（L0/L1/L2）

DSV4.1 使用 `test/assets/mock/config_hisparse_dsv41.json`（`top_k=512`、
`device_buffer=1024`、host ratio=8），通过 `--kv-profile` 或
`HISIM_HISPARSE_KV_PROFILE` 选择口径。设计和验证结果见
[`docs/hisparse_dsv41_simulation.md`](../../docs/hisparse_dsv41_simulation.md)：

- `dsv41`：890 B/token，四层均分，L0/L1 兼容；
- `dsv41_fp4_csa2`：890 B/token + 544 B/token indexer 假设 + L2 拓扑；
  绝对 page/allocator/SWA/compressor/admission 字节未知；
- `dsv41_vllm_bf16`：`.22` vLLM-Ascend BF16 布局，payload
  **2885 B/token/rank**、allocator/capacity **4096 B/token/rank**。逐 source
  layer main/index split、SWA/compressor 和 admission 公式已落地；HBM 池 sizing
  用 4096，IO/staging 用 2885，TP8 不再除 8。

```bash
# 三个 profile 的解析 sweep
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep --kv-profile dsv41
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep \
    --kv-profile dsv41_fp4_csa2
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep \
    --kv-profile dsv41_vllm_bf16

# DSV4.1 端到端（baseline/sparse 用同一个配置，靠 --enable 切换）
PYTHONPATH=src BENCHMARK_TEST_MODEL_PATH=/path/to/Qwen3-0.6B \
HISIM_CONFIG_PATH=test/assets/mock/config_hisparse_dsv41.json \
HISIM_VALIDATION_OUT_DIR=/tmp/hisparse_dsv41_results \
    bash benchmark/hisparse/run_validation.sh

# 用环境变量把同一份配置切到 BF16 L2 布局
HISIM_HISPARSE_KV_PROFILE=dsv41_vllm_bf16 \
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py bench --enable \
    --config test/assets/mock/config_hisparse_dsv41.json \
    --max-total-tokens 20000 --num-prompts 8 --input-len 8192 --output-len 64 \
    --output /tmp/hisparse_dsv41_results/sparse_bf16.json
```

`run_validation.sh` 自带的 sweep 仍是 legacy 几何；DSV4.1 sweep 请用上面的
`--kv-profile` 单跑。注意 `.22` 的 2885 B/token/rank 与官方 FP4/CSA2 的
890 B/token 是两个场景，不能混用；FP4 的 544 只是 indexer 设计假设。

profile 的动态 access model 默认每个 decode step 都重新选 Top-K
（`selection_cadence=every_step`），即使压缩长度不变；`compression_boundary`
只保留为敏感性对照。hook 在 prefill 的 radix matching 完成后才启动 staging，
因此 `prefix_indices` 已可读；direct-short 的 newest page 老化不产生主 KV H2D，
长序列 previous-newest 仍按 swap-in 计费；
hot-pool 容量和请求 overhead 使用同一个 per-worker 并发口径。可用
`HISIM_HISPARSE_SELECTION_CADENCE`、`HISIM_HISPARSE_HOT_BUFFER_INIT`、
`HISIM_HISPARSE_INDEX_READ_ON_HIT` 覆盖。
这些 profile 的字节布局属于 vLLM-Ascend/official 口径，不等于当前 sglang
C4/C128 的逐字节实现；真实 Top-K/hot-buffer/SSU 校准前不要把命中率或绝对
TPOT 当成实测预测。
长上下文 L1 验证命令见 DSV4.1 文档 §9 的复现命令。

## 复现注意事项

- hisim / sglang hook 必须在**模块顶层**导入：sglang 用 spawn 启动 scheduler 子进程，
  子进程以 `__mp_main__` 重新导入模块，只有顶层导入才能让 hook 生效。
- `sglang_bench.py` 会把 `HISIM_OUTPUT_DIR` 固定为 `/tmp/hisim/simulation`，所以脚本
  在每次 run 后把 `metrics.json` 拷到 `--output` 旁边。
- 基线/稀疏必须用**独立进程**运行：hisim 的池与状态是进程级全局的。
- constant predictor 只用于功能/统计验证，不代表真实时延；真实时延请用
  `predictor.name=aiconfigurator` 并挂载 AIC 数据库。
