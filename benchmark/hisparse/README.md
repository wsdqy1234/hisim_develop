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

## 复现注意事项

- hisim / sglang hook 必须在**模块顶层**导入：sglang 用 spawn 启动 scheduler 子进程，
  子进程以 `__mp_main__` 重新导入模块，只有顶层导入才能让 hook 生效。
- `sglang_bench.py` 会把 `HISIM_OUTPUT_DIR` 固定为 `/tmp/hisim/simulation`，所以脚本
  在每次 run 后把 `metrics.json` 拷到 `--output` 旁边。
- 基线/稀疏必须用**独立进程**运行：hisim 的池与状态是进程级全局的。
- constant predictor 只用于功能/统计验证，不代表真实时延；真实时延请用
  `predictor.name=aiconfigurator` 并挂载 AIC 数据库。
