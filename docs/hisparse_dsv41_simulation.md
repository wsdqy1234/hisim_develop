# DeepSeek-V4.1 HiSparse 仿真：L0/L1 实现与验证

日期：2026-11-05
分支：`develop_lfz_sparse_kv`
相关文档：[`docs/hisparse_simulation.md`](./hisparse_simulation.md)（HiSparse 基础模型）、
[`docs/sparse_kv_sglang_vllm.md`](./sparse_kv_sglang_vllm.md)（DSV4.1 版本/容器调研）

---

## 1. 目标与范围

上一版 HiSparse 仿真沿用 DeepSeek-V3.2 的 MLA 几何：

```text
HBM KV = physical_tokens × num_layers × bytes_per_token_per_layer
```

DeepSeek-V4.1-Flash 的全局 KV 只有约 **890 B/token**（FP4 + CSA2 跨层复用），且
`config.json` 没有 `kv_lora_rank`；如果继续走 legacy 公式，会落到 GQA 分支：

```text
2 × 1 KV head × 512 head_dim × 2 B × 40 layers = 81,920 B/token   # FP16 KV
```

即高估约 **92×**。128K context 的持久 KV 正确值约 **116.6 MB**，错误值约 **10.7 GB**
（FP16；若 KV cache 用 FP8，legacy 仍会高估约 46×）。

本次只做两级，不搬 CUDA/Triton kernel：

- **L0（几何校正）**：用 890 B/token 的全局 KV 替换 `num_layers × bytes_per_layer`；
  `top_k=512`；host/device ratio=8。
- **L1（压缩位置空间/节奏）**：按 KV source layer 的压缩比（ratio 0/1/2）分组，每组维护
  独立的压缩位置工作集；ratio=2 的组每 2 个 decode step 才产生一条新压缩条目，top-k 在
  **压缩候选**上选择；暂不计 indexer/SWA/MTP 的 HBM 常驻开销。

**L2 尚未实现**，本文最后列出待办。改动默认关闭：不配置 `kv_profile` 时，legacy 行为与
上一版完全一致。

---

## 2. DSV4.1 几何假设

来源：sglang `deepseek_v4.py` / `deepseek_v4_memory_pool.py`、V4.1 HF `config.json`、
vLLM / vLLM-Ascend 的 V4.1 实现。

| 项 | 值 | 说明 |
| --- | --- | --- |
| 层数 | 40 | 另加 3 层 MTP（L2 处理） |
| 压缩比分组 | 2 × ratio 0 + 18 × ratio 2 + 20 × ratio 1 | SWA / CSA / HCA |
| `kv_source_layer_ids` | `[2, 8, 14, 20]` | 3 个 ratio 2 + 1 个 ratio 1 |
| `index_source_layer_ids` | `[2, 8, 14, 20, 24, 28, 32, 36]` | 8 个 indexer 层（L2） |
| `index_topk` | 512 | 每步选中的压缩位置数 |
| `index_head_dim` | 128 | L2 indexer 池 |
| 全局 KV | **890 B/token** | FP4 + CSA2 跨层复用；host 持久池 |
| 128K 持久 KV | **116.65 MB** | `131072 × 890` |
| legacy GQA 估计 | 81,920 B/token | `2×1×512×2×40`，高估 92× |

L1 的逐 token 拆分（`source_layer_ratios=(2,2,2,1)`）：

```text
每个 source layer: 890 / 4 = 222.5 B/token
ratio 2 组（3 层）: 3 × 222.5 × 2 = 1335.0 B / 压缩条目
ratio 1 组（1 层）: 1 × 222.5 × 1 =  222.5 B / 压缩条目
每 token 合计:      3 × 222.5 + 1 × 222.5 = 890 B
```

一个 ratio-2 条目覆盖 2 个 token，所以条目字节是单层每 token 字节的 2 倍；每 token 的总
字节仍严格等于 890。**“4 个 source layer 均分 890”是 L1 的近似**；真实每层字节由
量化/布局决定，L2 从 `config.json` / 权重元数据解析。

---

## 3. 实现

### 3.1 变更文件

| 文件 | 变更 |
| --- | --- |
| `src/hisim/simulation/sglang/hisparse_sim.py` | 新增 `KVProfile` / `DSV41_PROFILE`；`HiSparseSimConfig.kv_profile`；profile 模式下的容量、staging、H2D/D2H、L1 分组工作集与 metrics |
| `src/hisim/simulation/sglang/sglang_hook.py` | `HiSparseSimulator.configure(..., tp_size=...)` |
| `test/test_hisparse_dsv41.py` | 新增 16 个 L0/L1 单测（legacy 回归仍由原 18 个测试覆盖） |
| `test/assets/mock/config_hisparse_dsv41.json` | DSV4.1 mock 配置 |
| `benchmark/hisparse/hisparse_bench.py` | `sweep --kv-profile legacy|dsv41` |

### 3.2 配置

```json
"hisparse": {
  "enable": false,
  "kv_profile": "dsv41",
  "top_k": 512,
  "device_buffer_size": 1024,
  "host_to_device_ratio": 8,
  "page_size": 1,
  "churn": 0.05,
  "recent_ratio": 0.5,
  "shared_index_freq": 1
}
```

- `kv_profile` 支持字符串（`dsv41` / `legacy`）或自定义 dict；环境变量
  `HISIM_HISPARSE_KV_PROFILE=dsv41` 可覆盖。
- profile 生效时 `top_k = min(config.top_k, index_topk)`，即不可超过 512；这是 indexer
  的物理上限，不是调参值。
- `device_buffer_size=0` 的默认值改为 `2 × effective_top_k`（DSV4.1 为 1024），与 sglang
  `device_buffer_size = 2 * top_k` 一致。
- `enable` 仍默认 `false`；baseline 与 sparse 用同一个配置，靠 `HISIM_HISPARSE=1` 切换。

### 3.3 L0：容量与总 IO

profile 模式下：

```text
HBM  = physical_tokens × (offload_bytes_per_token + indexer_bytes_per_token) + SWA 常驻
host = logical_tokens × offload_bytes_per_token,  logical = physical × host_to_device_ratio
staging = Σ_groups ceil(prompt_len / ratio) × entry_bytes = prompt_len × 890
H2D = Σ miss_entry × entry_bytes
D2H = Σ new_entry × entry_bytes
```

`tp_size > 1` 时，`offload_bytes_per_token` / `entry_bytes` / indexer 字节按 TP 均分
（MLA latent 在 TP 上切分）。`indexer_bytes_per_token` 与 SWA 字段已预留，但默认 0，
所以 L0/L1 的 HBM 只含 890 B/token 的全局 KV；L2 再打开。

### 3.4 L1：压缩位置空间与节奏

每个请求为每个 ratio 组建一个 `_WorkingSet`：

- 候选长度：`comp_len = ceil(seq_len / ratio)`；top-k 在 `[0, comp_len)` 上选。
- `ratio=1`：每个 decode step 新增 1 个压缩条目。
- `ratio=2`：每 2 个 decode step 新增 1 个压缩条目；未跨过压缩边界的 step **不推进**
  工作集，直接返回 `hits=min(top_k, comp_len), misses=0`。这避免了在真实 sglang 中
  不存在的“每个 token step 都 churn 一次”的额外 miss。
- 新条目才产生 D2H backup；第一个 decode step 的 prompt 压缩条目已在 prefill staging 中
  计过，不重复计 D2H。
- `churn` 现在按压缩 step 生效：ratio=2 的组每 2 step churn 一次。
- 新增 byte-weighted 命中率（`hisparse_hit_rate_bytes`），避免 ratio-2 的大条目和
  ratio-1 的小条目被按条目数等权平均。

legacy 路径（无 profile）保持原样：`_WorkingSet` 直接跑 token 位置，miss ×
`num_layers × bytes_per_layer` 计 H2D，每步 1 token × 该字节数计 D2H。

### 3.5 新增 metrics

| 字段 | 含义 |
| --- | --- |
| `hisparse_kv_profile` | `legacy` / `dsv41` / 自定义名 |
| `hisparse_offload_bytes_per_token` | 按 TP 缩放后的每 token 全局 KV |
| `hisparse_source_layer_ratios` / `hisparse_layer_groups` | 几何元数据 |
| `hisparse_profile_num_layers` | profile 描述的主模型层数（DSV4.1 = 40） |
| `hisparse_index_topk` / `hisparse_top_k` | indexer 上限 / 实际 top-k |
| `hisparse_io_groups` | `[[ratio, entry_bytes], ...]` |
| `hisparse_indexer_bytes_per_token` / `hisparse_hbm_swa_bytes` | L2 预留，默认 0 |
| `hisparse_hits_bytes` / `hisparse_misses_bytes` / `hisparse_hit_rate_bytes` | 字节加权命中 |
| `hisparse_new_compressed_entries` | 本 run 新增压缩条目数 |
| `hisparse_h2d_bytes_by_ratio` / `hisparse_d2h_bytes_by_ratio` | 按 ratio 拆分的搬运字节 |

---

## 4. 测试

### 4.1 单测

```bash
PYTHONPATH=src python -m pytest test/test_hisparse_sim.py test/test_hisparse_dsv41.py -q
# 34 passed
```

新增测试覆盖：

- 890 B/token 的 HBM/host 容量；128K = 116.65 MB；
- `io_groups()` = `[(1, 222.5), (2, 1335.0)]`；
- `top_k` 被 `index_topk` 截到 512，显式更小的 `top_k` 保留；
- prompt staging = `prompt_len × 890`；
- ratio-2 每 2 step 新增条目、未推进 step 不 churn、不产生 H2D；
- H2D 按压缩条目字节计费；
- TP 缩放；
- indexer 开销可选（L2 预演）；
- 无 profile 时 legacy 容量/层数/IO 完全不变。

### 4.2 解析 sweep（DSV4.1 几何）

```bash
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep \
    --kv-profile dsv41 --output /tmp/hisparse_dsv41_sweep.json
```

64 请求 × 32K prompt + 512 output，80 GB HBM 预算，`top_k=512`、`device_buffer=1024`、
host ratio 8、`recent_ratio=0.5`：

| churn | 命中率（条目） | 命中率（字节） | H2D | D2H | staging | mean load | 新压缩条目 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 98.05% | 98.50% | 0.392 GB | 0.029 GB | 1.866 GB | 6.89 µs | 49,088 |
| 0.20 | 92.62% | 94.39% | 1.465 GB | 0.029 GB | 1.866 GB | 7.49 µs | 49,088 |
| 0.50 | 81.68% | 86.13% | 3.626 GB | 0.029 GB | 1.866 GB | 8.70 µs | 49,088 |

HBM KV = 80 GB，host pool 容量 = 640 GB（`89.9M hot tokens × 8 × 890 B`，是配置的池容量，
不是本 workload 的实际占用；实际持久 KV ≈ staging 1.87 GB）。`max_reqs_hot_pool ≈ 87,695`。
ratio-2 的 H2D 约为 ratio-1 的 3 倍（条目字节 6×、cadence 1/2）。

### 4.3 端到端 replay（Qwen3-0.6B mock + DSV4.1 profile）

```bash
PYTHONPATH=src \
BENCHMARK_TEST_MODEL_PATH=/path/to/Qwen3-0.6B \
HISIM_CONFIG_PATH=test/assets/mock/config_hisparse_dsv41.json \
HISIM_VALIDATION_OUT_DIR=/tmp/hisparse_dsv41_results \
bash benchmark/hisparse/run_validation.sh
```

> `run_validation.sh` 自带的 sweep 仍用 legacy 几何；DSV4.1 sweep 用 §4.2 的
> `--kv-profile dsv41` 单独跑。

20 请求 × 1024 prompt + 64 output，constant predictor：

| run | physical HBM | TTFT | TPOT | req/s | 命中率（字节） | H2D | D2H | staging | mean load | hot 并发 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_capacity | 20,000 | 5.750 ms | 0.525 ms | 289.86 | — | — | — | — | — | — |
| sparse_capacity | 20,000 | 4.095 ms | 0.531 ms | 287.34 | 99.98% | 0.20 MB | 1.10 MB | 18.23 MB | 4.79 µs | 19 |
| baseline_roomy | 40,000 | 2.400 ms | 0.525 ms | 563.38 | — | — | — | — | — | — |
| sparse_roomy | 40,000 | 2.400 ms | 0.532 ms | 557.15 | 99.98% | 0.20 MB | 1.10 MB | 18.23 MB | 6.30 µs | 39 |
| sparse_roomy_freq4 | 40,000 | 2.400 ms | 0.532 ms | 557.19 | 99.98% | 0.20 MB | 1.10 MB | 18.23 MB | 6.26 µs | 39 |

对应容量：

- 20,000 physical → HBM 17.8 MB / host 142.4 MB；
- 40,000 physical → HBM 35.6 MB / host 284.8 MB；
- staging 18,227,200 B = 20 × 1024 × 890。

短 prompt（1024）下两份压缩工作集都装得进 1025-slot hot buffer，所以 H2D 接近 0；
这组结果主要验证“逻辑池 ×8 + 并发上限 + 890 B/token 几何”没有回归。

### 4.4 长上下文 replay（L1 真正起作用）

8 请求 × 8192 prompt + 64 output。baseline 用 physical 80,000（全部请求直接驻留），
sparse 用 physical 20,000（logical = 160,000，按 host ratio 8 放大）：

| 指标 | baseline | sparse (DSV4.1 L1) |
| --- | --- | --- |
| physical / logical tokens | 80,000 / 80,000 | 20,000 / 160,000 |
| TTFT / TPOT / 吞吐 | 5.00 ms / 0.5476 ms / 202.53 req/s | 5.00 ms / 0.5565 ms / 199.70 req/s |
| 命中率（条目 / 字节） | — | 97.62% / 98.11% |
| H2D | — | 7.61 MB |
| D2H | — | 0.44 MB |
| staging | — | 58.33 MB |
| mean load | — | 8.89 µs |
| 新压缩条目 | — | 744 |
| H2D by ratio | — | ratio 1: 1.755 MB；ratio 2: 5.859 MB |
| D2H by ratio | — | ratio 1: 0.110 MB；ratio 2: 0.331 MB |

即：用 1/4 的 physical HBM 换到同样 8 条 8K 请求全部驻留，TPOT 只增加约 1.6%
（0.5476 → 0.5565 ms），吞吐下降约 1.4%。

ratio-2 的 H2D ≈ 3.3× ratio-1，D2H ≈ 3.0× ratio-1，符合“条目 6×、频率 1/2”的 L1 设计。
staging 58,327,040 B = 8 × 8192 × 890，与 L0 公式精确一致。

---

## 5. 与 L0/L1 预期的对照

| 检查项 | 结果 |
| --- | --- |
| 890 B/token 几何 | ✅ HBM/host/staging 均按 890 计 |
| 128K 持久 KV | ✅ 116.65 MB（legacy 10.74 GB） |
| top-k=512 | ✅ 候选长度按 ratio 压缩，top-k 截到 512 |
| ratio=2 cadence | ✅ 每 2 step 新条目；未推进 step 无 churn/IO |
| 按组字节计费 | ✅ ratio 2 条目 1335 B，ratio 1 条目 222.5 B |
| host ratio 8 | ✅ 逻辑池 = physical × 8 |
| legacy 零回归 | ✅ 原 18 个单测全部通过 |
| TP 缩放 | ✅ `offload_bytes_per_token / tp_size` |
| indexer/SWA | ⏳ 字段已预留，L2 打开 |

---

## 6. L2 待办

1. **跨层复用与 indexer/source 映射**
   - 当前 L1 只按 source layer 的 ratio 分组，每个 ratio 组一个工作集；没有显式建模
     `kv_source_layer_ids=[2,8,14,20]` 被 18/20 个 consumer 层复用。
   - L2 需要按 layer id 建 source→consumer 映射，并区分 8 个 `index_source_layer_ids` 的
     indexer 选择；同一 source 的 KV 只搬一次，consumer 层复用。
   - `shared_index_freq` 目前是全局抽象；L2 应对应 sglang 的 anchor/follower 语义，
     follower 层 IO 走侧流预取。

2. **Indexer 常驻 HBM**
   - 8 层 × (128/2 + 4) = **544 B/token（FP4）** 或 8 × (128 + 4) = **1056 B/token（FP8）**。
   - 分别是 890 B/token 的 61% / 119%，必须计入 HBM 预算；`indexer_bytes_per_token`
     字段已存在，L2 设为 544 或 1056 后即可参与 `hbm_kv_bytes`。
   - 注意 indexer 不进入 host 持久池。

3. **SWA ring**
   - 2 个 ratio-0 层；每请求 `swa_window_tokens=128` 的 ring。
   - `swa_window_tokens` / `swa_bytes_per_token` 已预留；L2 需要按 per-request 计算
     `max_num_reqs × window × bytes`，而不是按 token 池线性放大。

4. **MTP / DSpark 草稿路径**
   - 3 个 MTP 层（ratio 0）和 DSpark 草稿 KV 未建模；需要单独的 draft pool / 草稿 token
     写入与回收，以及接受率对 IO 的影响。

5. **CSA2 / C4 / C128 块语义与页粒度**
   - L1 的压缩位置是 token 粒度，`page_size` 只进入 `padded_buffer_size`。
   - L2 需要按 C4/C128 的块大小做整块搬入/驱逐，并让 `device_buffer_size` 以压缩块计。

6. **容量模型细化**
   - 当前 `max_reqs_by_hot_pool = physical_tokens // padded_buffer_size` 是 token 级近似；
     真实实现中每个 ratio 组有独立 device buffer / physical pool，binding group 可能不同。
   - host 池目前固定为 `physical × 8`；真实部署应按 host 内存和最大 context 反推 ratio。
   - L2 建议输出每个 ratio 组的容量、并发上限和 host 占用。

7. **配置自动解析**
   - 目前 `dsv41` 是内置 profile；L2 应从 HF `config.json`（含 nested `text_config`）
     读取 `compress_ratios`、`kv_source_layer_ids`、`index_source_layer_ids`、
     `index_topk`、`quantization_config`，自动生成 `KVProfile`。
   - 这样同一套代码可覆盖 V4-Flash / V4.1，不需要手写常量。

8. **TP / DP / EP**
   - 已按 `tp_size` 均分 KV 字节；L2 需要明确 DP 复制、EP 对 KV 的影响，并在 metrics
     中区分 per-rank 与 global。

9. **真实校准**
   - 用独立容器 `quay.io/ascend/vllm-ascend:v0.30.0-a3`（A3，W8A8 权重
     `Eco-Tech/DeepSeek-V4.1-Flash-w8a8`）跑真实 V4.1，采集 hot-buffer 命中/swap-in
     计数器，校准 `churn` / `recent_ratio` / 实际 H2D 字节。
   - 校准时不要把 vLLM-Ascend 与 hisim 的 sglang 0.5.19 栈混装；hisim 侧继续用当前容器
     的 sglang 仿真后端。

10. **PD / direct RDMA 与 HiCache**
    - 当前只覆盖单实例本地 prefill→staging→decode；PD 分离、direct RDMA 直写 host、
      HiCache 共存（sglang 要求互斥）都未建模。

---

## 7. 复现命令速查

```bash
# 单测
PYTHONPATH=src python -m pytest test/test_hisparse_sim.py test/test_hisparse_dsv41.py -q

# DSV4.1 解析 sweep
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep \
    --kv-profile dsv41 --output /tmp/hisparse_dsv41_sweep.json

# 5 组端到端 replay（baseline / sparse / roomy / freq4）
PYTHONPATH=src \
BENCHMARK_TEST_MODEL_PATH=/path/to/Qwen3-0.6B \
HISIM_CONFIG_PATH=test/assets/mock/config_hisparse_dsv41.json \
HISIM_VALIDATION_OUT_DIR=/tmp/hisparse_dsv41_results \
bash benchmark/hisparse/run_validation.sh

# 长上下文 L1 验证（8 × 8192 prompt + 64 output）
PYTHONPATH=src BENCHMARK_TEST_MODEL_PATH=/path/to/Qwen3-0.6B \
HISIM_SIMULATION_MODE=OFFLINE \
python benchmark/hisparse/hisparse_bench.py bench --enable \
    --config test/assets/mock/config_hisparse_dsv41.json \
    --max-total-tokens 20000 --num-prompts 8 --input-len 8192 --output-len 64 \
    --output /tmp/hisparse_dsv41_results/sparse_longctx.json
```

> baseline 与 sparse 必须分进程运行；`run_validation.sh` 已处理。
> `HISIM_HISPARSE=1` 开启仿真，默认关闭；不设 `kv_profile` 时走 legacy 路径。

---

## 8. 结论

- L0 已修正 DSV4.1 的 KV 几何，128K 持久 KV 从“10.7 GB 量级”回到“116.6 MB 量级”。
- L1 已把 top-k 放到压缩位置空间，并实现 ratio=2 的 cadence；长上下文 replay 中
  ratio-2 的 H2D/D2H 与“条目 6×、频率 1/2”的预期一致。
- 短上下文 replay 的 H2D 接近 0，说明当前 `device_buffer=1024` 对 1K prompt 足够大；
  这不代表 L1 无效，长上下文和 sweep 才是 L1 的有效性证据。
- 当前仍是 **仿真级** L0/L1：没有真实 indexer、没有 indexer/SWA 常驻 HBM、没有 MTP 和
  PD。L2 的优先级建议是：indexer 544/1056 B/token → 跨层 source 复用 → SWA/MTP →
  容量模型细化 → 真实 vLLM-Ascend 校准。
