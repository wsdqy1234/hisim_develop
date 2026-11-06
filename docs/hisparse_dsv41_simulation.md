# DeepSeek-V4.1 HiSparse 仿真：L0/L1/L2 实现与验证

日期：2026-11-06
分支：`develop_lfz_sparse_kv`
相关文档：[`docs/hisparse_simulation.md`](./hisparse_simulation.md)（HiSparse 基础模型）、
[`docs/sparse_kv_sglang_vllm.md`](./sparse_kv_sglang_vllm.md)（DSV4.1 版本/容器调研）

---

## 1. 范围与结论

本文在已有 L0/L1 之上补齐 **L2 静态布局**，并明确区分两个不能混用的场景：

| 场景 | profile | 口径 | 本轮状态 |
| --- | --- | --- | --- |
| `.22` vLLM-Ascend **BF16** 实测部署 | `dsv41_vllm_bf16` | 每 rank 实测布局 | 静态字节、page/allocator、SWA/compressor、admission 公式已落地 |
| 官方 **FP4 + CSA2** 目标 | `dsv41_fp4_csa2` | 890 B/token + 544 B/token indexer 假设 | 只复用已知拓扑；绝对 page/allocator/SWA/compressor/admission 字节未知 |
| 旧 L0/L1 通用几何 | `dsv41` | 890 B/token，四层均分 | 保持向后兼容，用于历史结果对照 |

关键结论：

- `.22` BF16 的 **2885 B/token/rank** 不是 890 的另一种写法：它包含
  `2560 B/token/rank` main KV + `325 B/token/rank` index payload，allocator 斜率是
  **4096 B/token/rank**。两者混用会把容量低估约 3.2 倍。
- 逐 source layer 字节为 `(577, 577, 577, 1154)`，因此 IO 组是
  `[(1, 1154), (2, 3462)]`，不是 L1 的 `[(1, 222.5), (2, 1335)]`。
- BF16 的 SWA、compressor 与 admission 是三个不同口径：
  - 稳态 SWA ring：`8 req × 128 token × 40960 B/token = 41,943,040 B`；
  - 稳态 compressor：`8 req × 393,216 B = 3,145,728 B`；
  - admission 上界：1M context 单请求 **8533 blocks / 4,473,749,504 B**，其中
    非 long-KV 开销 **341 blocks / 178,782,208 B**。
- FP4/CSA2 能复用的是 **模型拓扑**（source→consumer、index 层、candidate 路径、
  SWA/compressor 层号、128-token block），不能复用 BF16 的绝对 page/allocator 字节。
- 本文所有 L2 动态行为仍是 **仿真**：`.22` 是 vLLM-Ascend 的静态布局实测，不是
  sglang HiSparse 的 online trace；top-k、hot-buffer 命中、host/SSU 带宽仍未校准。
  两个场景的缺口在最后一节分别列出。

改动默认关闭：不配置 `kv_profile` 时，legacy 路径与上一版完全一致。

---

## 2. 三个 profile 与字节口径

### 2.1 profile 选择

```text
legacy                 无 profile：num_layers × bytes_per_token_per_layer
dsv41                  890 B/token，四层均分，L0/L1 兼容
dsv41_fp4_csa2         890 B/token + 544 B/token indexer 假设 + L2 拓扑
dsv41_vllm_bf16        2885 B/token/rank 实测布局 + L2 静态字节
```

`KVProfile.bytes_scope` 决定 TP 缩放：

- `"global"`：`dsv41` / `dsv41_fp4_csa2`，字节按 `tp_size` 均分；
- `"per_rank"`：`dsv41_vllm_bf16`，字节已经是单 rank 值，TP8 不再除 8。

这不是风格差异：`.22` 的 `TP8/EP8/DCP1` 下每个 rank 持有一个 KV head 的副本，
再除 8 会直接算错。FP4/CSA2 目标没有实测，不能假定它也是 per-rank。

### 2.2 与 `.22` 实测的关系

`.22` 运行口径（来自 handover 报告）：

| 项 | 值 |
| --- | --- |
| 实现 | vLLM 0.27.1 / vLLM-Ascend |
| 并行 | TP8 / EP8 / DCP1 |
| KV dtype | BF16 |
| 权重 | W4A8 |
| 执行 | eager，SPEC=0 |
| HBM KV | 5.87 GiB/rank |
| 逻辑容量 | 1,478,304 tokens |
| main KV | 2560 B/token/rank |
| index | 325 B/token/rank |
| payload | **2885 B/token/rank** |
| allocator 斜率 | **4096 B/token/rank** |

注意：本文的 sweep 用 **80 GB 通用 HBM 预算** 做敏感性分析，不是 `.22` 的
5.87 GiB/rank；不要把 sweep 的绝对 `hbm_kv_gb` 当成 `.22` 容量。`.22` 的
1,478,304 tokens 是实测逻辑容量，当前仿真尚未复现这个绝对值。

---

## 3. `.22` BF16 L2 布局

### 3.1 逐 source layer 与 IO 组

```text
source layer ids     = [2, 8, 14, 20]
source layer ratios  = [2, 2, 2, 1]
per-token bytes      = (577, 577, 577, 1154)   # 实测 payload 拆分，合计 2885
ratio-2 组（3 层）   = 3 × 577 × 2 = 3462 B / 压缩条目
ratio-1 组（1 层）   = 1 × 1154 × 1 = 1154 B / 压缩条目
每 token 合计        = 3 × 577 + 1154 = 2885 B
```

因此：

```text
staging(prompt) = prompt_len × 2885
H2D(step)       = Σ miss_entry × {1154 或 3462}
D2H(step)       = Σ new_entry  × {1154 或 3462}
```

L1 的四层均分（222.5 B/token/layer）只是早期近似，**不能**用于 `.22` BF16
的 IO 计费。

### 3.2 page / block / allocator

| 项 | 值 | 说明 |
| --- | --- | --- |
| scheduler block | 128 tokens | admission 公式的 `L/128` |
| page | 131,072 B | 每 slot 一个 page |
| slots / block | 4 | `block_payload = 4 × 131072` |
| block payload | 369,280 B | `128 × 2885` |
| block allocator | 524,288 B | `128 × 4096` |
| packed file | 371,379 B | 导出统计 |
| FS alloc | 417,792 B | 文件系统分配 |

`payload/block = 369,280 B` 与 `allocator/block = 524,288 B` 的差值是 pool 的
对齐/元数据开销；容量模型用 allocator 斜率，IO 模型用 payload 字节。

### 3.3 跨层复用与 index candidate 路径

```text
source 2  -> consumers 2..7
source 8  -> consumers 8..13
source 14 -> consumers 14..19
source 20 -> consumers 20..39
```

同一 source 的 KV 只搬一次，consumer 层复用；仿真按 ratio 组聚合 source 字节，
所以不会重复计算每个 consumer 的 H2D。

已知的 index 结构：

| 项 | 值 |
| --- | --- |
| index storage layers | `[2, 8, 14, 20]` |
| index compute layers | `[2, 8, 14, 20, 24, 28, 32, 36]` |
| index_topk | 512 |
| candidate source layer | 20 |
| candidate blocks | 2048 |
| candidate block tokens | 8 |
| candidate filter layers | `[24, 28, 32, 36]` |

当前实现暴露这些拓扑和 `candidate_blocks × candidate_block_tokens` 的元数据，
但 **没有** 用 candidate 块限制 top-k 候选空间；真实的 candidate 选择仍需要
逐步 index trace。

### 3.4 SWA

| 项 | 值 |
| --- | --- |
| SWA layers | 40 |
| window | 128 tokens |
| bytes/token | 40 × 512 × 2 = 40,960 B |
| per-request steady | `128 × 40960 = 5,242,880 B` |
| admission window | `128 - 1 + 2 × 2048 = 4223 tokens` |

在仿真里：

```text
hbm_swa_bytes = max_num_reqs × 128 × 40960
```

这是 **稳态 ring**，不是每个 context 都按 `L × 40960` 线性放大。

### 3.5 compressor

| 项 | 值 |
| --- | --- |
| compressor layers | `[2, 8, 14]` |
| page | 131,072 B |
| per-request steady | `3 × 131072 = 393,216 B` |

```text
hbm_compressor_bytes = max_num_reqs × 393216
```

compressor 与 SWA 都是 per-request 状态，不能按 token 池线性放大。

### 3.6 admission 上界

实现公式（单请求）：

```text
B(L) = ceil(L/128) + 1
       + 10 × (ceil(min(L, 4223)/128) + 1)
```

每 block `524,288 B`：

| L | blocks | bytes |
| --- | ---: | ---: |
| 128 | 22 | 11,534,336 |
| 8,192 | 405 | 212,336,640 |
| 1,048,576 | **8,533** | **4,473,749,504** |

在 `L >= 4223` 后，非 long-KV 开销固定为 **341 blocks = 178,782,208 B**：
10 个 SWA 组各 34 blocks（33 个窗口块 + 1 个 null/对齐块），再加 1 个额外
block（compressor/其他状态）。8 个并发请求的 admission 上界是
`8 × 178,782,208 = 1,430,257,664 B`。

必须在报告里区分：

- `hisparse_hbm_kv_bytes`：稳态 long KV + 稳态 SWA + 稳态 compressor；
- `hisparse_hbm_admission_overhead_bytes`：admission 上界中的非 long-KV 部分；
- 两者不能直接相加后当作“实际常驻”，因为 admission 是预留上界，稳态是 decode 占用。

### 3.7 来源与可信边界

- 以上 page/slot/block/allocator、SWA/compressor、admission 值来自 handover 的
  `kv-layout.json`、`ssu-calibration-parameters.json`、`kv-storage-format.json` 和
  deployment measurement guide §9。
- 这些是 **从源码/布局重建** 的结果，不是 live worker tensor introspection；
  导出的 3 blocks × 8 ranks payload 一致，但这仍不等于真实 HiSparse 动态计数器。
- `.22` 是 vLLM-Ascend，不是 sglang HiSparse；不能把 vLLM 的静态布局当成
  sglang 的 top-k/swap-in 实测。

---

## 4. FP4/CSA2 L2：哪些能用，哪些不能用

### 4.1 已复用的模型拓扑

`dsv41_fp4_csa2` 与 BF16 profile 共用：

- source layers `[2, 8, 14, 20]`，ratios `(2, 2, 2, 1)`；
- source→consumer 映射；
- index storage `[2, 8, 14, 20]` / compute `[2, 8, 14, 20, 24, 28, 32, 36]`；
- candidate source layer 20，2048 blocks × 8 tokens，filters `[24, 28, 32, 36]`；
- SWA 40 层 / window 128，compressor layers `[2, 8, 14]`；
- scheduler block 128；
- `index_topk=512`。

这些是模型结构，不是 BF16 专属字节。

### 4.2 已实现的 FP4/CSA2 近似

| 项 | 值 | 口径 |
| --- | --- | --- |
| global KV | 890 B/token | 官方 FP4 + CSA2 |
| indexer | 544 B/token | **FP4 8 层设计假设**，单独计入 HBM |
| source 均分 | 222.5 B/token/layer | 近似，真实 FP4 逐层布局未知 |
| IO 组 | `[(1, 222.5), (2, 1335)]` | 与 L1 一致 |
| TP | global，按 `tp_size` 均分 | 目标未实测 |
| SWA/compressor 字节 | 0 | 未知，只保留层号/window |
| page/slot/allocator | 0 | 未知 |
| admission | 0 | 未知 |

544 的来源是 `8 × (128/2 + 4) = 544 B/token` 的 FP4 index 假设；它不是从
目标镜像测出来的。若目标实际是 4 个 source 的 325 B/token 口径，或 index 已包含
在 890 里，应通过自定义 profile 覆盖，而不是直接套用 544。

### 4.3 不能从 BF16 直接搬的东西

- `2885 / 4096 / 524288 / 131072 / 40960 / 393216` 都是 BF16/`.22` 口径；
- FP4 的 page、slot、allocator、payload/block、SWA/compressor 量化字节都可能不同；
- `.22` 的 admission 公式依赖 vLLM-Ascend allocator 的 4-slot/128-token 实现，
  FP4 目标镜像未验证。

---

## 5. 实现

### 5.1 变更文件

| 文件 | 变更 |
| --- | --- |
| `src/hisim/simulation/sglang/hisparse_sim.py` | `KVProfile` 增加 L2 字段/方法；新增 `DSV41_FP4_PROFILE`、`DSV41_VLLM_BF16_PROFILE`；per-rank TP 语义；SWA/compressor/admission 计入 metrics |
| `test/test_hisparse_dsv41.py` | 新增 12 个 L2 单测（BF16 布局/IO/admission、FP4 拓扑/TP、字符串别名） |
| `benchmark/hisparse/hisparse_bench.py` | `--kv-profile` 增加两个 L2 profile；BF16/FP4 sweep 几何；输出 SWA/compressor/admission 字段 |
| `docs/hisparse_dsv41_simulation.md` | 本文 |
| `benchmark/hisparse/README.md` | L2 profile 命令与口径说明 |

### 5.2 `KVProfile`

新增的关键字段：

```text
bytes_scope, source_layer_ids, source_layer_bytes_per_token,
source_consumers, index_storage_layers, index_compute_layers,
index_bytes_per_token, main_kv_bytes_per_token,
scheduler_block_tokens, page_bytes, slot_count,
allocator_bytes_per_token, payload_bytes_per_block,
candidate_blocks, candidate_block_tokens, candidate_source_layer,
candidate_filter_layers, swa_layers, swa_admission_tokens,
compressor_layers, compressor_page_bytes, compressor_bytes_per_request,
admission_swa_groups, admission_extra_groups, admission_bytes_per_block
```

新增方法：

- `KVProfile.io_groups()`：有逐层字节时按逐层计，否则保持旧的均分；
- `KVProfile.admission_blocks(L)` / `admission_bytes(L)` /
  `admission_overhead_bytes(L)`：admission 上界；
- `KVProfile.per_rank` / `source_layer_bytes_sum`：口径与校验辅助。

`KVProfile.from_value()` 支持：

```text
dsv41 / dsv4.1 / deepseek_v41
dsv41_fp4_csa2 / dsv41_fp4 / fp4_csa2
dsv41_vllm_bf16 / dsv41_bf16 / dsv41-bf16 / vllm_bf16
```

以及 dict 形式的逐层字节、source→consumer、index/candidate/compressor 拓扑。

### 5.3 Simulator

- `_tp_scale = 1.0`（`per_rank`）或 `1.0 / tp_size`（`global`）；
- `offload_bytes_per_token`、`io_groups`、`indexer_bytes_per_token` 统一走该缩放；
- `hbm_swa_bytes` / `hbm_compressor_bytes` 按 `max_num_reqs` 计算，不按 token 池；
- `hbm_kv_bytes` / `baseline_hbm_kv_bytes` 纳入 compressor；
- admission 开销单独放在 `hbm_admission_overhead_bytes`，不混入稳态 `hbm_kv_bytes`；
- metrics 增加 `bytes_scope`、逐层字节、source→consumer、index/candidate、
  page/allocator、SWA/compressor、admission 表等字段。

legacy 路径不变：没有 profile 时仍用
`physical_tokens × num_layers × bytes_per_token_per_layer`。

### 5.4 配置

沿用 `test/assets/mock/config_hisparse_dsv41.json`，用环境变量切 profile：

```bash
# BF16 .22 布局
HISIM_HISPARSE_KV_PROFILE=dsv41_vllm_bf16

# FP4/CSA2 拓扑近似
HISIM_HISPARSE_KV_PROFILE=dsv41_fp4_csa2
```

`kv_profile` 也可以是自定义 dict，覆盖 `source_layer_bytes_per_token`、
`indexer_bytes_per_token`、`page_bytes`、`admission_bytes_per_block` 等字段。

---

## 6. 测试

### 6.1 单测

```bash
PYTHONPATH=src python -m pytest test/test_hisparse_sim.py test/test_hisparse_dsv41.py -q
# 46 passed
```

本轮新增 12 个 L2 测试，覆盖：

- BF16 `2885 B/token/rank`、逐层 `(577, 577, 577, 1154)`、
  `io_groups = [(1, 1154), (2, 3462)]`；
- `bytes_scope="per_rank"` 时 TP1/TP8 的 `offload`、`io_groups` 完全一致；
- BF16 SWA `8 × 128 × 40960`、compressor `8 × 393216` 与
  `hbm_kv_bytes` 的合成；
- BF16 H2D 按 1154/3462 计费，staging = `prompt × 2885`；
- admission 表：128→22 blocks、8192→405、1M→8533、
  1M→4,473,749,504 B，4223 的非 long-KV 开销 341 blocks；
- FP4 profile 的 890+544、global TP 缩放、拓扑字段、未知字节为 0；
- 字符串别名 `dsv41_vllm_bf16` / `dsv41_fp4_csa2`；
- legacy 原 18 个测试 + L0/L1 原 16 个测试仍全部通过。

### 6.2 本轮没有做的事

- 没有在真实 `.22` 镜像上重跑 sglang HiSparse（`.22` 是 vLLM-Ascend）；
- 没有用真实 top-k trace 校准 `churn` / `recent_ratio`；
- BF16 L2 的 page/allocator/SWA/compressor/admission 来自源码/布局重建，
  不是 live worker 计数器。

---

## 7. 解析 sweep

命令：

```bash
# L0/L1 回归
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep \
    --kv-profile dsv41 --output /tmp/hisparse_dsv41_sweep.json

# FP4/CSA2 拓扑近似
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep \
    --kv-profile dsv41_fp4_csa2 --output /tmp/hisparse_fp4_sweep.json

# .22 BF16 布局
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep \
    --kv-profile dsv41_vllm_bf16 --output /tmp/hisparse_bf16_sweep.json
```

公共 workload：64 请求 × 32K prompt + 512 output，80 GB 通用 HBM 预算，
`top_k=512`、`device_buffer=1024`、host ratio 8、`recent_ratio=0.5`。
BF16/FP4 的 L2 sweep 设 `max_num_reqs=8`（对应 `.22` 的部署并发口径）；
`dsv41` 仍用原来的无 per-request SWA 口径。

### 7.1 `dsv41` 回归（890 B/token）

| churn | 命中率（条目） | 命中率（字节） | H2D | D2H | staging | mean load | 新压缩条目 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 98.05% | 98.50% | 0.392 GB | 0.029 GB | 1.866 GB | 6.89 µs | 49,088 |
| 0.20 | 92.62% | 94.39% | 1.465 GB | 0.029 GB | 1.866 GB | 7.49 µs | 49,088 |
| 0.50 | 81.68% | 86.13% | 3.626 GB | 0.029 GB | 1.866 GB | 8.70 µs | 49,088 |

HBM 80 GB、host 640 GB、`max_reqs_hot_pool ≈ 87,695`。这组数值与上一版
L0/L1 完全一致，证明新字段没有破坏 legacy/`dsv41` 路径。

### 7.2 `dsv41_fp4_csa2`（890 + 544 B/token）

| churn | 命中率（条目） | 命中率（字节） | H2D | D2H | staging | mean load |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 98.05% | 98.50% | 0.392 GB | 0.029 GB | 1.866 GB | 6.89 µs |
| 0.20 | 92.62% | 94.39% | 1.465 GB | 0.029 GB | 1.866 GB | 7.49 µs |
| 0.50 | 81.68% | 86.13% | 3.626 GB | 0.029 GB | 1.866 GB | 8.70 µs |

因为 access model 和 IO 组没变，命中率/H2D 与 `dsv41` 相同；indexer 544 只改了
HBM 容量分母：80 GB 下 physical tokens 从 89.89M 降到 **55.79M**，host pool 从
640 GB 降到 **397.21 GB**，`max_reqs_hot_pool ≈ 54,427`。SWA/compressor/admission
仍为 0（未知），不能把这组数当成 FP4 目标容量。

### 7.3 `dsv41_vllm_bf16`（2885 B/token/rank）

80 GB 下 physical tokens = **27,714,007**（已扣 8 请求的稳态 SWA/compressor）；
`max_num_reqs=8`。

| churn | 命中率（条目） | 命中率（字节） | H2D | D2H | staging | mean load | H2D ratio 1 / 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 98.05% | 98.36% | 1.268 GB | 0.094 GB | 6.050 GB | 7.38 µs | 0.501 / 0.767 GB |
| 0.20 | 92.62% | 93.86% | 4.755 GB | 0.094 GB | 6.050 GB | 9.34 µs | 1.910 / 2.845 GB |
| 0.50 | 81.68% | 84.79% | 11.779 GB | 0.094 GB | 6.050 GB | 13.28 µs | 4.754 / 7.025 GB |

对应容量：

- `hbm_kv_gb ≈ 80.0`（long KV + SWA 0.0419 GB + compressor 0.00315 GB）；
- `host_kv_gb ≈ 639.64`；
- `max_reqs_hot_pool ≈ 27,038`；
- admission 1M：8533 blocks / 4.4737 GB；
- admission 4223 非 long-KV 开销：0.1788 GB/request。

与 `dsv41` 相比：

- staging/H2D 约 **3.2×**，因为 2885/890 ≈ 3.24；
- D2H 约 **3.2×**；
- 字节命中率略低（同样 churn 下 98.50%→98.36%），因为 ratio-2 条目权重从
  1335 提到 3462，ratio-1 从 222.5 提到 1154，miss 的字节惩罚更接近实测布局。

再次强调：这里的 80 GB 是 sweep 的通用预算，不是 `.22` 的 5.87 GiB/rank；
这组数只用于比较三种 profile 的相对敏感性。

### 7.4 旧 L0/L1 端到端 replay（保留作回归）

上一轮用 Qwen3-0.6B mock + `config_hisparse_dsv41.json` 跑的 20×1024+64
replay 仍是当前 `dsv41` 路径的回归基线：

| run | physical HBM | TTFT | TPOT | req/s | 命中率（字节） | H2D | D2H | staging |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_capacity | 20,000 | 4.095 ms | 0.531 ms | 287.34 | 99.98% | 0.20 MB | 1.10 MB | 18.23 MB |
| sparse_roomy | 40,000 | 2.400 ms | 0.532 ms | 557.15 | 99.98% | 0.20 MB | 1.10 MB | 18.23 MB |

长上下文 8×8192+64、sparse physical 20,000 / logical 160,000：

| 指标 | baseline | sparse (DSV4.1 L1) |
| --- | --- | --- |
| TTFT / TPOT / 吞吐 | 5.00 ms / 0.5476 ms / 202.53 req/s | 5.00 ms / 0.5565 ms / 199.70 req/s |
| 命中率（条目 / 字节） | — | 97.62% / 98.11% |
| H2D / D2H / staging | — | 7.61 MB / 0.44 MB / 58.33 MB |

这组结果只证明 L0/L1 的容量/IO 路径没有回归；它 **不是** BF16 L2 的
sglang HiSparse 实测。

---

## 8. 覆盖矩阵

| 能力 | `dsv41_vllm_bf16` | `dsv41_fp4_csa2` |
| --- | --- | --- |
| 官方/实测总字节 | ✅ 2885 B/token/rank | ⚠️ 890 B/token + 544 假设 |
| 逐 source layer 字节 | ✅ `(577,577,577,1154)` | ❌ 仍按四层均分 |
| source→consumer | ✅ 已知映射 | ✅ 复用同一映射 |
| index 层/顶层 | ✅ storage/compute/topk | ✅ 复用拓扑 |
| index candidate 路径 | ⚠️ 仅有元数据，未限候选 | ⚠️ 仅有元数据 |
| page/slot/allocator | ✅ 131072/4/4096 | ❌ 未知 |
| block payload/allocator | ✅ 369280/524288 | ❌ 未知 |
| SWA 层/window | ✅ 40/128 | ✅ 层/window |
| SWA 字节 | ✅ 40960 B/token，稳态按请求 | ❌ 未知 |
| compressor 层/page/字节 | ✅ 3/131072/393216 | ⚠️ 仅层号 |
| admission 公式/表 | ✅ 22/405/8533 | ❌ 未知 |
| TP 语义 | ✅ per-rank，TP8 不再除 | ⚠️ global，按 TP 均分 |
| 真实 `.22` 静态布局 | ✅ 源码/布局重建 | ❌ 不适用 |
| sglang HiSparse online trace | ❌ 未跑 | ❌ 未跑 |
| top-k / hot-buffer 计数器 | ❌ 未校准 | ❌ 未校准 |
| host/SSU 带宽/延迟 | ❌ 用通用带宽模型 | ❌ 同左 |
| 1M / graph / SPEC / MTP | ❌ 未验证 | ❌ 未验证 |
| DCP>1 / 连续到达 / LRU 物理淘汰 | ❌ 未验证 | ❌ 未验证 |

---

## 9. 复现命令

```bash
# 单测
PYTHONPATH=src python -m pytest test/test_hisparse_sim.py test/test_hisparse_dsv41.py -q

# 三个 profile 的解析 sweep
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep \
    --kv-profile dsv41 --output /tmp/hisparse_dsv41_sweep.json
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep \
    --kv-profile dsv41_fp4_csa2 --output /tmp/hisparse_fp4_sweep.json
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep \
    --kv-profile dsv41_vllm_bf16 --output /tmp/hisparse_bf16_sweep.json

# 旧 L0/L1 端到端 replay（baseline/sparse 用同一配置，靠 --enable 切换）
PYTHONPATH=src \
BENCHMARK_TEST_MODEL_PATH=/path/to/Qwen3-0.6B \
HISIM_CONFIG_PATH=test/assets/mock/config_hisparse_dsv41.json \
HISIM_VALIDATION_OUT_DIR=/tmp/hisparse_dsv41_results \
    bash benchmark/hisparse/run_validation.sh

# 用环境变量切 BF16 L2 布局
HISIM_HISPARSE_KV_PROFILE=dsv41_vllm_bf16 \
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py bench --enable \
    --config test/assets/mock/config_hisparse_dsv41.json \
    --max-total-tokens 20000 --num-prompts 8 --input-len 8192 --output-len 64 \
    --output /tmp/hisparse_dsv41_results/sparse_bf16.json
```

> `HISIM_HISPARSE=1` 开启仿真，默认关闭；不设 `kv_profile` 时走 legacy 路径。
> baseline 与 sparse 必须分进程运行。

---

## 10. 结论

- L0/L1 的 890 B/token、压缩位置 cadence、TP 缩放和 legacy 零回归保持不变。
- L2 已把 `.22` BF16 的静态布局真正落到代码：逐层字节、page/allocator、
  SWA/compressor、admission 公式、per-rank TP 语义均有单测。
- FP4/CSA2 已把可复用的模型拓扑和 544 B/token index 假设落到
  `dsv41_fp4_csa2`，但绝对页/allocator/SWA/compressor/admission 字节仍是未知，
  不能与 BF16 数值混用。
- 当前仍是 **静态/解析级 L2**：没有真实 indexer、online top-k trace、
  hot-buffer 计数器或 sglang HiSparse 端到端实测。

---

## 11. 两种场景还需要什么

### 11.1 `.22` BF16 场景

1. **sglang HiSparse 动态实测**
   - `.22` 是 vLLM-Ascend，不是 sglang HiSparse；需要在 sglang 0.5.19
     HiSparse 路径上收集逐步 top-k ID/分数、hot-buffer hit/miss、swap-in/out
     字节和实际 H2D/D2H，校准 `churn` / `recent_ratio` / `sticky` 行为。
2. **host/SSU 真实带宽与延迟**
   - 当前 sweep 用通用 64 GB/s + 固定 6.67/4 µs 的传输模型；需要 `.22`
     的 host→device、device→host、SSU 路径实测，尤其是小 block 的尾延迟。
3. **真实 1M context**
   - admission 表是静态公式；需要跑 1M 请求，确认 8533 blocks / 4.4737 GB
     的实际上限、OOM 边界和最大并发，而不是只做算术推导。
4. **graph / SPEC / MTP / DCP>1**
   - `.22` 是 SPEC=0、DCP=1、eager；需要 CUDA graph、MTP 3 层、DSpark
     草稿 KV、DCP>1 的布局与并发实验。
5. **candidate block 选择**
   - 当前只暴露 2048×8 candidate 元数据；需要真实 indexer 的 candidate
     选择 trace，验证 top-k 是否真的优先落在这些块内。
6. **连续到达与物理淘汰**
   - 需要连续 batching、真实 LRU 物理淘汰、跨请求全局去重/生命周期实验；
     当前是顺序请求 + 抽象 access model。
7. **SWA/compressor 稳态 vs admission 验证**
   - 需要实测每请求 SWA ring、compressor page 的峰值/稳态，确认
     `hbm_kv_bytes` 与 admission 上界没有重复或漏算。

### 11.2 FP4/CSA2 场景

1. **FP4 逐 source layer 字节**
   - 用目标 FP4 权重/量化配置确认 `[2,8,14,20]` 各自的真实 B/token；
     当前 `222.5 B/token/layer` 只是均分近似。
2. **FP4 index 布局**
   - 确认 indexer 是 544 B/token（8 层×68）还是 4-source 的 325 B/token 口径，
     以及 index 是否已经包含在 890 里；当前 544 是独立 HBM 假设。
3. **FP4 page/slot/allocator**
   - 需要目标镜像的 page 大小、每 block slot 数、payload/block、allocator
     block 字节和对齐/元数据开销；当前这些字段全为 0。
4. **FP4 admission 公式**
   - 需要目标 FP4 allocator 的 block/slot 实现，确认是否仍是
     `ceil(L/128)+1+10×...`；当前 `admission_bytes(1M)=0` 是明确的“未知”。
5. **FP4 SWA/compressor 字节**
   - 需要确认 SWA/compressor 是否也用 FP4，量化后的每 token/page 字节；
     当前只保留层号和 window，字节为 0。
6. **目标镜像实测**
   - 需要目标 FP4/CSA2 镜像的静态布局导出和 1M/并发/带宽实验；不能把
     `.22` 的 BF16 绝对字节迁移过来。

### 11.3 两个场景共同缺的实验

- 逐步 top-k ID/分数（indexer 真实输出）；
- hot-buffer hit/miss/swap-in/swap-out 字节计数器；
- host/SSU 真实带宽、延迟、尾延迟；
- 真实 1M context 与最大并发；
- CUDA graph、SPEC/MTP、DSpark 草稿路径；
- DCP>1、连续到达吞吐、真实 LRU 物理淘汰；
- 生产级跨请求全局去重与 KV 生命周期；
- PD 分离 / direct RDMA / HiCache 共存（sglang 中 HiSparse 与 HiCache 互斥）。
