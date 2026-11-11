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

---

## 12. 2026-11-09 decode 稀疏实现复核（仅评审，未改代码）

### 12.1 总结与结论边界

本轮复核了 `_WorkingSet`、profile decode、hot buffer、IO/时延模型、sglang hook、
现有单测，并对照了当前 sglang coordinator/allocator/indexer/swap kernel 和
`.22` vLLM-Ascend 交接材料。结论是：**还有影响容量、命中率和时延结论的问题，
不能只在现有模型上继续调 `churn` 来凑结果。**

当前结果仍可用于比较抽象 access model 的相对敏感性，但在完成下述修复前：

- `dsv41_vllm_bf16` 的 HBM 物理 token 数和由其派生的并发结论需要重算；
- decode 命中率、H2D/D2H 和 load latency 只能视为合成模型输出，不能称为
  DSV4.1/sglang HiSparse 实测预测；
- `.22` vLLM-Ascend BF16 与官方 FP4/CSA2、以及当前 sglang C4/C128 路径必须继续
  分开描述，不能互相当作语义验证。

本节对动态 decode 和容量口径的复核结论优先于前文相关描述。尤其前文 7.3 的
BF16 physical tokens 及派生指标应在修复后重跑，不再作为容量结论引用。

问题分级：

- **P0**：已确认的公式/实现错误，或会使验证对象发生根本错位；先修复再校准。
- **P1**：已确认的重要语义缺口，绝对命中率、流量或时延可能明显偏差。
- **P2**：小边界问题或尚需 trace/实测判定的假设，不能伪装成已验证事实。

### 12.2 P0：先修正确定性错误

#### P0-1 BF16 把 2885 payload 错用于 HBM 容量，4096 allocator 只做了展示

证据：

- 交接报告明确区分：有效长程 payload 为 `2885 B/original-token/rank`，物理
  allocator 斜率为 `524288 / 128 = 4096 B/original-token/rank`，两者不能互换；
  SWA、压缩状态和空块另计。
- `hisparse_sim.py:995-1003` 的 `hbm_kv_bytes` 使用
  `physical_pool_tokens * (offload + indexer)`；BF16 此时就是 2885，而没有使用
  已定义的 `allocator_bytes_per_token=4096`。
- `benchmark/hisparse/hisparse_bench.py:146-156` 的 BF16 sweep 同样用 2885 估算
  physical tokens。`baseline_hbm_kv_bytes` 也有相同的容量口径风险。

影响：在相同 80 GB、8 请求稳态 SWA/compressor 扣减的简化预算下，当前为
27,714,007 physical tokens；改用 4096 后应为 **19,520,242**，当前约高估 42%。
这只是公式纠错，不是新的实测容量。IO/staging 仍应使用 2885 payload，不能把
所有字段一刀切改成 4096。

修复方向：增加明确的 `capacity_bytes_per_token` 口径，BF16 HBM 池 sizing 和稳态
HBM 容量使用 allocator 4096；IO 使用 payload 2885；index 是否另计由 profile 决定，
避免重复计费。FP4 的未知 allocator/page 字节继续保持未知，不套用 BF16 数值。

#### P0-2 profile 没有重定 e2e 物理池，scheduler 容量与 metrics 可能互相矛盾

`sglang_hook.py:247-330` 先用 legacy 模型几何或 `max_total_tokens` 得到
physical tokens，再把它交给 profile simulator，随后将 full/SWA/logical 池都设为
`physical * host_to_device_ratio`。profile 的 2885/4096、SWA/compressor 布局并没有
参与这次物理池重定。`max_num_reqs` 还来自通用启发式（最高 4096），不一定是
`.22` 的 8 或实际 server 并发上限。

因此解析 sweep 与端到端 replay 可能使用两套容量来源；统一把
`full_max_total_num_tokens`、`swa_max_total_num_tokens` 都设为 logical tokens，
也没有表达真实分离的 long-KV/SWA/C4/C128 池。

修复方向：要么由 HBM 预算和 profile 重新求 physical pool，再派生各 scheduler
池；要么要求显式传入 physical/logical/SWA 容量并在不一致时失败。并发和
SWA/compressor 稳态字节必须使用同一个实际 deployment 并发来源。

#### P0-3 对当前 sglang 路径，压缩候选长度用了 ceil 而真实 C4 使用 floor

`hisparse_sim.py:1167-1179` 使用：

```text
comp_len = max(1, ceil(past_len / ratio))
```

当前 sglang 的 `metadata_kernel.py:40-49` 则是
`c4_seq_lens_raw = seq_len // 4`，再仅为 kernel 下限 clamp 到 1；
`deepseek_v4_backend.py` 和 indexer 使用该长度做 C4 top-k/swap-in。
在非整除点，两者候选集合不同，仿真会把尚未完整产生的压缩条目提前纳入选择。

同时要区分验证对象：当前 profile 的 `(2,2,2,1)` 来自官方/vLLM-Ascend 布局；
当前 sglang `deepseek_v4.py:624-633` 只接受 `(0,4,128)`，DSV4 HiSparse allocator
固定 `compress_ratio=4`，indexer 的 HiSparse swap-in 也走 C4。现有 2/1 replay
可以保留为 vLLM 启发式模型，但不能称为当前 sglang DSV4.1 语义验证。

**D2H 不在本轮武断判成同样的 off-by-one**：ceil 长度的增量条件恰好是
`(L-1) % ratio == 0`，与 coordinator 的 backup 条件形式一致。因此已确认的是
“用于 top-k 的候选长度错误”；首步跳过、末步未 backup、hook 的 `past_len` 与真实
`seq_lens_cpu` 是否完全同相，仍须用同一请求逐步对拍后定论。

#### P0-4 把“压缩长度不推进”误当成“top-k 不变化且全命中”

`hisparse_sim.py:1173-1179` 在 `comp_len` 不推进时直接返回
`min(top_k, comp_len)` hits、0 misses，不调用 `_WorkingSet.step()`。
压缩条目产生 cadence 只限制候选集合，不保证下一步 query 的 indexer 分数和
Top-K ID 不变。真实 query 每个 decode step 都在变化，即使候选长度不变，也可能
选择不同的历史条目并产生 swap-in。

该 shortcut会系统性高估命中率、低估 H2D；修正 floor 后更不能继续沿用这个假设。
`test_cadence_does_not_churn_on_non_advance_steps` 当前是在固化待验证假设，
不是与真实实现对齐的证明。

修复方向：将 **query/top-k 选择 cadence**、**压缩条目产生 cadence**、
**D2H backup cadence** 拆成三个概念。默认每个 decode step 都更新选择；
`comp_len` 只限制候选范围，D2H 只由新完整条目触发。若需保留旧行为，应显式命名为
`frozen_selection_between_boundaries` 的敏感性对照，而非默认真实语义。

#### P0-5 churn 重采样没有排除 dropped，`churn=1` 并不等于完全换血

`hisparse_sim.py:728-733` 先从 sticky 集合删除 `dropped`，随后 `_sample()` 只排除
剩余 sticky，没有排除刚删除的位置，因此旧位置可以立即被重选。只读探针中，
`top_k=64、recent_ratio=0.5、churn=1` 的短历史场景一步后仍保留旧 sticky 的
30/32；这会使配置 churn 明显高于实际 churn，并高估命中率。

修复方向：候选重采样排除 `dropped`，历史不足时明确截断有效 churn，而不是静默
重选。新增集合级测试：历史充足时 `new_sticky.isdisjoint(dropped)`，并检查
`churn=0/1`、短历史和长时间稳态，而不只检查聚合 miss 数。

#### P0-6 初始 hot buffer seed 与所引用的 sglang 路径不一致

`hisparse_sim.py:649-652` 假定 prefill 把**最新** `capacity` 个位置放入 hot buffer。
当前 sglang `hisparse_coordinator.py:481-483` 的 staging 路径却把 regular device
buffer 初始化为 `arange(device_buffer_size)`，即最早的压缩位置；direct 长序列路径
在 `:410-415` 将槽位置为 `-1`，首次 top-k 全 miss，短序列则走 preload。

这不是无关紧要的初值差异：它决定首个 decode 的 miss 身份、后续 LRU 顺序和
staging/direct 的 H2D 峰值。修复时应分别建 staging-long、direct-long、direct-short
初始化，不要用一个“最新 token 全热”的 seed 覆盖所有路径。

### 12.3 P1：decode 稀疏和 IO/时延的重要语义缺口

#### P1-1 当前不是实际 LRU，且把 newest page 当作普通 hot slot

现有 `OrderedDict` 主要记录插入顺序；已有 hit 不执行 `move_to_end()`，所以它是
“受当前 recent/sticky 集合保护的插入顺序缓存”，不是真实 LRU。实际 swap kernel
会把 hits 移到 MRU、misses 插到 hits 前面（`hisparse.cuh:640-670`）。

此外 simulator 把 `padded_buffer_size=device_buffer_size+page_size` 全部作为通用
hot capacity；真实布局是 `[device_buffer_size 个 LRU slots] + [newest page]`，newest
page 不参与 LRU（`hisparse.cuh:314-315`）。修复时应使用分离的 regular slot/newest
slot 模型，并用小型手工 top-k 序列对拍每步淘汰顺序。

#### P1-2 每个 ratio 只有一条独立随机流，未表达真实层间选择与 candidate 路径

当前 ratio-2 的三个 source 被合成一个 working set，ratio-1 另用一个独立 RNG；
但 profile 自己声明有 4 个持久 source、8 个 index compute layers，以及
“source 20 的 2048×8 candidate -> filters”的路径。candidate/filter 目前只有元数据，
并未限制候选或产生流量。

这既不能证明各 source 应共享同一 Top-K，也不能证明它们应独立。后续应由真实
layer trace 决定共享关系，至少记录 per-source/per-index-layer Top-K ID、miss ID 和
复用关系。`shared_index_freq` 也不能替代这些结构信息。

#### P1-3 `shared_index_freq` 简单把关键路径字节除以 N，不代表 anchor/follower 时序

`_account_transfer()` 对聚合 H2D bytes 乘 `1/shared_index_freq`，总字节不变、
launch overhead 只计一次。真实 shared-index 路径是 anchor 同步 swap-in 并记录 miss
plan，skip layers 在侧流复用 plan；关键路径取决于 anchor 的字节、事件等待和计算
重叠，不是任意两组不同大小 IO 的统一 `1/N`。

修复方向：保留总流量与关键路径流量的区别，按 anchor/follower 建事件和传输，
不要用一个频率参数同时隐含选择共享、预取字节和延迟缩放。

#### P1-4 main KV 与 index IO 混算，命中时也没有 index 读取流量

当前 miss entry 一次性计 `main KV + index` 的 H2D，新条目也按同一 entry bytes 计
D2H。`.22` 交接报告则明确要求分别建：索引扫描、候选筛选、Top-K ID 返回、稀疏
主 KV 取回；Top-512 之前仍可能读取索引，**命中 main KV 不等于没有 index IO**。

因此当前模型既可能漏掉 index scan，又可能把 index 只在 miss 时随 main KV 搬运，
无法回答 SSU/HBM/host 各自瓶颈。修复时至少拆分
`index_scan_bytes`、`candidate/topk_bytes`、`main_kv_h2d_bytes`、
`main_kv_d2h_bytes` 和 resident index 容量；325 已包含在 2885 内，拆账时不能双计。
对 sglang C4 的 resident index 与 `.22` 远端 index 应使用不同 profile，而不是套同一
传输公式。

#### P1-5 prefix 复用和 host 生命周期没有建模

hook 在 `sglang_hook.py:1838-1845` 以完整 input+output 长度启动 staging，没有减去
已复用的 `prefix_indices`；`on_request_finish()` 又释放该请求全部 host tokens。
对于交接材料中的重复请求，这会把已有 prefix 再完整 staging，并不表达请求结束后
仍保留的共享 prefix。唯一 prefix 的 workload 受影响较小，重复/多轮 workload 的
staging、host 峰值和 TTFT 会明显失真。

修复方向：显式输入 prompt/prefix/new-token 长度，给 host prefix 建引用计数或明确
逐请求私有语义，并用“首次请求/重复请求”两个案例验收。

#### P1-6 传输与计算时间线仍过于理想化

- prefill staging 只累计 bytes，不计传输完成事件或 admission 等待，TTFT 偏乐观；
- backup 默认只累计 `backup_hidden_dur`，没有队列、event 或 backpressure，D2H 慢于
  decode 时仍可无限隐藏；
- 整个 batch 的所有请求/组只计一次 H2D fixed overhead，而真实 swap-in 按 layer/
  anchor 调用，小传输场景会低估 launch/tail；
- H2D 默认全部串行加到 forward，非 anchor 传输与 attention 的重叠没有按事件建模；
- 通用 64 GB/s 和固定 `t0` 不是 `.22` host/SSU 实测参数。

修复方向：先测单次 kernel/传输的 `t0`、有效带宽和并发流，再建 staging、anchor
H2D、follower prefetch、D2H backup、forward 的事件图；关键路径由依赖和
`max(compute, overlapped IO)` 决定，而不是固定全加或固定隐藏。

### 12.4 P2 与测试解释

- `recent_ratio=0` 仍由 `_recent_count()` 强制保留 1 个 recent entry；若 0 的语义应是
  纯 sticky，需要补边界测试并修正文档/实现之一。
- 模块注释称“page-granular LRU”，当前实际按抽象 entry 位置管理；page/entry 单位
  必须在目标 profile 中明确，不能顺手把 ratio-2 capacity 减半或加倍。
- `test_compressed_working_set_cadence`、`test_top_k_is_selected_on_compressed_positions`
  当前断言 ceil；BF16 测试断言 `physical_tokens * 2885` 的 HBM；这些测试证明的是
  现有公式自洽，不能作为真实语义 oracle。
- 现有 46 个目标测试通过不覆盖上述问题。后续应把“公式单元测试”和“参考实现/trace
  对拍测试”分开命名，避免再次把假设写成断言。

### 12.5 分阶段修复计划

#### 阶段 A：锁定目标与逐步 oracle（先于任何性能调参）

1. 分开声明 `vllm_ascend_dsv41` 与 `sglang_dsv4_hisparse` 目标；前者保留 2/1 和
   2885/4096 双口径，后者按 C4/C128 真实池与 indexer 路径建模。
2. 定义逐步 trace：`rid, seq_len, comp_len, layer/source, top-k IDs, hot slots,
   miss IDs, newest hit, H2D/D2H bytes, staging/backup events`。
3. 用同一请求对齐 hook 的 `past_len`、真实 `seq_lens_cpu`、压缩产生步和 backup 步；
   特别核对首步、整除边界、末步，不预先假定 D2H 总量错误。

验收：短小序列可手工逐步解释；floor/ceil、首步和最后一个 token 均有明确预期。

#### 阶段 B：修确定性错误和 hot-buffer 状态机

1. BF16 容量改用 4096 allocator，IO 保持 2885 payload；统一 hook 与 sweep 的
   physical/logical/SWA 池及 deployment 并发来源。
2. 修正目标对应的 `comp_len`，拆开选择/压缩/backup cadence；删除非推进 step 的
   默认全命中 shortcut。
3. churn 重选排除 dropped；实现 staging/direct 初始化、hit-to-MRU、miss 插入和
   独立 newest page。
4. 保留 legacy 路径和 HiSparse 默认关闭行为，不改旧几何默认值。

验收：集合级/LRU 顺序测试通过；churn=1 在历史充足时不重选；同一 trace 的逐步
hits/misses、hot slots、H2D/D2H 与 oracle 一致；BF16 容量测试显式检查
`tokens*4096 + SWA + compressor`，legacy 测试零回归。

#### 阶段 C：重建分层 IO 和事件时序

1. 按 source/index layer 建选择及共享关系，接入 candidate/filter，而非只按 ratio
   聚合并独立随机。
2. 拆 index scan、candidate/top-k、main KV H2D/D2H、resident bytes；明确每类流量的
   HBM/host/SSU 来源与去向。
3. 用实测 kernel/链路的 fixed overhead、带宽、anchor/follower 并发和 staging/backup
   event 替换统一 `1/N`、单 launch、全串行/全隐藏近似。

验收：总 bytes 守恒且无 index 双计；hit/miss 路径的 index IO 均按目标 profile
计费（resident index 可以有明确的 0 远程字节）；batch/layer 数变化时 launch 次数
正确；D2H 积压会产生可观察 backpressure；与实测分步时间线误差有报告。

#### 阶段 D：补生命周期、重跑实验并恢复结论

1. 加 prefix 复用、host 引用生命周期、连续到达和多请求并发。
2. 用真实 Top-K/hot-buffer/IO trace 校准 `access_model/churn/recent_ratio`，不再用
   这些参数掩盖结构性错误。
3. 重跑单测、目标 replay 和三种 profile sweep；将旧 BF16 容量、命中率和绝对时延
   结果标记为 superseded，保留仅用于解释修复前后差异。
4. 最后才恢复“容量/吞吐/TTFT/TPOT 收益”结论；没有实测校准的 FP4 page、allocator、
   SWA/compressor 和 SSU 参数继续留空或标假设。

### 12.6 建议执行顺序

最小安全顺序是：**A 语义对拍 -> B 容量/cadence/churn/LRU -> C 分层 IO/时序 ->
D 生命周期与实测校准**。不建议先改带宽或调 churn，因为那只会让错误状态机得到
看起来合理、但无法验证的曲线。

12.1-12.6 是 2026-11-09 的评审结论，最初未修改代码；修复落地、重跑结果和
仍缺的实测项见下一节。

### 12.7 第一次修复落地（2026-11-09）

> 本节保留第一次修复及其复现记录。二次评审发现的 prefix 调用时机、direct-short、
> 奇数 prompt 边界、LRU 混合顺序和并发口径问题已在 §12.8 修正；动态结果以 §12.8
> 为准，§12.7.2 仅用于解释修复前后差异。

#### 12.7.1 已完成

| 项 | 状态 | 实现 |
| --- | --- | --- |
| BF16 HBM 容量 | ✅ | 使用 `capacity_bytes_per_token=4096` allocator 斜率；IO/staging 继续用 2885 payload |
| e2e 物理池 | ✅ | hook 用同一字节预算和 profile capacity 重算 physical tokens；BF16 默认并发 8 |
| 压缩长度 | ✅ | ratio 组改用 `floor(past_len/ratio)` |
| 选择 cadence | ✅ | 默认 `every_step`；`compression_boundary` 仅作显式敏感性模式 |
| churn 重选 | ✅ | dropped 不再立即重选；历史不足时保持 Top-K 目标大小 |
| hot buffer 初始化 | ✅ | profile 默认 earliest；支持 earliest/latest/empty |
| hot buffer LRU | ✅ | hit-to-MRU、独立 newest slot、regular capacity 不含 extra page |
| main/index IO 分账 | ✅ | BF16 逐 source split：main `(512,512,512,1024)`、index `(65,65,65,130)` |
| index 命中读取 | ✅ | `index_read_on_hit=True` 时 hit/miss 都计 index payload 读取 |
| prefix staging | ✅ | `on_request_start(..., prefix_len=)` 只 staging 新增部分；hook 传入 `prefix_indices` |
| recent_ratio=0 | ✅ | 不再强制保留 1 个 recent entry |
| 传输 launch | ⚠️ | index scan 与 main KV 分两阶段计 fixed overhead；尚未按真实 layer 调用次数展开 |
| 其他指标 | ✅ | 新增 main/index H2D/D2H、critical H2D、staging latency、prefix tokens、runtime target |

目标测试：

```text
PYTHONPATH=src python -m pytest test/test_hisparse_sim.py test/test_hisparse_dsv41.py -q
53 passed
```

legacy 无 profile 路径仍保留原有 FIFO/hot 近似；HiSparse 默认关闭不变。
`hisparse_sim.py`、`sglang_hook.py`、`hisparse_bench.py`、mock 配置和测试已更新。

#### 12.7.2 重跑结果

`sweep` 公共 workload 仍为 64 请求 × 32K prompt + 512 output、80 GB 预算、
`top_k=512`、`device_buffer=1024`、host ratio 8、`recent_ratio=0.5`。
旧 7.1-7.3 的动态数字已被本节取代。

`dsv41`（890 B/token）：

| churn | 命中率（条目/字节） | H2D | D2H | staging | mean load |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 97.19% / 97.24% | 0.721 GB | 0.0290 GB | 1.866 GB | 7.07 µs |
| 0.20 | 89.96% / 90.05% | 2.600 GB | 0.0290 GB | 1.866 GB | 8.13 µs |
| 0.50 | 75.31% / 75.49% | 6.404 GB | 0.0290 GB | 1.866 GB | 10.26 µs |

physical tokens = 89,887,640；host = 640 GB；`max_reqs_hot_pool=87,695`。

`dsv41_fp4_csa2`（890 + 544 假设）：

| churn | 命中率（条目/字节） | H2D | mean load |
| --- | --- | --- | --- |
| 0.05 | 97.19% / 97.24% | 0.721 GB | 7.07 µs |
| 0.20 | 89.96% / 90.05% | 2.600 GB | 8.13 µs |
| 0.50 | 75.31% / 75.49% | 6.404 GB | 10.26 µs |

physical tokens = 55,788,005（capacity 1434 B/token）；host = 397.21 GB；
`max_reqs_hot_pool=54,427`。FP4 page/allocator/SWA/compressor 仍未测。

`dsv41_vllm_bf16`（4096 B/token/rank 容量）：

| churn | 命中率（条目/字节） | main H2D | index H2D | D2H | staging | mean load |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 97.19% / 97.23% | 1.906 GB | 8.724 GB | 0.0942 GB | 6.050 GB | 19.30 µs |
| 0.20 | 89.96% / 90.02% | 6.856 GB | 8.724 GB | 0.0942 GB | 6.050 GB | 22.08 µs |
| 0.50 | 75.31% / 75.44% | 16.879 GB | 8.724 GB | 0.0942 GB | 6.050 GB | 27.70 µs |

physical tokens = **19,520,242**（旧 27,714,007 高估 42%）；host = 450.53 GB；
`max_reqs_hot_pool=19,044`；admission 1M 仍为 8533 blocks / 4.4737 GB。
index H2D 在三个 churn 下相同，因为每步选中的 index entry 数不变；main H2D
随 miss 增加。这里的 index 读取是“选中 entry 的 payload”，**不是全候选扫描**，
因此是下界而非实测预测。

#### 12.7.3 仍不能宣称已验证

- 当前 profile 仍是 **vLLM-Ascend/official layout + sglang hook harness**；
  sglang 当前 C4/C128 的独立池、`compress_ratio=4` 和真实 candidate 路径尚未
  建 profile。`runtime_target=vllm_ascend` 已写入 metrics。
- `churn`/`recent_ratio` 仍是合成 access model，未用真实 Top-K ID 校准；不能把
  97% 或 75% 当目标模型的真实命中率。
- 仍按 ratio 聚合 source 层，未建模 8 个 index compute 层的共享/anchor 关系和
  source20 的 2048×8 candidate/filter 流量。
- index 目前只读选中的 payload；全候选 index scan、Top-K ID 返回和 resident
  index 的 HBM 带宽仍未建模。
- `shared_index_freq` 仍只是关键路径字节/延迟缩放；已有 critical bytes 指标，
  但没有 anchor event + follower 侧流事件图。
- prefix 复用已从 staging 中扣除，但 host 侧共享 prefix 的引用生命周期仍未完整
  建模；当前按逐请求新增 host token 计费。
- staging latency 已单独记录，但还没有作为 admission event 接入 scheduler；
  eager backup 仍只累计 hidden duration，没有队列/backpressure。
- FP4 的 page、allocator、SWA、compressor、SSU 带宽/尾延迟仍为未知或假设。

因此本轮修复解决了容量分母、cadence、churn、初始化/LRU 和 IO 分账等
**确定性错误**；绝对吞吐/TPOT 结论仍需真实 indexer/hot-buffer/SSU trace 后才能恢复。

### 12.8 第二次修复（2026-11-10）

#### 12.8.1 修复内容

二次评审在 `555fe9d` 上确认了以下确定性问题，并已修正：

| 项 | 修复 |
| --- | --- |
| prefix 调用时机 | 不再在 `TokenizedGenerateReqInput` 进入 dispatcher 前读取 `prefix_indices`；改为 prefill 批次生成后、radix 匹配已完成时启动 staging |
| direct-short | 仅当压缩长度不超过 device buffer 时，newest page 老化无 IO 进入 regular LRU，且不产生虚假主 KV H2D |
| 奇数 prompt | staging 改用 `floor(prompt_len/ratio)`；与 decode 的 floor 压缩长度对齐，partial entry 不再先 staging 后重复 D2H |
| 混合 hit/miss LRU | 先处理 hits 到 MRU，再把 misses 插到 hit block 之前；不再按 token position 交错 |
| recency 对照 | `access_model=recency` 固定选择最近 `top_k`，不再被 `recent_ratio` 缩短 |
| DP/显式容量并发 | server 的全局 `max_running_requests` 先换算为 per-worker；显式 `max_total_tokens` 也使用最终 worker 并发 |
| hot-pool clamp | 物理 token 数与请求 overhead 联立求解，clamp 后的并发就是实际预留 SWA/compressor 的并发 |

direct-short 的“无 IO”指主 KV swap-in 为零；BF16 中 `index_read_on_hit=True` 的
resident index 读取仍按选定 entry 计费，不是免费。这里的 newest 免费老化只适用于
direct-short；长序列 previous-newest 的语义由 §12.9 修正。

新增回归覆盖：redispatcher 后 prefix、direct-short/newest aging、mixed hit/miss
LRU 顺序、奇数与小于一个压缩块的 prompt、recency 选满 top_k、DP per-worker
并发和 hot-pool 固定点。目标测试：

```text
PYTHONPATH=src python -m pytest test/test_hisparse_sim.py test/test_hisparse_dsv41.py -q
60 passed
```

#### 12.8.2 重跑结果

公共 workload 不变：64 请求 × 32K prompt + 512 output、80 GB 预算、
`top_k=512`、`device_buffer=1024`、host ratio 8、`recent_ratio=0.5`。
以下数字取代 §12.7.2；其中长序列 newest 记账已被 §12.9 修正，动态结果以 §12.9.2
为准。

`dsv41`（890 B/token）：

| churn | 命中率（条目/字节） | H2D | D2H | staging | mean load |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 97.34% / 97.35% | 0.6918 GB | 0.0290 GB | 1.8665 GB | 7.06 µs |
| 0.20 | 90.10% / 90.16% | 2.5707 GB | 0.0290 GB | 1.8665 GB | 8.11 µs |
| 0.50 | 75.46% / 75.60% | 6.3749 GB | 0.0290 GB | 1.8665 GB | 10.25 µs |

physical tokens = 89,887,640；host = 640 GB；`max_reqs_hot_pool=87,695`。

`dsv41_fp4_csa2`（890 + 544 假设）：

| churn | 命中率（条目/字节） | H2D | mean load |
| --- | --- | --- | --- |
| 0.05 | 97.34% / 97.35% | 0.6918 GB | 7.06 µs |
| 0.20 | 90.10% / 90.16% | 2.5707 GB | 8.11 µs |
| 0.50 | 75.46% / 75.60% | 6.3749 GB | 10.25 µs |

physical tokens = 55,788,005；host = 397.21 GB；`max_reqs_hot_pool=54,427`。
FP4 page/allocator/SWA/compressor/SSU 参数仍未测。

`dsv41_vllm_bf16`（4096 B/token/rank 容量）：

| churn | 命中率（条目/字节） | main H2D | index H2D | D2H | staging | mean load |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 97.34% / 97.35% | 1.8226 GB | 8.7242 GB | 0.0942 GB | 6.0503 GB | 19.26 µs |
| 0.20 | 90.10% / 90.14% | 6.7725 GB | 8.7242 GB | 0.0942 GB | 6.0503 GB | 22.03 µs |
| 0.50 | 75.46% / 75.56% | 16.7953 GB | 8.7242 GB | 0.0942 GB | 6.0503 GB | 27.66 µs |

physical tokens = **19,520,242**；host = 450.53 GB；`max_reqs_hot_pool=19,044`；
admission 1M 仍为 8533 blocks / 4.4737 GB。index H2D 在三个 churn 下相同，因为
选中 entry 数不变；它仍是 selected-payload 下界，不是全候选扫描。

#### 12.8.3 仍未验证

§12.7.3 列出的真实 Top-K/hot-buffer/SSU 校准、C4/C128 独立池、candidate/filter
路径、anchor/follower 事件、shared prefix 生命周期和 backup backpressure 仍未完成。
§12.8.2 仍是合成 access model 的敏感性结果，不能当作真实模型命中率或实测 TPOT。

### 12.9 第三次修复（2026-11-11，本地时间）

#### 12.9.1 修复内容

| 项 | 修复 |
| --- | --- |
| 长序列 previous-newest | 不再无条件迁入 regular LRU。只有 direct-short（压缩长度不超过 device buffer）保留免费老化；长序列若上一 newest 仍被 Top-K 选中且不在 regular LRU，按正常 miss/H2D 计费 |
| 不可行 hot-pool 预算 | `physical_pool_tokens_for_budget()` 在预算不足时返回 0；`hot_pool_geometry_for_budget()` 无法容纳一个 padded buffer 时抛出 `ValueError`；simulator 拒绝 `physical_pool_tokens < padded_buffer_size` 的配置 |

第二条避免把 `budget=1 B` 之类的配置静默 clamp 成一个请求：BF16 一个请求至少需要
SWA/compressor 加 `padded_buffer_size` 个 hot-pool token。错误会直接中止配置流程，
而不再生成超预算的仿真结果。

新增回归覆盖：

- `top_k=2、capacity=4、recency` 的 direct-short 仍为 `(2 hits, 0 misses)`；
- 同一配置跨入长序列后，previous-newest 在 regular LRU 外时返回
  `(1 hit, 1 miss)`，并随后进入 regular LRU；
- 1 B 预算被拒绝，物理池小于一个 padded buffer 时 simulator 抛错。

目标测试：

```text
PYTHONPATH=src python -m pytest test/test_hisparse_sim.py test/test_hisparse_dsv41.py -q
# 63 passed
```

#### 12.9.2 重跑结果

公共 workload 与 §12.8.2 相同。长序列 newest 修正后，当前结果回到与 §12.7.2
接近的命中率/H2D 水平；之前的 §12.8.2 少算了一个 previous-newest miss/步。

`dsv41`（890 B/token）：

| churn | 命中率（条目/字节） | H2D | D2H | staging | mean load |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 97.19% / 97.24% | 0.7208 GB | 0.0290 GB | 1.8665 GB | 7.07 µs |
| 0.20 | 89.96% / 90.05% | 2.5997 GB | 0.0290 GB | 1.8665 GB | 8.13 µs |
| 0.50 | 75.31% / 75.49% | 6.4040 GB | 0.0290 GB | 1.8665 GB | 10.26 µs |

physical tokens = 89,887,640；host = 640 GB；`max_reqs_hot_pool=87,695`。

`dsv41_fp4_csa2`（890 + 544 假设）：

| churn | 命中率（条目/字节） | H2D | mean load |
| --- | --- | --- | --- |
| 0.05 | 97.19% / 97.24% | 0.7208 GB | 7.07 µs |
| 0.20 | 89.96% / 90.05% | 2.5997 GB | 8.13 µs |
| 0.50 | 75.31% / 75.49% | 6.4040 GB | 10.26 µs |

physical tokens = 55,788,005；host = 397.21 GB；`max_reqs_hot_pool=54,427`。
FP4 page/allocator/SWA/compressor/SSU 参数仍未测。

`dsv41_vllm_bf16`（4096 B/token/rank 容量）：

| churn | 命中率（条目/字节） | main H2D | index H2D | 总 H2D | D2H | staging | mean load |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 97.19% / 97.23% | 1.9062 GB | 8.7242 GB | 10.6303 GB | 0.0942 GB | 6.0503 GB | 19.30 µs |
| 0.20 | 89.96% / 90.02% | 6.8562 GB | 8.7242 GB | 15.5803 GB | 0.0942 GB | 6.0503 GB | 22.08 µs |
| 0.50 | 75.31% / 75.44% | 16.8790 GB | 8.7242 GB | 25.6031 GB | 0.0942 GB | 6.0503 GB | 27.70 µs |

physical tokens = **19,520,242**；host = 450.53 GB；`max_reqs_hot_pool=19,044`；
admission 1M 仍为 8533 blocks / 4.4737 GB。index H2D 在三个 churn 下相同，因为
选中 entry 数不变；它仍是 selected-payload 下界，不是全候选扫描。

#### 12.9.3 仍未验证

§12.8.3 的边界继续有效：真实 Top-K/hot-buffer/SSU trace、C4/C128 独立池、
candidate/filter、anchor/follower 事件、shared prefix 生命周期、backup
backpressure 和真实 e2e replay 均未完成。§12.9.2 仍是合成 access model 的敏感性
结果，不能当作真实模型命中率或实测 TPOT。
