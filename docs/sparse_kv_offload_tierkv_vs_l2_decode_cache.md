# 稀疏 KV 卸载方案详解：TierKV 与 L2 Decode Cache / KV Offload

编写日期：2026-11-12
用途：**自包含技术材料**。读者仅凭本文即可理解两个方案的完整设计、数据与差异。

资料范围：

| 编号 | 方案 | 形态 | 可信度 |
| --- | --- | --- | --- |
| S1 | TierKV | 全文整理 | 含原始团队测试数据；未在本文环境复现 |
| S2 | L2 Decode Cache / KV Offload | 架构图 + 匿名客户数据表 + 说明 | 第二手材料；架构与客户数据为转述，未提供完整推导 |
| S3 | 本仓库 `docs/tierkv_vs_hisparse.md` | 对比报告（含 §7 评审与 §8 复核） | 本仓库自产，含对 S1 的二次分析 |

> 说明：S1 的原文数字（如 TPS 23.3→45.8、命中率 97.76%）是**原始测试结果**，
> 本文只做转述与结构化，不代表在本仓库环境复现。S2 的客户吞吐数字同理。
> 凡属本文推导（如 GLM 层数反推）均已标注“推导”。

---

## 0. 速览：两个方案根本不是同一层的东西

一句话区分：

- **TierKV（S1）是“模型侧 + 单实例系统侧”的优化**：解决**单个推理实例内部**
  “HBM 装不下完整 KV”的问题，靠**稀疏索引加速 + 预取 + 计算传输重叠**把
  卸载的时延代价压下去。它不改变并行策略，不涉及多卡共享。
- **L2 Decode Cache / KV Offload（S2）是“集群侧 + 内存架构侧”的方案**：
  解决**多卡 TP 域内**“HBM 总量装不下 1M 序列、且 DCP 方案通信开销大”的问题，
  靠**多卡共享 DRAM 池（GVA 统一编址）+ 卡侧直访 + RD2H 直传**替代 DCP。

因此二者**不是竞争关系，而是可以叠加的两层**：S2 提供“更大的共享池 + 更短的
搬运路径”，S1 提供“更少的搬运量 + 更好的时延隐藏”。详见 §4。

| 维度 | TierKV（S1） | L2 Decode Cache（S2） |
| --- | --- | --- |
| 要解决的瓶颈 | 单实例 HBM 容量 → 并发/BS 受限 | TP 域 HBM 总量 → 1M 序列装不下；DCP 通信开销高 |
| 卸载目标介质 | DRAM → SSD → ASU（分层） | 本机 DRAM（MF 统一编址共享池）；SSU 为下一级 |
| 共享粒度 | 每请求私有 host 池 | **TP 域内一份 KV，多卡同地址共享** |
| 并行策略 | 保持 TP/EP，不做序列切分 | **用 Offload 替代 DCP（序列切分）** |
| 关键路径手段 | Indexer 加速、无损预取、分段 SFA + 多流 | H2D&SFA 融合算子、卡侧直访 DRAM、RD2H 直传 |
| 是否改模型算子 | **是**（`lightning_indexer_hi`、SFA 扩展） | 否（只改系统/内存路径） |
| 实测规模 | 64K/128K（1M 为目标） | **1M（客户已上线/POC）** |
| 收益口径 | TPS 23.3 → 45.8（约 1.97×）@80ms，单机 A3，64K | 1.2×~1.3× vs DCP @1M，多节点 |
| 精度影响 | 两阶段 indexer 有 IoU 损失（top-m=64 掉 2%） | 无（不改稀疏选择） |

---

## 1. 共同背景：为什么 KV Cache 必须卸载

### 1.1 KV Cache 的容量公式

对 MLA（Multi-head Latent Attention）模型，每 token 每层的 KV 只存
`kv_lora_rank + qk_rope_head_dim` 个元素（而非 `2 × heads × head_dim`）：

```
KV_bytes ≈ B × L × num_layers × (kv_lora_rank + qk_rope_head_dim) × itemsize
```

其中 `B` 是并发/批大小，`L` 是上下文长度。**`B` 和 `L` 都会线性放大 KV**，
而 HBM 容量增长远慢于模型上下文窗口与并发需求。

以 S2 的 GLM5.2/5.3 1M 场景为例（材料原文）：

```
1M MLA KV ≈ 90 GB
  = tokens × layers × (kv_lora_rank + qk_rope_head_dim) × bf16_bytes
```

> **推导**（本文）：若 `kv_lora_rank=512`、`qk_rope_head_dim=64`、BF16（2 B），
> 则每 token 每层 1152 B；`90e9 / (1e6 × 1152) ≈ 78` 层。即 GLM5.2/5.3 约 78 层。
> 材料未给出层数，此为反推，仅用于理解量级。

A3 单卡 64 GB HBM，典配场景单卡只剩约 25 GB 留给 KV Cache —— **装不下 90 GB**。
这就是两个方案共同的出发点。

### 1.2 稀疏注意力只解决“访问成本”，不解决“存储成本”

这是 S1 最关键的洞察，值得单独强调：

- DSA（DeepSeek Sparse Attention）类模型自带 indexer，每个 decode step 只选
  Top-K（如 2048 或 512）个历史 token 做注意力。
- 但**任意历史 token 都可能在后续 step 被 indexer 选中**，所以系统通常仍必须在
  HBM 里保留**完整** KV；部分模型还要额外常驻索引 Key。
- 结果：稀疏注意力降低了计算量（访问成本），但**没有降低 HBM 占用**（存储成本），
  于是 HBM 容量依然限制长请求并发数，稀疏带来的计算效率无法转化为端到端吞吐。

S2 的表述是同一件事的另一面：“DSA 稀疏命中（TopK 2048），**HBM 利用率不高**”——
因为 HBM 里 99% 的 KV 在某个 step 不会被访问，却必须常驻。

### 1.3 卸载范式

两个方案共享同一个基本范式：

```
完整 KV ──► 大容量介质（DRAM / SSD / SSU）
              ▲        │
   写回/溢出   │        │  按需召回（Top-K miss）
              │        ▼
HBM：只保留当前需要的热点 KV（hot pool）
              │
              └──► 稀疏 attention 只读 HBM
```

差别在于“大容量介质”的组织方式（私有 vs 共享）、“召回路径”的工程实现
（标准 H2D vs 卡侧直访/RD2H），以及“如何隐藏召回时延”（S1 的核心贡献）。

---

## 2. TierKV 方案详解（S1）

### 2.1 定位与总体架构

**全称**：TierKV：面向百万级长上下文的分层稀疏 KV 缓存卸载与推理加速架构。

**核心思路**：将完整历史 KV Cache 保存在大容量介质中，仅将当前注意力计算所需的
Top-K KV 和高频热点数据动态加载到 HBM，使 **KV Cache 的存储位置与实际访问位置解耦**。
由于每个解码步骤只访问完整上下文中的少量 KV，理论上可以用较低的数据传输量换取
显著的 HBM 容量节省，从而扩大 Batch Size 和请求并发度。

**分层结构**：

```
┌─────────────────────────────────────────────┐
│ HBM：hot pool（Top-K + 高频热点）            │  ← 每请求固定预算
│      + indexer 索引（部分模型常驻）           │
└───────────────▲──────────────┬──────────────┘
      H2D 召回  │              │  D2H eager backup
┌───────────────┴──────────────▼──────────────┐
│ DRAM（pinned host memory，全量 KV）           │
└───────────────▲──────────────┬──────────────┘
                │              │
┌───────────────┴──────────────▼──────────────┐
│ SSD / ASU（后续扩展，当前未实测）             │
└─────────────────────────────────────────────┘
```

**稀疏卸载带来的三个新系统挑战**（原文明确列出，很重要）：

1. **索引结果运行时动态生成，Top-K 分布离散** → 容易产生大量**随机小粒度传输**；
2. **KV 召回发生在索引计算之后** → Host-to-Device 传输直接暴露在**解码关键路径**上，
   显著增加 TPOT；
3. **动态缓存分配、淘汰和地址更新可能破坏入图**（graph capture）→ 引入额外
   CPU 调度与同步开销。

因此核心问题变成：**在保证稀疏选择结果和模型精度不变的前提下，尽可能隐藏数据移动
开销，把释放的 HBM 容量真正转化为更高的并发度和端到端吞吐。**

对应三大关键模块（下文 §2.2–§2.4）：

| 模块 | 解决的问题 | 手段 |
| --- | --- | --- |
| Top-K 检索加速 | indexer 本身成为瓶颈，且挤占传输时延窗口 | 块级粗筛 + Token 级精选（两阶段） |
| 无损预取 | 召回暴露在关键路径 | 预测下一层访问集，提前异步加载 |
| 计算与传输并行掩盖 | 串行的“检索+传输+注意力” | 分段 SFA + micro-batch + 多流 |

### 2.2 两阶段层级 Indexer（`lightning_indexer_hi`）

#### 2.2.1 动机：indexer 已成为 attention 的核心瓶颈

原文给出的三组证据：

| 场景 | 数据 |
| --- | --- |
| DS V3.2 稀疏模型 | 稀疏选择算子（`lightning_indexer`，简称 indexer）在 attention 中占比 **>30%**（prefill，大 batch，60K） |
| DS V4 稀疏模型 | 512K 输入长度下，indexer 占**总算子时延 >30%**，占**计算算子时延 >40%**（batch=1，seq_in=512K，seq_out=256） |
| Ascend 950PR 单算子 | 相同输入下，`lightning_indexer` 与 `sparse_flash_attention` 时延比 **>8×**（decode），序列越长越严重 |

结论：序列越长，indexer 越是 attention 的核心瓶颈。**降低 indexer 时延可直接带来
TPOT 收益，并为稀疏加载提供更大的时延窗口**（这句话是 TierKV 三大模块能协同的关键：
indexer 变快 → 留给 H2D 的时间窗口变长 → 预取/重叠才有空间）。

#### 2.2.2 核心依据：Top-K token 存在局部聚集性

原文通过分析发现：indexer 选出的 top-k token 在位置上**存在布局聚集性**
（不是均匀散布在整段历史中）。因此可以“先粗选出局部聚集的 top-m 个 block，
再从这些 block 中精确选出 top-k 个 token”。

#### 2.2.3 算法：块粗筛 + Token 精选

```
输入：query q，key cache（长度 L），权重 weights
参数：block 大小 B=128，block 数 M=L/B，保留 top-m 个 block

第一阶段（block 粗筛）：
  1. 把 key 按 B=128 分组池化，得到 M 个 block 表征向量（默认 mean）
  2. q 与 M 个 block 表征做矩阵乘，为每个 block 打分
  3. 取 top-m 个 block（外加 sink/recent 必选块）
     → 候选 token 数 = m × B

第二阶段（token 精选）：
  4. 在 m×B 个候选 token 上，用原始细粒度 token 级 indexer 逻辑精确打分
  5. 取 top-k 个 token 作为注意力输入
```

**复杂度**：`O(L²) → O(L/B + mB)`。

> 注意（来自本仓库评审 §7.2/§8.3）：上式是**一般 Q-query 形式**。decode 只有
> 1 个 query 时，原始 indexer 是 `O(L)`，两阶段是 `O(L/B + mB)`；prefill 有
> 约 `Q=L` 个 query 时，原始才是 `O(L²)`。引用时必须区分口径，否则会高估
> decode 下两阶段的相对收益。

#### 2.2.4 算子参数

`lightning_indexer_hi` 相比原始 indexer 新增 5 个输入参数（均为第一阶段 block 粗筛所需）：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `hi_block_size` | 第一阶段 block 大小，目前固定与 key block size 对齐 | 128 |
| `hi_block_num` | 第一阶段选择的 block 数（即 top-m），按 HISA 论文设默认 | 64 |
| `sink` | 第一阶段选择的最远必选 block 数 | 1 |
| `recent` | 第一阶段选择的最近必选 block 数，必须 ≥1 | 1 |
| `block_pooling_mode` | block 表征计算方式 | `mean` |

进入第二阶段的候选 token 数 = `hi_block_size × hi_block_num` = 128 × 64 = **8192**。

调用接口（原文）：

```python
topk_indices = torch.ops._C_ascend.npu_lightning_indexer_hi(
    query=q,
    key=kv_cache[2],
    weights=weights,
    actual_seq_lengths_query=actual_seq_lengths_query,
    actual_seq_lengths_key=actual_seq_lengths_key,
    block_table=block_table,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=2048,
    sparse_mode=3,
    hi_block_size=128,
    hi_block_num=64,
    sink=1,
    recent=1,
    block_pooling_mode="mean")
```

#### 2.2.5 进阶一：均值缓存（`lightning_indexer_hi_cached`）

问题：block 粗筛阶段要对 key 按 block 求均值，**每次调用都重算**，属于重复计算。

方案：把 key block 均值缓存起来，算子执行时直接读取。新增 2 个参数：

| 参数 | 形状 | dtype | 含义 |
| --- | --- | --- | --- |
| `key_mean_cache` | `(num_blocks, 1, kv_head, head_dim)` | 同 key cache | 每个 block 的均值 |
| `key_mean_cache_status` | `(num_blocks, kv_head)` | int32 | 每个 block 的均值状态 |

```python
topk_indices = torch.ops._C_ascend.npu_lightning_indexer_hi_cached(
    query=q, key=kv_cache[2], weights=weights,
    actual_seq_lengths_query=..., actual_seq_lengths_key=...,
    block_table=block_table,
    layout_query="TND", layout_key="PA_BSND",
    sparse_count=2048, sparse_mode=3,
    hi_block_size=128, hi_block_num=64,
    sink=1, recent=1, block_pooling_mode="mean",
    key_mean_cache=key_mean_cache,
    key_mean_cache_status=key_mean_cache_status)
```

#### 2.2.6 进阶二：配套算子 `scatter_nd_update_mean`

`key_mean_cache` 的生命周期必须与原始 key cache 完全一致：block 形成、更新、淘汰时，
均值也要同步形成、更新、淘汰。因此最佳时机是**新 key 写入 key_cache 时**，
若刚好写满一个 block，就同步计算并存储均值。

改造 `scatter_nd_update_` 算子，新增 3 个输入参数：

| 参数 | 形状 | dtype |
| --- | --- | --- |
| `key_mean_cache` | `(num_blocks, 1, kv_head, head_dim)` | 同 key cache |
| `key_mean_cache_status` | `(num_blocks, kv_head)` | int32 |
| `block_size` | key_cache 中 block 大小 | int32，默认 128 |

```python
torch.ops._C_ascend.npu_scatter_nd_update_mean(
    kv_cache[2].view(-1, k.shape[-1]),
    attn_metadata.slot_mapping.view(-1, 1),
    k.view(-1, k.shape[-1]),
    key_mean, key_mean_status, block_size)
```

#### 2.2.7 单算子实测（原文表 2.1 / 2.2）

Prefill（随机数据，IoU 偏低；原文说明真实 IoU 远高于随机数据）：

| key 长度 | top-m | batch | 原始 indexer | 两阶段 indexer | IoU | 加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| 32K | 64 | 6 | 14.511 ms | 6.789 ms | 0.0867 | **2.14×** |
| 64K | 128 | 6 | 29.579 ms | 17.261 ms | 0.1815 | **1.71×** |
| 128K | 256 | 6 | 59.453 ms | 33.031 ms | 0.1835 | **1.80×** |

Decode：

| key 长度 | top-m | batch | 原始 indexer | 两阶段 indexer | IoU | 加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| 32K | 64 | 64 | 306 µs | 220 µs | 0.1647 | **2.51×** |
| 64K | 128 | 64 | 1.040 ms | 386 µs | 0.1730 | **2.69×** |
| 128K | 256 | 64 | 2.012 ms | 686 µs | 0.1792 | **2.93×** |

原文另给出 head_ratio=32/64、topm=1/4 总 key 长时 decode 阶段不同 batch/不同 key 长度
的时延收益热力图（图 2.8/2.9），正文未列具体数值。

#### 2.2.8 精度：逐层 IoU 与回退策略

服务化推理时逐层对比原始 indexer 与两阶段 indexer 的 top-k indices IoU
（DS V3.2，vllm-ascend 0.13.0；原文说 0.18.0 上 IoU 会更高）：

| Layer | IoU | Layer | IoU | Layer | IoU | Layer | IoU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.463 | 16 | 0.684 | 32 | 0.794 | 48 | 0.777 |
| 1 | 0.416 | 17 | 0.554 | 33 | 0.796 | 49 | 0.749 |
| 2 | 0.684 | 18 | 0.631 | 34 | 0.741 | 50 | 0.794 |
| 3 | 0.745 | 19 | 0.846 | 35 | 0.832 | 51 | 0.765 |
| 4 | 0.582 | 20 | 0.853 | 36 | 0.731 | 52 | 0.803 |
| 5 | 0.660 | 21 | 0.777 | 37 | 0.779 | 53 | 0.637 |
| 6 | 0.753 | 22 | 0.561 | 38 | 0.836 | 54 | 0.732 |
| 7 | 0.779 | 23 | 0.601 | 39 | 0.822 | 55 | 0.810 |
| 8 | 0.592 | 24 | 0.750 | 40 | 0.694 | 56 | 0.772 |
| 9 | 0.650 | 25 | 0.690 | 41 | 0.710 | 57 | 0.727 |
| 10 | 0.751 | 26 | 0.856 | 42 | 0.847 | 58 | 0.767 |
| 11 | 0.785 | 27 | 0.856 | 43 | 0.722 | 59 | 0.852 |
| 12 | 0.713 | 28 | 0.829 | 44 | 0.604 | 60 | 0.585 |
| 13 | 0.631 | 29 | 0.819 | 45 | 0.781 | | |
| 14 | 0.632 | 30 | 0.741 | 46 | 0.789 | | |
| 15 | 0.808 | 31 | 0.746 | 47 | 0.838 | | |

**结论**：大部分层 IoU > 0.8；少数低 IoU 层可**回退使用原始 indexer**，
实践中回退 **0、1 两层**即可保证端到端精度。

> 这一点在工程上很重要：两阶段 indexer 不是“全层替换”，而是“逐层可选”。
> 回退层不享受加速，因此端到端收益会略低于单算子加速比。

#### 2.2.9 端到端效果

| 场景 | 配置 | 结果 |
| --- | --- | --- |
| DS V3.2 直接替换原始 indexer | — | 端到端性能 **+8~10%**，精度影响 **<1%** |
| 精度（LongBench V2，平均长度 64K） | top-m=128 | 几乎无损 |
| 精度（LongBench V2） | top-m=64 | 精度损失 **2%** |
| Decode 性能 | top-m=64，key 16K~64K，bs=64 | TPOT **−8~10%**，吞吐 **+8~10%** |
| Prefill 性能 | top-m=64，key 64K / 100K | TTFT **−5% / −13%** |

#### 2.2.10 场景二：用两阶段 indexer 预测下一层 top-k

原文图 2.10–2.12 给出一个有意思的观察：

| 方案 | 能降低的 token 数 |
| --- | --- |
| 原始 indexer | 381 |
| 层级 indexer（top-m=64） | 154 |
| 层级 indexer（top-m=128） | **382（与原始 indexer 效果相同）** |

含义：**top-m=128 时，两阶段 indexer 的选择效果与原始 indexer 等价**，但计算量更低。
这为“用两阶段 indexer 的输出直接作为下一层预取依据”提供了精度基础。

### 2.3 预测预取与动态淘汰

#### 2.3.1 问题

稀疏注意力长上下文推理中，各层访问的关键 token 动态变化。KV 卸载后容易因
**预测不准或预取不及时**造成频繁缺失；传统“逐层检索 + 静态缓存”难以同时利用：

- **跨层空间相关性**（相邻层选中的历史位置高度重叠）；
- **跨步时间局部性**（相邻 decode step 选中的 KV 高度重叠）。

结果：预测开销、数据搬移、访问等待三者并存。

#### 2.3.2 方案框架

基于 **Token 时空感知**的预测预取与缓存复用：

```
当前层（layer i）执行时：
  1. 用当前层的 Top-K 结果 / 层间相关性，预测下一层（layer i+1）将访问的 KV 集合
  2. 对 predicted Top-K 中不在 HBM 的部分，异步预取到 HBM
  3. 下一层实际执行时，diff load 仍未命中的部分（同步补捞）
```

关键点：**只改变传输时机，不改变最终结果** → 因此是“无损预取”。

#### 2.3.3 实验口径（原文明确给出，复现时必须对齐）

| 项 | 值 |
| --- | --- |
| 模型 | GLM-5.1-w8a8 |
| vllm-ascend 版本 | 0.18.0 |
| 数据集 | Open-SWE-Traces（先筛选 > 给定长度的序列，再按需截断） |
| 缓存容量 | 8K |
| decode step 步数 | 128 |
| batch size | 1 |
| 统计方式 | 无预热，取 10 步之后的结果 |

#### 2.3.4 静态预取效果

| 序列长度 | 指标 | 无预取（仅缓存策略） | 叠加预取 | 变化 |
| --- | --- | --- | --- | --- |
| 64K | 平均命中率 | 90.15% | **97.21%** | +7.06 pt |
| 64K | 历史最低命中率 | 12.1% | **59.69%** | +47.6 pt |
| 64K | 命中率 <95% 的层数 | 46 | **13** | −33（约 −72%） |
| 128K | 平均命中率 | 89.5% | **96.80%** | +7.3 pt |
| 128K | 历史最低命中率 | 4.8% | **50.45%** | +45.7 pt |
| 128K | 命中率 <95% 的层数 | 45 | **15** | −30（约 −67%） |

原文结论：长序列 + 小缓存（8K）的受限条件下，预取把平均命中率提升约 7%，
**同时大幅抬升最差层命中率，显著缓解长尾**；命中率 <95% 的层数下降约 70%，
说明预取不仅提升整体水平，还改善了跨层访问的不均衡性。

#### 2.3.5 叠加动态预测

| 序列长度 | 平均命中率 | 历史最低 | <95% 层数 |
| --- | --- | --- | --- |
| 64K | **97.76%** | **66.59%**（+6.9%） | **5** |
| 128K | **97.03%** | **55.71%**（+5.26%） | **13** |

原文说明：128K 场景是“根据层间统计信息”做动态调整；动态预取能有效缓解
**跨层资源分配不均**问题。

#### 2.3.6 层间预测复用加速（θ=0.6）

定义（原文表 3.1 的表头说明）：

- **缓存淘汰方案**：LRU 缓存淘汰策略；
- **层间复用策略**：利用上一层的 TopK Indices，提前捞取下一层的 KVCache；
- **预测预取方案**：在当前层提前异步捞取下一层预测的 TopK Indices 对应的 KVCache；
- **自适应加速的预测预取**：在预测预取基础上叠加回退/复用，降低预测开销。

实验结果（64K，缓存 8K，128 step，bs=1；逐层命中率 = 先逐层统计平均，再层间分析）：

| 方案 | 层间最低 | 层间最高 | 层间平均 | 历史最低命中率 | 异步捞取数 | 同步捞取数 |
| --- | --- | --- | --- | --- | --- | --- |
| 缓存淘汰（LRU） | 45.62 | 99.46 | 84.80 | 9.77 | — | 311.3 |
| 层间复用 | 61.21 | 99.77 | 87.13 | 27.44 | 394.60 | 263.55 |
| 预测预取 | **92.39** | 99.79 | **96.81** | **66.59** | 231.42 | **65.33** |
| 自适应加速的预测预取（θ=0.6，少 1/3 计算量） | 85.93 | 99.46 | 95.50 | 68.26 | 288.78 | 92.18 |

原文结论：

1. 简单 LRU 的未命中率最高；
2. 预测预取实现最低未命中率；
3. 利用层间相似度的自适应加速可**减少预测层数、降低 1/3 预测开销**，
   在 **73.2% 的异步加载量**下把 miss 率降低 **2.86×**（相对简单层间复用）。

**整体收益**：KV **同步加载量降低 79%**（摘要口径）。

#### 2.3.7 该模块的设计边界（重要）

- 预取依赖“预测”，预测不完美：有假阳性（预取了但没用到，浪费带宽+污染缓存）、
  假阴性（没预取到，仍要同步加载）。
- 预取要“及时完成”才有收益：若预取晚于使用时刻，等价于没预取。
- 因此**不能**把预取建模成“关键路径字节除以 N”的免费打折（本仓库 §7.3 明确批评了
  这种建模方式，见 `tierkv_vs_hisparse.md`）。

### 2.4 分段稀疏注意力（SFA）+ 多流并行

#### 2.4.1 问题：DSA decode 路径严格串行

DSA 长上下文 decode 路径可抽象为三步：

```
lightning_indexer → gather_selection_kv_cache → sparse_flash_attention
```

- `lightning_indexer`：根据当前 query 在长上下文 KV 中选择重要 token，输出稀疏索引；
- `gather_selection_kv_cache`：根据索引结果，把需要参与 SFA 的 KV 从 CPU 内存
  gather 到 device HBM；
- `sparse_flash_attention`：在选中的 KV 上执行稀疏注意力。

默认实现中三者严格串行。原文指出两个优化机会：

1. **相邻 decode step 选出的重要 KV 有很高重复比例——约 80%**；
2. `gather_selection_kv_cache` 内的通信传输可以和其他计算任务在不同流内并行。

#### 2.4.2 分段 SFA：把 hit 与 miss 拆开

每个 step 选出的 KV 分成两类：

```
H_t: hit / reuse KV，已经在 selection cache / device HBM 中
M_t: miss / tail KV，本轮需要 gather 才能进入 selection cache
```

因为约 80% 与上一 step 重复，`H_t` 已经在 HBM，其 SFA 计算**不需要等待 gather**。
串行方案的问题是：即使 `H_t` 已就绪，SFA 也会被 `M_t` 的传输卡住。

分段 SFA 把关键路径拆开：

```
Indexer(t)
  ├─ SFA-reuse(H_t)             # 不等待 gather miss 部分
  └─ Gather(M_t) -> SFA-tail(M_t) + merge
```

这样 `SFA-reuse` 可以和 miss KV 的 gather 传输重叠。

#### 2.4.3 为什么不能直接相加两个 attention 输出

softmax 的归一化分母来自**完整 KV 集合**，不是各子集独立归一化后的结果。
因此每个 SFA 分支必须导出 **partial attention state**：

```
O_i: 当前 KV 子集上的 partial attention output
m_i: row-wise softmax max
l_i: row-wise softmax sum under m_i
```

两个分支的稳定合并公式（原文）：

```
m   = max(m_0, m_1)
p_0 = l_0 * exp(m_0 - m)
p_1 = l_1 * exp(m_1 - m)
w_0 = p_0 / (p_0 + p_1)
w_1 = p_1 / (p_0 + p_1)
O   = w_0 * O_0 + w_1 * O_1
```

该合并等价于在 `H_t ∪ M_t` 上重新计算同一个 softmax attention，
但避免重新扫描已完成的 KV 子集。

#### 2.4.4 算法流程

```
1. Indexer 输出当前 step 的 sparse KV selection。
2. 对已在 HBM 中的复用 KV 启动 SFA-reuse：
     SFA-reuse(q, KV[H_t]) -> O_reuse, m_reuse, l_reuse
3. 同时 gather miss KV：
     Gather(M_t) -> KV[M_t] materialized in selection cache
4. 在 SFA-tail 中计算 miss KV，并把 SFA-reuse 的 partial state 作为 prior 输入：
     SFA-tail(q, KV[M_t], prior=(O_reuse, m_reuse, l_reuse)) -> O_final
```

第 4 步的 SFA-tail 不只是计算剩余 KV 的 attention，还在 **kernel 内部完成 softmax
state merge**，直接输出最终结果。

#### 2.4.5 两种 merge 实现方案

**方案一：独立 `DaAttentionMerge` 算子**

```
SFA-reuse -> O_0, m_0, l_0
SFA-tail  -> O_1, m_1, l_1
torch.ops.custom.npu_da_attention_merge(
    prev_attention_out, prev_softmax_max, prev_softmax_sum,
    cur_attention_out, cur_softmax_max, cur_softmax_sum)
```

- 优点：边界清晰，可单独测试、profile、替换；
- 缺点：多一次 operator launch；两个 SFA 分支都要把 partial output + softmax max/sum
  写回 HBM，再由 merge 算子读回；
- 关键缺点：为了让独立 merge 工作，SFA 仍需暴露 `softmax_max_out` / `softmax_sum_out`，
  所以从“对已有 SFA 的修改程度”看并不优于方案二。

**方案二：在后一个 SFA 内部完成 prior merge（最终采用）**

```python
attention_out, softmax_max, softmax_sum = run_sfa_export_group(sfa_reuse_inputs)

torch.ops.custom.npu_sparse_flash_attention(
    query, key, value, sparse_indices, scale_value,
    sparse_block_size=1,
    actual_seq_lengths_query=act_seq_q,
    actual_seq_lengths_kv=act_seq_kv,
    query_rope=query_rope, key_rope=key_rope,
    prior_attention_out=attention_out,
    prior_softmax_max=softmax_max,
    prior_softmax_sum=softmax_sum,
    layout_query="BSND", layout_kv="BSND", sparse_mode=3)
```

SFA-tail 计算当前 miss KV 的 partial state 后，直接在 kernel 内部与 prior state 合并。

采用方案二的三点理由（原文）：

1. **少一次 kernel launch**（不需要额外启动 `DaAttentionMerge`）；
2. **减少中间结果 HBM 读写**（方案一要把两个分支的 partial output 和 softmax state
   都写出再读入；方案二把 tail 的 partial state 和 merge 留在同一 kernel 内）；
3. **修改成本并不更高**（方案一为导出 partial state 也要改 SFA）。

#### 2.4.6 SFA 扩展接口

```python
npu_sparse_flash_attention(
    Tensor query, Tensor key, Tensor value, Tensor sparse_indices,
    float scale_value, int sparse_block_size,
    *,
    Tensor? block_table=None,
    Tensor? actual_seq_lengths_query=None,
    Tensor? actual_seq_lengths_kv=None,
    Tensor? query_rope=None, Tensor? key_rope=None,
    Tensor? softmax_max_out=None, Tensor? softmax_sum_out=None,
    Tensor? prior_softmax_max=None, Tensor? prior_softmax_sum=None,
    Tensor? prior_attention_out=None,
    str layout_query="BSND", str layout_kv="BSND", int sparse_mode=3
) -> Tensor
```

扩展点两类：

| 扩展参数 | 用途 |
| --- | --- |
| `softmax_max_out`, `softmax_sum_out` | 让当前 SFA 导出 partial softmax state，供后续 merge 使用 |
| `prior_attention_out`, `prior_softmax_max`, `prior_softmax_sum` | 让当前 SFA 接收上一段 SFA 的 partial state，并在 kernel 内完成 merge |

Decode BSND 场景下的典型形状：

| 参数 | 形状 | dtype |
| --- | --- | --- |
| `query` / return | `[B, S_q, N, D]` | fp16 / bf16 |
| `key`, `value` | `[B, S_kv, N_kv, D]` | fp16 / bf16 |
| `sparse_indices` | `[B, S_q, N_kv, topk]` | int32 |
| `query_rope` | `[B, S_q, N, D_rope]` | fp16 / bf16 |
| `key_rope` | `[B, S_kv, N_kv, D_rope]` | fp16 / bf16 |
| `actual_seq_lengths_query` | `[B]` | int32 |
| `actual_seq_lengths_kv` | `[B]` | int32 |
| `softmax_max_out`, `softmax_sum_out` | `[B * S_q, N]` | fp32 |
| `prior_attention_out` | 同 query | fp16 / bf16 |
| `prior_softmax_max`, `prior_softmax_sum` | `[B * S_q, N]` | fp32 |

#### 2.4.7 Batch 拆分与多 stream 并行

分段 SFA 解决“哪些 KV 可以先算”；多 stream 解决“这些可并行的工作如何同时跑”。

- 把完整 batch 切成**两个 micro-batch**，算子下发到不同计算流；
- 让一个 micro-batch 的 KV Gather 传输与另一个 micro-batch 的
  Lightning Indexer / hit SFA 并行；
- 原生 `gather_selection_kv_cache` 同时包含**计算**（确定哪些未命中、新读 KV 放哪）
  与**通信**（实际传输），无法直接与其他计算并行 → 拆成两个算子：
  - `KVSelect`：在 **AICPU** 上执行查找未命中的计算任务；
  - `KVGather`：仅包含传输任务。
  这样 gather 可以最大程度不抢占计算资源。

拆分后两个 micro-batch 在三条流上执行：

```
AIC/AIV 计算流：indexer0 → indexer1 → hit_sfa0 → hit_sfa1 → miss_sfa0+merge0 → miss_sfa1+merge1
AICPU 计算流：  kvselect0 → kvselect1
通信传输流：    kvgather0 → kvgather1
```

算子依赖与可并行对象（原文表）：

| 算子 | 依赖 | 可并行对象 |
| --- | --- | --- |
| indexer0 | step start | — |
| indexer1 | indexer0 后（当前实现串行） | mock_kv_select0 |
| mock_kv_select0 | indexer0_done | indexer1 |
| mock_kv_select1 | indexer1_done | hit_sfa0 |
| kvgather0 | mock_kv_select0_done | indexer1 / mock_kv_select1 / hit_sfa0 |
| kvgather1 | mock_kv_select1_done | hit_sfa0 / hit_sfa1 |
| miss_sfa_i + merge_i | 对应 hit_sfa_i 与 kvgather_i 都完成 | — |

> 注意：`KVSelect` 是 `KVGather` 的**前置依赖**，不能只把它列在并发流里而不进入关键路径
> （本仓库 §7.4 的批评点）。

#### 2.4.8 实测收益

当前阶段用 mock 实现的 `KVSelect` 代替尚未开发完成的真实算子（接口一致，
只模拟调度/延迟；真实 hit/miss plan 在 setup 阶段生成，timed path 复用该 plan）。

DS V3.2 模型，计算传输并行掩盖 vs 串行基线：

| seq_len | BS | baseline (ms) | pipe (ms) | 收益 |
| --- | --- | --- | --- | --- |
| 64K | 32 | 1.083202 | 0.979106 | **10.63%** |
| 64K | 64 | 1.951178 | 1.803496 | **8.19%** |
| 128K | 32 | 1.565792 | 1.478196 | **5.93%** |
| 128K | 64 | 2.924625 | 2.780956 | **5.17%** |

原文初步结论（很重要，说明收益边界）：

1. graph replay + stream limit + 3-stream 在 bs=32/64 能降低端到端延迟，64K/128K
   收益约 **5%~10%**；
2. **当前主要收益来自 overlap，而不是 split 路径本身更快**——split 路径的
   `KVGather + segmented SFA + merge` 总工作量比 fused serial **更大**；
3. **bs=96 时瓶颈转向 split 路径自身的 KVGather 和 SFA 工作量**，overlap 不能继续
   带来明显端到端收益；
4. 对 bs=96/128K，当前不同流之间的资源划分策略不再能带来正收益，后续需研究更合理的
   NPU 核划分方案；
5. 后续优化重点：把 hit/miss 查询基于 AIV 进一步加速，淘汰算法与元数据更新放到
   AICPU 异步执行，并通过淘汰算法优化进一步降低 kvgather 传输量。

### 2.5 多层索引共享与多 token 预测（模型侧信号）

#### 2.5.1 GLM-5.2 的 IndexShare

原文数据：

- IndexShare 将**同一组索引结果复用于连续四层**稀疏注意力；
- 1M 上下文场景下，单 token 计算量降低约 **2.9×**；
- 多 token 预测机制使 speculative decoding 的平均接受长度最高提升约 **20%**。

#### 2.5.2 共享的是“位置”，不是“KV Tensor”

原文明确强调（这是最容易被误读的一点）：

> 需要明确的是，共享的是**历史 token 的位置或稀疏索引结果**，而不是不同层之间的
> KV Tensor。由于各层具有独立的 K/V 投影参数，同一 token 在不同层对应的 KV 数据
> 并不相同。

因此：
- 索引计算可以省（组首层算一次，后续层复用）；
- **KV 搬运不能省**（每层仍要搬自己的 KV 数据）；
- 预取可以在组首层一次性解析地址，组内后续层直接复用地址。

#### 2.5.3 执行方式的变化

| 实现 | 执行方式 |
| --- | --- |
| 各层独立算 Top-K | 只能等某层开始执行后才知道该层要访问哪些 token，KV 只能逐层按需换入 |
| 多层共享索引 | 在 Layer Group 首层拿到索引结果后，即可提前组织后续若干层的数据访问 → **从“逐层被动换入”变为“面向层组的主动预取”** |

#### 2.5.4 多 token 预测对 Hot Buffer 生命周期的影响

传统自回归 decode 每轮只生成 1 个 token，缓存策略通常围绕“当前请求、当前层、
当前 step”设计，热数据保留周期短。

引入 speculative decoding 后，一轮可能先生成多个候选 token，再由主模型统一验证，
接受其中连续一段。于是：

- 历史 KV 的潜在复用窗口从**单个 token** 扩展到**整个候选 + 验证区间**；
- Hot Buffer 不应再采用“单层、单 step 固定短 TTL”，而应综合考虑
  **Request / Layer Group / Candidate Window / Accepted Window**；
- 多个候选共同访问的历史 KV 应尽量保留到验证阶段结束；
- 部分候选被拒绝时，只释放与被拒绝候选相关的临时状态，**不应淘汰仍属于正式历史
  上下文、且可能在后续步骤继续访问的热 KV**。

#### 2.5.5 模型信号汇总（原文表）

| 模型侧信号 | 对 KVCache 卸载的意义 |
| --- | --- |
| 长上下文窗口 | KVCache 容量随上下文线性增长，卸载从可选优化变为基础能力 |
| 稀疏注意力 | Decode 无需持续访问完整历史，为构建 Hot Set 和按需召回提供基础 |
| 多层共享索引 | Top-K 选择由单层局部信号提升为 Layer Group 级共享信号，Runtime 可提前完成地址解析与预取 |
| 多 token 预测与 speculative decoding | Decode 由单 token 串行变为 token burst，Hot Buffer 可在候选生成、验证、接受阶段持续复用 |

### 2.6 端到端实测与实验配置

**实验配置**（原文）：

| 项 | 值 |
| --- | --- |
| 推理框架 | vLLM 0.18 |
| 硬件 | 单机 A3 |
| 模型 | DeepSeek-3.2 w8a8c16 |
| 并行 | TP=16 / DP=1 / EP=16 |
| 序列长度 | 64K |
| 基线配置 | BS = 1（BS = 2 会 OOM） |
| 卸载后配置 | BS = 1 / 2 / 4 / 8 |

**结果**：

- 卸载 DRAM + 同步加载 KV 的情况下，卡 80 ms 时延；
- **TPS 从 23.3 提升到 45.8**（约 1.97×）。

**三大技术收益汇总**：

| 技术 | 收益 |
| --- | --- |
| 两阶段层级 Indexer | Prefill 加速比 >1.8×，Decode 加速比 >2.5×；端到端 +8~10%，精度 <1% |
| 无损预取 + 动态淘汰 | KV 同步加载量降低 **79%** |
| 分段 SFA + 多流并行 | 64K/128K 端到端时延下降 **5%~10%** |

### 2.7 TierKV 的局限与未完成项

1. **ASU/SSD 未实测**：原文明确“后续将拓展 ASU 卸载能力”，当前只到 DRAM。
2. **两阶段 indexer 有精度代价**：top-m=64 时 LongBench V2 掉 2%，需回退低 IoU 层。
3. **多流收益有边界**：bs=96 时 split 路径自身工作量成为瓶颈，overlap 不再有正收益。
4. **KVSelect 仍是 mock**：真实 AICPU 算子尚未完成，当前收益是“模拟调度/延迟”下的结果。
5. **每请求私有 host 池**：不涉及多卡共享，容量仍是“单实例 DRAM 总量”约束。
6. **未覆盖 PD 分离**：只讨论单实例“本地 prefill → staging → decode”。

---

## 3. L2 Decode Cache / KV Offload 方案详解（S2）

> 本节内容来自 L2 Decode Cache 方案的架构图、客户数据表和说明。材料未给出算法推导
> 与完整参数，标注“未给出”的地方即为信息缺口。

### 3.1 定位

**方案定位**：

> 通过 KV Offload 支持 GLM5.2/5.3 1M 长序列，吞吐约为 DCP 方案的 1.2 倍。

关键词解读：

- **L2 Decode Cache**：把 DRAM 作为 HBM 的下一级（L2）缓存，服务 **decode** 阶段；
- **KV Offload**：把 KV Cache 从 HBM 卸载到 DRAM；
- **1M 长序列**：目标场景是 100 万 token 上下文；
- **1.2 倍 DCP**：对比基线是 DCP（Decode Context Parallel）方案，吞吐是其 1.2 倍。

### 3.2 客户场景与痛点

**客户场景**（材料原文）：

> 客户 A/B 基于 A3 部署 AI Coding 场景 GLM 5.2/5.3 模型，业务场景要求推理服务
> 支持 1M 超长序列。

**痛点**（材料原文）：

> 1M MLA KV ≈ 90GB
> `[= tokens × layers × (kv_lora_rank + qk_rope_head_dim) × bf16_bytes]`，
> A3 单卡 64GB HBM，典配场景单卡剩余约 25GB HBM 预留给 KVCache，
> **无法支持 1M 长序列推理**。

即：单卡 HBM 物理上装不下 1M 序列的 KV，必须想别的办法。

### 3.3 Before：DCP（Decode Context Parallel）方案

材料用一张图描述 DCP：

```
R0 KV: 块0, 块4        R1 KV: 块1, 块5
R2 KV: 块2, 块6        R3 KV: 块3, 块7
        （KV 沿序列维度切分到 4 张卡，每卡 1/4）

TP 分片的 Q
    │
    ├─ AllGather 汇集 Q heads ──► 汇集后的 Q
    │
    ├─ 局部 SFA：Q × 本地 TopK KV ──► 局部 O + LSE
    │
    ├─ All-to-All 交换 O + LSE
    │
    └─ Softmax 加权合并 ──► 完整 O
```

**材料对 DCP 的评价**（原文）：

> Decode Context Parallel 方案：KV 沿序列维度切分到多卡，单卡 KVCache 为 1/DCP，
> 但仍然占用 HBM，且 DSA 稀疏命中（TopK 2048），HBM 利用率不高，DCP 通信开销较高。

拆解为三个问题：

1. **容量**：虽然单卡只存 1/DCP，但 KV 仍然全部在 HBM 里，总量没变，只是摊到多卡；
2. **利用率**：DSA 每步只命中 TopK 2048，HBM 里绝大部分 KV 在当前 step 不被访问，
   但依然占着 HBM；
3. **通信**：每步需要 `AllGather`（汇集 Q heads）+ `All-to-All`（交换 O 和 LSE），
   通信开销高。

### 3.4 KV Offload 方案：三大设计

材料用三个编号标注核心设计：

#### ① 多卡共享的 DRAM 池，支持 KV Offload

> GVA 统一编址，多卡使用相同内存池，**TP 域内只需一份 KV 缓存**。

- **GVA**（Global Virtual Address）统一编址：所有卡看到同一套虚拟地址；
- 多卡访问**同一块** DRAM 内存池；
- TP 域内只保留**一份**完整 KV，而不是每卡一份副本。

容量效果（材料原文，TP4 场景）：

| 方案 | 容量 |
| --- | --- |
| 无 MF 共享 | `16 × 90GB ≈ 1.5T` |
| 有 MF 共享 | `4 × 90GB ≈ 360GB` |

> **解读**（本文）：TP4 场景下，若无共享则需要 16 份 90GB 的 KV 副本（约 1.5TB），
> 有 MF（Memory Fabric）共享后只需 4 份（约 360GB），**容量需求降到 1/4**。
> 这里的“16”与“4”的具体构成材料未展开（可能与节点数 × TP rank 数有关），
> 引用时应保留原文口径。

#### ② 卡侧直访 DRAM，相比 `aclmemcpy`，消除 Host 开销

> 通过 **H2D & SFA 融合算子**，达成 **TPOT < 30ms**。

- 传统路径：`aclmemcpy` 需要 Host 侧参与/中转，有额外开销；
- 新路径：卡侧（NPU）直接访问 DRAM；
- 更进一步：把 **H2D 拷贝与 SFA（Sparse Flash Attention）融合成一个算子**，
  减少 kernel 边界与中间等待，目标 TPOT < 30 ms。

#### ③ P 侧 HBM KV 直传 D 侧 DRAM KV（D2RH）

> 减少 Decode HBM 占用；相比 `D2D + D2H`，**减少一次中转传输**。

- **P 侧**（Prefill 节点）的 KV 本来在 HBM；
- **D 侧**（Decode 节点）的 KV 目标位置在 DRAM；
- **RD2H**（Remote Device to Host）直接 P-HBM → D-DRAM，不经过 D 侧 HBM 中转；
- 对比传统 `D2D`（P-HBM → D-HBM）+ `D2H`（D-HBM → D-DRAM）两跳，
  **少一次中转**，同时减少 Decode 侧 HBM 占用。

### 3.5 部署拓扑与数据流（架构图解读）

材料下半部分的架构图（按位置还原）：

```
┌─────────────── Prefill ───────────────┐
│  NPU                                  │
│   └─ MLA KV (HBM)                     │
│         │ Layerwise                   │
│         │ MF: RD2H 直达               │   ← ③ P侧HBM KV 直传 D侧DRAM
└─────────┼─────────────────────────────┘
          ▼
┌─────────────── Decode (TP=4) ─────────────────────────────┐
│  rank0        rank1        rank2        rank3             │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐          │
│  │NPU     │  │NPU     │  │NPU     │  │NPU     │          │
│  │LI Cache│  │LI Cache│  │LI Cache│  │LI Cache│  ← HBM   │
│  │Hot TopK│  │Hot TopK│  │Hot TopK│  │Hot TopK│  ← HBM   │
│  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘          │
│      │ D2D       │           │           │                │
│      └───────────┴─────┬─────┴───────────┘                │
│                        │ ② H2D&SFA 融合算子                │
│                        ▼                                   │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ MF: DRAM 统一编址内存池                                │ │
│  │ TP 组内一份完整 KV，TP 域内所有卡同地址共享访问         │ │
│  │                                    TP组0              │ │
│  └──────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
          │ D2H（下一级）
          ▼
┌───────────────────────────────────────────────────────────┐
│ SSU – 下一级存储（具体型号与介质未给出）                    │
└───────────────────────────────────────────────────────────┘
```

**关键元素解读**：

| 元素 | 含义 |
| --- | --- |
| `LI Cache` | 每卡 HBM 中的 **L**ightning **I**ndexer 缓存（索引数据），与 KV 分离 |
| `Hot TopK` | 每卡 HBM 中的热点 Top-K KV（即 hot pool） |
| `D2D` | Decode 侧设备内/设备间搬运 |
| `D2H` | HBM → DRAM（卸载/写回） |
| `H2D & SFA 融合算子` | 从 DRAM 召回 KV 并直接做稀疏注意力，融合成一个算子 |
| `MF: RD2H 直达` | Prefill HBM → Decode DRAM 的直传路径 |
| `TP 组0` | 一个 TP 组共享一个 DRAM 池 |
| `SSU` | 下一级存储；具体型号、介质和 IOPS 未给出 |

**注意**：材料把 **LI Cache（索引）** 和 **Hot TopK（KV）** 都放在 HBM，这与
TierKV 的“indexer 索引常驻 HBM + hot pool 只放 Top-K KV”在结构上是一致的。

### 3.6 客户使用情况（匿名化数据表）

| 客户 | Decode 并行策略 | 上线规模（节点） | DCP 总吞吐 | Offload 总吞吐 | 吞吐收益 |
| --- | --- | --- | --- | --- | --- |
| 客户 A | tp4dp8 | A3 × 24 | 130,126 | 155,272 | **1.2× DCP @1M** |
| 客户 B | tp2dp16 | A3 × 40 | 111,603 | 145,084 | **1.3× DCP @1M** |
| 客户 C | tp2dp16 | A3 POC 中 | — | — | — |

> 吞吐单位材料未标注（推测为 tokens/s，但未确认，引用时不要臆断）。

### 3.7 材料备注

> 1M 真实的场景，TP 之间可以真实是多少，能共享 TP 要体现出来，KV 多大体现出来。

这说明材料编写方也认为：**TP 域内可共享的范围**与 **KV 实际大小**是这张图
最需要说清楚的两件事。当前材料只给了“16×90GB vs 4×90GB”这一组对比。

### 3.8 材料未给出/存疑的点（信息缺口清单）

为便于后续分析，明确列出材料**没有**回答的问题：

1. **MF/GVA 的硬件形态**：DRAM 池是节点内还是跨节点？走 PCIe、CXL 还是私有互联？
   带宽/延迟多少？材料只写“统一编址”和“卡侧直访”。
2. **共享的并发语义**：TP 域内一份 KV，多个 DP 组之间是否也共享？
   `tp4dp8` 中的 dp8 各自是否独立池？
3. **一致性/同步**：多卡同地址访问时，读写一致性如何保证？是否需要 fence？
4. **H2D&SFA 融合算子的内部结构**：是否等价于 TierKV 的“分段 SFA”？
   是否也做 hit/miss 拆分？材料未说明。
5. **TopK=2048 的来源**：是模型 indexer 输出还是框架固定？
   （TierKV 侧明确是模型 indexer，且 DSV4.1 是 512。）
6. **预取/淘汰策略**：Hot TopK 如何更新？LRU？是否有预取？材料未提。
7. **精度验证**：Offload 是否完全无损？材料未提精度数据。
8. **SSU 的角色**：是 DRAM 的下一级（容量扩展），还是仅用于持久化？
   “IOPS, 1yi, core” 的具体含义未解释。
9. **吞吐单位与测试 workload**：客户表只有总吞吐，没有 TTFT/TPOT、输入输出长度、
   并发数。
10. **1.2×/1.3× 的对比基线细节**：DCP 的 D 值（切分卡数）、是否同节点数、
    是否同 TP/DP 配置。

---

## 4. 两个方案的对比

### 4.1 定位差异：一个改“搬运量与时延隐藏”，一个改“搬运路径与共享范围”

| 视角 | TierKV（S1） | L2 Decode Cache（S2） |
| --- | --- | --- |
| 抽象层次 | 模型算子 + 单实例 runtime | 集群内存架构 + 并行策略 |
| 主要矛盾 | 单实例 HBM 装不下 → BS 上不去 | TP 域 HBM 总量装不下 1M；DCP 通信贵 |
| 解决方式 | 少搬、早搬、重叠搬 | 搬到共享池、走近路、少一跳 |
| 是否动模型 | 是（indexer 算子、SFA 扩展） | 否（只动内存与传输路径） |
| 是否动并行策略 | 否（保持 TP/EP） | **是（用 Offload 替代 DCP）** |
| 优化对象 | TPOT / 吞吐（单实例） | 吞吐（多节点集群）vs DCP 基线 |

### 4.2 逐维度详细对比

#### 4.2.1 容量与共享

| 项 | TierKV | L2 Decode Cache |
| --- | --- | --- |
| 容量公式 | `logical = physical × host_to_device_ratio`（每请求/每实例私有） | TP 域共享池，`TP 组内一份完整 KV` |
| 共享粒度 | 请求私有 host 池；不同请求不共享 | **TP 域内多卡共享同一份 KV** |
| 容量收益来源 | 把全量 KV 从 HBM 挪到 DRAM，HBM 只留 hot pool | ① 挪到 DRAM；② **多卡只存一份**（4×90GB vs 16×90GB） |
| 容量数量级 | 单机 DRAM（数百 GB 量级） | 共享池（材料示例 360GB / 1.5T） |
| 是否支持跨请求去重 | 未建模（TierKV 原文未提） | 未提（材料只讲 TP 域共享） |

**关键差异**：TierKV 的容量收益是“**同一份 KV 换个地方放**”；
S2 的容量收益是“**多卡不再各存一份**”，是**乘法级**的节省（4×）。
这是 S2 相比 TierKV 在容量维度上更强的地方。

#### 4.2.2 数据搬运路径

| 项 | TierKV | L2 Decode Cache |
| --- | --- | --- |
| 召回路径 | DRAM → HBM（标准 H2D） | DRAM → HBM，但用**卡侧直访 + H2D&SFA 融合算子** |
| 写回路径 | HBM → DRAM（eager backup，侧流） | D2H（图中标注） |
| Prefill→Decode | 未涉及（单实例） | **RD2H 直达**：P-HBM → D-DRAM，绕过 D-HBM |
| 中转次数 | 不涉及 | 比 `D2D+D2H` **少一次中转** |
| Host 参与 | 标准 host 内存路径 | **消除 Host 开销**（相比 `aclmemcpy`） |

**关键差异**：TierKV 假设“标准 H2D”，其带宽模型是 `t0 + bytes/(bw×0.85)`；
S2 用“卡侧直访 + 融合算子”，**带宽/延迟特性完全不同**（可能更快，也可能因
直访 DRAM 的随机访问而尾延迟更差）。二者不能共用同一套时延参数。

#### 4.2.3 关键路径优化手段

| 手段 | TierKV | L2 Decode Cache |
| --- | --- | --- |
| Indexer 加速 | ✅ 两阶段层级 indexer（1.8×/2.5×） | ❌ 未提（LI Cache 只是常驻 HBM） |
| 预取 | ✅ 预测下一层 + 动态预测（同步加载 −79%） | ❌ 未提 |
| hit/miss 分段 | ✅ 分段 SFA（H_t / M_t） | ❓ 可能隐含在“H2D&SFA 融合算子”中，未说明 |
| 多流并行 | ✅ 3 流 + 2 micro-batch（5~10%） | ❌ 未提 |
| 传输与计算融合 | ✅ SFA-tail 内 merge | ✅ H2D&SFA 融合算子 |
| 减少中转 | ❌ 不涉及 | ✅ RD2H 直达 |
| 消除 Host 开销 | ❌ 不涉及 | ✅ 卡侧直访 |

**关键差异**：TierKV 的优化集中在“**如何少搬、如何藏住搬运**”；
S2 的优化集中在“**如何让搬运走更短的路**”。二者正交，可叠加。

#### 4.2.4 稀疏性来源与索引处理

| 项 | TierKV | L2 Decode Cache |
| --- | --- | --- |
| 稀疏来源 | 模型自带 DSA indexer（DSV3.2/V4） | 模型自带 DSA（TopK 2048） |
| Top-K 大小 | DSV4.1 为 512；DSV3.2 为 2048 | **2048** |
| 索引存放 | 部分模型常驻 HBM（indexer K 全长） | **LI Cache 常驻 HBM** |
| 索引计算优化 | ✅ 两阶段 + 均值缓存 + 层间共享 | ❌ 未提 |
| 索引与 KV 是否分离 | 是（profile 中 main/index 分账） | 是（LI Cache vs Hot TopK） |

#### 4.2.5 精度影响

| 项 | TierKV | L2 Decode Cache |
| --- | --- | --- |
| 卸载本身 | 无损（只改存储位置） | 无损（只改存储位置） |
| 预取 | 无损（只改传输时机） | 未提 |
| Indexer 加速 | **有损**：top-m=64 掉 2%；需回退低 IoU 层 | 无（不改 indexer） |
| 端到端精度 | <1%（DSV3.2） | 未给 |

**关键差异**：TierKV 的**唯一精度风险来自两阶段 indexer**；S2 不碰算法，
理论上无精度风险（但材料未给验证数据）。

#### 4.2.6 实测规模与收益

| 项 | TierKV | L2 Decode Cache |
| --- | --- | --- |
| 硬件 | 单机 A3 | A3 集群（24/40 节点） |
| 模型 | DeepSeek-3.2 w8a8c16 | GLM5.2/5.3 |
| 序列长度 | 64K（实测）；1M 为目标 | **1M（实测）** |
| 并行 | TP16/DP1/EP16 | tp4dp8 / tp2dp16 |
| 基线 | 全量 KV 驻留 HBM（BS=1，BS=2 OOM） | **DCP** |
| 收益 | TPS 23.3 → 45.8（1.97×） | 1.2× / 1.3× vs DCP |
| 时延 | 80 ms 卡点 | TPOT < 30 ms（目标） |

**注意**：两者收益**不可直接比较**——基线不同（HBM 全驻留 vs DCP）、
模型不同、规模不同、指标不同（TPS vs 总吞吐）。

#### 4.2.7 工程依赖

| 依赖 | TierKV | L2 Decode Cache |
| --- | --- | --- |
| 算子改造 | `lightning_indexer_hi` / `_cached` / `scatter_nd_update_mean` / SFA 扩展 | H2D&SFA 融合算子 |
| 内存架构 | 无特殊要求（普通 pinned host memory） | **MF + GVA 统一编址 + 多卡共享 DRAM 池** |
| 互联 | 无特殊要求 | 需要支持卡侧直访 DRAM 的互联 |
| 存储 | DRAM（SSD/ASU 待扩展） | DRAM + SSU（具体型号未给出） |
| 框架 | vLLM-Ascend | 未标注（推测 vLLM-Ascend） |

### 4.3 互补关系：可以叠加的四条路径

#### (1) S2 的共享池 + S1 的 top-k 召回 = 更小的池 + 更少的搬

- S2 把“每卡一份”变成“TP 域一份”，池子总量降到 1/4；
- S1 的 top-k 召回让每步只搬 Top-K；
- 组合后：**共享池容量 × 稀疏召回率 = 实际搬运量**，两者相乘收益。

#### (2) S1 的 indexer 加速 + S2 的 H2D&SFA 融合 = 更宽的时延窗口

- S1 把 indexer 从 >30% 的 attention 占比降下来（1.8×/2.5×）；
- indexer 变快 → 留给 H2D 的时间窗口变长；
- S2 的融合算子又减少了 H2D 的 kernel 边界开销；
- 组合后：**关键路径 = 更短的 indexer + 更短的传输**。

#### (3) S1 的分段 SFA + S2 的融合算子 = 更彻底的 overlap

- S1 把 hit/miss 拆开，hit 部分不等 gather；
- S2 把 H2D 与 SFA 融合成一个算子；
- 二者结合需要明确：融合算子内部是否也做 hit/miss 拆分？若不拆分，
  S1 的分段收益会被融合算子的“整体等待”抵消。
- **这是最需要联合设计的一点**。

#### (4) S1 的层间共享索引 + S2 的 TP 域共享 = 两级共享

- S1：同一 TP rank 内，连续 4 层共享索引（省计算）；
- S2：TP 域内多卡共享 KV（省容量）；
- 组合后形成“**层间共享索引 + 卡间共享 KV**”的两级结构，但要注意：
  - S1 共享的是**位置**，S2 共享的是**数据**；
  - 两者共享维度不同，不冲突，但索引寻址需要同时考虑 layer group 和 TP 域地址映射。

### 4.4 张力与冲突点

| 冲突 | 说明 | 影响 |
| --- | --- | --- |
| 私有池 vs 共享池 | TierKV 假设每请求私有 host 池；S2 是 TP 域共享一份 KV | 若叠加，TierKV 的 `host_to_device_ratio` 容量公式需改为“TP 域共享池 ÷ 请求数” |
| 标准 H2D vs 卡侧直访 | 两者的带宽/延迟特性不同 | TierKV 的 `t0 + bytes/(bw×0.85)` 模型不能直接用于 S2 路径 |
| 分段 SFA vs 融合算子 | 分段要求 hit/miss 分离；融合要求 H2D 与 SFA 在同一 kernel | 需要明确融合算子内部是否保留分段语义，否则 S1 收益被抵消 |
| 索引计算优化 vs 索引常驻 | TierKV 优化 indexer 计算；S2 把 LI Cache 常驻 HBM | 不冲突，但 HBM 预算要在“LI Cache”与“Hot TopK”之间分配 |
| DCP 基线 vs HBM 基线 | 两者收益不可直接比较 | 不能把 1.97× 与 1.2× 放在同一张表里比大小 |
| 精度风险 | TierKV 有两阶段 indexer 的 IoU 损失；S2 无 | 若叠加，整体精度风险由 TierKV 承担 |

### 4.5 组合方案的建议架构（推导，非原文）

若要把两者叠加，建议的分层如下：

```
┌──────────────────────────────────────────────────────────────┐
│ 模型侧（S1）                                                  │
│  · 两阶段 Indexer（含均值缓存、层间共享、低 IoU 层回退）        │
│  · 预测预取（跨层 + 跨步，带 recall/precision 与完成事件）      │
│  · 分段 SFA（hit/miss 拆分 + partial softmax merge）           │
└───────────────────────────┬──────────────────────────────────┘
                            │ Top-K 位置 + hit/miss plan
┌───────────────────────────▼──────────────────────────────────┐
│ 系统侧（S2）                                                  │
│  · TP 域共享 DRAM 池（GVA 统一编址，一份 KV）                  │
│  · 卡侧直访 DRAM（消除 Host 开销）                             │
│  · H2D&SFA 融合算子（需保留 hit/miss 分段语义）                │
│  · RD2H 直达（P-HBM → D-DRAM，少一次中转）                     │
└───────────────────────────┬──────────────────────────────────┘
                            │ 容量不够时
┌───────────────────────────▼──────────────────────────────────┐
│ 存储侧                                                       │
│  · SSU（具体型号未给出）/ SSD / ASU                           │
└──────────────────────────────────────────────────────────────┘
```

关键设计问题（需要两边团队对齐）：

1. **共享池的容量口径**：TP 域共享一份 KV 后，TierKV 的
   `logical = physical × host_to_device_ratio` 应改为
   `TP 域总容量 = 共享 DRAM 池大小`，而 `physical` 仍是各卡 HBM hot pool 之和。
2. **融合算子的分段语义**：`H2D&SFA 融合算子` 是否接受 hit/miss 两个集合？
   若不接受，S1 的分段 SFA 无法生效。
3. **预取的地址解析**：S1 的预取需要“下一层访问集”的物理地址；
   S2 的 GVA 统一编址恰好让地址解析更简单（一份地址，多卡可用）。
4. **LI Cache 与 Hot TopK 的 HBM 预算分配**：S2 图中两者都占 HBM；
   若 indexer 加速后 LI Cache 可以缩小，能释放更多 HBM 给 Hot TopK。

---

## 5. 对仿真建模的启示（衔接本仓库 hisim）

> 本节供后续建模参考，细节见 `src/hisim/docs/tierkv_vs_hisparse.md`。

### 5.1 TierKV 侧需要补的模型

| 模块 | 当前 hisim 状态 | 需要补 |
| --- | --- | --- |
| 两阶段 indexer | 无时间模型 | `indexer_mode` / `hi_block_size` / `hi_block_num` / `Q` 区分 prefill/decode |
| 预测预取 | 仅 `shared_index_freq` 的 `1/N` 打折 | recall/precision、预取完成事件、FP 污染、`prefetch_lead_steps` |
| 分段 SFA | H2D 全量串行加到时钟 | hit/miss 拆分、partial merge、事件图 |
| 多流 | 无 | 3 流 + micro-batch 的 list scheduling |
| 层间共享 | 仅 `shared_index_freq` | 显式 `index_share_groups`（不能用 `source_consumers` 推导） |
| 动态淘汰 | 纯 LRU | `layer_aware` / TTL / candidate window |

### 5.2 S2 侧需要补的模型

| 模块 | 需要补 |
| --- | --- |
| TP 域共享池 | 容量口径从“每请求私有”改为“TP 域共享一份” |
| 卡侧直访 DRAM | 独立的带宽/延迟参数（不能复用标准 H2D 模型） |
| H2D&SFA 融合算子 | 融合后的 `t0` 与有效带宽；是否保留 hit/miss 分段 |
| RD2H 直传 | P→D 的跨节点带宽与延迟；与传统 `D2D+D2H` 的对比 |
| SSU 层 | 第三级存储的 IOPS/带宽/延迟 |

### 5.3 两个方案都不能由仿真直接回答的问题

- 真实最大并发与端到端 TPS（需要真实调度与 kernel）；
- 真实 Top-K 命中率（需要 indexer trace）；
- 卡侧直访 DRAM 的真实尾延迟（需要硬件实测）；
- MF/GVA 的一致性开销（需要架构文档或实测）。

---

## 6. 术语表

| 术语 | 全称/含义 |
| --- | --- |
| DSA | DeepSeek Sparse Attention，模型自带的稀疏注意力 |
| Indexer | 稀疏选择算子，输出 Top-K token 位置；DSV3.2 为 `lightning_indexer` |
| SFA | Sparse Flash Attention，稀疏注意力算子 |
| Top-K / top-k | 每步选中的历史 token 数（DSV3.2 2048，DSV4.1 512） |
| Hot pool / hot buffer | HBM 中只保留的热点 KV 区 |
| H2D / D2H | Host to Device / Device to Host |
| D2D | Device to Device |
| RD2H | Remote Device to Host，P 侧 HBM 直传 D 侧 DRAM |
| DCP | Decode Context Parallel，KV 沿序列维度切到多卡 |
| TP / DP / EP | Tensor / Data / Expert Parallel |
| MF | Memory Fabric，材料中的多卡共享内存池 |
| GVA | Global Virtual Address，统一编址 |
| LI Cache | Lightning Indexer Cache，HBM 中的索引缓存 |
| SSU | 下一级存储单元（具体型号和介质未给出） |
| ASU | 原文提到的后续卸载介质（未展开） |
| MLA | Multi-head Latent Attention，KV 只存 `kv_lora_rank + qk_rope_head_dim` |
| TPOT | Time Per Output Token |
| TTFT | Time To First Token |
| BS | Batch Size |
| IoU | Intersection over Union，两阶段 indexer 与原始 indexer 的 top-k 重合度 |
| LSE | Log-Sum-Exp，softmax 的稳定计算形式 |
| partial state | 分段 attention 的 `(O, m, l)` 中间状态 |

---

## 7. 信息可信度与引用建议

| 内容 | 来源 | 建议引用方式 |
| --- | --- | --- |
| TierKV 架构、算子接口、参数 | S1 原文 | 可直接引用，注明“原文” |
| TierKV 单算子加速比、端到端收益 | S1 原文实测 | 引用时注明“原文团队实测，未在本环境复现” |
| TierKV 预取命中率、层间复用表 | S1 原文实测 | 同上；注意实验口径（GLM-5.1-w8a8 / 8K 缓存 / 128 step / bs=1） |
| TierKV 分段 SFA 收益 | S1 原文实测 | 注意 `KVSelect` 是 mock，收益含模拟成分 |
| S2 架构三要素 | S2 材料 | 可直接引用；细节需向材料提供方确认 |
| S2 客户吞吐 1.2×/1.3× | S2 材料数据表 | 引用时注明“材料转述，单位与 workload 未标注” |
| S2 的 16×90GB / 4×90GB | S2 材料 | 原文口径；“16”与“4”的构成未展开 |
| GLM 约 78 层 | 本文推导 | 标注“推导”，不要当作事实 |
| 组合方案架构 | 本文推导 | 标注“推导，非原文” |

**引用说明**：本文已删除来源平台、材料文件名、人员姓名和客户名称。对外引用时只保留
方案名、匿名化配置与“未在本环境复现”的边界，不恢复上述标识。

---

## 8. 一页总结

| 问题 | TierKV | L2 Decode Cache |
| --- | --- | --- |
| 装不下怎么办 | 搬到 DRAM，HBM 只留 Top-K | 搬到 **TP 域共享** DRAM 池 |
| 搬得慢怎么办 | indexer 加速 + 预取 + 分段 SFA + 多流 | 卡侧直访 + H2D&SFA 融合 + RD2H 直传 |
| 搬多少 | Top-K（DSV4.1 512） | Top-K（2048） |
| 谁受益 | 单实例 BS/吞吐 | 多节点集群吞吐 vs DCP |
| 代价 | indexer 精度（top-m=64 掉 2%） | 依赖 MF/GVA 硬件 |
| 能否叠加 | **能**：S1 管“少搬+藏住”，S2 管“走近路+共享” | **能**：见 §4.5 |
| 最大不确定性 | 真实 Top-K trace 与 kernel 系数 | MF/GVA 的真实带宽/延迟与一致性 |

**最终判断**：

- 若目标是**单实例内**提升长上下文并发与吞吐，TierKV 的三件套（indexer 加速、
  预取、分段 SFA）是直接可用的；
- 若目标是**多节点 1M 长序列**且已有 DCP 部署，S2 的共享池 + 直访 + RD2H 是更
  对症的方案；
- 两者**不互斥**。最优工程路径很可能是：**S2 提供共享大池与短路径，
  S1 提供稀疏召回与重叠**，但需要先解决 §4.4 列出的四个张力点。

---

## 9. 评审意见（2026-11-12）

> 评审对象：本文去标识化前的首个版本。评审方式为表内数字重算、正文交叉引用和当前
> hisim 实现对照；未取得 S1 与 S2 原始材料，因此本节只判断内部一致性，不断言原始
> 数据哪一项才是正确值。若本节与上文冲突，以本节的复核结论为准。

### 9.1 总体结论

本文的方案分层、信息缺口和“不可直接比较收益”的边界写得较完整，尤其是明确标出
S1 的 `KVSelect` 仍为 mock、S2 的吞吐单位和 workload 未给出，以及组合方案属于推导。
但当前版本仍有三组表内可确定的数据问题，以及若干会改变工程工作量和仿真口径的
概念混用。修正这些问题前，相关表格不宜直接作为对外引用或实施规格。

### 9.2 高优先级：表格数字或结论不自洽

#### (1) 32K decode 的耗时与加速比不能同时成立

§2.2.7（`:294-300`）写原始 indexer 为 306 µs、两阶段为 220 µs、加速比为 2.51×，
但直接相除为：

```text
306 / 220 = 1.3909×
```

同一表其余五行的耗时比与标注加速比一致，因此这一行需要回查原表。若 220 µs 与
2.51× 正确，原始耗时应约为 552.2 µs；若 306 µs 与 2.51× 正确，两阶段耗时应约为
121.9 µs；若两个耗时都正确，加速比应改为约 1.39×。在拿到原始表前不应任选一种
改法冒充已核实事实。

同一问题还影响 §2.6（`:761`）的汇总：“Decode 加速比 >2.5×”不能被当前首行
数据支持；“Prefill 加速比 >1.8×”也与表中 64K 的 1.71×、128K 的约 1.7999×
不符。若原表没有其他口径，汇总应改成与各行一致的范围，而不能把部分行的最佳值
写成整组下界。

#### (2) “大部分层 IoU > 0.8”与本文表格不符

按 §2.2.8（`:305-333`）列出的 61 层数值重算：

- 仅 **15/61 = 24.6%** 的层满足 `IoU > 0.8`；
- **42/61 = 68.9%** 的层满足 `IoU > 0.7`；
- 61 层算术平均 IoU 约为 **0.7299**。

因此“多数层 >0.8”应改为“多数层 >0.7，但仅约四分之一层 >0.8”，或者直接引用
原文其他统计表并说明口径。“回退 0、1 两层即可保证端到端精度”只能作为原文的
经验结论保留，不能从当前这张 IoU 表自行推出；还需要对应的端到端消融结果支撑。

#### (3) SFA 的“收益”不是同口径的“时延下降”

§2.4.8（`:657-675`）的四个收益百分比实际符合：

```text
benefit = baseline / pipe - 1
```

而不是：

```text
latency_reduction = (baseline - pipe) / baseline
```

两种口径分别为：

| seq_len / BS | 文中收益（`baseline/pipe-1`） | 实际时延下降 |
| --- | --- | --- |
| 64K / 32 | 10.63% | 9.61% |
| 64K / 64 | 8.19% | 7.57% |
| 128K / 32 | 5.93% | 5.59% |
| 128K / 64 | 5.17% | 4.91% |

建议二选一：若保留当前百分比，列名改为“相对 pipe 的加速收益”，正文不要称为
“时延下降”；若要表达时延下降，则改用右列数字，并把 §2.4.8、§2.6 的范围改为约
**4.9%～9.6%**。两种口径不能混用。

### 9.3 中优先级：技术语义需要修正

| 位置 | 问题 | 建议 |
| --- | --- | --- |
| §0、§3.4、§4.1（`:42,869-874,989`） | S2 一处写“不改模型算子/只改系统路径”，另一处又要求新增 `H2D&SFA` 融合算子，工程边界矛盾 | 改为“不改变模型数学和稀疏选择，但需要新增或改造融合 kernel/runtime”；数值一致性仍需验证，不能仅凭不改选择就写成绝对无精度风险 |
| §2.5.2（`:702-705`） | IndexShare 共享位置被进一步写成“后续层直接复用地址”，容易误解为不同层共用 KV 物理地址 | 改为复用 Top-K 位置、block-table 遍历和可共享的地址解析中间结果；各层仍须结合自己的 KV base address 生成地址 |
| §4.2.3（`:1031`） | 把 TierKV 的 SFA-tail 内 merge 与 S2 的 H2D&SFA 融合都归为“传输与计算融合” | 拆成两类：前者是 attention 与 softmax-state merge 融合，后者才是传输与 attention 融合；TierKV 的 gather 仍是独立环节 |
| §4.3(1)（`:1087-1098`） | “共享池容量 × 稀疏召回率 = 实际搬运量”在量纲来源和因果上都不成立；共享池容量不决定每步 miss 字节。“正交、收益相乘”也与后文分段/融合相互制约的描述冲突 | 逐步按 `Σ miss_entries × entry_bytes` 计 H2D。共享池降低 DRAM 副本容量，稀疏与预取改变 miss/同步加载量，两者只能称为潜在互补，需共同 workload、容量和事件图验证 |
| §2.2.1、§4.3(2)（`:169-171,1093-1098`） | “indexer 变快 → 留给 H2D 的窗口变长”不是无条件的依赖关系；没有固定 deadline 或独立并发工作时，缩短前置 indexer 也可能只是让 gather 更早开始 | 用明确时间线说明被释放的独立计算窗口、固定 deadline 和跨流依赖，不能把 indexer 加速自动换算成可隐藏的 H2D 时间 |
| §4.2.1、§4.2.2、§4.4（`:999,1019,1120`） | 将 TierKV 原文、当前 hisim 的容量公式和“每请求私有 host 池”混为一谈 | 当前 hisim 是 worker/simulator 级共享 `host_pool_tokens`，用 `cur_host_tokens` 聚合所有请求；应区分“请求私有 KV 内容、无跨请求去重”与“物理私有 host 池”。`t0 + bytes/(bw×0.85)` 也应标为当前 hisim 近似，而非 TierKV 原文定律 |
| §4.2.4（`:1044-1046`） | 用 hisim profile 的 main/index 分账证明“索引与 KV 物理分离”，证据不足；当前 BF16 profile 还明确将 index 嵌入 2885 B/token offload payload | 将该行限缩为“main/index 是否分账”。物理布局、全长索引是否常驻 HBM 和 selected index payload 的 H2D 计费是三个独立问题，不能由计费字段互推 |
| §4.5（`:1160-1161`） | 将 GVA 统一虚拟编址直接当成预取所需物理地址解析已经完成，忽略物理映射、映射有效期、fence/一致性和 KV 驻留生命周期 | GVA 只能简化句柄与寻址元数据；仍需验证物理映射及一致性事件，不能用“一份地址，多卡可用”替代这些检查 |
| §0、§3.4、§3.8、§4.2.6（`:45,869-874,970,1066-1070`） | S2 的“达成/目标/实测/未验证”状态词不一致，精度一处写“无”，另一处又承认材料未给数据 | 统一为“材料宣称/客户表转述，非本环境实测”；TPOT 是宣称达成还是设计目标需回查材料。精度写为“算法上不改变稀疏选择，实际数值一致性未验证” |
| §4.5（`:1162-1163`） | “indexer 加速后 LI Cache 可以缩小”缺乏依据；两阶段均值缓存还可能增加存储 | 只有明确的索引压缩、共享存储或卸载设计才能据此缩减 HBM。计算加速本身不等于索引 footprint 缩小，应先列出 `key_mean_cache` 等新增占用 |

### 9.4 低优先级：表述、单位与引用修正

1. §2.2.3（`:196-201`）的 `O(L²) → O(L/B+mB)` 不宜称为“**一般 Q-query
   形式**”，因为式中没有 `Q`。应称为“未显式区分 query 数的形式”；一般形式是
   `O(Q·L) → O(Q·(L/B+mB))`，并另计 pooling / `mean_update`。§2.2.5 已说明无缓存
   版本每次调用都会重算 block 均值，因此无缓存时不能漏掉对全部 key 的 `O(L)` 池化
   成本；cached 版本也要计缓存维护与新增存储。本节关于 decode 下旧式会高估相对
   收益的方向是正确的。
2. §2.2.10（`:345-356`）仅凭“降低的 token 数 381≈382”不能推出 Top-K 集合或
   精度等价，更不能直接证明跨层预测有效。建议改为“数量效果接近”，并补同层 IoU、
   跨层 recall/precision 后再讨论预取精度基础。
3. §2.3.4 的“命中率提升约 7%”应写成“提升约 7 个百分点”；§2.3.5 的
   `+6.9%`、`+5.26%` 分别是 `+6.90`、 `+5.26` 个百分点，不是相对百分比。
4. §3.4 第③项标题（`:876`）的 `D2RH` 与全文术语不一致，应核对并统一为 `RD2H`。
5. S3 来源与 §5 引导（`:13,1169`）的路径 `src/hisim/docs/tierkv_vs_hisparse.md`
   在当前仓不存在，应为 `docs/tierkv_vs_hisparse.md`。
6. §8 的“直接可用”建议改成“技术方向更直接对应单实例瓶颈”。S1 仍依赖自定义
   kernel，真实 `KVSelect` 尚未完成，top-m 有精度损失，且 BS=96 时多流收益已消失，
   不应读成当前仓库可直接启用。同表“能否叠加：能”也应保留 §4.5 的“推导、联合
   设计、待验证”限定。
7. §2.2.3–§2.2.4（`:183-215`）的“top-m 个 block，外加 sink/recent”与候选数恰好
   `m×B=8192` 不能无条件同时成立。若 sink/recent 是额外块，候选数还可能增加；若
   包含在 top-m 内，应删除“外加”，并说明去重和序列首尾边界规则。
8. §4.2.1（`:1002`）的“共享池（材料示例 360GB / 1.5T）”容易把两个数字都读成
   共享后的容量。按 §3.4，360GB 是共享后的示例，约 1.5TB 是无 MF 共享的基线，建议
   改成“共享 360GB vs 无共享约 1.5TB”。
9. §8（`:1282`）称 §4.4 有“四个张力点”，但表格实际有六行。若只指前四项组合实现
   问题，应明确“前四项”；否则改为“六个张力点”。
10. §5.1（`:1176-1180`）可再精确：`shared_index_freq` 只缩放关键路径 H2D 字节和
    时延，总 H2D bytes 不变；profile 路径是 hit-to-MRU、特定 miss 插入顺序和独立
    newest page 的组合，不宜只写“纯 LRU”，legacy 无 profile 路径也不是同一状态机。

### 9.5 已核对且可以保留的内容

- 三行 prefill 和 64K/128K 两行 decode 的 indexer 耗时比与标注加速比一致。
- TPS 23.3→45.8 约为 1.97×；同步加载量从 311.3 降至 65.33 约为 −79.0%。
- 自适应方案的 `288.78/394.60≈73.2%`、`263.55/92.18≈2.86×` 与正文一致。
- S2 两组吞吐重算约为 1.193×、1.300×，可保留约 1.2×/1.3× 的写法，但仍须遵守
  “单位与 workload 未标注”的限制。
- `KVSelect → KVGather` 依赖、两种 merge 语义、预取 FP/FN 与完成事件、IndexShare
  不共享 KV Tensor，以及 S1/S2 实测收益不可直接比较等判断可以保留。

### 9.6 建议的最小后续动作

1. 先回查 S1 原表，修正 32K decode 行、IoU 总结和 SFA 收益口径；这三项不需要新增
   仿真或硬件实验即可消除内部矛盾。
2. 将“模型算法不变”与“需要改 kernel”分开写，分别评估数值一致性、算子工作量和
   系统路径收益。
3. 对组合方案只建立容量、miss 字节和事件时延的独立公式，不在缺少共同 workload
   trace 时宣称容量与稀疏收益可以直接相乘。
4. S1/S2 原始材料未重新核验前，继续保留“原文/材料转述、未在本环境复现”的标记；
   本节的重算只证明内部一致性，不替代原始实测验证。
