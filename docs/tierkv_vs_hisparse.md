# TierKV 与 `develop_lfz_sparse_kv` HiSparse 仿真对比与落地建议

日期：2026-11-10
分支：`develop_lfz_sparse_kv`（HEAD `cd22c02`）
对比对象：稼先社区《TierKV：面向百万级长上下文的分层稀疏 KV 缓存卸载与推理加速架构》
（2026-07-30，作者张弓、王鹏、杨庆庆、谭峰、黄宏鸣、王荧荧、周华漫、周卿、张䶮、章威、
高翼枭、游轩珂、高源、蒲贵友、谭程夫、付克博、陈仁海）

相关文档：

- [`hisparse_simulation.md`](./hisparse_simulation.md)：HiSparse 基础模型（L0/L1）
- [`hisparse_dsv41_simulation.md`](./hisparse_dsv41_simulation.md)：DSV4.1 L0/L1/L2 与两次评审修复
- [`sparse_kv_sglang_vllm.md`](./sparse_kv_sglang_vllm.md)：sglang / vLLM 稀疏 KV 方案分析

> 本文只做对比与方案建议，**不修改代码**。

---

## 0. 结论速览

| TierKV 技术 | 当前 HiSparse 仿真 | 差距 | 建议优先级 |
| --- | --- | --- | --- |
| 分层卸载（HBM hot + host 全量） | ✅ 已实现（`physical × host_to_device_ratio`） | 只到 DRAM，无 SSD/ASU 层 | P2 |
| 两阶段层级 Indexer（块粗筛 + Token 精选） | ❌ 完全没有 indexer 计算时间模型 | 时延模型只覆盖 attention/搬运，不覆盖 indexer | **P0** |
| 无损预取（预测下一层访问集） | ⚠️ 仅 `shared_index_freq` 的 `1/N` 字节缩放 | 无预测、无预取命中率、无预取流量 | **P0** |
| 动态淘汰策略 | ⚠️ 纯 LRU（hit-to-MRU + newest page） | 无时序/层间感知淘汰，无 TTL/候选窗口 | P1 |
| 分段 SFA + 多流并行掩盖 | ❌ H2D 全量串行加到关键路径 | 无 hit/miss 分段、无部分 softmax 合并、无多流 | **P0** |
| 多层索引共享（IndexShare） | ⚠️ 只有 `shared_index_freq` 参数 | 无 layer group 结构、无 anchor/follower 事件 | P1 |
| 多 token 预测 / Hot Buffer 生命周期 | ❌ 未建模 | 无 candidate/accepted window、无 spec decoding | P2 |
| 容量与并发（BS 由 HBM 决定） | ✅ 已实现（hot-pool clamp 固定点） | 无 SSD 容量层、无跨请求全局去重 | P1 |

一句话：**当前仿真的容量/IO 记账已经比较完整，但“时延”这一侧几乎只算了搬运，
没算 indexer 计算、没算预取、也没算重叠**。TierKV 恰好是围绕这三件事做的，
所以它的大部分优化在仿真里对应的不是“改参数”，而是“补模块”。

---

## 1. TierKV 做了什么

### 1.1 问题定义

- KV Cache 随 `B × L` 线性膨胀，HBM 容量成为并发瓶颈（不是算力瓶颈）。
- 原生稀疏注意力（DSA）只降低**访问成本**，不降低**存储成本**：任意历史 token 都可能
  被 indexer 选中，完整 KV 仍必须驻留 HBM；部分模型还要额外常驻索引 Key。
- 结论：需要把**存储位置**与**访问位置**解耦。

### 1.2 架构

```
完整 KV ──► DRAM / SSD / ASU（大容量）
              ▲        │
     eager backup        │  Top-K miss 时 H2D
              │        ▼
HBM：hot pool（Top-K + 高频热点），每请求固定预算
              │
              └──► 稀疏 attention 只读 hot pool
```

### 1.3 三大核心优化

#### (1) 两阶段层级 Indexer（`lightning_indexer_hi`）

- **动机**：DS V3.2 中 indexer 占 attention 时延 >30%；DS V4 在 512K 输入下占总算子
  >30%、占计算算子 >40%；Ascend 950PR 上 indexer 与 SFA 单算子时延比 >8×。
- **依据**：Top-K token 存在**局部聚集性**。
- **方案**：
  - 第一阶段 block 粗筛：key 按 `B=128` 分组池化（mean）成 `M=L/B` 个 block 表征，
    query 与 block 表征打分，取 top-m 个 block；
  - 第二阶段 token 精选：在 `m × B` 个候选 token 上做细粒度打分，取 top-k。
  - 复杂度 `O(L²) → O(L/B + mB)`。
  - 参数：`hi_block_size=128`、`hi_block_num=64`（默认，候选 8192）、`sink=1`、
    `recent>=1`、`block_pooling_mode=mean`。
- **进阶**：`lightning_indexer_hi_cached` 缓存 key block 均值
  （`key_mean_cache` / `key_mean_cache_status`），配套 `scatter_nd_update_mean`
  在写 key 时同步维护 block 均值。
- **效果**：
  - 单算子：Prefill 1.71–2.14×，Decode 2.51–2.93×（随机数据，真实 IoU 更高）；
  - 端到端 DS V3.2：性能 +8~10%，精度影响 <1%；
  - LongBench V2：top-m=128 几乎无损，top-m=64 掉 2%；
  - 服务化实测多数层 IoU >0.8，回退 0、1 两层保精度。

#### (2) 无损预取 + 动态淘汰

- **动机**：卸载后各层访问集动态变化，逐层按需召回会把 H2D 暴露在关键路径。
- **方案**：利用**跨层空间相关性** + **跨步时间局部性**，当前层预测下一层访问集，
  提前异步预取 predicted Top-K 中不在 HBM 的部分，并 diff load 未命中部分。
- **效果**（GLM-5.1-w8a8 / Open-SWE-Traces / 缓存 8K / 128 step）：
  - 64K：命中率 90.15% → 97.21%（叠加动态预测 → 97.76%），历史最低 12.1% → 59.69%
    （→66.59%），<95% 层数 46 → 13（→5）；
  - 128K：89.5% → 96.80%（→97.03%），历史最低 4.8% → 50.45%（→55.71%），
    <95% 层数 45 → 15（→13）；
  - 层间预测复用（θ=0.6，省 1/3 预测开销）：73.2% 异步加载量下 miss 率降低 2.86×；
  - 整体：**KV 同步加载量降低 79%**。

#### (3) 分段稀疏注意力 + 多流并行

- **动机**：`lightning_indexer → gather_selection_kv_cache → sparse_flash_attention`
  严格串行；相邻 decode step 选中的 KV 约 **80% 重复**。
- **分段 SFA**：把选中集拆成 `H_t`（hit，已在 HBM）与 `M_t`（miss，需 gather）：
  ```
  Indexer(t)
    ├─ SFA-reuse(H_t)          # 不等 gather
    └─ Gather(M_t) -> SFA-tail(M_t) + merge
  ```
  两个分支导出 partial state `(O, m, l)`，用稳定 softmax 合并：
  ```
  m = max(m0, m1); p_i = l_i * exp(m_i - m); w_i = p_i / (p0+p1); O = w0*O0 + w1*O1
  ```
  实现选**方案二**：SFA-tail kernel 内部接收 prior state 并完成 merge，
  省一次 kernel launch 和中间结果 HBM 往返。
- **多流并行**：batch 切两个 micro-batch；`gather_selection_kv_cache` 拆成
  AICPU 上的 `KVSelect`（查未命中/落位）与纯传输的 `KVGather`；三条流并行：
  AIC/AIV 计算流、AICPU 流、通信流。
- **效果**：64K/128K、BS=32/64 下 5.17%–10.63%；收益主要来自 overlap，
  split 路径本身工作量更大；BS=96 时瓶颈转为 split 路径自身工作量。

### 1.4 模型侧信号（Index 共享）

- GLM-5.2 的 IndexShare：一组索引结果复用于连续 4 层，1M 上下文下单 token 计算量
  降低约 **2.9×**；多 token 预测使 speculative decoding 平均接受长度最高提升约 **20%**。
- 共享的是**历史 token 位置/稀疏索引**，不是层间 KV Tensor（各层投影参数不同）。
- Hot Buffer 生命周期应从“单层单 step 固定短 TTL”改为考虑
  Request / Layer Group / Candidate Window / Accepted Window；被拒绝候选只释放临时状态。

### 1.5 端到端实测口径

vLLM-0.18，单机 A3，DeepSeek-3.2 w8a8c16，TP=16/DP=1/EP=16，64K 序列；
基线 BS=1（BS=2 OOM），卸载后 BS=1/2/4/8；80ms 时延下 **TPS 23.3 → 45.8**。

---

## 2. 当前 HiSparse 仿真做了什么

### 2.1 已有能力

| 能力 | 位置 | 说明 |
| --- | --- | --- |
| 分层容量 | `hisparse_sim.py:1209-1221` | `logical = physical × host_to_device_ratio`，host 池 = 逻辑池 |
| 每请求 hot buffer 预留 | `HiSparseSimConfig.padded_buffer_size` | `device_buffer_size + page_size` |
| 并发上限 | `max_reqs_by_hot_pool = physical // padded` | 与 hot-pool 固定点联立求解 |
| 静态 KV 布局 profile | `KVProfile`（`hisparse_sim.py:62-337`） | L0/L1/L2：逐 source layer 字节、ratio 组、page/allocator、SWA/compressor、admission |
| 压缩条目 cadence | `_decode_one_profile`（`:1528`） | `floor(past_len/ratio)`，与 sglang C4 对齐 |
| Top-K 选择 cadence | `selection_cadence` | 默认 `every_step`，`compression_boundary` 仅敏感性对照 |
| 抽象访问模型 | `_WorkingSet`（`:756`） | `sticky` / `recency` / `uniform`，`churn` / `recent_ratio` |
| hot buffer LRU | `_access_selected`（`:900`） | hit-to-MRU、miss 插到 hit block 前、独立 newest page |
| 初始化模式 | `hot_buffer_init` | `earliest` / `latest` / `empty` |
| main/index IO 分账 | `io_channel_groups` | BF16 逐 source split |
| 命中时 index 读取 | `index_read_on_hit` | hit/miss 都计 index payload |
| prefix 复用 | `on_request_start(prefix_len=)` | staging 只计新增部分 |
| 搬运时延 | `transfer_duration`（`:718`） | `t0 + bytes/(bw×0.85)`，H2D 6.67µs / D2H 4µs |
| 关键路径缩放 | `shared_index_freq` / `critical_fraction` | 关键路径只计 `1/N` 字节 |
| D2H 侧流 | `backup_on_critical` | 默认只记账、不加时钟 |
| 指标 | `metrics()`（`:1702`） | 容量/命中/IO/时延/admission 全量字段 |

### 2.2 当前**没有**的能力（与 TierKV 直接相关）

1. **没有 indexer 计算时间模型**。`time_predictor` 只预测 forward（attention/GEMM），
   `ConstantTimePredictor` 更是常数；`hisparse_sim` 只算搬运。indexer 的两阶段加速
   在仿真里无处体现。
2. **没有预取**。`shared_index_freq` 只是把聚合 H2D 字节乘 `1/N`，既不代表预测，
   也不产生预取流量、不改变命中率。
3. **没有 hit/miss 分段与重叠**。`_account_transfer` 把 batch 内全部 H2D 字节
   串行求和后一次加到全局时钟（`sglang_hook.py:2598-2626`），
   没有 `max(compute, overlapped IO)` 的时序。
4. **没有多流**。一个 batch 只计一次 launch overhead，所有请求/组共享同一带宽。
5. **没有 layer group / anchor-follower 事件**。`shared_index_freq` 一个参数同时
   隐含了“共享选择”“预取字节”“延迟缩放”三件事。
6. **没有 candidate/filter 路径**。`candidate_blocks=2048 × 8 tokens` 只是元数据。
7. **没有 SSD/ASU 层**。host 池只有 DRAM 一档，`host_to_device_ratio` 是唯一容量旋钮。
8. **没有 speculative decoding / candidate window**。Hot Buffer 只有单 step 语义。

---

## 3. 逐项对比与落地建议

### 3.1 【P0】两阶段 Indexer 时延模型

**TierKV 做法**：块粗筛 + Token 精选，复杂度 `O(L²)→O(L/B+mB)`，
Prefill 1.8×+ / Decode 2.5×+，端到端 +8~10%。

**当前仿真**：完全没有 indexer 时间项。这意味着：

- 无法评估“indexer 加速”对 TPOT 的贡献；
- 无法评估“indexer 变快后释放的时延窗口”能给预取/传输留多少余量；
- 文章里 indexer 占比 >30% 这一关键事实在仿真中不可见。

**建议改动**：

1. 新增 indexer 时延模型（解析式或查表），输入 `seq_len`、`batch`、`top_k`、
   `hi_block_size`、`hi_block_num`，输出 indexer 时间：
   ```
   t_index_full = a + b * L²          # 原始
   t_index_hi   = a + b * (L/B + mB)  # 两阶段
   ```
   系数从 Ascend 950PR 单算子实测拟合（文章表 2.1/2.2 可直接作为标定点）。
2. 在 `HiSparseSimConfig` 增加：
   - `indexer_mode: full | two_stage`
   - `hi_block_size`（默认 128）
   - `hi_block_num`（默认 64，即 top-m）
   - `indexer_launch_us` / `indexer_us_per_mtoken`
3. 在 `StepCost` 增加 `indexer_dur`，在 `_account_transfer` 或
   `sglang_hook` 的时钟推进处加入。
4. 增加 `key_mean_cache` 的收益项：命中缓存时 indexer 时间再乘一个
   `mean_cache_hit_ratio` 的折减（或直接减掉 block 池化项）。
5. 指标：`hisparse_indexer_dur_s`、`hisparse_indexer_share`（占 forward 比例）。

**预期收益**：让“indexer 加速 → TPOT 下降”这条链路在仿真里可量化；
文章 +8~10% 的端到端收益可作为首个校准目标。

**风险**：系数强依赖硬件与 kernel 实现；建议先做**相对收益**（full vs two_stage），
绝对时延等有实测再校准。

---

### 3.2 【P0】无损预取模型

**TierKV 做法**：当前层预测下一层访问集，异步预取，diff load；
命中率 90%→97%，同步加载量 −79%。

**当前仿真**：`shared_index_freq` 只是 `critical_fraction = 1/N`，
总字节不变、命中率不变、无预取流量。这会把“预取”错误地建模成“免费打折”。

**建议改动**：

1. 在 `HiSparseSimConfig` 增加预取参数：
   - `prefetch_enable: bool`
   - `prefetch_accuracy: float`（预测命中率，0~1）
   - `prefetch_depth: int`（预测下一层 or 下 N 层）
   - `prefetch_lead_steps: int`（提前几个 step）
   - `dynamic_prefetch: bool`（是否叠加动态预测）
   - `prefetch_theta: float`（层间复用阈值，文章 θ=0.6）
2. 在 `_WorkingSet` 之外增加一个**预测器视图**：
   - 每步先按 `prefetch_accuracy` 从下一层真实访问集中采样出“预测集”；
   - 预测集与 HBM 的差集作为**异步预取流量**（计入 `h2d_bytes`，但默认不计关键路径）；
   - 下一步真实访问时，若命中预测集则算 hit，否则算 miss（同步加载）。
3. 关键路径拆分：
   - `critical_h2d_bytes` = 未预取到的 miss 字节；
   - `prefetch_h2d_bytes` = 预取字节（默认侧流）；
   - 预取流量仍计入带宽竞争（可选 `prefetch_bandwidth_share`）。
4. 淘汰策略：增加 `eviction_policy: lru | ttl | layer_aware`，
   以及 `hot_ttl_steps`；层间感知可用“上一层的 Top-K 位置集合”作为保护集。
5. 指标：`hisparse_prefetch_bytes`、`hisparse_prefetch_hit_rate`、
   `hisparse_sync_load_bytes`、`hisparse_layer_min_hit_rate`（长尾层）。

**预期收益**：能复现文章“同步加载量 −79%”“最差层命中率 12%→60%”这两类结论；
也能解释 `shared_index_freq=4` 在旧结果里为什么把 TPOT 从 0.748 拉到 0.586 ms——
那是“假设免费”，不是“预测有效”。

**风险**：`prefetch_accuracy` 是新的合成旋钮，必须用真实 Top-K trace 校准，
否则只是把 `churn` 的问题复制一遍。建议与 §4 的 trace 采集一起做。

---

### 3.3 【P0】分段 SFA + 多流重叠时序

**TierKV 做法**：hit/miss 拆分，SFA-reuse 与 Gather 重叠，partial softmax 合并；
多 micro-batch + 三流并行；端到端 5%~10%。

**当前仿真**：`_account_transfer` 把 batch 全部 H2D 串行加到时钟，
`shared_index_freq` 只是字节打折。**没有任何 overlap 建模**。

**建议改动**：

1. 在 `StepCost` 增加分段字段：
   ```
   hit_kv_bytes / miss_kv_bytes
   hit_sfa_dur / miss_sfa_dur / merge_dur
   gather_dur / indexer_dur
   ```
2. 在 `_account_transfer` 用**事件图**替代简单求和：
   ```
   t_indexer  = ...
   t_hit_sfa  = f(hit_kv_bytes)
   t_gather   = transfer_duration(miss_kv_bytes)
   t_miss_sfa = f(miss_kv_bytes)
   step_dur   = t_indexer
              + max(t_hit_sfa, t_gather)      # 分段重叠
              + t_miss_sfa + t_merge
   ```
   这是文章分段 SFA 的最小可验证模型。
3. 多流：增加 `num_streams`（默认 1）与 `micro_batch`（默认 1）；
   `num_streams=3` 时按文章的三条流建模：
   - 计算流：`indexer0 → indexer1 → hit_sfa0 → hit_sfa1 → miss_sfa0+merge0 → ...`
   - AICPU 流：`kvselect0 → kvselect1`
   - 通信流：`kvgather0 → kvgather1`
   可用一个简化的 list scheduling（每流串行、流间并行）实现，不需要真实 kernel。
4. 把 `shared_index_freq` 拆成三个独立参数，避免一个旋钮隐含三件事：
   - `index_share_group`（每 N 层共享选择）
   - `index_prefetch_layers`（follower 预取层数）
   - `anchor_critical`（anchor 是否同步）
5. 指标：`hisparse_overlap_gain`、`hisparse_step_dur_us`、
   `hisparse_gather_dur_us`、`hisparse_sfa_dur_us`。

**预期收益**：能回答“分段 SFA 在什么 BS/序列长度下还有正收益、什么时候转为负收益”
（文章明确 BS=96 时 overlap 不再带来收益），这正是仿真最该回答的问题。

**风险**：需要 attention 计算时间模型。当前 `ConstantTimePredictor` 只有常数，
`aiconfigurator` 只给整体 forward。建议先按 `bytes/带宽` 与 `top_k × 层数` 的
解析式近似 SFA 时间，标注为“相对模型”。

---

### 3.4 【P1】多层索引共享（IndexShare）

**TierKV 做法**：一组索引复用于连续 4 层，1M 上下文单 token 计算量 −2.9×；
MTP 接受长度 +20%。

**当前仿真**：`shared_index_freq` 只有 `1/N` 字节缩放，没有 layer group 结构，
也没有“共享的是位置不是 KV”的语义。

**建议改动**：

1. 用 `KVProfile.source_consumers` / `index_compute_layers` 已有的拓扑
   建立 layer group（DSV4.1 已有 `[2,8,14,20]` 与 `[2,8,14,20,24,28,32,36]`）。
2. 增加 `index_share_group_size`（默认 1，GLM-5.2 场景为 4）。
3. 组内首层做完整 indexer（计 `indexer_dur` + 选择），后续层复用索引：
   - indexer 时间按组摊销（`indexer_dur / group_size`）；
   - KV 搬运仍按层独立（各层 KV Tensor 不同），但位置集合相同；
   - 预取可在组首层一次性解析地址，组内后续层直接复用。
4. 指标：`hisparse_indexer_dur_per_group`、`hisparse_index_share_gain`。

**预期收益**：把文章“1M 上下文 −2.9× 计算量”翻译成仿真里的 indexer 时间下降；
同时修正 `shared_index_freq` 的语义混淆。

---

### 3.5 【P1】动态淘汰与 Hot Buffer 生命周期

**TierKV 做法**：LRU → 时序/层间感知 + 动态预测；Hot Buffer 生命周期考虑
Request / Layer Group / Candidate Window / Accepted Window。

**当前仿真**：`_access_selected` 是标准 LRU（hit-to-MRU、miss 插到 hit block 前、
独立 newest page），`hot_buffer_init` 三选一；没有 TTL、没有保护集、没有候选窗口。

**建议改动**：

1. 增加 `eviction_policy`：
   - `lru`（现状）
   - `ttl`（`hot_ttl_steps`）
   - `layer_aware`（上一层 Top-K 位置作为保护集）
   - `frequency`（LFU/ARC 对照）
2. 增加 `protect_recent_steps`：最近 N 步被选中的位置不淘汰。
3. 为 speculative decoding 预留：
   - `spec_candidate_window: int`（候选 token 数）
   - `spec_accept_len: float`（平均接受长度）
   - 候选窗口内的历史 KV 保留到验证结束，被拒绝候选只释放临时状态。
4. 指标：`hisparse_evictions`、`hisparse_protected_evictions`、
   `hisparse_ttl_expired`。

**预期收益**：能对比“LRU vs 层间感知 vs TTL”的命中率/流量差异，
复现文章表 3.1（LRU 84.80% → 预测预取 96.81%）。

---

### 3.6 【P2】SSD / ASU 第三层

**TierKV 做法**：完整 KV 下沉 DRAM/SSD，后续拓展 ASU；当前实测是 DRAM + 同步加载。

**当前仿真**：只有 host DRAM 一档（`host_pool_tokens`），
`host_to_device_ratio` 是唯一容量旋钮。L3（SSD）能力在 hisim 里是给
HiCache 用的（`kvcm_optimizer`），与 HiSparse 互斥。

**建议改动**：

1. 增加 `tier_config`：
   - `dram_ratio`（相对 physical）
   - `ssd_ratio`（相对 physical，0 表示不启用）
   - `ssd_read_bw` / `ssd_write_bw` / `ssd_latency_us`
2. 访问路径：HBM miss → DRAM hit/miss → SSD hit/miss，逐层计流量与延迟。
3. 淘汰：DRAM → SSD 的 demotion，SSD → DRAM 的 promotion。
4. 指标：`hisparse_dram_hit_rate`、`hisparse_ssd_hit_rate`、
   `hisparse_ssd_read_bytes`、`hisparse_ssd_latency_s`。

**预期收益**：覆盖文章“后续支持 SSD/ASU”的方向；也能回答
“DRAM 容量不够时 SSD 尾延迟是否可接受”。

**风险**：SSD 延迟比 DRAM 高 2~3 个数量级，模型必须区分“预取可掩盖”与
“同步加载不可掩盖”，否则结论会过于乐观。

---

### 3.7 【P2】跨请求全局去重与 prefix 生命周期

**TierKV 做法**：文中未展开，但 Hot Buffer 生命周期与请求/层组绑定。

**当前仿真**：`on_request_finish` 释放该请求全部 host tokens；
prefix 复用只从 staging 扣除，host 侧共享 prefix 的引用生命周期未建模。

**建议改动**：

1. host 侧 prefix 建引用计数（或按 radix 前缀共享）；
2. 请求结束后保留仍被其他请求引用的 prefix；
3. 指标：`hisparse_shared_prefix_tokens`、`hisparse_host_reclaim_tokens`。

---

## 4. 建议的落地顺序

结合 `hisparse_dsv41_simulation.md` §12 的评审结论（P0 确定性错误已修，
仍有 trace/时序缺口），建议：

### 阶段 1：先补“时延侧”的三个模块（对应 TierKV 三大技术）

1. **Indexer 时延模型**（§3.1）——最小改动、收益最直接；
2. **预取模型**（§3.2）——需要与 trace 采集一起做，否则是新的合成旋钮；
3. **分段 SFA + 多流时序**（§3.3）——需要 attention 时间近似。

这三项完成后，`hisparse_mean_load_us` 这一个指标才会变成
“关键路径时延”，而不是“总搬运时间”。

### 阶段 2：语义与结构

4. **IndexShare layer group**（§3.4）——复用已有 `source_consumers` 拓扑；
5. **动态淘汰策略**（§3.5）——先做 `layer_aware`，再做 TTL/spec。

### 阶段 3：容量层扩展

6. **SSD/ASU 第三层**（§3.6）；
7. **跨请求去重**（§3.7）。

### 阶段 0（前置）：trace 与校准

- 真实 Top-K ID/分数（indexer 输出）；
- hot-buffer hit/miss/swap-in/swap-out 计数器；
- host/SSU 真实带宽、延迟、尾延迟；
- 单次 kernel launch 的 `t0` 与有效带宽。

没有这些，阶段 1 的 `prefetch_accuracy`、`indexer` 系数、overlap 收益
都只是新的假设。**建议先采 trace，再改模型**。

---

## 5. 关键差异总结（一句话版）

| 维度 | TierKV | 当前仿真 |
| --- | --- | --- |
| Indexer | 两阶段算子，Prefill 1.8×/Decode 2.5× | 无时间模型 |
| 预取 | 预测下一层 + diff load，同步加载 −79% | `1/N` 字节打折 |
| 重叠 | 分段 SFA + 三流，端到端 5~10% | 全串行加到时钟 |
| 索引共享 | layer group 复用，1M 上下文 −2.9× | 单个 `shared_index_freq` |
| 淘汰 | 时序/层间感知 + 动态预测 | 纯 LRU |
| 存储层 | DRAM / SSD / ASU | 仅 DRAM |
| 容量/并发 | BS 由 HBM 决定，TPS 23.3→45.8 | ✅ 已建模（hot-pool clamp） |
| IO 分账 | main/index 分离 | ✅ 已建模（BF16 逐 source split） |

**结论**：当前仿真的“容量与 IO 记账”已经能支撑容量/并发结论；
缺的是 TierKV 三大技术对应的**时延与调度模型**。建议按 §4 的阶段 1 优先补齐
indexer / 预取 / 重叠三项，再谈 SSD 与 IndexShare。

---

## 6. 参考

- 原文：稼先社区《TierKV：面向百万级长上下文的分层稀疏 KV 缓存卸载与推理加速架构》，
  postId `77ec4f82ff5b49239e06575182e70f0d`，2026-07-30。
- 同文：2012 网站系列 `1299532321674678272`；思想简报第 28866 期。
- 参考实现：vLLM KV Offloading Connector、NVIDIA Dynamo、LMCache、GLM-5.2、HISA
  （arXiv:2603.28458）。
- 本仓库：`src/hisim/simulation/sglang/hisparse_sim.py`、
  `src/hisim/simulation/sglang/sglang_hook.py`、
  `benchmark/hisparse/hisparse_bench.py`。

---

## 7. 评审建议更新（2026-11-10）

> 本节是评审补充，不改变前文作为原对比材料的记录；若与前文冲突，以本节为准。
> 评审基于分支 `develop_lfz_sparse_kv` 的实际 HEAD
> `a35283721c5b6d0cf67d6bf6455363743d90307e`。原文中的 TierKV 外部实测数字
> 仍属于转述，尚未在本仓库环境独立复现。

### 7.1 总体结论

对比方向和“先采 trace、再补时延模型”的落地顺序是合理的，但当前文档还不能直接
作为实现规格。主要问题是：Indexer 复杂度混用了 prefill/decode 口径；预取模型只
包含真阳性且默认传输必然及时完成；SFA 事件图漏掉 `KVSelect`/跨流依赖；把
DSV4.1 的 KV 复用拓扑误当成 GLM IndexShare 分组；以及把静态容量解析能力扩大为
一般容量、并发或 TPS 结论。

建议先修正以下语义和公式，再用真实 Top-K/hot-buffer/SSU trace 校准。否则新增的
`prefetch_accuracy`、indexer 系数和 overlap 收益只是第二组合成旋钮。

### 7.2 Indexer 复杂度必须区分 prefill 与 decode

前文 `O(L²) → O(L/B + mB)`（`:70`）及 `t_index_full = a + bL²`（`:192-193`）
混淆了两个口径：

- decode 通常只有一个 query，full indexer 是 `O(L)`，两阶段是
  `O(L/B + mB)`；
- prefill 有约 `Q=L` 个 query，full indexer 才是 `O(L²)`，两阶段应保留 `Q`。

建议至少改为：

```text
t_full = a + b × Q × L
t_hi   = a + b × Q × (L/B + mB) + t_mean_update
```

其中 `Q=1` 表示常规 decode；prefill 按实际 query 数展开，并另计 batch、Top-K、
均值缓存维护和 launch。标定系数时也要按 prefill/decode 分开，不能用一个 `b`
同时拟合两种阶段。

### 7.3 预取模型必须包含假阳性、完成事件和缓存副作用

`:232-245` 的方案从“下一层真实访问集”采样预测集，意味着预测集只包含真阳性：
没有 false positive，也没有预测错误造成的额外流量和缓存污染。下一步若命中预测集
就直接算 hit，又等价于假定所有异步预取都在使用前完成，使 `prefetch_lead_steps`
和带宽竞争失去意义。

建议用集合显式表达：

```text
P_t = 预测访问集
A_t = 下一层真实访问集
TP_t = P_t ∩ A_t
FP_t = P_t - A_t
FN_t = A_t - P_t
C_t  = 到本次访问前已经完成且仍驻留 HBM 的预取集
同步 miss = A_t - C_t
```

预取集必须进入热池容量和淘汰逻辑，并记录开始/完成时刻、排队、带宽占用和
false-positive 污染。若没有逐层 trace，应使用 recall/precision 两个参数做敏感性
分析，而不是只给一个“准确率”。

建议把 `:245` 的“能复现 −79%”改为：“该结构用于检验需要多高的预测
recall/precision 和提前量才能接近文章收益；最终复现程度由同 workload trace
校准后判定。”

### 7.4 SFA 事件图需要选定一种 merge 语义并补依赖

`:264-292` 同时描述了两套不同结构：

1. hit/miss 两支分别产生 partial state，最后独立 merge；
2. SFA-tail 接收 hit 的 prior state，并在 tail kernel 内完成 merge。

两者不能共用同一个公式。若采用独立 partial state：

```text
t_indexer
+ max(t_hit_sfa, t_kvselect + t_gather + t_miss_sfa)
+ t_merge
```

若采用 tail 接收 prior state，则 tail 必须同时等待 hit SFA 与 gather，且 merge
已包含在 tail 内，不能再无条件单列一次 `t_merge`。同时，`KVSelect` 是
`KVGather` 的前置依赖，不能只把它列在三条并发流中而不进入关键路径。

多流调度应按事件依赖实现，而不是简单给三条流各排一条串行链。现有
`hisparse_mean_load_us` 是 H2D-only 指标；补完 indexer/SFA 后应新增
`hisparse_step_dur_us` 或 `hisparse_critical_path_us`，不要改变原指标的含义。

### 7.5 IndexShare 不能从 DSV4.1 现有拓扑直接推导

`:313-319,413` 用 `KVProfile.source_consumers` / `index_compute_layers` 建立
GLM 式四层共享组，这是两种不同概念：

- `source_consumers` 是 KV Tensor 的 source→consumer 复用拓扑；
- `index_compute_layers` 是索引计算层集合；
- GLM IndexShare 共享的是同一组历史位置/Top-K，不是 KV Tensor。

因此不能据此推出“连续四层共享同一 Top-K”，也不能把 anchor 关键路径直接除以
组大小。建议新增显式的 `index_share_groups` 配置，只对已确认使用 IndexShare 的
模型启用；DSV4.1 在获得真实 trace 前保持 `group_size=1`。

### 7.6 结论强度需要收窄到静态解析边界

`:32,443,446-448` 将“给定布局和预算下的静态 hot-pool 解析上限”扩大为一般
容量/并发结论，并把它与 TierKV 的实测 TPS 并列，容易形成能力等价的误读。

当前仿真可以支持：

- 给定 profile、预算和部署并发下的物理/逻辑 token 上限；
- SWA、compressor、admission 上界的静态分账；
- 不同 churn/recent_ratio 假设下的相对敏感性。

当前不能直接支持：

- 真实最大并发、端到端 TPS 或 TPOT 预测；
- 未建模的 candidate/filter、shared prefix 生命周期和实际调度；
- 合成 Top-K 命中率与真实 kernel 命中率的等价性。

建议将 `:443` 的“✅ 已建模（hot-pool clamp）”改为“已建模静态 hot-pool 上界”，
并删除与实测 TPS 的等价暗示。

### 7.7 可复现信息与事实修正

- `:4` 的 HEAD `cd22c02` 不是有效 git 对象，实际为 `a352837`。
- `:165` 的“一个 batch 只计一次 launch overhead”不准确：当前按 batch 聚合后，
  非空 index/main 阶段分别计 fixed overhead；真正的问题是未按 layer/request/group
  展开，而不是只计一次。
- `:461` 的源码路径应为 `benchmark/hisparse/hisparse_bench.py`。
- 建议为原文补充可访问 URL、版本/快照日期。公开检索中存在同名但不同的 TierKV
  工作，仅有 `postId` 不足以稳定区分引用对象。

### 7.8 建议验收顺序

1. **语义 oracle**：用手工 Top-K 序列分别覆盖 direct-short、长序列 newest、
   多 hit/miss 顺序和不可行预算；逐步核对 hit/miss、H2D/D2H 与 hot slots。
2. **trace 采集**：逐层 Top-K ID/分数、hot-buffer 计数器、预取完成事件、
   host/SSU 带宽与尾延迟。
3. **模型接入**：先实现 indexer 时间项，再加入带完成事件的预取，最后实现带依赖
   的 SFA/多流事件图。
4. **校准与报告**：固定同一 workload 比较 full vs two_stage、LRU vs 预测预取、
   串行 vs 分段 overlap；在 trace 校准前，所有绝对命中率和时延都只标为敏感性结果。

在这四步完成前，本文的优先级建议可以保留，但不应把任何合成结果写成对 TierKV
实测收益的复现或验证。
