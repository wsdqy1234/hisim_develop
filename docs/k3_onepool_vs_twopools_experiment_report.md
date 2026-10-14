# One Pool vs Two Pools 收益验证实验报告

> **代码版本**：`develop_dqy` 分支, commit `2e08daf`
>
> **实验日期**：2026-10-13 ~ 2026-10-14

---

## 1. 实验背景

### 1.1 目标

在相同物理内存预算下，验证 One Pool（UnifiedRadixCache 动态共享）相对于 Two Pools（固定分区）的缓存收益优势及边界条件。

核心问题：**One Pool 能否通过双向动态空间借用适应变化的 MLA/KDA 工作集，获得比固定分区更好的有效缓存复用和更少的计算工作量？**

### 1.2 物理预算

```
Total = KDA_slots × KDA_entry_bytes + MLA_tokens × MLA_entry_bytes
      = 80 × 14,653,440 + 200,000 × 5,760
      = 2,324,275,200 Bytes (≈ 2.16 GB)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| KDA entry_bytes | 14,653,440 B/slot | 18 层 × 814,080 B/层 |
| MLA entry_bytes | 5,760 B/token | 5 层 × 288 × 2(K+V) × 2(bf16) |
| 总预算 | 2,324,275,200 B | 两种模式严格相同 |

### 1.3 两种模式

| 模式 | KDA 分配器 | MLA 分配器 | 容量管理 |
|------|-----------|-----------|---------|
| Two Pools | 固定 N slots | 固定 M tokens | 超限直接 Eviction |
| One Pool | MultiEndedAllocator | MultiEndedAllocator | Flush 水位线动态调整 |

One Pool 的 `max_slots = total_bytes // entry_bytes`：
- KDA: 158（可从 80 动态扩展到 158）
- MLA: 403,520（可从 200,000 动态扩展到 403,520）

### 1.4 Two Pools 固定分区表

| KDA Slots | MLA Tokens | KDA Bytes | MLA Bytes | Total |
|-----------|------------|-----------|-----------|-------|
| 40 | 301,760 | 586 MB | 1,738 MB | 2,324 MB |
| 60 | 250,880 | 879 MB | 1,445 MB | 2,324 MB |
| 80 | 200,000 | 1,172 MB | 1,152 MB | 2,324 MB |
| 100 | 149,120 | 1,465 MB | 859 MB | 2,324 MB |
| 120 | 98,240 | 1,758 MB | 566 MB | 2,324 MB |

### 1.5 统一实验约束

- L1-only，关闭 HiCache L2/L3
- 模型、TP、Scheduler、并发=1 全程一致
- 不修改 SGLang 原生 Prefix Match / Checkpoint / Eviction 算法
- SGLang 调度器设计性地将 Prefix Match 限制为 `input_len - 1`（为 logprob 保留最后 token），不修改此规则
- Reuse 输入 = Build 输入 + N 个新 token（自然对话延续），确保 full match
- `mamba_last_track_seqlen` = `(seq_len // C) * C`（正确模拟 Mamba Tracking 边界）

---

## 2. 边界验收

### 2.1 测试设计

在正式实验前，通过 3 组最小用例（1 Build + 1 Reuse）验证 COW 机制正确性：

| 测试 | Build | Reuse | 预期 | 验证点 |
|------|-------|-------|------|--------|
| 负例 | C=2048 | 2048 | COW 失败 | `input_len-1` 导致 partial match |
| 正例1 | C=2048 | 2049 | COW 成功 | full match, replay=0 |
| 正例2 | C=2048 | 2304 | COW 成功 | full match, new=256 |

### 2.2 验收结果

| 测试 | TP search_key | TP partial | TP COW-SRC | OP COW-SRC | 一致性 |
|------|-------------|-----------|-----------|-----------|--------|
| 负例 | 2047 | True | None | None | ✓ |
| 正例1 | 2048 | False | slot=5, pos=2048, replay=0 | slot=7, pos=2048, replay=0 | ✓ |
| 正例2 | 2048 | False | slot=5, pos=2048, replay=0, new=256 | slot=7, pos=2048, replay=0, new=256 | ✓ |

Two Pools 与 One Pool 行为完全一致（Slot ID 不同但语义相同）。COW 机制验收通过。

---

## 3. 实验 P5.1：KDA-heavy 正向扩容

### 3.1 实验设定

| 参数 | 值 |
|------|-----|
| Session 数 | 96（独立 1024-token 会话） |
| C (checkpoint interval) | 1024 |
| Build input_len | 1024 |
| Reuse input_len | 1025（Build + 1 new token） |
| Reuse sessions | 6（sessions 0, 16, 32, 48, 64, 80） |
| 理论 KDA 需求 | 96 checkpoints（每 Session 1 个） |
| 理论 MLA 需求 | 98,304 tokens（96 × 1024） |
| 对照 | Two Pools KDA=40/60/80/100/120 + One Pool |

**负载特征**：96 KDA Checkpoints > 80 slots（默认分区容量不足），MLA 需求远低于任何分区容量。验证 One Pool 能否将闲置 MLA 空间分配给 KDA。

### 3.2 结果

| 配置 | KDA Slots | COW Hit | COW Miss | prefix_ratio | peak_mamba | 说明 |
|------|-----------|---------|----------|-------------|------------|------|
| TP-40 | 40 | 2/6 | 4 | 0.333 | 40 | KDA 严重不足 |
| TP-60 | 60 | 3/6 | 3 | 0.500 | 60 | KDA 不足 |
| TP-80 | 80 | 4/6 | 2 | 0.666 | 80 | 默认分区 |
| TP-100 | 100 | **6/6** | 0 | 0.999 | 100 | 最优固定分区 |
| TP-120 | 120 | 5/6 | 1 | 0.833 | 120 | MLA 不足→级联 |
| **One Pool** | 80→97 | **6/6** | 0 | 0.999 | **97** | 动态扩展 |

所有 COW Hit 的 `checkpoint_pos=1024, replay_tokens=0`。

### 3.3 分析

- **One Pool 动态扩展**：从初始 80 KDA Slot 扩展到 97（3 次 flush），容纳全部 96 Checkpoints + 1 活跃请求
- **One Pool = 最优固定分区**：6/6 COW Hit = KDA=100，但无需预知最优分区
- **KDA=120 级联**：MLA=98,240 < 98,304 → MLA 驱逐 → 级联释放 KDA → 1 个 COW Miss
- **COW Miss 分析**：Two Pools KDA=80 淘汰了最早 18 个 Checkpoint（sessions 0-17），Reuse 的 sessions 0 和 16 落在淘汰范围内

---

## 4. 实验 P5.2：MLA-heavy 反向扩容

### 4.1 P0：反向扩容能力验证

发送 8 × 32768-token sessions（262,144 MLA tokens > 200,000 限制）到 One Pool：

| 指标 | 值 | 说明 |
|------|-----|------|
| **peak_mla_live_pages** | **253,952** | MLA 超过 200K，反向扩容成功 |
| peak_mamba_live_pages | 33 | KDA 仅用 33 slots |
| unified_flush | 2 | 2 次水位线调整完成 KDA→MLA 借用 |
| unified_alloc_fail | 0 | 无分配失败 |

**结论**：`max_total_tokens=200000` 仅影响物理预算计算，不是运行时 MLA 容量上限。One Pool 可通过 flush 从 KDA 反向借空间给 MLA。

### 4.2 实验设定

| 参数 | 值 |
|------|-----|
| Session 数 | 8（独立 32768-token 会话） |
| C | 8192 |
| Prefill chunk | 8192 |
| Build input_len | 32768 |
| Reuse input_len | 33024（Build + 256 new tokens） |
| Reuse sessions | 3（sessions 0, 3, 7） |
| 理论 KDA 需求 | 32 checkpoints（8 × 4，每 Session 在 8192/16384/24576/32768 各 1 个） |
| 理论 MLA 需求 | 262,144 tokens（8 × 32768） |
| 对照 | Two Pools KDA=40/60/80/100 + One Pool |

**负载特征**：262,144 MLA tokens > 200,000（默认 MLA 容量不足），KDA 需求 32 << 任何分区容量。验证 One Pool 能否将闲置 KDA 空间分配给 MLA。

### 4.3 结果

| 配置 | KDA | MLA | COW Hit | prefix_ratio | new_tokens | 说明 |
|------|-----|-----|---------|-------------|------------|------|
| TP-40 | 40 | 301,760 | **3/3** | 0.992 | 768 | MLA fits, KDA fits → 最优 |
| TP-60 | 60 | 250,880 | 3/3 | 0.827 | 8,704 | MLA 略不足 |
| TP-80 | 80 | 200,000 | 2/3 | 0.992 | 512 | MLA 不足→级联淘汰 KDA |
| TP-100 | 100 | 149,120 | 2/3 | 0.744 | 8,192 | MLA 严重不足 |
| **One Pool** | 80→33 | 200K→262K | **3/3** | **0.992** | **768** | 反向扩容 |

所有 COW Hit 的 `checkpoint_pos=32768, replay_tokens=0`。Per-request L1 breakdown 确认所有 Reuse 的 `L1=32768`（全 MLA Prefix Hit）。

### 4.4 分析

- **One Pool 反向扩容**：MLA 从 200K 扩展到 262K（2 次 flush），KDA 仅用 33 slots（远低于 80）
- **One Pool = 最优固定分区**：3/3 COW Hit = KDA=40，但无需预知最优分区
- **TP-80 级联**：MLA=200,000 < 262,144 → MLA 驱逐 → `_release_all_component_layers` 同时释放 KDA → 1 个 COW Miss
- **TP-100 严重级联**：MLA=149,120 << 262,144 → prefix_ratio 仅 0.744

---

## 5. 实验 P5.3：动态混合负载

### 5.1 实验设定

| 参数 | 值 |
|------|-----|
| C | 1024（全程不变） |
| Prefill chunk | 32768（全程不变） |
| 并发 | 1（全程不变） |
| 阶段间不重启 Server | 是 |
| 阶段间不清空 Radix Tree | 是 |

### 5.2 负载校准

| Phase | 内容 | KDA 需求 | MLA 需求 | KDA/MLA 比 | 特征 |
|-------|------|---------|---------|-----------|------|
| A | 96 × 1024 tokens | 96 | 98,304 | 1/1024 | KDA-heavy |
| B | 8 × 32768 tokens | 8 | 262,144 | 1/32768 | MLA-heavy |

确认 Phase A 的 KDA/MLA 需求比例远高于 Phase B，负载特征显著不同。

### 5.3 实验 Trace

| Phase | 描述 | 请求数 | 说明 |
|-------|------|--------|------|
| A Build | 96 × 1K | 96 | KDA-heavy Build |
| A Reuse | 6 × 1K+1 | 6 | 重访 sessions 0,16,32,48,64,80 |
| B Build | 8 × 32K | 8 | MLA-heavy Build（不清缓存） |
| B Reuse | 3 × 32K+256 | 3 | 重访 sessions 0,3,7 |
| C Revisit | 3×1K+1 + 3×32K+256 | 6 | 跨阶段重访 A(0,32,80) + B(0,3,7) |

总 MLA 需求 360,448 tokens + 总 KDA 需求 104 checkpoints，超过预算 → 必然淘汰。

### 5.4 对照配置

- Two Pools KDA=40 / MLA=301,760（MLA-optimal 固定分区）
- Two Pools KDA=80 / MLA=200,000（默认分区）
- One Pool：初始 80/200K，共享预算贯穿全程

### 5.5 逐阶段结果

| Phase | 指标 | One Pool | TP KDA=80 | TP KDA=40 |
|-------|------|----------|-----------|-----------|
| A Reuse | COW Hit | **6/6** | 4/6 | 2/6 |
| A Reuse | L1 prefix | 1024（全命中） | 1024(4) / 0(2) | 1024(2) / 0(4) |
| B Reuse | COW Hit | **3/3** | 2/3 | **3/3** |
| B Reuse | L1 prefix | 32768（全命中） | 32768（全命中） | 32768（全命中） |
| C A-revisit | L1 prefix | **1024** | **0（全 miss）** | **1024** |
| C A-revisit | COW Hit | **3/3** | **0/3** | **3/3** |
| C B-revisit | L1 prefix | 32768 | 32768 | 32768 |
| C B-revisit | COW Hit | 3/3 | 3/3 | 3/3 |

### 5.6 累计结果

| 指标 | One Pool | TP KDA=80 | TP KDA=40 |
|------|----------|-----------|-----------|
| 总 COW Hit | **15/15** | 9/15 | 14/15 |
| 总 MLA New tokens | **1,545** | ~39,433 | ~5,641 |
| 总 KDA Replay tokens | **0** | ~34,816 | ~4,096 |
| **总计算工作量** | **1,545** | **74,249** | **9,737** |
| peak_mla_live | 283,648 | 200,000（固定） | 301,760（固定） |
| peak_mamba_live | 98 | 80（固定） | 40（固定） |
| flush | 5 | N/A | N/A |
| alloc_fail | 0 | N/A | N/A |

### 5.7 分析

**One Pool 动态调整路径**：
1. Phase A（KDA-heavy）：KDA 从 80 扩展到 98 slots（从 MLA 借空间），MLA 使用 ~98K tokens
2. Phase B（MLA-heavy）：MLA 从 ~98K 扩展到 283K tokens（从 KDA 借空间），5 次 flush
3. Phase C（跨阶段重访）：两阶段缓存均存活，15/15 COW Hit

**TP KDA=80 的 Phase C 失效**：
- Phase B 的 262,144 MLA tokens > MLA 容量 200,000 → MLA 驱逐 62,144 tokens
- `_release_all_component_layers` 级联释放 KDA Checkpoint
- Phase C A-revisit 3 个 sessions 全部 L1=0（MLA 全 miss），COW 全 miss
- 需完整重新 Prefill 3 × 1025 = 3,075 tokens

**TP KDA=40 的 Phase A 失效**：
- KDA=40 < 96 → 56 个 KDA Checkpoint 被淘汰
- Phase A Reuse 仅 2/6 COW Hit
- 但 MLA=301,760 > 262,144 → Phase B MLA 全部保留
- Phase C A-revisit 全 hit（PA Reuse 创建了新 KDA Checkpoint，在 LRU 中更近）

---

## 6. 总体结论

### 6.1 三组实验汇总

| 场景 | One Pool COW Hit | 最优固定分区 | 最优 COW Hit | One Pool vs 最优 |
|------|-----------------|-------------|-------------|-----------------|
| KDA-heavy | 6/6 | KDA=100 | 6/6 | 持平 |
| MLA-heavy | 3/3 | KDA=40 | 3/3 | 持平 |
| 动态混合 | 15/15 | KDA=40 | 14/15 | **工作量减少 84.1%** |

### 6.2 动态混合负载计算工作量对比

| 配置 | 总 COW Hit | 总计算工作量 | 相对 One Pool |
|------|-----------|-------------|--------------|
| **One Pool** | **15/15** | **1,545 tokens** | 1× |
| TP KDA=40 | 14/15 | 9,737 tokens | 6.3× |
| TP KDA=80 | 9/15 | 74,249 tokens | 48× |

### 6.3 核心发现

1. **One Pool 在单一负载下达到最优固定分区性能**：
   - KDA-heavy：6/6 COW Hit = KDA=100
   - MLA-heavy：3/3 COW Hit = KDA=40
   - 但无需预知最优分区

2. **One Pool 在动态混合负载下显著优于所有固定分区**：
   - 总工作量 1,545 tokens vs 最优固定分区 KDA=40 的 9,737（减少 84.1%）
   - 15/15 COW Hit vs KDA=40 的 14/15

3. **无固定分区适应所有阶段**：
   - KDA=80：Phase A 好（4/6 COW），Phase B 差（MLA 不足→级联→Phase C 全 miss）
   - KDA=40：Phase A 差（2/6 COW），Phase B/C 好（MLA 充足）
   - One Pool：全阶段最优（动态调整 MLA/KDA 边界）

4. **双向动态空间借用**：
   - KDA-heavy 阶段：KDA 从 MLA 借空间（80→97/98 slots）
   - MLA-heavy 阶段：MLA 从 KDA 借空间（200K→262K/283K tokens）
   - 5 次 flush 完成边界调整，0 次 alloc_fail，0 次 compact

5. **级联淘汰是固定分区的根本限制**：
   - UnifiedTreeNode 的 FULL 和 MAMBA 共享节点生命周期
   - MLA 驱逐 → `_release_all_component_layers` 同时释放 KDA
   - 固定分区无法避免此问题，One Pool 通过动态扩容避免触发驱逐

### 6.4 最终回答

**One Pool 能否在相同总内存预算下，通过双向动态空间借用适应变化的 MLA/KDA 工作集，并获得比固定分区更好的有效缓存复用和更少的计算工作量？**

**是。**

- **KDA-heavy**：One Pool 动态扩展 KDA 到 97 slots，6/6 COW Hit，等于最优固定分区 KDA=100
- **MLA-heavy**：One Pool 动态扩展 MLA 到 262K tokens，3/3 COW Hit，等于最优固定分区 KDA=40
- **动态混合**：One Pool 全程 15/15 COW Hit，总计算工作量 1,545 tokens，比最优固定分区减少 84.1%

One Pool 的核心优势在于：无需预知负载特征即可达到最优性能，且在负载变化时通过低成本的水位线调整（flush）实现双向空间借用，避免了固定分区在不适配负载下的级联淘汰惩罚。

---

## 附录：实验配置

### 环境变量

```bash
export SGLANG_USE_CPU_ENGINE=1
export ASCEND_RT_VISIBLE_DEVICES=2
export HISIM_SIMULATION_MODE=OFFLINE
export HISIM_DISABLE_PROACTIVE_EVICT=1
export SGLANG_MAMBA_TRACK_INTERVAL=<C>
export HISIM_MAMBA_POOL_SIZE=<KDA>       # Two Pools only
```

### 启动命令

```bash
launch_server_benchmark.sh <mode> <KDA> <MLA> <C> <chunk> <label>
# mode: twopools / onepool
# KDA: KDA slots (Two Pools) or max-mamba-cache-size (One Pool)
# MLA: max-total-tokens
# C: SGLANG_MAMBA_TRACK_INTERVAL
# chunk: 0 = not set, >0 = --chunked-prefill-size
```

### Trace 设计规则

- Build input_len = C（使 track_boundary = C）
- Reuse input_len = C + N（N > 0，使 `key_limit = C+N-1 ≥ C`，确保 full match）
- Reuse tokens = Build tokens + N new tokens（自然对话延续）
- 不修改 SGLang 原生 `_compute_max_prefix_len = input_len - 1` 规则
