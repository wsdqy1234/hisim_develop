# Phase 2B.1 Correctness & Comparison Closure

## Task 1: Neutral Policy Equivalence Test — ALL PASSED ✓

| Trace | Native COW | Neutral COW | Native R | Neutral R | Equivalence |
|-------|-----------|------------|---------|----------|-------------|
| B3-Mig (1K) | 21 | 21 | 12288 | 12288 | ✓ |
| B1-8K | 8 | 8 | 131072 | 131072 | ✓ |
| B2-Fork | 21 | 21 | 0 | 0 | ✓ |
| B1-32K | 5 | 5 | 1359632 | 1359632 | ✓ |

**结论：Pre-sort Hook 在无评分时（Neutral）完全等价于 Native LRU。**
Hook 本身不改变淘汰顺序、不引入额外 eviction、不改变 write-back 行为。

## Task 2: B2-Fork 820447 Replay 定位

| Strategy | COW | Replay | Root Cause |
|----------|-----|--------|------------|
| Native | 21 | 0 | 基线 |
| Neutral | 21 | 0 | = Native（Hook 等价）|
| Hotness | 21 | 820447 | **频率重排导致不同 victim** |
| Oracle | 21 | 0 | = Native（无改善需要）|

**根因确认：B2-Fork Hotness Replay=820447 来自 Hotness 的频率评分重排，不是 Hook Bug。**

Hotness 在 Fork 场景中：
- 优先保留高频访问的**共享 prefix checkpoint**（32K tokens，被多个 session 访问）
- 牺牲低频但**高 replay cost 的分支专属 checkpoint**（40K tokens）
- 不同 victim 导致部分 reuse session miss → 累积 replay
- 这是**合法的策略退化**，反映了频率启发式在 Fork 场景中的局限

## Task 3: 补齐公平对照

### B3-Mig Native（重跑，解决端口冲突）
- COW=21, Replay=12288 ✓（`_drain_device_frees` 修复后正确）

### B1-8K Oracle（identity 修复后重跑）
| | 修复前 | 修复后 |
|---|--------|--------|
| COW | 8 (= Native) | **16** (= Hotness) |
| Replay | 131072 | **65536** |

修复内容：`load_future_map` 从 `ids[:1024]` 改为 `ids[:input_length]`（build）/ `ids[:input_length-1]`（re-access），使 hash 口径与 node prefix 一致。

## 完整结果总表（修复后，所有实验 56/56, DF=0）

### B3-Migration (1K tokens, 24 checkpoints, 8 evictions)

| Strategy | COW | Replay | vs Native |
|----------|-----|--------|-----------|
| Native | 21 | 12288 | — |
| Neutral | 21 | 12288 | = Native ✓ |
| Hotness | 25 | 8192 | +19% COW, -33% Replay |
| Oracle | 27 | 6144 | +29% COW, -50% Replay |

### B1-8K (8K tokens, 32 checkpoints, 16 evictions)

| Strategy | COW | Replay | vs Native |
|----------|-----|--------|-----------|
| Native | 8 | 131072 | — |
| Neutral | 8 | 131072 | = Native ✓ |
| Hotness | 16 | 65536 | +100% COW, -50% Replay |
| Oracle | 16 | 65536 | +100% COW, = Hotness |

### B2-Fork (32K+8K tokens, 23 checkpoints, 7 evictions)

| Strategy | COW | Replay | vs Native |
|----------|-----|--------|-----------|
| Native | 21 | 0 | — |
| Neutral | 21 | 0 | = Native ✓ |
| Hotness | 21 | 820447 | 0% COW, +∞ Replay (退化) |
| Oracle | 21 | 0 | = Native |

### B1-32K (32K tokens, 32 checkpoints, 16 evictions)

| Strategy | COW | Replay | vs Native |
|----------|-----|--------|-----------|
| Native | 5 | 1359632 | — |
| Neutral | 5 | 1359632 | = Native ✓ |
| Hotness | 5 | 1359632 | = Native (无效果) |
| Oracle | 7 | 1277712 | +40% COW, -6% Replay |

## Task 4: 32K Hotness 失效原因

B1-8K 和 B1-32K 都有 32 checkpoints、16 evictions，但 Hotness 在 8K 有效（+100%）、32K 无效（0%）。

**原因分析：**
- B1-8K: replay = 131072 / 8192 = **16 次完整 replay**（16 reuse × 1 次 each）→ 无 thrashing
- B1-32K: replay = 1359632 / 32768 = **41.5 次完整 replay**（16 reuse × 2.6 次 each）→ **thrashing**

**Thrashing 机制：**
1. Hotness 在第一轮 eviction 中保留热 checkpoint（freq=1）
2. 但 KDA=16 太小，后续 eviction 轮次仍需淘汰
3. 热 checkpoint 在后续轮次被淘汰 → replay（32K tokens）
4. replay 后的新 checkpoint 占用 slot → 触发更多 eviction
5. 循环往复 → 累积 replay = 41.5 × 32K = 1.36M

**不是算法预测不足**（Hotness 正确识别了热节点 priority=1.0），**是容量不足导致的 thrashing**。8K session 的 replay cost 较低（8K），单轮淘汰不会触发级联；32K session 的 replay cost 高（32K），单轮淘汰触发级联。

## Task 5: 下一阶段策略设计依据

### Q1: 是否存在不同 Checkpoint 的明显 Replay Cost 差异？

**是。** Checkpoint 深度 = session token 数：
- B3-Mig: 1K tokens/miss
- B1-8K: 8K tokens/miss
- B1-32K: 32K tokens/miss
- B2-Fork 共享: 32K tokens/miss，分支: 40K tokens/miss

8K 和 32K 差 4 倍，保留 32K checkpoint 的价值远高于 8K。

### Q2: Fork 共享 Checkpoint 是否比单 Session Checkpoint 更值得保护？

**不一定。**
- 共享 checkpoint（32K prefix）被多个 session 访问 → 频率高 → **LRU 自然保留**
- 分支专属 checkpoint（8K-40K suffix）被单个 session 访问 → 频率低 → **容易被淘汰**
- 但分支 checkpoint 的 replay cost 可能更高（40K tokens）
- **正确策略应考虑 replay cost × 访问概率，不只看频率**

### Q3: 热点迁移导致的历史热度失效有多严重？

**中度严重。**
- B1-8K（无迁移）：Hotness +100% COW
- B3-Migration：Hotness +19% COW（收益降至 1/5）
- 原因：Phase B 中 B0-B3 变热，A0-A3 历史频率仍高但不再被访问
- Hotness 的滑动窗口无法区分"历史热"和"当前热"
- 但 Hotness 仍优于 Native（+19% > 0%）

### Q4: Cost-Aware Eviction 是否有明确的潜在收益？

**是，且收益明确。**
- B1-8K: Hotness 保留 8 个 8K checkpoint → 节省 65536 replay tokens
- B1-32K: Hotness 无法防止 thrashing → Cost-Aware 应优先保留 32K checkpoint
- B2-Fork: Hotness 退化 → Cost-Aware 应优先保留 40K 分支 checkpoint 而非 32K 共享
- **Cost-Aware 公式：priority = replay_cost (token_depth) × next_access_probability**

### Q5: 多层级 Placement 是否比进一步优化 L1 Eviction 更值得优先研究？

**L1 Eviction 优先，Placement 其次。**
- L1 Eviction 已验证有效（B1-8K +100% COW）
- B1-32K 的 thrashing 可通过 Placement（L2/L3 缓存被淘汰 checkpoint）缓解
- L3 restore cost (24ms) << replay cost (439ms+) → Placement 有明确收益
- 但 Placement 依赖 L1 eviction 正确性（已验证）
- **建议：Cost-Aware L1 Eviction → Placement → One Pool 联合管理**
