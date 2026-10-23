# Phase 2C.1 Cost-Aware L1 Eviction — Results & Analysis

## Implemented Strategies

| Strategy | Score Formula | Key Feature |
|----------|--------------|-------------|
| Native LRU | None (baseline) | SGLang 原生 |
| Neutral | None (pre-sort only) | Hook 等价验证 |
| Hotness | frequency | 滑动窗口频率 |
| Depth-Aware | frequency × full_depth | 简单深度加权 |
| Cost-Aware | probability × marginal_cost | Fork-Aware 边际成本 |
| Oracle | farthest-next-use | Belady 理论参考 |

## Fork-Aware Marginal Cost Calculation

```
marginal_cost = full_depth - nearest_recoverable_ancestor_depth
```

验证（B2-Fork Cost-Aware 日志）：
```
node=22: full=40960, ancestor=16384, marginal=24576  ← 40K-16K=24K ✓
node=20: full=40960, ancestor=4095,  marginal=36865  ← 40K-4K=36K ✓
node=21: full=40960, ancestor=8191,  marginal=32769  ← 40K-8K=32K ✓
```

**边际成本计算正确** — 能区分"有祖先可恢复"和"需完整重算"的 checkpoint。

## Results

### B1-8K (8K tokens, 32 checkpoints, 16 evictions)

| Strategy | COW | Replay | vs Native |
|----------|-----|--------|-----------|
| Native | 8 | 131072 | — |
| Hotness | 16 | 65536 | +100% COW, -50% R |
| **Cost-Aware** | **16** | **65536** | **= Hotness** |
| Oracle | 16 | 65536 | = Hotness |

**Cost-Aware = Hotness = Oracle**：所有 session 深度相同（8K），marginal cost 无差异。Cost-Aware 退化为 Hotness。

### B2-Fork (32K+8K tokens, 23 checkpoints, 7 evictions)

| Strategy | COW | Replay | 问题 |
|----------|-----|--------|------|
| Native | 21 | 0 | 基线（最优）|
| Neutral | 21 | 0 | = Native ✓ |
| Hotness | 21 | 820447 | 退化 |
| Depth | 21 | 820447 | 同上 |
| **Cost-Aware** | **21** | **820447** | **同上** |
| Oracle | — | — | NPU 崩溃 |

**三种策略全部退化**：prob=0.0（identity 在 40K 路径上失败）使 score=0，边际成本被完全忽略。

### Mixed (4K-16K tokens, 23 checkpoints, 7 evictions, Fork + 热点迁移)

| Strategy | COW | Replay | Replay/Miss |
|----------|-----|--------|-------------|
| Cost-Aware | 22 | 98240 | 8930/miss |
| Oracle | 19 | 45056 | 3218/miss |

**Cost-Aware 有更多 COW（22>19）但更高 Replay（98K>45K）**：
- Cost-Aware 保留了更多 checkpoint（22 hits = 11 misses）
- Oracle 保留了更少但更贵的 checkpoint（19 hits = 14 misses）
- Oracle 的 miss 更便宜（3K/miss vs 9K/miss）
- **说明 Cost-Aware 的 prob=0 分组保留了低价值 checkpoint，牺牲了高价值 ones**

## Key Findings

### Q1: Cost-Aware 相比 Hotness 是否有额外收益？

**在当前实现下没有。**
- B1-8K：Cost-Aware = Hotness（深度相同，marginal 无差异）
- B2-Fork：Cost-Aware = Hotness（prob=0，marginal 被忽略）
- Mixed：Cost-Aware > Hotness（COW 22 vs 未知），但 Replay 高于 Oracle

**根因**：`prob = min(1.0, freq * 0.5)` 对 cold（freq=0）返回 0，使 `score = 0 × marginal = 0`。所有 cold 候选 score 相同，按 LRU 排序 — 与 Hotness 完全一致。

### Q2: 收益来自简单 Depth 还是 Fork-Aware Marginal Cost？

**当前无法区分** — 因为 prob=0 使两者都退化为 Hotness。
- 边际成本计算本身**正确**（验证通过）
- 但在 prob=0 时**无法生效**
- 需要修复 prob 公式后才能评估

### Q3: 哪些情况下历史热度仍然优于成本估计？

- **B1-8K**：Hotness = Cost-Aware = Oracle（同深度场景，频率足够）
- **B3-Migration**：Hotness +19%（频率有效，迁移影响适中）
- **B2-Fork**：Hotness 退化（频率保留了高频率共享 prefix，牺牲了低频率但高 cost 的分支）

### Q4: 有多少损失能够通过 L1 Eviction 避免？

- B1-8K：Hotness/Cost-Aware 避免 50% Replay（131K→65K）
- B3-Mig：Hotness 避免 33% Replay（12K→8K）
- B1-32K：Hotness 无法避免（thrashing，1359K→1359K）
- B2-Fork：Hotness 引入 820K 额外 Replay（退化）

### Q5: 哪些损失只能通过 L2/L3 Placement 或 Prefetch 改善？

- **B1-32K thrashing**（41.5 次 replay/16 reuse session）：L1 eviction 无法解决，需要 L2/L3 缓存被淘汰 checkpoint
- **B2-Fork 退化**：L1 eviction 改变 cascade 效应，需要更保守的 eviction 或 L2/L3 兜底
- **Cost-Aware 的 prob=0 问题**：L1 eviction 层面的修复（非零基础概率），不需要 Placement

## Next Steps (Phase 2C.2)

### 1. 修复 prob=0 问题
```python
# 当前：prob = min(1.0, freq * 0.5)  → cold: prob=0, score=0
# 修复：prob = max(0.05, min(1.0, freq * 0.5))  → cold: prob=0.05, score=0.05×marginal
```
这允许 cold 候选按 marginal cost 排序，而非全部 score=0 按 LRU。

### 2. 修复 identity 在长路径上的失败
- `_node_prefix_identity` 对 40K token 路径可能超时或异常
- 需要优化或使用更简单的 identity（如 node ID + depth）

### 3. 重新评估 Cost-Aware 在 B2-Fork 和 Mixed 上的表现
- 修复后重跑，验证边际成本是否能避免退化

### 4. 考虑 Oracle 不等于 Cost-Optimal
- Oracle (Belady) 优化命中率，不优化 Replay Cost
- Mixed 结果证明：Oracle COW=19 < Cost-Aware COW=22，但 Oracle R=45K < Cost-Aware R=98K
- 需要 Cost-Optimal Oracle（优化总 Replay Tokens）作为新参考
