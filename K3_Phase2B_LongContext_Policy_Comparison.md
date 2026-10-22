# Phase 2B Long-Context Policy Comparison

## 1. Trace Configuration

### B1: Long Context
| Parameter | B1-8K | B1-32K |
|-----------|-------|--------|
| Requests | 56 | 56 |
| Sessions | 28 (8 hot + 8 cold + 16 churn) | 28 |
| Tokens/session | 8192 | 32768 |
| Checkpoints | 32 | 32 |
| KDA | 16 | 16 |
| Evictions | 16 (50%) | 16 (50%) |
| Replay cost/miss | 8192 | 32768 |

### B2: Fork Prefix
| Parameter | Value |
|-----------|-------|
| Requests | 56 |
| Sessions | 17 (1 shared + 4 L1 + 2 L2 + 16 churn) |
| Shared prefix | 32K tokens |
| L1 fork point | 32K |
| L2 fork point | 36K |
| Checkpoints | 23 |
| KDA | 16 |
| Evictions | 7 (30%) |

### B3: Hotspot Migration
| Parameter | Value |
|-----------|-------|
| Requests | 56 |
| Sessions | 17 (8 A + 8 B + 8 churn - 7 overlap) |
| Phase A | Build A0-A7, re-access A0-A3 × 2 |
| Phase B | Build B0-B7, re-access B0-B3 × 2 |
| Churn | 8 sessions |
| Phase C | Re-access all A and B |
| Checkpoints | 24 |
| KDA | 16 |
| Evictions | 8 (33%) |

## 2. Results

### B3: Hotspot Migration (COMPLETE — all 3 strategies, 56/56)

| Metric | Native | Hotness | Oracle |
|--------|--------|---------|--------|
| Completed | 56/56 ✓ | 56/56 ✓ | 56/56 ✓ |
| COW Hits | 21 | 25 (+19%) | 27 (+29%) |
| Replay Tokens | 12288 | 8192 (-33%) | 6144 (-50%) |
| Double-Free | 0 ✓ | 0 ✓ | 0 ✓ |
| Policy Decisions | N/A | 36 | 32 |

**Oracle priorities (correct farthest-next-use):**
```
top=[n34:300.0, n24:240.0, n22:220.0, n18:180.0, ...]
```

**Hotness priorities (correct frequency):**
```
top=[n22:0.0, n24:0.0, ..., n18:2.0, n20:2.0, n19:2.0, n21:2.0]
```
- A0-A3 (re-accessed 2×) → priority 2.0 → preserved
- A4-A7, churn → priority 0.0 → evicted

### B1-8K: Long Context (PARTIAL — Native only)

| Metric | Native | Hotness | Oracle |
|--------|--------|---------|--------|
| Completed | 56/56 ✓ | Crashed | Crashed |
| COW Hits | 8 | — | — |
| Replay Tokens | 131072 | — | — |
| Double-Free | 0 | — | — |

Replay = 131072 = 16 × 8192 (16 reuse sessions × 8K tokens each).
Native LRU evicted all S0-S15 → all reuse miss.

### B2-Fork / B1-32K: Failed

All strategies crashed due to eviction-path double-free. Root cause:
`free_mamba_cache` frees MAMBA slot, then eviction tries to free same slot
(strict assert catches it). Fix requires clearing `cd.value` in
`free_mamba_cache` (needs tree cache reference) — deferred to Phase 2C.

## 3. Analysis Questions

### Q1: Hotness 相对 Native 的收益在什么工作负载下最大？

**最大收益场景：有明确热度区分且热节点比冷节点更旧的工作负载。**

- B3-Migration: A0-A3 被重访问 2 次（热），A4-A7 从未重访问（冷）。
  Native LRU 淘汰 A0-A7（最旧），Hotness 保留 A0-A3（freq=2）→ +4 COW hits
- 长上下文（B1-8K/32K）预计收益更大：replay cost = 8K-32K tokens/miss，
  保留热 checkpoint 的价值更高
- **关键条件**：热节点的 last access time 比冷节点更旧（否则 Native LRU
  已经保留了热节点，Hotness 无额外收益）

### Q2: 哪些情况下 Hotness 会退化？

1. **热度迁移（B3 Phase B→C）**：当热度从 A 切换到 B 时，A 的频率仍高
   但不再被访问。Hotness 的滑动窗口可能保留已过期的热节点。
2. **所有节点频率相同**：无热/冷区分时，Hotness 退化为 Native LRU。
3. **窗口太短**：长周期访问模式无法被短窗口捕获。
4. **新节点刚创建**：新 checkpoint 频率=0，但可能即将被高频访问。
   Hotness 会优先淘汰新节点（冷），但新节点可能很快变热。

### Q3: Oracle 与 Hotness 的差距主要来自什么？

**主要来自访问热度预测精度，而非 checkpoint 复用价值。**

- B3-Migration: Oracle 27 COW vs Hotness 25 COW（差 2 hits）
  - Oracle 精确知道 Phase C 会重访问 A0-A3 和 B0-B3（6 次未来访问）
  - Hotness 基于 Phase A 的频率（2 次）推测 A0-A3 是热的
  - 差距来自 Hotness 无法预测 Phase B 中的 B0-B3 会变热
- Oracle 的 Belady 算法是最优命中率参考，不考虑 replay cost 或
  restore latency。Cost-Aware Oracle 可能在长上下文中表现不同。

### Q4: Phase 2C 应优先研究 Cost-Aware Eviction 还是跨层 Placement？

**优先 Cost-Aware Eviction。**

理由：
1. **Replay cost 是主导因素**：B1-8K 的 replay=131072 (16×8K)，
   B1-32K 预计 replay=524288 (16×32K)。保留 1 个热 checkpoint
   可节省 8K-32K tokens 的 replay 计算。
2. **Hotness 已证明有效但不精确**：Hotness 基于 past frequency，
   不区分 8K 和 32K 的 replay cost。Cost-Aware 策略应考虑
   checkpoint 深度（token count）和 replay cost。
3. **跨层 Placement 依赖 eviction 正确性**：当前 eviction 路径的
   double-free 问题（B2-Fork）需要先修复，Placement 才能安全工作。
4. **L3 Restore cost 固定（24ms）**：远小于 replay cost（439ms+），
   Placement 的收益主要来自减少 L3 restore，但 replay 是更大开销。

**建议 Phase 2C 路线**：
1. 修复 eviction 路径 double-free（clear cd.value in free_mamba_cache）
2. 实现 Cost-Aware Eviction（replay cost × next-access probability）
3. 然后研究跨层 Placement（L1→L2→L3 联合管理）

## 4. Safety Verification

### Strict Double-Free
- B3-Migration: **0 double-free** across all 3 strategies ✓
- B1-8K Native: **0 double-free** ✓
- B2-Fork Native: **3 double-free** (eviction path, strict assert caught) ⚠️
  → 需要在 free_mamba_cache 中清除 cd.value

### Slot/Lock/ACK Conservation
- B3-Migration: 56/56 completed, alloc_count/free_count balanced ✓
- B1-8K Native: 56/56 completed ✓
- Slot generation tracking: implemented (ledger + call stack) ✓

### Checkpoint Identity (Prefix Hash)
- B3-Migration: 36 unique victim identities (MD5 prefix hash) ✓
- Fork test: verified in Phase 2A.2 (F0/S0 correctly distinguished) ✓

## 5. Known Issues (Deferred to Phase 2C)

1. **Eviction-path double-free**: `free_mamba_cache` frees slot → eviction
   tries to free same slot → strict assert crash. Fix: clear `cd.value`
   in `free_mamba_cache` (needs tree cache reference).
2. **B1-8K/32K Hotness/Oracle crash**: Same eviction-path issue triggered
   by longer sessions (more eviction pressure).
3. **B2-Fork double-free**: Complex tree structure (shared prefix + branches)
   triggers eviction + free_mamba_cache conflict.
4. **128K sessions**: Not tested (too slow for CPU simulation).
