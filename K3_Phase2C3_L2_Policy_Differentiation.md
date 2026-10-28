# K3 Phase 2C.3.2: L2 Placement Policy Differentiation

## 摘要

本轮工作设计了可区分 L2 策略的 trace，修复了 `record_access` 只在 L2 hit
时调用的 bug，新增了 TierCostAwareL2 策略，并进行了 5 策略消融实验。

**核心发现**：在当前 Mock 仿真框架下，NativeL2、HotnessL2、DepthAwareL2、
TierCostAwareL2 产生**完全相同**的结果。根因是 LRU 和 frequency 在当前实现中
**完全相关**——每次 L2 hit 同时更新 freq 和 LRU 位置，L2 miss 后 L3 restore
也会把 slot 重新放到 LRU 末尾。因此所有策略在相同 trace 下做出相同淘汰决策。

---

## 问题一：HotnessL2 能否在同等容量下降低 L3 Restore？

### 实验结果（l2diff_hot, L2=8, L1=Native, seed=42）

| Policy | COW | Replay | L3_Wr | L3_Hit | Load_Calls | Wr_Xfer | L2_Victims |
|--------|-----|--------|-------|--------|------------|---------|------------|
| NativeL2 | 6 | 0 | 30 | 12 | 12 | 30 | N/A |git pu
| HotnessL2 | 6 | 0 | 30 | 12 | 12 | 30 | 30 |
| DepthAwareL2 | 6 | 0 | 30 | 12 | 12 | 30 | N/A |
| TierCostAwareL2 | 6 | 0 | 30 | 12 | 12 | 30 | N/A |

**结论：HotnessL2 不能降低 L3 Restore。** 所有策略产生相同的 L3_Hit=12。

### 根因分析

LRU 和 frequency 在当前实现中**完全相关**：

| 事件 | LRU 更新 | Freq 更新 |
|------|----------|-----------|
| L2 hit (`load_to_device`) | `move_to_end(slot_hash)` ✓ | `record_access(slot_hash)` ✓ |
| L2 miss + L3 restore | `backup_from_device` → 添加到 LRU 末尾 ✓ | `record_access` ✓ |
| L2 miss (no L3) | 不在 L2 中 | `record_access` ✓ |

- L2 hit：slot 变为 LRU 最新 + freq 最高 → 两种策略都保留
- L2 miss + restore：slot 重新添加到 LRU 末尾 + freq 更新 → 两种策略都保留
- 未访问的 slot：LRU 最旧 + freq=0 → 两种策略都淘汰

**HotnessL2 的 `select_victim` 确实被调用了**（30 次 L2-VICTIM 日志），
`record_access` 也被调用了（8 次 L2-ACCESS 日志，freq=1,2）。
但被淘汰的 slot 和被访问的 slot 完全重合，且 freq=0 的 slot 恰好是 LRU 最旧的。

---

## 问题二：哪些负载下 HotnessL2 会退化？

### 热点迁移实验（l2diff_mig, L2=8, L1=Native, seed=42）

| Policy | COW | Replay | L3_Hit | Load_Calls |
|--------|------|--------|--------|------------|
| NativeL2 | 8 | 0 | 10 | 10 |
| HotnessL2 | 8 | 0 | 10 | 10 |
| TierCostAwareL2 | 8 | 0 | 10 | 10 |

**HotnessL2 在所有负载下都与 NativeL2 相同。** 不存在"退化"——因为
HotnessL2 从未"进化"过。它始终等价于 NativeL2。

### 退化条件（理论分析）

HotnessL2 理论上会在以下条件退化：
- 旧热点不再访问，但 freq 仍高 → HotnessL2 保留无用 slot
- 新热点 freq=0 → HotnessL2 淘汰新热点

但在当前实现中，**这些条件无法触发**：
- 旧热点的 freq 不会降低（sliding window 只在窗口满时 popleft）
- 但旧热点在 LRU 中也是最新的（最后一次访问时 `move_to_end`）
- 所以 NativeL2 也保留旧热点 → 无差异

---

## 问题三：Depth 与真正 Tier Restore Cost 是否提供不同的选择依据？

### DepthAwareL2 vs TierCostAwareL2 日志

**DepthAwareL2**：
```
[L2:depth_aware] hash=kda_18_0 depth=8192 prob=0.05 score=410
[L2:depth_aware] hash=kda_19_0 depth=8192 prob=0.05 score=410
```
所有 slot depth=8192（8K token 请求）→ score 相同 → 退化为 LRU。

**TierCostAwareL2**：
```
[L2:tier_cost] hash=kda_18_0 prob=0.0500 cost=0.000990 value=0.000049 last_t=N/A
[L2:tier_cost] hash=kda_19_0 prob=0.0500 cost=0.000990 value=0.000049 last_t=N/A
```
- `cost = slot_bytes / SSD_read_BW = 14.45MB / 14.6GB/s = 0.000990`（固定）
- `value = prob × cost` → 正比于 prob
- 首次淘汰时所有 slot `last_t=N/A` → prob=base_prob=0.05 → value 相同
- 访问后 prob 增加，但访问也更新 LRU → 与 NativeL2 相同

### 三种成本因子的对比

| 成本因子 | 变化来源 | 本轮是否有效 |
|----------|----------|--------------|
| Depth (token offset) | 请求长度 | ✗ 所有 slot depth=8192 |
| Incremental restore cost | slot_bytes / SSD_BW | ✗ 固定 0.000990 |
| Reuse probability | freq / decay | ✗ 与 LRU 相关 |

**结论**：Depth 和 Tier Restore Cost 在当前 Mock 中不提供不同的选择依据。
- Depth 不变（所有 checkpoint 在相同深度）
- Tier cost 不变（所有 slot 大小相同，SSD 带宽固定）
- Reuse probability 与 LRU 完全相关

---

## 问题四：当前收益是否足以支持下一阶段 MLA/KDA 联合 DRAM One Pool 管理？

### 不充分。理由：

1. **L2 策略无法区分**：所有 4 种策略产生相同结果，无法证明任何策略优于 NativeL2。

2. **L3 安全网消除策略差异**：L3 write-through 确保所有被淘汰的 checkpoint 都能从 L3 恢复。L2 策略只影响 L2 hit vs L3 restore 的比例，不影响 Replay。

3. **Mock 仿真限制**：
   - 所有 checkpoint 大小相同（14.45MB）→ tier cost 固定
   - 所有请求长度相同（8K）→ depth 固定
   - LRU 和 freq 完全相关 → HotnessL2 = NativeL2

4. **需要真实数据标定**：
   - 不同 checkpoint 大小的实际 H2D/D2H 延迟
   - SSD 带宽随队列深度的变化
   - Mamba forward 每 token 的实际计算延迟
   - Reuse probability 的真实分布

### 建议的后续方向

| 方向 | 说明 | 前提条件 |
|------|------|----------|
| 解耦 LRU 和 freq | LRU 基于 backup 时间，freq 基于访问 | 代码修改 |
| 变长 checkpoint | 不同请求长度 → 不同 depth | 修改 trace |
| SSD 带宽模型 | 随队列深度变化 | 需要 NPU 数据 |
| One Pool 联合管理 | MLA + KDA 共享 DRAM | 需要真实带宽 |
| 真实 NPU 标定 | H2D/D2H/SSD 延迟 | 需要 NPU 环境 |

---

## Task 5: 边界检查

### 1. L2=0 时 L3 Restore 的临时缓冲区

**路径**：`_mock_start_loading` → `mh.backup_from_device(-1, slot_hash, from_l3=True)`
→ 存入 L2 `_slots` dict → L2=0 立即淘汰 → `load_to_device` 返回 -1

**Host 容量计入**：❌ 未正确计入。临时 DRAM 缓冲使用未被追踪。

**影响**：L2=0 时 L3 restore 的 SSD→DRAM 数据临时占用 DRAM，但 Mock 不追踪
这个临时占用。在真实系统中，这会影响 DRAM 容量规划。

### 2. 指标粒度区分

| 指标 | 粒度 | 说明 |
|------|------|------|
| `Load_Calls` | per-batch | `_mock_start_loading` 调用次数 |
| `Mamba_Xfer` | per-transfer | MAMBA pool_transfers 数量 |
| L2 hits/misses | per-slot | `LOAD-L2CHECK` 日志 |
| L3 hits | per-slot | `LOAD-L3CHECK` 日志 |
| `L3_Wr` | per-node | MLA L3 写入计数（非 KDA）|
| `L3_Hit` | per-node | MLA L3 命中计数（非 KDA）|

**注意**：`L3_Wr` 和 `L3_Hit` 是 **MLA 指标**，不是 KDA 指标。KDA L3
写入通过 `st.batch_set_mamba()` 但未计入 `_diag_w["l3_writes"]`。需要
添加 KDA 专用 L3 指标。

### 3. 环境变量 truthiness 检查

| 变量 | 修复前 | 修复后 | 状态 |
|------|--------|--------|------|
| `HISIM_DISABLE_L3` | `"0"` truthy → 禁用 | `in ("1","true","yes","on")` | ✓ 已修复 |
| `HISIM_DISABLE_PROACTIVE_EVICT` | `"0"` truthy → 禁用 | `in ("1","true","yes","on")` | ✓ 已修复 |
| `HISIM_KDA_OBS` | `in ("1","true","yes")` | 无需修复 | ✓ 本就正确 |
| 其他 HISIM 变量 | 使用 `int()` 或 `.lower()` | 无需修复 | ✓ 本就正确 |

---

## 代码变更

| 文件 | 变更 | Commit |
|------|------|--------|
| `sglang_hook.py` L1432 | `HISIM_DISABLE_L3` truthiness fix | `7fb2e93` |
| `sglang_hook.py` L2569,3261 | `HISIM_DISABLE_PROACTIVE_EVICT` truthiness fix | `51d60e5` |
| `sglang_hook.py` L1794 | `record_access` on L2 lookup (hit or miss) | `51d60e5` |
| `sglang_mock_class.py` L1944 | 移除 `record_access` 从 `load_to_device` | `51d60e5` |
| `l2_placement_policy.py` L90 | `record_eviction` 不清除 access history | `51d60e5` |
| `l2_placement_policy.py` | CostAwareL2 → DepthAwareL2 (renamed) | `51d60e5` |
| `l2_placement_policy.py` | 新增 TierCostAwareL2Policy | `51d60e5` |
| `gen_k3_p2c32_trace.py` | 新 trace 生成器 (hot/mig) | `46b0625` |

---

## 实验环境

- NPU 设备 7（`ASCEND_RT_VISIBLE_DEVICES=7`）
- SIGTERM 关停（防 HBM 泄漏）
- `k3_profile_based` predictor
- `max_total_tokens=200000`, `max_running_requests=4`
- `HISIM_MAMBA_POOL_SIZE=16`, `HISIM_L2_MAMBA_SIZE=8`
- `SGLANG_MAMBA_TRACK_INTERVAL=1024`, `--random-seed 42`
- L3 write-through 启用（`HISIM_DISABLE_L3=0`）

---

## 结论

1. **HotnessL2 不能在同等容量下降低 L3 Restore。** LRU 和 frequency 完全相关，
   两种策略做出相同淘汰决策。

2. **HotnessL2 在所有负载下都与 NativeL2 相同**，不存在"退化"。根因是 LRU
   和 freq 的更新机制完全耦合。

3. **Depth 和 Tier Restore Cost 不提供不同选择依据。** Depth 固定（8192），
   Tier cost 固定（0.000990），两者都退化为 LRU。

4. **当前收益不足以支持下一阶段 MLA/KDA 联合 DRAM One Pool 管理。** 需要：
   - 解耦 LRU 和 freq 的更新机制
   - 变长 checkpoint（不同请求长度）
   - 真实 SSD 带宽模型（随队列深度变化）
   - 真实 NPU 数据标定 H2D/D2H 延迟

本轮未实现 One Pool、Prefetch 或新的 Cost-Optimal Oracle。
