# K3 Phase 2C.3.1: L3 Restore Correctness Closure

## 摘要

本轮工作定位并修复了 Phase 2C.3 中 L2=32、L3=On 时仍有 820447 Replay 的根因。
问题不在 Prefetch 逻辑或 Identity 不一致，而在 **`HISIM_DISABLE_L3` 环境变量的
Python 字符串 truthiness bug**：`"0"` 是非空字符串，被误判为 truthy，导致 L3
存储后端被禁用。修复后所有 L2 容量（0/3/32/128）均实现 Replay=0。

---

## 问题一：L2=32 为什么不能通过 L3 避免 Replay？

### 事件链追踪（B2-Fork, seed=42, L1=Hotness, L2=32）

```
Prefix Match (node 27)
  → KDA Checkpoint Identity: kda_27_0
  → L1 Miss: checkpoint 被 L1=Hotness cold-first 淘汰 ✓
  → L2 Miss: kda_27_0 被 L2=32 淘汰（L2 满，119 backups > 32 slots）✓
  → L3 Storage Lookup: ❌ BREAK — st is None
  → L3 Restore: 未发生 ✗
  → Forward 消费: Replay 820447 tokens ✗
```

### 断点：L3 Storage Lookup

`_mock_start_loading` 中的 `st`（storage backend）为 None。

**根因代码**（`sglang_hook.py` L1432，修复前）：

```python
if os.environ.get("HISIM_DISABLE_L3", ""):  # "0" 是非空字符串 → truthy!
    _storage = None                          # L3 被禁用！
```

`HISIM_DISABLE_L3=0` 在 Python 中 `"0"` 是 truthy（非空字符串），所以 L3 被
**错误禁用**。实验脚本设置 `HISIM_DISABLE_L3=0`（意为"L3 不禁用"），但代码
将其解释为"禁用 L3"。

### 日志证据

```
[LOAD-L2CHECK] hash=kda_27_0 l2_hit=False l2_cached=32 l2_cap=32 st_is_none=True st_type=None
[LOAD-RESULT] MAMBA restore FAILED — replay needed
```

`st_is_none=True` 确认 L3 存储后端不可用。

### 为何 L2=128 能工作

L2=128 容量足够大，所有 119 个原生 backup（`kda_{nid}_{i}` 格式）都保留在 L2 中，
不需要 L3。L2 lookup 直接命中 → restore OK → Replay=0。

---

## 问题二：问题出在 Storage、Lookup、Load-back 还是状态消费？

### 问题出在 Storage

| 环节 | 状态 | 说明 |
|------|------|------|
| Prefix Match | ✓ 正常 | node 27 正确匹配 |
| KDA Identity | ✓ 一致 | backup 和 restore 均用 `kda_{nid}_{i}` 格式 |
| L1 Miss | ✓ 正常 | L1=Hotness cold-first 淘汰 |
| L2 Miss | ✓ 正常 | L2=32 满，slot 被淘汰 |
| **L3 Storage** | **✗ 断点** | `st is None` — env var bug |
| L3 Restore | ✗ 未触发 | st 为 None |
| Forward 消费 | ✗ Replay | 状态未恢复 |

### Identity 一致性确认（非问题）

| 路径 | Hash 格式 | 代码位置 |
|------|-----------|----------|
| 原生 backup (`_mock_start_writing`) | `kda_{nid}_{i}` | L1498-1499 |
| 原生 restore (`_mock_start_loading`) | `kda_{nid}_{i}` | L1793-1794 |
| L3 write (`st.batch_set_mamba`) | `kda_{nid}_{i}` + evicted | L1614-1629 |
| L3 read (`st.batch_exists_mamba`) | `kda_{nid}_{i}` | L1802 |

所有路径使用相同的 `kda_{nid}_{i}` 格式，Identity 一致。

### 注意：HiSim proactive backup 使用不同格式

`handle_kda_backup_operation` 使用 `md5("kda_{req_id}_{token_offset}")` 格式。
这是一个**独立的路径**，不影响原生 restore。proactive backup 写入
`cls._kda_host_pool`（独立 L2 pool，固定 32 slots），而原生 restore 读取
`mh`（`self.mamba_pool_host`，由 `HISIM_L2_MAMBA_SIZE` 控制）。

---

## 修复

### 修复内容（`sglang_hook.py` L1432）

```python
# 修复前（bug）:
if os.environ.get("HISIM_DISABLE_L3", ""):     # "0" → truthy → 禁用 L3

# 修复后:
_l3_disable_val = os.environ.get("HISIM_DISABLE_L3", "").strip().lower()
if _l3_disable_val in ("1", "true", "yes", "on"):  # 只有明确禁用值才禁用
```

### 验证

```
L3 storage ENABLED (HISIM_DISABLE_L3='0') — st=MockHiCacheStorage
[LOAD-L2CHECK] hash=kda_27_0 l2_hit=False l2_cached=32 st_is_none=False st_type=MockHiCacheStorage
[LOAD-L3CHECK] hash=kda_27_0 l3_hit=1
[LOAD-RESULT] MAMBA restore OK — no replay needed
```

---

## 问题三：修复后 L2=32 能恢复多少 KDA Checkpoint？

### 容量扫描结果（B2-Fork, seed=42, L1=Hotness, native L2 policy）

| L2 Size | COW | Replay | L3_Wr(MLA) | L3_Hit(MLA) | L2_Cached | Load_Calls | Mamba_Xfer | Wr_Xfer |
|---------|-----|--------|------------|-------------|-----------|------------|------------|---------|
| 0 | 21 | **0** | 121 | 3 | 0 | 3 | 3 | 119 |
| 3 | 21 | **0** | 121 | 3 | 3 | 3 | 3 | 119 |
| 32 | 21 | **0** | 121 | 3 | 32 | 3 | 3 | 119 |
| 128 | 21 | **0** | 121 | 3 | 119 | 3 | 3 | 119 |

### 对照：L1=Native, L2=32

| COW | Replay | L3_Wr(MLA) | L3_Hit(MLA) | Load_Calls | Mamba_Xfer | Wr_Xfer |
|-----|--------|------------|-------------|------------|------------|---------|
| 21 | 0 | 128 | 0 | 0 | 0 | 128 |

### KDA L2/L3 restore 详情（从 LOAD-L2CHECK/LOAD-L3CHECK 日志）

| L2 Size | L2 hits | L2 misses | L3 checks | L3 hits | L3 restore |
|---------|---------|-----------|-----------|---------|------------|
| 0 | 0 | 6 | 6 | 6 | 6 |
| 3 | 0 | 6 | 6 | 6 | 6 |
| 32 | 4 | 2 | 2 | 2 | 2 |
| 128 | 6 | 0 | 0 | 0 | 0 |

### 分析

- **L2=0**: 所有 checkpoint 直接从 L3 恢复（6 次 L3 restore）
- **L2=3**: L2 太小，所有 lookup miss L2，从 L3 恢复（6 次 L3 restore）
- **L2=32**: 4 次 L2 hit + 2 次 L3 restore（node 27 和 19 被 L2 淘汰后从 L3 恢复）
- **L2=128**: 所有 checkpoint 在 L2 中（6 次 L2 hit，0 次 L3 restore）

### 两层策略贡献区分

| 配置 | Replay | L2 restore | L3 restore | 说明 |
|------|--------|------------|------------|------|
| L1=Native, L2=32 | 0 | 0 | 0 | L1 最优，无需 L2/L3 |
| L1=Hotness, L2=0 | 0 | 0 | 6 | L1 cascade 被 L3 完全捕获 |
| L1=Hotness, L2=32 | 0 | 4 | 2 | L2 减少了 4 次 L3 restore |
| L1=Hotness, L2=128 | 0 | 6 | 0 | L2 完全消除 L3 restore |

**结论**：L3 是消除 820K replay 的关键。L2 减少了 L3 restore 次数（从 6 到 0），
但不影响 Replay（L3 始终能捕获所有被淘汰的 checkpoint）。

---

## 问题四：L2 Native、Hotness、Cost-Aware 是否表现出真实差异？

### L2 策略 Hook 生效验证

| 验证项 | 结果 |
|--------|------|
| Policy 被设置 | ✓ `L2_Pol` 分别为 `native_l2`/`hotness_l2`/`cost_aware_l2` |
| `select_victim` 被调用 | ✓ HotnessL2: 89 次, CostAwareL2: 89 次（445 条候选日志）|
| Depth 被计算 | ✓ `KDA-DEPTH` 119 条，`node=18 depth=16384` 等 |
| 选择不同 victim | ✓ 见下表 |

### Victim 选择对比（L1=Hotness, L2=32）

| Policy | 前 3 个被淘汰的 slot | 淘汰依据 |
|--------|----------------------|----------|
| NativeL2 | kda_18_0, kda_19_0, kda_20_0 | LRU 顺序（最旧） |
| HotnessL2 | kda_18_0, kda_19_0, kda_20_0 | freq=0 退化为 LRU |
| CostAwareL2 | kda_33_0, kda_39_0, kda_45_0 | depth=8192 最浅（score=410 最低）|

**CostAwareL2 选择了不同的 victim！** 它淘汰最浅的 checkpoint（depth=8192），
而 NativeL2/HotnessL2 淘汰最旧的（node ID 顺序）。

### 但结果相同

| Policy | COW | Replay | L2 hits | L2 misses | L3 hits |
|--------|-----|--------|---------|-----------|---------|
| native | 21 | 0 | 4 | 2 | 2 |
| hotness | 21 | 0 | 4 | 2 | 2 |
| cost_aware | 21 | 0 | 4 | 2 | 2 |

### 为什么不同 victim 选择导致相同结果？

被重新访问的节点：
- **node 27**: depth=40896（较深）
- **node 19**: depth=32768（中等）

| Policy | node 27 被淘汰？ | 原因 |
|--------|------------------|------|
| NativeL2 | 是 | node 27 较旧（LRU） |
| HotnessL2 | 是 | freq=0（无访问历史）→ LRU |
| CostAwareL2 | 是 | depth=40896 < 40960（比 node 20/21/22 浅）|

三种策略都淘汰了 node 27 和 19，但原因不同。L3 安全网捕获了它们 → Replay=0。

### 差异不显现的三个原因

1. **无容量压力区分度**：被重新访问的节点（27, 19）在所有策略中都被淘汰
2. **L3 安全网**：L3 捕获所有被淘汰的 checkpoint，消除策略差异的影响
3. **访问历史为空**：HotnessL2 在首次访问前 freq=0，退化为 LRU

### 要看到策略差异需要的条件

需要被重新访问的节点在某种策略下被**保留**：
- **CostAwareL2**：重新访问的节点是最深的（depth=40960）→ 保留 → L2 hit
- **HotnessL2**：重新访问的节点有访问历史（freq>0）→ 保留 → L2 hit
- **NativeL2**：重新访问的节点是最近 backup 的 → 保留 → L2 hit

B2-Fork 不满足这些条件。

---

## 问题五：哪些成本需要进一步通过真实 NPU 数据标定？

| 成本项 | 当前模型 | 需要标定 | 说明 |
|--------|----------|----------|------|
| L2 backup D2H 延迟 | 14.45MB / 64GB/s × 0.85 | ✓ | 需真实 HBM→DRAM 带宽 |
| L2 load_back H2D 延迟 | 14.45MB / 64GB/s × 0.85 | ✓ | 需真实 DRAM→HBM 带宽 |
| L3 write (SSD) 延迟 | 14.45MB / 8GB/s | ✓ | 需真实 SSD 写带宽 |
| L3 read (SSD) 延迟 | 14.45MB / 14.6GB/s | ✓ | 需真实 SSD 读带宽 |
| COW (L1 D2D) 延迟 | 0（瞬时） | ✓ | 需真实 device-to-device copy 延迟 |
| Replay token 计算延迟 | 未建模 | ✓ | 需 Mamba forward 每 token 延迟 |
| CostAwareL2 score = prob × depth | prob=base_prob(0.05) | ✓ | 需真实 reuse probability |
| HotnessL2 window size | 20 | ✓ | 需真实访问频率分布 |

### 本轮未标定的原因

1. **CPU simulation 模式**：所有 NPU 操作都是 mock，不产生真实延迟
2. **NPU 设备 0-5 泄漏**：之前 11 次 SIGKILL 导致 HBM 泄漏，使用设备 7 绕过
3. **本轮目标**：验证 L3 restore correctness，不是性能标定

---

## 实验环境

### NPU Runtime 资源释放

| 项 | 状态 |
|----|------|
| SIGTERM 替代 SIGKILL | ✓ 已使用（`kill -TERM`）|
| NPU 设备 0-5 泄漏 | ⚠ 未修复（11.5GB HBM 泄漏）|
| 设备 7 绕过 | ✓ 临时恢复（`ASCEND_RT_VISIBLE_DEVICES=7`）|
| `aclFinalize` 调用 | ⚠ 未确认（SIGTERM 是否触发 graceful cleanup 待验证）|

### 设备切换说明

设备 7 只是暂时恢复实验的手段。HBM 泄漏的根因（SIGKILL 阻止 `aclFinalize`）
尚未完全修复。需要：
1. 主机重启清除泄漏的 HBM
2. 或 NPU 驱动重新加载（`rmmod/insmod`）
3. 长期：所有实验脚本使用 SIGTERM 确保 graceful shutdown

### 实验配置

```
ASCEND_RT_VISIBLE_DEVICES=7
HISIM_KDA_POLICY=hotness (L1)
HISIM_L2_MAMBA_SIZE={0,3,32,128}
HISIM_L2_POLICY={native,hotness,cost_aware}
HISIM_DISABLE_L3=0  (修复后正确启用 L3)
HISIM_MAMBA_POOL_SIZE=16
max_total_tokens=200000
max_running_requests=4
random_seed=42
predictor=k3_profile_based
```

---

## 修复代码变更

| 文件 | 变更 | Commit |
|------|------|--------|
| `sglang_hook.py` L1432 | env var truthiness bug 修复 | `7fb2e93` |
| `sglang_hook.py` L1498-1499 | 计算 depth 并传给 `backup_from_device` | `0cbac3e` |
| `l2_placement_policy.py` L107 | 添加 `[L2-VICTIM:hotness]` 诊断日志 | `0cbac3e` |

---

## 结论

1. **L2=32 不能通过 L3 避免 Replay 的根因**：`HISIM_DISABLE_L3=0` 的 Python 字符串
   truthiness bug 导致 L3 存储后端被禁用（`st is None`）。不是 Prefetch bug、
   不是 Identity 不一致、不是状态消费问题。

2. **问题出在 Storage**：L3 storage backend 为 None，L3 write 和 L3 read 都
   未发生。Identity 和 Lookup 逻辑本身是正确的。

3. **修复后 L2=32 恢复了 2 个 KDA Checkpoint**（node 27 和 node 19），
   从 L3 restore。L2=0 恢复 6 个，L2=128 恢复 0 个（全部 L2 hit）。

4. **L2 三种策略选择了不同的 victim**（CostAwareL2 选最浅，HotnessL2 退化为
   LRU），但结果相同，因为被重新访问的节点在所有策略中都被淘汰，L3 安全网
   捕获了它们。策略差异需要特殊 trace 才能显现。

5. **需要标定的成本**：L2/L3 带宽延迟、COW D2D 延迟、Replay token 计算延迟、
   CostAwareL2 的 reuse probability。这些需要真实 NPU 数据。
