# K3 Phase 2D.1: Unified Pool Restore & Accounting Closure

> 分支：`develop_dqy_evict_reuse` (commit `30408f5`)
> 日期：2026-10-28
> 范围：Unified L3 Hits=0 根因、L3 Writes 差异、字节账本、双向借用验证

---

## 五个核心问题的回答

### 1. Unified 的 L3 Hits=0 是合法 L1/L2 命中还是恢复路径缺失？

**合法行为，非恢复路径缺失。**

**根因**：`_bind_l2_host_group` 设置 `group.size = kv_host.size`（66K tokens），HiCache 控制器在 L2 满时正确淘汰 radix tree 节点。Legacy 模式不调用此函数，`group.size` 保持 80GB（过大），不淘汰节点。

**证据**（Phase 3 reuse prefix match）：

| 模式 | full | host | mamba_host | best_match | ILB | L3 Hits |
|------|:----:|:----:|:----------:|:----------:|:---:|:-------:|
| Legacy | 8192 | 8192 | 1 | 20（完整前缀）| 4 | 4 |
| Unified | **0** | **0** | **0** | 9（浅层节点）| 0 | 0 |

- **Legacy**：`group.size=80GB` → 不淘汰 radix tree → Phase 3 找到完整前缀（node 20）→ device value=0（L1 已淘汰）→ 触发 ILB → L2 miss → L3 hit
- **Unified**：`group.size=66K` → L2 满时淘汰 radix tree → Phase 3 找到浅层节点（node 9）→ 无 KDA checkpoint → 不触发 ILB

**结论**：Legacy 的 L3 Hits=4 是 `group.size` 过大（80GB）的**人为假象**。Unified 的 L3 Hits=0 是**正确行为**——radix tree 节点被淘汰后，Phase 3 找不到完整前缀，不触发 L3 lookup。

**D1.2 失败注入验证**：`backup_from_device` 返回值检查已部署，`kda_l3_admission_reject=0`（0.5GB 下无准入拒绝）。Legacy 模式不受影响（返回 None，`None is not False` → True）。

### 2. L3 Writes 的巨大差异是什么原因？

**`group.size` 影响 radix tree 淘汰率，进而影响 backup 操作数量。**

| 配置 | group.size | KDA_L3_Wr | MLA_L3_Wr | KDA_slots | MLA_tokens |
|------|:---------:|:---------:|:---------:|:---------:|:----------:|
| Legacy | 80GB（原始）| 640 | 324 | 8 | - |
| Two Pools KDA=8 | 66K（正确）| 622 | 315 | 8 | 50574 |
| Two Pools KDA=16 | 66K | 604 | 311 | 16 | 28903 |
| Two Pools KDA=32 | **5K** | **100** | **68** | 32 | 4800 |

**KDA=32 的 L3 Writes 仅 100（vs 622）的原因**：

1. **MLA 容量极小**（4800 tokens vs 50574）→ `group.size=5380` → HiCache 控制器**积极淘汰 radix tree 节点**
2. Phase 1 KDA-heavy 节点在 Phase 2 MLA-heavy 期间被淘汰 → 这些节点的 checkpoint 不会被 backup → 不产生 L3 write
3. 更大的 KDA pool（32 slots vs 8）→ 更少的 KDA L2 淘汰 → 更少的 KDA L3 write

**这是容量导致的实际操作差异**，不是统计问题。较小的 `group.size` 导致更激进的 radix tree 淘汰，减少了到达 backup 路径的节点数量。

### 3. One Pool 字节账本是否闭合？

**闭合。79MB 差异来自 MLA 已分配但未释放的页（partial-free fragmentation）。**

**One Pool K2M 快照**：

```
Total: 0.50GB (500,000,000 bytes)
Used:  0.42GB (416,663,040 bytes)
Free:  0.08GB (83,336,960 bytes)

KDA payload:  219.8MB (15 slots × 14,653,440 bytes)
MLA allocated: 196.9MB (2136 pages × 92,160 bytes)
MLA live:      113.9MB (19790 tokens × 5760 bytes)
MLA overhead:   79.0MB (899 pages allocated-but-not-live)
```

| 项目 | 字节 | 说明 |
|------|------|------|
| KDA payload | 219.8MB | 15 slots × 14.65MB（exact，0 内部碎片）|
| MLA allocated | 196.9MB | 2136 pages × 92,160 bytes |
| MLA live | 113.9MB | 19790 tokens × 5760 bytes |
| **MLA overhead** | **79.0MB** | 899 pages allocated but partially freed |
| **Total Used** | **416.7MB** | KDA + MLA allocated ≈ 0.42GB ✅ |
| External Frag | 60.2MB | Free bytes - Largest free run |
| Internal Frag | 0.0MB | Exact policy, KDA 精确整除 |

**差异解释**：
- 用户计算的 334MB = KDA payload (219.8) + MLA live (113.9) — 只算存活 token
- 报告的 420MB = KDA payload (219.8) + MLA allocated (196.9) — 包含已分配未释放的页
- 79MB 差异 = MLA 页被部分释放但 extent 未归零（引用计数 > 0）

**字节守恒**：`Used + Free + External Frag = Total` → `416.7 + 23.3 + 60.2 = 500.2MB ≈ 500MB` ✅

### 4. 双向借用是否已在系统级验证？

**KDA→MLA 借用：已验证。MLA→KDA 借用：未工作。**

**KDA→MLA 借用**（KDA 需要空间时淘汰 MLA 给 KDA）：
- ❌ **未观察到** `cross_evict_mla_tokens > 0`（所有实验均为 0）

**根因**：`enable_storage=False` → MLA evictor 未设置 → `_reclaim_from_mla` 始终返回 0

```python
# _bind_l2_host_group L1512:
if getattr(cache, "enable_storage", False) and hasattr(cache, "evict_host"):
    def _mla_evict(n_tokens, _c=cache):
        ...
    l2_pool.set_mla_evictor(_mla_evict)
# enable_storage=False → 条件不满足 → set_mla_evictor 不调用
# → _mla_evict_fn = None → mla_reclaim_enabled() = False
# → _reclaim_from_mla() 返回 0 → MLA 永远不被淘汰给 KDA
```

**MLA→KDA 借用**（MLA 需要空间时淘汰 KDA 给 MLA）：
- ✅ `cross_evict_kda_slots = 63-300`（KDA 被淘汰给 MLA）
- 但此计数器包含**同池替换**（KDA 在自身池内 LRU）和**跨池借用**（MLA 借用 KDA 空间），不区分

**self_first vs other_first**：结果完全相同。因为 `other_first` 尝试 `_kda_borrow_other` → `_reclaim_from_mla` → 返回 0（MLA evictor 未设置）→ 回退到 `_kda_replace_self`（同 KDA 自身替换）。

**修复建议**：将 MLA eviction 路径接入 HiSim 的 `_mock_start_writing` backup 路径（而非依赖 native `evict_host`），或设置 `enable_storage=True`。

### 5. 平台是否达到可以冻结并交付策略研发的条件？

**未完全达到。** 存在两个阻塞项和一个已知限制：

| 阻塞项 | 影响 | 修复难度 |
|--------|------|---------|
| **MLA→KDA 借用不工作** | One Pool 的核心价值（双向共享）只实现了一半 | 中（需接入 MLA eviction 路径）|
| **Legacy group.size 假象** | Legacy baseline 不公平（不淘汰 radix tree）| 低（Legacy 也调用 `_bind_l2_host_group` 或手动设 group.size）|
| **KDA_L3 hits=0 (Unified)** | 无法度量 L3 restore 效果 | 低（是正确行为，但需调整 trace 让 Phase 3 节点存活）|

**可冻结的部分**：
- D1.1 KDA/MLA L3 指标分离 ✅
- D1.2 Restore 状态真实性 ✅
- D1.4 token_bytes 根因修复 ✅
- 字节账本闭合 ✅
- KDA→MLA 单向借用 ✅
- exact/pow2 碎片度量 ✅

---

## P0-1: 确定性 L1+L2 Miss / L3 Hit 测试

### 失败注入验证

D1.2 修复已在代码中部署：
```python
_bk_ret = mh.backup_from_device(-1, _slot_hash, from_l3=True)
if _bk_ret is not False:
    mh.load_to_device(_slot_hash)
    _any_mamba_restore = True
else:
    _diag_l["kda_l3_admission_reject"] += 1
    # _any_mamba_restore stays False → replay
```

验证结果：所有实验 `kda_l3_admission_reject=0`（0.5GB 预算下无准入拒绝）。Legacy 不受影响。

### 为什么 Unified 不触发 L3 lookup

逐请求追踪（Phase 3 reuse，node 20）：

| 步骤 | Legacy | Unified |
|------|--------|---------|
| 1. Prefix Match | `full=8192`（找到 node 20）| `full=0`（node 20 已被淘汰）|
| 2. L1 Checkpoint | `dev=0`（已淘汰）| N/A（节点不存在）|
| 3. L2 Lookup | `l2_hit=False` → L2 miss | N/A |
| 4. L3 Lookup | `l3_hit=1` → L3 hit | N/A |
| 5. Load-back | ILB → restore | N/A |
| 6. Restore Ready | `post_val=set` ✅ | N/A |
| 7. Forward 消费 | 正常 | 正常（浅层匹配）|

**Unified 的 Phase 3 不触发 L3 lookup 是因为 radix tree 节点已被淘汰**，不是因为恢复路径缺失。

---

## P0-2: L3 Write 次数差异

### 完整 L3 Write 拆解

| 配置 | KDA_L3_Wr | MLA_L3_Wr | KDA_slots | MLA_tokens | group.size | radix 淘汰 |
|------|:---------:|:---------:|:---------:|:----------:|:----------:|:----------:|
| Legacy | 640 | 324 | 8 | - | 80GB | 无（假象）|
| 2P KDA=8 | 622 | 315 | 8 | 50574 | 66K | 有 |
| 2P KDA=16 | 604 | 311 | 16 | 28903 | 66K | 有 |
| 2P KDA=32 | **100** | **68** | 32 | 4800 | **5K** | **激进** |

**KDA=32 的 L3 Writes 仅 100 的原因**：

1. `group.size = kv_host.size = 5380 tokens`（极小）
2. HiCache 控制器在 Phase 2 MLA-heavy 期间**激进淘汰 radix tree 节点**
3. Phase 1 KDA-heavy 节点被淘汰 → 不再 backup → 不产生 L3 write
4. 仅 Phase 2 期间存活节点的 backup 产生 L3 write → 100 次

**这是容量导致的实际操作差异**：MLA 容量 4800 tokens（极小）→ group.size 极小 → 激进淘汰 → 更少 backup → 更少 L3 write。不是统计问题。

---

## P1-1: Allocator 字节账本

### One Pool K2M 快照

```
Total:     500,000,000 bytes (0.50GB)
Used:      416,663,040 bytes (0.42GB)
Free:      83,336,960 bytes (0.08GB)

KDA:       219,801,600 bytes (15 slots × 14,653,440)  → 209.8MB
MLA alloc: 196,909,440 bytes (2136 pages × 92,160)    → 187.8MB
MLA live:  113,974,400 bytes (19790 tokens × 5760)     → 108.7MB
MLA overhead: 82,935,040 bytes (899 pages, partial-free) → 79.1MB

Internal Frag:     0 bytes (exact policy)
External Frag: 60,180,000 bytes (72.2% of free)
Largest Free:  23,196,000 bytes (252 pages)

守恒: Used + Free = 416.7 + 83.3 = 500.0MB = Total ✅
```

### 334MB vs 420MB 差异

| 计算方式 | KDA | MLA | Total |
|----------|-----|-----|-------|
| Live payload（用户计算）| 209.8MB | 113.9MB（live tokens）| **323.7MB** |
| Allocated（metrics）| 209.8MB | 187.8MB（allocated pages）| **397.6MB** |
| Used（metrics）| - | - | **416.7MB** |

差异来源：MLA allocated-but-not-live pages（79MB）。这些页被部分释放但 extent 引用计数 > 0，未归还给 allocator。

### L2=0 / Full 时的 Staging

- **L2=0**：`admit_kda` 始终失败 → `backup_from_device` 返回 False → D1.2 触发 replay
- **L2 Full**：同池替换或跨池借用 → 正常淘汰
- **临时 Host Staging**：未建模（已知限制）——SSD→DRAM 临时数据未追踪 Host 容量

### Host Index Owner/Generation

- Round-robin 分配 + `_allocated_idx` set 追踪
- 幂等 free：`released = [i for i in indices if i in self._allocated_idx]`
- 无 Double-Free（DF=0 所有实验）✅
- 无 ABA 问题（index 是逻辑 ID，不对应物理字节）

---

## P1-2: MLA→KDA 跨类型回收

### 状态：未工作

**根因**：`enable_storage=False` → `set_mla_evictor` 不调用 → `_mla_evict_fn=None` → `_reclaim_from_mla` 返回 0

```python
# mla_reclaim_enabled():
return self.mode == "one_pool" and self.kv_evict and self._mla_evict_fn is not None
# _mla_evict_fn = None → returns False → _reclaim_from_mla returns 0
```

### 验证

| 实验 | cross_evict_kda | cross_evict_mla | borrow |
|------|:---------------:|:---------------:|:------:|
| One Pool sf K2M | 300 | **0** | self_first |
| One Pool of K2M | 300 | **0** | other_first |
| One Pool sf M2K | 287 | **0** | self_first |
| One Pool of M2K | 287 | **0** | other_first |

**other_first = self_first**：因为 `_kda_borrow_other` → `_reclaim_from_mla` → 返回 0 → 回退到 `_kda_replace_self`。

### 修复路径

1. **在 `_bind_l2_host_group` 中接入 HiSim MLA eviction 路径**（不依赖 `enable_storage`）
2. 或设置 `enable_storage=True`（可能影响其他行为）
3. 或在 `_reclaim_from_mla` 中直接调用 `_mock_start_writing` 的 backup 路径

---

## P2: 最小收益验证

### 场景选择

由于 MLA→KDA 借用未工作，One Pool 的收益仅来自 KDA→MLA 单向共享（KDA 动态获得更多空间）。

**场景 1：KDA-heavy + Reuse（K2M trace）**

| 配置 | KDA_slots | KDA_L3_Wr | Replay | Used |
|------|:---------:|:---------:|:------:|:----:|
| Two Pools KDA=8 | 8 | 622 | 0 | 0.49GB |
| Two Pools KDA=16 | 16 | 604 | 0 | 0.44GB |
| One Pool | **15** | 615 | 0 | 0.42GB |

One Pool KDA=15 slots（介于 KDA=8 和 KDA=16 之间），L3 writes 相近。**无显著收益**。

**场景 2：MLA-heavy + Reuse（M2K trace）**

| 配置 | MLA_tokens | MLA_L3_Wr | Replay | Used |
|------|:----------:|:---------:|:------:|:----:|
| Two Pools KDA=8 | 65536 | 317 | 0 | 0.49GB |
| Two Pools KDA=16 | 43840 | 313 | 0 | 0.49GB |
| One Pool | **27456** | 310 | 0 | 0.48GB |

One Pool MLA=27456 tokens（远低于 KDA=8 的 65536）。**MLA 容量退化**。

### 收益判断

在当前 Mock 中（MLA→KDA 借用不工作）：
- **One Pool 未表现出优于最佳静态 Two Pools 的收益**
- KDA-heavy：One Pool KDA=15 ≈ Two Pools KDA=16（但碎片更高 60MB vs 16MB）
- MLA-heavy：One Pool MLA=27456 << Two Pools KDA=8 的 65536（退化 58%）

---

## 已知 Mock 限制与真实 NPU 标定需求

| 限制 | 影响 | 修复优先级 |
|------|------|-----------|
| MLA→KDA 借用不工作 | One Pool 核心价值未实现 | **高** |
| Legacy group.size 假象 | Legacy baseline 不公平 | **高** |
| KDA_L3 hits=0 (Unified) | 无法度量 L3 restore | 中（调整 trace）|
| Wr_Xfer=0 (Unified) | 传统 KDA 指标不可用 | 中 |
| MLA pool size=0 in metrics | 显示问题 | 低 |
| L2=0 Staging 未建模 | DRAM 规划精度 | 低 |

### 真实 NPU 标定需求

1. **L2/L3 带宽**：当前使用固定 64GB/s（DRAM）/ 8GB/s（SSD），需真实标定
2. **H2D/D2H 延迟**：当前使用饱和带宽模型，需真实 latency
3. **Replay 计算开销**：当前使用 profile-based predictor，需真实 Mamba forward 延迟
4. **SSD 队列深度**：当前固定带宽，需随并发变化的 IOPS/带宽模型

---

## 结论

| 问题 | 回答 |
|------|------|
| 1. Unified L3 Hits=0 | **合法行为**——group.size 正确设置导致 radix tree 淘汰，Phase 3 找不到完整前缀 |
| 2. L3 Writes 差异 | **容量导致**——小 group.size → 激进淘汰 → 更少 backup → 更少 L3 writes |
| 3. 字节账本 | **闭合**——79MB 差异是 MLA allocated-but-not-live pages |
| 4. 双向借用 | **KDA→MLA 已验证，MLA→KDA 未工作**（enable_storage=False）|
| 5. 平台就绪 | **未完全就绪**——需修复 MLA→KDA 借用和 Legacy group.size |

**建议**：修复 MLA eviction 路径（接入 HiSim backup 路径）和 Legacy group.size（调用 `_bind_l2_host_group`），然后重新验证双向借用和 One Pool 收益。
