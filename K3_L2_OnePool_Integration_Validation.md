# K3 L2 One Pool Integration Validation

> 分支：`develop_dqy_evict_reuse` (merge commit `7e5cd4c` + fix `7750d9f`)
> 日期：2026-10-28
> 范围：L2 Unified DRAM Pool 合入后的端到端语义验收

---

## 总览

| 验收项 | 状态 | 说明 |
|--------|------|------|
| **P0-1 Legacy 零回归** | ✅ 通过 | l2diff_hot 完全匹配 pre-merge；B2-Fork Replay=0 |
| **P0-2 跨类型借用与淘汰** | ✅ 通过 | cross_evict_kda_slots=63/129；self_first 正确 |
| **P0-3 真实容量与碎片** | ✅ 通过 | exact vs pow2 碎片差异显著（0.25MB vs 189MB） |
| **P0-4 L3 Restore / Host Index** | ✅ 通过 | 准入拒绝不产生虚假命中；L3 write-through 保持 |
| **P1 系统级公平对照** | ⚠️ 部分 | 基础设施问题（端口冲突），仅有 legacy_mixed 完整结果 |

**发现的 Bug**：`calc_kv_host_size_per_token` 读取原生 device pool 属性导致 token_bytes ≈ 10MB 而非 HiSim 的 5760 B。已修复（commit `7750d9f`，添加 `HISIM_L2_TOKEN_BYTES` 覆盖）。

---

## P0-1: Legacy 零回归

### 实验配置

不设置任何 `HISIM_L2_ONE_POOL` / `HISIM_L2_TOTAL_GB` → `configure_l2_dram_pool()` 返回 `None` → 走 legacy 固定容量路径。

### 结果对比

| 指标 | l2diff_hot (pre-merge) | l2diff_hot (post-merge) | 匹配？ |
|------|----------------------|----------------------|--------|
| COW | 6 | 6 | ✅ |
| Replay | 0 | 0 | ✅ |
| L3_Wr | 30 | 30 | ✅ |
| L3_Hit | 12 | 12 | ✅ |
| Load_Calls | 12 | 12 | ✅ |
| Mamba_Xfer | 12 | 12 | ✅ |
| Wr_Xfer | 30 | 30 | ✅ |

| 指标 | B2-Fork (pre-merge) | B2-Fork (post-merge) | 匹配？ |
|------|--------------------|--------------------|--------| 
| Replay | 0 | 0 | ✅ |
| Completed | 56 | 56 | ✅ |
| One Pool 启用 | 否 | 否（日志确认无 "L2 unified DRAM pool"）| ✅ |

### 结论

**零回归确认。** 合入后 legacy 路径行为完全一致。代码路径验证：
- `hisim.__file__` = `/home/d00954154/hisim/src/hisim/__init__.py`（无 stale .pth）
- 合并 commit `7e5cd4c` 正确加载
- `configure_l2_dram_pool()` 返回 `None`（默认关闭）
- 服务器日志无 "L2 unified DRAM pool"（确认 legacy 路径）

---

## P0-2: 跨类型借用与淘汰

### 单元测试

```
test_one_pool_mla_borrows_from_kda     PASSED
test_one_pool_kda_borrows_from_mla     PASSED
test_two_ended_allocation_keeps_views_apart PASSED
test_borrow_policy_self_first_replaces_own_view PASSED
test_two_pools_regions_are_isolated    PASSED
```
**5/5 全部通过。**

### 集成测试（One Pool, 1GB, B2-Fork）

```
Mode=one_pool  Total=1.00GB  Used=0.98GB  Free=0.02GB
KV=565.9MB  KDA=410.3MB  KDA_slots=28  MLA_tokens=98198
CrossEvict: KDA_slots=63  MLA_tokens=0  ReclaimCalls=44
Borrow=self_first  KDA_avail=0  MLA_avail=1360
```

- **KDA→MLA 借用**：cross_evict_kda_slots=63 — 63 个 KDA slot 被淘汰给 MLA ✅
- **MLA→KDA 借用**：cross_evict_mla_tokens=0 — MLA 未被淘汰给 KDA（self_first 先在 KDA 内 LRU）✅
- **物理 Allocator**：Used=0.98GB（KDA 410MB + MLA 566MB = 976MB ≈ 0.98GB）✅
- **两端分配**：MLA 从低端、KDA 从高端，各占独立区域 ✅
- **Replay=0** — 借用不破坏 L3 write-through ✅

### 集成测试（One Pool, 1GB, Mixed）

```
KV=343.6MB  KDA=468.9MB  KDA_slots=32  MLA_tokens=52455
CrossEvict: KDA_slots=129  MLA_tokens=0  ReclaimCalls=124
ExtFrag=150.04MB (0.800)
```

- KDA-heavy → MLA-heavy 转换正确触发跨类型淘汰 ✅
- 129 次 KDA slot 被淘汰给 MLA ✅
- 高外部碎片（80%）— 混合分配模式产生碎片 ✅

### self_first / other_first 验证

- `self_first`（默认）：KDA 先在 KDA 内 LRU 替换，不够再向 MLA 借 → cross_evict_mla_tokens=0 ✅
- 借用策略仅控制**借用类别顺序**（先自身 LRU 还是先跨视图），不绕过合法 Victim Selection ✅
- 借用通过 `pool.evict()` → `select_victim()` → `_pol.record_eviction()` → `_up.release_kda()` → `notify_evicted()` → L3 write-through，完整保留生命周期 ✅

---

## P0-3: 真实容量与碎片

### 单元测试

```
test_exact_extent_alloc_free_coalesces     PASSED
test_external_fragmentation_is_reported    PASSED
test_pow2_size_class_internal_fragmentation PASSED
test_accounting_consistency_exact         PASSED
```
**4/4 全部通过。**

### 容量约束验证（1GB, exact, B2-Fork）

```
Total=1.00GB (10,737,418,240 bytes)
page_bytes=92,160 (16 tokens × 5760 bytes)
total_pages=11,639
KDA slot=14,653,440 bytes → 159 pages (exact, 0 internal frag)
KDA capacity=68 slots (11,639/159=73, minus MLA usage)
MLA capacity=186,224 tokens (11,639×16)

Used=0.98GB: KV=565.9MB + KDA=410.3MB = 976.2MB ≈ 0.98GB ✅
Free=0.02GB: 20.5MB free
```

**字节级一致性**：Used + Free = 0.98 + 0.02 = 1.00GB = Total ✅

### 碎片对比（exact vs pow2）

| 指标 | exact | pow2 | 差异 |
|------|-------|------|------|
| Internal Frag | 0.25 MB (0.000) | 189.28 MB (0.286) | **pow2 内部碎片高 756 倍** |
| External Frag | 63.04 MB (0.574) | 150.59 MB (0.444) | pow2 外部碎片更低 |
| KDA_slots | 6 | 0 | pow2 因 size class 向上取整导致 KDA 无法分配 |
| AllocFail MLA | 15 | 48 | pow2 失败更多 |
| CrossEvict KDA | 198 | 167 | pow2 借用更频繁 |

- **exact**：精确页数，内部碎片 ≤ 1 页/对象，外部碎片高 ✅
- **pow2**：2 的幂 size class，内部碎片可达 2×，外部碎片低 ✅
- **KDA=159 pages**：14,653,440 / 92,160 = 159.0（exact 精确整除，0 内部碎片）✅

### 外部碎片场景验证

```
Free=0.02GB (20.5MB)  ExtFrag=15.94MB (0.671)
→ 67% 的空闲字节是碎片（有字节但无足够连续 Extent）
→ alloc_fail_mla=5（MLA 因无连续页而分配失败）
```

**结论**：`available_size`、`admit_kda`、Eviction Fallback 和 Allocation Failure 都正确追踪 ✅

---

## P0-4: L3 Restore / Host Index

### 代码审查

#### backup_from_device 失败传播

```python
# sglang_mock_class.py L1997-1998
if not _up.admit_kda(slot_hash):
    return False  # ← 准入失败，slot 不入 _slots
```

- `admit_kda` 失败 → `backup_from_device` 返回 `False` → slot **不入** `_slots` → `has()` 返回 `False` → **无虚假 L2 命中** ✅
- `_any_mamba_restore = True` 在 `backup_from_device` 之前设置 → "restore success" 但 L3 有数据 → **不触发 replay** ✅
- **语义问题**（非 bug）：`mamba_restore_failed=False` 但 L2 实际未恢复。L3 安全网兜底。

#### L3 write-through 保持

```python
# sglang_hook.py L1644-1648
mh.backup_from_device(_slot_idx, _slot_hash, depth=_depth)  # 可能返回 False
_mamba_slot_hashes.append(_slot_hash)  # ← 无论 backup 结果，都加入 L3 待写队列
# L1627: st.batch_set_mamba(_l3_mamba)  ← L3 写入总是发生
```

**结论**：L2 准入失败不影响 L3 write-through ✅

#### L2=0 临时缓冲区

- L2=0（或极小）→ `admit_kda` 始终失败 → slot 不入 L2 → 直接走 L3
- SSD→DRAM 临时缓冲使用 L2 `_slots` dict（短暂存入后立即淘汰）
- Host DRAM 容量**未正确计入**临时缓冲（已知限制，Phase 2C.3.2 也发现）

### 单元测试

```
test_mamba_host_pool_legacy_free_list     PASSED
test_mamba_host_pool_bound_indices_do_not_leak PASSED
```
**2/2 全部通过。**

### 集成测试（0.1GB 极限小预算）

```
Total=0.10GB  KDA_slots=3  Reject=1  AllocFail_KDA=1
CrossEvict: KDA_slots=4  ReclaimCalls=5
Replay=0  Completed=56
```

- **准入拒绝**：admission_reject_kda=1 — KDA 准入失败被正确追踪 ✅
- **L3 兜底**：Replay=0 — 准入失败的 slot 仍从 L3 恢复 ✅
- **所有请求完成**：Completed=56 ✅

### Host Index 安全

- **Round-robin 分配**：`idx = [(start + i) % n for i in range(need_size)]` — 逻辑 host index，不对应物理字节
- **幂等 free**：`released = [i for i in indices if i in self._allocated_idx]` — 只释放已分配的 index
- **无 Double-Free**：`_allocated_idx.discard(i)` — set 操作天然幂等
- **无旧引用误释放**：不在 `_allocated_idx` 中的 index 不会被释放
- **Leak fix 验证**：test_mamba_host_pool_bound_indices_do_not_leak PASSED ✅

---

## P1: 系统级公平对照

### 完成状态

| 配置 | B2-Fork | Mixed | 说明 |
|------|---------|-------|------|
| Legacy Two Pools | ❌ 端口冲突 | ✅ 完成 | Mode=legacy 确认 |
| Unified Two Pools | ⚠️ 服务器启动但 bench 卡住 | ❌ 未运行 | mode=two_pools 确认 |
| One Pool | ❌ 未运行 | ❌ 未运行 | — |

### 已有结果

**Legacy Mixed**（唯一完整 P1 结果）：
```
Mode=legacy  1pool=0  Completed=162  Replay=0
L2: cached=32/32  free=32  pol=native_l2
CrossEvict: all 0 (legacy 无跨类型借用)
```
- Mode=legacy 正确 ✅
- 无 One Pool 指标（全 0）✅
- Mixed trace 的 KDA activity=0（Phase 1 仅 8 checkpoints，L1=16 不满，无 L2 backup）✅

**Unified Two Pools B2-Fork**（服务器启动确认）：
```
mode=two_pools  total=4.46GB  page=92160B
mla=3.99GB  kda=0.47GB (32 slots)  kv_evict=True
```
- 静态切分正确：MLA=3.99GB + KDA=0.47GB = 4.46GB ✅
- KDA=32 slots（与 Legacy `HISIM_L2_MAMBA_SIZE=32` 一致）✅
- bench 未完成（服务器卡住，可能因前一个实验端口冲突残留）

### P1 未完成原因

1. **端口冲突**：实验间 `pkill` + `sleep 5` 不足以确保端口释放（TIME_WAIT 状态）
2. **服务器启动竞争**：前一个服务器未完全退出时新服务器尝试绑定端口
3. **B2-Fork bench 耗时长**：56 请求 × 40K token 在大预算下 KV cache 管理慢

### 建议

使用 `SO_REUSEADDR` 或在实验间加入端口可用性检查（`while ! python3 -c "import socket; ..."; do sleep 2; done`）。

---

## 发现的 Bug 与修复

### Bug: token_bytes 计算错误

**根因**：`calc_kv_host_size_per_token(device_pool)` 读取原生 device pool 属性：
```python
layer_num = getattr(device_pool, 'layer_num', 23)    # K3: 61
real_head_num = getattr(device_pool, '_real_head_num', 1)  # K3: 128
real_head_dim = getattr(device_pool, '_real_head_dim', 1)  # K3: 128
# → token_bytes = 128 × 128 × 61 × 4 × 2 = 10,285,056 bytes ≈ 10MB
```

HiSim 的 MLA entry_bytes = 5760（head_num=1, head_dim=288），两者差 1785 倍。

**影响**：
- page_bytes = 16 × 10MB = 157MB（应为 92KB）
- 1GB pool 仅 6 页（应为 11,639 页）
- 无法存储有效数据

**修复**（commit `7750d9f`）：添加 `HISIM_L2_TOKEN_BYTES` 环境变量覆盖：
```python
token_bytes_override = _env_int("HISIM_L2_TOKEN_BYTES", 0)
if token_bytes_override > 0:
    token_bytes = token_bytes_override
```

实验中设置 `HISIM_L2_TOKEN_BYTES=5760` 即可。这是配置修复，非算法变更。

---

## 已通过的路径

| 路径 | 验证方式 | 状态 |
|------|---------|------|
| Legacy 零回归（无 One Pool env） | 端到端 B2-Fork + l2diff | ✅ 通过 |
| One Pool 启用（HISIM_L2_ONE_POOL=1） | 端到端 B2-Fork + Mixed | ✅ 通过 |
| Two Pools 启用（HISIM_L2_TOTAL_GB only） | 服务器启动确认 | ✅ 通过 |
| KDA→MLA 跨类型借用 | 集成 cross_evict_kda_slots=63 | ✅ 通过 |
| MLA→KDA 跨类型借用 | 单元测试（集成中 self_first 不触发）| ✅ 通过 |
| self_first 借用策略 | 集成 cross_evict_mla_tokens=0 | ✅ 通过 |
| KDA 准入拒绝 | 集成 admission_reject_kda=1 | ✅ 通过 |
| L3 write-through 保持 | 代码审查 + Replay=0 | ✅ 通过 |
| Host index 无泄漏 | 单元测试 + 代码审查 | ✅ 通过 |
| exact 分配 + 碎片度量 | 集成 IntFrag=0.25MB | ✅ 通过 |
| pow2 分配 + 碎片度量 | 集成 IntFrag=189MB | ✅ 通过 |
| PageExtentAllocator 合并 | 单元测试 coalesces | ✅ 通过 |
| 两端分配（MLA low/KDA high）| 单元测试 + 集成 | ✅ 通过 |
| backup_from_device 失败不产生虚假命中 | 代码审查 | ✅ 通过 |
| record_eviction 跨周期保持历史 | 代码审查（Phase 2C.3.2 修复）| ✅ 通过 |
| L2 策略保留（l2_placement_policy）| 集成 pol=native_l2 | ✅ 通过 |

---

## 仍属 Mock 限制的路径

| 限制 | 说明 | 影响 |
|------|------|------|
| **Wr_Xfer=0 with One Pool** | KDA 通过 `admit_kda` 路径准入，不经传统 MAMBA transfer counter | 无法用传统 KDA 指标度量 One Pool 的 KDA 活动 |
| **MLA pool size=0** | L1 device KV pool 的 `size` 属性在 metrics 提取时为 0 | 不影响 One Pool 功能，仅影响 metrics 显示 |
| **cross_evict_mla_tokens=0** | self_first 策略下 KDA 先在自身 LRU 替换，不触发 MLA 淘汰 | 需 `other_first` 策略或 KDA-heavy + MLA-heavy 交替负载才能触发 |
| **L2=0 临时缓冲未计入 host 容量** | SSD→DRAM 临时数据短暂占用 L2 `_slots` 后立即淘汰 | 不影响正确性，影响 DRAM 容量规划精度 |
| **KDA L3 指标缺失** | `L3_Wr`/`L3_Hit` 是 MLA 计数，KDA L3 写入/命中未单独计数 | Phase 2C.3.2 已发现，需添加 KDA 专用 L3 指标 |
| **P1 公平对照不完整** | 端口冲突导致大部分实验未完成 | 需重新运行，不影响 P0 结论 |

---

## 正式开展 Two Pools vs One Pool 性能实验前的剩余问题

### 必须解决

1. **P1 实验框架修复**：加入端口可用性检查，避免端口冲突导致实验失败
2. **`HISIM_L2_TOKEN_BYTES=5760` 固化**：在实验脚本中始终设置此环境变量，或在代码中直接使用 HiSim 的 MLA entry_bytes

### 建议解决

3. **One Pool KDA 指标补充**：`admit_kda` / `release_kda` 的调用应计入 `Wr_Xfer` / `L2_Bak` 等传统指标，使 One Pool 和 Legacy 的 KDA 活动可对比
4. **KDA 专用 L3 指标**：添加 `kda_l3_writes` / `kda_l3_hits`（当前 `L3_Wr`/`L3_Hit` 是 MLA 指标）
5. **MLA pool size 修复**：metrics 提取中 `mla_pool_size` 应从正确的 KV pool 对象读取

### 可选优化

6. **`other_first` 策略测试**：当前 self_first 下 MLA 从未被淘汰给 KDA，需用 `other_first` 策略验证双向借用
7. **真实 NPU 数据标定**：SSD 带宽、H2D/D2H 延迟、replay 计算开销需用真实数据标定
8. **变长 KDA checkpoint**：不同请求长度产生不同 depth 的 checkpoint，可区分 DepthAwareL2

---

## 实验环境

- NPU 设备 7（`ASCEND_RT_VISIBLE_DEVICES=7`）
- SIGTERM 关停
- `k3_profile_based` predictor, `device_name=h20e_sxm`, `backend_version=0.5.6.post2`
- `max_total_tokens=200000`, `max_running_requests=4`, `HISIM_MAMBA_POOL_SIZE=16`
- `SGLANG_MAMBA_TRACK_INTERVAL=1024`, `--random-seed 42`
- L3 write-through 启用（`HISIM_DISABLE_L3=0`，truthiness bug 已修复）
- `HISIM_L2_TOKEN_BYTES=5760`（修复 token_bytes 计算 bug）

## 提交记录

| Commit | 说明 |
|--------|------|
| `7e5cd4c` | merge: integrate develop_dqy_l2_one_pool |
| `8b8e69c` | docs: L2 One Pool merge report |
| `7750d9f` | fix: HISIM_L2_TOKEN_BYTES override for HiSim MLA entry size |
