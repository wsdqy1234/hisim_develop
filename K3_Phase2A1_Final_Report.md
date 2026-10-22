# Phase 2A.1 Correctness Hardening — Final Report

## 1. Native Eviction Hook 和调用链

**Hook point**: `MambaComponent._evict_device_start` / `_evict_device_next_node`

```
evict_for_alloc (capacity trigger, native)
  → _evict → _evict_components (native loop)
    → tree_core.evict_device_start(ct, request_cnt)           [NATIVE]
      → MambaComponent.evict_device_start(request_cnt)        [NATIVE wrapper]
        → MambaComponent._evict_device_start(request_cnt)     [HOOK: pre-sort]
          1. orig(request_cnt) — native cursor + request_cnt setup
          2. Walk LRU: get_lru_no_lock() → get_prev_no_lock() chain
          3. Filter: node in evictable_device_leaves && _can_evict_leaf_atomically
          4. Collect (node, str(_node_first_token(node))) candidates
          5. pol.select_victims(candidates) — sort by policy
          6. Store as _policy_evict_queue
    → UnifiedRadixCache._evict_device_next_node(ct, tracker) [NATIVE]
      → tree_core.evict_device_next_node(ct, tracker)        [NATIVE]
        → MambaComponent._evict_device_next_node(tracker,…) [HOOK: pop from queue]
          1. Pop next node from _policy_evict_queue
          2. Check still evictable: in_list && in evictable_device_leaves
          3. Return node.id (NO cursor advancement, NO frees)
          4. Queue exhausted → fall back to orig() (tombstone cleanup)
      → _free_values(device_frees, host_frees)               [NATIVE]
      → _accumulate_tracker(tracker, delta)                  [NATIVE]
    → _evict_device_leaf(node_id, tracker)                   [NATIVE, unmodified]
    → tree_core.evict_device_end(ct)                         [NATIVE]
      → MambaComponent._evict_device_end()                   [HOOK: clear queue]
```

## 2. 为什么不改变原生淘汰时机和生命周期

| 层面 | 是否改变 | 说明 |
|------|---------|------|
| 容量触发时机 | 否 | `evict_for_alloc` 由原生 `alloc()` 触发 |
| 可淘汰候选集 | 否 | `evictable_device_leaves` 原生维护 |
| Lock 检查 | 否 | `lock_ref == 0` 由原生 `get_prev_no_lock` 跳过 |
| ACK 链 | 否 | `_evict_device_leaf` 原生处理 |
| Allocator | 否 | `mamba_allocator.free` 在 `_free_values` 中原生调用 |
| Cascade | 否 | `_cascade_evict` 在原生 `_evict_device_next_node` 中 |
| Eviction 执行 | 否 | `_evict_device_leaf` 原生执行 |
| **Victim 优先级** | **是** | `select_victims` 重排候选顺序 |

## 3. 副作用分析

**Old approach (filter after native step)**:
- SKIP 前 cursor 已前进 → 热节点无法重新访问
- SKIP 前 tombstone 可能已 free → tracker 已更新
- `return (None, True)` 误报有效进展 → cursor 耗尽 → 分配失败

**New approach (pre-sort before native step)**:
- SKIP 前 cursor 未前进（使用独立 queue）
- SKIP 前无 device/host free、无 tracker 更新
- Queue 包含所有候选 → 无 cursor 耗尽
- Queue 耗尽时 fallback 到原生 walk（处理 tombstone）

## 4. Checkpoint Identity

**问题**: `hash_value` 在 CPU simulation 模式下为 None，导致 identity 为空。
**修复**: 使用 `_node_first_token(node)` — walk root-to-node，取第一个非空 key segment 的首 token。
- 稳定：相同 prefix → 相同首 token
- 跨 node split/reinsert 不变
- 在 simulation 模式下有效（key 包含实际 token IDs）

## 5. Double-Free 修复

**问题**: REPLAY-FALLBACK 路径释放 MAMBA slot 但不清除 `component_data[MAMBA].value`，
后续 eviction 再次 free 同一 slot → AssertionError 崩溃。
**修复**: `_StrictMambaAllocator.free()` 改为幂等 — 已释放的 slot 跳过并 warning。

## 6. 三策略结果（Phase 2A.2 最终回归）

### 配置
- Trace v4: 56 requests (8 hot × 3 access + 8 cold + 2 fork + 12 churn + 18 reuse)
- Fork test: F0 shares first token (100000) with S0, different suffix (999000 vs 100001)
- KDA=16, L1-only, strict free mode, _node_prefix_identity (MD5)

### 结果对照

| Metric | Native | Hotness | Oracle |
|--------|--------|---------|--------|
| Completed | 56/56 ✓ | 56/56 ✓ | 56/56 ✓ |
| COW Hits | 8 | 16 (+100%) | 20 (+150%) |
| Replay Tokens | 18432 | 10240 (-45%) | 6144 (-67%) |
| Double-Free (strict) | 0 ✓ | 0 ✓ | 0 ✓ |
| MA-STALE | 0 | 0 | 0 |
| Policy Decisions | N/A | 55 | 47 |
| Unique Victim Identities | N/A | 22 | 24 |

### P0: Double-Free 验收
- 根因: `free_mamba_cache` 释放 `req.kv.mamba_pool_idx`，slot 已被 eviction 释放
- 修复: `_safe_free()` 预检查 + 严格 `free()` assert + slot ledger
- **验收: 三种策略严格模式 Double-Free=0** ✓

### P1: Checkpoint Identity 验收
- `_node_prefix_identity`: MD5 hash of full root-to-node token sequence
- Fork 测试: F0 (首 token=100000, 第二=999000) 与 S0 (首=100000, 第二=100001) 有不同 identity ✓
- Hotness: F0 frequency=0, S0 frequency=1 — 未错误合并 ✓
- Oracle: F0 next-access=540, S0 next-access=400 — 正确区分 ✓

### P1: Pre-sort Hook 语义一致性
- Queue 弹出时重新检查: `in_list` + `evictable_device_leaves` + `_can_evict_leaf_atomically` ✓
- Queue 耗尽后 native fallback 处理 tombstone/cascade ✓
- 已淘汰节点从 LRU 移除，native 不重复访问 ✓
- FULL/MAMBA cascade 由 `_evict_device_leaf` 原生处理 ✓

## 7. Native EVICT=36 vs Hotness EVICT=4 解释

**Old filter approach (v1)**:
- Native: 0 SKIP, 36 EVICT — 所有候选通过，足够淘汰
- Hotness: 32 SKIP, 4 EVICT — 32 个热节点被跳过，仅 4 个冷节点淘汰
- 4 < 16 (需要淘汰数) → allocator 失败 → 服务器崩溃
- 原因：filter 在 cursor 前进后执行，cursor 耗尽无法恢复

**New pre-sort approach (v2)**:
- 所有候选在 queue 中，按策略排序
- 无 SKIP/EVICT 区分 — queue 中所有节点按顺序淘汰
- 足够淘汰保证 — queue 不空直到所有候选已考虑
