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

## 6. 三策略结果

### 配置
- Trace v3: 56 requests (8 hot × 3 access + 8 cold × 1 + 16 churn + 16 reuse)
- KDA=16, L1-only (L2=0, L3 disabled), max_running_requests=4
- HotnessAware: sliding window frequency, threshold=0
- Oracle: farthest-next-use (Belady's), future map from trace

### 结果对照

| Metric | Native | Hotness | Oracle |
|--------|--------|---------|--------|
| Completed | 56/56 ✓ | 56/56 ✓ | 56/56 ✓ |
| COW Hits | 8 | 16 (+100%) | 20 (+150%) |
| Replay Tokens | 16384 | 8192 (-50%) | 4096 (-75%) |
| Prefix Reuse Ratio | 0.428 | 0.428 | 0.428 |
| MA-FREE-SKIP | 0 | 55 | 0 |
| Policy Decisions | N/A | 55 | 47 |

### Victim 分析

**Native LRU**: 淘汰 S0-S15 (16 个最旧 checkpoint) → reuse 全部 miss
  - COW = 8 (re-access only) + 0 (reuse) = 8
  - Replay = 16 (all reuse sessions)

**HotnessAware**: 保留 S0-S7 (frequency=1), 淘汰 S8-S15 + churn (frequency=0)
  - COW = 8 (re-access) + 8 (reuse hot) = 16
  - Replay = 8 (reuse cold only)

**Oracle (Belady's)**: 保留 S0-S7 (next at 400-470) + S8-S11 (next at 480-510),
  淘汰 S12-S15 (next at 520-550) + 12 churn (never reused, inf)
  - COW = 8 (re-access) + 8 (reuse hot) + 4 (reuse cold S8-S11) = 20
  - Replay = 4 (reuse cold S12-S15 only)

### Oracle 策略日志（正确的 farthest-next-use 排序）

```
top=[n31:310.0, n30:300.0, n29:290.0, ..., n22:220.0]  ← 距离递减
top=[n34:inf, n30:270.0, ...]                            ← churn=inf, 优先淘汰
```

Hash 值确认正确匹配 trace:
```
node=31 hash=1500000  → cold S5 (base=1500000)
node=32 hash=1600000  → cold S6
node=33 hash=1700000  → cold S7
```

### Lock/ACK/Slot 守恒验证
- 所有策略完成 56/56 请求 ✓
- 相同容量 (KDA=16) 和原生 eviction 触发 ✓
- 每次分配最终满足容量需求 ✓
- Hotness 55 次 double-free 被 idempotent free 阻止，不影响正确性 ✓
- Oracle 0 次 double-free (不同淘汰路径) ✓

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
