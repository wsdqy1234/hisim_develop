# Phase 2A.1 Correctness Hardening — Side Effect Analysis

## 1. Eviction Hook Side Effect Analysis

### Old Approach (Filter After Native Step)

**Hook point**: `UnifiedRadixCache._evict_device_next_node`

**Call chain**:
```
_evict_components loop
  → UnifiedRadixCache._evict_device_next_node(ct, tracker) [HOOK]
    → tree_core.evict_device_next_node(ct, tracker)         [NATIVE]
      → MambaComponent._evict_device_next_node(tracker, device_frees, host_frees)
        Case A: Evictable leaf → cursor advances, return node_id
                NO device_frees, NO host_frees, NO tracker delta
        Case B: Internal/tombstone → _evict_component_and_detach_lru + _cascade_evict
                device_frees populated, host_frees populated, tracker updated
    → _free_values(result.device_frees, result.host_frees) [NATIVE]
    → _accumulate_tracker(tracker, result.tracker)         [NATIVE]
    → return (node_id, made_progress)
  [POLICY CHECK] should_evict(node_id)?
    → Yes: return (node_id, made_progress) — _evict_device_leaf called next
    → No:  return (None, True)             — SKIP, _evict_device_leaf NOT called
```

**Side effects before policy decision**:

| Side Effect | Case A (Leaf) | Case B (Tombstone) |
|---|---|---|
| Cursor advancement | YES (cursor moves past leaf) | YES (cursor moves past internal node) |
| Device tensor free | NO | YES (via _evict_component_and_detach_lru) |
| Host tensor free | NO | YES (via _cascade_evict) |
| Tracker update | NO | YES (delta from tombstone frees) |
| Slot free (allocator) | NO (happens in _evict_device_leaf) | YES (tombstone slot freed) |

**Key problem**: For Case A (leaf candidate), the cursor advances BEFORE the policy
decision. When the policy SKIPs a hot leaf, the cursor has already moved past it.
The hot leaf can't be re-visited in this walk.

**Consequence**: When all remaining candidates are hot (all skipped), the cursor
exhausts (node_id=None, made_progress=False). The eviction loop breaks. The
allocator doesn't get enough space → "alloc_req_slots runs out of memory" → crash.

**Native EVICT=36 vs Hotness EVICT=4 explanation**:
- Native: 0 SKIP, 36 EVICT — all candidates pass through, sufficient eviction
- Hotness: 32 SKIP, 4 EVICT — only 4 cold nodes evicted, 32 hot nodes skipped
- 4 evictions < 16 required → allocator fails → server crash
- Hotness metrics missing because crash before metrics write

### New Approach (Pre-sort Before Native Step)

**Hook point**: `MambaComponent._evict_device_start` / `_evict_device_next_node`

**Call chain**:
```
_evict_components loop
  → tree_core.evict_device_start(ct, request_cnt)           [NATIVE]
    → MambaComponent.evict_device_start(request_cnt)        [NATIVE]
      → MambaComponent._evict_device_start(request_cnt)     [HOOK]
        1. orig(request_cnt) — native cursor setup, request_cnt stored
        2. Walk LRU list: get_lru_no_lock() → get_prev_no_lock() chain
        3. Filter: node in evictable_device_leaves && _can_evict_leaf_atomically
        4. Collect (node, prefix_hash) candidates
        5. pol.select_victims(candidates) — sort by policy
        6. Store as _policy_evict_queue
  → UnifiedRadixCache._evict_device_next_node(ct, tracker) [NATIVE]
    → tree_core.evict_device_next_node(ct, tracker)        [NATIVE]
      → MambaComponent.evict_device_next_node(tracker, ...) [NATIVE]
        → MambaComponent._evict_device_next_node(tracker, ...) [HOOK]
          1. Pop next node from _policy_evict_queue
          2. Check still evictable: in_list && in evictable_device_leaves
          3. If yes: return node.id (NO cursor advancement)
          4. If no: skip to next in queue
          5. Queue exhausted: fall back to orig() (native tombstone cleanup)
    → _free_values(result.device_frees, result.host_frees) [NATIVE]
    → _accumulate_tracker(tracker, result.tracker)         [NATIVE]
  → _evict_device_leaf(node_id, tracker)                   [NATIVE, unmodified]
```

**Side effects before policy decision**: NONE

| Side Effect | Pre-sort Hook |
|---|---|
| Cursor advancement | NO (queue, not cursor) |
| Device tensor free | NO (happens in _evict_device_leaf) |
| Host tensor free | NO (happens in _evict_device_leaf) |
| Tracker update | NO (happens in _evict_device_leaf) |
| Slot free (allocator) | NO (happens in _evict_device_leaf) |

**Why no cursor exhaustion**:
- ALL evictable leaves are in the queue (walked upfront in _evict_device_start)
- Queue returns nodes in policy order (cold first for HotnessAware)
- If queue exhausted, native fallback handles tombstones + remaining nodes
- Sufficient eviction guaranteed: all evictable nodes are considered

**Lock/ACK/Slot conservation**:
- Lock: evictable_device_leaves only contains nodes with lock_ref == 0
- ACK: _evict_device_leaf handles ACK natively, unmodified
- Slot: each _evict_device_leaf call frees one slot, tracked by tracker
- Allocator: target_reached check is native, unmodified

## 2. Checkpoint Identity

**Old**: `node_id` (integer counter, changes on split/reinsert)
**New**: `_node_prefix_hash(node)` = `f"{get_last_hash_value()}_{len(hash_value)}"`

Properties:
- Stable across node split (same prefix → same last hash + length)
- Stable across eviction + reinsert (same prefix → same hash)
- NOT stable across different prefixes with same last hash + length (collision risk low)

Oracle uses `_node_first_token(node)` = root node's `key[0]` for trace matching.
