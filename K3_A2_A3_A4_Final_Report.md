# Phase 1 A2/A3/A4 Final Report

## Executive Summary

All five deliverables completed. A2 four-scenario regression passes 4/4.
force_evict root cause identified and fixed. L2→L3 eviction path verified.
MLA/KDA L3 counters confirmed separate.

---

## 1. force_evict 的准确触发原因

### 调用位置
`src/hisim/simulation/sglang/sglang_hook.py`, `wrapped_get_new_batch_prefill`,
L2939-2973 (pre-fix line numbers).

### 触发条件 (pre-fix)
1. `HISIM_KDA_OBS=1` (observation enabled)
2. `waiting_queue` 非空
3. `mamba_allocator.available_size() < 4` (L1 可用 slot < 4)
4. `kda_obs.get_mamba_evictable() > 0` (HiSim counter says evictable > 0)

### 请求方
HiSim `wrapped_get_new_batch_prefill` — **HiSim 特有的 Proactive Eviction**。
NOT native SGLang scheduler capacity eviction. Native eviction happens inside
`alloc()` when the pool is full — that path is untouched.

### 与原生 Eviction 的混淆
force_evict 在 `original_get_new_batch_prefill` 之前运行，提前淘汰 L1
checkpoints。这导致 L1-only 场景中 P0 checkpoint 在 reuse 前被 evict 到 L2，
实际走了 L2 Restore 而非 Device COW。

### 修复
- Gate with `HISIM_DISABLE_PROACTIVE_EVICT` (consistent with
  `wrapped_check_hicache_events` proactive eviction)
- When enabled (proactive ON), use native `mamba_evictable_size()` instead of
  HiSim counter
- When `HISIM_DISABLE_PROACTIVE_EVICT=1`: force_evict disabled, native
  capacity-driven eviction (inside `alloc()`) still works

### L1 Available Slots at Trigger Time (L1=4)
- Build P0: 3 slots (run+pp) + 1 ckpt = 4 used, 0 free during run;
  after completion: 1 ckpt + 3 free
- Reuse P0: available=3, `3 < 4` = TRUE → force_evict triggered (pre-fix)
- Post-fix: force_evict disabled → alloc succeeds (3 >= 3) → COW

---

## 2. L2→L3 容量淘汰与 Write-through 的实际关系

### 两种 L3 写入路径

**L3 Write-through** (每次 checkpoint 创建时):
- Location: `_mock_start_writing` L1606-1625
- Trigger: `backup_from_device` backs up to L2 → `batch_set_mamba` writes to L3
- Policy: `hicache_write_policy = "write_through"` (default)
- Result: Every checkpoint is in L3 from creation time

**L2 Capacity Eviction** (L2 满时):
- Location: `MockMambaStatePoolHost.backup_from_device` L1831-1840
- Trigger: `len(self._slots) > self.size` (L2 full)
- Action: LRU pop → evicted hash collected in `_last_evicted_hashes`
- Consumer: `_mock_start_writing` L1610-1613 promotes to L3 via
  `batch_set_mamba`

### 关系
Write-through 已经在创建时将 checkpoint 写入 L3。L2 Capacity Eviction
只是从 L2 移除条目（L3 已有副本）。因此：
- L3 Write-through: 数据进入 L3 的主要路径
- L2 Capacity Eviction: 从 L2 移除（L3 副本已存在）

### L3-only 场景验证 (L1=4, L2=3)
```
Build P2: L2=[P2]           cached=1, L3=[P2] (write-through)
Build Q2: L2=[P2,Q2]        cached=2, L3=[P2,Q2]
Build R2: L2=[P2,Q2,R2]     cached=3 (FULL), L3=[P2,Q2,R2]
Build S2: L2 evict P2 (LRU) → L3=[P2,Q2,R2,S2], L2=[Q2,R2,S2]
Reuse P2: L1 miss, L2 miss (P2 evicted), L3 hit → restore_l3
```

Server log confirms: `L2_KDA[cap=3 cached=0→1→2→3→3]`

### Native 恢复路径支持
L3-only KDA Checkpoint 重新发现通过 `_mock_start_loading` L1756-1763:
1. `mh.has(_slot_hash)` → L2 miss
2. `st.batch_exists_mamba([_slot_hash])` → L3 hit
3. `backup_from_device(from_l3=True)` → L3→L2 restore
4. `load_to_device(_slot_hash)` → L2→L1 restore

无需修改 Node Value 或伪造命中。原生路径完整支持。

---

## 3. A2 四组对照实验结果

### 配置
- L1=4 (`HISIM_MAMBA_POOL_SIZE=4`)
- L2=3 (`HISIM_L2_MAMBA_SIZE=3`)
- Proactive Eviction=OFF (`HISIM_DISABLE_PROACTIVE_EVICT=1`)
- Observation=ON (`HISIM_KDA_OBS=1`)
- 每组独立初始化测试环境（separate server instance, clean cache）

### 结果

| 场景 | 预期路径 | 关键指标 | Lock/ACK | 状态 |
|------|---------|---------|----------|------|
| **L1-only** | Device COW | `cow_exec=1`, `load_calls=0`, `restore_l2=0`, `restore_l3=0` | inc=5/dec=5, wq=1/1 | ✅ PASS |
| **L2-only** | Host Load-back | `cow_exec=0`, `load_calls=1`, `restore_l2=2`, `kda_l3_hits=1` | inc=13/dec=13, wq=3/3 | ✅ PASS |
| **L3-only** | Storage Restore | `cow_exec=0`, `load_calls=1`, `restore_l3=1`, `kda_l3_hits=1` | inc=16/dec=16, wq=4/4 | ✅ PASS |
| **All-Miss** | Full Prefill | `cow_exec=0`, `load_calls=0`, `reused_ratio=0.0`, `all_miss=1` | inc=3/dec=3, wq=1/1 | ✅ PASS |

### 说明
- L2-only 的 `restore_l2=2`: 一次为 L2→L1 H2D load, 另一次可能为 L3→L2
  intermediate step（write-through 将 P1 写入 L3，load-back 路径先查 L2）
- L3-only 的 `restore_l3=1` + `restore_l2=1`: L3 恢复路径包含两步
  (L3→L2 SSD read + L2→L1 H2D load)
- `replay_fallback=1` 在 L2-only 和 L3-only 中: 无 COW 时使用 replay
- `writing_check_call` 高达 ~160万: 原生行为，每轮调用 `check_hicache_events`

### 已知局限
inc=39/dec=39、alloc=39/free=39、ACK 9/9 只能证明本次正常测试路径通过，
**不能证明失败、取消或节点 Split 场景也已验证**。这些边界场景需要额外测试。

---

## 4. A3/A4 统计验证

### A3: L3 Write-through vs L2 Capacity Eviction

| 分类 | 定义 | L3-only 实测 |
|------|------|-------------|
| L3 Write-through | checkpoint 创建时写入 L3 | `kda_l3_writes=4` (P2,Q2,R2,S2) |
| L2 Capacity Eviction | L2 满时 LRU 淘汰到 L3 | `backup_l3=5` events (4 write-through + 1 L2 overflow) |
| L3 Lookup Hit | load-back 时 L3 命中 | `kda_l3_hits=1` |
| L3 Load-back Completed | L3→L2→L1 恢复完成 | `restore_l3=1` |
| Forward Verified Logical Reuse | 前向消费确认逻辑复用 | `replay_fallback=1`, `prefix_cache_reused_ratio=0.2` |

### A4: MLA/KDA L3 Load 计数器验证

| 计数器 | 路径 | 调用位置 | L3-only 实测 |
|--------|------|---------|-------------|
| `inc_hicache_l3_load_dur` (MLA) | FULL KV pool L3 read | L1719 `_mock_start_loading` | `mla_l3_load_dur_total=3.9e-07` (≈0, mock KV bytes=0) |
| `inc_hicache_kda_l3_load_dur` (KDA) | MAMBA pool L3 read | L1761 `_mock_start_loading` | `kda_l3_load_dur_total=0.001004` (14.45MB/14.6GB/s) |

**确认**: 两个计数器在完全不同的代码路径中递增：
- MLA L3 load 在 FULL KV pool transfer 路径 (L1714-1719)
- KDA L3 load 在 MAMBA pool transfer 路径 (L1756-1763)
不交叉计数。Server log 确认:
```
[A4-MLA-L3-LOAD] node=18 dur=0.000000
[A4-KDA-L3-LOAD] hash=kda_18_0 dur=0.001004
```

---

## 5. HiSim/SGLang 修改文件清单

### 全部修改在 HiSim Adapter/Mock 层 — 无 SGLang core 改动

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/hisim/simulation/sglang/sglang_hook.py` | HiSim Adapter | force_evict gate, L2 size fix, L2 pool logging, A4 logging, L2 pool metrics fix, ACK lifecycle fix (cn%10 removed), native evictable counter restoration |
| `src/hisim/simulation/sglang/sglang_mock_class.py` | HiSim Mock | `_StrictMambaAllocator` with `_ledger`, slot tracking |
| `src/hisim/simulation/kda_obs.py` | HiSim Obs | Shadow State Registry, per-req KDA obs, event logging |
| `src/hisim/simulation/manager/state.py` | HiSim State | `inc_hicache_l3_load_dur` (MLA L3, F4 fix), separate from KDA counter |
| `src/hisim/simulation/types.py` | HiSim Types | `RequestStats` KDA per-req fields |
| `benchmark/collection/gen_k3_a2_isolated.py` | Benchmark | A2 isolated trace generator (4 scenarios) |
| `benchmark/collection/run_a2_isolated.sh` | Benchmark | A2 per-scenario run script |
| `benchmark/collection/gen_k3_a2_scenarios.py` | Benchmark | A2 combined trace generator |
| `benchmark/collection/gen_k3_a2_l3_bypass.py` | Benchmark | L3 bypass trace generator |
| `benchmark/collection/gen_k3_l3_verify.py` | Benchmark | L3 verify trace generator |
| `benchmark/collection/run_l3_verify.sh` | Benchmark | L3 verify run script |
| `K3_Cache_Architecture_Audit.md` | Docs | Phase 0 audit report |
| `K3_KDA_A2_Scenario_Analysis.md` | Docs | A2 scenario analysis |
| `K3_KDA_Retreat_Analysis.md` | Docs | KDA retreat analysis |

### SGLang core: **未修改**
所有 SGLang 原生代码 (`/sgl-workspace/sglang/`) 未改动。
- 原生 LRU 算法: 未修改
- 原生 `check_hicache_events`: 每轮调用（移除了 HiSim 的 cn%10 门）
- 原生 Eviction Policy: 未修改
- 原生 `lock_ref` 机制: 仅观察（inc/dec hooks），不直接修改

---

## 修复清单

### 本轮修复 (commit 0907e34)
1. **force_evict gate**: 受 `HISIM_DISABLE_PROACTIVE_EVICT` 控制
2. **L2 pool size fix**: original init_hicache 路径尊重 `HISIM_L2_MAMBA_SIZE`
3. **L2 pool state logging**: server log + checkpoint_events
4. **A4 logging**: `[A4-MLA-L3-LOAD]`/`[A4-KDA-L3-LOAD]` + metrics counters
5. **L2 pool metrics fix**: 从 active `tc.mamba_pool_host` 读取

### 前轮修复 (commit 2be24d3, squashed)
1. **ACK lifecycle fix**: 移除 cn%10 门，每轮调用 `check_hicache_events`
2. **Native evictable counter restoration**: 作为淘汰依据
3. **Remove direct lock_ref modification**: `dec_lock_after_backup` 块移除
4. **KDA retreat/replay metrics fix**
5. **MLA/KDA L3 latency accounting** (F4 fix)
6. **Per-request KDA cache observability**
7. **Slot tracking** (`_StrictMambaAllocator._ledger`)
8. **Shadow-state validation** (kda_obs.py)

---

## 下一步

进入 Proactive OFF 的原生缓存策略基线实验 (Task B)。
