# Phase 2C.3 Tier-Aware KDA Placement — Design & NPU Diagnosis

## Task 2: Tier-Aware Placement Minimal Interface (DESIGNED)

### Architecture

```
L1 (NPU HBM)          L2 (DRAM Host)         L3 (SSD)
┌──────────────┐      ┌────────────────┐     ┌──────────┐
│ Native LRU   │─────▶│ Pluggable L2   │────▶│ Native   │
│ (unchanged)  │ backup│ Placement Policy│evict│Write-thru│
│              │◀─────│                │◀────│          │
│              │ load  │                │restore        │
└──────────────┘      └────────────────┘     └──────────┘
```

### Design Constraints (per user requirements)

1. **L1 stays Native LRU** — no modification to L1 eviction
2. **L2 policy only controls L2 eviction order** (which slots → L3)
3. **Cannot fabricate L2 hits** or bypass L3 restore path
4. **L3 stays native write-through**
5. **L2 capacity strictly limited** via `HISIM_L2_MAMBA_SIZE`

### Interface: `l2_placement_policy.py`

```python
class L2PlacementPolicy:
    def record_admission(self, slot_hash, slot_idx, depth)
    def record_access(self, slot_hash, virtual_time)
    def record_eviction(self, slot_hash)
    def select_victim(self, l2_pool) -> Optional[str]
    def get_priority(self, slot_hash) -> float
```

### Strategies

| Policy | Env Var | Eviction Logic | Key Feature |
|--------|--------|----------------|-------------|
| NativeL2Policy | `HISIM_L2_POLICY=native` | LRU (popitem) | Baseline |
| HotnessL2Policy | `HISIM_L2_POLICY=hotness` | Cold (low freq) first | Sliding window freq |
| CostAwareL2Policy | `HISIM_L2_POLICY=cost_aware` | Low replay cost first | Depth-based cost, `base_prob` configurable |
| OracleL2Policy | `HISIM_L2_POLICY=oracle` | Farthest-next-use first | Belady for L2 |

### Implementation (Committed)

**New file**: `src/hisim/simulation/l2_placement_policy.py`
- 4 policy classes + factory via `HISIM_L2_POLICY` env var
- `get_l2_policy()` / `reset_l2_policy()` for experiment isolation

**Modified**: `sglang_mock_class.py` — `MockMambaStatePoolHost`
- `backup_from_device()`: uses `policy.select_victim()` instead of `popitem(last=False)`
- `load_to_device()`: calls `policy.record_access()` for hotness tracking
- `evict()`: uses `policy.select_victim()` for L3 backup path
- `_get_policy()`: lazy initialization via `get_l2_policy()`
- Added `depth` parameter for CostAwareL2 cost estimation

**Modified**: `sglang_hook.py`
- `handle_kda_backup_operation()`: passes `depth=token_offset` to L2
- L2 policy initialization logging
- L2 metrics in `metrics.json`:
  - `kda_l2_backup_count` — L1→L2 backup count
  - `kda_l2_load_back_count` — L2→L1 hit count
  - `kda_l3_load_back_count` — L3→L2→L1 restore count
  - `kda_l3_hit_count` — L3 hit count
  - `kda_l2_policy` — policy name

### Correctness Guarantees (Task 3)

| Check | How Verified |
|-------|-------------|
| L2 eviction invalidates host_value | `select_victim` removes from `_slots` + `_lru` + calls `record_eviction` |
| L3 copies still discoverable | L2 eviction appends to `_last_evicted_hashes` → L3 `batch_set_mamba` |
| Restore identity/generation correct | L2 uses content-addressed `slot_hash` (same as L3) |
| Lock/ACK/Allocator conservation | L2 policy only touches `_slots`/`_lru`, not allocator or lock |
| L2 capacity strictly limited | `while len(self._slots) > self.size: evict()` |
| Native regression | `HISIM_L2_POLICY=native` (default) = LRU = original behavior |

## Task 1: B2-Fork Cascade Analysis (from P2C.2 data)

### First Victim Divergence (Position 9-12)

```
Native:    ...34, [30, 33, 38, 41], 37, 36  → R=0
Hotness:   ...34, [38, 41, 30, 33], 37, 40  → R=820447
Oracle:    ...34, [30, 33, 41, 38], 37, 36  → R=0
CostAware: ...34, [38, 41, 33, 30], 37, 40  → R=820447
```

**Same 4 nodes evicted, different ORDER.**
- Native/Oracle: evict 30,33 first → R=0 (optimal cascade)
- Hotness/CostAware: evict 38,41 first → R=820447 (cascade degradation)

### Causal Chain

1. L1 evicts node 38 (cold by frequency) instead of node 30 (LRU)
2. Node 38's slot is freed and re-allocated to a new checkpoint
3. The new checkpoint's eviction later triggers cascade evictions
4. Cascade evicts checkpoints that WILL be re-accessed → replay
5. Replay triggers more L1 allocations → more evictions → cascade amplifies

With **L2 Placement**: evicted L1 checkpoints are backed up to L2.
- If L2 has capacity: checkpoint restored from L2 (fast H2D, ~6ms)
- If L2 full: checkpoint evicted to L3, restored from SSD (slow, ~24ms)
- **L2 policy decides which L2 slots to keep** → reduces L3 restores

### Expected L2 Benefit

With L2=32 (moderate capacity):
- Some evicted L1 checkpoints caught by L2 → L2 hits
- L2 policy (hotness/cost_aware) keeps valuable slots → fewer L3 restores
- Replay reduced (L2 restore is fast, overlapped with inference)

With L2=3 (tiny):
- Most evicted L1 checkpoints go directly to L3
- L2 policy has minimal impact (too small to differentiate)

With L2=128 (large):
- Most evicted L1 checkpoints stay in L2 → many L2 hits
- L3 restores minimal
- L2 policy less important (capacity dominates)

## Task 5: NPU Runtime Error Diagnosis

### Root Cause

| Check | Result |
|-------|--------|
| NPU hardware health | ✓ OK (npu-smi: Health=OK, Temp=60°C) |
| Process resource leak | ✗ No process in device |
| HBM memory leak | ⚠ 11558/65536 MB leaked (driver-level) |
| ACL runtime state | ✗ Corrupted (`aclInit` fails, error 107001) |
| NPU reset | ✗ Failed (needs host-level permission) |

### Diagnosis

**Not a driver anomaly, not a process resource leak.**
The ACL (Ascend Computing Language) runtime state is corrupted at the
driver level. The `aclInit()` call fails with `ACL_ERROR_RT_INTERNAL_ERROR`
(error code 107001). This happens because:

1. Each SGLang server start calls `import torch_npu` → `aclInit()`
2. `aclInit` allocates driver resources (context, device handle)
3. When server is killed (SIGKILL via `pkill`), resources are NOT released
4. After ~7-11 starts, driver resource pool exhausted → `aclInit` fails
5. HBM memory (11558 MB) leaked at driver level, no process holds it

### Why CPU Mode Still Triggers NPU

SGLang imports `torch_npu` unconditionally during `import sglang.srt.*`.
The `torch_npu` import calls `aclInit()` which opens the NPU device.
`--device cpu` doesn't prevent this import.

Env vars tested (all ineffective):
- `CUDA_VISIBLE_DEVICES=""` — NPU uses CANN, not CUDA
- `TORCH_TRANSFER_TO_NPU=0` — prevents transfer, not init
- `TORCH_DEVICE_BACKEND_AUTOLOAD=0` — prevents auto-load, not explicit import
- `ASCEND_RT_VISIBLE_DEVICES=""` — not recognized by CANN runtime

Blocking `torch_npu` breaks `triton/backends/ascend` (requires `libtorch_npu.so`).

### Fix

1. **Host reboot** — resets NPU driver, clears leaked HBM
2. **Graceful shutdown** — use SIGTERM instead of SIGKILL to allow `aclFinalize`
3. **Driver reload** — `rmmod/insmod` NPU driver (needs host root)

## Task 4: Experiment Status — BLOCKED

| Experiment | Status | Blocker |
|------------|--------|---------|
| B2-Fork L2=32 (native/hotness/cost_aware) | ✗ Not run | NPU `aclInit` failure |
| B2-Fork L2=3 (native/cost_aware) | ✗ Not run | NPU crash |
| B2-Fork L2=128 (native/cost_aware) | ✗ Not run | NPU crash |
| Mixed L2=32 (native/hotness/cost_aware) | ✗ Not run | NPU crash |

**Environment recovery needed**: host reboot or NPU driver reload.
After recovery, experiments can run using the committed `run_p2c3_l2.sh` script.

## Commits

- `05784c3`: P2C.2 final report (cascade ORDER root cause)
- P2C.3 commits:
  - `l2_placement_policy.py` — 4 L2 strategies + factory
  - `sglang_mock_class.py` — MockMambaStatePoolHost policy integration
  - `sglang_hook.py` — depth passing + L2 metrics + policy logging
  - `run_p2c3_l2.sh` — experiment runner (L2=3/32/128 × 3 policies)
  - This design document
