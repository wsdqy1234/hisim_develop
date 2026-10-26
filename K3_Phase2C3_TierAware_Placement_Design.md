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

## Task 5: NPU Runtime Error Diagnosis — RESOLVED

### Root Cause

NPU devices 0-5 have leaked ~11.5GB HBM from previous SIGKILL kills.
Devices 6-15 are clean and functional.

| Check | Result |
|-------|--------|
| NPU hardware health | ✓ OK (npu-smi: Health=OK) |
| Process resource leak | ✗ No process in device |
| HBM memory leak | ⚠ 11.5GB leaked on devices 0-5 (driver-level) |
| ACL runtime state | ✗ rtSetDevice(0-5) = FAIL(507033) |
| rtSetDevice(6-15) | ✓ OK |
| Fix | `ASCEND_RT_VISIBLE_DEVICES=7` |

### Why CPU Mode Triggers NPU

SGLang imports `torch_npu` during module init → `aclInit()` → `rtSetDevice()`.
`--device cpu` doesn't prevent this import.
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` prevents auto-load but not conditional imports.

### Config Issues (also resolved)

1. Original config lost on container restart (/tmp cleared)
2. `aiconfigurator` predictor doesn't support `kimi_k3` model type
3. Fix: use `k3_profile_based` predictor
4. Required: `backend_version=0.5.6.post2`, `device_name=h20e_sxm`

### Prevention

Use `kill -TERM` (SIGTERM) instead of `pkill -9` (SIGKILL) to allow
`aclFinalize` graceful cleanup before process termination.

## Task 4: Experiment Results — COMPLETED

### L1=Native + L2=various (B2-Fork, seed=42, device 7)

| L1 | L2 Size | L2 Policy | COW | Replay | L2_Cap |
|----|---------|-----------|-----|--------|--------|
| Native | 32 | native | 21 | 0 | 32 |
| Native | 32 | hotness | 21 | 0 | 32 |
| Native | 32 | cost_aware | 21 | 0 | 32 |
| Native | 3 | native | 21 | 0 | 3 |
| Native | 3 | cost_aware | 21 | 0 | 3 |
| Native | 128 | native | 21 | 0 | 128 |
| Native | 128 | cost_aware | 21 | 0 | 128 |

**L1=Native already optimal (R=0)** — L2 doesn't change anything.

### L1=Hotness + L2=various (B2-Fork, seed=42, device 7) — KEY RESULTS

| L1 | L2 Size | L2 Policy | COW | Replay | Finding |
|----|---------|-----------|-----|--------|---------|
| Hotness | 0 | native | 21 | 0* | L3 enabled → R=0 |
| Hotness | 3 | native | 21 | **820447** | L2 too small |
| Hotness | 32 | native | 21 | **820447** | L2 too small |
| Hotness | 32 | cost_aware | 21 | **820447** | Same as native |
| **Hotness** | **128** | **native** | **21** | **0** | **L2=128 eliminates replay!** |

*Note: L2=0 with L3 enabled produces R=0 because all evicted checkpoints
go directly to L3 and are restored via L3 pre-fetch.

### Key Finding

**L2=128 (large) completely eliminates the 820K replay!**
- L1=Hotness alone: R=820447 (cascade degradation)
- L1=Hotness + L2=128: R=0 (L2 catches all evicted checkpoints)

**L2=32 and L2=3 do NOT help:**
- Checkpoints evicted from L2 to L3
- L3 pre-fetch only queries current round's slots (`_kda_l3_pending_writes`)
- Previous rounds' slots are in L3 but never queried → replay

**L2 policy (native vs cost_aware) doesn't differentiate:**
- With L2=32, the bottleneck is L3 pre-fetch scope, not L2 eviction order
- Need to fix L3 pre-fetch to query ALL needed slots, not just current round

### Environment Recovery

| Issue | Fix |
|-------|-----|
| NPU devices 0-5 leaked 11.5GB HBM | `ASCEND_RT_VISIBLE_DEVICES=7` |
| Config file lost (/tmp cleared) | `k3_profile_based` predictor + `h20e_sxm` + `backend_version=0.5.6.post2` |
| NPU leak from SIGKILL | `kill -TERM` (SIGTERM for graceful shutdown) |
| Container restart lost all /tmp files | Regenerated traces, scripts, check_b1_metrics.py |

## Commits

- `05784c3`: P2C.2 final report (cascade ORDER root cause)
- P2C.3 commits:
  - `l2_placement_policy.py` — 4 L2 strategies + factory
  - `sglang_mock_class.py` — MockMambaStatePoolHost policy integration
  - `sglang_hook.py` — depth passing + L2 metrics + policy logging
  - `run_p2c3_l2.sh` — experiment runner (L2=3/32/128 × 3 policies)
  - This design document
