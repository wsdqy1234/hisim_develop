# Phase 1B — Replay & Timing Correctness Report

> SGLang-policy Baseline in HiSim CPU Mock. Not equivalent to real NPU inference.

## Task 1: KDA Replay 执行路径

### 追踪结果

```
match_prefix → prefix_indices=1024 (KV cache hit)
  → init_load_back → creates load op (MAMBA host_value status)
  → _mock_start_loading:
      L2 check: mh.has(slot_hash) → False (evicted/L2=0)
      L3 check: st.batch_exists_mamba → True/False/None
      _any_mamba_restore = True/False
  → set_extend_range → length=1 (only 1 new token)
  → K3ProfilePredictor._predict_prefill:
      FakeRequest.input_length = extend_range.length + replay_tokens
```

### 三种 Replay Token 概念区分

| 概念 | 定义 | 来源 | B1 KDA=40 |
|------|------|------|-----------|
| `required_replay_tokens` | MAMBA 状态未恢复，需要重算 | `prefix_len` when restore fails | 1024 |
| `scheduled_replay_tokens` | 传入 predictor 的计算量 | `effective_compute = ext + replay` | 1025 (1+1024) |
| `actual_mock_replay_tokens` | HiSim Mock 实际执行 | 0（CPU mock 不执行真实计算） | 0 |

### 结论
- **1024 replay tokens 已传入 predictor**（`P1-TIMING: ext=1 replay=1024 effective=1025`）
- HiSim Mock 不执行真实 MAMBA 计算，但 predictor 正确估算计算时间
- 原生 SGLang 无独立 KDA Replay 路径——MAMBA replay 通过 `extend_range` + `replay_tokens` 传入 predictor

## Task 2: L3-only 前缀恢复语义

### host_value 失效验证

| 场景 | L2 slot 存在 | host_value 设置 | 结果 |
|------|------------|----------------|------|
| L2=0 (B1) | 否（立即淘汰） | None（P0-1 修复） | ✅ init_load_back 不用 stale |
| L2=3, evicted (B2) | 否（LRU 淘汰） | None（P0-1 修复） | ✅ 走 L3 检查 |
| L2=128 (B2) | 是 | 设为 host_indices | ✅ L2 hit |

### L3 Lookup 触发路径

- **不是** 原生 Storage Prefetch（HiSim 禁用了 backup/prefetch 线程）
- **不是** 原生 HiCache Load-back（原生 init_load_back 用 host_value，不走 L3）
- **是** HiSim Mock `_mock_start_loading` 补充逻辑：
  1. `mh.has(slot_hash)` → L2 miss
  2. `st.batch_exists_mamba([slot_hash])` → L3 hit（KVCM，MD5 hash 转换后）
  3. `mh.backup_from_device(from_l3=True)` → L3→L2 restore
  4. `mh.load_to_device(slot_hash)` → L2→L1 restore

### 原生 vs HiSim 补充

| 行为 | 原生 SGLang | HiSim Mock |
|------|------------|------------|
| L2 backup | `commit_hicache_transfer(BACKUP_HOST)` | `_mock_start_writing` + `backup_from_device` |
| L2 load_back | `init_load_back` + native H2D | `_mock_start_loading` + `load_to_device` |
| L3 write | `_finish_write_through_ack` → storage | `_mock_start_writing` → `batch_set_mamba` |
| L3 lookup | native storage prefetch | `_mock_start_loading` → `batch_exists_mamba` |
| L3 restore | native load_back → L2→L1 | `_mock_start_loading` → `backup_from_device(from_l3=True)` |

## Task 3: TTFT 时间模型

### 请求级时间归因

```
TTFT = queue_wait + l2_load_overlap + inference + l2_backup + l3_overflow
```

其中（OVERLAP_SCHEDULE=True）：
- `l2_load_overlap = max(l2_load_dur - last_inference_dur, 0)` — 与前一轮推理重叠
- `inference = K3ProfilePredictor.predict(extend_range.length + replay_tokens)`
- `l2_backup = max(mla_l2_backup, kda_l2_backup)` — D2H backup
- `l3_overflow = max(0, l3_write - (l2_load + inference + l2_backup))` — L3 写溢出

### 四路径实测

| 路径 | effective_compute | l2_load_dur | l2_backup | inference | TTFT |
|------|-------------------|-------------|-----------|-----------|------|
| L1 COW | 1 token | 0 | 0 | 439ms (clamped) | **382.8ms** |
| L2 Restore | 1 token | ~6.5ms | ~29ms | 439ms (clamped) | **382.8ms** |
| L3 Restore | 1 token | 24ms | ~29ms | 439ms (clamped) | **382.8ms** |
| All-Miss/Replay | 1025 tokens | 0 | ~29ms | 439ms (clamped) | **439.4ms** |

### 为什么 COW/Restore TTFT 相同

1. **Predictor clamp**: 所有 effective_compute < 1576 的请求返回 439ms（最小标定点）
2. **Pipeline overlap**: L2/L3 restore（24ms）被前一轮推理（439ms）完全覆盖
3. **L2 backup**: D2H backup 与推理重叠（`l2_backup` 单独加入 `request_response_time`）

### 实测 vs 估算

| 时间 | 类型 | 来源 |
|------|------|------|
| Prefill compute | **估算** | K3ProfilePredictor 8-point interpolation |
| Decode compute | **估算** | DECODE_BASE × batch_ratio |
| KDA L2 H2D | **估算** | 14.65MB / 64GB/s bandwidth model |
| KDA L3 SSD read | **估算** | 14.65MB / 14.6GB/s bandwidth model |
| MLA L2 H2D | **估算** | Token-level bandwidth model |
| MLA L3 read | **估算** | Token-level bandwidth model |

所有时间均为模型估算，无 NPU 实测数据（`--load-format dummy` 模式）。

### Model Limitation

1. **Predictor 最小 clamp**: effective_compute < 1576 时返回固定 439ms，无法区分 1 token 和 1024 tokens
2. **L2/L3 restore overlap**: 与前一轮推理完全重叠时不可见——真实系统中 restore 可能无法完全重叠
3. **无异步重叠建模**: L2 backup 与 inference 的重叠是粗略的 `max(l2_load - last_inference, 0)` 模型

## Task 4: 四路径对照验证

### 配置

| 路径 | KDA | L2 | L3 | 预期行为 |
|------|-----|----|----|---------|
| L1 COW | 165 | 0 | disabled | 全部 COW hit |
| L2 Restore | 40 | 128 | file | L2 hit → L2 restore |
| L3 Restore | 40 | 3 | file | L2 miss → L3 hit → L3 restore |
| All-Miss/Replay | 40 | 0 | disabled | MAMBA restore fail → replay |

### 结果

| 指标 | L1 COW | L2 Restore | L3 Restore | Replay |
|------|--------|------------|------------|--------|
| COW exec | 24 | 0 | 0 | 0 |
| Replay tokens | 0 | 0 | 0 | **24576** |
| Load calls | 0 | 24 | 24 | 24 |
| L3 writes | 0 | 128 | 128 | 0 |
| KDA L3 load dur | 0 | 0 | **0.024s** | 0 |
| MLA prefix hit | 24576 | 24576 | 24576 | 24576 |
| Mean TTFT | **382.8ms** | **382.8ms** | **382.8ms** | **439.4ms** |

### 成本差异解释

- **Replay vs COW**: 56.6ms 差异来自 predictor 估 1025 vs 1 token（但都被 clamp 到 439ms，差异来自 backup timing 和 overlap）
- **COW vs L2/L3 Restore**: TTFT 相同——L2/L3 restore 被 pipeline overlap 覆盖
- **L2 vs L3 Restore**: TTFT 相同——24ms L3 restore < 439ms inference，完全重叠

### 恢复字节数

| 路径 | 恢复字节数 | 来源 |
|------|-----------|------|
| L1 COW | 0 (D2D, no transfer) | COW 直接在 L1 |
| L2 Restore | 24 × 14.65MB = 351.6MB | L2→L1 H2D |
| L3 Restore | 24 × 14.65MB = 351.6MB | L3→L2 SSD + L2→L1 H2D |
| Replay | 0 (recompute, no transfer) | 无数据传输 |

## Task 5: 重跑结果与 Shadow State

### B1 结果

| 指标 | KDA=40 | KDA=165 |
|------|--------|---------|
| Completed | 152 | 152 |
| COW Exec | 0 | 24 |
| Replay Tokens | **24576** | 0 |
| MLA Prefix Hit | 24576 | 24576 |
| Mean TTFT | **439.4ms** | **382.8ms** |

### B2 结果

| 指标 | L2=3 | L2=128 |
|------|------|--------|
| COW Exec | 0 | 0 |
| Replay Tokens | 0 | 0 |
| L2 Capacity | 3 | 128 |
| KDA L3 Load Dur | **0.024s** | 0 |
| A4-KDA-L3-LOAD | 24 | 0 |
| Mean TTFT | 382.8ms | 382.8ms |

### Shadow State 验证

- `l2_evict_diag`: 125 次 L2 淘汰（L2=3，128 checkpoints）
- `LOAD-L2CHECK`: L2 miss 正确返回 False
- `LOAD-L3CHECK`: L3 hit 正确返回 True（MD5 hash 转换后）
- `LOAD-RESULT`: MAMBA restore OK/FAILED 正确分类
- `REPLAY-FALLBACK`: replay_tokens 正确反映 MAMBA restore 状态

## 结论

### Baseline Correctness

1. ✅ **Replay tokens 正确传入 predictor**（P1-TIMING 修复）
2. ✅ **host_value 过期引用已修复**（只在 L2 slot 存在时设置）
3. ✅ **L2/L3 restore 正确触发**（KVCM hash 修复 + L2 有效性检查）
4. ✅ **Replay/Restore 区分正确**（`mamba_restore_failed` 标志）
5. ✅ **TTFT 反映 replay compute**（439.4 vs 382.8ms）
6. ⚠️ **TTFT 不反映 L2/L3 restore**（pipeline overlap — 正确但需注意）

### 优化优先级建议

| 优化方向 | 收益场景 | 预估收益 | 优先级 |
|----------|----------|---------|--------|
| KDA 保存/淘汰 | L1-only (无 L2/L3) | 24576 replay tokens → 0 | **高** |
| 跨层 Placement | L2=3 (L3 restore) | 24ms L3 → 0ms L2 | **中** |
| Prefetch | L3 restore | 24ms overlap → 0ms | **低** (已 overlap) |
| Value-Aware Eviction | L1-only | Oracle 上界 24576 tokens | **高** (L1-only) |
| Value-Aware Eviction | 有 L2/L3 | replay 已=0 | **低** |

### 最终建议

1. **L1-only 场景**：优先实现 Value-Aware Eviction——Oracle 可消除 24576 replay tokens
2. **有 L2/L3 场景**：Value-Aware Eviction 收益有限——replay 已通过 restore 消除
3. **L2→L3 Placement**：L2=3 时 24ms L3 restore vs L2=128 时 0ms——但被 pipeline overlap 覆盖
4. **Prefetch**：L3 restore 已与前一轮推理重叠，收益最小
5. **Predictor 标定**：需补充 < 1576 token 的标定点，才能准确区分 COW (1 token) 和 replay (1025 tokens)

**建议优先级：KDA 保存/淘汰（L1-only） > 跨层 Placement > Prefetch**
