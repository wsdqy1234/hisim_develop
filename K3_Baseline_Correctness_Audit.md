# K3 Baseline Correctness Audit

> SGLang-policy Baseline in HiSim CPU Mock. Not equivalent to real NPU inference.

## 修复前后指标对比

### Bug 修复清单

| Bug | 根因 | 影响 | 修复 |
|-----|------|------|------|
| KVCM hash 转换 | `_pass_str_to_block_ids("mamba_kda_18_0")` 用 `int(,16)` 解析描述性字符串 | L3 读写全部静默失败，L2 容量对结果无影响 | MD5 转为 hex |
| `tree_core` 类属性 | `target.tree_core` 访问实例属性作类属性 | `C_HiRadixCacheHook` 安装失败，所有 HiCache 修复未执行 | `getattr(target, 'tree_core', None)` |
| `mamba_pool_host` 未设置 | 原 init_hicache 不设置此属性 | L2 KDA 池完全未使用 | 强制创建 `MockMambaStatePoolHost` |
| `.pth` 指向旧代码 | `/home/lfz/hisim/src` (Oct 15) 覆盖 PYTHONPATH | 所有代码修改未生效 | 指向 `/home/d00954154/hisim/src` |
| `host_value` 过期引用 | L2 淘汰时不清除 radix tree node 的 `host_value` | 原生 `init_load_back` 用 stale 数据"恢复" | 只在 L2 slot 存在时设置 `host_value` |
| `_is_restored` 不检查 MAMBA | 只看 `extend_range.length`（KV cache），不看 MAMBA 状态 | MAMBA 未恢复时 `replay_tokens=0`（错误） | 添加 `mamba_restore_failed` 标志检查 |
| L3 自动创建 | `_apply_mock_start_overrides` 总创建 `MockHiCacheStorage` | B1 "L1-only" 实际有 L3 | `HISIM_DISABLE_L3=1` |

### B1 修复前后对比

| 指标 | 修复前（错误） | 修复后（正确） | 说明 |
|------|----------------|----------------|------|
| `kda_historical_replay_tokens` (KDA=40) | 0 | **24576** | L1-only 无 L2/L3 → 需要 replay |
| `kda_historical_replay_tokens` (KDA=165) | 0 | **0** | 全部 COW hit → 无 replay（不受影响） |
| `kda_cow_exec_count` (KDA=165) | 24 | **24** | COW path 不受影响 |
| `kda_l3_writes` | 128 | **0** (L1-only) / **128** (B2) | L3 禁用正确 |

### B2 修复前后对比

| 指标 | L2=3 修复前 | L2=3 修复后 | L2=128 修复后 |
|------|-------------|------------|--------------|
| `kda_l3_load_dur_total` | 0.0 (错误) | **0.0241** | **0.0** |
| `A4-KDA-L3-LOAD` | 0 | **24** | **0** |
| `replay_tokens` | 0 (stale) | **0 (L3 restore)** | **0 (L2 restore)** |
| `LOAD-L3CHECK l3_hit` | 不触发 | **1** | N/A (L2 hit) |

## 事件时间线（B2 L2=3，修复后）

```
Build Phase (96 sessions):
  write_through → backup_from_device → L2 store → L2 evict (cap=3) → L3 write
  host_value: only set if L2 slot still exists (evicted → None)

Churn Phase (32 sessions):
  same as Build, more L2 evictions

Reuse Phase (24 sessions):
  match_prefix → prefix_len=1024 (KV cache hit)
  init_load_back → creates load op (MAMBA host_value=None or set)
  _mock_start_loading:
    L2 check: mh.has(slot_hash) → False (evicted)
    L3 check: st.batch_exists_mamba → True (write-through)
    L3 restore: backup_from_device(from_l3=True) + load_to_device
    _any_mamba_restore = True
  set_extend_range → length=1 (only 1 new token)
  REPLAY-FALLBACK:
    _is_restored = (1 <= 2) = True
    _mamba_failed = False (L3 restore succeeded)
    _is_restored stays True → replay_tokens=0 ✓
```

## 事件时间线（B1 KDA=40，L1-only，修复后）

```
Build Phase (96 sessions):
  write_through → backup_from_device → L2 store → L2 evict (cap=0) → L3 skip (disabled)
  host_value: None (L2=0, slot immediately evicted)

Reuse Phase (24 sessions):
  match_prefix → prefix_len=1024 (KV cache hit)
  init_load_back → creates load op
  _mock_start_loading:
    L2 check: mh.has(slot_hash) → False (L2=0)
    L3 check: st is None (HISIM_DISABLE_L3=1) → skip
    _any_mamba_restore = False
    mamba_restore_failed = True
  set_extend_range → length=1
  REPLAY-FALLBACK:
    _is_restored = (1 <= 2) = True (KV cache)
    _mamba_failed = True (MAMBA not restored)
    _is_restored = False (overridden by MAMBA check)
    replay_tokens = 1024 (NOT restored) ✓
    historical_replay_tokens += 1024
```

## P1: TTFT 时间模型

### 当前状态
- 全部实验 `mean_ttft_ms=382.8`, `P95=439.5` — 完全相同
- L2/L3 transfer 时延已记录（`kda_l3_load_dur_total=0.0241s`）但未反映在 TTFT 中

### 原因分析
1. **K3ProfilePredictor** 使用 input token 数估算 compute 时间，不包含 L2/L3 transfer
2. **虚拟时钟** 在 `wrapped_process_batch_result` 中通过 `StateManager.step_global_clock()` 累加 L2 backup/load + inference + L3 write overflow
3. 但 `--load-format dummy` 模式下 inference duration 为固定值，不随 KDA replay 量变化
4. KDA replay 的额外计算量未传递给 predictor

### Model Limitation
- **KDA Replay 开销**：当前 predictor 不接收 KDA replay token 数作为输入，无法反映 replay 对 TTFT 的影响
- **L2/L3 Restore 开销**：时延已记录在 StateManager 中，但未累加到 TTFT
- **异步重叠**：L2 backup 与 inference 可重叠，但当前模型未实现重叠逻辑

### 逐项耗时归因（Model Limitation）

| 成本类型 | 是否记录 | 是否反映在 TTFT | 状态 |
|----------|----------|----------------|------|
| MLA Prefix Hit | ✅ | ✅ (reduces new tokens) | 正常 |
| KDA COW | ✅ | ❌ (replay=0, but compute not reduced) | Model Limitation |
| KDA Replay | ✅ (24576 tokens) | ❌ (predictor 不接收) | Model Limitation |
| L2 Restore | ✅ (H2D timing) | ❌ (未累加到虚拟时钟) | Model Limitation |
| L3 Restore | ✅ (SSD read + H2D) | ❌ (未累加到虚拟时钟) | Model Limitation |
| L3 Write-through | ✅ (SSD write) | ❌ (overflow 只影响 next round) | 部分实现 |

## 完整实验结果

### B1: L1-only（KDA sweep，无 L2/L3）

| 指标 | KDA=40 | KDA=165 |
|------|--------|---------|
| Completed | 152 | 152 |
| MLA Prefix Hit | 24576 | 24576 |
| Prefix Reused Ratio | 0.158 | 0.158 |
| KDA COW Exec | 0 | 24 |
| KDA Checkpoint Hit | 0 | 24 |
| Historical Replay | **24576** | **0** |
| KDA Load Calls | 24 | 0 |
| KDA L3 Writes | 0 | 0 |
| MLA Prefill New | 131096 | 131096 |
| Mean TTFT (ms) | 382.8 | 382.8 |

### B2: HiCache 分层（L1=40，L2 sweep，L3 file）

| 指标 | L2=3 | L2=128 |
|------|------|--------|
| Completed | 152 | 152 |
| KDA COW Exec | 0 | 0 |
| Historical Replay | 0 | 0 |
| KDA Load Calls | 24 | 24 |
| KDA L3 Writes | 128 | 128 |
| KDA L2 Capacity | 3 | 128 |
| KDA L3 Load Dur | **0.0241** | **0.0** |
| A4-KDA-L3-LOAD | 24 | 0 |
| MLA L3 Load Dur | 4.5e-05 | 4.5e-05 |
| Mean TTFT (ms) | 382.8 | 382.8 |

## 结论

1. **L2 host_value 过期引用已修复**：只在 L2 slot 存在时设置 `host_value`
2. **Replay 统计已修正**：B1 KDA=40 从错误的 0 修正为 24576
3. **L2 容量正确区分 L2/L3 restore**：L2=3 → L3 restore (0.024s)，L2=128 → L2 restore (0.0s)
4. **TTFT 不反映 KDA/L2/L3 开销**：predictor 不接收 replay token 数 — Model Limitation
5. **Value-Aware Eviction 优先级**：L1-only 时 replay=24576（显著），有 L2/L3 时 replay=0（已消除）
