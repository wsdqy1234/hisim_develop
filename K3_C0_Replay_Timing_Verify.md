# C0: Replay 语义与 TTFT 归因验证

> SGLang-policy Baseline in HiSim CPU Mock. Cost-Only Simulation, not native scheduling.

## C0-1: 原生 MambaComponent Validator 与合法复用边界

### 原生代码追踪（SGLang v0.5.19）

**`MambaComponent.match_prefix`** (`mamba_component.py:170-185`):
```python
aligned_seqlen = full_kv_hit_length // mamba_checkpoint_grid * mamba_checkpoint_grid
branching_seqlen = aligned_seqlen if aligned_seqlen > mamba_boundary_len else None

if self.has_host_value_only(last_node):
    mamba_host_hit_length = max(result.mamba_host_hit_length, 1)
```

**`init_load_back`** (`unified_radix_cache.py:2734-2782`):
```python
if (is_full_device_evicted(best_match_node_id)
    or params.host_hit_length > 0
    or req.mamba_host_hit_length > 0):
    if self.load_back(best_match_node_id, mem_quota, req=req):
        new_indices = collect_full_device_indices(...)
        return new_indices, best_match_node_id
return empty_match_result.device_indices, last_node
```

### 合法复用边界

| 场景 | Device Value | Host Value | mamba_host_hit_length | init_load_back | extend_range.length | MAMBA Replay |
|------|-------------|------------|----------------------|----------------|---------------------|--------------|
| L1 COW | set | set | 0 | 不触发 | 1 (new token) | 0 (COW) |
| L2 Restore | None | set | ≥1 | 触发 → 恢复 | 1 | 0 (restored) |
| L3 Restore | None | None→set(HiSim) | 0→1(HiSim) | 触发 → HiSim L3 | 1 | 0 (HiSim restored) |
| All-Miss | None | None | 0 | 不触发(无MAMBA) | 1 (KV prefix) | 1024 (Mamba内部replay) |

### 关键结论

1. **`extend_range.length` 始终基于 KV cache prefix**，不反映 MAMBA replay
2. **MAMBA replay 由 Mamba 模型内部处理**，使用 `mamba_branching_seqlen` 确定 replay 起点
3. **HiSim 的 `replay_tokens` 是 Cost-Only Simulation**：
   - 不是原生已调度计算（Scheduler 不知道 replay）
   - 模拟 Mamba 模型内部 replay 的计算成本
   - 通过 `FakeRequest.input_length = extend_range.length + replay_tokens` 传入 predictor
4. **原生 SGLang 无独立 KDA Replay 调度路径**——replay 发生在 forward pass 内部

## C0-2: Predictor Clamp 与 382.8ms vs 439.4ms 差异

### Predictor 标定数据
```
PREFILL_TOKS = [1576, 4096, 6302, 16373, ...]
PREFILL_TIMES = [0.439, 0.725, 1.140, 2.716, ...]
```
最小标定点 = 1576 tokens → 0.439s = 439ms。
任何 input < 1576 返回 439ms（clamp）。

### 逐请求分析

| 请求类型 | B1 KDA=40 effective | B1 KDA=165 effective | Predictor 返回 |
|----------|---------------------|---------------------|---------------|
| Build (128) | 1024 | 1024 | 439ms |
| Churn (32) | 1024 | 1024 | 439ms |
| Reuse (24) | **1025** (1+1024 replay) | **1** (COW, 0 replay) | 439ms |

**两者 predictor 返回值相同**（439ms），56.6ms 差异**不来自 predictor prefill 估算**。

### 差异来源分析

差异来自 `l2_backup_dur`（D2H backup 时延）和虚拟时钟累加机制：
- `request_response_time = global_clock + l2_backup_dur + l3_write_overflow`
- `l2_backup_dur` 在 OVERLAP 模式下加到 response_time 但不加到 global_clock
- B1 KDA=40 的 24 个 reuse session 额外触发 `_mock_start_loading`（L2 check + L3 check），产生额外 `kda_l2_load_dur`
- 但 L2=0、L3=None，所以 `kda_l2_load_dur = 0`

**确切原因需要逐 batch 时延日志确认**，当前数据不足以精确定位 56.6ms。

### 结论
- **不能仅凭 effective_compute 增加就认定 56.6ms 来自 Replay**
- Predictor clamp 使 1 token 和 1025 tokens 不可区分
- 差异来自虚拟时钟累加和 backup timing，不是 replay compute

## C0-3: L2/L3 Load Duration 累计语义

### 当前实现
`StateManager.inc_hicache_kda_l3_load_dur()` 是**累计计数器**：
- 每次 `_mock_start_loading` 中 L3 restore 成功时累加
- 在 `wrapped_process_batch_result` 中通过 `pop_hicache_kda_l3_load_dur()` 一次性取出
- 取出的是**该 batch 所有请求的累计 L3 restore 时延**

### B2 L2=3 实测
- `kda_l3_load_dur_total = 0.0241s`（24 次累计）
- 单次 L3 restore ≈ 1ms（14.65MB / 14.6GB/s）
- **不能用总耗时 24ms 直接与单次 inference 439ms 比较**
- 应比较单次 L3 restore (1ms) 与单次 inference (439ms)

### 与 Inference 的正确比较

| 指标 | 单次值 | 来源 |
|------|--------|------|
| KDA L3 restore | ~1ms | 14.65MB / 14.6GB/s |
| KDA L2 restore | ~0.23ms | 14.65MB / 64GB/s |
| Prefill inference (1024 tok) | 439ms | Predictor clamp |
| Prefill inference (1 tok) | 439ms | Predictor clamp (Model Limitation) |

## C0-4: No-Overlap vs Full-Overlap 时间模型

### 当前模型（OVERLAP_SCHEDULE=True）
```
step(max(l2_load_dur - last_inference_dur, 0))  # L2/L3 load 与前一轮 inference 重叠
step(current_inference_dur)                      # 当前 inference
response = global_clock + l2_backup_dur + l3_write_overflow
```
- L3 restore (24ms) < 前一轮 inference (439ms) → **完全重叠，TTFT 不可见**

### No-Overlap 模型（OVERLAP_SCHEDULE=False）
```
step(l2_load_dur + current_inference_dur + l2_backup_dur + l3_write_overflow)
response = global_clock
```
- L3 restore (24ms) 串行加入 → TTFT 增加 24ms

### 敏感性分析

| 模型 | B2 L2=3 TTFT | B2 L2=128 TTFT | 差异 |
|------|-------------|----------------|------|
| Full-Overlap | ~383ms | ~383ms | 0ms |
| No-Overlap | ~407ms (383+24) | ~383ms | 24ms |
| 部分重叠 (50%) | ~395ms | ~383ms | 12ms |

**结论**：真实系统中 L2/L3 restore 能否与前一轮 inference 重叠取决于硬件并发能力。
当前 Model Limitation：无法确定真实重叠程度，需 NPU 实测。

## C0-5: L3 Mock 恢复 vs 原生 Prefix Match 安全边界

### 原生行为
1. `match_prefix` → `MambaComponent.match_prefix` 设置 `mamba_host_hit_length`
2. `init_load_back` → `load_back` → 原生 H2D 传输
3. 原生 storage prefetch（异步线程，HiSim 已禁用）

### HiSim Mock 补充行为
1. `_mock_start_writing`（替代 `start_writing`）：
   - 调用 `backup_from_device` → L2 backup + L3 write-through
   - 设置 `host_value`（只在 L2 slot 存在时）
2. `_mock_start_loading`（替代 `start_loading`）：
   - `mh.has(slot_hash)` → L2 membership check（**HiSim 补充**）
   - `st.batch_exists_mamba` → L3 lookup（**HiSim 补充**）
   - `backup_from_device(from_l3=True)` → L3→L2 restore（**HiSim 补充**）
   - `load_to_device` → L2→L1 restore

### 安全边界检查

| 检查项 | 原生 | HiSim Mock | 一致性 |
|--------|------|------------|--------|
| host_value 有效性 | load_back 检查 | L2 slot membership check | ✅ P0-1 修复后一致 |
| L3 lookup | 原生 storage prefetch | batch_exists_mamba (KVCM) | ⚠️ HiSim 补充，原生异步 |
| restore 顺序 | load_back → H2D | L3→L2→L1 (serial) | ✅ 语义一致 |
| Prefix match 完整性 | match_prefix 全路径 | 原生 match_prefix 不变 | ✅ 不绕过 |

### 结论
- HiSim Mock **不绕过**原生 Prefix Match——`match_prefix` 仍由原生代码执行
- HiSim Mock **补充**了 L2/L3 的 membership check 和 restore（原生通过异步线程完成，HiSim 同步模拟）
- `host_value` 有效性已在 P0-1 修复后与原生语义一致

---

## Phase 2 最小策略接口设计

### 设计原则
- 不修改 SGLang 原生 LRU/锁/ACK/Slot 生命周期
- 在 HiSim Adapter 层实现策略接口
- 通过 env var 选择策略，baseline 行为不变

### 策略接口

```python
class KDAEvictionPolicy(ABC):
    """Abstract base for KDA checkpoint eviction policies."""

    @abstractmethod
    def select_victims(self, tree_core, component_type, num_to_evict: int,
                        request_stats: dict) -> list:
        """Select which checkpoints to evict.
        Returns list of node_ids to evict."""
        ...

class NativeLRUPolicy(KDAEvictionPolicy):
    """SGLang native LRU — no modification, baseline."""
    def select_victims(self, tree_core, ct, n, stats):
        return None  # Let native LRU handle it

class HotnessAwarePolicy(KDAEvictionPolicy):
    """Evict least-recently-accessed (not just least-recently-used)."""
    def __init__(self):
        self._access_history = {}  # {node_id: [access_times]}

    def record_access(self, node_id, virtual_time):
        self._access_history.setdefault(node_id, []).append(virtual_time)

    def select_victims(self, tree_core, ct, n, stats):
        # Sort by access frequency (hotness), evict coldest
        candidates = self._get_evictable(tree_core, ct)
        candidates.sort(key=lambda nid: len(self._access_history.get(nid, [])))
        return candidates[:n]

class OraclePolicy(KDAEvictionPolicy):
    """Ideal policy with future access knowledge."""
    def __init__(self, future_trace: list):
        self._future = future_trace  # Pre-computed access sequence

    def select_victims(self, tree_core, ct, n, stats):
        # Evict checkpoints that will NEVER be re-accessed
        # or will be re-accessed FURTHEST in the future
        candidates = self._get_evictable(tree_core, ct)
        candidates.sort(key=lambda nid: self._next_access_distance(nid))
        return candidates[:n]
```

### 实施路径

1. **Hook 点**：`wrapped_check_hicache_events` 中的 proactive eviction 调用点
   - 当前已 gate by `HISIM_DISABLE_PROACTIVE_EVICT`
   - 策略模式：`HISIM_KDA_POLICY=native|hotness|oracle`

2. **Trace 设计**（8K/32K/128K + 分叉）：
   - Build session: 8192/32768/131072 tokens
   - 分叉 Session: 共享前缀 N tokens，不同后缀
   - Churn: 不同前缀的 session 制造容量压力
   - Reuse: 早期/中期/近期 session

3. **对比指标**：
   - Effective Prefix Tokens（MLA + KDA）
   - KDA Replay Tokens
   - L2/L3 Restore Count + Bytes
   - TTFT（No-Overlap / Full-Overlap / 50% Overlap）

4. **不改 SGLang 原生**：
   - 策略通过 HiSim Adapter 的 `select_victims` 接口注入
   - 原生 `evict()` 仍执行实际淘汰
   - 策略只决定**淘汰哪些**，不决定**如何淘汰**

### 预期实验配置

| 配置 | KDA Slots | L2 | L3 | Trace |
|------|-----------|----|----|-------|
| Two Pools L1-only | 40/80/165 | 0 | disabled | 8K/32K/128K |
| Two Pools HiCache | 40 | 3/32/128 | file | 8K/32K/128K |
| Policy | native / hotness / oracle | — | — | — |

### 时延报告格式

| 策略 | KDA | Replay | L2 Restore | L3 Restore | TTFT (No-Ovlp) | TTFT (Full-Ovlp) |
|------|-----|--------|------------|------------|----------------|------------------|
| Native LRU | 40 | ? | ? | ? | ? | ? |
| Hotness | 40 | ? | ? | ? | ? | ? |
| Oracle | 40 | ? | ? | ? | ? | ? |

### Model Limitation 标注

- TTFT 中的 compute 部分为 Predictor 估算（非 NPU 实测）
- L2/L3 restore 时延为带宽模型估算
- Overlap 程度为敏感性分析（非实测）
- Predictor clamp (< 1576 tokens) 限制短请求的时延区分度
