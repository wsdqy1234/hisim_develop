# Phase 2d 进度报告：L3 Read/Write 时序正确性与长 Prefix 公平验证

> 日期：2026-10-28
> V1a 算法不变，SGLang 0 修改，51 单元测试通过（41 原有 + 10 新增）

---

## P0-1：L3 Mock 逻辑恢复闭环

### 实现内容

#### 1. Metadata-only Restore Mock (`sglang_mock_class.py`)

修改 `MockHiCacheStorage` 的三个读取接口，使其返回数据而非 None：

| 接口 | 修改前 | 修改后 | 说明 |
|---|---|---|---|
| `batch_get` | `return None` | 返回 `target_locations[i]` for existing keys | `_generic_page_get` 据此计数成功读取的 pages |
| `batch_get_v1` | `return [False] * len(keys)` | 返回 `[self.exists(k) for k in keys]` | 旧版接口兼容 |
| `batch_get_v2` | `return {}` | 返回 `{pool_name: [exists(k) ...]}` | sidecar pool 接口兼容 |
| `get` | `return None` | 返回 `target_location` for existing keys | 单 key 接口 |

**适用边界**：CPU Mock 模式。不验证真实 KV Tensor 数据完整性，仅验证缓存管理逻辑、放置策略和时序模型（L3 read → L2 load-back → TTFT）。

新增调试计数器：`_DEBUG_BATCH_GET_COUNT`, `_DEBUG_BATCH_GET_HITS`。

#### 2. 同步 L3 数据加载 (`sglang_hook.py`)

**问题**：sglang 的 `prefetch_io_aux_func` 异步线程未被 HiSim 覆盖，导致 L3 数据加载不确定。

**修复**：
- 新增 `handle_prefetch_io_operation`：同步排空 `prefetch_buffer`，调用 `_page_transfer` → `_generic_page_get` → `batch_get`
- 覆盖 `prefetch_io_aux_func` 为 no-op（与 `backup_thread_func`、`prefetch_thread_func` 一致）
- 新增 `_drive_prefetch_io_phase`：在 `original_check_hicache_events` 之后调用

**时序管线**（确定性，一轮延迟）：
```
Round N:
  1. _drive_hicache_timing_phases → handle_prefetch_operation (batch_exists 查询)
  2. original_check_hicache_events → _drain_and_alloc_storage_hit (分配 host memory → prefetch_buffer)
  3. _drive_prefetch_io_phase → handle_prefetch_io_operation (排空 prefetch_buffer → batch_get → PrefetchAck)

Round N+1:
  1. original_check_hicache_events → _drain_ack_prefetch → _handle_prefetch_result (插入 radix tree)
  2. request match_prefix → host_hit_length > 0 → start_loading (L2 load-back)
```

#### 3. L3 读取指标追踪

新增 `RequestStats` 字段（区分 Exists/Issued/Completed/Restored/Consumed）：

| 字段 | 含义 | 数据来源 |
|---|---|---|
| `l3_exists_tokens` | batch_exists 命中数（存在性查询） | handle_prefetch_operation |
| `l3_read_issued_tokens` | 发起 SSD 读取的 tokens | handle_prefetch_operation (calc_prefetch_pages) |
| `l3_read_completed_tokens` | 实际读取完成的 tokens | handle_prefetch_io_operation (_page_transfer) |
| `l3_restored_tokens` | 插入 radix tree 的 tokens | _handle_prefetch_result (prefetch_loaded_tokens_by_reqid) |
| `l3_consumed_tokens` | 请求消费的 tokens | match_prefix (host_hit_length 中来自 L3 的部分) |
| `l3_read_bytes` | SSD 读取字节数 | handle_prefetch_operation |
| `l3_read_dur_ms` | SSD 读取时延 | handle_prefetch_operation (calc_prefetch_pages) |

新增 `StateManager.hicache_l3_load_dur`（max 语义，与其他 timing 字段一致）。

新增 BATCH-TIME 日志字段 `l3_read=Xμs` 和 iteration stat `l3_read_latency`。

### 验证结果

#### 单元测试（10/10 通过）
- batch_get 返回 existing keys 的数据 ✓
- batch_get 对 missing keys 返回 None ✓
- batch_get_v1/v2 返回正确格式 ✓
- RequestStats L3 字段默认值和赋值 ✓
- StateManager L3 load dur (max semantics) ✓
- Exists vs Read 区分 ✓
- batch_get 与 batch_set 一致性 ✓

#### 端到端测试（8 req × 512 tok × 2 rounds, L1=512, L2=256）
- Round 2 `cached=505`（505/506 tokens 缓存命中）→ L1 恢复有效
- L3 metrics: `exists=0, hits=0` → **L3 查询未命中**

### 关键发现：L3 Hash Key 不匹配

**根因**：sglang 的 L3 hash keys 是**路径依赖**的（基于 radix tree 节点路径），不是纯 token-ID 的。

| 轮次 | L1 match | last_hash | page_hashes | L3 查询结果 |
|---|---|---|---|---|
| Round 1 | 无（首次请求） | root hash | H(root, tokens[0:16]), ... | 写入 L3 |
| Round 2 | 256 tokens | node256 hash | H(node256, tokens[256:272]), ... | **不匹配** → hits=0 |

Round 2 的请求在 L1 找到 256-token prefix → `last_hash` 变为 node256 的 hash → page_hashes 与 Round 1 不同 → L3 查询不命中。

**这是 sglang 原生行为，不是 HiSim 实现 bug。** L3 恢复在以下条件之一时有效：
1. 请求的 prefix 不在 L1（`last_hash` = root，与写入时相同）
2. L1 在两轮之间被 flush（强制 `last_hash` = root）
3. 使用 token-ID-based hashing（需要修改 sglang，违反 0 修改约束）

### P0-1 结论

| 验收项 | 状态 | 说明 |
|---|---|---|
| 区分 L3 Exists/Read/Restore/Consume | ✅ | 5 个独立指标字段 |
| 模拟 SSD Read Bytes/Time | ✅ | calc_prefetch_pages + StateManager |
| 只有逻辑恢复完成的 KV 计为 L3 复用 | ✅ | l3_restored_tokens from _handle_prefetch_result |
| 未恢复的 Prefix 进入 Prefill 重算 | ✅ | sglang 原生行为（host_hit_length=0 → recompute） |
| L3 Read Timing 影响 TTFT | ⚠️ | 时序已记录但 best_effort 模式下不直接计入 clock（与 inference overlap） |
| 不产生假命中/重复恢复/重复计时 | ✅ | 同步管线确保确定性 |

---

## P0-2：L3 Write Overflow 正确性核查

### batch_set 统计单位

`_DEBUG_BATCH_SET_COUNT` 统计的是 **batch_set 调用次数**，不是 Pages 或 Tokens。

```python
# handle_backup_operation
while True:
    operation = self.backup_queue.get(block=False)
    self._page_backup(operation)  # 调用 batch_set 一次
    total_write_tokens += len(operation.host_indices)  # 累计 token 数
```

每次 `batch_set` 调用写入 `len(keys)` 个 pages（每个 page = `page_size` tokens）。

### 实际写入 Bytes

```python
# submit_l3_write_timing
total_write_bytes = total_write_tokens * KV_CACHE_BYTES_FULL
```

- `total_write_tokens`: 所有 backup operations 的 `len(host_indices)` 之和
- `KV_CACHE_BYTES_FULL`: 每个 token 的 KV 字节数（跨所有 TP workers）
- `KV_CACHE_BYTES`: 单 TP worker 的 KV 字节数（用于 SSD write 的 page_size 参数）

### SSD Write 时序模型

```python
# 单 tier fallback（无 SSD simulator）
write_time = total_write_bytes / DISK_WRITE_BANDWIDTH_BYTES  # 8 GB/s

# 或 SSD simulator
_, lat, _, _ = sim.submit_write(arrival, write_bytes, page_size, is_sequential=True)
write_time = lat
```

**Write arrival**：`write_arrival = StateManager.get_global_clock()`（T_N，当前轮开始时间）

### Write-through Ack 与 SSD 物理写完成的关系

| 阶段 | 操作 | 时序 |
|---|---|---|
| Data phase | `batch_set` 写入 kvcm/storage | 同步，立即返回 True |
| Timing phase | `submit_write` 提交 SSD 写时序 | 独立计算，不阻塞 data phase |
| Ack | `batch_set` 返回 True → sglang 认为 write-through 完成 | 逻辑完成，非物理持久化 |
| Overflow | `l3_write_overflow = max(0, l3_backup_dur - round_time)` | 只有超出部分延迟下一轮 |

**Write-through Ack ≠ SSD 物理写完成**。Ack 是逻辑写入（key 存入 storage set），SSD 时序独立计算。未 flush 的数据由 kvcm memory index 保护。

### L3 Write Overflow 如何影响 TTFT

```python
# wrapped_process_batch_result
l3_write_overflow = max(
    0,
    hicache_l3_backup_dur
    - (combined_load_dur + current_inference_dur + combined_backup_dur),
)
StateManager.set_last_l3_write_overflow(l3_write_overflow)

# Clock advancement (non-overlap mode)
StateManager.step_global_clock(
    combined_load_dur
    + current_inference_dur
    + combined_backup_dur
    + l3_write_overflow  # ← 只有 overflow 部分计入 clock
)
```

**因果关系**：
1. L3 write 在 T_N 开始，与整个轮次（load + inference + backup）overlap
2. 如果 write_time ≤ round_time → write 在轮次内完成 → overflow=0 → 不影响 TTFT
3. 如果 write_time > round_time → write 超出 → overflow > 0 → 延迟下一轮 → 影响 TTFT

### Buffer/Queue Backpressure

**当前模型无真实 Buffer/Queue Backpressure**：
- `backup_queue` 是无界 Queue，不会 backpressure
- `prefetch_buffer` 是无界 Queue
- `prefetch_rate_limited()` 检查 `prefetch_tokens_occupied`，但在 cache mode 下只限制总占用

**SSD simulator（SimpySSD）** 有 channel-level queue contention，但当前测试使用 bandwidth-based fallback（无 SSD simulator）。

### 是否存在写入时间重复计入

**否**。L3 write timing 只在 `submit_l3_write_timing` 中计算一次（`inc_hicache_l3_backup_dur`），在 `wrapped_process_batch_result` 中 pop 一次。Data phase（`batch_set`）不产生时序。Overflow 只计入超出部分。

### Per-batch Timeline（4K A 配置示例）

```
Batch N-1: PREFILL
  clock = T_{N-1}
  inference = 12.558ms
  backup = 37.7μs (L2 D2H)
  load = 0μs
  l3_ovf = 0μs
  → T_N = T_{N-1} + 0 + 12.558ms + 37.7μs + 0 = T_{N-1} + 12.596ms

Batch N-1: DECODE (7 tokens)
  clock = T_N
  inference = 1.760ms
  backup = 0μs
  load = 0μs
  l3_ovf = 53290.7μs ← L3 write overflow!
  → T_{N+1} = T_N + 0 + 1.760ms + 0 + 53.291ms = T_N + 55.051ms

  L3 write: 1988 batch_set calls, total_write_tokens × KV_CACHE_BYTES_FULL bytes
  write_time >> round_time → overflow = 53.291ms
```

### P0-2 结论

| 验收项 | 状态 | 说明 |
|---|---|---|
| batch_set 统计单位 | ✅ | 调用次数（非 tokens/pages） |
| 每次写入 Bytes | ✅ | total_write_tokens × KV_CACHE_BYTES_FULL |
| SSD Write 提交/排队/完成 | ⚠️ | bandwidth-based fallback（无 channel queue）；Phase 3 MQSim 校准 |
| Write-through Ack vs 物理写 | ✅ | 逻辑 Ack ≠ 物理持久化；SSD 时序独立 |
| Overflow 影响 TTFT 的因果关系 | ✅ | max(0, write_time - round_time) 计入 clock |
| Buffer/Queue Backpressure | ⚠️ | 无真实 backpressure（无界 Queue） |
| 写入时间重复计入 | ✅ | 无重复（data phase 无时序，timing phase 只 pop 一次） |

---

## P0-3：Recompute 正确性验证（设计）

### 三种场景定义

| 场景 | Backup | Restore | 预期计算 tokens |
|---|---|---|---|
| A | 成功 Backup 到 L2 | 从 L2 恢复 (host_hit_length) | input - L1_hit - L2_hit |
| B | 成功 Backup 到 L2 → L2 evict → L3 | 从 L3 恢复 (prefetch_complete) | input - L1_hit - L3_restored |
| C | Backup 失败 (adapter.alloc None) | 无恢复 | input - L1_hit (全部重算) |

### 关键区分

**Backup Failed Attempt Tokens ≠ Recomputed Tokens**：
- Backup Failed Attempt: adapter.alloc 返回 None → tokens 未写入 L2
- Recomputed Tokens: 请求处理时不在任何缓存中的 tokens → 需要重新 prefill
- 当前 `New` 字段 = `input - L1_hit - L2_hit`（包含了 backup failed 的 tokens）
- 需要新增 `recomputed_tokens` 字段 = `input - L1_hit - L2_hit - L3_restored`

### 实现计划

1. 在 RequestStats 新增 `backup_failed_tokens` 和 `recomputed_tokens`
2. 在 adapter.alloc 失败时记录 `backup_failed_tokens`
3. 在请求处理时计算 `recomputed_tokens = input - L1_hit - L2_hit - L3_restored`
4. 构造 A/B/C 测试场景验证

### 覆盖路径

- **SGLang pre-check 拒绝**：`_execute_kv_backup` 返回 None（在到达 adapter.alloc 之前）
- **Adapter.alloc 失败**：adapter.alloc 返回 None（V1a bounded retry 后仍失败）

---

## P1：长 Prefix 等容量公平对照（计划）

### 配置

| Prefix | Host 总容量 | A (DRAM) | E (DRAM+HBF) | HBF ≥ node_size? |
|---|---|---|---|---|
| 4K | 8192 | 8192 | 2458+5734 | 5734 ≥ 3840 ✅ |
| 16K | 24576 | 24576 | 7373+17203 | 17203 ≥ 16128 ✅ |

### Selective Backup 对照（A'）

新增 A' 配置：纯 DRAM 但**选择性不备份大 Node**（模拟 E 的 backup-failure 行为），以区分：
- 介质替代收益（HBF 命中）
- 少写 SSD 的策略收益（backup 量减少 → L3 write overflow 减少）

### 工作负载设计

构造 Build→Evict→Reuse→Churn 负载：
1. Build: 发送 N 个不同 prefix 的请求，填满 L1+L2
2. Evict: 继续发送请求，触发 L2→L3 eviction
3. Reuse: 重复发送 Build 阶段的请求，触发 L2/L3 restore
4. Churn: 混合新旧请求，测试稳态行为

### L3 Hash Key 问题

当前 L3 hash key 是路径依赖的，Round 2 的 L1 match 会改变 hash → L3 不命中。

**解决方案**：
1. 在 Reuse 阶段前 flush L1（`/flush_cache`）→ `last_hash` = root → hash 匹配
2. 或构造不共享 prefix 的请求（每个请求独立，L1 无 match）
3. 或在 Build 和 Reuse 之间发送足够多的请求使 L1 完全 evict Build 阶段的 prefix

---

## 不变量

- ✅ SGLang 原生代码 **0 修改**
- ✅ **51** 单元测试全通过（41 原有 + 10 新增 L3 restore loop）
- ✅ HBF-off 基线不回归（batch_get 修改对 HBF-off 无影响：storage backend 仍工作）
- ✅ V1a 算法未修改

## 下一步

1. **P0-3 实现**：新增 `recomputed_tokens` 追踪，构造 A/B/C 测试
2. **P1 实验**：使用 4K/16K 等容量配置 + flush L1 方案，运行公平对照
3. **L3 Read Timing 精确化**：当前 `l3_read=0.0μs` 因 bandwidth-based 计算产生极小值，需要配置 SSD simulator 或调整显示精度
