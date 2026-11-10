# Phase 2d-Closure 验证报告：Token 守恒、Read Completion 因果和 Write Async 对照

> 日期：2026-10-28
> SGLang 0 修改，51 单元测试通过，V1a 算法不变

---

## P0-1：Token 守恒 ✅ 已修复

### 问题根因

之前的 `New` 计算公式有三个错误：
1. `l1 = final_reused_tokens - l2_hit_tokens` 错误（`cached_tokens` 不包含 `host_hit_length`）
2. `l3_hit_tokens`（existence count=128）被从 prefill 中减去（但只有 7 tokens 实际被消费）
3. `input_length=512`（原始），不是 `506`（truncate 后）

### 修复后的公式

```
actual_input = len(origin_input_ids)  # 506 (truncated)
L1_device = final_reused_tokens       # 0 (cached_tokens, device-resident only)
L2_host = l2_hit_tokens               # 7 (host_hit_length, includes L3-restored)
Recomputed = actual_input - L1_device - L2_host  # 506 - 0 - 7 = 499
```

**互斥归因**：`L1_device + L2_host + Recomputed = actual_input`
```
0 + 7 + 499 = 506 ✅ (conservation holds)
```

### `l3_restored=7` vs `l3_exists=128` 差异解释

| 字段 | 值 | 含义 |
|---|---|---|
| `l3_exists_tokens` | 128 | kvcm `batch_exists` 找到 128 consecutive keys（kvcm 容量限制，非全部 505） |
| `l3_read_issued_tokens` | 128 | `calc_prefetch_pages` 发起 128 tokens SSD read |
| `l3_read_completed_tokens` | 128 | `batch_get` 成功读取 128 tokens |
| `pf_complete` | 128 | `prefetch_complete_tokens = 128` |
| `l3_restored_tokens` | 7 | `_handle_prefetch_result`: `loaded = completed(128) - prefix_len(121) = 7` |
| `l3_consumed_tokens` | 7 | `host_hit_length = 7`（只有 7 个有效 host_value 被 match_prefix 找到） |

**差异原因**：
- kvcm 只找到 128/505 keys（容量限制导致部分 key 被淘汰）
- 128 tokens 读取完成，但 121 个已有 host_value（L2 backup 残留），只有 7 个新增
- `match_prefix` 只找到 7 个有效 host_value（121 个可能指向已释放的 L2 slots）

**关键**：`l3_exists_tokens` 只能作为查询指标，**不能算实际复用**。实际复用 = `l3_consumed_tokens = 7`。

### 所有请求的守恒验证

| 请求 | L1 | DRAM | Recomputed | Conservation | 状态 |
|---|---|---|---|---|---|
| Warmup | 0 | 0 | 6 | 0+0+6=6/6 | ✅ |
| R0 Build | 0 | 0 | 506 | 0+0+506=506/506 | ✅ |
| R1 Evict | 0 | 0 | 506 | 0+0+506=506/506 | ✅ |
| R2 Evict | 0 | 0 | 506 | 0+0+506=506/506 | ✅ |
| R0 Reuse | 0 | 7 | 499 | 0+7+499=506/506 | ✅ |

**所有请求守恒一致，无负数、无重复统计、无缺失 Tokens。**

---

## P0-2：L3 Read Completion 时序因果 ✅ 已验证

### 事件时间戳

```
R0 Reuse L3-CAUSALITY:
  issue=3000.000ms     (SSD read 发起)
  complete=3001.835ms  (SSD read 完成 = issue + read_dur=1.835ms)
  restore=3005.000ms   (KV 插入 radix tree)
  consume=3005.000ms   (请求消费 restored KV)
  consume >= complete = OK ✅
```

### 因果关系验证

```
consume_time (3005ms) >= read_complete_time (3001.835ms) ✅
```

请求在 SSD read 完成后才消费 restored KV。不存在提前使用未完成数据的情况。

### 时间不重复计入

| 时序项 | 值 | 计入 clock | 说明 |
|---|---|---|---|
| L3 read (SSD) | 1835.0μs | ❌ 否 | best_effort 与 inference overlap |
| L2 load-back (H2D) | 65.7μs | ✅ 是 | 恢复数据从 host 加载到 device |
| L3 write overflow | 4446.3μs (barrier) | ✅ 是 | SSD 写入超出 |
| Inference | 2.734ms | ✅ 是 | 实际计算 |

L3 read 和 L2 load-back **不重复计入** clock。✅

### best_effort 语义

在 best_effort 模式下：
- L3 read 与 inference overlap（不阻塞）
- 若 read 在本轮 budget 内完成 → 数据下一轮可用
- 若 read 未完成 → 数据不可用 → 请求正常重算（不提前使用未完成数据）

---

## P0-3：Write Barrier vs Async Write ✅ 已实现和对照

### 实现方式

```python
_l3_write_mode = os.environ.get("HISIM_L3_WRITE_MODE", "barrier")
if _l3_write_mode == "async":
    l3_write_overflow = 0.0  # SSD writes proceed in background
else:
    l3_write_overflow = max(0, l3_backup_dur - round_time)  # Barrier
```

### A/B 对照结果（512-token, 4 请求）

| 指标 | Barrier | Async | 差异 |
|---|---|---|---|
| R1 PREFILL l3_ovf | 4512.0μs | 0.0μs | 4.5ms |
| R2 PREFILL l3_ovf | 4512.0μs | 0.0μs | 4.5ms |
| R0 Reuse l3_ovf | 4446.3μs | 0.0μs | 4.4ms |
| R0 Reuse TTFT | 12.3ms | 7.8ms | **4.5ms** |
| Token Conservation | 0+7+499=506 | 0+7+499=506 | 相同 ✅ |
| L3 read time | 1835.0μs | 1835.0μs | 相同 ✅ |
| L3 restored | 7 | 7 | 相同 ✅ |

### 敏感性分析

**当前长 Prefix 性能收益对 Write Barrier 假设高度敏感**：
- 每个 batch 的 l3_ovf = 4.5ms（512-token 工作负载）
- 4K/16K 场景的 l3_ovf = 53-232ms（之前测量）
- Async 模式下这些 overflow 全部消除 → TTFT 大幅降低

**结论**：4K/16K 的巨大性能差异主要来自 Write Barrier 的 overflow，不是 HBF 介质优势。Async Write 模式下，性能差异预计大幅缩小。

---

## P1：A/B/C Recompute 场景

### 场景 B（L3 restore）✅ 已验证

```
R0 Reuse: L1=0, DRAM=7 (L3 consumed), Recomputed=499
Conservation: 0+7+499=506/506 ✅
L3 read wait: 1.835ms (不直接计入 clock, best_effort overlap)
L2 load-back: 65.7μs (计入 clock)
TTFT: 12.3ms (barrier) / 7.8ms (async)
```

**验证**：L3 restore 减少 prefill 从 506 到 499（节省 7 tokens），读取等待正确计时。

### 场景 A（L2 命中）和 C（Backup 失败）— 待测

- **A (L2 命中)**：需要 L2 容量 ≥ 2 个请求（hicache_ratio ≥ 2.0），使 backup 保留在 L2
- **C (Backup 失败)**：需要 HBF on + 小 HBF（adapter.alloc 失败），验证 full recompute

这些场景需要额外的容量配置测试，在 4K/16K 公平实验中完成。

---

## 修改文件清单

| 文件 | 修改内容 |
|---|---|
| `types.py` | 新增 `actual_input_tokens`, `recomputed_tokens`, L3 causality timestamps |
| `sglang_hook.py` | Token conservation fix, L3 causality tracking, Async Write mode, recomputed_tokens from conservation |
| `sglang_mock_class.py` | (前阶段修改) batch_get Metadata-only Restore, exists() kvcm fix |
| `state.py` | (前阶段修改) hicache_l3_load_dur |

---

## 不变量

- ✅ SGLang 原生代码 **0 修改**
- ✅ **51** 单元测试全通过
- ✅ HBF-off 基线不回归
- ✅ V1a 算法未修改

## 验收总结

| 验收项 | 状态 | 说明 |
|---|---|---|
| Token 来源互斥统计完全对账 | ✅ | 0+7+499=506/506，解释 L1=-7→0、Restored=7、Read=128 |
| SSD Read 完成时间与消费时间因果正确 | ✅ | consume(3005ms) >= complete(3001.8ms) |
| Write Barrier / Async Write 模式区分 | ✅ | Barrier: l3_ovf=4.5ms, Async: l3_ovf=0ms |
| A/B/C 场景 Prefill 计算量与时延一致 | ⚠️ | B 已验证(499 tokens), A/C 待 4K 实验 |
| SGLang 0 修改, V1a 不变, 51 测试通过 | ✅ | 全部通过 |
