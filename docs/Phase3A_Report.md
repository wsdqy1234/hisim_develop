# Phase 3A 报告：真实 HBF/DRAM Host Hit 负载与 A/B/C 场景验证

> 日期：2026-10-29
> SGLang 0 修改，51 单元测试通过，V1a 算法不变

---

## 1. 工作负载设计

### Shared-Prefix Build→Evict→Reuse

```
R0: [shared S] + [unique U] = N tokens  (Build, backup to Host)
R1: [shared S] + [unique U] = N tokens  (Evict R0 from L1, R0 stays in Host)
R0 Reuse: same as R0                      (Host Hit → load-back)
```

关键设计：R0 和 R1 共享 prefix，R1 驱逐 R0 的 unique 部分从 L1，但 shared prefix 保留在 L1（R1 在用）。R0 的 unique 部分被 backup 到 Host，Reuse 时从 Host 恢复。

### 容量配置（等容量公平）

| Config | 512-token | 4K |
|---|---|---|
| A (DRAM) | hicache_ratio=2.0, DRAM=1025 | hicache_ratio=2.0, DRAM=8192 |
| E (HBF+DRAM) | hicache_ratio=0.6, DRAM=308+HBF=717, Total=1025 | hicache_ratio=0.6, DRAM=2458+HBF=5734, Total=8192 |
| C (Backup Fail) | hicache_ratio=0.3, DRAM=154+HBF=100, Total=254 | N/A |

**关键修复**：E 配置必须使用 `hicache_ratio=0.6`（不是 2.0），否则 DRAM 池不会被 HBF 扣减，总容量会膨胀到 1742（而非 1024）。

---

## 2. 核心结果：Host Hit 验证

### 512-token 实验

| 指标 | A_512 (DRAM) | E_512 (HBF) | 差异 |
|---|---|---|---|
| **HBF hit tokens** | 0 | **239** ✅ | HBF 命中！ |
| **DRAM hit tokens** | **239** ✅ | 0 | DRAM 命中！ |
| HBF load bytes | 0 | 27,410,432 | |
| DRAM load bytes | 27,410,432 | 0 | |
| **Host Load time** | **2030.8μs** | **53.5μs** | **38x 加速** |
| TTFT (R0 Reuse) | 29.1ms | 27.2ms | **1.9ms 节省** |
| Conservation | 240+239+1=480 ✅ | 240+239+1=480 ✅ | 守恒正确 |
| SchedPF | 1 | 1 | 与 Recomputed 一致 ✅ |
| Total Host KV | 1025 | 1025 | 等容量 ✅ |

### 4K 实验

| 指标 | A_4K (DRAM) | E_4K (HBF) | 差异 |
|---|---|---|---|
| **HBF hit tokens** | 0 | **2039** ✅ | HBF 命中！ |
| **DRAM hit tokens** | **2039** ✅ | 0 | DRAM 命中！ |
| HBF load bytes | 0 | 233,848,832 | |
| DRAM load bytes | 233,848,832 | 0 | |
| **Host Load time** | **17,224.9μs** | **456.7μs** | **37.7x 加速** |
| TTFT (R0 Reuse) | 67.6ms | 50.9ms | **16.7ms 节省** |
| Conservation | 2040+2039+1=4080 ✅ | 2040+2039+1=4080 ✅ | 守恒正确 |
| SchedPF | 1 | 1 | 与 Recomputed 一致 ✅ |
| Total Host KV | 8192 | 8192 | 等容量 ✅ |

### 时序验证

| 参数 | 值 | 来源 |
|---|---|---|
| bytes_per_token | ~114,688 bytes (112 KB) | Qwen3-0.6B: 28 layers × 8 kv_heads × 128 head_dim × 2 (K+V) × 2 (FP16) |
| DRAM bandwidth | 16 GB/s × 0.85 = 13.6 GB/s | config: memory_read_bandwidth_gb=16 |
| HBF bandwidth | 512 GB/s | HBFConfig: read_bandwidth_gb=512 |
| 带宽比 | 512 / 13.6 ≈ 37.6x | HBF 比 DRAM 快 ~38x |

**DRAM load (239 tokens)**: `6.67μs + 239 × 114688 / 13.6e9 = 6.67μs + 2.016ms = 2.023ms ≈ 2030.8μs` ✅
**HBF load (239 tokens)**: `max(10μs, 239 × 114688 / 512e9) = max(10μs, 53.5μs) = 53.5μs` ✅
**DRAM load (2039 tokens)**: `6.67μs + 2039 × 114688 / 13.6e9 = 17.19ms ≈ 17224.9μs` ✅
**HBF load (2039 tokens)**: `max(10μs, 2039 × 114688 / 512e9) = 456.7μs` ✅

---

## 3. A/B/C 场景验证

### Scenario A (L2 Host Hit) ✅

**描述**：KV 成功保存到 L2，下一次请求通过 HBF 或 DRAM 直接命中。

**验证**：
- R0 backup to Host (HBF or DRAM) ✅
- R0 Reuse: host_hit=239 (512) or 2039 (4K) ✅
- Prefill reduced: Recomputed=1 (only last token) ✅
- SchedPF=1 matches Recomputed ✅

### Scenario B (L3 Restore) ✅ (Phase 2d 已验证)

**描述**：KV 从 L2 淘汰到 L3，完成读取后被请求复用。

**验证**（Phase 2d-Closure）：
- L3[exists=128 consumed=7 restored=7] ✅
- Conservation: 0+7+499=506/506 ✅
- Causality: consume(3005ms) >= complete(3001.8ms) ✅

### Scenario C (Backup Fail → Full Recompute) ✅

**描述**：KV Backup 失败，没有其他有效缓存副本，后续请求需要重新 Prefill。

**配置**：HBF=100, DRAM=154, available_size=MAX(100,154)=154 < 480 → alloc fails

**验证**：
- R0 Reuse: `L1=240 HBF=0 DRAM=0 L3[exists=0 consumed=0] Recomputed=240 SchedPF=240`
- **无 Host Hit**（HBF=0, DRAM=0，无 HOST-LOAD 日志）✅
- **无 L3**（config.json 无 storage_tiers，L3[exists=0]）✅
- **Full Recompute of unique suffix**（Recomputed=240）✅
- Conservation: 240+0+240=480/480 ✅
- SchedPF=240 matches Recomputed ✅

**注意**：shared prefix (240 tokens) 仍在 L1（R1 在用），只有 unique suffix (240 tokens) 被重算。

---

## 4. Scheduler Prefill 独立核对 ✅

新增 `scheduler_prefill_tokens` 字段，从 `get_extend_input_len(req)` 独立获取，与守恒公式交叉验证：

| 场景 | Recomputed (守恒) | SchedPF (调度器) | 一致？ |
|---|---|---|---|
| A_512 R0 Build | 480 | 480 | ✅ |
| A_512 R0 Reuse | 1 | 1 | ✅ |
| E_512 R0 Reuse | 1 | 1 | ✅ |
| A_4K R0 Reuse | 1 | 1 | ✅ |
| E_4K R0 Reuse | 1 | 1 | ✅ |
| C_512 R0 Reuse | 240 | 240 | ✅ |

**守恒公式修复**：`L1_device = cached_tokens - l2_hit`（当 `cached_tokens >= l2_hit` 时）
- 之前：`L1=479` (错误，包含 host_hit)，守恒 `479+239+0=718≠480` ❌
- 之后：`L1=240` (正确，仅 device-resident)，守恒 `240+239+1=480=480` ✅

---

## 5. 收益来源区分

| 收益来源 | 存在？ | 证据 |
|---|---|---|
| **HBF 介质时延收益** | ✅ **存在！** | HBF load 38x 快于 DRAM (53.5μs vs 2030.8μs) |
| HBF 容量/命中率收益 | ❌ 不存在 | Hit tokens 相同 (239 or 2039) |
| 少写 SSD 策略收益 | ❌ 不存在 | 无 L3 写入 (Host Hit 不经过 L3) |
| Write Barrier 假设 | ❌ 不适用 | 无 L3 overflow (Host Hit 不触发 L3 write) |

**关键发现**：在真实 Host Hit 工作负载下，HBF 的介质时延优势**显著存在**：
- 512-token: 1.9ms TTFT 节省
- 4K: 16.7ms TTFT 节省
- 加速比 ~38x，与带宽比 (512/13.6≈37.6) 一致

---

## 6. HBF→HBM 有效互连带宽

当前模型中 HBF load 使用 **聚合带宽 512 GB/s**，这可能不代表真实端到端恢复带宽：

| 因素 | 当前模型 | 真实约束 |
|---|---|---|
| HBF 聚合带宽 | 512 GB/s | ✅ (HBFSim nominal) |
| HBF→HBM 互连 | 未建模 | PCIe/NVLink 带宽限制 |
| 并发数 | 无限 (max 语义) | Queue Depth 限制 |
| HBF 内部 HBM cache | 未建模 | 8GB cache 命中率 |

**Phase 3B MQSim 需要解决的问题**：
1. HBF→HBM 有效带宽受互连限制（不能直接用 512 GB/s）
2. 并发数受 Queue Depth 限制（不能无限并发）
3. HBF 内部 HBM cache 命中率影响延迟

---

## 7. 代码修改

| 文件 | 修改 | 说明 |
|---|---|---|
| types.py | +`scheduler_prefill_tokens` 字段 | 独立调度器 prefill 统计 |
| sglang_hook.py | +`scheduler_prefill_tokens` 赋值 | 在 `wrapped_get_new_batch_prefill` 中设置 |
| sglang_hook.py | +`SchedPF` / `LoadBytes` 日志 | 请求日志增加调度器和负载字节 |
| sglang_hook.py | 修复守恒公式 | `L1_device = cached - l2_hit` (当 cached ≥ l2_hit) |
| sglang_hook.py | 修复 recomputed_tokens | 同上启发式 |
| sglang_hook.py | 门控 HBF-STATS 日志 | `HISIM_HBF_STATS=1` 环境变量控制 |

---

## 8. 不变量

- ✅ SGLang 原生代码 **0 修改**
- ✅ **51** 单元测试全通过
- ✅ HBF-off 基线不回归
- ✅ V1a 算法未修改

---

## 9. 验收标准

> "同一批有效 Host KV 可以分别从 HBF 和 DRAM 恢复，而且模拟时间随介质时延参数变化。"

- ✅ A 配置：R0 从 DRAM 恢复 (dram_hit=239/2039)
- ✅ E 配置：R0 从 HBF 恢复 (hbf_hit=239/2039)
- ✅ 模拟时间随介质变化：DRAM load=2030.8μs vs HBF load=53.5μs (38x 差异)

---

## 10. 下一步

### Phase 3B：HBFSim MQSim Reference Backend

1. 核查 HBFSim 的 CPU-only MQSim API（`MqsimOnlineEngine`）
2. 微基准：Read/Write × I/O Size(16KB/64KB/1MB/64MB) × QD(1/4/16/64)
3. 比较 fast vs MQSim reference 的 P50/P95/P99 Latency 和有效带宽
4. 接入 HiSim HBF Backup/Load 时序（每次 I/O 包含 Arrival/Size/Type/Completion/ReqID）
5. 明确 HBF→HBM 有效互连带宽约束
6. DRAM 保持独立时序模型，SSD L3 保持自身模型

### Phase 3C：重新验证 HBF 替代 DRAM

1. 工作负载：真实 Host Hit（Phase 3A 的 shared-prefix workload）
2. 配置：100% DRAM vs 30%DRAM+70%HBF vs 10%DRAM+90%HBF
3. 时序：fast vs MQSim reference
4. Write：Barrier vs Async（标注 Async 的简化语义）
5. 回答三个问题：QoS、最小带宽需求、收益来源
