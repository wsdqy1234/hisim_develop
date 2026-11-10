# Phase 3D-P1 报告：共享带宽竞争、并发恢复与 QoS 硬件需求

> 日期：2026-10-29
> SGLang 0 修改，61 单元测试通过（51 原有 + 10 共享资源），V1a 算法不变

---

## 1. P0：单请求基线冻结 ✅

### 有效带宽计算修正

| 互连 BW | 实际有效 BW | 说明 |
|---|---|---|
| 8 GB/s | 8.0 GB/s | interconnect 瓶颈 |
| 16 GB/s | 16.0 GB/s | interconnect 瓶颈 |
| 32 GB/s | 32.0 GB/s | interconnect 瓶颈 |
| 51.2 GB/s | 51.2 GB/s | media 瓶颈（channel aggregate） |

`max(media_time, interconnect_time)` 标注为**理想流水线近似**：假设媒体读取和互连传输可以完全重叠。真实物理完成时间可能更长（串行流水线）。

### DRAM 基线带宽说明

DRAM 有效带宽 13.6 GB/s = `memory_read_bandwidth_gb(16) × 0.85`，这是 **Host→HBM 有效带宽**（含 15% 协议开销），不是介质带宽。DRAM 和 HBF 共享相同的 HBM 目标，但使用不同的源介质和互连路径。

### 单请求回归基线

| 场景 | DRAM Load | HBF Load (channel, BW=51.2) | HBF Load (channel, BW=16) |
|---|---|---|---|
| 512 (239tk, 27MB) | 2030.8μs | 535.4μs | 1713.2μs |
| 4K (2039tk, 233MB) | 17224.9μs | 4567.4μs | 14615.6μs |

**冻结为 Phase 3D Channel Baseline。**

---

## 2. P1：共享资源模型 ✅

### SharedHbfResourceManager 设计

两个共享资源（各自串行内，跨资源并行）：
1. **Media**：channel aggregate BW (51.2 GB/s) — NAND 读取
2. **Interconnect**：HBF→HBM link BW (可配置) — 数据传输

并行流水线模型：
- 单 I/O：`total = max(media_time, ic_time)` — 匹配 Phase 3D-P0
- N 并发 I/O：资源共享 → 后续 I/O 排队等待 → 更长完成时间

### 正确性验证（10 单元测试）

| 测试 | 验证内容 | 结果 |
|---|---|---|
| single_io_matches_independent_model | 并发=1 匹配独立模型 | ✅ |
| concurrent_ios_share_bandwidth | 4 并发：聚合 BW ≤ min(media, ic) | ✅ 16.0 GB/s |
| completion_times_non_negative | 所有时间 ≥ 0 | ✅ |
| read_write_share_resources | 读写共享资源 | ✅ |
| queue_depth_limits_outstanding | QD 限制并发 | ✅ peak=QD |
| interconnect_bottleneck | ic < media 时 ic 瓶颈 | ✅ |
| media_bottleneck | media < ic 时 media 瓶颈 | ✅ |
| later_io_has_longer_duration | 后续 I/O 更长（排队） | ✅ |
| batch_duration_equals_max_individual | batch = max(individual) | ✅ |
| reset_clears_state | 重置清除状态 | ✅ |

### 不变量验证

- ✅ 并发=1：结果与 Phase 3D-P0 一致（误差 < 0.2%）
- ✅ 聚合吞吐 ≤ min(media_bw, ic_bw)
- ✅ Queue Wait、Completion Time ≥ 0
- ✅ 读写不分别独占互连
- ✅ StateManager 使用 MAX（非 SUM）→ 正确反映并发完成时间

---

## 3. P2：并发负载实验 ✅

### 工作负载设计

- N 个完全独立的 Build 请求（无共享 prefix）→ 强制 L1 驱逐
- N 个 Reuse 同时到达（t=5.0）→ 同一 prefill batch（bs=N）
- max_total_tokens=1024, input=340, output=16
- E90: HBF=1843 (90%), DRAM=205 → 所有 backup 到 HBF

### N=2 并发结果（BW=16 GB/s）

| Config | bs | Load Time | TTFT | HBF Hit | DRAM Hit | QoS |
|---|---|---|---|---|---|---|
| A (DRAM) | 2 | 5743.1μs | 7.4ms | 0 | 339×2 | baseline |
| E90 (HBF) | 2 | **4869.4μs** | **6.5ms** | 339×2 | 0 | **87.8% ✅** |

### 共享资源效应验证

| 指标 | 单请求 HBF | 2 并发 HBF | 倍数 |
|---|---|---|---|
| Load Time | 2431μs | 4869μs | **2.0x** |
| 有效 BW | 16.0 GB/s | 8.0 GB/s | 0.5x（共享） |

**2 并发 HBF load 时间 = 2 × 单请求** → 共享带宽效应精确验证 ✅

### HBF vs DRAM 并发对比

| 指标 | DRAM (SUM) | HBF (MAX+shared) | HBF 优势 |
|---|---|---|---|
| 单请求 Load | 2871μs | 2431μs | 1.18x |
| 2 并发 Load | 5743μs | 4869μs | **1.18x** |
| 有效并发 BW | 6.8 GB/s | 8.0 GB/s | 1.18x |

**HBF 在并发下仍比 DRAM 快 18%**，因为 HBF 互连 (16 GB/s) > DRAM 有效 (13.6 GB/s)。

### DRAM 模型说明

DRAM 使用 SUM（`inc_hicache_l2_load_dur += dur`），HBF 使用 MAX（`if dur > current`）。
- DRAM SUM ≈ 2 loads 共享 13.6 GB/s（accidental fairness）
- HBF MAX + shared = 2 loads 共享 16 GB/s
- 对比公平：两者都反映了并发共享效应

---

## 4. 瓶颈归因（N=2 并发）

| 组件 | 时间 | 占总 Load % | 瓶颈? |
|---|---|---|---|
| HBF 媒体 Service | 759μs (per I/O) | 16% | 否 |
| HBF→HBM 互连 | 2431μs (per I/O) | 50% | **是** |
| 队列等待（第2个 I/O） | 2431μs | 50% | **是** |
| **总 Load (HBF)** | **4869μs** | 100% | interconnect + queue |

### 瓶颈链

```
单请求: media (51.2 GB/s) vs interconnect (16 GB/s) → interconnect 瓶颈
2 并发: interconnect 共享 → 每个得 8 GB/s → queue wait 增加
N 并发: interconnect 分 N 份 → 每个 16/N GB/s → 线性增长
```

---

## 5. QoS 门槛（N=2 并发，基于实验数据）

| 互连 BW | HBF Load (2 并发) | HBF TTFT | DRAM TTFT | QoS? |
|---|---|---|---|---|
| 16 GB/s | 4869μs | 6.5ms | 7.4ms | ✅ (87.8%) |
| 8 GB/s | ~9738μs | ~11.6ms | 7.4ms | ❌ (157%) |
| 32 GB/s | ~2431μs | ~4.1ms | 7.4ms | ✅ (55.4%) |

### 临界带宽（N=2 并发，QoS ≤ 110% DRAM）

| 负载 | DRAM TTFT | QoS 阈值 | 临界 BW |
|---|---|---|---|
| N=2, 340tk (38.9MB) | 7.4ms | 8.1ms | **~11 GB/s** |

对比单请求临界带宽：
| 负载 | 单请求临界 BW | N=2 临界 BW |
|---|---|---|
| 512 (27MB) | 5.6 GB/s | ~11 GB/s |
| N=2 (38.9MB) | — | ~11 GB/s |

**并发使临界带宽提高约 2x**（从 ~5.6 到 ~11 GB/s），因为互连带宽被 N 个请求共享。

---

## 6. 70%/90% DRAM 替代能力评估

### 单请求（Phase 3D-P0 基线）

| 替代率 | 512 QoS | 4K QoS | 条件 |
|---|---|---|---|
| 70% (E) | ✅ | ✅ (BW≥10) | interconnect ≥ 10 GB/s |
| 90% (E90) | ✅ | ✅ (BW≥10) | 同上 |

### N=2 并发（Phase 3D-P1）

| 替代率 | QoS | 条件 |
|---|---|---|
| 90% (E90) | ✅ (87.8%) | interconnect ≥ 11 GB/s |

### 高并发未验证

N=4/8/16/32 需要更大 L1 容量和更复杂的工作负载设计。当前仿真在多数配置下处理 bs=1（单请求/batch），N=2 需要 E90 + 特定容量配置才能实现 bs=2。

---

## 7. 代码修改

| 文件 | 修改 |
|---|---|
| `hbf_shared_resource.py` | **新增**：SharedHbfResourceManager（共享带宽模型） |
| `sglang_hook.py` | `_hbf_channel_transfer_duration` 接入共享管理器 |
| `sglang_hook.py` | `wrapped_check_hicache_events` 批次开始时重置管理器 |
| `sglang_hook.py` | HBF adapter 安装时配置管理器参数 |
| `test/test_hbf_shared_resource.py` | **新增**：10 个共享资源单元测试 |

环境变量：
- `HISIM_HBF_TIMING_MODEL=fast|channel`
- `HISIM_HBF_INTERCONNECT_BW_GB=N`
- 共享管理器自动启用（channel 模式下）

---

## 8. 不变量

- ✅ SGLang 原生 **0 修改**
- ✅ **61** 单元测试全通过（51 + 10）
- ✅ HBF-off 基线不回归
- ✅ V1a 算法未修改
- ✅ 单请求结果与 Phase 3D-P0 一致

---

## 9. 当前限制

1. **并发测试覆盖有限**：仅验证 N=2（bs=2）。N=4+ 需要更大 L1 或修改仿真逻辑
2. **DRAM 无显式共享模型**：DRAM 使用 SUM（≈ accidental sharing），未显式建模共享带宽
3. **简化通道模型 ≠ MQSim**：无 FTL/wear-leveling/HBM cache
4. **理想流水线近似**：`max(media, ic)` 可能低估真实延迟
5. **数值非真实硬件**：HBFSim nominal 假设值

---

## 10. 三个问题的更新回答

### Q1: HBF 替代 70%/90% DRAM 在并发下是否满足 QoS？

**是，在 interconnect ≥ 11 GB/s 时满足。**
- N=2 并发：HBF TTFT = 6.5ms vs DRAM TTFT = 7.4ms → 87.8% ✅
- HBF 在并发下仍比 DRAM 快 18%（互连 16 > DRAM 13.6 GB/s）

### Q2: HBF 在并发下需要多少有效带宽？

| 并发数 | 临界 BW | 安全 BW |
|---|---|---|
| 1 | ~5.6 GB/s (512) / ~9.7 (4K) | ≥10 GB/s |
| 2 | ~11 GB/s | ≥12 GB/s |
| 4 (推算) | ~22 GB/s | ≥24 GB/s |
| N | ~5.6 × N GB/s | ≥6 × N GB/s |

### Q3: HBF 收益主要来自容量还是性能？

**两者都有，性能收益在并发下仍然存在：**
- 并发性能收益：HBF 18% 快于 DRAM（互连 16 vs 13.6 GB/s）
- 容量收益：70%~90% DRAM 被替代
- 瓶颈：interconnect 带宽（非 media），并发时被 N 请求共享

---

## 11. 下一步建议

1. **N=4/8/16/32 并发测试**：修改仿真支持更大 batch size，或使用多轮交错负载
2. **DRAM 共享模型**：为 DRAM 添加显式共享带宽，使对比完全公平
3. **Read/Write 竞争**：构建同时包含 backup 和 load 的批次，验证读写干扰
4. **MQSim 校准**：获取 MQSim 源码，用真实 FTL 模型校准简化模型
5. **V1b Hotness Placement**：基于访问频率优化 HBF/DRAM 放置
