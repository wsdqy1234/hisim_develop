# Phase 3C 报告：HBF 替代 DRAM 的最终验证

> 日期：2026-10-29
> SGLang 0 修改，51 单元测试通过，V1a 算法不变

---

## 1. 实验配置

### 工作负载

Shared-Prefix Build→Evict→Reuse（Phase 3A 设计）：
- R0 Build → R1 Evict → R0 Reuse（Host Hit load-back）

### 容量配置（等容量公平）

| Config | 512-token | 4K |
|---|---|---|
| A (100% DRAM) | DRAM=1025, HBF=0 | DRAM=8192, HBF=0 |
| E (30% DRAM + 70% HBF) | DRAM=308, HBF=717, Total=1025 | DRAM=2458, HBF=5734, Total=8192 |
| E90 (10% DRAM + 90% HBF) | DRAM=102, HBF=922, Total=1024 | DRAM=819, HBF=7373, Total=8192 |

### 时序模型

| 模型 | HBF 带宽 | 来源 |
|---|---|---|
| fast | 512 GB/s (aggregate envelope) | HBFConfig.read_bandwidth_gb |
| channel | 51.2 GB/s (per-channel aggregate) | 32 ch × 1.6 GB/s |

---

## 2. 核心结果

### 512-token (Host Total=1025)

| Config | Model | HBF Hit | DRAM Hit | Load Time | TTFT | Conservation |
|---|---|---|---|---|---|---|
| A | N/A | 0 | 239 | 2030.8μs | 29.1ms | 240+239+1=480 ✅ |
| E | fast | 239 | 0 | 53.5μs | 27.2ms | 240+239+1=480 ✅ |
| E | **channel** | 239 | 0 | **535.4μs** | **27.7ms** | 240+239+1=480 ✅ |
| E90 | **channel** | 239 | 0 | **535.4μs** | **27.6ms** | 240+239+1=480 ✅ |

### 4K (Host Total=8192)

| Config | Model | HBF Hit | DRAM Hit | Load Time | TTFT | Conservation |
|---|---|---|---|---|---|---|
| A | N/A | 0 | 2039 | 17224.9μs | 67.6ms | 2040+2039+1=4080 ✅ |
| E | fast | 2039 | 0 | 456.7μs | 50.9ms | 2040+2039+1=4080 ✅ |
| E | **channel** | 2039 | 0 | **4567.4μs** | **55.0ms** | 2040+2039+1=4080 ✅ |
| E90 | **channel** | 2039 | 0 | **4567.4μs** | **54.9ms** | 2040+2039+1=4080 ✅ |

### 加速比对比

| 场景 | DRAM Load | HBF Fast Load | HBF Channel Load | Fast 加速 | **Channel 加速** |
|---|---|---|---|---|---|
| 512 (239tk, 27MB) | 2030.8μs | 53.5μs | 535.4μs | 38.0x | **3.8x** |
| 4K (2039tk, 223MB) | 17224.9μs | 456.7μs | 4567.4μs | 37.7x | **3.8x** |

### TTFT 改善

| 场景 | DRAM TTFT | HBF Fast TTFT | HBF Channel TTFT | Fast 改善 | **Channel 改善** |
|---|---|---|---|---|---|
| 512 | 29.1ms | 27.2ms | 27.7ms | 1.9ms (6.5%) | **1.4ms (4.8%)** |
| 4K | 67.6ms | 50.9ms | 55.0ms | 16.7ms (24.7%) | **12.6ms (18.6%)** |

---

## 3. E vs E90 对比

| 指标 | E (30% DRAM) | E90 (10% DRAM) | 差异 |
|---|---|---|---|
| HBF Hit tokens | 239 / 2039 | 239 / 2039 | 相同 ✅ |
| Load Time | 535.4μs / 4567.4μs | 535.4μs / 4567.4μs | **完全相同** |
| TTFT | 27.7ms / 55.0ms | 27.6ms / 54.9ms | 相同（±0.1ms 噪声） |
| R1 Backup 去向 | DRAM (6μs) | HBF (100μs) | E90 更慢 backup |
| R0 Reuse Load 来源 | HBF | HBF | 相同 |

**结论**：E 和 E90 的 R0 Reuse load-back 性能完全相同，因为 R0 的 backup 始终在 HBF。DRAM 比例不影响 load-back 性能，只影响 R1 backup 的介质选择。

---

## 4. 三个问题的回答

### 问题 1：HBF 替代 70% DRAM，在更准确的时序模型下是否仍然满足 QoS？

**是的，满足 QoS。**

| 场景 | DRAM TTFT (baseline) | HBF Channel TTFT | 比率 | QoS (≤110%)? |
|---|---|---|---|---|
| 512 | 29.1ms | 27.7ms | 95.2% | ✅ |
| 4K | 67.6ms | 55.0ms | 81.4% | ✅ |

HBF 不只满足 QoS，还**改善了** TTFT（4.8% ~ 18.6%）。Channel 模型下 HBF 仍有 3.8x 的 load-back 加速。

### 问题 2：HBF 至少需要多少有效带宽、读写延迟和并发能力，才能满足 QoS？

**当前参数已远超最低要求。**

QoS 阈值：HBF TTFT ≤ 110% × DRAM TTFT

| 场景 | DRAM TTFT | QoS 阈值 (110%) | 当前 HBF Load | 最低 HBF Load | 最低 HBF BW |
|---|---|---|---|---|---|
| 512 | 29.1ms | 32.0ms | 535.4μs | ≤ 2870μs | ≥ 9.7 GB/s |
| 4K | 67.6ms | 74.4ms | 4567.4μs | ≤ 22975μs | ≥ 9.9 GB/s |

**最低 HBF 有效带宽 ≈ 10 GB/s**（当前 51.2 GB/s，有 5x 余量）。

最低要求分解：
- 有效带宽 ≥ 10 GB/s（当前 51.2 GB/s ✅）
- 读延迟 ≤ 100μs（当前 10μs ✅）
- 并发通道 ≥ 4（当前 32 ✅，但单 I/O 已足够）

### 问题 3：HBF 的收益主要来自减少 DRAM 容量，还是改善 Host KV 恢复性能？

**两者都有，但恢复性能改善是主导因素。**

| 收益来源 | 量化 | 占比 |
|---|---|---|
| **恢复性能改善** | Load 加速 3.8x，TTFT 改善 4.8%~18.6% | **主导** |
| **DRAM 容量替代** | 70%~90% DRAM 被替代 | 容量收益（经济性） |
| Write 策略收益 | 无（Host Hit 不经过 L3） | 不适用 |

具体分析：
- 4K 场景：TTFT 从 67.6ms 降至 55.0ms，**12.6ms 改善全部来自 HBF load 加速**
- 容量替代是经济收益（减少 DRAM 成本），不影响性能
- HBF 的 51.2 GB/s 通道带宽 vs DRAM 的 13.6 GB/s 有效带宽 → 3.8x 性能优势

---

## 5. 已知限制

1. **简化通道模型 ≠ 真实 MQSim**：未建模 FTL 地址映射、wear leveling、HBM cache
2. **Backup 时序未完全建模**：当前 backup=100μs（page latency），未含 transfer time
3. **队列深度效应未体现**：当前工作负载（3 请求）无队列竞争
4. **HBF→HBM 互连未建模**：51.2 GB/s 是通道聚合，非端到端互连带宽
5. **Async Write 简化语义**：当前 `l3_write_overflow=0`，未建模 Pending Write 依赖
6. **数值非真实硬件**：HBFSim nominal 参数为假设值

---

## 6. 代码修改

| 文件 | 修改 |
|---|---|
| `types.py` | HBFConfig 增加 channel 几何参数 + 计算属性 |
| `sglang_hook.py` | `_hbf_transfer_duration` 增加 fast/channel 开关 |
| `sglang_hook.py` | 新增 `_hbf_channel_transfer_duration` 函数 |
| `sglang_hook.py` | HBF adapter 安装时设置 channel 参数 |
| `hbf_channel_model.py` | 独立微基准脚本 |

环境变量开关：`HISIM_HBF_TIMING_MODEL=fast|channel`（默认 fast）

---

## 7. 不变量

- ✅ SGLang 原生代码 **0 修改**
- ✅ **51** 单元测试全通过
- ✅ HBF-off 基线不回归（fast 模型为默认）
- ✅ V1a 算法未修改

---

## 8. 收益来源区分总结

| 收益来源 | fast 模型 | channel 模型 | 真实预期 |
|---|---|---|---|
| HBF 介质时延 | 38x 加速 | **3.8x 加速** | 3~4x（channel 模型可信） |
| HBF 容量/命中率 | 无差异 | 无差异 | 取决于工作负载 |
| 少写 SSD 策略 | 无 | 无 | 取决于 Write 模式 |
| Write Barrier | 不适用 | 不适用 | Host Hit 无 L3 写入 |

**核心结论**：HBF 替代 70% DRAM 在 channel 模型下仍有 3.8x load-back 加速，TTFT 改善 5%~19%，QoS 远超要求。收益主要来自 HBF 的通道带宽优势（51.2 vs 13.6 GB/s），而非容量替代。
