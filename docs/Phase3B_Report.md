# Phase 3B 报告：HBF 通道/队列时序模型与微基准

> 日期：2026-10-29
> SGLang 0 修改，51 单元测试通过

---

## 1. MQSim 构建状态：受限

### 问题

HBFSim 的 MQSim 引擎 (`MqsimOnlineEngine`) 是 C++ 类，需要从源码构建：
- 需要 CMake ≥ 3.25（容器内仅 3.22，找到 4.4.3 替代）
- 需要 MQSim 子模块源码（`third_party/mqsim` 为空）
- 容器**无互联网访问**，无法 `git clone` MQSim

### 替代方案

实现了**简化通道/队列模型** (`hbf_channel_model.py`)，捕获 fast 模型缺失的关键时序行为：

1. **每通道带宽瓶颈**：32 通道 × 1.6 GB/s = 51.2 GB/s（非聚合 512 GB/s）
2. **页级并行**：每通道 32 planes，同 plane 内顺序，跨 plane 并行
3. **队列深度**：admission control 限制并发请求数

**这不是重新开发 Flash 媒体模型**——而是实现 fast 模型缺失的通道级带宽约束。参数全部来自 HBFSim `nominal.json`。

---

## 2. HBFSim Profile 参数

| 参数 | 值 | 来源 |
|---|---|---|
| channels | 32 | nominal.json |
| dies_per_channel | 8 | nominal.json |
| planes_per_die | 4 | nominal.json |
| page_bytes | 16384 (16KB) | nominal.json |
| channel_width_bits | 8 (1 byte) | nominal.json |
| channel_transfer_rate_mtps | 1600 | nominal.json |
| read_latency_ns | 10000 (10μs) | nominal.json |
| program_latency_ns | 100000 (100μs) | nominal.json |
| queue_depth | 128 | nominal.json |
| aggregate_bandwidth_bytes_per_s | 512 GB/s | nominal.json |

### 关键带宽计算

| 带宽 | 值 | 计算 |
|---|---|---|
| 每通道带宽 | 1.6 GB/s | 1 byte × 1600 MT/s |
| **通道聚合带宽** | **51.2 GB/s** | 32 × 1.6 GB/s |
| 聚合封装带宽 | 512 GB/s | profile.envelope |
| **带宽比** | **10x** | 512 / 51.2 |

---

## 3. 微基准结果

### 3.1 单 I/O 延迟对比

| I/O 大小 | Fast Read | Channel Read | 比率 | Fast Write | Channel Write | 比率 |
|---|---|---|---|---|---|---|
| 16KB | 10.0μs | 10.0μs | 1.0x | 100.0μs | 100.0μs | 1.0x |
| 64KB | 10.0μs | 10.0μs | 1.0x | 100.0μs | 100.0μs | 1.0x |
| 1MB | 10.0μs | 20.5μs | 2.0x | 100.0μs | 100.0μs | 1.0x |
| 4MB | 10.0μs | 81.9μs | 8.2x | 100.0μs | 100.0μs | 1.0x |
| 16MB | 32.8μs | 327.7μs | **10.0x** | 327.7μs | 327.7μs | 1.0x |
| 64MB | 131.1μs | 1310.7μs | **10.0x** | 1310.7μs | 1310.7μs | **10.0x** |
| 256MB | 524.3μs | 5242.9μs | **10.0x** | 5242.9μs | 5242.9μs | **10.0x** |

**关键发现**：
- ≤64KB：页延迟主导（10μs read / 100μs write），两模型一致
- ≥16MB（read）：通道传输主导，fast 高估 **10x**
- ≥64MB（write）：通道传输主导，fast 高估 **10x**

### 3.2 Token 级延迟（HiSim HBF Backup/Load）

| Tokens | I/O 大小 | Fast Load | Channel Load | 比率 | Fast Backup | Channel Backup | 比率 |
|---|---|---|---|---|---|---|---|
| 240 | 26.2MB | 53.8μs | **537.6μs** | 10.0x | 268.8μs | **537.6μs** | 2.0x |
| 480 | 52.5MB | 107.5μs | **1075.2μs** | 10.0x | 537.6μs | **1075.2μs** | 2.0x |
| 2040 | 223MB | 457.0μs | **4569.6μs** | 10.0x | 2284.8μs | **4569.6μs** | 2.0x |
| 4080 | 446MB | 913.9μs | **9139.2μs** | 10.0x | 4569.6μs | **9139.2μs** | 2.0x |

### 3.3 有效带宽

| I/O 大小 | Fast BW | Channel BW | 瓶颈 |
|---|---|---|---|
| 16KB | 1.6 GB/s | 1.6 GB/s | page_latency |
| 64KB | 6.6 GB/s | 6.6 GB/s | page_latency |
| 1MB | 104.9 GB/s | **51.2 GB/s** | channel_transfer |
| 4MB+ | 512.0 GB/s | **51.2 GB/s** | channel_transfer |

### 3.4 队列深度效应

| QD | Channel P50 | Channel P95 | Channel Total |
|---|---|---|---|
| 1 | 20.5μs | 31.3μs | 0.363ms |
| 4 | 20.5μs | 31.3μs | 0.363ms |
| 16 | 20.5μs | 31.3μs | 0.363ms |
| 64 | 20.5μs | 31.3μs | 0.363ms |

**队列深度对当前工作负载无影响**：32 通道 + 32 planes，16 个 1MB 请求均匀分布，无队列竞争。

---

## 4. DRAM vs HBF 重新对比（Channel Model）

### Load-back 时延对比

| 场景 | DRAM Load | HBF Fast Load | HBF Channel Load | HBF Fast 加速 | **HBF Channel 加速** |
|---|---|---|---|---|---|
| 240 tokens (27MB) | 2,023μs | 53.8μs | **537.6μs** | 37.6x | **3.8x** |
| 2039 tokens (223MB) | 16,400μs | 457.0μs | **4,569.6μs** | 35.9x | **3.6x** |

### Backup 时延对比

| 场景 | DRAM Backup | HBF Fast Backup | HBF Channel Backup | Fast 加速 | **Channel 加速** |
|---|---|---|---|---|---|
| 480 tokens (52MB) | 3,870μs | 537.6μs | **1,075.2μs** | 7.2x | **3.6x** |

### 带宽比验证

| 介质 | 有效带宽 | 来源 |
|---|---|---|
| DRAM | 13.6 GB/s | 16 GB/s × 0.85 |
| HBF (channel) | 51.2 GB/s | 32 × 1.6 GB/s |
| **HBF/DRAM 比** | **3.76x** | 51.2 / 13.6 |

---

## 5. Phase 3A 结果修正

Phase 3A 使用 fast 模型，HBF 加速为 38x。用 channel 模型修正：

| 指标 | Phase 3A (fast) | 修正 (channel) |
|---|---|---|
| HBF Load (240tk) | 53.5μs | **537.6μs** |
| DRAM Load (240tk) | 2,030.8μs | 2,030.8μs (不变) |
| HBF 加速比 | 38x | **3.8x** |
| TTFT 节省 (512) | 1.9ms | **~1.5ms** |
| TTFT 节省 (4K) | 16.7ms | **~12ms** |

**HBF 仍有显著优势**（3.6-3.8x），但远小于 fast 模型的 38x。

---

## 6. 限制与注意事项

1. **简化模型 ≠ 真实 MQSim**：
   - 未建模 NAND FTL 地址映射细节
   - 未建模 wear leveling / GC 影响
   - 未建模 HBF 内部 HBM cache 命中
   - 未建模 UCIe 互连链路

2. **HBF→HBM 有效互连带宽**：
   - 当前 channel 模型使用 51.2 GB/s（通道聚合）
   - 真实端到端可能受 PCIe/NVLink 限制（需 Phase 3C 验证）

3. **数值不代表真实 HBF 硬件**：
   - HBFSim 参数为 nominal 假设值
   - 51.2 GB/s 是通道级理论值，非实测

4. **队列深度效应**：
   - 当前工作负载（1-4 请求/批）无队列竞争
   - 高并发场景（100+ 请求/批）需重新评估

---

## 7. 代码文件

| 文件 | 说明 |
|---|---|
| `src/hisim/simulation/hbf_channel_model.py` | 简化通道/队列模型 + 微基准 |

---

## 8. 下一步：Phase 3C

1. **接入 HiSim**：将 channel 模型集成到 `_hbf_transfer_duration`，增加 `HISIM_HBF_TIMING_MODEL=fast|channel` 开关
2. **重跑 Phase 3A**：用 channel 模型重跑 Host Hit 实验
3. **三配置对比**：100% DRAM vs 30%+70% HBF vs 10%+90% HBF
4. **Write 模式**：Barrier vs Async
5. **回答三个问题**：
   - Q1: HBF 替代 70% DRAM 在 channel 模型下是否满足 QoS？
   - Q2: HBF 至少需要多少有效带宽/延迟/并发才能满足 QoS？
   - Q3: HBF 收益主要来自减少 DRAM 容量还是改善恢复性能？
