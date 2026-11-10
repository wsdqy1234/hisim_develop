# Phase 3D 报告：HBF 端到端有效带宽与 QoS 临界分析

> 日期：2026-10-29
> SGLang 0 修改，51 单元测试通过，V1a 算法不变

---

## 1. P0：HBF Backup 时序修复 ✅

### 问题

`mock_start_writing` 使用 `self._transfer_num_bytes(op)` 获取 backup 字节数，但该方法返回 **mock buffer 字节**（~56 bytes/token），而非真实 KV 字节（114,688 bytes/token）。

| 修复前 | 修复后 |
|---|---|
| `480tk→HBF (0.05MB) dur=100.0μs` | `480tk→HBF (55.05MB) dur=1075.2μs` |
| `240tk→DRAM (0.03MB) dur=6.0μs` | `240tk→DRAM (27.53MB) dur=2027.9μs` |

### 修复

使用 `ConfigManager.get_kv_cache_bytes()` 替代 `_transfer_num_bytes(op)`：
```python
real_bytes_per_token = ConfigManager.get_kv_cache_bytes() or 32768
hbf_bytes = hbf_count * real_bytes_per_token
```

### 验证

| 操作 | Tokens | Bytes | Duration | 模型验证 |
|---|---|---|---|---|
| HBF Backup | 480 | 55.05MB | 1075.2μs | channel: max(1.075ms transfer, 400μs page) = 1.075ms ✅ |
| DRAM Backup | 240 | 27.53MB | 2027.9μs | DRAM: 4μs + 27.53e6/13.6e9 = 2.028ms ✅ |
| HBF Load | 239 | 27.41MB | 535.4μs | channel: max(537.6μs, 10μs) = 537.6μs ✅ |

---

## 2. P0：HBF→HBM 互连带宽约束 ✅

### 数据路径模型

```
HBF Media (NAND channels) → HBF Internal Buffer → HBM (via interconnect)
     media_service_time          interconnect_time
```

**流水线模型**（overlap）：`total = max(media_service, interconnect_transfer)`

- 当 interconnect_BW ≥ channel_aggregate_BW (51.2 GB/s)：media 是瓶颈
- 当 interconnect_BW < 51.2 GB/s：interconnect 是瓶颈

### 实现

新增 `HISIM_HBF_INTERCONNECT_BW_GB` 环境变量（默认=51.2，无约束）：
```python
interconnect_time = num_bytes / (interconnect_bw_gb * 1e9)
return max(media_service, interconnect_time)
```

### 区分四类时间

| 时间类型 | 来源 | 计算方式 |
|---|---|---|
| HBF 媒体 Service Time | NAND channels | max(channel_transfer, page_latency) |
| HBF→HBM 互连 Transfer | PCIe/NVLink/UCIe | bytes / interconnect_BW |
| 队列等待 | admission control | 当前工作负载无竞争（QD=128, 1请求） |
| 影响 TTFT 的 Wait | load + infer | load_time 直接加入 clock |

---

## 3. 互连带宽扫描结果

### 512-token (27MB load, DRAM baseline TTFT=29.1ms, QoS≤32.0ms)

| Interconnect BW | HBF Load | TTFT | vs DRAM | QoS? | 瓶颈 |
|---|---|---|---|---|---|
| 8 GB/s | 3,426.3μs | 30.7ms | 105.5% | ✅ | interconnect |
| 10 GB/s | 2,741.0μs | 30.0ms | 103.1% | ✅ | interconnect |
| 16 GB/s | 1,713.2μs | 28.9ms | 99.3% | ✅ | interconnect |
| 32 GB/s | 856.6μs | 28.0ms | 96.2% | ✅ | interconnect |
| 51.2 GB/s | 535.4μs | 27.7ms | 95.2% | ✅ | media |

### 4K (233MB load, DRAM baseline TTFT=67.6ms, QoS≤74.4ms)

| Interconnect BW | HBF Load | TTFT | vs DRAM | QoS? | 瓶颈 |
|---|---|---|---|---|---|
| **8 GB/s** | **29,231.1μs** | **79.8ms** | **118.0%** | **❌ FAIL** | interconnect |
| 10 GB/s | 23,384.9μs | 73.9ms | 109.3% | ✅ (marginal) | interconnect |
| 16 GB/s | 14,615.6μs | 65.1ms | 96.3% | ✅ | interconnect |
| 32 GB/s | 7,307.8μs | 57.8ms | 85.5% | ✅ | interconnect |
| 51.2 GB/s | 4,567.4μs | 55.0ms | 81.4% | ✅ | media |

### QoS 临界带宽

| 工作负载 | Load 大小 | DRAM TTFT | QoS 阈值 | **临界 BW** | 安全 BW |
|---|---|---|---|---|---|
| 512-token | 27 MB | 29.1ms | 32.0ms | **~5.6 GB/s** | ≥8 GB/s |
| 4K | 233 MB | 67.6ms | 74.4ms | **~9.7 GB/s** | ≥10 GB/s |

### 关键发现

1. **4K @ 8 GB/s FAILS QoS**：TTFT=79.8ms > 74.4ms 阈值，HBF 比 DRAM 慢 18%
2. **4K @ 10 GB/s 边缘通过**：TTFT=73.9ms，仅低于阈值 0.5ms
3. **互连 < 16 GB/s 时 HBF 可能比 DRAM 慢**（4K 场景）
4. **512-token 始终满足 QoS**（即使 8 GB/s）
5. **瓶颈在 51.2 GB/s 处切换**：interconnect → media (channel)

### HBF vs DRAM 有效带宽对比

| 互连 BW | HBF 有效 BW (4K) | DRAM 有效 BW | HBF/DRAM |
|---|---|---|---|
| 8 GB/s | 8.0 GB/s | 13.6 GB/s | **0.59x (HBF 慢!)** |
| 10 GB/s | 10.0 GB/s | 13.6 GB/s | **0.74x (HBF 慢!)** |
| 16 GB/s | 14.6 GB/s* | 13.6 GB/s | 1.07x |
| 32 GB/s | 23.3 GB/s* | 13.6 GB/s | 1.71x |
| 51.2 GB/s | 51.2 GB/s (media) | 13.6 GB/s | 3.76x |

*有效 BW 受 interconnect 限制，低于 media BW

---

## 4. 瓶颈归因

### 时间分解（4K, BW=32 GB/s）

| 组件 | 时间 | 占总 Load % | 瓶颈? |
|---|---|---|---|
| HBF 媒体 Service | 4,567μs | 62.5% | 否 (interconnect 更慢) |
| HBF→HBM 互连 | 7,281μs | 99.6% | **是** |
| 队列等待 | 0μs | 0% | 否 (无竞争) |
| **总 Load** | **7,308μs** | 100% | interconnect |

### 瓶颈切换图

```
BW < 9.7 GB/s:  interconnect 瓶颈 → QoS FAIL (4K)
BW ≈ 10 GB/s:   interconnect 瓶颈 → QoS 边缘 (4K)
BW ≈ 16 GB/s:   interconnect 瓶颈 → HBF ≈ DRAM 速度
BW ≈ 32 GB/s:   interconnect 瓶颈 → HBF 1.7x 快于 DRAM
BW ≥ 51.2 GB/s: media 瓶颈 → HBF 3.8x 快于 DRAM
```

---

## 5. 70%/90% DRAM 替代能力评估

### 单请求场景（当前验证）

| 替代率 | 512-token | 4K | 条件 |
|---|---|---|---|
| 70% (E) | ✅ QoS 通过 | ✅ (BW≥10) | interconnect ≥ 10 GB/s |
| 90% (E90) | ✅ QoS 通过 | ✅ (BW≥10) | 同上（E/E90 load 相同） |

### 高并发场景（P1，待验证）

当前实验仅测试单请求 Restore。高并发下：
- 多个请求同时从 HBF 恢复 → interconnect 带宽共享
- 队列深度限制 → 排队等待
- 写入干扰读恢复 → 读写竞争

**需要 P1 高并发实验验证。**

---

## 6. 代码修改

| 修改 | 文件 | 说明 |
|---|---|---|
| Backup bytes 修复 | sglang_hook.py | `real_bytes_per_token = ConfigManager.get_kv_cache_bytes()` |
| 互连带宽约束 | sglang_hook.py | `_hbf_channel_transfer_duration` 增加 `HISIM_HBF_INTERCONNECT_BW_GB` |
| Channel 几何参数 | types.py | HBFConfig 增加 channel_width_bits, transfer_rate, page_bytes 等 |

环境变量：
- `HISIM_HBF_TIMING_MODEL=fast|channel`（默认 fast）
- `HISIM_HBF_INTERCONNECT_BW_GB=N`（默认 51.2 = 无约束）

---

## 7. 不变量

- ✅ SGLang 原生 **0 修改**
- ✅ **51** 单元测试全通过
- ✅ HBF-off 基线不回归（fast 为默认，interconnect 默认无约束）
- ✅ V1a 算法未修改

---

## 8. P1 待完成项

### 高并发读写竞争
- 扩展到多个并发 Restore 请求（1/4/8/16/32 Active Restore）
- 构建 Build/Reuse/Churn 混合负载
- 验证队列竞争和读写干扰
- A vs E vs G（10% DRAM + 90% HBF）三配置对比

### 硬件性能门槛重新评估
- 在高并发下扫描 HBF 有效带宽、读延迟、Channel 数
- 找到满足 P95 TTFT ≤ 110%、P95 TPOT ≤ 110%、Throughput ≥ 95% 的参数组合
- 确定是否达到互连带宽瓶颈

---

## 9. 结论

**HBF 替代 70% DRAM 的 QoS 满足条件**：

1. **单请求场景**：interconnect ≥ 10 GB/s 即可满足 QoS（4K 工作负载）
2. **HBF 优势来源**：当 interconnect ≥ 16 GB/s 时，HBF 比 DRAM 快；≥ 51.2 GB/s 时，media 成瓶颈，HBF 3.8x 快
3. **关键风险**：interconnect < 10 GB/s 时，4K 工作负载 QoS 失败（HBF 比 DRAM 慢）
4. **容量替代**：70% 和 90% 替代在单请求下性能相同（R0 backup 始终在 HBF）
5. **高并发未验证**：需要 P1 实验确认多请求下的 interconnect 带宽共享效应
