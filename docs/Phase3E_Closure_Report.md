# Phase 3E-Closure 报告：容量配置修正、拓扑对照与基线冻结

> 日期：2026-11-03
> SGLang 0 修改，70 单元测试通过，V1a 算法不变

---

## 1. 容量配置问题确认与修正 ✅

### 问题

Phase 3E 报告中"E"配置实际使用 HBF=717/2048 = **35%**（非 70%）。原因：`HISIM_HBF_LOGICAL_SIZE=717` 从 512-token 实验直接复制，未按 Host Total=2048 重新计算。

### 修正后配置（统一 Runtime Capacity）

| Config | HBF | DRAM | Total | HBF 占比 | hicache_ratio |
|---|---|---|---|---|---|
| A | 0 | 2048 | 2048 | 0% | 2.0 |
| E35 | 717 | 1331 | 2048 | **35%** | 1.3 |
| **E70** | **1433** | **615** | **2048** | **70%** | **0.6** |
| E90 | 1843 | 205 | 2048 | 90% | 0.2 |

---

## 2. N=2 重新实验（正确配置） ✅

### Independent 拓扑（HBF 和 DRAM 独立 Host→HBM 路径）

| Config | Load | TTFT | vs DRAM (7.4ms) | Load 分布 | Backup 分布 |
|---|---|---|---|---|---|
| A (DRAM) | 5743μs | 7.4ms | baseline | 2 DRAM (shared) | 全 DRAM |
| E35 | 2884μs | **4.5ms** | **60.8%** | **1 HBF + 1 DRAM** | R0→HBF, R1→DRAM |
| **E70** | **2884μs** | **4.5ms** | **60.8%** | **1 HBF + 1 DRAM** | R0→HBF, R1→DRAM, R2/R3→HBF |
| E90 | 4869μs | 6.5ms | 87.8% | 2 HBF (shared) | 全 HBF |

**关键发现**：E70（正确 70% 配置）仍产生 1 HBF + 1 DRAM 并行 Load！

**原因**：PlacementPolicyV0 维护 70/30 比例：
- R0→HBF（第一次分配，目标 HBF）
- R1→DRAM（平衡比例，DRAM 占比需要提升）
- R2→HBF, R3→HBF（维持 70/30 总体比例）

前两个 backup 分别到不同介质 → Reuse 时并行 Load。

### Shared Downstream 拓扑（HBF+DRAM 共享 Host→HBM 路径，13.6 GB/s）

| Config | Load | TTFT | vs DRAM | 并行优势 |
|---|---|---|---|---|
| E35 | 5749μs | 7.4ms | 100% | **消失** |
| E70 | 5749μs | 7.4ms | 100% | **消失** |
| E90 | 5729μs | 7.4ms | 100% | **消失** |

**关键发现**：共享下游时，所有配置 TTFT=7.4ms（= DRAM 基线）。

**原因**：共享 13.6 GB/s 下游 → 2 个 Load（无论来源）串行共享带宽 → 总时间 = 2×38.9MB/13.6GB/s = 5.72ms，与 DRAM 基线相同。

---

## 3. 资源拓扑对照结论

### 拓扑敏感性

```
Independent 拓扑:
  E70 并行优势 = DRAM_load(2.87ms) 与 HBF_load(2.43ms) 独立
  combined = max(2.87, 2.43) = 2.87ms → TTFT = 4.5ms (60.8%)

Shared Downstream 拓扑 (13.6 GB/s):
  E70 并行优势消失 = 两个 Load 共享 13.6 GB/s
  combined = 2×38.9MB / 13.6GB/s = 5.72ms → TTFT = 7.4ms (100%)
```

### 结论

| 拓扑假设 | E70 vs DRAM | 说明 |
|---|---|---|
| **Independent**（默认） | **39% 更快** | HBF 和 DRAM 有独立 Host→HBM 路径 |
| **Shared DS** | 0%（相同） | HBF 和 DRAM 共享下游 DMA/PCIe |

**硬件拓扑未知时，两种假设都应标注为假设模型。**

当前 HiSim 默认 Independent 拓扑。如需 Shared DS，设置 `HISIM_SHARED_DOWNSTREAM_BW_GB=N`。

---

## 4. Phase 3E 基线冻结

### 已验证（可冻结）

| 能力 | 验证方法 | 结果 |
|---|---|---|
| 共享带宽管理 | 70 单元测试 | N=1→32 线性扩展 ✅ |
| 跨批次 Busy Until | 反例测试 | 正确保留，无超卖 ✅ |
| DRAM 显式共享 | 单元+端到端 | 与 HBF 对称公平 ✅ |
| N=2 QoS (E90) | 端到端扫描 | 临界 BW ≈ 12 GB/s ✅ |
| N=2 并行 Load (E70, Independent) | 端到端 | TTFT=4.5ms (60.8%) ✅ |
| Backup Bytes 修正 | 端到端日志 | 真实 KV Bytes (114,688/tk) ✅ |
| 拓扑敏感性 | 两拓扑对照 | Independent vs Shared DS ✅ |

### 未验证（需后续工作）

| 项目 | 状态 | 需要的工作 |
|---|---|---|
| N=4+ 端到端 QoS | 未验证 | 修改仿真支持更大 batch size |
| HBF 0.8 带宽效率 | 未测量 | 需要真实硬件或 MQSim 数据 |
| 真实 MQSim 标定 | 未完成 | 需获取 MQSim 源码并构建 |
| Full-duplex 建模 | 未实现 | 分离 read/write busy_until |
| V1b Hotness Placement | 未开发 | 基于访问频率的放置优化 |

### 冻结基线配置

```
# 环境变量
HISIM_HBF_TIMING_MODEL=channel       # channel 模型
HISIM_HBF_INTERCONNECT_BW_GB=16      # 互连带宽
HISIM_SHARED_DRAM=1                  # DRAM 共享模型
# HISIM_SHARED_DOWNSTREAM_BW_GB=13.6  # 可选：共享下游拓扑

# 容量配置（Host Total=2048, max_total_tokens=1024）
A:   hicache_ratio=2.0, HBF=0
E35: hicache_ratio=1.3, HBF=717
E70: hicache_ratio=0.6, HBF=1433
E90: hicache_ratio=0.2, HBF=1843
```

---

## 5. 修正后的结论

### 70% DRAM 替代（E70，Independent 拓扑）

| 指标 | E70 | vs DRAM 基线 |
|---|---|---|
| TTFT (N=2) | 4.5ms | **60.8%**（39% 改善） |
| Load Time | 2884μs | **50.2%**（49.8% 改善） |
| 临界 BW | ~6 GB/s | — |
| Backup Fail | 0 | — |

### 70% 替代（E70，Shared DS 拓扑）

| 指标 | E70 | vs DRAM 基线 |
|---|---|---|
| TTFT (N=2) | 7.4ms | **100%**（无改善） |
| 并行优势 | 消失 | — |

### 最终结论

1. **E70 在 Independent 拓扑下有显著优势**：TTFT 改善 39%，因为并发 Load 分散到不同介质
2. **E70 在 Shared DS 拓扑下无优势**：所有 Load 共享下游带宽
3. **拓扑假设是 HBF 价值主张的关键前提**：需确认真实硬件拓扑
4. **E70 与 E35 性能相同**：PlacementPolicyV0 在两种配置下都产生 1 HBF + 1 DRAM 前两个 backup
5. **E90 在 Independent 拓扑下仍满足 QoS**（87.8%），但无并行优势

---

## 6. 代码修改

| 文件 | 修改 |
|---|---|
| `hbf_shared_resource.py` | 新增 shared downstream interconnect 支持 |
| `hbf_shared_resource.py` | `SharedHbfResourceManager.submit` 支持 shared DS |
| `hbf_shared_resource.py` | `SharedDramResourceManager.submit` 支持 shared DS |
| `sglang_hook.py` | 集成 `HISIM_SHARED_DOWNSTREAM_BW_GB` 配置 |
| `sglang_hook.py` | batch reset 时重置 downstream cursor |

环境变量：
- `HISIM_SHARED_DOWNSTREAM_BW_GB=N`（启用共享下游拓扑，默认 unset = Independent）

---

## 7. 不变量

- ✅ SGLang 原生 **0 修改**
- ✅ **70** 单元测试全通过
- ✅ HBF-off 基线不回归
- ✅ V1a 算法未修改
- ✅ 默认 Independent 拓扑（新增功能 opt-in）

---

## 8. 下一步：Phase 4

Phase 3E Baseline 冻结后，进入：
1. **KV Cache 容宽比需求识别**：确定 HBF 池的最优容量比例
2. **HBF 池化带宽效率评估**：测量 HBF 在池化配置下的实际带宽利用
3. **N=4+ 端到端**：修改仿真支持更大 batch
4. **真实 MQSim 标定**：获取源码后校准简化模型
5. **V1b Value-aware Placement**：基于访问频率优化放置
