# Phase 2c 实验报告：V1a 大负载稳定性与 HBF 容量替代验证

> 日期：2026-10-28
> V1a 版本：available_size=MAX + 有界 Eviction Retry (max_retries=3)
> 模型：Qwen3-0.6B (CPU, dummy load)
> 时序模型：fast (max 语义，无队列竞争)

---

## 1. 256 请求稳定性结果

**工作负载**：256 reqs, 512 input (256 shared + 256 unique), 16 output, max_total_tokens=1024, total L2=1024, prefetch_threshold=1

| Config | DRAM% | P95_TTFT(ms) | P95_TPOT(ms) | Tput(req/s) | Comp | Fail% | L3 hits |
|---|---|---|---|---|---|---|---|
| A(基线) | 100% | 3395 | 2.4 | 52.5 | 256 | N/A | 24320 |
| C(50%) | 50% | **3124** | 2.1 | **55.8** | 256 | 4.9% | 21632 |
| E(30%) | 30% | **3211** | 2.2 | **54.8** | 256 | 2.0% | 9088 |
| F(20%) | 20% | **3211** | 2.2 | **54.8** | 256 | 1.9% | 10112 |
| G(10%) | 10% | **3211** | 2.2 | **54.8** | 256 | 1.0% | 10368 |

**QoS 约束**（P95_TTFT ≤ 3735ms, P95_TPOT ≤ 2.64ms, Tput ≥ 49.9）：**全部满足** ✅

**结论**：256 请求下无恶性循环，失败率 1-4.9%（no_evictable_victim），所有配置通过 QoS。

---

## 2. 不同 Prefix 长度下的容量替代对照

**测试配置**：A(100% DRAM) vs E(30% DRAM + 70% HBF)，default prefetch_threshold=256

### 2a. 1K Input (64 reqs, max_total_tokens=2048, total L2=2048)

| Config | P95_TTFT(ms) | P95_TPOT(ms) | Tput | Fail% | L3 hits |
|---|---|---|---|---|---|
| A(基线) | 1351 | 3.4 | 36.3 | N/A | 6912 |
| E(30%) | **1031** (↓24%) | **2.6** (↓24%) | **45.1** (↑24%) | 0% | 3456 |

QoS: P95_TTFT ≤ 1486 ✅, P95_TPOT ≤ 3.74 ✅, Tput ≥ 34.5 ✅

### 2b. 4K Input (32 reqs, max_total_tokens=5120, total L2=5120)

| Config | P95_TTFT(ms) | P95_TPOT(ms) | Tput | Fail% | L3 hits |
|---|---|---|---|---|---|
| A(基线) | 2717 | 5.6 | 10.3 | N/A | 1536 |
| E(30%) | **1147** (↓58%) | **2.1** (↓63%) | **23.1** (↑124%) | 0% | 0 |

QoS: P95_TTFT ≤ 2989 ✅, P95_TPOT ≤ 6.16 ✅, Tput ≥ 9.8 ✅

### 2c. 16K Input (16 reqs, max_total_tokens=20480, total L2=20480)

| Config | P95_TTFT(ms) | P95_TPOT(ms) | Tput | Fail% | L3 hits |
|---|---|---|---|---|---|
| A(基线) | 5033 | 17.9 | 2.8 | N/A | 256 |
| E(30%) | **1770** (↓65%) | **2.7** (↓85%) | **8.1** (↑189%) | 0% | 0 |

QoS: P95_TTFT ≤ 5536 ✅, P95_TPOT ≤ 19.7 ✅, Tput ≥ 2.7 ✅

---

## 3. DRAM 缩减比例与性能关系

### 3a. 256 请求 × 512 Input — DRAM 扫描

| DRAM% | P95_TTFT(ms) | P95_TPOT(ms) | Tput | Fail% | L3 hits |
|---|---|---|---|---|---|
| 100% (基线) | 3395 | 2.4 | 52.5 | N/A | 24320 |
| 50% | 3124 | 2.1 | 55.8 | 4.9% | 21632 |
| 30% | 3211 | 2.2 | 54.8 | 2.0% | 9088 |
| 20% | 3211 | 2.2 | 54.8 | 1.9% | 10112 |
| 10% | 3211 | 2.2 | 54.8 | 1.0% | 10368 |

**趋势**：DRAM 从 100% 降到 10%，P95_TTFT 先降后稳（3395→3124→3211），吞吐持续优于基线。L3 hits 随 HBF 增大而减少（24320→10368）。

### 3b. Prefix 长度 × 70% 替代（E config）

| Prefix | A P95_TTFT | E P95_TTFT | 改善 | A Tput | E Tput | 改善 |
|---|---|---|---|---|---|---|
| 512 | 3395 | 3211 | 5% | 52.5 | 54.8 | +4% |
| 1K | 1351 | 1031 | 24% | 36.3 | 45.1 | +24% |
| 4K | 2717 | 1147 | 58% | 10.3 | 23.1 | +124% |
| 16K | 5033 | 1770 | 65% | 2.8 | 8.1 | +189% |

**趋势**：Prefix 越长，HBF 的优势越大。16K 时 E 的吞吐是基线的 2.9 倍。

---

## 4. 缓存命中、Backup Failure 和容量统计

### 4a. Backup 成功率

| 工作负载 | Config | success_tk | failed_tk | fail_rate | retry_success |
|---|---|---|---|---|---|
| 256×512 | C(50%) | 158592 | 8192 | 4.9% | — |
| 256×512 | E(30%) | 147497 | 3072 | 2.0% | — |
| 256×512 | G(10%) | 149239 | 1536 | 1.0% | — |
| 64×1K | E(30%) | 50381 | 0 | 0% | — |
| 32×4K | E(30%) | 269 | 0 | 0% | — |
| 16×16K | E(30%) | 269 | 0 | 0% | — |

**注**：Backup Failed Tokens ≠ Recomputed Prefix Tokens。failed_tk 是 alloc 失败的尝试量，实际重算 tokens 取决于 prefix miss 后的 prefill 计算。长 prefix 下 backup 主要覆盖 shared prefix（256tk），大 unique suffix 节点可能不触发 backup（L1 容量足够）。

### 4b. L3 读取统计

| 工作负载 | Config | L3 queries | L3 hits | L3 prefetched |
|---|---|---|---|---|
| 256×512 | A | ~240 | 24320 | 24320tk |
| 256×512 | E | ~90 | 9088 | 9088tk |
| 64×1K | A | ~60 | 6912 | 0tk |
| 64×1K | E | ~30 | 3456 | 0tk |
| 32×4K | A | ~16 | 1536 | 0tk |
| 32×4K | E | 0 | 0 | 0tk |
| 16×16K | A | ~2 | 256 | 0tk |
| 16×16K | E | 0 | 0 | 0tk |

**趋势**：HBF 显著减少 L3 读取。4K+ prefix 下 E 完全消除 L3 读取。

### 4c. 容量峰值（256×512）

| Config | HBF Peak | DRAM Peak | Total Peak |
|---|---|---|---|
| C(50%) | 444/512 | 1024/1024 | 1468/1536 |
| E(30%) | 685/717 | 615/615 | 1300/1332 |
| F(20%) | 813/819 | 384/410 | 1197/1229 |
| G(10%) | ~920/922 | ~102/102 | ~1022/1024 |

---

## 5. 70% DRAM 替代阶段性判断

### 结论

**在当前 V1a 缓存管理策略下，HBF 可以在以下条件下替代 70% DRAM 容量而不损失 QoS：**

| 条件 | 70% 替代可行性 | 备注 |
|---|---|---|
| 256 reqs × 512 input | ✅ 通过 | P95_TTFT 比基线低 5%，2% 失败（no_evictable_victim） |
| 64 reqs × 1K input | ✅ 通过 | P95_TTFT 比基线低 24%，0% 失败 |
| 32 reqs × 4K input | ✅ 通过 | P95_TTFT 比基线低 58%，0% 失败，L3 读取为 0 |
| 16 reqs × 16K input | ✅ 通过 | P95_TTFT 比基线低 65%，0% 失败，L3 读取为 0 |

**80%/90% 替代**（256×512）：均通过 QoS，失败率 1-2%。

### 性能下降归因

当性能下降时（如 256×512 下 C 的 4.9% 失败）：
1. **Host 容量**：L2=1024，working set 69888 → 68x 过载 → 结构性容量不足
2. **L3 读取**：A 的 L3 hits=24320 → L3 读延迟主导 TTFT
3. **KV 重算**：backup 失败的 tokens 在 round 2 重算（New tokens），但重算时延未单独建模
4. **HBF 恢复时延**：10μs-1ms 无影响（被 inference 重叠）

### V1a 主要瓶颈

1. **no_evictable_victim**（2-5%）：write-through pending 锁定节点，无法安全淘汰
2. **Backup Failed Tokens vs Recomputed Tokens 口径不完整**：adapter 统计不含 SGLang 层 pre-check 失败
3. **V0 比例放置**：不区分 hot/cold，HBF 可能存放低频数据

### 适用工作负载和假设

- **适用**：长 prefix（≥1K）、高复用、中等并发 — HBF 显著减少 L3 读取
- **假设**：fast 时序模型（max 语义，无队列竞争），HBF read_latency=10μs
- **待验证**：MQSim 时序校准、更大并发下的队列竞争、HBF 延迟 >1ms 的影响

---

## 6. 不变量

- ✅ SGLang 原生代码 0 修改
- ✅ 41 单元测试全通过
- ✅ HBF-off 基线不回归
- ✅ V1a 算法未修改（available_size=MAX + bounded retry）
- ✅ 快速时序模型已标注
