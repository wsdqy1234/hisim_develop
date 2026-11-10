# Phase 2c-Fix 报告：容量配置与性能归因校验

> 日期：2026-10-28
> V1a 版本不变，SGLang 0 修改，41 测试通过

---

## P0-1：容量配置核查

### 256 请求测试（wl7_256req, max_total_tokens=1024）

从 HBF-STATS 读取的实际运行时容量：

| Config | hicache_ratio | HBF Cap | DRAM Cap | Total | 预期 | 正确？ |
|---|---|---|---|---|---|---|
| A | 1.0 | 0 | 1024 | 1024 | 1024 | ✅ |
| C | 0.5 | 512 | 513 | 1025 | 1024 | ✅ |
| E | 0.3 | 717 | 308 | 1025 | 1024 | ✅ |
| F | 0.2 | 819 | 205 | 1024 | 1024 | ✅ |
| G | 0.1 | 922 | 103 | 1025 | 1024 | ✅ |

**结论**：256 请求测试所有配置 Total ≈ 1024-1025，容量一致 ✅

用户表中的不一致（DRAM=1024 for C）来自 **128 请求测试**，该测试使用了不同的 hicache_ratio（C: ratio=1.0 而非 0.5），导致 Total=1536 ≠ 1024。128 请求测试的容量比较不公平，但 Phase 2c 报告的核心结论基于 256 请求测试（容量正确）。

### 长 Prefix 测试

| Prefix | max_total_tokens | A DRAM | E DRAM | E HBF | E Total | A Total | 一致？ |
|---|---|---|---|---|---|---|---|
| 1K | 2048 | 2048 | 614 | 1434 | 2048 | 2048 | ✅ |
| 4K | 5120 | 5120 | 1536 | 3584 | 5120 | 5120 | ✅ |
| 16K | 20480 | 20480 | 6144 | 14336 | 20480 | 20480 | ✅ |

**结论**：所有长 Prefix 测试 Total 容量一致 ✅

---

## P0-2：L3 Hits 与 Prefetched 不一致

### 根因

`MockHiCacheStorage.batch_get()` 返回 `None`（mock 后端不加载真实数据）：

```python
def batch_get(self, keys, ...):
    """The mock backend only tracks keys (no real KV data),
    so every load is reported as a miss (None -> 0 pages loaded)."""
    return None
```

### 统计语义

| 指标 | 含义 | 值 |
|---|---|---|
| `L3 total_hits` | `batch_exists` 找到的块数（**存在性查询**） | > 0 |
| `L3 prefetched` | `batch_get` 实际加载的 token 数 | **0**（mock 返回 None） |

**结论**：`L3 hits > 0, prefetched = 0` 是 **正确的统计语义**：
- L3 存在性查询成功（数据在 kvcm 索引中）
- 但实际数据未恢复（mock 不加载）
- L3 读延迟 **未计入 TTFT**（因为没有实际读取）
- 这些 "hits" **不应视为实际缓存恢复**

---

## P0-3：4K/16K 性能收益归因

### 根因：L3 Write Overflow

从 BATCH-TIME 日志中发现的 `l3_ovf`（L3 写溢出）：

| 工作负载 | Config | batch_set | max l3_ovf | TTFT |
|---|---|---|---|---|
| 4K | A(100% DRAM) | **1988** | **53.3ms** | 2717ms |
| 4K | E(30% DRAM) | 4 | 0ms | 1147ms |
| 16K | A(100% DRAM) | **4068** | **232ms** | 5033ms |
| 16K | E(30% DRAM) | 4 | 0ms | 1770ms |
| 256×512 | A | — | 5.7ms | 3395ms |
| 256×512 | E | — | 5.0ms | 3211ms |

### 根因链

```
A (DRAM ≥ node_size):
  alloc(3840) 成功 → 备份大量 token → 1988 次 L3 写入
  → hicache_l3_backup_dur 巨大 → l3_write_overflow = 53-232ms
  → 延迟下一轮 → TTFT 大幅增加

E (HBF MAX < node_size):
  alloc(3840) 失败 (MAX=3584 < 3840) → 大节点不备份
  → 仅 4 次 L3 写入（shared prefix 256tk）
  → 无 l3_write_overflow → TTFT 不受影响
```

### 公平性判断

| 工作负载 | 公平性 | 原因 |
|---|---|---|
| 256×512 | ✅ 公平 | 两者都有 l3_ovf（5.7 vs 5.0ms），容量一致 |
| 4K | ❌ 不公平 | E 的 MAX(3584) < node_size(3840) → 备份失败 → 无 overflow |
| 16K | ❌ 不公平 | E 的 MAX(14336) < node_size(16128) → 同上 |

**4K/16K 的巨大收益主要来自 L3 write overflow 差异，不是 HBF 介质优势。**

如果 E 的 HBF ≥ node_size（能备份大节点），E 也会有 L3 write overflow，性能差异会大幅缩小。

### Per-request 缓存命中对照（4K round 2）

| Config | L1 hit | HBF hit | DRAM hit | L3 hit | New (recompute) |
|---|---|---|---|---|---|
| A | 256 | N/A | 0 | 0 (存在性查询有但未恢复) | 3840 |
| E | 256 | 0 | 0 | 0 | 3840 |

**两者缓存命中完全相同**：L1=256（shared prefix），New=3840（unique suffix 重算）。性能差异完全来自 l3_write_overflow。

---

## 修订结论

### 仍然成立的结论

1. **256×512（公平比较）**：E(30% DRAM) 通过 QoS，P95_TTFT 比基线低 5%，吞吐高 4%。**70% DRAM 替代在 256 请求下有效。** ✅

2. **V1a 容量管理**：available_size=MAX + 有界重试有效消除了恶性循环。256 请求下失败率 1-4.9%（no_evictable_victim）。✅

3. **L3 Read-back 机制**：prefetch_from_storage → batch_exists 链路正常工作（存在性查询成功）。batch_get 返回 None 是 mock 限制，不影响仿真正确性。✅

### 需要修订的结论

1. **4K/16K 的巨大性能优势（58-65% TTFT 降低）需要修订**：
   - 主要来自 L3 write overflow 差异（A 有 53-232ms overflow，E 有 0ms）
   - 不是 HBF 介质优势，而是 E 的 per-pool 容量 < node_size 导致备份量少
   - **不能作为 HBF 容量替代的有效性证据**

2. **L3 "hits" 不代表实际缓存恢复**：
   - batch_exists 找到数据但 batch_get 不加载
   - L3 读延迟未计入 TTFT
   - 后续分析中不应将 L3 hits 等同于 L3 读取收益

3. **128 请求测试（V1a 验证）的容量不一致**：
   - C: Total=1536（比 A 多 50%），E: Total=1332（比 A 多 30%）
   - 这些测试的 QoS 对照不公平
   - 但 V1a 的核心修复（恶性循环消除）仍然有效（基于 256 请求公平测试验证）

### 需要补充的实验

1. **4K/16K 公平比较**：增大 HBF 使 MAX ≥ node_size（如 4K: total L2=8192, HBF=5734 ≥ 3840），重新比较 A vs E
2. **Recompute tokens 时序**：backup 失败的 tokens 的重算时间是否正确计入 TTFT
3. **L3 write overflow 时序建模**：确认 l3_backup_dur 的计算是否正确反映 SSD 写入时间

---

## 不变量

- ✅ SGLang 原生代码 0 修改
- ✅ 41 单元测试全通过
- ✅ HBF-off 基线不回归
- ✅ V1a 算法未修改
- ✅ 快速时序模型已标注
