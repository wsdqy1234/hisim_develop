# Phase 2e 验证报告：缓存正确性最终验收与 4K 公平性能实验

> 日期：2026-10-28
> SGLang 0 修改，51 单元测试通过，V1a 算法不变

---

## P0-1：121 个 host_value 有效性核查 ✅ 无 Stale

### 根因分析

通过阅读 SGLang `unified_tree_core.py` 的 `insert_host` 实现（行 1853-1920），确认：

```python
def insert_host(self, node_id, key, host_value, hash_value):
    matched_length = 0
    while len(key) > 0 and child_key in node.children:
        node = node.children[child_key]
        prefix_len = node.key.match(key, page_size=self.page_size)
        key = key[prefix_len:]
        host_value = host_value[prefix_len:]  # ← slice, NOT set on existing node
        matched_length += prefix_len
    
    result = InsertResult(prefix_len=matched_length, ...)
    
    if len(key) == 0:
        # All matched: return WITHOUT setting host_values
        return result
    
    # Only NEW tokens get host_value:
    new_node.component_data[BASE_COMPONENT_TYPE].host_value = host_value.clone()
```

### `prefix_len=121` 的真正语义

`prefix_len` = **tree node match count**（已存在的 radix tree 子节点匹配数），**不是** host_value 计数。

| 字段 | 值 | 含义 |
|---|---|---|
| `prefix_len` | 121 | 121 tokens 匹配了已存在的 tree nodes（R0 Build 时创建，evict_backuped 保留） |
| 这些 nodes 的 host_value | None | L2 eviction 清除了 host_value，但 tree node 保留（backuped=True） |
| `loaded_from_storage` | 7 | 128 - 121 = 7 NEW tokens 创建了新 node 并设置 host_value |
| `host_hit_length` | 7 | match_prefix 只找到 7 个有效 host_value（只有 7 个新 node 有） |

### 为什么 128 read → 只有 7 consumed

1. kvcm `batch_exists` 找到 128 consecutive keys（kvcm 容量限制）
2. `batch_get` 读取 128 tokens 到 host_indices
3. `insert_host` 遍历 tree：
   - 121 tokens 匹配已存在 tree nodes → **不设置 host_value**（SGLang 设计）
   - 7 tokens 是新 token → 创建新 node + 设置 host_value
4. `match_prefix` 只找到 7 个有效 host_value → `host_hit_length=7`
5. 请求消费 7 tokens（从 host 加载到 device），recompute 499 tokens

### 结论

**不存在 Stale host_value。** 121 是 tree node 匹配数（非 host_value 数），所有 121 个 node 的 host_value=None。只有 7 个新 node 有有效 host_value。这是 SGLang `insert_host` 的设计行为：只为新 token 设置 host_value，不为已存在 node 补充 host_value。

---

## P0-2：A/B/C Prefill 验证

### 场景 B（L3 restore）✅ 已验证（512-token 和 4K）

**512-token 实验**：
```
R0 Reuse: L1=0, DRAM=7(L3 consumed), Recomputed=499
Conservation: 0+7+499=506/506 ✅
```

**4K 实验**：
```
R0 Reuse: L1=128, DRAM=128(L3 consumed), Recomputed=3834
Conservation: 128+128+3834=4090/4090 ✅
```

### 场景 A（L2 命中）和 C（Backup 失败）

在 4K 公平实验中，A 和 E 配置都**没有 L2 命中**（所有请求 `HBF=0, DRAM=0` except L3 consumed）。原因是工作负载使用完全独立的 token IDs（无共享 prefix），不产生 L2 命中。

场景 C（Backup 失败）需要 HBF on + 小 HBF（adapter.alloc 失败），在 4K 公平实验中 HBF=5734 ≥ 4090（node_size），所以 backup 不会失败。

**结论**：A/C 场景需要特殊的共享 prefix 工作负载或小 HBF 配置，留待后续测试。

---

## P0-3：Read/Write 完成时间验证 ✅

### L3 Read Causality（所有配置一致）

```
issue=5000.000ms → complete=5001.835ms → restore=5005.000ms → consume=5005.000ms
consume >= complete = OK ✅
```

### Write Barrier vs Async（4K 对照）

| 指标 | Barrier | Async | 差异 |
|---|---|---|---|
| l3_ovf/batch | 46ms | 0ms | 46ms |
| TTFT | 63.6ms | 18.5ms | **45.1ms** |
| L3 read time | 1835μs | 1835μs | 0 |
| Token conservation | ✅ | ✅ | 相同 |

**TTFT 差异 = 45.1ms ≈ l3_ovf (46ms)**，证明 Write Barrier 假设直接且精确地影响 TTFT。

---

## P1：4K 公平性能实验 ✅

### 配置

| Config | Host Total | DRAM | HBF | Write Mode | HBF ≥ node_size? |
|---|---|---|---|---|---|
| A_barrier | 8192 | 8192 | 0 | barrier | N/A |
| A_async | 8192 | 8192 | 0 | async | N/A |
| E_barrier | 8192 | 2458 | 5734 | barrier | 5734 ≥ 4090 ✅ |
| E_async | 8192 | 2458 | 5734 | async | 5734 ≥ 4090 ✅ |

工作负载：4 个独立 4096-token 请求（R0 Build → R1 Evict → R2 Evict → R0 Reuse）

### 核心结果

| Config | R0 Reuse TTFT | L3 consumed | Recomputed | Conservation | l3_ovf/batch |
|---|---|---|---|---|---|
| **A_barrier** | **63.6ms** | 128 | 3834 | 128+128+3834=4090 ✅ | 46ms |
| **A_async** | **18.5ms** | 128 | 3834 | 128+128+3834=4090 ✅ | 0ms |
| **E_barrier** | **63.6ms** | 128 | 3834 | 128+128+3834=4090 ✅ | 46ms |
| **E_async** | **18.5ms** | 128 | 3834 | 128+128+3834=4090 ✅ | 0ms |

### 关键发现

1. **A 和 E 的 TTFT 完全相同**（barrier: 63.6ms, async: 18.5ms）
2. **L3 restore 完全相同**（exists=128, consumed=128, restored=128）
3. **Token conservation 完全一致**（128+128+3834=4090）
4. **batch_set 完全相同**（129 calls in all configs）
5. **性能差异仅来自 Write 模式**（barrier vs async），不是 HBF vs DRAM

### 收益来源区分

| 收益来源 | 存在？ | 证据 |
|---|---|---|
| HBF 替代 DRAM 的介质时延收益 | ❌ 不存在 | A 和 E TTFT 相同 |
| HBF 容量和放置改变带来的命中率收益 | ❌ 不存在 | L3 consumed 相同（128） |
| 少做 Backup、少写 SSD 的策略收益 | ❌ 不存在 | batch_set 相同（129） |
| Write Barrier 假设差异 | ✅ 主要来源 | barrier vs async: 45.1ms 差异 |

---

## 修订后的 70% DRAM 替代结论

### 适用条件

在以下条件**全部满足**时，HBF 替代 70% DRAM **不产生性能差异**：

1. ✅ 总 Host KV 容量一致（A 和 E 总 slots 相同）
2. ✅ HBF 单池容量 ≥ Backup Node 大小（能完整备份）
3. ✅ 工作负载不产生 L2 命中差异（独立 prefix 或相同 prefix pattern）
4. ✅ Write 模式一致（Barrier 或 Async）

### 限制

1. **Write Barrier 假设敏感**：当前模型中 L3 write overflow 是主要性能影响因素（46ms/batch for 4K）
2. **L3 restore 部分恢复**：`insert_host` 只为新 tokens 设置 host_value，已存在 tree nodes 不恢复
3. **kvcm 容量限制**：L3 查询可能只找到部分 keys（128/505），影响恢复量
4. **无 L2 命中优势**：独立 prefix 工作负载下，HBF 和 DRAM 的 L2 命中率相同

### Phase 2c 结论修订

Phase 2c 报告中 4K/16K 的巨大性能优势（65% TTFT 降低）**不再成立**：
- 根因是**不公平配置**（E 的 HBF < node_size → backup 失败 → 少写 SSD → 少 overflow）
- 在公平配置下（HBF ≥ node_size），A 和 E 性能**完全相同**
- **70% DRAM 替代在公平条件下不产生性能收益**

### 正确的收益来源

当前模型中，性能差异的唯一来源是 **Write Barrier 假设**：
- Barrier 模式：L3 write overflow 阻塞下一轮（46ms/batch for 4K）
- Async 模式：L3 write 不阻塞（0ms overflow）
- 这与 HBF vs DRAM 无关，与 SSD 写入模型有关

**要获得真实的 HBF 介质优势，需要 Phase 3 MQSim 校准**：
- MQSim 的 channel/queue 模型可以区分 HBF（高带宽、低延迟）和 DRAM 的真实时序差异
- 当前 fast timing model 使用相同的 bandwidth 模型，无法体现介质差异

---

## 不变量

- ✅ SGLang 原生代码 **0 修改**
- ✅ **51** 单元测试全通过
- ✅ HBF-off 基线不回归
- ✅ V1a 算法未修改

## 下一步

1. **Phase 3 MQSim**：用 channel/queue 模型校准 HBF vs DRAM 的真实时序差异
2. **16K 公平实验**：预期与 4K 结果一致（A=E，差异仅在 Write mode）
3. **共享 Prefix 工作负载**：测试 L2 命中场景下 HBF 是否有优势
4. **Async Write 完善化**：增加 Pending Write 状态追踪，确保数据一致性
