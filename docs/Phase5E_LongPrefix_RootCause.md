# Phase 5E.1：4K/16K Prefix 只恢复 128 Tokens 的根因分析

> 日期：2026-11-05 · 方法：阶段级插桩（`HISIM_LPF_DEBUG=1`：树转储 + match 逐节点追踪）+ 最小复现 + 源码核查（sglang 0.5.19 unified_cache / cache_controller / sim_storage）
> 原始数据：`/tmp/p5e_probe_*.log`、`/tmp/p5e_p*.log`（14 组前缀扫描 + 对照）

---

## 1. 结论先行（A/B/C 判定）

**5D 现象定性：Case B 的容量亚型 + 环境伪影，而非缓存/介质/对象机制的缺陷。**

- **A（缓存只存了 128 tokens）？否**。树转储证明 4K 前缀作为单一节点完整备份：
  `N14 p=5 key_len=4089 host_len=4089 host=(13,4101) evicted=1 backuped=1`——backup op 全量（`4090tk→HBF (469.07MB)`、16K 同理 `16378tk→1878MB`）。
- **B（存了完整、但匹配/恢复只找到 128）？是，但机制是容量-驱逐-retraction 级联**：reuse 的**第一次 match 是完整的**（`host_hit=4089, N15(key=4089, host=4089)`），第二次 match 变 0——两次之间对象被摧毁（见 §3 级联链）。
- **C（存恢完整、消费只算 128）？否**。被消费的 128 tokens 是真实的部分命中（`pf_complete=128`、LoadBytes=14MB、守恒 `0+128+3962=4090/4090` 全对）——缓存语义诚实，无虚计。

**128 的来源**：sglang 存储层常量 `STORAGE_BATCH_SIZE = 128`（`mem_cache/hicache_storage.py:23`）——被驱逐对象经**异步线程**按 128 token/批写 L3；reuse 探测（`_storage_hit_query`）落在写入进行中，只找到已完成的**第一批** → 恰好 128。4K 与 16K 出现**完全相同的 128**（时序竞态不会两次恰好相等）证实了批粒度机制。

## 2. 证据链

### 2.1 容量充足时全量恢复（决定性对照）

| 配置 | max_total | host | 复用命中 |
|---|---|---|---|
| 2 builds + 1 reuse，ratio 3.0，无 L3 | 4096 | 12288 | **4089/4090（469MB）全量** |
| 同上 + storage 后端 | 4096 | 12288 | **4089 全量** |
| 5D 原负载（3 builds + 2 reuses），ratio 2.0 | 4096 | 8192 | 128 + 0（复现 5D） |
| 5D 负载 + **正确配比**（max_total 8192） | 8192 | 16384 | **2×4095 全量（2×469MB）** |

### 2.2 5D 失败的机制级联（插桩还原）

```
L1(4096)=1 个 4K 前缀；host(8192)=2 个 4K 对象（8186/8192，零余量）
├─ build1 驱逐 build0 → backup#1(4090→L2) ✓
├─ build2 驱逐 build1 → backup#2(4090→L2) ✓（L2=8186，满）
├─ R0/R1(t=50) 同时到达，各自 match 到 4089（完整！）
├─ 批次组建：2×4089+16 > 4096(剩余) → 调度器先驱逐 build2 → backup#3
│    └─ host 余量 6 → 驱逐 LIVE L2 对象（LRU）→ 对象写入 L3（异步 128/批）
├─ 另一 reuse 的对象被驱逐 → 其 match 变 0 → 探测 L3 → 命中第一批=128
└─ 在途 host-load 的请求被 retraction → retraction_discard 释放其 host backup
     → 该请求第二次 match=0 → 全量重算（58.6ms）
```

关键代码（sglang 原生，零修改）：
- `unified_radix_cache.retraction_discard()`：retraction 释放 host_indices（被中止请求的加载作废——正确语义）
- `full_component.drive_host_eviction()`：host LRU 驱逐
- `cache_controller._page_backup()`：按 `STORAGE_BATCH_SIZE=128` 分批异步写
- `full_component.finalize_match_result_in_tree_core()`：host_hit = 逐节点完整 host_value 求和（无粒度截断）

## 3. 前缀尺寸扫描（正确配比：max_total ≥ 2×prefix，host ≥ 3×prefix 有效容量）

| prefix | A（纯 DRAM） | B70（HBF/DRAM 分池） | HBF-only | L1-only |
|---|---|---|---|---|
| 128 | 254（纯 L1，未触发 host） | 254 | — | — |
| 256 | **510 = 2×255 全量** | **510** | — | — |
| 512 | **1022 = 2×511** | 1022 | — | — |
| 1024 | **2046 = 2×1023** | 2046 | — | — |
| 4096 | **8190 = 2×4095** | 8190 | 8190（2×4095 全 HBF） | 0 ✓ |
| 16384 | **32766 = 2×16383** | 32766 | — | — |

- 每组守恒（`L1+L2+RC = actual`）全 True；host_loads 明细见 `/tmp/p5e_parse_p*.log`
- **A 与 B70 覆盖率逐 token 相等**——HBF 分池不损失任何可复用前缀
- 128tk 场景未触发 host 路径（max_total 512 ≥ 3×128 无驱逐），其复用走 L1（127/128）——符合预期

## 4. 故障域定位（用户要求的六选一）

| 候选域 | 判定 | 证据 |
|---|---|---|
| 原生 Radix Cache | **否** | 节点级 host_value 完整（4089）；finalize 求和无截断 |
| Chunked Prefill | **否** | chunked_prefill_size=16384 ≥ 单请求，无分块残留 |
| 节点 Write-through | **否** | backup op 一次成节点（4090tk 单 op） |
| HBF Backup Adapter | **否** | HBF-only 对照全量恢复；backup 日志全量 |
| Restore Matching | **否** | 第一次 match 4089 完整 |
| **容量/淘汰过程** | **是** | L1 零余量→retraction 丢弃在途加载；host 零余量→LRU 驱逐 LIVE 对象→异步 L3 写第一批 128 |

## 5. 修复（P2：最小干预，SGLang 0 修改，V1a 不变）

1. **Prefix Capacity Rule（配置规则 + 运行时守卫）**：
   - `max_total_tokens ≥ n_concurrent_reuse × prefix + margin`（否则 retraction 摧毁在途 host 加载）
   - `host ≥ (live_objects + reinsert_wave) × prefix`（否则复用波的自插入驱逐 LIVE 对象）
   - 实现：`[HBF-WARN] prefix-capacity rule` 一次性运行时告警（hook 层，backup 时检测对象 >50% host 池）+ 本文档规则
2. **G4 wait-then-read**（顺带修复 5E.0 审计 #6 的潜在假快恢复）
3. **不做的**（按用户约束）：不强制全前缀命中（部分 L3 命中 + 诚实重算是正确缓存语义）；不改 V1a/调度/驱逐算法（V1b 留待后续）；SGLang 零修改

## 6. 期望可复用范围（P2 验收项 2）

按实际 Prefix Cache 匹配规则（节点边界 + `page_size=1` + auto-truncate 边界）：
- prefix P（P ≤ max_total−8）：backup = P（单节点），restore = P−1（首 token 不命中，SGLang 匹配语义）± truncate 6
- 多对象并发驻留要求单池连续容量 ≥ 对象大小（TieredHostKVCache `available_size=MAX` 语义）
- 128 = L3 异步写批粒度（STORAGE_BATCH_SIZE），仅在对象被 L2 驱逐后作为部分命中出现——非错误

## 7. 对 5D 报告 §4 的修正

5D "大前缀限制（V1a 备份路径节点覆盖缺陷）"的表述**错误**：备份路径无缺陷；真因是实验配比（L1/host 零余量）触发的 retraction/驱逐级联。5D §4 表中 p4k/p16k 的 0.6%/0.2% 命中是配比结果，非机制结果；p3 修复后矩阵（§P3 报告）给出正确配比下的真实数字。
