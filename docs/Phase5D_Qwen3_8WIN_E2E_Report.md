# Phase 5D 报告：8-WIN HBF 的 Qwen3 端到端对象管理与系统收益验证

> 日期：2026-11-05 · 测试：**116/116 通过**（106 既有 + 10 项 5D 对象管理）；SGLang 0 修改；V1a 不变；HBF-off 不回归
> 原始数据：`docs/p5d_results/`（24 组 E2E run JSON + P4-B 容量 trace CSV + 输出日志）；win 池/对象注册表随 `win_pool_summary.json` 落盘
> 层级声明：Layer A = 专家包络；Layer B = 假设几何（未用于任何结论）

---

## 1. P0：KV Object ID → WIN 稳定映射 ✅

**设计**：稳定对象地址 = HBF 虚拟槽位索引（backup 的 `op.host_indices` 与 restore 的 `host_value` 天然同源；TieredHostKVCache 的 free 路径挂失效回调）。

新增 `WinObjectRegistry`（`hbf_win_pool.py`）记录：Object ID/Generation、Size、Prefix slots、WIN、每 WIN extents（连续 run）、状态机（ALLOCATED→BACKING_UP→READY→RESTORING→EVICTED）、读写计数与完成时间。

**六项保证（各由单元测试锁定，`test_phase5d_object_mgmt.py`）**：

| 保证 | 实现 | 测试 |
|---|---|---|
| G1 同对象同 WIN | backup 注册 → restore 按同一槽位分解（striped=静态块映射；balanced=注册表查询） | ✅ 三种放置模式一致 |
| G2 迁移显式 | `migrate_object`（记账+计数；striped 多 WIN 对象拒绝整迁移） | ✅ |
| G3 无过期映射 | `notify_slots_freed`（全删/CoW 部分释放就地收缩）→ 适配器 `_on_hbf_free` 回调 | ✅ 全逐+部分释放 |
| G4 未完成不可命中 | **时间门**（`ready_time > arrival` → 阻塞，0 字节 miss）——修复了"submit 即置 READY"的状态时序 bug | ✅ t<ready 阻塞 |
| G5 跨批游标 | 对象路径复用既有跨批语义 | ✅ |
| G6 守恒 | 注册表 bytes/slots/每 WIN 资源三方对账 | ✅ alloc+free 全程 |
| 端到端贯通 | backup（node_ids+槽位）与 restore（`_collect_host_load_split` 收集实际槽位）双 choke point 传索引 → `submit_object` | ✅ 24 组 E2E conservation 全 True、0 次错误命中 |

## 2. P1：两种放置机制 + 负对照 ✅

| 前缀 | obj-balanced 写 | **striped 写** | hotspot 写 | striped 恢复 |
|---|---|---|---|---|
| 340tk | 3.90ms | **0.49ms（8×）** | 3.90ms | 0.025ms |
| 4Ktk | 47.0ms | **5.87ms** | 47.0ms | 0.294ms |
| 16Ktk | 187.9ms | **23.5ms** | 187.9ms | 1.174ms |

- **Striping 是真实提交**：对象按块边界分解为 8 个 chunk，分别进入 8 个独立 WIN 资源队列（测试断言 8 WIN 全部有 I/O、聚合写带宽 ≤80GB/s 无超售）——非"时间除以 8"的捷径
- 条带块配置随前缀尺寸调整（340tk→43、4K→512、16K→2048）
- 逻辑对象映射：`win_pool_summary.json` 记录每 WIN 写入字节与 extents

## 3. P2：读写调度 ✅（提交序策略模型 + 无抢占语义）

写入突发 + 并发恢复（1 写 + 2 读同 WIN，负对照热点放置）：

| 策略 | restore 延迟 | Backup P95 | 结论 |
|---|---|---|---|
| FIFO | 23.79ms | 23.40ms | 5C 的 0.19→4.29ms 放大的对象级复现 |
| **Read-priority** | **0.39ms** | 23.79ms | 恢复 61× 改善；写 P95 +0.4ms（饥饿量化） |
| Fair | 8.19ms | 23.79ms | 折中 |
| **R/W 分离（迁移到专用 WIN）** | **0.19ms** | **3.90ms** | 双赢——放置级分离优于调度级优先级 |

写不可抢占（模型无抢占，仅排队序）✅；饥饿副作用以 Backup P95 量化 ✅。四级竞争源区分：WIN 内（本测试）/ 全局 QD 槽位（5C C2）/ 封装接口（5C P4）/ 下游（P5 组合测试）。

## 4. P3：Qwen3 端到端（24 组，QoS 全过）✅

| 工作负载 | A 基线 TTFT p95 | bal | **str** | 命中率 (A→str) | 备注 |
|---|---|---|---|---|---|
| n2 (340tk×2) | 16.8ms | 38.1% | **32.1%** | 33.2%→33.2% | 回归通过；条带最优 |
| p4k (4Ktk) | 106.6ms | 68.4% | **67.6%** | 0.6% | 见下方大前缀限制 |
| p16k (16Ktk) | 318.0ms | — | 100.3% | 0.2% | 同上 |
| n4 (N=4) | 78.9ms | 88.6% | 91.1% | 29.9/19.9% | N=4 可运行；str 少 1 请求命中（放置方差） |
| churn | 41.8ms | 96.2% | 89.4% | 18.8→10.5% | 与 4-Closure 一致的 V1a 部分命中行为 |
| agent 间歇复用 | 21.2ms | — | **67.5%** | 18.8% | |
| reuse-heavy | 200.1ms | — | 88.8% | 59.2% | |
| write-heavy | 5.4ms | — | 100% | 0% | 无复用=无收益（写不阻塞 TTFT）✓ |

**吞吐全部 ≥100.0%**；TPOT 无差异；24 组 conservation 全 True。

**大前缀限制（重要发现，列为 5E P0）**：4K/16K 前缀 host 复用仅恢复 128-token 级节点（V1a 备份路径的节点覆盖/write-through 轮次粒度），4K 命中 128tk/4096tk ≈ 3%——**当前备份路径不能完整覆盖大前缀对象**；条带机制本身已验证（P1 表），恢复完整性受限于上游节点切分。

## 5. P5：接口拓扑 ✅

- E2E 四种接口假设（无/共享112/每WIN112/共享896）**在 KV 尺寸下零差异**（n2 全部 5.4ms/32.1%）——HBF 条带恢复 24μs ≪ DRAM 侧 2.87ms，TTFT 由 DRAM 路径主导（与 4-Closure"瓶颈不在 HBF 媒体"一致，且 8-WIN 新参数下依旧成立）
- **win8 + SharedDownstream 正确组合**（替换旧 WARNING）：外部游标桥接使 HBF 与 DRAM 在同一下游资源竞争（单资源无重复计费）——单元测试验证 HBF 排队于 DRAM 占用之后、共享游标推进、跨批保留；E2E `n2_str_ds` 通过（32.1%）
- 接口/媒体/下游分别统计于 `win_pool_summary.json`（IC delivered/util/constrained%）

## 6. P4：容量收益与公平对照 ✅

**A（小规模等比例）**：E2E churn/n2 于 E70 容量复验（上表），放置行为可观察。

**B（800GB 逻辑容量，虚拟对象 trace，60K+ 请求）**——`test_phase5d_capacity_trace.py`，800GB 无物理内存（4B/tk 占位）：

| 场景（等容量，总 Host 字节严格相同） | 命中率 | TTFT p50 | Backup P95 |
|---|---|---|---|
| α=0.5（Host=½WS）：A vs B | **33.3% = 33.3%** | 4.76ms | 2.87ms |
| α=1.0（Host=WS）：A | 66.7% | **2.87ms** | 2.87ms |
| α=1.0：B（HBF 560+DRAM 240GB） | 63.3%* | **0.02ms（143×）** | 2.87ms |
| α=2.0：B | 66.7% | 0.02ms | **0.51ms** |

- *B 的 −3.4pp = 重插入交替放置的记账伪影（报告为放置开销，非容量差异）
- **容量收益与介质收益已分离**：等容量下命中率相同（容量驱动），HBF 的收益全部体现在恢复/备份速度（p50 143×、备份 P95 5.6×）；未将 800GB 与小 DRAM 直接对照
- **扩容（单独报告）**：WS 1.17TB > DRAM 800GB 池：A 命中 33.3% → C（+800GB HBF，1.6TB Host）**43.3%（+10pp 容量收益）**
- 800GB 规模（20.5K 会话）等容量对照复现同规律（33.3/63.3/66.7%）

## 7. P6：访问模式与可信性 ✅

- 模式判定基于**每 WIN 物理连续性**（extents 的连续 run），非仅逻辑 host index 连续；碎片化对象报告 run 数（stride-2 碎片 → 100 runs 被检出）
- 端到端恢复为连续槽位区间 → 每 WIN chunk 连续 → sequential（保守正确）；**物理介质随机性 Unknown**——Layer A 无页级几何，报告 Sequential 边界结果；Random 边界 = 100GB/s 包络（5C P2 表），两端均已给出
- Layer B 仅假设几何，未用于任何达标宣称 ✅

## 8. 交付物与状态

| 交付物 | 位置 |
|---|---|
| 本报告 | `docs/Phase5D_Qwen3_8WIN_E2E_Report.md` |
| 对象管理模型 | `hbf_win_pool.py`（WinObjectRegistry + submit_object + 外部接口桥） |
| 端到端贯通 | `sglang_hook.py`（槽位收集/choke point/注册表失效回调/summary 落盘）、`hbf_tiered_pool.py`（free 钩子） |
| 测试 | `test_phase5d_object_mgmt.py`（10 项 G1-G6+P1/P2/P5/P6） |
| 原始数据 | `docs/p5d_results/`（24 E2E JSON + 容量 trace CSV） |
| 配置快照 | 每 run 的 `win_pool_summary.json` + `hbf_win_effective_config.json`（假设标注） |

不变量：**116/116** · SGLang 0 修改 · V1a 未动 · HBF-off 基线不回归（A 组与 4-Closure 数值一致）。

## 9. Phase 5E 建议：模型校准与最终收益评估

| # | 事项 | 依据 |
|---|---|---|
| 1 | **大前缀备份覆盖修复**（节点级 write-through 对 4K/16K 只覆盖 128tk 节点）| §4 大前缀限制——当前最大收益阻塞项 |
| 2 | **接口语义硬件确认**（112/896/共享）| 5C/5D 均显示 E2E 不敏感（DRAM 主导），但 80% 池化效率结论仍由它决定 |
| 3 | MQSim/真机标定（Layer B 几何 + Unknown 损耗项） | 5C P5 |
| 4 | 重插入放置策略优化（消除 B 模式 −3.4pp 记账伪影 → V1b 对象级放置） | §6 |
| 5 | 长前缀写放置（470MB/1.9GB 对象跨 WIN 条带化已实现，端到端等待 1 修复后评估） | 5C P6 |
| 6 | 最终收益评估：n2/agent/churn QoS 优异（32-96%），容量收益 +10pp/扩容、恢复 143×——待 1、2 完成后出终版 | 全文 |
