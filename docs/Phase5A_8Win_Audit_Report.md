# Phase 5A：《8-WIN HBF 参数能力审计报告》

> 日期：2026-11-05 · 审计基线：分支 `develop_dqy_0.5.19_qwen3_hbf`（Phase 4-Closure 后，86 测试通过）
> 审计对象：`types.py HBFConfig` / `manager/state.py` / `hbf_page_sim.py` / `hbf_shared_resource.py` / `hbf_io_trace.py` / `sglang_hook.py` / 既有测试与配置脚本 / HBFSim 参考源
> 专家目标（单 WIN）：容量 100GB，顺序读 200GB/s，随机读 100GB/s，写 10GB/s，读延迟 5–10μs，P/E >100K，接口 UB×4@224G；8-WIN 聚合：800GB / 1600 / 800 / 80 GB/s

---

## 1. 逐项审计结论（对应八项审计问题）

### Q1：HBF 容量如何计算？能否支持 800GB 逻辑容量而不申请同等物理内存？

**结论：✅ 支持，无需改动物理内存。**

- 容量入口是 **token 数**（`HISIM_HBF_LOGICAL_SIZE`，默认 100000），`capacity_gb` 为派生显示值（`hbf_size × size_per_token / 1e9`，`sglang_hook.py:120`）
- `MockHBFPool` 物理分配为 **微型占位缓冲**：`torch.zeros((2,1,logical_size,1,1), fp16)` ≈ **4 B/token**（`hbf_tiered_pool.py:125`），注释明确 "No real memory allocation"
- 800GB @ Qwen3（114,688 B/token）= **6,975,660 tokens** → mock 缓冲仅 ~28MB + free-list 索引 ~63MB，完全可行
- 唯一注意点：`hicache_ratio` 换算公式 `(Host总容量−hbf_size)/max_total_tokens` 需按 800GB 等效 token 数重算

### Q2：是否支持 8 个 WIN 分别管理容量、队列和 Busy Until？

**结论：❌ 不支持。当前是单聚合资源视图。**

- `SharedHbfResourceManager` 全局单例：**一个** `media_busy_until` + **一个** `ic_busy_until` 游标（`hbf_shared_resource.py:115-116`）
- 所有 I/O（不论逻辑归属哪个对象）共享同一游标 → 无法表达"WIN 间并行、WIN 内串行"
- 容量侧 `TieredHostKVCache` 也是两个整体池（HBF/DRAM），无 WIN 划分

### Q3：PageSim 的 Channel/Plane/Page 几何能否适配单 WIN 新性能？

**结论：⚠️ 参数化上可适配，但新规格的底层几何未知，不能反推。**

- `PageLevelMediaSim(channels, planes_per_channel, page_bytes, read_latency, write_latency, channel_rate)` 全参数化——机制上可以配任何几何
- 当前默认（承自 HBFSim nominal.json）：32ch × 1.6GB/s = 51.2GB/s、16KB 页、8 die × 4 plane、tR 10μs / tW 100μs
- 专家目标单 WIN 顺序读 200GB/s：若用 32 通道需 6.25GB/s/通道——**这是"调数字凑结果"，明确禁止**。新 HBF 的通道数/页大小/Plane 数/介质速率均未知 → 层 B 只能作显式假设参数，不能作为效率结论依据

### Q4：是否支持顺序读、随机读、写入三种独立性能参数？

**结论：❌ 不支持。**

- 现有只有二维：`read_bandwidth_gb`（512，仅 fast 模型用）与 `write_bandwidth`（=read×0.2，仅 fast 模型用）；`PageSim` 读共用 tR/通道速率，写共用 tW/通道速率
- **无 Sequential/Random 之分**：任何读 I/O 用同一套参数

### Q5：能否通过实际地址访问模式确定 Sequential/Random？

**结论：❌ 当前 I/O 路径不带地址，无法判定。**

- 时序入口只收 `(arrival, bytes, is_read)`（`mgr.submit(arrival, int(num_bytes), is_read=...)`，`sglang_hook.py` `_hbf_channel_transfer_duration`）
- **地址信息其实存在但未接入**：备份路径有 `op.host_indices`（虚拟索引张量，`mock_start_writing`），恢复路径 HOST-LOAD 有 rid/token 数；`PageSim` 内部有全局页序游标（隐含顺序性）
- 判定机制（连续性检测）需要把 host_indices 的掩码/连续性传入时序层——属于 5B 新增能力，5B 先提供显式 mode + 对象级流游标自动判定，端到端地址贯通列为 5C 工作

### Q6：媒体与互连时间计算、资源竞争、流水线是否存在重复计算？

**结论：✅ 无重复计算，但有一处语义需在新模型中保持一致。**

- 媒体时间（WIN/NAND 内部数据通路）与互连时间（HBF→HBM 链路）是**两个不同资源**，`total = max(media, ic) − arrival`（`hbf_shared_resource.py:268`）建模流水重叠——同一字节的介质读出与链路传输重叠，取 max 而非求和，与 HBFSim calibrator.cpp:46-64 的 `fast_service_ns` 语义一致（该注释明确"later of the two bounds, not their sum"）
- 页级模型中通道数据输出（page/rate）计入媒体时间；互连游标独立推进——若未来在 WIN 内部再加一层"数据输出到 WIN 边界"的传输，**必须**与互连合并为一层，否则重复计费。5B 设计：WIN 内部（media cursor）+ 封装接口（interface cursor）各一层，max 语义

### Q7：UB×4@224G 接口信息缺少哪些条件？

**结论：❌ 无法定参，至少缺 7 项**（详见第 3 节缺失清单）：
①"G"是 Gbps 还是 GB/s；②"×4"是 4 条 lane 还是 4 个独立接口；③单 WIN 接口还是封装级共享（专家已提示"优先理解为 8 个 Die 组成的资源池/堆叠，不得当作 8 个独立对外 Stack"）；④全双工/半双工（读写并发）；⑤协议编码效率（有效载荷比）；⑥接口固定延迟；⑦224G 是单向还是双向聚合。
**数量级推演**（供参考，不作结论）：若 4 lane × 224Gbps = 112GB/s 单向——若为单 WIN 独享则与 200GB/s 顺序读矛盾（112 < 200），若为封装共享则 8-WIN 聚合被压到 112GB/s ≪ 1600GB/s 目标；若 224G=GB/s/lane 则单 WIN 896GB/s 不受限。三种解读结果完全不同 → **必须作为可配置假设项**，默认不启用接口约束，实验显式开启

### Q8：HBFSim Reference / MQSim 是否有可复用时序或标定能力？

**结论：✅ HBFSim 源码可用（本机 `/home/d00954154/HBFSim_src/HBFSim-main`），可复用两项；MQSim 仍不可用。**

- **可复用 ①**：`configs/profiles/nominal.json` 完整 profile（1TiB/512GB/s/32ch/16KB 页/tR 10μs/tW 100μs/QD128）——已被 Phase 1 复用为默认值
- **可复用 ②**：`src/hybrid/calibrator.cpp` 的 `fast_service_ns = max(latency, transfer)` 时序语义与 HybridSampler 采样标定框架（sample_rate 0.01、warmup 1024）——标定框架**需要真实设备测量数据**才能运转，当前无硬件 → 无法产出标定表
- MQSim：容器无网络，`third_party/mqsim` 为空，**依旧阻塞**（Phase 4 遗留）
- 专家 8-WIN 规格（100GB/WIN、200/100/10 GB/s）与 HBFSim nominal（1TiB、512GB/s）**数值体系完全不同**——HBFSim 只作方法论参考，不作参数来源

---

## 2. 完整参数表（现状 → 专家目标 → 必要改动）

| # | 参数 | 单位 | 当前默认值 | 配置入口 | 代码使用位置 | 真正生效？ | 专家目标 | 必要改动 |
|---|---|---|---|---|---|---|---|---|
| 1 | HBF 逻辑容量 | token | 100000（≈11MB@Qwen3） | `HISIM_HBF_LOGICAL_SIZE` | `hbf_tiered_pool.py:104`（MockHBFPool）、hook:120（capacity_gb 派生） | ✅ | 800GB=6,975,660tk | 加 `HISIM_HBF_CAPACITY_GB` 换算入口；物理内存无需变（4B/tk 占位） |
| 2 | 顺序读带宽 | GB/s | 512（仅 fast 模型） | `HISIM_HBF_READ_BW_GB` | `_hbf_fast_transfer_duration`（hook:415）；channel 模型**不用**它 | ⚠️ fast 才生效 | **200/WIN** | 新增独立参数（当前无 seq/rand 之分） |
| 3 | 随机读带宽 | GB/s | ❌ 无此参数 | — | — | ❌ | **100/WIN** | 5B 新增 |
| 4 | 写带宽 | GB/s | read×0.2=102.4（fast）；channel 模型读写共用媒体游标 | `HISIM_HBF_WRITE_BW_GB` | fast 模型 | ⚠️ | **10/WIN** | 新增独立参数（写成为显著瓶颈项） |
| 5 | 读延迟 | μs | 10 | `HISIM_HBF_READ_LATENCY_US` | fast 基座；PageSim tR | ✅ | **5–10** | 取中点 7.5 为默认（范围参数，实验扫） |
| 6 | 写延迟 | μs | 100（承 HBFSim nominal，**非专家给定**） | `HISIM_HBF_WRITE_LATENCY_US` | fast 基座；PageSim tW | ✅ | 未给出 | 保留为**假设参数**并标注 |
| 7 | Channel 数 | 个 | 32 | `HISIM_HBF_CHANNELS` | PageSim/manager 几何 | ✅ | **未知** | 层 B 假设参数，显式标注 |
| 8 | 单通道速率 | GB/s | 1.6（8b×1600MT/s） | `HISIM_HBF_CHANNEL_RATE_MTPS`/`WIDTH_BITS` | `per_channel_bw_bytes_per_s`（types.py:147） | ✅ | **未知** | 同上 |
| 9 | 聚合媒体带宽 | GB/s | 51.2（channel 模型实际用值） | `HISIM_HBF_MEDIA_BW_GB`（4-Closure 新增覆盖） | `mgr.configure(media_bw=...)` | ✅ | 1600/800/80（8-WIN 聚合，分模式） | 5B 由 8×单 WIN 分模式参数聚合得到 |
| 10 | 页大小 | KB | 16 | `HISIM_HBF_PAGE_BYTES` | PageSim 页拆分 | ✅ | **未知** | 层 B 假设参数 |
| 11 | Die×Plane | — | 8×4=32/WIN 内 plane | `HISIM_HBF_DIES_PER_CHANNEL`/`PLANES_PER_DIE` | PageSim plane 游标 | ✅ | **未知** | 层 B 假设参数 |
| 12 | P/E 寿命 | 次 | ❌ 不建模（时序无关） | — | — | ❌ | >100K | 不影响时序；影响 GC/寿命建模（5C+ 议题，若写量大会引入 GC 损耗——**Unknown**） |
| 13 | 互连带宽 | GB/s | 51.2（默认=媒体聚合） | `HISIM_HBF_INTERCONNECT_BW_GB` | manager IC 游标 | ✅ | UB×4@224G **解读未定**（112 或 896 或共享） | 5B：`per_win_interface_bw` + `package_shared_interface_bw` 两个可配置假设项，默认不启用 |
| 14 | 共享下游带宽 | GB/s | 不设（独立拓扑） | `HISIM_SHARED_DOWNSTREAM_BW_GB` | 下游游标 | ✅ | —（拓扑问题，独立于 WIN） | 保持，与 WIN 接口层正交 |
| 15 | 队列深度 | — | 128（承 nominal） | HBFConfig.queue_depth | manager admission（仅统计近似） | ⚠️ 准入逻辑简化 | **未知** | 5B 每 WIN 独立队列计数 |
| 16 | WIN 数量 | 个 | 1（隐式，无 WIN 概念） | — | — | ❌ | **8** | 5B 核心新增：`n_wins` |
| 17 | 访问模式判定 | — | ❌ 无（所有读同参数） | — | — | ❌ | 需按地址模式定 seq/random | 5B：显式 mode + 对象流游标自动判定；地址贯通 5C |
| 18 | 媒体模型 | — | formula（Theory Bound）/ pagesim | `HISIM_HBF_MEDIA_MODEL` | manager dispatch | ✅ | 层 A 包络 / 层 B 几何 | 5B 新增 `win8` 模式（内含 A/B 两层） |
| 19 | 放置策略 | — | V0 比例分流（HBF↔DRAM 层面） | `HISIM_HBF_RATIO` | `PlacementPolicyV0` | ✅ | WIN 间均衡/热点集中 | 5B 新增 WIN 级 placement |
| 20 | 跨批状态 | — | 忙游标 max(…,batch_start) 保留 | — | manager.reset/begin_batch | ✅（3E 修复） | 需保持 | 5B 每 WIN 游标同语义 |

**StateManager 审计**：`_hicache_hbf_load/backup_dur`、`_last_hbf_*` 字段（state.py:15-18）与 WIN 数量无关（聚合计量），5B 无需改动；I/O Trace 的 `hbf_io_trace.py` 按 I/O 记录、天然兼容多 WIN（需在 entry 中加 `win` 字段——5B 顺手补上）。

---

## 3. 缺失参数与硬件不确定性清单（交付物 2）

| 类别 | 缺失项 | 影响 | 当前处置 |
|---|---|---|---|
| **接口** | UB×4@224G 的 G 单位/lane 数/归属（单 WIN vs 封装共享）/双工/编码效率/接口延迟 | 决定 8-WIN 聚合能否到 1600GB/s（112~896GB/s 三种解读） | 双假设参数可配置，默认不启用约束；实验显式开启 |
| **几何** | 通道数、页大小、Die/Plane 数、介质内部速率 | 层 B（Page/Channel/Plane 损耗）无法标定 | 显式假设参数（默认沿用 HBFSim nominal，标注"假设，非专家给定"）；**禁止调参凑 80%** |
| **写路径** | 写延迟、写内部机制（SLC cache? 编程模式） | 小写 I/O 时序、读写竞争强度 | 写延迟=假设参数（100μs ballpark） |
| **模式判定** | 真实 KV 访问的地址分布（顺序/随机占比） | seq/rand BW 选择 | 5B 对象流游标自动判定 + 显式 mode；端到端 host_indices 贯通列为 5C |
| **寿命/GC** | P/E>100K 下的 GC 触发率、写放大 | 长稳态写带宽衰减 | **Unknown，不建模不赋值** |
| **标定** | 真机测量数据（供 calibrator 框架） | 效率结论只能停留在模型层 | HBFSim HybridSampler 框架就绪但无数据；MQSim 依旧阻塞（无网络） |
| **拓扑** | 8 WIN 是 8 Die 堆叠还是 8 独立 Stack（专家已警示） | 并行度上界与接口归属 | 按专家提示默认"Die 池/堆叠"建模（媒体级并行、接口归属待定） |

---

## 4. 审计总结：现状能力 vs 5B 需求差距

| 5B 需求 | 现状 | 差距 |
|---|---|---|
| 8×100GB 逻辑容量无物理内存 | ✅ 已支持（token 逻辑 + 4B/tk 占位） | 加 GB→token 入口 |
| 每 WIN 独立容量/队列/Busy | ❌ 单聚合游标 | **核心新增** |
| seq/rand/write 三带宽 | ❌ 仅 read/write 二维（且 channel 模型混用） | **核心新增** |
| 地址判定 seq/random | ❌ 地址未进时序层 | 机制新增 + 贯通延后 |
| WIN 间并发 + 共享接口 | ❌ | **核心新增** |
| Object 级放置（均衡/热点） | ❌（V0 是 HBF↔DRAM 池间） | 新增 WIN 级 |
| 读写竞争 + 跨批连续性 | ✅ 语义已有（单资源级） | 复制到每 WIN |
| 参数集中配置 + 快照 | ⚠️ 环境变量散落（16 个） | 新模型收敛为单一 `Hbf8WinConfig` + JSON |
| 旧 86 测试/HBF-off 不回归 | ✅ | 新模式 env 门控、默认关闭 |

**结论**：现有代码提供可复用的骨架（时序 choke point、跨批游标语义、trace、虚拟池），但 8-WIN 语义（多资源并行、三模式带宽、WIN 级放置）必须新建。5B 按"层 A（专家包络）/层 B（假设几何）"双层实现，所有新参数收敛进 `Hbf8WinConfig` 单一配置对象。
