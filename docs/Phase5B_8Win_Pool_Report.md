# Phase 5B 报告：最小 8-WIN Pool 原型 + 第一轮验证

> 日期：2026-11-05 · 前置：Phase 5A 审计（`Phase5A_8Win_Audit_Report.md`）
> 测试：**98/98 通过**（既有 86 + 新增 12 WIN 测试）；SGLang 0 修改；HBF-off 基线不受影响（新模式 env 门控）
> ⚠️ 本阶段只验证**模型正确性与硬件配置可行性**，不宣称真实 HBF 达到 80% 效率

---

## 1. 交付物清单

| 交付物 | 位置 | 状态 |
|---|---|---|
| 1. 8-WIN 参数能力审计报告 | `docs/Phase5A_8Win_Audit_Report.md` | ✅ |
| 2. 缺失参数与硬件不确定性清单 | 审计报告 §3（7 类：接口/几何/写路径/模式判定/寿命 GC/标定/拓扑） | ✅ |
| 3. 最小 WIN-aware Pool 模型 | `src/hisim/simulation/hbf_win_pool.py` | ✅ |
| 4. 单 WIN / 8 WIN 正确性测试 | `test/test_hbf_win_pool.py`（12 项） | ✅ 12/12 |
| 5. Runtime Effective Config | `Hbf8WinConfig.dump_effective_json` → `hbf_win_effective_config.json`（含假设标注） | ✅ |
| 6. Phase 5C 实验计划 | 本报告 §5 | ✅ |
| 第一轮验证（6 场景） | `test/test_phase5b_validation.py` | ✅ 7/7 检查通过 |

---

## 2. 模型实现（`hbf_win_pool.py`）

```
Hbf8WinConfig（唯一参数对象，JSON 可加载）
 ├─ n_wins=8, win: WinSpec{100GB, 200/100/10 GB/s, 7.5μs/100μs}
 ├─ layer: "envelope"（层 A：专家标定包络）| "pagesim"（层 B：假设几何）
 ├─ default_access_mode: sequential / random / auto（对象流游标连续性判定）
 ├─ placement: balanced（轮转）| hotspot（指定 WIN 先填满再溢出）
 ├─ per_win_interface_bw / package_shared_interface_bw（UB×4@224G 未定 → 双假设项，默认不启用）
 ├─ interface_full_duplex（默认 false=半双工，保守假设）
 └─ pagesim_assumptions（层 B 几何，全部显式假设，默认 HBFSim nominal 占位，
    注释明令"禁止调参凑 200GB/s/80% 效率"）
      │
HbfWinPoolModel
 ├─ 8 × WinResource：每 WIN 独立媒体忙游标（读写竞争=同 WIN 串行）、
 │    队列计数、流游标、吞吐统计（seq/rand/write 字节分开）、接口受限计数
 ├─ 对象注册表 + 容量分配（alloc_object/free_object，字节级）
 ├─ 共享接口游标（读写同游标=半双工；full_duplex 时分游标）
 └─ begin_batch 保留全部游标（跨批未完成 I/O 状态连续）
```

**接入方式**（原始 Phase 4 模型不变，可选模式）：
- `HISIM_HBF_MEDIA_MODEL=win8` → `SharedHbfResourceManager.configure(media_model="win8", win_config=…)`
- WIN pool 同时接管**媒体与接口**时序（管理器的 IC 路径旁路，避免双重计费——审计 Q6 结论）
- `HISIM_HBF_WIN_CONFIG=path.json` 覆盖参数（**所有新参数集中在配置对象，不再散落 env**）
- `HISIM_HBF_CAPACITY_GB=800` → 逻辑容量按真实 KV 字节换算 token（800GB@114,688B/tk=6,975,660tk，物理内存仍是 4B/tk 占位——审计 Q1 结论落地）
- 启动时自动输出 effective config 快照（假设标注）到 `Envs.output_dir()`
- win8 + SharedDownstream 组合会打 WARNING（未验证组合，留 5C）

---

## 3. 最小测试结果（12/12 通过）

| # | 测试 | 结果 |
|---|---|---|
| T1 | 单 WIN 容量与带宽生效 | ✅ 38.88MB seq read = 194.4μs（@200GB/s），100GB 容量 |
| T2 | 8 WIN 总逻辑容量 800GB | ✅ 8×1GB 对象均衡分布到 8 个 WIN |
| T3 | 单 WIN 三模式不超标定带宽 | ✅ seq 200 / rand 100 / write 10 GB/s 均不被突破 |
| T4 | 8 WIN 聚合不超 1600/800/80 | ✅ 聚合 1600.0 / 80.0 GB/s——**上限由每 WIN 限值自然涌现，无独立强制** |
| T5 | 单 WIN 负载不误用聚合带宽 | ✅ 单 I/O 194.4μs（200GB/s），非 24.3μs（1600） |
| T6 | 跨 WIN 独立 + 共享接口生效 | ✅ 双 WIN 并行；共享 112GB/s → 聚合压到 112.0（8 I/O 全部受限标记）；每 WIN 独立接口 → 聚合 896 |
| T7 | 跨批未完成 I/O 不被重置 | ✅ 批 2 读等待在途写（3.9ms 游标保留） |
| + | 放置策略 | ✅ balanced 分散 4→4 WIN；hotspot 30GB×5 → [0,0,0,1,1]（先填满再溢出） |
| + | 读写竞争 | ✅ 同 WIN 读等待写完成 |
| + | auto 模式判定 | ✅ 连续=sequential ×2，跳跃=random |
| + | effective config JSON 往返 | ✅ 全参数保留 + assumptions 标注 |
| + | 管理器 win8 分发 | ✅ 双载并行各 194.4μs，跨批保留 |
| T8 | 旧 86 测试回归 | ✅ 98/98 总通过，SGLang 0 修改 |

## 4. 第一轮验证（6 场景 + 变体，全部通过）

| 场景 | 结果 | 受接口约束？ |
|---|---|---|
| V1 单 WIN 38.88MB seq read | 194.4μs，200GB/s | 否 |
| V2 单 WIN 38.88MB random read | 388.8μs，100GB/s | 否 |
| V3 单 WIN 38.99MB write | **3.899ms，10GB/s** | 否 |
| V4a 8 WIN 各 1×38.88MB seq read | 194.4μs，**聚合 1600GB/s** | 否 |
| V4b 同上 + 共享 112GB/s 接口 | 2.78ms，**聚合 112GB/s** | **是（8/8 I/O 标记受限）** |
| V5 8 WIN 各 1×38.99MB write | 3.9ms，聚合 80GB/s | 否 |
| V6 混合 4R+2W + hotspot@win0 + 共享 112 | 8.58ms，读 18.1/写 9.1GB/s（全部挤在 win0：媒体串行+接口串行双重排队，qw=4.68ms） | 是 |

**端到端集成冒烟**（Qwen3 N=2 Host Hit，`HISIM_HBF_MEDIA_MODEL=win8`）：服务器启动、effective config 落盘、38.99MB 备份 = 3899.4μs（精确等于 10GB/s 写包络）、TTFT=4.5ms、守恒 0+339+1=340/340 ✓。

**新规格带来的两个结构性变化**（Phase 4 模型没有的）：
1. **写成为显著瓶颈**：每 WIN 写仅 10GB/s——38.99MB 备份 3.9ms/WIN（旧模型共享 51.2GB/s 时 ~0.76ms 媒体）。写吞吐完全依赖**跨 WIN 并行放置**，写放置策略的权重远大于旧模型
2. **读极快**：200GB/s/WIN，38.88MB 恢复 194μs——恢复侧瓶颈转移到 DRAM 侧/接口假设，与 Phase 4-Closure"互连是瓶颈"的结论需要在新参数体系下重新评估

---

## 5. Phase 5C 实验计划：池化带宽效率扫描（详细）

### 5.0 目标与口径

**问题**：8-WIN 池化配置下，有效带宽效率（Pool Useful BW ÷ 聚合标定 BW）能否超过 0.8？在什么条件（I/O 尺寸 / QD / 读写比 / 访问模式 / 放置 / 接口假设）下达到？

**效率定义（沿用 Phase 4-Closure 口径，适配 WIN 结构）**：
- 单 WIN 效率 = I/O bytes ÷ media_time ÷ 该 WIN 标定 BW
- 池化有效带宽 = 总 bytes ÷ makespan；**池化效率 = 池化有效带宽 ÷ (活跃 WIN 数 × 单 WIN 标定)**（分母必须按活跃 WIN 数缩放——否则闲置 WIN 会虚低效率，这是 5B 测试 T5 锁定的语义）
- 每_WIN 利用率 = 该 WIN media busy ÷ span
- 接口受限率 = interface_constrained I/O 占比

**红线（继承 5A 审计）**：层 B 几何是假设，**不得调参凑 80%**；结论必须标注层（A 包络 / B 假设几何）与接口假设状态。

### 5.1 实验矩阵

| 维度 | 取值 | 目的 |
|---|---|---|
| E1 I/O 尺寸 | 4KB/16KB/64KB/256KB/1MB/4MB/16MB/38.9MB | 小 I/O 延迟受限 vs 大 I/O 带宽受限的分界（层 A 无页概念 → 预期平；层 B 假设几何下复现 Phase 4 形状） |
| E2 QD（并发 I/O 数） | 1/4/8/16/32/64 | 并发填 WIN；QD≥8 时 8 WIN 全激活 |
| E3 读写比 | R100 / W100 / R50W50 | 写 10GB/s 的拖尾效应（预期混合大幅降池化效率） |
| E4 访问模式 | seq / rand / auto | 100 vs 200GB/s 差异；auto 判定命中率 |
| E5 活跃 WIN 数 | 1/2/4/8 | 效率分母缩放验证；单 WIN 高负载 vs 多 WIN 分摊 |
| E6 放置策略 | balanced / hotspot | 热点集中 → 单 WIN 串行化损失（V6 已见 8.58ms） |
| E7 接口假设 | 无 / 每WIN 112 / 共享 112 / 共享 896 | **UB×4@224G 三种解读的敏感性**——这是 80% 结论的最大不确定源 |
| E8 层 | envelope / pagesim(假设几何) | 层 B 损耗分解（页/通道/Plane）对照层 A 包络 |

### 5.2 扫描执行方式

- 全部基于 `HbfWinPoolModel` 独立扫描（复用 5B 骨架），每格输出：单 WIN 效率、池化效率、每 WIN 利用率、qw、接口受限率
- 固定 Trace（同字节同请求序）逐机制开关，沿 Phase 4-Closure P2 瀑布法归因
- KV 相关性场景：38.9MB 恢复突发（N=2/4/8）、混合备份+恢复（写瓶颈的真实 KV 形态）

### 5.3 5C 成功判据（预先声明）

1. 找到 ≥0.8 池化效率的**条件集合**（预期：seq 读 + 大 I/O + QD≥8 + balanced + 无共享接口约束；写侧大概率 <0.8 除非跨 WIN 并行写）
2. 明确"距 0.8 的损耗构成"实测归因（非拍脑袋数字；Unknown 项不赋值）
3. 接口三解读对 80% 结论的**敏感性结论**（若共享 112GB/s 成立，则读侧 80% 池化效率在接口层即被否决——这本身就是重要结论）
4. 输出 `Phase5C_Efficiency_Report.md`，明确标注模型层与假设状态

### 5.4 后续（5D/5E 预告）

- **5D**：Qwen3 端到端（B70/C 容量配置 × 8-WIN 时序 × 独立/共享拓扑；重点验证写瓶颈对 TTFT 的影响与备份放置策略）
- **5E**：KV 容宽比重估（800GB 视角）+ HBF 引入收益终版 + 数据可信性分级

---

## 6. 遗留与风险

| 项 | 状态 |
|---|---|
| UB×4@224G 解读（112/896/共享） | **未决**——5C E7 做敏感性，需向专家回询确认 |
| 层 B 几何 | 假设占位（HBFSim nominal），只作损耗分解参考 |
| 写延迟 | 非专家给定（100μs 假设），对大写无影响（带宽主导），小写敏感 |
| 端到端对象级放置贯通 | 5B 集成为 I/O 级轮转；备份/恢复对象同一 WIN 的映射（node_id 贯通）留 5D |
| MQSim / 真机标定 | 依旧阻塞（无网络/无硬件）——效率结论永远标注"模型层" |
