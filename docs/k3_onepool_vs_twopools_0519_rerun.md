# K3 One Pool vs Two Pools — 0.5.19 复跑报告（Rerun）

> 目标：在本地权威目标 **sglang 0.5.19** 上复跑
> `docs/k3_onepool_vs_twopools_experiment_report.md`（docker fork `0.5.18.dev922`）
> 的关键实验，记录结果并逐项归因与 docker 报告的差异。

## 0. 结论摘要

1. **P5.1（KDA-heavy）与 docker 报告完全一致**：KDA=40/60/80/100/120 的
   COW Hit = 2/3/4/6/5，prefix_ratio = 0.333/0.500/0.666/0.999/0.833，全部逐格吻合
   （TP-120 需按 docker 的共享预算语义设 MLA=98,240，见 §2 归因）。
2. **P5.2（MLA-heavy）4 格中 3 格一致**；TP-80 复跑为 3/3（docker 2/3），
   归因于 docker fork 的 KDA L2/L3 回收链路损坏（`f8985f4` 修复）。
3. **P5.3（动态混合）Two Pools 与 docker 报告逐阶段表（§5.5）完全一致**：
   KDA=80 总 COW 9/15（分阶段 4/6+3/3+0/3+2/3，docker 为 4/6+2/3+0/3+3/3，总数相同）；
   KDA=40 总 COW 11/15 = 2/6+3/3+3/3+3/3。**docker 报告 §6.2 的 "KDA=40 → 14/15"
   与其自身 §5.5 逐阶段表（求和 11/15）矛盾，复跑证实 11/15。**
4. **L2/L3 修复后 Two Pools 的 MLA prefix 达到理想值**（P5.3 复用新 token = 1,545，
   与 One Pool 理想相同），因为被驱逐的 prefix/checkpoint 能从 L2/L3 正确恢复；
   docker 报告中 Two Pools 为 39,433/5,641。
5. **One Pool 在 0.5.19 上跑通**（`606c9cb`，修 3 个 0.5.18→0.5.19 API 断点）：
   P5.1 6/6、P5.2 3/3、P5.3 **15/15 COW + 1,545 理想工作量**，与 docker 报告的
   One Pool 结论完全一致（§6.1）。统一池真正生效（`MultiEndedAllocator`
   grow=down/grow=up，`Components (FULL, MAMBA)`，`UnifiedTreeCore`）。
6. **重新评估**（§6.3）：L2/L3 修复后，One Pool vs Two Pools 的 "计算工作量"
   差距消失（都是 1,545）；剩余优势 = KDA COW 命中率（15/15 vs 9/15、11/15）
   + 零 L2→L1 回载流量（0 vs 37,888/4,096 token）。

## 1. 环境

| 项 | 值 |
|----|-----|
| sglang | 0.5.19（本地 `/sgl-workspace/sglang`，权威目标） |
| 代码基线 | `f8985f4` + `c142441` + `606c9cb`（One Pool bring-up） |
| 模型 | mock K3-23L（`$WORKDIR/k3-model`，`gen_k3_model.py` 生成） |
| Trace | `$WORKDIR/k3_p5{1,2,3}_*.jsonl`（`benchmark/collection/k3_onepool_twopools/gen_k3_exp_traces.py` 生成） |
| 模式 | Two Pools：`--enable-hierarchical-cache`，K3 自动 force `extra_buffer`；KDA=`HISIM_MAMBA_POOL_SIZE`，MLA=`--max-total-tokens` |

### 1.1 相对 docker 报告的语义差异（读数前必读）

- **共享预算**：docker fork 的 Two Pools 满足 `MLA = 403520 − 2544×KDA`
  （1 KDA slot = 14,653,440 B = 2,544 个 MLA entry，entry=5,760 B；总预算 2.324 GB）。
  本地 0.5.19 的 Two Pools 是**独立两个池**（MLA 与 KDA 互不挤占）。
  P5.2 复跑按 docker 的共享预算设置 MLA；P5.1 主扫固定 MLA=200,000，另跑
  TP-120/MLA=98,240 归因 cell。
- **L3 存储**：复跑默认开 `--hicache-storage-backend file --hicache-size 80`
  （mamba L3 走 KVCM `batch_set_mamba`/`batch_exists_mamba`）。

### 1.2 关键代码修复

| commit | 内容 |
|--------|------|
| `c142441` | `allocation.alloc_req_slots` 对 `HybridReqToTokenPool` 是顶层 from-import，只重绑 `memory_pool` 会让该模块持有旧类 → isinstance 判 False → **原生 mamba 预驱逐被跳过** → mamba 压力下 `alloc_req_slots runs out of memory`。改为遍历 `sys.modules` 重绑定（不 force-import，否则 scheduler 导入顺序变化会破坏 host pool 构造，曾复现 1.47 PB 分配）。 |
| `606c9cb` | `handle_unified_memory_pool` / `handle_page_major_kv_layout` 在 0.5.19 是 `arg_groups.kv_cache_hook` 的模块函数（0.5.18 是 ServerArgs 方法），旧补丁全部失效。按模块函数形式重打补丁 + 重绑定 `arg_groups.pipeline`；page_major 补丁只放行 backend 白名单断言（HiSim forward 全 mock，backend 不执行）。 |
| `606c9cb` | One Pool 手动 hicache 分支：0.5.19 的 `HybridCacheController` 要求 `mem_pool_host` 是 `HostPoolGroup`（读 `.anchor_entry.host_pool`/`.entries`），传普通 mock host pool 崩溃。 |

## 2. P5.1 KDA-heavy（C=1024）

Trace：96×1024 build + 6×1025 reuse（sessions 0,16,32,48,64,80）。

| 配置 | KDA | MLA | COW Hit | prefix_ratio¹ | 完成 | alloc_fail | docker COW |
|------|-----|-----|---------|---------------|------|-----------|------------|
| TP-40 | 40 | 200,000 | 2/6 | 0.333 | 102/102 | 0 | 2/6 ✓ |
| TP-60 | 60 | 200,000 | 3/6 | 0.500 | 102/102 | 0 | 3/6 ✓ |
| TP-80 | 80 | 200,000 | 4/6 | 0.667 | 102/102 | 0 | 4/6 ✓ |
| TP-100 | 100 | 200,000 | 6/6 | 1.000 | 102/102 | 0 | 6/6 ✓ |
| TP-120 | 120 | 200,000 | 6/6 | 1.000 | 102/102 | 0 | 5/6 ✗ |
| **TP-120（共享预算）** | 120 | **98,240** | **5/6** | **0.833** | 102/102 | 0 | 5/6 ✓ |

¹ prefix_ratio = `mla_prefix_hit_l1_tokens_agg / mla_prefix_hit_tokens_agg`。

**TP-120 归因**：docker 的 MLA=403520−2544×120=98,240 < 需求 98,304 → MLA 驱逐 →
级联释放 KDA → 1 个 COW miss（复跑 l1_hit=5120/6144，正好少 1,024 = 1 个 session）。
本地独立池模式下 MLA=200,000 无压力 → 6/6。**语义差异，非行为回归。**
所有 COW Hit 的 `checkpoint_pos=1024, replay_tokens=0`，与 docker 报告一致。

## 3. P5.2 MLA-heavy（C=8192, chunk=8192）

Trace：8×32768 build + 3×33024 reuse（sessions 0,3,7）；MLA=403520−2544×KDA。

| 配置 | KDA | MLA | COW Hit | l1_hit¹ | docker COW | 一致 |
|------|-----|-----|---------|---------|------------|------|
| TP-40 | 40 | 301,760 | 3/3 | 294,912 | 3/3 | ✓ |
| TP-60 | 60 | 250,880 | 3/3 | 294,912 | 3/3 | ✓ |
| TP-80 | 80 | 200,000 | **3/3** | 294,912 | 2/3 | ✗ |
| TP-100 | 100 | 149,120 | 2/3 | 262,144 | 2/3 | ✓ |

¹ `mla_prefix_hit_l1_tokens_agg`（含 build 分块自命中 8×24,576=196,608；
reuse 全命中 = +98,304 → 294,912；TP-80 差额 32,768 由 L2/L3 恢复补齐）。

**TP-80 归因（3/3 vs 2/3）**：MLA=200,000 < 262,144 → MLA 驱逐确实发生
（L2 host pool `L2_avail` 下降 ~196,608 tokens，即 ~6 个 session 被降级），
reuse session 的 checkpoint 被驱逐后**从 L2/L3 正确恢复**
（with-L3：`kda_l3_hits=1`；NOL3 对照：`kda_l3_hits=0` 但 L2 host mamba pool
恢复，仍 3/3；串行化对照 trace `p52_tp80_seq` 仍 3/3，排除到达次序影响）。
docker fork 的 KDA L3 写路径损坏（`MockHiCacheStorage.set` 引用未定义的
`self.storage` → `l3_writes=0`）且 L2→L1 load-back 任意取前 N 个 LRU，
驱逐后无法恢复 → 2/3。**复跑更优是 `f8985f4` 修复的预期结果，非回归。**

## 4. P5.3 动态混合负载（C=1024, chunk=32768）

Trace：A(96×1K build + 6×1K+1 reuse) → B(8×32K build + 3×32K+256 reuse) →
C(重访 A 0,32,80 + B 0,3,7)；阶段间不重启、不清 radix tree。

| Phase | 指标 | TP KDA=80 | TP KDA=40 | docker KDA=80 | docker KDA=40 |
|-------|------|-----------|-----------|---------------|----------------|
| A Reuse | COW Hit | 4/6 | 2/6 | 4/6 ✓ | 2/6 ✓ |
| B Reuse | COW Hit | 3/3 | 3/3 | 2/3（总数同） | 3/3 ✓ |
| C A-revisit | COW Hit | 0/3 | 3/3 | 0/3 ✓ | 3/3 ✓ |
| C B-revisit | COW Hit | 2/3 | 3/3 | 3/3（总数同） | 3/3 ✓ |
| **累计** | **总 COW** | **9/15** | **11/15** | 9/15 ✓ | §5.5=11/15（§6.2 写 14/15，自相矛盾） |
| 累计 | MLA 复用新 token² | **1,545** | **1,545** | ~39,433 | ~5,641 |
| 累计 | KDA replay tokens³ | 145,408 | 111,616 | ~34,816 | ~4,096 |
| 累计 | L2→L1 回载 token | 37,888 | 4,096 | — | — |

² `mla_prefill_new_tokens_agg` − build 基线 360,448 = 1,545 = 理想值
（6×1 + 3×256 + 3×1 + 3×256）。两个配置的 MLA prefix 均全命中
（`mla_prefix_hit_tokens_agg=205,824` = 理想），被驱逐部分由 L2/L3 恢复
（回载量见上行，`mla_prefix_hit_l1` 分别为 167,936 / 201,728）。

³ **该指标与 docker 不可直接对比**：本地 `kda_historical_replay_tokens`
聚合所有 `KDA-REPLAY` 日志事件（含 chunked prefill 的请求内自重放与
deferred 重试），One Pool 也有 107,520 ≠ 0；docker 报告只统计 COW miss
导致的重放。COW 事件粒度的 `replay_tokens` 才是可比口径（所有 COW Hit
均为 0，miss 的才重放）。

**与 docker 的核心差异**：docker 报告中 Two Pools 的复用新 token 为
39,433/5,641（大量 prefix miss）；复跑中 L2/L3 恢复链路修复后，Two Pools 的
MLA prefix 命中率达到 One Pool 理想值。KDA replay tokens 仍显著非零
（COW miss 的 checkpoint 需重放），这是 Two Pools 与 One Pool 的剩余差距来源。

## 5. 差异归因总表

| 现象 | 归因 | 结论 |
|------|------|------|
| P5.1 TP-120: 6/6 vs 5/6 | docker 共享预算 MLA=98,240<98,304 → 级联；本地独立池 | 语义差异；按共享预算复跑 → 5/6 完全一致 |
| P5.2 TP-80: 3/3 vs 2/3 | docker KDA L3 写损坏 + L2→L1 load-back 任意取前 N；修复后驱逐 checkpoint 可恢复 | 修复的预期改进 |
| P5.3 TP-40: 11/15 vs §6.2 的 14/15 | docker 报告 §6.2 与 §5.5 内部矛盾 | 复跑支持 §5.5（11/15） |
| P5.3 B/C-B 阶段划分 3/3+2/3 vs 2/3+3/3 | 同 batch 内驱逐顺序的时序差异 | 总数一致（9/15） |
| P5.3 MLA new tokens 1,545 vs 39,433/5,641 | L2/L3 恢复链路修复 | Two Pools 达到 One Pool 的 MLA 理想值 |
| P5.3 replay tokens 145K/112K vs 34,816/4,096 | 指标口径不同（本地聚合所有 KDA-REPLAY 日志事件，含 chunk 自重放；One Pool 也有 107,520） | 不可直接对比，见 §4 脚注 ³ |
| One Pool 无 "80→97 扩容/5 次 flush" 轨迹 | 0.5.19 统一池 mamba 预算由内存推导（6589 slots）而非初始 80；借还机制改为 MultiEndedAllocator grow=down/up | docker 轨迹是其 mock 语义；原生反向借用已在 P5.2 生效（§6.2） |

## 6. One Pool 本地可行性（0.5.19）

**结论：One Pool 已在 0.5.19 上跑通**（commit `606c9cb`）。修复的三个断点见 §1.2。
启动日志确认统一内存池真正生效：

```
[unified-memory-pool] UNIFIED MEMORY POOL ENABLED -- path=Mamba hybrid (MLA full side)
[unified-memory-pool]   full_layers=5, mamba_layers=18, kv_lora_rank=512, ...
[unified-memory-pool] MultiEndedAllocator('full')  ready: grow=down, ...
[unified-memory-pool] MultiEndedAllocator('mamba') ready: grow=up, ...
Components: (FULL, MAMBA)  Tree Core: UnifiedTreeCore
```

### 6.1 One Pool 三组实验结果（复跑 vs docker）

| 实验 | 指标 | 复跑 | docker 报告 | 一致 |
|------|------|------|------------|------|
| P5.1 KDA-heavy | COW Hit | **6/6** | 6/6 | ✓ |
| P5.1 | `checkpoint_pos`/`replay_tokens` | 1024 / 0 | 1024 / 0 | ✓ |
| P5.2 MLA-heavy | COW Hit | **3/3** | 3/3 | ✓ |
| P5.2 | reuse L1 命中 | 3×32768 全命中 | 全命中 | ✓ |
| P5.3 dynamic | **总 COW Hit** | **15/15** | 15/15 | ✓ |
| P5.3 | `kda_new_tokens` | **1,545**（理想） | 1,545 | ✓ |
| P5.3 | 复用 MLA new tokens | **1,545**（理想） | 1,545 | ✓ |
| P5.3 | 逐阶段 COW | A 6/6, B 3/3, C-A 3/3, C-B 3/3 | 同 | ✓ |

P5.3 的关键细节：One Pool 的 205,824 复用 token **全部命中 L1**
（`mla_prefix_hit_l1 = mla_prefix_hit = 205,824`，`l3_hits=0`，`load_calls=0`），
无需任何 L2/L3 恢复；Two Pools 则分别需要 37,888（KDA=80）/ 4,096（KDA=40）
token 的 L2→L1 回载。

### 6.2 与 docker One Pool 的语义差异

0.5.19 原生统一池的 `max_mamba_cache_size` 由内存预算推导
（CPU 模拟下 `total_bytes=772 GB → mamba 子池 max_slots=6589`），
**不是** docker fork 的 "初始 80 slots + 水位线 flush 扩容到 97"。
docker 报告的 "80→97 扩容"、"5 次 flush" 轨迹是其 mock 语义，在 0.5.19 上
不可直接对比（P5.1 负载 96 checkpoints ≪ 6589，无压力）。原生 0.5.19 的
动态借还是 `MultiEndedAllocator` 的 grow=down/grow=up 共享区域：
P5.2 中 MLA 需求 262,144 > 名义上限 200,000，而复用全部 L1 命中且
L2/L3 载入为 0（`l1=hit=294,912`，`load_calls=0`）→ 推断 full 子池向下
借用了共享区域，原生反向扩容路径生效（0.5.19 无 flush 观测计数器，
未直接仪表化，见遗留 P2）。

### 6.3 0.5.19 上 One Pool vs Two Pools 的重新评估

复跑（L2/L3 修复后）的核心发现：

| P5.3 指标 | One Pool | TP KDA=80 | TP KDA=40 |
|-----------|----------|-----------|-----------|
| 总 COW Hit | **15/15** | 9/15 | 11/15 |
| 复用新 token（计算量） | **1,545** | 1,545 | 1,545 |
| KDA COW miss 数 | **0** | 6 | 4 |
| L2→L1 回载 token | **0** | 37,888 | 4,096 |

docker 报告中 One Pool 的优势（工作量 1,545 vs 39,433/5,641）主要来自
Two Pools 的 L2/L3 回收链路损坏；链路修复后，**Two Pools 的 "计算工作量"
差距消失**（都是 1,545），剩余差距是：

1. **KDA COW miss**（9/15、11/15 vs 15/15）→ miss 的 checkpoint 需重放
   （KDA replay），这是固定分区在动态负载下的真实惩罚，与 docker 结论一致；
2. **L2→L1 回载流量**（37,888/4,096 token 的 D2H/H2D 传输），One Pool 为 0
   —— 统一池通过共享区域避免了跨池搬运。

即：**修复 L2/L3 后，One Pool 的剩余优势从 "MLA prefix 命中率" 转移到
"KDA COW 命中率 + 零回载流量"**。



## 附录 A：复跑命令

工具链已入库：`benchmark/collection/k3_onepool_twopools/`（用法详见其 `README.md`）。

```bash
# 1. 生成 trace（P5.1 / P5.2 / P5.2-串行归因变体 / P5.3）
python3 benchmark/collection/k3_onepool_twopools/gen_k3_exp_traces.py

# 2. 单 cell：run_exp.sh <mode> <KDA> <MLA> <C> <chunk> <trace> <n> <label> [port]
#    mode = twopools | onepool；NOL3=1 关 L3 storage backend（归因用）
bash benchmark/collection/k3_onepool_twopools/run_exp.sh twopools 80 200000 8192 8192 \
  /tmp/opencode/k3_p52_mla_heavy.jsonl 11 p52_tp80

# 3. 全量 Two Pools（P5.1 5+1 格 / P5.2 4 格 / P5.3 2 格）+ One Pool 3 格
MODE=all      bash benchmark/collection/k3_onepool_twopools/run_all_exp.sh
MODE=onepool  bash benchmark/collection/k3_onepool_twopools/run_all_exp.sh

# 4. 汇总
python3 benchmark/collection/k3_onepool_twopools/analyze_exp.py
```

前置：mock K3 模型与 sim config 在 `$WORKDIR`（默认 `/tmp/opencode`，可用
`MODEL_DIR`/`SIM_CONF`/`WORKDIR` 覆盖）；每格重启 server（CPU 上 ~2.5–3 分钟）。

## 附录 B：归因对照 cell

| cell | 配置 | 目的 |
|------|------|------|
| `p51_tp120_b98k` | TP, KDA=120, **MLA=98,240** | 复现 docker 共享预算语义（98,240<98,304 → 级联 → 5/6） |
| `p52_tp80_nol3` | TP-80, `NOL3=1` | 排除 L3 storage backend（仍 3/3，恢复来自 L2） |
| `p52_tp80_seq` | TP-80, 串行 trace | 排除到达次序/并发度对 LRU victim 的影响（仍 3/3） |

## 附录 C：数据文件

- 每 cell 指标快照：`$WORKDIR/exp_results/<label>.json`（17 份，含归因 cell）
- 原始输出：`/tmp/hisim/<label>/metrics.json`（canonical）+ `metrics_phase1.json`
- server/bench 日志：`$WORKDIR/<label>_server.log` / `<label>_bench.log`
