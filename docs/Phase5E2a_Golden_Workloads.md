# Phase 5E.2a Golden Workloads（冻结工作负载与配置协议）

> 日期：2026-11-05 · 26 Case 全部带 Case ID + config hash（sha256 前 16 位）+ 完整产物归档 `docs/p5e2a_results/gold_cases/<CASE_ID>/`
> 每目录：`manifest.json`（参数+hash）· `win_config.json` · `parse.json`（请求级）· `ledger.json`（批/请求/I-O 时序账本）· `ledger_batches.csv` · `ledger_requests.csv` · `io_trace.jsonl`（逐 I-O arrival/media/ic/queue）· `win_pool_summary.json` · `hbf_win_effective_config.json` · `request.jsonl` · 原始 log

## 协议规则（全部 Case 强制）

1. **Prefix Capacity Rule**：`max_total ≥ n_reuse×prefix+margin`（合批余量）；host ≥ (live+reinsert)×prefix
2. **`HISIM_SHARED_DRAM=1` 全场景**（含 A 基线——修正 p3e 驱动缺陷）
3. **确定性复用顺序**：reuse i 时间戳 = base+0.001×i（消除 heap salt 平局随机批序；p95 请求身份可复现）
4. **唯一 Case ID + config hash**：sha256(effective env + win cfg + 启动参数 + 工作负载参数)，杜绝"n2"类歧义
5. 完整恢复组守恒校验（每请求 L1+L2+RC=actual 全过）；Agent/Churn 组保留真实容量压力并记录部分命中原因
6. 接口/媒体参数全部标注 Hypothesis（UB 语义未确认；0.8 效率为人工设定）

## 冻结工作负载

| Workload | 几何 | 用途 | 验收要点 |
|---|---|---|---|
| `n2` | 340tk，N=2 复用，MAXT 1024 | 回归 + 短 I/O + 同介质串行域 | 命中 2×339；合批 bs=2 |
| `n4` | 340tk，N=4，MAXT 1700 | 多请求并发 | 命中差异记录（V1a 放置效应） |
| `agent` | 8 builds + 8 间隔复用 | 真实容量压力/部分命中 | 部分命中原因=host churn（8>6 对象） |
| `p4k` | 4096tk×3 builds+2 reuses，MAXT 8300 | 完整 Host Restore | 命中 2×4095 |
| `p16k` | 16384tk×3+2，MAXT 33100(合批域)/32768(串行域) | 完整 Host Restore + 接口敏感性 | 命中 2×16383 |

## Golden 结果总表

### 配对基线（A=纯 DRAM 等总 Host；B70=HBF/DRAM 混合；HBFO=纯 HBF）

| Case | config hash | 命中 | reuse TTFT | p95 | 对 A |
|---|---|---|---|---|---|
| n2_A | f830a4b4 | 678 | 7.4/7.4 | 7.4 | — |
| n2_str | 75546318 | 678 | 4.5/4.5 | 4.5 | **−39.2%** |
| n4_A | 24fa5d4f | 1017 | 10.6/46.1/46.1 | 46.1 | — |
| n4_str | 3f7ad307 | 678 | …/34.7 | 34.7 | −24.7% |
| ag_A | e2574eb8 | 1024 | 26.2 p95 | 26.2 | — |
| ag_str | de0e9abe | 1024 | 23.9 p95 | 23.9 | −8.8% |
| p4k_A_co | 0b892614 | 8190 | 59.7/123.5 | 123.5 | — |
| p4k_str_co | 4f479cf0 | 8190 | 59.7/123.5 | 123.5 | **0%** |
| p16k_A_co | 257a84a6 | 32766 | 235.9/419.6 | 419.6 | — |
| p16k_str_co | 39058700 | 32766 | 235.9/419.6 | 419.6 | **0%** |
| **p16k_hbfo_none** | 33306be3 | 32766 | 235.9/282.5 | 282.5 | **−32.7%** |

### 接口假设矩阵（16K，Hypothesis）

| Case | hash | p95 | Δ | Case | hash | p95 |
|---|---|---|---|---|---|---|
| hbfo_none | 33306be3 | 282.5 | — | hbfo_s896 | 90e6e63c | 283.4 |
| hbfo_s112 | 3351ef0f | 298.1 | +15.6 | hbfo_s1280 | efc44c25 | 282.8 |
| hbfo_s1600 | a8d6701a | 282.5 | 0 | b70_s112 | (co_s112 同构) | 419.6 |
| b70_s1280 | — | 419.6 | 0 | str_ser_s112 | e6c1976e | 465.7（串行域传导）|

### 80% 媒体/写效率假设（Hypothesis）

| Case | hash | p95 | 说明 |
|---|---|---|---|
| b70_m160 / b70_m160_s112 | (ab3853b0 等) | 419.6 | seq 160GB/s=0.8 假设，B70 域被 L3 屏障吸收 |
| b70_rand80 / rand80_s112 | 0af62a15 / ec2f37fa | 419.6 | rand 80GB/s=0.8 假设 |
| hbfo_w8 | 7ed87d118 | 282.6 | 写 8GB/s=0.8 假设，复用 TTFT 无感 |

## 关键路径与瓶颈归因（每 Case ledger.json 可复查）

- **16K 复用波**：批1=HBF 请求（load 1.18/16.8ms ∥ media/接口 max + L3 写屏障 231ms 门控）→ TTFT 235.9；批2=DRAM 请求（批1 时长 + DRAM 138.3ms）→ p95
- **n2**：合批 load=同介质串行 makespan（A 5743μs）vs max(DRAM,HBF)（str 2875μs）
- **媒体/接口/写效率变体**在 B70 域 p95 不动（屏障吸收）、HBFO 域 1:1 进自身加载（T3 算术）

## 部分命中记录（Agent/Churn 类，不要求全命中）

- agent：128tk/req（8 对象 > host 6 对象容量 → L2 LRU churn → 部分 L3 部分命中）——守恒全过，原因记录于 ledger
- 完整恢复组（n2/n4/p4k/p16k）：Match=Backup=Restore=Consume 逐 token 相等（parse.json + io_trace 可对账）

## 复现

```bash
source /tmp/gold_driver.sh && run_gold <CASE_ID> <PORT> wl=p16k scen=B70 iface=s112
# 产物 → /tmp/gold_cases/<CASE_ID>/（含 manifest/ledger/trace）
```
