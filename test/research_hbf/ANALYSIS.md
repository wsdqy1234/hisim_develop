# Host Prefix 需求与 HBF 效率研究入口

运行环境：现有 `dqy-hisim-qwen3` 容器，仓库 `/home/d00954154/HiSim_Qwen3`。所有 Python 命令建议使用 `-B`、`PYTHONDONTWRITEBYTECODE=1`，测试增加 `-p no:cacheprovider`。程序不会自动安装依赖。

`run_host_case.py` 负责安全运行、精确业务冻结和严格验收；详见 `RUNNER.md`。实验输出必须使用新的 case ID/目录，已有案例不会被覆盖。以下从服务器宿主运行；图表使用宿主已有 matplotlib，仿真使用已有容器。

```bash
docker exec -e PYTHONPATH=/home/d00954154/HiSim_Qwen3/src \
  -e PYTHONDONTWRITEBYTECODE=1 dqy-hisim-qwen3 python3 -B \
  /home/d00954154/HiSim_Qwen3/test/research_hbf/run_host_case.py \
  --case-id example_new --port 30300 \
  --output-root /home/d00954154/hbf_research_results/new_cases \
  --scenario mixed --prefix-tokens 340 --concurrency 2 --reuse 2 \
  --max-total-tokens 1024 --host-capacity-tokens 2312 --stripe-block-tokens 128
```

**工具职责**

| 文件 | 输入/输出与含义 |
|---|---|
| `analyze_host.py CASE_DIR` | 完整案例→`demand_analysis/`：逻辑 byte-LRU 参考曲线、复用距离、Host 实际 I/O、到达/完成窗口及在途数 |
| `supply_scan.py --output NEW_DIR` | 176 行闭环单 WIN I/O 包络扫描；固定整池分母；非硬件标定 |
| `replay_host_trace.py CASE_DIR --output NEW_DIR` | 固定 HBF slots/到达/顺序，改变共享接口；必须先检查原配置 no-op 误差 |
| `placement_replay.py --case CASE_DIR --output NEW_DIR --repo REPO` | 固定 Trace 改条带大小，输出单次恢复效率，不能替代饱和效率 |
| `summarize_matrix.py --root CASE_ROOT --output SUMMARY --repo REPO` | 仅汇总完整验收案例，输出实际容量曲线及条件容量/带宽表 |
| `plot_results.py --supply SUPPLY --cases CASE_ROOT --summary SUMMARY --output FIGURES` | 导出六张 PNG；需要 matplotlib/numpy |

```bash
docker exec -e PYTHONPATH=/home/d00954154/HiSim_Qwen3/src \
  -e PYTHONDONTWRITEBYTECODE=1 dqy-hisim-qwen3 python3 -B \
  /home/d00954154/HiSim_Qwen3/test/research_hbf/analyze_host.py \
  /home/d00954154/hbf_research_results/new_cases/example_new

python3 -B /home/d00954154/HiSim_Qwen3/test/research_hbf/summarize_matrix.py \
  --root /home/d00954154/hbf_research_results/new_cases \
  --output /home/d00954154/hbf_research_results/new_summary \
  --repo /home/d00954154/HiSim_Qwen3
```

**口径**

- 逻辑 Prefix 引用含 miss，suffix/output 分列；离线 LRU 是独立参考策略，不能当作 SGLang Host 真实策略。
- 到达字节率使用 requested bytes，完成字节率使用 served bytes；二者不是同一个指标。在途数按半开区间 `[arrival,completion)` 计数，父对象和 WIN 子 I/O 分开。
- `metadata.source=l3_restore` 标记 SSD 恢复后新增的 HBF program；既有 `backup` 字段表示写方向，不保证都来自 HBM。L3 consumed 可能与 Host Restore 重叠，不能直接相加。
- `mock_lbk` 是零权威耗时的占位诊断，真实流量/资源对账筛选 `path=real`。
- 固定整池效率分母为 1600/800/80GB/s。合成扫描每次 I/O 只针对一个 WIN；端到端一个条带对象可覆盖多 WIN。
- 失败/运行中案例默认拒绝分析；`--allow-incomplete`仅用于调试，不能纳入成果。
- 本轮固定回放未包含 Free 事件与调度反馈，因此只用于映射/时序隔离；实际容量结论来自配对 E2E。
- Trace v2 缺失细分是 null，旧 `gold_ledger.py` 等直接对 `media_s`乘法的历史脚本需显式适配，不能将缺失当作零。

最终首轮数据为 `20261106_host_v4`、`20261106_supply_v4`、`20261106_summary_v4`；早期 v1/v2/v3 目录只保留开发证据。
