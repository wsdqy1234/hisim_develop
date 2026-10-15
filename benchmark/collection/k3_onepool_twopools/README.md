# K3 One Pool vs Two Pools — 0.5.19 复跑工具链

复跑 `docs/k3_onepool_vs_twopools_experiment_report.md`（docker fork
`0.5.18.dev922`）的关键实验，结果与差异归因见
`docs/k3_onepool_vs_twopools_0519_rerun.md`。

## 前置

- 本地 sglang **0.5.19**（权威目标），代码含 `c142441`（isinstance 重绑）与
  `606c9cb`（One Pool bring-up）。
- mock K3 模型目录与 sim config（`$WORKDIR/k3-model`、`$WORKDIR/config_k3_0519.json`），
  可用 `MODEL_DIR` / `SIM_CONF` 覆盖。
- `WORKDIR`（默认 `/tmp/opencode`）：trace / 日志 / 结果快照的落盘目录。

## 用法

```bash
# 1. 生成 trace（P5.1 / P5.2 / P5.2-串行归因变体 / P5.3）
python3 benchmark/collection/k3_onepool_twopools/gen_k3_exp_traces.py

# 2a. 全量 Two Pools 扫描（P5.1 5+1 格 / P5.2 4 格 / P5.3 2 格；每格重启 server，~4 分钟/格）
MODE=all bash benchmark/collection/k3_onepool_twopools/run_all_exp.sh
# 2b. 或分模式：MODE=p51|p52|p53|onepool
# 2c. 或单 cell：
#    run_exp.sh <mode> <KDA> <MLA> <C> <chunk> <trace> <n> <label> [port]
bash benchmark/collection/k3_onepool_twopools/run_exp.sh twopools 80 200000 8192 8192 \
  /tmp/opencode/k3_p52_mla_heavy.jsonl 11 p52_tp80

# 3. 汇总
python3 benchmark/collection/k3_onepool_twopools/analyze_exp.py
```

## 关键语义

- **Two Pools**：KDA=`HISIM_MAMBA_POOL_SIZE`，MLA=`--max-total-tokens`，两池独立。
  docker fork 的 Two Pools 是共享预算 `MLA = 403520 − 2544×KDA`（1 KDA slot
  = 14,653,440 B = 2,544 MLA entry）；P5.2 复跑按此设置 MLA，P5.1 的 TP-120
  归因 cell（`p51_tp120_b98k`）单独按共享预算设 MLA=98,240。
- **One Pool**：`--enable-unified-memory`（0.5.19 原生统一池，mamba 预算由内存
  推导，与 docker 的 "初始 80 + 水位线扩容" mock 语义不同）。
- **NOL3=1**：关 L3 storage backend 的归因模式（显式 `--hicache-ratio 2`，
  否则 `hicache_ratio=None` 会让 mock host pool 定容崩溃）。
- 服务器启动 ~2.5–3 分钟（CPU dummy load），bench 完成标志为
  `All prompts done|RuntimeError|requests failed|Throughput`。

## 已知坑

- **pkill 自匹配**：`run_exp.sh` 开头的 `pkill -f '...launch[_]serv'` 会匹配任何
  命令行含 `launch_server` 文本的 shell（例如 py_compile 的路径参数）——不要从
  这种命令行里调用本脚本。
- `kda_historical_replay_tokens` 聚合所有 `KDA-REPLAY` 日志事件（含 chunked
  prefill 请求内自重放），与 docker 报告的 "COW miss 重放" 口径不同，不可直接
  对比；COW 事件粒度的 `replay_tokens` 才可比。
