# HiSim 三层缓存命中率理论分析工具 v3（HiSim 对齐模型）

`hisim_cache_analyzer_v3.py` 在 v2（sglang radix tree 模型）基础上叠加 HiSim 仿真特有的 KV 复用逻辑，是三个 analyzer 中与 hisim 仿真实际行为对齐度最高的版本。

## v3 相对 v2 新增的 HiSim 机制

- **write_through backup**：首次命中即备份并加锁，直到 ACK 才解锁
- **load_back**：被逐出链从 L2 重新物化回 L1
- **连续前缀不变量**：父节点必须先于子节点备份
- **16384-token 分块插入**（对齐 HiSim `chunked_prefill_size`）
- **Phase 3 重物化**（churn 循环：evict → match → insert re-materialize），浅优先遍历
- **output 叶子流**：请求完成后插入 output tokens
- **L3 仅经 prefetch_from_storage 查询**（非直接 hash 查找）；L3 hash 延迟到 ACK 时写入（write_backup_storage）
- **prefetch buffer**：为剩余完整 input 分配 L2（非树节点）

## 使用方法

```bash
# hisim-collection 格式 jsonl
python hisim_cache_analyzer_v3.py -i trace.jsonl

# Weka Agentic Trace（真实 Claude Code 会话数据集）
python hisim_cache_analyzer_v3.py \
    --weka-traces /path/to/traces.jsonl \
    --num-traces 4 --max-concurrent-traces 2 --trace-start-interval 0.5 \
    --max-rounds-per-trace 10 [--max-round-tokens 100000]
```

## 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `-i, --input` | 输入 JSONL（hisim-collection 格式）；给了 `--weka-traces` 可省略 | — |
| `--l1-tokens` | L1 (HBM) 容量（tokens），0=无限 | 200000 |
| `--l2-gb` | L2 (DRAM) 容量（GB），0=无限 | 80 |
| `--kv-bytes-per-token` | 每 token KV cache 字节数 | 262144（Qwen3-32B FP16） |
| `--page-size` | 页大小（tokens，影响 L3 hash 粒度） | 1 |
| `--max-requests` | 最大分析请求数（0=全部） | 0 |
| `--weka-traces` | weka traces.jsonl 路径，可多个（`;` 分隔） | — |
| `--num-traces` | weka：总共读取的 trace 数 | -1（全部） |
| `--max-concurrent-traces` | weka：同时并发的 trace 数（槽位制） | 1 |
| `--trace-start-interval` | weka：并发 trace 首轮间隔（秒） | 0.5 |
| `--max-rounds-per-trace` | weka：每条 trace 最多分析多少轮 | -1（无限） |
| `--max-round-tokens` | weka：单轮长度上限（in+out），超限前截断该 trace | -1（无限） |

## Weka 模式语义要点

- **前缀复用**：每 trace 分配全局唯一 block base，块 h 展开为 tokens `[(base+h)*64, +64)`；共享 hash 前缀 ⟺ 共享 token 前缀，缓存命中结构与真实会话一致。
- **output 推导**：weka 模式下 `analyze()` 使用从下一轮 input 推导的真实 `output_ids`（hisim-collection 老路径仍用全 1 的 mock output），多轮前缀链不断裂。
- **调度**：静态时间轴——并发 trace 首轮按 `trace_start_interval` 错峰、trace 内轮次保持录制时间间隔，请求按该时间轴排序处理。
- **subagent**：当前仅分析主流请求（subagent 组暂不展开）。
- 无仿真依赖：loader 模块（`src/hisim/dataset/weka_trace.py`）纯标准库，自动通过 importlib 加载，无需安装 transformers/sglang。

## 实测结果（cc-traces-weka-062126-256k）

4 traces / 并发 2 / 间隔 0.5s / 每 trace 10 轮（40 rounds / 3.5M input tokens）：

| Analyzer | L1 | L2 | L3 | New |
|---|---|---|---|---|
| v1（block 级，`--block-size 64`） | 82.29% | 1.84% | 1.84% | — |
| v2（radix tree） | 84.88% | 0.52% | 1.87% | 12.73% |
| v3（v2 + HiSim 逻辑） | 83.96% | 0.52% | 1.87% | 13.64% |

L1 82~85% 为 agentic 多轮前缀复用的预期水平（每轮 ~13-14% 为新增上下文）；v2/v3 的 L3 完全一致（逐出后经 prefetch 恢复的整轮），v3 的 prefetch buffer 逻辑带来 ~1pp 的 L1 差异。
