# HiSim 三层缓存命中率理论分析工具 v2（Radix Tree）

`hisim_cache_analyzer_v2.py` 是基于 radix tree 的三层缓存命中率理论分析工具，替代 v1 的 block 级模型。

## 为什么需要 v2

v1（block 级模型）用固定大小的 block（默认 16 tokens）模拟缓存，每个 block 在 LRU 中独立管理。这导致 **L2 命中率被高估 88 倍**（理论 7.97% vs 实际 0.09%）。

根因：radix tree 的**内部节点保护机制**——有子节点的内部节点不可从 L1 驱逐，只有叶子节点可驱逐。共享前缀的节点因为后续请求在其下创建子节点，变成内部节点，永久留在 L1（L1 命中），不会进入 L2。block 级模型里每个 block 独立可驱逐，共享前缀最终被挤出 L1 进入 L2，导致 L2 命中被高估。

v2 用真正的 radix tree 结构模拟，正确建模：
- 变长节点（非固定 block）
- 节点分裂（partial match 时 split）
- 内部节点保护（有非驱逐子节点的节点不可从 L1 驱逐）
- 叶子优先驱逐（bottom-up LRU，从最深叶子开始）
- 分块插入（chunk_size=16384，对齐 HiSim 的 chunked prefill）

## 模型架构

```
请求 tokens → match_prefix（遍历 radix tree）
  ├─ L1 命中：非驱逐节点 → l1_tokens
  ├─ L2 命中：驱逐节点（在树中，backuped）→ host_hit_tokens
  └─ 剩余 tokens → L3 hash 查找 → l3_tokens

存储：剩余 tokens 按 chunk_size 分块插入树（创建变长节点链）
备份：从最深节点向上遍历，对未 backuped 的非驱逐节点执行 write_backup
  ├─ L2 分配（evictable_host_leaves LRU 驱逐）
  └─ L3 写入（hash 加入持久集）

L1 驱逐：收集 evictable_leaves（叶子，lock_ref=0），按 last_access LRU 驱逐
  ├─ backuped=True → value=None（留树，进 L2）
  └─ backuped=False → 从树删除（_evict_regular）

L2 驱逐：收集 evictable_host_leaves（已从 L1 驱逐，backuped，子节点全驱逐）
  → 从树删除，host_value=None（数据只在 L3）
```

## 使用方法

```bash
# 默认配置（Qwen3-32B, L1=200K, L2=80GB, L3=无限）
python hisim_cache_analyzer_v2.py -i trace.jsonl

# 自定义容量
python hisim_cache_analyzer_v2.py -i trace.jsonl --l1-tokens 200000 --l2-gb 80

# 限制请求数（快速测试）
python hisim_cache_analyzer_v2.py -i trace.jsonl --max-requests 30
```

### Weka Agentic Trace 输入（`--weka-traces`）

v1/v2/v3 三个 analyzer 均可直接分析 SemiAnalysis cc-traces-weka 真实 Claude Code 会话数据集（无需安装仿真依赖，loader 模块纯标准库，自动通过 importlib 加载）：

```bash
python hisim_cache_analyzer_v2.py \
    --weka-traces /path/to/traces.jsonl \
    --num-traces 4 --max-concurrent-traces 2 --trace-start-interval 0.5 \
    --max-rounds-per-trace 10 [--max-round-tokens 100000]
```

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--weka-traces` | 一个或多个 traces.jsonl 路径（`;` 分隔）；给定后 `-i` 可省略 | — |
| `--num-traces` | 总共读取的 trace 数 | -1（全部） |
| `--max-concurrent-traces` | 同时并发的 trace 数（槽位制） | 1 |
| `--trace-start-interval` | 并发 trace 首轮之间的间隔（秒） | 0.5 |
| `--max-rounds-per-trace` | 每条 trace 最多分析多少轮 | -1（无限） |
| `--max-round-tokens` | 单轮长度上限（in+out tokens），首个超限轮前截断该 trace | -1（无限） |

weka 模式语义要点：
- **前缀复用**：每 trace 分配全局唯一 block base，块 h 展开为 tokens `[(base+h)*64, +64)`；共享 hash 前缀 ⟺ 共享 token 前缀，radix tree 命中结构与真实会话一致。
- **output 推导**：weka 模式下 `analyze()` 使用从下一轮 input 推导的真实 `output_ids`（而非虚拟 token），多轮前缀链不断裂；否则 L1 复用率会被系统性低估。
- **调度**：静态时间轴——并发 trace 首轮按 `trace_start_interval` 错峰，trace 内轮次保持录制时间间隔；请求按该时间轴排序处理。
- **subagent**：当前仅分析主流请求（subagent 组暂不展开）。
- 实测（4 traces / 并发 2 / 10 轮，40 rounds / 3.5M tokens）：v2 L1=84.88% / L2=0.52% / L3=1.87%；v3 L1=83.96% / L2=0.52% / L3=1.87%。

## 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `-i, --input` | 输入 JSONL trace 文件（hisim-collection 格式） | — |
| `--l1-tokens` | L1 (HBM) 容量（tokens） | 200000 |
| `--l2-gb` | L2 (DRAM) 容量（GB） | 80 |
| `--kv-bytes-per-token` | 每 token KV cache 字节数 | 262144（Qwen3-32B FP16: 8×128×64×2×2） |
| `--page-size` | 页大小（tokens，影响 L3 hash 粒度） | 1 |
| `--max-requests` | 最大分析请求数（0=全部） | 0 |

## 输出示例

```
Radix Tree Cache v2: L1=200000 L2=305175 page=1

  #133: inp=129739 L1=120157 L2=8294 L3=0 New=1288
  #137: inp=130569 L1=119144 L2=10913 L3=0 New=512
  ...
  500 reqs 74s

============================================================
Total input: 49,370,238
L1: 43,846,377 (88.81%)
L2: 0 (0.00%)  [0 reqs]
L3: 3,073,071 (6.22%)
New:2,450,790 (4.96%)

HiSim:  L1=87.34% L2=0.09% L3=2.39%
Block:  L1=84.39% L2=7.97% L3=2.67%
```

## 与 v1（block 级模型）对比

| 指标 | HiSim 实际 | v1 (block) | v2 (radix tree) |
|---|---|---|---|
| L1 | 87.34% | 84.39% | **88.81%** |
| L2 | 0.09% | 7.97% (88× 偏高) | **0.00%** |
| L3 | 2.39% | 2.67% | 6.22% |

- **L2 对齐**：v2 的 L2=0.00% 与 HiSim 的 0.09% 一致，根因是 radix tree 内部节点保护
- **L1 接近**：v2 的 L1=88.81% 与 HiSim 的 87.34% 在 1.5pp 以内
- **L3 偏高**：v2 的 L3=6.22% 高于 HiSim 的 2.39%，可能来自 hash chain 或 backup 范围的细节差异

## 输入文件格式

HiSim `hisim-collection` 格式（JSONL）：

```json
{"rid": "21e5xx", "timestamp": 732.31, "input_length": 1024, "output_length": 1024, "input_ids": [925, 3911, ...], "output_ids": [244, 129, ...]}
```

请求按 `timestamp` 排序后处理（对齐 HiSim OFFLINE 模式的虚拟时钟调度）。

## 关键实现细节

### 节点分裂（_split）

当请求部分匹配一个已有节点时，节点被分裂为：
- 新父节点（匹配部分）：继承 `host_value`（L2 数据）、`hash_vals`（L3 哈希）
- 原节点（未匹配部分）：保留剩余 key 和 hash

分裂后，新父节点可能有子节点（来自后续请求）→ 变成内部节点 → 不可从 L1 驱逐。

### 分块插入（chunk_size=16384）

剩余 tokens 按 16384 分块插入，创建变长节点链（parent → child → grandchild）。这模拟 HiSim 的 `cache_unfinished_req`，使树结构更接近实际——更多内部节点，更多保护。

### L1 驱逐（叶子优先）

只驱逐 `is_evictable_leaf()` 的节点：
- `lock_ref == 0`
- 非驱逐状态（value != None）
- 所有子节点都已驱逐（或无子节点）

这模拟 sglang HiRadixCache 的 `evict()` 方法，从叶子开始 bottom-up 驱逐。

### L2 驱逐（evictable_host_leaves）

只驱逐 `is_evictable_host_leaf()` 的节点：
- 已从 L1 驱逐（evicted=True）
- `backuped=True`（有 L2 数据）
- 所有子节点已驱逐

驱逐后从树中删除，L2 数据释放。数据只在 L3（hash 持久存储）。
