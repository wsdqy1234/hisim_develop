# HiSim 三层缓存命中率理论分析工具

`hisim_cache_analyzer.py` 用于在给定 L1/L2/L3 容量配置下，理论计算 HiSim 仿真中各层缓存的命中率。

## 核心概念

### 三层缓存层次

| 层级 | 介质 | 匹配机制 | 驱逐策略 |
|---|---|---|---|
| L1 (GPU HBM) | Radix Cache | 累积 hash 前缀匹配（连续） | LRU，驱逐时写入 L2+L3 |
| L2 (Host DRAM) | Host Memory | 前缀匹配（L1 miss 后连续） | LRU，驱逐时数据只在 L3 |
| L3 (SSD/kvcm) | SSD/远端 | Hash 查找（非连续） | 持久存储（默认无限） |

### 匹配逻辑

每个请求的 tokens 按 block_size 切分为 block，每个 block 的 hash 包含从头到该 block 的所有 token（累积 hash），确保前缀正确性。

```
请求 tokens: [t1..t16] [t17..t32] [t33..t48]  (block_size=16)

Block 0: hash = rolling_hash(t1..t16)
Block 1: hash = rolling_hash(Block 0 hash, t17..t32)  ← 包含 Block 0
Block 2: hash = rolling_hash(Block 1 hash, t33..t48)  ← 包含 Block 0+1
```

对每个请求：
1. **L1 命中**：从 Block 0 开始连续匹配 L1 中的 block，直到第一个 miss
2. **L2 命中**：从 L1 miss 处继续连续匹配 L2 中的 block
3. **L3 命中**：从 L2 miss 处逐个 hash 查找 L3（不需连续）
4. **New**：未在任何层命中，需要 prefill 计算

### 请求完成时的 backup

当请求完成时（模拟 `cache_finished_req`），该请求的所有 block 被写入 L2 和 L3。这确保后续请求能在 L2/L3 中找到之前请求的数据。

## 安装

```bash
# 无额外依赖，仅需 Python 3.8+
```

## 使用方法

```bash
# 默认配置（Qwen3-32B, L1=200K tokens, L2=80GB, L3=无限）
python hisim_cache_analyzer.py -i trace.jsonl

# 自定义容量
python hisim_cache_analyzer.py -i trace.jsonl --l1-tokens 200000 --l2-gb 80

# 无限 L1（理论上限命中率）
python hisim_cache_analyzer.py -i trace.jsonl --l1-tokens 0

# 包含 output_ids
python hisim_cache_analyzer.py -i trace.jsonl --use-output-ids

# 输出逐请求结果
python hisim_cache_analyzer.py -i trace.jsonl -o results.jsonl
```

## 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `-i, --input` | 输入 JSONL trace 文件 | — |
| `-o, --output` | 输出逐请求结果 JSONL | None |
| `--block-size` | Block 大小（tokens） | 16（kvcm block size） |
| `--kv-bytes-per-token` | 每 token KV cache 字节数 | 262144（Qwen3-32B FP16: 8×128×64×2×2） |
| `--l1-tokens` | L1 (HBM) 容量（tokens），0=无限 | 200000 |
| `--l2-gb` | L2 (DRAM) 容量（GB），0=无限 | 80 |
| `--l3-gb` | L3 (SSD) 容量（GB），0=无限 | 0 |
| `--use-output-ids` | 包含 output_ids 在 token 序列中 | False（仅 input_ids） |
| `--max-requests` | 最大分析请求数 | 0（全部） |

## 输出示例

```
================================================================================
3-Tier Cache Hit Rate Summary
================================================================================
  Total requests: 500
  Total input tokens: 49,370,238
  Total blocks: 3,085,410

  L1 (Radix Cache):
    Hit blocks: 2,574,570  Hit tokens: 41,193,120
    Hit rate:   83.44%  (83.44% of input)
    Evictions: 498,340
    Final cached: 12,500 blocks

  L2 (Host DRAM):
    Hit blocks: 268,875  Hit tokens: 4,302,000
    Hit rate:   8.71%  (8.71% of input)
    Evictions: 222,892
    Final cached: 19,073 blocks

  L3 (SSD/kvcm):
    Hit blocks: 88,789  Hit tokens: 1,420,624
    Hit rate:   2.88%  (2.88% of input)
    Final stored: 153,176 blocks

  New (need computation):
    New blocks: 153,176  New tokens: 2,450,816

  Hierarchy check: L1=12500 < L2=19073 < L3=153176
================================================================================
```

## 理论 vs 实际对比（500 请求 trace）

### 配置：L1=200K tokens, L2=80GB (305K tokens), L3=无限

| 层级 | 理论命中率 | 实际命中率 | 差异原因 |
|---|---|---|---|
| L1 | 83.44% | 87.3% | 理论用 block_size=16（粗粒度），实际 page_size=1（细粒度，更多匹配机会） |
| L2 | 8.71% | ~0% (1 次 load_back) | 理论用 hash 查找 L2，实际需 radix tree 路径穿过驱逐节点（更严格） |
| L3 | 2.88% | ~2.39% (73,730 blocks) | 接近，都是 hash 查找 |
| New | 4.97% | ~10.3% | 实际有更多 L1 miss 未被 L2/L3 命中 |

### L2 理论 vs 实际差异分析

理论分析中 L2 命中 8.71%，但实际仅 1 次。差异原因：

- **理论模型**：L2 命中 = L1 miss 后的 block 在 L2 的 LRU 中（hash 查找，连续匹配）
- **HiSim 实际**：L2 命中需要 radix tree 路径穿过被驱逐节点（`host_hit_length > 0`）。公共前缀常驻 L1（LRU 高优先级），私有前缀互不相同，路径不穿过驱逐节点 → `host_hit_length = 0`

理论模型是 L2 命中率的上界估计。实际 L2 命中受 radix tree 路径匹配机制限制，远低于理论值。

### 旧配置对比（修正 mock size_per_token 前）

| 层级 | 理论 (L2=15.6M mock) | 实际 (L2=15.6M mock) |
|---|---|---|
| L1 | 87.53% | 94.4% |
| L2 | 7.50% | 69 次 load_back |
| L3 | 0.00% | 0 |

L2=15.6M tokens（mock 放大 1024 倍）→ L2 永不满 → L2 永不驱逐 → L3 从未被查询 → L3=0。理论与实际一致。

## 输入文件格式

HiSim `hisim-collection` 格式（JSONL）：

```json
{"rid": "21e5xx", "timestamp": 732.31, "input_length": 1024, "output_length": 1024, "input_ids": [925, 3911, ...], "output_ids": [244, 129, ...]}
```

| 字段 | 说明 |
|---|---|
| `input_ids` | 输入 token ID 列表（用于 block hash 计算） |
| `output_ids` | 输出 token ID 列表（可选，`--use-output-ids` 时包含） |
| `input_length` | 输入长度（用于统计） |
| `timestamp` | 请求时间戳 |
