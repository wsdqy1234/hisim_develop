# HiSparse 稀疏 KV 仿真（hisim）

日期：2026-11-03
分支：`develop_lfz_sparse_kv`（基于 `develop_lfz_0.5.19`）
依赖分析：[`docs/sparse_kv_sglang_vllm.md`](./sparse_kv_sglang_vllm.md)
DSV4.1 L0/L1 扩展：[`docs/hisparse_dsv41_simulation.md`](./hisparse_dsv41_simulation.md)

---

## 1. 目标与定位

sglang 的 HiSparse（`--enable-hisparse`）把**全量 KV 放在 pinned host DRAM**，HBM 里只保留
每请求固定大小的 hot buffer（`device_buffer_size`），decode 时由 DSA indexer 选出 top-k，
把 miss 的位置从 host 搬到 HBM。它依赖三样 CPU 环境没有的东西：

- DSA indexer 的真实 top-k 分数（需要模型前向 + CUDA kernel）；
- `sgl_kernel.kvcacheio` 的 swap-in kernel；
- `flashmla_sparse` / `flashinfer_sparse_mla` 等稀疏 MLA attention backend。

因此 hisim **不执行** `--enable-hisparse`，而是把 HiSparse 的**容量结构、数据搬运和关键路径
时延**建模进现有的 replay 循环：模型前向仍然是 mock，但 scheduler 看到的池子大小、并发上限、
每步 H2D/D2H 字节数和额外时延都按 HiSparse 的语义计算。开关是 `HISIM_HISPARSE=1`
（或配置里的 `hisparse.enable`），默认关闭，对现有仿真零影响。

这是一种**仿真级移植**：能回答“给定 HBM 预算和 top-k 命中率，HiSparse 能多放多少请求、每步
多花多少时间、产生多少搬运”，但不能替代真实 GPU 上的正确性/精度验证。

---

## 2. 与 sglang 实现的对应关系

| sglang HiSparse 语义 | 源码位置 | hisim 实现 |
| --- | --- | --- |
| `padded_buffer_size = device_buffer_size + page_size` | `hisparse_coordinator.py:186-188` | `HiSparseSimConfig.padded_buffer_size` |
| 每请求预留 `padded_buffer_size` 个 hot slot | `hisparse_coordinator.py:191-227` | `_WorkingSet.capacity`、`max_reqs_by_hot_pool` |
| 逻辑池 = 物理池 × `host_to_device_ratio` | `mem_cache/hisparse_memory_pool.py:58`、allocator | `logical_pool_tokens = physical × ratio`，scheduler 的 `max_total_num_tokens` |
| hot buffer 布局 `[LRU slots] + [newest page]` | `kvcacheio/hisparse.cuh:314-316` | LRU `OrderedDict` + 每步新增 token 直接记命中 |
| prefill KV 整体 D2H staging | `hisparse_coordinator.py:348-386` | `on_request_start()` 记录 `staging_bytes` |
| decode 每步 top-k swap-in（miss H2D） | `hisparse_coordinator.py:1005+` | `on_decode_step()`：working set vs hot buffer，miss 计 H2D |
| eager backup 上一步新 token（D2H，侧流） | `hisparse_coordinator.py:665-753` | 每步 1 token 的 D2H，默认不进入关键路径 |
| shared-index 预取：anchor 同步、skip 层侧流 | `hisparse_coordinator.py:1005-1045` | `shared_index_freq`，只有 `1/freq` 的字节计入关键路径 |
| 并发上限受 hot 池约束 | `hisparse_req_budget` / coordinator | `max_num_reqs = min(原上限, physical // padded_buffer_size)` |

关键文件：

- `src/hisim/simulation/sglang/hisparse_sim.py`：容量/访问/时延模型；
- `src/hisim/simulation/sglang/sglang_hook.py`：
  - `C_ModelRunnerHook.override_initialize`：放大逻辑池、限制并发；
  - `C_SchedulerHook.wrapped_run_batch`：decode 批次驱动仿真；
  - `wrapped_process_batch_result`：把 load/backup 时延注入全局时钟、写 iteration 字段；
  - `record_request_stats` / 请求完成：host 池记账；
  - `dump_profile_results`：合并 metrics 并写 `hisparse.json`。
- `src/hisim/simulation/manager/state.py`：per-iteration 累加器。

---

## 3. 容量模型

设：

- `P` = hisim 按 HBM 预算算出的物理 KV token 数（`max_total_num_tokens`，可被
  `--max-total-tokens` 覆盖）；
- `R` = `host_to_device_ratio`（默认 2）；
- `D` = `device_buffer_size`（默认 `2 * top_k`）；
- `page_size` = 1（当前 hisim 的 KV pool page 粒度）。

则：

```
logical_pool_tokens = P * R                 # scheduler 可见 / host 池容量
padded_buffer_size  = D + page_size         # 每请求在物理池里的预留
max_reqs_by_hot_pool = P // padded_buffer_size
```

`max_num_reqs` 取原 hisim 上限与 `max_reqs_by_hot_pool` 的较小值。仿真只把 **物理池** 记为
HBM KV（`hisparse_hbm_kv_bytes = P × layers × bytes_per_token_per_layer`），把 **逻辑池**
记为 host KV；`hisparse_hbm_saving_bytes = (逻辑池 - 物理池) × layers × bpt`。

MLA 模型的每 token 每层字节数取 `(kv_lora_rank + qk_rope_head_dim) × kv_dtype.itemsize`；
MHA/GQA 取 `2 × (kv_heads / tp_size) × head_dim × itemsize`。FP8 KV 按 1 字节/元素，
BF16/FP16 按 2 字节。DeepSeek-V3.2 因此是 576 B/token/layer（61 层），和 sglang 的
`bytes_per_token` 口径一致。

---

## 4. top-k 访问模型（可校准）

真实 top-k 由 indexer 产生，replay 中不可观测。hisim 用一条**每请求 canonical 访问流**建模，
再按层数放大字节/命中数（各层视为 i.i.d.）。`access_model` 三选一：

| 模型 | 行为 | 用途 |
| --- | --- | --- |
| `sticky`（默认） | 选择集 = 最近 `recent_ratio × top_k` 个位置 + 一个固定大小的旧位置集合；每步按 `churn` 比例替换旧集合 | 近似 DSA 的“局部注意力 + 少量长期重要 token” |
| `recency` | 只选最近 `top_k` 个位置 | 下界（对 HiSparse 最友好），用于对照 |
| `uniform` | 每步在历史上均匀抽 `top_k` | 上界（最不友好），用于对照 |

`sticky` 的关键参数：

- `recent_ratio`（默认 0.5）：top-k 中固定为“最近窗口”的比例；
- `churn`（默认 0.05）：每步被替换的旧集合比例。**这是校准主旋钮**：
  `churn=0` 等价于完美稳定工作集，`churn=1` 等价于每步完全换血。

hot buffer 是容量 `padded_buffer_size` 的 LRU。被替换掉的位置不会立即从 buffer 删除，
而是在 LRU 驱逐时自然淘汰——因此 buffer 比工作集大时，回访仍可能命中。这正是
`device_buffer_size ≥ top_k` 带来的余量。

每层视为 i.i.d. 意味着 `bytes = canonical_misses × num_layers × bpt`，不建模层间
index 共享的具体 pattern，只用 `shared_index_freq = N` 表示“每 N 层一个 anchor，
其余层在侧流预取”，关键路径只计 `1/N` 的 miss 字节。

**校准方法**：在真实 sglang HiSparse 部署上采 `swap_in` 的 miss token 数 / `top_k`、
每步 H2D 字节、hot-buffer 命中率，然后调 `churn`（以及必要时 `recent_ratio`）使仿真的
稳态命中率与实测一致。`sweep` 模式给出该参数的敏感性，便于先估计区间。

---

## 5. 时延模型

搬运时延沿用 hisim 现有的 L2 host 带宽模型（`sglang_hook._l2_transfer_duration`）：

```
duration = t0 + bytes / (raw_bandwidth × 0.85)
t0 = 6.67 µs (H2D) / 4 µs (D2H)
```

- **H2D swap-in**：`misses × num_layers × bpt × critical_fraction`，加到 decode 步的
  `process_batch_result` 全局时钟；一个 decode batch 只做一次 kernel（`t0` 计一次），
  带宽按 batch 内字节求和。overlap 时序模型（`OVERLAP_SCHEDULE=True`，默认）里
  swap-in 位于 forward 内部，因此也进入时钟；D2H backup 仍按侧流处理。
- **D2H eager backup**：1 个新 token × `num_layers × bpt`；`backup_on_critical=false`
  （默认）时只记账、不加时钟（对应 sglang 的侧流），`true` 时加到时钟；
- staging（prefill KV 整体 D2H）只记字节，不计时延（对应 staging 与 decode 重叠；
  真实预填充阶段的时延仍由 hisim 的前向模型给出）。

`sweep` 里可直接看到：DeepSeek-V3.2 几何（61 层、576 B/token/layer、物理 2.27M tokens、
top_k=2048、device_buffer=4096、64 请求 × 32k prompt）下，`churn` 从 5% → 50% 时，
稳态命中率 97.6% → 77.0%，H2D 56.7 → 542.5 GB，单请求每步 load 38.5 → 311.0 µs。

---

## 6. 配置

### 6.1 配置文件

```json
{
  "hisparse": {
    "enable": false,
    "top_k": 2048,
    "device_buffer_size": 4096,
    "host_to_device_ratio": 2,
    "page_size": 1,
    "access_model": "sticky",
    "churn": 0.05,
    "recent_ratio": 0.5,
    "shared_index_freq": 1,
    "backup_on_critical": false,
    "seed": 0
  }
}
```

`device_buffer_size: 0` 表示 `2 * top_k`（与 sglang 默认一致）；小于 `top_k` 会被抬到
`top_k`。

### 6.2 环境变量（优先级高于配置文件）

| 变量 | 对应字段 |
| --- | --- |
| `HISIM_HISPARSE` | `enable`（`1/true/yes/on`） |
| `HISIM_HISPARSE_TOP_K` | `top_k` |
| `HISIM_HISPARSE_DEVICE_BUFFER_SIZE` | `device_buffer_size` |
| `HISIM_HISPARSE_HOST_TO_DEVICE_RATIO` | `host_to_device_ratio` |
| `HISIM_HISPARSE_PAGE_SIZE` | `page_size` |
| `HISIM_HISPARSE_ACCESS_MODEL` | `access_model` |
| `HISIM_HISPARSE_CHURN` | `churn` |
| `HISIM_HISPARSE_RECENT_RATIO` | `recent_ratio` |
| `HISIM_HISPARSE_SHARED_INDEX_FREQ` | `shared_index_freq` |
| `HISIM_HISPARSE_BACKUP_ON_CRITICAL` | `backup_on_critical` |
| `HISIM_HISPARSE_SEED` | `seed` |

---

## 7. 输出指标

开启后 `metrics.json` 会多出 `hisparse_*` 字段，同时写出 `hisparse.json`（内容相同）：

| 指标 | 含义 |
| --- | --- |
| `hisparse_physical_pool_tokens` / `logical_pool_tokens` / `host_pool_tokens` | 三个容量口径 |
| `hisparse_padded_buffer_size` / `max_reqs_hot_pool` | buffer 与并发上限 |
| `hisparse_hbm_kv_bytes` / `host_kv_bytes` / `hbm_saving_bytes` | KV 占用（字节） |
| `hisparse_hits` / `misses` / `hit_rate` | top-k 命中（按层累计） |
| `hisparse_h2d_bytes` / `d2h_bytes` / `staging_bytes` / `total_io_bytes` | 搬运量 |
| `hisparse_load_latency_s` / `backup_latency_s` / `backup_hidden_latency_s` | 累计时延 |
| `hisparse_mean_load_us` / `mean_backup_us` | 每个 **decode batch** 平均时延 |
| `hisparse_peak_host_tokens` / `host_overcommit_tokens` | host 池占用/超卖 |
| `hisparse_decode_steps` / `decode_batches` / `requests` / `finished_requests` | 规模 |

一个 decode batch 由一次 swap-in kernel 服务，所以带宽按 batch 内所有请求的字节求和、
固定启动开销 `t0` **只计一次**；`hisparse_mean_load_us = 总 load 时延 / decode_batches`。
单请求场景（含 `sweep`）batch size 为 1，与按请求计的口径一致。

`ITERATION_STATS` 里每个 batch 也带 `hisparse_load_latency` / `hisparse_hits` /
`hisparse_misses` / `hisparse_h2d_bytes` / `hisparse_d2h_bytes`。

host 池按请求的 prompt+已生成 token 记账；超出 `logical_pool_tokens` 会打 warning 并累计
`host_overcommit_tokens`，不会中断仿真。

---

## 8. 验证

### 8.1 单元测试

```bash
PYTHONPATH=src python -m pytest test/test_hisparse_sim.py -q
# 18 passed
```

覆盖：配置解析/夹取、容量几何（`padded_buffer_size`、`P×R`、并发上限）、MLA/MHA 每层
字节数、请求生命周期与 staging 字节、重复 start/finish 幂等、sticky/recency/uniform 的
命中率排序、`churn` 单调性、seed 确定性、H2D/D2H 字节与关键路径时延、shared-index 缩放、
batch 聚合/单次启动开销、metrics 字段。

### 8.2 端到端 replay（Qwen3-0.6B，mock 权重，constant predictor）

工作负载：20 请求 × (1024 prompt + 64 output)；`top_k=256`、`device_buffer=512`、
`ratio=2`、`churn=0.05`。HBM 物理池 20000 tokens 时，逻辑池 40000。

| 场景 | completed | TTFT (ms) | TPOT (ms) | throughput (req/s) | mean load (µs/batch-step) |
| --- | --- | --- | --- | --- | --- |
| baseline（HBM=20000） | 20 | 5.75 | 0.525 | 289.9 | – |
| HiSparse（HBM=20000, logical=40000） | 20 | 2.40 | 0.748 | 403.7 | 223.0 |
| baseline（HBM=40000，无容量压力） | 20 | 2.40 | 0.525 | 563.4 | – |
| HiSparse（HBM=40000, logical=80000） | 20 | 2.40 | 0.748 | 403.7 | 223.0 |
| HiSparse（HBM=40000, `shared_index_freq=4`） | 20 | 2.40 | 0.586 | 508.6 | 60.8 |

HiSparse 运行（20000 物理）统计：hit rate **98.0%**、H2D **741 MB**、D2H **142 MB**、
staging **2.35 GB**；host 峰值 21740 tokens（< 40000，无超卖）。

结论：

1. **容量收益**：同样 20000-token HBM 预算，baseline 装不下 21760 逻辑 token 而排队
   （TTFT 5.75 ms），HiSparse 逻辑池 40000 可全部并发（TTFT 2.40 ms），吞吐
   289.9 → 403.7 req/s（**+39%**）。
2. **时延代价**：TPOT 0.525 → 0.748 ms（**+42%**）。20 并发下每步 H2D 约 11.8 MB，
   按 54.4 GB/s 有效带宽（64 × 0.85）约 216 µs，再加一次 6.67 µs 启动开销，
   与实测 223 µs/batch-step 一致。
   因此**无容量压力时 HiSparse 反而更慢**（563.4 → 403.7 req/s）：它是拿带宽换容量。
3. **shared-index 的杠杆**：`shared_index_freq=4` 时只有 1/4 的 miss 字节在关键路径，
   mean load 223 → 60.8 µs，TPOT 0.586 ms（较 baseline 仅 +12%）、吞吐 508.6 req/s
   （较 baseline −10%），几乎把开销藏进侧流。是否真能这样隐藏取决于模型的
   `index_topk_freq` / 预取实现，需实测校准。
4. 该 Qwen3 用例只验证 hook 链路与统计口径；真实的 DeepSeek-V3.2 规模收益看下面的 sweep。

### 8.3 解析 sweep（DeepSeek-V3.2 几何）

61 层、576 B/token/layer（FP8 MLA）、物理 2.27M tokens（≈80 GB HBM KV）、
top_k=2048、device_buffer=4096、ratio=2、64 请求 × (32768 prompt + 512 output)：

| churn | hit rate | H2D (GB) | staging (GB) | mean load (µs/request-step) |
| --- | --- | --- | --- | --- |
| 0.05 | 97.6% | 56.7 | 73.7 | 38.5 |
| 0.20 | 90.6% | 220.5 | 73.7 | 130.4 |
| 0.50 | 77.0% | 542.5 | 73.7 | 311.0 |

对应 HBM KV 79.8 GB（物理）/ host KV 159.5 GB（逻辑），hot 池可容纳 554 个并发请求
（`2270000 // 4097`）。`sweep` 逐请求调用（batch size 1），因此该表是**单请求每步**；
一个 64 请求的 decode batch 的 H2D 是它的 64 倍（churn=0.05 时约 110 MB/step，
≈2.0 ms @ 54.4 GB/s（64 × 0.85），`t0` 仍只计一次），这是 HiSparse 在大 batch 下的主要代价，
也是 `churn` 必须实测校准的原因。

### 8.4 复现

```bash
# 单测
PYTHONPATH=src python -m pytest test/test_hisparse_sim.py -q

# 端到端 + sweep（需要 sglang 源码、torch、一个本地模型目录）
PYTHONPATH=src BENCHMARK_TEST_MODEL_PATH=/path/to/Qwen3-0.6B \
    bash benchmark/hisparse/run_validation.sh
# 结果：benchmark/hisparse/results/{baseline_capacity,sparse_capacity,
#       baseline_roomy,sparse_roomy,sparse_roomy_freq4}.json
#       + sweep.json + summary.json
```

`benchmark/hisparse/hisparse_bench.py` 也支持单独跑：

```bash
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py sweep
PYTHONPATH=src python benchmark/hisparse/hisparse_bench.py bench --enable \
    --output /tmp/sparse.json --max-total-tokens 20000
```

> 端到端脚本里 hisim / sglang 的 hook 必须在**模块顶层**导入，且入口要有
> `if __name__ == "__main__":`——sglang 用 spawn 拉起 scheduler 子进程，子进程会以
> `__mp_main__` 重新导入模块；顶层导入才能让 hook 在子进程里生效。

---

## 9. 限制与后续

1. **没有真实 indexer**：top-k 是抽象流，`churn`/`recent_ratio` 需要真实部署校准；
   层间 i.i.d. 假设会低估/高估特定模型的层间相关性。
2. **token 粒度**：miss/IO 按 token 计，未建模 `page_size > 1` 的整页搬入；`page_size`
   目前只进入 `padded_buffer_size`。sglang 的 `swap_in_block_size` 是 kernel block size，
   不是 IO 粒度，故未建模。
3. **没有 L2/L3**：HiSparse 与 HiCache 在 sglang 中互斥；同时开启时 hisim 会 warning
   并只保留 HiSparse 的容量模型，L2/L3 仍可能被现有 hook 触发，建议验证时
   `enable_hierarchical_cache=False`。
4. **没有 radix cache**：sglang 要求 `--disable-radix-cache`；hisim 的 L1 prefix
   reuse 仍按原逻辑运行，HiSparse 的 host 池未与 radix 前缀共享打通。
5. **eager backup 不计关键路径**（sglang 用侧流）。若目标硬件侧流带宽与主带宽共享，
   需要把 `backup_on_critical` 设为 `true`。
6. **PD 分离未建模**：只覆盖单实例“本地 prefill → staging → decode”路径；direct
   （RDMA 直写 host）路径的 host 池跨节点容量/带宽不在本模型内。
7. **KV dtype**：按 `kv_cache_dtype` 的元素大小计算；FP8 per-tensor scaling 的额外开销、
   indexer K 常驻 HBM 的部分都未单列。

后续如果要提高保真度，优先做两件事：(a) 用真实 sglang HiSparse 计数器校准 `churn`；
(b) 把 `page_size` 真正用于 IO 粒度与 host 页分配，以覆盖 DSv4 的压缩布局。
