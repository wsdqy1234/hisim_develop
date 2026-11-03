# sglang / vLLM 稀疏 KV 方案分析：是否存在「KV 驻留 DRAM/SSD + top-k 搬回 HBM」

分析日期：2026-11-03
分析分支：`develop_lfz_sparse_kv`（基于 `develop_lfz_0.5.19` @ `f3723a1`）

代码基线：

- sglang：`/sgl-workspace/sglang`，`release/v0.5.19` cherry-pick，commit `0bcd822377`
- vLLM：`/home/l00998189/vllm`，v0.30.0，commit `ced6857afa`（2026-09-21）

---

## 0. 结论速览

**有。两边的对应特性都叫 HiSparse（Hierarchical Sparse Attention），且 vLLM 的实现明确是从 sglang 移植的**
（`csrc/libtorch_stable/hisparse_kernels.cu:1-30` 注释：*"The swap-in algorithm is a port of SGLang's hisparse load_cache_to_device_buffer kernel"*）。

共同范式：

```
全量 KV ──► pinned host DRAM pool（每层一份）
                 ▲   │
   D2H 写回/溢出  │   │  top-k miss 时 H2D 拷入
                 │   ▼
HBM：per-request 固定大小 hot / device buffer（+ 仍在 HBM 的 indexer KV）
                 │
                 └──► 稀疏 MLA attention kernel 直接读 top-k 对应的 HBM 行
```

关键前提：**top-k 不是框架自己估出来的，而是 DSA（DeepSeek Sparse Attention）模型自带的 indexer 输出**
（DeepSeek-V3.2 / GLM-5.x / DeepSeek-V4）。因此「只把 top-k KV 放 HBM」不损失精度，只是把原先
「HBM 里全量 KV + 稀疏 gather」换成「HBM 里只放 hot 集 + 稀疏 gather 前先换页」。

SSD 层：

- 两个 HiSparse **都不含 SSD 层**，只有 pinned host DRAM。
- SSD 只出现在各自的**通用 prefix-cache offload** 里（sglang HiCache storage / vLLM
  `TieringOffloadingSpec` + `fs` tier）。那是「前缀块命中→整块搬回 HBM」，**不是 top-k 选择**。

---

## 1. sglang：HiSparse

### 1.1 开关与参数

`python/sglang/srt/server_args.py:2824-2832`

```
--enable-hisparse                     # 默认关闭
--hisparse-config '{"top_k":2048,"device_buffer_size":4096,
                    "host_to_device_ratio":2,"swap_in_block_size":960}'
```

默认值见 `mem_cache/sparsity/factory.py:60-117`：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `top_k` | 2048 | 每步保留的 top-k token 数；若 `hf_config.index_topk` 存在则以模型值为准（`model_runner.py:929-931`） |
| `device_buffer_size` | `2*top_k` = 4096 | **每请求**在 HBM 的 device buffer 行数（必须 ≥ top_k） |
| `host_to_device_ratio` | 2 | host 池容量 = device 池容量 × ratio；同时决定逻辑索引池大小 |
| `swap_in_block_size` | 960 | swap-in CUDA kernel 的 **thread-block 大小**（≤1024），不是 token 粒度 |

官方文档：`docs/docs/advanced_features/hisparse_guide.mdx`。

### 1.2 存储层级与容量

- **host 侧**：`MLATokenToKVPoolHost`（`mem_cache/pool_host/mla.py:51`），
  `pin_memory=True`（`memory_pool_host.py` → `pool_host/base.py:113-200`），
  容量 `size = device_pool.size * host_to_device_ratio`（`host_size=0` 时走该公式）。
  host 侧按 page 分配 slot（`pool_host/hisparse.py:11-40`），按层各一份 buffer。
- **HBM 侧**：`HiSparseDSATokenToKVPool`（`mem_cache/hisparse_memory_pool.py:28-59`）
  - MLA KV：只有 `size` 个 slot（所有请求共享的 hot 池，每请求 `device_buffer_size`）。
  - **indexer K buffer 仍是全长的**：`index_buf_size = size * host_to_device_ratio`
    （同一文件 58 行）。也就是说 indexer 需要的 K 始终在 HBM，被 offload 的只是
    体积大的 MLA KV。
- **逻辑 → 物理映射**：`HiSparseTokenToKVPoolAllocator`
  （`mem_cache/allocator/hisparse.py:15-70`）
  - `logical_attn_allocator` 管 `size_full = size * ratio` 的逻辑槽位；
  - `hisparse_attn_allocator` 管 `size` 个 HBM device buffer 槽位；
  - `full_to_hisparse_device_index_mapping[logical_idx] = device_idx`。

### 1.3 请求准入：两条路

`managers/hisparse_coordinator.py`

1. **staging（本地 prefill 完成）** — `admit_request_into_staging()`（348-386）
   prefill KV 本来在 HBM；在与 decode 并行的 `write_staging_stream` 上
   `backup_from_device_all_layer(..., io_backend="kernel")` 把整段 KV 拷到 host pool，
   然后把 ack 放进 `ack_staging_queue`，请求进入 decode。
2. **direct（PD 分离）** — `admit_request_direct()`（388-419）
   prefill 节点通过 RDMA 把 KV **直接写进 decode 节点的 host pool**（绕过 decode GPU），
   decode 侧只 `alloc_device_buffer()`；长序列把 `device_buffer_tokens` 置 `-1`（全 miss），
   短序列 `_preload_to_device_buffer()` 预载。

> 文档明确写「currently requires PD disaggregation mode, enabled on the decode instance」；
> 代码里 staging 分支说明非 PD 单实例也能跑本地 prefill→decode，但官方部署形态是 PD。
> 另有校验：只支持 DSA / DeepSeek-V4，必须 `--disable-radix-cache`，KV dtype 必须是
> `bfloat16` / `fp8_e4m3`（`arg_groups/hisparse_hook.py`），DSA backend 限
> `flashmla_sparse` / `flashmla_kv` / `flashinfer_sparse_mla`（CUDA）或 `tilelang` / `aiter`（ROCm）。

### 1.4 decode 关键路径：top-k → swap-in kernel

调用点：`layers/attention/dsa_backend.py:2259-2270`

```python
if self.hisparse_coordinator is not None:
    page_table_1 = self.hisparse_coordinator.swap_in_selected_pages(...)
```

流程（`hisparse_coordinator.py:759-850` → `937-990`）：

1. 模型 indexer 每层给出 top-k 的**逻辑 token 位置**（`top_k_result`）。
2. `swap_in_selected_pages` 按请求拼出 `top_k_device_locs`：
   - 短序列（`seq_len ≤ device_buffer_size`）：直接 `req_to_device_buffer[req, selected]`，无 IO；
   - 长序列：逐请求、逐 token 判断 `device_buffer_tokens` 里是否已有该 token（命中），
     未命中的进入 miss 计划。
3. 调 `load_cache_to_device_buffer_mla(...)`（`kernels/ops/kvcache/hisparse.py:271+`，
   CUDA 实现在 `kernels/jit/csrc/kvcacheio/hisparse.cuh`）：
   - 每个请求一个 CUDA block（`__launch_bounds__(BLOCK_SIZE)`，模板参数就是
     `swap_in_block_size`）；
   - 共享内存里建 **open-addressing hash**（Knuth 乘法哈希）做 top-k token → buffer slot 命中判断；
   - 维护 LRU slot 表，找不到空位时选 LRU victim；
   - 对 miss 的 token，从 `host_cache[host_loc]` 按 `item_size_bytes = kv_cache_dim*dtype.itemsize`
     逐行 H2D 拷到 device buffer 的 slot；
   - 输出 `top_k_device_locs`（即 `page_table_1`），供 `flashmla_sparse` / `flashinfer_sparse_mla` /
     `tilelang` 直接当稀疏 MLA 的 KV 索引。
4. device buffer 布局：`[HOT_BUFFER_SIZE 个 LRU slot] + [page_size 个 newest-token slot]`
   （`hisparse.cuh:314-316`），最新 token 固定放在 `HOT_BUFFER_SIZE` 处。

### 1.5 写回（D2H）

- **eager backup**：`_eager_backup_previous_token()`（665-753）在每步 decode 后把**上一步新写的
  token KV** 异步 D2H 到 host pool（与 forward 重叠），保证 host 侧始终有最新 token。
- `_grow_device_buffers()`（488-560）：序列变长需要更多 hot slot 时扩容并重映射。
- 新 token 的 KV 先落 HBM（逻辑池 → 预留 device-buffer slot），随后由 eager backup 同步到 host。

### 1.6 共享索引预取（shared-index prefetch）

`resolve_shared_index_layers()` + `_init_shared_index_prefetch()`（265-317）、
`_run_copy_only_kernel()`。

当模型在若干层间复用 anchor 层的 `index_topk`（DSA `index_topk_freq`/`index_topk_pattern`，
GLM-5.2 的 IndexShare）时：anchor 层的 swap-in kernel 把 miss 计划（host slot → device slot）
记录下来，后续 skip 层直接在 **side stream** 上重放同一份拷贝计划，把 H2D IO 藏进中间层的计算里。
可用 `SGLANG_DISABLE_HISPARSE_PREFETCH=1` 关闭；PP>1 或投机解码时退化为同步 swap-in。

### 1.7 限制

- 仅 DSA / DSv4；必须关 radix cache；官方要求 PD decode 实例。
- `sgl_kernel.kvcacheio` 仅 CUDA/ROCm；`get_cpu_copy`/`load_cpu_copy` 在 HiSparse device pool 上
  直接 `NotImplementedError`（`hisparse_memory_pool.py:118-122`）。
- DSv4 走 C4 压缩布局（`compress_ratio`，host 页数 = `size_full // compress_ratio / page_size`），
  ROCm 下 unified-KV 模式不支持。
- 指标：`HiSparseTokenStats`（device/host token 数与占用率，`hisparse_coordinator.py:331-346`）。

### 1.8 另一套「稀疏框架」：`SparseCoordinator`

`mem_cache/sparsity/core/sparse_coordinator.py`（Quest / PQCache / SnapKV 类算法）目前只做
**top-k 选择 + 改 attention metadata**；`forward_begin`/`forward_end` 的 offload 明确写着 TODO，
`on_request_end` 的 host index 释放也是 TODO。即：**通用稀疏框架还没有 DRAM→HBM 搬运，只有 HiSparse 有。**

---

## 2. vLLM：HiSparse

### 2.1 启用方式

```bash
vllm serve <DSA model> \
  --kv-transfer-config '{
    "kv_connector": "HiSparseConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"host_pool_gib": 200}
  }'
```

- `host_pool_gib` **必填**，是每个 DP replica 的 host 容量
  （`config/kv_transfer.py:16-42`）。
- 配置了该 connector 后会自动推断出 `attention_config.hisparse_config`
  （`config/vllm.py:1630-1642`）；也可显式设 `HiSparseConfig` 调参
  （`config/attention.py:18-34`）：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `device_buffer_size` | `(max_decode_query_len+1)*top_k` | 每请求 GPU hot 行数，下限 `max_query_len*top_k`，上限 int16 |
| `eager_host_mirror` | True | forward 中把 decode 新写的 KV 行镜像到 host，使 page spill 不再搬数据 |

硬约束（`config/vllm.py:1640-1680`、`2050-2062`）：

- 仅 NVIDIA CUDA；`Model Runner V2`；需要 HMA（hybrid KV cache manager）；
- 不支持 PP、不支持 DCP；
- 模型必须有 `hf_config.index_topk`（DSA 模型），`top_k` 直接取该值；
- P/D 场景只有 `NixlConnector` / `MooncakeStoreConnector` / `MultiConnector` 是验证过的组合。

设计文档：`docs/design/hisparse.md`（"experimental"）。

### 2.2 架构：connector + scheduler coordinator + worker runtime

与 sglang「一个大 coordinator 管所有池」不同，vLLM 把它拆成 KV connector 体系：

```
scheduler 进程                          worker 进程
HiSparseConnector                       HiSparseConnector
  └─ HiSparseCoordinator                  └─ HiSparseWorker
       - 逻辑 host block 分配                  - pinned host 物理池
       - source/prefix identity                - 拷贝调度（enqueue/complete）
       - resident lease / spill 事务            - 每个 cache 的 HiSparseRuntime
```

- `HiSparseCoordinator`（`v1/hisparse/coordinator.py`，778 行）：**只管逻辑 host block**、
  residency 策略、spill 事务、block-table 更新；不分配 GPU block。
- `HiSparseRuntime`（`v1/hisparse/runtime.py`，1291 行）：per-cache 的 host/hot 视图、GPU LRU、
  fused 解析；`HiSparseCacheHandle` 暴露给 attention 层。
- GPU 侧：**resident pages（普通 KV block，HMA 分配）+ 每请求固定 hot 区**共享一个 packed allocation；
  host 侧：`allocate_pinned_host_pool()` 分配 pinned + 显式注册（`runtime.py:181-232`）。
- 单机 MP 张量并行时，所有 TP rank 映射**同一份** pinned host pool（MLA source KV 在 TP 间复制），
  由 rank 0 写、其余 rank 等 IPC event 后读；其他并行形态每 rank 私有。

### 2.3 decode 关键路径：fused resolver（resident → hot → host）

`runtime.py:881-1067` + `csrc/libtorch_stable/hisparse_kernels.cu:190+`

`hisparse_resolve_residency_kernel` 每个 batch row 一个 block，对 top-k 的每个位置：

```
indexer 输出 top-k 逻辑位置
        │  source block table → global KV slot id
        ▼
resident block table 命中？ ──是──► resident GPU 行（直接返回，kernel 内提前退出）
        │否
        ▼
hot LRU hash 命中？ ──是──► 复用 hot 行 + 更新 GPU LRU
        │否
        ▼
选 GPU LRU victim ──► 从 pinned host 行 H2D 拷入 hot 行
```

关键设计点（`hisparse_kernels.cu:1-30`）：

- **寻址用全局 KV slot id**，而不是「请求内位置」。host pool 的第 i 行就是全局 slot i，
  因此不需要 sglang 那种 per-request 的 host 位置表，只需 block_table 转换。
- 每个 batch row 拥有固定 `region_stride` 的 hot 行；`[0, hot_size)` 由 LRU 管理，
  `hot_size` 永远是本行最新 token（由 KV-cache update 直接写）。
- 解析完全在 GPU 上做，**没有 CPU 侧决策、没有 device→CPU 回读**，可 CUDA graph 捕获。
- 按 index-sharing 分组：每个 group 只解析一次（leader），followers 复用同一份 miss plan；
  `_stage_this_layer` + `wait_for_staged_swap` 在 side stream 上提前拷贝。

### 2.4 spill / 写回

- `eager_host_mirror=True`（默认）：decode forward 时就把新写行镜像到 host，
  之后 GPU block 被回收时「spill」无需再搬数据。
- 关闭时：decode 行只留在 GPU，页被换出时才 D2H（`config/attention.py:29-33`）。
- spill 事务有严格顺序（`docs/design/hisparse.md:166-195`）：
  scheduler pin 源/目的 lease → 下发 `SparseKVPageTransfer` → worker 入队后 ack（此时
  stream 顺序已保证 GPU block 可复用）→ 拷贝 event 完成后 ack → 才发布 host page 供前缀复用。
- `SparseKVOffloadCommand` / `SparseKVRowMirror`（`v1/hisparse/types.py`）通过
  `kv_connector_metadata` 在 scheduler↔worker 间传递。

### 2.5 P/D 与 indexer KV

- P/D 导入目标**每请求只选一次**：完整前缀能放进 device pool → NIXL 直接写 resident GPU 页；
  否则（host 固定足迹 + host source block 放得下）导入 host tier，经有界 staging pool 中转，
  需要立即用的页顺带镜像到 resident 目的页。
- **indexer KV 不归 HiSparse 管**：它仍是普通 prefix-cacheable GPU cache group；
  若同时配了 `OffloadingConnector`，indexer 组走通用 KV offload 路径
  （`docs/design/hisparse.md:152-164`）。

### 2.6 与「通用 KV offload」的区别

vLLM 另有一套 `OffloadingConnector`（`docs/features/kv_offloading_usage.md`）：

- `CPUOffloadingSpec`：GPU block → pinned host，LRU/ARC 驱逐，**按 prefix 命中整块 promotion**；
- `TieringOffloadingSpec`：CPU primary + 二级 tier（`fs` 本地文件系统 / 对象存储），
  GPU↔二级必须经 CPU 中转。

这是「prefix-cache 扩展」，与 top-k 无关；SSD 也只在这条路径里。
`kv_offload/` 目录下还有 `cpu/`、`fs/`、`tiering/` 等实现，可作 SSD 层的参考，
但它解决的是「前缀复用/容量扩展」，不是「稀疏注意力工作集换页」。

### 2.7 指标

`v1/kv_connector/v1/hisparse/stats.py`：

- `cache_hits` / `cache_misses` / `host_to_device_bytes`（CLI 日志三项）；
- Prometheus：`vllm:hisparse_*` 计数器（另有通用的 `vllm:kv_offload_load_bytes`）。

---

## 3. 对比表

| 维度 | sglang HiSparse | vLLM HiSparse |
| --- | --- | --- |
| 开关 | `--enable-hisparse` + `--hisparse-config` | `--kv-transfer-config` 里配 `HiSparseConnector`（`host_pool_gib` 必填） |
| 代码形态 | 一个大 `HiSparseCoordinator` + 专用 pool/allocator | KV connector + scheduler coordinator + worker runtime 三层拆分 |
| 支持模型 | DSA（V3.2 / GLM-5.x）、DeepSeek-V4 | DSA（必须有 `index_topk`），CUDA only |
| 全量 KV 位置 | pinned host DRAM（每层一份，`host_to_device_ratio` 决定容量） | pinned host DRAM（`host_pool_gib` 每 DP replica，TP 间可共享一份物理内存） |
| HBM 侧结构 | 每请求 `device_buffer_size` 行的 device buffer；indexer K 仍全长在 HBM | resident 普通 KV block（HMA）+ 每请求固定 hot 区；indexer KV 是独立 GPU group |
| top-k 来源 | 模型 indexer（`index_topk`） | 模型 indexer（`index_topk`） |
| 命中层级 | device buffer 命中 → miss 走 host | resident → hot LRU → host（三级，resident 命中在 kernel 内提前退出） |
| 搬运单位 | token 行（`kv_cache_dim*dtype.itemsize`），host 侧按 page 分配 slot | token 行（全局 slot id），hot 区按 `region_stride` 分块 |
| 换页 kernel | `load_cache_to_device_buffer_mla`（JIT CUDA），open-addressing hash + LRU | `hisparse_resolve_residency_kernel`（libtorch stable CUDA，sglang 内核移植） |
| 新 token | eager backup 上一步 token D2H；newest slot 固定在 `HOT_BUFFER_SIZE` | `eager_host_mirror` 在 forward 内镜像；newest slot 在 `hot_size` |
| 预取 | shared-index prefetch：anchor 记录 miss plan，skip 层 side stream 重放 | 同样按 index-sharing 分组，leader 解析一次，followers 复用 plan + side stream staging |
| 写回时机 | 每步 decode 后异步 D2H（与 forward 重叠） | spill 事务（enqueue / complete 两级 ack）；默认 forward 内镜像 |
| P/D | 官方要求 PD decode 实例；direct-to-host 用 RDMA 直写 host pool | 可选；NIXL（能放下就直写 resident GPU，否则 host tier）/ Mooncake 已验证 |
| 平台 | CUDA + ROCm（部分路径） | 仅 NVIDIA CUDA |
| 限制 | 必须关 radix cache；DSv4 有 C4 压缩专用布局；PP/投机解码下预取退化 | 需 Model Runner V2 + HMA；不支持 PP/DCP；`device_buffer_size` 受 int16 限制 |
| 指标 | `HiSparseTokenStats`（device/host tokens、usage） | hits/misses/host_to_device_bytes + `vllm:hisparse_*` |
| SSD 层 | 无（SSD 在 HiCache storage，prefix 粒度） | 无（SSD 在 `TieringOffloadingSpec` 的 `fs` tier，prefix 粒度） |

---

## 4. 要点与差异

1. **两者都是「模型原生稀疏」而非「框架估算稀疏」**：top-k 来自 DSA indexer，
   与 Quest/PQCache/SnapKV 这类「用 query 估分数再选块」的路线不同；后者在 sglang 里
   还没有 offload 实现（`SparseCoordinator` 的 `forward_end` 仍是 TODO）。
2. **indexer KV 必须留在 HBM**：被 offload 的只是体积占大头的 MLA KV（DSv4 下是 C4/C128 压缩后的部分）。
   这是「能省多少 HBM」的上限来源。
3. **寻址方式是最主要的实现差异**：sglang 用「请求内位置 + per-request host 表」，
   vLLM 用「全局 KV slot id」，后者省掉一张 per-request 表，但要先做 block_table 转换；
   vLLM 的内核注释也承认这是对 sglang 的改写。
4. **vLLM 多一层 resident（普通 KV block）**：能在 device pool 放下的前缀就不进 host，
   只有放不下的部分才落到 host tier，解析时 resident 命中直接短路；sglang 则是
   「device buffer / host」两层。
5. **都没有 SSD 参与 top-k 换页**。若要做「DRAM/SSD 两级 + top-k」，需要在 HiSparse 的
   host 层之下再加一级（可参考 vLLM `TieringOffloadingSpec` 的 `fs` tier 或 sglang HiCache
   的 storage backend），并把 host miss 变成二级 IO —— 目前两边都没做。
6. **对模拟器（hisim）的含义**：这两个实现给出了一个可建模的参考结构——
   per-request hot 集合、token 行粒度搬运、LRU 淘汰、anchor/skip 层共享 miss plan、
   eager D2H 写回；容量参数分别是 `device_buffer_size × 并发请求数`（HBM）
   与 `device_size × host_to_device_ratio` / `host_pool_gib`（DRAM）。

---

## 5. 关键源码索引

### sglang

| 文件 | 作用 |
| --- | --- |
| `python/sglang/srt/server_args.py:2824-2832` | `--enable-hisparse` / `--hisparse-config` |
| `python/sglang/srt/mem_cache/sparsity/factory.py:60-117` | 配置解析与默认值 |
| `python/sglang/srt/model_executor/model_runner.py:919-947` | 构造 `HiSparseCoordinator`，`top_k` 取 `index_topk` |
| `python/sglang/srt/managers/hisparse_coordinator.py` | 准入、device buffer 管理、swap-in、eager backup、共享索引预取 |
| `python/sglang/srt/mem_cache/hisparse_memory_pool.py:28-59` | HBM device pool + `index_buf_size=size*ratio` |
| `python/sglang/srt/mem_cache/allocator/hisparse.py:15-70` | 逻辑池 / device buffer 双 allocator |
| `python/sglang/srt/mem_cache/pool_host/hisparse.py:11-40` | host 侧按页分配 token slot |
| `python/sglang/kernels/ops/kvcache/hisparse.py:271+` | `load_cache_to_device_buffer_mla` 封装 |
| `python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh` | swap-in CUDA kernel（hash + LRU + host 拷贝） |
| `python/sglang/srt/layers/attention/dsa_backend.py:2259-2270` | 每层调用 `swap_in_selected_pages` |
| `python/sglang/docs/docs/advanced_features/hisparse_guide.mdx` | 官方使用文档 |
| `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` | 通用稀疏框架（offload 仍 TODO） |

### vLLM

| 文件 | 作用 |
| --- | --- |
| `vllm/config/attention.py:18-34` | `HiSparseConfig` |
| `vllm/config/vllm.py:1630-1680` / `2050-2062` | 启用条件与约束校验 |
| `vllm/config/kv_transfer.py:16-42` | `host_pool_gib` 解析 |
| `vllm/distributed/kv_transfer/kv_connector/factory.py:201` | `HiSparseConnector` 注册 |
| `vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py` | scheduler/worker 边界与 metadata |
| `vllm/v1/hisparse/coordinator.py` | 逻辑 host block、spill 事务、residency 策略 |
| `vllm/v1/hisparse/runtime.py` | host/hot 视图、LRU、fused resolver 调用 |
| `vllm/v1/hisparse/types.py` | `SparseKVPageTransfer` / `SparseKVRowMirror` / `SparseKVOffloadCommand` |
| `vllm/csrc/libtorch_stable/hisparse_kernels.cu` | resolve / gather / invalidate kernel（sglang 移植） |
| `vllm/v1/attention/backends/mla/flashmla_sparse.py:300-320` | sparse MLA 与 HiSparse 的联动 |
| `vllm/docs/design/hisparse.md` | 设计文档（ownership、spill、hot lookup） |
| `vllm/docs/features/kv_offloading_usage.md` | 通用 CPU/SSD tier offload（非 top-k） |
