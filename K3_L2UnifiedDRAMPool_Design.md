# K3 L2 Unified DRAM Pool — 最小版本设计（KDA / MLA 共享物理 Allocator）

> 分支：`develop_dqy_l2_one_pool`（基线 `develop_dqy_evict_reuse` @ `199b2a8`）
> 状态：最小可用实现，默认关闭（未设置任何 `HISIM_L2_*` 时行为与基线完全一致）
> 测试：`test/test_l2_dram_pool.py`（9 项）、`benchmark/collection/l2_pool_frag_study.py`

---

## 1. 目标与范围

| 要求 | 本设计的落点 |
|---|---|
| Two Pools 与 One Pool 使用**相同 DRAM 总字节预算** | 两者都由 `HISIM_L2_TOTAL_GB`（未设则 `--hicache-size`）给出同一个 `total_bytes`；Two Pools 只把这份预算**静态切开**，One Pool 动态共享 |
| KDA 仍以 Checkpoint/Slot 为逻辑对象 | 逻辑身份仍是 content-addressed `slot_hash`；`_slots` / `_lru` / `load_to_device` / L3 提升路径不变 |
| MLA 仍以 Token/Block 为逻辑对象 | 逻辑身份仍是 token index（`int64`）；`host_value` / `get_data_page(index)` / 恢复路径不变 |
| 共享物理 Allocator，保留各自逻辑 Metadata、Identity、Restore | 物理层 `PageExtentAllocator` 唯一；上层 `MLAView` / `KDAView` 各自维护 extent 表、台账与计价点 |
| 研究 Size Class / Page / Extent，明确内部/外部碎片 | `exact`（精确页数）与 `pow2`（2 的幂 size class）两种策略；内部/外部碎片有定义、有计数器、有实测（§8） |
| 不改 sglang 原生 L1 Eviction / Lock / ACK 生命周期 | 只改 L2 host 池的字节来源（`alloc/free`）与容量；`evict_host` / `write` / `_finish_*_ack` 全部沿用原生实现 |
| L3 保留既有 Write-through / Restore | 触发 L2 驱逐的路径仍走 `_last_evicted_hashes` / `_kda_l3_pending_writes` / `st.batch_set*`；本设计不新增也不绕过 L3 |
| 不加 Cost-Aware Placement / Prefetch | 驱逐**顺序**仍由 `l2_placement_policy` 决定；本设计只改"有多少字节、分给谁" |

非目标：地址压缩/搬运、best-fit 之外的分配优化、跨进程预算、真实 DMA 缓冲（L2 池仍是 mock 计时 + 逻辑视图）。

---

## 2. 架构

```
                    HISIM_L2_TOTAL_GB  (= --hicache-size 默认)
                                 │
        ┌────────────────────────┴────────────────────────┐
        │        Physical layer: PageExtentAllocator       │
        │  page_bytes = HISIM_L2_PAGE_TOKENS × token_bytes │
        │  free runs (address-sorted) → first-fit extent   │
        │  MLA 从低地址、KDA 从高地址两端分配（two-ended）  │
        │  policy: exact | pow2 (size class)               │
        └───────────┬──────────────────────────┬───────────┘
                    │                          │
        ┌───────────┴───────────┐  ┌───────────┴────────────┐
        │ MLAView               │  │ KDAView                │
        │  object: token/block  │  │  object: slot hash     │
        │  index : token idx    │  │  key   : kda_<...>     │
        │  charge: alloc/free   │  │  charge: admit/evict   │
        │  owner[] token→extent │  │  slots{hash→extent}    │
        └───────────┬───────────┘  └───────────┬────────────┘
                    │                          │
        MockTokenToKVPoolHost      MockMambaStatePoolHost (×2)
                    └──────── 原生 HostPoolGroup.entry_map ─────┘
                          (KV / MAMBA entry 重指到上面两个 mock)
```

| 层 | 职责 | 代码 |
|---|---|---|
| 预算 | 决定 `total_pages`，Two Pools 切分 / One Pool 单区 | `UnifiedDRAMPool.__init__` |
| 物理 | 页 → 连续 extent，first-fit，free 时相邻合并；MLA 低端 / KDA 高端 | `PageExtentAllocator` |
| 逻辑 MLA | token 索引分配/释放、extent 引用计数、可用量 | `MLAView` |
| 逻辑 KDA | hash 准入/释放、slot→extent | `KDAView` |
| 适配 | 把预算接进 sglang 的 host pool 与 `HostPoolGroup` | `sglang_mock_class.py` / `sglang_hook.py` |

---

## 3. 逻辑对象与身份保持

### 3.1 MLA（Token / Block）

- `MockTokenToKVPoolHost.alloc(n)` → `MLAView.alloc_tokens(n)`：
  - 预留 `ceil(n / page_tokens)` 页，返回该 extent 的前 `n` 个 token id（`int64` tensor）。
  - token id 在 extent 内连续，因此 `host_value` 仍是普通整数索引；`get_data_page(index)` 语义不变（mock 内是 dummy page）。
- `free(indices)` → `free_tokens(indices)`：
  - 用 `owner[]`（`int32`，长度 = 区域页数 × `page_tokens`）把 token 映射回 extent；
  - extent 内仍有存活 token 时不归还页（引用计数 `live`），归零才 `free`，支持"分块 free"。
- 计价点：`alloc` / `free`。**每一页只属于一个 extent，绝不重复计价。**

### 3.2 KDA（Checkpoint / Slot）

- 逻辑身份仍是 `kda_{req_id}_{token_offset}`（class-level 池）与 `kda_{nid}_{i}`（tree 路径）两套命名空间，**不合并**。
- 计价点：`backup_from_device(hash)` 准入（1 slot extent = `ceil(slot_bytes/page_bytes)` 页）/ `evict()` 释放。
- `MockMambaStatePoolHost.alloc(n)` 只发**逻辑 host index**（index 池），不产生物理字节；这是刻意的语义划分：
  - "L2 是否真的驻留 checkpoint" 由 `has(hash)` 决定，而 `has` 由物理准入决定 → 不会出现"有 host index 但没有 DRAM"的假命中；
  - 准入失败时调用方仍把 hash 追加到 L3 待写队列（`_kda_l3_pending_writes`），L3 write-through 不丢。
- `_last_evicted_hashes` / `_kda_l3_pending_writes` / `load_to_device` / `notify_evicted` 全部保留；跨视图回收触发的驱逐也走 `notify_evicted`，因此 L3 提升与原生一致。

### 3.3 两个 mamba L2 池

| 池 | 用途 | hash 形态 |
|---|---|---|
| `C_HiCacheController._kda_host_pool` | HiSim 自己的 KDA L1 驱逐备份 | `kda_{req_id}_{token_offset}` |
| `cache.mamba_pool_host` | 原生 tree 组件（`MAMBA`）备份/预取 | `kda_{nid}_{i}` |

两者都注册到**同一个** `KDAView`（同一份字节预算），但各自保留 dict/LRU/策略对象，互不干扰。

---

## 4. 物理层：Page / Extent / Size Class

### 4.1 页

```
page_bytes = HISIM_L2_PAGE_TOKENS × mla_token_bytes
```

- `mla_token_bytes` 由 `calc_kv_host_size_per_token(device_pool)` 从真实 device 池推导（与 mock host 池的 `size_per_token` 同源，保证"预算 / 每 token 字节"与既有时序模型一致）。
- 默认 `page_tokens=16`。K3 真实口径（5,760 B/entry）下一页 = 92,160 B，且 **KDA slot 14,653,440 B = 159 页整除**；本机 mock 池口径（52,992 B/token）下一页 = 847,872 B、slot = 18 页（17.28 → 向上取整，见 §8 内部碎片）。

### 4.2 Extent

- 一次请求 = 一段**连续页**（DMA/`copy_` 需要连续）。
- 空闲空间用地址有序 run 列表维护；分配 first-fit 从 run 头部切；`free` 时按地址插入并与左右相邻 run 合并（coalescing）。
- **两端分配（two-ended）**：MLA 从空闲 run 的**头部**（低地址）取，KDA 从**尾部**（高地址）取。两种对象形状（KDA 恒为 `slot_pages`，MLA 16–512 页）于是从两端向中间生长，只在相遇处互相干扰，避免长寿命 KDA extent 把 MLA 的空闲空间切碎（§8 实测：同样负载下 one_pool 占用率因此从 85–93% 提升到 96–98%）。
- 因此 `free_pages` 足够但 `largest_free_run` 不足时，大请求会失败——这就是外部碎片的物理含义。

### 4.3 Size Class

| 策略 | 行为 | 碎片特征 |
|---|---|---|
| `exact`（默认） | 精确 `ceil(bytes/page)` 页 | 内部碎片 ≤ 1 页/对象；外部碎片相对高 |
| `pow2` | 向上取 2 的幂页数（classic size-class/buddy 风格） | 内部碎片可达 2×；外部碎片低 |

### 4.4 碎片定义（有计数器、可直接进 metrics）

```
internal_frag_bytes = Σ_extent ( reserved_pages × page_bytes − requested_bytes )
                     （存活 extent 的尾部浪费）
external_frag_bytes = free_bytes − largest_free_run_bytes
                     （"有字节但凑不出一个连续大请求"的部分）
internal_frag_ratio = internal / used_bytes
external_frag_ratio = external / free_bytes
```

> 说明：`external_frag_ratio` 的分母是**空闲**字节而不是总量，因此它回答"剩下的空间里有多少是碎片"，不随占用率变化而失真。

---

## 5. 两种模式，同一预算

### 5.1 启用条件

```bash
# 默认（都不设）：完全走基线，零回归
HISIM_L2_TOTAL_GB=8                 # Two Pools：同一总预算，静态切分
HISIM_L2_TOTAL_GB=8 HISIM_L2_ONE_POOL=1   # One Pool：同一总预算，动态共享
```

- `HISIM_L2_TOTAL_GB` 未设置且 `HISIM_L2_ONE_POOL=0` → **不创建**统一池，`MockTokenToKVPoolHost.alloc/free` 走原来的 `free_slots` 路径。
- 只要启用，两种模式都会被 `_bind_l2_host_group` 把原生 `HostPoolGroup` 的 `KV`/`MAMBA` entry 指向 mock 池。否则 Two Pools 下 `cache_controller.write()` 仍走原生固定池，预算只影响计时模型——那会让对比失真。

### 5.2 Two Pools（静态切分）

```
kda_region = min(HISIM_L2_MAMBA_SIZE × slot_bytes, total)   # 或 HISIM_L2_SPLIT_KDA_GB
kda_pages  = kda_region // page_bytes
mla_pages  = total_pages − kda_pages
```

两个独立 `PageExtentAllocator`，互不借用；`mla_capacity_bytes` / `kda_capacity_slots` 即各自硬上限。
这就是"同样的总预算、不同的切分策略"的公平基线。

### 5.3 One Pool（动态共享）

- 单个 `PageExtentAllocator`（`total_pages`），MLA 低端 / KDA 高端两端分配。
- 可选 KDA 保底：`HISIM_L2_MAMBA_MIN_SLOTS=N` → `mla_cap_pages = total_pages − ceil(N×slot_bytes/page_bytes)`，即这 N 个 slot 的字节 **MLA 永远拿不到**（KDA 可以拿）。
- 预算用满时谁让位由 `HISIM_L2_BORROW_POLICY` 决定（默认 `self_first`，见 §6）：双方各自 LRU 替换是稳定态；`other_first` 让"最后写入者"把另一方的缓存挤到 L3。

---

## 6. 借用 / 回收规则

| 方向 | 触发 | 动作 | 计数 |
|---|---|---|---|
| MLA ← KDA | MLA 请求需要的连续页不足 | 逐个 `KDAView.evict()`，走既有 placement policy；每次驱逐调 `notify_evicted` → L3 待写队列 | `cross_evict_kda_slots/bytes` |
| KDA ← MLA | KDA 准入需要的连续页不足 | `cache.evict_host(deficit_tokens, FULL)`（仅 `enable_storage`）；MLA host value 被 `group.free` 后由 `MLAView` 归还页 | `cross_evict_mla_tokens` |
| KDA 拒绝 | 上面两步都无法腾出足够连续页 | `backup_from_device` 返回 `False`，hash 不入 `_slots`（`has()==False`），调用方仍写 L3 | `admission_reject_kda` / `alloc_fail_kda` |

实现细节：

- **顺序**：先同视图替换（LRU / placement policy），再跨视图借用；由 `HISIM_L2_BORROW_POLICY` 选择，默认 `self_first`。
  - `self_first`：MLA 先在 MLA 内 LRU，KDA 先在 KDA 内 LRU；只有当本视图已无可驱逐对象、或驱逐后仍凑不出连续页时才向对方借。稳定、不 ping-pong；代价是切分在预热期定形（§8）。
  - `other_first`：每次握手都优先驱逐对方视图，需求驱动、弹性最强；但同时写入少的一方会被逐步挤到 0（§8 的 `mla_heavy` 只用 0.62 GB KV），需要配合 `HISIM_L2_MAMBA_MIN_SLOTS` 之类的水位保护。
- MLA 的驱逐顺序由原生 `_execute_kv_backup` 决定（先 `evict_host` 自己），统一池只在 `alloc` 失败后跨视图借 KDA，因此 MLA 侧天然是 `self_first`。
- 回收量按**缺口**算（`need_pages − largest_free_run`），而不是按整个请求，避免过度驱逐；
- `_reclaim_from_kda` 用"无进展即退出 + 512 次上限"保证终止；`release_kda` 幂等（`dict.pop`），owner 池自己释放过也不会重复归还；
- MLA 回收不新增算法：`cache.evict_host` 就是原生 L2 驱逐入口，victim 仍由 radix tree 的 host LRU 决定；
- Two Pools 下跨视图借用关闭（区域隔离），只做区域内 LRU。


---

## 7. 正确性不变量

1. **L1 不动**：没有修改 L1 驱逐、`req_to_token` 锁、`ack` 生命周期；本设计只影响 host 池的字节来源。
2. **不造假命中**：KDA 命中由 `has(hash)`（物理准入）决定；MLA 命中由 radix tree 的 `host_value` 决定，而 `host_value` 只可能来自成功的 `alloc`。
3. **L3 不旁路**：所有 L2 驱逐仍然进入 `_last_evicted_hashes` / `_kda_l3_pending_writes` → `MockHiCacheStorage.batch_set*`；恢复仍走 `batch_exists` / `batch_get` → `from_l3=True` 的 `backup_from_device`。
4. **字节守恒**：`used_bytes + free_bytes == total_bytes`，`used_bytes == kv_bytes + kda_bytes`（`kv_bytes`/`kda_bytes` 都是**物理页**字节）。
5. **页对齐**：预算向下对齐到整页；区域按页切分；`MLAView` 的 token 索引与页边界同构（`page_tokens` 个 token/页）。
6. **幂等释放**：MLA `free` 对未分配 token 无操作；KDA `free(indices)` 只回收仍在 `_allocated_idx` 的 index；`release_kda` 对未知 hash 返回 `False`。
7. **不重复计价**：KDA host index 分配不计字节；checkpoint 只在准入时计一次；MLA 只在 `alloc` 计一次。
8. **默认零回归**：未启用时所有新增分支都短路（`_unified is None`）。

---

## 8. 碎片实测

脚本：`benchmark/collection/l2_pool_frag_study.py`（结果 `l2_pool_frag_study_results.json`，60 行）。
参数：预算 8 GB、Two Pools 的 KDA 区域 330 slots、6000 次操作；工作负载 `kda_heavy / balanced / mla_heavy`（KDA 概率 0.7/0.4/0.1），MLA 请求 256–8192 token（16–512 页）；压力下先本视图 LRU、再跨视图回收（与 §6 默认一致）。

### 8.1 代表行

| workload | mode | page | alloc | used% | KV GB | KDA slots | 内部碎片% | 外部碎片% | 最大连续空闲 MB |
|---|---|---|---|---|---|---|---|---|---|
| kda_heavy | two_pools | 16 | exact | 99.9 | 3.16 | 330 | 0.00 | 23.81 | 4.4 |
| kda_heavy | two_pools | 16 | pow2 | 99.6 | 3.16 | **204** | **22.88** | 4.85 | 22.7 |
| kda_heavy | two_pools | 64 | exact | 99.8 | 3.16 | 327 | 0.38 | **0.00** | 13.6 |
| kda_heavy | one_pool+self | 16 | exact | 98.2 | 0.06 | 532 | 0.00 | 87.83 | 17.5 |
| kda_heavy | one_pool+other | 16 | exact | 97.5 | 0.00 | 532 | 0.00 | 93.00 | 14.2 |
| balanced | two_pools | 16 | exact | 99.9 | 3.15 | 330 | 0.00 | 11.81 | 10.3 |
| balanced | two_pools | 256 | exact | 99.7 | 3.15 | 327 | 0.38 | 0.00 | 13.3 |
| balanced | one_pool+self | 16 | exact | 96.5 | 0.16 | 516 | 0.00 | 95.05 | 13.9 |
| balanced | one_pool+other | 16 | exact | 95.6 | 0.13 | 513 | 0.00 | 96.18 | 13.5 |
| mla_heavy | two_pools | 16 | exact | 99.9 | 3.16 | 330 | 0.00 | 59.49 | 2.9 |
| mla_heavy | one_pool+self | 16 | exact | 93.6 | 3.21 | 292 | 0.00 | 97.39 | 13.4 |
| mla_heavy | one_pool+other | 16 | exact | 85.1 | 0.62 | 422 | 0.00 | 98.02 | 23.6 |
| mla_heavy | two_pools | 16 | pow2 | 99.6 | 3.16 | 204 | 22.88 | 14.46 | 22.7 |

### 8.2 解读

1. **`pow2` 对 KDA 是灾难性的**：KDA slot = 159 页（5,760 B entry 口径）→ 256 页，单 slot 多付 61%，KDA 容量 330 → 204，内部碎片 22.9%。**这是默认选 `exact` 的直接证据。**（`exact` 下 159 页整除，内部碎片为 0；本机 52,992 B/token 口径是 17.28 → 18 页，约 +4%。）
2. **`exact` 的代价是外部碎片，且随 page 增大而下降**：page=1/16 时 `two_pools` 的最大连续空闲只有 4.4–10 MB，外部碎片占空闲的 12–60%；page=64/256 时 `two_pools` 外部碎片降到 0。原因是 KDA slot 159 页在大页下更容易落到同一个 run。
3. **两端分配（§4.2）是 One Pool 可用的前提**：单 region、纯 first-fit 时 KDA 的 159 页 extent 会把 MLA 的空闲空间切碎，one_pool 占用率只有 85–93%；MLA 低端 / KDA 高端之后就回到 96–98%，与 two_pools 的 99.9% 只差 1–4 个百分点（相遇处的碎片）。
4. **One Pool 的收益是"切分随负载"，不是"容量更大"**：`self_first` 下总占用 93.6–98.2%（two_pools 99.7–99.9%），切分由预热期需求决定（kda_heavy → 532 slots，mla_heavy → 292 slots + 3.21 GB KV）。这比手工设 `HISIM_L2_MAMBA_SIZE` 更省事，但**不会**凭空多出容量。
5. **`other_first` 是需求驱动的弹性模式，但会饿死一方**：`mla_heavy` 下 KDA 每次准入都抢 MLA，KV 只剩 0.62 GB（`self_first` 是 3.21 GB）；反之 `kda_heavy` 下 KV 直接归 0。它适合"确定某一方是冷数据"的场景，并且需要 `HISIM_L2_MAMBA_MIN_SLOTS` 之类的保底。
6. **占用率与碎片是两回事**：one_pool 的 `used%` 96–98%，但 `外部碎片%` 常 >87%——分母是剩余空闲，说明"剩下那点空间（0.1–0.2 GB）已经凑不出大请求"，与总占用率不矛盾。

### 8.3 参数建议

| 场景 | 建议 |
|---|---|
| 通用 | `HISIM_L2_ALLOC_POLICY=exact`、`HISIM_L2_PAGE_TOKENS=16`、`BORROW_POLICY=self_first` |
| MLA 大块写入多、外部碎片敏感 | `HISIM_L2_PAGE_TOKENS=64`（two_pools 外部碎片 0） |
| 负载形状稳定、要最高占用率 | 用 Two Pools + `HISIM_L2_MAMBA_SIZE` 手工切分（99.9%） |
| 负载形状未知 / 想少调一个参数 | One Pool（`self_first`），占用率损失 1–4% |
| 确信某一方是冷数据 | One Pool + `other_first`（+ 保底水位） |
| 想复现 size-class 对照实验 | `HISIM_L2_ALLOC_POLICY=pow2`（仅用于研究） |


---

## 9. 开关与 metrics

### 9.1 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `HISIM_L2_TOTAL_GB` | 未设（= `--hicache-size`） | 设了才启用统一池；Two Pools 静态切分它，One Pool 共享它 |
| `HISIM_L2_ONE_POOL` | `0` | `1` = One Pool 动态共享 |
| `HISIM_L2_PAGE_TOKENS` | `16` | 页大小 = `page_tokens × token_bytes` |
| `HISIM_L2_ALLOC_POLICY` | `exact` | `exact` / `pow2` |
| `HISIM_L2_MAMBA_SIZE` | `330` | Two Pools 的 KDA 区域 slot 数（也用作默认 KDA 索引池上限） |
| `HISIM_L2_SPLIT_KDA_GB` | `0` | 非 0 时直接指定 KDA 区域字节数（覆盖 `MAMBA_SIZE`） |
| `HISIM_L2_MAMBA_MIN_SLOTS` | `0` | One Pool 的 KDA 保底 slot 数（MLA 不可用） |
| `HISIM_L2_KV_EVICT` | `1` | One Pool 下允许 KDA 通过 `evict_host` 回收 MLA |
| `HISIM_L2_BORROW_POLICY` | `self_first` | 预算用满时的让位顺序：`self_first`（先本视图 LRU，再跨视图借）/ `other_first`（先抢对方，需求驱动） |
| `HISIM_L2_KV_SIZE_GB` | `--hicache-size` | 仅未启用统一池时保留给旧实验的 KV 计时容量覆盖 |

### 9.2 metrics（`metrics.json` / `metrics_phaseN.json` 双写，均已并入既有 `metrics` dict）

`l2_mode`、`l2_one_pool`、`l2_total_bytes`、`l2_page_bytes`、`l2_page_tokens`、`l2_alloc_policy`、`l2_total_pages`、`l2_used_pages`、`l2_used_bytes`、`l2_kv_bytes`、`l2_kda_bytes`、`l2_free_bytes`、`l2_largest_free_run_bytes`、`l2_free_run_count`、`l2_internal_frag_bytes`、`l2_external_frag_bytes`、`l2_internal_frag_ratio`、`l2_external_frag_ratio`、`l2_mla_extents`、`l2_mla_live_tokens`、`l2_mla_available_tokens`、`l2_kda_cached_slots`、`l2_kda_logical_bytes`、`l2_kda_available_slots`、`l2_kda_slot_pages`、`l2_kda_floor_bytes`、`l2_mla_capacity_bytes`、`l2_kda_capacity_bytes`、`l2_alloc_fail_mla`、`l2_alloc_fail_kda`、`l2_admission_reject_kda`、`l2_cross_reclaim_calls`、`l2_cross_evict_kda_slots`、`l2_cross_evict_kda_bytes`、`l2_cross_evict_mla_tokens`、`l2_kv_evict_enabled`、`l2_borrow_policy`

健康口径：正常运行时 `l2_used_bytes + l2_free_bytes == l2_total_bytes`、`l2_used_bytes == l2_kv_bytes + l2_kda_bytes`、`l2_internal_frag_bytes` 不随时间单调增长（释放会回冲）。

---

## 10. 实现与验证

### 10.1 文件

| 文件 | 变更 |
|---|---|
| `src/hisim/simulation/l2_dram_pool.py` | 新增：`Extent` / `PageExtentAllocator` / `MLAView` / `KDAView` / `UnifiedDRAMPool` / env 单例 |
| `src/hisim/simulation/sglang/sglang_mock_class.py` | `calc_kv_host_size_per_token`；KV mock 的 `alloc/free/available_size` 走统一池；KDA mock 的准入/驱逐计价、`notify_evicted`、index 释放幂等 |
| `src/hisim/simulation/sglang/sglang_hook.py` | `_setup_l2_dram_pool`、`_bind_l2_host_group`（重指 `HostPoolGroup` + `storage_host_pool`）、两处容量来源、`_init_kda_pools` 注册、metrics |
| `test/test_l2_dram_pool.py` | 11 项单测（分配/合并/外部碎片/pow2/两端分配/token 往返/部分释放/KDA 准入/两池隔离/双向借用/借用顺序/守恒） |
| `benchmark/collection/l2_pool_frag_study.py` | 碎片 / 借用顺序对照实验（60 行结果，输出 JSON + markdown 表） |

### 10.2 已验证

```bash
pytest test/test_l2_dram_pool.py -q          # 11 passed
python benchmark/collection/l2_pool_frag_study.py --ops 6000
```

另有两个临时集成脚本（不入库，可重放）验证适配层：

- mock 池 + 统一池：`alloc/free` 往返、分块释放、KDA 准入/`load_to_device`/`evict`、index 幂等、MLA 借 KDA（`cross_evict_kda_slots>0`）、末态归零；
- hook 适配：`_setup_l2_dram_pool` + `_bind_l2_host_group` 之后 `HostPoolGroup` 的 `KV/MAMBA` entry 确实指向 mock，`group.size/logical_size` 与 mock 一致，`cc.storage_host_pool` 同步；One Pool 下 KDA 先借 MLA（KDA 视图为空时，`cross_evict_mla_tokens>0`）、随后 MLA 借 KDA（`cross_evict_kda_slots>0`）、`self_first` 下第二次 KDA 准入只替换自己的 slot（slot 数不变、MLA 不再被驱逐）；Two Pools 下 4 GB 预算切出 3.12 GB KV + 0.88 GB KDA（57 slots），两侧都填满后 `used ≤ total`、`external_frag=0`。

### 10.3 未做（本机无法完成）

- **服务端 smoke 未跑**：`/tmp/opencode/k3-model`（mock K3-23L）与 `config_k3_0519.json`、预测器 profile CSV 都不在本机（此前会话已丢），`benchmark/collection/k3_onepool_twopools/` 只能生成 trace，不能重建模型。复跑前需恢复这三样，然后：
  1. 不设 `HISIM_L2_*` 跑一格确认零回归；
  2. `HISIM_L2_TOTAL_GB=8`（Two Pools）与 `+HISIM_L2_ONE_POOL=1`（One Pool）各跑一格，检查 `l2_*` 指标、`l2_alloc_fail_*`、`l2_used+free=total`、L3 命中不劣化。

---

## 11. 限制与后续

1. **外部碎片没有治理**：first-fit + 合并 + 两端分配只保证不泄漏、减少相互切碎，不保证紧凑；page=16 时 two_pools 外部碎片仍可达空闲的 ~24%，one_pool 相遇处剩 1–4% 的占用率损失。可选后续：best-fit、页迁移/压缩（超出最小版本）。
2. **不区分视图碎片**：`stats()` 在 Two Pools 下给出两区域之和；若要"谁在浪费"，需按区域拆 `internal/external`。
3. **One Pool 的切分不是公平调度**：`self_first` 让切分在预热期定形，`other_first` 是最后写入者赢；两者都没有权重/水位线，唯一的保护是 `HISIM_L2_MAMBA_MIN_SLOTS`（且只在 MLA 触到 cap 时生效，本轮实测负载下未触顶）。
4. **KDA←MLA 回收依赖 L3**：`enable_storage=False` 时不做 MLA 回收，KDA 准入失败即只留 L3（这是安全的，但会降低 L2 命中）。
5. **两套 mamba 池共同计价**：`_kda_host_pool` 与 `mamba_pool_host` 都计入 KDA 视图，若未来确认二者模拟的是同一份物理 checkpoint，需要去重。
6. **token_bytes 与真实 K3 口径的差异**：本机 mock 池 `size_per_token`（52,992 B）来自 `_real_head_dim × layer_num`，比真实 MLA entry 口径（5,760 B/entry，5 个 full-attn 层）大；预算与计时同源所以自洽，但与 docker 报告的"403,520 entries"不能直接换算。
7. 未引入任何新的 placement / prefetch 算法；`l2_placement_policy` 一行未改。
