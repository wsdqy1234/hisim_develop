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
- **两端分配（two-ended）**：MLA 从空闲 run 的**头部**（低地址）取，KDA 从**尾部**（高地址）取。两种对象形状（KDA 恒为 `slot_pages`，MLA 16–512 页）于是从两端向中间生长，只在相遇处互相干扰，避免长寿命 KDA extent 把 MLA 的空闲空间切碎（§8 实测：同样负载下 one_pool 占用率因此从 85–93% 提升到 96–98%；与 MemCache best-fit / 双端阈值的对拍见 §8.4；三类及以上 block 大小时的结构选择见 §8.5）。
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


### 8.4 分配策略对拍：两端 first-fit vs MemCache best-fit

§4.2 的两端分配是"按对象类别切分地址空间"，MemCache 的 `MmcBlobAllocator::Alloc`
是"在全部空闲 run 里选尺寸最接近的块"（best-fit）。两者不是同一层的取舍，可以组合，
所以需要分别回答：**要不要换掉两端？要不要在端内换 best-fit？**

脚本：`benchmark/collection/l2_pool_alloc_policy_study.py`，结果
`l2_pool_alloc_policy_results.json`（96 行聚合 + 480 行原始数据）。

方法（开环，避免闭环 harness 的伪差异）：分配/释放序列由指数到达 + 指数寿命生成，
与策略无关，**不做淘汰换空间**——放不下的请求直接丢弃，丢弃就是碎片代价。
每个 case 跑 5 个 seed 取均值（闭环"边分配边淘汰"的写法里，淘汰谁依赖策略本身，
各策略的活跃集不同，占用率不可横向比较；早期闭环数据已废弃）。
预算/几何取两组：

* `server_1656KB`：服务端 mock 口径，page 27.1 MB，3132 页（85 GB），KDA slot = 1 页；
  池里只有 ~20–30 个对象。
* `real_5760B`：真实 entry 口径（5,760 B/token），page 92 KB，86,805 页（8 GB），
  KDA slot = 159 页，池里 ~500–700 个活跃 extent。

负载 ρ=0.95（offered bytes / 池容量），MLA:KDA 字节比 70/50/30 三档；
`frag MLA%` = 空闲足够但没有连续块而失败的 MLA 请求占比；`big-blocked%` =
采样时刻"空闲 ≥512 页但最大空闲 run <512 页"（即整批写会被挡）的比例。

| geometry | workload | policy | used% | frag MLA% | frag KDA% | big-blocked% | runs |
|---|---|---|---|---|---|---|---|
| server | mla_heavy | two_ended_ff | 72.9 | 8.26 | 0.00 | 50.4 | 25 |
| server | mla_heavy | best_fit | 73.5 | 7.73 | 0.00 | 48.2 | 23 |
| server | balanced | two_ended_ff | 77.6 | 8.13 | 0.00 | **45.6** | 30 |
| server | balanced | best_fit | 78.3 | 9.06 | 0.00 | 49.8 | 29 |
| server | kda_heavy | two_ended_ff | **83.4** | **8.54** | 0.00 | **31.6** | 34 |
| server | kda_heavy | best_fit | 82.1 | 8.75 | 0.00 | 46.1 | 31 |
| server | 任一 | ideal_compact | +1.8~7.1 | 0 | 0 | 0.0 | 0 |
| real | mla_heavy | two_ended_ff | 90.5 | 1.28 | **0.11** | 20.6 | **66** |
| real | mla_heavy | best_fit | **91.6** | **1.01** | 0.24 | **15.0** | 103 |
| real | balanced | two_ended_ff | 91.4 | 1.49 | **0.06** | 24.2 | **57** |
| real | balanced | best_fit | 91.4 | 1.31 | 0.41 | 18.0 | 120 |
| real | kda_heavy | two_ended_ff | **92.0** | 1.47 | **0.08** | 22.6 | **46** |
| real | kda_heavy | best_fit | 91.1 | **1.23** | 0.31 | **16.4** | 123 |
| real | 任一 | ideal_compact | +1.5~3.4 | 0 | 0 | 0.0 | 0 |

解读：

1. **部署几何（server）下，换 best-fit 不划算**：差异 0.6–1.3 点占用、<1 点 MLA
   碎片，多数落在 seed 噪声内（`big-blocked%` 的 seed 标准差 4–13 点，池里只有 20–30
   个对象）。逐 seed 看，KDA-heavy 档两端的占用率 5/5 领先（83.0–84.0 vs 81.4–82.9），
   整批被挡 4/5 更低；balanced 档 3/5 更低。**没有可复现的 best-fit 优势。**
2. **best-fit 只在"两类尺寸重叠"的真实口径下有稳定收益**：`real_5760B` 里 KDA slot
   =159 页，正好卡在 MLA 的 128/256 页之间，两类同场竞争。best-fit 的"保大洞"把整批
   可放置率抬高约 5 个点（20.6/24.2/22.6 → 15.0/18.0/16.4，6 个 case 里 5 个同向，
   只有 KDA-heavy@0.99 反向），代价是 ≤1.6 点占用、**空闲 run 数翻倍**
   （46–66 → 103–123）、KDA 自身碎片 0.06–0.11% → 0.24–0.41%（best-fit 会把 159 页的
   KDA 塞进 MLA 中段的洞）。
3. **端内 best-fit（MemCache 文档方案 D）没有稳定收益**：`two_ended_bf` 在 real 几何
   接近 best-fit，但在 server 几何 KDA-heavy 档反而更差（53.8% vs 46.1% vs 两端 31.6%）——
   它一边保留分端、一边又把两个端内部打散，等于丢掉两端唯一的紧凑性收益。
4. **MemCache 的双端阈值不可直接移植**：它按字节阈值（默认 1 MiB）分端，
   "≤1 MiB → 头，>1 MiB → 尾"。我们最小的请求就是 16 页 = 1.47 MB（真实口径）/
   434 MB（mock 口径），KDA slot 14.65 MB，**全部 > 1 MiB** → 照搬会全部走去尾端，
   退化成单端 best-fit。阈值扫描 @16/@64/@159 之间最多差 6–10 个点，也印证其文档
   §7.1 "阈值位置敏感"的结论。我们的类别在调用点已知（`MLAView` / `KDAView`），
   不需要这个参数。
5. **最大差距不在策略，在粒度**：`ideal_compact`（零外部碎片上界）比任何策略好
   1.5–7.1 点占用，且"整批可放置"是 0% vs 30–55%（server 口径）。和我们服务端观测一致
   （mock 口径 KDA slot 只占 1 页却浪费 46%，MLA 一批 13.9 GB），下一步应该先做
   "分批写 + 更细的 KDA slot 粒度"，再谈 allocator 换代。

结论：**保留两端 first-fit 作为默认**（无参数、KDA 碎片最低、run 数最少、
部署几何下逐 seed 不落后），不切 MemCache 的全局 best-fit。端内 best-fit 暂不实现；
若将来切到 `real_5760B` 这类"两类尺寸重叠"的口径，再在 `PageExtentAllocator` 上
加一个默认关闭的 `policy="best_fit"` 开关即可（改动局限在 `_pick` 一处）。


### 8.5 三种及以上 block 大小：两端不够用之后怎么办

两端之所以成立，是因为地址空间只有两个端点可以"锚定"一个同质段：MLA 贴低端、KDA 贴高端，
各自向中间生长，只在相遇处互相干扰。这条性质有个硬上限——**两个端最多维持两个同质段**
（MemCache `one_pool_multi_size_class_design.md` §2.4 称之为"同质连续段不变量"）。
一旦同一份预算里出现第三类对象（sglang 的 tri-pool 形状：full-attention KV + sliding-window
KV + mamba/SSM state），第三类没有"端"可站，必须显式选一种结构。

**先确认是不是真的三类。** 不要按模型名猜，按 `hf_config` 判：

* sglang 0.5.19 只有 `mambaish_config is not None and is_hybrid_swa and not is_dsv4`
  才走 tri-pool（`kv_cache_configurator.py:415-440` → `init_unified_mamba_swa_pools`），
  也就是"既有 SWA 层、又有线性/SSM 层"的模型（注释里叫 Inkling-class）。
  只有 MLA/MHA + 滑窗、没有 SSM 的，是 2 池不是 3 池。
* 上游对 tri-pool 还有一条限制：**PD 分离直接报错**（"no whole-envelope transfer scheme
  for the SWA sub-pool"）。我们本地没有 GLM-5.3-Flash 的 config，无法确认它是否落在该分支；
  落地前先按 `hf_config`（`sliding_window` / `layer_types` + mamba/linear 层）判一次，
  再决定要不要动。

上游的做法（sglang 0.5.19，`unified_memory_pool.py` / `multi_ended_allocator.py`）：
`UnifiedKVPool` 支持 N 个子池，`SubPoolSpec.grow_direction ∈ {"up","down","float"}`，
链序 `[up end, floats..., down end]`；tri-pool 是 `[mamba(up) | swa(float) | full(down)]`，
注释写明取舍：**mamba/conv 放最低字节、full KV 放最高、SWA 浮在中间；两端从不搬迁，
SWA 的窗口受限 span 最便宜搬**。也就是说第三类被放进中段，靠按需搬迁
（`FloatMultiEndedAllocator.make_room` / `compact_holes`）换弹性。这套机制的前提是
v2p 表（`MultiEndedAllocator.virtual_to_physical`）——搬的只是表项加一次页拷贝；
我们的 L2 没有这一层（`Extent.start_page` 直接进 view / L3），**不能照抄**。
上游还有一道启动期 `_check_bs1_feasibility_floor`（bs=1 最坏请求必须放得下，
否则运行期 retract 活锁），这是正确性下限，不是运行期预留。

脚本：`benchmark/collection/l2_pool_multiclass_study.py`，结果
`l2_pool_multiclass_results.json`（24 行聚合 + 120 行原始，5 seeds）。方法与 §8.4 同源
（开环 trace、失败即丢弃、不做淘汰换空间），三类字节份额 MLA/SWA/state = 60/15/25，
MLA 16–512 页、state 恒 159 页，第三类按三种尺寸口径扫：`small` 4–16 页、
`mid` 32–128 页、`big` 128–512 页；每类独立指数寿命（300 / 800 / 3000 步），ρ=0.95。
指标：`used%`、512 页 MLA 请求失败率（整批写能不能放）、各类失败率、
`frag`（全局空闲够但没有连续块）、run 数、每次分配的平均搬迁页数。

| 记号 | 结构 |
|---|---|
| `ends2_ff` | 现状的自然延伸：第三类再挂一个 view 到低端 |
| `ends2_spill` | 两端不动，第三类不切段，best-fit 到任意空洞（溢出） |
| `global_bf` | 全池 best-fit（MemCache 现状口径） |
| `chain3` | 三段有序，边界只能推进到邻段的**空闲**边缘页，不搬任何数据 |
| `chain3_move` | 同上，但可把边界上的**整个 extent**搬进该段自己的洞（extent 必须连续，只能整对象搬） |
| `chain3_cascade` | 同上，且可把邻段整体往外推（sglang `make_room` 式级联搬迁） |
| `static3` | 三段固定切分（按 60/15/25，等于"比例猜对"的上界） |
| `ideal` | 零外部碎片上界 |

`chain3` 的中段初始为 0 页、靠借；`static3` 是另一个极端（比例已知）。下表为
`real_5760B`、三种第三类尺寸口径下的**区间**（min–max over small/mid/big）：

| policy | used% | 512p MLA 失败% | SWA 失败% | STATE 失败% | frag MLA% | runs | 搬迁页/次分配 |
|---|---|---|---|---|---|---|---|
| `chain3` | 83.3–86.4 | **36.1–53.2** | 0.00–1.15 | 0.00–1.17 | 3.35–5.18 | 63–93 | 0 |
| `chain3_move` | 89.9–90.9 | 10.9–18.6 | 0.00–1.74 | **2.35–4.41** | 0.98–1.67 | 60–86 | 0.67–3.99 |
| `chain3_cascade` | 89.8–90.9 | **7.9–12.1** | 0.01–1.76 | **4.45–10.2** | 0.68–1.09 | 60–89 | 0.82–3.95 |
| `static3`（比例已知） | 88.5–91.1 | 13.4–14.4 | 0.31–**6.09** | 2.18–3.33 | 1.30–1.44 | 40–71 | 0 |
| `ends2_ff` | 89.7–90.9 | 11.3–17.0 | 0.00–3.67 | 0.00–0.02 | 0.96–1.49 | 65–130 | 0 |
| `ends2_spill` | 89.8–91.0 | 10.6–16.6 | 0.00–3.78 | 0.00–0.14 | 0.90–1.46 | **63–89** | 0 |
| `global_bf` | 90.7–91.7 | 7.6–12.8 | 0.00–2.77 | 0.20–0.56 | 0.70–1.21 | 95–154 | 0 |
| `ideal` | 92.3–93.7 | 1.9–2.7 | 0.11–1.24 | 0.63–1.54 | 0 | 0 | 0 |

解读：

1. **专用中段 + 不许搬迁是最差的**：512 页整批请求失败 36–53%，`used` 掉 4–7 点。
   原因不是"切分比例猜错"，而是结构性的——中段夹在两个端之间，端类无法跨过它互相借用，
   而我们的池没有"flush 对端"手段（§8.4 已说明）。这和 MemCache 文档 E1（硬分段 42.8）
   的结论一致，但更严重：那里的失败有 force-evict 兜底，我们直接丢弃。
2. **搬整个 extent 不贵，但搬不来容量**：`chain3_move` 把 used 拉到 89.9–90.9
   （与溢出打平）、512 页失败降到 10.9–18.6%（也与溢出打平），代价只有 0.67–3.99 页/次分配
   （真实口径 92 KB/页 → 60–370 KB/次分配的 host 拷贝）。**它没有换来任何优势**，
   反而把长寿命端类饿到 `STATE` 失败 2.35–4.41%：中段只会在端类的空闲边缘上"单向棘轮"——
   端类让出的前沿，中段不会主动还回去。
3. **级联搬迁（`make_room` 式）能把 512 页失败压到 7.9–12.1%（全场最好），但饿死端类更严重**：
   `STATE` 失败 4.45–10.2%。中段可以一路把端类的前沿往外推，端类却推不回来——
   这正是上游注释"ends never relocate"要防的事；上游靠"端类永不被搬 + 中段是窗口有界的
   SWA"把损失限制在 1 个类，代价是它必须假设三类之间的比例长期稳定。
4. **固定三段只在比例猜对时能看**：`static3` 用了真实字节比 60/15/25，仍会把第三类
   饿到 0.3–6.1%、把 STATE 饿到 2.2–3.3%；比例一旦偏离（真实负载必然偏离）只会更差。
   不要用硬分区。
5. **全局 best-fit 容量最好**（512p 7.6–12.8%、used 90.7–91.7），但 run 数 95–154，
   是两端方案的 1.5–2 倍；且 §8.4 已测它在"两类尺寸重叠"时会把 KDA 塞进 MLA 中段的洞
   （KDA 自身碎片 0.06–0.11% → 0.24–0.41%）。它适合"只有一种尺寸/几乎没有大请求"的场景，
   不适合我们。
6. **溢出几乎免费**：`ends2_spill` 与 `ends2_ff` 在容量上打平（used 89.8–91.0，
   512p 10.6–16.6%），但 run 数从 65–130 降到 63–89，且第三类拿到全局空洞的 best-fit，
   不再和 MLA 抢低端。它同时保住了两端语义（KDA 碎片 0.00–0.14%）。
7. **差距仍在粒度**：`ideal` 比最好的实用策略还好 1.3–3.4 点 used、512 页失败
   7.6–12.8% → 1.9–2.7%。第三类的出现不改变 §8.4 结论 5：先做分批写和更细的
   KDA/state 粒度，再谈 allocator 换代。

结论（三类及以上）：

* **默认仍然是两个端，第三类及以后全部当"溢出类"**（`ends2_spill`），不新增端、不切中段。
  落地面很小：给第三个 `*View` 一个 `prefer_end=None` 分支，`_pick` 走 best-fit；
  仍放在默认关闭的开关后（例如 `HISIM_L2_ALLOC_POLICY=spill` 或独立的 class 列表）。
* **谁上端、谁溢出，按"单对象页数 + 寿命"排，不按语义排**：最大的两个类占两端，
  最小、寿命最短/最灵活（SWA 这类窗口有界的）当溢出类。上游的规则是"最便宜的搬动对象
  放 float"；我们不能搬，所以规则换成"最便宜的放置对象当溢出"。
* **若第三类又大又长寿**（单对象页数 ≥ 中位数的 8 倍、寿命 ≥ 端类），它必须占一端，
  把原来的 KDA 降级为溢出类。判据用运行期统计（`p50/p99 extent pages`、平均驻留时间），
  不要写死模型名。
* **不做的事**：不加第三个端/固定中段（不搬迁时 512 页失败 36–53%）；不为三类引入搬迁
  （搬家本身只要 0.7–4 页/次分配，但它换不来容量、还会让端类单向棘轮饿死 2.4–10.2%）；
  不把全局 best-fit 设为默认（run 翻倍、端语义丢失，保留为已有开关）。
* **正确性下限要有**：启动时检查"bs=1 最坏请求放得下"（对应上游
  `_check_bs1_feasibility_floor`），否则第三类可能被饿到不可用；运行期用软下限
  （`HISIM_L2_*_MIN_PAGES`，默认 0）而不是硬切分。
* **N>3 同一规则**：地址只有两个端，第三个及以后都溢出。若溢出类多到互相干扰，
  先做"减档"（把尺寸接近的类归并，MemCache 方案 I′ 的 4 档/octave），而不是继续加段。

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
| `test/test_l2_dram_pool.py` | 13 项单测（分配/合并/外部碎片/pow2/两端分配/token 往返/部分释放/KDA 准入/两池隔离/双向借用/借用顺序/守恒/mock mamba host index 生命周期） |
| `benchmark/collection/l2_pool_frag_study.py` | 碎片 / 借用顺序对照实验（60 行结果，输出 JSON + markdown 表） |

### 10.2 已验证

单测与碎片实验：

```bash
pytest test/test_l2_dram_pool.py -q          # 13 passed
python benchmark/collection/l2_pool_frag_study.py --ops 6000
```

另有两个临时集成脚本（不入库，可重放）验证适配层：

- mock 池 + 统一池：`alloc/free` 往返、分块释放、KDA 准入/`load_to_device`/`evict`、index 幂等、MLA 借 KDA（`cross_evict_kda_slots>0`）、末态归零；
- hook 适配：`_setup_l2_dram_pool` + `_bind_l2_host_group` 之后 `HostPoolGroup` 的 `KV/MAMBA` entry 确实指向 mock，`group.size/logical_size` 与 mock 一致，`cc.storage_host_pool` 同步；One Pool 下 KDA 先借 MLA（KDA 视图为空时，`cross_evict_mla_tokens>0`）、随后 MLA 借 KDA（`cross_evict_kda_slots>0`）、`self_first` 下第二次 KDA 准入只替换自己的 slot（slot 数不变、MLA 不再被驱逐）；Two Pools 下 4 GB 预算切出 3.12 GB KV + 0.88 GB KDA（57 slots），两侧都填满后 `used ≤ total`、`external_frag=0`。

#### 服务端 smoke（mock K3-23L / 910C / sglang 0.5.19）

命令：`benchmark/collection/k3_onepool_twopools/run_exp.sh twopools 80 200000 8192 8192 /tmp/opencode/k3_p52_mla_heavy.jsonl 11 <label>`；
预算 85 GB ≈ baseline 的 `--hicache-size 80`（mock 池口径 47,177 tokens）。

| cell | `HISIM_L2_*` | 完成 | MLA prefix hits（L1 / L2） | KDA write→L3 | e2e (ms) |
|---|---|---|---|---|---|
| baseline | 不设 | 11/11 | 294,912（294,912 / 0） | 35 → 35 | 22,819.65 |
| two_pools | `TOTAL_GB=85` | 11/11 | 294,912（294,912 / 0） | 6 → 6 | 22,817.36 |
| one_pool | `TOTAL_GB=85 ONE_POOL=1` | 11/11 | 294,912（294,912 / 0） | 7 → 7 | 22,817.44 |

基线在最终代码上逐位复现（同 trace 两次运行完全相同：命中数、KDA 计数、e2e 到 1e-5 ms），零回归。

池状态（dump 时刻；总预算 84,977,123,328 B = 3,132 页 × 27,131,904 B，page_tokens=16）：

| 指标 | two_pools | one_pool |
|---|---|---|
| MLA 区域 | 80.15 GB（2,954 页） | 共享全部 84.98 GB |
| KDA 区域 | 4.83 GB（178 页，1 页/slot） | 共享（KDA floor = 0） |
| MLA 常驻 tokens | 41,216 | 49,408（**+19.9%**） |
| MLA 可再分配 tokens | 6,048 | 592 |
| KDA cached slots | 6（0.163 GB） | 7（0.190 GB） |
| used / free | 70.05 / 14.92 GB | 83.97 / 1.00 GB（98.8% 占用） |
| internal frag | 74.9 MB（0.107%，全部来自 KDA 页内取整） | 87.3 MB（0.104%） |
| external frag | 0 | 0 |
| alloc_fail / admission_reject | 0 / 0 | 0 / 0 |

三条不变量全部成立：`used+free=total`、`used=kv+kda`、`MLA live+avail` 与页数一致
（Two Pools 47,264 = 2,954 × 16；One Pool 49,408 + 592 + 7 个 KDA 页 = 3,132 页）。

KDA 密集 trace（`k3_p51_kda_heavy.jsonl`，102 请求，`TOTAL_GB=85` two_pools）：
102/102，MLA hits 6,144（L1 4,096 + **L2 2,048**），KDA write 46 → L3 46，L3 hit 2，donate 96，
KDA 常驻 46 slots（物理 1.248 GB / 逻辑 0.674 GB），e2e 418.0 ms。

### 10.3 服务端验证中发现并修复的问题

1. **预算口径**：服务端 mock 池 `size_per_token = 1,695,744 B`（1656 KB/token），因此
   `HISIM_L2_TOTAL_GB=8` 只切出 1,856 个 MLA token，小于一个写批次（8,192 token）。
   原生 `HybridCacheController.write()` 先 `mem_pool_host.alloc(len)` 再入队，分配失败直接
   `return None`，于是 L2 写入与搭车的 KDA 备份都不发生——表现为 `l2_used_bytes=0`、
   `kda_write_calls=0`、`l2_kda_free_slots` 缓慢下降（逻辑 index 已分配但从未提交）。
   **这不是分配器错误，而是预算小于写粒度的退化形态**；预算应至少容纳一个写批次。
2. **KDA host index 泄漏**（已修）：mock 的 `start_writing` 同步消费写队列，原生 ack 路径不再
   运行，`resolve_host_transfers` 为 MAMBA 分配的 host index 永远收不回来。P5.1 实测 96 次
   donate 泄漏 96 个（`kda_l2_free_slots` 178→36），再跑 80 余个 checkpoint 就会耗尽并静默
   停止 KDA 备份。修法：绑定统一池后，host index 退化为纯逻辑 id（round-robin 发放，
   容量由 `KDAView`/`admit_kda` 裁决），`available_size()`/`get_stats()['free_slots']`
   改为回报共享池的可用 slot 数；legacy 路径的有限 free list 行为一字未改。修复后
   `kda_l2_free_slots=132 = 178 − 46 cached`，命中/时延逐位不变。
3. **基线与统一池的容量来源不同**：基线不重指 `HostPoolGroup`，`Cache state` 里的
   `L2_avail=1,680,784` 来自**原生** host pool（44.53 GB，26,492 B/token）；统一池路径把
   group 的 `KV/MAMBA` entry 换成预算 mock，容量按 1656 KB/token 计算。两者不能直接比
   token 数，对齐时应比字节预算（85 GB ↔ `--hicache-size 80` 在 mock 口径下同为 ~47k tokens）。

### 10.4 未做

- **长时间压测**：本次最长 102 请求；KDA host index 泄漏已修，但未验证万级 checkpoint 下
  L3 write-through 与统一池回收的长期稳定性。
- **KV 满导致 KDA 备份停摆**：KDA 备份搭在 KV 写操作上，当 MLA 侧连一个批次都放不下时，
  `write()` 返回 None，KDA checkpoint 也就不会再进 L2/L3（P5.2 里 35→6）。这是原生
  `write()` 的语义，未改动；若要解耦需单独给 MAMBA 建写路径（超出最小版本）。

---

## 11. 限制与后续

1. **外部碎片没有治理**：first-fit + 合并 + 两端分配只保证不泄漏、减少相互切碎，不保证紧凑；page=16 时 two_pools 外部碎片仍可达空闲的 ~24%，one_pool 相遇处剩 1–4% 的占用率损失。§8.4 对 best-fit 做过对拍：在真实 5,760 B/token 口径下整批可放置率高 ~5 点，但 run 数翻倍、KDA 碎片上升，部署口径下无稳定收益，故未采用；页迁移/压缩仍超出最小版本。
2. **不区分视图碎片**：`stats()` 在 Two Pools 下给出两区域之和；若要"谁在浪费"，需按区域拆 `internal/external`。
3. **One Pool 的切分不是公平调度**：`self_first` 让切分在预热期定形，`other_first` 是最后写入者赢；两者都没有权重/水位线，唯一的保护是 `HISIM_L2_MAMBA_MIN_SLOTS`（且只在 MLA 触到 cap 时生效，本轮实测负载下未触顶）。
4. **KDA←MLA 回收依赖 L3**：`enable_storage=False` 时不做 MLA 回收，KDA 准入失败即只留 L3（这是安全的，但会降低 L2 命中）。
5. **两套 mamba 池共同计价**：`_kda_host_pool` 与 `mamba_pool_host` 都计入 KDA 视图，若未来确认二者模拟的是同一份物理 checkpoint，需要去重。
6. **token_bytes 与真实 K3 口径的差异**：服务端 mock 池实测 `size_per_token = 1,695,744 B`
   （1656 KB/token）= `_real_head_dim(576) × _real_head_num(64) × layer_num(23) × itemsize(2)`，
   而原生 MLA host pool 是 26,492 B/token、真实 entry 口径是 5,760 B/token（576 × 5 个 full-attn
   层 × 2）。预算与计时同源所以自洽，但与 docker 报告的"403,520 entries"不能直接换算；
   一个 85 GB 的预算在 mock 口径下只有 47,264 个 token。
7. **Two Pools 的"切分成本"不是 external frag**：两个区域各自 first-fit，`stats()` 里的
   `external_frag` 是两区域之和（各自内部都只有一段连续空闲，所以是 0），而
   `largest_free_run_bytes` 取的是两区域最大值。区域间不可挪用体现为
   `l2_mla_available_tokens` 被区域边界卡住（P5.2：two_pools 剩 6,048 tokens 却放不下
   一个 8,192 的批次），不要读成碎片。
8. 未引入任何新的 placement / prefetch 算法；`l2_placement_policy` 一行未改。
9. **三类及以上 block 大小时两端不够用**：地址只有两个端点，最多维持两个同质段。
   §8.5 对拍了八种结构，默认应保持「两个端 + 第三类溢出（best-fit 任意空洞）」：
   不加第三个端/固定中段（512 页整批失败 36–53%），不为三类引入搬迁（1–9 页/次分配，
   仍不如溢出），也不把全局 best-fit 设为默认（run 数翻倍）。是否真的有三类取决于
   `hf_config`（`mambaish` + `is_hybrid_swa`），落地前先判型；tri-pool 在 sglang 里
   目前还不支持 PD 分离。
