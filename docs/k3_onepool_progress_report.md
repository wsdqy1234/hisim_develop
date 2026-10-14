# HiSim + Kimi-K3 One Pool vs Two Pools 容量与缓存收益验证报告

> **文档目的**：总结 HiSim 框架在 Kimi-K3 模型上对 One Pool（UnifiedRadixCache 动态共享）与 Two Pools（固定分区 MockReqToTokenPool + _StrictMambaAllocator）两种 L1 部署模式的全部验证工作，包括背景、代码修改、实验设计、结果分析和优化方向评估。
>
> **前置文档**：`k3_hicache_progress_report.md`（HiCache L2/L3 适配，commit `a811a06`）
>
> **更新历史**：
> - 2026-10-13：初始版本（P4.3-A Eviction 边界验证 + P4.3-B One Pool vs Two Pools 对比 + MLA/KDA 级联淘汰核查，commit `c07ca98`）

---

## 1. 环境信息

| 项目 | 值 |
|------|-----|
| 硬件 | Ascend 910C（8 chip，每 chip 2 die） |
| OS | openEuler 22.03 LTS |
| Docker 容器 | `sglang-kimi-k3-dqy` |
| sglang 版本 | `0.5.18.dev922+ga41da991c` |
| PyTorch | `2.10.0+cpu`（CPU 引擎模式） |
| torch_npu | `2.10.0` |
| 模型 | Kimi-K3-23L（23 层：5 层 Full Attention + 18 层 KDA/Mamba） |
| Git 分支 | `develop_dqy` |
| 最新 commit | `c07ca98` |
| 前置基线 | `a811a06`（k3_hicache_progress_report.md 的最新 commit） |

### 关键环境变量

```bash
export SGLANG_USE_CPU_ENGINE=1
export ASCEND_RT_VISIBLE_DEVICES=2          # NPU 2（避免与其他会话冲突）
export TORCH_TRANSFER_TO_NPU=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HISIM_SIMULATION_MODE=OFFLINE
export HISIM_OUTPUT_DIR=/tmp/hisim/output
export HISIM_SSD_SIM=simpyssd
export HISIM_CONFIG_PATH=.../config_910c_k3.json
export HISIM_DISABLE_PROACTIVE_EVICT=1      # 禁用 proactive eviction（实验控制）

# Two Pools 模式
export HISIM_MAMBA_POOL_SIZE=80             # KDA Slot 容量（默认 80）

# One Pool 模式
# --enable-unified-memory --max-mamba-cache-size 80
```

---

## 2. 两种 L1 部署模式

### 模式定义

| 模式 | sglang 参数 | L1 结构 | KDA 分配器 | MLA 分配器 |
|------|-------------|---------|-----------|-----------|
| **Two Pools** | （无特殊参数） | MockReqToTokenPool + MockTokenToKVPool | _StrictMambaAllocator（固定容量） | MockTokenToKVPool（固定容量） |
| **One Pool** | `--enable-unified-memory` | UnifiedKVPool（KV+Mamba 统一） | MultiEndedAllocator（动态水印线） | MultiEndedAllocator（动态水印线） |

### 物理容量计费

两种模式使用**相同的总物理预算**：

```
Total = KDA_slots × KDA_entry_bytes + MLA_tokens × MLA_entry_bytes
      = 80 × 14,653,440 + 200,000 × 5,760
      = 1,172,275,200 + 1,152,000,000
      = 2,324,275,200 Bytes (≈ 2.16 GB)
```

| 参数 | Two Pools | One Pool |
|------|-----------|----------|
| KDA entry_bytes | 14,653,440 B/slot (18 层 × 814,080) | 14,653,440 B/slot（同） |
| MLA entry_bytes | 5,760 B/token (5 层 × 288 × 2 × 2) | 5,760 B/token（同） |
| KDA 初始容量 | 80 slots（`HISIM_MAMBA_POOL_SIZE`） | 80 slots（`--max-mamba-cache-size 80`） |
| KDA 最大容量 | 80（硬上限） | 158（`max_slots`，可从 80 扩展） |
| MLA 容量 | 200,000 tokens（`--max-total-tokens`） | 200,000 tokens（同） |
| 物理池大小 | mock（分离） | UnifiedKVPool `total_bytes=2,324,275,200` |

### 关键差异

| 维度 | Two Pools | One Pool |
|------|-----------|----------|
| 虚拟 Slot ID | 1-80（1:1 物理映射） | 1-156+（虚拟→物理映射，可增长） |
| 容量管理 | 直接 Eviction（`cache.evict(mamba_num=N)`） | 水位线 Flush（`MultiEndedAllocator._flush`） |
| Eviction 触发 | `_StrictMambaAllocator.alloc()` 返回 None | `MultiEndedAllocator.alloc()` 水位到达 |
| MLA/KDA 耦合 | 独立池，但 Radix Tree 节点级联 | 共享池，水位线联动 + 节点级联 |

---

## 3. 代码修改总览

### 3.1 修改的文件

| 文件 | 说明 | 改动量 |
|------|------|--------|
| `src/hisim/src/hisim/simulation/sglang/sglang_mock_class.py` | MockReqToTokenPool + _StrictMambaAllocator | ~160 行新增/修改 |
| `src/hisim/src/hisim/simulation/sglang/sglang_hook.py` | KDA-TRACK/COW-EXEC-SIM/REPLAY-FALLBACK + _cache 引用 | ~60 行新增/修改 |

### 3.2 修改分类

#### A. _StrictMambaAllocator 观测日志（`sglang_mock_class.py`，commit `46e982c`）

| 修改 | 目的 |
|------|------|
| `MA-INIT` | 记录分配器初始化：capacity、device、free_slots |
| `MA-ALLOC` | 记录每次 alloc：n、slots、remaining、alloc_count |
| `MA-ALLOC-FAIL` | 记录 alloc 失败：n、available、free、alloc_count、free_count |
| `MA-FREE` | 记录每次 free：slots、remaining、alloc_count、free_count |
| `MA-PP-FAIL` | 记录 pp_buf alloc(2) 失败后回滚：rid、pool_idx |
| `MA-PP-EVICT` | 记录 pp_buf alloc(2) 失败触发 eviction：rid |
| `get_stats()` | 返回 capacity、available、alloc_count、free_count、free_slots |

#### B. mock 模式 Mamba 追踪模拟（`sglang_hook.py`，commit `3cf2212`）

**问题**：mock `wrapped_forward_v2` 不运行真实 Mamba 模型，`batch_result_processor` 永远不会设置 `req.mamba_last_track_seqlen`。导致 `prepare_for_caching_req` 返回 `cache_len=None` → 提前返回，donation/eviction 路径不被执行。

**修复**：在 `wrapped_run_batch` 中，`original_run_batch` 调用前，为每个请求设置 `mamba_last_track_seqlen`：

```python
if getattr(_req, 'mamba_last_track_seqlen', None) is None:
    _sl = getattr(_req, 'kv_committed_len', None)
    if _sl is None or _sl == 0:
        _sl = len(getattr(_req, 'input_ids', []))
    if _sl and _sl > 0:
        _req.mamba_last_track_seqlen = _sl
```

仅在 `mamba_last_track_seqlen` 为 None 时设置，不覆盖 `batch_result_processor` 的值。

#### C. free_mamba_cache 实现（`sglang_mock_class.py`，commit `0e2e066` + `549a5ee`）

**问题**：`MockReqToTokenPool.free_mamba_cache` 原为 `pass`（no-op），导致：
1. non-keep pp_buf slot 不被释放 → Slot 泄漏
2. `mamba_pool_idx`（COW dst slot）不被释放 → COW dst 累积泄漏

**修复**：匹配 sglang 原生 `HybridReqToTokenPool.free_mamba_cache` 的行为：

```python
def free_mamba_cache(self, req=None, mamba_ping_pong_track_buffer_to_keep=None, ...):
    # 1. 始终释放 mamba_pool_idx（COW dst 或 alloc-time slot）
    _mpi = getattr(req, 'mamba_pool_idx', None)
    if _mpi is not None:
        _mpi_val = _mpi.item() if hasattr(_mpi, 'item') else int(_mpi)
        if _mpi_val != -1:
            self.mamba_allocator.free(_t.tensor([_mpi_val], dtype=_t.int64))
        req.mamba_pool_idx = None

    # 2. 释放 non-keep pp_buf slot（过滤 -1）
    _pp = getattr(req, 'mamba_ping_pong_track_buffer', None)
    if _pp is not None:
        if mamba_ping_pong_track_buffer_to_keep is not None:
            # 释放 non-keep slot
            ...
        else:
            # 释放所有 pp_buf slot
            ...

    # 3. 清除 pp_buf 属性
    req.mamba_ping_pong_track_buffer = None
    req.mamba_next_track_idx = None
    req.mamba_last_track_idx = None
```

#### D. pp_buf alloc(2) 失败处理（`sglang_mock_class.py`，commit `549a5ee` + `f4d9bec`）

**问题**：`_alloc_v2` 中 `mamba_allocator.alloc(2)` 失败时写入 `[-1,-1]` 并继续执行，导致后续 donation 将 -1 登记为 mamba_value，COW 使用 src=-1 作为源 Slot。

**修复**：触发 eviction + retry + 安全回滚：

```python
if _pp_slots is None:
    _cache = getattr(self, '_cache', None)
    if _cache is not None:
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams
        _cache.evict(EvictParams(num_tokens=0, mamba_num=2))
        _pp_slots = self.mamba_allocator.alloc(2)  # retry
if _pp_slots is not None:
    # 正常分配
else:
    # 回滚所有 req_pool_idx 分配，返回 None（请求被 defer）
    for _rr in reqs:
        if _rr.req_pool_idx in select_index:
            self.free_slots.append(_rr.req_pool_idx)
            _rr.req_pool_idx = None
    return None
```

`_cache` 引用在 `wrapped_get_new_batch_prefill` 中设置：
```python
_rtp = getattr(self, 'req_to_token_pool', None)
if _rtp is not None and getattr(_rtp, '_cache', None) is None:
    _rtp._cache = _tc  # tree_cache
```

#### E. COW src=-1 防御（`sglang_hook.py`，commit `549a5ee`）

**问题**：COW 检查只跳过 `src_index is None`，不跳过 `src_index == -1`。

**修复**：在 `wrapped_run_batch` 的 COW-EXEC-SIM 之前检查：

```python
if _batch_cow_src is not None and len(_batch_cow_src) > 0:
    for _ci in range(len(_batch_cow_src)):
        _csv = int(_batch_cow_src[_ci].item()) if ... else int(_batch_cow_src[_ci])
        if _csv == -1:
            logger.warning("[COW-INVALID] batch_idx=%d src_slot=-1 — clearing COW", _ci)
            batch.mamba_cow_src_indices = None
            batch.mamba_cow_dst_indices = None
            for _req in batch.reqs:
                _req.mamba_cow_src_index = None
                _req.mamba_needs_clear = False
            break
```

#### F. REPLAY-FALLBACK 修复（`sglang_hook.py`，commit `c07ca98`）

**问题**：REPLAY-FALLBACK 代码（来自 commit `71aa2b1`）在 `wrapped_run_batch` 的 `else` 分支（无 COW 时）对**所有批次**检查 `prefix_len > 0 and mamba_pool_idx is not None`。Decode 批次请求已有 Active Mamba State（来自 Prefill/COW），不需要 Replay，但被错误计入 `replay_tokens = prefix_len = 2304`。

**虚假统计**：`replay_tokens = 34,560 = 15 × 2304`（15 个 Reuse session 的 Decode 批次）

**修复**：增加 `forward_mode.is_extend()` 检查：

```python
_fm = getattr(batch, 'forward_mode', None)
_is_extend = hasattr(_fm, 'is_extend') and _fm.is_extend()
if not _is_extend:
    pass  # Decode batch — skip REPLAY-FALLBACK
else:
    # Prefill/Extend — 统计 replay
```

修复后 `REPLAY-FALLBACK = 0`，COW-EXEC-SIM 的 `replay=0` 是真实值。

#### G. KDA entry_bytes 对齐（`sglang_mock_class.py`，commit `3592a35`）

**问题**：mock 的 `get_kv_size_bytes()` 返回 15,151,104（硬编码，过时），sglang `MambaSubPoolSpec.entry_bytes()` 返回 14,653,440。

**修复**：`return 14653440  # 18 layers × 814,080 B/layer`

---

## 4. MLA/KDA 级联淘汰机制

### 4.1 UnifiedTreeNode 结构

```python
class UnifiedTreeNode:
    component_data: list[ComponentData]  # 按 ComponentType 索引
    # ComponentType.FULL  = 0 → MLA KV cache (value = token indices)
    # ComponentType.MAMBA = 2 → KDA checkpoint (value = mamba slot ID)
```

**同一节点**同时保存 FULL 和 MAMBA 的 ComponentData，共享节点生命周期。

### 4.2 级联淘汰调用链

**L1-only 模式（无 HiCache，`node.backuped=False`）**：

```
cache.evict(EvictParams(num_tokens=N))           ← MLA 驱逐请求
  → _evict_components({FULL: N, MAMBA: 0, ...})
    → evict_device_start(FULL)
    → while node_id = _evict_device_next_node(FULL):
        _evict_device_leaf(node_id)
          → node.backuped == False
          → _delete_unbacked_device_leaf(node)
            → _release_all_component_layers(node)     ← 释放所有组件
              → for comp in [FULL, MAMBA]:
                  [RELEASE-ALL] node=X full_before=True mamba_before=True
                  _evict_component_and_detach_lru(node, FULL)   ← MLA freed
                  _evict_component_and_detach_lru(node, MAMBA)  ← KDA freed
            → _remove_leaf_from_parent(node)           ← 节点从树中删除
            → _iteratively_delete_tombstone_leaf()     ← 级联向上删除空父节点
```

**生产模式（有 HiCache，`node.backuped=True`）**：

```
_demote(node)
  → _evict_component_and_detach_lru(node, FULL, DEVICE)   ← 淘汰 FULL 设备值
  → _cascade_evict(node, FULL, ...)                        ← 级联淘汰
    → for comp in components:
        if comp.eviction_priority(is_leaf) <= FULL.priority:  ← MAMBA(0) ≤ FULL(2)
          [CASCADE-EVICT] node=X trigger=FULL evicting=MAMBA
          _evict_component_and_detach_lru(node, MAMBA)     ← KDA 设备值 = None
```

### 4.3 eviction_priority 规则

| 节点类型 | FULL | MAMBA | 级联方向 |
|----------|------|-------|----------|
| 叶子节点 | 0 | 0 | 任一组件被驱逐 → `_release_all_component_layers` → 节点删除 |
| 内部节点 | 2 | 0 | FULL 驱逐 → `_cascade_evict` → MAMBA 也被驱逐；反向不成立 |

### 4.4 One Pool Flush 语义

| 操作 | 含义 | 是否影响 Radix Tree |
|------|------|-------------------|
| `flush` (215-315 次) | 移动 MLA/Mamba 水位线边界 | 间接：MLA 缩小 → MLA 驱逐 → 节点删除 |
| `flush_peer` (0-3 次) | 从 MLA 子池驱逐页给 Mamba | 间接：同上 |
| `compact` (0 次) | Mamba 物理页碎片整理 | 不影响 |
| `alloc_fail` (0-3 次) | Mamba 分配器耗尽物理页 | 直接：触发 `_alloc_mamba_slot` eviction |

**关键区分**：`flush` 是水位线调整（不是缓存淘汰），但水位线调整导致 MLA 缩小，**间接触发** MLA 缓存淘汰 → 级联到 KDA。

### 4.5 容器临时补丁

| 补丁 | 文件 | 行 | 目的 |
|------|------|-----|------|
| RELEASE-ALL 日志 | `unified_tree_core.py` | 1331 | 追踪节点删除时 FULL+MAMBA 同时释放 |
| CASCADE-EVICT 日志 | `unified_tree_core.py` | 1571 | 追踪 `_cascade_evict` 级联淘汰 |
| COW-ENTRY/SRC/SET/SKIP | `mamba_component.py` | 198-228 | 追踪 COW 流程（来自早期会话） |

这些补丁在容器内直接修改 sglang 代码，**重启容器后丢失**。

---

## 5. 实验设计与结果

### 5.1 P4.3-A：Eviction 边界验证

**配置**：`HISIM_MAMBA_POOL_SIZE=11`，Two Pools，9 Build + 3 Reuse

#### 修复的 P0 问题

| 问题 | 根因 | 修复 | Commit |
|------|------|------|--------|
| pp_buf alloc(2) 失败写入 [-1,-1] | `_alloc_v2` 无 eviction 重试 | 触发 `cache.evict()` → retry → 失败则回滚 | `549a5ee` |
| `free_mamba_cache` 是 no-op | 不释放 mamba_pool_idx | 匹配真实实现：始终释放 mamba_pool_idx | `0e2e066`/`549a5ee` |
| COW 使用 src=-1 | COW 检查只跳过 None | 防御性检查：src=-1 时清除 COW | `549a5ee` |
| mock 模式 `mamba_last_track_seqlen` 未设置 | forward 不运行 Mamba 模型 | 手动设置模拟 Mamba 追踪 | `3cf2212` |

#### 验收结果（6 个场景全部通过）

| 场景 | 结果 |
|------|------|
| 容量充足 Build→Reuse | ✅ COW src=1 dst=3 |
| pp_buf alloc(2) 失败→Eviction→Retry | ✅ MA-PP-EVICT 触发，9/9 Build 完成 |
| 无可淘汰资源时安全回滚 | ✅ 请求被 defer，不写入 [-1,-1] |
| 淘汰后重访 COW 正确跳过 | ✅ COW-SKIP reason=src_index_is_None |
| 未淘汰状态 COW 正常命中 | ✅ COW-SET cow_src=valid |
| 无 Slot -1、无 COW 非法源、无泄漏 | ✅ |

### 5.2 P4.3-B：One Pool vs Two Pools 对比

#### 第一轮：容量不匹配（发现）

| 发现 | 影响 |
|------|------|
| One Pool Mamba 物理 max_slots=58,988 | 737 倍于 Two Pools 80 slots |
| UnifiedKVPool total_bytes=806 GB | One Pool 使用全系统内存 |
| 修正：`--max-mamba-cache-size 80` | One Pool max_slots=158，初始 80 |

#### 第二轮：统一 80 Slot 初始容量

**KDA-heavy（独立前缀，100 Build + 6 Reuse）**

| 配置 | KDA Slots | MLA Tokens | COW Hit | COW Miss | KDA Evict/Flush |
|------|-----------|------------|---------|----------|-----------------|
| Two Pools KDA=60 | 60 | 250,880 | 2 | 4 | 25 evict |
| Two Pools KDA=80 | 80 | 200,000 | 3 | 3 | 19 evict |
| Two Pools KDA=100 | 100 | 149,120 | 3 | 3 | 0 evict (MLA cascade) |
| One Pool (80→156) | 80→156 | 200,000 | 3 | 3 | 215 flush |

**MLA-heavy（共享 2000-token 前缀，100 Build + 6 Reuse）**

| 配置 | KDA Slots | MLA Tokens | COW Hit | COW Miss | KDA Evict/Flush |
|------|-----------|------------|---------|----------|-----------------|
| Two Pools KDA=80 | 80 | 200,000 | 104 | 2 | 17 evict |
| One Pool (80→156) | 80→156 | 200,000 | 104 | 2 | 315 flush, 0 fail |

### 5.3 级联淘汰证据（100 Session）

#### RELEASE-ALL 统计

| 统计 | 值 |
|------|-----|
| RELEASE-ALL 事件总数 | 56（28 节点 × 2 组件） |
| `mamba_before=True` 的节点 | 28 个（node 18-45） |
| `mamba_before=False` 的节点 | 0 个 |
| CASCADE-EVICT 事件 | 0（L1-only 模式无 `_demote`） |

**28 个节点在被删除时全部同时持有 FULL=True 和 MAMBA=True**。无一例独立 KDA 淘汰。

#### Session→Node→COW Miss 对应关系

| Session | Build pool_idx | Radix Node | RELEASE-ALL | Reuse COW |
|---------|---------------|------------|-------------|-----------|
| 0 | 2 | ~18 | mamba_before=True | **Miss** |
| 1 | 6 | ~19 | mamba_before=True | **Miss** |
| 2 | 10 | ~20 | mamba_before=True | **Miss** |
| 3 | 14 | ~21 | mamba_before=True | **Miss** |
| 4 | 18 | ~22 | mamba_before=True | **Miss** |
| 5 | 22 | ~23 | mamba_before=True | **Miss** |
| 49 | 98 | ~66+ | survived | **Hit** (src=56) |
| 50 | 102 | ~68+ | survived | **Hit** (src=61) |
| 99 | 198 | ~118+ | survived | **Hit** (src=84) |

### 5.4 核心结论

#### 结论 1：One Pool 动态扩展未产生有效收益

| 负载类型 | Two Pools KDA=80 | One Pool (80) | 差异 |
|----------|-----------------|----------------|------|
| KDA-heavy | 3 Hit / 3 Miss | 3 Hit / 3 Miss | **无差异** |
| MLA-heavy | 104 Hit / 2 Miss | 104 Hit / 2 Miss | **无差异** |

One Pool 的 flush（215-315 次）只移动水位线，不增加有效 Checkpoint 驻留。级联淘汰将 MLA 缩小导致的节点删除传递到 KDA，抵消了扩展收益。

#### 结论 2：级联淘汰是根本限制

```
One Pool 扩展 Mamba → flush 移动水位线 → MLA 子池缩小
→ MLA 驱逐 → _release_all_component_layers → FULL + MAMBA 同时释放
→ KDA Checkpoint 丢失 → COW Miss
```

#### 结论 3：共享前缀是缓存收益的关键驱动

| 负载 | COW Hit | 原因 |
|------|---------|------|
| KDA-heavy（独立前缀） | 3/6 | 每个 Session 独立节点，LRU 淘汰早期节点 |
| MLA-heavy（共享前缀） | 104/106 | 共享前缀节点高频访问，LRU 优先保留 |

#### 结论 4：固定分区调优可匹配 One Pool

| KDA Slots | COW Hit | COW Miss | 瓶颈 |
|-----------|---------|----------|------|
| 60 | 2 | 4 | KDA 容量不足 |
| 80 | 3 | 3 | KDA/MLA 接近平衡（最优） |
| 100 | 3 | 3 | MLA 容量不足（级联淘汰） |

KDA=80 是最优固定分区。KDA=100 不优于 KDA=80，因为 MLA 容量减少导致级联淘汰。

---

## 6. REPLAY-FALLBACK 修复详解

### 问题

REPLAY-FALLBACK 代码（commit `71aa2b1`）在 `wrapped_run_batch` 的 `else` 分支（无 COW 时）对所有批次检查 `prefix_len > 0 and mamba_pool_idx is not None`。

每个 Reuse session（output_length=16）产生 1 个 prefill batch + 15 个 decode batch：
- Prefill batch：COW 已调度 → `else` 分支不执行 → REPLAY-FALLBACK 不触发
- Decode batch：`mamba_cow_src_indices = None` → `else` 分支执行 → `_prefix_len=2304, _mpi is not None` → `replay_tokens = 2304`

虚假统计：`replay_tokens = 34,560 = 15 × 2304`（15 个 Reuse session 的 decode 批次）

### 修复

增加 `batch.forward_mode.is_extend()` 检查，仅 Prefill/Extend 批次统计 REPLAY-FALLBACK。

### 验证

修复后 `REPLAY-FALLBACK = 0`。COW-EXEC-SIM 的 `replay=0` 是真实的 Replay 工作量——COW Hit 时无需 Replay。

### 三条路径验证

| 路径 | REPLAY-FALLBACK | 实际 Replay |
|------|----------------|-------------|
| Checkpoint Hit (COW) | 不触发（COW 已调度） | 0 |
| Checkpoint Miss (Evicted) | 触发（仅 prefill） | prefix_len（需 full replay） |
| Decode（已有 Active State） | 不触发（修复后跳过） | 0 |

---

## 7. 优化方向评估

### 方案 A：MLA+KDA 综合价值感知 Eviction

**现状**：Eviction 按 component 独立 LRU 驱逐。FULL 驱逐级联到 MAMBA，但决策时不考虑 MAMBA 价值。

**设计**：修改 eviction walk 的 LRU 排序，使同时持有 FULL+MAMBA 的节点获得更高保留优先级。

| 维度 | 评估 |
|------|------|
| 实现复杂度 | 中 — 修改 `evictable_device_leaves` 的 LRU 排序逻辑 |
| 正确性风险 | 中 — 可能导致 MLA 饥饿 |
| 理论收益 | 高 — KDA-heavy 负载下可减少 COW Miss |
| 适用场景 | KDA-heavy（独立前缀） |

### 方案 B：MLA 和 KDA 独立驻留

**现状**：`_release_all_component_layers` 同时释放 FULL + MAMBA。节点从树中删除。

**设计 B2（推荐先探索）**：当 FULL 被驱逐但 MAMBA 仍有价值时，不删除节点。仅 tombstone FULL（`cd.value = None`），保留 MAMBA。

| 维度 | 评估 |
|------|------|
| 实现复杂度 | 高 — 修改删除逻辑 + prefix match 语义 |
| 正确性风险 | 高 — 树碎片化、孤立 MAMBA 节点 |
| 理论收益 | 非常高 — 完全解耦 MLA/KDA 淘汰 |
| 关键挑战 | prefix match 需区分"节点存在但 FULL=None"和"节点不存在"；COW 检查需独立于 prefix match |

---

## 8. 容量守恒验证

### 物理预算守恒

```
Total = MLA_tokens × MLA_entry_bytes + KDA_slots × KDA_entry_bytes
      = 200,000 × 5,760 + 80 × 14,653,440
      = 1,152,000,000 + 1,172,275,200
      = 2,324,275,200 Bytes
```

| 维度 | Two Pools | One Pool (80) |
|------|-----------|---------------|
| 总物理预算 | 2,324,275,200 B | 2,324,275,200 B（同） |
| 稳态 KDA 使用 | 78/80 Slot | ~92/158 物理页（v2p_valid=0.58） |
| 稳态 MLA 使用 | ≤200K tokens | ≤200K tokens（被 Mamba 挤占） |
| 预算守恒 | ✅ 严格 | ✅ 水位线保证 |
| 超额计费 | 无 | 无 |

### Virtual vs Physical Slot

| 维度 | Two Pools | One Pool |
|------|-----------|----------|
| 虚拟 Slot 范围 | 1-80（1:1 物理映射） | 1-156（虚拟→物理映射） |
| 物理 Slot 范围 | 1-80 | 1-158（max_slots） |
| 虚拟 ID 复用 | 无（free 后回到 free_slots） | 无（虚拟 ID 只增不减） |
| 物理 Slot 回收 | 直接 free | 内部 flush 回收 |
| V2P 有效率 | N/A（1:1） | 0.58-0.72 |

---

## 9. 未覆盖项与后续计划

| 项 | 状态 | 原因 |
|------|------|------|
| 200+ Session 区分实验 | 待执行 | 100 Session 已证明两种模式行为一致 |
| HiCache 模式级联验证 | 待执行 | 需启用 `--enable-hierarchical-cache`，验证 `_demote` → `_cascade_evict` 路径 |
| 方案 A/B2 实现 | 设计阶段 | 需评估正确性风险后再实现 |
| TTFT/TPOT | 不可用 | Mock sampler 无 decode batch |
| Mock Sampler 修复 | 后续阶段 | 需修改 `wrapped_forward`/`wrapped_sample` |
| donate_count 准确性 | 后续修复 | 旧逻辑（=hit_count），不影响 COW 正确性 |
| Mixed Churn 负载 | 待执行 | 50% 共享前缀 + 50% 独立前缀 |

---

## 10. 本轮全部 Commit

| Commit | 修改内容 |
|--------|----------|
| `46e982c` | _StrictMambaAllocator alloc/free 日志 + get_stats() |
| `3cf2212` | mock 模式设置 mamba_last_track_seqlen |
| `0e2e066` | 实现 free_mamba_cache（原 no-op） |
| `549a5ee` | P0 修复：pp_buf eviction+retry+rollback, free_mamba_cache 始终释放 mamba_pool_idx, COW src=-1 防御, 设置 _cache 引用 |
| `f4d9bec` | 修正 EvictParams 导入路径 |
| `3592a35` | 统一 KDA entry_bytes 为 14,653,440 |
| `c07ca98` | REPLAY-FALLBACK 仅 prefill 批次统计 |

### 前置 Commit（来自 HiCache 报告 a811a06 及更早）

| Commit | 修改内容 |
|--------|----------|
| `e3316d7` | One Pool Unified Memory 部署模式支持（Phase 1） |
| `29c390d` | proactive evict guard + max_running_requests override |
| `03c11a8` | 跳过 NPU profiler in OFFLINE mode |
| `c9af26c` | bypass Triton attention backend for unified memory |
| `16c736b` | _flush counter for lazy compaction tracking |
| `113b3a6` | One Pool vs Two Pools advantage comparison report |
| `2508169` | K3 层编号 0-indexed + slot_bytes 修正 |
| `46e7578` | MAMBA slot_hash 用 node_id |
| `5abe1cd` | MAMBA value 遍历 transfer |
| `237f879` | KDA 时延模型修复 |
| `a811a06` | KDA-REPLAY 统计 + 清理诊断 |
