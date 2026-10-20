# K3 Cache Architecture Audit — Phase 0 代码审查报告

> 审查基线：HiSim `develop_dqy_evict_reuse` 分支（HEAD=`1747b4c`）+ sglang 0.5.19（容器 `lfz-sglang`）
> 审查方法：基于实际源码（非文档），逐方法审查关键调用链，所有结论引用具体行号
> 审查范围：HiSim 侧 `sglang_hook.py`(3359行)/`sglang_mock_class.py`(2143行)/`launch_server.py`(890行)/`k3_predictor.py`(132行)；sglang 上游 `mamba_component.py`(948行)/`unified_tree_core.py`(2655行)/`unified_radix_cache.py`(3120行)/`hybrid_cache_controller.py`(915行)/`multi_ended_allocator.py`(3995行)/`unified_memory_pool.py`(2026行)
>
> **修订记录**：
> - v1（初始）：基于子代理并行审查产出
> - v2（补充）：根据审查意见 1-7 修正 F5 结论、补充路径状态限制审查、容量核算验证、指标定义修订。标记每个结论为 `[静态审查]` 或 `[运行验证]`

---

## 0. 执行摘要：关键发现

### 0.1 重大架构问题（影响基线正确性）

| # | 严重度 | 问题 | 位置 | 影响 |
|---|---|---|---|---|
| F1 | 🔴 高 | **KDA 8 阶段流水线是死代码**：`handle_kda_backup_operation`/`handle_kda_prefetch_operation`/`submit_kda_l3_write_timing` 定义但从未被调用 | `sglang_hook.py:L864,918,966` | KDA L2/L3 backup/restore 完全由 `_mock_start_writing`/`_mock_start_loading` 内联处理，与"8阶段"设计文档不一致 |
| F2 | 🔴 高 | **两套并行 KDA pool 实现**：`C_HiCacheController._kda_pool`（`MockMambaStatePool`，默认70 slots）从未参与模拟；活跃路径用 `MockMambaStatePoolHost`（`mh`，默认330 slots） | `sglang_hook.py:L828` vs `L1757` | 容量配置混淆，两 pool 默认值不同 |
| F3 | 🔴 高 | **metrics.json 从死 pool 读取**：`kda_active_slots`/`kda_alloc_count`/`kda_evict_count`/`kda_free_slots` 读自 `_kda_pool`（从未 alloc），永远为 0 | `sglang_hook.py:L3087-3093` | KDA L1 容量观测完全失效 |
| F4 | 🔴 高 | **MLA L3 hit 污染 KDA 计数器**：`_mock_start_loading` 中 MLA L3 命中用 `inc_hicache_kda_l3_load_dur` 记录延迟 | `sglang_hook.py:L1627` | KDA L3 load 统计被 MLA 延迟污染 |
| F5 | 🟡 中 | ~~**预测器不含 KDA Replay 成本**~~ **[v2 修正]**：KDA Replay 成本已被 Predictor **隐含**——KDA Retreat → `prefix_indices` 缩短 → `extend_range.length` 增大 → `FakeRequest.input_length` 增大 → Predictor 用更大值做插值。**不是漏算**，但无法区分 MLA-only prefill 与 KDA Replay prefill，per-token 速率存在精度偏差 | `k3_predictor.py:L94-101`; `sglang_hook.py:L2763-2767`; `schedule_batch.py:L1468-1484` | [静态审查] |

### 0.2 配置/语义不一致（影响实验复现）

| # | 严重度 | 问题 | 位置 |
|---|---|---|---|
| F6 | 🟡 中 | `SGLANG_MAMBA_TRACK_INTERVAL` 环境变量在 sglang 0.5.19 不存在；实际是 ServerArg `mamba_track_interval`（默认256），HiSim 代码读取此 env var | `sglang_hook.py:L2582` |
| F7 | 🟡 中 | `node.evicted` 属性语义：原生只判断 `parent is not None and FULL.value is None`，**不含** `host_value is not None`；"device evict + host 仍有" = `evicted and backuped` | `unified_tree_core.py:L151-157` |
| F8 | 🟡 中 | `slot_hash` 构造不一致：活跃路径 `f"kda_{nid}_{i}"`，死代码 `md5(f"kda_{req_id}_{token_offset}")` | `sglang_hook.py:L1490` vs `L904-906` |
| F9 | 🟡 中 | `MAMBA_SLOTS_PER_REQ=5` 与实际不符：只在死代码用；活跃 `_alloc_v3` 每 req 用 3 slot（1 active + 2 pp_buf） | `sglang_hook.py:L585` vs `sglang_mock_class.py:L718,409` |
| F10 | 🟡 中 | `slot_bytes=14653440` 多处硬编码重复 | `sglang_hook.py:L1546/1665/1757/1870`、`launch_server.py:L232` |
| F11 | 🟡 中 | `_kda_diag['cached_checkpoint_slots']` 永远为 0：init 时无此 key，L3078 写 `.get(...,0)` | `sglang_hook.py:L2450,3078` |
| F12 | 🟡 中 | `replay_fallback_count`/`no_cow_count` 在 `_kda_diag` 累计但未写入 metrics.json | `sglang_hook.py:L3076-3084` |
| F13 | 🟡 中 | COW 路径 `replay_tokens` 硬编码为 0：`_rt = max(0, _prefix_len - _prefix_len)` | `sglang_hook.py:L2457` |

### 0.3 Proactive Eviction 行为变更

| # | 严重度 | 问题 | 位置 |
|---|---|---|---|
| F14 | 🟠 中-高 | **proactive eviction 改变原生调度行为**（非纯硬件 mock）：每轮把 `evictable_size()` 全降 L2 | `sglang_hook.py:L1910-1932` |
| F15 | 🟠 中-高 | `wrapped_check_hicache_events` 节流：原生每轮跑，HiSim 每 10 轮才调原生 `check_hicache_events`（影响 L2/L3 backup/restore 时机） | `sglang_hook.py:L1933-1934` |
| F16 | 🟢 低 | proactive eviction 与 L2/L3 backup 节奏不同步：前者每轮，后者每 10 轮 | `sglang_hook.py:L1914-1932 vs L1933` |

### 0.4 Phase 0 补充审查结论 [v2 新增]

基于审查意见 1-7 的深度补充审查，以下结论已由静态源码审查确认：

**审查意见 2 回应 — F5 修正**：`[静态审查]` KDA Replay 成本已被 Predictor 隐含。完整数据流链：
1. KDA Retreat 发生 → MAMBA validator 在深节点失败 → `best_match_node` 停在祖先 X（`unified_tree_core.py:L763-768, L843`）
2. → `device_indices` 长度 = X → `req.prefix_indices` = X（`schedule_batch.py:L1468-1484`）
3. → `extend_range.length = total - X`（比无 Retreat 增大 Y - X）（`schedule_policy.py:L1140-1143`）
4. → `FakeRequest.input_length = req.extend_range.length`（**关键传递点** `sglang_hook.py:L2763-2767`）
5. → `K3ProfilePredictor._predict_prefill` 用更大值做 8 点插值（`k3_predictor.py:L98-99`）

**Replay 的 token 数（Y - X）已隐含在增大的 input_length 中。** F5 应修正为："Predictor 已通过 extend_range.length 隐含 Replay 成本，但无法区分 MLA-only prefill 与 KDA Replay prefill，per-token 速率存在 ~10% 量级精度偏差（方向取决于 prefix 长度差），不是漏算。"

**诊断计数器的真实 bug**（不影响 Predictor，只影响 KDA 诊断日志）：
- `sglang_hook.py:L2457`：`_rt = max(0, _prefix_len - _prefix_len)` 恒为 0（笔误，应为 `max(0, _prefix_len - checkpoint_position)`）
- `sglang_hook.py:L2744`：`_replay = max(0, _prefix_len - _branching)` 符号反转，应为 `max(0, _branching - _prefix_len)`；`else` 分支在 host load_back 已恢复时过度报告

**审查意见 3 回应 — 指标定义修订**：`[静态审查]`
- `mamba_branching_seqlen`：**本次前向需要额外 snapshot 的分叉点 seqlen**（目标 Checkpoint 位置，不代表已存在可恢复状态）。由 `finalize_match_result_in_tree_core` 计算（`mamba_component.py:L174-185`）
- `mamba_last_track_seqlen`：**请求当前已 snapshot 的 mamba 状态对应的序列长度**（checkpoint_grid 对齐），不代表 Radix Tree 的历史可恢复位置
- **实际可恢复 Checkpoint 位置**需根据 `best_match_node.component_data[MAMBA]` 的 `value`（L1 命中）/ `host_value`（L2 命中）/ L3 `batch_exists_mamba` 结果分别记录
- `full_kv_hit_length` **能**正确代表当前树中 MLA 可用的潜在前缀（含 device + host + evicted 但 backuped 的部分，`unified_tree_core.py:L787`）

**审查意见 4 回应 — 路径状态限制**：`[静态审查]`
- `mamba_max_states_per_path`：默认 **-1（无限制）**（`server_args.py:L2595-2602`），HiSim 未覆盖
- `_evict_excess_path_states`：因 cap=-1 完全休眠（`mamba_component.py:L258` 立即 return）
- **fork 点保护只存在于路径超额驱逐**（`mamba_component.py:L295`: `if len(node.children) != 1: continue`）
- **LRU 驱逐（proactive eviction 走的路径）无 fork 保护** — `_evict_device_next_node` 只检查 `evictable_device_leaves` 原子性和锁，不检查子节点数
- 这意味着 proactive eviction 会**无差别驱逐 fork 点 checkpoint**，导致所有子路径在该深度的复用失效

**审查意见 7 回应 — 容量核算**：`[静态审查 + 运行验证]`
- `slot_bytes = 14,653,440` 对 K3-23L（18 KDA 层）**正确**，与 sglang 上游 `MambaSubPoolSpec.entry_bytes()` 完全一致
- 公式：`18 × (13824×2 + 196608×4) = 18 × 814080 = 14,653,440`
  - `conv_dim=13824 = prod((3, 4608))`，来自 `KimiLinearStateShape.create(tp=8, num_heads=96, head_dim=128, conv_kernel=4)`
  - `temporal_dim=196608 = prod((12, 128, 128))`
  - `conv=bf16(itemsize=2), temporal=fp32(itemsize=4)` — K3 config 未设 `mamba_ssm_dtype`，走默认
- **80 slots + 32 running requests 不可行**：native 每请求 3 slots（1 active + 2 pp_buf）→ 需 96 slots；HiSim config 假设 5 slots/req → 需 160 slots
- **建议**：`HISIM_MAMBA_POOL_SIZE=165`（对齐 `config_910c_k3.json` 的 `max_mamba_cache_size=165`），可支持 32 running requests（HiSim 5-slots 模型下余 5 slots 给 radix donation）
- MLA 200K tokens 可支持 32 running requests（前提：平均 context ≤ 6.25K tokens）
- `config_910c_k3.json:43` 的 `max_mamba_cache_size=165` vs mock 默认 `80` 不一致，启动时必须显式 `export HISIM_MAMBA_POOL_SIZE=165`

---

## 1. KDA Checkpoint 的产生与保存

### 1.1 KDA Checkpoint 生成时机（原生 sglang 0.5.19）

**Track 机制**：sglang 在 extend/decode 阶段根据 `mamba_track_interval` 建立 Tracking State。

- **extend 阶段**（`schedule_batch.py:L2851`）：
  ```python
  mamba_track_seqlen_aligned = len(req.prefix_indices) + (req.extend_range.length // checkpoint_grid) * checkpoint_grid
  req.kv.mamba_last_track_seqlen = mamba_track_seqlen_aligned
  ```
  其中 `checkpoint_grid = mamba_checkpoint_grid(page_size) = lcm(mamba_cache_chunk_size(), page_size)`（`runtime_context.py:L1920-1925`），`mamba_cache_chunk_size() = max(FLA_CHUNK_SIZE=64, page_size)`。

- **decode 阶段**（`schedule_batch.py:L3345-3366`）：
  ```python
  mamba_track_interval = mamba_track_grid(self.tree_cache.page_size)  # = lcm(checkpoint_grid, mamba_track_interval arg)
  track_remainders_cpu = self.seq_lens_cpu % mamba_track_interval
  track_mask_cpu = track_remainders_cpu == 0  # remainder==0 时触发 track
  ```

- **物理写入**：`hybrid_linear_attn_backend` 前向中根据 `batch.mamba_track_mask` 把 ssm 状态快照写入 ping-pong track buffer 的 keep slot（`hybrid_linear_attn_backend.py:L428-430`）。

**关键配置**：
- `mamba_track_interval`：ServerArg（`server_args.py:L2641-2646`），默认 256，`NS("exec.mamba")`。**注意：原生 0.5.19 不存在 `SGLANG_MAMBA_TRACK_INTERVAL` 环境变量**（全仓 grep 零命中）。
- 实际 track grid = `lcm(mamba_checkpoint_grid, mamba_track_interval)`（`runtime_context.py:L1928-1932`）。
- 约束：`mamba_track_interval >= speculative_num_draft_tokens`、`% page_size == 0`、`>= mamba_chunk_size`（`mamba_hook.py:L120-122`、`hybrid_linear_attn_backend.py:L884-885`）。

### 1.2 HiSim 中的 Track 模拟

**位置**：`sglang_hook.py:L2572-2616`（`wrapped_run_batch` 内）

HiSim mock 前向（`wrapped_forward_v1/v2`，`L493-518`）只返回空 logits，**不运行真实 Mamba 模型**，因此 `batch_result_processor` 不会设置 `mamba_last_track_seqlen`。HiSim 以 fallback 方式补设：

```python
_track_interval = int(os.environ.get("SGLANG_MAMBA_TRACK_INTERVAL", "256"))  # L2582
for _req in batch.reqs:
    _lts_before = req_mamba_get(_req, 'mamba_last_track_seqlen', None)
    _kcl = getattr(_req, 'kv_committed_len', None)
    _seq_len = _kcl or len(getattr(_req, 'origin_input_ids', []))
    if (_lts_before is None or _lts_before == 0) and _seq_len and _seq_len > 0:
        _track_boundary = (_seq_len // _track_interval) * _track_interval  # L2608
        if _track_boundary > 0:
            req_mamba_set(_req, 'mamba_last_track_seqlen', _track_boundary)  # L2610
```

**关键限制**：
- 只有当 `_lts_before is None or 0` 时才补设（L2603-2604），即只在 sglang 原生路径未设置时介入。
- HiSim 读取 `SGLANG_MAMBA_TRACK_INTERVAL` 环境变量（L2582），但**原生 0.5.19 不识别此 env var**——这是一个 HiSim 专有扩展，需确认是否与原生 `mamba_track_interval` ServerArg 一致。

### 1.3 `mamba_radix_cache_strategy`：extra_buffer vs auto vs no_buffer

**配置定义**（`server_args.py:L2621-2630`）：`choices=["auto", "no_buffer", "extra_buffer", "extra_buffer_lazy"]`，默认 `"auto"`。

**auto 解析**（`arg_groups/overrides.py:L553-594`）：
```python
if view.mamba_radix_cache_strategy == "auto":
    wants_overlap = not view.disable_overlap_schedule
    wants_paging = view.page_size is not None and view.page_size > 1
    if (wants_overlap or wants_paging) and supports_mamba_cache_extra_buffer(view, model_arch):
        declared["mamba_radix_cache_strategy"] = "extra_buffer"
    else:
        declared["mamba_radix_cache_strategy"] = "no_buffer"
        declared["disable_overlap_schedule"] = True
```

**派生 boolean**（`runtime_context.py:L1672-1692`）：
- `enable_mamba_extra_buffer = strategy in {"extra_buffer", "extra_buffer_lazy"} and not disable_radix_cache`
- `enable_mamba_extra_buffer_lazy = strategy == "extra_buffer_lazy" and not disable_radix_cache`

**对 Checkpoint 保存的影响**（`mamba_component.py:L536-606`）：
- **启用 extra_buffer**：unfinished req 用 `mamba_last_track_seqlen` 截断 radix key 并 donate ping-pong slot（L536-537, L571-606）。
- **未启用**（no_buffer）：用 `token_ids_len` 走 copy-from 拷贝路径（L538-552, L596-604）。

**HiSim 强制设置**（`launch_server.py:L849-869`）：K3 被强制设为 `extra_buffer`，因为 KDA checkpoint donation 依赖 ping-pong track buffer。

### 1.4 Track / Donate / COW 三者关系

#### Track — 建立 Tracking State

- **时机**：seq_len 跨过 `mamba_track_interval` 边界时
- **动作**：`hybrid_linear_attn_backend` 把 ssm 状态写入 ping-pong track buffer 的 keep slot
- **结果**：`req.kv.mamba_last_track_seqlen` 被设为 track_boundary（grid 对齐）

#### Donate — 捐赠 Checkpoint 到 Radix Tree

**原生实现**（`memory_pool.py:L1495-1515`）：
```python
def donate_mamba_ping_pong_slot(self, req, new_slot):
    donate_idx = self.get_mamba_ping_pong_keep_idx(req)  # = req.kv.mamba_last_track_idx
    mamba_value_donated = req.kv.mamba_ping_pong_track_buffer[donate_idx].unsqueeze(-1).clone()
    self.set_mamba_ping_pong_slot(req, donate_idx, new_slot[0])  # 新 slot 替换 keep 位置
    return mamba_value_donated
```

**HiSim mock 实现**（`sglang_mock_class.py:L479-487`）：逻辑与原生一致。

- **捐赠什么**：keep idx（`mamba_last_track_idx`）位置的 pp_buf slot 值，clone 为 shape `[1]` 张量
- **返回什么**：被捐赠的 slot 值 `mamba_value_donated`，用于插入 Radix Tree
- **捐赠后归属**：归 **Radix Tree**（捐赠值被 sglang 原生 insert 到 radix node 的 `component_data[MAMBA].value`）
- **替换**：用 `new_slot[0]` 覆盖 keep idx 位置，请求继续 tracking

**调用时机**（`mamba_component.py:L571-606`，`prepare_for_caching_req` unfinished 分支）：
- 启用 extra_buffer 时：分配新 slot → `donate_mamba_ping_pong_slot` → 设 `insert_params.mamba_value = donated`

#### COW — Copy-on-Write 状态复用

**原生触发**（`mamba_component.py:L187-216`，`finalize_match_result_in_cache`）：
```python
src_index = self.tree_core.get_component_device_value(result.best_match_node, MAMBA)  # 从 radix node 取
if src_index is None:
    return result  # fallback: 不做 COW
dst_index = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
# alloc 失败 → inc_lock_ref + evict_for_alloc + 重试
req.kv.mamba_pool_idx = dst_index[0]
req.kv.mamba_cow_src_index = src_index
```

- **src**：`best_match_node.component_data[MAMBA].value`（一个已缓存的 checkpoint slot id）
- **dst**：请求自己新分配的 mamba pool slot
- **COW 失败 fallback**（`src_index is None`）：静默跳过 COW（L196-197），后续由 `prepare_load_back`（L664-684）处理 host-only 情形，或由前向 extend 重新建立状态
- **dst 分配失败 fallback**：`inc_lock_ref + evict_for_alloc + 重试`，重试仍失败则 `assert`（L206-212）

**HiSim mock COW 模拟**（`sglang_hook.py:L2645-2714`）：
- 轻量级，无 NPU kernel
- 防御性检查：若 `src == -1`，清空 COW 并回退到 REPLAY-FALLBACK（L2649-2661）
- COW 执行模拟：累计 `cow_exec_count`/`hit_count`，打印 `[COW-EXEC-SIM]`（L2704）
- **`replay_tokens` 硬编码为 0**（L2697）：`_rt = max(0, _prefix_len - checkpoint_position)` 其中 `checkpoint_position = prefix_len`（L2694），所以恒为 0 — 这假设 COW 完全覆盖 prefix

**REPLAY-FALLBACK 路径**（`sglang_hook.py:L2715-2753`）：
- 当无 COW 且 batch 是 extend 模式时
- `_replay = max(0, _prefix_len - mamba_branching_seqlen)`（L2744）或 `_prefix_len`（L2746，branching 为 None 时）
- Decode batch 跳过（L2724-2726）

### 1.5 Checkpoint 与 Radix Tree Node / Token Position 的映射

**存储位置**：`UnifiedTreeNode.component_data[ComponentType.MAMBA].value`

**value 类型**：**`torch.Tensor`，shape `[1]`，dtype int64，内容是一个 mamba pool slot id**（不是直接的 state tensor）

**写入点**（`mamba_component.py:L218-250`，`commit_insert_component_data`）：
- 新叶节点：`node.component_data[MAMBA].value = params.mamba_value` + 插入 device LRU MRU + 增加 `component_evictable_size_[MAMBA]`
- 已有节点 value 为 None：写入 + 从 host LRU 移到 device LRU
- 已有 value：只 reset LRU MRU，置 `result.mamba_exist = True`

**与 Token Position 的关系**：
- `checkpoint_pos`（语义上对应 `mamba_track_seqlen_aligned`）：donate 出来的 checkpoint 在序列中的位置 = `prefix_len + (extend_len // checkpoint_grid) * checkpoint_grid`
- `mamba_branching_seqlen`：**本次前向需要额外 snapshot 的分叉点 seqlen**（由 `finalize_match_result_in_tree_core` 计算，`mamba_component.py:L174-185`），表示 Full KV 命中超过现有 mamba 边界的、grid 对齐的位置

### 1.6 `mamba_last_track_seqlen` vs `kv_committed_len` vs `mamba_branching_seqlen`

| 字段 | 定义位置 | 设置者 | 语义 |
|---|---|---|---|
| `mamba_last_track_seqlen` | `schedule_batch.py:L873`（KVCache 字段） | `_mamba_radix_cache_v2_req_prepare_for_extend`（L2851）；HiSim fallback（`sglang_hook.py:L2610`） | 请求当前已 snapshot 的 mamba 状态所对应的序列长度（checkpoint_grid 对齐） |
| `kv_committed_len` | `schedule_batch.py:L857` | 前向/overlap 提交点更新 | KV 内容已提交的长度（≤ kv_allocated_len） |
| `mamba_branching_seqlen` | `schedule_batch.py:L1083`（Req 字段） | 由 `match_result.mamba_branching_seqlen` 拷贝（L1475-1484） | 本次前向需要额外 snapshot 的分叉点 seqlen |

层级关系：`mamba_last_track_seqlen`（已有 checkpoint 位置） ≤ `mamba_branching_seqlen`（本次要新建 checkpoint 的位置，若存在） ≤ `kv_committed_len`

### 1.7 可复用 Checkpoint vs 临时运行状态

| 状态类型 | 存储位置 | 可复用？ | 生命周期 |
|---|---|---|---|
| Radix Tree 节点的 `component_data[MAMBA].value` | Radix Tree | ✅ 可复用 | 节点被 evict/delete 时结束 |
| `mamba_ping_pong_track_buffer[keep_idx]` | 请求 KVCache | ❌ 临时 | 请求结束时 donate 或 free |
| `mamba_pool_idx`（active slot） | 请求 KVCache | ❌ 临时 | 请求结束时 free |
| L2 host pool 的 `host_value` | Radix Tree 节点 | ✅ 可复用（L2 命中时 load_back） | host pool LRU evict 时结束 |
| L3 storage 的 `slot_hash` | MockHiCacheStorage | ✅ 可复用（L3 命中时 restore） | 永久（除非显式 clear） |

---

## 2. KDA Checkpoint 的匹配、淘汰与恢复

### 2.1 请求到达后的 KDA State 匹配

**核心方法**：`UnifiedTreeCore.match_prefix`（`unified_tree_core.py:L698-723`）→ `_match_prefix_helper`（L725-810）→ `_match_post_processor`（L812-868）

**遍历逻辑**：
- 从 root 出发，按 `child_key` 逐层匹配
- 维护两个游标：`best_match_node`（所有 validator 通过的最深节点）与 `best_match_device_node`（device validator 通过的最深节点）
- 每个被访问节点调 `_update_best_if_valid(node)`：仅当所有 component 的 validator 都通过才更新 `best_match_node`

**FULL validator**（`tree_component.py:L357-369`）：
```python
lambda node: node.component_data[FULL].value is not None or node.backuped
# match_device_only=True 时只检查 value is not None
```

**MAMBA validator**（`mamba_component.py:L142-153`）：
```python
if match_device_only:
    return lambda node: node.component_data[MAMBA].value is not None
return lambda node: (
    node.component_data[MAMBA].value is not None
    or node.component_data[MAMBA].host_value is not None  # HiCache: host_value 也算命中
)
```

**MLA Prefix Match 和 KDA State Match 如何共同决定有效复用长度**：
- `full_kv_hit_length`：在 `_match_prefix_helper` 中累加（含 device + host + evicted 但仍 backuped 的部分）
- `device_indices`：device-resident KV slot id 拼接，长度 = `best_match_device_value_len`
- `host_hit_length`：初始化为 0（L851），由 HiCache controller 在 `finalize_match_result_in_tree_core` 填充
- 三者关系：`full_kv_hit_length ≥ len(device_indices) + host_hit_length`

**KDA Retreat 机制**：
- 当树深层某节点的 MAMBA checkpoint 已被全层淘汰（`value` 与 `host_value` 都为 None），MAMBA validator 失败
- `_update_best_if_valid` 不更新 `best_match_node`，保留为上一个全部 validator 通过的祖先节点
- 后续 insert 从该祖先处分叉新建节点，新请求需从此处重新计算 mamba 状态
- `finalize_match_result_in_tree_core` 基于 `best_match_node` 计算 `mamba_branching_seqlen`（A.5）：若 Full KV 命中超过 mamba 边界，标记一个分叉点供本次前向补一个 checkpoint

**HiSim 的 `wrapped_match_prefix`**（`sglang_hook.py:L1948-1968`）：
- **不计算** `host_hit_length`，只读取并日志记录（注释 L1946: "Do NOT modify host_hit_length — just log for diagnosis."）
- `host_hit_length` 和 `mamba_host_hit_length` 均由 sglang 原生 `match_prefix` 计算

### 2.2 KDA Pool 容量不足时的淘汰决策

**Two Pools 模式**（`_StrictMambaAllocator`，HiSim mock）：
- `alloc(n)` 返回 None（`sglang_mock_class.py:L577-587`），**不主动 evict**
- 调用方 `_alloc_v3`（L719-726）：打印 `[MA-ALLOC-FAIL]`，调用 `_rollback()` 回滚已分配 slot，返回 None → scheduler 推迟请求
- Eviction 由 sglang scheduler 原生逻辑触发（检测 alloc 失败后调用 radix cache evict）

**One Pool 模式**（`MultiEndedAllocator`，sglang 原生）：
- `alloc(n)`（`multi_ended_allocator.py:L1081-1120`）：检测 `need_size > available_size()` → 调 `_relieve_for_alloc`（L216-243）
- `_relieve_for_alloc` 短缺阶梯：(1) flush peer sub-pool（`m._flush(urgent=True)`）；(2) `_ask_float_for_room`
- **不直接驱逐缓存数据**，只调整 watermark 边界（实际数据搬运由 `move_kv_cache` 做）

**原生 sglang eviction 调用链**（`unified_radix_cache.py`）：
- `evict(EvictParams(num_tokens, mamba_num))` → `_evict`（L610-639）→ `_evict_components`（L696-759）
- `_evict_components` 按 component 遍历：`evict_device_start` → 循环 `evict_device_next_node` → `evict_device_leaf`
- `_evict_request_by_type`（L585-592）拆 EvictParams：`FULL: num_tokens`、`MAMBA: mamba_num`

### 2.3 淘汰粒度

**粒度**：Radix Tree 叶子节点（leaf node），每个 leaf 可持有 FULL + MAMBA 两个 component 的 data。

**L1-only 模式（无 HiCache）** — `_release_all_component_layers`（`unified_tree_core.py:L1405-1426`）：
- 对节点上每个 component 调 `_evict_component_and_detach_lru(target=ALL)`
- 同时从 device/host LRU 和 evictable leaf set 移除
- 后续 `_remove_leaf_from_parent` + `_iteratively_delete_tombstone_leaf` 物理删除节点
- **FULL 和 MAMBA 同时释放**（节点被删除）

**HiCache 模式** — `_demote`（`unified_tree_core.py:L1586-1612`）：
- 只驱逐 DEVICE 层的 FULL（trigger=FullComponent）
- `_cascade_evict` 级联到同节点低优先级组件的 device 层
- 把 aux 组件插入 host LRU
- 节点保留在树中，变为 host-only

### 2.4 级联淘汰（Cascade Eviction）

**方向**：`Full(2) → SWA(1) → MAMBA(0)`（高优先级被驱逐时级联到低优先级）

**`eviction_priority` 规则**（`tree_component.py:L482-507`）：
- 叶子节点：所有组件 priority=0（驱逐任一就级联全部，因为节点将被删除）
- 内部节点：Full=2 > SWA=1 > MAMBA=0
- Mamba 不覆写 `eviction_priority`，继承基类返回 0

**`_cascade_evict`**（`unified_tree_core.py:L1614-1655`）：
- trigger → 所有 `eviction_priority ≤ trigger.eviction_priority` 的组件
- 即 Full 触发级联到 SWA + MAMBA；MAMBA 触发不级联（没有更低优先级）
- **不存在 MAMBA→Full 的反向级联**

### 2.5 L1→L2→L3 的备份、下沉、恢复、删除

#### L1→L2 backup（D2H）

**原生链路**：
1. `write(device_indices, ...)` → `mem_pool_host.alloc` → `write_queue.append(CacheOperation)` → `start_writing()`
2. `start_writing()`（`cache_controller.py:L805-827`）：`submit_device_to_host` → `ack_write_queue.append(HiCacheAck)`
3. `_finish_write_through_ack(ack_id)`（`unified_radix_cache.py:L1420-1435`）：pop `ongoing_write_through` → `tree_core.finish_write_through` → `dec_lock_ref`

**HiSim mock**（`sglang_hook.py:L1444-1584`，`_mock_start_writing`）：
- **绕过 ack 链**，直接执行 D2H + L3 write
- FULL pool backup：`host.backup_from_device_all_layer(...)`（L1464）
- MAMBA sidecar backup：`mh.backup_from_device(_slot_idx, _slot_hash)`（L1491），hash = `f"kda_{_nid}_{_i}"`（L1490）
- **L3 write 直接执行**：`st.set(str(_hash))` + `st.batch_set_mamba(_l3_mamba)`（L1529, L1545）
- 设置 `node.component_data[MAMBA].host_value`（L1570 primary / L1572 fallback 用 device value clone）
- **构造 fake ack** 塞入 `cc.ack_write_queue`（L1576-1581）

#### L2→L1 load-back（H2D）

**原生链路**：
1. `load(host_indices, ...)` → `load_queue.append` → 返回 device_indices（不立即触发）
2. `start_loading()`（`cache_controller.py:L917-954`）：`submit_host_to_device` → `ack_load_queue.append`
3. `loading_check`（`unified_radix_cache.py:L2679-2730`）：轮询 → `finish_load_back`

**HiSim mock**（`sglang_hook.py:L1586-1704`，`_mock_start_loading`）：
- MLA L3 命中检查：`st.exists(str(_hash))`（L1623）→ **⚠️ 用 `inc_hicache_kda_l3_load_dur` 记录 MLA L3 延迟（L1627，应改为 MLA 专用计数器）**
- MAMBA L2 命中检查：`mh.has(_slot_hash)`（L1659）
  - hit → `mh.load_to_device(_slot_hash)`（L1661）→ `inc_hicache_kda_l2_load_dur`
  - miss → `st.batch_exists_mamba([_slot_hash])`（L1663）
    - hit → `inc_hicache_kda_l3_load_dur`（L1667）+ `mh.backup_from_device(-1, _slot_hash, from_l3=True)`（L1668，L3→L2 无 D2H 计时）+ `mh.load_to_device(_slot_hash)`（L1669，L2→L1 H2D 计时）
- **L3→L2→L1 串行**（非并行）

#### L2→L3 overflow

**原生**：`_execute_kv_backup`（`unified_radix_cache.py:L1370-1382`）检测 host_avail < kv_tokens → `evict_host(needed)` → `_finish_write_through_ack` 触发 `write_backup_storage` → `storage_backend.batch_set`

**HiSim mock**：在 `_mock_start_writing` 中**直接触发 L3 write**（L1510-1550），不走原生 ack 链。注释明确（L1418-1420）: "Also directly trigger L3 (SSD) writes/reads to bypass the broken ack processing chain."

#### L3→L2 restore

**HiSim mock**（`_mock_start_loading` 内联）：
- L3 命中 → `mh.backup_from_device(-1, _slot_hash, from_l3=True)`（L1668，L3→L2 restore，无 D2H 计时）
- 然后 `mh.load_to_device(_slot_hash)`（L1669，L2→L1 H2D 计时）

#### 永久删除

**位置**：`_remove_leaf_from_parent`（`unified_tree_core.py:L1699-1708`）+ `_iteratively_delete_tombstone_leaf`（L1740-1756）+ `_iteratively_delete_tombstone_ancestors`（L1758-1812）

- `_remove_leaf_from_parent`：从 parent.children 弹出 key、清 `full_host_duplicates`、`_unregister_node`
- `_iteratively_delete_tombstone_ancestors`：循环删除 **childless + unlocked + Full device 缺失 + Full host 缺失** 的祖先
- 这是 checkpoint 真正从树中消失的唯一路径

### 2.6 HiSim 的主动淘汰（Proactive Eviction）

**位置**：`sglang_hook.py:L1907-1938`（`wrapped_check_hicache_events`）

```python
def wrapped_check_hicache_events(self, *args, **kwargs):
    _cn = getattr(self, '_hicache_cn', 0) + 1
    self._hicache_cn = _cn
    # === Proactive eviction（每次都跑，不受 %10 节流）===
    _disable_proactive_evict = os.environ.get("HISIM_DISABLE_PROACTIVE_EVICT", "")
    if not _disable_proactive_evict:
        _es = self.evictable_size() if hasattr(self, 'evictable_size') else 0
        if _es > 0:
            self.evict(EvictParams(num_tokens=_es, mamba_num=_es))  # 原生 evict
    # === 节流：仅每 10 轮调用原生 check_hicache_events ===
    if _cn % 10 != 0:
        return
    original_check_hicache_events(self, *args, **kwargs)
```

**关键行为**：
- **proactive eviction 每轮都执行**（不受 %10 节流），调用 sglang 原生 `self.evict()`
- **原生 `check_hicache_events` 每 10 轮才调一次**（影响 L2/L3 backup/restore 时机）
- `HISIM_DISABLE_PROACTIVE_EVICT=1` 可关闭 proactive eviction

**对基线的影响**：proactive eviction 是**改变调度/淘汰行为**的 patch（非纯硬件 mock）。基线复现必须明确：
1. **"原生 Baseline"**：`HISIM_DISABLE_PROACTIVE_EVICT=1`（关闭主动淘汰，只保留原生 sglang LRU）
2. **"HiSim 默认"**：`HISIM_DISABLE_PROACTIVE_EVICT` 不设（启用主动淘汰）

### 2.7 Mock vs 原生逻辑对照表

| 方法 / 路径 | 状态 | 位置 |
|---|---|---|
| `wrapped_forward_v1/v2` | 🟡 Mock（空 logits） | `sglang_hook.py:L493-518` |
| `wrapped_run_batch` COW 模拟 | 🟡 Mock（轻量，无 kernel） | `sglang_hook.py:L2645-2714` |
| `wrapped_run_batch` 设 `mamba_last_track_seqlen` | 🟡 Mock fallback | `sglang_hook.py:L2581-2616` |
| `_alloc_v3` `mamba_pool_idx` + `pp_buf` | 🟡 Mock | `sglang_mock_class.py:L718,734` |
| `donate_mamba_ping_pong_slot` | 🟡 Mock | `sglang_mock_class.py:L479-487` |
| `free_mamba_cache` | 🟡 Mock | `sglang_mock_class.py:L494-549` |
| `_StrictMambaAllocator.alloc/free` | 🟡 Mock | `sglang_mock_class.py:L577-601` |
| `_mock_start_writing`（D2H backup） | 🟡 Mock（绕过 ack 链） | `sglang_hook.py:L1444-1584` |
| `_mock_start_loading`（H2D load） | 🟡 Mock（绕过 ack 链） | `sglang_hook.py:L1586-1704` |
| `MockMambaStatePoolHost.backup_from_device/load_to_device` | 🟡 Mock | `sglang_mock_class.py:L1790-1835` |
| `MockHiCacheStorage.batch_set_mamba/batch_exists_mamba` | 🟡 Mock | `sglang_mock_class.py:L2099-2143` |
| `handle_kda_backup_operation` | ❌ **死代码** | `sglang_hook.py:L864` |
| `handle_kda_prefetch_operation` | ❌ **死代码** | `sglang_hook.py:L918` |
| `submit_kda_l3_write_timing` | ❌ **死代码** | `sglang_hook.py:L966` |
| `get_kda_stats` | ❌ **死代码** | `sglang_hook.py:L1004` |
| `MockMambaStatePool`（`cls._kda_pool`） | ❌ **仅被死代码引用** | `sglang_mock_class.py:L1615` |
| `wrapped_match_prefix` | 🟢 原生透传 + 日志 | `sglang_hook.py:L1948-1968` |
| Proactive `self.evict()` | 🟢 原生 sglang | `sglang_hook.py:L1925` |
| `original_check_hicache_events`（含 start_writing/loading 调用） | 🟢 原生（节流 1/10） | `sglang_hook.py:L1936` |
| `MultiEndedAllocator.alloc/free/flush` | 🟢 原生（One Pool 模式） | `multi_ended_allocator.py` |

---

## 3. Two Pools vs One Pool 差异

### 3.1 容量限制对比

| 维度 | Two Pools | One Pool |
|---|---|---|
| 物理存储 | `MHATokenToKVPool` + `MambaPool` 两个独立 buffer | `UnifiedKVPool` 一个 byte buffer，多 sub-pool view |
| ReqToTokenPool | `HybridReqToTokenPool`（持有 `mamba_allocator`） | `UnifiedHybridReqToTokenPool`（mamba sub-allocator 是 `MultiEndedAllocator`） |
| MLA 容量 | `max_total_tokens`（硬上限） | 初始 slot 上限，运行时由 compaction 动态调整 |
| KDA 容量 | `max_mamba_cache_size` 或 `HISIM_MAMBA_POOL_SIZE`（硬上限） | 初始 slot 上限，运行时动态调整 |
| MLA/KDA 容量耦合 | 独立池，树级 cascade 同步释放 | 共享池，水位线联动 + flush_peer 跨子池回收 |
| Eviction 触发 | `alloc` 返回 None → scheduler 调 `evict_for_alloc` | `alloc` 检测短缺 → `_relieve_for_alloc` → flush peer |

### 3.2 Unified Memory 动态分配和碎片整理

**`MultiEndedAllocator` 核心数据结构**（`multi_ended_allocator.py:L246-422`）：
- `virtual_to_physical` / `physical_to_virtual`：双向映射
- `watermark_physical`：水位线物理页指针（grow-up 从低端向上长，grow-down 从高端向下长）
- MLA（grow-down，byte 高端）与 Mamba（grow-up，byte 低端）互为 growth-side 邻居

**`alloc(n)`**（L1081-1120）：
- 检测 `need_size > available_size()` → 调 `_relieve_for_alloc`
- `_relieve_for_alloc`（L216-243）：flush peer sub-pool → `_ask_float_for_room`
- **不直接驱逐缓存数据**，只调整 watermark 边界

**`_flush(urgent)`**（L1787-1858+）：
- 批 compaction pass：把存活 slot 朝对侧 packed，关闭内部 holes
- survivor walk（`_topmost_survivor`，L1696-1750）：two-pointer 把最顶活页搬到下一个 hole

**`flush_peer`**：实际机制是 `_relieve_for_alloc` 调 `short_pool._flush_targets()` 返回 growth-side 邻居，然后 `m._flush(urgent=True)`。**注意：`launch_server.py:L357` 的 `_MEA._flush_peer_for_alloc` 在 0.5.19 不存在此方法名**，diag 计数器静默失效。

**`compact`**：即 `_compact_pending`（L1422），eager 模式下 `free()` 末尾触发，lazy 模式下由 `_flush` 触发。

### 3.3 One Pool HiCache 的 L2 管理

**One Pool HiCache 模式**（`--enable-unified-memory --enable-hierarchical-cache`）：
- L1 是 `UnifiedKVPool`（KV+Mamba 统一）
- L2 仍由 HiSim mock 的 `MockTokenToKVPoolHost`（MLA）+ `MockMambaStatePoolHost`（KDA）分别管理
- 即 **L2 仍分别管理 MLA/KDA**，只有 L1 是统一的

### 3.4 影响淘汰策略公平性的差异

| 差异 | 对淘汰策略公平性的影响 |
|---|---|
| Two Pools 硬隔离 vs One Pool 动态软划分 | One Pool 可通过 flush_peer 动态借用空间，固定分区无法做到 |
| Two Pools cascade 同步释放 vs One Pool flush_peer 跨池回收 | One Pool 的 cascade 机制更复杂，可能影响淘汰决策 |
| `max_mamba_cache_size` 硬上限 vs 软约束 | Two Pools 达上限即 evict；One Pool 可动态扩展 |
| proactive eviction 对两种模式影响不同 | One Pool 有 518K 空闲页时仍触发 proactive evict（见 `k3_hicache_progress_report.md` §6.8） |

---

## 4. Phase 1 指标收集 Gap 分析

### 4.1 当前已收集的指标

**`metrics.json`** 已写入的 KDA 字段（`sglang_hook.py:L3036-3098`）：
- `kda_write_calls` / `kda_write_mamba_xfers`
- `kda_load_calls` / `kda_load_mamba_xfers`
- `kda_l3_writes` / `kda_l3_hits`
- `kda_checkpoint_donate_count` / `kda_checkpoint_commit_count`
- `kda_cached_checkpoint_slots`（⚠️ 永远为 0）
- `kda_checkpoint_hit_count` / `kda_cow_exec_count`
- `kda_checkpoint_position` / `kda_prefix_length`
- `kda_historical_replay_tokens` / `kda_new_tokens`
- `kda_active_slots` / `kda_alloc_count` / `kda_evict_count` / `kda_free_slots`（⚠️ 读自死 pool，永远为 0）
- `kda_l2_cached_slots` / `kda_l2_capacity`

**`iteration.jsonl`** 已写入（`sglang_hook.py:L2972-2981`）：
- `forward_latency`
- `l2_load_latency` / `l2_backup_latency` / `l3_backup_latency`
- `l3_direct_load_latency`（条件写入）
- **缺 KDA 独立延迟**：`kda_l2_load_dur` / `kda_l2_backup_dur` / `kda_l3_backup_dur` / `kda_l3_load_dur` 已计算（L2896-2899）但只折叠进 MLA max，未单独持久化

**`request.jsonl`** 已写入（`RequestStats` dataclass，`types.py:L55-67`）：
- `rid, input_length, output_length, final_reused_tokens, l2_hit_tokens, l3_hit_tokens, prefetch_complete_tokens, gen_token_latencies`
- **无任何 KDA 字段**

### 4.2 Phase 1 要求指标 vs 现状

| 要求指标 | 已收集？ | 采集点 | 缺什么 |
|---|---|---|---|
| `mla_potential_hit_tokens` | ❌ | 无 | match_prefix 前 MLA KV 树中存在的最大可命中前缀（未驱逐前）。当前只有 `final_reused_tokens`（命中后实际复用）—— 是「有效」非「潜在」 |
| `kda_recoverable_position` | ❌ | 无 | KDA checkpoint 在 L1/L2/L3 哪一层、可恢复到哪个 token 位置。`_kda_diag['checkpoint_position']` 只存最后一个 req 的 prefix_len，非 per-req、非分层 |
| `effective_prefix_hit_tokens` | ⚠️ 部分 | `final_reused_tokens` + `l2_hit_tokens` + `l3_hit_tokens` | 缺 KDA 前缀命中维度——当前「effective」只指 MLA KV，未含 KDA checkpoint 的等效前缀 |
| `retreat_gap_tokens` | ❌ | 无 | KDA Retreat 的「回退 gap」。`replay_fallback_count` 只有计数无 token 量；`historical_replay_tokens` ≈ `prefix_len - branching` 是粗估，且全局非 per-req |
| `kda_replay_tokens` | ⚠️ 部分 | `_kda_diag['historical_replay_tokens']` | 全局累加，非 per-req；且 COW 路径 `replay_tokens` 硬编码为 0 |
| `kda_l1/l2/l3_hit` | ⚠️ 部分 | L3: `_kda_stats['l3_hit']` 全局；L2: `load_back_l2` count；L1: `_kda_diag['hit_count']` 全局 | 非_per-req；三层未统一到 per-req |
| `kda_loadback_bytes` | ❌ | 仅有 `load_back_l2`/`load_back_l3` 计数 | 缺 × `MAMBA_SLOT_BYTES`（14,653,440 B）的换算，且无 per-req |
| `mla_loadback_bytes` | ❌ | 有 `l2_hit_tokens`/`l3_hit_tokens` per-req | 未 × `_MLA_ENTRY=5760` 换算为 bytes |
| `ttft` | ✅ | 虚拟时钟计算 | 但预测器不含 KDA Replay 成本，TTFT 在 Retreat 场景低估 |

### 4.3 per-req 维度缺口总览

所有 KDA 相关指标（`kda_replay_tokens`、`kda_l1/l2/l3_hit`、`kda_loadback_bytes`）当前都是**全局累加**（`_kda_stats` / `_kda_checkpoint_diag`），未进 `RequestStats`。Phase 1 若要请求级分析，需：
1. 扩展 `RequestStats` dataclass（`types.py:L55-67`）加 KDA 字段
2. 在 `sglang_hook.py:L2442-2469` 的 per-req 循环里把值写入 `C_SchedulerHook.REQUEST_STATS[req.rid]`
3. 在 `_mock_start_writing`/`_mock_start_loading` 的 per-slot 循环里补 `node_id → rid` 映射以聚合 loadback bytes

---

## 5. 基线配置考虑

### 5.1 Two Pools HiCache 基线配置

基于研究计划要求（TP=8, max_total_tokens=200K, max_mamba_cache_size=80, max_running_requests=32, chunked_prefill_size=16K, KDA Checkpoint Interval=4096），验证与代码兼容性：

| 参数 | 计划值 | 代码默认 | 兼容性 |
|---|---|---|---|
| TP=8 | ✅ | config_910c_k3.json 已设 `tp_size=8` | ✅ |
| max_total_tokens=200K | ✅ | 命令行 `--max-total-tokens 200000` | ✅ |
| max_mamba_cache_size=80 | ✅ | `HISIM_MAMBA_POOL_SIZE=80`（`sglang_mock_class.py:L38` 默认 80） | ✅ |
| max_running_requests=32 | ✅ | 命令行 `--max-running-requests 32` | ✅（需确认 sglang 0.5.19 是否支持此参数名） |
| chunked_prefill_size=16K | ✅ | 命令行 `--chunked-prefill-size 16384` | ✅ |
| KDA Checkpoint Interval=4096 | ⚠️ | 需用 ServerArg `--mamba-track-interval 4096`（非 `SGLANG_MAMBA_TRACK_INTERVAL` env var） | ⚠️ 需验证 HiSim 代码读取路径 |

**需特别注意**：
1. HiSim 代码读取 `SGLANG_MAMBA_TRACK_INTERVAL` 环境变量（`sglang_hook.py:L2582`），但 sglang 0.5.19 原生不识别此 env var。需确认两者是否一致：
   - 如果 sglang scheduler 用 ServerArg `mamba_track_interval`（默认256）设置 `mamba_last_track_seqlen`，HiSim 的 fallback 不会触发（因为 `_lts_before` 不为 None）
   - 如果 HiSim 的 env var 与 ServerArg 不一致，可能导致 track_boundary 计算不一致
2. **建议**：Phase 1 实验中统一使用 ServerArg `--mamba-track-interval 4096`，同时设 `SGLANG_MAMBA_TRACK_INTERVAL=4096` env var 以保证 HiSim fallback 一致。

### 5.2 KDA Slot Bytes 一致性

**硬编码位置**：
- `sglang_hook.py:L1546, L1665, L1757, L1870`：`slot_bytes=14653440`
- `launch_server.py:L232`：`slot_bytes=14653440`
- `sglang_mock_class.py`：`slot_bytes=14653440`（多处）

**计算公式**：`18 layers × (13824×2 + 196608×4) = 14653440 B/slot`（TP=8, conv=bf16 itemsize=2, temporal=fp32 itemsize=4）

**一致性**：所有硬编码位置一致（14653440），但**分散重复**，建议提取为常量。

### 5.3 KDA State 逻辑分配但未进入可复用状态的风险

**已确认的风险场景**：
1. **`_alloc_v3` 分配 `mamba_pool_idx` 后请求崩溃**：slot 已分配但未 donate 到 radix tree → slot 泄漏。HiSim 的 `_rollback()`（`sglang_mock_class.py:L725-726`）应处理此情况。
2. **`donate_mamba_ping_pong_slot` 返回 None**：曾导致 `assert params.mamba_value is not None` 失败（见 `K3_HISIM_ADAPTATION_SUMMARY.md` 问题36）。已修复为返回 `dst_req`。
3. **COW src=-1**：防御性检查已添加（`sglang_hook.py:L2649-2661`），清空 COW 并回退到 REPLAY-FALLBACK。

### 5.4 HiCache 的 KDA L2/L3 Backup/Load-back 完整性

**已知不完整之处**：
1. **KDA 8 阶段流水线是死代码**（F1）：`handle_kda_backup_operation` 等从未被调用
2. **KDA L2/L3 backup/restore 完全由 `_mock_start_writing`/`_mock_start_loading` 内联处理**：绕过原生 ack 链，时序可能与原生行为不一致
3. **MLA L3 hit 污染 KDA 计数器**（F4）：`sglang_hook.py:L1627`
4. **check_hicache_events 节流**（F15）：每 10 轮才执行原生 `check_hicache_events`，影响 KDA L2/L3 backup/restore 时机
5. **slot_hash 构造**：活跃路径用 `f"kda_{nid}_{i}"`（L1490），与死代码的 md5 hash 不一致（F8）

### 5.5 预测器对 KDA Replay 的表达

**当前状态**：`K3ProfilePredictor._predict_prefill`（`k3_predictor.py:L94-101`）只按 `input_length` 查 MLA prefill 表，**无任何 KDA Replay 项**。

**影响**：
- 当发生 KDA Retreat（COW 未命中需 replay 前缀的 KDA 状态）时，replay 计算 ≈ `replay_tokens` 个 token 的额外前向
- `_kda_checkpoint_diag['historical_replay_tokens']`（`sglang_hook.py:L2467`）粗估 `prefix_len - branching`，但**此值未注入预测器**
- 按 prefill 表插值，`replay_tokens` 个 token 的成本约 `interp(PREFILL_TOKS, PREFILL_TIMES, replay_tokens)`
- TTFT 被低估该量级

**建议**：Phase 1 中先记录 `kda_replay_tokens` per-req，待数据积累后再评估是否在预测器中加入 KDA Replay 成本项。

---

## 6. Checkpoint 生命周期状态机

```
                      ┌─────────────────────────────────────────────────────┐
                      │ 1. CREATE (Track)                                    │
                      │   触发：seq_len 跨过 mamba_track_interval 边界       │
                      │   schedule_batch.py:L3357  track_mask=True           │
                      │   schedule_batch.py:L2851  mamba_last_track_seqlen = │
                      │     prefix_len + (extend_len // grid) * grid         │
                      │   HiSim fallback: sglang_hook.py:L2610               │
                      │   hybrid_linear_attn_backend 写入 ping-pong keep slot │
                      └──────────────────────────┬──────────────────────────┘
                                                 │
                                                 ▼
                ┌──────────────────────────────────────────────────────┐
                │ 2. DONATE / CLONE (prepare_for_caching_req)           │
                │   unfinished: donate_mamba_ping_pong_slot              │
                │     memory_pool.py:L1495-1515 (原生)                  │
                │     sglang_mock_class.py:L479-487 (HiSim mock)        │
                │     keep slot → mamba_value (shape [1] int64 slot id) │
                │     新 slot 替换 keep 位置，请求继续 tracking          │
                │   finished: clone keep idx / mamba_pool_idx            │
                │     mamba_component.py:L557-570                        │
                └──────────────────────────┬───────────────────────────┘
                                           │
                                           ▼
              ┌──────────────────────────────────────────────────────┐
              │ 3. COMMIT (commit_insert_component_data)              │
              │   mamba_component.py:L218-250                          │
              │   新叶：node.component_data[MAMBA].value = mamba_value │
              │         + insert_mru(device LRU) + evictable_size += 1 │
              │   旧叶 None→有：写入 + 从 host LRU 移到 device LRU     │
              │   旧叶已有：仅 reset LRU MRU，mamba_exist=True         │
              └──────────────────────────┬───────────────────────────┘
                                         │
              ┌──────────────────────────┴────────────────────────────┐
              │                                                         │
              ▼                                                         ▼
   ┌───────────────────────────┐                    ┌──────────────────────────────┐
   │ 4a. MATCH (match_prefix)  │                    │ 4b. EVICT (evict_for_alloc)   │
   │   create_match_validator: │                    │   mamba_component.py:L493-500 │
   │   MAMBA.value is not None  │                    │   _evict_components 遍历      │
   │     OR host_value not None│                     │   evict_device_start/next/leaf│
   │   validator 失败 → RETREAT│                    │   LRU 游标 get_lru_no_lock    │
   │   best_match 停在祖先     │                    │                               │
   └─────────────┬─────────────┘                    └───────────────┬──────────────┘
                 │                                                   │
                 ▼                                                   ▼
   ┌───────────────────────────┐         ┌──────────────────────────────────────┐
   │ 5a. COW / LOAD_BACK        │         │ 5b. EVICT path 分叉                   │
   │   finalize_match_in_cache: │         │   not backuped (L1-only):             │
   │     src=best_match.MAMBA   │         │     _release_all_component_layers    │
   │     dst=req.mamba_pool_idx │         │       (target=ALL, 释放 device+host) │
   │   HiSim mock: COW-EXEC-SIM │         │     + _remove_leaf_from_parent       │
   │     sglang_hook.py:L2645-  │         │     + _iteratively_delete_tombstone  │
   │     2714                   │         │       → 永久删除                     │
   │   host-only: prepare_load_ │         │   backuped (HiCache):                │
   │     back + H2D            │         │     _demote (target=DEVICE only)     │
   │   HiSim mock: _mock_start_│          │       + cascade to SWA+MAMBA device  │
   │     loading L1586-1704    │         │       + aux 插入 host LRU            │
   │                           │         │       → 节点变 host-only（保留）       │
   └─────────────┬─────────────┘         └──────────────────────────────────────┘
                 │
                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 6. BACKUP STORAGE (HiSim mock 内联)                          │
   │   _mock_start_writing: st.set() + st.batch_set_mamba()       │
   │     sglang_hook.py:L1510-1550                                │
   │   原生 ack 链被绕过 (fake ack)                               │
   └──────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 7. HOST EVICT / PERMANENT DELETE                             │
   │   drive_host_eviction (原生, mamba_component.py:L860-912)    │
   │   host leaf: _evict_host_leaf (atomic, ALL layers)           │
   │     → _remove_leaf_from_parent + _iteratively_delete         │
   │   internal: tombstone host_value + cascade (target=HOST)     │
   │   → 永久删除                                                 │
   └──────────────────────────────────────────────────────────────┘
```

**关键不变式**：
- `node.component_data[MAMBA].value` 要么 None 要么是 shape[1] 的 slot id tensor
- Mamba 是 **leaf-only data**（`redistribute_on_node_split` 把 new_parent 的 MAMBA value 置 None，`mamba_component.py:L310-320`）；split 后 mamba value 留在原 leaf（child）
- Mamba LRU 只 refresh best_match 单节点（`mamba_component.py:L134-136`），不像 Full 会 refresh 整条 matched path
- `node.backuped` = `FULL.host_value is not None`（只看 FULL，不看 MAMBA）
- `node.evicted` = `parent is not None and FULL.value is None`（**不含 host_value 条件**，F7）

---

## 7. 关键代码位置索引

### 7.1 HiSim 侧

| 功能 | 文件 | 行号 |
|---|---|---|
| sgl_kernel 5 层 mock | `launch_server.py` | L36-129 |
| cudart mock | `launch_server.py` | L134-143 |
| `kimi_linear_config` patch | `launch_server.py` | L169-195 |
| `mambaish_config` patch | `launch_server.py` | L197-256 |
| `support_triton=False` patch | `launch_server.py` | L491-500 |
| `zero_pages` CPU fallback | `launch_server.py` | L502-539 |
| `get_available_gpu_memory` patch | `launch_server.py` | L541-574 |
| `ModelConfig.__init__` override | `launch_server.py` | L585-606 |
| `MambaPool` data ops no-op | `launch_server.py` | L616-634 |
| `handle_page_major_kv_layout` 放宽 | `launch_server.py` | L645-678 |
| `handle_unified_memory_pool` 绕过 | `launch_server.py` | L690-726 |
| `HybridCacheController` 线程 no-op | `launch_server.py` | L733-739 |
| K3 强制 `extra_buffer` | `launch_server.py` | L849-869 |
| /health 永返 200 | `launch_server.py` | L813-829 |
| `wrapped_forward_v1/v2` (mock) | `sglang_hook.py` | L493-518 |
| `_init_kda_pools` (死代码) | `sglang_hook.py` | L810-861 |
| `handle_kda_backup_operation` (死代码) | `sglang_hook.py` | L864 |
| `handle_kda_prefetch_operation` (死代码) | `sglang_hook.py` | L918 |
| `submit_kda_l3_write_timing` (死代码) | `sglang_hook.py` | L966 |
| `get_kda_stats` (死代码) | `sglang_hook.py` | L1004 |
| `handle_backup_operation` (MLA, 活跃) | `sglang_hook.py` | L1049 |
| `submit_l3_write_timing` (MLA, 活跃) | `sglang_hook.py` | L1080 |
| `handle_prefetch_operation` (MLA, 活跃) | `sglang_hook.py` | L1165 |
| `wrapped_init_hicache` | `sglang_hook.py` | L1406-1899 |
| `_apply_mock_start_overrides` | `sglang_hook.py` | L1413-1722 |
| `_mock_start_writing` (L1→L2 backup) | `sglang_hook.py` | L1444-1584 |
| `_mock_start_loading` (L2→L1 load) | `sglang_hook.py` | L1586-1704 |
| `wrapped_check_hicache_events` | `sglang_hook.py` | L1907-1938 |
| proactive eviction | `sglang_hook.py` | L1910-1932 |
| `wrapped_match_prefix` (透传+日志) | `sglang_hook.py` | L1948-1968 |
| `wrapped_get_new_batch_prefill` | `sglang_hook.py` | L2397-2469 |
| KDA COW/REPLAY 诊断 | `sglang_hook.py` | L2440-2469 |
| `wrapped_run_batch` | `sglang_hook.py` | L2545-2760 |
| mock `mamba_last_track_seqlen` 设置 | `sglang_hook.py` | L2581-2616 |
| COW-EXEC-SIM | `sglang_hook.py` | L2645-2714 |
| REPLAY-FALLBACK | `sglang_hook.py` | L2715-2753 |
| `wrapped_process_batch_result` | `sglang_hook.py` | L2840-2855 |
| iteration.jsonl 写入 | `sglang_hook.py` | L2972-2981, L3202-3204 |
| metrics.json 写入 | `sglang_hook.py` | L3036-3098, L3199-3200 |
| request.jsonl 写入 | `sglang_hook.py` | L3206-3208 |
| `_StrictMambaAllocator` | `sglang_mock_class.py` | L551-615 |
| `_alloc_v3` | `sglang_mock_class.py` | L663-755 |
| `donate_mamba_ping_pong_slot` | `sglang_mock_class.py` | L479-487 |
| `free_mamba_cache` | `sglang_mock_class.py` | L494-549 |
| `MockMambaStatePoolHost.backup_from_device` | `sglang_mock_class.py` | L1790-1819 |
| `MockMambaStatePoolHost.load_to_device` | `sglang_mock_class.py` | L1821-1835 |
| `MockHiCacheStorage.batch_set_mamba` | `sglang_mock_class.py` | L2099-2143 |
| `K3ProfilePredictor.predict_infer_time` | `k3_predictor.py` | L85-92 |
| `_predict_prefill` | `k3_predictor.py` | L94-101 |
| `_predict_decode` | `k3_predictor.py` | L103-115 |
| `ScheduleBatch` | `base.py` | L18-71 |

### 7.2 sglang 上游（容器内 `/sgl-workspace/sglang/python/sglang/`）

| 功能 | 文件 | 行号 |
|---|---|---|
| `MambaComponent.prepare_for_caching_req` | `srt/mem_cache/unified_cache/components/mamba_component.py` | L529-606 |
| `MambaComponent.commit_insert_component_data` | 同上 | L218-250 |
| `MambaComponent.finalize_match_result_in_cache` (COW) | 同上 | L187-216 |
| `MambaComponent.finalize_match_result_in_tree_core` | 同上 | L155-185 |
| `MambaComponent.create_match_validator` | 同上 | L142-153 |
| `MambaComponent._alloc_mamba_slot` | 同上 | L493-500 |
| `MambaComponent._free_mamba_value` | 同上 | L523-527 |
| `MambaComponent.cleanup_after_caching_req` | 同上 | L615-644 |
| `MambaComponent.prepare_load_back` | 同上 | L664-684 |
| `MambaComponent.drive_host_eviction` | 同上 | L860-912 |
| `UnifiedTreeCore.match_prefix` | `srt/mem_cache/unified_cache/unified_tree_core.py` | L698-723 |
| `_match_prefix_helper` | 同上 | L725-810 |
| `_match_post_processor` | 同上 | L812-868 |
| `node.backuped` property | 同上 | L146-149 |
| `node.evicted` property | 同上 | L151-157 |
| `_release_all_component_layers` | 同上 | L1405-1426 |
| `_demote` | 同上 | L1586-1612 |
| `_cascade_evict` | 同上 | L1614-1655 |
| `_remove_leaf_from_parent` | 同上 | L1699-1708 |
| `_iteratively_delete_tombstone_ancestors` | 同上 | L1758-1812 |
| `evict` / `_evict` / `_evict_components` | `srt/mem_cache/unified_radix_cache.py` | L563-759 |
| `_execute_kv_backup` | 同上 | L1370-1382 |
| `_finish_write_through_ack` | 同上 | L1420-1435 |
| `writing_check` / `loading_check` | 同上 | L2622-2730 |
| `HybridCacheController.write` | `srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` | L309-338 |
| `HybridCacheController.load` | 同上 | L500-540 |
| `HiCacheController.start_writing` | `srt/managers/cache_controller.py` | L805-827 |
| `HiCacheController.start_loading` | 同上 | L917-954 |
| `MultiEndedAllocator.alloc` | `srt/mem_cache/multi_ended_allocator.py` | L1081-1120 |
| `MultiEndedAllocator.free` | 同上 | L1278-1330 |
| `_flush` | 同上 | L1787-1858 |
| `_topmost_survivor` | 同上 | L1696-1750 |
| `_relieve_for_alloc` | 同上 | L216-243 |
| `UnifiedKVPool` | `srt/mem_cache/unified_memory_pool.py` | L293- |
| `init_unified_mamba_pools` | 同上 | L1159-1398 |
| `donate_mamba_ping_pong_slot` (原生) | `srt/mem_cache/memory_pool.py` | L1495-1515 |
| `_mamba_radix_cache_v2_req_prepare_for_extend` | `srt/managers/schedule_batch.py` | L2851 |
| track_mask (decode) | `srt/managers/schedule_batch.py` | L3345-3366 |

---

## 8. Phase 1 指标收集点设计建议

基于以上分析，Phase 1 最小侵入注入点：

### 8.1 请求级指标

- **`mla_potential_hit_tokens`**：在 `sglang_hook.py:L2398`（`final_reused_tokens` 写入处）旁，从 `wrapped_match_prefix` 的日志中提取 `full_kv_hit_length`（match_prefix 前的候选长度）
- **`kda_recoverable_position`**：在 `sglang_hook.py:L2442-2469` 的 per-req 循环里，从 `mamba_branching_seqlen` 和 `mamba_last_track_seqlen` 派生
- **`effective_prefix_hit_tokens`**：当前 `final_reused_tokens` + 需加 KDA checkpoint 等效前缀
- **`retreat_gap_tokens`**：在 COW 失败 / REPLAY-FALLBACK 路径，`retreat_gap = prefix_len - kda_recoverable_position`
- **`kda_replay_tokens`**：从 `_kda_diag` 全局改为 per-req，写入 `RequestStats`
- **`kda_l1/l2/l3_hit`**：在 `_mock_start_writing`/`_mock_start_loading` 的 per-slot 循环里，按 `node_id → rid` 映射聚合
- **`kda_loadback_bytes`**：`loadback_slots × MAMBA_SLOT_BYTES`，per-req 聚合
- **`mla_loadback_bytes`**：`l2_hit_tokens × MLA_ENTRY_BYTES`，per-req

### 8.2 Checkpoint 生命周期事件

建议在以下位置埋点（用环境变量 `HISIM_KDA_OBS=1` 开关）：

| 事件 | 埋点位置 | 记录字段 |
|---|---|---|
| Checkpoint Create/Track | `sglang_hook.py:L2610`（HiSim 设 `mamba_last_track_seqlen`） | rid, node_id, track_pos, virtual_time |
| Checkpoint Donate/Commit | `sglang_hook.py:L2851-2855`（donation 检测） | rid, node_id, slot_id, checkpoint_pos |
| Checkpoint Match | `sglang_hook.py:L1948-1968`（`wrapped_match_prefix`） | rid, best_match_node, full_kv_hit, host_hit, mamba_host_hit |
| Checkpoint Backup (L1→L2) | `sglang_hook.py:L1491`（`mh.backup_from_device`） | node_id, slot_hash, bytes, virtual_time |
| Checkpoint Evict from L1 | `sglang_hook.py:L1925`（proactive `self.evict`）+ 原生 `_demote` | node_id, reason(proactive/alloc_fail), target_layer |
| Checkpoint Restore from L2/L3 | `sglang_hook.py:L1661/L1669`（`mh.load_to_device`） | node_id, slot_hash, from_layer(L2/L3), bytes |
| Checkpoint Permanent Delete | 原生 `_remove_leaf_from_parent`（需 hook） | node_id, reason |

### 8.3 修复建议（Phase 1 前置）

| 修复项 | 优先级 | 位置 |
|---|---|---|
| 删除或明确标注 KDA 死代码 | 高 | `sglang_hook.py:L807-1011` |
| 修复 metrics.json 从死 pool 读取 | 高 | `sglang_hook.py:L3087-3093` → 改读 `_StrictMambaAllocator.get_stats()` |
| 修复 MLA L3 hit 污染 KDA 计数器 | 高 | `sglang_hook.py:L1627` → 改用 MLA 专用计数器 |
| iteration.jsonl 加 KDA 延迟 | 中 | `sglang_hook.py:L2972-2981` |
| RequestStats 加 KDA 字段 | 中 | `types.py:L55-67` |
| `replay_fallback_count`/`no_cow_count` 写入 metrics | 中 | `sglang_hook.py:L3076-3084` |
| 统一 `SGLANG_MAMBA_TRACK_INTERVAL` 与 ServerArg | 中 | `sglang_hook.py:L2582` |

---

## 9. 下一步（Phase 1）准备

基于本审查，Phase 1 的实施路径：

1. **修复死代码/计数器污染**（前置必做）：删除 KDA 死代码或明确标注；修复 metrics.json 读取死 pool；修复 MLA L3 hit 污染 KDA 计数器
2. **扩展 RequestStats**：加 KDA per-req 字段（`kda_replay_tokens`、`kda_l1/l2/l3_hit`、`kda_loadback_bytes`、`mla_loadback_bytes`、`retreat_gap_tokens`、`kda_recoverable_position`、`mla_potential_hit_tokens`）
3. **埋点请求级指标**：在 `wrapped_get_new_batch_prefill` / `wrapped_run_batch` / `wrapped_match_prefix` / `_mock_start_writing` / `_mock_start_loading` 中填充 per-req 字段
4. **Checkpoint 事件日志**：用 `HISIM_KDA_OBS=1` 开关，输出 `checkpoint_events.jsonl`
5. **验证基线不变**：关闭 `HISIM_KDA_OBS` 后，metrics.json 与未修改前完全一致

所有诊断代码必须：
- 默认关闭（环境变量开关）
- 不改变调度/淘汰决策
- 不影响虚拟时钟推进
- 输出独立文件（`checkpoint_events.jsonl`），不污染 `metrics.json`/`iteration.jsonl`/`request.jsonl` 的原有字段

---

> **审查完成。请确认本审计报告的结论是否与您的理解一致，特别是：**
> 1. KDA 8 阶段流水线是死代码（F1）这一发现
> 2. 两套并行 KDA pool 实现（F2）这一发现
> 3. 预测器不含 KDA Replay 成本（F5）这一限制
> 4. proactive eviction 需用 `HISIM_DISABLE_PROACTIVE_EVICT=1` 区分基线（F14）
> 5. `SGLANG_MAMBA_TRACK_INTERVAL` env var 与 ServerArg `mamba_track_interval` 的关系（F6）
>
> 确认后我将进入 Phase 1 实施。
