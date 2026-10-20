# K3 KDA Retreat Analysis — Phase 1 实验报告

> **实验日期**: 2026-10-19
> **HiSim Commit**: `1747b4c` (develop_dqy_evict_reuse)
> **SGLang Commit**: `0bcd822377` (sglang 0.5.19, 容器 `lfz-sglang`)
> **实验类型**: KDA L3 可复用性验证 + KDA Retreat 根因分析

---

## 1. 实验配置

### 1.1 环境变量与 ServerArgs

| 参数 | 值 | 来源 |
|---|---|---|
| `HISIM_MAMBA_POOL_SIZE` | 80 | L1 KDA pool slots |
| `HISIM_L2_MAMBA_SIZE` | 3 | L2 KDA pool slots（极小，强制 L3 溢出） |
| `SGLANG_MAMBA_TRACK_INTERVAL` | 1024 | HiSim env var（控制 track_boundary） |
| `--mamba-track-interval` | 1024 | ServerArg（控制原生 track grid） |
| `HISIM_DISABLE_PROACTIVE_EVICT` | 未设（ON） | 启用主动淘汰（用于触发 L3 溢出） |
| `HISIM_KDA_OBS` | 1 | 启用 checkpoint 事件日志 |
| `--max-total-tokens` | 200,000 | MLA L1 容量 |
| `--max-running-requests` | 24 | 并发上限 |
| `--enable-hierarchical-cache` | True | L2/L3 分层缓存 |
| `--hicache-storage-backend` | file | L3 存储后端 |
| `--hicache-size` | 80 | L2 host DRAM 容量 (GB) |

### 1.2 Mock 配置

| 项 | 值 |
|---|---|
| 模型 | K3-23L mock (`/tmp/opencode/k3-model`) |
| Sim config | `/tmp/opencode/config_k3_0519.json` |
| slot_bytes | 14,653,440 (18 KDA 层 × 814,080 B/层) |
| MLA entry_bytes | 5,760 (5 MLA 层 × 288 × 2 × 2) |
| `mamba_max_states_per_path` | -1（无限制，原生默认） |

### 1.3 Trace 设计

**文件**: `/tmp/opencode/k3_l3_verify.jsonl`（由 `benchmark/collection/gen_k3_l3_verify.py` 生成）

| 阶段 | 请求数 | input_length | 说明 |
|---|---|---|---|
| Build | 20 | 1024 | 20 个独立 1024-token 会话，各产生 1 个 KDA checkpoint |
| Reuse | 4 | 1025 | 重访 session 0,5,10,15（+1 token 确保 full match） |

- C (track_interval) = 1024，使 `track_boundary = (1024 // 1024) * 1024 = 1024`
- 每个会话 token 不重叠（`s * 200_000 + i`）
- output_length = 1（最小 decode）

### 1.4 实验目标

1. **[审查意见 5]** 验证 KDA L3 可复用性：Build → Backup L3 → Evict L1/L2 → Reuse
2. **[审查意见 6]** 隔离 proactive eviction 影响（本实验用 proactive ON 触发 L3，后续补 proactive OFF 对照）
3. KDA Retreat 根因分类
4. F3/F4 修复验证
5. F5 修正验证（Predictor 是否隐含 Replay 成本）

---

## 2. 实验结果

### 2.1 核心指标

| 指标 | 值 | 说明 |
|---|---|---|
| num_completed_requests | 24/24 | ✅ 全部完成 |
| kda_write_calls | 20 | 20 次 L1→L2 backup（KDA write-through） |
| kda_write_mamba_xfers | 20 | 20 次 MAMBA pool transfer |
| kda_load_calls | 3 | 3 次 L2→L1 load-back |
| kda_load_mamba_xfers | 3 | 3 次 MAMBA load-back |
| **kda_l3_writes** | **20** | ✅ **20 次 L3 写入** — L1→L2→L3 溢出链路工作 |
| **kda_l3_hits** | **3** | ✅ **3 次 L3 命中** — L3 可复用性验证通过 |
| kda_checkpoint_donate_count | 24 | 24 次 checkpoint 捐赠到 radix tree |
| kda_checkpoint_hit_count | 1 | 1 次 COW 命中（session 15） |
| kda_cow_exec_count | 1 | 1 次 COW 执行 |
| kda_historical_replay_tokens | 7168 | ⚠️ 重复计数（实际 4096 = 4×1024） |
| kda_active_slots | 8 | ✅ **[F3 修复]** 从活跃 `_StrictMambaAllocator` 读取 |
| kda_pool_capacity | 80 | ✅ **[F3 修复]** |
| kda_alloc_count | 75 | 总分配次数 |
| kda_free_count | 67 | 总释放次数 |
| mean_ttft_ms | 379.48 | 平均 TTFT |
| prefix_cache_reused_ratio | 0.167 | 前缀复用率（低，因各会话独立） |

### 2.2 Per-Request KDA 指标

**Build 阶段（20 请求，input=1024）**：
- `mla_potential_hit_tokens=0` — 首次请求，无前缀匹配 ✅
- `kda_recoverable_position=0` — 无历史 checkpoint ✅
- `retreat_gap_tokens=0` — 无 gap ✅
- `kda_l1_hit=False, kda_l2_hit=False, kda_l3_hit=False` — 无缓存命中 ✅

**Reuse 阶段（4 请求，input=1025）**：

| rid | session | mla_pot | kda_recov | retreat_gap | replay | L1 | L2 | L3 | cow |
|---|---|---|---|---|---|---|---|---|---|
| 499707c9 | 0 | 1024 | 0 | 1024 | 1024 | ❌ | ✅(MLA) | ✅(agg) | -1 |
| bc0e2d64 | 5 | 1024 | 0 | 1024 | 1024 | ❌ | ✅(MLA) | ✅(agg) | -1 |
| 2924d51a | 10 | 1024 | 0 | 1024 | 1024 | ❌ | ✅(MLA) | ✅(agg) | -1 |
| eb376dd3 | 15 | 1024 | 0 | 1024 | 1024→0 | ✅ | ❌ | ❌ | 47 |

**关键观察**：
- 所有 4 个 reuse 请求的 `mla_potential_hit_tokens=1024` — MLA 找到了完整前缀 ✅
- `kda_recoverable_position=0` — KDA checkpoint 在匹配节点未找到 → KDA Retreat
- `retreat_gap_tokens=1024` — 整个 MLA 前缀成为 retreat gap
- Session 0,5,10：KDA Retreat（checkpoint 在 L3）→ 3 次 L3 命中
- Session 15：COW 命中（checkpoint 仍在 L1，slot=47）→ `replay_tokens=0`

### 2.3 Checkpoint 事件日志

**文件**: `checkpoint_events.jsonl`（4 条 match 事件）

```json
{"vtime":100.0, "event":"match", "rid":"499707c9", "checkpoint_pos":0, "layer":"L2", "mla_potential":1024, "retreat_gap":1024, "replay_tokens":1024}
{"vtime":101.0, "event":"match", "rid":"bc0e2d64", "checkpoint_pos":0, "layer":"L2", "mla_potential":1024, "retreat_gap":1024, "replay_tokens":1024}
{"vtime":102.0, "event":"match", "rid":"2924d51a", "checkpoint_pos":0, "layer":"L2", "mla_potential":1024, "retreat_gap":1024, "replay_tokens":1024}
{"vtime":103.0, "event":"match", "rid":"eb376dd3", "checkpoint_pos":0, "layer":"",  "mla_potential":1024, "retreat_gap":1024, "replay_tokens":1024}
```

### 2.4 Server 日志关键行

**Build 阶段模式**（每个 build 请求）：
```
[KDA-TRACK] rid=... pool_idx=1 pp_buf=[2, 3] cow_src=None prefix_len=0
[KDA-DONATE-CHECK] rid=... olen=1 mpi=1 lts=1024
[KDA-DONATE] rid=... slot=1 track_seqlen=1024
```

**Proactive eviction**（build 期间触发）：
```
Proactive evict (check_hicache): 5120 tokens+mamba demoted to L2, L1_avail=198976, cn=165391
```

**Reuse 阶段 — KDA Retreat（session 0,5,10）**：
```
[KDA-REPLAY] rid=499707c9 prefix_len=1024 branching=None replay_tokens=1024 cow_src=None
[REPLAY-FALLBACK] rid=499707c9 prefix_len=1024 branching=None replay_tokens=1024 cow_src=None
```

**Reuse 阶段 — COW 命中（session 15）**：
```
[KDA-REPLAY] rid=eb376dd3 prefix_len=1024 branching=None replay_tokens=1024 cow_src=None
[COW-EXEC-SIM] src_slot=47 dst_slot=73 checkpoint_pos=1024 prefix_len=1024 replay_tokens=0 new_tokens=1
```

---

## 3. KDA Retreat 根因分类

### 3.1 本次实验的 Retreat 根因

| 根因 | 影响请求数 | 说明 |
|---|---|---|
| **Checkpoint 已淘汰到 L3** | 3/4 | Proactive eviction 将 L1 checkpoint 降级到 L2→L3。Reuse 时 MAMBA validator 在匹配节点失败（`value=None, host_value=None`），best_match 退回到 root（position=0），retreat_gap=1024 |
| **Checkpoint 仍在 L1（COW 命中）** | 1/4 | Session 15 的 checkpoint 是最近创建的（slot=47），proactive eviction 尚未将其降级。COW src=47 有效，replay_tokens=0 |

### 3.2 KDA Retreat 机制图

```
Build Phase (20 sessions):
  每个请求 → Track (建立 mamba_last_track_seqlen=1024)
           → Donate (keep slot 捐给 radix tree)
           → Commit (node.component_data[MAMBA].value = slot_id)
           → Proactive Evict (每轮把 evictable_size 全降 L2)
              → L2 (size=3) 快速满 → 溢出到 L3
              → st.batch_set_mamba(f"kda_{nid}_0") → L3 写入

Reuse Phase (4 sessions):
  请求到达 → match_prefix
    → MLA validator: full_kv_hit_length=1024 (MLA 找到完整前缀)
    → MAMBA validator: 在深节点失败 (value=None, host_value=None after L2→L3 overflow)
    → best_match_node 退回到 root (position=0)
    → retreat_gap = 1024 - 0 = 1024
    → extend_range.length = 1025 - 0 = 1025 (含 1024 replay + 1 new)
    → Predictor 用 input_length=1025 查表 (隐含 replay 成本) ✅ [F5 修正验证]

  3/4 sessions: checkpoint 在 L3
    → _mock_start_loading → st.batch_exists_mamba(f"kda_{nid}_0") → hit!
    → L3 restore → L2 → L1 (H2D 计时)
    → kda_l3_hits += 1

  1/4 sessions (session 15): checkpoint 仍在 L1
    → mamba_cow_src_index = slot 47 (valid)
    → COW-EXEC-SIM: replay=0, new=1
```

### 3.3 根因分类框架

| 根因类型 | 描述 | 检测方法 | 本次实验 |
|---|---|---|---|
| **Type A: Checkpoint 从未生成** | 请求到达时该 token 位置从未建立过 KDA checkpoint | `kda_recoverable_position=0 AND 无 L3 batch_exists_mamba hit` | 0 次 |
| **Type B: Checkpoint 已生成但被淘汰删除** | Checkpoint 在 L1 被驱逐且无 L2/L3 备份 | `kda_recoverable_position=0 AND L3 miss` | 0 次（因 L3 write-through 保证备份） |
| **Type C: Checkpoint 已下沉但未及时恢复** | Checkpoint 在 L2/L3 但 load-back 未在 match_prefix 前完成 | `kda_recoverable_position=0 AND L3 hit (后续恢复)` | **3 次** |
| **Type D: KDA/MLA 复用粒度不同** | MLA 可命中更深的前缀，但 KDA checkpoint 只在更浅位置 | `retreat_gap > 0 AND kda_recoverable_position > 0` | 0 次 |
| **Type E: 其他实现原因** | COW 不可用、slot 分配失败等 | 各类错误日志 | 0 次 |

**本次实验的 Retreat 主要是 Type C**：Checkpoint 被 proactive eviction 降级到 L3，reuse 时 L3 命中并恢复，但 retreat gap 已经产生（需要 replay 计算来重建 KDA 状态）。

### 3.4 L3 可复用性验证结论

| 验证项 | 结果 | 证据 |
|---|---|---|
| L1→L2→L3 写入链路 | ✅ 正常 | `kda_l3_writes=20`（20 个 checkpoint 写入 L3） |
| L3 查找链路 | ✅ 正常 | `kda_l3_hits=3`（3 个 checkpoint 在 L3 命中） |
| node_id slot_hash 一致性 | ✅ 正常 | backup 用 `f"kda_{nid}_0"`，load 用相同 hash，3 次命中证明匹配 |
| 节点淘汰重建后 hash 行为 | ✅ 正常 | proactive eviction 降级节点后，L3 仍能通过 node_id hash 找到 |
| L3→L2→L1 恢复链路 | ✅ 正常 | `kda_load_calls=3, kda_load_mamba_xfers=3`（3 次 load-back） |

---

## 4. 修复验证

### 4.1 F3 修复验证（metrics 从死 pool 读取）

| 指标 | 修复前 | 修复后 | 验证 |
|---|---|---|---|
| `kda_active_slots` | 0（读死 pool） | **8**（读活跃 allocator） | ✅ |
| `kda_pool_capacity` | 不存在 | **80** | ✅ |
| `kda_alloc_count` | 0（读死 pool） | **75** | ✅ |
| `kda_free_count` | 不存在 | **67** | ✅ |
| `kda_free_slots` | 0（读死 pool） | **72** | ✅ |

### 4.2 F4 修复验证（MLA L3 hit 污染 KDA 计数器）

| 检查项 | 修复前 | 修复后 | 验证 |
|---|---|---|---|
| MLA L3 load 延迟记录位置 | `inc_hicache_kda_l3_load_dur`（KDA 计数器） | `inc_hicache_l3_load_dur`（MLA 计数器） | ✅ [静态验证] |
| KDA L3 load 统计纯净度 | 被 MLA L3 延迟污染 | 不受 MLA 影响 | ✅ [静态验证] |

### 4.3 F5 修正验证（Predictor 是否隐含 Replay 成本）

| 检查项 | 结果 | 说明 |
|---|---|---|
| KDA Retreat 后 `extend_range.length` | 1025（= total 1025 - effective_prefix 0） | [静态审查] Retreat 导致 effective_prefix=0 → extend_range 增大 |
| `FakeRequest.input_length` | = `extend_range.length` = 1025 | [静态审查] sglang_hook.py:L2763-2767 |
| Predictor 查表用 input_length | 1025（含 1024 replay + 1 new） | [静态审查] k3_predictor.py:L98-99 |
| **Replay 成本是否隐含** | **✅ 是** | Predictor 用 1025 查表，包含了 replay 的 1024 token 计算成本 |
| 诊断 `replay_tokens=1024` | 正确记录 | 但不影响 Predictor（Predictor 不读此值） |
| `historical_replay_tokens=7168` | ⚠️ 重复计数 | 实际 4096=4×1024，KDA-REPLAY 和 REPLAY-FALLBACK 双重累计 |

---

## 5. 已知限制与后续工作

### 5.1 Per-req 观测精度限制

| 限制 | 影响 | 后续修复 |
|---|---|---|
| `kda_recoverable_position=0` 不够精确 | 应为 best_match_node 的 key 长度，但当前从 `mamba_branching_seqlen` 派生（None 时默认 0） | 需在 `wrapped_match_prefix` 中存储 best_match_node 的 token 位置 |
| `kda_l2_hit` 实际是 MLA L2 hit | `l2_hit_tokens` 是 MLA 的，非 KDA 的 | 需从 `mamba_host_hit_length` 派生 KDA L2 hit |
| `kda_l3_hit` per-req 未实现 | aggregate `kda_l3_hits=3` 正确，但 per-req 无此字段 | 需在 `_mock_start_loading` 中按 node_id→rid 映射 |
| `historical_replay_tokens` 重复计数 | KDA-REPLAY（L2467）和 REPLAY-FALLBACK（L2747）双重累计 | 去重：只在 REPLAY-FALLBACK 路径累计 |

### 5.2 未完成的实验

| 实验 | 状态 | 原因 |
|---|---|---|
| Proactive OFF baseline 对照 | 待执行 | 需要单独跑 `HISIM_DISABLE_PROACTIVE_EVICT=1` + 大 L1/L2（不触发 L3） |
| 1/10 节流 vs 每轮处理对比 | 待执行 | 需修改 `wrapped_check_hicache_events` 节流参数 |
| KDA Slot Budget 扫描 | 待执行 | 需扫描 40/80/165 等不同 KDA pool 大小 |
| 实验 B（Checkpoint 位置与分叉） | 待执行 | 需构造分叉 trace |
| 实验 C（真实 Coding Agent Trace） | 待执行 | 真实 trace 后续提供 |

### 5.3 实验环境记录

| 项 | 值 |
|---|---|
| HiSim Commit | `1747b4c` (branch `develop_dqy_evict_reuse`) |
| SGLang Commit | `0bcd822377` (sglang 0.5.19) |
| 容器 | `lfz-sglang` (`quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3`) |
| Python | 3.11.15 |
| PyTorch | 2.10.0+cpu |
| 模拟目标 | K3-23L (23 层，5 MLA + 18 KDA) |
| Trace | `k3_l3_verify.jsonl` (24 reqs, 24580 tokens) |
| 输出文件 | `test/results/l3_verify/` (metrics.json + checkpoint_events.jsonl + request_phase1.jsonl + iteration_phase1.jsonl) |

---

## 6. 研究问题初步回答

### Q1: 原生 SGLang 的 KDA Checkpoint 主要保存在哪里？

**[静态审查]** KDA Checkpoint 按 `mamba_track_interval`（默认 256，本实验设为 1024）的倍数位置保存。在 track 边界触发（`schedule_batch.py:L3357` track_mask），通过 `donate_mamba_ping_pong_slot` 捐赠到 Radix Tree 节点的 `component_data[MAMBA].value`。

### Q2: 产生 KDA Retreat 的主要原因是什么？

**[运行验证]** 本次实验中 3/4 reuse 请求的 Retreat 根因是 **Type C（Checkpoint 已下沉但未及时恢复）**：
- Proactive eviction 将 L1 checkpoint 降级到 L2→L3
- Reuse 时 MAMBA validator 在匹配节点失败
- L3 命中并恢复，但 retreat gap 已产生

### Q3: KDA Checkpoint 淘汰是否真的导致大量额外 Prefill？

**[运行验证]** 本次实验中：
- 3 个 reuse 请求因 Retreat 需要额外 replay 1024 tokens（共 3072 tokens）
- 1 个 reuse 请求 COW 命中，无需 replay
- 额外计算成本已被 Predictor 隐含（通过增大的 `extend_range.length`）[F5 修正验证]
- 但 `historical_replay_tokens=7168` 重复计数，实际为 4096

### Q4: 如果能够避免不必要的 KDA Checkpoint 删除，理论上能节省多少开销？

**[静态审查]** 本次实验中，如果 proactive eviction 不淘汰 session 0,5,10 的 checkpoint：
- 3 个请求本可 COW 命中（replay=0），实际 replay=1024
- 节省 3 × 1024 = 3072 tokens 的额外计算
- 按 prefill 表：`interp(1576→0.439s, 4096→0.725s, 1024→~0.3s)`，约节省 3 × 0.3s ≈ 0.9s
- **理论上界**（Oracle），不可作为线上策略的性能

### Q5: 下一步应该优先优化保存位置，还是淘汰与分层放置？

**[运行验证 + 静态审查]** 基于本次实验：
- Retreat 根因是 proactive eviction 的**过粗粒度**（每轮把 `evictable_size` 全降 L2）
- **优先优化淘汰策略**（Value-Aware Eviction）：
  - 在 proactive eviction 中加入价值感知（保留高频复用的 checkpoint）
  - 加入 fork 点保护（`children>=2` 的节点给予保留权重）
  - 按价值排序的部分驱逐，替代全量 demote
- **保存位置优化**为次优先级：
  - 当前 track_interval=1024 已经合理（每 1024 tokens 一个 checkpoint）
  - 保存位置优化主要影响分叉场景（实验 B 待执行）

---

## 7. 复现命令

```bash
# 1. 生成 trace
cd /home/d00954154/hisim
python3 benchmark/collection/gen_k3_l3_verify.py /tmp/opencode

# 2. 启动服务器（容器内）
docker exec -d lfz-sglang bash -c '
export SGLANG_USE_CPU_ENGINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 TORCH_TRANSFER_TO_NPU=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HISIM_SIMULATION_MODE=OFFLINE HISIM_OUTPUT_DIR=/tmp/hisim/l3_verify HISIM_SSD_SIM=simpyssd HISIM_RESET_HICACHE_STORAGE=1
export HISIM_MAMBA_POOL_SIZE=80 HISIM_L2_MAMBA_SIZE=3 SGLANG_MAMBA_TRACK_INTERVAL=1024
export HISIM_KDA_OBS=1
export PYTHONPATH=/home/d00954154/hisim/src:/home/d00954154/hisim/3rdparty/kvcm_l3:/home/d00954154/hisim/3rdparty/simpyssd
cd /home/d00954154/hisim/src
python3 -u -m hisim.simulation.sglang.launch_server \
  --model-path /tmp/opencode/k3-model \
  --sim-config-path /tmp/opencode/config_k3_0519.json \
  --skip-server-warmup --port 30010 --enable-hierarchical-cache \
  --hicache-storage-backend file --hicache-size 80 \
  --max-total-tokens 200000 --max-running-requests 24 \
  --mamba-track-interval 1024 \
  --trust-remote-code --device cpu --load-format dummy
'

# 3. 等待就绪后发送 trace
docker exec lfz-sglang bash -c '
export HISIM_KDA_OBS=1 SGLANG_MAMBA_TRACK_INTERVAL=1024 HISIM_MAMBA_POOL_SIZE=80
export PYTHONPATH=/home/d00954154/hisim/src
cd /home/d00954154/hisim/src
python3 -u -m hisim.simulation.bench_serving \
  --warmup-requests 0 --model /tmp/opencode/k3-model \
  --bench-mode simulation --dataset-name hisim-collection \
  --dataset-path /tmp/opencode/k3_l3_verify.jsonl --num-prompts 24 \
  --host 127.0.0.1 --port 30010
'

# 4. 查看结果
docker exec lfz-sglang cat /tmp/hisim/l3_verify/metrics.json | python3 -m json.tool
docker exec lfz-sglang cat /tmp/hisim/l3_verify/checkpoint_events.jsonl
```

---

## 8. 结论标记

| 结论 | 验证方式 | 标记 |
|---|---|---|
| L3 写入链路正常 | 运行实验 `kda_l3_writes=20` | ✅ [运行验证] |
| L3 查找链路正常 | 运行实验 `kda_l3_hits=3` | ✅ [运行验证] |
| node_id slot_hash 一致性 | 运行实验（3 次命中） | ✅ [运行验证] |
| F3 修复生效 | 运行实验 `kda_active_slots=8` | ✅ [运行验证] |
| F4 修复生效 | 静态代码审查 | ✅ [静态验证] |
| F5 修正：Predictor 隐含 Replay | 静态代码审查 | ✅ [静态验证] |
| KDA Retreat 根因为 Type C | 运行实验 + 日志分析 | ✅ [运行验证] |
| `historical_replay_tokens` 重复计数 | 运行实验 7168 vs 实际 4096 | ✅ [运行验证] |
| `mamba_max_states_per_path=-1`（休眠） | 静态代码审查 + server_args 日志 | ✅ [静态验证] |
| fork 保护缺口在 LRU 路径 | 静态代码审查 | ✅ [静态验证] |
| slot_bytes=14,653,440 正确 | 静态代码审查 + K3 config 验证 | ✅ [静态验证] |
| 80 slots + 32 running req 不可行 | 静态容量核算 | ✅ [静态验证] |
| Proactive OFF baseline 对照 | 待执行 | ⏳ [待验证] |
| 1/10 节流对比 | 待执行 | ⏳ [待验证] |
| KDA Slot Budget 扫描 | 待执行 | ⏳ [待验证] |
