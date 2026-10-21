ase | Checkpoint Backup (L1→L2) | `sglang_hook.py:L1491`（`mh.backup_from_device`） | node_id, slot_hash, bytes, virtual_time |
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
