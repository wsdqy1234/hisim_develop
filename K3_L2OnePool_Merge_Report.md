# L2 Unified DRAM Pool 合入报告

> 分支：`develop_dqy_l2_one_pool` → `develop_dqy_evict_reuse`
> 日期：2026-10-28
> 合并 commit：`7e5cd4c`

---

## 1. 合并背景

另一个 Coding Agent 基于 `develop_dqy_evict_reuse` 的早期版本（commit `199b2a8`，Phase 2C.3 设计文档）创建了 `develop_dqy_l2_one_pool` 分支，实现了 Local DRAM One Pool（KDA/MLA 共享物理 DRAM 预算）。

我们的 `develop_dqy_evict_reuse` 分支在 `199b2a8` 之后又进行了 Phase 2C.3.1（L3 Restore 修复）和 Phase 2C.3.2（L2 策略分化）共 11 个 commit 的工作。需要将 One Pool 的改动合入到当前最新版本。

### 分支拓扑

```
199b2a8  (merge base — Phase 2C.3 Tier-Aware L2 Placement 设计)
│
├── develop_dqy_evict_reuse (我们的分支, +11 commits)
│   ├── e16c5d5  ASCEND_RT_VISIBLE_DEVICES=7 (NPU 泄漏绕过)
│   ├── 433850b  实验脚本 v2 (device 7 + SIGTERM + config)
│   ├── 63440b3  config 修复 (k3_profile_based predictor)
│   ├── 322b458  L2=128 消除 820K replay
│   ├── 7fb2e93  ★ L3 env var truthiness 修复 (820K replay 根因)
│   ├── 0cbac3e  depth 传递 + L2 victim 诊断日志
│   ├── 52f6375  Phase 2C.3.1 L3 Restore Closure 报告
│   ├── 51d60e5  ★ L2 策略分化修复 + TierCostAwareL2 + trace gen
│   ├── 46b0625  l2diff trace 重设计 (20 build requests)
│   ├── 8525740  record_access 诊断日志
│   └── 62b82d7  Phase 2C.3.2 L2 Policy Differentiation 报告
│
└── develop_dqy_l2_one_pool (另一 agent, +2 commits)
    ├── e2d938d  unified KDA/MLA DRAM pool (PageExtentAllocator)
    └── 4430405  host-index leak 修复 + 13 测试 + server 验证
```

### 各分支改动统计

| 分支 | 新增文件 | 修改文件 | 新增行数 |
|------|---------|---------|---------|
| develop_dqy_evict_reuse (vs merge base) | 7 | 3 | +1,011 |
| develop_dqy_l2_one_pool (vs merge base) | 5 | 3 | +3,550 |

**共同修改的文件**（潜在冲突点）：
- `src/hisim/simulation/sglang/sglang_hook.py`
- `src/hisim/simulation/sglang/sglang_mock_class.py`

---

## 2. 合并前分析

### 2.1 冲突检查

执行 `git merge --no-commit --no-ff` dry-run：

```
Auto-merging src/hisim/simulation/sglang/sglang_hook.py
Auto-merging src/hisim/simulation/sglang/sglang_mock_class.py
Automatic merge went well; stopped before committing as requested
```

**结果：零冲突。** Git 自动合并成功。

### 2.2 改动区域分析

虽无冲突，但需验证自动合并在语义上是否正确（即两边改动是否保留了各自的逻辑）。

#### sglang_hook.py 改动区域对比

| 我们的改动（行号） | 所在函数 | One Pool 改动（行号） | 所在函数 |
|-------------------|---------|----------------------|---------|
| L1558 | `_apply_mock_start_overrides` (L3 修复) | L1410 | `_setup_l2_dram_pool` (新函数) |
| L1640 | `_mock_start_writing` (depth 传递) | L1457 | `_bind_l2_host_group` (新函数) |
| L1943 | `_mock_start_loading` (record_access) | L2011+ | `__init__` host pool 创建 |
| L2596 | `check_hicache_events` (proactive evict) | L4438 | metrics 输出 |
| L3290 | `schedule_running` (proactive evict) | | |

**结论：完全不重叠。** 我们的改动在 L1/L3 恢复路径和策略钩子，One Pool 的改动在 host pool 创建和绑定阶段。

#### sglang_mock_class.py 改动区域对比

| 我们的改动 | One Pool 改动 |
|-----------|--------------|
| `load_to_device` (L2048)：注释掉 record_access（移至 _mock_start_loading） | `calc_kv_host_size_per_token` (新函数) |
| | `MockTokenToKVPoolHost`: unified 集成 (attach/alloc/free/available_size) |
| | `MockMambaStatePoolHost.__init__`: _unified/_evict_hook/_allocated_idx |
| | `backup_from_device`: admit_kda 准入检查 |
| | `evict`/`select_victim`: release_kda 释放 |
| | `alloc`/`free`/`available_size`/`get_stats`: unified 路径 |

**结论：完全不重叠。** 我们只改了 `load_to_device`，One Pool 改了 `__init__`/`alloc`/`free`/`backup_from_device`/`evict`/`select_victim`。

### 2.3 语义兼容性预检

| 检查项 | 结果 | 说明 |
|--------|------|------|
| L3 write-through | ✅ 兼容 | One Pool 设计文档明确"不新增也不绕过 L3" |
| L2 淘汰顺序 | ✅ 兼容 | One Pool 明确保留 `l2_placement_policy` 决定淘汰顺序 |
| KDA 身份 | ✅ 兼容 | `slot_hash` 格式不变（`kda_{nid}_{i}`） |
| record_access | ✅ 兼容 | One Pool 不修改 `_mock_start_loading` 和 `load_to_device` |
| record_eviction | ✅ 兼容 | One Pool 调用 `_pol.record_eviction()`，与我们的修复一致 |
| L3 env var 修复 | ✅ 兼容 | One Pool 不触碰 `_apply_mock_start_overrides` |
| 默认关闭 | ✅ 零回归 | 仅 `HISIM_L2_ONE_POOL=1` 或 `HISIM_L2_TOTAL_GB` 设置时启用 |
| env var truthiness | ✅ 正确 | `_env_bool` 正确处理 `"0"` → False |
| leak fix | ✅ 兼容 | round-robin host index + idempotent free，legacy 路径不变 |

---

## 3. 合并执行

### 3.1 合并命令

```bash
git merge --no-ff origin/develop_dqy_l2_one_pool -m "merge: integrate develop_dqy_l2_one_pool ..."
```

### 3.2 合并结果

```
Merge: 62b82d7 4430405
commit 7e5cd4c4ed273c3a26274848492f2565475aa7dc

 8 files changed, 3550 insertions(+), 12 deletions(-)
 create mode 100644 K3_L2UnifiedDRAMPool_Design.md
 create mode 100644 benchmark/collection/l2_pool_frag_study.py
 create mode 100644 benchmark/collection/l2_pool_frag_study_results.json
 create mode 100644 src/hisim/simulation/l2_dram_pool.py
 create mode 100644 test/test_l2_dram_pool.py
```

**新增文件（来自 One Pool 分支）：**

| 文件 | 大小 | 说明 |
|------|------|------|
| `src/hisim/simulation/l2_dram_pool.py` | 850 行 / 34KB | One Pool 核心实现（PageExtentAllocator + MLA/KDA Views） |
| `test/test_l2_dram_pool.py` | 315 行 | 13 个单元测试 |
| `K3_L2UnifiedDRAMPool_Design.md` | 379 行 | 设计文档 |
| `benchmark/collection/l2_pool_frag_study.py` | 279 行 | 碎片研究脚本 |
| `benchmark/collection/l2_pool_frag_study_results.json` | 1382 行 | 碎片研究结果数据 |

**修改文件（双方合并）：**

| 文件 | 我们的改动 | One Pool 改动 | 合并状态 |
|------|-----------|--------------|---------|
| `sglang_hook.py` | L3 修复 + record_access + proactive evict | _setup_l2_dram_pool + _bind_l2_host_group | ✅ 自动合并 |
| `sglang_mock_class.py` | load_to_device record_access 移除 | unified 集成 + admit/release_kda | ✅ 自动合并 |
| `history.md` | （未修改） | 追加项目历史 | ✅ 直接采用 |

---

## 4. 合并后验证

### 4.1 双方改动保留验证

| 标记 | 合并后位置 | 状态 |
|------|-----------|------|
| **我们的改动** | | |
| L3 truthiness fix (`_l3_disable_val in`) | sglang_hook.py L1558 | ✅ 保留 |
| record_access on L2 lookup (`P2C.3.2`) | sglang_hook.py L1943 | ✅ 保留 |
| proactive evict fix (`_dpe_val in`) | sglang_hook.py L2596, L3290 | ✅ 保留 (2处) |
| TierCostAwareL2Policy | l2_placement_policy.py | ✅ 保留 |
| l2diff trace 生成器 | gen_k3_p2c32_trace.py | ✅ 保留 |
| record_access 移除注释 | sglang_mock_class.py L2048 | ✅ 保留 |
| **One Pool 改动** | | |
| `_setup_l2_dram_pool` | sglang_hook.py L1410 | ✅ 保留 |
| `_bind_l2_host_group` | sglang_hook.py L1457 | ✅ 保留 |
| `l2_dram_pool.py` | 34KB 新文件 | ✅ 保留 |
| `admit_kda` | sglang_mock_class.py L1997 | ✅ 保留 (2处) |
| `release_kda` | sglang_mock_class.py L2024/2069/2079 | ✅ 保留 (3处) |
| `_env_bool` | l2_dram_pool.py | ✅ 保留 |
| `test_l2_dram_pool.py` | 13 个测试 | ✅ 保留 |

### 4.2 单元测试

```
PYTHONPATH=src python3 -m pytest test/test_l2_dram_pool.py -v

============================= test session starts ==============================
collected 13 items

test_l2_dram_pool.py::test_exact_extent_alloc_free_coalesces PASSED          [  7%]
test_l2_dram_pool.py::test_external_fragmentation_is_reported PASSED         [ 15%]
test_l2_dram_pool.py::test_pow2_size_class_internal_fragmentation PASSED     [ 23%]
test_l2_dram_pool.py::test_mla_view_token_roundtrip_and_partial_free PASSED [ 30%]
test_l2_dram_pool.py::test_kda_view_admit_release PASSED                     [ 38%]
test_l2_dram_pool.py::test_two_pools_regions_are_isolated PASSED            [ 46%]
test_l2_dram_pool.py::test_one_pool_mla_borrows_from_kda PASSED             [ 53%]
test_l2_dram_pool.py::test_one_pool_kda_borrows_from_mla PASSED             [ 61%]
test_l2_dram_pool.py::test_two_ended_allocation_keeps_views_apart PASSED    [ 69%]
test_l2_dram_pool.py::test_borrow_policy_self_first_replaces_own_view PASSED [ 76%]
test_l2_dram_pool.py::test_accounting_consistency_exact PASSED              [ 84%]
test_l2_dram_pool.py::test_mamba_host_pool_legacy_free_list PASSED          [ 92%]
test_l2_dram_pool.py::test_mamba_host_pool_bound_indices_do_not_leak PASSED [100%]

======================= 13 passed, 14 warnings in 8.50s ========================
```

**结果：13/13 全部通过。**

### 4.3 模块 Import 验证

```
=== 我们的模块 ===
l2_placement_policy: OK
  default policy: native_l2

=== One Pool 模块 ===
l2_dram_pool: OK
  default (no env): None              ← 默认关闭，零回归 ✅
  enabled: mode=one_pool total=8.00GB ← 启用时正确创建 ✅
    mla_tokens=1388880 kda_slots=545

=== sglang_hook (双方共存) ===
sglang_hook: OK
  _setup_l2_dram_pool: True           ← One Pool 函数存在 ✅
  _bind_l2_host_group: True           ← One Pool 函数存在 ✅
```

**结论：所有模块正常加载，双方改动共存无冲突。**

---

## 5. One Pool 功能概述

### 5.1 核心设计

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
        └───────────┬───────────┘  └───────────┬────────────┘
                    │                          │
        MockTokenToKVPoolHost      MockMambaStatePoolHost
```

### 5.2 启用条件

| 环境变量 | 作用 | 默认 |
|----------|------|------|
| `HISIM_L2_ONE_POOL=1` | 启用 One Pool（动态共享） | 未设置（关闭） |
| `HISIM_L2_TOTAL_GB=N` | 设置总 DRAM 预算（也启用 Two Pools 基线） | 未设置（用 hicache_size） |
| `HISIM_L2_PAGE_TOKENS=N` | 每页 token 数 | 16 |
| `HISIM_L2_ALLOC_POLICY` | 分配策略：`exact` / `pow2` | `exact` |
| `HISIM_L2_MAMBA_MIN_SLOTS=N` | KDA 保底 slot 数（One Pool） | 0 |
| `HISIM_L2_BORROW_POLICY` | 借用策略：`self_first` / `other_first` | `self_first` |

**默认行为**：不设置任何 `HISIM_L2_*` 时，`configure_l2_dram_pool()` 返回 `None`，所有 `if _l2_pool is not None` 分支不执行，走 legacy 固定容量路径，**与合并前完全一致**。

### 5.3 与我们工作的关系

| 我们的工作 | One Pool 的关系 |
|-----------|----------------|
| Phase 2C.3.1 L3 修复 | One Pool 不触碰 L3 路径，保留 write-through |
| Phase 2C.3.2 record_access | One Pool 不修改 L2 lookup 路径 |
| TierCostAwareL2 | One Pool 保留 `l2_placement_policy` 决定淘汰顺序 |
| l2diff trace | One Pool 不影响 trace 和实验 |
| L2 策略等价性结论 | One Pool 改的是"有多少字节、分给谁"，不改"淘汰谁" |

**互补关系**：我们的工作回答了"L2 淘汰策略能否区分"（答案：当前不能，LRU 与 freq 完全相关）；One Pool 回答了"KDA 和 MLA 能否共享 DRAM 预算"（答案：可以，通过 PageExtentAllocator 两端分配）。

---

## 6. 次要注意事项

| 编号 | 注意事项 | 严重度 | 说明 |
|------|---------|--------|------|
| 1 | `_mock_start_loading` 不检查 `backup_from_device` 返回值 | 低 | 既有模式（L2=0 时也存在），L3 安全网兜底，非 One Pool 引入 |
| 2 | `HISIM_L2_MAMBA_SIZE` 双重语义 | 低 | legacy 模式是 slot 数；One Pool Two Pools 模式是 KDA region 字节计算依据 |
| 3 | L2=0 临时缓冲区计入 | 低 | 我们 2C.3.2 发现的问题在 One Pool 启用时同样存在 |
| 4 | push 认证 | 待处理 | 本地合并已完成，push 需用户在 VS Code 中手动执行 |

---

## 7. 当前状态

| 项目 | 状态 |
|------|------|
| 本地合并 | ✅ 完成（commit `7e5cd4c`） |
| 单元测试 | ✅ 13/13 通过 |
| 模块 import | ✅ 全部正常 |
| 双方改动保留 | ✅ 完整验证 |
| 远端 push | ⏳ 待用户在 VS Code 中执行 |
| 总 commit 数 | 94（合并前 93 + 合并 commit 1） |

### 合并后的 commit 历史

```
7e5cd4c  merge: integrate develop_dqy_l2_one_pool ← 合并 commit
├── 62b82d7  phase2c.3.2: deliver L2 Policy Differentiation report (我们的最新)
└── 4430405  l2: fix KDA host-index leak in bound mock pool (One Pool 最新)
```

---

## 8. 建议的后续步骤

1. **Push 到远端**：在 VS Code 中执行 `git push origin develop_dqy_evict_reuse`

2. **Baseline 回归验证**：清除 `__pycache__` 后跑一次 baseline 实验（不设 `HISIM_L2_*`），确认零回归：
   ```bash
   find src -name "__pycache__" -type d -exec rm -rf {} +
   # 跑 B2-Fork 或 l2diff_hot，不设 HISIM_L2_ONE_POOL / HISIM_L2_TOTAL_GB
   ```

3. **One Pool 对比实验**（可选）：
   ```bash
   # Two Pools 基线（同预算，静态切分）
   export HISIM_L2_TOTAL_GB=8
   # One Pool（同预算，动态共享）
   export HISIM_L2_TOTAL_GB=8 HISIM_L2_ONE_POOL=1
   ```

4. **碎片度量采集**：One Pool 提供了 `internal_frag_bytes` / `external_frag_bytes` 计数器，可加入 metrics 输出。

---

## 附录：合并命令完整记录

```bash
# 1. 确认当前分支和工作树状态
git status --short                    # → 干净
git branch --show-current             # → develop_dqy_evict_reuse
git log --oneline -1                  # → 62b82d7

# 2. Fetch 远端 One Pool 分支
git fetch origin                      # → new branch develop_dqy_l2_one_pool

# 3. 冲突预检（dry-run）
git merge --no-commit --no-ff origin/develop_dqy_l2_one_pool
# → Auto-merging sglang_hook.py / sglang_mock_class.py
# → Automatic merge went well
git merge --abort                     # 取消 dry-run

# 4. 执行合并
git merge --no-ff origin/develop_dqy_l2_one_pool -m "merge: integrate ..."
# → 8 files changed, 3550 insertions(+), 12 deletions(-)

# 5. 验证双方改动保留
grep -c "_l3_disable_val in" src/.../sglang_hook.py        # → 1 (我们的 L3 修复)
grep -c "P2C.3.2.*Record L2 lookup" src/.../sglang_hook.py # → 1 (我们的 record_access)
grep -c "def _setup_l2_dram_pool" src/.../sglang_hook.py   # → 1 (One Pool)
grep -c "admit_kda" src/.../sglang_mock_class.py           # → 2 (One Pool)

# 6. 运行单元测试
docker exec lfz-sglang bash -c '
  cd /home/d00954154/hisim
  PYTHONPATH=src python3 -m pytest test/test_l2_dram_pool.py -v
'
# → 13 passed

# 7. Import 验证
docker exec lfz-sglang bash -c '
  PYTHONPATH=src python3 -c "
    from hisim.simulation.l2_dram_pool import configure_l2_dram_pool
    import os; os.environ.pop(\"HISIM_L2_ONE_POOL\", None)
    print(configure_l2_dram_pool(8.0, 5760, 14653440))  # → None (默认关闭)
  "
'
# → None (零回归确认)
```
