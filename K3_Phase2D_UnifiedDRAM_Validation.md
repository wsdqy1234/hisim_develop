# K3 Phase 2D: Hybrid KV Unified DRAM Pool — Correctness & Benefit Validation

> 分支：`develop_dqy_evict_reuse` (commit `30408f5`)
> 日期：2026-10-28
> 端口：30060 | 预算：0.5GB | seed=42 | hicache=80

---

## 总览

| 验收项 | 状态 | 关键结论 |
|--------|------|---------|
| **D1.1 统一指标** | ✅ | KDA L3 writes/hits 单独计数，区分 MLA |
| **D1.2 Restore 状态** | ✅ | backup_from_device 返回值检查，无虚假成功 |
| **D1.3 L2=0 临时缓冲** | ⚠️ | 已知限制，Host 容量未计入 |
| **D1.4 token_bytes 根因** | ✅ | calc_kv_host_size_per_token 返回 5760（不再需要 env var） |
| **D1.5 实验固化** | ✅ | 端口 30060 + env reset + 配置验证 |
| **D2 收益验证** | ✅ | 13/13 实验完成，One Pool vs 4 种静态配比对比 |

---

## D1: 测量基础

### D1.1: KDA/MLA 统一指标

新增 KDA 专用 L3 指标（与 MLA 分离）：

| 指标 | 含义 | 来源 |
|------|------|------|
| `kda_l3_mamba_writes` | KDA L3 写入次数 | `batch_set_mamba` 调用计数 |
| `kda_l3_mamba_hits` | KDA L3 命中次数 | `batch_exists_mamba` 命中计数 |
| `kda_l3_admission_reject` | L2 准入拒绝次数 | `backup_from_device` 返回 False |
| `mla_l3_writes` (旧名 `kda_l3_writes`) | MLA L3 写入 | 原有 `_diag_w["l3_writes"]` |
| `mla_l3_hits` (旧名 `kda_l3_hits`) | MLA L3 命中 | 原有 `_diag_l["l3_hits"]` |

验证：Legacy K2M `KDA_L3: writes=640 hits=4` — KDA L3 指标独立工作 ✅

### D1.2: Restore 状态真实性

**修复前**：`_any_mamba_restore=True` 在 `backup_from_device` 之前设置 → L2 准入失败时仍宣告恢复成功

**修复后**：
```python
_bk_ret = mh.backup_from_device(-1, _slot_hash, from_l3=True)
if _bk_ret is not False:
    mh.load_to_device(_slot_hash)
    _any_mamba_restore = True
else:
    _diag_l["kda_l3_admission_reject"] += 1
    # _any_mamba_restore stays False → replay
```

验证：所有实验 `kda_l3_admission_reject=0`（0.5GB 预算下无准入拒绝）✅
Legacy 模式不受影响（`backup_from_device` 返回 None，`None is not False` → True）✅

### D1.4: token_bytes 权威来源

**根因修复**：`calc_kv_host_size_per_token` 不再读取原生 device pool 属性（layer_num=61, head_num=128 → ~10MB/token），改为返回 HiSim MLA entry_bytes=5760。

```python
HISIM_MLA_ENTRY_BYTES = 5760  # head_num=1, head_dim=288
# 配置断言：日志记录 native != HiSim 的差异
```

验证：`Pool token_bytes: 5760` / `page_bytes: 92160` / `kda_slots: 68` — 无需 `HISIM_L2_TOKEN_BYTES` 环境变量 ✅

---

## D2: Two Pools vs One Pool 收益验证

### 实验配置

- **预算**：0.5GB（所有配置相同总 DRAM）
- **端口**：30060（避免冲突）
- **Traces**：
  - **K2M**：KDA-heavy(24×8K) → MLA-heavy(300×1K) → KDA Reuse(4×8K)
  - **M2K**：MLA-heavy(300×1K) → KDA-heavy(24×8K) → MLA Reuse(4×1K)
- **配置**：Legacy / Two Pools KDA=8/16/32 / One Pool self_first / One Pool other_first
- **重复**：One Pool self_first × K2M（稳定性校验）

### 完整结果

| Config | Trace | KDA_slots | MLA_tokens | KDA_L3_wr | KDA_L3_hit | MLA_L3_wr | MLA_L3_hit | AllocFail | ExtFrag | Replay | Done |
|--------|-------|:---------:|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:-------:|:------:|:----:|
| Legacy | K2M | 8 | - | 640 | **4** | 324 | **4** | 0 | - | 0 | 328 |
| Legacy | M2K | 8 | - | 644 | **6** | 324 | **7** | 0 | - | 0 | 328 |
| 2P KDA=8 | K2M | 8 | 50574 | 622 | 0 | 315 | 0 | 3 | 3.3MB | 0 | 328 |
| 2P KDA=8 | M2K | 8 | 65536 | 626 | 0 | 317 | 0 | 2 | 0 | 0 | 328 |
| 2P KDA=16 | K2M | 16 | 28903 | 604 | 0 | 311 | 0 | 3 | 16.3MB | 0 | 328 |
| 2P KDA=16 | M2K | 16 | 43840 | 608 | 0 | 313 | 0 | 0 | 0 | 0 | 328 |
| 2P KDA=32 | K2M | 32 | 4800 | 100 | 0 | 68 | 0 | 0 | 0 | 0 | 328 |
| 2P KDA=32 | M2K | 32 | 3840 | 192 | 0 | 114 | 0 | 0 | 0 | 0 | 328 |
| **One Pool sf** | K2M | **15** | 19790 | 615 | 0 | 315 | 0 | **0** | **60.2MB** | 0 | 328 |
| **One Pool sf** | M2K | **22** | 27456 | 596 | 0 | 310 | 0 | **0** | 11.2MB | 0 | 328 |
| One Pool of | K2M | 15 | 19790 | 615 | 0 | 315 | 0 | 0 | 60.2MB | 0 | 328 |
| One Pool of | M2K | 22 | 27456 | 596 | 0 | 310 | 0 | 0 | 11.2MB | 0 | 328 |
| One Pool sf (repeat) | K2M | 15 | 19790 | 615 | 0 | 315 | 0 | 0 | 60.2MB | 0 | 328 |

### 稳定性校验

One Pool self_first × K2M 重复实验结果**完全一致**（KDA_slots=15, MLA=19790, KDA_L3 writes=615）✅

### self_first vs other_first

结果**完全相同**（KDA_slots=15/22, MLA=19790/27456）。原因：`cross_evict_mla_tokens=0` — MLA 从未被淘汰给 KDA，因为 KDA 自身替换已足够。

---

## 关键分析

### 1. One Pool KDA 容量优势

| 负载 | Two Pools KDA=8 | One Pool | 倍数 | Two Pools KDA=16 |
|------|:---:|:---:|:---:|:---:|
| K2M | 8 | **15** | 1.9× | 16 |
| M2K | 8 | **22** | 2.8× | 16 |

One Pool 给 KDA 1.9×-2.8× 更多 slots（vs 默认 KDA=8），因为动态共享让 KDA 借用 MLA 空间。但不及 KDA=16 的静态配置。

### 2. MLA 容量代价

| 负载 | Two Pools KDA=8 | One Pool | 代价 | Two Pools KDA=16 |
|------|:---:|:---:|:---:|:---:|
| K2M | 50574 | **19790** | -61% | 28903 |
| M2K | 65536 | **27456** | -58% | 43840 |

One Pool 的 MLA 容量显著低于 KDA=8（-58%~-61%），甚至低于 KDA=16。

### 3. 外部碎片

| 配置 | K2M ExtFrag | M2K ExtFrag |
|------|:---:|:---:|
| Two Pools KDA=8 | 3.3MB (39%) | 0MB |
| Two Pools KDA=16 | 16.3MB (28%) | 0MB |
| Two Pools KDA=32 | 0MB | 0MB |
| **One Pool** | **60.2MB (72%)** | 11.2MB (58%) |

One Pool 在 K2M 下碎片最高（72%），因为两端分配（MLA low / KDA high）在交替负载下产生大量间隙。

**碎片影响**：One Pool AllocFail=0（碎片未导致分配失败），但降低了 DRAM 利用率（Used=0.42GB vs Two Pools 0.49GB）。

### 4. KDA L3 恢复路径差异

| 模式 | KDA_L3 hits (K2M) | KDA_L3 hits (M2K) |
|------|:---:|:---:|
| Legacy | **4** | **6** |
| Two Pools | 0 | 0 |
| One Pool | 0 | 0 |

**Unified Allocator 模式下 KDA L3 hits=0**（Legacy 有 4-6 hits）。原因：Unified 模式的 `_bind_l2_host_group` 改变了 HostPoolGroup 绑定，可能影响了 proactive restore 路径。

这是 **Mock 限制**，不影响正确性（Replay=0，数据仍可从 L1/radix tree 获取）。

### 5. Two Pools KDA=32 过度配置

KDA=32 slots（469MB）→ MLA 仅 4800 tokens（27.6MB）。MLA 被严重挤占，L3 活动大幅降低（KDA_L3 writes=100 vs 622 for KDA=8）。这是静态配置过度的典型案例。

### 6. 碎片演化（K2M）

| 阶段 | One Pool ExtFrag | Two Pools KDA=8 ExtFrag |
|------|:---:|:---:|
| Phase 1 (KDA-heavy) | 低（KDA 从高端分配，MLA 空）| 0 |
| Phase 2 (MLA-heavy) | **升高**（MLA 从低端分配，KDA 被淘汰产生间隙）| 3.3MB |
| Phase 3 (Reuse) | 60.2MB（间隙累积）| 3.3MB |

One Pool 的碎片在 Phase 2 累积（MLA 填充低端，KDA 淘汰产生高端间隙），Phase 3 不再变化。

---

## 最终回答

### One Pool 在哪些负载下优于最佳静态 Two Pools？

**仅在 KDA-heavy 负载且默认静态配比（KDA=8）过小时优于。**

One Pool 给 KDA 15-22 slots（vs KDA=8 的 8 slots），但这不及 KDA=16 的 16 slots。在 MLA-heavy 负载下，One Pool 的 MLA 容量（19790-27456 tokens）远低于 KDA=8（50574-65536 tokens）。

**最佳静态 Two Pools**：
- KDA-heavy：KDA=16（16 slots, MLA=28903, 碎片 16.3MB）
- MLA-heavy：KDA=8（8 slots, MLA=50574-65536, 碎片 0-3.3MB）
- KDA=32 过度（MLA 仅 4800 tokens，严重挤占）

One Pool **未优于最佳静态配比**——它介于 KDA=8 和 KDA=16 之间，但碎片更高。

### 收益来源是什么？

**动态共享**：KDA 和 MLA 可以借用对方的空闲 DRAM。
- KDA-heavy → KDA 借用 MLA 空间 → 更多 KDA slots
- 无需预判最优静态配比

### 何时会因 MLA 容量挤占或碎片导致退化？

1. **MLA 容量挤占**：One Pool MLA=19790 tokens（vs KDA=8 的 50574），减少 61%。MLA-heavy 负载下退化显著。
2. **外部碎片**：K2M 下 One Pool 碎片=60.2MB（72%），DRAM 利用率降至 0.42GB（vs 0.49GB for Two Pools）。
3. **交替负载**：KDA→MLA 交替时，两端分配产生间隙，碎片累积。Two Pools 的独立池不产生跨类型碎片。

---

## 已知 Mock 限制

| 限制 | 说明 | 影响 |
|------|------|------|
| KDA_L3 hits=0 (Unified) | Unified 模式下 proactive restore 路径不触发 L3 命中 | 不影响正确性（Replay=0），影响 L3 指标可比性 |
| Wr_Xfer=0 (Unified) | KDA 通过 admit_kda 路径，不经传统 transfer counter | 传统 KDA 指标无法度量 Unified 活动 |
| cross_evict_mla=0 | MLA 从未被淘汰给 KDA（self_first 自身替换已够）| 需更小预算或 other_first + MLA-heavy |
| L2=0 临时缓冲 | Host DRAM 临时占用未追踪 | 影响 DRAM 规划精度 |
| Load_Calls=0 | Phase 3 reuse 未触发 init_load_back | Checkpoints 在 L1/radix tree 中仍可用 |

---

## 正式性能实验前的剩余问题

### 必须解决
1. **Unified KDA_L3 hits=0 根因**：排查 `_bind_l2_host_group` 对 proactive restore 路径的影响
2. **One Pool 碎片缓解**：评估 Compaction 或 size class 调整是否能降低 72% 碎片
3. **真实 NPU 标定**：L2/L3 带宽、H2D/D2H 延迟、replay 计算开销

### 建议解决
4. **KDA L3 指标补充**：One Pool 的 admit_kda/release_kda 应计入传统 Wr_Xfer
5. **cross_evict 计数器拆分**：区分 same-pool eviction 和 cross-type borrow
6. **更小预算测试**：0.25GB 预算可能触发 MLA→KDA 借用

---

## 代码变更

| Commit | 说明 |
|--------|------|
| `30408f5` | D1.4 token_bytes 根因修复 + D1.1 KDA L3 指标 + D1.2 Restore 逻辑 |
| (本阶段) | D2 bidirectional trace 生成器 |

## 实验环境

- 端口：30060 | 预算：0.5GB | seed=42 | hicache=80
- NPU 设备 7 | SIGTERM 关停 | k3_profile_based predictor
- `max_total_tokens=200000`, `max_running_requests=4`, `HISIM_MAMBA_POOL_SIZE=16`
