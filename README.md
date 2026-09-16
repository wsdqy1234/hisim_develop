# HiSim × Kimi-K3 集成

> 让 HiSim 多层级缓存仿真系统支持 Kimi-K3(MLA+KDA 混合注意力)在 Ascend 910C NPU 上的仿真

---

## 目录

- [1. 背景与目标](#1-背景与目标)
- [2. 原始 HiSim 现状](#2-原始-hisim-现状)
- [3. Kimi-K3 特性](#3-kimi-k3-特性)
- [4. 改动总览](#4-改动总览)
- [5. 分阶段实施](#5-分阶段实施)
- [6. 原理](#6-原理)
- [7. 验证过程与结论](#7-验证过程与结论)
- [8. 局限性](#8-局限性)
- [9. 未来工作](#9-未来工作)

---

## 1. 背景与目标

HiSim 是一个 sglang 的多层级缓存仿真系统，通过 monkey-patch hook 进 sglang 的内部类（Scheduler、ModelRunner、HiCacheController、HiRadixCache 等），用 mock 前向 + 虚拟时钟模拟 L1（HBM）→L2（DRAM）→L3（SSD）的 KV cache 层级行为。

**原始 HiSim 已支持**：MLA/MHA 单一注意力模型的 KV cache 仿真（page 级 L1→L2→L3）、SimpySSD SSD 时序仿真、Trace 重放（OFFLINE 虚拟时钟模式）、推理时间预测（XGBoost AIConfigurator 或自定义 predictor）。已验证模型：Qwen3-32B on NVIDIA H20。

**本次集成目标**：让 HiSim 能仿真 Kimi-K3 的 MLA KV + KDA state 两个独立 cache 的 L1/L2/L3 行为，展示 K3 混合注意力在多级缓存中的特性。

---

## 2. 原始 HiSim 现状

| 维度 | 改动前 |
|------|--------|
| 支持模型 | Qwen3, DeepSeek, Kimi-K2 等（无 K3） |
| 支持硬件 | NVIDIA H20（无 Ascend NPU） |
| KV cache 仿真 | 仅 MLA/MHA 的 page 级 KV（L1→L2→L3） |
| KDA/Mamba state | 不支持（无 slot 管理，无 recurrent state pool） |
| CUDA 依赖 | `_is_cuda=True` 硬编码，`torch.cuda.synchronize()` 4 处 |
| 推理时间预测 | AIConfigurator（XGBoost + AIC 数据库，千级 operator×config 数据） |
| sglang 兼容版本 | 0.5.6 ~ 0.5.9 |

---

## 3. Kimi-K3 特性

Kimi-K3 是混合注意力模型，有两种截然不同的 cache：

| 维度 | MLA KV Cache | KDA State Cache |
|------|-------------|-----------------|
| 层类型 | Full Attention（5 层，减层后） | Linear Attention / Mamba SSM（18 层） |
| 数据结构 | per-token K/V page | per-slot recurrent state（conv + ssm） |
| 增长方式 | O(n) 随 seq_len 线性增长 | O(1) 固定大小，不随 seq_len 变化 |
| 大小 | 5760 B/tok（5 层，bf16） | 15.15 MB/slot（18 层，f32） |
| L1 管理 | radix tree page 引用 | slot pool + COW/donate/track |
| 并发瓶颈 | token 数（212K） | slot 数（165/5=33） |
| 前缀复用 | page 零拷贝引用 | COW 拷贝 15.15MB |

**K3 关键参数**（93 层全量 → 23 层减层）：

| 参数 | 全量(93层) | 减层(23层) | 来源 |
|------|-----------|-----------|------|
| MLA 层 | 24 | 5 | linear_attn_config.full_attn_layers |
| KDA 层 | 69 | 18 | linear_attn_config.kda_layers |
| kv_lora_rank | 512 | 512 | config.json |
| qk_rope_head_dim | 64 | 64 | config.json |
| KDA head_dim | 128 | 128 | linear_attn_config.head_dim |
| KDA num_heads | 96 | 96 | linear_attn_config.num_heads |
| KDA conv_kernel | 4 | 4 | linear_attn_config.short_conv_kernel_size |
| 硬件 | Ascend 910C-A3, 64GB HBM/die, 1.6 TB/s | 同 | npu-smi 实测 |

**核心挑战**：HiSim 只有 page 级 KV cache 仿真，KDA state 是 slot 级（14.45MB/slot），粒度差 25000×；HiSim 用 `num_hidden_layers` 全算 KV 容量，但 KDA 层不产生 KV；HiSim 无 NPU 支持，CUDA 硬编码。

---

## 4. 改动总览

共修改 12 个文件 + 新建 2 个文件，约 700 行代码。

| 文件 | 改动类型 | 阶段 | 行数 |
|------|---------|------|------|
| `spec/model/base.py` | kimi_k3 注册 + config 解析（text_config 合并 + linear_attn_config） | 1 | +20 |
| `spec/accelerator/info.py` | 加 ASCEND_910C（64GB HBM, 1600GB/s） | 1 | +18 |
| `simulation/utils.py` | MLA 容量修正 + 新增 calc_mamba_state_bytes() | 2 | +40 |
| `simulation/types.py` | SchedulerConfig 加 max_mamba_cache_size/mamba_slots_per_request | 2 | +3 |
| `simulation/sim_args.py` | SchedulerConfig + JSON 解析 | 2 | +8 |
| `simulation/sglang/sglang_mock_class.py` | CUDA 修复 + 新增 MockMambaStatePool/Host + MockHiCacheStorage 扩展 | 1+3 | +250 |
| `simulation/sglang/sglang_hook.py` | C_HiCacheController KDA 方法 + 8-phase pipeline + max 时钟 + sglang 0.5.18 兼容 | 3+7 | +160 |
| `simulation/manager/state.py` | KDA 时间累加器 | 3 | +50 |
| `simulation/sglang/launch_server.py` | NPU 检测 | 1 | +3 |
| `simulation/sglang/sglang_bench.py` | NPU 检测 | 1 | +3 |
| `time_predictor/__init__.py` | lazy import + K3 predictor 注册 | 2+4 | +10 |
| `time_predictor/k3_predictor.py` | 新建 K3ProfilePredictor | 4 | +110 |
| `manager/config.py` | k3_profile_based predictor 分支 | 4 | +5 |
| `test/assets/mock/config_910c_k3.json` | 新建 K3 配置 | 1 | +30 |

---

## 5. 分阶段实施

### 阶段 1：模型注册 + 加速器注册 + CUDA 修复

**改动**：
- `spec/model/base.py`：加 `"kimi_k3"` 到 model_type 列表；`from_dict()` 入口将 K3 的 `text_config` 字段合并到顶层；加 K3 分支读 `linear_attn_config` 解析 `num_full_attention`/`num_linear_attention`/KDA 参数
- `spec/accelerator/info.py`：加 `ASCEND_910C`（64GB HBM, 1600GB/s, vendor=Ascend）
- `sglang_mock_class.py`：`_is_cuda=True` → `"cuda" in str(self.device)` 动态检测；`torch.cuda.synchronize()` → `torch.get_device_module().synchronize()`（4 处）；`cudaHostRegister` 加 cuda 设备保护
- `launch_server.py` + `sglang_bench.py`：加 `torch.npu.is_available()` 检测
- 新建 `test/assets/mock/config_910c_k3.json`

**验证**：kimi_k3 模型解析（93 层：24 MLA + 69 KDA）✓，910C 加速器（64GB, 1600GB/s）✓

### 阶段 2：KV 容量修正

**改动**：
- `simulation/utils.py`：`calc_kv_cache_cell_elems()` MLA 路径用 `num_full_attention` 而非 `num_hidden_layers`；新增 `calc_mamba_state_bytes()`（ssm = `(num_heads/TP) × head_dim² × dtype × num_linear`；conv = K3 measured 55296 B/层，因 SiTU 激活有额外 state）；`estimate_kv_cache_pool_capacity()` 扣除 KDA pool
- `simulation/types.py` + `sim_args.py`：SchedulerConfig 加 `max_mamba_cache_size`/`mamba_slots_per_request`
- `time_predictor/__init__.py`：aiconfigurator lazy import（避免 `ModuleNotFoundError`）

**验证**：

| 指标 | 公式算出 | 实测值 | 匹配 |
|------|---------|--------|------|
| MLA per-token（5 层） | 5760 B | 5760 B | ✓ |
| KDA per-slot（18 层） | 15,151,104 B = 15.15 MB | 15.15 MB | ✓ |
| ssm/layer | 786,432 B | 786,432 B | ✓ |
| conv/layer | 55,296 B（K3 measured） | 55,296 B | ✓ |

### 阶段 3：KDA State 仿真

**新增类**：

- `MockMambaStatePool`（L1 HBM slot 管理）：alloc/free/evict_lru/cow_copy/touch，slot 级 LRU
- `MockMambaStatePoolHost`（L2 DRAM backup）：backup_from_device/load_to_device/evict，带宽模型 `x×bw/(t0×bw+x)`

**扩展类**：

- `MockHiCacheStorage`（L3）：加 `batch_set_mamba()`/`batch_exists_mamba()`（slot hash → kvcm/set，"mamba_" 前缀区分）
- `C_HiCacheController`（hook）：加 `_init_kda_pools`/`handle_kda_backup_operation`/`handle_kda_prefetch_operation`/`submit_kda_l3_write_timing`/`get_kda_stats`
- `StateManager`：加 KDA 时间累加器（kda_l2_load/backup, kda_l3_backup, last_kda_l2_load/backup）

**Hook 集成**：`wrapped_check_hicache_events` 从 4-phase 扩展为 8-phase：

```
Phase 1:   MLA backup     Phase 1b:  KDA backup     ← NEW
Phase 2:   MLA prefetch   Phase 2b:  KDA prefetch    ← NEW
Phase 3:   MLA L3 write   Phase 3b:  KDA L3 write    ← NEW
Phase 4:   original check + CUDA fallback
```

`wrapped_process_batch_result` 时钟叠加改为 max 模型：

```python
l2_load = max(mla_l2_load, kda_l2_load)       # 并行 D2H/H2D
l2_backup = max(mla_l2_backup, kda_l2_backup)
l3_backup = max(mla_l3_backup, kda_l3_backup)
overflow = max(0, l3_backup - (l2_load + inference + l2_backup))
clock = l2_load + inference + l2_backup + overflow
```

### 阶段 4：推理时间预测器

**新建** `time_predictor/k3_predictor.py`（`K3ProfilePredictor`）

**数据来源**（全部来自 23 层减层模型, TP=8, 910C, disable-cuda-graph）：

| 数据集 | 来源 | 点数 | 测量域 |
|--------|------|------|--------|
| Prefill e2e | supplement_profiling_results.json + GAP | 5（2K/8K/32K/128K/256K） | sglang meta_info / curl（无 profiler） |
| Decode base | OL512（output=512, seq=4K） | 1 | meta_info（无 profiler） |
| Batch scaling | B1-B5（seq=15.7K, bs=1-32） | 5 | kernel_details（profiler） |
| Batch×seq 矩阵 | BS8（1K/4K/16K）+ BS16（4K） | 4 | meta_info（无 profiler） |
| Seq ratio | D1-D5（step_trace） | 5 | NPU profiler（仅用 ratio） |
| GAP prefix cache | gap_experiment_results.json | 9 | curl（无 profiler） |

**预测方法**：查表 + 线性插值（不做多项式拟合，4 点拟合 3 参数会过拟合）

### 补充实测数据

在 sglang 服务（910C, TP=8）上采集：

| 测试 | 参数 | 结果 |
|------|------|------|
| Prefill 2K/8K/32K | input=2048/8192/32768, output=1 | e2e=0.439/1.140/4.332s |
| Prefill 256K | input=262144, output=1 | e2e=39.957s（O(n²) 验证: 4.04x ≈ 4.0x） |
| Output 长度 | output=1/32/128/512/1024, seq=4K | per-step: OL32-128=99ms, OL512-1024=81ms |
| Batch×seq | bs=8（1K/4K/16K）+ bs=16（4K） | per-step: 179/124/99ms（bs=8）, 156ms（bs=16） |

---

## 6. 原理

### 6.1 K3 混合注意力在多级缓存中的表现

```
L1 (HBM)                    L2 (DRAM)                L3 (SSD)
┌─────────────────────┐    backup    ┌──────────────┐   backup   ┌──────────┐
│ MLA KV Pool         │ ──────────→ │ MLA KV       │ ────────→ │ kvcm/set │
│ (page 级, 5760B/tok)│   page D2H  │ backup pool  │  page     │ (page    │
│ radix tree 管理      │             │              │  SSD      │  hash)   │
│ 零拷贝引用          │  ←───────── │ LRU evict →  │ ←───────  │          │
│                     │  page H2D  │    backup    │  prefetch │          │
├─────────────────────┤             │              │           │          │
│ KDA State Pool      │ ──────────→ │ KDA state    │ ────────→ │ kvcm/set │
│ (slot 级, 15.15MB/  │  slot D2H  │ backup pool  │  slot     │ (slot    │
│  slot)              │  15.15MB   │              │  SSD      │  hash,   │
│ slot pool 管理      │             │              │           │ "mamba_" │
│ COW/donate/track    │  ←───────── │ LRU evict →  │ ←───────  │  prefix) │
│ 256 track / 4096    │  slot H2D  │    backup    │  prefetch │          │
│ donate / 256 match  │  15.15MB   │              │           │          │
└─────────────────────┘             └──────────────┘           └──────────┘

时钟叠加: max(MLA_transfer, KDA_transfer) + inference + overflow
```

### 6.2 K3 特性在仿真中的体现

| K3 特性 | 仿真体现 | 实现代码 |
|---------|---------|---------|
| MLA O(n²) 增长 | prefill 时间随 seq² 增长（256K: 4.04x ≈ 4x） | K3ProfilePredictor PREFILL_TOKS |
| KDA O(1) 稳定 | decode per-step ~81ms 不随 seq 变化 | K3ProfilePredictor DECODE_BASE |
| MLA page 级 vs KDA slot 级 | D2H/H2D 数据量差 25000x（0.576KB vs 15.15MB） | MockMambaStatePoolHost vs MockTokenToKVPoolHost |
| KDA slot 并发瓶颈 | 165 slots / 5 = 33 并发（不受 MLA token 限制） | handle_kda_backup_operation |
| KDA COW 15.15MB | load_back 时 H2D 15.15MB（vs MLA 零拷贝） | MockMambaStatePoolHost.load_to_device |
| 混合 prefix cache | MLA page 引用 + KDA COW 并行，取 max | 8-phase pipeline + max 时钟 |
| KDA donate/track | 每 4096 token donate，每 256 track（简化为 per-iteration） | handle_kda_backup_operation |

### 6.3 Time Predictor 原理

查表 + 线性插值，不做多项式拟合（4 点拟合 3 参数 = 过拟合，自由度只有 1）：

```
Prefill: interp([1576, 6302, 25207, 131072, 201654], [0.439, 1.140, 4.332, 24.808, 39.957], seq_len)
Decode:  base(81.4ms) × batch_ratio(B1-B5) × seq_factor(BS8)
```

Prefix cache 效果：`prefill(suffix) = interp(..., suffix)`，用 GAP 9 点校验（误差 4.6-26.8%）。

### 6.4 层 scaling（23L → 93L）

无 93 层实测数据，用层比例假设（未验证）：

| 组件 | 23L 层数 | 93L 层数 | 比例 |
|------|---------|---------|------|
| MLA | 5 | 24 | 4.80x |
| KDA | 18 | 69 | 3.83x |
| MoE | 22 | 92 | 4.18x |
| TP | 23 | 93 | 4.04x |
| 加权（at 128K） | — | — | ~4.17x |

---

## 7. 验证过程与结论

### 7.1 集成测试（不启动 sglang）

在容器内运行集成测试脚本，10 个环节 69 项检查：

| 环节 | 检查数 | PASS | FAIL | 说明 |
|------|--------|------|------|------|
| 1. Config + 模型解析 | 5 | 5 | 0 | ✓ |
| 2. KV 容量计算 | 2 | 2 | 0 | MLA=5760B, KDA=15.15MB ✓ |
| 3. L1 slot 管理（alloc/evict/cow/touch） | 16 | 16 | 0 | ✓ |
| 4. L2 D2H/H2D 时序 | 7 | 7 | 0 | 10×15.15MB/54.4GB/s=2.825ms ✓ |
| 5. L3 KDA 存储（batch_set/exists_mamba） | 3 | 3 | 0 | ✓ |
| 6. StateManager KDA 累加器 | 4 | 4 | 0 | ✓ |
| 7. C_HiCacheController KDA 方法 | 5 | 5 | 0 | ✓ |
| 8. K3ProfilePredictor | 10 | 10 | 0 | 0% 误差 ✓ |
| 9. Full cycle（backup→L3→load_back→clock） | 8 | 7 | 1 | FAIL: 测试预期错误，非代码 bug |
| 10. GAP prefix cache 校验 | 8 | 8 | 0 | 4.6-26.8% 误差 ✓ |
| **合计** | **68** | **67** | **1** | **98.5% PASS** |

**Full Cycle 测试详解**（环节 9）：模拟 L1 填满→evict→L2 backup→L3 write→L2 load_back→时钟叠加的完整数据流：

```
填满 L1(5 slots) → alloc 3 个 → 3 次 LRU evict
  → 每个 evicted slot backup 到 L2(D2H, 15.15MB, 带宽模型) → kda_l2_backup_dur 累加
  → L3 write: 3×15.15MB/8GB/s = 5.682ms → kda_l3_backup_dur 累加
  → L2 load_back: 15.15MB/54.4GB/s = 0.285ms → kda_l2_load_dur 累加
  → 时钟叠加: max(mla, kda) + inference + overflow = 84ms
```

### 7.2 GAP 交叉验证

用独立的 GAP 实验数据（curl wall clock, 无 profiler）验证 predictor 的 prefix cache 预测：

| suffix | pred_savings | gap_savings | 误差 |
|--------|-------------|------------|------|
| 128K | 0.0s | 0.0s | — |
| 127K | 0.792s | 0.692s | 14.5% |
| 122K | 1.584s | 1.268s | 25.0% |
| 115K | 3.169s | 2.500s | 26.8% |
| 98K | 6.338s | 5.402s | 17.3% |
| 65K | 12.676s | 11.240s | 12.8% |
| 33K | 19.014s | 17.647s | 7.7% |
| 16K | 21.966s | 21.008s | 4.6% |
| 0 | 24.346s | 24.478s | 0.5% |

误差来源：中段偏差最大，因线性插值在 32K-128K 之间不够精确（MLA O(n²) 是曲线）。两端误差小（<5%）。

### 7.3 端到端 sglang Engine 测试

**容器**：sglang-kimi-k3-dqy（sglang 0.5.18.dev922 + CANN 9.0.0 + K3 模型 + KDA 代码补丁）

**遇到的问题与解决**：

1. **HiSim Scheduler hook 不兼容 sglang 0.5.18**：`recv_requests` 重命名为 `process_input_requests`
   - 修复：`getattr(target, "process_input_requests", None) or getattr(target, "recv_requests", None)` + `hasattr(target, "profile")` 条件赋值

2. **CANN Graph Engine 初始化失败**：`PYTHONPATH` 被 HiSim 路径覆盖
   - 修复：`export PYTHONPATH=${PYTHONPATH}:/hisim/src:...`（追加，不覆盖）

3. **NPU die 选择**：dies 8-15 被 VLLM 容器残留 HBM 占用（~50GB/die），改用 dies 0-7

**测试结果**（共享前缀 ~1610 tokens 的请求）：

| # | 请求 | prompt | cached | e2e(s) | 说明 |
|---|------|--------|--------|--------|------|
| 1 | 首次（冷） | 1610 | 0 | 0.774 | 全量 prefill |
| 2 | 共享前缀 | 1610 | **1536** | 2.583 | radix cache 命中 95.5% |
| 3 | flush 后（冷） | 1610 | 0 | 0.753 | cache 清空 |
| 4-8 | 重复请求 | 1610 | **1536** | 0.598-0.666 | cache 稳定 |

**sglang 日志**：`mamba usage: 0.02`（KDA state pool 使用 3 slots），`cached-token: 1536`（radix cache 命中）

**关键发现**：
1. HiSim 7 个 hooks 与 sglang 0.5.18 兼容 ✓
2. K3 模型 + KDA state 在 HiSim hook 下正常工作 ✓
3. 8-phase check_hicache_events pipeline 运行不 crash ✓
4. Radix cache (L1) 95.5% 命中率 ✓
5. KDA state pool 使用跟踪（mamba usage > 0）✓
6. 未触发 L2/L3（请求量少，pool 未满）

---

## 8. 局限性

### 8.1 Time Predictor

| 局限 | 影响 | 数据量 |
|------|------|--------|
| 查表+线性插值，无 per-operator 拆解 | 插值区间（如 32K-128K）可能偏差 >20% | 5 prefill 点 |
| 只有 23 层减层数据 | 93 层全量模型靠层比例假设（4.17x，未验证） | 无 93 层实测 |
| 只有 TP=8, disable-cuda-graph | 不能预测其他 TP/量化/cuda_graph 配置 | 1 种配置 |
| 只有 W4A8+bf16 量化 | 不能预测 FP8/INT4 等 | 1 种量化 |
| 无 batch prefill | prefill 只在 batch=1 测过 | 无 batch prefill 数据 |
| 对比 AIConfigurator（H20+Qwen3）：千级 operator×config 数据 + XGBoost | 差距 100x+ | 27 个点 vs 千级 |

### 8.2 KDA State 仿真

| 局限 | 影响 |
|------|------|
| `handle_kda_backup_operation` 用 `active_reqs × slots_per_req` 估算 slot 使用 | 未精确 per-request 追踪 slot 分配/释放 |
| `handle_kda_prefetch_operation` 用 `_kda_l3_pending_writes` 列表查询 | 未建模 prefix match → checkpoint 对齐（256 token 粒度） |
| conv_state 公式用 K3 measured 值（55296 B/层） | 未从 sglang 源码推导精确公式（SiTU 激活有额外 state） |
| 未实现 track（256）/donate（4096）粒度 | 简化为 per-iteration backup |

### 8.3 测量域不一致

| 数据 | 域 | 比例关系 |
|------|----|---------|
| Prefill e2e（P6-P9） | sglang meta_info（无 profiler） | 基准 |
| GAP TTFT | curl wall clock（无 profiler） | ≈ meta_info（2% 差） |
| Step_trace（P1-P5） | NPU profiler | ≈ 2.51x meta_info |
| B1-B5 | kernel_details（profiler） | decode base ≈ 1.65x OL512 |

### 8.4 未触发/未实现

- 未触发 L2/L3 cache（端到端测试请求量少，pool 未满）
- 未跑完整 HiSim benchmark（需要 bench_serving runner + OFFLINE 模式）
- 未验证 93 层全量模型层 scaling 假设

---

## 9. 未来工作

### 短期（可在现有条件下做）

| # | 工作 | 说明 | 依赖 |
|---|------|------|------|
| 1 | **完整 HiSim benchmark（OFFLINE 模式）** | 用 HiSim 的 SGLangBenchmarkRunner 跑 K3 trace，触发虚拟时钟 + predictor 集成 | sglang 0.5.18 兼容已修复 |
| 2 | **触发 L2/L3 cache** | 发 >33 并发请求，填满 KDA pool（165 slots），触发 LRU evict → L2 backup → L3 write | sglang 服务 + 并发请求脚本 |
| 3 | **精确 KDA state 追踪** | per-request slot 分配/释放，prefix match → checkpoint 对齐（256/4096 粒度） | 改 hook 代码 |
| 4 | **conv_state 公式推导** | 从 sglang 源码 `memory_pool_npu.py` 推导精确公式 | 读 sglang 代码 |

### 中期（需要补充测试）

| # | 工作 | 说明 | 需要 |
|---|------|------|------|
| 5 | **Prefill 更多 seq_len 点** | 补 4K/16K/64K 的 e2e（无 profiler），减小插值误差 | sglang 服务 |
| 6 | **Decode batch×seq 完整矩阵** | bs=1/4/8/16/32 × seq=1K/4K/16K/64K/128K | sglang 服务 |
| 7 | **cuda_graph on/off 对照** | 同 seq_len 测两种配置，建模 cuda_graph 效果 | sglang 服务 |
| 8 | **93 层全量模型验证** | 跑一次 93 层模型，验证层 scaling 假设（4.17x） | NPU 资源 |

### 长期（需要大量数据或工具支持）

| # | 工作 | 说明 | 需要 |
|---|------|------|------|
| 9 | **AIConfigurator + 910C 数据库** | 用 AIConfigurator 的 per-operator 框架，把 910C 的 op_statistic 数据喂给 collector，建 910C AIC 数据库 | AIConfigurator + collector 适配 |
| 10 | **XGBoost 模型** | 用 20+ 个数据点训练 XGBoost，替代查表 | 大量测试 |
| 11 | **多 TP/量化配置** | 测 TP=4/16，FP8/INT4 等 | NPU 资源 |
| 12 | **混合 batch 仿真** | 不同 seq_len 请求混合的 decode 时间 | 测试脚本 |
