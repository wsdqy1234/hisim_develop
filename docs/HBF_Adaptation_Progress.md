# HiSim HBF L1.5 适配 — 进展总结

> 日期：2026-10-27  
> 分支：`develop_dqy_0.5.19_qwen3_hbf`（从 `develop_lfz_0.5.19` 切出）  
> 容器：`dqy-hisim-qwen3`（镜像 `quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3`）

---

## 一、总体目标

在 HiSim 多层级 KV Cache 仿真框架中引入 **HBF（High-Bandwidth Flash）** 作为 L1.5 高带宽大容量缓存介质，与 L2 Host DRAM **并列**，替代部分 DRAM 容量，目标在基本保持端到端性能的条件下**降低 70%+ 的 Host KV DRAM 需求**。

HBF 介质特征（来自 HBFSim `nominal.json`，为假设配置非实测）：

| 参数 | 值 | 对照 |
|---|---|---|
| 容量 | 1 TiB | DRAM ~几百GB |
| 聚合读带宽 | 512 GB/s | DRAM 16GB/s（HiSim config）→ HBF 高 32 倍 |
| 读延迟 | 10 μs | DRAM t0≈6.67μs → HBF 略高 |
| 写延迟 | 100 μs | — |

---

## 二、已完成工作阶段

### 阶段 0：Qwen3 仿真验证 ✅

**目标**：验证 `develop_lfz_0.5.19` 分支支持 Qwen3 在 sglang 0.5.19 上的仿真。

**结果**：
- 分支**未被 kimi-k3 污染**（零 k3/mamba/kda 代码）
- 端到端仿真成功：`completed=4, TPOT=457.6ms, TTFT=2.31ms`
- TPOT 与 history.md 记录的 456ms 几乎一致
- L2 分层缓存（UnifiedRadixCache + hierarchical KV host pool）正常工作

**关键发现**（正确运行方式，详见 `docs/Qwen3_Simulation_Guide.md`）：
- 必须用 `launch_server.py`（server 模式），不能用 `SGLangBenchmarkRunner`（in-process，hook 时机问题）
- 必须 `--device cpu`（mock tensor 在 CPU，hook 处理 `_init_cpu_threads_env`）
- 必须 `TORCH_TRANSFER_TO_NPU=0`（禁 transfer_to_npu 重定向，避免 NPU 不支持 uint64）
- **不能**设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`（会导致 `is_npu()` raise RuntimeError）
- 用 NPU 7 卡（`ASCEND_RT_VISIBLE_DEVICES=7`）

### 阶段 0.5：源码调查与架构设计 ✅

**调查范围**：SGLang 0.5.19 原生状态机、HiSim hook 层、kvcm C++ 后端、HBFSim backend 能力。

**关键源码确认事实**：

#### 1. SGLang 0.5.19 多层级缓存状态机

- 0.5.19 使用 `UnifiedRadixCache` + `HybridCacheController` + `HostPoolGroup`（多 pool 门面）
- **DRAM 槽位分配点**：`HybridCacheController.write()`（hybrid_cache_controller.py:312）的 `mem_pool_host.alloc()`，在 L1 驱逐时（`_execute_kv_backup`）触发，**远早于** `handle_backup_operation`（L2→L3）
- **`host_value` 生命周期**：由 `commit_hicache_transfer(BACKUP_HOST)` 创建（`tree_component.py:688`），`evict_component(HOST)` 清除（`full_component.py:184`）
- **对象位置索引分散在三套系统**：
  - L1: radix 树节点 `ComponentData.value`
  - L2: radix 树节点 `ComponentData.host_value`
  - L3: `storage_backend.batch_exists(page_hashes)` 内容寻址，每次现查
- **副本语义**：write-through 时 token 可同时存在 L1+L2+L3（L3 写 ack 后 L2 副本保留作可回收副本）

#### 2. Host Pool 引用关系（方案 A 的关键）

确认了所有直接持有 anchor host pool 引用的位置：
1. `cache_controller.mem_pool_host`（HostPoolGroup 门面）
2. `cache_controller.storage_host_pool`（`HybridCacheController.__init__:139` 独立缓存）
3. `storage_backend.mem_pool_host`（`register_mem_pool_host` 注册）
4. `_l2_transfers`（每次 backup 直接访问 `anchor_entry.host_pool`）
5. `get_data_page`/`set_from_flat_data_page`（通过 `storage_host_pool`）

**结论**：仅替换 `mem_pool_host` 不足；必须同时替换 `storage_host_pool` 和重新注册 `storage_backend`。

#### 3. HBFSim 集成边界

- HBFSim 是 NVIDIA GPU 侧仿真器（PTX 重写 + eBPF 拦截），**C API 需 CUDA** → HiSim 不能用
- **但** `fast` 时序公式是纯算术：`max(base_latency, ceil(bytes*1e9/bandwidth))` — 5 行 Python 可移植
- `reference`(MQSim) 后端 CPU-only、可从 GPU 路径分离（`MqsimOnlineEngine` C++ 类）
- 8GiB HBM cache 是 Capacity-mode 页缓存，**非**媒体命中模型，Timing 模式不参与
- 写路径缺陷在高层（PTX 注入层），MQSim reference 本身正确

#### 4. kvcm 后端

- kvcm `RadixTreeIndex` 已支持多 tier 位置链（`AppendBlockLocation`）
- `TierWriteMode`: WRITE_THROUGH / WRITE_THROUGH_SELECTIVE
- HiSim ↔ kvcm 通过 `storage_backend.batch_set`/`batch_exists_per_tier` 交互
- HBF 可独立于 kvcm（按 L2 的 `host_value` 模式管理，kvcm 只管 L3）

### 阶段 0.5 PoC：Virtual Adapter 验证 ✅

**选定方案**：方案 A — Virtual/Tiered Host Pool Adapter

**PoC 结果**：8/8 单元测试通过 + 集成测试 adapter 安装成功 + metrics 与基线一致（100% 透明）。详见前一版进展文档。

### Phase 1：配置 + HBF 时序骨架 ✅

**完成内容**：

| 组件 | 文件 | 状态 |
|---|---|---|
| `HBFConfig` dataclass | `types.py` | ✅ 参数来自 HBFSim nominal.json |
| `PlatformConfig.hbf`/`enable_hbf` | `types.py` + `sim_args.py` | ✅ 双处定义 |
| Config 解析 (`hbf` 段) | `sim_args.py` + `config.py` | ✅ `_parse_hbf_config()` |
| `_hbf_transfer_duration` (fast 公式) | `sglang_hook.py` | ✅ `max(latency, bytes/bw)` |
| `_record_hbf_backup/load_timing` | `sglang_hook.py` | ✅ 记入 StateManager |
| StateManager HBF dur 字段 | `state.py` | ✅ inc/pop/last/reset |
| Config 示例 | `config_qwen3_hbf.json` | ✅ 64GB HBF, 512GB/s |
| 时序单元测试 | `test_hbf_timing.py` | ✅ 5/5 PASS |

**时序测试验证**：
- fast 公式 `max(latency, transfer)` 正确（非 sum，重叠感知）
- 小传输：延迟主导（10μs）；大传输：带宽主导（100MB→204.8μs）
- 交叉点：5.12MB（latency == transfer）
- `enable_hbf=False` → `_hbf_transfer_duration` 返回 0（no-op，向后兼容）

**基线回归**（`enable_hbf=False`，无 `HISIM_HBF_ENABLE`）：
- PlatformConfig 正确显示 `enable_hbf=False, hbf=None`
- completed=4, TPOT=457.6（与基线一致）
- 无 HBF-DEBUG 输出（adapter 未安装）

### Phase 2b-2：L3 Write-through 闭环 + Host Load 真实归因

**A. Host Load 真实来源归因与时序验证（已完成 ✅）**

1. ✅ **真实 host_value 归因**：Hook `match_prefix`，从 `best_match_node` 向上走到 `last_device_node`，收集 `node.component_data[FULL].host_value`（虚拟索引），按 `dram_offset` 拆分为 HBF/DRAM。
   - **不再使用容量比例估算**——改为遍历 Radix Tree 收集真实 `host_value` 张量
   - `_collect_host_load_split(tree_core, result, dram_offset)` 实现
   - `_HOST_LOAD_SPLIT` dict 按 `rid` 缓存，scheduler hook 消费

2. ✅ **反例测试**（`test/test_hbf_load_attribution.py`，8/8 PASS）：
   - All-HBF hit → only HBF Load（DRAM=0）
   - All-DRAM hit → only DRAM Load（HBF=0）
   - Mixed HBF/DRAM → 按真实位置拆分（4 HBF + 3 DRAM）
   - 10ms HBF 延迟但只命中 DRAM → **无 HBF Overflow** ✅
   - 10ms HBF 延迟且确实命中 HBF → **Overflow=9ms** ✅
   - Overlap 语义：load < last_inference → overflow=0；first round → full load
   - Token 顺序保持（链式遍历正确）

3. ✅ **集成验证**（eviction_test_8req 数据集，L1=20）：
   - **HOST-LOAD 事件触发**：8 个请求，每个 `host_hit=4 HBF=4tk DRAM=0tk`（HBF_LOGICAL_SIZE=10000）
   - **HBF/DRAM 位置正确切换**：HBF=10000 → 全 HBF hit（HBF=4tk）；HBF=5 → 全 DRAM hit（DRAM=4tk）
   - **TTFT 受控变化**：10μs → TTFT=2.7ms；10ms → TTFT=12.7ms（**+10ms = HBF 读延迟**）
   - **combined_load_dur bug 修复**：时钟推进从 `hicache_l2_load_dur` → `combined_load_dur`（含 HBF）

4. ✅ **基线回归**：HBF off → TPOT=50.6, TTFT=2.7（与 HBF 10μs 相同，10μs 可忽略）

**B. L3 Write-through 生命周期闭环（部分完成）**

1. ✅ **L3 Storage 开启**：`config_hbf_l3.json` + `--hicache-storage-backend sim`
   - `Overrode kvcm storages with 1 HiSim tiers`
   - `Creating hijacked cache storage backend` → MockHiCacheStorage

2. ✅ **HBF/DRAM → L3 Write-through**：
   - `batch_set=18-19`（L3 写入 18-19 次）
   - `handle_backup_operation()` → `_page_backup()` → `storage_backend.batch_set()`
   - HBF adapter 与 storage_backend 协同（`Re-registered adapter with storage_backend`）

3. ✅ **副本保护**：`handle_backup_operation` → `ack_backup_queue.put(operation)`（Ack 语义）
   - 写入完成后才放入 ack queue

4. ✅ **L3 Read-back 闭环验证**（已修复，原根因是 prefetch_threshold）：
   - 原 `batch_exists=0` 根因：`prefetch_threshold=256`（默认）阻断短 prefix（5 tokens < 256）
   - 修复：`--hicache-storage-backend-extra-config '{"prefetch_threshold": 1}'`
   - **不需要 UnifiedCacheLinker** — 原生 HiCache prefetch 路径正常工作
   - 验证结果：HBF+L3 `hits=24, prefetched=24tk`；Baseline+L3 `hits=64, prefetched=64tk`

5. ✅ **无 double-free / false hit**：
   - HBF pool `assert not _slot_used[select].any()`（防 double-alloc）
   - DRAM pool `assert _slot_used[indices].all()`（防 double-free）
   - HOST-LOAD 真实归因防止 false hit（不按比例估算）

**关键配置文件**：
- `test/assets/mock/config_hbf_l3.json`：含 storage_tiers（local_nvme）的 HBF+L3 配置
- `test/assets/dataset/eviction_test_8req.jsonl`：8 请求确定性驱逐测试
- `test/assets/dataset/eviction_test_16req.jsonl`：16 请求大压力测试

**全部测试**：26 单元测试通过（8+5+5+8），SGLang 0 修改，HBF-off 基线不回归。

**HBF Backup 时序（已验证）**：

1. ✅ **HBF BACKUP 事件确认**：`[HBF-BACKUP] Ntk→HBF dur=Xμs` 日志出现，8 次 backup 事件，timing 正确记录
2. ✅ **ConfigManager.hbf 修复**：env-var 安装时同步设置 HBFConfig（之前 bug：timing 全返回 0）
3. ✅ **HBF timing 用 max（并行）不是 sum**：同一轮多个 backup 事件并行，`inc_hicache_hbf_backup_dur` 改为 max 语义
4. ✅ **HBF/DRAM backup 分介质日志**：每次 backup 打印 `[HBF-BACKUP]` 或 `[DRAM-BACKUP]`，含 token 数 + bytes + duration

**HBF Load 时序独立建模（已验证）**：

5. ✅ **根因定位**：HiSim simulation 模式下 `load_queue=0`（不做真实 load_back 传输）。仿真通过 `host_hit_length` 元数据知道 L2 命中，但不调 `mock_start_loading`。
6. ✅ **L2 Load 时序消费点确认**：`wrapped_process_batch_result`（line 2562-2567）：
   ```python
   # OVERLAP mode: load overflow delays clock
   step_global_clock(max(hicache_l2_load_dur - last_inference_dur, 0))
   step_global_clock(current_inference_dur)
   ```
   **L2 load timing 确实被仿真时钟消费**——但之前 `hicache_l2_load_dur=0` 因为 `mock_start_loading` 不调。
7. ✅ **修复方案**：从 `host_hit_length`（prefix match 元数据）计算 load timing，不从 `mock_start_loading`。在 `host_hit_length > 0` 时通过 `_collect_host_load_split` 遍历 Radix Tree 收集真实 `host_value` 虚拟索引，按 `dram_offset` 拆分 HBF/DRAM，记录 `_record_hbf_load_timing` / `_record_l2_load_timing`。
8. ✅ **独立 PoC 验证**（`test/test_hbf_load_poc.py`，5/5 PASS）：

| 测试 | 延迟 | host_hit | HBF load | DRAM load | overflow | TPOT 影响 |
|---|---|---|---|---|---|---|
| Normal | 10μs | 20 | 10μs | 16.3μs | 0μs | 无（load < inference, 完全重叠） |
| High | 10ms | 20 | 10ms | 0.02ms | **9ms** | **TPOT +9ms**（load > inference, overflow） |
| Overlap A | 500μs | 20 | — | — | 0μs | 无（load=500μs < inference=2000μs） |
| Overlap B | 500μs | 20 | — | — | **400μs** | TPOT +400μs（load > inference=100μs） |
| Baseline | — | 20 | 0 | 54.9μs | — | 无 HBF timing |

**关键结论**：
- HBF Load timing **独立于 L3 Storage**——从 `host_hit_length` 元数据计算，通过仿真时钟 `step_global_clock` 消费
- **重叠感知模型**：load 与 inference 并行，只有 overflow（load > inference）延迟下一轮 → 影响 TPOT
- 不机械累加 load 到 TPOT——只有超出 inference 的部分才影响
- 10ms 延迟 → overflow=9ms → TPOT 增加 9ms（验证时序正确消费）

**集成测试限制**：hisim_collection 数据集（4 个短请求）太小，L1 从不满 → `host_hit_length=0` → 集成测试无法触发 HBF Load。需更大工作负载（Qwen3-32B 或多轮负载）在 Phase 2b-5 验证。

**并行时序假设记录**（为 Phase 3 MQSim 预留）：
- 当前简化模型：同轮 backup/load 取 max（假设完全并行）
- HBF 内部多个读写请求可能共享带宽及队列 → 不能始终假设独立并行
- 已记录：请求数量、总字节数、模拟起止时间（通过 HBF-STATS 日志）
- Phase 3 用 MQSim 的 queue_depth/channel 并发模型替代简化 max

---

## 三、修订架构图

```
                          SGLang UnifiedRadixCache (0.5.19)
                               │
                    HybridCacheController
                               │
                     TieredHostKVCache ★adapter (方案A)
                      ┌────────┴────────┐
                      │                 │
                 HBF Pool            DRAM Pool (原有 anchor)
                 (MockHBFPool)       (SGLang 原生 host pool)
                 容量: HBFConfig     容量: hicache_ratio 控制
                 时序: HBFTiming     时序: _l2_transfer_duration
                 索引: 虚拟[0,hbf)    索引: 虚拟[hbf,total)
                      │                 │
                 放置策略路由 ★新增      │
                 V0: 比例分流           │
                 V1: 对象特征感知       │
                 V2: 代价感知(预留)     │
                      │                 │
                      └────────┬────────┘
                      共同溢出 ↓
                      storage_backend.batch_set → kvcm L3 (现有)
```

**位置索引层级**：
- L1: 树节点 `value`（SGLang，现有）
- HBF/DRAM: 虚拟索引（adapter 内部路由，树节点 `host_value` 存虚拟索引）
- L3: `storage_backend.batch_exists`（kvcm，现有）

---

## 四、当前文件清单

### 新增文件
| 文件 | 行数 | 内容 | 阶段 |
|---|---|---|---|
| `src/hisim/simulation/sglang/hbf_tiered_pool.py` | 470 | TieredHostKVCache adapter + MockHBFPool + PlacementPolicyV0 + 事件追踪 + install_tiered_host_pool() | PoC + Phase 2a |
| `test/test_hbf_tiered_pool.py` | 302 | 8 个 standalone 单元测试 (P1-P4 + 索引/split/容量) | PoC |
| `test/test_hbf_timing.py` | 210 | 5 个时序单元测试 (fast公式/transfer_duration/StateManager/config解析/基线回归) | Phase 1 |
| `test/test_hbf_load_poc.py` | 210 | 5 个 HBF Load 时序 PoC 测试 (独立于L3, 重叠感知, 10ms延迟→overflow=9ms) | Phase 2b-1 |
| `test/assets/mock/config_qwen3_hbf.json` | 35 | HBF 配置示例 (64GB HBF, 512GB/s, ratio=0.7) | Phase 1 |
| `docs/Qwen3_Simulation_Guide.md` | 350 | Qwen3 仿真运行指南 | 阶段 0 |
| `docs/HBF_Adaptation_Progress.md` | 380 | 本进展文档 | 持续更新 |

### 修改文件
| 文件 | 改动 | 说明 | 阶段 |
|---|---|---|---|
| `src/hisim/simulation/types.py` | +25 行 | `HBFConfig` dataclass + `PlatformConfig.hbf`/`enable_hbf` + bandwidth properties | Phase 1 |
| `src/hisim/simulation/sim_args.py` | +20 行 | `_parse_hbf_config()` + `PlatformConfig` HBF 字段 + `from_json` HBF 解析 | Phase 1 |
| `src/hisim/simulation/manager/config.py` | +5 行 | `get_platform_config` HBF 解析 | Phase 1 |
| `src/hisim/simulation/manager/state.py` | +40 行 | HBF dur 字段 + inc/pop/last/reset 方法 | Phase 1 |
| `src/hisim/simulation/sglang/sglang_hook.py` | +80 行 | `_hbf_transfer_duration` + `_record_hbf_*_timing` + `_install_hbf_adapter_if_enabled` + `mock_start_writing/loading` HBF 分流 + `wrapped_process_batch_result` HBF timing 合并 + HBF-STATS 日志 | PoC + Phase 1 + Phase 2a |

### 未修改
- SGLang 原生代码（`/sgl-workspace/sglang/`）：**0 行修改**
- kvcm C++ 源码：**0 行修改**
- HBFSim 源码：**0 行修改**（只读参考）

---

## 五、单 KV Page 生命周期（方案 A 设计）

以 page P（3 tokens）为例：

```
Build:    L1✓ (value=[dev_5,6,7])  L2✗  HBF✗  L3✗
            ↓ L1 驱逐
Backup:   adapter.alloc(3) → 放置策略选 HBF → 虚拟索引[v1001,1002,1003]
          host_value=[v1001..]  value=None
          L1✗  L2✓(虚拟)  HBF✓(物理)  L3✗
          容量: HBF_used+=3, DRAM_used+=0  ← ★ DRAM 未占
            ↓ HBF 满
Evict:    evict_host → host_value=None → write_backup_storage → L3
          L1✗  L2✗  HBF✗  L3✓(batch_exists 可查)
          容量: HBF_used-=3, L3_used+=3
            ↓ prefix 命中 L3
Reuse:    batch_exists 命中 → adapter.alloc → restore to HBF
          L1✗  L2✓(虚拟)  HBF✓  L3✓(副本保留)
            ↓ 需计算
Restore:  load_back(host_value) → adapter 路由 → HBF→L1
          L1✓  L2✓(write-through 副本)  HBF✓  L3✓
```

---

## 六、HBFSim 集成方案（已确认）

| Tier | 方式 | 工作量 | 读精度 | 写精度 | 阶段 |
|---|---|---|---|---|---|
| 1 | Python `fast_service_ns` 公式 | 5 分钟 | 一阶(无并发) | 一阶(无 program unit) | Phase 1 |
| 3 | 链接 `hbfsim_core`/`MqsimOnlineEngine` | 1-3 天 | channel/queue 级 | MQSim FTL 级 | Phase 3 |

**fast 公式**（`hbf_device.cuh:746-816` 源码确认）：
```python
latency = max(read_latency_ns, ceil(bytes * 1e9 / aggregate_bandwidth))
# 读: base = read_latency_ns (10μs)
# 写: base = program_latency_ns (100μs)
# 注意: 是 max 不是 sum (重叠感知模型)
```

---

## 七、待完成工作（后续阶段）

### Phase 2a：最小真实数据路径验证 ✅ 已完成
- ✅ HBF backup 实际发生（8 allocs, 33 tokens）
- ✅ DRAM 未过度占用（7/61 = 11.5%）
- ✅ Load-back 成功（prefix_reuse=0.7778）
- ✅ 混合放置（HBF+DRAM 同时有数据）
- ✅ HBF timing 接入 read_arrival 推导链
- ✅ 基线回归（HBF off 结果一致）

### Phase 2b-1：HBF Load 时序独立建模 ✅ 已完成
- ✅ 根因定位：`load_queue=0`（sim 模式不做真实 load_back），L2 load timing 从 `host_hit_length` 注入
- ✅ 时序消费点：`wrapped_run_batch` line 2677 — `step_global_clock(max(combined_load - last_inference, 0))`
- ✅ 独立 PoC：5/5 测试（10ms 延迟 → overflow=9ms）
- ✅ 重叠感知：load < inference → overflow=0（不机械累加）

### Phase 2b-2：L3 Write-through + Host Load 真实归因（进行中）
- ✅ Host Load 真实归因（host_value 遍历，非比例估算）— 8/8 反例测试
- ✅ 集成验证：HOST-LOAD 事件 + TTFT +10ms（10ms 延迟）
- ✅ combined_load_dur bug 修复（时钟推进用 combined 不用 l2_load）
- ✅ L3 Storage 开启（config_hbf_l3.json + --hicache-storage-backend sim）
- ✅ HBF/DRAM → L3 Write-through（batch_set=34-98）
- ✅ 副本保护（ack_backup_queue 语义）
- ✅ **L3 Read-back 闭环验证**（prefetch_threshold=1 修复后）：
  - 根因：`prefetch_threshold=256`（默认）阻断短 prefix（5 tokens < 256）的 storage query
  - 修复：`--hicache-storage-backend-extra-config '{"prefetch_threshold": 1}'`
  - **不需要 UnifiedCacheLinker** — 原生 HiCache prefetch 路径正常工作
  - `prefetch_from_storage → cache_controller.prefetch → handle_prefetch_operation → batch_exists → submit_read(SSD sim)`
  - HBF+L3: batch_exists=31, hits=24, prefetched=24tk, TTFT=18.2ms
  - Baseline+L3: batch_exists=33, hits=64, prefetched=64tk, TTFT=33.5ms
  - HBF 减少 L3 读取（24 vs 64 hits）→ TTFT 更低（18.2 vs 33.5）
- ✅ **L3 Write Ack 时序语义**（简化模型，已标注）：
  - `batch_set` 调 `storage_manager.WriteCache()` 同步返回 → Ack 在**逻辑写入**后发出
  - SSD 写入时序独立计算（`submit_l3_write_timing` → SSDTimingSimulator）
  - Host 副本可能在 SSD 写完成前释放（但逻辑写入已完成，kvcm 有数据）
  - `batch_set` 返回 False 时 Ack 不发出 → 不会误认为 L3 有副本
  - **简化标注**：瞬时逻辑 Ack + 独立 SSD 时序，不模拟写入排队延迟
- ⚠️ `batch_get` 返回 None（mock 不加载真实数据）— 预期行为，时序通过 SSD sim 独立计算
### V1a：容量感知 Placement — available_size 修复（已完成）

**P0: 首次失败精确机制**

SGLang `_execute_kv_backup` (unified_radix_cache.py:1376):
```python
host_avail = mem_pool_host.available_size()  # adapter 返回 SUM
if host_avail < kv_tokens:                    # SUM >= n → 跳过 eviction
    needed = kv_tokens - host_avail
    if self.evict_host(needed) < needed:
        return None
return cache_controller.write(...)            # alloc(n) 要求单一 Pool 容纳 n
```

根因链：
1. `available_size()` 返回 `hbf_free + dram_free`（SUM）
2. `alloc(n)` 要求**单一 Pool** 完整容纳 n（`hbf_pool.alloc(n)` 或 `dram_pool.alloc(n)`）
3. 当 `HBF_free=3, DRAM_free=3, n=5`：SUM=6≥5 → 不触发 eviction → alloc(5) 两池都<5 → 失败
4. alloc 失败 → 无 host_value → LRU 驱逐不调 free() → 池保持满 → **恶性循环死锁**

**V1a-1 修复：available_size() 返回 MAX 而非 SUM**

```python
def available_size(self):
    return max(self.hbf_pool.available_size(), self.dram_pool.available_size())
```

- `MAX < n` → 正确触发 `evict_host(needed)` → free() 被调用 → 释放空间 → alloc 成功
- 新增 `total_available_size()` 返回 SUM（供 metrics 使用）

**V1a 验证结果（128 请求，wl5_128req）**：

| Config | V0 fail_rate | V1a fail_rate | V0 backed_up | V1a backed_up | V1a L3 hits | V1a TTFT | V1a Tput |
|---|---|---|---|---|---|---|---|
| A(100%基线) | N/A | N/A | N/A | N/A | 15616 | 924 | 51.4 |
| C(50%) | **99.4%** | **0.3%** ✅ | 1082tk | **80397tk** | 10752 | 919 | 51.4 |
| E(30%) | **99.4%** | **3.3%** ✅ | 1082tk | **78992tk** | 9472 | 899 | 51.9 |
| F(20%) | 0% | 0% | 73869tk | 73869tk | 4224 | 910 | 51.6 |

- **恶性循环打破**：C/E fail_rate 从 99.4% 降到 0.3%/3.3%
- **QoS 全部满足**：TTFT ≤ 110%基线，吞吐 ≥ 95%基线
- 残余失败（C: 0.3%, E: 3.3%）→ both_full 边缘情况（victim 被 lock/pending）
- V1a-2（tier-aware eviction）可进一步降低残余失败

### V1a-Complete：容量感知 + 有界重试（已完成）

**V1a-1**（available_size=MAX）+ **V1a-2**（evict callback）+ **V1a-3**（bounded retry）

**残余失败追踪**（E config, 128 reqs）：
- `[ALLOC-FAIL] need=128 HBF_avail=62-77 DRAM_avail=103` → 两池都 <128（碎片化）
- `[EVICT-HOST] requested=87 freed=284` → eviction 释放了空间，但被其他 backup 竞争消耗
- 残余失败根因：**释放空间被并发 backup 消耗**（非错误介质回收）

**V1a-2/V1a-3 实现**：
- Adapter 安装时设置 `_evict_callback = lambda n: cache_obj.evict_host(n)`
- `alloc()` 失败时：计算 deficit → 调 `_evict_callback(deficit)` → 重试 alloc（max 3 次）
- 终止条件：eviction 返回 0（无 victim）/ alloc 成功 / 达到 max_retries
- 不绕过锁、Pending Write、Load-back 保护（通过原生 `evict_host`）

**V1a-Complete 验证**（128 reqs）：

| Config | V0 fail_rate | V1a-1 fail_rate | **V1a-Complete fail_rate** | V1a-Complete failed_tk |
|---|---|---|---|---|
| C(50%) | 99.4% | 0.3% | **0.0%** ✅ | **0tk** |
| E(30%) | 99.4% | 3.3% | **2.8%** | 2304tk |
| F(20%) | 0% | 0% | 0% | 0tk |

- C: **0% 失败** — retry 完全消除残余失败
- E: 3.3% → 2.8% — 残余为结构性容量不足（working set 68000tk >> L2 1332tk）
- QoS 全部满足：TTFT ≤ 110%基线，吞吐 ≥ 95%基线

**V1a-2/V1a-3 单元测试**（6/6 PASS）：

| 测试 | 场景 | 验证 |
|---|---|---|
| A | 错误介质 victim → retry 释放正确池 | alloc 成功 ✅ |
| B | 多 victim，只有目标介质 victim 有效 | alloc 成功 ✅ |
| C | 全部 victim 锁定 → 安全退出 | 无无限循环 ✅ |
| D | Eviction→realloc 状态一致 | 无 double-alloc ✅ |
| E | Build→Evict→Reuse 无假命中 | 无 double-free ✅ |
| Extra | 无进展终止（eviction 返回 0） | ≤max_retries ✅ |

### V1a-Final：残余失败归因 + 重入安全 + QoS 验证（已完成）

**E 的 16 次残余失败精确归因**：

每次失败的 RETRY 日志模式：
```
[RETRY] retry=0 need=128 deficit=25 freed=15 HBF: 62→77 DRAM: 103→103
[RETRY] retry=1 need=128 deficit=25 freed=0 HBF: 77→77 DRAM: 103→103
[RETRY] no_progress_exit: evict returned 0
[RETRY] exhausted: retries=3 final HBF=77 DRAM=103
```

归因：
1. 第一次 eviction 释放 15 tokens（HBF），但 deficit=25，不够
2. 第二次 eviction 返回 **0** — **no_evictable_victim**
3. 剩余 host 节点全部被 write-through pending / load-back 锁定
4. **不是错误介质回收、不是分配竞争、不是锁冲突** — 是 genuine structural exhaustion

Retry 统计：attempts=34, success=11, fail=16, rate=32.4%

**重入安全性检查**：
1. ✅ `alloc()` 持有 `self._lock`（RLock），`evict_host→free()` 重入同一锁 — RLock 允许
2. ✅ `evict_host` 检查 `lock_ref`/`host_lock_ref` — 被备份节点不可淘汰
3. ✅ 当前备份节点正在创建 host_value → 不在 host LRU → 不会被淘汰
4. ✅ 无递归（evict_host 不调 alloc）、无死循环（max_retries=3 终止）
5. ✅ 无 stale host_value（evict 通过原生 `_free_values` 链路，host_value 设 None）
6. ✅ 无 double-free（`free()` 有 `_slot_used` assert）

**Tier-aware Victim Selection 判断**：
- 残余失败根因是 **no_evictable_victim**（eviction 返回 0），不是错误介质
- 第一次 eviction 释放了正确介质（HBF），但量不够
- 第二次 eviction 无 victim 可选 — Tier-aware 选择无法解决
- **不需要实现 Tier-aware Victim Selection** — 当前安全降级是正确的

**V1a-Complete QoS 对照**（128 reqs, wl5_128req）：

| Config | DRAM% | P95_TTFT | P95_TPOT | Tput | Comp | Fail% | L3 hits | DRAM Peak | Retry |
|---|---|---|---|---|---|---|---|---|---|
| A(基线) | 100% | 1742 | 2.4 | 51.4 | 128 | N/A | 15232 | N/A | N/A |
| C(50%) | 50% | 1740 ✅ | 2.4 ✅ | 51.4 ✅ | 128 | **0%** | 12160 | 1024 | 1/1 |
| E(30%) | 30% | 1718 ✅ | 2.4 ✅ | 51.9 ✅ | 128 | 2.5% | 9600 | 615 | 11/34 |
| F(20%) | 20% | 1733 ✅ | 2.4 ✅ | 51.6 ✅ | 128 | **0%** | 4224 | 384 | 0/0 |

QoS 约束（P95_TTFT ≤ 1916, P95_TPOT ≤ 2.64, Tput ≥ 48.8）：**全部满足** ✅

**70% DRAM 替代阶段性结论**：
- E（30% DRAM = 70% 替代）：P95_TTFT **1718ms vs 基线 1742ms（1.4% 更优）**
- 残余 2.5% 失败为 no_evictable_victim（结构性容量不足），不影响 QoS
- HBF 减少 L3 读取 37%（9600 vs 15232）
- DRAM Peak 仅 615/615（100% 利用率），HBF Peak 685/717（95%）
- **70% DRAM 替代在 128 请求下稳定通过 QoS** ✅

**P0: E/F 非单调行为根因**

E(DRAM=307,HBF=717): alloc fail_rate=**97.6%**, both_full=174, freed=0 → **恶性循环**
F(DRAM=205,HBF=819): alloc fail_rate=**0%**, both_full=0, freed=18720 → **良性循环**

根因链条：
1. 池填满 → `alloc()` 返回 None → backup 失败 → 无 host_value
2. LRU 驱逐无 host_value 的节点 → 不调 `free()` → 池保持满
3. 新 alloc 继续失败 → 更多节点无 host_value → **死锁**

F 逃出死锁：所有 alloc 成功 → 所有节点有 host_value → LRU 驱逐时调 `free()` → 释放空间 → 良性循环

**P0: 缓存守恒指标已补齐**
- `conservation_stats()`: alloc_attempts/success/failed, fallback counts, both_full, peak_used, free_tokens
- iteration.jsonl: 全部 conservation 字段
- `[HBF-CONSERV]` 日志：每次 check_hicache_events 输出

**128 请求 V0 扫描**（wl5_128req: 128 reqs, 512 input, 16 output）：

| Config | DRAM% | success_tk | failed_tk | fail_rate | L3 hits | L2 hits | TTFT(ms) | Tput |
|---|---|---|---|---|---|---|---|---|
| A(基线) | 100% | N/A | N/A | N/A | 15488 | 0 | 924 | 51.4 |
| C(50%+HBF) | 50% | 1082 | 192000 | **99.4%** | 0 | 765 | 722 | 60.9 |
| E(30%+HBF) | 30% | 1082 | 192000 | **99.4%** | 0 | 765 | 724 | 60.8 |
| **F(20%+HBF)** | 20% | **73869** | **0** | **0%** | 4224 | 0 | 910 | 51.6 |
| G(10%+HBF) | 10% | 811 | 193536 | 99.6% | 0 | 510 | 720 | 60.9 |
| H(5%+HBF) | 5% | 811 | 193536 | 99.6% | 0 | 510 | 720 | 60.9 |

**关键发现**：
- 128 请求下恶性循环更严重：C/E/G/H 99.4-99.6% 失败率
- **只有 F(20% DRAM, hbf_ratio=0.8) 逃出死锁**（0% 失败）
- F 的 TTFT(910ms) ≈ 基线 A(924ms) → L3 读取主导
- C/E 的 TTFT(722ms) < 基线 → 192000 recompute tokens 无缓存读取（但需重算）

**HBF 延迟敏感性**（config F, 128 reqs）：

| HBF read_latency | TTFT(ms) | TPOT | Tput |
|---|---|---|---|
| 10μs | 910.2 | 2.4 | 51.6 |
| 100μs | 910.2 | 2.4 | 51.6 |
| 1ms | 910.2 | 2.4 | 51.6 |
| 10ms | 超时 | — | — |

- **10μs-1ms 无影响**：L3 读取主导 TTFT，HBF load 被 inference 重叠
- 10ms 超时：HBF load overflow 过大

**70%+ DRAM 替代的阶段性判断**：

| 工作负载 | 70% 替代 | 80% 替代 | 结论 |
|---|---|---|---|
| 32 reqs | ✅ E(30%DRAM) TTFT降47% | ✅ F(20%) 满足QoS | 小负载可行 |
| 128 reqs | ❌ C/E 99.4%失败 | ⚠️ F 0%失败但TTFT≈基线 | **V0 放置策略不足** |

**结论**：
- V0 比例放置策略存在**恶性循环缺陷**：池满后 alloc 失败 → 无 host_value → 无法 free → 死锁
- 32 请求下 70% 替代可行（负载小，死锁未触发）
- 128 请求下大部分配置陷入死锁（99%+ 失败率）
- **需要 V1a（容量感知放置 + 定向淘汰）修复恶性循环**
- V1a 目标：池满时主动淘汰 victim（free host_value）而非失败 alloc

**限制标注**：
- 快速时序模型（max 语义，无队列竞争）
- prefetch_threshold=1（非默认 256）
- HBF read_latency=10μs（1ms 以内不影响 TTFT）

**三个问题调查**：
1. ✅ **C(64 L3 hits) vs D(0 L3 hits) 根因**：D 的 DRAM=16 太小 → HBF 填满后 fallback 不足 → `alloc()` 返回 None → backup 失败 → 无 L2 副本 → 无 L3 写入。C 的 DRAM=21 更大 → 82 allocs vs D 的 10 → L3 写入正常。
2. ✅ **E2E = TTFT + TPOT**：output_length=2 时 trivially correct（1 prefill + 1 decode）。已用 output_length=16 验证：E2E = TTFT + TPOT × 15。
3. ✅ **l3_prefetch_wait_ms**：预留字段（=0.0），**未实际计算**，已标记不可用于收益分析。

**V0 容量替代扫描**（wl1b_512tok: 32 reqs, 512 input, 16 output, max_total_tokens=1024, prefetch_threshold=1）：

| Config | DRAM% | DRAM | HBF | L3 hits | L2 hits | TTFT(ms) | P95_TTFT | TPOT | Tput |
|---|---|---|---|---|---|---|---|---|---|
| A(基线) | 100% | 1024 | 0 | 3840 | 0 | 189.8 | 319.1 | 2.3 | 48.1 |
| B(纯缩) | 50% | 512 | 0 | 3968 | 0 | 144.6 | 269.3 | 2.3 | 52.0 |
| C(HBF补) | 50% | 512 | 512 | 0 | 765 | **97.9** | 188.1 | 2.0 | 60.9 |
| D(HBF补) | 35% | 358 | 666 | 0 | 765 | **99.9** | 190.2 | 2.0 | 60.7 |
| E(HBF补) | 30% | 307 | 717 | 0 | 765 | **99.9** | 190.2 | 2.0 | 60.7 |
| F(HBF补) | 20% | 205 | 819 | 2176 | 0 | 135.8 | 247.0 | 2.1 | 54.0 |

**QoS 约束**（P95_TTFT ≤ 351ms, P95_TPOT ≤ 2.64ms, Tput ≥ 45.7）：
- **C/D/E 全部满足**，TTFT 比基线低 47%，吞吐高 26%
- **E（70% DRAM 替代）**：TTFT 99.9ms vs 基线 189.8ms → **性能更优**
- **F（80% DRAM 替代）**：L3 读取回来，TTFT 退化但仍满足 QoS
- **拐点**：DRAM=205(F) 时 HBF 不能完全吸收 → L3 读取回来

**容量替代结论**：
- 在当前工作负载（32 reqs, 512 token, 高复用）下，**70% DRAM 替代可行且性能更优**
- HBF ≥ 512 时完全吸收 L2 容量 → L3 读取为 0 → TTFT 大幅降低
- DRAM 过小（<205）时 fallback 不足 → 部分 token 无 L2 副本 → L3 读取回来

**限制标注**：
- 快速时序模型（max 语义，无队列竞争）→ Phase 3 MQSim 校准
- 工作负载较小（32 reqs）→ Phase 2b-5+ 更大负载验证
- HBF read_latency=10μs（快速）→ 需测试不同延迟下的拐点
- prefetch_threshold=1（非默认 256）→ 需调查 rate limiting 影响
- ✅ **Request 级指标**（request.jsonl + 日志）：
  - `hbf_hit_tokens`, `dram_hit_tokens`：HBF/DRAM 真实命中拆分
  - `hbf_load_bytes`, `dram_load_bytes`：每介质加载字节数
  - `l3_hit_tokens`, `prefetch_complete_tokens`：L3 命中和预取完成
  - `gen_token_latencies[0]=TTFT`, `[1]=TPOT`：每请求时延
  - 日志格式：`Req #rid: input=5 L1=0 HBF=0 DRAM=4 L3=0 New=1 TTFT=3.6ms TPOT=1.6ms pf_complete=0`
- ✅ **Iteration 级指标**（iteration.jsonl）：
  - `hbf_used_tokens/bytes`, `dram_used_tokens/bytes`：每介质容量
  - `hbf_load_latency`, `hbf_backup_latency`：HBF 时序
  - `combined_load_latency`, `combined_backup_latency`：合并时序(max)
  - `l3_write_overflow`：L3 写溢出
  - `hbf_capacity_tokens`, `dram_capacity_tokens`, `hbf_available`, `dram_available`
- ✅ **TTFT 差异归因**（A vs C，同总量不同介质）：
  - 稳态请求(1-14) TTFT 模式相似（交替 56.9/8.9ms）
  - 边界请求差异：req0(106.9 vs 61.9)、req15(171.9 vs 8.9) — 调度边界效应
  - L3 hits/pf_complete 完全一致(64/64) → L3 行为不变
  - 差异来自请求到达时间与 batch 调度的交互，非时序 bug
- ✅ **V0 容量替代预实验**（A/B/C/D，16请求 2轮）：

| Config | DRAM | HBF | Total | L3 hits | L2 hits | TPOT | TTFT | E2E |
|---|---|---|---|---|---|---|---|---|
| A(基线) | 41 | 0 | 41 | 64 | 0 | 25.2 | 25.5 | 50.7 |
| B(减DRAM) | 21 | 0 | 21 | 64 | 0 | 15.2 | 260.6 | 275.9 |
| C(HBF补足) | 21 | 20 | 41 | 64 | 0 | 11.0 | 28.2 | 39.2 |
| D(更大HBF) | 15 | 26 | 41 | 0 | 16 | 25.6 | 2.7 | 28.3 |

  - B vs A: DRAM 减半 → TTFT 10x 恶化
  - C vs B: HBF 补足 → TTFT 恢复(260→28)，**DRAM 减 50% 不损失 QoS**
  - D: DRAM 减 63% + 大 HBF → TTFT 最佳(2.7ms)，L3 读取降为 0
- ✅ **L3 Write Ack 简化文档**：
  - `batch_set` 同步逻辑写 → Ack 立即发出（不等 SSD 写完成）
  - SSD 写时序独立计算（`submit_l3_write_timing` → SSDTimingSimulator）
  - **不等价于物理持久化完成** — 未落盘数据由 kvcm 内存索引保持可见
  - Host 副本在逻辑写后释放，kvcm 缓存保留索引 → 数据安全
  - `submit_read` 用前一轮 write avail_time → 正确因果顺序
- ✅ **Fallback vs Eviction vs Migration 区分**：
  - **Fallback**：alloc 首选池满 → 自动尝试另一池（已实现 ✅）
  - **Eviction**：双池满 → SGLang 原生 LRU evict_host → `adapter.free()` 路由到正确物理池（已验证 ✅）
  - **Migration**：HBF→DRAM 或 DRAM→HBF 主动迁移 → **V0 不支持**（V1 放置策略范围）
  - 当前淘汰由 SGLang LRU 驱动，不区分介质选择 victim
- ✅ **TPOT 差异归因**：A/B/C 容量公平对照
  - A(DRAM=41,HBF=0) TPOT=2.5 ; B(DRAM=41,HBF=5) TPOT=32.3 ; C(DRAM=36,HBF=5) TPOT=2.6
  - **A≈C（同总量 41）→ TPOT 相同 → 无时序 bug**
  - B 的 TPOT 高因 L2 增容改变驱逐模式 → DECODE bs=1 → prefill/decode 间有 ~50ms 请求到达间隔
  - per-batch 时序验证：DECODE `load=0μs`（无重复计时），backup 0-100μs（无重复）
- ✅ **L3 Write Ack 因果**（简化模型，已标注）：
  - `_drive_hicache_timing_phases` 调用顺序：`handle_backup`(逻辑写) → `handle_prefetch`(读) → `submit_l3_write_timing`(SSD 时序)
  - `batch_set` 同步返回 → Ack 在逻辑写入后 → Host 副本可释放
  - `submit_read` 用前一轮 `avail_time` → 正确因果（不读当前轮未提交的写）
  - Host 副本可能在 SSD 写完成前释放（但 kvcm 有数据，无丢失）
  - **简化标注**：瞬时逻辑 Ack + 独立 SSD 时序
- ✅ **HBF/DRAM 定向淘汰**（9/9 测试 PASS）：
  - HBF 满 → DRAM fallback ✅
  - DRAM 满 → HBF fallback ✅
  - 双满 → return None → 调用方淘汰到 L3 ✅
  - `free()` 按 `dram_offset` 拆分释放正确物理 Pool ✅
  - 混合 HBF+DRAM 索引一次 `free()` 正确拆分 ✅
  - Double-free assert 保护 ✅
  - 20 次 alloc/free 循环无泄漏 ✅
  - 虚拟索引一致性（HBF < offset, DRAM ≥ offset, 无重叠）✅
- ✅ **per-batch 时序日志**（`[BATCH-TIME]` 诊断日志）：load/infer/backup/l3_ovf/clock
- ✅ **per-medium 容量统计**（`[HBF-STATS]`）：hbf_used/dram_used/allocs/getData/setData

### Phase 2b-3：HBF/DRAM 定向淘汰与容量压力验证
- [ ] HBF 满→DRAM fallback；DRAM 满→HBF；双满→L3
- [ ] 容量压力下的正确性和无死循环

### Phase 2b-4：完整指标输出
- [ ] iteration.jsonl HBF 字段（backup/load/hit/capacity per medium）
- [ ] metrics.json HBF 聚合统计
- [ ] HBF/DRAM 逻辑容量 vs CPU Mock 内存分开统计

### Phase 2b-5：V1 放置策略及更大负载
- [ ] V1 对象特征感知放置（KV 大小/访问粒度/复用）
- [ ] 更大工作负载（Qwen3-32B 或多轮 hisim-collection）
- [ ] 小规模 DRAM 容量替代实验（V0 策略能否达到 70% DRAM 降低）

### Phase 3：HBFSim 时序适配
- [ ] `HBFTimingSimulator` 类（同 `SSDTimingSimulator` 接口）
- [ ] 评估链接 `hbfsim_core` 或 shell out `hbf_mqsim_service`
- [ ] 对比 fast vs MQSim 时序差异
- [ ] channel/queue 级并发建模

### Phase 4：DRAM 容量替代实验
- [ ] 实验矩阵（A/B/C/D：DRAM 100/30/30/30 GiB，HBF 0/0/70/256 GiB）
- [ ] 验收指标：DRAM 降低比例 ≥70%、TTFT/TPOT/吞吐、各层命中
- [ ] 工作负载：Qwen3 基线 + Build/Hold/Reuse/Churn 多阶段
- [ ] 区分逻辑 Host KV DRAM 占用 vs 仿真器 CPU 内存占用

### Phase 5：Hybrid KV 扩展（预留）
- [ ] KDA State / MLA KV Block 等异构缓存对象接口预留

---

## 八、关键设计决策记录

| 决策 | 选择 | 理由 | 状态 |
|---|---|---|---|
| HBF 层级位置 | 并列 L2（非串行 L1.5） | 用户要求"并列替代部分 DRAM"，非级联缓存 | ✅ 用户确认 |
| 时序保真度 | B2: 移植 fast backend 为 Python | 一阶精度够用，MQSim 留 Phase 3 | ✅ 用户确认 |
| 温度模型 | Defer | 先做容量/带宽/延迟 | ✅ 用户确认 |
| 分支策略 | `develop_dqy_0.5.19_qwen3_hbf` | 保持 Qwen3 基线干净 | ✅ 用户确认 |
| Pool 方案 | A: Virtual/Tiered Host Pool Adapter | 低侵入（0 行 SGLang 改动），透明性已验证 | ✅ PoC 验证 |
| kvcm 关系 | 2a: HBF 独立于 kvcm | HBF 按 L2 模式管理，kvcm 只管 L3 | ✅ 用户确认 |
| 放置策略 | V0 比例分流 → V1 特征感知 → V2 代价感知 | 渐进式，V0 先验证链路 | ✅ 用户确认 |
| 路由插入点 | `mem_pool_host.alloc` 层（非 `handle_backup`） | DRAM 在 alloc 时分配，路由必须在 alloc 前 | ✅ 源码确认 |

---

## 九、运行方式

### Qwen3 仿真（已验证）
```bash
export PYTHONPATH=/home/d00954154/HiSim_Qwen3/src
export TORCH_TRANSFER_TO_NPU=0
export ASCEND_RT_VISIBLE_DEVICES=7
export HISIM_CONFIG_PATH=/home/d00954154/HiSim_Qwen3/test/assets/mock/config.json

python3 -m hisim.simulation.sglang.launch_server \
  --model-path /home/d00832329/Qwen3-0.6B \
  --load-format dummy --device cpu \
  --enable-hierarchical-cache --port 30010
```

### HBF PoC + Phase 2a（adapter 安装 + 真实数据路径）
```bash
export PYTHONPATH=/home/d00954154/HiSim_Qwen3/src
export TORCH_TRANSFER_TO_NPU=0
export ASCEND_RT_VISIBLE_DEVICES=7
export HISIM_CONFIG_PATH=/home/d00954154/HiSim_Qwen3/test/assets/mock/config.json
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HISIM_HBF_ENABLE=1
export HISIM_HBF_LOGICAL_SIZE=10000   # HBF 虚拟槽位数
export HISIM_HBF_RATIO=0.9            # V0 放置比例

# 用 --max-total-tokens 限制 L1 制造驱逐压力
python3 -m hisim.simulation.sglang.launch_server \
  --model-path /home/d00832329/Qwen3-0.6B \
  --load-format dummy --device cpu \
  --enable-hierarchical-cache --port 30010 \
  --max-total-tokens 30
```

### Phase 1 单元测试
```bash
cd /home/d00954154/HiSim_Qwen3
PYTHONPATH=src python3 test/test_hbf_tiered_pool.py   # 8 tests (PoC adapter)
PYTHONPATH=src python3 test/test_hbf_timing.py          # 5 tests (fast 公式 + config)
```

---

## 十、风险与待验证项

| 风险/待验证 | 状态 | 缓解/计划 |
|---|---|---|
| HBF 路由未在集成中触发（工作负载太小） | ⚠️ 待验证 | 用 Qwen3-32B 或多轮负载触发 L1 驱逐 |
| HBF 副本与 L3 ack 语义一致性 | ❓ 待验证 | 确认现有 L2→L3 ack 的 `dec_host_lock_ref` 适用 HBF |
| `HostPoolGroup` sidecar 与 adapter 交互 | ❓ 待验证 | PoC 用 PlainKv（无 sidecar），多 component 待测 |
| HBF 并发对 TPOT 的影响 | ❓ 待实测 | Phase 4 实验验证 |
| nominal.json 参数非实测 | ❓ 已标注 | 结果标 CONDITIONAL_SIMULATED |
| fast 公式不建模 channel/queue | 已知 | Phase 3 用 MQSim reference 补 |
