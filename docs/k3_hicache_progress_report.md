# HiSim + Kimi-K3 多级缓存适配进展报告

> **文档目的**：总结 HiSim 框架适配 Kimi-K3（MLA + KDA/Mamba 多级缓存）的全部技术进展，包括背景、架构、问题定位、代码修改和验证结果。供在其他机器、其他 sglang/hisim 版本上适配时参考。
>
> **更新历史**：
> - 2026-09-xx：初始版本（Phase 4C-1 基础，commit `0c62f61`）
> - 2026-10-13：追加 Phase 4C-1 后续修复（层编号/slot_bytes/slot_hash/MAMBA.value）、4C-2 动态内存管理、4C-3 时延模型修复、4C-4 容量对照实验（commit `a811a06`）

---

## 1. 环境信息

| 项目 | 值 |
|------|-----|
| 硬件 | Ascend 910C（8 chip，每 chip 2 die） |
| OS | Ubuntu 22.04.5 LTS |
| Docker 容器 | `sglang-kimi-k3-dqy` |
| sglang 版本 | `0.5.18.dev922+ga41da991c` |
| PyTorch | `2.10.0+cpu`（CPU 引擎模式） |
| torch_npu | `2.10.0` |
| 模型 | Kimi-K3-23L（23 层：5 层 Full Attention + 18 层 KDA/Mamba） |
| Git 分支 | `develop_dqy` |
| 最新 commit | `a811a06` |

### 关键环境变量

```bash
export SGLANG_USE_CPU_ENGINE=1          # CPU 模拟模式（不执行真实 NPU kernel）
export ASCEND_VISIBLE_DEVICES=2          # 使用 chip 2（避免与另一会话设备 3 冲突）
export TORCH_TRANSFER_TO_NPU=0           # 禁止数据传输到 NPU
export TORCH_DEVICE_BACKEND_AUTOLOAD=0    # 禁止自动加载设备后端
export HISIM_SIMULATION_MODE=OFFLINE      # 离线时间戳调度模式
export HISIM_OUTPUT_DIR=/tmp/hisim/output
export HISIM_SSD_SIM=simpyssd             # SSD 模拟后端
export HISIM_CONFIG_PATH=.../config_910c_k3.json
export HISIM_RESET_HICACHE_STORAGE=1      # 每次启动清空 L3 存储

# Phase 4C-2+ 新增
export HISIM_L2_MAMBA_SIZE=330           # L2 Mamba 缓存大小（默认 330）
export HISIM_L2_KV_SIZE_GB=10             # L2 KV 缓存大小 GB（默认 = hicache_size）
export HISIM_DISABLE_PROACTIVE_EVICT=0   # 1=禁用 proactive eviction（实验控制）
```

---

## 2. 四种部署模式

HiSim 通过不同的 sglang 启动参数组合，支持四种部署模式：

| 模式 | sglang 参数 | L1 结构 | L2 结构 | 说明 |
|------|-------------|---------|---------|------|
| **Two Pools Plain** | （无特殊参数） | MockKV + MockMamba | 无 | 基础模式，纯 L1 |
| **Two Pools HiCache** | `--enable-hierarchical-cache --hicache-storage-backend file --hicache-size 80` | MockKV + MockMamba | MockHostKV + MockHostMamba | 独立 L1/L2 双池架构 |
| **One Pool** | `--enable-unified-memory` | UnifiedKVPool（KV+Mamba 统一） | 无 | 统一内存池，MultiEndedAllocator 管理 KV/Mamba 动态划分 |
| **One Pool HiCache** | `--enable-unified-memory --enable-hierarchical-cache --hicache-size 10 --max-mamba-cache-size 1000` | UnifiedKVPool | MockHostKV + MockHostMamba | 统一 L1 + mock L2，是最复杂模式 |

### K3 模型架构

> **重要修正（Phase 4C-1）**：config 使用 1-indexed 层编号，sglang 通过 `(i+1) in kda_layers` 转换为 0-indexed。原 mock 使用 1-indexed 导致第 23 层（0-indexed）被排除，实际只有 17 层 KDA。已修复为 0-indexed。

```
Layer 0-22 (共 23 层，0-indexed)
  Full Attention: [3, 7, 11, 15, 19]     — 5 层 MLA（kv_lora_rank=512, qk_rope_head_dim=64）
  KDA/Mamba:      [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22] — 18 层

MLA entry_bytes = 5 × 1 × 288 × 2(K+V) × 2(bf16) = 5760 B/token
KDA slot_bytes  = 18 × (13824×2 + 196608×4) = 14653440 B/slot (TP=8, conv=bf16, temporal=fp32)
                 （旧错误值: 15151104，假设 conv itemsize=4 实际 bf16=2）
```

#### 23 层 Mapping 对照表

| sglang 层 (0-idx) | 类型 | Sub-Pool | Index |
|-------------------|------|----------|-------|
| 0, 1, 2 | KDA | Mamba | 0, 1, 2 |
| **3** | **Full Attention** | **KV** | **0** |
| 4, 5, 6 | KDA | Mamba | 3, 4, 5 |
| **7** | **Full Attention** | **KV** | **1** |
| 8, 9, 10 | KDA | Mamba | 6, 7, 8 |
| **11** | **Full Attention** | **KV** | **2** |
| 12, 13, 14 | KDA | Mamba | 9, 10, 11 |
| **15** | **Full Attention** | **KV** | **3** |
| 16, 17, 18 | KDA | Mamba | 12, 13, 14 |
| **19** | **Full Attention** | **KV** | **4** |
| 20, 21, 22 | KDA | Mamba | 15, 16, 17 |

总计：5 Full Attention + 18 KDA = 23 层，无交叉映射。

---

## 3. 代码修改总览

### 3.1 修改的文件

| 文件 | 说明 | 改动量 |
|------|------|--------|
| `src/hisim/src/hisim/simulation/sglang/launch_server.py` | sglang 启动前 monkey-patch | ~370 行新增 |
| `src/hisim/src/hisim/simulation/sglang/sglang_hook.py` | HiCache 路由树 hook（mock write/load） | ~200 行新增/修改 |
| `src/hisim/src/hisim/simulation/sglang/sglang_mock_class.py` | Mock host pool 类 | ~50 行修改 |

### 3.2 修改分类

修改按目的分为以下几类：

#### A. 基础设施 Patch（`launch_server.py`）

这些 patch 在 sglang import 之前执行，为 CPU 模拟环境创建必要条件。

| Patch 名称 | 目的 | 原理 |
|-----------|------|------|
| `kimi_linear_config` 强制检测 | 让 sglang 识别 K3 模型为 hybrid_ssm | sglang 原生依赖 model_config.json 中的 `model_type`，但 K3 的配置在某些版本中不匹配 |
| `mambaish_config` 强制返回 | 为 unified memory 提供 `mamba2_cache_params` | `init_unified_mamba_pools` 需要此参数计算 Mamba sub-pool 大小 |
| `support_triton = False` | 禁用 Triton backend | Triton kernel 在 NPU 模拟中崩溃，使用 Python fallback |
| `get_available_gpu_memory` patch | CPU 模式使用全部系统内存 | 原生代码除以 NUMA 节点数，限制为单节点（~212GB） |
| `ModelConfig.__init__` override | `head_num=1, head_dim=288` | HiSim MLA 容量模拟：用 MHA 尺寸模拟 MLA 字节数（5760 B/token） |
| `MambaPool` data ops no-op | `copy_from/get_cpu_copy/load_cpu_copy` 空实现 | 这些方法触发 NPU fused kernel，在 CPU 模拟中崩溃 |
| `_handle_page_major_kv_layout` 放宽 | 允许 `torch_native` backend | 原生代码 assert 要求 Triton，HiSim 前向是 mock 不需要 |
| `_handle_unified_memory_pool` 绕过 | 临时关闭 `enable_hierarchical_cache` 过 assert | 原生代码不允许 unified + hierarchical 同时启用 |
| `MultiEndedAllocator` 观测计数 | 统计 alloc/free/compact/flush | 可选的诊断计数器，不影响功能 |

#### B. 崩溃修复 Patch（`launch_server.py`）

| Patch 名称 | 目的 | 原理 |
|-----------|------|------|
| **`zero_pages` CPU fallback** | 修复 One Pool 模式 SIGABRT | 见下方问题 1 详解 |
| **`HybridCacheController` 线程函数 no-op** | 修复 `backup_thread_func` SIGABRT | 见下方问题 2 详解 |

#### C. HiCache 路由树 Hook（`sglang_hook.py`）

| Hook 函数 | 目的 |
|----------|------|
| `C_ModelRunnerHook` unified memory 分支 | 在 unified 模式下让 sglang 创建真实 UnifiedKVPool，但 mock 化 MambaPool |
| `wrapped_init_hicache` | 统一模式下跳过原生 init_hicache，手动设置 mock host pool |
| `_apply_mock_start_overrides` | 替换 `cache_controller.start_writing` 和 `start_loading` 为 mock 实现 |
| `_mock_start_writing` | mock L1→L2 备份：mock host pool backup_from_device + 设置 node.component_data host_value |
| `_mock_start_loading` | mock L2→L1 加载：mock host pool load_to_device + 恢复 node.component_data value |
| `wrapped_check_hicache_events` | 主动驱逐：在 check_hicache 时驱逐 evictable 节点，触发 L1→L2→L3 链 |
| `wrapped_match_prefix` | P2 诊断：打印 match_prefix 结果（full/dev/host/mamba_host/best_match） |
| `wrapped_reset` | 记录 tree reset 调用（诊断） |
| OFFLINE 调度 | 基于时间戳的请求释放（heap queue + virtual clock） |

#### D. Mock Host Pool 修改（`sglang_mock_class.py`）

| 修改 | 目的 |
|------|------|
| `MockTokenToKVPoolHost` 使用 `getattr` | 兼容 UnifiedKVPool（部分属性不存在） |
| `MockMambaStatePoolHost._last_evicted_hashes` | 记录 L2 驱逐的 hash，供 L3 提升使用 |
| `MockMambaStatePoolHost.get_stats()` | 返回 L2 池统计快照 |

#### E. L2 Host Hit 修复（`sglang_hook.py`，Phase 4C-1 初始修复）

| 修改 | 目的 |
|------|------|
| `component_data` 索引访问 | 见下方问题 3 详解 |
| MAMBA `host_value` fallback | 见下方问题 3 详解 |

#### F. 层编号与 slot_bytes 修正（`launch_server.py`，Phase 4C-1 后续修复）

| 修改 | 目的 | Commit |
|------|------|--------|
| K3 层编号 1-indexed → 0-indexed | config 使用 1-indexed，sglang 通过 `(i+1) in kda_layers` 转为 0-indexed。mock 原使用 1-indexed → 第 23 层（0-idx）被排除 → 17 层而非 18 层 KDA | `2508169` |
| slot_bytes 修正 | 旧值 15151104（假设 conv itemsize=4）→ 新值 14653440（conv=bf16 itemsize=2, temporal=fp32 itemsize=4） | `2508169` |
| `layer_mapper` 精确映射 | 旧 `lambda i: i - 5 if 5 <= i < 23 else None` → 新 dict 映射，精确指定 Full/KDA 层的 sub-pool index | `2508169` |
| `full_attention_layer_ids` 修正 | 旧 `[4,8,12,16,20]` (1-indexed) → 新 `[3,7,11,15,19]` (0-indexed) | `2508169` |

#### G. MAMBA slot_hash 与 value 修复（`sglang_hook.py`，Phase 4C-1 后续修复）

| 修改 | 目的 | Commit |
|------|------|--------|
| MAMBA `slot_hash` 用 `node_id` 不含 `slot_idx` | 旧 `f"kda_{nid}_{slot_idx}"`：backup 和 load 的 slot_idx 不同 → hash 不匹配 → L3 restore 找不到 | `46e7578` |
| MAMBA `value` 恢复遍历所有 transfer | 旧 `_mamba_loads[0].device_indices` 为 None（首个 transfer）→ `value=None` → validator 失败。改为遍历所有 transfer 找非 None | `5abe1cd` |

#### H. 时延模型修复（`sglang_hook.py` + `sglang_mock_class.py`，Phase 4C-3）

| 修改 | 目的 | Commit |
|------|------|--------|
| `backup_from_device(from_l3=True)` 不计 D2H | L3→L2 restore 路径中，数据已在 DRAM，无需额外 D2H。每次减少 273.4µs 多计时 | `237f879` |
| KDA L3 read + H2D 串行化 | 旧 `max(MLA, KDA_H2D, KDA_L3_read)` 将 KDA L3 和 H2D 视为并行 → 改为 `max(MLA, KDA_L3_read + KDA_H2D)` 串行求和 | `237f879` |

#### I. 诊断与环境控制（`sglang_hook.py` + `launch_server.py`）

| 修改 | 目的 | Commit |
|------|------|--------|
| `HISIM_DISABLE_L3` 环境变量 | 负面测试：禁用 L3 后请求以 fresh state 完成 | `b13e4b1`（后移除） |
| `HISIM_L2_MAMBA_SIZE` / `HISIM_L2_KV_SIZE_GB` | 可配置 L2 容量 | — |
| `HISIM_DISABLE_PROACTIVE_EVICT` | 禁用 proactive eviction，用于实验控制 | — |
| KDA-REPLAY 统计 | 在 `wrapped_get_new_batch_prefill` 中打印 replay_tokens | `a811a06` |
| P2 match COW 诊断 | 移除（已确认 COW 不可用） | `a811a06` |

---

## 4. 核心问题详解

### 问题 1：`zero_pages` SIGABRT（影响所有 One Pool 模式）

**现象**：One Pool 模式启动后，第一个 prefill 请求触发 SIGABRT，日志显示 `aivec error`（D-cache 总线错误）。

**根因**：

sglang 的 `zero_pages`（`kernels/ops/kvcache/zero_pages.py`）用于清零物理页。它**总是启动 Triton kernel**，没有 `is_cuda` 检查和 CPU fallback。

而同模块的 `alloc_bind_inplace` 有正确的 fallback：

```python
# alloc_bind_inplace (正确)
def alloc_bind_inplace(...):
    if v_pages.is_cuda:
        _alloc_bind_kernel(...)  # Triton kernel
    else:
        # CPU fallback
        for i in range(...):
            ...

# zero_pages (缺失 fallback)
def zero_pages(raw, pages, num_pages, page_bytes):
    _zero_pages_kernel[grid](...)  # 总是启动 Triton kernel!
```

在 One Pool 模式下：
- `--enable-unified-memory` 使 kvcache 变为 `UnifiedMLATokenToKVPool`
- `MultiEndedAllocator._zero_pages_on_alloc = isinstance(kvcache, UnifiedMLATokenToKVPool)` → `True`
- 每次 alloc 都调用 `_maybe_zero_pages` → `zero_pages`
- `UnifiedKVPool._raw.device = cpu`（因为 `SGLANG_USE_CPU_ENGINE=1`）
- Triton kernel 在 NPU 上执行但访问 CPU buffer → aivec error

**修复**（`launch_server.py`）：

```python
from sglang.kernels.ops.kvcache.zero_pages import _zero_pages_kernel, _BLOCK
import sglang.kernels.ops.kvcache.zero_pages as _zp_mod

_orig_zero_pages = _zp_mod.zero_pages

def _zero_pages_cpu_fallback(raw, pages, num_pages, page_bytes):
    m = int(pages.numel())
    if m == 0:
        return
    page_words = page_bytes // 8
    words = raw[: num_pages * page_bytes].view(torch.int64)
    pg_list = pages.to(torch.int64).tolist()
    for pg in pg_list:
        words[pg * page_words : (pg + 1) * page_words].zero_()

def _zero_pages_patched(raw, pages, num_pages, page_bytes):
    if not raw.is_cuda:
        _zero_pages_cpu_fallback(raw, pages, num_pages, page_bytes)
        return
    _orig_zero_pages(raw, pages, num_pages, page_bytes)

_zp_mod.zero_pages = _zero_pages_patched
# 关键：还要 patch unified_memory_pool 中的 import 引用！
import sglang.srt.mem_cache.unified_memory_pool as _ump
_ump.zero_pages = _zero_pages_patched
```

**重要注意**：`from ... import zero_pages` 在 import 时绑定函数引用。如果不 patch `unified_memory_pool` 模块中的引用，该模块仍会使用原始函数。需要同时 patch **模块级**和**import 绑定处**。

---

### 问题 2：`backup_thread_func` SIGABRT（影响 One Pool HiCache 模式）

**现象**：One Pool HiCache 模式在 `prepare_for_decode` 时 SIGABRT，日志显示 `backup_thread_func` 访问 NPU 失败。

**根因**：

`HybridCacheController`（不是 `HiCacheController`）有自己的 `backup_thread_func` 和 `prefetch_thread_func` 方法，这些方法访问 NPU 进行真实的 KV cache 备份。

HiSim 的 `C_HiCacheController.hook` 在 `HiCacheController` 类级别替换了这些方法，但 `HybridCacheController` 继承自 `HiCacheController` 并**重写了**这些方法，所以实例方法查找时找到的是 `HybridCacheController` 自己的版本。

**修复**（两处，都需要）：

```python
# launch_server.py — class-level patch（在 import 前执行）
import sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller as _hcc_mod
_hcc_mod.HybridCacheController.backup_thread_func = lambda self, *a, **kw: None
_hcc_mod.HybridCacheController.prefetch_thread_func = lambda self, *a, **kw: None

# sglang_hook.py wrapped_init_hicache — instance-level patch（运行时执行）
self.cache_controller.backup_thread_func = lambda *a, **kw: None
self.cache_controller.prefetch_thread_func = lambda *a, **kw: None
```

---

### 问题 3：One Pool HiCache L2 Host Hit=0（L2→L1 load-back 不生效）

这是最复杂的 bug，需要 6 步逐步定位。

**现象**：One Pool HiCache 模式，batch 1 写入 200 token prefix 后驱逐到 L2，batch 2 使用相同 prefix。`match_prefix` 返回 `full=200, dev=0, host=0`（host 应该是 192 但返回 0），导致 L2→L1 load-back 不触发。

**根因链**：

```
1. component_data 是 LIST（按 ComponentType int enum 索引），不是 dict
   ↓
2. _HA_CT.MAMBA in node.component_data → 对 list 检查"元素是否在列表中"
   → 总是返回 False（因为列表元素是 ComponentData 对象，不是 enum 值）
   ↓
3. MAMBA host_value 从未被 mock _mock_start_writing 设置
   ↓
4. 原生 commit_backup 返回的 comp_xfers 中 MAMBA transfer 存在
   但 host_indices=None（原生 write() 无法为 MAMBA 分配 host 内存）
   ↓
5. 节点被驱逐后：MAMBA.value=None, MAMBA.host_value=None
   ↓
6. MAMBA validator: "value is not None or host_value is not None" → False
   （注意：MAMBA validator 检查自己的 host_value，不是 node.backuped
    node.backuped 只检查 FULL.host_value）
   ↓
7. _all_valid 返回 False → best_match 不更新 → host_hit_length=0
```

**排查路径**（6 步）：

| 步骤 | 诊断手段 | 发现 |
|------|----------|------|
| P3-POSTEVICT | 驱逐后遍历树 | root 有 5 children，`v=None hv=set bk=True` ✅ 驱逐正确 |
| P3-KEYS | 对比 child_key 和 prefix | `child_key_in_children=True` ✅ key 匹配 |
| P5-HELPER | 在 `_match_prefix_helper` 内部追踪 | `fkhl=200` 但 `best_match=root` → while loop 执行但 validator 失败 |
| PRE-MATCH | match 前检查节点状态 | `F.val=set F.hv=set M.val=set M.hv=None` → MAMBA host_value 未设置 |
| P5-VAL | 包装 `create_match_validator` 打印每个 validator 返回值 | `comp=FULL result=True` `comp=MAMBA result=False` → MAMBA validator 失败 |
| MAMBA-CHK | 检查 MAMBA transfer | `mamba_xfers=1` 但 `host_indices=None` → 原生 write 无法分配 MAMBA host 内存 |

**修复**（`sglang_hook.py`，3 处，净增 19 行）：

```python
# 修复 1+2: _mock_start_writing 中的 host_value 设置
# 原:
if _HA_CT.FULL in node.component_data and _hi is not None and len(_hi) > 0:
    node.component_data[_HA_CT.FULL].host_value = _hi
if _mamba_xfers and _HA_CT.MAMBA in node.component_data:
    _mhi = getattr(_mamba_xfers[0], 'host_indices', None)
    if _mhi is not None:
        node.component_data[_HA_CT.MAMBA].host_value = _mhi.clone() ...

# 修复后:
try:
    _fv = node.component_data[_HA_CT.FULL]      # 索引访问，不是 in 检查
except (KeyError, IndexError, TypeError):
    _fv = None
try:
    _mv = node.component_data[_HA_CT.MAMBA]     # 索引访问
except (KeyError, IndexError, TypeError):
    _mv = None
if _fv is not None and _hi is not None and len(_hi) > 0:
    _fv.host_value = _hi
if _mamba_xfers and _mv is not None:
    _mhi = getattr(_mamba_xfers[0], 'host_indices', None)
    if _mhi is not None:
        _mv.host_value = _mhi.clone() ...
# 关键：独立 if（不是 elif），当 _mhi=None 时从 device value 设置 fallback
if _mv is not None and _mv.value is not None and _mv.host_value is None:
    _mv.host_value = _mv.value.clone() if hasattr(_mv.value, 'clone') else _mv.value

# 修复 3: _mock_start_loading 中的 MAMBA value 恢复
# 原:
if _mamba_loads and _HA_CT.MAMBA in node.component_data:
    ...
# 修复后:
if _mamba_loads:
    _mdi = getattr(_mamba_loads[0], 'device_indices', None)
    if _mdi is not None:
        try:
            node.component_data[_HA_CT.MAMBA].value = _mdi.clone() ...
        except (KeyError, IndexError, TypeError):
            pass
```

**为什么 fallback 从 device value 设置 host_value 是正确的**：

1. 原生 `commit_hicache_transfer(BACKUP_HOST)` 会将 device 上的 KV 数据拷贝到 host 并设置 `host_value` 指向 host buffer
2. HiSim 中 mock 的 `mh.backup_from_device()` 已经在 L2 池中完成了 Mamba 状态的备份
3. 设置 `host_value = value.clone()` 只是让 validator 通过（`host_value is not None` → True）
4. 实际的 L2→L1 加载由 `mh.load_to_device()` 执行，不依赖 `host_value` 的具体内容

---

### 问题 4：K3 层编号 1-indexed vs 0-indexed（影响 KDA 层数和 slot_bytes）

**现象**：mock 使用 config 的 1-indexed 层列表，但 sglang 的 `kv_cache_configurator` 通过 `(i+1) in kda_layers` 将 config 的 1-indexed 转为 0-indexed。mock 直接使用 1-indexed 导致实际只有 17 层 KDA 而非 18 层。

**根因**：

```
config (1-indexed):  kda_layers = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23]
sglang 转换:          (i+1) in kda_layers → 0-indexed: [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
mock 原代码:          直接使用 config 的 1-indexed 列表 → 层映射偏移
                      → 实际 17 层而非 18 层
```

**修复**（`launch_server.py`）：将 mock 的层列表改为 0-indexed，同时修复 `slot_bytes`（旧值 15151104 → 新值 14653440，conv=bf16 itemsize=2）和 `layer_mapper`（旧 lambda → 精确 dict 映射）。Commit: `2508169`

---

### 问题 5：MAMBA slot_hash 不匹配（影响 L3 restore）

**现象**：KDA 节点从 L2 驱逐到 L3 后，L3 restore 时 `batch_exists_mamba(slot_hash)` 找不到数据。

**根因**：`slot_hash = f"kda_{nid}_{slot_idx}"` 中 backup 和 load 使用不同的 `device_indices`（slot 在驱逐后会被重分配），导致 slot_hash 不一致。

**修复**（`sglang_hook.py`）：slot_hash 只用 `node_id`：`f"kda_{nid}"`。Commit: `46e7578`

---

### 问题 6：MAMBA.value=None 导致 validator 失败（影响 L2→L1 load-back）

**现象**：L3 restore 后，MAMBA.value 仍为 None → MAMBA validator 失败 → best_match 不更新 → L2→L1 load-back 不触发。

**根因**：`_mamba_loads[0].device_indices` 为 None（首个 transfer 无法分配 host 内存），导致整个 value 设置被跳过。

**修复**（`sglang_hook.py`）：遍历所有 transfer 找非 None 的 `device_indices`。Commit: `5abe1cd`

---

### 问题 7：时延模型 — L3 restore 多余 D2H（Phase 4C-3 Issue 1）

**现象**：KDA L3→L2 restore 的时延被高估 ~273.4µs。

**根因**：`backup_from_device(from_l3=True)` 不区分 `from_l3` 参数，总是计入 D2H 时延。但 L3 restore 时数据已在 DRAM（SSD→DRAM 步骤已完成），不需要额外 D2H。

**修复**（`sglang_mock_class.py`）：`from_l3=True` 时不计入 D2H。每次 L3 restore 减少 273.4µs。Commit: `237f879`

---

### 问题 8：时延模型 — KDA L3 read 与 H2D 错误并行化（Phase 4C-3 Issue 2）

**现象**：KDA L3→L1 的总时延被低估 ~276µs。

**根因**：`max(MLA_H2D, KDA_H2D, KDA_L3_read)` 将 KDA L3 read（SSD→DRAM）和 H2D（DRAM→HBM）视为并行，但实际是串行的：SSD → DRAM → HBM。

**修复**（`sglang_hook.py`）：改为 `kda_total = KDA_L3_read + KDA_H2D`（串行），`l2_load = max(MLA_H2D, kda_total)`。修复后 KDA L3→L1 总时延从 1003.7µs 修正为 1279.7µs。Commit: `237f879`

---

## 5. 数据流详解

### 5.1 Two Pools HiCache 数据流

```
                    L1 (Device)                    L2 (Host/DRAM)              L3 (SSD)
              ┌─────────────────┐           ┌─────────────────────┐      ┌──────────────┐
              │ MockTokenToKVPool│           │MockTokenToKVPoolHost│      │ StorageBackend│
              │ (FULL: MLA KV)  │           │ (FULL: MLA KV host) │      │ (file backend)│
              ├─────────────────┤           ├─────────────────────┤      ├──────────────┤
              │ MockMambaStatePool│          │MockMambaStatePoolHost│      │ kvcm_l3      │
              │ (KDA: Mamba)    │           │ (KDA: Mamba host)   │      │ (PaceStorage) │
              └────────┬────────┘           └──────────┬──────────┘      └──────┬───────┘
                       │                                │                        │
                       │  evict (LRU)                   │ overflow (LRU)         │
                       └────────→─────────────────────→┘                        │
                       │  backup_from_device            │ batch_set_mamba         │
                       │  load_to_device                │ batch_exists_mamba      │
                       │                                └←──────────────────────┘
                       │                                  L3→L2 restore
                       
RadixTree (UnifiedRadixCache)
  node.component_data[ComponentType.FULL]  → .value (device), .host_value (host)
  node.component_data[ComponentType.MAMBA]  → .value (device), .host_value (host)
  
  backuped property = FULL.host_value is not None
  evicted property  = FULL.value is None and FULL.host_value is not None
  
  FULL validator:    value is not None or node.backuped
  MAMBA validator:   value is not None or MAMBA.host_value is not None  ← 注意：检查自己的 host_value
```

### 5.2 One Pool HiCache 数据流

```
                    L1 (Unified Device Pool)          L2 (Host/DRAM)              L3 (SSD)
              ┌───────────────────────────┐     ┌─────────────────────┐      ┌──────────────┐
              │ UnifiedKVPool             │     │MockTokenToKVPoolHost│      │ StorageBackend│
              │ ├─ _raw (uint8 buffer)    │     │ (FULL: MLA KV host) │      │ (file backend)│
              │ ├─ MHASubPoolSpec (KV)    │     ├─────────────────────┤      ├──────────────┤
              │ └─ MambaSubPoolSpec       │     │MockMambaStatePoolHost│      │ kvcm_l3      │
              │   (KDA, mock data ops)    │     │ (KDA: Mamba host)   │      │ (PaceStorage) │
              ├───────────────────────────┤     └──────────┬──────────┘      └──────┬───────┘
              │ MultiEndedAllocator       │                │                        │
              │ ├─ virtual_to_physical    │                │                        │
              │ ├─ _free_phys_pages        │                │                        │
              │ ├─ alloc/free/compact      │                │                        │
              │ └─ _zero_pages_on_alloc   │                │                        │
              │   (True for MLA pool)     │                │                        │
              └───────────┬───────────────┘                │                        │
                          │  evict (LRU)                   │                        │
                          └→─────────────────────────────→┘                        │
                           │  backup_from_device            │ batch_set_mamba         │
                           │  load_to_device                │ batch_exists_mamba      │
                           │                                └←──────────────────────┘
                           │                                  L3→L2 restore

关键差异：
  - L1 是统一池（KV+Mamba 共享 _raw buffer），由 MultiEndedAllocator 动态划分
  - MambaPool 被 mock 化（_HiSimMambaPoolMock），但 mamba_allocator 保持真实
  - zero_pages 在 alloc 时被调用（_zero_pages_on_alloc=True for MLA）
  - HybridCacheController 的 async 线程被禁用，备份/加载由 mock 同步执行
```

### 5.3 Mock Write/Load 调用链

```
sglang scheduler → prepare_for_decode
  → cache_controller.start_writing  (mock: _mock_start_writing)
    → for each backup operation:
      → parse pool_transfers → _full_xfers / _mamba_xfers
      → FULL host backup: host.backup_from_device(indices)
      → MAMBA host backup: mh.backup_from_device(slot_idx, slot_hash)
      → L3 write: st.batch_set(indices)
      → L3 KDA write: st.batch_set_mamba(slot_hashes)
      → set node.component_data[FULL].host_value = host_indices
      → set node.component_data[MAMBA].host_value = host_indices or device value (fallback)
      → ack_write_queue.append(ack)

  → cache_controller.start_loading  (mock: _mock_start_loading)
    → for each load_back operation:
      → parse pool_loads → _full_loads / _mamba_loads
      → L3 read: st.batch_get(indices)
      → FULL host load: host.load_to_device_per_layer(...)
      → MAMBA host load: mh.load_to_device(slot_hash)
      → L3→L2 restore: if not in L2, check L3, backup_from_device, load_to_device
      → set node.component_data[MAMBA].value = device_indices
      → ack_load_queue.append(ack)
```

---

## 6. 验证结果

### 6.1 四模式回归测试

| 模式 | 启动 | Batch 1 (5 reqs) | Batch 2 (5 reqs, same prefix) | L2 HIT | 状态 |
|------|------|-------------------|-------------------------------|--------|------|
| Two Pools Plain | ✅ | 5/5 | 5/5 | N/A | ✅ |
| Two Pools HiCache | ✅ | 5/5 | 5/5 | `host=192, mamba_host=1` | ✅ |
| One Pool | ✅ | 5/5 | 5/5 | N/A | ✅ |
| One Pool HiCache | ✅ | 5/5 | 5/5 | `host=192, mamba_host=1` | ✅ |

### 6.2 P2 诊断日志示例（验证通过时的日志）

```
# Batch 1 (首次写入，无 L2 hit)
P2 match_prefix: full=200 dev=200 host=0 mamba_host=0 last_dev=9 last_host=0 best_match=9

# Batch 2 (相同 prefix，L2 hit)
P2 match_prefix: full=200 dev=0 host=192 mamba_host=1 last_dev=9 last_host=18 best_match=18(int)
```

字段含义：
- `full=200`：prefix 完整匹配 200 token
- `dev=0`：device 上 0 token（已被驱逐）
- `host=192`：host 上 192 token（L2 命中）
- `mamba_host=1`：MAMBA host 有 1 个 slot 命中
- `best_match=18`：best match node id=18

### 6.3 测试方法

```bash
# 1. 启动服务器（以 One Pool HiCache 为例）
docker exec -d sglang-kimi-k3-dqy bash -c \
  'export ASCEND_VISIBLE_DEVICES=3; bash /home/d00954154/sglang-k3_docker/start_onepool_hicache.sh'

# 2. 等待就绪
for i in 1 2 3 4 5 6 7 8; do
  sleep 30
  status=$(docker exec sglang-kimi-k3-dqy bash -c \
    "curl -s --max-time 10 http://127.0.0.1:30000/get_server_info >/dev/null 2>&1 && echo READY || echo WAIT")
  echo "Check $i: $status"
  [ "$status" = "READY" ] && break
done

# 3. 发送 batch 1（写入 prefix）
python3 send_requests_v2.py trace_batch1.jsonl 5 127.0.0.1 30000
# 预期: completed: 5/5

# 4. 等待 5 秒（让 L1→L2 驱逐生效）
sleep 5

# 5. 发送 batch 2（相同 prefix，测试 L2 hit）
python3 send_requests_v2.py trace_batch2_late.jsonl 5 127.0.0.1 30000
# 预期: completed: 5/5

# 6. 检查 P2 日志
grep "P2 match" /tmp/hisim_onepool_hicache.log | grep "full=200" | head -5
# 预期: host=192 mamba_host=1
```

### 6.4 测试数据

- `trace_batch1.jsonl`：5 个请求，各有唯一的 200 token prefix，`timestamp=0.0`
- `trace_batch2_late.jsonl`：5 个请求，使用与 batch1 **相同**的 prefix，`timestamp=100.0`
- `output_length=10`（每个请求只生成 10 token，快速完成）

### 6.5 Phase 4C-1 后续验证：L2 eviction + L3 restore 闭环

在修复层编号/slot_bytes/slot_hash/MAMBA.value 后的验证结果：

| 验证项 | 方法 | 结果 |
|--------|------|------|
| L2 LRU 驱逐链 | L2_MAMBA_SIZE=2，写入 4 个 KDA slot | `kda_18 → kda_19 → kda_20 → kda_21` ✅ |
| L3 backup | L2 满后 overflow 到 L3 | `st.batch_set_mamba(slot_hashes)` ✅ |
| L3 restore | L2_MISS → L3_HIT → restore → L2_HIT | slot_hash=`kda_18` 一致 ✅ |
| Device slot 重用 | backup slot=3, load slot=26 | slot_hash 不含 slot → 无混淆 ✅ |
| 四模式回归 | One Pool / Two Pools × HiCache | 全部 5/5×2 ✅ |

### 6.6 Phase 4C-2：动态内存管理验证

#### `max_total_tokens` 在 One Pool 中不作为严格运行时上限

```
One Pool:  max_total_tokens=500 → MLA allocator max_slots=254,900 (>> 500)
Two Pools: max_total_tokens=500 → MLA allocator max_slots=500 (= max_total_tokens)
```

调度器使用 `allocator.available_size()`（物理空闲页数），不检查 `max_total_tokens`。存在上游类似 Bug 报告，**暂不定性**。

#### One Pool vs Two Pools 容量对照（E1 配置）

| 指标 | One Pool | Two Pools |
|------|----------|-----------|
| Total L1 Bytes | 2,999,998,080 (2.79 GB) | 2,999,998,080 (2.79 GB) |
| MLA allocatable pages | **518,289** | **~12,033** |
| Mamba allocatable slots | 203 | 199 |
| max_running_requests | 20 (mamba-capped=36) | 20 (set explicitly) |
| MLA capacity advantage | **43×** | 1× |

#### flush_peer 不触发的原因

- `lazy_compaction=True`（默认启用）
- Buffer 容量 = 两个子池最大容量之和 → 物理空间不存在竞争
- Proactive eviction 在逻辑限制达到时驱逐，不触发物理空间竞争
- **设计正确行为**

#### KDA Replay 统计

| Checkpoint 状态 | COW 设置 | replay_tokens | 在 Virtual Clock？ |
|----------------|----------|---------------|-------------------|
| L1 有 | ✅ cow_src ≠ None | 0 | 不需要 |
| L2 有 | ❌ cow_src=None | prefix_len | ❌ 不在 clock |
| L3 有 | ❌ cow_src=None | prefix_len | ❌ 不在 clock |
| 全部丢失 | ❌ cow_src=None | prefix_len | ❌ 不在 clock |

**COW 不可用的根因**：`backup_thread_func`/`prefetch_thread_func` 被 no-op → load-back 不在 `match_prefix` 之前执行 → `MAMBA.value=None` → COW 不设置。HiSim mock 的 `replay_tokens=prefix_len` 是**保守上界**。

#### Proactive Eviction 机制

- **触发条件**：`evictable_size() > 0`（存在 lock ref=0 的 idle 节点）
- **与空闲容量无关**：One Pool 有 307K-518K 空闲页时仍触发
- **策略驱动，非容量驱动**
- 可通过 `HISIM_DISABLE_PROACTIVE_EVICT=1` 禁用

### 6.7 Phase 4C-3：时延模型验证

#### 五个合成测试结果

| 测试 | 场景 | 修复前 | 修复后 | 预期 |
|------|------|--------|--------|------|
| A | KDA L2-only (D2H+H2D) | 549.4µs | 549.4µs | 549.4µs ✅ |
| B | KDA L3 restore | 1553.1µs (+273.4µs) | **1279.7µs** | 1279.7µs ✅ |
| C | MLA+KDA 同时 | 1003.7µs (-276µs) | **1279.7µs** | 1279.7µs ✅ |
| D | D2H+SSD Write | 2378.4µs | 2378.4µs | 2378.4µs ✅ |
| E | Cross-round Overflow | overflow=0 | overflow=0 | 0 ✅ |

#### 时延模型总结

```
每轮 Virtual Clock:
  clock += L2_load + inference + L2_backup + L3_write_overflow

  L2_load    = max(MLA_H2D, KDA_L3_read + KDA_H2D)     ← MLA/KDA 独立并行, KDA 内部串行
  inference  = AIC roofline predictor
  L2_backup  = max(MLA_D2H, KDA_D2H)
  L3_write_overflow = max(0, L3_write - round_dur)

带宽模型:
  D2H: 饱和曲线 bw(x)=x·B/(t0·B+x), t0=4µs, B=54.4 GB/s
  H2D: 同上, t0=6.67µs
  SSD: 线性 bytes/bandwidth, read=14.6 GB/s, write=8 GB/s

不支持的重叠:
  - OVERLAP_SCHEDULE 强制禁用（跨轮不重叠）
  - D2H/H2D 不双向（串行）
  - KDA SSD write 无排队/固定延迟
```

#### 带宽参数

| 通道 | 方向 | 模型 | 参数 |
|------|------|------|------|
| D2H (HBM→DRAM) | Device→Host | 饱和曲线 | t0=4µs, B=54.4 GB/s |
| H2D (DRAM→HBM) | Host→Device | 饱和曲线 | t0=6.67µs, B=54.4 GB/s |
| SSD Read | SSD→DRAM | 线性 | 14.6 GB/s |
| SSD Write | DRAM→SSD | 线性 | 8 GB/s |

### 6.8 Phase 4C-4：E1 实验（MLA 容量紧张配置）

**配置**：

```
max_total_tokens=12,033, max_mamba_cache_size=200
total=3.0GB, --max-running-requests=20, L2_MAMBA_SIZE=330, hicache_size=10
Workload: 50 reqs × 300 tokens = 15K MLA tokens (> 12K Two Pools MLA 容量)
两轮: Round 1 (unique prefix, ts=0.0-0.49) → Round 2 (相同 prefix, ts=100.0-100.49)
```

#### 四组对照表（Proactive ON/OFF × One Pool/Two Pools）

| Group | Mode | Proactive | Round 2 结果 | L2 hits (host=256) | Proactive Evict | KDA-REPLAY |
|-------|------|-----------|-------------|-------------------|-----------------|------------|
| **G1** | One Pool | **ON** | 50/50 (7.3s) | **100** | 85 | 50 |
| **G2** | One Pool | **OFF** | 50/50 (6.7s) | **22** | **0** | 50 |
| **G3** | Two Pools | **ON** | 50/50 processed | **ALL** (50 unique) | 90 | 43 |
| **G4** | Two Pools | **OFF** | 50/50 (5.0s) | **ALL** (50 unique) | **0** | 43 |

#### 关键发现

**1. Proactive Eviction 限制了 One Pool 的额外容量利用**

```
One Pool + ON:  100 L2 hits (proactive 驱逐 idle 节点，即使有 518K 空闲)
One Pool + OFF:  22 L2 hits (无 proactive，大部分 prefix 保留在 L1)
差异: 78 次不必要的 L2 hit = 78 × 276µs H2D = ~21.5ms 额外搬运时间
```

**2. Two Pools 的 L2 hit 是容量驱动的，Proactive 不影响结果**

```
Two Pools + ON:  ALL L2 hits (proactive + native LRU)
Two Pools + OFF: ALL L2 hits (native LRU only, 容量 12K < 15K 需求)
无差异: Proactive ON/OFF 对 Two Pools 结果相同
```

**3. One Pool 的优势来自两个独立来源**

| 优势来源 | 描述 | 量化 |
|---------|------|------|
| **共享 buffer 容量** | MLA allocator 518K vs 12K | 即使有 proactive eviction，仍有部分 L1 hit |
| **避免不必要驱逐** | Proactive OFF 时不驱逐 idle 节点 | 减少 78 次 L2 hit（21.5ms H2D） |

#### E1 Min L1_avail（Proactive ON 时）

| 模式 | Min L1_avail | 空闲率 |
|------|-------------|--------|
| One Pool | 307,194 | 59.2% free |
| Two Pools | 315 | 2.6% free |

### 6.9 Phase 4C-4：E2 实验（MLA 容量充足配置）

**配置**：`max_total_tokens=138,889, max_mamba_cache_size=150, total=3.0GB`

| 规模 | One Pool | Two Pools | 差异 |
|------|----------|-----------|------|
| 50 reqs | 50 L1 hits | 50 L1 hits | **无差异** |
| 500 reqs | 部分 L2 hits | 部分 L2 hits | **无差异** |

**原因**：并发 MLA 使用量（20 × 300 = 6K pages）远低于两种模式的 MLA 容量。proactive eviction 在批次间驱逐 idle 节点，防止 L1 累积。

### 6.10 容量压力公式

```
MLA 压力 = 并发 MLA 使用 / MLA 容量
并发 MLA 使用 ≈ max_running_requests × prefix_length + overhead

E1: 6K / 12K = 50%  → 压力显著 → One Pool 优势明显
E2: 6K / 138K = 4%  → 无压力 → 无差异
```

---

## 7. 已知限制

### 7.1 原有限制（Phase 4C-1）

| 限制 | 影响 | 规避方式 |
|------|------|----------|
| NPU chip 0 die 0 有 aivec 硬件错误 | 所有模式在 chip 0 上崩溃 | `ASCEND_VISIBLE_DEVICES=2` |
| `HybridCacheController` async 线程完全禁用 | 备份/加载由 mock 同步执行，不模拟真实异步时序 | 功能不受影响，仅影响时序模拟精度 |
| MAMBA `host_value` 从 device value 设置 | host_value 指向 device tensor 而非真实 host buffer | validator 只检查 `is not None`；实际 L2 数据由 `mh.backup_from_device` 管理 |
| `support_triton=False` 全局禁用 | 所有 Triton kernel 被替换为 Python fallback | HiSim 前向是 mock，不影响功能 |
| `use_mla_backend=False`（在 ModelRunner 上） | 不使用 MLA kernel 路径 | 通过 `head_num=1, head_dim=288` 使 MHA 容量等价于 MLA |
| `MambaPool` data ops 全部 no-op | Mamba 状态数据不真实 | 前向 mock 不读取数据；`mamba_allocator` 保持真实 |

### 7.2 新增限制（Phase 4C-2 ~ 4C-4）

| 限制 | 影响 | 状态 |
|------|------|------|
| **COW 不可用** | replay_tokens=prefix_len（保守上界），COW-HIT=0 | 根因已定位（async 线程禁用），mock 无法修复 |
| **KDA Replay Compute Cost 不在 Virtual Clock** | TTFT 低估（mock 中为保守上界，真实系统可能为 0） | 待接入，需 Profiling 数据 |
| **`max_total_tokens` 不作为严格上限** | One Pool MLA 容量 >> max_total_tokens | 设计行为？上游 Bug？暂不定性 |
| **Proactive eviction 限制 One Pool 容量利用** | 78 次不必要 L2 hit（~21.5ms H2D） | 已确认，可通过 `HISIM_DISABLE_PROACTIVE_EVICT=1` 禁用 |
| **KDA SSD write 线性模型** | 无排队/固定延迟，可能低估 | 已标注边界，暂不修复 |
| **MLA/KDA 带宽不共享** | 可能高估（共享 memory bus） | 简化假设，暂不修复 |
| **OFFLINE 模式无跨轮重叠** | D2H/H2D 不双向，OVERLAP_SCHEDULE 禁用 | 简化假设 |
| **P2 match 统计口径** | raw line count 包含多轮数据 | 需按 Round 去重 |
| **另一会话修改容器内文件** | `unified_tree_core.py` 语法错误已修复，但可能再次冲突 | 需协调 |

---

## 8. 适配指南（其他环境）

### 8.1 适配到新 sglang 版本

1. **检查 `zero_pages` 是否已有 fallback**：
   - 文件：`sglang/kernels/ops/kvcache/zero_pages.py`
   - 如果已有 `is_cuda` 检查 → 删除 `launch_server.py` 中的 zero_pages patch
   - 如果仍无 → 保留 patch，但检查函数签名是否变化

2. **检查 `HybridCacheController` 是否仍存在**：
   - 文件：`sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`
   - 如果类名变化 → 更新 `launch_server.py` 中的 patch 目标
   - 如果 `backup_thread_func` 已被移除或重构 → 删除对应 patch

3. **检查 `component_data` 数据结构**：
   - 文件：`sglang/srt/mem_cache/unified_cache/unified_tree_core.py`
   - 确认 `component_data` 仍然是 list（按 `ComponentType` int enum 索引）
   - 如果变为 dict → 删除 `sglang_hook.py` 中的 indexed access 修复（改回 `in` 检查）

4. **检查 validator 逻辑**：
   - 文件：`sglang/srt/mem_cache/unified_cache/components/mamba_component.py`
   - 确认 MAMBA validator 仍检查 `MAMBA.host_value is not None`
   - 如果改为检查 `node.backuped` → 可能不需要 host_value fallback

5. **检查 `MultiEndedAllocator` API**：
   - 文件：`sglang/srt/mem_cache/multi_ended_allocator.py`
   - 确认 `_zero_pages_on_alloc`、`_maybe_zero_pages`、`alloc`、`free` 等方法签名
   - 确认 `_compact_pending`、`_flush`、`_flush_peer_for_alloc` 仍存在

6. **检查 `init_unified_mamba_pools` 参数**：
   - 文件：`sglang/srt/mem_cache/unified_memory_pool.py`
   - 确认 `max_mamba_cache_size`、`use_mla_backend`、`kv_lora_rank` 等参数名

### 8.2 适配到新硬件

1. **aivec 错误**：如果新硬件没有此问题，可以移除 `ASCEND_VISIBLE_DEVICES=3`
2. **`SGLANG_USE_CPU_ENGINE=1`**：如果新硬件支持在模拟器中执行真实 kernel，可能不需要此标志，但仍需要 `support_triton=False` 和 MambaPool no-op
3. **内存预算**：`get_available_gpu_memory` patch 使用全部系统内存。如果新环境内存有限，需要调整 `hicache-size` 参数

### 8.3 适配到新模型

1. **`kimi_linear_config` / `mambaish_config` patch**：如果新模型不是 K3，需要修改模型检测逻辑
2. **KDA layers**：`[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]` 和 Full Attention layers `[3,7,11,15,19]` 是 K3 的 0-indexed 层列表。新模型需要根据 config 中的 1-indexed 列表转换为 0-indexed
3. **`entry_bytes` 计算**：`head_num=1, head_dim=288` 对应 K3 MLA（kv_lora_rank=512, qk_rope_head_dim=64）。新模型需要重新计算
4. **`slot_bytes` 计算**：`14653440` 是 K3 KDA 的 slot 大小（TP=8, conv=bf16 itemsize=2, temporal=fp32 itemsize=4）。新模型需要根据 `num_mamba_layers × bytes/slot/layer` 重新计算。**注意：不要假设 conv itemsize=4**
5. **`layer_mapper`**：使用精确 dict 映射，指定每层在 sub-pool 中的 index。不要使用 lambda 简写，避免层映射偏移

### 8.4 关键诊断手段

| 诊断 | 实现方式 | 用途 |
|------|----------|------|
| P2 match_prefix 日志 | `wrapped_match_prefix` 打印 `full/dev/host/mamba_host/best_match` | 确认 L2 hit 是否生效 |
| P0-DIAG alloc 配置 | `_diag_alloc` 首次调用打印 allocator 配置 | 确认 `_zero_pages_on_alloc`、kvcache 类型 |
| P2-ZERO zero_pages 追踪 | `_diag_zero` 打印 phys_pages 和 kvcache 内部状态 | 确认 zero_pages 是否被调用、buffer device |
| Unified memory diag | flush 时打印 alloc/free/compact/flush 计数 | 确认 MultiEndedAllocator 活跃度 |
| KDA write/load diag | flush 时打印 `calls/mamba_pt/l3_writes/l3_hits` | 确认 L1→L2→L3 闭环 |

### 8.5 常见问题排查

| 问题 | 可能原因 | 排查方法 |
|------|----------|----------|
| SIGABRT on startup | NPU chip 错误 | `ASCEND_VISIBLE_DEVICES=3` |
| SIGABRT on first prefill | `zero_pages` 无 CPU fallback | 检查 `launch_server.py` 中 zero_pages patch |
| SIGABRT on decode | `backup_thread_func` 访问 NPU | 检查 `HybridCacheController` 线程 patch |
| `completed: 0/5` | 服务器崩溃或超时 | 检查 `/tmp/hisim_*.log` 中的 traceback |
| `host=0` (L2 不命中) | MAMBA host_value 未设置 | 检查 `sglang_hook.py` 中 indexed access + fallback |
| `mamba_ping_pong_track_buffer is None` | 诊断代码副作用或 pycache 过期 | 清除 `__pycache__`，重启容器 |

---

## 9. 文件索引

### 修改的文件（HiSim 侧）

| 文件路径 | 说明 |
|----------|------|
| `src/hisim/src/hisim/simulation/sglang/launch_server.py` | sglang 启动前 patch（~370 行新增） |
| `src/hisim/src/hisim/simulation/sglang/sglang_hook.py` | HiCache 路由树 hook（~200 行新增/修改） |
| `src/hisim/src/hisim/simulation/sglang/sglang_mock_class.py` | Mock host pool 类（~50 行修改） |

### 参考的文件（sglang 侧，只读）

| 文件路径 | 说明 |
|----------|------|
| `python/sglang/kernels/ops/kvcache/zero_pages.py` | `zero_pages()` — 无 `is_cuda` 检查（问题 1 根因） |
| `python/sglang/kernels/ops/memory/virtual_slot.py` | `alloc_bind_inplace()` — 有 `is_cuda` + CPU fallback（参考模式） |
| `python/sglang/srt/mem_cache/unified_memory_pool.py` | `UnifiedMLATokenToKVPool.zero_physical_pages`（调用 zero_pages）；`UnifiedKVPool._raw` |
| `python/sglang/srt/mem_cache/multi_ended_allocator.py` | `_zero_pages_on_alloc`、`_maybe_zero_pages`、`alloc` |
| `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py` | `_match_prefix_helper`（树遍历+validator）；`backuped`/`evicted` property |
| `python/sglang/srt/mem_cache/unified_cache/components/full_component.py` | `create_match_validator`：`value is not None or node.backuped` |
| `python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py` | `create_match_validator`：`MAMBA.value is not None or MAMBA.host_value is not None` |
| `python/sglang/srt/mem_cache/unified_cache/component_type.py` | `ComponentType` enum：FULL=0, SWA=1, MAMBA=2, C128=3 |
| `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` | `HybridCacheController`（有独立 `backup_thread_func`） |
| `python/sglang/srt/managers/cache_controller.py` | `HiCacheController`（基类） |
| `python/sglang/srt/mem_cache/unified_radix_cache.py` | `_reset_full`、`writing_check`、`_execute_kv_backup` |
| `python/sglang/srt/managers/schedule_batch.py` | `prepare_for_decode`、`_mamba_radix_cache_v2_req_prepare_for_extend` |

### 启动脚本

| 文件路径 | 模式 |
|----------|------|
| `/home/d00954154/sglang-k3_docker/start_p1b.sh` | Two Pools HiCache |
| `/home/d00954154/sglang-k3_docker/start_onepool_hicache.sh` | One Pool HiCache |
| `/home/d00954154/sglang-k3_docker/start_e1.sh` | E1 实验（One Pool/Two Pools 参数化） |
| `/home/d00954154/sglang-k3_docker/start_e2.sh` | E2 实验（大容量配置） |
| `/home/d00954154/sglang-k3_docker/start_e1_noevict.sh` | E1 Proactive OFF 启动器 |
| `/home/d00954154/sglang-k3_docker/start_p0_cow.sh` | E1 One Pool 启动器（端口 30002） |

### 测试数据

| 文件路径 | 说明 |
|----------|------|
| `/home/d00954154/sglang-k3_docker/trace_batch1.jsonl` | 5 reqs, 各有唯一 200 token prefix, timestamp=0.0 |
| `/home/d00954154/sglang-k3_docker/trace_batch2_late.jsonl` | 5 reqs, 相同 prefix, timestamp=100.0 |
| `/home/d00954154/sglang-k3_docker/trace_e1_r1.jsonl` | 50 reqs × 300 tokens, unique prefix, ts=0.0-0.49 |
| `/home/d00954154/sglang-k3_docker/trace_e1_r2.jsonl` | 50 reqs, 相同 prefix, ts=100.0-100.49 |
| `/home/d00954154/sglang-k3_docker/trace_e2_500_r1.jsonl` | 500 reqs × 300 tokens, Round 1 |
| `/home/d00954154/sglang-k3_docker/trace_e2_500_r2.jsonl` | 500 reqs, Round 2 |
| `/home/d00954154/sglang-k3_docker/trace_kda_reuse.jsonl` | 10 reqs, 50% 重用, 200 tokens |
| `/home/d00954154/sglang-k3_docker/send_requests_v2.py` | HTTP 请求发送脚本（含 OFFLINE simulation 参数） |

### 验证脚本

| 文件路径 | 说明 |
|----------|------|
| `/tmp/test_timing_model.py` | 确定性时延验证脚本（测试 A-E） |
| `/tmp/test_de_verify.py` | 测试 D 事件表 + 测试 E 正溢出验证 |

---

## 10. Git 提交历史（关键提交）

```
a811a06 cleanup: remove P2 match COW diagnostic, keep KDA-REPLAY stats     ← HEAD
237f879 fix: KDA timing model — remove extra D2H from L3 restore + serial L3+H2D
b550ea0 cleanup: remove RUN-BATCH diagnostic, keep KDA COW/Replay stats
4cce0de revert: remove sync-load (load queue empty before match_prefix)
5fbc32d fix: synchronous load-back before match_prefix (async threads disabled)
71553bf diag: KDA COW/Replay stats in get_new_batch_prefill
a387aea diag: add RUN-BATCH entry log
b13e4b1 cleanup: remove HISIM_DISABLE_L3 (negative test completed)
5abe1cd fix: MAMBA value restore uses correct transfer
46e7578 fix: MAMBA slot_hash use node_id only
2508169 fix: correct K3 layer numbering (1-indexed→0-indexed) and slot_bytes
0c62f61 fix: MAMBA host_value indexed access + fallback (Phase 4C-1 基线)
        ← 初始修复：component_data indexed access + MAMBA host_value fallback

45e161d fix: also patch zero_pages reference in unified_memory_pool module
        ← zero_pages CPU fallback + unified_memory_pool import 绑定

1416886 fix: add CPU fallback for zero_pages kernel (match alloc_bind_inplace)
        ← zero_pages CPU fallback 基础 patch

b787157 fix: patch HybridCacheController thread funcs at class level (not instance)
        ← backup_thread_func class-level patch

4363526 fix: One Pool HiCache — unit fix + use_mla_backend + assert bypass
        ← One Pool 基础架构支持

e583578 feat: KDA L2→L1 load-back verified — two-batch test success
        ← Two Pools HiCache L2→L1 首次验证通过

778f94e fix: KDA L1→L2→L3 closed loop — hash, overflow, L3→L2 restore
        ← L1→L2→L3 闭环修复

f01e834 feat: OFFLINE mode timestamp-based request scheduling
        ← OFFLINE 时间戳调度

61eced6 feat(k3): complete Kimi-K3 MLA+KDA multi-level cache adaptation for HiSim
        ← K3 适配初始版本

92cb52b hisim+kimi-k3_v0
        ← 项目初始版本
```

### 净改动（相对 `0c62f61` 基线）

| 文件 | 改动内容 |
|------|---------|
| `launch_server.py` | K3 层编号 0-indexed，slot_bytes 修正，layer_mapper 精确映射，L2 容量 env vars |
| `sglang_hook.py` | MAMBA slot_hash=node_id，host_value 索引访问，MAMBA.value 正确 transfer，KDA timing 修复，KDA-REPLAY 统计 |
| `sglang_mock_class.py` | getattr 兼容，_last_evicted_hashes，backup_from_device(from_l3) 修复 |

### 注意：另一会话的提交

另一会话 (`ses_fa8ed6366ffeyZjmzZrT6jcxDS`) 在同一分支上有并存提交 (`3cf2212`, `0e2e066`, `549a5ee`, `c91e5b5`/`2382125` revert, `f4d9bec`)。其捐赠检测/COW 修改曾导致 `completed: 0/5` 崩溃，已恢复。其容器内 `unified_tree_core.py` 修改曾导致 IndentationError，已修复。

---

## 11. 总结

### 11.1 已完成

1. **四种部署模式全部可用**：Two Pools Plain/HiCache + One Pool/HiCache
2. **L1→L2→L3 闭环**：write-through + evict + L3 write/read + L3→L2 restore
3. **L2→L1 load-back**：在 Two Pools HiCache 和 One Pool HiCache 均验证通过
4. **八个核心问题修复**：
   - `zero_pages` CPU fallback（影响 One Pool）
   - `backup_thread_func` class-level no-op（影响 One Pool HiCache）
   - MAMBA `host_value` indexed access + fallback（影响 One Pool HiCache L2 hit）
   - K3 层编号 1-indexed→0-indexed（影响 KDA 层数和 slot_bytes）
   - MAMBA `slot_hash` 用 node_id（影响 L3 restore）
   - MAMBA `value` 遍历所有 transfer（影响 L2→L1 load-back）
   - 时延模型 L3 restore 多余 D2H（每次减少 273.4µs）
   - 时延模型 KDA L3+H2D 错误并行化（修正 +276µs）
5. **OFFLINE 时间戳调度**：支持按 timestamp 顺序释放请求
6. **Phase 4C-2 动态内存管理验证**：确认 `max_total_tokens` 在 One Pool 中不作为严格上限，确认 proactive eviction 是策略驱动（非容量驱动）
7. **Phase 4C-3 时延模型验证**：五个合成测试全部通过，确认带宽模型和重叠规则
8. **Phase 4C-4 容量对照实验**：
   - E1（MLA 紧张）：One Pool 有 L1 hit（~50% ON / ~90% OFF），Two Pools 为 0%
   - E2（MLA 充足）：无差异（并发使用远低于两种模式的 MLA 容量）
   - 四组对照（Proactive ON/OFF × One Pool/Two Pools）：确认优势来自共享容量和避免不必要驱逐两个独立来源

### 11.2 核心结论

1. **One Pool 动态共享收益已确认**：在 MLA 容量紧张时（E1: 15K vs 12K），One Pool Round 2 有 L1 hit（~50% ON / ~90% OFF），Two Pools 为 0%
2. **优势来自两个独立来源**：共享 buffer 物理容量（518K vs 12K）+ proactive eviction 策略（ON 限制利用，OFF 最大化保留）
3. **优势在 MLA 容量充足时消失**：当并发使用远低于两种模式的 MLA 容量时（E2），proactive eviction 控制了 L1 驻留，两模式行为一致
4. **时延模型已修复两个确认错误**：L3 restore 多余 D2H（-273.4µs/次）和 KDA L3+H2D 错误并行（+276µs/次）
5. **TTFT 暂不作为最终性能结论**：KDA Replay Compute Cost 未接入 Virtual Clock，COW 不可用导致保守上界
6. **Proactive eviction 是 One Pool 的潜在优化点**：关闭后可将 L1 保留率从 ~50% 提升至 ~90%+，减少 78 次不必要 L2 hit（~21.5ms H2D）

### 11.3 核心设计决策

| 决策 | 原因 | 影响 |
|------|------|------|
| `use_mla_backend=False` + `head_num=1, head_dim=288` | MLA kernel 在 NPU 模拟中崩溃 | 用 MHA 尺寸模拟 MLA 容量（5760 B/token） |
| MambaPool data ops no-op | `copy_from/get_cpu_copy` 触发 NPU kernel | 数据不真实，但 `mamba_allocator` 保持真实 |
| `support_triton=False` | Triton kernel 在 NPU 模拟中崩溃 | 使用 Python fallback |
| mock `start_writing`/`start_loading` | 同步执行备份/加载 | 不模拟真实异步时序 |
| `component_data` indexed access | list 不是 dict，`in` 检查无效 | 使用 try/except 索引访问 |
| MAMBA `host_value` 从 device value fallback | 原生 `write()` 无法分配 MAMBA host 内存 | 让 validator 通过，实际 L2 数据由 mock 管理 |
| K3 层编号使用 0-indexed | sglang 通过 `(i+1) in kda_layers` 转换 config 的 1-indexed | mock 需匹配 0-indexed |
| slot_hash 只用 `node_id` | backup/load 的 slot_idx 不同 | 避免 hash 不匹配 |
| KDA L3 read + H2D 串行化 | SSD→DRAM→HBM 是物理串行路径 | 正确计算 KDA L3→L1 总时延 |
| `from_l3=True` 不计 D2H | L3 restore 时数据已在 DRAM | 消除多余 D2H 计时 |

### 11.4 下一步建议

#### 立即执行（P0）

1. **Task 3：Two Pools 固定分区扫描** — 固定 total=3GB, E1 workload，扫描 MLA=12K/40K/80K Tokens，找到 Two Pools 的最优静态分区
2. **Task 1 剩余：H2D/D2H/SSD 字节统计** — 从 StateManager 累积器读取 Duration，按带宽模型反推 Bytes

#### 短期执行（P1）

3. **Task 4：工作负载敏感性** — Prefix=300/1000/3000, 并发=5/10/20，找到 One Pool 优势消失的临界点
4. **Task 5：真实 Trace 准备** — 选取高复用 Coding Agent Session

#### 中期执行（P2）

5. **KDA Replay Compute Cost 接入** — 需 K3 Profiling 数据，校准后才能发布最终 TTFT
6. **与另一会话协调** — 避免同时修改容器内文件
7. **`max_total_tokens` 行为定性** — 确认是设计行为还是实现缺陷
