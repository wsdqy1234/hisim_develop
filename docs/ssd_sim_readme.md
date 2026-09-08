# SSD 仿真模块

## 概述

HiSim 的 L3 层（SSD 存储）prefetch 时序默认用固定带宽模型 `time = data_size / bandwidth`，忽略了 NAND 读延迟、多通道并行、IOPS vs 带宽两区间等真实 SSD 行为。

本模块提供 SimpySSD 仿真器替换固定带宽模型，通过环境变量 `HISIM_SSD_SIM` 控制：

| 模式 | 仿真器 | 建模内容 | 精度 |
|------|--------|---------|------|
| 原始（默认） | 无 | `time = data / bandwidth` | 低 |
| SimpySSD | Python | NAND 延迟 + 多 channel 并行 | 中（规格书 ±5%） |
| **多 Tier SimpySSD** | Python + kvcm | **每 tier 独立 SimpySSD 配置** | 中（多介质混合） |

## 安装

SimpySSD 是纯 Python，无需编译：

```bash
cd 3rdparty/simpyssd
pip install -e .
```

kvcm 需编译（含 per-tier 命中数支持）：

```bash
cd 3rdparty/kvcm_l3
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
pip install build/kvcm_py_optimizer-*.whl  # 或直接用 PYTHONPATH
```

## 使用

### 环境变量 `HISIM_SSD_SIM`

| 值 | 模式 | 说明 |
|----|------|------|
| 未设置 / 空 | 原始 | 固定带宽模型 |
| `1` 或 `simpyssd` | SimpySSD | 使用内置 `es3500p_pcie5.cfg` |
| `/path/to.cfg` | SimpySSD | 使用指定配置文件 |

### 单 Tier 示例

```bash
HISIM_SSD_SIM=simpyssd python -m hisim.simulation.bench_serving ...
```

### 多 Tier 示例

在 config JSON 的 `platform` 中配置 `storage_tiers`：

```json
{
  "platform": {
    "tier_read_mode": "parallel",
    "storage_tiers": [
      {
        "name": "local_nvme",
        "ssd_config": "3rdparty/simpyssd/simpyssd/configs/es3500p_pcie5.cfg",
        "read_bandwidth_gb": 14.6,
        "kvcm_storage_type": "pace",
        "kvcm_bandwidth_mbps": 14600,
        "kvcm_capacity_gb": 100,
        "kvcm_priority": 0
      },
      {
        "name": "remote_nfs",
        "ssd_config": "3rdparty/simpyssd/simpyssd/configs/nfs_ssd.cfg",
        "read_bandwidth_gb": 3.0,
        "kvcm_storage_type": "nfs",
        "kvcm_bandwidth_mbps": 3000,
        "kvcm_capacity_gb": 500,
        "kvcm_priority": 1
      }
    ]
  }
}
```

```bash
HISIM_SSD_SIM=simpyssd python -m hisim.simulation.sglang.launch_server \
  --enable-hierarchical-cache ...
```

## 多 Tier 工作原理

```
HiSim config JSON (唯一配置源)
  ├─ storage_tiers[].name           → kvcm storages[].unique_name (init 时动态覆盖)
  ├─ storage_tiers[].ssd_config     → SimpySSD SSDTimingSimulator(cfg) per tier
  ├─ storage_tiers[].kvcm_*         → kvcm storages[].storage_type/bandwidth/capacity/priority
  └─ tier_read_mode                 → "parallel" (各 tier 并行读, 取 max) 或 "serial" (串行, 取 sum)

运行时:
  batch_exists() → kvcm.GetCacheLocation()
    → 返回 per_tier_hit_block_num + tier_names (不改返回类型, 缓存内部)
  
  batch_exists_per_tier() → 返回缓存的 {tier_name: hit_count}
  
  calc_prefetch_pages(per_tier_hits, ...)
    → 对每个 tier 用对应 SimpySSD config 算读时间
    → parallel: 总时间 = max(各 tier 时间)
    → serial:   总时间 = sum(各 tier 时间)
```

### 配置参数

| 参数 | 说明 | 来源 |
|------|------|------|
| `name` | tier 名称，与 kvcm `unique_name` 对应 | HiSim config |
| `ssd_config` | SimpySSD 配置路径 | HiSim config |
| `read_bandwidth_gb` | fallback 带宽 (SimpySSD 未启用时) | HiSim config |
| `kvcm_storage_type` | kvcm 存储类型 (pace/nfs/hf3fs 等) | HiSim config → kvcm |
| `kvcm_bandwidth_mbps` | kvcm tier 带宽 | HiSim config → kvcm |
| `kvcm_capacity_gb` | kvcm tier 容量 | HiSim config → kvcm |
| `kvcm_priority` | kvcm tier 优先级 (0=最快) | HiSim config → kvcm |
| `tier_read_mode` | 读模式: `parallel` 或 `serial` | HiSim config |

### 向后兼容

| 场景 | 行为 |
|------|------|
| 不配 `storage_tiers` + 不设 `HISIM_SSD_SIM` | 原始带宽模型（不变） |
| 不配 `storage_tiers` + `HISIM_SSD_SIM=simpyssd` | 单 SimpySSD（不变） |
| 配 `storage_tiers` + `HISIM_SSD_SIM=simpyssd` | 多 tier，每 tier 用各自 config |
| 配 `storage_tiers` + 不设 `HISIM_SSD_SIM` | 多 tier，每 tier 用 `read_bandwidth_gb` |

`batch_exists` 返回值不变，sglang 无感知。`batch_exists_per_tier` 是新方法，只 hisim hook 调用。

## 验证结果

### vs ES3500P V7 PCIe 5.0 规格书（单 tier）

| 指标 | 规格 | SimpySSD | 比值 |
|------|------|----------|------|
| 4K 读延迟 @1QD | 60μs | 62.2μs | 1.04x |
| 4K 随机读 IOPS | 3,100,000 | 2,980,000 | 0.96x |
| 128K 顺序读带宽 | 14,600 MB/s | 14,581 MB/s | 1.00x |

## 文件结构

```
3rdparty/
  simpyssd/                        # SimpySSD（纯 Python）
    simpyssd/
      __init__.py
      sim.py, ssd_config.py, util.py, ssd_estimate.py, ssd_timing_adapter.py
      configs/es3500p_pcie5.cfg
    setup.py
  kvcm_l3/                         # kvcm（C++ pybind11）
    kvcm/kv_cache_manager/...      # 含 per_tier_hit_block_num 支持
    kvcm_py_optimizer/             # Python binding
```
