# SSD 仿真模块

## 概述

HiSim 的 L3 层（SSD 存储）prefetch 时序默认用固定带宽模型 `time = data_size / bandwidth`，忽略了 NAND 读延迟、多通道并行、IOPS vs 带宽两区间等真实 SSD 行为。

本模块提供 SimpySSD 仿真器替换固定带宽模型，通过环境变量 `HISIM_SSD_SIM` 控制：

| 模式 | 仿真器 | 建模内容 | 精度 |
|------|--------|---------|------|
| 原始（默认） | 无 | `time = data / bandwidth` | 低 |
| SimpySSD | Python | NAND 延迟 + 多 channel 并行 | 中（规格书 ±5%） |

## 安装

SimpySSD 是纯 Python，无需编译：

```bash
cd 3rdparty/simpyssd
pip install -e .
```

或通过 `PYTHONPATH` 直接使用（无需安装）：

```bash
export PYTHONPATH=/path/to/src/hisim/3rdparty:$PYTHONPATH
```

## 使用

### 环境变量 `HISIM_SSD_SIM`

| 值 | 模式 | 说明 |
|----|------|------|
| 未设置 / 空 | 原始 | 固定带宽模型 |
| `1` 或 `simpyssd` | SimpySSD | 使用内置 `es3500p_pcie5.cfg` |
| `/path/to.cfg` | SimpySSD | 使用指定配置文件 |

### 示例

```bash
# 原始模式（默认）
python -m hisim.simulation.bench_serving --sim-config-path config.json ...

# SimpySSD 模式
HISIM_SSD_SIM=simpyssd python -m hisim.simulation.bench_serving --sim-config-path config.json ...

# 自定义 SSD 配置
HISIM_SSD_SIM=/path/to/my_ssd.cfg python -m hisim.simulation.bench_serving ...
```

## 配置文件

### 默认配置：ES3500P V7-3840 PCIe 5.0

位置：`3rdparty/simpyssd/configs/es3500p_pcie5.cfg`

基于规格书参数：

| 规格书参数 | 配置参数 | 值 |
|-----------|---------|-----|
| 平均读延时 @1QD: 60μs | `read_latency_us` | 60 |
| 顺序读带宽: 14,600 MB/s | `channel_bw_mbps` | 1825 (= 14600/8) |
| 随机 4K 读 IOPS: 3,100,000 | `num_chip` × `num_channel` | 24 × 8 = 192 (192/60μs ≈ 3.2M) |
| NAND 类型: 3D TLC | `read_latency_us` = TLC 平均 | 60 (LSB=40/CSB=60/MSB=80, avg=60) |
| 页大小: 4KB | `pg_sz_kb` | 4 |

### SimpySSD 配置参数详解

```ini
[flash]
channel_bw_mbps:     1825      # 每 channel DMA 带宽 (MB/s) = 顺序读带宽 ÷ channel 数
read_latency_us:     60        # NAND 读延迟 (μs) = 规格书"平均读延时 @1QD"
write_latency_us:    500       # NAND 写延迟 (μs) = TLC program 典型值
pg_sz_kb:            4         # NAND 物理页大小 (KB)
num_chip:            24        # 每 channel 芯片数 = IOPS × 读延迟 ÷ channel 数
num_channel:         8         # NAND 通道数，企业级 NVMe 典型 8
```

### 如何为不同 SSD 配置

1. 查 SSD 规格书，获取：
   - 平均读延时 @1QD (μs) → `read_latency_us`
   - 顺序读带宽 (MB/s) → `channel_bw_mbps` = 带宽 ÷ 8
   - 随机 4K 读 IOPS → `num_chip` = IOPS × 读延迟 ÷ 8
   - NAND 类型 → 写延迟: SLC=200, MLC=300, TLC=500 (μs)

2. 写配置文件并使用：
   ```bash
   HISIM_SSD_SIM=/path/to/your_config.cfg python -m hisim.simulation.bench_serving ...
   ```

## 集成原理

替换 `sglang_hook.py` 中 `C_HiCacheController.calc_prefetch_pages()`：

```
原始:  time = required_pages × page_size / bandwidth

SimpySSD:
  1. 构造 N 个读命令 (N = required_pages)
  2. 分配到 8×24=192 个并行单元 (顺序: 条带分配, 随机: 随机分配)
  3. 事件驱动模拟 NAND 读 + channel 传输
  4. 返回总时间 (含并行重叠)
```

## 验证结果

### vs ES3500P V7 PCIe 5.0 规格书

| 指标 | 规格 | SimpySSD | 比值 |
|------|------|----------|------|
| 4K 读延迟 @1QD | 60μs | 62.2μs | 1.04x |
| 4K 随机读 IOPS | 3,100,000 | 2,980,000 | 0.96x |
| 128K 顺序读带宽 | 14,600 MB/s | 14,581 MB/s | 1.00x |

三项指标均在规格书 ±5% 以内。

## 文件结构

```
3rdparty/
  simpyssd/                        # SimpySSD（纯 Python，无需编译）
    __init__.py                    # 导出 SSDTimingSimulator
    sim.py                         # 离散事件引擎
    ssd_config.py                  # 配置解析
    util.py                        # 辅助函数
    ssd_estimate.py                # SSD 模型（Chip/Channel/SSD/Cmd）
    ssd_timing_adapter.py          # 适配器: predict_read_time()
    setup.py                       # pip install -e .
    configs/
      es3500p_pcie5.cfg            # ES3500P V7 PCIe 5.0 规格（默认）
```
