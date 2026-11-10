# HiSim Qwen3 仿真运行指南

> 分支：`develop_lfz_0.5.19` | SGLang：`0.5.19` | 验证日期：2026-10-26

## TL;DR

```bash
# 1. 启动容器（镜像见下文）
# 2. 容器内设环境变量 + 装 4 个依赖包（见步骤 2-3）
# 3. 启动仿真 server
export PYTHONPATH=/home/d00954154/HiSim_Qwen3/src
export TORCH_TRANSFER_TO_NPU=0
export ASCEND_RT_VISIBLE_DEVICES=7

python3 -m hisim.simulation.sglang.launch_server \
  --model-path /home/d00832329/Qwen3-0.6B \
  --load-format dummy \
  --device cpu \
  --enable-hierarchical-cache \
  --port 30010

# 4. 另开终端跑 benchmark
python3 -m hisim.simulation.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30010 \
  --model /home/d00832329/Qwen3-0.6B \
  --dataset-name hisim-collection \
  --dataset-path test/assets/dataset/hisim_collection_requests.jsonl \
  --num-prompts 4 --bench-mode simulation
```

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| **容器镜像** | `quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3` | Ascend CANN 9.0.0 + sglang 0.5.19 |
| **sglang** | 0.5.19 | 镜像内置（editable 装于 `/sgl-workspace/sglang`） |
| **torch** | 2.10.0+cpu | 镜像内置 |
| **torch_npu** | 2.10.0 | 镜像内置，NPU 可用时自动加载 |
| **aiconfigurator** | 0.5.0 | ❌ 镜像无，需从 lfz-sglang 拷贝（含 systems 数据） |
| **xgboost** | 3.2.0 | ❌ 镜像无，需从 lfz-sglang 拷贝 |
| **simpyssd** | 0.1.0 | ❌ 镜像无，从 HiSim_Qwen3/3rdparty 装 |
| **kvcm_py_optimizer** | 0.1.0 | ❌ 镜像无，从 HiSim_Qwen3/3rdparty 装（需拷 _native.so） |
| **HiSim 分支** | `develop_lfz_0.5.19` | 代码在 `/home/d00954154/HiSim_Qwen3` |
| **Qwen3 模型** | 0.6B / 32B 等 | `--load-format dummy` 只需 config+tokenizer，不加载权重 |

### 可用的 Qwen3 模型（本机）

| 路径 | 说明 |
|------|------|
| `/home/d00832329/Qwen3-0.6B` | 0.6B，28层，适合快速验证 |
| `/home/d00832329/Qwen3-32B` | 32B |
| `/home/models/Qwen3-30B-A3B-W8A8` | 30B MoE W8A8 |
| `/home/weights/Qwen3-30B-A3B` | 30B MoE |

---

## 2. 启动容器

```bash
docker run --name dqy-hisim-qwen3 -it -d --net=host --shm-size=500g \
  --privileged=true \
  -w /home \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/devmm_svm \
  --entrypoint=bash \
  -v /mnt:/mnt \
  -v /data:/data \
  -v /dev:/dev \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home:/home \
  -v /tmp:/tmp \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
  quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3
```

> **注意**：必须挂载 `/home` 和 `/tmp`（HiSim_Qwen3 代码和 Qwen3 模型在 `/home` 下）。NPU 设备挂载（davinci_manager 等）是 torch_npu 检测 NPU 硬件所必需的。

---

## 3. 容器内安装缺失依赖

镜像里没有 `aiconfigurator`、`xgboost`、`simpyssd`、`kvcm_py_optimizer`（lfz-sglang 容器是创建后单独装的，不在镜像里）。需手动安装：

### 3a. 从 lfz-sglang 拷贝 aiconfigurator + xgboost（含真实 systems 数据）

```bash
SITE=/usr/local/python3.11.15/lib/python3.11/site-packages

# 从 lfz-sglang 拷到共享 /tmp（两容器都挂载 /tmp）
docker cp lfz-sglang:$SITE/aiconfigurator /tmp/aiconfigurator
docker cp lfz-sglang:$SITE/xgboost /tmp/xgboost
docker cp lfz-sglang:$SITE/xgboost.libs /tmp/xgboost.libs
docker cp lfz-sglang:$SITE/aiconfigurator-0.5.0.dist-info /tmp/aiconfigurator.dist-info
docker cp lfz-sglang:$SITE/xgboost-3.2.0.dist-info /tmp/xgboost.dist-info

# 新容器读入 site-packages
docker exec dqy-hisim-qwen3 bash -c "
  cp -r /tmp/aiconfigurator /tmp/xgboost /tmp/xgboost.libs $SITE/ &&
  cp -r /tmp/aiconfigurator.dist-info $SITE/aiconfigurator-0.5.0.dist-info &&
  cp -r /tmp/xgboost.dist-info $SITE/xgboost-3.2.0.dist-info
"
```

> **为什么从 lfz-sglang 拷而不 pip install？** aiconfigurator 的 `systems/` 性能数据库是 git-lfs 数据，PyPI wheel 里是 LFS 指针，`pip install` 后加载会 KeyError。lfz-sglang 里的是真实数据。

### 3b. 从 3rdparty 装 simpyssd + kvcm_py_optimizer

```bash
docker exec dqy-hisim-qwen3 bash -c "
  cd /home/d00954154/HiSim_Qwen3/3rdparty &&
  pip install -e ./simpyssd -e ./kvcm_l3
"
```

### 3c. 拷贝 kvcm 的 _native.so（编译产物，editable 装不编译）

```bash
docker cp lfz-sglang:$SITE/kvcm_py_optimizer/_native.cpython-311-aarch64-linux-gnu.so /tmp/_native.so

docker exec dqy-hisim-qwen3 bash -c "
  cp /tmp/_native.so /home/d00954154/HiSim_Qwen3/3rdparty/kvcm_l3/kvcm_py_optimizer/_native.cpython-311-aarch64-linux-gnu.so
"
```

### 3d. 验证全部导入

```bash
docker exec dqy-hisim-qwen3 python3 -c "
import aiconfigurator, xgboost, simpyssd, kvcm_py_optimizer, sglang, torch
print('全导入OK', sglang.__version__, torch.__version__, xgboost.__version__)
"
```

---

## 4. 环境变量

写一个 env 脚本方便复用：

```bash
docker exec dqy-hisim-qwen3 bash -c 'cat > /tmp/qwen3_env.sh <<EOF
export PYTHONPATH=/home/d00954154/HiSim_Qwen3/src
export BENCHMARK_TEST_MODEL_PATH=/home/d00832329/Qwen3-0.6B
export HISIM_CONFIG_PATH=/home/d00954154/HiSim_Qwen3/test/assets/mock/config.json
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCH_TRANSFER_TO_NPU=0
export ASCEND_RT_VISIBLE_DEVICES=7
EOF'
```

| 环境变量 | 值 | 作用 |
|------|------|------|
| `PYTHONPATH` | HiSim_Qwen3/src | **覆盖**镜像里 hisim 的 editable 安装（默认指向 K3 分支的 `/home/d00954154/hisim/src`） |
| `TORCH_TRANSFER_TO_NPU` | **0** | 禁止 transfer_to_npu 重定向 tensor 操作到 NPU（mock tensor 的 `torch.cat` uint64 在 NPU 上会崩） |
| `ASCEND_RT_VISIBLE_DEVICES` | 7 | 用 7 卡（避免干扰 lfz-sglang 容器的 K3 工作） |
| `HISIM_CONFIG_PATH` | test/assets/mock/config.json | 仿真配置（accelerator/predictor/scheduler） |
| `FLASHINFER_DISABLE_VERSION_CHECK` | 1 | 禁 flashinfer 版本检查 |

### ⚠️ 不要设的环境变量

| 环境变量 | 为什么不能设 |
|------|------|
| `TORCH_DEVICE_BACKEND_AUTOLOAD=0` | 会导致 torch_npu 不加载 → sglang 的 `is_npu()` 检测到 torch_npu 但 `torch.npu` 不存在 → raise RuntimeError "torch_npu detected, but NPU device is not available" |

---

## 5. 运行仿真

### 5a. 启动 server（launch_server 模式）

```bash
docker exec -d dqy-hisim-qwen3 bash -c '
  source /tmp/qwen3_env.sh &&
  cd /home/d00954154/HiSim_Qwen3 &&
  exec python3 -m hisim.simulation.sglang.launch_server \
    --model-path /home/d00832329/Qwen3-0.6B \
    --load-format dummy \
    --device cpu \
    --enable-hierarchical-cache \
    --port 30010 \
    > /tmp/qwen3_server.log 2>&1
'
```

等约 40 秒，日志出现 `The server is fired up and ready to roll!` 即成功。

### 关键参数说明

| 参数 | 值 | 为什么 |
|------|------|------|
| `--model-path` | Qwen3-0.6B 等 | Qwen3 模型路径 |
| `--load-format` | dummy | 只读 config+tokenizer，不加载权重（HiSim 是仿真器，forward 被 mock） |
| **`--device`** | **cpu** | **关键！** mock tensor 必须在 CPU 上。不指定则默认 NPU → `torch.cat` uint64 在 NPU 上崩。分支的 `wrapped_init_torch_distributed` hook 会处理 CPU 路径的 `_init_cpu_threads_env` |
| `--enable-hierarchical-cache` | (flag) | 开启 L1+L2(+L3) 分层缓存仿真 |
| `--port` | 30010 | 避开 30000（lfz-sglang 可能占用） |

### ⚠️ 必须用 launch_server（server 模式），不要用 SGLangBenchmarkRunner（in-process）

- `launch_server.py`（server 模式）：hook 在主进程安装，scheduler 子进程 fork 继承 → **hook 正常生效**
- `SGLangBenchmarkRunner`（sglang_bench.py，in-process Engine）：Engine fork scheduler 子进程时 hook 时机有问题，`wrapped_init_torch_distributed` 可能没挂上 → `_init_cpu_threads_env` 崩

history.md 的验证也是用 launch_server 模式跑的。

### 5b. 运行 benchmark

```bash
docker exec dqy-hisim-qwen3 bash -c '
  source /tmp/qwen3_env.sh &&
  cd /home/d00954154/HiSim_Qwen3 &&
  python3 -m hisim.simulation.bench_serving \
    --backend sglang \
    --base-url http://127.0.0.1:30010 \
    --model /home/d00832329/Qwen3-0.6B \
    --dataset-name hisim-collection \
    --dataset-path test/assets/dataset/hisim_collection_requests.jsonl \
    --num-prompts 4 \
    --output-file /tmp/qwen3_metrics.json \
    --bench-mode simulation
'
```

> dataset 名是 `hisim-collection`（连字符），不是 `hisim_collection`（下划线）。

---

## 6. 验证结果

### 成功标志

server 日志出现：
```
The server is fired up and ready to roll!
All requests received. Starting simulation now.
Simulation results saved to /tmp/hisim/output.
```

### 期望指标（Qwen3-0.6B, 4 请求 hisim-collection）

| 指标 | 期望值 | 实测值 | 说明 |
|------|------|------|------|
| `completed` | 4 | 4 | 全部完成 |
| `mean_tpot_ms` | ~456 (history.md) | 457.59 | 与历史记录一致 |
| `mean_ttft_ms` | <5 | 2.31 | 正常 |
| `prefix_cache_reused_ratio` | >0 | 0.0556 | 前缀复用生效（数值取决于 dataset 重叠度） |
| L2 backup latency | >0（部分 iteration） | 3 个 iteration 有 | L2 分层缓存在工作 |
| `hybrid_ssm` | False | False | 正确（Qwen3 无 Mamba） |

### 指标文件位置

- 仿真指标：`/tmp/hisim/output/metrics.json`
- 迭代日志：`/tmp/hisim/output/iteration.jsonl`（查 L2/L3 backup/load/hit）
- 请求日志：`/tmp/hisim/output/request.jsonl`

---

## 7. 常见坑与排查

### 坑 1：`_init_cpu_threads_env` AttributeError

```
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'init_cpu_threads_env'
```

**原因**：aarch64 CPU 路径调 `torch.ops.sgl_kernel.*`，但镜像没编译 sgl_kernel。分支的 `wrapped_init_torch_distributed` hook 能处理，但只在 launch_server 模式生效。

**解决**：用 `launch_server.py`（server 模式），不要用 `SGLangBenchmarkRunner`（in-process）。

### 坑 2：`torch.cat` uint64 NPU 错误

```
RuntimeError: cat: NPU function error: call aclnnCat failed
AclNN_Parameter_Error: tensor 0 not implemented for DT_UINT64
```

**原因**：没指定 `--device cpu`，mock tensor 在 NPU 上，NPU 不支持 uint64 的 `torch.cat`。

**解决**：加 `--device cpu` + `TORCH_TRANSFER_TO_NPU=0`。

### 坑 3：`is_npu()` RuntimeError

```
RuntimeError: torch_npu detected, but NPU device is not available or visible.
```

**原因**：设了 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，torch_npu 不加载，`torch.npu` 不存在，但 sglang 检测到 torch_npu 包。

**解决**：**不要设** `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。让 torch_npu 正常 auto-load（NPU 硬件可用，`is_npu()` 返回 True）。

### 坑 4：hisim 导入到 K3 分支

**原因**：镜像里 hisim 是 editable 安装，指向 `/home/d00954154/hisim/src`（K3 分支 `develop_dqy_evict_reuse`）。

**解决**：设 `PYTHONPATH=/home/d00954154/HiSim_Qwen3/src` 覆盖。验证：`python3 -c "import hisim; print(hisim.__file__)"` 应显示 HiSim_Qwen3 路径。

### 坑 5：aiconfigurator `PackageNotFoundError`

**原因**：只拷了 aiconfigurator 目录，没拷 `.dist-info` 元数据。

**解决**：同时拷 `aiconfigurator-0.5.0.dist-info` 和 `xgboost-3.2.0.dist-info`（见步骤 3a）。

### 坑 6：kvcm `_native` ModuleNotFoundError

**原因**：`pip install -e kvcm_l3` 是 editable 装，不编译 C 扩展。

**解决**：从 lfz-sglang 拷 `_native.cpython-311-aarch64-linux-gnu.so` 到 editable 源目录（见步骤 3c）。

### 坑 7：check_framework skip 测试

`test_simulation_sglang_runner.py` 的 `check_framework("vllm")` 返回 False（大小写敏感，包名是 `vLLM`）→ 测试被 skip。

**解决**：直接调测试函数绕过 pytest skipif：
```bash
python3 -c "import test_simulation_sglang_runner as t; t.test_benchmark_sglang()"
```
（但注意：in-process 模式有坑 1 的问题，推荐用 launch_server 模式）

---

## 8. 清理

```bash
# 停止 server
docker exec dqy-hisim-qwen3 pkill -f "launch_server.*30010"

# 停止并删除容器
docker stop dqy-hisim-qwen3 && docker rm dqy-hisim-qwen3
```

---

## 9. 分支与 K3 适配的关系

| | develop_lfz_0.5.19 (Qwen3) | develop_dqy_evict_reuse (K3) |
|---|---|---|
| 模型 | Qwen3 (纯 MLA/MHA) | Kimi-K3 (MLA + KDA/Mamba) |
| `mamba2_config` | 不设（干净） | 无条件设 `True` |
| hook 目标 | 双 hook: `HiRadixCache` + `UnifiedRadixCache` | 仅 `UnifiedRadixCache` |
| sgl_kernel mock | 仅 `load_utils`（hook 够用） | 5 层 mock（launch_server 920 行） |
| launch_server.py | 84 行 | 920 行 |
| predictor | `aiconfigurator`（Qwen3 用） | `k3_profile_based`（K3 专用） |
| K3 代码 | 无 | 有（k3_predictor/kda_obs/kda_policy） |

两个分支同源（共同祖先 `79c2973`），独立分叉，互不污染。
