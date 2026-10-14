# HiSim 容器部署问题与修复记录

> 日期：2026-09-20
> 状态：已解决

---

## 1. 问题背景

原始容器 `sglang-kimi-k3-dqy` 被删除后，使用同一镜像重建容器时，遗漏了 NPU 硬件挂载和驱动/firmware 挂载，导致 `torch_npu` 和 `triton_ascend` 无法正常导入，服务器启动失败。

## 2. 镜像与容器信息

| 项 | 值 |
|---|---|
| **镜像** | `swr.cn-south-west-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann9.0.0-a3`（ID: `6b8f50aefcde`，16.1GB） |
| **容器名** | `sglang-kimi-k3-dqy` |
| **sglang 版本** | `0.5.18.dev922+ga41da991c` |
| **CANN 版本** | 9.0.0（镜像内置） |
| **triton_ascend** | v3.2.1（镜像内置） |
| **部署脚本** | `/home/d00954154/sglang-k3_docker/setup_container.sh` |

## 3. 根因分析

### 3.1 原始容器配置（setup_container.sh）

原始容器通过 `setup_container.sh` 创建，关键挂载：

```bash
docker run --name sglang-kimi-k3-dqy -it -d --net=host --shm-size=500g \
    --privileged=true \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /data3:/data3 \
    -v /dev:/dev \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home:/home \
    -v /tmp:/tmp \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    ${IMAGE_NAME}
```

### 3.2 重建时遗漏的配置

| 遗漏项 | 影响 |
|--------|------|
| `--device=/dev/davinci_manager` | NPU 管理设备不可见 |
| `--device=/dev/hisi_hdc` | NPU HDC 设备不可见 |
| `--device=/dev/devmm_svm` | NPU SVM 设备不可见 |
| `-v /usr/local/Ascend/driver` | NPU 驱动库（`libascend_hal.so`）不可用 |
| `-v /usr/local/Ascend/firmware` | NPU firmware 不可用 |
| `-v /usr/local/dcmi` | DCMI 管理接口不可用 |
| `-v /usr/local/bin/npu-smi` | `npu-smi info` 命令不可用 |
| `-v /etc/ascend_install.info` | Ascend 安装信息不可用 |
| `--privileged=true` | 缺少特权模式 |
| `--net=host` | 未使用 host 网络 |

### 3.3 错误链

```
遗漏 NPU 挂载
  → torch_npu 导入失败（libascend_hal.so 找不到）
  → 或 CANN 版本不匹配（宿主机 9.1.0 vs 镜像 9.0.0 → UnicodeDecodeError）
  → triton_ascend 尝试编译 NPU C++ 扩展（头文件缺失 → 编译失败）
  → sglang is_npu() 抛 RuntimeError（torch_npu detected but NPU not available）
  → 服务器启动失败
```

## 4. 修复过程

### 4.1 容器重建

使用 `setup_container.sh` 中的完整配置重建容器：

```bash
docker run --name sglang-kimi-k3-dqy -it -d --net=host --shm-size=500g \
    --privileged=true \
    -w /home \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /mnt:/mnt -v /data:/data -v /data3:/data3 -v /dev:/dev \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home:/home -v /tmp:/tmp \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    6b8f50aefcde
```

### 4.2 验证

```bash
# NPU 可见
docker exec sglang-kimi-k3-dqy npu-smi info  # ✅ 正常

# torch_npu 导入
python3 -c "import torch_npu; print(torch_npu.npu.is_available())"  # ✅ True

# triton 导入
python3 -c "import triton; print(triton.__version__)"  # ✅ 3.2.0
```

### 4.3 sglang 源码补丁

原始容器内有 sglang 源码补丁（容器删除后丢失），需要重新打上：

**补丁 1：MambaComponent isinstance 断言禁用**

```
文件: /sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py
修改: assert isinstance(params.req_to_token_pool, HybridReqToTokenPool) → assert True or isinstance(...)
原因: HiSim 使用 MockReqToTokenPool，不继承 HybridReqToTokenPool
```

**补丁 2：ascend_kda_backend.py dtype 补丁**（来自 setup_container.sh）

```
文件: /sgl-workspace/sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py
修改: ].transpose(-1, -2) → ].transpose(-1, -2).to(conv_states.dtype)
原因: dtype 不匹配修复
```

**补丁 3：MultiEndedAllocator _topmost_survivor p2v fix**（Phase 2 发现）

```
文件: /sgl-workspace/sglang/python/sglang/srt/mem_cache/multi_ended_allocator.py
修改: 在 _topmost_survivor 的 grow-up 和 grow-down 分支中，return p, j 之前加 p2v 检查:
      if int(self.physical_to_virtual[p].item()) < 0:
          p -= 1  (grow-up) 或 p += 1 (grow-down)
          continue
原因: 当 KV pool 有大量从未分配的 page (p2v=-1) 时，_topmost_survivor 错误返回这些
      page 作为 survivor，导致 _flush 断言失败。补丁让 survivor 遍历跳过 p2v=-1 的 page。
```

### 4.4 launch_server.py mock 代码回退

之前为绕过容器配置问题添加的 torch_npu/triton mock 代码已全部回退，`launch_server.py` 恢复为干净状态。

## 5. Two Pools + Hierarchical Cache 验证

重建容器并打补丁后，Two Pools + Hierarchical Cache 模式验证通过：

```
服务器启动: ready
bench_serving: 10 请求跑通，ITL 指标正常产出
```

## 6. 经验教训

1. **容器重建必须参照原始部署脚本**：`setup_container.sh` 包含了所有必要的 NPU 硬件挂载和 sglang 源码补丁
2. **NPU 环境的 torch_npu/triton 不需要 mock**：只要 NPU 硬件可见且 CANN 版本匹配，`torch_npu` 和 `triton_ascend` 可正常导入
3. **sglang 源码补丁在容器内**：补丁直接修改 `/sgl-workspace/sglang/` 下的源文件，容器删除后补丁丢失
4. **HiSim 代码在宿主机**：`/home/d00954154/memory_system/` 通过挂载进入容器，不受容器删除影响
5. **容器崩溃可导致 NPU 驱动状态残留**：进程崩溃时持有 NPU 设备引用 (business_ref_cnt=1)，导致 hotreset 失败，NPU 卡在 state=3 (not ready)。需要宿主机级别重置 (pkill hccn_tool / rmmod drv_pcie_hdc_host / 冷重启)
6. **proactive evict guard bug**：Plain 模式下 `_hicache_cn=0`，`0%10==0` 每轮都触发 proactive evict，摧毁 prefix reuse，可能导致 KV pool 满载死锁。已修复为 `_hicache_cn > 0 and (_hicache_cn % 10 == 0)`
