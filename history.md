# memory_system 工作历史

> 本文件记录对本项目所做的关键操作，供后续 agent 读取并继续追加新条目（最新置底）。
> 每条尽量含：日期、commit、做了什么、关键文件/位置、注意事项。

---

## 2026-07-07 — 从 tair-kvcache 迁移 hisim 并集成 kvcm_optimizer L3 仿真

**背景**：把 tair-kvcache 下的 hisim 仿真组件提取到 memory_system 独立维护，并保留其 L3 多层级缓存仿真能力（依赖 tair-kvcache 的 kvcm_optimizer 模块）。

**依赖分析结论**：
- hisim 本身高度自包含（纯 pip 包，无 Bazel），运行期隐式依赖 sglang/torch/psutil/transformers/requests/packaging。
- 唯一外部耦合 = L3 功能，全部集中在 `src/hisim/src/hisim/simulation/sglang/sglang_mock_class.py` 的 `MockHiCacheStorage`：导入 `kvcm_py_optimizer` 并调用 `OptimizerConfigLoader`/`OptimizerManager`。
- L3 调用链为纯 CPU/内存（RadixTree 前缀匹配 + 驱逐 + Stats），不依赖网络/磁盘/redis，故可脱离 Bazel 用 CMake 重建。

**完成的事项**：

1. **搬迁 hisim**（commit `38a1e28` `init hisim`）：`tair-kvcache/hisim` → `memory_system/src/hisim`，整目录复制，不改源码逻辑。hisim 自带 `.gitignore`（已覆盖 `*.so`/`build/`/`dist/`/`*.egg-info`）。

2. **vendor kvcm_optimizer 源码**到 `src/hisim/3rdparty/kvcm/`：73 个文件（30 .cc + 43 .h），来自 `kv_cache_manager/{optimizer,meta,data_storage,common}`。只取 common_define 子集，剔除所有 storage backend（dummy/hf3fs/mooncake/nfs/vineyard/tair_mempool）。
   - 唯一源码改动：`optimizer/pybind/py_optimizer_binding.cc` 的 `PYBIND11_MODULE(kvcm_py_optimizer, ...)` 改名 `_native`，以便作为 `kvcm_py_optimizer._native` 子模块被包 `__init__.py` 重新导出。

3. **vendor 依赖**：rapidjson（header-only，自本地 `/home/wty/OpenViking/third_party/rapidjson`，打了 `0001-document_h.patch`）；`alog_stub`/`autil_stub` 最小桩头（hisim 不调 `LoggerBroker.InitLogger`，alog 路径运行时不执行，桩仅满足编译链接）。剔除 redis/hiredis/boost/3fs/mooncake/protobuf/grpc 等全部重依赖。

4. **独立 wheel 构建**（commit `be6df5f` `支持kvcm_optimizer L3多层级仿真功能`）：`3rdparty/CMakeLists.txt`（`pybind11_add_module(_native ...)`，C++17，输出到 `kvcm_py_optimizer/`）+ `setup.py`（打 `kvcm_py_optimizer` wheel）。.so 不入 hisim 包，作为独立 wheel `pip install`。`kvcm_py_optimizer/__init__.py` 导出符号 + `get_config_path()`。

5. **patch `sglang_mock_class.py`**：
   - import：`from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer` → `import kvcm_py_optimizer`
   - config 路径：硬编码 `parents[4]` → `os.environ.get("HISIM_KVCM_CONFIG_PATH") or kvcm_py_optimizer.get_config_path()`
   - **API 适配器**（见下"注意事项"）：`WriteCache` 去 `write_token_ids`；`GetCacheLocation` 传 `block_mask=0`、`input_len=len(keys)*_KVCM_BLOCK_SIZE`（`_KVCM_BLOCK_SIZE=16` 取自 bundled config）。

6. **文档**：`src/hisim/README.md`/`README_zh.md` 加 L3 章节；`3rdparty/README.md` 详解构建/维护/适配器原理。

7. **.gitignore**：memory_system 根补 `__pycache__/`/`*.pyc`/`*.egg-info/`/`build/`/`dist/`/`*.so`。

8. **测试**：cmake 编译一次通过；wheel 安装后从 site-packages 独立 import；`OptimizerConfigLoader.load` + `OptimizerManager.Init` + `ClearAllCaches` 通过；端到端 `MockHiCacheStorage` 的 `batch_set`/`batch_exists`(命中长度 3)/`clear` 全跑通。

9. **push** 到 `origin/develop_lfz`，远程 hooks PASSED。

**注意事项 / 后续需关注**：
- ⚠️ **hisim↔kvcm pybind 预先存在的 API 版本错位**（upstream hisim#53/#61 与 kvcm#168 不同步）：原 hisim `batch_set`/`batch_exists` 调用签名与 vendored kvcm pybind 不匹配。已做 best-effort 适配器（见上第 5 条 + `3rdparty/README.md`），`block_mask=0`/`input_len` 取值为语义推断，未来与更新 upstream kvcm API 对齐时需重新审视。
- ⚠️ wheel 当前 tag 是 `py3-none-any`（含 aarch64 .so，仅本机 cp311-aarch64 可 import）；跨平台/跨 Python 版本需重建。
- ⚠️ `sglang_mock_class.py` 里 `_CURRENT_DIR`（L27）现已无引用（仅定义），保留未删。
- ⚠️ bundled `optimizer_startup_config_load.json` 的 `trace_file_path`/`output_result_path` 仍是 upstream 的 `/mnt/baiyi/...` 硬编码路径；hisim 不调 `DirectRun`/`AnalyzeResults` 故不触发，但自定义场景注意。
- 构建依赖：cmake>=3.18、g++>=9、pybind11>=2.12（含 `native_enum.h`，本机为 3.0.3）、setuptools+wheel。rapidjson/alog/autil 已 vendor，无需联网。

**关键路径速查**：
- hisim 包：`src/hisim/src/hisim/`
- L3 后端源码：`src/hisim/3rdparty/kvcm/kv_cache_manager/`
- wheel 构建：`src/hisim/3rdparty/`（`cmake -B build -S . && cmake --build build && python setup.py bdist_wheel --dist-dir dist`）
- 适配器所在：`src/hisim/src/hisim/simulation/sglang/sglang_mock_class.py`（`MockHiCacheStorage`）
- 配置覆盖：环境变量 `HISIM_KVCM_CONFIG_PATH`

---

## 2026-07-07 — 对接 SimpySSD：SSD 读时序仿真替换固定带宽模型

**背景**：HiSim L3 prefetch 时序默认用固定带宽模型 `time = data_size / bandwidth`，忽略 NAND 读延迟、多通道并行、IOPS vs 带宽两区间。引入 SimpySSD（纯 Python SSD 仿真器）替换。

**完成的事项**：

1. **vendor SimpySSD** 到 `3rdparty/simpyssd/`（commit `658ce28`）：从 `/home/lfz/test/SimpySSD/src/` 提取核心 SSD 模型（已从 GNN 解耦），6 个 .py 文件 + 配置。目录结构为 `simpyssd/simpyssd/`（包内包），`pip install -e .` 可安装。相对导入（`.sim`/`.ssd_estimate` 等）已修正。

2. **修改 `sglang_hook.py`**：`C_HiCacheController` 加 `_ssd_simulator` 懒加载 + `_calc_prefetch_pages_ssd()` + `calc_prefetch_pages()` 开头检查 SSD sim。环境变量 `HISIM_SSD_SIM` 控制开关：
   - 未设置 → 原始带宽模型（不变）
   - `simpyssd`/`1` → 用内置 `es3500p_pcie5.cfg`
   - 自定义路径 → 用指定 cfg

3. **配置文件** `configs/es3500p_pcie5.cfg`：基于 ES3500P V7-3840 PCIe 5.0 规格书（8ch×24chip=192 并行, TLC 60μs read, 1825 MB/s/ch）。验证：4K延迟 1.04x、IOPS 0.96x、带宽 1.00x vs 规格书。

4. **README** `docs/ssd_sim_readme.md`：使用说明 + 配置参数详解 + 验证结果。

**关键路径速查**：
- SimpySSD 包：`src/hisim/3rdparty/simpyssd/simpyssd/`
- 集成点：`src/hisim/src/hisim/simulation/sglang/sglang_hook.py`（`C_HiCacheController.calc_prefetch_pages`）
- 配置：`3rdparty/simpyssd/simpyssd/configs/es3500p_pcie5.cfg`
- 开关：环境变量 `HISIM_SSD_SIM`

---

## 2026-07-07 — 修复 hisim 在 sglang dev 版本上的运行兼容性

**背景**：本地 sglang 为 dev build `0.1.dev11386`，不在 hisim 兼容列表 `['0.5.6'...'0.5.9']` 内，hook 注册到 sglang 内部类时多处不兼容。

**完成的事项**（commit `3cfe8f2`）：

1. **`sglang_hook.py` 兼容修复**：
   - `C_TokenizerManagerHook._send_one_request`：改为 `*args` 可变参数签名，兼容 2 参数（新版）和 4 参数（旧版）。
   - `C_ModelRunnerHook.override_initialize`：补充 `use_ngram_embedding=False`/`is_hybrid_swa=False`/`is_draft_worker=False` 三个 sglang dev 版本需要的属性。
   - `StorageMetricsCollector` 导入容错：`sglang.srt.metrics.collector` 模块在 dev 版本不存在，用 try/except 降级为 `None`。

2. **SimpySSD 目录重构**：原 `3rdparty/simpyssd/` 根目录直接放 .py 文件，`pip install` 报 `package directory 'simpyssd' does not exist`。重构为 `simpyssd/simpyssd/` 子目录，`pip install -e .` 正常工作。

3. **新增 `config_h20.json`**：使用 `h20_sxm` 设备 + 本地 aic 数据路径 `3rdparty/aic_data`（数据从 `/home/lfz/tmp/hisim/aic/` 拷贝，64MB，.gitignore 排除）。

4. **.gitignore** 加 `3rdparty/aic_data/`。

5. **验证**：基础 benchmark（5 req, 11.34 req/s）、kvcm hierarchical cache（3 req, 5.01 req/s）、simpyssd（3 req, 5.01 req/s）均通过。模型用 `/mnt/data1/Qwen3-Reranker-4B`（qwen3, 36 层, 4B）。

**注意事项**：
- aic 数据（`3rdparty/aic_data/`）不入 git，需从 `/home/lfz/tmp/hisim/aic/` 拷贝或自行下载 H20 AIC 数据。
- sglang dev 版本会产生版本不兼容警告但不阻塞运行（VersionDispatcher 用 newest fallback）。

---

## 2026-07-08 — 多 tier L3 存储仿真：kvcm per-tier 命中 + SimpySSD 多配置 + 并行/串行模式

**背景**：HiSim 原始 L3 仿真用单一带宽/SSD 配置，无法建模多 tier 存储（如 local NVMe + remote NFS）。kvcm 内部已有 per-tier 数据但 Python API 未暴露。需要：HiSim config 统一管理 tier 配置，kvcm 做数据淘汰/查询，SimpySSD 按 tier 配置模拟预取时间。

**完成的事项**（commit `f867278`，10 文件，+396/-138 行）：

1. **kvcm C++ 改动（3 文件）**——暴露 per-tier 命中数：
   - `insight_simulator_types.h`：`GetCacheLocationRes` 加 `per_tier_hit_block_num`（`vector<int64_t>`）和 `tier_names`（`vector<string>`）。
   - `optimizer_manager.cc`：`GetCacheLocation()` 从 `ReadRecord` 拷贝 `per_tier_hit_blocks` 和 `tier_names` 到 result（数据已存在于 `ReadRecord`，只需透传）。
   - `py_optimizer_binding.cc`：加两个 `def_readonly` 暴露新字段。`pybind11/stl.h` 已 include，自动转 Python list。

2. **HiSim 配置类型（3 文件）**——`types.py`/`sim_args.py`/`config.py`：
   - 新增 `StorageTierConfig` dataclass：`name`/`ssd_config`/`read_bandwidth_gb`/`kvcm_storage_type`/`kvcm_bandwidth_mbps`/`kvcm_capacity_gb`/`kvcm_priority`。
   - `PlatformConfig` 加 `storage_tiers: list[StorageTierConfig]` 和 `tier_read_mode: str`（`"parallel"` 或 `"serial"`）。
   - `sim_args.py` 的 `from_json()` 和 `config.py` 的 `get_platform_config()` 都解析 `storage_tiers`。

3. **`sglang_mock_class.py`**——`MockHiCacheStorage` 改造：
   - `init_kvcm()`：读 HiSim config 的 `storage_tiers`，动态覆盖 kvcm 模板 JSON 的 `storages[]`（用 HiSim 的 `name`/`kvcm_*` 参数），写临时文件给 `OptimizerConfigLoader` 加载。HiSim 是唯一配置源。
   - `batch_exists()`：返回值不变（`int`），但内部缓存 `self._last_per_tier_hits = dict(zip(res.tier_names, res.per_tier_hit_block_num))`。
   - 新增 `batch_exists_per_tier() -> dict[str, int]`：返回缓存的 per-tier 命中数，不额外调 kvcm。只 hisim hook 调用，sglang 无感知。

4. **`sglang_hook.py`**——多 tier SimpySSD + 并行/串行：
   - `_init_ssd_simulator()`：如果有 `storage_tiers`，为每个 tier 创建独立 `SSDTimingSimulator(tier.ssd_config)`，存入 `_tier_sims` dict。无 `storage_tiers` 退化为单 tier（现有逻辑）。
   - `_calc_prefetch_multi_tier(per_tier_hits, page_size, max_dur)`：对每个 tier 用对应 SimpySSD 算读时间；`parallel` 模式取 `max`，`serial` 模式取 `sum`。
   - `calc_prefetch_pages()`：`required_pages` 是 `dict` 时走多 tier 路径，是 `int` 时走单 tier 路径。
   - `handle_prefetch_operation()`：`_storage_hit_query` 后调用 `batch_exists_per_tier()` 获取 per-tier dict；chunked 续传时按比例缩减 per-tier hits。

5. **示例 config** `test/assets/mock/config_h20_multitier.json`：两 tier（local_nvme + remote_nfs），`tier_read_mode: "parallel"`。

6. **README** 更新：多 tier 工作原理、配置参数、向后兼容说明。

**验证**：
```
Multi-tier parallel: 100 tokens, 240us  (各 tier 并行读, 取 max)
Multi-tier serial:   100 tokens, 353us  (各 tier 串行读, 取 sum)
Single-tier:         100 tokens, 410us  (单 SimpySSD, 现有行为)
Original bandwidth:  100 tokens, 410us  (固定带宽模型, 不变)
```

**向后兼容**：不配 `storage_tiers` → 退化为单 tier；`batch_exists` 返回类型不变，sglang 无感知。

**关键路径速查**：
- kvcm per-tier 改动：`3rdparty/kvcm_l3/kvcm/kv_cache_manager/optimizer/{config/insight_simulator_types.h, manager/optimizer_manager.cc, pybind/py_optimizer_binding.cc}`
- 配置类型：`src/hisim/simulation/types.py`（`StorageTierConfig`/`PlatformConfig`）
- init_kvcm 动态覆盖：`src/hisim/simulation/sglang/sglang_mock_class.py`（`MockHiCacheStorage.init_kvcm`）
- 多 tier 计算：`src/hisim/simulation/sglang/sglang_hook.py`（`C_HiCacheController._calc_prefetch_multi_tier`）
- 示例 config：`test/assets/mock/config_h20_multitier.json`

**注意事项**：
- kvcm `.so` 需重新编译才能使用 per-tier 功能（`cmake -B build -S . && cmake --build build`）。
- HiSim config 的 `storage_tiers[].name` 必须与 kvcm 返回的 `tier_names` 一致（init 时由 HiSim 覆盖 kvcm config 保证）。
- `batch_exists_per_tier()` 返回的是上次 `batch_exists()` 的缓存结果，不会额外调 kvcm。

---

## 2026-07-18 — 添加 L2→L3 写时间仿真（SimpySSD predict_write_time + 与推理重叠）

**背景**：之前 L2→L3（DRAM→SSD）写的仿真时间为 0——`DISK_WRITE_BANDWIDTH_BYTES` 声明了但从未赋值或使用。L1→L2 backup 有 PCIe 带宽模型，L3→L2 prefetch 有 SimpySSD 读时间模型，唯独 L2→L3 写缺失。需要补上。

**完成的事项**（commit `7623706`，2 文件，+48/-4 行）：

1. **`state.py`** — 新增 L3 写时间累加器：
   - `_hicache_l3_backup_dur: float = 0`
   - `inc_hicache_l3_backup_dur(dur)` / `pop_hicache_l3_backup_dur()` — 与 L2 的 `inc/pop_hicache_l2_backup_dur` 完全对称
   - `reset()` 中加入清零

2. **`sglang_hook.py` `handle_backup_operation`** — 收集写数据量 + 调 SimpySSD：
   - 遍历 `backup_queue`，收集 `len(operation.host_indices)`（token 数）
   - 写数据量 = `total_write_tokens * KV_CACHE_BYTES`（与 prefetch 的 `storage_hit_count * KV_CACHE_BYTES` 完全对称，都是 token 级 × 每 token 字节数）
   - 调 `sim.predict_write_time(total_bytes, KV_CACHE_BYTES, is_sequential=True)`
   - fallback：`DISK_WRITE_BANDWIDTH_BYTES` 带宽模型（`total_bytes / bandwidth`）
   - 结果存入 `StateManager.inc_hicache_l3_backup_dur(write_time)`

3. **`sglang_hook.py` `wrapped_process_batch_result`** — L3 写时间与下一次推理重叠：
   - `l3_write_overflow = max(0, l3_write - current_inference_dur)`
   - 非 overlap：`step_clock(l2_load + inference + l2_backup + l3_write_overflow)`
   - overlap：`response_time += l3_write_overflow`
   - `iteration_stats` 新增 `l3_backup_latency` 字段

**时间模型**：
```
非overlap:  T_N ─→ l2_load ─→ inference ─→ l2_backup ─→ [overflow] ─→ T_{N+1}
                                                         ↑
                                                         max(0, l3_write - inference_dur)
                                                         写≤推理 → 完全隐藏
                                                         写>推理 → 溢出延迟下一轮
```

L3_preload（读）与 inference 重叠（用 `inference_dur` 做预算，不在时钟里），L3_write（写）也与 inference 重叠（只有溢出在时钟里）。两者不在同一时间段——读在推理期间，写在推理后（但溢出部分延迟下一轮）。

**验证结果**：
```
Write 500 tokens (16.4MB): 2113μs, BW=7.8GB/s  vs 规格 8.2GB/s (0.95x) ✅
Write 100 tokens (3.3MB):  733μs  < inference 1000μs → overflow=0（完全隐藏）✅
Write 500 tokens (16.4MB): 2113μs > inference 1000μs → overflow=1113μs ✅
Fallback (no SSD sim):     2048μs (BW=8.0GB/s) ✅
```

SimpySSD 写性能对齐规格书 0.95x，读性能 0.97x（之前验证过）。`write_latency_us=500`（TLC NAND program）保持不变。

**关键路径速查**：
- L3 写时间累加：`src/hisim/simulation/manager/state.py`（`_hicache_l3_backup_dur`）
- L3 写时间计算：`src/hisim/simulation/sglang/sglang_hook.py`（`handle_backup_operation`）
- L3 写溢出到时钟：`src/hisim/simulation/sglang/sglang_hook.py`（`wrapped_process_batch_result`，`l3_write_overflow`）
- SimpySSD 写接口：`3rdparty/simpyssd/simpyssd/ssd_timing_adapter.py`（`predict_write_time`）

**注意事项**：
- L3 写和 L3 读目前不在同一时间段，没有读写争用建模（需后续 SSD 带时序仿真支持）。
- SimpySSD 的 `_predict()` 每次调用 `_reset()` 清空状态，不支持跨调用的读写争用。如需争用建模，需去掉 `_reset()` 并保留 `avail_time`（见 SimpySSD test/timing_overlap_demo.py 的验证）。
- SimpleSSD 原生支持跨调用状态保持（`Engine.simTick` 不重置），天然支持读写争用，后续可替代 SimpySSD 做时序仿真。

---

## 2026-07-23 — SSD 带时序仿真：SimpySSD 支持读写重叠争用 + 提交即返回延迟

**背景**：之前的 SimpySSD `predict_read_time`/`predict_write_time` 每次调用 `_reset()` 清空状态，无法建模跨调用的读写争用。需要"提交即返回延迟"的时序仿真模式——请求带到达时间，提交后返回该请求的延迟，引擎状态跨调用保持，后续请求受前面请求影响。

**完成的事项**（3 个 commit，共 4 文件）：

### commit `35dfb13` — SimpySSD 时序仿真核心 + HiSim 集成

1. **`ssd_estimate.py`** — Chip 改为 ContendedChip（3 行改动）：
   - `read_begin`：`start = max(engine.now, self.avail_time)` → 等 chip 空闲再开始读
   - `write_begin`：同上
   - 两者完成后设 `self.avail_time = start + latency` → 推进 chip 可用时间
   - `avail_time` 跨调用保持（不再 `_reset()` 清零）

2. **`ssd_timing_adapter.py`** — 新增 5 个时序 API：
   - `submit_write(arrival_us, data_bytes, io_block_size, is_sequential)` → `(comp, lat, wait, exec)`
     - 推进所有受影响 chip 的 `avail_time`（写影响保留）
   - `submit_read(arrival_us, data_bytes, io_block_size, is_sequential)` → `(comp, lat, wait, exec)`
     - 也推进 `avail_time`，但调用后可用 `set_all_avail_time()` 恢复（读后重置）
   - `get_all_avail_time()` → 所有 chip 的 `{(ch,chip): time}` 快照
   - `set_all_avail_time(snapshot)` → 恢复所有 chip 的 `avail_time`（读后重置用）
   - `reset_avail_time()` → 全部归零（初始化用）
   - 保留原有 `predict_read_time`/`predict_write_time`（向后兼容）

3. **`state.py`** — 新增跨迭代保存的 L2 时间值：
   - `last_l2_load_dur` / `last_l2_backup_dur` + setter/getter
   - 在 `process_batch_result` 里保存（供下一轮 `handle_backup`/`handle_prefetch` 用）

4. **`sglang_hook.py`** — 三处改动：
   - `handle_backup_operation`：用 `submit_write(write_arrival, ...)` 带到达时间提交
     - `write_arrival = global_clock + last_l2_load + current_inference + last_l2_backup`
   - `_calc_prefetch_pages_ssd`：用 `submit_read(read_arrival, ...)` 带到达时间提交
     - `read_arrival = global_clock + last_l2_load`
     - 分离 `wait`（等写完成的固定开销）和 `exec`（实际读时间）
     - `remaining_budget = max_dur - wait`，用 `exec` 做 ratio 估算完成量
     - 读后 `set_all_avail_time(avail_snapshot)` 恢复（消除读影响，保留写影响）
   - `wrapped_process_batch_result`：保存 `last_l2_load_dur` / `last_l2_backup_dur`

### commit `6cad374` — 修复 `_submit_timed` bug

- **Bug**：TrackingSet 在第一个 I/O 完成时就 break，但后续 I/O 在 drain 循环里继续跑，推进了 `avail_time`。导致 write 报告 `lat=518us`（第一个 I/O），但 `avail_time=51018us`（全部跑完），两者矛盾。
- **修复**：去掉 TrackingSet 和 break，直接跑完所有事件后用 `engine.now`。
- 修复后：`Write 100 tokens: lat=50018us, avail=51018us`，一致。

### commit `76e914c` — 修复 I/O 分配：跨 channel/chip 条带

- **Bug**：`submit_write`/`submit_read` 的 `ch_id`/`chip_id` 默认为 0，导致 100 个 I/O 全部串行到 ch0,chip0，100×500μs=50000μs。真实 SSD 应跨 8ch×24chip=192 并行单元条带分配。
- **修复**：改为 `is_sequential` 参数，内部按 `i % num_channel` 条带分配。新增 `get_all_avail_time`/`set_all_avail_time` 处理所有 chip 的快照/恢复。
- 修复后：`Write 100 tokens: 50018us → 733us`（1 轮并行写 500μs + DMA 233μs）。

**时序模型**：
```
轮 N:
  handle_backup → submit_write(write_arrival=T_N + last_l2_load + inference + last_l2_backup)
    → I/O 跨 192 chip 条带分配
    → chip avail_time 推进到 write_done
  handle_prefetch → submit_read(read_arrival=T_N + last_l2_load)
    → 读看到写的 avail_time → wait = max(0, write_done - read_arrival)
    → exec = 实际 NAND 读 + 传输
    → 读后 set_all_avail_time 恢复（消除读影响）

轮 N+1:
  handle_backup → submit_write(...)
    → 看到 N 的 write_done（写-写争用保留）
  handle_prefetch → submit_read(...)
    → 看到当前轮 write_done（写-读争用保留）
    → 读后恢复
```

**争用矩阵**：
| | 写-写 | 写-读 | 读-读 | 读-写(下轮) |
|--|:---:|:---:|:---:|:---:|
| 当前实现 | ✅ | ✅ | ❌(读后恢复) | ❌(读后恢复) |
| 未来多worker | ✅ | ✅ | ✅(不恢复) | ✅(不恢复) |

**验证结果**：
```
Write 100 tokens: lat=733us (500μs NAND + 233μs DMA), wait=0, exec=733
Read 100 tokens:  lat=811us, wait=733us (等写完成), exec=78us (100×60μs+传输)
→ 写-读争用生效：读延迟增加了 733μs 等待时间
```

**关键路径速查**：
- SimpySSD 时序核心：`3rdparty/simpyssd/simpyssd/ssd_timing_adapter.py`（`submit_write`/`submit_read`/`get_all_avail_time`/`set_all_avail_time`）
- Chip 争用实现：`3rdparty/simpyssd/simpyssd/ssd_estimate.py`（`read_begin`/`write_begin` 用 `avail_time`）
- L3 写时序集成：`sglang_hook.py`（`handle_backup_operation` → `submit_write`）
- L3 读时序集成：`sglang_hook.py`（`_calc_prefetch_pages_ssd` → `submit_read`）
- 跨迭代状态：`state.py`（`last_l2_load_dur`/`last_l2_backup_dur`）

**注意事项**：
- 读后 `set_all_avail_time` 恢复对齐 sglang 的 abort prefetch 行为。未来多 worker 场景去掉恢复即可保留读-读争用。
- `write_arrival` 和 `read_arrival` 用上一轮的 `last_l2_load`/`last_inference`/`last_l2_backup` 近似（方案 B），不是当前轮的精确值。误差可接受（HiSim 本来就是粗粒度仿真）。
- SimpleSSD 原生支持时序仿真（`Engine.simTick` 不重置），后续可替代 SimpySSD 做更精确的时序仿真（含 CPU/FTL/ICL 全栈开销）。

---

## 2026-07-27 — 多层级 L3 存储时序仿真（SimpySSD 独立引擎 + kvcm 每层级写入统计）

**背景**：将单层级 SSD 时序扩展为多层级（local_nvme + remote_nfs），每个层级拥有独立的 SimpySSD Engine 实例，实现真正的多层级读写争用建模。

**完成的事**：

### Part 1: SimpySSD Engine 实例化（消除全局单例）
- `sim.py`：移除全局单例 `engine`，`Engine` 改为普通类，每个实例独立。
- `ssd_estimate.py`：SSD/Channel/Chip 链式持有 `_engine`，`read_begin`/`write_begin` 用 `max(engine.now, avail_time)` 实现争用。
- `ssd_timing_adapter.py`：重写为 `SSDTimingSimulator(cfg_path)`，`__init__` 中 `self._engine = Engine()`，`self._ssd = SSD(StubSystem(), self._engine)`，所有 `engine.*` 替换为 `self._engine.*`。
- 验证：两个实例 sim1/sim2 各自写 100 块，avail_time 独立（互不影响），读看到各自写的争用。

### Part 2: kvcm C++ 每层级写入块计数
- `stats_record.h`：`WriteRecord` 新增 `per_tier_write_blocks`（`vector<size_t>`）和 `tier_names`（`vector<string>`）。
- `optimizer_runner.cc`：`HandleWriteCache` 填充 `record.per_tier_write_blocks = eviction_manager_->GetCurrentInstanceUsagePerTier(instance_id)` 和 `record.tier_names = indexer->GetTierNames()`。
- `insight_simulator_types.h`：`WriteCacheRes` 新增 `per_tier_write_block_num`（`vector<int64_t>`）和 `tier_names`（`vector<string>`）。
- `optimizer_manager.cc`：`WriteCache` 从 `last_write` 填充 `res.per_tier_write_block_num` 和 `res.tier_names`。
- `py_optimizer_binding.cc`：暴露 `per_tier_write_block_num` 和 `tier_names` 为 `def_readonly`。
- 编译验证：`cmake --build build` 通过，`WriteCacheRes` attrs 包含新字段。

### Part 3: MockHiCacheStorage 每层级写入缓存
- `sglang_mock_class.py`：
  - `__init__`：新增 `self._last_per_tier_write = None`。
  - `batch_set`（kvcm 路径）：捕获 `WriteCache` 返回的 `per_tier_write_block_num` + `tier_names`，存入 `_last_per_tier_write`（`dict(zip(...))`）。
  - 新增 `get_last_write_per_tier() -> dict`：返回最后一次 `batch_set` 的每层级写入块数（无额外 kvcm 调用，镜像 `batch_exists_per_tier()` 模式）。

### Part 4: handle_backup_operation 多层级时序写入
- `types.py`：`StorageTierConfig` 新增 `write_bandwidth_gb: Optional[float] = None`。
- `config.py`：解析 `write_bandwidth_gb` 字段。
- `sglang_hook.py`：
  - `_init_ssd_simulator`：新增 `cls._tier_write_bandwidths = {}`，每层级存 `write_bandwidth_gb * 1e9`（fallback 到 `read_bandwidth_gb`）。
  - `handle_backup_operation`：
    - backup 循环中，每次 `_page_backup` 后调用 `self.storage_backend.get_last_write_per_tier()` 累积每层级写入块数。
    - 循环结束后，若 `_tier_sims` 存在且有累积数据：对每层级 `submit_write(write_arrival, write_blocks * KV_CACHE_BYTES, ...)` 到独立 SSD 引擎，`write_time = max(lat)` (parallel) 或 `sum(lat)` (serial)。
    - 无 SSD sim 的层级用 `write_bandwidth` fallback。
    - 单层级路径保持不变（else 分支）。

### Part 5: _calc_prefetch_multi_tier 多层级时序读取
- `sglang_hook.py` `_calc_prefetch_multi_tier`：
  - `read_arrival = global_clock + last_l2_load_dur`（所有层级相同）。
  - 每层级：`submit_read(read_arrival, total_bytes, ...)` → 得到 `(wait, exec_t, lat)`，然后 `set_all_avail_time(snapshot)` 恢复（对齐 sglang abort prefetch）。
  - 无 `submit_read` 的 fallback 到 `predict_read_time` 或 `read_bandwidth`。
  - Parallel 模式：`total_dur = max(lat)`，partial 时每层级独立计算 `remaining = max_dur - wait_i`，`completed_i = hits_i * min(1, remaining / exec_i)`。
  - Serial 模式：`total_dur = sum(lat)`，`ratio = max_dur / total_dur`，`completed = sum(max(1, int(hits_i * ratio)))`。

**验证结果**：
```
Multi-tier read (100+50 blocks): completed=150, dur=293.4us
Multi-tier write (100+50 blocks): local=733.4us, remote=625.7us, parallel=733.4us
Read after write: local wait=933.4us, remote wait=825.7us (write-read contention persists)
→ 多层级独立引擎争用生效
kvcm per-tier write: {'shared': 100} (tier override works with proper config)
```

**关键路径速查**：
- 多层级初始化：`sglang_hook.py` `_init_ssd_simulator`（`_tier_sims` / `_tier_bandwidths` / `_tier_write_bandwidths`）
- 多层级写时序：`sglang_hook.py` `handle_backup_operation`（per-tier `submit_write`）
- 多层级读时序：`sglang_hook.py` `_calc_prefetch_multi_tier`（per-tier `submit_read` + snapshot/restore）
- 每层级写入计数 C++：`optimizer_runner.cc` `HandleWriteCache` → `stats_record.h` `WriteRecord`
- 每层级写入计数 Python：`sglang_mock_class.py` `get_last_write_per_tier()`
- SSD 引擎实例化：`ssd_timing_adapter.py` `SSDTimingSimulator.__init__`（`self._engine = Engine()`）

---

## 2026-07-27 — 拆分读写策略字段 + 顺序模式到达时间链式传播 + 截断逻辑修正

**背景**：`tier_read_mode` 同时控制读写策略不合理。写入应默认并发（数据同时下发各层级），读取应默认顺序（先读快层级再读慢层级）。顺序模式下后一层级的到达时间应叠加前一层级耗时。

**完成的事**：

### 1. 拆分配置字段
- `types.py`：`PlatformConfig` 拆分 `tier_read_mode`（默认 `"serial"`）和 `tier_write_mode`（默认 `"parallel"`）。
- `config.py`：解析两个字段，默认值 read=serial, write=parallel。
- `config_h20_multitier.json`：更新为 `"tier_read_mode": "serial", "tier_write_mode": "parallel"`。
- `sglang_hook.py`：`_init_ssd_simulator` 读取两个字段到 `_tier_read_mode` / `_tier_write_mode`。

### 2. 顺序写入 — 到达时间链式传播
- `handle_backup_operation`：serial 模式下，`arrival_i = write_arrival + sum(lat_0..lat_{i-1})`；parallel 模式下所有层级用同一 `write_arrival`。
- `write_time = sum(lat)` (serial) 或 `max(lat)` (parallel)。

### 3. 顺序读取 — 到达时间链式传播 + 逐层级截断
- `_calc_prefetch_multi_tier` serial 模式：
  - 按 kvcm 优先级顺序遍历层级（dict 迭代顺序 = GetTierNames 优先级顺序）。
  - `arrival_i = read_arrival + accumulated`，其中 `accumulated` 是前序层级 lat 之和。
  - 每层级 submit_read 后恢复 avail_time 快照。
  - **截断逻辑**：`budget_i = max_dur - accumulated`：
    - `budget_i <= wait_i`：该层级无法启动等待 → break，后续层级全 0。
    - `budget_i >= lat_i`（= wait_i + exec_i）：完整完成，`accumulated += lat_i`。
    - 否则：部分完成 `completed_i = hits_i * (budget_i - wait_i) / exec_i`，return `max_dur`。
  - 全部完成：return `total_completed, accumulated`。

### 4. 并发读取 — 同一到达时间 + 独立截断
- parallel 模式：所有层级 `submit_read(read_arrival, ...)` 同一时间点。
- `total_dur = max(lat)`，全部完成时 return `total_completed, total_dur`。
- **截断**：每层级独立计算 `remaining_i = max_dur - wait_i`：
  - `remaining_i <= 0`：0 token。
  - `remaining_i >= exec_i`：完整。
  - 否则：`hits_i * remaining_i / exec_i`。

**验证结果**：
```
Serial read (chained):
  tier1 arrival=0   lat=911.4 (wait=833.4 exec=78.0)
  tier2 arrival=911.4 lat=185.7 (wait=0.0 exec=185.7)  ← 链式传播生效
  serial total = 1097.1us

Serial read (budget=200, wait=833.4): completed=0  ← 无法启动等待
Serial read (budget=lat1+50): completed=113.5     ← tier1全完成 + tier2部分(50/185.7)

Parallel read (budget=850): completed=71.3, dur=850 ← 各层级独立截断
Parallel write: write_time = max(733.4, 625.7) = 733.4us
Serial write:   tier2 arrival = 1250+733.4 = 1983.4, total = 1359.1us
```

**策略总结**：
| | 写入 (默认 parallel) | 读取 (默认 serial) |
|---|---|---|
| **parallel** | 所有层级同一 arrival，`write_time = max(lat)` | 所有层级同一 arrival，每层级独立截断 |
| **serial** | arrival 链式传播，`write_time = sum(lat)` | arrival 链式传播，逐层级顺序截断 |

---

## 2026-07-27 — 数据/时序解耦 + 2 轮滞后时序模型 + submit_write 移到 submit_read 之后

**背景**：发现 `handle_backup_operation` 中 `submit_write` 在 `handle_prefetch_operation` 的 `submit_read` 之前执行，导致读看到**本轮写**的 avail_time（时序倒置）。实际数据流中 L3 写的是 2 轮前的 KV cache（`backup_queue` 有 2 轮滞后），应该建模为"第三轮 L3 读与第一轮 L3 写并发"。

**数据流确认**（2 轮滞后）：
```
check_hicache_events #N:
  handle_backup_operation → batch_set 写入 round N-2 的数据
  handle_prefetch_operation → batch_exists 查询 round N 的请求前缀
  original → writing_check() → backup_queue.put(round N-1 的 demotion)
```

**时序模型**：
- `write_arrival = T_N`（数据 2 轮前的 L2 backup 已完成，直接从轮起始开始）
- `read_arrival = T_{N-1} + l2_load_{N-1} = T_N - inference_{N-1} - l2_backup_{N-1} - l3_overflow_{N-1}`
  （上一轮 L2 load 结束时刻 = 上一轮结束前）
- `write_arrival > read_arrival`：写在读之后到达 ✓
- 读后 snapshot/restore（sglang abort prefetch 行为）✓
- 每轮读看到**上一轮** submit_write 的 avail_time（1 轮争用滞后匹配 2 轮数据滞后）✓

**争用管线**：
| check_hicache_events | submit_read (arrival) | submit_write (arrival) | 读看到的 avail_time |
|---|---|---|---|
| #2 | — | round 0 写, T_2 | — |
| #3 | T_2+l2_load₂ | round 1 写, T_3 | round 0 写的 avail |
| #4 | T_3+l2_load₃ | round 2 写, T_4 | round 1 写的 avail |

### 完成的改动

**1. StateManager (state.py)**
- 新增 `_last_l3_write_overflow` + `set_last_l3_write_overflow()` / `get_last_l3_write_overflow()` + reset

**2. 数据/时序拆分 (sglang_hook.py)**
- `handle_backup_operation`：拆为**数据阶段**——只做 batch_set + 累积 per_tier_write / total_write_tokens，存到 `self._pending_write_data`
- 新增 `submit_l3_write_timing`：**时序阶段**——从 `self._pending_write_data` 取数据，`write_arrival = get_global_clock()`（T_N），含多层级 + 单层级 + bandwidth 三条路径，调用 `inc_hicache_l3_backup_dur`

**3. 调用顺序更新**
- `wrapped_check_hicache_events`：① `handle_backup_operation`(数据) → ② `handle_prefetch_operation`(数据+submit_read) → ③ `submit_l3_write_timing`(submit_write)
- `wrapped_reset`：① `handle_backup_operation`(数据) → ② `submit_l3_write_timing`(submit_write)
- `target.submit_l3_write_timing = submit_l3_write_timing` 新增 target hook

**4. read_arrival 更新（单层级 + 多层级）**
- `_calc_prefetch_pages_ssd`：`read_arrival = T_N - last_inference - last_l2_backup - last_l3_write_overflow`
- `_calc_prefetch_multi_tier`：同上

**5. overflow 计算更新 (wrapped_process_batch_result)**
- 旧：`max(0, l3_backup - inference)`（L3 写只与 inference 重叠）
- 新：`max(0, l3_backup - (l2_load + inference + l2_backup))`（L3 写从 T_N 开始，与整轮重叠）
- 新增 `set_last_l3_write_overflow(overflow)` 供下轮计算 `read_arrival`

**验证结果**：
```
R1: read_arrival=0,    wait=0   (无争用)
    write_arrival=0,   avail=518, overflow=453.4, T2=733.4
R2: read_arrival=50   (= T_1 + l2_load = 0 + 50 ✓)
    wait=683.4 (看到 R1 写的 avail_time ✓)
    write_arrival=733.4 (= T_2 ✓)
R3: read_arrival=783.4 (= T_2 + l2_load = 733.4 + 50 ✓)
    wait=683.4 (看到 R2 写的 avail_time ✓)
→ 每轮读看到上一轮写 ✓，数据/时序解耦 ✓
```

---

## 2026-08-04 — 多 TP worker 单 worker 模拟：L3 数据量用全量 KV_CACHE_BYTES_FULL

**背景**：单 worker 模拟多 TP 行为。TP 组 N 个 worker 共享同一 NVMe 设备，KV cache 按 kv_heads 分片（每 worker 存 1/N）。需在 L3 时序中建模全量数据（所有 worker 的 L3 读写总和）。

**关键发现**：
- `KV_CACHE_BYTES`（per-worker）= `num_kv_heads // tp_size × head_dim × layers × 2 × dtype`（除 tp_size）
- 全量 = `KV_CACHE_BYTES × tp_size` = `num_kv_heads × head_dim × layers × 2 × dtype`（不除 tp_size）
- **MLA 模型**：sglang `backup_skip = True` for rank≠0，仅 rank 0 写 L3，`KV_CACHE_BYTES` 本身就是全量（MLA 不分片）
- **MHA 模型**：所有 rank 都写 L3，每 rank 存 1/N，总和 = 全量
- 两种模型 L3 存的都是"全量 KV cache"，`KV_CACHE_BYTES_FULL` 天然适配两者

**方案 B**：新增 `KV_CACHE_BYTES_FULL`（不除 tp_size），用于 L3 数据量计算；`KV_CACHE_BYTES`（per-worker）继续用于 L2 和 io_block_size。

### 改动清单

| 文件 | 改动 |
|---|---|
| `utils.py` | 新增 `calc_kv_cache_cell_elems_full(model_info, pp_size)` — MLA 不变，MHA 用原始 `num_kv_heads`（不除 tp_size） |
| `config.py` | 新增 `get_kv_cache_bytes_full()` + import |
| `sglang_hook.py` `C_HiCacheController` | 新增 `KV_CACHE_BYTES_FULL` class 变量 + 懒初始化 |
| `sglang_hook.py` `submit_l3_write_timing` | write_bytes 用 `KV_CACHE_BYTES_FULL`（多层级 + 单层级），io_block_size 保持 `KV_CACHE_BYTES` |
| `sglang_hook.py` `_calc_prefetch_pages_ssd` | total_bytes 用 `KV_CACHE_BYTES_FULL`，submit_read 的 io_block_size 保持 `page_size_byte`（= `KV_CACHE_BYTES`） |
| `sglang_hook.py` `_calc_prefetch_multi_tier._read_tier` | total_bytes 用 `KV_CACHE_BYTES_FULL`，io_block_size 保持 `page_size_byte` |
| `sglang_hook.py` `calc_prefetch_pages` 带宽回退 | `_prefetch_dur` 和 `_completed_pages` 用 `KV_CACHE_BYTES_FULL` |
| `sglang_hook.py` `handle_prefetch_operation` | 新增 `KV_CACHE_BYTES_FULL` 懒初始化 |

### 关键约束

- **io_block_size 始终用 `KV_CACHE_BYTES`（per-worker）**：决定 SSD 仿真器的 I/O 数量和条带化。tp=2 时 200 个 73728B I/O vs tp=1 时 100 个 147456B I/O，总数据量相同但 I/O 模式不同。
- **`completed` 返回值公式不变**：`ratio × required_pages` 给出 per-worker 完成页数（因为 ratio 是全量数据的完成比例，per-worker = 全量/tp_size）。
- **L2、kvcm 数据层、StateManager、overflow 公式不变**：L2 是 per-worker 并行无争用；kvcm 共享 L3 hash 去重自动生效。

### 验证结果

```
MHA tp=1: per=73728, full=73728, ratio=1.0  (FULL == per ✓)
MHA tp=2: per=36864, full=73728, ratio=2.0  (FULL constant, per halved ✓)
MHA tp=4: per=18432, full=73728, ratio=4.0  (FULL constant, per quartered ✓)
MLA tp=2: per=34560, full=34560, equal=True  (MLA: per==full, only rank 0 writes ✓)

L3 write data: tp=1 = 100×147456 = 14745600 bytes, tp=2 = 100×147456 = 14745600 bytes (same total ✓)
L3 I/O count: tp=1 = 100 I/Os × 147456B, tp=2 = 200 I/Os × 73728B (more I/Os, smaller each ✓)
```

---

## 2026-08-11 — Trace 重放仿真 + 时间戳单位检测修复

**背景**：使用生产环境 trace（`hisim_qwen3_32b_trace.jsonl`，1740 请求，Qwen3-32B）测试 HiSim trace 重放功能。发现仿真时间异常放大 1000 倍。

### 1. Trace 截断脚本

trace 中 1027 个请求的 `input_length` 超过模型上下文限制（131072 tokens）。编写截断脚本 `/tmp/opencode/truncate_trace.py`：
- 截断策略：保留输入尾部（最近内容），截掉开头（历史/旧内容）
- 输入：1740 请求，最大 input 193625 tokens
- 输出：`hisim_qwen3_32b_trace_truncated.jsonl`，1027 个被截断，最大 131046 tokens
- 截掉 30,540,867 tokens（总量的 13%）

### 2. 时间戳单位检测 Bug 修复（`bench_serving.py`）

**根因**：`timestamp_scale_s=1` 硬编码，但 trace 时间戳是 Unix epoch 毫秒（`1785657570617`）。`sample_hisim_collection_requests` 归一化后（减去最小值），时间戳范围 `0-50024`（50 秒的毫秒值被当作秒），导致 `Benchmark duration = 50025.63s`（放大 1000 倍）。

**修复**：在 `bench_serving.py` 第 2173 行，读取 trace 文件首行原始时间戳，如果 `>1e9`（Unix epoch ms），使用 `timestamp_scale_s=1000`（ms→s 转换）。

```python
# Auto-detect: read first line raw timestamp
_raw_ts = first_line.get("timestamp", 0)
ts_scale = 1000 if _raw_ts > 1e9 else 1
```

### 3. HiSim Trace 仿真机制

HiSim OFFLINE 模式用**虚拟时钟**（`StateManager.global_clock`）推进，不用真实时间：
1. bench_serving 发送所有请求到服务器（附带 `created_time = trace_timestamp / ts_scale`）
2. 服务器 `recv_requests` 将请求按 `created_time` 入 `FUTURE_QUEUE`（最小堆）
3. 每轮调度：弹出 `created_time ≤ global_clock` 的请求，predictor 预测时间，`step_global_clock(推理时长)`
4. 无批次时：`set_global_clock(next_created_time + 1e-6)`（跳到下一请求时间）

### 4. 端到端验证（修复前）

| 指标 | 值 | 说明 |
|---|---|---|
| Successful requests | 8/8 | 全部完成 |
| Benchmark duration | 50025.63s | 放大 1000 倍（bug） |
| Mean TTFT | 83.19ms | 4B 模型短请求 |
| Trace 时间戳重放 | ✓ | 虚拟时钟驱动 |
| L1 前缀缓存命中 | ✓ | `cached-token: 2082` |

---

## 2026-08-14 — L2/L3 命中统计 + ack_write_queue fallback + 根因分析

**背景**：为验证 L2/L3 多层级缓存在 trace 重放下的命中情况，添加统计追踪代码并调试 0 命中问题。

**完成的事项**：

### 1. L2/L3 命中统计追踪（`sglang_hook.py` + `sglang_mock_class.py`）

- `C_SchedulerHook` 新增类级计数器：`_L3_QUERY_COUNT`、`_L3_TOTAL_HITS`、`_L3_HIT_STATS`、`_L3_PREFETCHED_TOKENS`
- `handle_prefetch_operation` 中 `batch_exists_per_tier()` 调用后累加 per-tier 命中数
- `MockHiCacheStorage` 新增 `_DEBUG_BATCH_SET_COUNT`/`_DEBUG_BATCH_EXISTS_COUNT`/`_DEBUG_BATCH_EXISTS_HITS` 计数器
- 仿真结束时输出 `Storage stats: batch_set=N, batch_exists=N, hits=N`
- `wrapped_profile` 中重置所有统计计数器

### 2. ack_write_queue fallback（`sglang_hook.py`）

**根因分析**：在 CPU 仿真模式下，`writing_check()` 检查 `torch.cuda.Event.query()` 判断 GPU→host 拷贝是否完成。在有 GPU 的机器上 CUDA 事件可能正常完成，但在纯 CPU 环境下不会。

**修复**：在 `wrapped_check_hicache_events` 的 `original_check_hicache_events` 之后添加 fallback：遍历 `ack_write_queue` 中未完成的条目，跳过 CUDA 事件检查，直接调用 `write_backup_storage()`。

```python
result = original_check_hicache_events(self, *args, **kwargs)
if hasattr(self, 'cache_controller') and self.enable_storage:
    cc = self.cache_controller
    for ack in cc.ack_write_queue:
        for ack_id in ack.node_ids:
            node = self.ongoing_write_through.pop(ack_id, None)
            if node is not None:
                self.write_backup_storage(node)
    cc.ack_write_queue.clear()
```

注意：不在 fallback 中调用 `handle_backup_operation()`，而是留给下一轮 step 1 处理，确保 `batch_set`（写）在 `batch_exists`（读）之前执行。

### 3. L3 0 命中根因分析

通过详细调试确认数据流正确：
- `batch_set=6316`：L3 写入正常（每批 128 blocks，来自 `handle_backup_operation` → `_page_backup`）
- `batch_exists=97`：L3 查询正常（来自 `handle_prefetch_operation` → `_storage_hit_query`）
- `hits=0`：所有查询返回 0 命中

**根因**：`batch_exists`（step 2）在 `batch_set`（step 1 下一轮）之前执行。同一请求的读写在同一轮内，读先于写。不同请求的 block ID 不匹配。L2 容量充足（`--hicache-size 4`），L3 从未被真正需要。

**结论**：0 命中是当前 workload 下的正确行为。L2 容量足以覆盖所有 backed-up 数据，L3 查询总是针对未被 L1/L2 覆盖的新 token，这些 token 从未被备份到 L3。

### 4. 测试结果（100 请求，Qwen3-32B FP8，L1+L2+L3）

| 指标 | 值 |
|---|---|
| batch_set | 6316 |
| batch_exists | 97 |
| hits | 0 |
| L1 reuse | ~91% |
| Mean TTFT | 85061.35ms |
| Mean TPOT | 27.72ms |

**关键文件**：
- `src/hisim/src/hisim/simulation/sglang/sglang_hook.py` — L3 统计 + ack_write_queue fallback

---

## 2026-08-14 — 修复 L2/L3 缓存零复用：ack_write_queue 缺 dec_lock_ref + 新增 ack_load_queue fallback

**背景**：仿真测试中 L2/L3 缓存始终零复用（`batch_exists` hits=0，无 L2 load_back 触发），即使 100 条请求跑两次也没有改善。根因分析发现两个 bug。

### 根因

sglang 的 `HiRadixCache` 用 CUDA event 跟踪异步 D2H/H2D 拷贝完成：
- `writing_check()`：检查 `ack_write_queue` 中的 CUDA event，完成后调 `dec_lock_ref`（解锁节点）+ `write_backup_storage`（写 L3）
- `loading_check()`：检查 `ack_load_queue` 中的 CUDA event，完成后调 `dec_lock_ref`（解锁 load_back 节点）

在 CPU 仿真模式下 CUDA event 永不完成，HiSim 之前为 `ack_write_queue` 加了 fallback 但**漏掉了 `dec_lock_ref`**，且**完全没有为 `ack_load_queue` 加 fallback**。

### Bug 1：ack_write_queue fallback 缺少 dec_lock_ref（主因）

`sglang_hook.py` 的 `wrapped_check_hicache_events` fallback 只调了 `write_backup_storage(node)` 但没调 `self.dec_lock_ref(node)`。对比 sglang 原版 `writing_check()`（hiradix_cache.py:702-705）：

```python
# sglang 原版：
backuped_node = self.ongoing_write_through.pop(ack_id)
self.dec_lock_ref(backuped_node)           # ← HiSim 漏了这行
if self.enable_storage:
    self.write_backup_storage(backuped_node)
```

**后果链**：
1. `write_backup()` 时 `inc_lock_ref(node)` 锁定节点 → fallback 不解锁 → `lock_ref` 永远 > 0
2. `lock_ref > 0` 的节点不能被 `evict()` 驱逐 → `node.value` 不为 None → `evicted = False`
3. `match_prefix()` 中 `host_hit_length` 永远为 0（因为 `last_node.evicted` 永远 False）
4. L2 load_back 从不触发 → **L2 零复用**
5. L1 miss 的 token 从未进过 L1 → 从未 backup → **不在 L3** → `batch_exists` hits=0 → **L3 零复用**

### Bug 2：ack_load_queue 完全没有 fallback（次因）

`loading_check()` 用 CUDA event 判断 L2→L1 load_back 是否完成。CPU 仿真下 event 永不完成，`ongoing_load_back` 中的节点永远不被清理，`lock_ref` 永不递减。即使 Bug 1 修复后 L2 load_back 能触发，load_back 的节点也会永久锁定，最终 L1 被锁满，后续 load_back 失败。

### 修复

`sglang_hook.py` `wrapped_check_hicache_events` 的 fallback：

1. **ack_write_queue fallback 补 `dec_lock_ref`**：在 `write_backup_storage(node)` 之前加 `self.dec_lock_ref(node)`，与 sglang 原版 `writing_check()` 对齐
2. **新增 ack_load_queue fallback**：镜像 `ack_write_queue` 的处理方式，遍历 `ack_load_queue`，pop `ongoing_load_back` 节点，调 `dec_lock_ref`

### 修复后的预期效果

- backed-up 节点解锁 → 可被 `evict()` 驱逐 → `evicted = True`
- 后续请求匹配到被驱逐的节点 → `host_hit_length > 0` → L2 load_back 触发
- L2 load_back 完成 → `ack_load_queue` fallback 清理 `ongoing_load_back`
- L1 miss 的 token 如果曾被 backup 过 → 在 L3 中能找到 → `batch_exists` hits > 0
- 跑两次同一 trace 时，第二次 L2/L3 应有命中

**关键文件**：
- `src/hisim/src/hisim/simulation/sglang/sglang_hook.py` — `wrapped_check_hicache_events` fallback 修复

---

## 2026-08-15 — L2/L3 缓存复用验证 + HISIM_RESET_HICACHE_STORAGE kvcm 路径修复

**背景**：修复 L2/L3 零复用 bug 后，跑 30 请求 trace 两遍验证 L2/L3 是否有复用。

### 1. HISIM_RESET_HICACHE_STORAGE kvcm 路径修复

`sglang_mock_class.py` 的 `__init__` 中，`HISIM_RESET_HICACHE_STORAGE` 环境变量生效时尝试 `open(self.storage_file_path)` 清空文件，但使用 kvcm 后端时 `self.storage_file_path` 从未设置（只有 `set()` 后端才设置），导致 `AttributeError` 崩溃。

**修复**：检测 `hasattr(self, "storage_manager")`，kvcm 路径调 `ClearAllCaches()`，set 路径才清文件。

### 2. 验证结果（30 请求 trace，Qwen3-32B FP8，L1+L2+L3 全开）

**Run 1（冷启动）**：
| 指标 | 值 |
|---|---|
| batch_set | 2852（L3 写入） |
| batch_exists | 30（L3 查询） |
| L3 hits | 0（冷启动，L3 为空） |
| host_hit_length > 0 | 5 次（1002, 14371, 14371, 28581, 28581 tokens） |
| load_back | 2 次（evicted=True, backuped=True） |
| token usage | 达到 1.00（L1 满，eviction 发生） |

**Run 2（`--flush-cache` 清 L1/L2，保留 L3）**：
| 指标 | 值 |
|---|---|
| batch_set | 5704（+2852，新写入） |
| batch_exists | 288（+258，查询大幅增加） |
| **L3 hits** | **32,984 blocks** ✓ |
| L3 prefetched | 88 tokens ✓ |
| load_back | 4（+2，Run 2 新增 2 次 L2 load_back） |

### 3. 关键发现

1. **L2 load_back 生效**：Run 1 有 5 次 `host_hit_length > 0`，2 次 `load_back` 成功。节点被 `_evict_backuped` 驱逐后，后续请求匹配到 `evicted=True` 的节点，触发 L2→L1 load_back。
2. **L3 复用生效**：Run 2 用 `--flush-cache` 清空 L1/L2 但保留 L3 (kvcm)，L3 命中 32,984 blocks，预取 88 tokens。
3. **`writing_check()` 在 CPU 上正常工作**：PyTorch CUDA event 在 CPU 模式下立即完成（`finish_event.query()` 返回 True），因此 `ack_write_queue` fallback 虽然加了 `dec_lock_ref`，但实际未被触发（fallback log = 0 行）。fallback 作为安全网保留。
4. **跑两次不 flush 的原因**：`wrapped_profile` 只调 `StateManager.reset()`（重置仿真时钟），不清 RadixCache。第二次跑时 L1 仍有第一次的全部数据，L1 命中率接近 100%，无 L1 miss → 无 L2/L3 查询。需要用 `--flush-cache` 清 L1/L2 才能观察 L3 复用。

**关键文件**：
- `src/hisim/src/hisim/simulation/sglang/sglang_mock_class.py` — `HISIM_RESET_HICACHE_STORAGE` kvcm 路径修复
- `src/hisim/src/hisim/simulation/sglang/sglang_mock_class.py` — batch_set/batch_exists 计数器

---

## 2026-08-15 — 修复 SSD 时序 API 单位不一致 + arrival 参数名整改

**背景**：code review 发现 `SSDTimingSimulator` 的 timing-aware API（`submit_read`/`submit_write`）返回**微秒**，但 HiSim 调用方（`sglang_hook.py`）将返回值当作**秒**使用，与同文件的 `predict_read_time`/`predict_write_time`（正确返回秒）不一致，造成 **1,000,000× 误差**。

### Bug：单位不一致（1e6× 误差）

`ssd_timing_adapter.py` 的两套 API 返回不同单位：

| API | 返回单位 | HiSim 期望 | 状态 |
|---|---|---|---|
| `predict_read_time`/`predict_write_time` | 秒（`engine.now * 1e-6`） | 秒 | ✓ 正确 |
| `submit_read`/`submit_write` | 微秒（`comp - arrival`，raw engine us） | 秒 | ✗ **1e6× 偏高** |

**根因**：`_submit_timed` 直接返回 `comp - arrival_us`（engine 内部 us 单位），未像 `_predict` 那样做 `* 1e-6` 秒转换。

**后果链**（HiSim 调用点）：

1. `sglang_hook.py:643` `inc_hicache_l3_backup_dur(write_time)`：write_time 是 us 当秒存 → `l3_backup_dur` 偏高 1e6×
2. `sglang_hook.py:1274` `l3_write_overflow = max(0, l3_backup_us - (秒级 l2_load + inference + l2_backup))` → overflow 恒为天文数字
3. `sglang_hook.py:381` `_calc_prefetch_pages_ssd`：`remaining_budget = max_dur(秒) - wait(us)` → 恒负 → L3 prefetch 永远 0 完成
4. `sglang_hook.py:455` `_calc_prefetch_multi_tier`：`budget = max_dur(秒) - accumulated(us)` → 同上

**为何当前测试未爆**：
- **写路径**：`get_last_write_per_tier()` 在 kvcm 无 `tier_names` 时返回 `{"default": N}`，与 `_tier_sims` 的 `"local_nvme"/"remote_nfs"` 不匹配 → 走带宽 fallback（`write_bytes / bw`，正确秒），`submit_write` 未被实际调用
- **读路径**：冷启动 L3 hits=0 → `submit_read` 从未被调用
- 两个路径都恰好绕过了 timing-aware API，所以 bug 是**潜伏**的，第二次跑同一 trace（L3 有命中）或 kvcm 返回 tier_names 时才会触发

### 整改：arrival_us 参数名误导

`submit_write(arrival_us, ...)` / `submit_read(arrival_us, ...)` 的参数名暗示微秒，但 HiSim 实际传入 `StateManager.get_global_clock()`（秒）。参数名与实际用法不符。

### 修复

`3rdparty/simpyssd/simpyssd/ssd_timing_adapter.py`：

1. **参数重命名**：`submit_write`/`submit_read`/`_submit_timed` 的 `arrival_us` → `arrival_sec`，docstring 标注"in seconds"
2. **单位转换**：`_submit_timed` 内部 `arrival_us = arrival_sec * 1e6`（秒→us 供 SSD sim 内部用），返回值 `comp/total_lat/wait/exec_t` 全部 `* 1e-6`（us→秒）
3. **结果**：`submit_*` 与 `predict_*` 返回值现在一致（秒），`submit_write(0, ...)` == `predict_write_time(...)`

### 验证

```
predict_write=529.2us  submit_write lat=529.2us wait=0.0us exec=529.2us  (ratio=1.00)
predict_read=89.2us   submit_read  lat=89.2us  wait=0.0us exec=89.2us  (ratio=1.00)
budget=0.04s - wait=0.0s = 0.04s  (positive, prefetch proceeds)
cross-round: R1 write@0 → R2 read@0.001s wait=0.000649s (sees contention) ✓
```

HiSim 调用方（`sglang_hook.py`）无需改动——所有调用点已传秒、已按秒使用返回值，修复后自动正确。

**关键文件**：
- `src/hisim/3rdparty/simpyssd/simpyssd/ssd_timing_adapter.py` — `submit_write`/`submit_read`/`_submit_timed` 单位修复 + 参数重命名

---

## 2026-08-15 — 500 条请求仿真 + 三层缓存命中量化 + 运行方法文档

**背景**：修复 L2/L3 缓存复用 bug 后，跑 500 条请求 trace 量化 L1/L2/L3 各层命中情况，并在 `HiSim_Simulation_Test_Report.md` 中补充第 9 章（500 请求结果）和第 10 章（运行方法）。

### 500 条请求仿真结果（L1+L2+L3 全开，冷启动）

| 层级 | 命中情况 | 数据 |
|---|---|---|
| L1 (Radix Cache) | 94.4% 命中 | 46.9M cached / 49.4M total input |
| L2 (Host DRAM) | 69 次 load_back | 175,531 迭代中 69 次有 L2 load（总 1066s） |
| L3 (SSD/kvcm) | 0 命中（冷启动） | batch_set=23046, batch_exists=476, hits=0 |

L3 冷启动验证（30 请求跑两遍，第二遍 `--flush-cache` 清 L1/L2）：
- Run 1: L3 hits=0（冷启动）
- Run 2: **L3 hits=32,984 blocks**，88 tokens prefetched

### 文档更新

- `HiSim_Simulation_Test_Report.md` 第 9 章：500 请求三层命中汇总 + 与 30/100 请求对比 + 冷启动 vs 预热 L3 验证
- `HiSim_Simulation_Test_Report.md` 第 10 章：运行方法（环境准备、配置文件、环境变量、启动服务、运行压测、查看统计、trace 格式、参数说明）

---

## 2026-08-15 — 修复 Mock size_per_token + 容量层次设计 L1<L2<L3

**背景**：500 请求测试中 L3 hits=0（冷启动），根因是 `MockTokenToKVPoolHost.get_size_per_token()` 继承了 mock `head_num=1, head_dim=1`，导致 L2 容量计算放大 1024 倍（15.6M tokens vs 真实 15K tokens）。L2 过大→L2 总能命中→L3 查询范围只剩新 token→L3 必然 miss。

### 修复

1. **`MockTokenToKVPool.__init__`**：保存 `_real_head_num` / `_real_head_dim`（原始构造参数），buffer 仍用 mock 1×1（省内存）
2. **`MockTokenToKVPoolHost.get_size_per_token()`**：用 `_real_head_num` / `_real_head_dim` 算真实 per-token 大小（262,144 bytes），不用 mock 值（256 bytes）
3. **Host 内存检查**：用 mock buffer 实际大小（256 bytes/token），不用虚拟大小（262,144 bytes/token）
4. **assert(size > device_pool.size) 改为 warning**：允许 L2 < L1 的测试配置

### 容量层次设计

| 层级 | 虚拟容量 | Token 数 | 参数 |
|---|---|---|---|
| L1 (HBM) | ~52 GB | 200,000 | `--max-total-tokens 200000` |
| L2 (DRAM) | 80 GB | 305,176 | `--hicache-size 80` |
| L3 (SSD) | 600 GB | ~2,300,000 | kvcm 配置 |

L1=200K > 最大单请求 149K（131K input + 18K output），可放下 1-2 条请求。

### 500 请求仿真结果对比

| 指标 | 修正前 (L1=217K L2=15.6M) | 修正后 (L1=200K L2=305K) |
|---|---|---|
| L1 reuse | 94.4% | 87.3% |
| L3 hits | **0** | **73,730** |
| L2 load_back | 69 | 1 |
| Mean TTFT | 913,879ms | 1,799,839ms |

**关键结果**：L3 命中从 0 提升到 73,730（~1.18M tokens），证明 L2 容量修正后三层层次化缓存正确工作。

**关键文件**：
- `src/hisim/src/hisim/simulation/sglang/sglang_mock_class.py` — `_real_head_num/dim` + `get_size_per_token` + host 内存检查

---

## 2026-08-27 — L3 直通 L1 读路径（绕过 L2 中转，L2/L3 并发拉取取 max）

**背景**：当前 L3 读路径是三级串行（L3→L2 预取 + L2→L1 H2D 加载），L3→L2 SSD 读延迟靠搭推理便车隐藏。新增"L3 直通 L1"模式：L3 数据绕过 L2 直接传输到 L1（模拟 NVLink/CXL 直连），L2-backup 数据仍走 PCIe H2D。两者**并发执行**，总拉取时间 = max(H2D, direct)。

### 配置

`PlatformConfig`（`types.py`）新增：
- `enable_l3_direct_to_l1: bool = False` — 直通模式开关
- `l3_to_l1_bandwidth_gb: Optional[float] = None` — 直通带宽（GB/s），线性模型 `data / bandwidth`
- `l3_to_l1_bandwidth` 属性（`* 1e9` 转 bytes/s）

JSON 配置示例（`config_h20_multitier.json`）：
```json
"enable_l3_direct_to_l1": false,
"l3_to_l1_bandwidth_gb": 64
```

### 互斥语义

- `enable_l3_direct_to_l1=True`：L3→L2 SSD 读时序**不计算**（预取仅数据路由），L2→L1 按 H2D 计算，L3-direct 按直通带宽计算，两者并发取 max
- `enable_l3_direct_to_l1=False`：现有 L3→L2（SSD 读，搭推理便车）+ L2→L1（H2D）两级路径

### 数据流说明

L3 数据**仍预取到 L2**（使 `host_hit_length > 0` 触发 sglang 的 `load_back` → `load_to_device_per_layer`），但 L2 在直通模式下是"零成本路由层"——数据经过 L2 但不产生 SSD 读延迟，时序按 L3→L1 直通计算。不预取到 L2 会导致 L3 命中 token 走 prefill 路径（AIConfigurator 预测的 prefill 时间远高于缓存加载），因此 L2 路由不可省略。

### 实现

#### 1. StateManager（`state.py`）新增状态

- `_l3_direct_load_count: int` — 本轮 L3 命中 token 数，供 `load_to_device_per_layer` 拆分用
  - `inc_l3_direct_load_count(count)` / `get_l3_direct_load_count()` / `pop_l3_direct_load_count()`
- `_hicache_l3_direct_load_dur: float` — 本轮 L3 直通传输延迟累计（per-layer 累加）
  - `inc_hicache_l3_direct_load_dur(dur)` / `pop_hicache_l3_direct_load_dur()`
- `reset()` 中一并清零

#### 2. 预取阶段（`sglang_hook.py:handle_prefetch_operation`）

直通模式开启时走简化路径：
- 仍调 `_storage_hit_query`（查 L3 命中数）+ `batch_exists_per_tier`（L3 命中统计）
- 仍标记 `operation.completed_tokens = storage_hit_count`，`mark_terminate()`（数据路由到 L2，触发 `load_back`）
- **跳过** `calc_prefetch_pages` / `submit_read`（不计算 SSD 读时序，不消耗 `remain_dur`）
- `StateManager.inc_l3_direct_load_count(storage_hit_count)` 累计本轮 L3 命中数
- `while True`（无 `remain_dur` 预算约束，处理完所有 prefetch_queue）

非直通：现有逻辑不变。

#### 3. 同步阶段（`sglang_mock_class.py:load_to_device_per_layer`）

直通模式开启时，按 `l3_direct_load_count` 拆分 `host_indices`：
- `l3_count = min(get_l3_direct_load_count(), total)` — L3 来源 token 数
- `l2_count = total - l3_count` — L2-backup token 数
- **L2-backup**：H2D 非线性模型（单段近似），`h2d_dur = t0 + l2_bytes / eff_bw`（`t0=6.67μs`, `eff_bw=memory_read_bw*0.85`）→ `inc_hicache_l2_load_dur(h2d_dur)`
- **L3-direct**：直通线性模型，`direct_dur = l3_bytes / direct_bw` → `inc_hicache_l3_direct_load_dur(direct_dur)`
- 两者**分别累计**到不同 StateManager 字段

非直通：现有 per-segment H2D 模型（`est_bandwidth_batch` + `np.sum(size / bandwidth)`）不变。

#### 4. 时钟推进（`sglang_hook.py:wrapped_process_batch_result`）

```python
if enable_l3_direct_to_l1:
    l3_direct_dur = pop_hicache_l3_direct_load_dur()
    hicache_l2_load_dur = max(hicache_l2_load_dur, l3_direct_dur)  # 并发取max
pop_l3_direct_load_count()  # 复位
```

后续 `set_last_l2_load_dur`、`l3_write_overflow`、OVERLAP/非OVERLAP 时钟计算全部不变——`hicache_l2_load_dur` 已是 `max(H2D, direct)`，语义不变（"本轮加载延迟"）。

### 验证

- 200 tokens 拆分（120 L3 + 80 L2, 64 layers）：H2D=12.8ms, direct=15.7ms, concurrent max=15.7ms (vs sequential 28.5ms) ✓
- 边界：L3 零命中 → max(h2d, 0) = h2d ✓；全 L3 命中 → max(0, direct) = direct ✓；count > total → cap at total ✓
- 非直通模式回归：`direct_bw=None` → 走现有 per-segment H2D 路径 ✓

### 未来改进

直通带宽当前用**线性模型**（`data / bandwidth`，适合 NVLink/CXL）。后续可接入 **SimpySSD** 的 `submit_read` 时序仿真器，用 NAND 级别的非线性模型（启动开销 + 多通道并行 + 读写争用）替换线性公式，与 L3→L2 路径的 SSD 时序模型统一。具体做法：在 `load_to_device_per_layer` 直通分支中调用 `sim.submit_read(arrival, l3_bytes, io_block_size)` 获取含争用的延迟，替代 `l3_bytes / direct_bw`。

**关键文件**：
- `src/hisim/src/hisim/simulation/types.py` — `PlatformConfig` 加 `enable_l3_direct_to_l1` + `l3_to_l1_bandwidth_gb` + `l3_to_l1_bandwidth` 属性
- `src/hisim/src/hisim/simulation/manager/config.py` — JSON 加载 2 个新字段
- `src/hisim/src/hisim/simulation/manager/state.py` — `_l3_direct_load_count` + `_hicache_l3_direct_load_dur` 及 inc/get/pop/reset
- `src/hisim/src/hisim/simulation/sglang/sglang_hook.py` — `handle_prefetch_operation` 直通分支 + `wrapped_process_batch_result` max 逻辑
- `src/hisim/src/hisim/simulation/sglang/sglang_mock_class.py` — `load_to_device_per_layer` 直通分支（拆分 L2/L3 + 分别累计）

---

## 2026-09-08 — 将 memory_system 中的 HiSim 修改迁移到 hisim 独立仓库

**背景**：memory_system 仓库中 `src/hisim/` 下的所有修改（22 个 commit）是在 hisim 原始代码基础上做的。现在要将这些修改按 commit 迁移回 hisim 独立仓库（`/home/lfz/push/hisim`，develop 分支），使 hisim 仓库成为唯一维护源。

**前提验证**：
- hisim 仓库 develop 分支 HEAD（`96b2716`，4月8日）与 memory_system 的 init commit（`38a1e28`，7月6日）的 base 完全一致（sglang_hook.py、sglang_mock_class.py 等 6 个关键文件 MD5 全部相同）。
- 迁移方式：对 memory_system 的每个 commit 生成 diff patch，去掉 `src/hisim/` 前缀后 `git apply` 到 hisim 仓库。零冲突。

**完成的事项**：

1. **22 个 HiSim 代码修改 commit**（按时间序逐个 apply，零冲突）：
   - `支持kvcm_optimizer L3多层级仿真功能` → `对接SimpySSD` → `修复sglang dev兼容性` → `多tier L3存储仿真` → SSD时序仿真系列（读写重叠/条带分配/multi-tier timing/split modes/decouple layers/Multi-TP） → `Fix trace timestamp` → L2/L3 hit stats + ack fallback 系列 → `Fix mock size_per_token` → `Add L2 host_hit_length tracking` → `Track L2 hit tokens` → `Add per-request L1/L2/L3 hit tracking + cache state logging` → `Add L3 direct-to-L1 read path`

2. **迁移 history.md**（commit `ec671f7`）：从 memory_system 根目录拷到 hisim 仓库根目录。

3. **迁移 HiSim_Simulation_Test_Report.md**（commit `846bd72`）：从 memory_system 根目录拷到 `docs/` 目录。

4. **迁移 cache analyzer v1/v2/v3 到 test/**（commit `6c4d973`）：
   - `test/hisim_cache_analyzer_v1.py` — block-level prefix cache 模型（vllm 风格 LRU），修复了 docstring（原误写"Radix Cache"→改为"block-level prefix cache, vllm-style"）
   - `test/hisim_cache_analyzer_v2.py` — radix tree 模型（匹配 sglang HiRadixCache），原样迁移
   - `test/hisim_cache_analyzer_v3.py` — v2 + HiSim 逻辑，清理了实验性代码后迁移
   - `test/README_hisim_cache_analyzer_v1.md` + `test/README_hisim_cache_analyzer_v2.md`

**V3 清理详情**：
- 删除：`ack_delay_n`/`ack_delay_queues`/`_process_delayed_acks`（ACK 延迟实验，证明无效）
- 删除：`prefetch_to_free`（死代码，声明后从未使用）
- 删除：`prefetch_enabled` flag（调试工具，非 HiSim 逻辑）
- 删除：`evictable_size` 属性 + `_evictable_size()` 方法（未使用）
- 删除：注释掉的"permanent allocation"实验代码
- 保留：`pending_l3` 延迟 L3 hash 写入（匹配 HiSim 的 ACK 时 `write_backup_storage` 语义）
- 简化：ACK 处理改为简单 `_process_acks()`（请求开头直接处理 pending_acks，不走 delay queue）
- 保留的核心 HiSim 逻辑：write_through backup + lock、load_back、contiguous prefix invariant、16384 chunking、Phase 3 统一 backup（chunked=True 跳过）、Phase 3 重物化（churn 循环）、shallow-first walk、output 叶子流、L3 条件写入（backup 成功才写）、L3 仅通过 prefetch 查询、prefetch buffer（按整个 remaining 分配 L2）

**总新增 commit 数**：25（22 代码 + history.md + 测试报告 + cache analyzer）

**注意事项**：
- hisim 仓库 develop 分支现在是唯一维护源，memory_system 中的 `src/hisim/` 不再继续维护
- 未 push 远端，用户自行 push
- hisim 仓库的 `3rdparty/` 目录在 git 中未跟踪（.so 文件等），需要单独安装

---

## 2026-09-01 ~ 09-08 — HiSim 三层缓存命中率分析（memory_system 阶段）

**背景**：在 memory_system 的 `src/hisim/` 上，用 trace 数据对标 HiSim 仿真结果，分析 L1/L2/L3 三层缓存的命中率差异和根因。

**开发的工具**：
- `hisim_cache_analyzer.py`（V1）：block-level 前缀缓存模型，模拟 vllm 风格的 LRU 逐出。用 `OrderedDict` + 累积 hash 做前缀匹配，block_size 可配。简单快速但不建模 radix tree 的节点分裂和级联驱逐。
- `hisim_cache_analyzer_v2.py`（V2）：radix tree 模型，匹配 sglang HiRadixCache 的行为（变长节点、节点分裂、级联驱逐、host leaf 驱逐、L3 hash 查询）。验证了全量数据集 L1=40M 与 L1=400K+L2=1600GB 的 Total 一致性（97.78% = 97.78%）。
- `hisim_cache_analyzer_v3.py`（V3）：V2 + HiSim 特有逻辑。建模 write_through backup+lock、load_back 重物化、连续前缀不变量、16384 chunking、Phase 3 统一 backup、churn 循环、output 叶子流、prefetch buffer、延迟 L3 hash 写入。

**HiSim 实测结果（对标基准）**：

| 配置 | L1 | L2 | L3 | Total |
|------|----|----|----|----|
| L1=200K L2=80GB | 87.34% | 0.09% | 2.39% | 89.82% |
| L1=200K L2=160GB | 87.49% | 1.97% | 0.00% | 89.46% |
| L1=200K L2=400GB | 85.80% | 8.60% | 0.00% | 94.40% |
| L1=200K L2=1600GB | 85.80% | 8.60% | 0.00% | 94.40% |
| L1=160K L2=70GB | 81.25% | 0.03% | 0.15% | 81.43% |
| L1=160K L2=70GB（禁 prefetch） | 79.85% | 12.37% | 0.15% | 92.37% |

**V3 校准结果（无 prefetch，对标 HiSim 无 prefetch）**：

| | L1 | L2 | L3 | Total |
|--|----|----|----|----|
| V3 | 82.70% | 9.63% | 0.00% | 92.33% |
| HiSim | 79.85% | 12.37% | 0.15% | 92.37% |
| 差距 | +2.85% | -2.74% | -0.15% | +0.04% |

**关键发现**：

1. **Total 在无 prefetch 下几乎完美对齐**（0.04% 差距），验证了 V3 的核心缓存逻辑正确性。

2. **L2=0.03%（L1=160K L2=70GB）的根因是 prefetch buffer**：HiSim 的 `prefetch_from_storage` 按整个 remaining input 长度预分配 L2（非树节点，`evict_host` 无法回收），导致 L2 结构性超卖。禁掉 prefetch 后 L2 恢复到 12.37%。这是 SGLang 真实代码的行为，非仿真 artifact。

3. **L1=160K 比 L1=200K 的 L2/L3 更低**是反直觉的非单调现象：更小 L1/L2 → death spiral 更严重（backup 失败 → 连续前缀级联 skip → non-backuped 节点被 `_evict_regular` 删除 → 树缩小 → host leaf 断档 → 更多 backup 失败）。这是螺旋类机制独有的阈值/相变指纹。

4. **HiSim 的 L2/L3 偏低不是 ACK 延迟问题**（L3>0 直接反证 ACK 链路畅通），也不是 CUDA event artifact（CPU 模拟下 mock D2H 是同步瞬时完成）。根因是 SGLang 原生的 write_through 钉死 + prefetch buffer 超卖 + evict_host 只能收树上的 host leaf 三重结构性压力。

5. **V3 与 HiSim 的剩余差距**（有 prefetch 时 11pp）来自 host leaf 粒度差异：V3 的 host leaf ~16K/个（input chunk 叶子），HiSim 的 ~1.6K/个（output 叶子 ~1K + 分裂尾）。粒度差异导致 V3 的 `_evict_l2` 总能找到足够的 host leaf（2-3 个即可），backup 永不失败；HiSim 一次 burst 需要 15+ 个小 host leaf，容易耗尽触发 death spiral。

**关键文件**：
- `test/hisim_cache_analyzer_v1.py` — block-level 模型（vllm 风格）
- `test/hisim_cache_analyzer_v2.py` — radix tree 模型（sglang 风格）
- `test/hisim_cache_analyzer_v3.py` — V2 + HiSim 逻辑（write_through/load_back/prefetch/contiguous prefix）
- `docs/HiSim_Simulation_Test_Report.md` — 完整测试报告（含配置说明、运行方法、结果对比）

---

## 2026-09-09 — 接入 SemiAnalysis cc-traces-weka Agentic 追踪数据集（weka_trace loader）

**背景**：InferenceX/AgentX 的 Agentic 追踪回放数据集（`semianalysisai/cc-traces-weka-062126-256k`，真实 Claude Code 会话，393 traces / 68,266 请求 / 6.89B input tokens / 平均 ISL≈100K）需要接入 hisim。该数据集**无文本内容**，只有 token 计数 + 64-token KV 前缀块哈希序列（`hash_ids`），多轮前缀复用关系隐含在共享 hash 前缀中；subagent 组与主流时间上并发重叠。

**完成的事项**：

### 1. weka_trace.py（`src/hisim/dataset/weka_trace.py`，纯标准库零依赖）

核心规划模块，可被 analyzer 通过 importlib 文件加载直接复用（无需安装 transformers/sglang）：

- **`plan_weka_traces()`**：解析一个或多个 traces.jsonl（`;` 分隔），构建**静态时间轴调度**（由 hisim 虚拟时钟消费）：
  - `num_traces`：总共读几条 trace（-1 全部）
  - `max_concurrent_traces`：并发 trace 槽位；第 j 个初始槽起始时间 = `j * trace_start_interval`
  - `trace_start_interval`（默认 0.5s）：初始 trace 第一轮之间的间隔；trace 结束（最后保留轮的 `t + api_time`）后隔该间隔由下一条 trace 接替槽位（选最早空闲槽）
  - `max_rounds_per_trace`（-1 无限）：截断 trace 到前 N 轮
  - `max_round_tokens`（-1 无限）：轮长 = `in + out`，首个超限轮之前截断 trace，下一条接替
  - trace 内后续轮保持录制相对时间间隔（录制时间已编码"等回复+思考"的依赖）
- **`expand_hash_ids()`**：hash → 合成 token 展开。每 trace 分配全局唯一 block base（`max_hash+1` 递增），块 h 映射到 tokens `[(base+h)*64, +64)`。两请求共享 token 前缀 ⟺ 共享 hash 前缀 ⟺ 真实 KV 前缀复用结构，无损进入 radix tree。
- **`derive_output_tokens()`**：**跨轮 output 推导**（关键正确性设计）——轮 i 的 output tokens 取自轮 i+1 input 超出轮 i input 的部分（对齐到 `out` 长度），使 output 插入树后下一轮前缀匹配不断链；无后继轮/上下文压缩时回退为唯一虚拟 token（`-(global_seq+1)`）。
- **`iter_round_entries()`**：时间排序流式生成器（逐轮惰性展开，内存峰值 = 单请求大小），产出 analyzer 兼容 entry（timestamp/input_ids/output_ids/长度字段）。
- **`add_weka_arguments()` / `weka_plan_from_args()`**：共享 argparse 参数组。

### 2. WekaTraceDataset（`src/hisim/dataset/weka_trace_dataset.py`）

注册名 `weka_trace`（`dataset/__init__.py`），`__getitem__` 惰性展开 token，`custom_params={"created_time", "rid"}` 供 sglang_hook 的 FUTURE_QUEUE 虚拟时钟调度。`DatasetArgs` 新增 5 个 weka 字段。**内存只存 hash_ids（~每块 1 int），不存展开 token**——避免 6.9B token 全量展开的 ~55GB 内存问题。

### 3. analyzer V1/V2/V3 接入（复用同一 loader）

- 三个 analyzer 均加 `--weka-traces` 参数组（`-i` 变为可选，与 `--weka-traces` 互斥）；`_load_weka_module()` 先试常规 import，失败则 importlib 按文件加载 `src/hisim/dataset/weka_trace.py`（绕开 `hisim.dataset.__init__` 的 transformers 依赖）。
- **V2/V3**：`analyze()` 加 `output_ids` 参数——weka 模式插入推导的真实 output（多轮复用不断链），优先于原虚拟 output（V2 `[-(rid+1)]`、V3 `[1]*n`）；hisim_collection 老路径行为不变。
- **V1**：weka 模式强制 `use_output=True`（tokens = input + derived output 一起入树）。
- 逐请求打印加 `tr{trace_idx}r{round_idx}` 标签。

### 4. 测试

- **loader 单元测试**（`test/test_weka_trace_loader.py`，10/10 通过）：单 trace 时间保持、num_traces、并发槽调度（初始错峰 + 最早空闲槽接替）、max_rounds/max_round_tokens 截断（含整 trace 跳过）、hash 展开前缀语义（同 trace 跨轮前缀一致、跨 trace 不相交）、output 推导对齐、多文件输入、流式生成器。
- **真实数据完整测试**（`cc-traces-weka-062126-256k`，4 traces / 并发 2 / 间隔 0.5s / 每 trace 10 轮，40 rounds，3.5M input tokens）：

| Analyzer | L1 | L2 | L3 | New |
|---|---|---|---|---|
| V1（block 级，--block-size 64） | 82.29% (83.51% of input) | 1.84% | 1.84% | — |
| V2（radix） | 84.88% | 0.52% | 1.87% | 12.73% |
| V3（V2+HiSim 逻辑） | 83.96% | 0.52% | 1.87% | 13.64% |

V2/V3 的 L3 完全一致（65,664 tokens = 一整轮逐出后恢复）；L1 82-85% 符合 agentic 多轮前缀复用预期（每轮 New ~13-14% 为新增上下文）。参数组合测试（并发 1 / 间隔 2 / 轮长 100K 截断 → 9 rounds，串行无逐出 L2/L3=0）与 hisim_collection 老 jsonl 路径回归均通过。

**关键设计决策记录**：
- **静态时间轴 vs 动态等待**：轮次间"max(trace 间隔, 上轮完成)"的依赖已编码在录制的 `t` 序列中；trace 间接替用录制 `api_time` 估算结束时间。不做动态完成时间反馈（与"先靠 hisim 虚拟时钟"的决策一致），后续如需动态化需在 sglang_hook 层实现。
- **output_ids 必须真实推导**：若用虚拟 output（如 hisim 仿真 mock 全 1），下一轮 input 中对应上轮 output 的块全部 miss，L1 复用率被系统性大幅低估（前缀链在上轮 input 末尾断掉）。
- **subagent 暂未展开**：当前只取主流请求（`type != "subagent"`）；subagent 并发对缓存竞争的影响留待后续版本。
- **`in` 与 hash 100% 块对齐**（实测 3307 主流请求 0 mismatch），`input_length` 取 `len(hash_ids)*block_size`。

**使用示例**：
```
# hisim 仿真（DatasetArgs: name="weka_trace", filepath="a.jsonl;b.jsonl", ...）
# analyzer V3:
python test/hisim_cache_analyzer_v3.py --weka-traces <traces.jsonl路径> \
  --num-traces 4 --max-concurrent-traces 2 --trace-start-interval 0.5 \
  --max-rounds-per-trace 10 [--max-round-tokens 100000]
```

**关键文件**：
- `src/hisim/dataset/weka_trace.py` — 核心规划模块（零依赖，analyzer 复用入口）
- `src/hisim/dataset/weka_trace_dataset.py` — WekaTraceDataset（仿真侧）
- `src/hisim/dataset/dataset_args.py` / `__init__.py` — 参数扩展与注册
- `test/hisim_cache_analyzer_v{1,2,3}.py` — weka 参数组 + output_ids 支持
- `test/test_weka_trace_loader.py` — loader 单元测试（10 用例）

---

## 2026-09-09 — weka loader 文档化 + subagent 展开 / 动态完成时间反馈特性分析

**文档更新**：
- `README.md` / `README_zh.md`：Step 1 新增 "Agentic 多轮追踪数据集（weka_trace）" 章节（数据集格式、DatasetArgs 用法、analyzer CLI、已知限制：bench_serving HTTP CLI 未接线、subagent 未展开）
- `test/README_hisim_cache_analyzer_v1.md` / `v2.md`：新增 "Weka Agentic Trace 输入" 章节（参数表 + 语义要点 + 实测结果）
- `test/README_hisim_cache_analyzer_v3.md`：新建（v3 相对 v2 的 HiSim 机制、参数表、weka 语义、实测结果）

### 特性一：subagent 请求展开 —— 必要性：高（对容量竞争研究）

**数据实测（cc-traces-weka-062126-256k，393 traces）**：

| 指标 | 数值 | 含义 |
|---|---|---|
| 含 subagent 组的 trace | 175/393（44.5%） | 近半会话有 subagent |
| subagent 请求数 | 39,822 / 68,266（**58.3%**） | 丢掉 subagent 丢掉过半负载 |
| subagent 输入 token | 2.99B / 6.89B（**43.4%**） | 近半 token 流量 |
| 平均 subagent 组 | 23.5 请求，单请求 ISL≈75K | 独立多轮会话 |
| 缓存足迹（unique blocks） | 1.86M → 4.12M（**+121%**） | 容量压力翻倍 |
| 主流/subagent 共享块 | 仅 0.9%（union 口径） | subagent ≈ 独立缓存压力 |

结论：subagent 使 L1/L2/L3 的容量竞争压力翻倍以上，不展开会**系统性低估逐出与 L2/L3 复用率**——正是 reuse rate 研究的核心变量；并发重叠还影响调度与排队指标。

**实现方案（改动集中在 weka_trace.py，~100 行，analyzer 核心零改动）**：
1. `WekaRound` 增加 `stream_id`（主流 = "main"，subagent = agent_id）；`_extract_main_rounds` 改为可选展开（`include_subagents` 参数，默认 False 保持兼容）
2. subagent 内部请求扁平化进同一 trace 的 rounds（`t` 已是 trace 级相对时间，天然与主流重叠）
3. **关键**：output 推导的轮次链接从 `trace_idx` 改为 `(trace_idx, stream_id)` —— 每个 subagent 是独立会话流，轮内 output 对齐本流下一轮 input；主流与各 subagent 各自成链
4. 槽位调度：trace 时长取 max(主流末轮结束, 所有 subagent 内部末请求结束)
5. hash 空间无需改（subagent 与主流共享 trace 本地 hash 空间，block_base 机制已兼容）
6. analyzer 加 `--include-subagents` 开关

**特性二：动态完成时间反馈 —— 必要性：中高（完整仿真必需，analyzer 不需要）**

静态时间轴的失真场景：仿真预测慢于录制 api_time（换硬件/配置探索——正是 hisim 的核心用途）时，下一轮在上一轮完成前到达：
1. **缓存复用失真（最关键）**：上一轮未 finish → output KV 未 insert（`cache_finished_req` 未执行）→ 下一轮 input 中对齐上一轮 output 的 token 段 miss → L1 复用率被系统性低估
2. 排队/TTFT 失真：会话内轮次本应串行（agent 等回复），静态轴产生假并发

analyzer 不受影响（逐请求顺序处理，隐含"等完成"语义）；hisim OFFLINE 仿真受影响。

**实现方案（sglang_hook 层，完成事件驱动的重定键 + 停靠，~200 行 / 5 文件，默认关闭）**：
- 元数据透传：`weka_trace.py` 导出每轮 think 间隔 → `weka_trace_dataset.py` custom_params 增加 session_id/round_idx/think_time → `sglang_bench.py get_request`（复用 queue_start 通道模式）→ `sglang_hook.py` 收集阶段解析登记
- 核心机制（`sglang_hook.py`）：
  - 释放循环（L1257-1267）：堆顶到期但本会话上一轮未完成 → 弹出到 `PARKED[session]` 停靠（防队头阻塞其他会话）
  - 完成挂钩（`wrapped_process_batch_result` L1470-1481）：`req.finished` 时取虚拟完成时刻，把 PARKED 中下一轮以 `max(静态时间, 完成时刻+间隔)` 重新 heappush
  - 两个陷阱已识别：① 不能用拉取式门控（空闲跳跃 L1344-1350 会死锁，停靠条目必须在完成时获得数值化 key）；② 不能复用 L1293-1297 的 queue_start 时钟回拨（debug 专用）
  - 防御：堆空且 PARKED 非空且 running 为空时强制放行
- `types.py` BenchmarkConfig 加 `dynamic_completion_feedback` 开关，默认 False 对现有静态回放零影响
- 无死锁证明：停靠期间上一轮必在 running_batch，循环持续；完成后停靠条目获得数值 key，空闲跳跃可跳达

**优先级建议**：先做 subagent 展开（analyzer 侧立刻可用、直接改善 reuse rate 研究的负载保真度），动态完成时间反馈在跑完整 weka 仿真前实现。

## 2026-09-09 — 实现 subagent 展开 + 动态完成时间反馈

两个特性均已实现并测试通过（默认关闭，对现有行为零影响）。

### 一、subagent 请求展开

按上一节方案落地，`include_subagents` 默认 False：

- `weka_trace.py`：`_extract_streams` 取代 `_extract_main_rounds`，返回 `[(stream_id, reqs)]` 流列表（主流 = "main"，subagent 按 `agent_id` 分组合并排序）；`WekaRound` 增加 `stream_id`；output 推导链按 `(trace_idx, stream_id)` 逐流链接；trace 时长取所有流末轮结束的最大值；`add_weka_arguments` 增加 `--include-subagents`
- `dataset_args.py`：`include_subagents` 字段；`weka_trace_dataset.py` 透传给 planner
- analyzer V1/V2/V3：`--include-subagents` 开关（复用 `add_weka_arguments`），L2/L3 命中输出标签带 `tr{N}.{stream_id}r{M}`

**实测（cc-traces-weka-062126-256k，5 traces / 每流 20 轮上限，V3）**：
- 不含 subagent：100 轮，10.18M input tokens，L1 88.58%
- 含 subagent：327 轮（+227），22.10M input tokens（+117%，与前期"缓存足迹 +121%"的统计一致），L1 88.13%，subagent 流的 L2 命中正常出现在输出中
- V1/V2 同样本验证通过（V2 含 subagent：22.10M tokens，L1 83.10%）

### 二、动态完成时间反馈

按上一节方案落地（完成事件驱动的停靠 + 重定键），`BenchmarkConfig.dynamic_completion_feedback` 默认 False：

- **元数据链**：`iter_round_entries`/`WekaTraceDataset` 输出 `session=f"{trace_idx}:{stream_id}"`、`round_idx`、`recorded_finish`（= timestamp + api_time，计划时间轴上的录制完成时刻）→ `sglang_bench.get_request(dynamic_feedback=True)` 透传到 simulation_params → `sglang_hook` 收集阶段登记
- **核心机制（sglang_hook.py）**：
  - 释放循环（`wrapped_recv_requests`）：堆顶到期但本会话上一轮仍在飞行 → 停靠 `PARKED[session]`（不阻塞其他会话）；上一轮已完成 → 以 `shift = max(0, finish_vtime - recorded_finish)` 延迟释放
  - 完成挂钩（`wrapped_process_batch_result`）：`req.finished` 时记录虚拟完成时刻，释放该会话全部停靠轮（`enqueue_time + shift` 重新入堆）；等价于 max(静态时间, 实际完成 + 录制间隔)
  - 防死锁：堆空 + PARKED 非空 + running/waiting 均空时强制放行最早停靠轮（防御完成事件丢失）
- **session 元数据语义**：`recorded_finish` 直接携带计划轴上的录制完成时刻，下一轮静态时间已含录制 think 间隔，shift 平移即可，无需单独导出 think_time

**验证**：loader 侧 session 元数据由单测覆盖（`test_iter_round_entries_stream`：session 格式、recorded_finish = timestamp + api_time、时间单调性）；hook 层逻辑因本机无 sglang 环境未跑完整仿真，待有环境后用 `BenchmarkConfig(dynamic_completion_feedback=True)` + weka 数据集验证。

### 三、测试与文档

- `test/test_weka_trace_loader.py`：14/14 通过（新增 4 个 subagent 用例 + 会话元数据断言）
- `README.md`/`README_zh.md`：补 `include_subagents`、动态完成时间反馈说明、`--include-subagents` 示例及实测数据


## 2026-09-10 — weka_trace 仿真联调修复（develop_lfz，sglang 0.5.6.post2）

用 cc-traces-weka-062126 真实 trace（/home/lfz/data/cc-traces-weka-062126/traces.jsonl，393 会话）端到端验证 weka_trace 数据集 + SGLangBenchmarkRunner 仿真链路，发现并修复 5 个问题：

1. **WekaTraceDataset 未注册**：`dataset/__init__.py` 只 import 未加进 `dataset_registry`，`get_dataset("weka_trace")` 报 unknown dataset name → 补注册
2. **transformers 5.x 兼容**：`base_dataset.py` 从 `transformers.tokenization_utils` import `PreTrainedTokenizerBase`（5.x 已移除该导出）→ 改从顶层 `transformers` import（4.x/5.x 均稳定）
3. **is_hybrid_swa=False 破坏 0.5.6**：`override_initialize` 里 `is_hybrid_swa=False` 使 `tp_worker.is_hybrid_swa`（`is not None` 语义）恒为 True，调度器读 `sliding_window_size` 直接 AttributeError → 与 develop_lfz_0.5.19 对齐：`is_hybrid_swa=None` + 防御性 `sliding_window_size=None`
4. **flush_cache 死锁**：进程内 Engine 的 `flush_cache()`/`clear_hicache_storage()` 依赖 TokenizerManager.handle_loop 分发应答，而 handle_loop 由第一个 generate_request 惰性创建；首次 benchmark 前调用直接死锁 → `sglang_bench.py` 增加 `_ensure_handle_loop()`（在 engine loop 上先建 handle_loop）；`reset_storage_cache` 改走 `tokenizer_manager.clear_hicache_storage()`（`Engine.clear_hicache_storage` 为新版 API）
5. **非层级缓存崩溃**：`wrapped_get_new_batch_prefill` 的 Cache state 日志直接访问 `tc.evictable_host_leaves`（仅 HIRadixCache 有），关闭 --enable-hierarchical-cache 时 AttributeError（1d9eedb 引入，getattr 防护遗漏）→ 全部字段加默认值与 getattr 防护

**验证**（venv 0.5.6.post2 + mock model，max_position_embeddings 调至 262144 匹配 256k 数据集）：
- loader 单测 14/14；真实 trace 加载正常（4 traces/40 rounds）
- weka 端到端仿真（2 traces×6 rounds，并发 2，round ≤64K tokens，L1=64K L2=2x + L3 file 后端）：12 轮全部完成，L1 复用 79.1%，L2 命中 30.7%（L1 驱逐后回灌），动态完成反馈开/关结果一致（mock 即时计算下预期相同），无死锁无崩溃
- pytest：17 passed / 1 skipped；`test_dataset`（需 HF 网络）、`test_aic_xgb_predictor`（Abort，aiconfigurator 环境问题）为本机网络/环境限制，非代码回归

**已知说明**：bench_serving HTTP CLI 未接 weka 数据集（用 Python API 或 analyzer CLI）；`num_traces` 取文件前 N 个会话。
## 2026-09-09 — HiSim 适配 sglang 0.5.19（分支 develop_lfz_0.5.19）

**背景**：hisim 原只支持 sglang 0.5.6~0.5.9（`COMPATIBLE_VERSIONS`），本机环境是 sglang 0.5.19（editable 安装于 `/sgl-workspace/sglang`），直接启动 server 在 Scheduler hook 安装阶段即失败。目标是让 hisim 在 0.5.19 上跑通，同时不破坏 0.5.6~0.5.9 的兼容（新分支 `develop_lfz_0.5.19`，单 commit 合入全部改动）。

**0.5.19 关键破坏性变化（相对 0.5.6，经逐点比对确认）**：
- `Scheduler.recv_requests` 方法被移除：请求接收移到独立组件 `SchedulerRequestReceiver`（`srt/managers/scheduler_components/request_receiver.py`），事件循环改为 `recv_reqs = self.request_receiver.recv_requests(); self.process_input_requests(recv_reqs)`。
- `get_new_batch_prefill` 返回 `NextBatchPlan(batch_to_run, running_batch)`（msgspec.Struct），不再是 ScheduleBatch。
- `Req.extend_input_len`/`req.is_chunked` 被移除：改为 `req.extend_range.length`（`Range` NamedTuple）与 `req is self.chunked_req` 判断。
- `ReqToTokenPool.alloc/free` 语义变化：req pool slot 从 `req.req_pool_idx` 移到 `req.kv.req_pool_idx`（holds_kv/kv_allocated_len）。
- `Scheduler.profile` 被移除，profile 走 `SchedulerProfilerManager._profile`（新 io_struct `ProfileReq/ProfileReqType`）；`ProfileReqOutput` 变 kw-only msgspec struct（不能位置传参）。
- `ModelRunner.init_torch_distributed` 从 `initialize()` 移到 `__init__`（hook 接管 initialize 前就会执行）；显存池分配从 initialize 内移到 `Scheduler.init_memory_pools → tp_worker.alloc_memory_pool`（会覆盖 hook 的 mock pool）；另新增 `Scheduler.init_all_attention_backends/init_all_cuda_graphs` 两条调用链。
- CPU 启动路径：`bootstrap._init_cpu_threads_env` 在 ARM 上（`_is_cpu_arm64`）无条件调用 `torch.ops.sgl_kernel.init_cpu_threads_env/initialize`，sgl-kernel 无 aarch64 CPU wheel → 必崩（且上游把 `Optional[List[int]]` 传成了 str）。
- `ServerArgs` 解析后只读（setattr 报 "assigned after resolution"）；CPU 下 `mem_fraction_static` 默认 None（0.5.6 同）。
- `Scheduler.__init__` 中 `is_hybrid_swa` 判定是 `tp_worker.is_hybrid_swa = model_runner.is_hybrid_swa is not None`——mock 里设 `False` 会误入 hybrid-SWA 分支（0.5.6 语义：非 SWA 模型该值为 None）。
- allocator/host pool/kv pool 新增一批被无条件调用的接口：`check_decode_capacity`/`evict_to_free_tokens`/`free_segment(s)`/`flush_opportunistic`/`verify_byte_accounting`/`set_disagg_move_gate`/`schedulable_token_capacity`/`alloc_aux_to_lengths`/`reset_aux_cache_allocator`/`post_capture_active`/host pool `destroy` 等。

**修改的文件（6 个，+653/-127）**：

1. `src/hisim/simulation/sglang/version.py`：`COMPATIBLE_VERSIONS` 加 `"0.5.19"`；ModelRunner `forward` dispatcher 注册 0.5.19（沿用 0.5.9 的 `wrapped_forward_v2`，`ModelRunnerOutput` 字段兼容）。

2. `src/hisim/simulation/sglang/sglang_hook.py`（核心）：
   - **C_SchedulerHook 双路径接线**：hook 安装时用 `hasattr(target, "recv_requests")` 判版本。≤0.5.9 走原路径（recv_requests/profile hook，行为不变）；≥0.5.10 改挂 `Scheduler.process_input_requests`：OFFLINE 缓冲逻辑（收满 `total_request` 后按仿真时钟释放）重写为 `_offline_collect_and_release`，BLOCKING/OFFLINE 的 per-request 统计抽成 `record_request_stats`；`get_new_batch_prefill` 适配 `NextBatchPlan`（`getattr(result, "batch_to_run", result)`）；`run_batch` 用 `get_extend_input_len()`（extend_range 优先、extend_input_len 回退）；`process_batch_result` 用 `is_chunked_req()`（is_chunked 属性优先、`req is chunked_req` 回退）。
   - **profile 落盘**：原 `wrapped_profile` 大函数抽成 `C_SchedulerHook.dump_profile_results()` classmethod（metrics.json/iteration.jsonl/request.jsonl + StateManager.reset），legacy 路径仍挂 `Scheduler.profile`，新路径由新增 `C_SchedulerProfilerManagerHook` 挂 `SchedulerProfilerManager._profile`（`ProfileReqOutput` 改 kw 传参）。
   - **C_ModelRunnerHook**：`override_initialize` 补 0.5.10+ 下游必需属性 stub——`lora_manager=None`、`loader=MockWeightLoader`（`preloaded_weights_bytes` 属性被 `effective_max_total_num_tokens` 链路读取）、`canary_manager=None`、`prefill_aware_swa=False`、`attn_backend=MockAttentionBackend`（0.5.19 `ForwardBatch.init_new` 在 CPU 分支无条件调 `get_cpu_graph_seq_len_fill_value()`）、`kv_index_translator=MockKVIndexTranslator`（init_new 无条件调 `rebind_write_loc`）、`ngram_embedding_manager=MockNgramEmbeddingManager`（读 `.enabled` + `prepare_for_forward`）、`prefill/decode_attention_backend_str="torch_native"`、`sampler`（try import create_sampler）、`expert_location_updater`（try import）、`memory_pool_config=MemoryPoolConfig(max_total_num_tokens=...)`（非 None 使 `init_target_memory_pool` 短路，保住 mock pool 不被真实分配覆盖）。另外：对 `target.init_torch_distributed` 包一层——调用期间临时把 `bootstrap._init_cpu_threads_env` 换成 no-op（CPU 无 sgl-kernel 时绕开 torch.ops.sgl_kernel 崩溃）；`init_attention_backends`/`init_cuda_graphs`（0.5.10+ 新方法）no-op。同时保留此前的 0.5.6 修复：`sliding_window_size=None`、`full/swa_max_total_num_tokens`、`is_hybrid_swa=None`（注释说明 None 语义）。
   - **C_HiCacheController**：构造 `HiCacheController` 时用 `inspect.signature` 探测新参数（`enable_storage_metrics`/`host_memory_mode`）按版本传参。
   - hook 注册尾部按 `legacy_request_path` 分叉挂接。

3. `src/hisim/simulation/sglang/sglang_mock_class.py`：
   - 新增 stub 类：`MockAttentionBackend`/`MockWeightLoader`/`MockKVIndexTranslator`/`MockNgramEmbeddingManager`。
   - `MockReqToTokenPool`：注册 `_alloc_v3`/`_free_v3`（0.5.19，走 `req.kv.holds_kv/req_pool_idx`，兼容旧 `req.req_pool_idx`）；补 `reset_aux_cache_allocator`/`schedulable_token_capacity`/`alloc_aux_to_lengths`。
   - `MockBaseTokenToKVPoolAllocator`：补 `free_segment`/`free_segments`/`flush_opportunistic`/`set_disagg_move_gate`/`verify_byte_accounting`/`set_latest_forward_done_event`/`set_inflight_forward`/`evict_to_free_tokens`（复用 sglang `evict_from_tree_cache`）/`check_decode_capacity`（evict 后判 available_size，对齐 stock 语义）。
   - `MockTokenToKVPool`：类属性 `post_capture_active=False`（0.5.19 Scheduler `init_model_worker` 无条件读）。
   - `MockTokenToKVPoolHost`：补 `destroy()`（flush_cache 路径）。
   - `MockHiCacheStorage`：补 `batch_get()`（返回 None = miss，0.5.19 `_generic_page_get` 语义；此前只跟踪 key 无数据，属既有缺口的显式化）。

4. `src/hisim/simulation/sglang/launch_server.py`：`ServerArgs.from_cli_args` 后立即 `disable_overlap_schedule=True`（必须在 resolution 前改，只读后 setattr 会炸；仿真强制非 overlap 调度）；hook 安装列表加 `C_SchedulerProfilerManagerHook`。`wrapped_init` 里的 setattr 改 try/except（0.5.19 只读时忽略，靠 launch 期提前改 + 实例属性兜底：`original_init` 后 `self.enable_overlap=False`，同时驱动 `dispatch_event_loop` 和 `run_batch` 分支选择）。

5. `src/hisim/simulation/sglang/sglang_bench.py`：hook 安装列表加 `C_SchedulerProfilerManagerHook`。

6. `src/hisim/simulation/utils.py`：`estimate_kv_cache_pool_capacity` 对 `mem_fraction_static=None` 兜底 0.9（CPU 默认不填该值，0.5.6/0.5.19 皆然）。

**验证结果（本机 aarch64，sglang 0.5.19，系统环境；mock model = 本地 config+tokenizer + `--load-format dummy`）**：
- server 启动：`health: 200`（"The server is fired up and ready to roll!"），hook 全部安装成功、无 Scheduler hook 失败告警。
- 回放模式（`--dataset-name hisim-collection`）：4/4 请求成功，Mean TTFT 0.44ms / TPOT 456ms（含 trace 时间戳回放的排队）。
- random 模式（10 prompts, 256/256, rate 4）：10/10 成功，Mean TTFT 0.51ms / TPOT 0.45ms / E2E 114ms。
- server 端 profile 落盘正常（`/tmp/hisim/output/{metrics,iteration,request}`，确认 ProfilerManager hook 生效）。
- **0.5.6 回归**（/tmp/opencode/venv 内 sglang 0.5.6.post2）：回放 4/4 成功、TTFT 0.37ms / TPOT 0.31ms，与改动前基线一致，legacy 路径未破坏。

**注意事项 / 后续需关注**：
- ⚠️ L2/L3 分层场景（`--enable-hierarchical-cache`）在 0.5.19 上尚未端到端实测：`HiRadixCache.__init__/check_hicache_events/evictable_host_leaves` 等已确认存在且 hook 兼容，`HiCacheController` 的 `backup_queue` 等队列在 0.5.19 改为 `attach_storage_backend` 懒创建（storage_backend 非空时构造即 attach，理论兼容），但 prefetch/backup 时序管线未跑真实数据验证。
- ⚠️ `mock model`（/tmp/opencode/mock-model）与 0.5.6 venv（/tmp/opencode/venv）都是临时产物，重建环境时需：aiconfigurator 安装必须用 git clone + git-lfs 拉真实数据（PyPI wheel 的 systems/data 是 LFS 指针，性能数据库加载会 KeyError）；sglang 0.5.6 从 GitHub tag 克隆装 venv（`--system-site-packages`）。
- ⚠️ 0.5.19 默认 `hicache_write_policy="write_through_selective"`（0.5.6 是 "write_through"），行为差异未评估。
- ⚠️ 本机为 torch_npu 环境（aarch64），`_is_cpu_arm64=True` 导致 0.5.19 CPU 启动必走 sgl_kernel 路径；x86 非 AMX 机器走 warning 分支，不受此问题影响，但 hook 的 no-op 补丁同样无害。
- 分支 `develop_lfz_0.5.19` 基于 `develop_lfz`，未 push 远端。

**关键文件**：
- `src/hisim/simulation/sglang/sglang_hook.py` — 双路径 Scheduler hook、ModelRunner stub、ProfilerManager hook
- `src/hisim/simulation/sglang/sglang_mock_class.py` — Mock 类 0.5.10+ 接口补齐
- `src/hisim/simulation/sglang/launch_server.py` — overlap 禁用时机（resolution 前）
- 对照参考：sglang 0.5.6 源码 `/tmp/opencode/sglang056`、0.5.19 源码 `/sgl-workspace/sglang`

---

## 2026-09-09（下午） — 0.5.19 L2/L3 分层缓存全场景验证与修复

**背景**：上一条记录的 0.5.19 适配只验证了 L1 基线。本条记录 L2（host 分层缓存）、L3（storage backend）、多 tier 配置、L3 direct-to-L1 四类场景在 0.5.19 上的验证过程、发现的缺口与修复，以及 0.5.6 legacy 路径的回归。

**0.5.19 hierarchical cache 的架构性变化（重要）**：
- tree cache 不再是 `HiRadixCache`，而是 **`UnifiedRadixCache`**（`srt/mem_cache/unified_radix_cache.py`）+ `init_hicache()`（`registry.py: create_tree_cache` 路由）。hisim 的 `C_HiRadixCacheHook`（挂 HiRadixCache 类）在 0.5.19 hierarchical 路径上**不生效**。
- controller 是 **`HybridCacheController`**（`hybrid_cache/hybrid_cache_controller.py`，继承 `HiCacheController`）→ hisim 在 HiCacheController 类上的 hook（mock start_writing/start_loading、handle_backup_operation 等）**通过继承继续生效**；但 HybridCacheController override 了 `backup_thread_func`（hisim 的 no-op 被覆盖，stock 线程真实运行）。
- host pool 不再由 cache 层构造：`kv_cache_builder → hybrid_pool_assembler.build_kv_host_pool` 在 hook 之前构造**真实 host pool**（读 `device_pool.head_dim/v_head_dim` 选类），并包成 `HostPoolGroup`（`pool_host/group.py`，`mem_pool_host` 变为组对象）。
- L2 transfer 从异步线程改为同步 `start_writing/start_loading`（直驱 CUDA kernel + CUDA events）；storage 队列（prefetch/backup）仍由后台线程消费，ack 由 `check_hicache_events → _sync_hicache_ready_counts/writing_check/loading_check`（CPU event 可同步）收割。

**发现并修复的缺口（本轮 5 个）**：
1. `MockTokenToKVPool` 缺 `v_head_dim`（host pool 类选择读取）→ 补 `v_head_dim=1`（与 mock 的 1x1 head 对称，选 MHA host pool）。
2. 真实 host pool 分配走 `ALLOC_MEMORY_FUNCS["cpu"] → alloc_with_host_register → cudaHostRegister`，纯 CPU 必崩 → `C_ModelRunnerHook.hook` 里给 `pool_host.common.ALLOC_MEMORY_FUNCS` 注册 CPU 平铺分配（`torch.empty(pin_memory=False)`）。
3. L2 写/读同步路径调用 CUDA kernel → `C_HiCacheController.hook` 里为 0.5.10+ 替换 `start_writing/start_loading` 为 mock 版：保留 stock 的队列合并/host 索引分配/bookkeeping，ack events 用 `device_module.Event()` 立即 record（writing_check/loading_check 的 `finish_event.synchronize()` 即时通过）。
4. `MockHiCacheStorage` 缺 0.5.10+ 接口：`register_mem_host_pool_v2`（HybridCacheController attach 必调）、`get_stats`、`close`、`batch_get_v1/v2`、`batch_set_v1/v2`、`batch_exists_v2`、`get` → 全部补齐（v2 为空实现，页级 v1 接口为主路径）。
5. `override_generic_page_set`（L3 页写）里 `mem_pool_host.get_data_page` 在 0.5.19 是 HostPoolGroup → 加 `get_pool(None)`/`entries[0].host_pool` 解包，取主 KV host pool。

**行为对齐调整**：
- legacy 的 `prefetch_thread_func/backup_thread_func` no-op 改为**仅 0.5.9- 生效**：0.5.19 下 storage 队列由 stock 线程消费（纯 CPU 操作），否则 prefetch_queue 无人消费（根因：batch_exists 始终为 0）。
- `wrapped_check_hicache_events` 的 ack fallback（跳过 CUDA event 直接收割）限定 legacy：0.5.19 的 mock events 立即完成，stock writing_check/loading_check 正常收割，fallback 反而会重复处理。
- `MockHiCacheStorage` 的 set() fallback 分支补 `_DEBUG_BATCH_SET_COUNT/_DEBUG_BATCH_EXISTS_*/_last_per_tier_hits` 统计（原只挂在 kvcm 分支，导致 Storage stats 恒 0 的误报）。

**验证结果（0.5.19，L1=16 tokens 强制逐出，prefetch_threshold=1）**：
- **L2 场景**（`--enable-hierarchical-cache --hicache-ratio 4`）：第二轮起 `L2 cache stats: hit_requests=4, hit_tokens=14`，逐出→write_through→load_back 链路完整。
- **L3 场景**（`--hicache-storage-backend file`）：RUN1 `batch_set=8`（storage_keys.txt 27 keys 持久化）；**重启 server 后**（L1/host 清空）RUN `batch_exists=4, hits=13`（L3 冷启动查询命中）；下一轮 `L2 hit_requests=4`（L3→host→L2→请求复用，端到端 ✓）。
- **多 tier + direct-to-L1**（storage_tiers[nvme/nfs] + `enable_l3_direct_to_l1: true`）：启动正常、3 轮 bench 全部 4/4、L3 写/查/L2 命中同上。
- **0.5.6 回归**（venv）：同 L3 场景，per-request `L3=1` 请求级命中、`L2 hit 4/4`，legacy 时序管线（handle_backup/prefetch_operation + StateManager）完好。

**注意事项 / 已知差异（0.5.19）**：
- ⚠️ **hisim 时序仿真层在 0.5.19 L2/L3 路径上暂不参与**：`check_hicache_events` 的 wrapped 版挂在 HiRadixCache 上（0.5.19 用 UnifiedRadixCache），因此 handle_backup_operation/handle_prefetch_operation/submit_l3_write_timing（L2/L3 带宽与 SSD 时序、per-tier 统计、L3 direct-to-L1 的时序/统计）不会被驱动；L2/L3 功能正确但时延由 stock 即时完成（预置 mock ack）。若要在 0.5.19 恢复完整时序仿真，需针对 UnifiedRadixCache + HybridCacheController 重做 hook 管线（后续工作）。
- ⚠️ 0.5.19 默认 `hicache_storage_prefetch_threshold=256` tokens，小 prefix 数据集测 L3 需传 `--hicache-storage-backend-extra-config '{"prefetch_threshold": 1}'`。
- ⚠️ dump_profile 的 per-request `L1 = final_reused_tokens - L2` 在"仅 L2 命中"（final_reused=0）时显示负数，属统计口径显示瑕疵，不影响 L2/L3 命中本身。
- L3 prefetch 是异步的：本轮查询命中，数据下一轮才可被同前缀请求复用（0.5.6/0.5.19 一致）。

**关键文件**：
- `src/hisim/simulation/sglang/sglang_hook.py` — CPU alloc 补丁、start_writing/start_loading mock、线程 no-op 版本分叉、HostPoolGroup 解包
- `src/hisim/simulation/sglang/sglang_mock_class.py` — v_head_dim、MockHiCacheStorage v2 接口、debug 计数
- 对照参考：`unified_radix_cache.py`（init_hicache/prefetch_from_storage/check_hicache_events）、`hybrid_cache/hybrid_cache_controller.py`、`pool_host/{group.py,mha.py,common.py}`

---

## 2026-09-09 ~ 09-10 — 0.5.19 时序仿真全面恢复（Step 0-6）

**背景**：上一轮 0.5.19 适配只做到 L2/L3 "功能正确、时序旁路"。本条记录按计划恢复完整时序仿真（时序生产端接回 StateManager/全局时钟），并完成 kvcm/simpyssd 环境与全场景回归。分支 `develop_lfz_0.5.19`，每步独立 commit。

**分步内容与 commit**：

1. **Step 0（fe7d8f6）L3 持久化改选开**：set() 后端默认内存态（跨 bench/重启隔离、可复现），`HISIM_PERSIST_L3_STORAGE=1` 开启温存储模拟，`HISIM_L3_STORAGE_PATH` 可覆盖路径；保留 `HISIM_RESET_HICACHE_STORAGE`。改动 `env.py`（两个类方法）+ `sglang_mock_class.py`（构造/set/clear 三处门控）。单元测试验证默认/开启/重载/清除四态。

2. **Step 1（7d57a04）时序驱动接回**：新增 `C_UnifiedRadixCacheHook`（wrap `check_hicache_events`/`reset`，复刻 0.5.6 四段式驱动：handle_backup_operation → handle_prefetch_operation → submit_l3_write_timing → original；逻辑抽为模块级 `_drive_hicache_timing_phases` 与 HiRadixCache 路径共享）+ `C_HybridCacheControllerHook`（no-op 子类 override 的 `backup_thread_func`，避免与数据相双消费 backup_queue）+ 基类 prefetch/backup 线程 no-op 恢复无条件。保留 `prefetch_io_aux_thread`（host 分配后的 ack/插树载体）。验证：iteration.jsonl 出现非零 l3_backup_latency（~63-token 节点 129-262us，与 8GB/s 写带宽手算吻合）。

3. **Step 2（64245f2）prefetch 读时序门控**：0.5.19 分支重构——查 `batch_exists` 后不再立即放行，按 `calc_prefetch_pages`（带宽/SSD/多 tier）+ 本轮推理预算做 **chunked 跨轮推进**（进度存 chunked stash 的 `progress` 字段；`operation.completed_tokens` 留给 stock io_aux ack 路径——drain 处有 `completed_tokens <= acked` 断言，占用会崩）；全量"读完"才放行 `prefetch_hit_queue`（host 分配/插树交 stock drain）；同时修复 direct 模式在 0.5.19 误入 legacy 分支的 bug（`operation.host_indices` 未分配时解引用）。慢盘（0.001GB/s）+ 持久化 keys 冷启动验证：13 hit tokens 跨多轮推进，TTFT 0.45ms → 321ms 真实体现读时延，下轮 L2 命中 4/14、TTFT 回落 0.45ms。

4. **Step 3（c7dfa62）L2 传输时序注入**：0.5.19 下真实 host pool 顶替了 MockTokenToKVPoolHost（其 load/backup 时序记账失效），改为在 `mock_start_writing`（D2H：t0=4us + bytes/(mem_write_bw×0.85) → inc_l2_backup_dur）和 `mock_start_loading`（H2D：t0=6.67us + bytes/(mem_read_bw×0.85) → inc_l2_load_dur；**direct 拆分**：L3 直读 token 按 l3_to_l1_bandwidth 计 → inc_l3_direct_load_dur，消费端 max() 并发语义已就位）记账，带宽模型照搬 `est_bandwidth_batch`（0.85 有效系数 + 固定延迟），抽为模块级 `_l2_transfer_duration`/`_record_l2_backup_timing`/`_record_l2_load_timing`。单元断言全部通过（D2H/H2D 数学、direct 拆分与 StateManager 记账）。

5. **Step 4（随 Step 1 验证）L3 写时序**：`submit_l3_write_timing` 由驱动点按轮调用，非零时延进 iteration.jsonl ✓。

6. **Step 5 环境补齐 + kvcm GIL 段错误修复**：
   - `pip install -e 3rdparty/simpyssd` ✓。
   - kvcm wheel 构建安装后 **GetCacheLocation 必崩（SIGSEGV）**（空树单 key 最小复现；WriteCache 偶发崩）。gdb 栈显示 `_native.so` 内在无 GIL 状态调 `PyErr_Occurred`——pybind `py::call_guard<py::gil_scoped_release>()` 与 C++ 异常翻译（LogAndThrowReplayError 路径）冲突。**修复**：`py_optimizer_binding.cc` 去掉全部 8 处 call_guard（模拟器单线程串行调用，释放 GIL 无收益），重编 wheel 后 20/20 write+get 命中、空树查询/清除后查询均正常。**binding 文件在 git 跟踪内，本次 commit；重装环境务必用修复后的源码构建**。
   - ⚠️ 注意：venv（0.5.6）`--system-site-packages` 会继承系统的 kvcm/simpyssd——0.5.6 回归现在也默认走 kvcm 后端（本次顺带验证 0.5.6+kvcm 组合正常：batch_set=3/exists=5、L2 hit 4/14、请求级 L1/L2 拆分）。要回退 set() 后端需卸载系统 kvcm。

7. **Step 6 回归矩阵**（全部通过，无异常无段错误）：
   - 0.5.19：L1 基线（4/4，TTFT 0.44ms）｜L2+L3-kvcm（2 轮 4/4，batch_set=8/exists=4，L2 hit 4/14）｜多 tier+SSD sim（2 个 SSDTimingSimulator 启用、kvcm storages 按 HiSim tiers 覆盖，4/4×2）｜direct-to-L1 config（4/4×2，时序拆分单测验证）｜慢盘 chunked（TTFT 321ms 体现）｜持久化开（跨重启 hits=13）/关（默认内存态）。
   - 0.5.6 venv：L2+L3（kvcm 后端，2 轮 4/4、L3 写/查、L2 hit 4/14、请求级 L2 拆分）；无 hook 失败告警。

**遗留与注意事项**：
- ⚠️ **L2 load 时序的端到端 iteration.jsonl 记录**需要 host hit ≥ load_back_threshold（10 tokens，0.5.6/0.5.19 同值）的 load_back——小测试集只产生 1-token 碎片命中；真实大前缀 trace（0.5.6 L2 测试同条件）才能触发。函数挂点与数学已验证（`ready_to_load_host_cache → start_loading` 必经、单元断言通过）。
- ⚠️ iteration.jsonl 的 dump 发生在**下一轮 bench 开始**（start_profile 触发 dump+clear），看第 N 轮数据要跑第 N+1 轮；bench 结束后空闲轮的时序 inc 会被 profile 边界的 StateManager.reset 清掉（小数据集边界效应）。
- ⚠️ kvcm 后端内存态，跨重启 L3 命中需 set() 后端 + 持久化开关。
- 时序管线（含 SSD submit_read/write、多 tier 串并行）在 0.5.19 的驱动结构与 0.5.6 完全一致（同一组 handle_* 函数），差异仅在挂载类（UnifiedRadixCache vs HiRadixCache）与 chunked stash 的 progress 字段。

**关键文件**：
- `src/hisim/simulation/sglang/sglang_hook.py` — 共享驱动、chunked 门控、L2 时序注入、mock start_writing/loading
- `src/hisim/simulation/manager/env.py` + `sglang_mock_class.py` — 持久化选开、kvcm 后端
- `3rdparty/kvcm_l3/.../py_optimizer_binding.cc` — GIL 段错误修复
- 测试配置：`/tmp/opencode/config_slowdisk.json`（慢盘 chunked）、`config_tiers_ssd.json`（多 tier+SSD）、`config_direct.json`（direct-to-L1）、`/tmp/opencode/l2load_dataset.jsonl`（12-token 共享前缀，构造 L2 命中；注意 token id 必须在 mock tokenizer 词表内 ~207，否则 bench 侧截断）

## 2026-09-10 — develop_lfz rebase 到 develop_lfz_0.5.19 + weka_trace 在 0.5.19 联调

将 develop_lfz 的 weka_trace loader/动态反馈/联调修复（d34f6fa..a55f2d1，4 个 commit）rebase 到 develop_lfz_0.5.19（原 7 个 0.5.19 适配 commit 重放其上），冲突解决：

1. **sglang_hook.py `wrapped_profile`**：0.5.19 分支已把 dump 主体抽为 `C_SchedulerHook.dump_profile_results()`（legacy 路径与 SchedulerProfilerManager 共用），采用该重构，并把 weka 的 SESSION_STATE/PARKED/REQ_SESSION 清理补进该方法（原重构基于 weka 改动之前）
2. **Cache state 日志**：采用 develop_lfz 的 getattr 全防护版本（`n_host_leaves` 默认 0）
3. **history.md**：两侧条目均保留

**0.5.19 新发现问题与修复**：`start_profile(profile_prefix="reset")` 在 0.5.19 报 TypeError（>=0.5.10 的 start_profile 收 ProfileReq 对象而非 kwargs）→ `sglang_bench.py` 以 `inspect.signature` 判别，旧版走 kwargs、新版走 `start_profile()`（`profile_prefix` 本就只影响 sglang profiler 命名，hisim hook 不消费，可安全省略）

**验证**（system python + sglang 0.5.19，与 0.5.6 同一 weka 数据集/配置对照）：
- 12/12 轮完成，L1 复用 79.1%（与 0.5.6 完全一致），L2 命中 23.1%（0.5.6 为 30.7%，host_hit_length 统计口径差异），动态反馈开/关一致，全程 0 异常
- 0.5.19 时序管线（storage/L3/L2 stats）在 rebase 后完整保留

## 2026-09-10 — 任务3：hisim 仿真 vs analyzer V1/V2/V3 复用率对比（同 L1/L2/L3 大小）

**口径**：同一 weka 工作负载（/tmp/opencode/weka_small_traces.jsonl 选取的 4 个会话，`num_traces=4 max_concurrent_traces=2 max_rounds_per_trace=6 max_round_tokens=65536`，24 轮 / 808,960 input tokens，两侧用同一 planner、同顺序）；同一容量：L1=65,536 tokens（`max_total_tokens`）、L2=131,072 tokens（`hicache_ratio=2`；analyzer 侧 `--l2-gb 2.147483648 --kv-bytes-per-token 16384`）、L3=无限（file 后端 / `--l3-gb 0`）；kv bytes/token=16384（mock 4 层×8 kv头×128 head_dim×2(KV)×bf16）。

| 层级 | hisim 仿真 (0.5.19) | V1 | V2 | V3 |
|---|---|---|---|---|
| L1 | **44.07%** (356,479) | 36.06% | 43.43% | **44.07%** (356,480) |
| L2 | 30.52% (246,912) | 34.45% | 12.53% | 24.82% |
| L3 | 0.02% (128) | 6.31% | 6.37% | 7.96% |
| New | 25.40% | 23.17% | 37.67% | 23.16% |

**结论**：
1. **L1 完全对齐**：仿真与 V3 的 L1 命中 tokens 仅差 1（356,479 vs 356,480）——V3（HiSim 感知 radix 树模型）就是按 hisim 语义校准的，交叉验证通过；V2 接近（-0.6pp）；V1 偏低 8pp（块粒度 LRU L1，无 radix 精确前缀/重插语义，属 V1 模型固有近似）。
2. **L2 仿真偏高 (+5.7pp vs V3)**：仿真 write_through 把每条 L1 驱逐链都备份到 host 且 mock ACK 即时完成，host 池保留更多；V3 的 pending-ack 锁/连续前缀不变量/host-leaf 驱逐更严格。且 host 命中的 token 不再走 L3。
3. **L3 仿真≈0 vs 理论 6~8%**：主因是 (1) 仿真 L2 保留多——请求在前缀里就命中了 L2，无需查 L3（对比 #24：仿真 l2=32,768/l3=128，V3 L2=16,384/L3=34,624）；(2) sglang `prefetch_from_storage` 只查 host match 之后的"新尾部"，而多轮 agent 会话的新尾部本来就是新内容。仿真 23 次 batch_exists 仅最后 1 次命中 128 tokens，机制正常但触发面窄。**若要仿真体现 L3 复用，需减小 L2（hicache_ratio<2）或跨 pass 持久化（HISIM_PERSIST_L3_STORAGE=1）**。
4. 仿真 L2 统计口径注意：request.jsonl 的 `l2_hit_tokens` 在 chunked prefill 下后 chunk 覆盖前 chunk（总量 246,912），scheduler 日志的全局累加器（307,328）按 chunk 事件累计；对比以 per-request 口径为准（与 V3 L1 对齐佐证其一致性）。

**复现命令**：
- 仿真：`weka_compare_sim.py`（ServerArgs: max_total_tokens=65536, hicache_ratio=2, hicache_storage_backend=file, prefetch_policy=timeout, prefetch_threshold=1）
- V3：`python3 test/hisim_cache_analyzer_v3.py --weka-traces /tmp/opencode/weka_small_traces.jsonl --num-traces 4 --max-concurrent-traces 2 --max-rounds-per-trace 6 --max-round-tokens 65536 --l1-tokens 65536 --l2-gb 2.147483648 --kv-bytes-per-token 16384 --page-size 1`
- V2 同 V3；V1 用 `--block-size 1`（其余同）

## 2026-09-10 — 分析：hisim 仿真 vs V2 复用率差异根因 + V3 修正验证

**问题**：同为"sglang KV 复用"模型，同负载同容量（L1=65536/L2=131072，24 轮 808,960 tokens）下 hisim 仿真（0519）L1/L2/L3/New = 44.07/30.52/0.02/25.40%，V2 = 43.43/12.53/6.37/37.67%——L2 差 18pp、New 差 12pp。逐请求 diff + 代码对照 + 仿真内窥（临时日志）定位根因。

### 一、V2 偏离真实 sglang 的实现缺陷（主因，工具测得）

1. **split 锁泄漏**：`_split` 把 `lock_ref` 复制到两个节点，pending-ACK/匹配路径锁被滞留 → `_evict_l1` 收集到 0 个可驱逐叶子（实测 evictable_leaves=0）→ **L1 静默溢出**（最高 126435/65536）→ 破坏"非驱逐节点构成根前缀"不变量：出现 [被驱逐祖先][L1 叶子] 状态（tr1 的 chunk1-3 被驱逐、带泄漏锁的 out_r2 叶子留在 L1）。
2. **插入不再物化**：sglang 0.5.6 `_insert_helper` 对匹配到的已驱逐节点恢复 device 值（churn cycle）；V2 只按键穿越 → 新节点挂在已驱逐父下，加剧 1。
3. **host 命中只算匹配路径"连续被驱逐后缀"**（此逻辑与 sglang 一致：`while last_node.evicted: host_hit_length += len(host_value)`），但 1+2 的不变量破坏使大段 host 命中落入 New（#11 丢失 46144）→ V2 New=37.67%。
4. **备份时机**：sglang chunked 插入（`cache_unfinished_req(chunked=True)`）跳过 `_inc_hit_count`，备份只在 `cache_finished_req`（write_through 阈值 1）；V2 在插入期就备份 → L2 提前填满，驱逐拿了链头节点（tr1.r0 于 #12 被逐）→ 后续轮总匹配崩溃（#11: 391 vs 仿真 46144；#14: 4864 vs 50880）。
5. **output 处理**：V2 插 weka 真实 output_ids（能多匹配每轮十几~几百 token 的 output 段）；hisim 仿真的 mock 采样输出全 1 → 下一轮在 output 段断链。这是 V2 唯一高于仿真的点。

### 二、hisim 仿真侧机制（决定 L2/L3 形态，实测确认）

- **host 驱逐仅由 prefetch buffer 分配失败触发**（`write_backup` 从未失败：buffer 在请求调度时已释放，先于自身备份）；驱逐从叶子端剥离（全程仅 5 次失败事件、~89K tokens），链主体存活 → L2 命中高。
- **L3 命中在 weka 负载下结构性趋零**：prefetch 目标 = `input[L1命中+host命中:]` = 上轮 output+工具结果——从未以匹配哈希链写入存储（存储里是全 1 output 链，第一个 token 即分叉）；只有 host 驱逐造成的树断点才可能命中。
- **0.5.19 时序门控再压一层**（Step 2 chunked prefetch gate）：mock 计算瞬时 → 每轮推理预算内 SSD 读进度≈0 → 存储命中也读不回（#24 仅 128 tokens）。
- 版本差异：sim056（HIRadixCache）L2=36.18%/L3=1.61%（0.5.6 预分配 buffer + #24 有 L2/L3 重叠计数伪影 new=-7104）；sim0519（UnifiedRadixCache，惰性分配）30.52%/0.02%。

### 三、V3 修正与验证（本次改动，V2 按约定不动）

V3 原把 prefetch buffer 记为跨请求占用的 L2（下一请求才释放），使其与本请求 Phase 3 备份叠加、过度驱逐 host（#11 错把 tr1 chunk2/3 计为 L3）。修正为与仿真一致：**接收时按需驱逐分配、L3 查询后立即释放（先于备份）**：

| 层级 | V3(修正后) | sim0519 | 逐请求 L1/L2 |
|---|---|---|---|
| L1 | 356,480 (44.07%) | 356,479 (44.07%) | **全部 24 请求一致** |
| L2 | 246,912 (30.52%) | 246,912 (30.52%) | **全部 24 请求完全相等** |
| L3 | 18,240 (2.25%) | 128 (0.02%) | 差值=时序门控压制的结构潜力 |
| New | 187,328 (23.16%) | 205,441 (25.40%) | 差值=L3 差 |

结论：V3（HiSim 感知模型）修正后与 hisim 仿真的缓存结构行为**逐 token 对齐**；剩余 L3 差异是仿真的时序效应（瞬时计算下 SSD 读不完），属结构模型不覆盖的时间维度，非缓存语义差异。V2 保持原样作为"理论 sglang"参考，其与真实 sglang 的偏差（锁泄漏/无再物化/插入期备份）已记录在案。

**调试方法备注**：venv sglang hiradix_cache.py 加 HISIM_DEBUG_HOST_EVICT 临时日志（已还原）；V2 侧用 /tmp 副本插桩（仓库 V2 未动）。
