# kvcm_py_optimizer (hisim L3 多层级仿真后端)

本目录把 tair-kvcache 的 `kv_cache_manager/optimizer` 模块（C++ + pybind11）以
**脱离 Bazel 的独立 CMake 构建** 重新打包，作为 `kvcm_py_optimizer` wheel 独立安装。
hisim 在运行时 `import kvcm_py_optimizer` 即可启用 **L3 多层级缓存仿真**；未安装时
hisim 自动退化为基于 `set()` 的简易后端（不报错，仅 L3 前缀匹配功能不可用）。

## 功能等价性

L3 仿真调用链（`OptimizerManager.Init/WriteCache/GetCacheLocation/ClearAllCaches`）为
**纯 CPU/内存计算**（RadixTree 前缀匹配 + 驱逐策略 + Stats），不依赖任何外部网络/磁盘/redis。
配置中的 `pace`/`hf3fs` 仅作为 tier 标签解析为 enum，不实例化 storage backend。因此本目录
剔除了 redis/hiredis/boost/3fs/mooncake/protobuf/grpc 等重依赖，仅保留 rapidjson（解析配置）
与 pybind11，行为与原 Bazel 构建完全一致。

## 目录结构

```
3rdparty/
├── CMakeLists.txt              # 独立 CMake 构建（pybind11_add_module(_native ...)）
├── setup.py                    # 把 .so + config 打成 wheel
├── README.md                   # 本文件
├── kvcm/                       # vendored C++ 源码（73 个 .cc/.h，来自 tair-kvcache/kv_cache_manager）
│   └── kv_cache_manager/{optimizer,meta,data_storage,common}/...
├── rapidjson/include/rapidjson/  # vendored rapidjson 头（header-only，已打 0001-document_h.patch）
├── alog_stub/alog/             # 最小 alog 桩头（hisim 不调 InitLogger，运行时不执行）
├── autil_stub/autil/           # 最小 autil 桩头（StringUtil::fromString）
└── kvcm_py_optimizer/          # wheel 包目录
    ├── __init__.py             # from ._native import * ; get_config_path()
    ├── optimizer_startup_config_load.json   # 默认 optimizer 启动配置
    └── _native.<EXT_SUFFIX>.so  # ← cmake 产物（*.so 已 gitignore，不入库）
```

## 构建依赖

- CMake >= 3.18
- C++17 编译器（g++ >= 9 / clang >= 10）
- Python 3.9+，且已 `pip install pybind11`（>= 2.12，需含 `native_enum.h`）
- setuptools + wheel（用于打包）

rapidjson 与 alog/autil 桩均已 vendored，**无需联网**。

## 编译与安装

```bash
cd 3rdparty

# 1. 构建 C++ 扩展（.so 直接落到 kvcm_py_optimizer/ 下）
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 2. 打成 wheel
python setup.py bdist_wheel --dist-dir dist
#   或： pip wheel . --no-build-isolation -w dist

# 3. 安装（独立于 hisim）
pip install dist/kvcm_py_optimizer-*.whl

# 4. 验证
python -c "import kvcm_py_optimizer as k; \
l=k.OptimizerConfigLoader(); print(l.load(k.get_config_path())); \
m=k.OptimizerManager(l.config()); m.Init(); print('L3 OK')"
```

随后按 hisim 根目录的 README 安装 hisim 即可（`pip install .`）。hisim 的
`sglang_mock_class.py` 会在 `MockHiCacheStorage` 中 `import kvcm_py_optimizer`，
导入成功则走 kvcm 后端，失败则自动退化。

## 自定义配置

默认读取本包自带的 `optimizer_startup_config_load.json`。如需替换，设置环境变量：

```bash
export HISIM_KVCM_CONFIG_PATH=/path/to/your/config.json
```

## 维护说明

- vendored 源码与上游 tair-kvcache 字节一致，**仅** `optimizer/pybind/py_optimizer_binding.cc`
  中的 `PYBIND11_MODULE(kvcm_py_optimizer, ...)` 改名为 `PYBIND11_MODULE(_native, ...)`，
  以便作为 `kvcm_py_optimizer._native` 子模块被包 `__init__.py` 重新导出。
- rapidjson 的 `include/rapidjson/document.h` 应用了 `0001-document_h.patch`
  （移除 `GenericStringRef::operator=`，与上游 Bazel 构建一致）。
- alog_stub / autil_stub 为最小桩；hisim 不调用 `LoggerBroker::InitLogger`，故 `alog` 路径
  运行时不执行，桩仅满足编译链接。

### hisim ↔ kvcm pybind 调用适配

上游 tair-kvcache 当前源树中，`hisim/src/hisim/simulation/sglang/sglang_mock_class.py`
的 `MockHiCacheStorage.batch_set/batch_exists` 与本目录 vendored 的 kvcm pybind 签名存在
**预先存在的版本错位**（hisim 侧 #53/#61 与 optimizer 侧 #168 演进不同步）：

| 调用 | kvcm pybind 签名 | hisim 原始传参 | 问题 |
|---|---|---|---|
| `WriteCache` | `(instance_id, trace_id, timestamp, block_ids, ttl_seconds=0)` | 多传 `write_token_ids=[1]` | 第 5 参被当 `ttl_seconds`，TypeError |
| `GetCacheLocation` | `(..., block_ids, block_mask, input_len)` | 传 `read_token_ids=[1]` 当 `block_mask`、`mask_offset=0` 当 `input_len` | 类型不符 + `input_len<=0` 抛异常 |

为使 L3 端到端可用，`sglang_mock_class.py` 做了如下兼容适配：

- `WriteCache`：去掉 `write_token_ids`（kvcm 本就不接受该参，丢失信息为零）。
- `GetCacheLocation`：传 `block_mask=0`（`BlockMaskOffset=0` 表示无本地块——hisim 调用前已
  过滤 L1/L2，故 0 个本地块语义正确，全部 key 均查 L3）；`input_len = len(keys) * block_size`
  （`block_size=16` 取自 bundled 配置；`input_len` 仅参与 stats 统计，`kvcm_hit_length` 由
  `PrefixQuery` 决定，与 `input_len` 无关，故取值只需通过校验）。

该适配是 best-effort 桥接，未来与 upstream kvcm API 对齐时需重新审视。
