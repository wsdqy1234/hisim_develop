cmake_minimum_required(VERSION 3.18)
project(kvcm_py_optimizer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# pybind11 is provided by the environment (pip install pybind11).
# Override the cmake module path with the installed pybind11's cmake dir.
find_program(PYBIND11_PYTHON_EXECUTABLE python3)
execute_process(
    COMMAND ${PYBIND11_PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)
if(PYBIND11_RESULT EQUAL 0 AND PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()
find_package(pybind11 CONFIG REQUIRED)

set(KVCM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/kvcm)
set(KVCM_INCLUDE_DIRS
    ${KVCM_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/rapidjson/include
    ${CMAKE_CURRENT_SOURCE_DIR}/alog_stub
    ${CMAKE_CURRENT_SOURCE_DIR}/autil_stub
)

set(KVCM_SOURCES
    ${KVCM_ROOT}/kv_cache_manager/optimizer/manager/optimizer_manager.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/manager/optimizer_runner.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/manager/eviction_manager.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/manager/optimizer_loader.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/manager/indexer_manager.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/config/types.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/config/optimizer_config.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/config/tier_config.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/config/eviction_config.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/config/instance_config.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/config/instance_group_config.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/config/optimizer_config_loader.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/index/radix_tree_index.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/eviction_policy/lru.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/eviction_policy/random_lru.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/eviction_policy/leaf_aware_lru.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/eviction_policy/ttl.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/eviction_policy/policy_factory.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/trace_loader/standard_trace_loader.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/trace_loader/trace_util.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/analysis/tracker/hit_rate_tracker.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/analysis/tracker/template_prefix_tracker.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/analysis/tracker/block_lifecycle_tracker.cc
    ${KVCM_ROOT}/kv_cache_manager/optimizer/pybind/py_optimizer_binding.cc
    ${KVCM_ROOT}/kv_cache_manager/meta/cache_location.cc
    ${KVCM_ROOT}/kv_cache_manager/data_storage/storage_config.cc
    ${KVCM_ROOT}/kv_cache_manager/common/logger.cc
    ${KVCM_ROOT}/kv_cache_manager/common/jsonizable.cc
    ${KVCM_ROOT}/kv_cache_manager/common/env_util.cc
    ${KVCM_ROOT}/kv_cache_manager/common/standard_uri.cc
)

pybind11_add_module(_native ${KVCM_SOURCES})
target_include_directories(_native PRIVATE ${KVCM_INCLUDE_DIRS})
target_compile_definitions(_native PRIVATE
    RAPIDJSON_HAS_STDSTRING=1
    RAPIDJSON_WRITE_DEFAULT_FLAGS=2
)

# Emit the compiled extension directly into the python package dir so that
# `pip wheel .` / `python setup.py bdist_wheel` packages it together with the
# bundled config json and __init__.py.
set(KVCM_PKG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kvcm_py_optimizer)
set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${KVCM_PKG_DIR}
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(_native PRIVATE stdc++fs)
endif()
