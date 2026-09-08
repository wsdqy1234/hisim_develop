#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace kv_cache_manager {

struct GetCacheLocationRes {
    std::string trace_id;
    int64_t kvcm_hit_length;
    std::vector<int64_t> per_tier_hit_block_num;
    std::vector<std::string> tier_names;
};

struct WriteCacheRes {
    std::string trace_id;
    int64_t kvcm_write_length;
    int64_t kvcm_write_hit_length;
};

} // namespace kv_cache_manager