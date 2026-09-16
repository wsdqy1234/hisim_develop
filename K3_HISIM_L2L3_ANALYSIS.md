# HiSim + Kimi-K3 适配架构与 L1/L2/L3 分层缓存分析

## 1. 适配架构框图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        bench_serving (HTTP 客户端)                        │
│                   发送 /generate 请求 + /start_profile                   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────────────┐
│                    sglang HTTP Server (主进程)                           │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ TokenizerManager │  │  FastAPI     │  │  TokenizerWorker          │  │
│  │ (tokenize 请求)  │  │  (HTTP 路由)  │  │  (分词子进程)              │  │
│  └────────┬────────┘  └──────────────┘  └───────────────────────────┘  │
│           │ ZMQ PUSH (IPC socket)                                        │
└───────────┼─────────────────────────────────────────────────────────────┘
            │ fork + ZMQ
┌───────────▼─────────────────────────────────────────────────────────────┐
│                  Scheduler 子进程 (sglang 0.5.18)                      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  event_loop_normal (每轮循环)                                    │    │
│  │                                                                 │    │
│  │  1. recv_requests [HiSim hook: FUTURE_QUEUE 收集/释放]           │    │
│  │  2. process_input_requests [HiSim hook: 初始化 REQUEST_STATS]    │    │
│  │  3. get_next_batch_to_run                                       │    │
│  │     → get_new_batch_prefill [HiSim hook: Mamba属性 + 触发器]     │    │
│  │       → _get_new_batch_prefill_raw                              │    │
│  │         → check_hicache_events ← L2/L3 驱动入口 (每轮调用)       │    │
│  │         → check_prefetch_progress [patch: 恒True]               │    │
│  │         → add_one_req → can_run_list                            │    │
│  │         → ScheduleBatch.init_new + prepare_for_extend           │    │
│  │       → ngram.prepare_for_forward [mock: 原样返回batch]         │    │
│  │       → return NextBatchPlan(batch_to_run)                      │    │
│  │  4. if batch:                                                   │    │
│  │     → run_batch [HiSim hook: predictor 计算延迟 + 推进虚拟时钟] │    │
│  │       → ModelRunner.forward [mock: 空logits]                   │    │
│  │       → sample [mock: 全1 token IDs]                           │    │
│  │     → process_batch_result [HiSim hook: 统计L2/L3延迟 +         │    │
│  │                              gen_token_latencies + 完成触发器]   │    │
│  │  5. else: on_idle() (无batch时空转)                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  K3ProfilePredictor (延迟预测, 不做真实计算)                     │    │
│  │  Prefill: 8点查表插值 (1576→0.439s, 4096→0.725s, ...)        │    │
│  │  Decode: DECODE_BASE=0.0805s × batch_ratio × seq_factor        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  HiRadixCache (真实 sglang 代码, 非 mock)                       │    │
│  │  ┌───────────────────────────────────────────────────────────┐  │    │
│  │  │  L1: HBM KV Cache Pool (radix tree 共享前缀)                │  │    │
│  │  │  MockTokenToKVPoolAllocator (alloc/free 页面管理)          │  │    │
│  │  │  容量 = max_total_tokens (默认 200000)                     │  │    │
│  │  │  ↓ 池满时 evict()                                          │  │    │
│  │  └───────────────────────────────────────────────────────────┘  │    │
│  │  ┌───────────────────────────────────────────────────────────┐  │    │
│  │  │  L2: DRAM Host Memory Pool (MockTokenToKVPoolHost)         │  │    │
│  │  │  backup_from_device_all_layer → inc_hicache_l2_backup_dur  │  │    │
│  │  │  load_to_device_per_layer → inc_hicache_l2_load_dur        │  │    │
│  │  │  ↓ 池满时 evict_host() → L3                                │  │    │
│  │  └───────────────────────────────────────────────────────────┘  │    │
│  │  ┌───────────────────────────────────────────────────────────┐  │    │
│  │  │  L3: SSD Storage (MockHiCacheStorage + SimpySSD)           │  │    │
│  │  │  batch_set → kvcm WriteCache → SSD 写延迟模拟             │  │    │
│  │  │  batch_exists → L3 命中查询 → SSD 读延迟模拟              │  │    │
│  │  └───────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  HiCacheController (8阶段流水线, 每轮 check_hicache_events 执行) │    │
│  │  Phase 1:  MLA handle_backup_operation    (排空 backup_queue)   │    │
│  │  Phase 1b: KDA handle_kda_backup_operation (KDA backup)        │    │
│  │  Phase 2:  MLA handle_prefetch_operation  (排空 prefetch_queue)  │    │
│  │  Phase 2b: KDA handle_kda_prefetch_operation (KDA prefetch)     │    │
│  │  Phase 3:  MLA submit_l3_write_timing     (L3 写延迟)           │    │
│  │  Phase 3b: KDA submit_kda_l3_write_timing (KDA L3 写延迟)       │    │
│  │  Phase 4:  ack-queue fallback (dec_lock_ref, 释放锁定节点)      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  KDA/Mamba State Pool (hybrid_ssm=False → 跳过)                │    │
│  │  MockMambaStatePoolHost (L2 KDA state DRAM pool)               │    │
│  │  MAMBA_POOL_SIZE=165, MAMBA_SLOT_BYTES=15.15MB               │    │
│  │  ← hybrid_ssm=False 导致 _kda_pool=False, 所有 KDA 方法 early-return │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. L1/L2/L3 分层缓存的完整触发链路

```
                    ┌──────────────────────────────────┐
                    │   新请求到达, add_one_req()       │
                    │   检查: total_tokens < pool容量?  │
                    └──────────┬───────────┬───────────┘
                          够   │           │ 不够
                    ┌──────────▼┐    ┌─────▼──────────────┐
                    │ 正常调度   │    │ NO_TOKEN, 请求等待  │
                    │ alloc()    │    │ batch_is_full=True │
                    │ 分配 KV 页  │    │ 不触发 eviction!    │
                    └───────────┘    └────────────────────┘
                                          │
                          ← 调度器选择等待, 不 evict ┘

    ┌─────────────────────────────────────────────────────────────┐
    │  只有当 running_batch 正在 decode + check_decode_mem()     │
    │  检测到 available_size < num_tokens 时, 才触发 eviction:   │
    └──────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  evict_from_tree_cache(tree_cache, num_tokens)              │
    │  → HiRadixCache.evict(EvictParams(num_tokens))             │
    │    遍历 evictable_leaves (lock_ref==0 的叶子节点)            │
    └──────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────────┐ ┌────────────┐ ┌──────────────────┐
    │ write_back   │ │ _evict_     │ │ _evict_backuped  │
    │ 策略:        │ │ regular()   │ │ (已 backup 的)   │
    │ write_backup │ │ 直接删除    │ │ 释放 HBM KV     │
    │ (D2H→L2)    │ │ 不进 L2    │ │ 保留 host_value  │
    └──────┬───────┘ └────────────┘ │ evicted=True    │
           │                          └──────────────────┘
           ▼
    ┌──────────────────────────────────────────┐
    │  L2 BACKUP (HBM → DRAM)                  │
    │  cache_controller.write()                │
    │  → MockTokenToKVPoolHost                 │
    │    .backup_from_device_all_layer()       │
    │  → inc_hicache_l2_backup_dur  ✓          │
    │                                          │
    │  如果 L2 DRAM 池满:                       │
    │  → evict_host() → L3 写入                 │
    │  → write_backup_storage()                │
    │  → MockHiCacheStorage.batch_set()        │
    │  → submit_l3_write_timing()              │
    │  → inc_hicache_l3_backup_dur  ✓         │
    └──────────────────────────────────────────┘

    ┌──────────────────────────────────────────┐
    │  L2 LOAD_BACK (DRAM → HBM)               │
    │                                          │
    │  后续请求 match_prefix() 匹配到          │
    │  evicted 节点 → host_hit_length > 0      │
    │                                          │
    │  → init_load_back() → load_back()       │
    │  → cache_controller.load()               │
    │  → MockTokenToKVPoolHost                 │
    │    .load_to_device_per_layer()           │
    │  → inc_hicache_l2_load_dur  ✓           │
    │                                          │
    │  L3 PREFETCH (SSD → DRAM)               │
    │  → handle_prefetch_operation()           │
    │  → MockHiCacheStorage.batch_exists()    │
    │  → if hit: SSD→DRAM load → L3 命中 ✓    │
    │  → prefetch_complete_tokens > 0         │
    │  → disk_prefetch_ratio > 0  ✓           │
    └──────────────────────────────────────────┘
```

## 3. 为什么 L2/L3 在 e2e 仿真中未触发

### 根本原因：L1 池从不溢出 → eviction 从不触发

```
    L1 池容量 = max_total_tokens = 200,000 tokens (200K)

    trace_basic: 10 请求 × 1412 tokens, 共享 1610-token 前缀
    → unique KV 占用 ≈ 1412 + 9×1 = 1,421 tokens
    → 1,421 << 200,000 → L1 远远没满 ❌

    trace_pressure: 40 请求 × 4097 tokens, 共享 4096-token 前缀
    → unique KV 占用 ≈ 4097 + 39×1 + 400(output) = 4,536 tokens
    → 4,536 << 200,000 → L1 远远没满 ❌

    trace_mixed: 20 请求, 3 组 (803/3919/16416 tokens)
    → unique KV ≈ 803 + 3919 + 16416 + 17×1 + 620(output) ≈ 21,975 tokens
    → 21,975 << 200,000 → L1 远远没满 ❌

    即使缩小到 max_total_tokens=1500:
    → 调度器自动限制 batch 大小 (每批 2 请求而非 5 请求)
    → add_one_req 返回 NO_TOKEN, 请求等待
    → 但不触发 eviction! 调度器选择"等待"而非"驱逐"
    → available_size 始终 >= num_tokens → evict() 从不调用 ❌
```

### 调度器的防溢出机制（正常行为，非 bug）

```
    sglang 调度器有两层保护机制防止 L1 溢出:

    第一层: add_one_req (prefill 准入)
    ┌─────────────────────────────────────────────────┐
    │  if total_tokens >= rem_total_tokens:            │
    │      return NO_TOKEN   ← 请求留在 waiting_queue  │
    │                                                    │
    │  → 不调用 evict(), 不触发 L2 backup               │
    │  → 调度器只是"不调度新请求", 等运行中的请求完成   │
    │  → batch_is_full=True, 但没有 eviction            │
    └─────────────────────────────────────────────────┘

    第二层: check_decode_mem (decode 内存检查)
    ┌─────────────────────────────────────────────────┐
    │  evict_from_tree_cache(tree_cache, num_tokens)   │
    │  if allocator.available_size() < num_tokens:     │
    │      tree_cache.evict(...)  ← 这里才真正 evict!  │
    │                                                    │
    │  → 但 num_tokens (decode 需要) 极小 (几页)        │
    │  → available_size 始终 >> num_tokens             │
    │  → evict() 从不被调用                            │
    └─────────────────────────────────────────────────┘
```

### 两个 K3 特有的额外阻断

```
    阻断 1: check_prefetch_progress = 恒True (patch)
    ┌─────────────────────────────────────────────────┐
    │  原本: check_prefetch_progress 检查 L2/L3 预取   │
    │        是否完成 (CUDA event 机制)               │
    │        CPU 下 CUDA event 永不完成 → 恒False     │
    │        → 请求被 continue 跳过 → can_run_list=0  │
    │                                                    │
    │  Patch: 强制返回 True, 让请求能被调度             │
    │  副作用: 跳过了 terminate_prefetch 真实逻辑      │
    │        → prefetch_loaded_tokens 永远为 0          │
    │        → req.storage_hit_length = 0              │
    │        → disk_prefetch_ratio = 0 (L3 结果丢失)    │
    │                                                    │
    │  注意: 不影响 L2/L3 延迟累加器 (那些在 mock      │
    │        transfer 函数里, 独立于此 patch)            │
    └─────────────────────────────────────────────────┘

    阻断 2: hybrid_ssm = False (K3 未被识别为 Mamba 模型)
    ┌─────────────────────────────────────────────────┐
    │  sglang 判断: model_runner 没有 mamba2_config     │
    │  → is_hybrid_ssm = False                         │
    │  → 使用 HiRadixCache (非 HiMambaRadixCache)      │
    │  → _init_kda_pools() 失败 → _kda_pool = False   │
    │  → handle_kda_backup_operation() early-return   │
    │  → handle_kda_prefetch_operation() early-return │
    │                                                    │
    │  影响: KDA/Mamba 状态的 L2/L3 分层完全跳过        │
    │  trace_pressure 设计的 840 slots > 165 pool 溢出  │
    │  针对的是 KDA pool, 但 KDA 路径被跳过了           │
    └─────────────────────────────────────────────────┘
```

## 4. 对比：Qwen3 仿真中 L2/L3 确实触发了

```
    Qwen3 报告: 500 请求, 平均输入 ~98K tokens, L1=200K
    → L1 被填满 (token usage 达到 1.00)
    → evict() 被调用 → write_backup() → L2 backup 触发 ✓
    → 69 次 L2 load_back (host_hit_length > 0) ✓
    → 23,046 次 L3 write ✓

    K3 仿真: 10-40 请求, 平均输入 ~1.4K-16K tokens, L1=200K
    → L1 远远没满 (usage < 1%)
    → evict() 从不调用 → 全链路不触发 ❌

    结论: 不是适配代码的 bug, 是工作量不够大 / L1 容量太大
```

## 5. 如何让 L2/L3 在 e2e 中触发

```
    方案 A: 增大工作量到填满 L1
    ┌─────────────────────────────────────────────────┐
    │  需要 unique KV 总量 > max_total_tokens          │
    │  = 需要 ~500+ 请求, 每个有大量不共享的 token     │
    │  或使用长序列 (128K+ tokens) 让单请求占满 L1     │
    │                                                    │
    │  优点: 最真实的验证                                │
    │  缺点: 仿真时间长 (虚拟时钟也要走很久)            │
    └─────────────────────────────────────────────────┘

    方案 B: 修复 check_prefetch_progress patch
    ┌─────────────────────────────────────────────────┐
    │  不再强制 True, 而是让 ack-queue fallback 正常    │
    │  完成 CUDA event 模拟 → terminate_prefetch 能跑  │
    │  → storage_hit_length > 0 → disk_prefetch_ratio  │
    │                                                    │
    │  优点: 修复 L3 结果传播, 不需要大工作量           │
    │  缺点: 只修复 disk_prefetch_ratio, 不触发         │
    │        L2/L3 延迟 (那些仍需要 eviction)           │
    └─────────────────────────────────────────────────┘

    方案 C: 启用 KDA/Mamba 路径
    ┌─────────────────────────────────────────────────┐
    │  让 supports_mamba() 返回 True                   │
    │  → 使用 HiMambaRadixCache                        │
    │  → _init_kda_pools() 成功                        │
    │  → handle_kda_backup/prefetch 不再 early-return  │
    │  → trace_pressure 的 840 slots > 165 能触发 KDA  │
    │    L2 backup                                     │
    │                                                    │
    │  优点: 能验证 KDA 分层 (trace_pressure 专为它设计) │
    │  缺点: 改动较大, 可能引入新问题                   │
    └─────────────────────────────────────────────────┘

    方案 D: 修改 mock 让 alloc() 在特定场景强制返回 None
    ┌─────────────────────────────────────────────────┐
    │  在 MockTokenToKVPoolAllocator.alloc 里,         │
    │  当 free_pages 使用超过 X% 时返回 None           │
    │  → 强制 evict() 被调用                           │
    │                                                    │
    │  优点: 最小改动, 可控                             │
    │  缺点: 不真实, 可能影响其他逻辑                   │
    └─────────────────────────────────────────────────┘
```

## 6. 总结

```
    L1 (HBM KV Cache)  ─── radix tree 共享前缀, 200K slots ─── ✅ 正常工作 (87-90% prefix reuse)
         │
         │ evict() ← 只在 L1 池满时触发 ← 永远没满 ❌
         │
    L2 (DRAM Host)     ─── backup_from_device / load_to_device ── 未触发 (无 eviction → 无 backup)
         │
         │ evict_host() ← 只在 L2 池满时触发 ← L2 从没被写入 ❌
         │
    L3 (SSD Storage)   ─── batch_set / batch_exists ──────────── 未触发 (无 L2 backup → 无 L3 写)
```

**核心结论**: L2/L3 分层缓存的代码链路完全正确（单元测试 41/41 PASS，Qwen3 e2e 曾产出 69 次 load_back），但在 K3 e2e 仿真中，**工作量远小于 L1 容量（~2K-22K vs 200K），L1 从不溢出，eviction 从不触发，整个 L2/L3 链路是"死代码"**。这不是适配 bug，而是工作量/容量配置问题。要触发 L2/L3，需要用足够大的工作量填满 L1，或启用 KDA 路径配合 trace_pressure。
