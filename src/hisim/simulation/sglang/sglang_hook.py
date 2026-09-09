import torch
import time
from dataclasses import asdict
from collections import defaultdict
import json
import os
import heapq
import importlib
import threading
from queue import Empty

from hisim.utils import get_logger

from hisim.hook import BaseHook
from hisim.simulation.types import (
    MockSimulationMode,
    RequestStats,
)
from hisim.hook.utils import get_obj_from_args
from hisim.utils.json import CustomJsonEncoder
from hisim.simulation.manager import StateManager, ConfigManager, Envs
from hisim.time_predictor import (
    InferTimePredictor,
    FakeRequest,
    ScheduleBatch as HisimScheduleBatch,
)
from hisim.simulation.sglang.sglang_mock_class import (
    MockAttentionBackend,
    MockHiCacheStorage,
    MockKVIndexTranslator,
    MockNgramEmbeddingManager,
    MockPagedTokenToKVPoolAllocator,
    MockReqToTokenPool,
    MockTokenToKVPool,
    MockTokenToKVPoolAllocator,
    MockTokenToKVPoolHost,
    MockWeightLoader,
)
from hisim.simulation.utils import (
    calc_metrics,
    estimate_kv_cache_pool_capacity,
)
from hisim.simulation.sglang.version import VersionDispatcher


logger = get_logger("hisim")


def _is_legacy_sglang() -> bool:
    """True for sglang <= 0.5.9 (async hicache transfer threads + CUDA events
    that never complete on CPU); False for >= 0.5.10 (synchronous transfers,
    event-driven acks)."""
    try:
        from sglang.version import __version__ as _v
    except Exception:
        return True
    try:
        from packaging.version import InvalidVersion, Version

        return Version(_v) < Version("0.5.10")
    except InvalidVersion:
        return True


def _drive_hicache_timing_phases(cache_obj) -> None:
    """Run the hisim storage data/timing phases on a tree cache:

    1. Data phase: batch_set writes the previous round's KV to the storage.
    2. Data + read timing: batch_exists queries the storage (+ SSD read
       timing on simulators).
    3. Write timing: submit_write advances the SSD simulation state.

    Called from the hooked check_hicache_events before the stock event
    processing. Shared by the HiRadixCache (<= 0.5.9) and UnifiedRadixCache
    (>= 0.5.10) hooks."""
    cc = getattr(cache_obj, "cache_controller", None)
    if cc is None or not getattr(cc, "enable_storage", False):
        return
    cc.handle_backup_operation()
    cc.handle_prefetch_operation()
    cc.submit_l3_write_timing()


class C_EngineHook(BaseHook):
    HOOK_CLASS_NAME = "Engine"
    HOOK_MODULE_NAME = "sglang.srt.entrypoints.engine"

    @classmethod
    def hook(cls, target):
        def hook_clear_hicache_storage(self):
            return self.loop.run_until_complete(
                self.tokenizer_manager.clear_hicache_storage()
            )

        target.clear_hicache_storage = hook_clear_hicache_storage


class C_TokenizerManagerHook(BaseHook):
    HOOK_CLASS_NAME = "TokenizerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.tokenizer_manager"

    @classmethod
    def hook(cls, target):
        original_send_one_request = target._send_one_request

        # When running with blocking mode, send the created time to schedule.
        def wrapped_send_one_request(self, *args):
            import time as _time

            if len(args) >= 1 and args[0].__class__.__name__ == "GenerateReqInput":
                obj = args[0]
                tokenized_obj = args[1] if len(args) >= 2 else None
                if tokenized_obj is not None:
                    custom_params = getattr(
                        tokenized_obj.sampling_params, "custom_params", None
                    )
                    if custom_params and "simulation" in custom_params:
                        custom_params["simulation"]["server_created_time"] = (
                            _time.time()
                        )
            return original_send_one_request(self, *args)

        target._send_one_request = wrapped_send_one_request


class C_ModelRunnerHook(BaseHook):
    HOOK_CLASS_NAME = "ModelRunner"
    HOOK_MODULE_NAME = "sglang.srt.model_executor.model_runner"

    @classmethod
    def hook(cls, target):
        _version_dispatcher = VersionDispatcher()

        def override_initialize(self, *args, **kwargs):
            class MockModel:
                def forward(self):
                    pass

            self.model = MockModel()

            # Attributes that may not be set by the hook but are expected by sglang
            self.use_ngram_embedding = False
            # None matches the non-SWA semantics of sglang >= 0.5.6, where
            # `is_hybrid_swa is not None` is used as the hybrid-SWA check.
            self.is_hybrid_swa = None
            self.is_draft_worker = False
            self.sliding_window_size = None

            self.dtype = self.model_config.dtype
            self.kv_cache_dtype = (
                self.dtype
            )  # FIXME: get kv cache dtype from server args

            model = ConfigManager.get_model_info(self.model_config.hf_config.__dict__)
            hw = ConfigManager.get_accelerator_info()
            config = ConfigManager.get_scheduler_config(
                self.server_args.__dict__,
                "sglang",
                self.model_config.hf_config.__dict__,
            )

            assert model is not None and hw is not None and config is not None

            if self.server_args.max_total_tokens is not None:
                self.max_total_num_tokens = self.server_args.max_total_tokens
            else:
                self.max_total_num_tokens = estimate_kv_cache_pool_capacity(
                    model, hw, config
                )

            if hasattr(self, "page_size") and self.page_size > 1:
                self.max_total_num_tokens = (
                    self.max_total_num_tokens // self.page_size * self.page_size
                )
            self.full_max_total_num_tokens = self.max_total_num_tokens
            self.swa_max_total_num_tokens = self.max_total_num_tokens

            max_num_reqs = min(
                max(
                    int(self.max_total_num_tokens / model.max_seq_len * 512),
                    2048,
                ),
                4096,
            )
            logger.info(
                f"Model runner initialized with {self.max_total_num_tokens} tokens. Maximum number of requests: {max_num_reqs}"
            )

            model_has_mtp_layers = (
                self.model_config.num_nextn_predict_layers is not None
            )
            model_num_layers = (
                self.model_config.num_nextn_predict_layers
                if self.is_draft_worker and model_has_mtp_layers
                else max(
                    self.model_config.num_hidden_layers,
                    self.model_config.num_attention_layers,
                )
            )
            self.start_layer = getattr(self.model, "start_layer", 0)
            self.end_layer = getattr(self.model, "end_layer", model_num_layers)
            self.num_effective_layers = self.end_layer - self.start_layer

            self.req_to_token_pool = MockReqToTokenPool(
                size=max_num_reqs,
                max_context_len=model.max_seq_len,
                device=self.device,
                enable_memory_saver=False,
            )

            self.token_to_kv_pool = MockTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(
                    1  # get_attention_tp_size()
                ),
                head_dim=self.model_config.head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )

            if self.page_size == 1:
                self.token_to_kv_pool_allocator = MockTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=1,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )
            else:
                self.token_to_kv_pool_allocator = MockPagedTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )

            # self.init_memory_pool(50)
            # sglang >= 0.5.10 dereferences attn_backend and several other
            # attributes in ForwardBatch.init_new / downstream code paths even
            # though the forward pass itself is intercepted, so provide stubs.
            self.attn_backend = MockAttentionBackend()
            self.graph_mem_usage = 0
            self.weight_load_mem_usage = 10
            self.lora_manager = None
            self.loader = MockWeightLoader()
            self.canary_manager = None
            self.prefill_aware_swa = False
            self.ngram_embedding_manager = MockNgramEmbeddingManager()
            self.prefill_attention_backend_str = "torch_native"
            self.decode_attention_backend_str = "torch_native"
            self.kv_index_translator = MockKVIndexTranslator()
            try:
                from sglang.srt.layers.sampler import create_sampler

                self.sampler = create_sampler()
            except Exception:
                self.sampler = None
            try:
                from sglang.srt.eplb.expert_location_updater import (
                    ExpertLocationUpdater,
                )

                self.expert_location_updater = ExpertLocationUpdater()
            except Exception:
                self.expert_location_updater = None
            # sglang >= 0.5.10 allocates the memory pools from
            # Scheduler.init_memory_pools via ModelRunner.alloc_memory_pool,
            # which would overwrite the mock pools. A non-None
            # memory_pool_config makes init_target_memory_pool() skip it.
            try:
                from sglang.srt.model_executor.pool_configurator import (
                    MemoryPoolConfig,
                )

                self.memory_pool_config = MemoryPoolConfig(
                    max_total_num_tokens=self.max_total_num_tokens
                )
            except Exception:
                import types as _types

                self.memory_pool_config = _types.SimpleNamespace(
                    max_total_num_tokens=self.max_total_num_tokens
                )

            self.max_running_requests = min(
                (
                    self.max_total_num_tokens // 2
                    if self.server_args.max_running_requests is None
                    else self.server_args.max_running_requests
                    // (
                        self.server_args.dp_size
                        if self.server_args.enable_dp_attention
                        else 1
                    )
                ),
                self.req_to_token_pool.size,
            )

            return

        def wrapped_forward_v1(self, *args, **kwargs):
            batch = args[0]
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            output = LogitsProcessorOutput(
                next_token_logits=torch.empty(
                    size=(batch.batch_size, self.model_config.vocab_size),
                    device=self.device,
                )
            )

            return output, False

        _version_dispatcher.register_method(
            "forward", ["0.5.6", "0.5.6.post1", "0.5.6.post2"], wrapped_forward_v1
        )

        def wrapped_forward_v2(self, *args, **kwargs):
            from sglang.srt.model_executor.model_runner import ModelRunnerOutput

            output, _ = wrapped_forward_v1(self, *args, **kwargs)
            return ModelRunnerOutput(
                logits_output=output,
                can_run_graph=False,
                expert_distribution_metrics=None,
            )

        _version_dispatcher.register_method(
            "forward",
            ["0.5.7", "0.5.8", "0.5.8.post1", "0.5.9", "0.5.19"],
            wrapped_forward_v2,
        )

        def wrapped_sample(self, *args, **kwargs):
            logits = args[0]
            ids = torch.ones(
                size=(logits.next_token_logits.shape[0],),
                device=self.device,
                dtype=torch.int64,
            )
            return ids

        def wrapped_compute_logprobs_only(*args, **kwargs):
            return None

        target.initialize = override_initialize
        target.forward = _version_dispatcher.get_compat_method("forward")
        target.sample = wrapped_sample
        target.compute_logprobs_only = wrapped_compute_logprobs_only

        # sglang >= 0.5.10 runs torch distributed init inside
        # ModelRunner.__init__ (before `initialize`), and its CPU bootstrap
        # path calls torch.ops.sgl_kernel.* ops that are unavailable in
        # CPU-only builds without the sgl-kernel extension. Neutralize
        # `_init_cpu_threads_env` for the duration of the call.
        if hasattr(target, "init_torch_distributed"):
            _original_init_torch_distributed = target.init_torch_distributed

            def wrapped_init_torch_distributed(self, *args, **kwargs):
                try:
                    from sglang.srt.distributed import bootstrap as _bootstrap
                except ImportError:
                    return _original_init_torch_distributed(self, *args, **kwargs)
                if not hasattr(_bootstrap, "_init_cpu_threads_env"):
                    return _original_init_torch_distributed(self, *args, **kwargs)
                _original = _bootstrap._init_cpu_threads_env
                _bootstrap._init_cpu_threads_env = lambda *a, **k: None
                try:
                    return _original_init_torch_distributed(self, *args, **kwargs)
                finally:
                    _bootstrap._init_cpu_threads_env = _original

            target.init_torch_distributed = wrapped_init_torch_distributed

        # sglang >= 0.5.10 builds real attention backends from
        # Scheduler.init_all_attention_backends; the mock forward pass only
        # needs the stub backend installed above.
        if hasattr(target, "init_attention_backends"):
            target.init_attention_backends = lambda self, *args, **kwargs: None
        # sglang >= 0.5.10 triggers cuda graph capture from
        # Scheduler.init_all_cuda_graphs; graphs are irrelevant for the
        # simulation (the forward pass is intercepted).
        if hasattr(target, "init_cuda_graphs"):
            target.init_cuda_graphs = lambda self, *args, **kwargs: None

        # sglang >= 0.5.10 host pools allocate their buffers with
        # cudaHostRegister by default, which is unavailable on CPU-only
        # builds; route CPU allocations to plain (non-pinned) host tensors.
        try:
            import torch as _torch
            from sglang.srt.mem_cache.pool_host import common as _host_common

            def _cpu_host_alloc(
                dims,
                dtype,
                device,
                pin_memory,
                allocator=None,
                registration_granularity_bytes=None,
            ):
                return _torch.empty(dims, dtype=dtype, device=device, pin_memory=False)

            _host_common.ALLOC_MEMORY_FUNCS.setdefault("cpu", _cpu_host_alloc)
        except Exception:  # pragma: no cover - older sglang has no pool_host pkg
            pass


class C_HiCacheController(BaseHook):
    HOOK_CLASS_NAME = "HiCacheController"
    HOOK_MODULE_NAME = "sglang.srt.managers.cache_controller"

    KV_CACHE_BYTES: int = None
    KV_CACHE_BYTES_FULL: int = None
    DISK_READ_BANDWIDTH_BYTES: float = None
    DISK_WRITE_BANDWIDTH_BYTES: float = None

    _ssd_simulator = None
    _ssd_simulator_initialized = False
    _tier_sims = None
    _tier_bandwidths = None
    _tier_write_bandwidths = None
    _tier_read_mode = "serial"
    _tier_write_mode = "parallel"

    @classmethod
    def _get_ssd_simulator(cls):
        if not cls._ssd_simulator_initialized:
            cls._init_ssd_simulator()
        return cls._ssd_simulator

    @classmethod
    def _init_ssd_simulator(cls):
        cls._ssd_simulator_initialized = True
        env = os.environ.get("HISIM_SSD_SIM", "")

        # Check for multi-tier config
        try:
            from hisim.simulation.manager import ConfigManager

            platform_cfg = ConfigManager.get_platform_config()
            tiers = platform_cfg.storage_tiers
            cls._tier_read_mode = platform_cfg.tier_read_mode
            cls._tier_write_mode = platform_cfg.tier_write_mode
        except Exception:
            tiers = []

        if tiers:
            cls._tier_sims = {}
            cls._tier_bandwidths = {}
            cls._tier_write_bandwidths = {}
            for tier in tiers:
                cls._tier_bandwidths[tier.name] = tier.read_bandwidth_gb * 1e9
                wbg = tier.write_bandwidth_gb or tier.read_bandwidth_gb
                cls._tier_write_bandwidths[tier.name] = wbg * 1e9
                if env in ("1", "default", "true", "True", "simpyssd", "simpyssd_multi"):
                    try:
                        from simpyssd import SSDTimingSimulator

                        cls._tier_sims[tier.name] = SSDTimingSimulator(tier.ssd_config)
                        get_logger().info(
                            f"SSD simulator enabled for tier '{tier.name}' with config: {tier.ssd_config}"
                        )
                    except Exception as e:
                        get_logger().warning(
                            f"Failed to init SSD simulator for tier '{tier.name}': {e}"
                        )
            return

        # Single-tier fallback (existing behavior)
        if not env:
            return
        try:
            if env in ("1", "default", "true", "True", "simpyssd"):
                from simpyssd import SSDTimingSimulator
                import simpyssd as _pkg

                cfg_path = os.path.join(
                    os.path.dirname(_pkg.__file__), "configs", "es3500p_pcie5.cfg"
                )
                cls._ssd_simulator = SSDTimingSimulator(cfg_path)
                get_logger().info(
                    f"SSD simulator enabled (SimpySSD) with config: {cfg_path}"
                )
            else:
                from simpyssd import SSDTimingSimulator

                cls._ssd_simulator = SSDTimingSimulator(env)
                get_logger().info(
                    f"SSD simulator enabled (SimpySSD) with config: {env}"
                )
        except Exception as e:
            get_logger().warning(f"Failed to init SSD simulator: {e}")

    @staticmethod
    def _calc_prefetch_pages_ssd(
        sim, required_pages: int, page_size_byte: int, max_dur: float
    ) -> tuple[float, float]:
        # KV_CACHE_BYTES_FULL = total across all TP workers
        full_bytes = C_HiCacheController.KV_CACHE_BYTES_FULL or page_size_byte
        # Try timing-aware API first
        if hasattr(sim, 'submit_read'):
            total_bytes = required_pages * full_bytes
            # read_arrival = T_{N-1} + l2_load_{N-1}
            #              = T_N - inference_{N-1} - l2_backup_{N-1} - l3_overflow_{N-1}
            read_arrival = (
                StateManager.get_global_clock()
                - StateManager.get_last_inference_dur()
                - StateManager.get_last_l2_backup_dur()
                - StateManager.get_last_l3_write_overflow()
            )
            avail_snapshot = sim.get_all_avail_time()
            comp, lat, wait, exec_t = sim.submit_read(
                read_arrival, total_bytes, page_size_byte, is_sequential=True
            )
            # Restore all chips' avail_time to undo read's effect (sglang aborts prefetch)
            sim.set_all_avail_time(avail_snapshot)
            # Separate wait and exec for accurate partial completion
            remaining_budget = max_dur - wait
            if remaining_budget <= 0:
                return 0.0, max_dur
            if exec_t <= remaining_budget:
                return float(required_pages), lat
            ratio = remaining_budget / exec_t
            completed = max(1.0, required_pages * ratio)
            return completed, max_dur
        # Fallback: original non-timing API
        total_bytes = required_pages * full_bytes
        read_time = sim.predict_read_time(
            total_bytes, page_size_byte, is_sequential=True
        )
        if read_time <= max_dur:
            return float(required_pages), read_time
        ratio = max_dur / read_time
        completed = max(1, int(required_pages * ratio))
        return float(completed), max_dur

    @staticmethod
    def _calc_prefetch_multi_tier(
        per_tier_hits: dict, page_size_byte: int, max_dur: float
    ) -> tuple[float, float]:
        # read_arrival = T_{N-1} + l2_load_{N-1}
        #              = T_N - inference_{N-1} - l2_backup_{N-1} - l3_overflow_{N-1}
        read_arrival = (
            StateManager.get_global_clock()
            - StateManager.get_last_inference_dur()
            - StateManager.get_last_l2_backup_dur()
            - StateManager.get_last_l3_write_overflow()
        )

        def _read_tier(tier_name, hit_count, arrival):
            """Submit a read to one tier, return (wait, exec_t, lat)."""
            full_bytes = C_HiCacheController.KV_CACHE_BYTES_FULL or page_size_byte
            total_bytes = hit_count * full_bytes
            sim = (
                C_HiCacheController._tier_sims.get(tier_name)
                if C_HiCacheController._tier_sims
                else None
            )
            if sim and hasattr(sim, 'submit_read'):
                avail_snapshot = sim.get_all_avail_time()
                _, lat, wait, exec_t = sim.submit_read(
                    arrival, total_bytes, page_size_byte, is_sequential=True
                )
                sim.set_all_avail_time(avail_snapshot)
                return wait, exec_t, lat
            elif sim:
                t = sim.predict_read_time(
                    total_bytes, page_size_byte, is_sequential=True
                )
                return 0.0, t, t
            else:
                bw = (
                    C_HiCacheController._tier_bandwidths.get(tier_name, 8e9)
                    if C_HiCacheController._tier_bandwidths
                    else 8e9
                )
                t = total_bytes / bw
                return 0.0, t, t

        active = {n: c for n, c in per_tier_hits.items() if c > 0}
        if not active:
            return 0.0, 0.0

        if C_HiCacheController._tier_read_mode == "serial":
            # Serial: tiers read in priority order, each arrival chains
            # off the previous tier's completion.
            accumulated = 0.0
            completed = 0.0
            for tier_name, hit_count in active.items():
                arrival = read_arrival + accumulated
                wait, exec_t, lat = _read_tier(tier_name, hit_count, arrival)
                budget = max_dur - accumulated
                if budget <= wait:
                    # Can't even finish waiting — this and all later tiers get 0
                    break
                if budget >= lat:
                    # Full completion
                    completed += hit_count
                    accumulated += lat
                else:
                    # Partial: only (budget - wait) time for exec
                    if exec_t > 0:
                        partial = hit_count * (budget - wait) / exec_t
                        completed += max(1.0, partial) if hit_count > 0 else 0
                    return completed, max_dur
            return completed, accumulated
        else:
            # Parallel: all tiers start at the same read_arrival
            tier_results = {}
            for tier_name, hit_count in active.items():
                wait, exec_t, lat = _read_tier(tier_name, hit_count, read_arrival)
                tier_results[tier_name] = (hit_count, wait, exec_t, lat)

            total_completed = sum(c for c, _, _, _ in tier_results.values())
            total_dur = max(lat for _, _, _, lat in tier_results.values())
            if total_dur <= max_dur:
                return float(total_completed), total_dur
            # Partial: each tier independently checks budget after its own wait
            completed = 0.0
            for _, (hit_count, wait, exec_t, lat) in tier_results.items():
                remaining = max_dur - wait
                if remaining <= 0:
                    continue
                if exec_t <= remaining:
                    completed += hit_count
                elif exec_t > 0:
                    completed += max(1.0, hit_count * (remaining / exec_t))
            return completed, max_dur

    @staticmethod
    def calc_prefetch_pages(
        required_pages, page_size_byte: int, max_dur: float, bandwidth: float
    ) -> tuple[float, float]:
        # Multi-tier mode: required_pages is dict[str, int]
        if isinstance(required_pages, dict):
            return C_HiCacheController._calc_prefetch_multi_tier(
                required_pages, page_size_byte, max_dur
            )
        # Single-tier SSD sim
        sim = C_HiCacheController._get_ssd_simulator()
        if sim is not None:
            return C_HiCacheController._calc_prefetch_pages_ssd(
                sim, required_pages, page_size_byte, max_dur
            )
        # Original bandwidth model
        full_bytes = C_HiCacheController.KV_CACHE_BYTES_FULL or page_size_byte
        _prefetch_dur = required_pages * full_bytes / bandwidth
        if _prefetch_dur > max_dur:
            _completed_pages = max(max_dur * bandwidth / full_bytes, 1)
            return _completed_pages, max_dur
        else:
            return required_pages, _prefetch_dur

    @classmethod
    def hook(cls, target):
        def override_backup_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_backup_operation`
            pass

        def override_prefetch_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_prefetch_operation`
            pass

        def handle_backup_operation(self):
            """Data phase only: dequeue, batch_set to kvcm, accumulate write info."""
            if not self.enable_storage:
                return
            total_write_tokens = 0
            per_tier_write_accum = {}
            while True:
                try:
                    operation = self.backup_queue.get(block=False)
                    if operation is None:
                        break

                    if not self.backup_skip:
                        self._page_backup(operation)
                    if hasattr(operation, "host_indices"):
                        total_write_tokens += len(operation.host_indices)
                    if C_HiCacheController._tier_sims:
                        try:
                            per_tier = self.storage_backend.get_last_write_per_tier()
                            for tn, cnt in per_tier.items():
                                per_tier_write_accum[tn] = per_tier_write_accum.get(tn, 0) + cnt
                        except Exception:
                            pass
                    self.ack_backup_queue.put(operation)

                except Empty:
                    break

            # Stash for submit_l3_write_timing() to consume after prefetch
            self._pending_write_data = (per_tier_write_accum, total_write_tokens)

        def submit_l3_write_timing(self):
            """Timing phase: submit_write to SSD sims + update state.
            Called AFTER handle_prefetch_operation so that submit_read
            sees the previous round's write avail_time, not this round's."""
            per_tier_write_accum, total_write_tokens = getattr(
                self, "_pending_write_data", ({}, 0)
            )
            if total_write_tokens <= 0:
                return

            if C_HiCacheController.KV_CACHE_BYTES is None:
                C_HiCacheController.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()
            if C_HiCacheController.KV_CACHE_BYTES_FULL is None:
                C_HiCacheController.KV_CACHE_BYTES_FULL = ConfigManager.get_kv_cache_bytes_full()

            # L3 write starts at T_N (data's L2 backup was 2 rounds ago, already done)
            write_arrival = StateManager.get_global_clock()

            # Multi-tier: per-tier SSD timing with independent engines
            if C_HiCacheController._tier_sims and per_tier_write_accum:
                tier_lats = {}
                accumulated = 0.0
                for tier_name, write_blocks in per_tier_write_accum.items():
                    if write_blocks <= 0:
                        continue
                    write_bytes = write_blocks * C_HiCacheController.KV_CACHE_BYTES_FULL
                    sim = C_HiCacheController._tier_sims.get(tier_name)
                    if C_HiCacheController._tier_write_mode == "serial":
                        arrival = write_arrival + accumulated
                    else:
                        arrival = write_arrival
                    if sim is not None and hasattr(sim, 'submit_write'):
                        _, lat, _, _ = sim.submit_write(
                            arrival, write_bytes,
                            C_HiCacheController.KV_CACHE_BYTES,
                            is_sequential=True
                        )
                        tier_lats[tier_name] = lat
                    elif sim is not None:
                        lat = sim.predict_write_time(
                            write_bytes,
                            C_HiCacheController.KV_CACHE_BYTES,
                            is_sequential=True,
                        )
                        tier_lats[tier_name] = lat
                    else:
                        bw = (
                            C_HiCacheController._tier_write_bandwidths.get(tier_name, 8e9)
                            if C_HiCacheController._tier_write_bandwidths
                            else 8e9
                        )
                        lat = write_bytes / bw
                        tier_lats[tier_name] = lat
                    accumulated += tier_lats[tier_name]

                if C_HiCacheController._tier_write_mode == "serial":
                    write_time = sum(tier_lats.values())
                else:
                    write_time = max(tier_lats.values()) if tier_lats else 0.0
            else:
                # Single-tier fallback
                total_write_bytes = total_write_tokens * C_HiCacheController.KV_CACHE_BYTES_FULL
                sim = C_HiCacheController._get_ssd_simulator()
                if sim is not None and hasattr(sim, 'submit_write'):
                    _, lat, _, _ = sim.submit_write(
                        write_arrival, total_write_bytes,
                        C_HiCacheController.KV_CACHE_BYTES,
                        is_sequential=True
                    )
                    write_time = lat
                elif sim is not None:
                    write_time = sim.predict_write_time(
                        total_write_bytes,
                        C_HiCacheController.KV_CACHE_BYTES,
                        is_sequential=True,
                    )
                else:
                    if C_HiCacheController.DISK_WRITE_BANDWIDTH_BYTES is None:
                        C_HiCacheController.DISK_WRITE_BANDWIDTH_BYTES = (
                            ConfigManager.get_platform_config().disk_write_bandwidth
                        )
                    bw = C_HiCacheController.DISK_WRITE_BANDWIDTH_BYTES or 8e9
                    write_time = total_write_bytes / bw
            StateManager.inc_hicache_l3_backup_dur(write_time)
            logger.debug(
                f"L3 write timing: tokens={total_write_tokens}, "
                f"write_time={write_time:.3e}s"
            )

        def handle_prefetch_operation(self):
            if not self.enable_storage:
                return

            if C_HiCacheController.KV_CACHE_BYTES is None:
                C_HiCacheController.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()
            if C_HiCacheController.KV_CACHE_BYTES_FULL is None:
                C_HiCacheController.KV_CACHE_BYTES_FULL = ConfigManager.get_kv_cache_bytes_full()
            if C_HiCacheController.DISK_READ_BANDWIDTH_BYTES is None:
                C_HiCacheController.DISK_READ_BANDWIDTH_BYTES = (
                    ConfigManager.get_platform_config().disk_read_bandwidth
                )

            direct_mode = False
            try:
                direct_mode = ConfigManager.get_platform_config().enable_l3_direct_to_l1
            except Exception:
                pass

            if direct_mode:
                chunked = getattr(self, "chunked_prefetch_operation", None)
                if chunked is not None:
                    operation = chunked["operation"]
                    storage_hit_count = chunked["storage_hit_count"]
                    operation.completed_tokens = int(storage_hit_count)
                    operation.mark_terminate()
                    setattr(self, "chunked_prefetch_operation", None)
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                    req_stats.prefetch_complete_tokens = operation.completed_tokens
                    StateManager.inc_l3_direct_load_count(storage_hit_count)

                while True:
                    try:
                        operation = self.prefetch_queue.get(block=False)
                        if operation is None:
                            return
                        hash_value, storage_hit_count = self._storage_hit_query(operation)
                        if storage_hit_count > 0:
                            try:
                                req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                                req_stats.l3_hit_tokens = storage_hit_count
                            except Exception:
                                pass
                        if (
                            self.prefetch_threshold is not None
                            and storage_hit_count < self.prefetch_threshold
                        ):
                            operation.mark_terminate()
                            self.append_host_mem_release(operation.host_indices)
                            continue
                        operation.hash_value = hash_value[
                            : (storage_hit_count // self.page_size)
                        ]
                        storage_hit_count = (
                            storage_hit_count // self.page_size * self.page_size
                        )
                        if C_HiCacheController._tier_sims:
                            try:
                                per_tier_hits = self.storage_backend.batch_exists_per_tier()
                                C_SchedulerHook._L3_QUERY_COUNT += 1
                                for tn, cnt in per_tier_hits.items():
                                    C_SchedulerHook._L3_HIT_STATS[tn] = C_SchedulerHook._L3_HIT_STATS.get(tn, 0) + cnt
                                    C_SchedulerHook._L3_TOTAL_HITS += cnt
                            except Exception:
                                pass
                        operation.completed_tokens = int(storage_hit_count)
                        operation.mark_terminate()
                        C_SchedulerHook._L3_PREFETCHED_TOKENS += int(storage_hit_count)
                        req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                        req_stats.prefetch_complete_tokens = operation.completed_tokens
                        self.append_host_mem_release(
                            operation.host_indices[storage_hit_count:]
                        )
                        StateManager.inc_l3_direct_load_count(storage_hit_count)
                    except Empty:
                        return
                return

            # TODO: Overlap schedule
            remain_dur = StateManager.get_current_inference_dur()

            if not _is_legacy_sglang():
                # sglang >= 0.5.10: the storage query result is handed to the
                # stock drain path (prefetch_hit_queue -> host allocation ->
                # ack), simulating an instantly completed storage query. The
                # legacy chunked-prefetch machinery does not apply.
                while True:
                    try:
                        operation = self.prefetch_queue.get(block=False)
                        if operation is None:
                            return
                        hash_value, storage_hit_count = self._storage_hit_query(
                            operation
                        )
                        try:
                            req_stats = C_SchedulerHook.REQUEST_STATS[
                                operation.request_id
                            ]
                            req_stats.l3_hit_tokens = storage_hit_count
                        except Exception:
                            pass
                        C_SchedulerHook._L3_QUERY_COUNT += 1
                        C_SchedulerHook._L3_TOTAL_HITS += storage_hit_count
                        try:
                            per_tier = self.storage_backend.batch_exists_per_tier()
                            for tn, cnt in per_tier.items():
                                C_SchedulerHook._L3_HIT_STATS[tn] = (
                                    C_SchedulerHook._L3_HIT_STATS.get(tn, 0) + cnt
                                )
                        except Exception:
                            pass
                        # Mirror prefetch_thread_func (minus the real IO).
                        operation.hash_value = hash_value[
                            : (storage_hit_count // self.page_size)
                        ]
                        operation.storage_hit_count = storage_hit_count
                        try:
                            storage_hit_count_tensor = torch.tensor(
                                storage_hit_count, dtype=torch.int
                            )
                            self._all_reduce(
                                storage_hit_count_tensor,
                                torch.distributed.ReduceOp.MIN,
                                self.prefetch_hits_sync_groups,
                            )
                        except Exception:
                            pass
                        C_SchedulerHook._L3_PREFETCHED_TOKENS += storage_hit_count
                        direct_mode = False
                        try:
                            direct_mode = (
                                ConfigManager.get_platform_config().enable_l3_direct_to_l1
                            )
                        except Exception:
                            pass
                        if direct_mode and storage_hit_count > 0:
                            StateManager.inc_l3_direct_load_count(storage_hit_count)
                        self.prefetch_hit_queue.put(operation)
                    except Empty:
                        return
                return

            chunked_prefetch_operation = getattr(
                self, "chunked_prefetch_operation", None
            )
            if chunked_prefetch_operation is not None:
                operation = chunked_prefetch_operation["operation"]
                storage_hit_count = chunked_prefetch_operation["storage_hit_count"]
                per_tier_hits = chunked_prefetch_operation.get("per_tier_hits")
                remaining = storage_hit_count - operation.completed_tokens
                if per_tier_hits and storage_hit_count > 0 and C_HiCacheController._tier_sims:
                    ratio = remaining / storage_hit_count
                    calc_arg = {k: max(0, int(v * ratio)) for k, v in per_tier_hits.items()}
                else:
                    calc_arg = remaining
                completed_tokens, prefetch_dur = (
                    C_HiCacheController.calc_prefetch_pages(
                        calc_arg,
                        C_HiCacheController.KV_CACHE_BYTES,
                        remain_dur,
                        C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                    )
                )
                if completed_tokens < storage_hit_count - operation.completed_tokens:
                    operation.completed_tokens += completed_tokens
                    remain_dur = 0
                else:
                    operation.completed_tokens = int(storage_hit_count)
                    operation.mark_terminate()
                    remain_dur -= prefetch_dur
                    setattr(self, "chunked_prefetch_operation", None)
                    # Release host memory after current operation is finished
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                # update request states
                req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                req_stats.prefetch_complete_tokens = operation.completed_tokens

            while remain_dur > 0:
                try:
                    operation = self.prefetch_queue.get(block=False)
                    if operation is None:
                        return

                    hash_value, storage_hit_count = self._storage_hit_query(operation)
                    # Track L3 hit tokens per request (page_size=1, so count=tokens)
                    if storage_hit_count > 0:
                        try:
                            req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                            req_stats.l3_hit_tokens = storage_hit_count
                        except Exception:
                            pass
                    # not to prefetch if not enough benefits
                    if (
                        self.prefetch_threshold is not None
                        and storage_hit_count < self.prefetch_threshold
                    ):
                        operation.mark_terminate()
                        self.append_host_mem_release(operation.host_indices)
                        continue

                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    storage_hit_count = (
                        storage_hit_count // self.page_size * self.page_size
                    )

                    # Get per-tier hits if multi-tier is configured
                    per_tier_hits = None
                    if C_HiCacheController._tier_sims:
                        try:
                            per_tier_hits = self.storage_backend.batch_exists_per_tier()
                            C_SchedulerHook._L3_QUERY_COUNT += 1
                            for tn, cnt in per_tier_hits.items():
                                C_SchedulerHook._L3_HIT_STATS[tn] = C_SchedulerHook._L3_HIT_STATS.get(tn, 0) + cnt
                                C_SchedulerHook._L3_TOTAL_HITS += cnt
                        except Exception:
                            pass
                    calc_arg = per_tier_hits if per_tier_hits else storage_hit_count

                    completed_tokens, prefetch_dur = (
                        C_HiCacheController.calc_prefetch_pages(
                            calc_arg,
                            C_HiCacheController.KV_CACHE_BYTES,
                            remain_dur,
                            C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                        )
                    )
                    if completed_tokens < storage_hit_count:
                        # Continue to prefetch data next time.
                        operation.completed_tokens = int(completed_tokens)
                        C_SchedulerHook._L3_PREFETCHED_TOKENS += int(completed_tokens)
                        setattr(
                            self,
                            "chunked_prefetch_operation",
                            {
                                "operation": operation,
                                "storage_hit_count": storage_hit_count,
                                "per_tier_hits": per_tier_hits,
                            },
                        )
                        remain_dur = 0
                    else:
                        operation.completed_tokens = int(
                            storage_hit_count // self.page_size * self.page_size
                        )
                        operation.mark_terminate()
                        remain_dur -= prefetch_dur
                        C_SchedulerHook._L3_PREFETCHED_TOKENS += int(
                            storage_hit_count // self.page_size * self.page_size
                        )
                    # update request states
                    req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                    req_stats.prefetch_complete_tokens = operation.completed_tokens
                    # Release host memory after current operation is finished
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )

                except Empty:
                    return

        def override_generic_page_set(
            self, hash_values, host_indices, extra_info=None
        ) -> bool:
            # Always pass extra_info to storage_backend.
            host_pool = self.mem_pool_host
            # sglang >= 0.5.10 wraps host pools in a HostPoolGroup; unwrap the
            # primary (KV) pool for page-level access.
            get_pool = getattr(host_pool, "get_pool", None)
            if get_pool is not None:
                try:
                    host_pool = get_pool(None)
                except Exception:
                    try:
                        host_pool = host_pool.entries[0].host_pool
                    except Exception:
                        pass
            data = [
                host_pool.get_data_page(host_indices[i * self.page_size])
                for i in range(len(hash_values))
            ]
            return self.storage_backend.batch_set(hash_values, data, extra_info)

        # The async backup/prefetch threads must not run: the data/timing
        # phases are driven by handle_backup_operation /
        # handle_prefetch_operation from the hooked check_hicache_events
        # instead (scheduler-thread lockstep, deterministic timing). On
        # sglang >= 0.5.10 HybridCacheController overrides backup_thread_func
        # (shadowing this no-op); C_HybridCacheControllerHook handles that.
        target.prefetch_thread_func = override_prefetch_thread_func
        target.backup_thread_func = override_backup_thread_func
        target.handle_backup_operation = handle_backup_operation
        target.handle_prefetch_operation = handle_prefetch_operation
        target.submit_l3_write_timing = submit_l3_write_timing
        target._generic_page_set = override_generic_page_set

        # sglang >= 0.5.10 performs L2 transfers synchronously via
        # start_writing/start_loading (CUDA kernels + events). The mock host
        # pool holds no real KV data, so the "transfer" is instantaneous:
        # keep the stock bookkeeping but complete the ack events right away.
        if not _is_legacy_sglang() and hasattr(target, "start_writing"):
            def mock_start_writing(self):
                from sglang.srt.managers.cache_controller import (
                    CacheOperation,
                    HiCacheAck,
                )
                from sglang.srt.utils import get_device_module

                if len(self.write_queue) == 0:
                    return
                op = CacheOperation.merge_ops(self.write_queue)
                host_indices, device_indices, pool_transfers = (
                    self._move_write_operation(op)
                )
                self.write_queue.clear()
                device_module = get_device_module()
                start_event = device_module.Event(enable_timing=False)
                finish_event = device_module.Event(enable_timing=False)
                start_event.record()
                finish_event.record()
                self.ack_write_queue.append(
                    HiCacheAck(
                        start_event=start_event,
                        finish_event=finish_event,
                        node_ids=op.node_ids,
                        num_tokens=len(op.device_indices),
                        timing_enabled=False,
                        num_tokens_by_pool=self._num_tokens_by_pool(op),
                        num_bytes=self._transfer_num_bytes(op),
                    )
                )

            def mock_start_loading(self):
                from sglang.srt.managers.cache_controller import (
                    CacheOperation,
                    HiCacheAck,
                )
                from sglang.srt.utils import get_device_module

                if len(self.load_queue) == 0:
                    return -1
                producer_id = self.layer_done_counter.update_producer()
                op = CacheOperation.merge_ops(self.load_queue)
                host_indices, device_indices, pool_transfers = self._move_op_indices(
                    op
                )
                self.load_queue.clear()
                producer_event = self.layer_done_counter.events[producer_id]
                producer_event.start_event.record()
                # Simulate the per-layer completion of the host->device copy.
                for layer_index in range(self.layer_num):
                    producer_event.complete(layer_index)
                device_module = get_device_module()
                start_event = device_module.Event(enable_timing=False)
                finish_event = device_module.Event(enable_timing=False)
                start_event.record()
                finish_event.record()
                self.ack_load_queue.append(
                    HiCacheAck(
                        start_event=start_event,
                        finish_event=finish_event,
                        node_ids=op.node_ids,
                        num_tokens=len(op.device_indices),
                        timing_enabled=False,
                        num_tokens_by_pool=self._num_tokens_by_pool(op),
                        num_bytes=self._transfer_num_bytes(op),
                    )
                )
                return producer_id

            target.start_writing = mock_start_writing
            target.start_loading = mock_start_loading


class C_HiRadixCacheHook(BaseHook):
    HOOK_CLASS_NAME = "HiRadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.hiradix_cache"

    @classmethod
    def hook(cls, target):
        original_check_hicache_events = target.check_hicache_events
        original_reset = target.reset

        def wrapped_reset(self):
            if hasattr(self, "cache_controller"):
                self.cache_controller.handle_backup_operation()
                self.cache_controller.submit_l3_write_timing()
            original_reset(self)

        def override_init(self, params, server_args):
            if server_args.hicache_io_backend == "direct":
                # FIXME: move this logic into server_args parsing
                if server_args.hicache_mem_layout == "page_first":
                    server_args.hicache_mem_layout = "page_first_direct"
                    logger.warning(
                        "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                    )

            self.page_size = params.page_size
            self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
            # Replace the host pool
            self.token_to_kv_pool_host = MockTokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                pin_memory=False,
                device="cpu",
            )

            self.tp_group = params.tp_cache_group
            self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
            self.enable_storage = server_args.hicache_storage_backend is not None
            self.enable_storage_metrics = self.enable_storage and params.enable_metrics

            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                server_args.hicache_storage_backend_extra_config
            )
            self.prefetch_threshold = prefetch_threshold
            self.prefetch_timeout_base = prefetch_timeout_base
            self.prefetch_timeout_per_page = (
                self.page_size / 1024 * prefetch_timeout_per_ki_token
            )
            self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
            # TODO: support more timeout check functions
            self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
            self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

            HiCacheController = getattr(
                importlib.import_module("sglang.srt.managers.cache_controller"),
                "HiCacheController",
            )
            try:
                StorageMetricsCollector = getattr(
                    importlib.import_module("sglang.srt.metrics.collector"),
                    "StorageMetricsCollector",
                )
            except (ImportError, ModuleNotFoundError):
                StorageMetricsCollector = None

            self.load_cache_event = threading.Event()
            controller_kwargs = dict(
                load_cache_event=self.load_cache_event,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=self.prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
            )
            # sglang >= 0.5.10 introduced extra constructor parameters.
            import inspect as _inspect

            _ctor_params = _inspect.signature(HiCacheController.__init__).parameters
            if "enable_storage_metrics" in _ctor_params:
                controller_kwargs["enable_storage_metrics"] = (
                    self.enable_storage_metrics
                )
            if "host_memory_mode" in _ctor_params:
                controller_kwargs["host_memory_mode"] = getattr(
                    server_args, "hicache_host_memory_mode", "cache"
                )
            self.cache_controller = HiCacheController(
                params.token_to_kv_pool_allocator,
                self.token_to_kv_pool_host,
                self.page_size,
                self.tp_group,
                **controller_kwargs,
            )
            if self.enable_storage_metrics and StorageMetricsCollector is not None:
                # TODO: support pp
                labels = {
                    "storage_backend": server_args.hicache_storage_backend,
                    "tp_rank": self.cache_controller.tp_rank,
                    "dp_rank": self.cache_controller.dp_rank,
                }
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)

            # Record the nodes with ongoing write-through
            self.ongoing_write_through = {}
            # Record the node segments with ongoing load-back
            self.ongoing_load_back = {}
            # Record the ongoing prefetch requests
            self.ongoing_prefetch = {}
            self.ongoing_backup = {}
            # TODO: Dynamically adjust the threshold
            self.write_through_threshold = (
                1 if server_args.hicache_write_policy == "write_through" else 2
            )
            self.load_back_threshold = 10
            # Version: 0.5.9
            self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
            self.evictable_host_leaves = set()
            # super().__init__(params=params)
            target.__mro__[1].__init__(self, params=params)

        def wrapped_check_hicache_events(self, *args, **kwargs):
            # 1-3. Data + timing phases (storage backend only); see
            # _drive_hicache_timing_phases for the phase ordering rationale.
            _drive_hicache_timing_phases(self)
            result = original_check_hicache_events(self, *args, **kwargs)
            # 4. Fix (legacy sglang <= 0.5.9 only): In CPU simulation, CUDA
            # events in ack_write_queue and ack_load_queue never complete, so
            # writing_check() and loading_check() never process their queues.
            # We process them here as a fallback, skipping the CUDA event
            # check. On sglang >= 0.5.10 the transfers are synchronous with
            # instantly-completed mock events, and the stock checks above
            # already reap the acks.
            if (
                _is_legacy_sglang()
                and hasattr(self, 'cache_controller')
                and self.enable_storage
            ):
                cc = self.cache_controller
                # ack_write_queue fallback: mirrors sglang's writing_check()
                # normal path (hiradix_cache.py:698-706). Must call
                # dec_lock_ref to unlock the node so it can be evicted later.
                # Without dec_lock_ref, backed-up nodes stay locked in L1
                # forever, host_hit_length is always 0, and L2 load_back is
                # never triggered.
                n_write_acks = 0
                for ack in cc.ack_write_queue:
                    for ack_id in ack.node_ids:
                        node = self.ongoing_write_through.pop(ack_id, None)
                        if node is not None:
                            n_write_acks += 1
                            self.dec_lock_ref(node)
                            self.write_backup_storage(node)
                cc.ack_write_queue.clear()
                # ack_load_queue fallback: mirrors sglang's loading_check()
                # (hiradix_cache.py:708-721). Without this, nodes in
                # ongoing_load_back stay locked forever, eventually
                # exhausting evictable L1 space and breaking L2 load_back.
                n_load_acks = 0
                for ack in cc.ack_load_queue:
                    for ack_id in ack.node_ids:
                        end_node = self.ongoing_load_back.pop(ack_id, None)
                        if end_node is not None:
                            n_load_acks += 1
                            self.dec_lock_ref(end_node)
                cc.ack_load_queue.clear()
                if n_write_acks or n_load_acks:
                    logger.debug(
                        f"check_hicache_events fallback: write_acks={n_write_acks}, "
                        f"load_acks={n_load_acks}, "
                        f"ongoing_write_through={len(self.ongoing_write_through)}, "
                        f"ongoing_load_back={len(self.ongoing_load_back)}, "
                        f"backup_queue={cc.backup_queue.qsize() if hasattr(cc.backup_queue, 'qsize') else 'N/A'}, "
                        f"prefetch_queue={cc.prefetch_queue.qsize() if hasattr(cc.prefetch_queue, 'qsize') else 'N/A'}"
                    )
            return result

        target.__init__ = override_init
        target.check_hicache_events = wrapped_check_hicache_events
        target.reset = wrapped_reset

        # Node lifecycle tracking (enabled via HISIM_NODE_TRACK env var)
        if os.environ.get("HISIM_NODE_TRACK", ""):
            _orig_write_backup = target.write_backup
            _orig_evict_backuped = target._evict_backuped
            _orig_evict_regular = target._evict_regular
            _orig_evict_host = target.evict_host
            _orig_load_back = target.load_back

            def _tracked_write_backup(self, node, write_back=False):
                r = _orig_write_backup(self, node, write_back)
                if r > 0:
                    logger.info(f"NODE_BACKUP id={node.id} key_len={len(node.key)} backuped={node.backuped}")
                return r

            def _tracked_evict_backuped(self, node):
                r = _orig_evict_backuped(self, node)
                logger.info(f"NODE_EVICT_L1 id={node.id} key_len={len(node.key)} backuped={node.backuped} evicted={node.evicted}")
                return r

            def _tracked_evict_regular(self, node):
                logger.info(f"NODE_DELETE id={node.id} key_len={len(node.key)} backuped={node.backuped}")
                return _orig_evict_regular(self, node)

            def _tracked_evict_host(self, num_tokens):
                # Log before eviction (nodes still in tree)
                leaves = list(self.evictable_host_leaves)
                r = _orig_evict_host(self, num_tokens)
                # Log what changed
                after = list(self.evictable_host_leaves)
                removed = [n for n in leaves if n not in after]
                for n in removed:
                    logger.info(f"NODE_EVICT_L2 id={n.id} key_len={len(n.key)} backuped={n.backuped} evicted={n.evicted}")
                return r

            def _tracked_load_back(self, node, mem_quota=None):
                logger.info(f"NODE_LOAD_BACK_START id={node.id} key_len={len(node.key)} evicted={node.evicted} backuped={node.backuped}")
                r = _orig_load_back(self, node, mem_quota)
                if r is not None:
                    logger.info(f"NODE_LOAD_BACK_OK id={node.id} tokens={len(r)}")
                else:
                    logger.info(f"NODE_LOAD_BACK_FAIL id={node.id}")
                return r

            target.write_backup = _tracked_write_backup
            target._evict_backuped = _tracked_evict_backuped
            target._evict_regular = _tracked_evict_regular
            target.evict_host = _tracked_evict_host
            target.load_back = _tracked_load_back
            logger.info("Node lifecycle tracking enabled (HISIM_NODE_TRACK)")


class C_HybridCacheControllerHook(BaseHook):
    """sglang >= 0.5.10: HybridCacheController overrides backup_thread_func
    (sidecar pools must be backed up on every TP rank), shadowing the no-op
    installed on HiCacheController. The storage data phase is driven by
    handle_backup_operation from the hooked check_hicache_events instead, so
    the thread must stay idle to avoid double-consuming backup_queue."""

    HOOK_CLASS_NAME = "HybridCacheController"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller"

    @classmethod
    def hook(cls, target):
        def override_backup_thread_func(self, *args, **kwargs):
            # Async thread: perform no action.
            # The action will be performed by `handle_backup_operation`.
            pass

        target.backup_thread_func = override_backup_thread_func


class C_UnifiedRadixCacheHook(BaseHook):
    """sglang >= 0.5.10 builds the hierarchical tree cache as
    UnifiedRadixCache (registry.create_tree_cache) instead of HiRadixCache,
    so C_HiRadixCacheHook never fires. Drive the same storage data/timing
    phases from UnifiedRadixCache.check_hicache_events / reset."""

    HOOK_CLASS_NAME = "UnifiedRadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.unified_radix_cache"

    @classmethod
    def hook(cls, target):
        original_check_hicache_events = target.check_hicache_events
        original_reset = target.reset

        def wrapped_check_hicache_events(self, *args, **kwargs):
            _drive_hicache_timing_phases(self)
            return original_check_hicache_events(self, *args, **kwargs)

        def wrapped_reset(self):
            # Flush the pending storage writes and their write timing before
            # the tree state is destroyed.
            cc = getattr(self, "cache_controller", None)
            if cc is not None and getattr(cc, "enable_storage", False):
                cc.handle_backup_operation()
                cc.submit_l3_write_timing()
            original_reset(self)

        target.check_hicache_events = wrapped_check_hicache_events
        target.reset = wrapped_reset


class C_StorageBackendFactory(BaseHook):
    HOOK_CLASS_NAME = "StorageBackendFactory"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.storage.backend_factory"

    @classmethod
    def hook(cls, target):
        def override_create_backend(cls, *args, **kwargs):
            logger.info("Creating hijacked cache storage backend.")
            return MockHiCacheStorage()

        target.create_backend = override_create_backend


class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    INFERENCE_PREDICTOR: InferTimePredictor = None

    REQUEST_STATS: dict[str, RequestStats] = defaultdict(RequestStats)
    ITERATION_STATS: list[dict] = []
    LAST_CPU_TS: float = 0
    LAST_FLUSH_TS: float = 0
    HISIM_BATCH: HisimScheduleBatch = None

    OVERLAP_SCHEDULE: bool = False

    SIM_MODE = MockSimulationMode(Envs.simulation_mode())
    OFFLINE_RECV_ALL_REQUEST: bool = False
    FUTURE_QUEUE: list[
        tuple[float, int, RequestStats]
    ] = []  # tuple(created time, salt, request)

    # --- Dynamic completion feedback (session-aware replay) ---
    # Active only for requests carrying `session` metadata in their simulation
    # params (e.g. weka_trace rounds). The next round of a session is released
    # at max(static time, previous round's actual finish + recorded gap).
    # session -> {"round_idx", "recorded_finish", "finished", "finish_vtime"}
    SESSION_STATE: dict = {}
    # session -> [(enqueue_time, salt, req)] rounds whose static time arrived
    # but whose session predecessor has not finished yet
    PARKED: dict = {}
    # rid -> session, populated when a session-aware request is released
    REQ_SESSION: dict = {}

    # L2/L3 cache hit statistics
    _L3_HIT_STATS: dict = {}  # tier_name -> total hit blocks
    _L3_QUERY_COUNT: int = 0  # total batch_exists calls
    _L3_TOTAL_HITS: int = 0  # total hit blocks across all tiers
    _L3_PREFETCHED_TOKENS: int = 0  # total tokens actually prefetched from L3
    _L2_HIT_TOKENS: int = 0  # total tokens matched in L2 (host_hit_length sum)
    _L2_HIT_COUNT: int = 0  # number of requests with host_hit_length > 0

    SCHEDULE_REQ_STATS = []

    # ------------------------------------------------------------------
    # Helpers shared by the legacy (<= 0.5.9) and new (>= 0.5.10) request
    # paths. `sampling_params` is a `SamplingParams` object on old versions
    # and may arrive as a plain dict on newer msgspec-based IO structs.
    # ------------------------------------------------------------------

    @staticmethod
    def get_simulation_params(req):
        sp = getattr(req, "sampling_params", None)
        if sp is None:
            return None
        if isinstance(sp, dict):
            custom_params = sp.get("custom_params")
        else:
            custom_params = getattr(sp, "custom_params", None)
        if not isinstance(custom_params, dict):
            return None
        return custom_params.get("simulation")

    @staticmethod
    def get_max_new_tokens(req):
        sp = getattr(req, "sampling_params", None)
        if isinstance(sp, dict):
            return sp.get("max_new_tokens")
        return getattr(sp, "max_new_tokens", None)

    @staticmethod
    def get_extend_input_len(req):
        """Number of new tokens processed by the current extend batch."""
        extend_range = getattr(req, "extend_range", None)  # sglang >= 0.5.10
        if extend_range is not None:
            return extend_range.length
        extend_input_len = getattr(req, "extend_input_len", None)  # sglang <= 0.5.9
        if extend_input_len is not None:
            return extend_input_len
        return len(req.origin_input_ids) - len(req.prefix_indices)

    @staticmethod
    def is_chunked_req(scheduler, req):
        is_chunked = getattr(req, "is_chunked", None)  # sglang <= 0.5.9
        if is_chunked is not None:
            return is_chunked != 0
        return req is scheduler.chunked_req  # sglang >= 0.5.10

    @classmethod
    def record_request_stats(cls, req, now):
        req_stats = cls.REQUEST_STATS[req.rid]
        req_stats.rid = req.rid
        req_stats.input_length = len(req.input_ids)
        req_stats.output_length = cls.get_max_new_tokens(req)
        simulation_args = cls.get_simulation_params(req)
        if cls.SIM_MODE == MockSimulationMode.BLOCKING:
            if simulation_args is None:
                # Health-check / warm-up request without simulation parameters.
                logger.warning(
                    "The request's simulation parameters are missing, using the "
                    "current time as the creation time."
                )
                simulation_args = {}
            req_stats.created_time = simulation_args.get("server_created_time", now)
            req_stats.last_event_time = req_stats.created_time
            req_stats.queue_start = now
        elif cls.SIM_MODE == MockSimulationMode.OFFLINE:
            if simulation_args is None:
                # Warm-up requests are excluded from the statistics.
                cls.REQUEST_STATS.pop(req.rid, None)
                return
            req_stats.created_time = simulation_args["created_time"]
            req_stats.last_event_time = req_stats.created_time
            # Align with the real queue start timestamp if queue_start is not None. For debugging only.
            queue_start = simulation_args.get("queue_start")
            if queue_start is not None:
                StateManager.set_global_clock(queue_start)
            req_stats.queue_start = StateManager.get_global_clock()

    @classmethod
    def _offline_collect_and_release(cls, scheduler, recv_reqs, dispatch):
        """OFFLINE-mode buffering for sglang >= 0.5.10.

        Generate requests are collected into FUTURE_QUEUE until all of them
        (`total_request`) have arrived; afterwards the requests whose simulated
        arrival time is due are released to the dispatcher. Returns the
        requests that should be dispatched in this call."""
        if not cls.OFFLINE_RECV_ALL_REQUEST:
            gen_requests = []
            extra_requests = []
            time.sleep(0.05)  # waiting requests

            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    gen_requests.append(req)
                else:
                    # Such as: /profile_start, /flush_cache, etc.
                    extra_requests.append(req)

            for req in gen_requests:
                sim_params = cls.get_simulation_params(req)
                if sim_params is None:
                    # There are some warm-up requests when starting the server without --skip-server-warmup.
                    extra_requests.append(req)
                    logger.warning(
                        "Failed to extract the simulation parameters required for simulation from the request. Ignore this warning if the request is a warm-up request."
                    )
                    continue
                if sim_params.get("queue_start"):
                    logger.debug(
                        "Add request to waiting queue with custom queue start timestamp."
                    )
                cls.FUTURE_QUEUE.append(
                    (
                        sim_params.get("queue_start") or sim_params["created_time"],
                        time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                        req,
                    )
                )

            if len(cls.FUTURE_QUEUE) != 0:
                _, _, gen_req = cls.FUTURE_QUEUE[-1]
                total_request = cls.get_simulation_params(gen_req)["total_request"]

                if len(cls.FUTURE_QUEUE) == total_request:
                    cls.OFFLINE_RECV_ALL_REQUEST = True
                    heapq.heapify(cls.FUTURE_QUEUE)
                    logger.info("All requests received. Starting simulation now.")
                else:
                    logger.info(
                        f"Offline simulation mode enabled. {total_request} requests expected in total. "
                        f"Received {len(cls.FUTURE_QUEUE)} requests so far."
                    )

            if extra_requests:
                # Schedule the extra requests immediately.
                dispatch(scheduler, extra_requests)
            return []

        # All generate requests have been buffered. Pass through newly arrived
        # requests (flush, abort, health check, etc.) and release the buffered
        # generate requests that are due.
        released = list(recv_reqs)
        current_timestamp = StateManager.get_global_clock()
        while len(cls.FUTURE_QUEUE) > 0:
            enqueue_time, _, req = cls.FUTURE_QUEUE[0]
            if enqueue_time > current_timestamp:
                break
            released.append(req)
            heapq.heappop(cls.FUTURE_QUEUE)
        return released

    @classmethod
    def dump_profile_results(cls):
        """Dump the simulation results (metrics/iteration/request files) and
        reset the state. Shared by the legacy `Scheduler.profile` hook and the
        sglang >= 0.5.10 `SchedulerProfilerManager` hook."""
        stats: list[RequestStats] = []
        for item in cls.REQUEST_STATS.values():
            if item.rid is not None and item.input_length > 0:
                stats.append(item)

        stats = sorted(stats, key=lambda req: req.created_time)

        output_dir = Envs.output_dir()
        os.makedirs(output_dir, exist_ok=True)

        if len(stats) > 0:
            # Remove warmup requests.
            if len(stats) > Envs.num_warmup():
                metrics_stats = stats[Envs.num_warmup() :]
            else:
                metrics_stats = stats

            min_created_time = metrics_stats[0].created_time
            # Align timestamps
            for item in stats:
                item.created_time -= min_created_time
                item.queue_start -= min_created_time
                item.queue_end -= min_created_time
                item.last_event_time -= min_created_time

            metrics = calc_metrics(metrics_stats)
            metrics["time_cost"] = time.time() - cls.LAST_FLUSH_TS

            try:
                with open(f"{output_dir}/metrics.json", "w") as f:
                    f.write(json.dumps(metrics, cls=CustomJsonEncoder) + "\n")

                with open(f"{output_dir}/iteration.jsonl", "w") as f:
                    for item in cls.ITERATION_STATS:
                        f.write(json.dumps(item) + "\n")

                with open(f"{output_dir}/request.jsonl", "w") as f:
                    for item in stats:
                        f.write(json.dumps(asdict(item)) + "\n")

                logger.info(f"Simulation results saved to {output_dir}.")

                # Log L2/L3 cache hit statistics
                from hisim.simulation.sglang.sglang_mock_class import MockHiCacheStorage
                logger.info(
                    f"Storage stats: batch_set={MockHiCacheStorage._DEBUG_BATCH_SET_COUNT}, "
                    f"batch_exists={MockHiCacheStorage._DEBUG_BATCH_EXISTS_COUNT}, "
                    f"hits={MockHiCacheStorage._DEBUG_BATCH_EXISTS_HITS}"
                )
                if cls._L3_QUERY_COUNT > 0:
                    logger.info(
                        f"L3 cache stats: queries={cls._L3_QUERY_COUNT}, "
                        f"total_hits={cls._L3_TOTAL_HITS} blocks, "
                        f"prefetched={cls._L3_PREFETCHED_TOKENS} tokens, "
                        f"per_tier={cls._L3_HIT_STATS}"
                    )
                logger.info(
                    f"L2 cache stats: hit_requests={cls._L2_HIT_COUNT}, "
                    f"hit_tokens={cls._L2_HIT_TOKENS}"
                )
                # Log per-request L1/L2/L3 hit breakdown
                for item in metrics_stats:
                    # final_reused_tokens (cached_tokens) includes both L1 prefix match
                    # AND L2 load_back (init_load_back extends prefix_indices).
                    # Subtract L2 to get L1-only.
                    l2 = item.l2_hit_tokens
                    l1 = item.final_reused_tokens - l2
                    l3 = item.l3_hit_tokens
                    new_tok = item.input_length - l1 - l2 - l3
                    logger.info(
                        f"Req #{item.rid}: input={item.input_length} "
                        f"L1={l1} L2={l2} L3={l3} New={new_tok}"
                    )

            except Exception as e:
                logger.error(f"Failed to dump results. Error: {e}")
        else:
            logger.warning("No request statistics available.")

        StateManager.reset()
        cls.REQUEST_STATS.clear()
        cls.ITERATION_STATS.clear()
        cls.LAST_CPU_TS = 0
        cls.LAST_FLUSH_TS = time.time()
        cls.OFFLINE_RECV_ALL_REQUEST = False
        cls._L3_HIT_STATS = {}
        cls._L3_QUERY_COUNT = 0
        cls._L3_TOTAL_HITS = 0
        cls._L3_PREFETCHED_TOKENS = 0
        cls._L2_HIT_TOKENS = 0
        cls._L2_HIT_COUNT = 0
        cls.SESSION_STATE.clear()
        cls.PARKED.clear()
        cls.REQ_SESSION.clear()

        ProfileReqOutput = getattr(
            importlib.import_module("sglang.srt.managers.io_struct"),
            "ProfileReqOutput",
        )
        result = {
            "total_request": len(stats),
            "output_directory": output_dir,
        }

        # sglang >= 0.5.10 defines ProfileReqOutput as a kw-only msgspec
        # struct, so keyword arguments are required.
        return ProfileReqOutput(success=True, message=json.dumps(result))

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_get_new_batch_prefill = target.get_new_batch_prefill
        original_run_batch = target.run_batch
        original_process_batch_result = target.process_batch_result
        original_event_loop_normal = target.event_loop_normal

        # sglang >= 0.5.10 removed `Scheduler.recv_requests`: requests are now
        # received by a separate `SchedulerRequestReceiver` and dispatched via
        # `Scheduler.process_input_requests`.
        legacy_request_path = hasattr(target, "recv_requests")
        if legacy_request_path:
            original_recv_requests = target.recv_requests
        else:
            original_process_input_requests = target.process_input_requests

        def override_event_loop_overlap(self, *args, **kwargs):
            # To reduce the complexity of the simulation, the overlapping schedule is not needed.
            return original_event_loop_normal(self, *args, **kwargs)

        def wrapped_init(self, *args, **kwargs):
            # Disable overlap schedule
            server_args = get_obj_from_args(
                "sglang.srt.server_args.ServerArgs", *args, **kwargs
            )
            C_SchedulerHook.OVERLAP_SCHEDULE = not getattr(
                server_args, "disable_overlap_schedule", False
            )
            # sglang >= 0.5.10 resolves ServerArgs into a read-only struct, so
            # the mutation may fail there; `launch_server` disables overlap
            # before resolution and the instance attribute is overridden below.
            try:
                setattr(server_args, "disable_overlap_schedule", True)
            except (AttributeError, TypeError):
                pass
            logger.debug(
                f"Overlap schedule simulation mode: {C_SchedulerHook.OVERLAP_SCHEDULE}."
            )

            original_init(self, *args, **kwargs)

            # sglang >= 0.5.10: ServerArgs may be read-only, so disable the
            # overlap schedule on the scheduler instance instead. This drives
            # both dispatch_event_loop() and the run_batch() branch selection.
            if getattr(self, "enable_overlap", False):
                self.enable_overlap = False
                self.enable_overlap_mlx = False

            try:
                model = ConfigManager.get_model_info(
                    self.model_config.hf_config.__dict__
                )
                hw = ConfigManager.get_accelerator_info()
                sched_config = ConfigManager.get_scheduler_config(
                    self.server_args.__dict__,
                    "sglang",
                    self.model_config.hf_config.__dict__,
                )
                ConfigManager.set_scheduler_config(sched_config)
                ConfigManager.set_model_info(model)

                C_SchedulerHook.INFERENCE_PREDICTOR = (
                    ConfigManager.get_inference_time_predictor(model, hw, sched_config)
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize inference time predictor. Error: {e}"
                )
                raise e

        def wrapped_recv_requests(self, *args, **kwargs) -> list:
            recv_reqs = []

            if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                recv_reqs.extend(original_recv_requests(self, *args, **kwargs))
            elif C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                # Initializing
                if not C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST:
                    gen_requests = []
                    extra_requests = []
                    time.sleep(0.05)  # waiting requests

                    reqs = original_recv_requests(self, *args, **kwargs)

                    for req in reqs:
                        if req.__class__.__name__ == "TokenizedGenerateReqInput":
                            gen_requests.append(req)
                        else:
                            # Such as: /profile_start, /flush_cache, etc.
                            extra_requests.append(req)

                    # Add requests to future queue
                    for req in gen_requests:
                        sim_params = None
                        if req.sampling_params.custom_params is not None:
                            sim_params = req.sampling_params.custom_params.get(
                                "simulation"
                            )
                        if sim_params is None:
                            # There are some warm-up requests when starting the server without --skip-server-warmup.
                            extra_requests.append(req)
                            logger.warning(
                                "Failed to extract the simulation parameters required for simulation from the request. Ignore this warning if the request is a warm-up request."
                            )
                            continue
                        if sim_params.get("queue_start"):
                            logger.debug(
                                "Add request to waiting queue with custom queue start timestamp."
                            )

                        C_SchedulerHook.FUTURE_QUEUE.append(
                            (
                                sim_params.get("queue_start")
                                or sim_params["created_time"],
                                time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                                req,
                            )
                        )

                    if len(C_SchedulerHook.FUTURE_QUEUE) != 0:
                        _, _, gen_req = C_SchedulerHook.FUTURE_QUEUE[-1]
                        total_request = gen_req.sampling_params.custom_params[
                            "simulation"
                        ]["total_request"]

                        if len(C_SchedulerHook.FUTURE_QUEUE) == total_request:
                            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = True
                            heapq.heapify(C_SchedulerHook.FUTURE_QUEUE)
                            logger.info(
                                "All requests received. Starting simulation now."
                            )
                        else:
                            logger.info(
                                f"Offline simulation mode enabled. {total_request} requests expected in total. Received {len(C_SchedulerHook.FUTURE_QUEUE)} requests so far."
                            )

                    if len(extra_requests) != 0:
                        # Schedule the extra requests immediately.
                        return extra_requests
                else:
                    # Extra requests include: flush request, abort request, etc.
                    recv_reqs.extend(original_recv_requests(self, *args, **kwargs))

                # Process the arrived requests only after all requests have been added to the future queue
                current_timestamp = StateManager.get_global_clock()
                while (
                    C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST
                    and len(C_SchedulerHook.FUTURE_QUEUE) > 0
                ):
                    enqueue_time, salt, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    if enqueue_time > current_timestamp:
                        break
                    heapq.heappop(C_SchedulerHook.FUTURE_QUEUE)

                    # Dynamic completion feedback: a round whose session
                    # predecessor is still in flight is parked (not blocking
                    # other sessions); a finished predecessor may shift the
                    # release time by the accumulated virtual delay.
                    sim_args = (
                        req.sampling_params.custom_params.get("simulation")
                        if req.sampling_params.custom_params is not None
                        else None
                    )
                    session = sim_args.get("session") if sim_args else None
                    if session is not None:
                        st = C_SchedulerHook.SESSION_STATE.get(session)
                        if st is not None and not st["finished"]:
                            # Previous round of this session still in flight.
                            C_SchedulerHook.PARKED.setdefault(
                                session, []
                            ).append((enqueue_time, salt, req))
                            continue
                        if st is not None and st["finished"]:
                            # Shift by the virtual delay of the predecessor:
                            # release = static + max(0, actual - recorded).
                            shift = max(
                                0.0,
                                st["finish_vtime"] - st["recorded_finish"],
                            )
                            if shift > 0 and enqueue_time + shift > current_timestamp:
                                heapq.heappush(
                                    C_SchedulerHook.FUTURE_QUEUE,
                                    (enqueue_time + shift, salt, req),
                                )
                                continue
                        C_SchedulerHook.SESSION_STATE[session] = {
                            "round_idx": sim_args.get("round_idx"),
                            "recorded_finish": sim_args.get(
                                "recorded_finish", enqueue_time
                            ),
                            "finished": False,
                            "finish_vtime": 0.0,
                        }
                    recv_reqs.append(req)

            now = time.time()
            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.rid = req.rid
                    req_stats.input_length = len(req.input_ids)
                    req_stats.output_length = req.sampling_params.max_new_tokens
                    simulation_args = req.sampling_params.custom_params["simulation"]
                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        if "server_created_time" not in simulation_args:
                            logger.warning(
                                "The request's creation time is missing, which may cause the TTFT to be inaccurate."
                            )
                        req_stats.created_time = simulation_args.get(
                            "server_created_time", now
                        )
                        req_stats.last_event_time = req_stats.created_time
                        req_stats.queue_start = now
                    elif C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                        req_stats.created_time = simulation_args["created_time"]
                        req_stats.last_event_time = req_stats.created_time
                        # Align with the real queue start timestamp if queue_start is not None. For debugging only.
                        queue_start = simulation_args.get("queue_start")
                        if queue_start is not None:
                            StateManager.set_global_clock(queue_start)
                        req_stats.queue_start = StateManager.get_global_clock()
                        # Register the session for the completion-event hook
                        # (dynamic completion feedback).
                        if simulation_args.get("session") is not None:
                            C_SchedulerHook.REQ_SESSION[req.rid] = (
                                simulation_args["session"]
                            )

            if recv_reqs and C_SchedulerHook.LAST_CPU_TS == 0:
                C_SchedulerHook.LAST_CPU_TS = time.time()
                StateManager.set_global_clock(0)

            return recv_reqs

        def wrapped_process_input_requests(self, recv_reqs, *args, **kwargs):
            """sglang >= 0.5.10 request path.

            Replaces the legacy `recv_requests` interception: requests are
            dispatched through `process_input_requests`, so the OFFLINE
            buffering (collect all generate requests first, release them by
            simulation timestamp) and the per-request statistics recording are
            implemented around the original dispatcher."""
            recv_reqs = list(recv_reqs) if recv_reqs else []

            if C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                recv_reqs = C_SchedulerHook._offline_collect_and_release(
                    self, recv_reqs, original_process_input_requests
                )

            now = time.time()
            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    C_SchedulerHook.record_request_stats(req, now)

            if recv_reqs and C_SchedulerHook.LAST_CPU_TS == 0:
                C_SchedulerHook.LAST_CPU_TS = time.time()
                StateManager.set_global_clock(0)

            if recv_reqs:
                original_process_input_requests(self, recv_reqs, *args, **kwargs)

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            result = original_get_new_batch_prefill(self, *args, **kwargs)
            # sglang >= 0.5.10 returns NextBatchPlan(batch_to_run, running_batch)
            # instead of the ScheduleBatch itself.
            new_batch = getattr(result, "batch_to_run", result)
            now = time.time()
            if new_batch is not None:
                for req in new_batch.reqs:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.final_reused_tokens = req.cached_tokens
                    if req_stats.queue_end == -1:
                        if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                            req_stats.queue_end = now
                        else:
                            req_stats.queue_end = StateManager.get_global_clock()
                    else:
                        # Chunked request
                        pass
                    # Track L2 host_hit_length (set by _prefetch_kvcache -> init_next_round_input)
                    host_hit = getattr(req, 'host_hit_length', 0)
                    if host_hit > 0:
                        req_stats.l2_hit_tokens = host_hit
                        C_SchedulerHook._L2_HIT_TOKENS += host_hit
                        C_SchedulerHook._L2_HIT_COUNT += 1
                    # Log cache state per prefill batch (L1/L2 pool sizes)
                    tc = getattr(self, 'tree_cache', None)
                    l1_evict = l1_total = l2_avail = owt = olb = -1
                    n_host_leaves = 0
                    if tc is not None and hasattr(tc, 'evictable_size_'):
                        cc = getattr(tc, 'cache_controller', None)
                        l2_avail = cc.mem_pool_host.available_size() if cc and hasattr(cc, 'mem_pool_host') else -1
                        l1_evict = tc.evictable_size_
                        l1_total = tc.token_to_kv_pool_allocator.available_size() + l1_evict if hasattr(tc, 'token_to_kv_pool_allocator') else -1
                        owt = len(getattr(tc, 'ongoing_write_through', {}))
                        olb = len(getattr(tc, 'ongoing_load_back', {}))
                        # HIRadixCache only (absent on a plain RadixCache)
                        n_host_leaves = len(getattr(tc, 'evictable_host_leaves', {}))
                    logger.info(
                        f"Cache state: L1_evict={l1_evict} L1_total={l1_total} "
                        f"L2_avail={l2_avail} L2_host_leaves={n_host_leaves} "
                        f"owt={owt} olb={olb}"
                    )
            elif len(self.running_batch.reqs) == 0 and len(self.waiting_queue) > 0:
                # Prefetching
                StateManager.step_global_clock(0.005)
                StateManager.set_current_inference_dur(0.005)
            else:
                if C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE and (
                    len(C_SchedulerHook.FUTURE_QUEUE) != 0
                    and len(self.running_batch.reqs) == 0
                ):
                    next_created_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    StateManager.set_global_clock(next_created_time + 1e-6)
                elif (
                    C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE
                    and len(C_SchedulerHook.FUTURE_QUEUE) == 0
                    and len(C_SchedulerHook.PARKED) != 0
                    and len(self.running_batch.reqs) == 0
                    and len(self.waiting_queue) == 0
                ):
                    # Defensive: everything left is parked but nothing is
                    # running (e.g. a finish event was missed). Force-release
                    # the earliest parked round to avoid a deadlock.
                    session = min(
                        C_SchedulerHook.PARKED,
                        key=lambda s: C_SchedulerHook.PARKED[s][0][0],
                    )
                    enqueue_time, salt, req = C_SchedulerHook.PARKED.pop(session)[0]
                    logger.warning(
                        f"Dynamic feedback: force-releasing parked round of "
                        f"session {session} (no running request)."
                    )
                    StateManager.set_global_clock(enqueue_time + 1e-6)
                    heapq.heappush(
                        C_SchedulerHook.FUTURE_QUEUE, (enqueue_time, salt, req)
                    )
            logger.debug(
                f"Get new batch prefill: global iteration={StateManager.get_iteration()}, "
                f"new batch={new_batch.batch_size() if new_batch is not None else 0}, "
                f"waiting queue={len(self.waiting_queue)}"
            )

            return result

        def wrapped_run_batch(self, *args, **kwargs):
            ret = original_run_batch(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )

            if ret.__class__.__name__ == "GenerationBatchResult":
                hisim_batch = HisimScheduleBatch(reqs=[])
                if batch.forward_mode.is_extend():
                    for req in batch.reqs:
                        hisim_batch.reqs.append(
                            FakeRequest(
                                input_length=C_SchedulerHook.get_extend_input_len(req),
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )
                elif batch.forward_mode.is_decode():
                    for req in batch.reqs:
                        hisim_batch.reqs.append(
                            FakeRequest(
                                input_length=1,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )

                if not hisim_batch.is_empty():
                    StateManager.inc_iteration()
                    predicted_latency = (
                        C_SchedulerHook.INFERENCE_PREDICTOR.predict_infer_time(
                            hisim_batch
                        )
                    )
                    predicted_latency = float(predicted_latency)

                    forward_latency = 0
                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        now = time.time()
                        time.sleep(abs(predicted_latency))
                        now = time.time()
                        forward_latency = now - C_SchedulerHook.LAST_CPU_TS
                        C_SchedulerHook.LAST_CPU_TS = now
                    else:
                        now = time.time()
                        forward_latency = predicted_latency

                    StateManager.set_current_inference_dur(forward_latency)

                C_SchedulerHook.HISIM_BATCH = hisim_batch

            return ret

        def wrapped_process_batch_result(self, *args, **kwargs):
            ret = original_process_batch_result(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            if batch is not None:
                if len(batch.reqs) == 0:
                    return ret

                hicache_l2_load_dur = StateManager.pop_hicache_l2_load_dur()
                hicache_l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()
                hicache_l3_backup_dur = StateManager.pop_hicache_l3_backup_dur()
                current_inference_dur = StateManager.get_current_inference_dur()

                try:
                    if ConfigManager.get_platform_config().enable_l3_direct_to_l1:
                        l3_direct_dur = StateManager.pop_hicache_l3_direct_load_dur()
                        hicache_l2_load_dur = max(hicache_l2_load_dur, l3_direct_dur)
                except Exception:
                    pass
                StateManager.pop_l3_direct_load_count()

                # Save last values for next iteration's timing-aware SSD simulation
                StateManager.set_last_l2_load_dur(hicache_l2_load_dur)
                StateManager.set_last_l2_backup_dur(hicache_l2_backup_dur)

                # L3 write starts at T_N (round start), overlaps with entire round:
                # l2_load + inference + l2_backup. Only overflow delays next round.
                l3_write_overflow = max(
                    0,
                    hicache_l3_backup_dur
                    - (hicache_l2_load_dur + current_inference_dur + hicache_l2_backup_dur),
                )
                StateManager.set_last_l3_write_overflow(l3_write_overflow)

                if C_SchedulerHook.OVERLAP_SCHEDULE:
                    StateManager.step_global_clock(
                        max(
                            hicache_l2_load_dur - StateManager.get_last_inference_dur(),
                            0,
                        )
                    )
                    StateManager.step_global_clock(current_inference_dur)
                    request_response_time = (
                        StateManager.get_global_clock()
                        + hicache_l2_backup_dur
                        + l3_write_overflow
                    )
                else:
                    StateManager.step_global_clock(
                        hicache_l2_load_dur
                        + current_inference_dur
                        + hicache_l2_backup_dur
                        + l3_write_overflow
                    )
                    request_response_time = StateManager.get_global_clock()
                # Request statistics
                for req in batch.reqs:
                    if not C_SchedulerHook.is_chunked_req(self, req):
                        req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                        req_stats.gen_token_latencies.append(
                            request_response_time
                            - req_stats.last_event_time  # queue duration
                        )
                        req_stats.last_event_time = request_response_time
                    else:
                        # Chunked request: nothing to do
                        pass
                # Dynamic completion feedback: when a session round finishes,
                # release its parked successor rounds with the accumulated
                # virtual delay (actual finish - recorded finish) applied.
                if C_SchedulerHook.SESSION_STATE:
                    for req in batch.reqs:
                        finished = getattr(req, "finished", False) or getattr(
                            req, "finished_status", None
                        )
                        if not finished:
                            continue
                        session = C_SchedulerHook.REQ_SESSION.get(req.rid)
                        if session is None:
                            continue
                        st = C_SchedulerHook.SESSION_STATE.get(session)
                        if st is None or st["finished"]:
                            continue
                        st["finished"] = True
                        st["finish_vtime"] = request_response_time
                        parked = C_SchedulerHook.PARKED.pop(session, [])
                        if parked:
                            shift = max(
                                0.0,
                                st["finish_vtime"] - st["recorded_finish"],
                            )
                            for enqueue_time, salt, preq in parked:
                                heapq.heappush(
                                    C_SchedulerHook.FUTURE_QUEUE,
                                    (enqueue_time + shift, salt, preq),
                                )
                # Iteration statistics
                C_SchedulerHook.ITERATION_STATS.append(
                    {
                        "requests": C_SchedulerHook.HISIM_BATCH.request_info(),
                        "forward_latency": current_inference_dur,
                        "l2_load_latency": hicache_l2_load_dur,
                        "l2_backup_latency": hicache_l2_backup_dur,
                        "l3_backup_latency": hicache_l3_backup_dur,
                    }
                )
            C_SchedulerHook.LAST_CPU_TS = time.time()
            return ret

        def wrapped_profile(self, req, *args, **kwargs):
            return C_SchedulerHook.dump_profile_results()

        target.event_loop_overlap = override_event_loop_overlap
        target.__init__ = wrapped_init
        target.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_batch_result
        if legacy_request_path:
            # sglang <= 0.5.9: requests are received by Scheduler.recv_requests
            # and profile requests are handled by Scheduler.profile.
            target.recv_requests = wrapped_recv_requests
            target.profile = wrapped_profile
        else:
            # sglang >= 0.5.10: requests are dispatched via
            # Scheduler.process_input_requests; profile requests are handled by
            # SchedulerProfilerManager (hooked by C_SchedulerProfilerManagerHook).
            target.process_input_requests = wrapped_process_input_requests


class C_SchedulerProfilerManagerHook(BaseHook):
    """sglang >= 0.5.10 moved profile handling from `Scheduler.profile` to
    `SchedulerProfilerManager._profile`; hook it to dump the simulation
    results (the legacy path keeps hooking `Scheduler.profile`)."""

    HOOK_CLASS_NAME = "SchedulerProfilerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler_components.profiler_manager"

    @classmethod
    def hook(cls, target):
        def wrapped_profile(self, recv_req, *args, **kwargs):
            return C_SchedulerHook.dump_profile_results()

        target._profile = wrapped_profile
