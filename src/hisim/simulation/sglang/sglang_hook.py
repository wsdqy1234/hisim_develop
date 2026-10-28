import torch
import time
import math
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
from hisim.simulation import kda_obs
from hisim.time_predictor import (

    InferTimePredictor,
    FakeRequest,
    ScheduleBatch as HisimScheduleBatch,
)
from hisim.simulation.sglang.sglang_mock_class import (
    MockReqToTokenPool,
    MockTokenToKVPool,
    MockTokenToKVPoolAllocator,
    MockPagedTokenToKVPoolAllocator,
    MockTokenToKVPoolHost,
    MockMambaStatePoolHost,
    MockHiCacheStorage,
    MockKVIndexTranslator,
)
from hisim.simulation.utils import (
    calc_metrics,
    estimate_kv_cache_pool_capacity,
)
from hisim.simulation.sglang.version import VersionDispatcher


logger = get_logger("hisim")


def req_mamba_get(req, name, default=None):
    """Read a mamba attribute from req.kv (sglang >= 0.5.18) or bare req (0.5.9).

    sglang 0.5.18 moved ``mamba_pool_idx`` / ``mamba_ping_pong_track_buffer`` /
    ``mamba_last_track_idx`` / ``mamba_next_track_idx`` / ``mamba_last_track_seqlen``
    / ``mamba_cow_src_index`` onto ``req.kv``. ``mamba_branching_seqlen`` stays a
    bare Req attribute.
    """
    _kv = getattr(req, "kv", None)
    if _kv is not None and hasattr(_kv, name):
        return getattr(_kv, name)
    return getattr(req, name, default)


def req_mamba_set(req, name, value):
    """Write a mamba attribute to req.kv (0.5.18+) or bare req (0.5.9)."""
    _kv = getattr(req, "kv", None)
    if _kv is not None and hasattr(_kv, name):
        setattr(_kv, name, value)
    else:
        setattr(req, name, value)


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

            # sglang >= 0.5.19: ModelRunner.preloaded_weights_bytes reads
            # self.loader (set by the real weight-loading path that this hook
            # replaces). Provide a stub loader.
            class _StubModelLoader:
                preloaded_weights_bytes = 0

            self.loader = _StubModelLoader()

            # Attributes that may not be set by the hook but are expected by sglang
            self.use_ngram_embedding = False
            self.is_hybrid_swa = False
            self.is_draft_worker = False

            # K3 is a hybrid MLA+Mamba model. Set mamba2_config to non-None
            # so sglang's scheduler sets is_hybrid_ssm=True during init,
            # causing UnifiedRadixCache to include ComponentType.MAMBA in
            # tree_components (two-pool hierarchical cache: separate L2 host
            # pools for MLA KV and Mamba/KDA state).
            self.mamba2_config = True
            self.hybrid_gdn_config = None

            # ModelRunner-level ngram_embedding_manager. ForwardBatch.init_new
            # reads model_runner.ngram_embedding_manager.enabled; a None here
            # raises AttributeError once run_batch actually executes.
            class _NoopNgramEmbeddingManager:
                enabled = False
            self.ngram_embedding_manager = _NoopNgramEmbeddingManager()

            # sglang >= 0.5.19: ForwardBatch/scheduler read the KV index
            # translator and attention backend strings; real translation only
            # matters for attention kernels, which are intercepted.
            self.kv_index_translator = MockKVIndexTranslator()
            self.prefill_attention_backend_str = "torch_native"
            self.decode_attention_backend_str = "torch_native"


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

            # In unified memory mode, let sglang's kv_cache_configurator
            # create the real UnifiedHybridReqToTokenPool + UnifiedKVPool.
            # Only create mock pools for Two Pools (hierarchical/plain) mode.
            _is_unified = getattr(self.server_args, 'enable_unified_memory', False)
            if _is_unified:
                logger.info("Unified memory mode: skipping mock pool creation, "
                            "letting sglang configurator create real unified pools")
                # Set attributes needed by init_kv_cache_configurator
                self.pre_model_load_memory = 0
                self.sliding_window_size = None
                self._proc_scaled_bwd_llk_shm = None
                self._proc_scaled_bwd_glk_shm = None
                self.is_hybrid_swa_compress = False
                self.use_mla_backend = False  # Keep False: MLA kernel paths crash on NPU simulation
                # HiSim MLA Capacity Simulation: override head_num/head_dim so
                # MHASubPoolSpec entry_bytes matches real MLA (5760 B/token).
                # MHA: entry_bytes = 5 × head_num × head_dim × 2(K+V) × 2(bf16) = 5760
                # With head_dim=288 (set by launch_server patch): head_num = 1
                # Override get_num_kv_heads to return 1 for correct sizing.
                _orig_get_nkh = type(self.model_config).get_num_kv_heads
                def _hisim_get_nkh(self_mc, tensor_parallel_size, dcp_size=1):
                    if getattr(self_mc, 'kv_lora_rank', None) is not None:
                        return 1  # HiSim MLA capacity: 1 head × 288 dim = 288 (MLA = 576/2)
                    return _orig_get_nkh(self_mc, tensor_parallel_size, dcp_size)
                type(self.model_config).get_num_kv_heads = _hisim_get_nkh
                logger.info("HiSim MLA capacity: head_num=1, head_dim=288 → entry_bytes=5760")
                from sglang.srt.model_executor.model_runner_components.layer_setup import ModelLayerInfo
                self.layer_info = ModelLayerInfo(
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    num_effective_layers=self.num_effective_layers,
                )
                self.forward_stream = None
                self.memory_pool_config = None
                self.draft_model_idx = None
                self.post_capture_kv_active = False
                self.kv_cache_dtype_str = (
                    getattr(self.server_args, "kv_cache_data_type", None) or "auto"
                )
                self.spec_aux_config = None
                self.spec_algorithm = None
                try:
                    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
                    self.spec_algorithm = SpeculativeAlgorithm.from_string(
                        getattr(self.server_args, "spec_algorithm", None) or "none"
                    )
                except Exception:
                    self.spec_algorithm = None
                # Resolve spec_aux_config (even with no spec, returns a valid config)
                try:
                    from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
                        resolve_spec_aux_hidden_state_config,
                    )
                    self.spec_aux_config = resolve_spec_aux_hidden_state_config(
                        server_args=self.server_args,
                        model_config=self.model_config,
                        spec_algorithm=self.spec_algorithm,
                        is_draft_worker=self.is_draft_worker,
                    )
                except Exception:
                    from types import SimpleNamespace as _SSNS
                    self.spec_aux_config = _SSNS(
                        eagle_use_aux_hidden_state=False,
                        eagle_aux_hidden_state_layer_ids=[],
                        dflash_use_aux_hidden_state=False,
                        dflash_target_layer_ids=[],
                    )
                # Initialize the configurator and configure real unified pools
                self.init_kv_cache_configurator()
                # Ensure use_mla_backend is True on the configurator (K3 uses MLA).
                # The configurator may not inherit this from ModelRunner, causing
                # init_unified_mamba_pools to use MHASubPoolSpec instead of MLA.
                self.kv_cache_configurator.use_mla_backend = True
                result = self.kv_cache_configurator.configure(
                    pre_model_load_memory=self.pre_model_load_memory
                )
                self.max_total_num_tokens = result.max_total_num_tokens
                self.max_running_requests = result.max_running_requests
                self.req_to_token_pool = result.req_to_token_pool
                # Override max_running_requests to force Mamba pool overflow.
                # The configurator caps max_running_requests to
                # max_mamba_cache_size/5, which prevents overflow by design.
                # For pressure testing, we need more concurrent requests than
                # the configurator allows, so the Mamba sub-pool overflows and
                # triggers compact/flush_peer in MultiEndedAllocator.
                _user_mrr = getattr(self.server_args, 'max_running_requests', None)
                if _user_mrr is not None and _user_mrr > self.max_running_requests:
                    logger.info(
                        f"Overriding max_running_requests: {self.max_running_requests}"
                        f" -> {_user_mrr} (user-specified, for pressure testing)"
                    )
                    self.max_running_requests = _user_mrr
                    # Resize req_to_token_pool to support more concurrent requests.
                    # ReqToTokenPool has _alloc_size = size + 1 (padding row at 0),
                    # req_to_token tensor, req_generation tensor, and free_slots list.
                    # HybridReqToTokenPool also has req_index_to_mamba_index_mapping
                    # and req_index_to_mamba_ping_pong_track_buffer_mapping.
                    import torch
                    _pool = self.req_to_token_pool
                    _new_alloc = _user_mrr + 1
                    _pool.req_to_token = torch.zeros(
                        (_new_alloc, _pool.max_context_len),
                        dtype=_pool.req_to_token.dtype,
                        device=_pool.req_to_token.device,
                    )
                    _pool.req_generation = torch.zeros(
                        _new_alloc, dtype=torch.int64,
                        device=_pool.req_generation.device,
                    )
                    _pool._alloc_size = _new_alloc
                    _pool.size = _user_mrr
                    _pool.free_slots = list(range(1, _new_alloc))
                    # Resize HybridReqToTokenPool mamba mapping tensors
                    if hasattr(_pool, 'req_index_to_mamba_index_mapping'):
                        _pool.req_index_to_mamba_index_mapping = torch.zeros(
                            _new_alloc, dtype=torch.int32, device=_pool.device,
                        )
                    if hasattr(_pool, 'req_index_to_mamba_ping_pong_track_buffer_mapping'):
                        _pp_size = _pool.mamba_ping_pong_track_buffer_size
                        _pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
                            (_new_alloc, _pp_size), dtype=torch.int64, device=_pool.device,
                        )
                self.token_to_kv_pool = result.token_to_kv_pool
                self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
                self.memory_pool_config = result.memory_pool_config

                # HiSim Mamba State Mock: Replace UnifiedMambaPool with a catch-all
                # mock (like Two Pools' _MockMambaPool). The forward pass is mocked,
                # so actual Mamba state data is irrelevant. But mamba_component.py
                # calls mamba_pool methods (copy_from, get_cpu_copy, replayssm_*)
                # that trigger NPU kernels and crash in CPU simulation.
                # Keep mamba_allocator (MultiEndedAllocator) fully functional —
                # alloc/free/compact/flush_peer still track the dynamic KV/Mamba split.
                _real_mamba_pool = self.req_to_token_pool.mamba_pool
                _real_allocator = self.req_to_token_pool.mamba_allocator

                class _HiSimMambaPoolMock:
                    """Catch-all mock for MambaPool data operations.
                    Passes through size/attributes but no-ops all methods.
                    mamba_allocator (MultiEndedAllocator) remains real."""
                    def __init__(self, real_pool):
                        self._real = real_pool
                        self.size = getattr(real_pool, 'size', 0)
                        self.num_mamba_layers = getattr(real_pool, 'num_mamba_layers', 17)
                        self.layer_transfer_counter = None
                        self.replayssm_write_pos = None
                        self.replayssm_cache_base = None
                        self.replayssm_is_flush = None
                        # Provide mamba_cache with tiny CPU tensors (for attribute reads)
                        import torch as _t
                        _n = self.num_mamba_layers
                        class _MockCache:
                            conv = [_t.zeros(_n, 1, 1, 4, dtype=_t.float32)]
                            temporal = _t.zeros(_n, 1, 1, 4, dtype=_t.float32)
                        self.mamba_cache = _MockCache()
                    def copy_from(self, src, dst):
                        pass  # no-op: data doesn't matter (forward mocked)
                    def get_cpu_copy(self, indices):
                        return None
                    def load_cpu_copy(self, cache_cpu, indices):
                        pass
                    def __getattr__(self, name):
                        return None  # catch-all: any other method → None

                self.req_to_token_pool.mamba_pool = _HiSimMambaPoolMock(_real_mamba_pool)
                # mamba_allocator stays real (MultiEndedAllocator)
                logger.info(f"Unified pools created: req_to_token_pool={type(self.req_to_token_pool).__name__}, "
                            f"token_to_kv_pool={type(self.token_to_kv_pool).__name__}, "
                            f"max_total_tokens={self.max_total_num_tokens}, "
                            f"mamba_pool=mocked (NPU bypass), mamba_allocator=real")
            else:
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
            self.attn_backend = None
            self.graph_mem_usage = 0
            self.weight_load_mem_usage = 10

            # Attributes expected by sglang 0.5.18's post-init code paths
            # (cuda graph setup, canary, etc.) — HiSim skips real init so these
            # need to be set to safe defaults
            self.canary_manager = None
            self.decode_cuda_graph_runner = None
            self.prefill_cuda_graph_runner = None
            self.torch_compile_runner = None
            self.enable_torch_compile = False
            self.is_embedding = False
            self.enable_two_batch_overlap = False
            self.mtp_layer_sync_manager = None
            self.remote_instance_transfer_engine = None
            self.ngram_embedding_manager = None
            self.encoder_decoder_manager = None
            self.prefill_aware_swa = False
            self.mtp_layer_ids = []
            self.enable_prefill_cp = False
            self.attn_cp_size = 1

            # Attributes expected by sglang 0.5.18's init_kv_cache_configurator
            # and other post-init methods (HiSim skips original_initialize, so
            # we must set these attributes manually)
            self.kv_cache_dtype_str = (
                getattr(self.server_args, "kv_cache_data_type", None) or "auto"
            )
            self.spec_algorithm = None
            try:
                from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
                self.spec_algorithm = SpeculativeAlgorithm.from_string(
                    getattr(self.server_args, "spec_algorithm", None) or "none"
                )
            except Exception:
                self.spec_algorithm = None
            self.sliding_window_size = getattr(
                self.model_config, "sliding_window", None
            )
            self.pp_group = None  # no distributed for CPU simulation
            self.layer_info = type("LayerInfo", (), {
                "start_layer": self.start_layer,
                "end_layer": self.end_layer,
                "num_effective_layers": self.num_effective_layers,
            })()

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
            "forward", ["0.5.7", "0.5.8", "0.5.8.post1", "0.5.9"], wrapped_forward_v2
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

        # sglang >= 0.5.10: the scheduler separately initializes attention
        # backends and CUDA graphs after ModelRunner init. Both are irrelevant
        # for the simulation (the forward pass is intercepted), and building
        # real backends would require real devices — no-op them.
        if hasattr(target, "init_attention_backends"):
            target.init_attention_backends = lambda self, *args, **kwargs: None
        if hasattr(target, "init_cuda_graphs"):
            target.init_cuda_graphs = lambda self, *args, **kwargs: None

        # Mock alloc_memory_pool: HiSim already set up mock pools in override_initialize.
        # The real alloc_memory_pool triggers kv_cache_configurator which needs real GPU/NPU.
        # Just return the already-set pools as the "result".
        def override_alloc_memory_pool(self, memory_pool_config=None):
            from collections import namedtuple
            PoolResult = namedtuple("PoolResult", [
                "max_total_num_tokens", "max_running_requests",
                "req_to_token_pool", "token_to_kv_pool",
                "token_to_kv_pool_allocator",
            ])
            return PoolResult(
                max_total_num_tokens=self.max_total_num_tokens,
                max_running_requests=getattr(self, "max_running_requests", 2048),
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                token_to_kv_pool_allocator=getattr(self, "token_to_kv_pool_allocator", None),
            )
        target.alloc_memory_pool = override_alloc_memory_pool

        # Mock init_cuda_graphs: not needed for CPU simulation (forward is mocked)
        target.init_cuda_graphs = lambda self, *args, **kwargs: None
        target.sample = wrapped_sample
        target.compute_logprobs_only = wrapped_compute_logprobs_only


class C_HiCacheController(BaseHook):
    HOOK_CLASS_NAME = "HiCacheController"
    HOOK_MODULE_NAME = "sglang.srt.managers.cache_controller"

    KV_CACHE_BYTES: int = None
    KV_CACHE_BYTES_FULL: int = None
    DISK_READ_BANDWIDTH_BYTES: float = None
    DISK_WRITE_BANDWIDTH_BYTES: float = None

    # KDA/Mamba state pool parameters (lazily initialized)
    MAMBA_SLOT_BYTES: int = None          # per-slot bytes (14.45MB for K3-23L)
    MAMBA_POOL_SIZE: int = None           # max slots in L1 (70 or 165)
    MAMBA_SLOTS_PER_REQ: int = 5          # active slots per request (ping-pong + spec)
    MAMBA_TRACK_INTERVAL: int = 256      # sglang mamba_track_interval (track every N tokens)
    _kda_pool = None                      # MockMambaStatePool instance
    _kda_host_pool = None                 # MockMambaStatePoolHost instance
    _kda_slots_in_use: int = 0            # track total slots in use
    _kda_l3_pending_writes: list = None   # slot hashes pending L3 write
    _kda_l3_write_count: int = 0          # snapshot of pending count for L3 timing
    _kda_stats = None

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

    # ==================== KDA/Mamba State Methods ====================

    @classmethod
    def _init_kda_pools(cls):
        """Lazily initialize KDA state pool (L1) and host pool (L2)."""
        if cls._kda_pool is not None:
            return
        try:
            from hisim.simulation.manager import ConfigManager
            from hisim.simulation.utils import calc_mamba_state_bytes
            from hisim.simulation.sglang.sglang_mock_class import (
                MockMambaStatePool, MockMambaStatePoolHost,
            )

            sched = ConfigManager.get_scheduler_config()
            model = ConfigManager.get_model_info()
            platform = ConfigManager.get_platform_config()

            cls.MAMBA_SLOT_BYTES = calc_mamba_state_bytes(
                model, sched.tp_size, sched.pp_size
            )
            cls.MAMBA_POOL_SIZE = sched.max_mamba_cache_size or 70
            cls.MAMBA_SLOTS_PER_REQ = sched.mamba_slots_per_request
            # Read mamba_track_interval from sglang server args (default 256)
            # Used to calculate checkpoint slot count: ceil(seq_len / track_interval)
            import os as _os
            _track_interval = _os.environ.get("SGLANG_MAMBA_TRACK_INTERVAL", "256")
            try:
                cls.MAMBA_TRACK_INTERVAL = int(_track_interval)
            except ValueError:
                cls.MAMBA_TRACK_INTERVAL = 256

            # L1 pool (HBM slots)
            cls._kda_pool = MockMambaStatePool(
                size=cls.MAMBA_POOL_SIZE,
                dtype=torch.float32,
                layer_num=model.num_linear_attention,
                device="cpu",  # simulation, no real device
                slot_bytes=cls.MAMBA_SLOT_BYTES,
            )
            # L2 pool (DRAM backup) — size = 2x L1 (can hold evicted + working set)
            # With a unified L2 DRAM pool the shared budget replaces this
            # per-object cap (checkpoint admission charges the KDA view).
            try:
                from hisim.simulation.l2_dram_pool import get_l2_dram_pool as _get_l2up
                _l2up = _get_l2up()
            except Exception:
                _l2up = None
            _kda_l2_size = _l2up.kda_capacity_slots if _l2up is not None else cls.MAMBA_POOL_SIZE * 2
            cls._kda_host_pool = MockMambaStatePoolHost(
                size=_kda_l2_size,
                slot_bytes=cls.MAMBA_SLOT_BYTES,
            )
            cls._kda_l3_pending_writes = []
            cls._kda_stats = {"backup": 0, "load_back_l2": 0, "load_back_l3": 0, "l3_hit": 0}
            if _l2up is not None:
                cls._kda_host_pool.attach_unified(_l2up)
                cls._kda_host_pool.set_evict_hook(
                    lambda h: cls._kda_l3_pending_writes.append(h)
                )
                _l2up.register_kda_pool(cls._kda_host_pool)
                logger.info(
                    f"KDA host pool bound to L2 unified DRAM pool "
                    f"({_l2up.mode}, {_kda_l2_size} max slots)"
                )
            logger.info(
                f"KDA pools initialized: L1={cls.MAMBA_POOL_SIZE} slots "
                f"({cls.MAMBA_SLOT_BYTES/1e6:.2f}MB/slot), "
                f"L2={_kda_l2_size} slots"
            )
        except Exception as e:
            logger.warning(f"KDA pool init failed (non-hybrid model?): {e}")
            cls._kda_pool = False  # mark as not-applicable

    @classmethod
    def handle_kda_backup_operation(cls, self):
        """KDA state backup: when L1 MambaPool is full, evict LRU slots to L2.
        Called from wrapped_check_hicache_events after MLA backup.

        Slot count per request = active_slots + ceil(max_seq_len / track_interval).
        Uses content-addressed hash: hash(req_id + token_offset) for L3 dedup.
        """
        if cls._kda_pool is None:
            cls._init_kda_pools()
        if not cls._kda_pool:
            return

        # Calculate actual slot usage from running batch (not fixed estimate)
        running_batch = getattr(self, "running_batch", None)
        if running_batch is not None and hasattr(running_batch, "reqs"):
            slots_in_use = 0
            for req in running_batch.reqs:
                # Active slots (ping-pong + speculative) + checkpoint slots
                seq_len = getattr(req, "seq_len", 0) or getattr(req, "input_length", 0)
                num_checkpoints = max(1, seq_len // cls.MAMBA_TRACK_INTERVAL)
                slots_in_use += cls.MAMBA_SLOTS_PER_REQ + num_checkpoints
        else:
            # Fallback: use num_running_reqs * estimated slots
            active_reqs = getattr(self, "num_running_reqs", 0)
            slots_in_use = active_reqs * (cls.MAMBA_SLOTS_PER_REQ + 2)

        cls._kda_slots_in_use = slots_in_use
        excess = slots_in_use - cls.MAMBA_POOL_SIZE
        if excess <= 0:
            return

        # Backup excess slots to L2 (each slot = MAMBA_SLOT_BYTES D2H)
        import hashlib
        running_reqs = getattr(running_batch, "reqs", []) if running_batch else []
        for i in range(excess):
            slot_idx = cls._kda_pool.alloc_slot()  # triggers LRU evict if full
            # Content-addressed hash: deterministic based on slot_idx + iteration
            # so same logical checkpoint gets same hash → L3 dedup works
            req_id = running_reqs[i % len(running_reqs)].rid if running_reqs else f"req_{i}"
            token_offset = (slot_idx * cls.MAMBA_TRACK_INTERVAL) if running_reqs else slot_idx
            slot_hash = hashlib.md5(
                f"kda_{req_id}_{token_offset}".encode()
            ).hexdigest()
            cls._kda_host_pool.backup_from_device(slot_idx, slot_hash,
                                                   depth=token_offset)
            cls._kda_l3_pending_writes.append(slot_hash)
            cls._kda_stats["backup"] += 1

        # If L2 is full, evict to L3
        evicted = cls._kda_host_pool.evict()
        while evicted is not None:
            cls._kda_l3_pending_writes.append(evicted)
            evicted = cls._kda_host_pool.evict()

    @classmethod
    def handle_kda_prefetch_operation(cls, self):
        """KDA state load_back: when a prefix match finds evicted checkpoint in L2/L3.
        Called from wrapped_check_hicache_events after MLA prefetch.

        B2 fix: snapshot pending count BEFORE clearing, pass to L3 write timing.
        B3 fix: for L3 hits, do SSD→DRAM (D2H) then DRAM→HBM (H2D) with timing.
        """
        if not cls._kda_pool:
            return
        if not cls._kda_l3_pending_writes:
            return

        # B2: snapshot count for L3 write timing (before prefetch clears list)
        cls._kda_l3_write_count = len(cls._kda_l3_pending_writes)

        # Write to L3 (kvcm) first — these slots are being backed up
        try:
            slot_hashes = list(cls._kda_l3_pending_writes)
            if hasattr(self, 'storage_backend') and self.storage_backend:
                self.storage_backend.batch_set_mamba(slot_hashes)
        except Exception as e:
            logger.debug(f"KDA L3 write skipped: {e}")

        # Query L3 for KDA checkpoint lookups
        try:
            slot_hashes = list(cls._kda_l3_pending_writes)
            l3_hits = self.storage_backend.batch_exists_mamba(slot_hashes)
            l3_hit_count = sum(l3_hits.values())
            cls._kda_stats["l3_hit"] += l3_hit_count

            for slot_hash, hit in l3_hits.items():
                if hit:
                    # B3: L3 hit → SSD→DRAM (D2H) then DRAM→HBM (H2D)
                    # First: SSD read → DRAM (timed as kda_l3_load_dur)
                    cls._kda_host_pool.backup_from_device(-1, slot_hash, from_l3=True)
                    # Then: DRAM → HBM (H2D, COW)
                    cls._kda_host_pool.load_to_device(slot_hash)
                    cls._kda_stats["load_back_l3"] += 1
                elif cls._kda_host_pool.has(slot_hash):
                    # Found in L2: load_back H2D
                    cls._kda_host_pool.load_to_device(slot_hash)
                    cls._kda_stats["load_back_l2"] += 1
        except Exception as e:
            logger.debug(f"KDA prefetch skipped: {e}")
        finally:
            cls._kda_l3_pending_writes.clear()

    @classmethod
    def submit_kda_l3_write_timing(cls):
        """KDA L3 write timing: SSD write for KDA slots (MAMBA_SLOT_BYTES/slot).
        Called after MLA submit_l3_write_timing.

        B2 fix: use _kda_l3_write_count (snapshot taken before prefetch clear)
        instead of reading len(_kda_l3_pending_writes) which may already be 0.
        """
        if not cls._kda_pool or cls._kda_l3_write_count == 0:
            return

        write_arrival = StateManager.get_global_clock()
        total_write_bytes = cls._kda_l3_write_count * cls.MAMBA_SLOT_BYTES

        sim = cls._get_ssd_simulator()
        if sim is not None and hasattr(sim, 'submit_write'):
            _, lat, _, _ = sim.submit_write(
                write_arrival, total_write_bytes,
                cls.MAMBA_SLOT_BYTES,
                is_sequential=True,
            )
            write_time = lat
        elif sim is not None:
            write_time = sim.predict_write_time(
                total_write_bytes, cls.MAMBA_SLOT_BYTES, is_sequential=True,
            )
        else:
            if cls.DISK_WRITE_BANDWIDTH_BYTES is None:
                cls.DISK_WRITE_BANDWIDTH_BYTES = (
                    ConfigManager.get_platform_config().disk_write_bandwidth
                )
            bw = cls.DISK_WRITE_BANDWIDTH_BYTES or 8e9
            write_time = total_write_bytes / bw

        StateManager.inc_hicache_kda_l3_backup_dur(write_time)
        # Reset snapshot
        cls._kda_l3_write_count = 0

    @staticmethod
    def get_kda_stats() -> dict:
        """Return KDA state simulation statistics."""
        if C_HiCacheController._kda_stats is None:
            return {}
        stats = dict(C_HiCacheController._kda_stats)
        if C_HiCacheController._kda_pool:
            stats.update(C_HiCacheController._kda_pool.get_stats())
        return stats

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
            data = [
                self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
                for i in range(len(hash_values))
            ]
            return self.storage_backend.batch_set(hash_values, data, extra_info)

        target.prefetch_thread_func = override_prefetch_thread_func
        target.backup_thread_func = override_backup_thread_func
        target.handle_backup_operation = handle_backup_operation
        target.handle_prefetch_operation = handle_prefetch_operation
        target.submit_l3_write_timing = submit_l3_write_timing
        target._generic_page_set = override_generic_page_set


def _setup_l2_dram_pool(server_args, params):
    """Create the shared L2 DRAM pool (KDA/MLA one budget) when enabled.

    Enabled by ``HISIM_L2_ONE_POOL=1`` (dynamic sharing) or by setting
    ``HISIM_L2_TOTAL_GB`` (same total budget for the static Two Pools
    baseline).  Returns ``None`` when disabled, which keeps the legacy
    fixed-size host pools byte-for-byte unchanged.
    """
    try:
        from hisim.simulation.l2_dram_pool import configure_l2_dram_pool
        from hisim.simulation.sglang.sglang_mock_class import (
            calc_kv_host_size_per_token,
        )
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning(f"L2 unified pool unavailable: {exc}")
        return None

    try:
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        token_bytes = calc_kv_host_size_per_token(kvcache)
    except Exception as exc:
        logger.warning(f"L2 unified pool: cannot size MLA token ({exc}); disabled")
        return None

    slot_bytes = 14653440
    try:
        from hisim.simulation.utils import calc_mamba_state_bytes
        _model = ConfigManager.get_model_info()
        _sched = ConfigManager.get_scheduler_config()
        slot_bytes = int(calc_mamba_state_bytes(_model, _sched.tp_size, _sched.pp_size))
    except Exception as exc:
        logger.debug(f"L2 unified pool: using default KDA slot bytes ({exc})")

    hicache_gb = float(getattr(server_args, "hicache_size", 0) or 0)
    pool = configure_l2_dram_pool(hicache_gb, token_bytes, slot_bytes)
    if pool is not None:
        logger.info(
            "HiSim L2 unified DRAM pool: mode=%s total=%.2fGB page=%dB mla_tokens=%d kda_slots=%d",
            pool.mode,
            pool.total_bytes / 1e9,
            pool.page_bytes,
            pool.mla_capacity_tokens,
            pool.kda_capacity_slots,
        )
    return pool


def _bind_l2_host_group(cache, l2_pool, kv_host, kda_host):
    """Point the native HostPoolGroup at the budgeted host pools.

    Without this the controller would keep allocating from the native
    fixed-size host pools (Two Pools path) and the shared budget would only
    affect the timing mocks.
    """
    from sglang.srt.mem_cache.hicache_storage import PoolName

    group = getattr(cache, "host_pool_group", None)
    cc = getattr(cache, "cache_controller", None)
    if group is None and cc is not None:
        group = getattr(cc, "mem_pool_host", None)
    if group is None or not hasattr(group, "entry_map"):
        logger.warning("L2 unified pool: no HostPoolGroup; budget on mocks only")
    else:
        kv_entry = group.entry_map.get(PoolName.KV)
        if kv_entry is not None:
            kv_entry.host_pool = kv_host
        kda_entry = group.entry_map.get(PoolName.MAMBA)
        if kda_entry is not None and kda_host is not None:
            kda_entry.host_pool = kda_host
        try:
            group.size = kv_host.size
            group.logical_size = kv_host.logical_size
            group.page_size = kv_host.page_size
            group.layout = getattr(kv_host, "layout", group.layout)
            group.device = getattr(kv_host, "device", group.device)
            group._refresh_transfer_capabilities()
        except Exception as exc:
            logger.debug(f"L2 unified pool: group refresh failed: {exc}")
        if cc is not None and hasattr(cc, "storage_host_pool"):
            cc.storage_host_pool = kv_host

    if kv_host is not None:
        kv_host.attach_unified(l2_pool)
        l2_pool.bind_mla_pool(kv_host)
    if kda_host is not None:
        kda_host.attach_unified(l2_pool)
        l2_pool.register_kda_pool(kda_host)

    # [P2D2] KDA -> MLA reclaim uses the native host-eviction path.
    # The enable_storage guard was overly conservative: evict_host is a
    # radix-tree operation that works regardless of L3 storage.  In HiSim
    # Mock, L3 writes happen in the backup path (_mock_start_writing), so
    # evicted L2 entries already have L3 copies.  Removing the guard allows
    # MLA→KDA cross-type borrow in One Pool mode without enabling native
    # Storage.
    if hasattr(cache, "evict_host"):
        from sglang.srt.mem_cache.unified_cache.component_type import (
            ComponentType as _P2D2_CT,
        )

        # [P2D2] Wrap evict_host to count ALL MLA evictions (for the
        # evict_mla_for_mla counter = total - evict_mla_for_kda).
        _orig_evict_host = cache.evict_host
        _l2_ref = l2_pool  # capture for counter update

        def _wrapped_evict_host(*args, **kwargs):
            result = _orig_evict_host(*args, **kwargs)
            try:
                # component_type is the 2nd positional arg or kwarg
                # If not specified, the original default was used (FULL for K3)
                ct = args[1] if len(args) > 1 else kwargs.get("component_type")
                if result and (ct is None or ct == _P2D2_CT.FULL):
                    _l2_ref._evict_mla_total_tokens += int(result)
            except Exception:
                pass
            return result

        cache.evict_host = _wrapped_evict_host

        def _mla_evict(n_tokens, _c=cache):
            try:
                return int(_c.evict_host(int(n_tokens), _P2D2_CT.FULL) or 0)
            except Exception as exc:
                logger.debug(f"L2 unified pool: MLA reclaim failed: {exc}")
                return 0

        l2_pool.set_mla_evictor(_mla_evict)
        logger.info(
            "[P2D2] MLA evictor registered (enable_storage=%s, evict_host wrapped for tracking)",
            getattr(cache, "enable_storage", False),
        )
    else:
        logger.warning("[P2D2] cache.evict_host not found — MLA→KDA borrow disabled")


class C_HiRadixCacheHook(BaseHook):
    HOOK_CLASS_NAME = "UnifiedRadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.unified_radix_cache"

    @classmethod
    def hook(cls, target):
        original_init_hicache = target.init_hicache
        original_check_hicache_events = target.check_hicache_events
        original_reset = target.reset
        original_count_ready_acks = target._count_ready_acks

        def wrapped_reset(self):
            if hasattr(self, "cache_controller") and self.cache_controller is not None:
                self.cache_controller.handle_backup_operation()
                self.cache_controller.submit_l3_write_timing()
            original_reset(self)

        def wrapped_init_hicache(self, server_args, params):
            # Try the real init_hicache first. If it fails (because mocks
            # don't have all the real mamba internals like mamba_cache.conv),
            # fall back to manual setup of mock host pools + cache_controller.
            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _HA_CT
            from sglang.srt.managers.cache_controller import HiCacheAck as _HA_CKACK

            def _apply_mock_start_overrides(cache, full_host, mamba_host=None):
                # Override start_writing (D2H backup) & start_loading (H2D
                # load_back) to bypass CUDA l2_transfer_engine, record timing
                # via MockTokenToKVPoolHost / MockMambaStatePoolHost, set
                # host_value on BOTH FULL (MLA) and MAMBA (KDA) components,
                # and create fake acks (torch.Event so query/synchronize work).
                # Also directly trigger L3 (SSD) writes/reads to bypass the
                # broken ack processing chain (write_backup_storage never called).
                _diag_w = {"calls": 0, "mamba_pt": 0, "full_only": 0, "l3_writes": 0, "kda_l3_writes": 0, "kda_l2_admissions": 0}
                _diag_l = {"calls": 0, "mamba_pt": 0, "full_only": 0, "l3_hits": 0, "kda_l3_hits": 0, "kda_l3_admission_reject": 0, "kda_l2_load_fail": 0, "kda_l3_stage_fail": 0}

                # Get storage backend for L3 writes/reads
                # Always use MockHiCacheStorage — the real storage_backend may be
                # HiCacheFile (if C_StorageBackendFactory hook didn't fire) which
                # lacks batch_set/batch_exists.
                # [P0-2] HISIM_DISABLE_L3=1 forces _storage=None for true L1-only
                # [P2C.3.1-FIX] "0" is a non-empty string and is truthy in Python,
                # so the old check `if os.environ.get("HISIM_DISABLE_L3", ""):` wrongly
                # disabled L3 when the user set HISIM_DISABLE_L3=0 (meaning L3 enabled).
                # Only treat explicit truthy values ("1", "true", "yes") as disable.
                _l3_disable_val = os.environ.get("HISIM_DISABLE_L3", "").strip().lower()
                if _l3_disable_val in ("1", "true", "yes", "on"):
                    _storage = None
                    logger.info(f"L3 storage disabled (HISIM_DISABLE_L3={_l3_disable_val!r}) — true L1-only mode")
                else:
                    _storage = getattr(cache.cache_controller, 'storage_backend', None)
                    if not isinstance(_storage, MockHiCacheStorage):
                        _storage = MockHiCacheStorage()
                    logger.info(f"L3 storage ENABLED (HISIM_DISABLE_L3={_l3_disable_val!r}) — st={type(_storage).__name__}")

                # L3 timing helpers
                _disk_write_bw = 8e9  # 8 GB/s default
                _disk_read_bw = 14.6e9  # 14.6 GB/s default
                try:
                    from hisim.simulation.manager import ConfigManager as _CM
                    _pcfg = _CM.get_platform_config()
                    _disk_write_bw = getattr(_pcfg, 'disk_write_bandwidth', 8e9) or 8e9
                    _disk_read_bw = getattr(_pcfg, 'disk_read_bandwidth', 14.6e9) or 14.6e9
                except Exception:
                    pass


                def _mock_start_writing(cc=cache.cache_controller, host=full_host, c=cache, mh=mamba_host, st=_storage):
                    if len(cc.write_queue) == 0:
                        return
                    try:
                        op = cc.write_queue[0]
                        if hasattr(type(op), 'merge_ops'):
                            op = type(op).merge_ops(cc.write_queue)
                        cc.write_queue.clear()
                        _hi = getattr(op, 'host_indices', None)
                        _di = getattr(op, 'device_indices', None)
                        _ni = getattr(op, 'node_ids', [])
                        _pt = getattr(op, 'pool_transfers', None) or []
                    except Exception:
                        cc.write_queue.clear()
                        return
                    _diag_w["calls"] += 1
                    _tc = getattr(c, 'tree_core', None)
                    # FULL pool backup (D2H)
                    if _di is not None and _hi is not None:
                        try:
                            host.backup_from_device_all_layer(host.device_pool, _hi, _di, "kernel")
                        except Exception as e:
                            logger.debug(f"backup_from_device_all_layer failed: {e}")
                    # Process sidecar pool transfers (MAMBA/KDA)
                    _mamba_xfers = []
                    for _xfer in _pt:
                        _xn = getattr(_xfer, 'name', None)
                        _xv = getattr(_xn, 'value', str(_xn))
                        if _xv == 'mamba':
                            _mamba_xfers.append(_xfer)
                    _mamba_slot_hashes = []
                    if _mamba_xfers and mh is not None:
                        _diag_w["mamba_pt"] += 1
                        for _xfer in _mamba_xfers:
                            _mhi = getattr(_xfer, 'host_indices', None)
                            _mdi = getattr(_xfer, 'device_indices', None)
                            try:
                                _n = len(_mhi) if hasattr(_mhi, '__len__') else 1
                                _nid = _ni[0] if _ni else 0
                                for _i in range(_n):
                                    _slot_idx = int(_mdi[_i]) if _mdi is not None and hasattr(_mdi, '__getitem__') else _i
                                    # Hash key is node-id based (not physical
                                    # slot idx: backup and load use different
                                    # physical slots, so slot idx would never
                                    # match). The loop index disambiguates
                                    # multiple slots within one node.
                                    _slot_hash = f"kda_{_nid}_{_i}"
                                    # [P2C.3.1] Compute checkpoint depth
                                    # (prefix length) from tree node for
                                    # CostAwareL2Policy eviction scoring.
                                    # depth = sum of key lengths from root
                                    # to this node = total token offset.
                                    _depth = 0
                                    try:
                                        _tnode = _tc.node_by_id(_nid) if (_tc is not None and hasattr(_tc, 'node_by_id')) else None
                                        _walker = _tnode
                                        while _walker is not None and getattr(_walker, 'key', None) is not None:
                                            try:
                                                _depth += len(_walker.key)
                                            except TypeError:
                                                _depth += 1
                                                break
                                            _walker = getattr(_walker, 'parent', None)
                                    except Exception:
                                        pass
                                    mh.backup_from_device(_slot_idx, _slot_hash, depth=_depth)
                                    if kda_obs.is_obs_enabled() and _i == 0:
                                        logger.info(f"[KDA-DEPTH] node={_nid} depth={_depth} hash={_slot_hash}")
                                    _mamba_slot_hashes.append(_slot_hash)
                                    # [A2] Shadow: register + backup_l2
                                    if kda_obs.is_obs_enabled():
                                        if _slot_hash not in kda_obs._SHADOW:
                                            kda_obs.shadow_register_checkpoint(
                                                _slot_hash, node_id=_nid,
                                                device_slot=_slot_idx)
                                        kda_obs.shadow_backup_l2(_slot_hash, _slot_idx)
                                    # [A2-fix] Manually update tree_core's
                                    # component_evictable_size_[MAMBA] and LRU
                                    # list, because commit_hicache_transfer
                                    # (BACKUP_HOST) doesn't update them.
                                    try:
                                        _tcore = getattr(c, 'tree_core', None)
                                        _has_lru = hasattr(_tcore, 'lru_lists') if _tcore else False
                                        if _tcore is not None and _has_lru:
                                            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _ACT
                                            _mnode = _tcore.node_by_id(_nid)
                                            if _mnode is not None:
                                                _mcd = _mnode.component_data[_ACT.MAMBA]
                                                _mval = _mcd.value
                                                if _mval is not None:
                                                    _mlru = _tcore.lru_lists.get(_ACT.MAMBA)
                                                    if _mlru is not None:
                                                        _in_list = _mlru.in_list(_mnode)
                                                        # Force-update counter: node is in LRU but
                                                        # counter may be 0 due to init mismatch
                                                        _cur = _tcore.component_evictable_size_.get(_ACT.MAMBA, 0)
                                                        _needed = len(_mval)
                                                        if _cur < _needed:
                                                            _tcore.component_evictable_size_[_ACT.MAMBA] = _needed
                                                        if not _in_list:
                                                            _mlru.insert_mru(_mnode)
                                                        # [A2-root-cause-fix] store_component_device_value
                                                        # doesn't call _update_evictable_leaf_sets, so the
                                                        # node is NOT in evictable_device_leaves and the
                                                        # MambaComponent eviction can't find it.
                                                        # Manually add it here.
                                                        try:
                                                            _tcore._update_evictable_leaf_sets(_mnode)
                                                            if _mnode.parent is not None:
                                                                _tcore._update_evictable_leaf_sets(_mnode.parent)
                                                        except Exception:
                                                            pass
                                                        # [A2] Also increment HiSim-managed counter
                                                        kda_obs.inc_mamba_evictable(len(_mval))
                                                        if kda_obs.is_obs_enabled():
                                                            _in_edl = _mnode in _tcore.evictable_device_leaves
                                                            kda_obs.log_event("lru_update", node_id=_nid,
                                                                extra={"old": _cur, "new": max(_cur, _needed),
                                                                       "in_list": _in_list, "in_edl": _in_edl,
                                                                       "hism_cnt": kda_obs.get_mamba_evictable()})
                                            if kda_obs.is_obs_enabled():
                                                _vm = "set" if (_mnode is not None and _mnode.component_data[_ACT.MAMBA].value is not None) else "None"
                                                _mlru_check = _tcore.lru_lists.get(_ACT.MAMBA) if _tcore else None
                                                _in_list_check = _mlru_check.in_list(_mnode) if (_mlru_check is not None and _mnode is not None) else None
                                                kda_obs.log_event("lru_update_attempt", node_id=_nid,
                                                    extra={"tcore_ok": True, "has_lru": True,
                                                           "node_found": _mnode is not None,
                                                           "mamba_val": _vm,
                                                           "mlru_ok": _mlru_check is not None,
                                                           "in_list": _in_list_check})
                                        elif kda_obs.is_obs_enabled():
                                            kda_obs.log_event("lru_update_attempt", node_id=_nid,
                                                extra={"tcore_ok": _tcore is not None, "has_lru": False})
                                    except Exception as _me:
                                        if kda_obs.is_obs_enabled():
                                            kda_obs.log_event("lru_update_error", node_id=_nid,
                                                extra={"error": str(_me)[:200]})
                                        logger.debug(f"mamba LRU update failed: {_me}")
                            except Exception as e:
                                logger.debug(f"mamba backup_from_device failed: {e}")
                        logger.debug(
                            f"[KDA-WRITE] nodes={len(_ni)} dev={0 if _di is None else len(_di)} "
                            f"mamba_slots={len(_mamba_slot_hashes)} hashes={_mamba_slot_hashes[:4]}"
                        )
                    elif _pt:
                        _diag_w["full_only"] += 1
                    # L3 (SSD) backup — mirror real sglang's write_backup_storage
                    # which is called from _finish_write_through_ack
                    if st is None:
                        logger.debug("[KDA-L3] skip: storage backend is None")
                    elif not _ni:
                        logger.debug(
                            f"[KDA-L3] skip: empty node_ids (di={_di is not None}, "
                            f"mamba_slots={len(_mamba_slot_hashes)})"
                        )
                    if st is not None and _ni:
                        _kv_bytes = float(getattr(MockTokenToKVPoolHost, 'KV_CACHE_BYTES', 0) or 0)
                        if _kv_bytes == 0:
                            try:
                                from hisim.simulation.manager import ConfigManager
                                _kv_bytes = float(ConfigManager.get_kv_cache_bytes())
                            except Exception:
                                _kv_bytes = 1e5
                        for nid in _ni:
                            try:
                                _hash = None
                                if _tc is not None and hasattr(_tc, 'node_by_id'):
                                    _node = _tc.node_by_id(nid)
                                    if _node is not None:
                                        _hash = getattr(_node, 'hash_value', None)
                                        if _hash:
                                            _hash = _hash[-1] if isinstance(_hash, list) else str(_hash)
                                if not _hash:
                                    _hash = f"node_{nid}"
                                st.set(str(_hash))
                                _diag_w["l3_writes"] += 1
                                _l3_dur = _kv_bytes / _disk_write_bw if _di is not None else 0
                                StateManager.inc_hicache_l3_backup_dur(_l3_dur)
                            except Exception as e:
                                logger.debug(f"L3 write failed node {nid}: {e}")
                        # KDA L3 backup: write-through all backed-up slots
                        # + overflow promotion from L2 eviction
                        _l3_mamba = list(_mamba_slot_hashes)
                        if mh is not None:
                            _evicted = getattr(mh, '_last_evicted_hashes', None) or []
                            if _evicted:
                                _l3_mamba.extend(_evicted)
                                mh._last_evicted_hashes = []
                                # [P0-1 DIAG] Log L2 evictions for analysis
                                # (do NOT clear host_value yet — causes node removal)
                                if kda_obs.is_obs_enabled():
                                    for _eh in _evicted:
                                        kda_obs.log_event("l2_evict_diag",
                                            virtual_time=StateManager.get_global_clock(),
                                            extra={"slot_hash": _eh})
                        if _l3_mamba:
                            try:
                                st.batch_set_mamba(_l3_mamba)
                                _slot_bytes = getattr(mh, 'slot_bytes', 14653440) if mh else 14653440
                                _kda_l3_dur = (_slot_bytes * len(_l3_mamba)) / _disk_write_bw
                                StateManager.inc_hicache_kda_l3_backup_dur(_kda_l3_dur)
                                # [D1.1] KDA L3 write counter (separate from MLA l3_writes)
                                _diag_w["kda_l3_writes"] += len(_l3_mamba)
                                # [A2] Shadow: backup_l3 for each evicted hash
                                if kda_obs.is_obs_enabled():
                                    for _sh in _l3_mamba:
                                        kda_obs.shadow_backup_l3(_sh)
                            except Exception as e:
                                logger.debug(f"KDA L3 write failed: {e}")
                    # Set host_value on components (FULL + MAMBA)
                    if _tc is not None and hasattr(_tc, 'node_by_id') and _ni:
                        for nid in _ni:
                            try:
                                node = _tc.node_by_id(nid)
                                if node is not None and hasattr(node, 'component_data'):
                                    try:
                                        _fv = node.component_data[_HA_CT.FULL]
                                    except (KeyError, IndexError, TypeError):
                                        _fv = None
                                    try:
                                        _mv = node.component_data[_HA_CT.MAMBA]
                                    except (KeyError, IndexError, TypeError):
                                        _mv = None
                                    if _fv is not None and _hi is not None and len(_hi) > 0:
                                        _fv.host_value = _hi
                                    if _mamba_xfers and _mv is not None:
                                        _mhi = getattr(_mamba_xfers[0], 'host_indices', None)
                                        if _mhi is not None:
                                            # [P0-1 FIX] Only set host_value if the slot
                                            # is still in L2 after backup. With L2=0 or
                                            # L2 full, the slot is immediately evicted,
                                            # and host_value would point to stale data.
                                            # The native init_load_back uses host_value
                                            # to "restore" MAMBA state, bypassing the
                                            # HiSim mock's L2/L3 membership check.
                                            _l2_still_has = True
                                            if mh is not None:
                                                for _sh in _mamba_slot_hashes:
                                                    if not mh.has(_sh):
                                                        _l2_still_has = False
                                                        break
                                            if _l2_still_has:
                                                _mv.host_value = _mhi.clone() if hasattr(_mhi, 'clone') else _mhi
                                            else:
                                                _mv.host_value = None  # L2 evicted
                                    if _mv is not None and _mv.value is not None and _mv.host_value is None:
                                        # Fallback: only set if L2 didn't evict
                                        _l2_ok = True
                                        if mh is not None:
                                            for _sh in _mamba_slot_hashes:
                                                if not mh.has(_sh):
                                                    _l2_ok = False
                                                    break
                                        if _l2_ok:
                                            _mv.host_value = _mv.value.clone() if hasattr(_mv.value, 'clone') else _mv.value
                            except Exception as e:
                                logger.debug(f"set host_value failed node {nid}: {e}")
                    _ack_wq_before = len(cc.ack_write_queue) if hasattr(cc, 'ack_write_queue') else -1
                    _cc_id = id(cc)
                    _cc_wq_id = id(cc.ack_write_queue) if hasattr(cc, 'ack_write_queue') else -1
                    try:
                        _evt = torch.Event(); _evt.record()
                        cc.ack_write_queue.append(_HA_CKACK(
                            start_event=_evt, finish_event=_evt, node_ids=list(_ni) if _ni else [],
                            num_tokens=len(_di) if _di is not None else 0,
                            timing_enabled=False, num_bytes=0,
                        ))
                    except Exception as e:
                        logger.warning(f"fake write ack failed: {e}")
                    _ack_wq_after = len(cc.ack_write_queue) if hasattr(cc, 'ack_write_queue') else -1
                    # [ACK-LIFECYCLE] Log ACK creation with node_ids for matching
                    # against _track_write_through_node and _finish_write_through_ack.
                    if kda_obs.is_obs_enabled():
                        kda_obs.log_event("ack_create",
                            virtual_time=StateManager.get_global_clock(),
                            extra={"node_ids": list(_ni) if _ni else [],
                                   "wq_before": _ack_wq_before,
                                   "wq_after": _ack_wq_after,
                                   "owt_size": len(getattr(c, 'ongoing_write_through', {})),
                                   "cc_id": _cc_id,
                                   "wq_list_id": _cc_wq_id,
                                   "self_cc_id": id(getattr(c, 'cache_controller', None))})
                cache.cache_controller.start_writing = _mock_start_writing

                def _mock_start_loading(cc=cache.cache_controller, host=full_host, c=cache, mh=mamba_host, st=_storage):
                    if len(cc.load_queue) == 0:
                        return -1
                    try:
                        op = cc.load_queue[0]
                        if hasattr(type(op), 'merge_ops'):
                            op = type(op).merge_ops(cc.load_queue)
                        cc.load_queue.clear()
                        _hi = getattr(op, 'host_indices', None)
                        _di = getattr(op, 'device_indices', None)
                        _ni = getattr(op, 'node_ids', [])
                        _pt = getattr(op, 'pool_transfers', None) or []
                    except Exception:
                        cc.load_queue.clear()
                        return -1
                    _diag_l["calls"] += 1
                    _tc = getattr(c, 'tree_core', None)
                    # L3 (SSD) read — check if data is in L3
                    if st is not None and _ni:
                        _kv_bytes = float(getattr(MockTokenToKVPoolHost, 'KV_CACHE_BYTES', 0) or 0)
                        if _kv_bytes == 0:
                            try:
                                from hisim.simulation.manager import ConfigManager
                                _kv_bytes = float(ConfigManager.get_kv_cache_bytes())
                            except Exception:
                                _kv_bytes = 1e5
                        for nid in _ni:
                            try:
                                _hash = None
                                if _tc is not None and hasattr(_tc, 'node_by_id'):
                                    _node = _tc.node_by_id(nid)
                                    if _node is not None:
                                        _hash = getattr(_node, 'hash_value', None)
                                        if _hash:
                                            _hash = _hash[-1] if isinstance(_hash, list) else str(_hash)
                                if not _hash:
                                    _hash = f"node_{nid}"
                                _hit_count = 1 if st.exists(str(_hash)) else 0
                                if _hit_count > 0:
                                    _diag_l["l3_hits"] += 1
                                    _l3_dur = _kv_bytes / _disk_read_bw if _di is not None else 0
                                    # [v2-fix F4] Use MLA L3 counter, not KDA counter
                                    StateManager.inc_hicache_l3_load_dur(_l3_dur)
                                    # [A4] Log MLA L3 load (separate from KDA counter)
                                    logger.info(f"[A4-MLA-L3-LOAD] node={nid} dur={_l3_dur:.6f}")
                                    C_SchedulerHook._mla_l3_load_total = getattr(C_SchedulerHook, '_mla_l3_load_total', 0.0) + _l3_dur
                                    # Set L3 direct load count so load_to_device_per_layer
                                    # splits the H2D timing between L2 and L3 direct path
                                    _ntok = len(_di) if _di is not None else 0
                                    if _ntok > 0:
                                        StateManager.inc_l3_direct_load_count(_ntok)
                            except Exception as e:
                                logger.debug(f"L3 read failed node {nid}: {e}")
                    # FULL pool load (H2D)
                    if _hi is not None and _di is not None:
                        try:
                            host.load_to_device_per_layer(host.device_pool, _hi, _di, 0, "kernel")
                        except Exception as e:
                            logger.debug(f"load_to_device_per_layer failed: {e}")
                    # Process sidecar pool transfers (MAMBA/KDA) with L3→L2→L1 restore
                    _mamba_loads = []
                    for _xfer in _pt:
                        _xn = getattr(_xfer, 'name', None)
                        _xv = getattr(_xn, 'value', str(_xn))
                        if _xv == 'mamba':
                            _mamba_loads.append(_xfer)
                    # [P0-1 DIAG] Log whether MAMBA transfers exist
                    logger.info(f"[LOAD-DIAG] calls={_diag_l['calls']} pt_count={len(_pt)} mamba_loads={len(_mamba_loads)} ni={list(_ni)[:3]}")
                    if _mamba_loads and mh is not None:
                        _diag_l["mamba_pt"] += 1
                        _any_mamba_restore = False
                        for _xfer in _mamba_loads:
                            _mhi = getattr(_xfer, 'host_indices', None)
                            _mdi = getattr(_xfer, 'device_indices', None)
                            try:
                                _n = len(_mhi) if hasattr(_mhi, '__len__') else 1
                                _nid = _ni[0] if _ni else 0
                                for _i in range(_n):
                                    _slot_idx = int(_mdi[_i]) if _mdi is not None and hasattr(_mdi, '__getitem__') else _i
                                    _slot_hash = f"kda_{_nid}_{_i}"
                                    _l2_hit = mh.has(_slot_hash)
                                    logger.info(f"[LOAD-L2CHECK] hash={_slot_hash} l2_hit={_l2_hit} l2_cached={len(getattr(mh, '_slots', {}))} l2_cap={getattr(mh, 'size', -1)} st_is_none={st is None} st_type={type(st).__name__ if st else 'None'}")
                                    # [P2C.3.2] Record L2 lookup for HotnessL2 policy
                                    # (hit OR miss — the slot was REQUESTED).
                                    # Previously record_access was only called
                                    # from load_to_device (L2 hits), so evicted
                                    # slots that were re-requested kept freq=0.
                                    _l2_pol = mh._get_policy() if hasattr(mh, '_get_policy') else None
                                    if _l2_pol is not None:
                                        try:
                                            _l2_pol.record_access(_slot_hash, StateManager.get_global_clock())
                                        except Exception:
                                            pass
                                    if _l2_hit:
                                        # [P2D2] Check load_to_device return value.
                                        # load_to_device returns -1 if the slot
                                        # is not actually in L2 (race / stale).
                                        # Only mark restore success on valid return.
                                        _ld_ret = mh.load_to_device(_slot_hash)
                                        if _ld_ret is not None and _ld_ret >= 0:
                                            _any_mamba_restore = True
                                            if kda_obs.is_obs_enabled():
                                                kda_obs.shadow_restore_l2(_slot_hash, _slot_idx)
                                        else:
                                            logger.info(f"[P2D2-L2-LOAD-FAIL] hash={_slot_hash} load_to_device={_ld_ret} — restore NOT confirmed")
                                            _diag_l["kda_l2_load_fail"] = _diag_l.get("kda_l2_load_fail", 0) + 1
                                    elif st is not None:
                                        _m_hits = st.batch_exists_mamba([_slot_hash])
                                        _l3_hit_val = _m_hits.get(_slot_hash, 0) if _m_hits else 0
                                        logger.info(f"[LOAD-L3CHECK] hash={_slot_hash} l3_hit={_l3_hit_val} raw={_m_hits}")
                                        if any(_m_hits.values()):
                                            _slot_bytes = getattr(mh, 'slot_bytes', 14653440)
                                            _kda_l3_dur = _slot_bytes / _disk_read_bw
                                            StateManager.inc_hicache_kda_l3_load_dur(_kda_l3_dur)
                                            # [D1.1] KDA L3 hit counter (separate from MLA l3_hits)
                                            _diag_l["kda_l3_hits"] += 1
                                            # [A4] Log KDA L3 load (separate from MLA counter)
                                            logger.info(f"[A4-KDA-L3-LOAD] hash={_slot_hash} dur={_kda_l3_dur:.6f}")
                                            C_SchedulerHook._kda_l3_load_total = getattr(C_SchedulerHook, '_kda_l3_load_total', 0.0) + _kda_l3_dur
                                            # [D1.2] Check backup_from_device return — don't declare
                                            # restore success just because L3 has data. If L2 admission
                                            # fails (One Pool budget exhausted), the data is NOT consumable
                                            # (no Direct Staging path in current Mock).
                                            _bk_ret = mh.backup_from_device(-1, _slot_hash, from_l3=True)
                                            if _bk_ret is not False:
                                                # [P2D2] Verify the staged slot is actually loadable.
                                                # backup_from_device may succeed (admission OK) but
                                                # the slot might not be in _slots if there was a
                                                # race or internal error.
                                                _ld_ret = mh.load_to_device(_slot_hash)
                                                if _ld_ret is not None and _ld_ret >= 0:
                                                    _any_mamba_restore = True
                                                else:
                                                    logger.info(f"[P2D2-L3-STAGE-FAIL] hash={_slot_hash} staged but load returned {_ld_ret}")
                                                    _diag_l["kda_l3_stage_fail"] = _diag_l.get("kda_l3_stage_fail", 0) + 1
                                            else:
                                                # L2 admission rejected — data in L3 but not staged to L2
                                                logger.info(f"[D1.2-L3-REJECT] hash={_slot_hash} L3 hit but L2 admission failed")
                                                _diag_l["kda_l3_admission_reject"] += 1
                                                # _any_mamba_restore stays False → replay triggered
                                            if kda_obs.is_obs_enabled():
                                                kda_obs.shadow_restore_l3(_slot_hash, _slot_idx)
                            except Exception as e:
                                logger.info(f"[LOAD-ERROR] mamba load_to_device failed: {e}", exc_info=True)
                        # [P0-2] Track if MAMBA restore failed for this batch
                        if not _any_mamba_restore:
                            _diag_l["mamba_restore_failed"] = True
                            logger.info("[LOAD-RESULT] MAMBA restore FAILED — replay needed")
                        else:
                            _diag_l["mamba_restore_failed"] = False
                            logger.info("[LOAD-RESULT] MAMBA restore OK — no replay needed")
                    elif _pt:
                        _diag_l["full_only"] += 1
                    # Set value on MAMBA component (restore from L2)
                    if _tc is not None and hasattr(_tc, 'node_by_id') and _ni:
                        for nid in _ni:
                            try:
                                node = _tc.node_by_id(nid)
                                if node is not None and hasattr(node, 'component_data'):
                                    if _mamba_loads:
                                        _mdi = None
                                        for _ml in _mamba_loads:
                                            _tmp = getattr(_ml, 'device_indices', None)
                                            if _tmp is not None:
                                                _mdi = _tmp
                                                break
                                        if _mdi is not None:
                                            try:
                                                node.component_data[_HA_CT.MAMBA].value = _mdi.clone() if hasattr(_mdi, 'clone') else _mdi
                                            except (KeyError, IndexError, TypeError):
                                                pass
                            except Exception as e:
                                logger.debug(f"set mamba value failed node {nid}: {e}")
                    try:
                        _evt = torch.Event(); _evt.record()
                        cc.ack_load_queue.append(_HA_CKACK(
                            start_event=_evt, finish_event=_evt, node_ids=_ni,
                            num_tokens=len(_di) if _di is not None else 0,
                            timing_enabled=False, num_bytes=0,
                        ))
                    except Exception as e:
                        logger.warning(f"fake load ack failed: {e}")
                    return 0
                cache.cache_controller.start_loading = _mock_start_loading

                # Expose diagnostic counters for verification
                cache._hismim_diag_w = _diag_w
                cache._hismim_diag_l = _diag_l

                # Tracking dicts required by _finish_write_through_ack /
                # _finish_load_back_ack (called from writing_check/loading_check).
                if not hasattr(cache, 'ongoing_write_through'):
                    cache.ongoing_write_through = {}
                if not hasattr(cache, 'ongoing_load_back'):
                    cache.ongoing_load_back = {}
                if not hasattr(cache, 'ongoing_prefetch'):
                    cache.ongoing_prefetch = {}
                if not hasattr(cache, 'ongoing_backup'):
                    cache.ongoing_backup = {}
                if not hasattr(cache, 'buffer_pipeline'):
                    cache.buffer_pipeline = None
                logger.debug("Applied mock start_writing/start_loading overrides")

                # [A2-root-cause-fix] Add MAMBA to tree_components if missing.
                # Without this, _evict_components skips MAMBA entirely,
                # causing mamba slot exhaustion.
                _tc_m = getattr(cache, 'tree_core', None)
                if _tc_m is not None:
                    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _MCT
                    _mtcs = getattr(_tc_m, 'tree_components', None)
                    if _mtcs is not None and _MCT.MAMBA not in _mtcs:
                        _mtcs.append(_MCT.MAMBA)
                        logger.info(f"[A2-fix] Added MAMBA to tree_components: now {[getattr(c, chr(118)+chr(97)+chr(108)+chr(117)+chr(101), str(c)) for c in _mtcs]}")
                    if kda_obs.is_obs_enabled():
                        kda_obs.log_event("tree_components_fix",
                            extra={"tree_components": [getattr(c, chr(118)+chr(97)+chr(108)+chr(117)+chr(101), str(c)) for c in (_mtcs or [])],
                                   "mamba_in": _MCT.MAMBA in (_mtcs or [])})

            # Unified L2 DRAM pool (KDA + MLA share one byte budget).
            # None when HISIM_L2_ONE_POOL / HISIM_L2_TOTAL_GB are unset.
            _l2_pool = _setup_l2_dram_pool(server_args, params)

            try:
                _is_unified = getattr(server_args, 'enable_unified_memory', False)
                if _is_unified:
                    raise Exception("Skip original init_hicache in unified memory mode")
                original_init_hicache(self, server_args, params)
            except Exception as e:
                logger.warning(f"original init_hicache failed ({e}), using manual setup")

                # Manually create cache_controller and host pools
                from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
                    HostPoolGroup, PoolEntry, PoolName,
                )
                from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
                    HybridCacheController,
                )
                _kvcache = params.token_to_kv_pool_allocator.get_kvcache()
                _page_size = params.page_size

                # Create mock host pools
                # L2 size configurable for eviction testing (default: 330 Mamba slots, hicache_size GB KV)
                if _l2_pool is not None:
                    _l2_mamba_size = int(_l2_pool.kda_capacity_slots)
                    _l2_kv_size = float(_l2_pool.mla_capacity_bytes) / 1e9
                else:
                    _l2_mamba_size = int(os.environ.get("HISIM_L2_MAMBA_SIZE", "330"))
                    _l2_kv_size = float(os.environ.get("HISIM_L2_KV_SIZE_GB", str(server_args.hicache_size)))
                _full_host = MockTokenToKVPoolHost(
                    _kvcache,
                    server_args.hicache_ratio,
                    _l2_kv_size,
                    _page_size,
                    server_args.hicache_mem_layout,
                    pin_memory=False,
                    device="cpu",
                )
                _mamba_host = MockMambaStatePoolHost(
                    size=_l2_mamba_size,
                    slot_bytes=14653440,
                )
                logger.info(f"L2 pools: mamba_size={_l2_mamba_size} kv_size_gb={_l2_kv_size}")

                # Create host_pool_group with PoolEntry objects
                # K3 0-indexed layer mapping:
                # Full Attention (KV): layers [3,7,11,15,19] → sub-pool indices [0,1,2,3,4]
                # KDA (MAMBA): layers [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22] → [0..17]
                _k3_kv_map = {3: 0, 7: 1, 11: 2, 15: 3, 19: 4}
                _k3_mamba_map = {
                    0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 8: 6, 9: 7, 10: 8,
                    12: 9, 13: 10, 14: 11, 16: 12, 17: 13, 18: 14, 20: 15, 21: 16, 22: 17,
                }
                self.host_pool_group = HostPoolGroup(entries=[
                    PoolEntry(
                        name=PoolName.KV,
                        host_pool=_full_host,
                        device_pool=_kvcache,
                        layer_mapper=lambda i, _m=_k3_kv_map: _m.get(i),
                    ),
                    PoolEntry(
                        name=PoolName.MAMBA,
                        host_pool=_mamba_host,
                        device_pool=getattr(params.req_to_token_pool, 'mamba_pool', _kvcache),
                        layer_mapper=lambda i, _m=_k3_mamba_map: _m.get(i),
                    ),
                ])

                # Create HybridCacheController.
                # 0.5.19: the controller expects a HostPoolGroup as mem_pool_host
                # (it reads mem_pool_host.anchor_entry.host_pool and .entries);
                # passing the plain mock host pool crashes with
                # "'NoneType' object has no attribute 'host_pool'".
                self.load_cache_event = threading.Event()
                self.cache_controller = HybridCacheController(
                    token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
                    mem_pool_host=self.host_pool_group,
                    page_size=_page_size,
                    tp_group=params.tp_cache_group,
                    load_cache_event=self.load_cache_event,
                    write_policy=server_args.hicache_write_policy,
                    io_backend=server_args.hicache_io_backend,
                    storage_backend=server_args.hicache_storage_backend,
                    prefetch_threshold=256,
                    model_name=server_args.served_model_name,
                    storage_backend_extra_config=None,
                    transfer_layer_num=23,
                    host_memory_mode=server_args.hicache_host_memory_mode,
                 )

                # Apply mock start_writing/start_loading overrides (defined above)
                _apply_mock_start_overrides(self, _full_host, _mamba_host)

                # Override backup/prefetch thread funcs on the controller instance
                # to prevent real async threads from running (causes NPU crash).
                # C_HiCacheController.hook overrides these on HiCacheController class,
                # but HybridCacheController has its own methods that shadow them.
                self.cache_controller.backup_thread_func = lambda *a, **kw: None
                self.cache_controller.prefetch_thread_func = lambda *a, **kw: None

                # Set host pool attributes on cache and components
                from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _CT
                self.full_kv_pool_host = _full_host
                self.mamba_pool_host = _mamba_host
                if _CT.FULL in self.components:
                    self.components[_CT.FULL]._full_kv_pool_host = _full_host
                if _CT.MAMBA in self.components:
                    self.components[_CT.MAMBA]._mamba_pool_host = _mamba_host

                self.enable_storage = server_args.hicache_storage_backend is not None
                self.is_write_back = server_args.hicache_write_policy == "write_back"
                self.write_through_threshold = (
                    1 if server_args.hicache_write_policy == "write_through" else 2
                )
                self.load_back_threshold = 10
                self.prefetch_loaded_tokens_by_reqid = {}
                # Tracking dicts required by _finish_write_through_ack and
                # _finish_load_back_ack (called from writing_check/loading_check).
                self.ongoing_write_through = {}
                self.ongoing_load_back = {}
                self.ongoing_prefetch = {}
                self.ongoing_backup = {}
                self.buffer_pipeline = None

                # Tag tree_core
                if hasattr(self, 'tree_core'):
                    self.tree_core.set_hicache_enabled()

                logger.info("Manual hicache setup completed (fallback mode)")

            # If original succeeded, swap host pools to mock versions
            if not hasattr(self, 'full_kv_pool_host') or self.full_kv_pool_host is None:
                # Original succeeded but didn't set full_kv_pool_host (shouldn't happen)
                pass
            elif not isinstance(self.full_kv_pool_host, MockTokenToKVPoolHost):
                # Original succeeded — swap real host pools to mock
                from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _CT
                _kvcache = params.token_to_kv_pool_allocator.get_kvcache()
                if _l2_pool is not None:
                    _l2_kv_size_orig = float(_l2_pool.mla_capacity_bytes) / 1e9
                else:
                    _l2_kv_size_orig = server_args.hicache_size
                self.full_kv_pool_host = MockTokenToKVPoolHost(
                    _kvcache,
                    server_args.hicache_ratio,
                    _l2_kv_size_orig,
                    self.page_size,
                    server_args.hicache_mem_layout,
                    pin_memory=False,
                    device="cpu",
                )
                if _CT.FULL in self.components:
                    self.components[_CT.FULL]._full_kv_pool_host = self.full_kv_pool_host

                # Mamba pool host: ALWAYS create a MockMambaStatePoolHost with
                # the correct L2 size, regardless of what original init set.
                _l2_mamba_size_orig = (
                    int(_l2_pool.kda_capacity_slots) if _l2_pool is not None
                    else int(os.environ.get("HISIM_L2_MAMBA_SIZE", "330"))
                )
                self.mamba_pool_host = MockMambaStatePoolHost(
                    size=_l2_mamba_size_orig,
                    slot_bytes=14653440,
                )
                if _CT.MAMBA in self.components:
                    self.components[_CT.MAMBA]._mamba_pool_host = self.mamba_pool_host
                logger.info(f"Force-set mamba_pool_host (L2 size={_l2_mamba_size_orig})")

                logger.info("Swapped real host pools to mock versions")
                _apply_mock_start_overrides(self, self.full_kv_pool_host, getattr(self, 'mamba_pool_host', None))

            # Unified L2 DRAM pool: bind mocks to the shared budget and point
            # the native HostPoolGroup at them (Two Pools + One Pool share the
            # same total byte budget; only the split policy differs).
            if _l2_pool is not None and getattr(self, 'full_kv_pool_host', None) is not None:
                _bind_l2_host_group(
                    self,
                    _l2_pool,
                    self.full_kv_pool_host,
                    getattr(self, 'mamba_pool_host', None),
                )
                logger.info(
                    "L2 unified DRAM pool bound: mode=%s total=%.2fGB "
                    "mla=%.2fGB kda_slots=%d page=%dB",
                    _l2_pool.mode,
                    _l2_pool.total_bytes / 1e9,
                    _l2_pool.mla_capacity_bytes / 1e9,
                    _l2_pool.kda_capacity_slots,
                    _l2_pool.page_bytes,
                )
            else:
                # [P2D2] Legacy mode: normalize HostPoolGroup capacity to
                # the mock pool's logical size.  Without this, group.size
                # stays at the native default (hicache_size=80GB), which
                # prevents radix tree eviction and creates an unfair baseline
                # (Legacy never evicts radix nodes → inflated L3 hits).
                _kv_h = getattr(self, 'full_kv_pool_host', None)
                _mb_h = getattr(self, 'mamba_pool_host', None)
                if _kv_h is not None:
                    from sglang.srt.mem_cache.hicache_storage import PoolName as _P2D2_PN
                    _grp = getattr(self, "host_pool_group", None)
                    _cc_p2d2 = getattr(self, "cache_controller", None)
                    if _grp is None and _cc_p2d2 is not None:
                        _grp = getattr(_cc_p2d2, "mem_pool_host", None)
                    if _grp is not None and hasattr(_grp, "entry_map"):
                        _kv_ent = _grp.entry_map.get(_P2D2_PN.KV)
                        if _kv_ent is not None:
                            _kv_ent.host_pool = _kv_h
                        _mb_ent = _grp.entry_map.get(_P2D2_PN.MAMBA)
                        if _mb_ent is not None and _mb_h is not None:
                            _mb_ent.host_pool = _mb_h
                        try:
                            _grp.size = _kv_h.size
                            _grp.logical_size = getattr(_kv_h, 'logical_size', _kv_h.size)
                            _grp.page_size = getattr(_kv_h, 'page_size', 1)
                            _grp._refresh_transfer_capabilities()
                        except Exception as _exc:
                            logger.debug(f"[P2D2] Legacy group.size normalization failed: {_exc}")
                        logger.info(
                            "[P2D2] Legacy HostPoolGroup normalized: "
                            "group.size=%d tokens (%.3fGB), kv_host.size=%d, "
                            "mamba_size=%d",
                            _grp.size,
                            _grp.size * 5760 / 1e9,
                            _kv_h.size,
                            getattr(_mb_h, 'size', 0) if _mb_h else 0,
                        )

            # [P2D2] Capacity verification — print and validate at startup
            _kv_h_v = getattr(self, 'full_kv_pool_host', None)
            _grp_v = getattr(self, "host_pool_group", None)
            _cc_v = getattr(self, "cache_controller", None)
            if _grp_v is None and _cc_v is not None:
                _grp_v = getattr(_cc_v, "mem_pool_host", None)
            _grp_size_v = getattr(_grp_v, 'size', None) if _grp_v else None
            _kv_size_v = getattr(_kv_h_v, 'size', None) if _kv_h_v else None
            logger.info(
                "[P2D2] Capacity check: group.size=%s kv_host.size=%s "
                "enable_storage=%s hicache_size=%sGB mode=%s",
                _grp_size_v,
                _kv_size_v,
                getattr(self, 'enable_storage', 'N/A'),
                getattr(server_args, 'hicache_size', 'N/A'),
                "unified" if _l2_pool is not None else "legacy",
            )
            if _grp_size_v is not None and _kv_size_v is not None:
                if _grp_size_v != _kv_size_v:
                    logger.warning(
                        "[P2D2] MISMATCH: group.size=%d != kv_host.size=%d "
                        "— radix tree eviction will be incorrect",
                        _grp_size_v, _kv_size_v,
                    )

            logger.info(
                f"HiSim hicache attached: components={set(self.components.keys())}, "
                f"write_through_threshold={getattr(self, 'write_through_threshold', 'N/A')}"
            )

            # [ACK-LIFECYCLE] Wrap ack_write_queue with a logging list to
            # catch ALL modifications (append, pop, clear, __init__).
            _cc_for_wq = getattr(self, 'cache_controller', None)
            if _cc_for_wq is not None and hasattr(_cc_for_wq, 'ack_write_queue'):
                _orig_wq = _cc_for_wq.ack_write_queue
                _cc_ref = _cc_for_wq
                class _LoggingAckQueue(list):
                    def append(self, item):
                        r = super().append(item)
                        if kda_obs.is_obs_enabled():
                            kda_obs.log_event("wq_append",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"len": len(self),
                                       "node_ids": list(getattr(item, 'node_ids', [])),
                                       "cc_id": id(_cc_ref)})
                        return r
                    def pop(self, idx=0):
                        if kda_obs.is_obs_enabled():
                            kda_obs.log_event("wq_pop",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"len_before": len(self), "idx": idx,
                                       "cc_id": id(_cc_ref)})
                        return super().pop(idx)
                    def clear(self):
                        if kda_obs.is_obs_enabled():
                            kda_obs.log_event("wq_clear",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"len_before": len(self),
                                       "cc_id": id(_cc_ref)})
                        return super().clear()
                _logging_wq = _LoggingAckQueue(_orig_wq)
                _cc_for_wq.ack_write_queue = _logging_wq

            # [P2B-FIX] Hook _drain_device_frees — deferred to after policy check.
            # Only installed when pre-sort policy is active (non-native).
            # See below after _policy is loaded.
            _tc_ednn = getattr(self, 'tree_core', None)

            if _tc_ednn is not None and kda_obs.is_obs_enabled():
                _orig_ednn_inst = getattr(_tc_ednn, 'evict_device_next_node', None)
                if _orig_ednn_inst is not None:
                    def _wrapped_ednn(ct, tracker, orig=_orig_ednn_inst):
                        result = orig(ct, tracker)
                        try:
                            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _ECT
                            if ct == _ECT.MAMBA:
                                kda_obs.log_event("ednn_trace",
                                    virtual_time=StateManager.get_global_clock(),
                                    extra={"ct": getattr(ct, 'value', str(ct)),
                                           "node_id": result.node_id if result.node_id else -1,
                                           "made_progress": result.made_progress})
                        except Exception:
                            pass
                        return result
                    _tc_ednn.evict_device_next_node = _wrapped_ednn

            # [Phase2] Pre-sort eviction hook on MambaComponent.
            # Hook point: MambaComponent._evict_device_start / _evict_device_next_node
            # (NOT UnifiedRadixCache._evict_device_next_node — that was the old
            # filter approach which had cursor-advancement side effects).
            #
            # Design:
            #   _evict_device_start: after native cursor setup, walk LRU list,
            #     collect ALL evictable MAMBA leaves, sort by policy, store as
            #     _policy_evict_queue. NO side effects at this point (cursor is
            #     set up but not walked).
            #   _evict_device_next_node: pop from pre-sorted queue (NO cursor
            #     advancement for policy-selected nodes). When queue exhausted,
            #     fall back to native walk (for tombstone cleanup).
            #   _evict_device_end: clear the queue.
            #
            # Safety:
            #   - Same evictable candidate set (evictable_device_leaves)
            #   - Same capacity trigger (evict_for_alloc → _evict_components)
            #   - Same eviction execution (_evict_device_leaf, native, unmodified)
            #   - NO cursor advancement before policy decision
            #   - NO device/host frees, NO tracker updates for non-evicted nodes
            #   - ALL candidates in queue → no cursor exhaustion, no alloc failure
            #   - Does NOT re-enable HiSim Proactive Eviction
            #   - Does NOT call allocator.free directly
            #   - Does NOT modify Lock/ACK/Cascade semantics
            from hisim.simulation import kda_policy as _kda_pol
            from hisim.simulation.kda_policy import _node_prefix_identity
            _policy = _kda_pol.get_policy()

            # [P2C.3] Initialize L2 placement policy (Tier-Aware Placement)
            try:
                from hisim.simulation.l2_placement_policy import get_l2_policy as _get_l2
                _l2_policy = _get_l2()
                logger.info(f"[P2C.3] L2 placement policy: {_l2_policy.name}")
            except Exception as _l2e:
                logger.warning(f"[P2C.3] L2 policy init skipped: {_l2e}")
                _l2_policy = None

            try:
                from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _PCT
                _mamba_comp = _tc_ednn.components_by_type.get(_PCT.MAMBA) if _tc_ednn else None
            except Exception:
                _mamba_comp = None

            logger.info(f"[Phase2-DEBUG] self_type={type(self).__name__} "
                        f"mamba_comp={_mamba_comp is not None} policy={_policy.name}")

            if _mamba_comp is not None and _policy.name != "native":
                # [P2B-FIX] Install _drain_device_frees pre-filter (only for non-native).
                # Prevents eviction-path double-free when free_mamba_cache already
                # freed a slot. For native, this hook is NOT installed because it
                # creates new tensors from tombstone cleanup, changing eviction behavior.
                _orig_drain = getattr(self, '_drain_device_frees', None)
                if _orig_drain is not None:
                    def _safe_drain_dev(device_frees, orig=_orig_drain, _self=self):
                        try:
                            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _DCT
                            if _DCT.MAMBA in device_frees:
                                _alloc = getattr(_self.cache.req_to_token_pool, 'mamba_allocator', None)
                                if _alloc is not None and hasattr(_alloc, 'free_slots'):
                                    _free_set = set(_alloc.free_slots)
                                    _tensors = device_frees[_DCT.MAMBA]
                                    _filtered = []
                                    _skipped = 0
                                    for _t in _tensors:
                                        _vals = _t.tolist() if hasattr(_t, 'tolist') else [int(_t)]
                                        _remaining = [v for v in _vals if v not in _free_set]
                                        _skipped += len(_vals) - len(_remaining)
                                        if len(_remaining) == len(_vals):
                                            _filtered.append(_t)
                                        elif _remaining:
                                            import torch as _torch
                                            _filtered.append(_torch.tensor(_remaining, dtype=_t.dtype, device=_t.device))
                                    if _skipped > 0:
                                        logger.info(f"[EVICT-PREFILTER] {_skipped} MAMBA slots already freed, filtered")
                                    if _filtered:
                                        device_frees[_DCT.MAMBA] = _filtered
                                    else:
                                        del device_frees[_DCT.MAMBA]
                        except Exception:
                            pass
                        return orig(device_frees)
                    self._drain_device_frees = _safe_drain_dev
                    logger.info("[P2B-FIX] _drain_device_frees pre-filter installed (non-native only)")

                # Capture bound methods before replacing
                _orig_start_b = _mamba_comp._evict_device_start
                _orig_next_b = _mamba_comp._evict_device_next_node
                _orig_end_b = _mamba_comp._evict_device_end

                def _policy_start(request_cnt, orig=_orig_start_b,
                                   comp=_mamba_comp, pol=_policy, tc=_tc_ednn):
                    """Pre-sort: collect evictable leaves, sort by policy."""
                    orig(request_cnt)  # Native cursor + request_cnt setup
                    try:
                        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _CT
                        _lru = tc.lru_lists.get(_CT.MAMBA)
                        _evictable = tc.evictable_device_leaves
                        _enabled = getattr(tc, 'enable_session_radix_cache', False)

                        _candidates = []
                        _node = _lru.get_lru_no_lock() if _lru else None
                        _wc = 0
                        while _node is not None and _wc < 10000:
                            if _node in _evictable:
                                _ce = True
                                if _enabled:
                                    _ce = comp._can_evict_leaf_atomically(_node)
                                if _ce:
                                    _candidates.append((_node, _node_prefix_identity(_node)))
                            _node = _lru.get_prev_no_lock(_node) if _lru else None
                            _wc += 1

                        # Set current time for Oracle policy
                        if hasattr(pol, 'set_current_time'):
                            pol.set_current_time(StateManager.get_global_clock())
                        # [P2C.2-DIAG] Log candidate order BEFORE sort
                        _before_ids = [getattr(n, 'id', -1) for n, _ in _candidates]
                        _sorted = pol.select_victims(_candidates, _lru, tc)
                        _after_ids = [getattr(n, 'id', -1) for n, _ in _sorted]
                        _order_changed = _before_ids != _after_ids
                        if _order_changed and len(_candidates) > 1:
                            logger.info(f"[POLICY:{pol.name}] SORT-CHANGED "
                                        f"before={_before_ids[:10]} "
                                        f"after={_after_ids[:10]} "
                                        f"ncand={len(_candidates)}")
                        comp._policy_evict_queue = [n for n, _ in _sorted]
                        comp._policy_evict_hashes = [h for _, h in _sorted]
                        comp._policy_evict_idx = 0
                        _prio_list = []
                        for n, h in _sorted[:10]:
                            _prio_list.append(f"n{n.id}:{pol.get_priority(h):.1f}")
                        _suffix = " ORDER_CHANGED" if _order_changed else ""
                        logger.info(f"[POLICY:{pol.name}] start: req={request_cnt} "
                                    f"cand={len(_candidates)} q={len(_sorted)} "
                                    f"top=[{', '.join(_prio_list)}]{_suffix}")
                        kda_obs.log_event("policy_start",
                            virtual_time=StateManager.get_global_clock(),
                            extra={"policy": pol.name, "request_cnt": request_cnt,
                                   "candidates": len(_candidates),
                                   "queue_size": len(_sorted)})
                    except Exception as e:
                        logger.warning(f"[POLICY:{pol.name}] start error: {e}")
                        comp._policy_evict_queue = None

                def _policy_next(tracker, device_frees, host_frees,
                                 orig=_orig_next_b, comp=_mamba_comp,
                                 pol=_policy, tc=_tc_ednn):
                    """Pop from pre-sorted queue. Fall back to native."""
                    _queue = getattr(comp, '_policy_evict_queue', None)
                    if _queue is not None:
                        try:
                            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _CT
                            _lru = tc.lru_lists.get(_CT.MAMBA)
                            _evictable = tc.evictable_device_leaves
                            _enabled = getattr(tc, 'enable_session_radix_cache', False)

                            while comp._policy_evict_idx < len(_queue):
                                _node = _queue[comp._policy_evict_idx]
                                _ph = comp._policy_evict_hashes[comp._policy_evict_idx]
                                comp._policy_evict_idx += 1
                                if _lru and _lru.in_list(_node) and _node in _evictable:
                                    _ce = True
                                    if _enabled:
                                        _ce = comp._can_evict_leaf_atomically(_node)
                                    if _ce:
                                        pol.record_eviction(_node.id, _ph)
                                        logger.info(f"[POLICY:{pol.name}] EVICT "
                                                    f"node={_node.id} hash={_ph[:8]}")
                                        kda_obs.log_event("policy_evict",
                                            virtual_time=StateManager.get_global_clock(),
                                            extra={"policy": pol.name, "node_id": _node.id,
                                                   "hash": _ph[:16], "action": "evict"})
                                        return _node.id
                            # Queue exhausted
                            comp._policy_evict_queue = None
                            logger.info(f"[POLICY:{pol.name}] queue exhausted → native fallback")
                            return orig(tracker, device_frees, host_frees)
                        except Exception as e:
                            logger.warning(f"[POLICY:{pol.name}] next error: {e}")
                            comp._policy_evict_queue = None
                            return orig(tracker, device_frees, host_frees)
                    else:
                        return orig(tracker, device_frees, host_frees)

                def _policy_end(orig=_orig_end_b, comp=_mamba_comp):
                    """Clear policy queue on eviction end."""
                    comp._policy_evict_queue = None
                    comp._policy_evict_idx = 0
                    orig()

                _mamba_comp._evict_device_start = _policy_start
                _mamba_comp._evict_device_next_node = _policy_next
                _mamba_comp._evict_device_end = _policy_end
                logger.info(f"[Phase2] Pre-sort eviction policy '{_policy.name}' "
                            f"installed on MambaComponent")
                # Hook inc_lock_ref and dec_lock_ref
                _orig_inc = getattr(_tc_ednn, 'inc_lock_ref', None)
                _orig_dec = getattr(_tc_ednn, 'dec_lock_ref', None)
                if _orig_inc is not None:
                    def _wrapped_inc(node_id, *a, **kw):
                        r = _orig_inc(node_id, *a, **kw)
                        kda_obs.log_event("inc_lock_ref", node_id=node_id,
                            virtual_time=StateManager.get_global_clock(),
                            extra={"lock_ref": r.lock_ref if hasattr(r, 'lock_ref') else '?'})
                        return r
                    _tc_ednn.inc_lock_ref = _wrapped_inc
                if _orig_dec is not None:
                    def _wrapped_dec(node_id, *a, **kw):
                        r = _orig_dec(node_id, *a, **kw)
                        kda_obs.log_event("dec_lock_ref", node_id=node_id,
                            virtual_time=StateManager.get_global_clock(),
                            extra={"lock_ref": r.lock_ref if hasattr(r, 'lock_ref') else '?'})
                        return r
                    _tc_ednn.dec_lock_ref = _wrapped_dec
                # Hook _finish_write_through_ack on the cache instance
                _orig_fwt = getattr(self, '_finish_write_through_ack', None)
                if _orig_fwt is not None:
                    def _wrapped_fwt(ack_id, orig=_orig_fwt):
                        _owt = getattr(self, 'ongoing_write_through', {})
                        _has = ack_id in _owt
                        kda_obs.log_event("fwt_ack_call",
                            virtual_time=StateManager.get_global_clock(),
                            extra={"ack_id": ack_id, "in_owt": _has,
                                   "owt_size": len(_owt)})
                        try:
                            orig(ack_id)
                            kda_obs.log_event("fwt_ack_done",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"ack_id": ack_id})
                        except Exception as e:
                            kda_obs.log_event("fwt_ack_error",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"ack_id": ack_id, "error": repr(e)[:200]})
                            raise
                    self._finish_write_through_ack = _wrapped_fwt
                # [A2] Hook cache_finished_req to verify it's called
                _orig_cfr = getattr(self, 'cache_finished_req', None)
                if _orig_cfr is not None:
                    def _wrapped_cfr(req, *a, **kw):
                        _ln = getattr(req, 'last_node', None)
                        _skip = getattr(req, 'skip_lock_node_ids', None)
                        kda_obs.log_event("cache_finished_req",
                            virtual_time=StateManager.get_global_clock(),
                            extra={"last_node": _ln, "skip_ids": str(_skip)[:100],
                                   "rid": getattr(req, 'rid', '')[:8]})
                        try:
                            r = _orig_cfr(req, *a, **kw)
                            kda_obs.log_event("cache_finished_done",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"last_node": _ln})
                            return r
                        except Exception as e:
                            kda_obs.log_event("cache_finished_error",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"last_node": _ln, "error": repr(e)[:200]})
                            raise
                    self.cache_finished_req = _wrapped_cfr

                # [ACK-LIFECYCLE] Hook _execute_and_commit_kv_backup to trace
                # write_back param, node_id, lock_ref transitions, owt transitions.
                _orig_eckb = getattr(self, '_execute_and_commit_kv_backup', None)
                if _orig_eckb is not None:
                    def _wrapped_eckb(action, write_back=False, orig=_orig_eckb):
                        _nids = list(getattr(action, 'node_ids', []) or [])
                        _owt_before = len(getattr(self, 'ongoing_write_through', {}))
                        _lr_before = {}
                        _tc_eckb = getattr(self, 'tree_core', None)
                        if _tc_eckb is not None:
                            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _ECKCT
                            for _nid in _nids:
                                try:
                                    _n = _tc_eckb.node_by_id(_nid)
                                    if _n is not None:
                                        _cd = _n.component_data.get(_ECKCT.MAMBA)
                                        _lr_before[_nid] = _cd.lock_ref if _cd else -1
                                except Exception:
                                    _lr_before[_nid] = -2
                        if kda_obs.is_obs_enabled():
                            kda_obs.log_event("eckb_call",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"node_ids": _nids, "write_back": write_back,
                                       "owt_before": _owt_before,
                                       "lr_before": str(_lr_before)})
                        _r = orig(action, write_back=write_back)
                        _owt_after = len(getattr(self, 'ongoing_write_through', {}))
                        _lr_after = {}
                        if _tc_eckb is not None:
                            for _nid in _nids:
                                try:
                                    _n = _tc_eckb.node_by_id(_nid)
                                    if _n is not None:
                                        _cd = _n.component_data.get(_ECKCT.MAMBA)
                                        _lr_after[_nid] = _cd.lock_ref if _cd else -1
                                except Exception:
                                    _lr_after[_nid] = -2
                        if kda_obs.is_obs_enabled():
                            kda_obs.log_event("eckb_done",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"node_ids": _nids, "write_back": write_back,
                                       "owt_after": _owt_after,
                                       "lr_after": str(_lr_after),
                                       "written": _r})
                        return _r
                    self._execute_and_commit_kv_backup = _wrapped_eckb

                # [ACK-LIFECYCLE] Hook _track_write_through_node to confirm
                # pending registration (ack_id == node_id).
                _orig_twt = getattr(self, '_track_write_through_node', None)
                if _orig_twt is not None:
                    def _wrapped_twt(node_id, lock_params=None, orig=_orig_twt):
                        _owt_before = len(getattr(self, 'ongoing_write_through', {}))
                        orig(node_id, lock_params)
                        _owt_after = len(getattr(self, 'ongoing_write_through', {}))
                        _has = node_id in getattr(self, 'ongoing_write_through', {})
                        if kda_obs.is_obs_enabled():
                            kda_obs.log_event("track_wt",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"node_id": node_id,
                                       "lock_params": lock_params is not None,
                                       "owt_before": _owt_before,
                                       "owt_after": _owt_after,
                                       "registered": _has})
                    self._track_write_through_node = _wrapped_twt

                # [ACK-LIFECYCLE] Hook writing_check to see when it runs,
                # with what params, and ack_wq / owt transitions.
                _orig_wc = getattr(self, 'writing_check', None)
                if _orig_wc is not None:
                    def _wrapped_wc(write_back=False, finish_count=None, orig=_orig_wc):
                        _cc_wc = getattr(self, 'cache_controller', None)
                        _wq_before = len(_cc_wc.ack_write_queue) if _cc_wc and hasattr(_cc_wc, 'ack_write_queue') else -1
                        _owt_before = len(getattr(self, 'ongoing_write_through', {}))
                        _cc_id_wc = id(_cc_wc) if _cc_wc else -1
                        _wq_list_id_wc = id(_cc_wc.ack_write_queue) if _cc_wc and hasattr(_cc_wc, 'ack_write_queue') else -1
                        if kda_obs.is_obs_enabled():
                            kda_obs.log_event("writing_check_call",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"write_back": write_back,
                                       "finish_count": finish_count,
                                       "wq_before": _wq_before,
                                       "owt_before": _owt_before,
                                       "cc_id": _cc_id_wc,
                                       "wq_list_id": _wq_list_id_wc})
                        try:
                            _r = orig(write_back=write_back, finish_count=finish_count)
                        except Exception as _wce:
                            if kda_obs.is_obs_enabled():
                                kda_obs.log_event("writing_check_error",
                                    virtual_time=StateManager.get_global_clock(),
                                    extra={"error": repr(_wce)[:200]})
                            raise
                        _wq_after = len(_cc_wc.ack_write_queue) if _cc_wc and hasattr(_cc_wc, 'ack_write_queue') else -1
                        _owt_after = len(getattr(self, 'ongoing_write_through', {}))
                        if kda_obs.is_obs_enabled():
                            kda_obs.log_event("writing_check_done",
                                virtual_time=StateManager.get_global_clock(),
                                extra={"write_back": write_back,
                                       "wq_after": _wq_after,
                                       "owt_after": _owt_after})
                        return _r
                    self.writing_check = _wrapped_wc

            # P0: Check tree_core hicache state and related params
            _tc0 = getattr(self, 'tree_core', None)
            if _tc0 is not None:
                _hicache_en = getattr(_tc0, '_hicache_enabled', None)
                if _hicache_en is None:
                    _hicache_en = getattr(_tc0, 'enable_hicache', None)
                if _hicache_en is None:
                    _hicache_en = getattr(_tc0, 'hicache_enabled', None)
                logger.info(
                    f"P0 tree_core: hicache_enabled={_hicache_en}, "
                    f"load_back_threshold={getattr(self, 'load_back_threshold', 'N/A')}, "
                    f"write_through_threshold={getattr(self, 'write_through_threshold', 'N/A')}, "
                    f"is_write_back={getattr(self, 'is_write_back', 'N/A')}, "
                    f"enable_storage={getattr(self, 'enable_storage', 'N/A')}"
                )
            else:
                logger.info("P0 tree_core is None!")

        def wrapped_count_ready_acks(self, ack_queue):
            try:
                _n = len(ack_queue)
                if kda_obs.is_obs_enabled() and _n > 0:
                    kda_obs.log_event("count_ready_acks",
                        virtual_time=StateManager.get_global_clock(),
                        extra={"count": _n, "q_id": id(ack_queue),
                               "cc_id": id(getattr(self, 'cache_controller', None)),
                               "q_type": type(ack_queue).__name__})
                return _n
            except TypeError:
                return 0

        def wrapped_check_hicache_events(self, *args, **kwargs):
            _cn = getattr(self, '_hicache_cn', 0) + 1
            self._hicache_cn = _cn
            # [ACK-LIFECYCLE-FIX] Always run original_check_hicache_events
            # (native SGLang runs it every iteration). The previous cn%10 gate
            # delayed writing_check (which processes ACKs and releases backup
            # locks). When slots are full, the scheduler can't schedule new
            # requests, and the delayed ACK processing causes a deadlock:
            # slots full → can't schedule → ACKs not processed → locks not
            # released → slots stay full.
            # Logging is gated to cn%100 to reduce output volume.
            _log_diag = (_cn % 100 == 0)
            try:
                if _log_diag:
                    _cc_diag = getattr(self, 'cache_controller', None)
                    _wq_size = len(_cc_diag.ack_write_queue) if _cc_diag and hasattr(_cc_diag, 'ack_write_queue') else -1
                    _owt_size = len(getattr(self, 'ongoing_write_through', {}))
                    if kda_obs.is_obs_enabled():
                        kda_obs.log_event("pre_check_hicache",
                            virtual_time=StateManager.get_global_clock(),
                            extra={"ack_wq": _wq_size, "owt": _owt_size, "cn": _cn})
                original_check_hicache_events(self, *args, **kwargs)
                if _log_diag:
                    _cc_diag = getattr(self, 'cache_controller', None)
                    _wq_after = len(_cc_diag.ack_write_queue) if _cc_diag and hasattr(_cc_diag, 'ack_write_queue') else -1
                    _owt_after = len(getattr(self, 'ongoing_write_through', {}))
                    if kda_obs.is_obs_enabled():
                        kda_obs.log_event("post_check_hicache",
                            virtual_time=StateManager.get_global_clock(),
                            extra={"ack_wq": _wq_after, "owt": _owt_after, "cn": _cn})
            except Exception as e:
                    import traceback as _tb
                    if kda_obs.is_obs_enabled():
                        kda_obs.log_event("check_hicache_error",
                            virtual_time=StateManager.get_global_clock(),
                            extra={"error": repr(e)[:200], "tb": _tb.format_exc()[-400:]})
                    logger.debug(f"check_hicache_events partial failure: {e}")
            # Proactive eviction: self IS the UnifiedRadixCache instance.
            # [P2C.3.2-FIX] Same truthiness bug as HISIM_DISABLE_L3: "0" is truthy.
            _dpe_val = os.environ.get("HISIM_DISABLE_PROACTIVE_EVICT", "").strip().lower()
            _disable_proactive_evict = _dpe_val in ("1", "true", "yes", "on")
            if not _disable_proactive_evict:
                try:
                    _es = self.evictable_size() if hasattr(self, 'evictable_size') else 0
                except Exception:
                    _es = 0
                _ms = 0
                try:
                    _ms = self.mamba_evictable_size() if hasattr(self, 'mamba_evictable_size') else 0
                except Exception:
                    _ms = 0
                # [ACK-LIFECYCLE-FIX] Use native evictable counter as primary
                # decision source. HiSim counter is diagnostic only.
                # Previously HiSim counter replaced native due to timing mismatch,
                # but now that ACK lifecycle is fixed (check_hicache_events runs
                # every iteration), the native counter is reliable.
                _hism_ms = kda_obs.get_mamba_evictable()
                if kda_obs.is_obs_enabled() and _cn % 100 == 0:
                    kda_obs.log_event("proactive_evict_check",
                        virtual_time=StateManager.get_global_clock(),
                        extra={"es": _es, "ms": _ms, "hism_ms": _hism_ms, "cn": _cn})
                if _es > 0 or _ms > 0:
                    try:
                        from sglang.srt.mem_cache.base_prefix_cache import EvictParams
                        self.evict(EvictParams(num_tokens=_es, mamba_num=_ms))
                        logger.info(
                            f"Proactive evict (check_hicache): tokens={_es} mamba={_ms}, "
                            f"L1_avail={self.token_to_kv_pool_allocator.available_size() if hasattr(self, 'token_to_kv_pool_allocator') else -1}, "
                            f"cn={_cn}"
                        )
                    except Exception as e:
                        logger.warning(f"Proactive evict failed: {e}")

        target.init_hicache = wrapped_init_hicache
        target.check_hicache_events = wrapped_check_hicache_events
        target._count_ready_acks = wrapped_count_ready_acks
        target.reset = wrapped_reset

        # P2: Observe match_prefix result fields for L2 load-back diagnosis.
        # Do NOT modify host_hit_length — just log for diagnosis.
        # [v2-fix] Also store per-rid match result for Phase 1 metrics.
        original_match_prefix = target.match_prefix
        def wrapped_match_prefix(self, params, orig=original_match_prefix):
            result = orig(self, params)
            try:
                _di = result.device_indices
                _dev = len(_di) if _di is not None else 0
                _full = result.full_kv_hit_length or 0
                _host = getattr(result, 'host_hit_length', 0) or 0
                _mhost = getattr(result, 'mamba_host_hit_length', None)
                _ldn = getattr(result, 'last_device_node', None)
                _lhn = getattr(result, 'last_host_node', None)
                _bmn = getattr(result, 'best_match_node', None)
                logger.info(
                    f"P2 match_prefix: full={_full} dev={_dev} host={_host} "
                    f"mamba_host={_mhost} last_dev={_ldn} last_host={_lhn} "
                    f"best_match={_bmn}({type(_bmn).__name__})"
                )
                # [v2-fix A1] Store per-rid match result for Phase 1 metrics
                if kda_obs.is_obs_enabled():
                    _rid = getattr(params, 'rid', None) or str(getattr(params, 'request_id', ''))
                    if _rid:
                        kda_obs.store_match_result(
                            rid=_rid,
                            full_kv_hit=_full,
                            device_hit=_dev,
                            host_hit=_host,
                            mamba_host_hit=int(_mhost) if _mhost is not None else 0,
                            best_match_node=int(_bmn) if _bmn is not None else -1,
                        )
                # [Phase2] Record access for HotnessAware policy
                if _bmn is not None:
                    try:
                        from hisim.simulation import kda_policy as _kp
                        from hisim.simulation.kda_policy import _node_prefix_identity
                        _pol = _kp.get_policy()
                        _vt = StateManager.get_global_clock()
                        _node = self.tree_core.node_by_id(_bmn) if hasattr(self, 'tree_core') else None
                        if _node is not None:
                            _ident = _node_prefix_identity(_node)
                            if _ident:
                                _pol.record_access(int(_bmn), _ident, _vt)
                            # [P2C.2-FIX] Also record for all ANCESTOR nodes
                            # with MAMBA checkpoints. When best_match is a deep
                            # leaf, the ancestor checkpoints are implicitly used
                            # (SSM state at ancestor is the replay starting point).
                            # When best_match is a shallow ancestor (e.g. 32K
                            # shared root), this records the ancestor itself.
                            try:
                                from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _RACT
                                _ct_m = _RACT.MAMBA
                                _cur = getattr(_node, 'parent', None)
                                _anc_count = 0
                                while _cur is not None:
                                    _cd = getattr(_cur, 'component_data', None)
                                    if _cd is not None:
                                        _mcd = _cd.get(_ct_m) if hasattr(_cd, 'get') else None
                                        if _mcd is not None and getattr(_mcd, 'value', None) is not None:
                                            _a_ident = _node_prefix_identity(_cur)
                                            if _a_ident:
                                                _pol.record_access(getattr(_cur, 'id', -1), _a_ident, _vt)
                                                _anc_count += 1
                                    _cur = getattr(_cur, 'parent', None)
                                if _anc_count > 0 and _kp.logger.isEnabledFor(10):
                                    _kp.logger.debug(
                                        f"[RECORD-ACCESS] bmn={_bmn} ident={_ident[:8]} "
                                        f"ancestors_recorded={_anc_count}")
                            except Exception as _ae:
                                pass
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"P2 match_prefix error: {e}")
            return result
        target.match_prefix = wrapped_match_prefix
        logger.debug("Wrapped match_prefix to compute host_hit_length")

        # [A2] Hook evict() to trace the full eviction chain
        original_evict = getattr(target, 'evict', None)
        if original_evict is not None and kda_obs.is_obs_enabled():
            def wrapped_evict(self, params, orig=original_evict):
                _nt = getattr(params, 'num_tokens', 0)
                _mn = getattr(params, 'mamba_num', 0)
                _rtp = getattr(self, 'req_to_token_pool', None)
                _ma = getattr(_rtp, 'mamba_allocator', None) if _rtp else None
                _before_avail = _ma.available_size() if _ma else -1
                # [A2] Log tree_components to check if MAMBA is included
                _tc_ev = getattr(self, 'tree_core', None)
                _tcs = []
                if _tc_ev is not None:
                    try:
                        _tcs = [getattr(ct, 'value', str(ct)) for ct in _tc_ev.component_types]
                    except Exception:
                        pass
                kda_obs.log_event("evict_call",
                    virtual_time=StateManager.get_global_clock(),
                    extra={"num_tokens": _nt, "mamba_num": _mn,
                           "mamba_avail_before": _before_avail,
                           "tree_components": _tcs})
                result = orig(self, params)
                _after_avail = _ma.available_size() if _ma else -1
                _evicted_mamba = getattr(result, 'mamba_num_evicted', 0)
                _evicted_tokens = getattr(result, 'num_tokens_evicted', 0)
                kda_obs.log_event("evict_done",
                    virtual_time=StateManager.get_global_clock(),
                    extra={"mamba_avail_after": _after_avail,
                           "mamba_evicted": _evicted_mamba,
                           "tokens_evicted": _evicted_tokens,
                           "avail_delta": _after_avail - _before_avail})
                return result
            target.evict = wrapped_evict
            logger.debug("Wrapped evict for eviction chain tracing")

            # [A2] Hook tree_core.evict_device_next_node to trace cursor state
            # tree_core is an instance attribute, not a class attribute — use getattr
            _tc_cls = getattr(target, 'tree_core', None)
            _orig_ednn = getattr(_tc_cls, 'evict_device_next_node', None) if _tc_cls else None
            if _orig_ednn is not None and kda_obs.is_obs_enabled():
                def wrapped_ednn(self_tc, component_type, tracker, orig=_orig_ednn):
                    _ct_val = getattr(component_type, 'value', str(component_type))
                    result = orig(self_tc, component_type, tracker)
                    _nid = result.node_id if result.node_id else -1
                    _dev_frees = {str(k): len(v) for k, v in result.device_frees.items()} if result.device_frees else {}
                    _tracker_delta = dict(result.tracker) if result.tracker else {}
                    kda_obs.log_event("ednn_trace",
                        virtual_time=StateManager.get_global_clock(),
                        extra={"ct": _ct_val, "node_id": _nid,
                               "made_progress": result.made_progress,
                               "device_frees": _dev_frees,
                               "tracker_delta": _tracker_delta})
                    return result
                target.tree_core.evict_device_next_node = wrapped_ednn

        # [v2-fix A1] Hook init_load_back to verify L2→L1 restoration timing
        original_init_load_back = getattr(target, 'init_load_back', None)
        if original_init_load_back is not None:
            def wrapped_init_load_back(self, params, orig=original_init_load_back):
                _rid = ""
                _bmn = -1
                _hhl = 0
                _mhl = 0
                try:
                    _req = getattr(params, 'req', None)
                    if _req is not None:
                        _rid = getattr(_req, 'rid', '')
                    _bmn = getattr(params, 'best_match_node', -1) or -1
                    _hhl = getattr(params, 'host_hit_length', 0) or 0
                    _mhl = getattr(_req, 'mamba_host_hit_length', 0) or 0 if _req is not None else 0
                except Exception:
                    pass
                # Pre-call: check node MAMBA state
                _pre_mamba_val = None
                _pre_mamba_hval = None
                try:
                    _tc = getattr(self, 'tree_core', None)
                    if _tc is not None and _bmn >= 0:
                        _node = _tc.node_by_id(_bmn)
                        if _node is not None:
                            _cd = _node.component_data
                            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _CT
                            _mc = _cd[_CT.MAMBA] if len(_cd) > _CT.MAMBA else None
                            if _mc is not None:
                                _pre_mamba_val = "set" if _mc.value is not None else "None"
                                _pre_mamba_hval = "set" if _mc.host_value is not None else "None"
                except Exception:
                    pass
                result = orig(self, params)
                # Post-call: check node MAMBA state
                _post_mamba_val = None
                _post_indices_len = 0
                try:
                    _tc = getattr(self, 'tree_core', None)
                    if _tc is not None and _bmn >= 0:
                        _node = _tc.node_by_id(_bmn)
                        if _node is not None:
                            _cd = _node.component_data
                            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _CT
                            _mc = _cd[_CT.MAMBA] if len(_cd) > _CT.MAMBA else None
                            if _mc is not None:
                                _post_mamba_val = "set" if _mc.value is not None else "None"
                    _post_indices_len = len(result[0]) if result and result[0] is not None else 0
                except Exception:
                    pass
                if kda_obs.is_obs_enabled():
                    # Determine KDA tier from pre-call state
                    if _bmn < 0:
                        _kda_tier = "miss"
                    elif _pre_mamba_val == "set":
                        _kda_tier = "L1"
                    elif _pre_mamba_hval == "set":
                        _kda_tier = "L2"
                    else:
                        _kda_tier = "L3"
                    # Store match result (init_load_back has rid, unlike match_prefix)
                    if _bmn >= 0:
                        kda_obs.store_match_result_minimal(
                            _rid, _bmn, host_hit=_hhl, mamba_host_hit=_mhl)
                        # [A2] For L1 hit, mark shadow as ready (no transfer needed)
                        if _kda_tier == "L1":
                            kda_obs.shadow_mark_ready(f"kda_{_bmn}_0", tier="L1")
                    kda_obs.update_match_result(_rid, "init_load_back_tier", _kda_tier)
                    kda_obs.log_event(
                        "init_load_back", rid=_rid,
                        virtual_time=StateManager.get_global_clock(),
                        node_id=_bmn,
                        extra={"host_hit": _hhl, "mamba_host_hit": _mhl,
                               "pre_mamba_value": _pre_mamba_val,
                               "pre_mamba_host_value": _pre_mamba_hval,
                               "post_mamba_value": _post_mamba_val,
                               "restored_indices": _post_indices_len,
                               "kda_tier": _kda_tier},
                    )
                    logger.info(
                        f"[ILB] rid={_rid[:8]} node={_bmn} tier={_kda_tier} "
                        f"host_hit={_hhl} mamba_host_hit={_mhl} "
                        f"pre_val={_pre_mamba_val} pre_hval={_pre_mamba_hval} "
                        f"post_val={_post_mamba_val} restored={_post_indices_len}"
                    )
                return result
            target.init_load_back = wrapped_init_load_back
            logger.debug("Wrapped init_load_back for L2 restoration tracking")

        # HiSim runs as a single CPU process (TP=1): no cross-rank distributed
        # all_reduce is needed. The real _all_reduce can hang inside
        # transfer_to_npu/torch.distributed in CPU simulation during
        # check_hicache_events -> _sync_hicache_ready_counts. Override to no-op.
        target._all_reduce = lambda self, data, tp_reduce_op: data
        logger.debug("Overrode UnifiedRadixCache._all_reduce to no-op (single-proc CPU)")

        logger.debug("C_HiRadixCacheHook applied to UnifiedRadixCache")


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
    # Cross-phase accumulator for the canonical metrics.json consumed by
    # bench_serving.load_simulation_metrics(). REQUEST_STATS is cleared after
    # every phase dump (so metrics_phaseN.json stay per-phase), but this list
    # keeps raw RequestStats copies so the canonical file aggregates all phases.
    ALL_REQUEST_STATS: list[RequestStats] = []
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

    # L2/L3 cache hit statistics
    _L3_HIT_STATS: dict = {}  # tier_name -> total hit blocks
    _L3_QUERY_COUNT: int = 0  # total batch_exists calls
    _L3_TOTAL_HITS: int = 0  # total hit blocks across all tiers
    _L3_PREFETCHED_TOKENS: int = 0  # total tokens actually prefetched from L3
    _L2_HIT_TOKENS: int = 0  # total tokens matched in L2 (host_hit_length sum)
    _L2_HIT_COUNT: int = 0  # number of requests with host_hit_length > 0

    SCHEDULE_REQ_STATS = []

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_recv_requests = getattr(target, "process_input_requests", None) or getattr(target, "recv_requests", None)
        original_get_new_batch_prefill = target.get_new_batch_prefill
        original_run_batch = target.run_batch
        original_process_batch_result = target.process_batch_result
        original_event_loop_normal = target.event_loop_normal

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
            try:
                setattr(server_args, "disable_overlap_schedule", True)
            except AttributeError:
                # sglang 0.5.18+ makes server_args read-only after resolution
                pass
            logger.debug(
                f"Overlap schedule simulation mode: {C_SchedulerHook.OVERLAP_SCHEDULE}."
            )

            original_init(self, *args, **kwargs)

            # sglang 0.5.18 CPU simulation: force prefetch-progress to succeed.
            # In CPU sim the L2/L3 hierarchical cache is mocked; requests get stuck
            # in `ongoing_prefetch` and are skipped by check_prefetch_progress (False),
            # keeping can_run_list==0 forever. Patch directly on the instance so the
            # forked scheduler subprocess definitely inherits it.
            try:
                import zmq as _dummy  # noqa
                _tc = getattr(self, 'tree_cache', None)
                if _tc is not None and hasattr(_tc, 'check_prefetch_progress'):
                    _tc.check_prefetch_progress = lambda req_id: True
                    _tc.pop_prefetch_loaded_tokens = lambda req_id: 0
                    logger.info("Patched tree_cache.check_prefetch_progress in scheduler")
            except Exception:
                pass

            # sglang 0.5.18 compat: force disable overlap schedule
            # (setattr on server_args fails because 0.5.18 is read-only)
            self.enable_overlap = False
            import queue as _queue
            if not hasattr(self, "result_queue"):
                self.result_queue = _queue.Queue()

            # sglang 0.5.18 compat: ensure ngram_embedding_manager is not None.
            # CRITICAL: prepare_for_forward must return the passed-in batch, NOT
            # None. scheduler.get_next_batch_to_run does
            #   ret = ngram_embedding_manager.prepare_for_forward(ret, ...)
            # which overwrites the batch with the return value; returning None
            # nulls the batch -> event_loop takes on_idle() -> scheduler hang.
            if getattr(self, "ngram_embedding_manager", None) is None:
                class _MockNgramManager:
                    def prepare_for_forward(self, *a, **kw):
                        return a[0] if a else None
                self.ngram_embedding_manager = _MockNgramManager()

            # sglang 0.5.18 compat: disable invariant checker (mock pools
            # don't track token accounting precisely enough)
            class _NoopInvariantChecker:
                def self_check_during_busy(self, *a, **kw): pass
                def _check_all_pools(self, *a, **kw): return False, []
                def _check_req_pool(self, *a, **kw): pass
                def _check_tree_cache(self, *a, **kw): pass
                def _report_leak(self, *a, **kw): pass
            self.invariant_checker = _NoopInvariantChecker()

            # sglang 0.5.18 (docker): rebuild PULL socket to fix fork+IPC
            # connection issues after fork. OFF by default on sglang >= 0.5.19
            # where the inherited socket works fine: rebuilding re-binds the
            # endpoint while the deferred-closed old socket keeps LISTENING on
            # it, so the tokenizer's PUSH round-robins messages between the
            # live and the dead listener and silently drops half the requests.
            # Set HISIM_REBUILD_PULL_SOCKET=1 to enable (0.5.18 docker only).
            if (
                hasattr(self, "request_receiver")
                and os.getenv("HISIM_REBUILD_PULL_SOCKET", "0") == "1"
            ):
                try:
                    import zmq as _zmq
                    _rr = self.request_receiver
                    _old_sock = getattr(_rr, "recv_from_tokenizer", None)
                    if _old_sock is not None:
                        _endpoint = _old_sock.getsockopt(_zmq.LAST_ENDPOINT)
                        if isinstance(_endpoint, bytes):
                            _endpoint = _endpoint.decode()
                        logger.info(f"Rebuilding PULL socket: old endpoint={_endpoint}")
                        try:
                            # linger=0: drop the old connection immediately.
                            # A lazy close leaves the old PULL attached to the
                            # endpoint and the tokenizer's PUSH round-robins
                            # messages between the live and the dead socket,
                            # silently dropping every other request.
                            _old_sock.close(linger=0)
                        except Exception:
                            pass
                        _new_ctx = _zmq.Context(2)
                        _new_sock = _new_ctx.socket(_zmq.PULL)
                        # The old socket owned (bound) the endpoint — re-bind
                        # it, otherwise the tokenizer's PUSH has no listener
                        # and everything after the first few in-flight
                        # messages is silently queued forever.
                        for _attempt in range(20):
                            try:
                                _new_sock.bind(_endpoint)
                                break
                            except _zmq.ZMQError:
                                time.sleep(0.1)
                        object.__setattr__(_rr, "recv_from_tokenizer", _new_sock)
                        logger.info(f"PULL socket rebuilt and bound to {_endpoint}")
                except Exception as e:
                    logger.warning(f"PULL socket rebuild failed: {e}")

            # sglang 0.5.18: request fetching moved from process_input_requests
            # to request_receiver.recv_requests(). Hook the receiver for OFFLINE
            # FUTURE_QUEUE logic.
            if hasattr(self, "request_receiver"):
                from sglang.srt.managers.scheduler_components.request_receiver import SchedulerRequestReceiver
                _orig_recv = SchedulerRequestReceiver.recv_requests

                def _wrapped_recv_requests(self_rr, *args, **kwargs):
                    raw_reqs = _orig_recv(self_rr, *args, **kwargs)
                    if raw_reqs is None:
                        raw_reqs = []

                    if os.getenv("HISIM_DEBUG_RECV"):
                        logger.info(
                            f"[RECV] {len(raw_reqs)} reqs: "
                            f"{[r.__class__.__name__ for r in raw_reqs]}"
                        )

                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        return raw_reqs

                    # OFFLINE mode: collect into FUTURE_QUEUE, then release
                    if not C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST:
                        gen_requests = []
                        extra_requests = []
                        for req in raw_reqs:
                            if req.__class__.__name__ == "TokenizedGenerateReqInput":
                                sim_params = None
                                if req.sampling_params.custom_params is not None:
                                    sim_params = req.sampling_params.custom_params.get("simulation")
                                if sim_params is None:
                                    extra_requests.append(req)
                                    continue
                                gen_requests.append(req)
                                C_SchedulerHook.FUTURE_QUEUE.append(
                                    (
                                        sim_params.get("queue_start") or sim_params["created_time"],
                                        time.time_ns(),
                                        req,
                                    )
                                )
                            else:
                                extra_requests.append(req)

                        if len(C_SchedulerHook.FUTURE_QUEUE) != 0:
                            _, _, gen_req = C_SchedulerHook.FUTURE_QUEUE[-1] if len(C_SchedulerHook.FUTURE_QUEUE) > 0 else (None, None, None)
                            if gen_req is not None:
                                total_request = gen_req.sampling_params.custom_params[
                                    "simulation"
                                ]["total_request"]
                                if len(C_SchedulerHook.FUTURE_QUEUE) == total_request:
                                    C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = True
                                    # New run: reset the cross-phase canonical
                                    # accumulator and phase counter.
                                    C_SchedulerHook.ALL_REQUEST_STATS = []
                                    C_SchedulerHook._offline_phase = 0
                                    heapq.heapify(C_SchedulerHook.FUTURE_QUEUE)
                                    logger.info("All requests received. Starting simulation now.")
                                else:
                                    logger.info(
                                        f"Offline simulation mode enabled. {total_request} requests expected. "
                                        f"Received {len(C_SchedulerHook.FUTURE_QUEUE)} so far."
                                    )
                        return extra_requests
                    else:
                        # Release GENERATE requests from FUTURE_QUEUE based on
                        # the virtual clock.
                        current_timestamp = StateManager.get_global_clock()
                        released = []
                        while len(C_SchedulerHook.FUTURE_QUEUE) > 0:
                            enqueue_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                            if enqueue_time > current_timestamp:
                                break
                            released.append(req)
                            heapq.heappop(C_SchedulerHook.FUTURE_QUEUE)
                        # Always forward CONTROL messages (ProfileReq,
                        # AbortReq, flush, ...). They are not part of the
                        # virtual-clock schedule, and once
                        # OFFLINE_RECV_ALL_REQUEST is set they must NOT be
                        # dropped: bench_serving issues a second
                        # /start_profile right after the workload, and if it
                        # lands in the window before the auto-trigger resets
                        # the flag it would otherwise vanish and the client
                        # would block forever waiting for a response.
                        extra_requests = [
                            r
                            for r in raw_reqs
                            if r.__class__.__name__
                            not in (
                                "TokenizedGenerateReqInput",
                                "BatchTokenizedGenerateReqInput",
                            )
                        ]
                        if os.getenv("HISIM_DEBUG_RECV") and extra_requests:
                            logger.info(
                                f"[RECV-CTRL] forwarding "
                                f"{[r.__class__.__name__ for r in extra_requests]} "
                                f"(released={len(released)})"
                            )
                        return released + extra_requests

                # Patch at class level (not instance level, since msgspec.Struct
                # doesn't allow instance setattr)
                SchedulerRequestReceiver.recv_requests = _wrapped_recv_requests
                logger.info("Hooked SchedulerRequestReceiver.recv_requests for OFFLINE mode")

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
            recv_reqs = args[0] if args else kwargs.get("recv_reqs", [])
            if recv_reqs is None:
                recv_reqs = []
            if recv_reqs:
                pass

            # Note: K3 Mamba track attributes are set on Req objects (not
            # TokenizedGenerateReqInput) in wrapped_get_new_batch_prefill,
            # because TokenizedGenerateReqInput is msgspec.Struct (frozen,
            # cannot setattr). Req is a regular Python class (can setattr).

            # Initialize REQUEST_STATS
            now = time.time()
            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    if req.rid not in C_SchedulerHook.REQUEST_STATS:
                        C_SchedulerHook.REQUEST_STATS[req.rid] = RequestStats()
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.rid = req.rid
                    req_stats.input_length = len(req.input_ids)
                    req_stats.output_length = req.sampling_params.max_new_tokens
                    _cp = getattr(req.sampling_params, 'custom_params', None) or {}
                    simulation_args = _cp.get("simulation", {}) if isinstance(_cp, dict) else {}
                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        req_stats.created_time = simulation_args.get("server_created_time", now)
                        req_stats.last_event_time = req_stats.created_time
                        req_stats.queue_start = now
                    elif C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                        req_stats.created_time = simulation_args.get("created_time", 0)
                        req_stats.last_event_time = req_stats.created_time
                        queue_start = simulation_args.get("queue_start")
                        if queue_start is not None:
                            StateManager.set_global_clock(queue_start)
                        req_stats.queue_start = StateManager.get_global_clock()
                        if C_SchedulerHook.LAST_CPU_TS == 0:
                            C_SchedulerHook.LAST_CPU_TS = time.time()
                            StateManager.set_global_clock(0)

            if recv_reqs and C_SchedulerHook.LAST_CPU_TS == 0:
                C_SchedulerHook.LAST_CPU_TS = time.time()
                StateManager.set_global_clock(0)

            # Call original process_input_requests to process the requests
            try:
                original_recv_requests(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"process_input_requests error: {e}", exc_info=True)

            return recv_reqs

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            # Do NOT initialize mamba_ping_pong_track_buffer here.
            # sglang's alloc() checks `if req.mamba_ping_pong_track_buffer is None:`
            # to trigger _alloc_ping_pong_buffer, which allocates real Mamba
            # slot IDs from mamba_allocator. Setting [-1,-1] prevents that.

            # Proactive eviction now runs in wrapped_check_hicache_events
            # (after process_batch_result unlocks nodes, before next match_prefix).
            _tc = getattr(self, 'tree_cache', None)
            _hicache_cn = getattr(_tc, '_hicache_cn', 0) if _tc is not None else 0

            # OFFLINE mode: release requests from FUTURE_QUEUE based on
            # virtual clock timestamp. Requests are released when their
            # arrival_time <= current virtual clock, regardless of whether
            # a batch is running. The scheduler decides batching policy.
            if (C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE
                and C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST
                and len(C_SchedulerHook.FUTURE_QUEUE) > 0):
                import heapq
                _now = StateManager.get_global_clock()
                # If scheduler is idle (no running, no waiting) and next
                # request hasn't arrived yet, advance clock to next request
                if (len(self.running_batch.reqs) == 0
                    and len(self.waiting_queue) == 0):
                    _next_time = C_SchedulerHook.FUTURE_QUEUE[0][0]
                    if _next_time > _now:
                        StateManager.set_global_clock(_next_time + 1e-6)
                        _now = StateManager.get_global_clock()
                # Release all requests whose arrival_time <= clock
                _released = 0
                _to_process = []
                while len(C_SchedulerHook.FUTURE_QUEUE) > 0:
                    _et, _, _req = C_SchedulerHook.FUTURE_QUEUE[0]
                    if _et > _now:
                        break
                    heapq.heappop(C_SchedulerHook.FUTURE_QUEUE)
                    _to_process.append(_req)
                    _released += 1
                if _to_process:
                    # Process through the normal scheduler path to convert
                    # TokenizedGenerateReqInput → Req (handle_generate_request)
                    # before adding to waiting_queue. Direct append bypasses
                    # the conversion and causes AttributeError when MLA backend
                    # calls req.init_next_round_input().
                    try:
                        self.process_input_requests(_to_process)
                    except Exception as e:
                        logger.error(f"OFFLINE release process_input_requests error: {e}", exc_info=True)
                    logger.info(
                        f"OFFLINE release: {_released} reqs at clock={_now:.4f}, "
                        f"waiting={len(self.waiting_queue)}, "
                        f"future={len(C_SchedulerHook.FUTURE_QUEUE)}"
                    )

            # OFFLINE release done above — let scheduler decide batching.
            # No prefill preemption guard: the scheduler's native batching
            # policy (enable_mixed_chunk, etc.) determines concurrency.

            # [ACK-LIFECYCLE-FIX] force_evict is a HiSim-specific proactive
            # eviction safety net. It was added as a workaround for the slot
            # exhaustion deadlock (caused by the cn%10 gate delaying ACK
            # processing). Now that check_hicache_events runs every iteration,
            # native capacity-driven eviction (inside alloc() when pool is full)
            # handles slot pressure. force_evict is gated by
            # HISIM_DISABLE_PROACTIVE_EVICT so it can be turned off for clean
            # baselines. It is NOT native scheduler capacity eviction — that
            # happens inside _tc.evict() / alloc() when the pool is full.
            # [P2C.3.2-FIX] Same truthiness bug as HISIM_DISABLE_L3.
            _dpe_val = os.environ.get("HISIM_DISABLE_PROACTIVE_EVICT", "").strip().lower()
            _disable_proactive_evict = _dpe_val in ("1", "true", "yes", "on")
            if not _disable_proactive_evict and kda_obs.is_obs_enabled():
                try:
                    _rtp = getattr(self, 'req_to_token_pool', None)
                    _tc = getattr(self, 'tree_cache', None)
                    _wq = getattr(self, 'waiting_queue', None)
                    if _rtp is not None and _tc is not None and _wq and len(_wq) > 0:
                        _ma = getattr(_rtp, 'mamba_allocator', None)
                        if _ma is not None and _ma.available_size() < 4:
                            # Use native evictable counter as primary (consistent
                            # with wrapped_check_hicache_events). HiSim counter
                            # is diagnostic only.
                            _mes = _tc.mamba_evictable_size() if hasattr(_tc, 'mamba_evictable_size') else 0
                            _hism_ms = kda_obs.get_mamba_evictable()
                            if kda_obs.is_obs_enabled():
                                kda_obs.log_event(
                                    "force_evict_check",
                                    virtual_time=StateManager.get_global_clock(),
                                    extra={"mamba_avail": _ma.available_size(),
                                           "mamba_evictable_native": _mes,
                                           "mamba_evictable_hism": _hism_ms,
                                           "waiting": len(_wq)})
                            if _mes > 0:
                                from sglang.srt.mem_cache.base_prefix_cache import EvictParams
                                _tc.evict(EvictParams(num_tokens=0, mamba_num=_mes))
                                logger.info(
                                    f"[A2-FORCE-EVICT] mamba_avail={_ma.available_size()} "
                                    f"< 4, evicted {_mes} mamba slots (native cnt), "
                                    f"hism_cnt={_hism_ms}, waiting={len(_wq)}"
                                )
                                if kda_obs.is_obs_enabled():
                                    kda_obs.log_event(
                                        "force_evict", virtual_time=StateManager.get_global_clock(),
                                        extra={"mamba_avail_before": _ma.available_size() + _mes,
                                               "evicted": _mes, "waiting": len(_wq)})
                except Exception as e:
                    logger.debug(f"[A2-FORCE-EVICT] error: {e}")

            new_batch = original_get_new_batch_prefill(self, *args, **kwargs)
            now = time.time()
            # sglang 0.5.18: returns NextBatchPlan with .batch_to_run
            # older sglang: returns ScheduleBatch directly
            actual_batch = getattr(new_batch, "batch_to_run", new_batch)
            if actual_batch is not None:
                for req in actual_batch.reqs:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.final_reused_tokens = (
                        req.cached_tokens
                        if getattr(req, 'cached_tokens', 0) > 0
                        else len(req.prefix_indices)
                    )
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
                        # Manually record L2 load_back (H2D) timing, since the
                        # real load() may fail when device pool is full (alloc
                        # returns None). This ensures l2_load_latency is recorded
                        # whenever host_hit is detected.
                        tc = getattr(self, 'tree_cache', None)
                        if tc is not None and hasattr(tc, 'full_kv_pool_host'):
                            try:
                                import numpy as np
                                tc.full_kv_pool_host.load_to_device_per_layer(
                                    tc.full_kv_pool_host.device_pool,
                                    torch.arange(host_hit, dtype=torch.int64),
                                    torch.arange(host_hit, dtype=torch.int64),
                                    0, "kernel",
                                )
                            except Exception:
                                pass
                        # KDA/Mamba L2→L1 load_back is handled by
                        # _mock_start_loading, which uses the load op's
                        # node_ids to resolve the correct per-node hash
                        # (kda_<node>_<i>). Do NOT fish arbitrary LRU keys out
                        # of the L2 pool: that restored unrelated checkpoints
                        # and double-counted the H2D timing.
                    # [v2-fix A1] Direct node MAMBA state check after match+init_load_back
                    # This provides runtime evidence (not just source code tracing) that
                    # init_load_back restored MAMBA value before set_extend_range.
                    if kda_obs.is_obs_enabled():
                        _tc = getattr(self, 'tree_cache', None)
                        _tc_core = getattr(_tc, 'tree_core', None) if _tc is not None else None
                        for req in actual_batch.reqs:
                            _mr = kda_obs.get_match_result(req.rid)
                            _bmn = _mr.get("best_match_node", -1)
                            if _tc_core is not None and _bmn >= 0:
                                try:
                                    _node = _tc_core.node_by_id(_bmn)
                                    if _node is not None:
                                        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as _CT
                                        _cd = _node.component_data
                                        _mc = _cd[_CT.MAMBA] if len(_cd) > _CT.MAMBA else None
                                        _fv = _cd[_CT.FULL] if len(_cd) > _CT.FULL else None
                                        _mamba_val = "set" if (_mc is not None and _mc.value is not None) else "None"
                                        _mamba_hval = "set" if (_mc is not None and _mc.host_value is not None) else "None"
                                        _full_val = "set" if (_fv is not None and _fv.value is not None) else "None"
                                        _full_hval = "set" if (_fv is not None and _fv.host_value is not None) else "None"
                                        _er = getattr(req, 'extend_range', None)
                                        _ext_len = getattr(_er, 'length', 0) if _er else 0
                                        _prefix_len = len(getattr(req, 'prefix_indices', []))
                                        _cow_src = req_mamba_get(req, 'mamba_cow_src_index', None)
                                        logger.info(
                                            f"[NODE-STATE] rid={req.rid[:8]} node={_bmn} "
                                            f"MAMBA val={_mamba_val} hval={_mamba_hval} "
                                            f"FULL val={_full_val} hval={_full_hval} "
                                            f"prefix={_prefix_len} extend={_ext_len} "
                                            f"cow_src={'set' if _cow_src is not None else 'None'}"
                                        )
                                        kda_obs.log_event(
                                            "node_state", rid=req.rid,
                                            virtual_time=StateManager.get_global_clock(),
                                            node_id=_bmn,
                                            extra={"mamba_value": _mamba_val,
                                                   "mamba_host_value": _mamba_hval,
                                                   "full_value": _full_val,
                                                   "full_host_value": _full_hval,
                                                   "prefix_len": _prefix_len,
                                                   "extend_len": _ext_len,
                                                   "cow_src": 'set' if _cow_src is not None else 'None'},
                                        )
                                        # [A2] Shadow forward_consume + tier hit
                                        _slot_hash = f"kda_{_bmn}_0"
                                        _ver = kda_obs.shadow_verify(req.rid, _slot_hash)
                                        kda_obs.shadow_mark_forward_consume(req.rid, _slot_hash)
                                        # Determine KDA tier hit from init_load_back event
                                        _ilb = kda_obs.get_match_result(req.rid).get("init_load_back_tier", "")
                                        if _mamba_val == "set" and _mamba_hval == "set" and not _ilb:
                                            _kda_tier = "L1"
                                        elif _ilb == "L2":
                                            _kda_tier = "L2"
                                        elif _ilb == "L3":
                                            _kda_tier = "L3"
                                        elif _prefix_len > 0:
                                            _kda_tier = "unknown"
                                        else:
                                            _kda_tier = "miss"
                                        logger.info(
                                            f"[FWD-CONSUME] rid={req.rid[:8]} tier={_kda_tier} "
                                            f"identity_ok={_ver.get('identity_ok',False)} "
                                            f"ready={_ver.get('ready',False)} "
                                            f"shadow_state={_ver.get('state','n/a')}"
                                        )
                                except Exception as e:
                                    logger.debug(f"[NODE-STATE] error: {e}")
                    # KDA COW/Replay diagnostics: check mamba_cow_src_index and
                    # mamba_branching_seqlen set during init_next_round_input→match_prefix
                    for req in actual_batch.reqs:
                        _cow_src = req_mamba_get(req, 'mamba_cow_src_index', None)
                        _branching = getattr(req, 'mamba_branching_seqlen', None)
                        _prefix_len = len(getattr(req, 'prefix_indices', []))
                        _mpi = req_mamba_get(req, 'mamba_pool_idx', None)
                        if _prefix_len > 0 and _mpi is not None:
                            _kda_diag = getattr(C_SchedulerHook, '_kda_checkpoint_diag', None)
                            if _kda_diag is None:
                                _kda_diag = {'donate_count':0,'commit_count':0,'hit_count':0,'cow_exec_count':0,'checkpoint_position':0,'prefix_length':0,'historical_replay_tokens':0,'new_tokens':0,'no_cow_count':0,'replay_fallback_count':0}
                                C_SchedulerHook._kda_checkpoint_diag = _kda_diag
                            if _cow_src is not None:
                                _kda_diag['hit_count'] += 1
                                _kda_diag['cow_exec_count'] += 1
                                _kda_diag['checkpoint_position'] = _prefix_len
                                _kda_diag['prefix_length'] = _prefix_len
                                # [v2-fix A1] COW hit: replay=0, no accumulation here
                                # (actual accumulation done in COW-EXEC-SIM in wrapped_run_batch)
                                logger.info(f"[KDA-COW-HIT] rid={req.rid} prefix_len={_prefix_len} cow_src={_cow_src} replay_tokens=0")
                            else:
                                # No COW → replay needed
                                # [v2-fix A1] Do NOT accumulate historical_replay_tokens here.
                                # This is a preliminary check; actual accumulation is in
                                # REPLAY-FALLBACK (wrapped_run_batch) to avoid double-counting.
                                if _branching is not None:
                                    _replay = max(0, _prefix_len - _branching)
                                else:
                                    _replay = _prefix_len
                                _kda_diag['no_cow_count'] = (_kda_diag.get('no_cow_count', 0)) + 1
                                logger.info(f"[KDA-REPLAY] rid={req.rid} prefix_len={_prefix_len} branching={_branching} replay_tokens={_replay} cow_src=None")
                            # [v2-fix A1] Per-req metrics are now populated in wrapped_run_batch
                            # (COW-EXEC-SIM or REPLAY-FALLBACK) after the final COW/replay
                            # decision is made. Populating here is premature because cow_src
                            # can change between get_new_batch_prefill and run_batch.
                    # Log cache state per prefill batch (L1/L2 pool sizes)
                    tc = getattr(self, 'tree_cache', None)
                    l1_evict = l1_total = l2_avail = 0
                    _l2_mamba_cap = _l2_mamba_cached = _l2_mamba_free = _l2_mamba_lru = -1
                    if tc is not None:
                        try:
                            l1_evict = tc.evictable_size() if hasattr(tc, 'evictable_size') else 0
                        except Exception:
                            l1_evict = 0
                        cc = getattr(tc, 'cache_controller', None)
                        if cc is not None and hasattr(cc, 'mem_pool_host'):
                            l2_avail = cc.mem_pool_host.available_size()
                        elif hasattr(tc, 'full_kv_pool_host') and tc.full_kv_pool_host is not None:
                            l2_avail = tc.full_kv_pool_host.available_size()
                        l1_total = (tc.token_to_kv_pool_allocator.available_size() + l1_evict) if hasattr(tc, 'token_to_kv_pool_allocator') else -1
                        # L2 KDA (MAMBA) host pool stats — for A3/L3-only diagnosis
                        _mph = getattr(tc, 'mamba_pool_host', None)
                        if _mph is not None and hasattr(_mph, 'get_stats'):
                            try:
                                _khs = _mph.get_stats()
                                _l2_mamba_cap = _khs.get('capacity', -1)
                                _l2_mamba_cached = _khs.get('cached_slots', -1)
                                _l2_mamba_free = _khs.get('free_slots', -1)
                                _l2_mamba_lru = _khs.get('lru_entries', -1)
                            except Exception:
                                pass
                    logger.info(
                        f"Cache state: L1_evict={l1_evict} L1_total={l1_total} "
                        f"L2_avail={l2_avail} "
                        f"L2_KDA[cap={_l2_mamba_cap} cached={_l2_mamba_cached} "
                        f"free={_l2_mamba_free} lru={_l2_mamba_lru}]"
                    )
                    # Also log to kda_obs for structured analysis
                    if kda_obs.is_obs_enabled():
                        kda_obs.log_event("l2_kda_pool_state",
                            virtual_time=StateManager.get_global_clock(),
                            extra={"capacity": _l2_mamba_cap,
                                   "cached_slots": _l2_mamba_cached,
                                   "free_slots": _l2_mamba_free,
                                   "lru_entries": _l2_mamba_lru,
                                   "l1_evict": l1_evict,
                                   "l2_avail": l2_avail})
            elif len(self.running_batch.reqs) == 0 and len(self.waiting_queue) > 0:
                # Prefetching
                StateManager.step_global_clock(0.005)
                StateManager.set_current_inference_dur(0.005)
            else:
                # OFFLINE mode virtual clock advancement now handled above
                # (before original_get_new_batch_prefill) to ensure requests
                # are released in timestamp order before match_prefix.
                pass

            logger.debug(
                f"Get new batch prefill: global iteration={StateManager.get_iteration()}, "
                f"new batch={actual_batch.batch_size() if actual_batch is not None else 0}, "
                f"waiting queue={len(self.waiting_queue)}"
            )

            # OFFLINE mode: trigger metrics collection when all requests have
            # finished (running_batch empty = all decode done, waiting_queue
            # empty, FUTURE_QUEUE empty). This is checked here because
            # wrapped_get_new_batch_prefill is called every event_loop
            # iteration, including when the scheduler goes idle (batch=None).
            # wrapped_process_batch_result is NOT called when batch is None.
            # NOTE: `actual_batch is None` is essential: a freshly-built prefill
            # batch has already drained waiting_queue but is not merged into
            # running_batch yet, so the queues transiently look empty. Without
            # this guard the dump fires mid-simulation, clears REQUEST_STATS,
            # and all subsequent request latencies are silently lost (seen with
            # output_len=1 traces where the last prefill batch is the final work).
            if (
                C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE
                and C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST
                and len(C_SchedulerHook.FUTURE_QUEUE) == 0
                and len(self.waiting_queue) == 0
                and (self.running_batch is None or len(self.running_batch.reqs) == 0)
                and getattr(self, "chunked_req", None) is None
                and actual_batch is None
                and len(C_SchedulerHook.REQUEST_STATS) > 0
            ):
                _rb = getattr(self, "running_batch", None)
                _rb_count = len(_rb.reqs) if _rb else 0
                _wq = len(getattr(self, "waiting_queue", []))
                _fq = len(C_SchedulerHook.FUTURE_QUEUE)
                logger.info(
                    f"[AUTO-TRIGGER] running={_rb_count} waiting={_wq} future={_fq} "
                    f"actual_batch={'None' if actual_batch is None else 'set'} "
                    f"req_stats={len(C_SchedulerHook.REQUEST_STATS)}"
                )
                _phase = getattr(C_SchedulerHook, "_offline_phase", 0) + 1
                C_SchedulerHook._offline_phase = _phase
                logger.info(
                    f"[HiSim] OFFLINE phase {_phase} complete: dumping metrics "
                    f"(completed={len(C_SchedulerHook.REQUEST_STATS)})"
                )
                wrapped_profile(self, None, _phase=_phase)

            return new_batch

        def wrapped_run_batch(self, *args, **kwargs):
            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            # KDA Checkpoint tracking: log mamba state for each request
            # Also ensure _kda_checkpoint_diag exists from initialization
            if not hasattr(C_SchedulerHook, '_kda_checkpoint_diag') or C_SchedulerHook._kda_checkpoint_diag is None:
                C_SchedulerHook._kda_checkpoint_diag = {
                    'donate_count':0,'commit_count':0,
                    'hit_count':0,'cow_exec_count':0,
                    'checkpoint_position':0,'prefix_length':0,
                    'historical_replay_tokens':0,
                    'new_tokens':0,
                }
            if batch is not None and hasattr(batch, 'reqs') and batch.reqs:
                for _req in batch.reqs:
                    _mpi = req_mamba_get(_req, 'mamba_pool_idx', None)
                    _ppb = req_mamba_get(_req, 'mamba_ping_pong_track_buffer', None)
                    _cow = req_mamba_get(_req, 'mamba_cow_src_index', None)
                    _ppb_str = _ppb.tolist() if _ppb is not None else 'None'
                    _mpi_str = _mpi.item() if _mpi is not None and hasattr(_mpi, 'item') else str(_mpi)
                    _cow_str = _cow.item() if _cow is not None and hasattr(_cow, 'item') else str(_cow)
                    logger.info(
                        f"[KDA-TRACK] rid={_req.rid} pool_idx={_mpi_str} "
                        f"pp_buf={_ppb_str} cow_src={_cow_str} "
                        f"prefix_len={len(getattr(_req,'prefix_indices',[]))}"
                    )
            # In mock mode, the Mamba model never runs, so
            # batch_result_processor never sets mamba_last_track_seqlen.
            # Without it, prepare_for_caching_req returns early (cache_len
            # is None) and the donation/eviction path is never exercised.
            # Set mamba_last_track_seqlen to the correct track boundary.
            # In real sglang, the Mamba model tracks at every C tokens during forward.
            # The scheduler may set mamba_last_track_seqlen to mamba_branching_seqlen
            # (from prefix match) or kv_committed_len, but neither is the actual
            # track boundary. We must compute it as (seq_len // C) * C.
            if batch is not None and hasattr(batch, 'reqs') and batch.reqs:
                _track_interval = int(os.environ.get("SGLANG_MAMBA_TRACK_INTERVAL", "256"))
                for _req in batch.reqs:
                    _lts_before = req_mamba_get(_req, 'mamba_last_track_seqlen', None)
                    _lts_bare = getattr(_req, 'mamba_last_track_seqlen', None)
                    _kcl = getattr(_req, 'kv_committed_len', None)
                    _iil = len(getattr(_req, 'input_ids', []))
                    _oir = len(getattr(_req, 'origin_input_ids', []))
                    _prefix = len(getattr(_req, 'prefix_indices', []))
                    logger.info(
                        f"[LTS-BEFORE] rid={_req.rid} kv_lts={_lts_before} "
                        f"bare_lts={_lts_bare} kv_committed_len={_kcl} "
                        f"input_ids_len={_iil} origin_input_ids_len={_oir} "
                        f"prefix_indices_len={_prefix} track_interval={_track_interval}"
                    )
                    # Fallback only: on sglang 0.5.18+ prepare_for_extend sets
                    # req.kv.mamba_last_track_seqlen natively; only fill it in
                    # when the native path left it unset (mock forward never
                    # runs the real Mamba model).
                    _seq_len = _kcl
                    if _seq_len is None or _seq_len == 0:
                        _seq_len = _oir
                    if (
                        (_lts_before is None or _lts_before == 0)
                        and _seq_len
                        and _seq_len > 0
                    ):
                        _track_boundary = (_seq_len // _track_interval) * _track_interval
                        if _track_boundary > 0:
                            req_mamba_set(_req, 'mamba_last_track_seqlen', _track_boundary)
                            logger.info(
                                f"[LTS-SET] rid={_req.rid} seq_len={_seq_len} "
                                f"track_interval={_track_interval} "
                                f"track_boundary={_track_boundary} "
                                f"(was {_lts_before})"
                            )
                # --- Logging point 2: Prefill complete (mamba_last_track_seqlen set) ---
                for _req in batch.reqs:
                    _lts = req_mamba_get(_req, 'mamba_last_track_seqlen', None)
                    _mpi = req_mamba_get(_req, 'mamba_pool_idx', None)
                    _ppb = req_mamba_get(_req, 'mamba_ping_pong_track_buffer', None)
                    _keep = req_mamba_get(_req, 'mamba_last_track_idx', None)
                    _mpi_v = int(_mpi.item()) if _mpi is not None and hasattr(_mpi, 'item') else (_mpi if _mpi is not None else '?')
                    _ppb_v = _ppb.tolist() if _ppb is not None else 'None'
                    logger.info(
                        f"[PREFILL] rid={_req.rid} mamba_pool_idx={_mpi_v} "
                        f"pp_buf={_ppb_v} keep_idx={_keep} "
                        f"mamba_last_track_seqlen={_lts}"
                    )
            # Defensive fallback: ensure model_runner.ngram_embedding_manager is a
            # disabled mock. ForwardBatch.init_new reads
            # model_runner.ngram_embedding_manager.enabled; in the forked
            # scheduler the model_runner may be re-instantiated and left None,
            # raising "AttributeError: 'NoneType' object has no attribute
            # 'enabled'" before any forward runs.
            _mr = getattr(getattr(self, "tp_worker", None), "model_runner", None)
            if _mr is not None and getattr(_mr, "ngram_embedding_manager", None) is None:
                class _NoopNgram:
                    enabled = False
                try:
                    _mr.ngram_embedding_manager = _NoopNgram()
                except Exception:
                    object.__setattr__(_mr, "ngram_embedding_manager", _NoopNgram())
            # Log COW scheduled state and simulate COW execution
            if batch is not None and hasattr(batch, 'reqs') and batch.reqs:
                _batch_cow_src = getattr(batch, 'mamba_cow_src_indices', None)
                _batch_cow_dst = getattr(batch, 'mamba_cow_dst_indices', None)
                # Defensive check: reject invalid COW src (-1 or out of range)
                if _batch_cow_src is not None and len(_batch_cow_src) > 0:
                    for _ci in range(len(_batch_cow_src)):
                        _csv = int(_batch_cow_src[_ci].item()) if hasattr(_batch_cow_src[_ci],'item') else int(_batch_cow_src[_ci])
                        if _csv == -1:
                            logger.warning("[COW-INVALID] batch_idx=%d src_slot=-1 — clearing COW, falling back to full replay", _ci)
                            batch.mamba_cow_src_indices = None
                            batch.mamba_cow_dst_indices = None
                            for _req in batch.reqs:
                                req_mamba_set(_req, 'mamba_cow_src_index', None)
                                req_mamba_set(_req, 'mamba_needs_clear', False)
                            _batch_cow_src = None
                            _batch_cow_dst = None
                            break
                _src_str = _batch_cow_src.tolist() if _batch_cow_src is not None else 'None'
                _dst_str = _batch_cow_dst.tolist() if _batch_cow_dst is not None else 'None'
                if _batch_cow_src is not None and len(_batch_cow_src) > 0:
                    logger.info(f"[COW-COLLECTED] src={_src_str} dst={_dst_str}")
                    # Lightweight State Copy simulation (no NPU kernel)
                    _kda_diag = getattr(C_SchedulerHook, '_kda_checkpoint_diag', None)
                    if _kda_diag is None:
                        _kda_diag = {
                            'donate_count':0,'commit_count':0,
                            'hit_count':0,'cow_exec_count':0,
                            'checkpoint_position':0,'prefix_length':0,
                            'historical_replay_tokens':0,
                            'new_tokens':0,
                        }
                        C_SchedulerHook._kda_checkpoint_diag = _kda_diag
                    for _i in range(len(_batch_cow_src)):
                        _ss = int(_batch_cow_src[_i].item()) if hasattr(_batch_cow_src[_i],'item') else int(_batch_cow_src[_i])
                        _ds = int(_batch_cow_dst[_i].item()) if hasattr(_batch_cow_dst[_i],'item') else int(_batch_cow_dst[_i])
                        # Find req by pool_idx to get prefix_len and extend_len
                        _pl = 0
                        _nl = 0
                        for _req in batch.reqs:
                            _mpi = req_mamba_get(_req,'mamba_pool_idx',None)
                            if _mpi is not None:
                                _mvi = _mpi.item() if hasattr(_mpi,'item') else int(_mpi)
                                if _mvi == _ds:
                                    _pl = len(getattr(_req,'prefix_indices',[]))
                                    _er = getattr(_req,'extend_range',None)
                                    _nl = getattr(_er,'length',0) if _er else 0
                                    break
                        # checkpoint_position = prefix_length (COW source is
                        # best_match_node which covers the full matched prefix)
                        _cp = _pl
                        # historical_replay = prefix_len - checkpoint_position
                        # (0 when checkpoint covers the full prefix)
                        _rt = max(0, _pl - _cp)
                        _kda_diag['hit_count'] += 1
                        _kda_diag['cow_exec_count'] += 1
                        _kda_diag['checkpoint_position'] = _cp
                        _kda_diag['prefix_length'] = _pl
                        _kda_diag['historical_replay_tokens'] += _rt
                        _kda_diag['new_tokens'] += _nl
                        logger.info(
                            f"[COW-EXEC-SIM] src_slot={_ss} dst_slot={_ds} "
                            f"checkpoint_pos={_cp} prefix_len={_pl} "
                            f"replay_tokens={_rt} new_tokens={_nl}"
                        )
                        # [v2-fix A1] Populate per-req metrics for COW hit
                        if kda_obs.is_obs_enabled() and _req is not None:
                            _mr = kda_obs.get_match_result(_req.rid)
                            _mla_pot = _mr.get("full_kv_hit", _pl)
                            _mhost = _mr.get("mamba_host_hit", 0)
                            _total_in = len(getattr(_req, 'origin_input_ids', []))
                            _kda_l1 = _pl  # COW hit → checkpoint at matched node
                            _kda_l2 = _pl if _mhost > 0 else 0
                            _kda_best = max(_kda_l1, _kda_l2)
                            _req_st = C_SchedulerHook.REQUEST_STATS.get(_req.rid)
                            if _req_st is not None:
                                _req_st.mla_potential_hit_tokens = _mla_pot
                                _req_st.kda_l1_recoverable_position = _kda_l1
                                _req_st.kda_l2_recoverable_position = _kda_l2
                                _req_st.kda_cow_src_slot = _ss
                                _req_st.effective_prefix_hit_tokens = min(_mla_pot, _kda_best)
                                _req_st.retreat_gap_tokens = max(0, _mla_pot - _kda_best)
                                _req_st.actual_extra_recompute_tokens = 0  # COW covers, no replay
                                _req_st.scheduled_extend_tokens = _nl
                                _req_st.unavoidable_new_tokens = max(0, _total_in - _mla_pot)
                                _req_st.predictor_input_tokens = _nl
                                # Legacy aliases
                                _req_st.kda_replay_tokens = 0
                                _req_st.kda_new_tokens = _nl
                                _req_st.kda_l1_hit = True
                                _req_st.kda_l2_hit = _kda_l2 > 0
                                _req_st.kda_recoverable_position = _kda_best
                                # [A2] New metrics
                                _req_st.scheduler_effective_hit_tokens = int(getattr(_req, 'cached_tokens', 0) or _pl)
                                _req_st.verified_logical_reuse = True  # COW = device-resident
                                _req_st.historical_tokens_actually_scheduled = 0
                                _req_st.kda_restore_tier = "L1"
                                _req_st.kda_restore_latency_ms = 0.0  # D2D, no H2D
                            kda_obs.log_event(
                                "cow_exec", rid=_req.rid,
                                virtual_time=StateManager.get_global_clock(),
                                checkpoint_pos=_cp, layer="L1",
                                extra={"mla_potential": _mla_pot, "retreat_gap": 0,
                                       "replay_tokens": 0, "extend_tokens": _nl},
                            )
                    # Clear deferred COW
                    batch.mamba_cow_src_indices = None
                    batch.mamba_cow_dst_indices = None
                    for _req in batch.reqs:
                        req_mamba_set(_req, 'mamba_cow_src_index', None)
                        req_mamba_set(_req, 'mamba_needs_clear', False)
                else:
                    # No COW — either no prefix match or checkpoint unavailable.
                    # Track replay tokens ONLY for prefill/extend batches where
                    # replay is actually needed. Decode batches already have
                    # active Mamba state and should NOT be counted.
                    _fm = getattr(batch, 'forward_mode', None)
                    _is_extend = False
                    if _fm is not None:
                        _is_extend = hasattr(_fm, 'is_extend') and _fm.is_extend()
                    if not _is_extend:
                        # Decode batch — skip REPLAY-FALLBACK
                        pass
                    else:
                        _kda_diag = getattr(C_SchedulerHook, '_kda_checkpoint_diag', None)
                        if _kda_diag is None:
                            _kda_diag = {'donate_count':0,'commit_count':0,
                                         'hit_count':0,'cow_exec_count':0,
                                         'checkpoint_position':0,'prefix_length':0,
                                         'historical_replay_tokens':0,'new_tokens':0,
                                         'no_cow_count':0,'replay_fallback_count':0}
                            C_SchedulerHook._kda_checkpoint_diag = _kda_diag
                        _kda_diag['no_cow_count'] = (_kda_diag.get('no_cow_count', 0)) + 1
                        for _req in batch.reqs:
                            _branching = getattr(_req, 'mamba_branching_seqlen', None)
                            _prefix_len = len(getattr(_req, 'prefix_indices', []))
                            _mpi = req_mamba_get(_req, 'mamba_pool_idx', None)
                            if _prefix_len > 0 and _mpi is not None:
                                # [v2-fix A1] CORRECTED: No COW does NOT mean replay needed.
                                # sglang 0.5.19 has TWO restoration paths:
                                #   1. COW: copies device-resident checkpoint to new slot (cow_src set)
                                #   2. init_load_back → load_back → commit_load_back: restores
                                #      from L2 host to L1 device BEFORE set_extend_range
                                #      (schedule_policy.py:L1290 before L1337)
                                # If init_load_back succeeded, prefix_indices was extended
                                # and extend_range.length = total - prefix_len (small).
                                # No replay needed in this case.
                                #
                                # The OLD code assumed prefix_len > 0 + cow_src None = replay
                                # needed, but this is a FALSE POSITIVE when load_back succeeded.
                                #
                                # Correct check: if extend_range.length is small (= total - prefix_len),
                                # the state was restored (either COW or load_back), no replay.
                                _er = getattr(_req, 'extend_range', None)
                                _ext_len = getattr(_er, 'length', 0) if _er else 0
                                _total_in = len(getattr(_req, 'origin_input_ids', []))
                                _expected_ext_if_restored = _total_in - _prefix_len
                                _is_restored = (_ext_len <= _expected_ext_if_restored + 1)  # +1 tolerance
                                # [P0-2 FIX] Also check if MAMBA state was actually restored.
                                # extend_range.length only reflects KV cache prefix match,
                                # NOT MAMBA state restore. If MAMBA restore failed (L2 miss,
                                # L3 miss/None), replay IS needed even if extend_range is small.
                                _tc_rf = getattr(self, 'tree_cache', None)
                                _diag_l_rb = getattr(_tc_rf, '_hismim_diag_l', {}) if _tc_rf else {}
                                _mamba_failed = _diag_l_rb.get('mamba_restore_failed', False)
                                if _is_restored and _mamba_failed:
                                    _is_restored = False  # MAMBA not restored → replay needed
                                if _is_restored:
                                    _replay = 0  # State restored via load_back, no replay needed
                                    logger.info(
                                        f"[REPLAY-FALLBACK] rid={_req.rid} prefix_len={_prefix_len} "
                                        f"branching={_branching} replay_tokens=0 (load_back restored) "
                                        f"cow_src={getattr(_req, 'mamba_cow_src_index', None)} "
                                        f"extend={_ext_len}"
                                    )
                                else:
                                    # State NOT restored — extend_range is larger than expected
                                    # This means init_load_back failed or wasn't called
                                    if _branching is not None:
                                        _replay = max(0, _prefix_len - _branching)
                                    else:
                                        _replay = _prefix_len
                                    _kda_diag['historical_replay_tokens'] += _replay
                                    logger.info(
                                        f"[REPLAY-FALLBACK] rid={_req.rid} prefix_len={_prefix_len} "
                                        f"branching={_branching} replay_tokens={_replay} "
                                        f"cow_src={getattr(_req, 'mamba_cow_src_index', None)} "
                                        f"extend={_ext_len} (NOT restored)"
                                    )
                                _kda_diag['replay_fallback_count'] = (_kda_diag.get('replay_fallback_count', 0)) + 1
                                # [v2-fix A1] Populate per-req metrics for REPLAY-FALLBACK
                                if kda_obs.is_obs_enabled():
                                    _mr = kda_obs.get_match_result(_req.rid)
                                    _mla_pot = _mr.get("full_kv_hit", _prefix_len)
                                    _kda_l1 = 0  # No COW → no L1 checkpoint at match time
                                    # If load_back restored, L2 checkpoint was used (prefix_len > 0)
                                    _kda_l2 = _prefix_len if _prefix_len > 0 else 0
                                    _kda_best = max(_kda_l1, _kda_l2)
                                    _req_st = C_SchedulerHook.REQUEST_STATS.get(_req.rid)
                                    if _req_st is not None:
                                        _req_st.mla_potential_hit_tokens = _mla_pot
                                        _req_st.kda_l1_recoverable_position = _kda_l1
                                        _req_st.kda_l2_recoverable_position = _kda_l2
                                        _req_st.kda_cow_src_slot = -1
                                        _req_st.effective_prefix_hit_tokens = min(_mla_pot, _kda_best)
                                        _req_st.retreat_gap_tokens = max(0, _mla_pot - _kda_best)
                                        # [v2-fix] actual_extra_recompute = 0 if restored, _replay if not
                                        _req_st.actual_extra_recompute_tokens = _replay
                                        _req_st.scheduled_extend_tokens = _ext_len
                                        _req_st.unavoidable_new_tokens = max(0, _total_in - _mla_pot)
                                        _req_st.predictor_input_tokens = _ext_len
                                        _req_st.kda_replay_tokens = _replay
                                        _req_st.kda_new_tokens = max(0, _total_in - _mla_pot)
                                        _req_st.kda_l1_hit = False
                                        _req_st.kda_l2_hit = _kda_l2 > 0
                                        _req_st.kda_recoverable_position = _kda_best
                                        # [A2] New metrics
                                        _req_st.scheduler_effective_hit_tokens = int(getattr(_req, 'cached_tokens', 0) or _prefix_len)
                                        _ilb_tier = _mr.get("init_load_back_tier", "L2" if _is_restored else "unknown")
                                        _req_st.kda_restore_tier = _ilb_tier
                                        _req_st.historical_tokens_actually_scheduled = _replay
                                        _slot_hash = f"kda_{_mr.get('best_match_node', -1)}_0"
                                        _ver = kda_obs.shadow_verify(_req.rid, _slot_hash)
                                        _req_st.verified_logical_reuse = bool(
                                            _ver.get("identity_ok", False) and
                                            _ver.get("ready", False))
                                    kda_obs.log_event(
                                        "replay_fallback", rid=_req.rid,
                                        virtual_time=StateManager.get_global_clock(),
                                        checkpoint_pos=_kda_best, layer="L2" if _kda_l2 > 0 else "",
                                        extra={"mla_potential": _mla_pot,
                                               "retreat_gap": max(0, _mla_pot - _kda_best),
                                               "replay_tokens": _replay,
                                               "extend_tokens": _ext_len,
                                               "predictor_input": _ext_len,
                                               "restored": _is_restored,
                                               "kda_tier": _ilb_tier,
                                               "verified": _req_st.verified_logical_reuse if _req_st else False},
                                    )
                            else:
                                # [A2] All-Miss: no prefix match, full prefill
                                if kda_obs.is_obs_enabled():
                                    _total_in = len(getattr(_req, 'origin_input_ids', []))
                                    _er = getattr(_req, 'extend_range', None)
                                    _ext_len = getattr(_er, 'length', _total_in) if _er else _total_in
                                    _req_st = C_SchedulerHook.REQUEST_STATS.get(_req.rid)
                                    if _req_st is not None:
                                        _req_st.mla_potential_hit_tokens = 0
                                        _req_st.kda_l1_recoverable_position = 0
                                        _req_st.kda_l2_recoverable_position = 0
                                        _req_st.kda_cow_src_slot = -1
                                        _req_st.effective_prefix_hit_tokens = 0
                                        _req_st.retreat_gap_tokens = 0
                                        _req_st.actual_extra_recompute_tokens = 0
                                        _req_st.scheduled_extend_tokens = _ext_len
                                        _req_st.unavoidable_new_tokens = _total_in
                                        _req_st.predictor_input_tokens = _ext_len
                                        _req_st.scheduler_effective_hit_tokens = 0
                                        _req_st.verified_logical_reuse = False
                                        _req_st.historical_tokens_actually_scheduled = 0
                                        _req_st.kda_restore_tier = "miss"
                                    kda_obs.log_event(
                                        "all_miss", rid=_req.rid,
                                        virtual_time=StateManager.get_global_clock(),
                                        extra={"total_input": _total_in,
                                               "extend_tokens": _ext_len},
                                    )

            ret = original_run_batch(self, *args, **kwargs)

            if ret.__class__.__name__ == "GenerationBatchResult":
                hisim_batch = HisimScheduleBatch(reqs=[])
                if batch.forward_mode.is_extend():
                    # [P1-FIX] Get MAMBA restore status for replay token accounting
                    _tc_p1 = getattr(self, 'tree_cache', None)
                    _diag_l_p1 = getattr(_tc_p1, '_hismim_diag_l', {}) if _tc_p1 else {}
                    _mamba_restore_failed = _diag_l_p1.get('mamba_restore_failed', False)
                    for req in batch.reqs:
                        _ext_len = getattr(
                            getattr(req, "extend_range", None),
                            "length",
                            len(req.origin_input_ids),
                        )
                        _prefix_len = len(req.prefix_indices)
                        _cow_src = req_mamba_get(req, 'mamba_cow_src_index', None)
                        # Calculate replay tokens: only when MAMBA not restored
                        if _cow_src is not None:
                            _replay = 0  # COW hit
                        elif _mamba_restore_failed and _prefix_len > 0:
                            _replay = _prefix_len  # MAMBA restore failed
                        else:
                            _replay = 0  # MAMBA restored via L2/L3
                        _effective_compute = _ext_len + _replay
                        hisim_batch.reqs.append(
                            FakeRequest(
                                input_length=_effective_compute,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )
                        if _replay > 0:
                            logger.info(
                                f"[P1-TIMING] rid={req.rid[:8]} ext={_ext_len} "
                                f"replay={_replay} effective={_effective_compute} "
                                f"cow={'Y' if _cow_src is not None else 'N'} "
                                f"mamba_failed={_mamba_restore_failed}"
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
            # CPU simulation: original_process_batch_result may fail because
            # the detokenizer subprocess is not running (ZMQ send fails).
            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            # Capture mamba_pool_idx BEFORE original_process_batch_result,
            # because cleanup_after_caching_req (called inside) frees it.
            _pre_donations = {}
            if batch is not None and hasattr(batch, 'reqs'):
                for _r in batch.reqs:
                    _mpi = req_mamba_get(_r, "mamba_pool_idx", None)
                    _lts = req_mamba_get(_r, "mamba_last_track_seqlen", None)
                    _olen = len(getattr(_r, "output_ids", []))
                    if _mpi is not None and _olen >= 0:
                        _pre_donations[_r.rid] = (_mpi, _lts, _olen)
            try:
                ret = original_process_batch_result(self, *args, **kwargs)
            except Exception as e:
                logger.warning(f"process_batch_result exception: {e}")
                ret = None
            # Log running_batch state after process_batch_result
            _rb = getattr(self, "running_batch", None)
            _rb_count = len(_rb.reqs) if _rb else 0
            _mode = getattr(getattr(batch, "forward_mode", None), "value", "?") if batch else "?"
            if batch is not None and hasattr(batch, "reqs"):
                for _r in batch.reqs:
                    _fin = getattr(_r, "finished", None)
                    _olen = len(getattr(_r, "output_ids", []))
                    _rid = _r.rid
                    logger.info(f"[PBR] rid={_rid} mode={_mode} output_len={_olen} finished={_fin} running={_rb_count}")

            # KDA Donation detection: use pre-captured mamba_pool_idx
            # (before cleanup_after_caching_req freed it).
            _kda_diag = getattr(C_SchedulerHook, '_kda_checkpoint_diag', None)
            if _kda_diag is not None and batch is not None and hasattr(batch, 'reqs'):
                for _r in batch.reqs:
                    _rid = _r.rid
                    _olen = len(getattr(_r, "output_ids", []))
                    if _rid in _pre_donations:
                        _mpi, _lts, _pre_olen = _pre_donations[_rid]
                        _mpi_val = _mpi.item() if hasattr(_mpi, 'item') else int(_mpi)
                        _lts_val = _lts if _lts is not None else 0
                        logger.info(f"[KDA-DONATE-CHECK] rid={_rid} olen={_olen} mpi={_mpi_val} lts={_lts_val}")
                        if _lts_val > 0:
                            _kda_diag['donate_count'] += 1
                            _kda_diag['commit_count'] += 1
                            logger.info(f"[KDA-DONATE] rid={_rid} slot={_mpi_val} track_seqlen={_lts_val}")
            # --- Logging point 3: Cache Commit — verify Radix Tree has MAMBA value ---
            _tc = getattr(self, 'tree_cache', None)
            if _tc is not None and batch is not None and hasattr(batch, 'reqs'):
                _tree_core = getattr(_tc, 'tree_core', None)
                if _tree_core is not None:
                    for _r in batch.reqs:
                        _prefix_indices = getattr(_r, 'prefix_indices', None)
                        _token_ids = getattr(_r, 'input_ids', None) or getattr(_r, 'all_token_ids', None)
                        if _token_ids is None:
                            continue
                        _token_len = len(_token_ids)
                        if _token_len == 0:
                            continue
                        try:
                            # match_prefix to find the node that should hold this req's checkpoint
                            _match = _tc.match_prefix(_token_ids[:1].tolist() if hasattr(_token_ids, 'tolist') else [_token_ids[0]])
                            _best = getattr(_match, 'best_match_node', None) if _match else None
                            if _best is not None:
                                _node = _tree_core._node_arena.get(_best)
                                if _node is not None:
                                    _cd_mamba = _node.component_data[2] if len(_node.component_data) > 2 else None
                                    _mamba_val = getattr(_cd_mamba, 'value', None) if _cd_mamba else None
                                    _mamba_exist = _mamba_val is not None and (
                                        int(_mamba_val.item()) if hasattr(_mamba_val, 'item') else int(_mamba_val)
                                    ) != -1 if _mamba_val is not None else False
                                    logger.info(
                                        f"[CACHE-COMMIT] rid={_r.rid} node={_best} "
                                        f"mamba_value={_mamba_val} mamba_exist={_mamba_exist} "
                                        f"token_len={_token_len}"
                                    )
                        except Exception as _e:
                            logger.debug(f"[CACHE-COMMIT] rid={getattr(_r,'rid','?')} match failed: {_e}")
            if batch is not None:
                if len(batch.reqs) == 0:
                    return ret

                hicache_l2_load_dur = StateManager.pop_hicache_l2_load_dur()
                hicache_l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()
                hicache_l3_backup_dur = StateManager.pop_hicache_l3_backup_dur()
                # KDA state durations
                kda_l2_load_dur = StateManager.pop_hicache_kda_l2_load_dur()
                kda_l2_backup_dur = StateManager.pop_hicache_kda_l2_backup_dur()
                kda_l3_backup_dur = StateManager.pop_hicache_kda_l3_backup_dur()
                kda_l3_load_dur = StateManager.pop_hicache_kda_l3_load_dur()
                # MLA and KDA have independent data paths (different memory regions),
                # so their H2D/D2H can overlap — take max.
                # But KDA L3 read (SSD→DRAM) must finish BEFORE KDA H2D (DRAM→HBM):
                # these are serial, so sum them first, then take max with MLA.
                kda_load_total = kda_l3_load_dur + kda_l2_load_dur  # Serial: SSD read + H2D
                l2_load_dur = max(hicache_l2_load_dur, kda_load_total)
                l2_backup_dur = max(hicache_l2_backup_dur, kda_l2_backup_dur)
                l3_backup_dur = max(hicache_l3_backup_dur, kda_l3_backup_dur)
                current_inference_dur = StateManager.get_current_inference_dur()
                l3_direct_dur = 0.0

                try:
                    if ConfigManager.get_platform_config().enable_l3_direct_to_l1:
                        l3_direct_dur = StateManager.pop_hicache_l3_direct_load_dur()
                        l2_load_dur = max(l2_load_dur, l3_direct_dur)
                except Exception:
                    pass
                StateManager.pop_l3_direct_load_count()

                # Save last values for next iteration's timing-aware SSD simulation
                StateManager.set_last_l2_load_dur(l2_load_dur)
                StateManager.set_last_l2_backup_dur(l2_backup_dur)
                StateManager.set_last_kda_l2_load_dur(kda_l2_load_dur)
                StateManager.set_last_kda_l2_backup_dur(kda_l2_backup_dur)

                # L3 write starts at T_N (round start), overlaps with entire round:
                # l2_load + inference + l2_backup. Only overflow delays next round.
                l3_write_overflow = max(
                    0,
                    l3_backup_dur
                    - (l2_load_dur + current_inference_dur + l2_backup_dur),
                )
                StateManager.set_last_l3_write_overflow(l3_write_overflow)

                if C_SchedulerHook.OVERLAP_SCHEDULE:
                    StateManager.step_global_clock(
                        max(
                            l2_load_dur - StateManager.get_last_inference_dur(),
                            0,
                        )
                    )
                    StateManager.step_global_clock(current_inference_dur)
                    request_response_time = (
                        StateManager.get_global_clock()
                        + l2_backup_dur
                        + l3_write_overflow
                    )
                else:
                    StateManager.step_global_clock(
                        l2_load_dur
                        + current_inference_dur
                        + l2_backup_dur
                        + l3_write_overflow
                    )
                    request_response_time = StateManager.get_global_clock()
                # Request statistics
                for req in batch.reqs:
                    if getattr(req, "is_chunked", 0) == 0:
                        req_stats = C_SchedulerHook.REQUEST_STATS.get(req.rid)
                        if req_stats is None:
                            continue
                        _lat = request_response_time - req_stats.last_event_time
                        if not math.isfinite(_lat) or _lat < 0:
                            # Guard against NaN/negative from virtual-clock
                            # anomalies (e.g. eviction timing); skip bad samples.
                            continue
                        req_stats.gen_token_latencies.append(_lat)
                        req_stats.last_event_time = request_response_time
                    else:
                        # Chunked request: nothing to do
                        pass
                # Iteration statistics
                _iter_stat = {
                    "requests": C_SchedulerHook.HISIM_BATCH.request_info(),
                    "forward_latency": current_inference_dur,
                    "l2_load_latency": l2_load_dur,
                    "l2_backup_latency": l2_backup_dur,
                    "l3_backup_latency": l3_backup_dur,
                }
                if l3_direct_dur > 0:
                    _iter_stat["l3_direct_load_latency"] = l3_direct_dur
                C_SchedulerHook.ITERATION_STATS.append(_iter_stat)

            C_SchedulerHook.LAST_CPU_TS = time.time()
            return ret

        def wrapped_profile(self, req, *args, **kwargs):
            _phase = kwargs.pop("_phase", None)
            C_SchedulerHook._profile_call_count = getattr(C_SchedulerHook, "_profile_call_count", 0) + 1
            stats: list[RequestStats] = []
            _rs_total = len(C_SchedulerHook.REQUEST_STATS)
            for item in C_SchedulerHook.REQUEST_STATS.values():
                if item.rid is not None and item.input_length > 0 and (len(item.gen_token_latencies) > 0 or item.queue_end != -1):
                    stats.append(item)
            if _rs_total > 0 and len(stats) == 0:
                pass

            stats = sorted(stats, key=lambda req: req.created_time)

            output_dir = Envs.output_dir()
            os.makedirs(output_dir, exist_ok=True)

            if len(stats) > 0:
                # Remove warmup requests.
                if len(stats) > Envs.num_warmup():
                    metrics_stats = stats[Envs.num_warmup() :]
                else:
                    metrics_stats = stats

                # Accumulate raw (unaligned) copies for the canonical
                # metrics.json. The phase file below uses the live objects,
                # which the alignment step mutates; keeping pristine copies
                # lets metrics.json aggregate every phase.
                import copy as _copy
                C_SchedulerHook.ALL_REQUEST_STATS.extend(
                    _copy.copy(_s) for _s in metrics_stats
                )

                # Ensure gen_token_latencies has at least one entry (TTFT).
                # Requests that finish after extend without decode batches
                # (mock forward produces stop tokens) have empty latencies.
                for item in metrics_stats:
                    if len(item.gen_token_latencies) == 0:
                        item.gen_token_latencies.append(0.0)

                min_created_time = metrics_stats[0].created_time
                # Align timestamps
                for item in stats:
                    item.created_time -= min_created_time
                    item.queue_start -= min_created_time
                    item.queue_end -= min_created_time
                    item.last_event_time -= min_created_time

                metrics = calc_metrics(metrics_stats)
                metrics["time_cost"] = time.time() - C_SchedulerHook.LAST_FLUSH_TS

                # --- Add KDA/Unified diagnostics to metrics.json ---
                # Cumulative counters (reset after each OFFLINE dump)
                _ud = getattr(C_SchedulerHook, '_unified_diag', None)
                if _ud is not None:
                    metrics["unified_alloc"] = _ud.get('alloc', 0)
                    metrics["unified_free"] = _ud.get('free', 0)
                    metrics["unified_compact"] = _ud.get('compact', 0)
                    metrics["unified_compact_pages"] = _ud.get('compact_pages_moved', 0)
                    metrics["unified_flush_peer"] = _ud.get('flush_peer', 0)
                    metrics["unified_flush"] = _ud.get('flush', 0)
                    metrics["unified_flush_pages"] = _ud.get('flush_pages_moved', 0)
                    metrics["unified_alloc_fail"] = _ud.get('alloc_fail', 0)
                    metrics["unified_gap_bytes"] = _ud.get('gap_bytes', 0)
                    metrics["unified_kv_watermark"] = _ud.get('kv_watermark', 0)
                    metrics["unified_mamba_watermark"] = _ud.get('mamba_watermark', 0)
                    metrics["unified_mla_live_pages"] = _ud.get('mla_live_pages', 0)
                    metrics["unified_mamba_live_pages"] = _ud.get('mamba_live_pages', 0)
                    metrics["peak_mla_live_pages"] = _ud.get('peak_mla_live_pages', 0)
                    metrics["peak_mamba_live_pages"] = _ud.get('peak_mamba_live_pages', 0)
                    metrics["peak_mla_watermark"] = _ud.get('peak_mla_watermark', 0)
                    metrics["peak_mamba_watermark"] = _ud.get('peak_mamba_watermark', 0)
                # else: Two Pools mode — unified diag not applicable, fields omitted (not zeroed)

                # KDA radix cache interaction (L1 checkpoint donation/hit)
                _tc = getattr(self, 'tree_cache', None)
                if _tc is not None:
                    _dw = getattr(_tc, '_hismim_diag_w', {})
                    _dl = getattr(_tc, '_hismim_diag_l', {})
                    metrics["kda_write_calls"] = _dw.get('calls', 0)        # radix tree insert calls
                    metrics["kda_write_mamba_xfers"] = _dw.get('mamba_pt', 0)  # L1 checkpoint donations (state donated to radix tree)
                    metrics["kda_load_calls"] = _dl.get('calls', 0)        # radix tree match_prefix calls
                    metrics["kda_load_mamba_xfers"] = _dl.get('mamba_pt', 0)  # L1 checkpoint hits (state loaded back from radix tree)
                    # HiCache-specific (L2/L3 backup). When HiCache disabled, these remain 0
                    # because no backup/load_back occurs. This is genuinely 0, not "missing".
                    # [D1.1] Note: l3_writes/l3_hits are MLA counters (legacy naming).
                    # KDA-specific counters added below.
                    metrics["kda_l3_writes"] = _dw.get('l3_writes', 0)      # MLA L3 writes (legacy name)
                    metrics["kda_l3_hits"] = _dl.get('l3_hits', 0)           # MLA L3 hits (legacy name)
                    metrics["kda_l3_mamba_writes"] = _dw.get('kda_l3_writes', 0)  # KDA L3 writes (D1.1)
                    metrics["kda_l3_mamba_hits"] = _dl.get('kda_l3_hits', 0)     # KDA L3 hits (D1.1)
                    metrics["kda_l3_admission_reject"] = _dl.get('kda_l3_admission_reject', 0)  # D1.2
                    # [P2D2] Restore Ready failure boundary counters
                    metrics["kda_l2_load_fail"] = _dl.get('kda_l2_load_fail', 0)
                    metrics["kda_l3_stage_fail"] = _dl.get('kda_l3_stage_fail', 0)
                    # [A4] L3 load duration counters (MLA vs KDA, separately accumulated)
                    metrics["mla_l3_load_dur_total"] = getattr(C_SchedulerHook, '_mla_l3_load_total', 0.0)
                    metrics["kda_l3_load_dur_total"] = getattr(C_SchedulerHook, '_kda_l3_load_total', 0.0)

                # KDA Checkpoint native diagnostics (from COW simulation)
                _kda_diag = getattr(C_SchedulerHook, '_kda_checkpoint_diag', None)
                if _kda_diag is not None:
                    metrics["kda_checkpoint_donate_count"] = _kda_diag.get('donate_count', 0)
                    metrics["kda_checkpoint_commit_count"] = _kda_diag.get('commit_count', 0)
                    metrics["kda_cached_checkpoint_slots"] = _kda_diag.get('cached_checkpoint_slots', 0)
                    metrics["kda_checkpoint_hit_count"] = _kda_diag.get('hit_count', 0)
                    metrics["kda_cow_exec_count"] = _kda_diag.get('cow_exec_count', 0)
                    metrics["kda_checkpoint_position"] = _kda_diag.get('checkpoint_position', 0)
                    metrics["kda_prefix_length"] = _kda_diag.get('prefix_length', 0)
                    metrics["kda_historical_replay_tokens"] = _kda_diag.get('historical_replay_tokens', 0)
                    metrics["kda_new_tokens"] = _kda_diag.get('new_tokens', 0)

                # KDA pool instantaneous stats (snapshot at dump time)
                # [v2-fix F3] Read from active _StrictMambaAllocator, not dead _kda_pool
                _rtp = getattr(self, 'req_to_token_pool', None)
                _active_alloc = getattr(_rtp, 'mamba_allocator', None) if _rtp is not None else None
                if _active_alloc is not None and hasattr(_active_alloc, 'get_stats'):
                    try:
                        _aps = _active_alloc.get_stats()
                        metrics["kda_active_slots"] = _aps.get("capacity", 0) - _aps.get("available", 0)
                        metrics["kda_alloc_count"] = _aps.get("alloc_count", 0)
                        metrics["kda_free_count"] = _aps.get("free_count", 0)
                        metrics["kda_free_slots"] = _aps.get("available", 0)
                        metrics["kda_pool_capacity"] = _aps.get("capacity", 0)
                    except Exception:
                        pass  # allocator not yet initialized
                _kda_host_pool = getattr(_tc, 'mamba_pool_host', None) if _tc is not None else None
                if _kda_host_pool is not None and hasattr(_kda_host_pool, 'get_stats'):
                    try:
                        _khs = _kda_host_pool.get_stats()
                        metrics["kda_l2_cached_slots"] = _khs.get('cached_slots', 0)
                        metrics["kda_l2_capacity"] = _khs.get('capacity', 0)
                        metrics["kda_l2_free_slots"] = _khs.get('free_slots', 0)
                        metrics["kda_l2_lru_entries"] = _khs.get('lru_entries', 0)
                    except Exception:
                        pass
                # [P2C.3] L2 placement stats (backup/load_back/eviction counters)
                _kda_stats = C_HiCacheController.get_kda_stats()
                if _kda_stats:
                    metrics["kda_l2_backup_count"] = _kda_stats.get("backup", 0)
                    metrics["kda_l2_load_back_count"] = _kda_stats.get("load_back_l2", 0)
                    metrics["kda_l3_load_back_count"] = _kda_stats.get("load_back_l3", 0)
                    metrics["kda_l3_hit_count"] = _kda_stats.get("l3_hit", 0)
                # [P2C.3] L2 policy name
                try:
                    from hisim.simulation.l2_placement_policy import get_l2_policy as _gl2p
                    metrics["kda_l2_policy"] = _gl2p().name
                except Exception:
                    metrics["kda_l2_policy"] = "unknown"

                # L2 unified DRAM pool (KDA+MLA one budget) metrics
                try:
                    from hisim.simulation.l2_dram_pool import get_l2_dram_pool as _gl2u
                    _l2u = _gl2u()
                    if _l2u is not None:
                        _l2s = _l2u.stats()
                        for _k, _v in _l2s.items():
                            metrics[f"l2_{_k}"] = _v
                        metrics["l2_one_pool"] = int(_l2s.get("mode") == "one_pool")
                        # [P2D2] MLA page-level diagnostics for overhead attribution
                        try:
                            _mla_pd = _l2u.mla_page_diagnostics()
                            for _pk, _pv in _mla_pd.items():
                                metrics[f"l2_mla_pd_{_pk}"] = _pv
                        except Exception as _pd_err:
                            logger.debug(f"MLA page diagnostics unavailable: {_pd_err}")
                except Exception as _l2e:
                    logger.debug(f"L2 unified metrics unavailable: {_l2e}")

                # MLA instantaneous: capture current KV usage from server log context
                _kv_pool = getattr(self, 'token_to_kv_pool', None) or getattr(
                    getattr(self, 'model_runner', None), 'token_to_kv_pool', None)
                if _kv_pool is not None and hasattr(_kv_pool, 'size'):
                    metrics["mla_pool_size"] = _kv_pool.size
                    if hasattr(_kv_pool, 'available_size'):
                        metrics["mla_available_tokens"] = _kv_pool.available_size()
                        metrics["mla_resident_tokens"] = _kv_pool.size - _kv_pool.available_size()

                # Budget and resident bytes
                _MLA_ENTRY = 5760
                _KDA_ENTRY = 14653440
                _TOTAL_BUDGET = 2324275200
                metrics["budget_total_bytes"] = _TOTAL_BUDGET
                metrics["budget_mla_bytes"] = metrics.get("mla_pool_size", 0) * _MLA_ENTRY
                metrics["budget_kda_bytes"] = metrics.get("kda_free_slots", -1)
                # For Two Pools: kda capacity = active + free
                if "kda_active_slots" in metrics and "kda_free_slots" in metrics:
                    _kda_cap = metrics["kda_active_slots"] + metrics["kda_free_slots"]
                    metrics["kda_capacity_slots"] = _kda_cap
                    metrics["budget_kda_bytes"] = _kda_cap * _KDA_ENTRY
                    metrics["kda_resident_bytes"] = metrics["kda_active_slots"] * _KDA_ENTRY
                    metrics["mla_resident_bytes"] = metrics.get("mla_resident_tokens", 0) * _MLA_ENTRY
                    metrics["peak_resident_bytes"] = max(
                        metrics["kda_resident_bytes"], metrics["mla_resident_bytes"]
                    ) + min(
                        metrics["kda_resident_bytes"], metrics["mla_resident_bytes"]
                    )
                # For One Pool: resident from live_pages
                if _ud is not None:
                    _mla_pages = _ud.get('mla_live_pages', 0)
                    _mamba_pages = _ud.get('mamba_live_pages', 0)
                    _peak_mla = _ud.get('peak_mla_live_pages', 0)
                    _peak_mamba = _ud.get('peak_mamba_live_pages', 0)
                    metrics["kda_active_slots"] = _mamba_pages  # page_size=1 for Mamba
                    metrics["mla_resident_tokens"] = _mla_pages  # page_size=1 for our mock
                    metrics["kda_resident_bytes"] = _mamba_pages * _KDA_ENTRY
                    metrics["mla_resident_bytes"] = _mla_pages * _MLA_ENTRY
                    metrics["peak_kda_resident_bytes"] = _peak_mamba * _KDA_ENTRY
                    metrics["peak_mla_resident_bytes"] = _peak_mla * _MLA_ENTRY
                    metrics["peak_total_resident_bytes"] = (
                        _peak_mamba * _KDA_ENTRY + _peak_mla * _MLA_ENTRY
                    )

                # Aggregated MLA prefix-hit metrics. Computed BEFORE the files
                # are written (the save below happens earlier in the function
                # than the per-request diagnostic logging used to sit), so the
                # values actually persist to metrics_phaseN.json and, via the
                # `_canon` borrow loop, to the canonical metrics.json.
                _agg_prefix_hit = 0
                _agg_prefix_hit_l1 = 0
                _agg_prefill_new = 0
                for _item in metrics_stats:
                    _l2 = _item.l2_hit_tokens
                    _l1 = _item.final_reused_tokens - _l2
                    _l3 = _item.l3_hit_tokens
                    _new_tok = _item.input_length - _l1 - _l2 - _l3
                    _agg_prefix_hit += (_l1 + _l2 + _l3)
                    _agg_prefix_hit_l1 += _l1
                    _agg_prefill_new += max(0, _new_tok)
                metrics["mla_prefix_hit_tokens_agg"] = _agg_prefix_hit
                metrics["mla_prefix_hit_l1_tokens_agg"] = _agg_prefix_hit_l1
                metrics["mla_prefill_new_tokens_agg"] = _agg_prefill_new

                # --- Canonical metrics.json (aggregated across all phases) ---
                # bench_serving.load_simulation_metrics() reads metrics.json;
                # the per-phase metrics_phaseN.json files are kept separately
                # for multi-phase analysis.
                _canon_items = [
                    _copy.copy(_s) for _s in C_SchedulerHook.ALL_REQUEST_STATS
                ]
                if _canon_items:
                    for _s in _canon_items:
                        if len(_s.gen_token_latencies) == 0:
                            _s.gen_token_latencies.append(0.0)
                    _canon_items.sort(key=lambda _r: _r.created_time)
                    _cmin = _canon_items[0].created_time
                    for _s in _canon_items:
                        _s.created_time -= _cmin
                        _s.queue_start -= _cmin
                        _s.queue_end -= _cmin
                        _s.last_event_time -= _cmin
                    _canon = calc_metrics(_canon_items)
                    _canon["time_cost"] = metrics.get("time_cost", 0.0)
                    _canon["num_completed_requests"] = len(_canon_items)
                    # Borrow phase-scoped diagnostics (instantaneous counters)
                    # for keys the aggregate calc does not produce.
                    for _k, _v in metrics.items():
                        if _k not in _canon:
                            _canon[_k] = _v
                else:
                    _canon = metrics

                try:
                    _suffix = f"_phase{_phase}" if _phase is not None else ""
                    # [A2] Shadow state snapshot
                    if kda_obs.is_obs_enabled():
                        _shadow_snap = kda_obs.shadow_get_all()
                        metrics["kda_shadow_checkpoints"] = len(_shadow_snap)
                        metrics["kda_shadow_states"] = {
                            k: v.get("state", "unknown")
                            for k, v in _shadow_snap.items()
                        }
                        with open(f"{output_dir}/shadow_state{_suffix}.json", "w") as f:
                            f.write(json.dumps(_shadow_snap, cls=CustomJsonEncoder) + "\n")
                    with open(f"{output_dir}/metrics{_suffix}.json", "w") as f:
                        f.write(json.dumps(metrics, cls=CustomJsonEncoder) + "\n")

                    # Canonical file consumed by bench_serving.
                    with open(f"{output_dir}/metrics.json", "w") as f:
                        f.write(json.dumps(_canon, cls=CustomJsonEncoder) + "\n")

                    with open(f"{output_dir}/iteration{_suffix}.jsonl", "w") as f:
                        for item in C_SchedulerHook.ITERATION_STATS:
                            f.write(json.dumps(item) + "\n")

                    with open(f"{output_dir}/request{_suffix}.jsonl", "w") as f:
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
                    if C_SchedulerHook._L3_QUERY_COUNT > 0:
                        logger.info(
                            f"L3 cache stats: queries={C_SchedulerHook._L3_QUERY_COUNT}, "
                            f"total_hits={C_SchedulerHook._L3_TOTAL_HITS} blocks, "
                            f"prefetched={C_SchedulerHook._L3_PREFETCHED_TOKENS} tokens, "
                            f"per_tier={C_SchedulerHook._L3_HIT_STATS}"
                        )
                    logger.info(
                        f"L2 cache stats: hit_requests={C_SchedulerHook._L2_HIT_COUNT}, "
                        f"hit_tokens={C_SchedulerHook._L2_HIT_TOKENS}"
                    )
                    # Log KDA/MAMBA pool transfer diagnostics
                    _tc = getattr(self, 'tree_cache', None)
                    if _tc is not None:
                        _dw = getattr(_tc, '_hismim_diag_w', {})
                        _dl = getattr(_tc, '_hismim_diag_l', {})
                        if _dw or _dl:
                            logger.info(
                                f"KDA diag: write_calls={_dw.get('calls',0)} "
                                f"mamba_write_xfers={_dw.get('mamba_pt',0)} "
                                f"full_only_writes={_dw.get('full_only',0)} "
                                f"l3_writes={_dw.get('l3_writes',0)} | "
                                f"load_calls={_dl.get('calls',0)} "
                                f"mamba_load_xfers={_dl.get('mamba_pt',0)} "
                                f"full_only_loads={_dl.get('full_only',0)} "
                                f"l3_hits={_dl.get('l3_hits',0)}"
                            )
                    # Log unified memory diagnostics
                    _ud = getattr(C_SchedulerHook, '_unified_diag', None)
                    if _ud is not None:
                        logger.info(
                            f"Unified memory diag: alloc={_ud['alloc']} "
                            f"alloc_extend={_ud['alloc_extend']} "
                            f"alloc_decode={_ud['alloc_decode']} "
                            f"free={_ud['free']} compact={_ud['compact']} "
                            f"compact_pages={_ud.get('compact_pages_moved',0)} "
                            f"flush_peer={_ud['flush_peer']} "
                            f"flush={_ud.get('flush',0)} "
                            f"flush_pages={_ud.get('flush_pages_moved',0)} "
                            f"alloc_fail={_ud['alloc_fail']} "
                            f"gap_bytes={_ud['gap_bytes']} "
                            f"kv_wm={_ud.get('kv_watermark',0)} "
                            f"v2p_valid={_ud.get('v2p_valid_ratio',0):.2f}"
                        )
                    # Log per-request L1/L2/L3 hit breakdown. The aggregated
                    # MLA prefix-hit metrics are computed earlier (before the
                    # files are written) so they persist.
                    for item in metrics_stats:
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
            C_SchedulerHook.REQUEST_STATS.clear()
            C_SchedulerHook.ITERATION_STATS.clear()
            C_SchedulerHook.LAST_CPU_TS = 0
            C_SchedulerHook.LAST_FLUSH_TS = time.time()
            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = False
            C_SchedulerHook._kda_checkpoint_diag = None
            C_SchedulerHook._L3_HIT_STATS = {}
            C_SchedulerHook._L3_QUERY_COUNT = 0
            C_SchedulerHook._L3_TOTAL_HITS = 0
            C_SchedulerHook._L3_PREFETCHED_TOKENS = 0
            C_SchedulerHook._L2_HIT_TOKENS = 0
            C_SchedulerHook._L2_HIT_COUNT = 0

            ProfileReqOutput = getattr(
                importlib.import_module("sglang.srt.managers.io_struct"),
                "ProfileReqOutput",
            )
            result = {
                "total_request": len(stats),
                "output_directory": output_dir,
            }

            return ProfileReqOutput(success=True, message=json.dumps(result))

        target.event_loop_overlap = override_event_loop_overlap
        target.__init__ = wrapped_init
        if hasattr(target, "process_input_requests"):
            target.process_input_requests = wrapped_recv_requests
        elif hasattr(target, "recv_requests"):
            target.recv_requests = wrapped_recv_requests
        target.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_batch_result
        if hasattr(target, "profile"):
            target.profile = wrapped_profile

        # Always wrap Scheduler.__init__ to patch profiler_manager._profile
        # sglang 0.5.18 replaced Scheduler.profile with SchedulerProfilerManager._profile
        # (dispatched via ZMQ), so we hook the profiler_manager after it's created.
        _orig_scheduler_init = target.__init__

        def patched_scheduler_init(self, *args, **kwargs):
            _orig_scheduler_init(self, *args, **kwargs)
            logger.debug("patched_scheduler_init post-init starting")
            try:
                if hasattr(self, "profiler_manager"):
                    _orig_pm_profile = self.profiler_manager._profile

                    def wrapped_pm_profile(recv_req, *a, **kw):
                        if C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                            logger.info("OFFLINE mode: skipping _orig_pm_profile, running HiSim dump only")
                            try:
                                _phase = getattr(C_SchedulerHook, "_offline_phase", 0) + 1
                                C_SchedulerHook._offline_phase = _phase
                                _dump_result = wrapped_profile(self, recv_req, _phase=_phase)
                            except Exception as e:
                                logger.error(f"HiSim profile dump failed: {e}")
                                _dump_result = None
                            # Must return a ProfileReqOutput: the scheduler's
                            # ProfileReq handler sends the return value back to
                            # the HTTP layer. Returning None leaves
                            # /start_profile without a response and the client
                            # (bench_serving) blocks forever.
                            return _dump_result
                        result = _orig_pm_profile(recv_req, *a, **kw)
                        try:
                            wrapped_profile(self, recv_req, *a, **kw)
                        except Exception as e:
                            logger.debug(f"HiSim profile dump failed: {e}")
                        return result
                    self.profiler_manager._profile = wrapped_pm_profile
                    logger.debug("patched profiler_manager._profile for 0.5.18")
                else:
                    logger.warning("Scheduler has no profiler_manager")
            except Exception as e:
                logger.error(f"HiSim profiler patch error: {e}")

        target.__init__ = patched_scheduler_init
