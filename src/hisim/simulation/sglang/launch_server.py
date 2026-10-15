import os
import json
import sys
import argparse
import torch
import types
import hisim.hook as hisim_hook
from hisim.simulation.sglang import sgl_kernel_hook, sglang_hook
from hisim.simulation.sim_args import SimulationArgs
from hisim.utils import get_logger
logger = get_logger("hisim")


# =============================================================================
# CPU Simulation Mode: comprehensive sgl_kernel mock
# =============================================================================
# sglang 0.5.18 requires sgl_kernel (compiled NPU/GPU kernel library) at
# multiple levels. This mock handles all levels so HiSim can run on pure CPU:
#
# Layer 1: module imports (from sgl_kernel import X) → mock sys.modules
# Layer 2: torch.ops.sgl_kernel.X property access → patch _OpNamespace.__getattr__
# Layer 3: torch.library.register_fake("sgl_kernel::X") → wrap to catch RuntimeError
# Layer 4: torch.library.define/impl for sgl_kernel operators → wrap to catch errors
#
# HiSim does NOT need real model weights — it mocks the forward pass and uses
# K3ProfilePredictor for timing. The mock just needs to let sglang's Engine
# initialize enough for the scheduler + hooks to work.
# =============================================================================

_is_cuda = torch.cuda.is_available()
_is_npu = hasattr(torch, "npu") and torch.npu.is_available()
_force_cpu = os.getenv("SGLANG_USE_CPU_ENGINE", "0") == "1"

if (not _is_cuda and not _is_npu) or _force_cpu:

    # --- Layer 1: mock sgl_kernel module ---
    _mock_kernel = types.ModuleType("sgl_kernel")
    _mock_kernel.__path__ = []
    _mock_kernel.__all__ = []

    def _kernel_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)

        def _noop(*args, **kwargs):
            return None
        return _noop
    _mock_kernel.__getattr__ = _kernel_getattr
    sys.modules["sgl_kernel"] = _mock_kernel

    _mock_load_utils = types.ModuleType("sgl_kernel.load_utils")
    _mock_load_utils._load_architecture_specific_ops = lambda *a, **kw: None
    sys.modules["sgl_kernel.load_utils"] = _mock_load_utils

    # Also mock common sub-modules that sglang might import
    for _submod in ["sgl_kernel.kv_cache_utils", "sgl_kernel.allreduce"]:
        _sm = types.ModuleType(_submod)
        _sm.__getattr__ = _kernel_getattr
        sys.modules[_submod] = _sm

    # --- Layer 2: patch _OpNamespace.__getattr__ ---
    # This handles torch.ops.sgl_kernel.X access at class level so forked
    # subprocesses inherit the patch. Returns a no-op callable for any
    # attribute on the sgl_kernel namespace.
    try:
        import torch._ops
        _orig_opns_getattr = torch._ops._OpNamespace.__getattr__

        def _mock_opns_getattr(self, name):
            if getattr(self, "name", "") == "sgl_kernel":
                def _noop(*args, **kwargs):
                    return None
                return _noop
            return _orig_opns_getattr(self, name)
        torch._ops._OpNamespace.__getattr__ = _mock_opns_getattr
    except Exception as e:
        logger.warning(f"could not patch _OpNamespace: {e}")

    # --- Layer 3: wrap torch.library.register_fake ---
    # sglang's register_fake_if_exists decorator checks hasattr(torch.ops.sgl_kernel,
    # op_name) — our __getattr__ mock makes this True, then register_fake tries
    # to check the C++ registry and raises RuntimeError if the operator doesn't
    # exist. This wrapper catches that RuntimeError and silently skips.
    try:
        _orig_register_fake = torch.library.register_fake

        def _safe_register_fake(op_name, func=None, **kwargs):
            try:
                return _orig_register_fake(op_name, func, **kwargs)
            except (RuntimeError, Exception):
                pass  # operator not in C++ registry — skip fake registration
        torch.library.register_fake = _safe_register_fake
    except Exception:
        pass

    # --- Layer 4: wrap torch.library.define and impl ---
    # Some sglang code paths may try to define or implement sgl_kernel operators.
    # Wrap these to silently skip when the namespace doesn't exist in C++.
    try:
        _orig_define = torch.library.define

        def _safe_define(op_name, schema, **kwargs):
            try:
                return _orig_define(op_name, schema, **kwargs)
            except (RuntimeError, Exception):
                # Return a dummy object to avoid None issues downstream
                class _DummyDef:
                    def __init__(self):
                        self.default = lambda *a, **kw: None
                return _DummyDef()
        torch.library.define = _safe_define
    except Exception:
        pass

    # --- Layer 5: wrap torch._C._dispatch_has_kernel_for_dispatch_key ---
    # This is the low-level C++ check that raises RuntimeError. By wrapping it
    # for sgl_kernel operators, we prevent the error at its source.
    try:
        _orig_has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key

        def _safe_has_kernel(qualname, dispatch_key):
            if "sgl_kernel" in str(qualname):
                return False  # pretend operator doesn't exist
            return _orig_has_kernel(qualname, dispatch_key)
        torch._C._dispatch_has_kernel_for_dispatch_key = _safe_has_kernel
    except Exception:
        pass

    hisim_hook.install_module_hooks([sgl_kernel_hook.M_SGLangKernelLoadUtilHook])

    # Patch: mock torch.cuda.cudart for CPU mode
    # sglang 0.5.18's hierarchical cache calls cudart.cudaHostRegister which
    # is None on CPU. Provide a mock that returns cudaSuccess (0).
    class _MockCudart:
        def cudaHostRegister(self, *a, **kw): return 0
        def cudaHostUnregister(self, *a, **kw): return 0
        def cudaGetErrorString(self, *a, **kw): return "mocked"
    _orig_cudart_fn = torch.cuda.cudart
    def _mock_cudart_fn():
        if torch.cuda.is_available():
            return _orig_cudart_fn()
        return _MockCudart()
    torch.cuda.cudart = _mock_cudart_fn

    # Inject no-op transfer_kv_* into mha module namespace.
    # sgl 0.5.18's pool_host/mha.py conditionally imports these from
    # sgl_kernel.kvcacheio only when _is_cuda or _is_hip. On CPU, the import
    # is skipped but the calls remain, causing NameError during prefill
    # process_batch_result (KV host write-back).
    import sglang.srt.mem_cache.pool_host.mha as _mha_mod
    for _fn in (
        "transfer_kv_all_layer", "transfer_kv_all_layer_direct_lf_pf",
        "transfer_kv_all_layer_lf_pf", "transfer_kv_all_layer_lf_ph",
        "transfer_kv_all_layer_mla_lf_pf", "transfer_kv_direct",
        "transfer_kv_per_layer",
    ):
        if not hasattr(_mha_mod, _fn):
            setattr(_mha_mod, _fn, lambda *a, **kw: None)

    logger.info("CPU simulation mode: sgl_kernel fully mocked (5 layers) + cudart mocked + transfer_to_npu disabled")

    # Force K3 to be detected as a hybrid_ssm model (MLA + Mamba/KDA).
    # sglang's kv_cache_builder calls kimi_linear_config(model_config) to detect
    # K3's KDA layers. In the mock environment, hf_config may not be parsed as
    # KimiLinearConfig, so the detection fails → hybrid_ssm=False → MAMBA
    # component not added to tree_components → no two-pool hierarchical cache.
    # Patch kimi_linear_config in the kv_cache_builder namespace to force
    # detection for K3.
    try:
        import sglang.srt.mem_cache.kv_cache_builder as _kvcb_mod
        _orig_klc = _kvcb_mod.kimi_linear_config
        def _forced_klc(model_config):
            result = _orig_klc(model_config)
            if result is not None:
                return result
            # K3's top-level model_type is "kimi_k3", text_config has
            # model_type="kimi_linear". In mock env, text_config may be a
            # plain dict so getattr fails — use model_type as fallback.
            _hf = getattr(model_config, "hf_config", None)
            _model_type = getattr(_hf, "model_type", "")
            _tc = getattr(_hf, "text_config", None)
            _tc_mt = ""
            if isinstance(_tc, dict):
                _tc_mt = _tc.get("model_type", "")
            else:
                _tc_mt = getattr(_tc, "model_type", "")
            if _model_type == "kimi_k3" or _tc_mt == "kimi_linear":
                return True  # non-None → is_hybrid_ssm=True
            return None
        _kvcb_mod.kimi_linear_config = _forced_klc
        # Also patch the source module — kv_cache_configurator imports directly
        # from hybrid_arch, not from kv_cache_builder, so the patch above
        # doesn't affect the configurator's copy.
        import sglang.srt.configs.hybrid_arch as _ha_mod
        _ha_mod.kimi_linear_config = _forced_klc

        # For unified memory mode, mambaish_config needs mamba2_cache_params
        # with layers/mamba_cache_per_req/shape. Use the real K3 values from
        # config: num_heads=96, head_dim=128, conv_kernel=4, TP=8.
        # KDA layers (0-indexed): [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
        # Full attn layers (0-indexed): [3,7,11,15,19]
        # Note: K3 config uses 1-indexed layer numbers, but sglang internally
        # uses 0-indexed. The configurator filters layers with start_layer <= i < end_layer
        # where end_layer=23 (num_hidden_layers). Layer 23 (1-indexed = layer 22 0-indexed)
        # would pass, but layer 23 as a raw int fails i < 23.
        # We must use 0-indexed layer IDs to match sglang's internal convention.
        from types import SimpleNamespace as _NS
        _k3_kda_layers_0idx = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
        _k3_full_attn_layers_0idx = [3,7,11,15,19]
        try:
            from sglang.srt.configs.mamba_utils import (
                KimiLinearCacheParams, KimiLinearStateShape, mamba2_state_dtype,
            )
            _k3_shape = KimiLinearStateShape.create(
                tp_world_size=8,  # Match real TP8 deployment (12 heads/die instead of 96)
                num_heads=96,
                head_dim=128,
                conv_kernel_size=4,
            )
            _mock_mamba2 = KimiLinearCacheParams(
                shape=_k3_shape,
                layers=_k3_kda_layers_0idx,  # 0-indexed, 18 layers
                dtype=mamba2_state_dtype(None),
            )
            logger.info("Created real KimiLinearCacheParams for K3 unified memory (0-indexed layers)")
        except Exception as _e:
            logger.warning(f"Failed to create KimiLinearCacheParams: {_e}, using SimpleNamespace fallback")
            _mock_mamba2 = _NS(
                layers=_k3_kda_layers_0idx,  # 0-indexed, 18 layers
                shape=_NS(conv=[(3, 4608)], temporal=(12, 128, 128), num_k_heads_per_tp=12, is_kda=True, conv_slice_axis=1, disable_conv_window_dedup=True, num_heads=96, head_dim=128, num_k_heads=96, head_k_dim=128, conv_kernel=4, num_spec=0),
                dtype=_NS(conv=torch.bfloat16, temporal=torch.float32, itemsize=4),
                mamba_cache_per_req=14653440,  # TP=8: 18 layers × 814080 B/slot/layer
                replayssm_ring_bytes_per_req=lambda *a, **k: 0,
                is_kda=True,
            )
        _ha_mod_mambaish_orig = _ha_mod.mambaish_config
        def _forced_mambaish(model_config):
            result = _ha_mod_mambaish_orig(model_config)
            # Only return if result is a real config (not bool from our kimi_linear patch)
            if result is not None and not isinstance(result, bool):
                return result
            # If kimi_linear_config detected K3, return mock with mamba2_cache_params
            if _forced_klc(model_config) is not None:
                return _NS(
                    mamba2_cache_params=_mock_mamba2,
                    full_attention_layer_ids=_k3_full_attn_layers_0idx,
                    architectures=["KimiK3ForConditionalGeneration"],
                    model_type="kimi_linear",
                )
            return None
        _ha_mod.mambaish_config = _forced_mambaish
        # Also patch in kv_cache_configurator (it imports mambaish_config at class level)
        import sglang.srt.mem_cache.kv_cache_configurator as _kcc_mod
        _kcc_mod.KVCacheConfigurator.__init__.__wrapped__ = None  # no-op if wrapped
        _kcc_mod.mambaish_config = _forced_mambaish
        logger.info("Patched kimi_linear_config + mambaish_config to force K3 unified memory detection")
    except Exception as e:
        logger.warning(f"Failed to patch kimi_linear_config: {e}")

    # Patch _MambaStrategy.matches to accept MockTokenToKVPool.
    # The real check is isinstance(kvcache, HybridLinearKVPool), but HiSim
    # uses MockTokenToKVPool. We relax the check to also accept our mock.
    try:
        import sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler as _hpa
        _orig_mamba_matches = _hpa._MambaStrategy.matches
        def _mocked_mamba_matches(self, kvcache, components):
            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
            if components == {ComponentType.FULL, ComponentType.MAMBA}:
                if type(kvcache).__name__ == "MockTokenToKVPool":
                    return True
            return _orig_mamba_matches(self, kvcache, components)
        _hpa._MambaStrategy.matches = _mocked_mamba_matches
        logger.info("Patched _MambaStrategy.matches to accept MockTokenToKVPool")
    except Exception as e:
        logger.warning(f"Failed to patch _MambaStrategy: {e}")

    # Patch: relax MambaComponent's HybridReqToTokenPool isinstance assert.
    # sglang 0.5.19's MambaComponent.__init__ asserts
    # isinstance(params.req_to_token_pool, HybridReqToTokenPool); HiSim passes
    # its MockReqToTokenPool. Rebind the memory_pool attribute to a subclass
    # whose metaclass accepts the mock in isinstance checks (real pools still
    # pass; the class remains constructible).
    #
    # IMPORTANT: several sglang modules do a *top-level*
    # `from ...memory_pool import HybridReqToTokenPool`, which binds the
    # ORIGINAL class object at import time. Rebinding only memory_pool is not
    # enough: e.g. allocation.alloc_req_slots does
    # `isinstance(req_to_token_pool, HybridReqToTokenPool)` against its own
    # stale binding, that check fails for the mock, and the native mamba
    # pre-eviction is silently skipped -> `alloc` returns None ->
    # "alloc_req_slots runs out of memory" as soon as mamba is under pressure.
    # Rebind every module that already captured the original.
    try:
        import sglang.srt.mem_cache.memory_pool as _mp
        _OrigHybridPool = _mp.HybridReqToTokenPool

        class _HybridPoolMeta(type(_OrigHybridPool)):
            def __instancecheck__(cls, instance):
                if type(instance).__name__ == "MockReqToTokenPool":
                    return True
                # Any genuine HybridReqToTokenPool (or subclass) instance.
                return isinstance(instance, _OrigHybridPool)

        _HybridPoolAlias = _HybridPoolMeta(
            "HybridReqToTokenPool", (_OrigHybridPool,), {}
        )
        _mp.HybridReqToTokenPool = _HybridPoolAlias

        # Sweep all *already-loaded* modules for stale top-level bindings.
        # NOTE: do NOT force-import extra sglang modules here. Doing so during
        # launch_server's module body changes import ordering for the scheduler
        # process and corrupts later device/host pool construction (observed:
        # MockTokenToKVPoolHost sized the host KV buffer from mock 1x1 head dims
        # -> a 1.47 PB allocation). Only touch what is already imported; the
        # scheduler imports allocation/common well before it builds the cache.
        import sys as _sys

        _rebound = []
        for _mname, _m in list(_sys.modules.items()):
            try:
                if getattr(_m, "HybridReqToTokenPool", None) is _OrigHybridPool:
                    setattr(_m, "HybridReqToTokenPool", _HybridPoolAlias)
                    _rebound.append(_mname)
            except Exception:
                pass
        logger.info(
            "Patched HybridReqToTokenPool isinstance to accept MockReqToTokenPool "
            f"(rebound: {_rebound})"
        )
    except Exception as e:
        logger.warning(f"Failed to patch HybridReqToTokenPool: {e}")

    # Patch MultiEndedAllocator with observation counters.
    # In unified memory mode, MultiEndedAllocator manages the dynamic split
    # between KV and Mamba sub-pools. These counters verify unified memory
    # is actually used and provide data for fragmentation analysis.
    try:
        from sglang.srt.mem_cache.multi_ended_allocator import MultiEndedAllocator as _MEA
        _unified_diag = {
            "alloc": 0, "alloc_extend": 0, "alloc_decode": 0,
            "free": 0, "compact": 0, "flush_peer": 0,
            "gap_bytes": 0, "alloc_fail": 0,
            "compact_pages_moved": 0,
            "flush": 0, "flush_pages_moved": 0,
            "kv_watermark": 0, "mamba_watermark": 0,
            "mla_live_pages": 0, "mamba_live_pages": 0,
            "peak_mla_live_pages": 0, "peak_mamba_live_pages": 0,
            "peak_mla_watermark": 0, "peak_mamba_watermark": 0,
            "v2p_valid_ratio": 0.0,
        }
        _orig_mea_alloc = _MEA.alloc
        _orig_mea_alloc_extend = getattr(_MEA, 'alloc_extend', None)
        _orig_mea_alloc_decode = getattr(_MEA, 'alloc_decode', None)
        _orig_mea_free = _MEA.free
        _orig_mea_compact = _MEA._compact_pending
        _orig_mea_flush = _MEA._flush
        _orig_mea_flush_peer = _MEA._flush_peer_for_alloc

        def _sample_watermarks(self):
            try:
                _unified_diag["gap_bytes"] = self._current_gap_bytes()
                _wm = int(getattr(self, 'watermark_physical', 0))
                _spn = getattr(self, 'sub_pool_name', '')
                _lc = int(getattr(self, 'live_page_count', 0))
                if _spn == 'full':
                    _unified_diag["kv_watermark"] = _wm
                    _unified_diag["mla_live_pages"] = _lc
                    if _lc > _unified_diag["peak_mla_live_pages"]:
                        _unified_diag["peak_mla_live_pages"] = _lc
                    if _wm > _unified_diag["peak_mla_watermark"]:
                        _unified_diag["peak_mla_watermark"] = _wm
                elif _spn == 'mamba':
                    _unified_diag["mamba_watermark"] = _wm
                    _unified_diag["mamba_live_pages"] = _lc
                    if _lc > _unified_diag["peak_mamba_live_pages"]:
                        _unified_diag["peak_mamba_live_pages"] = _lc
                    if _wm > _unified_diag["peak_mamba_watermark"]:
                        _unified_diag["peak_mamba_watermark"] = _wm
            except Exception:
                pass
            try:
                _unified_diag["v2p_valid_ratio"] = float(
                    (self.virtual_to_physical >= 0).sum().item()
                ) / max(1, self.virtual_to_physical.numel())
            except Exception:
                pass

        def _diag_alloc(self, need_size):
            _unified_diag["alloc"] += 1
            _sample_watermarks(self)
            # P0 diagnostics: print allocator config on first alloc
            if _unified_diag["alloc"] == 1:
                try:
                    _zc = getattr(self, '_zero_pages_on_alloc', 'N/A')
                    _kvc = type(self._kvcache).__name__ if hasattr(self, '_kvcache') else 'N/A'
                    _ps = getattr(self, 'page_size', 'N/A')
                    _lc = getattr(self, 'lazy_compaction', 'N/A')
                    _np = getattr(self, 'num_pages', 'N/A')
                    _wm = int(getattr(self, 'watermark_physical', 0))
                    _av = self.available_size() if hasattr(self, 'available_size') else 'N/A'
                    _fvi = self.free_virtual_ids.numel() if hasattr(self, 'free_virtual_ids') else 'N/A'
                    _gd = getattr(self, 'grow_direction', 'N/A')
                    _fpp = self._free_phys_pages.numel() if hasattr(self, '_free_phys_pages') else 'N/A'
                    logger.info(f"[P0-DIAG] alloc#1: _zero_pages_on_alloc={_zc} kvcache={_kvc} page_size={_ps} lazy_compaction={_lc} num_pages={_np} watermark={_wm} available={_av} free_virt={_fvi} grow_dir={_gd} free_phys_pages={_fpp} need={need_size}")
                except Exception as _e:
                    logger.warning(f"[P0-DIAG] failed to print alloc config: {_e}")
            _r = _orig_mea_alloc(self, need_size)
            if _r is None:
                _unified_diag["alloc_fail"] += 1
            return _r
        def _diag_alloc_extend(self, *a, **kw):
            _unified_diag["alloc_extend"] += 1
            _sample_watermarks(self)
            return _orig_mea_alloc_extend(self, *a, **kw)
        def _diag_alloc_decode(self, *a, **kw):
            _unified_diag["alloc_decode"] += 1
            return _orig_mea_alloc_decode(self, *a, **kw)
        def _diag_free(self, free_index):
            _unified_diag["free"] += 1
            return _orig_mea_free(self, free_index)
        def _diag_compact(self, freed_physical_pages):
            _unified_diag["compact"] += 1
            try:
                _unified_diag["compact_pages_moved"] += len(freed_physical_pages)
            except Exception:
                pass
            _r = _orig_mea_compact(self, freed_physical_pages)
            _sample_watermarks(self)
            return _r
        def _diag_flush_peer(self, need_tokens):
            _unified_diag["flush_peer"] += 1
            _r = _orig_mea_flush_peer(self, need_tokens)
            _sample_watermarks(self)
            return _r
        def _diag_flush(self, urgent=False):
            _unified_diag["flush"] += 1
            _wm_before = int(getattr(self, 'watermark_physical', 0))
            _r = _orig_mea_flush(self, urgent=urgent)
            _wm_after = int(getattr(self, 'watermark_physical', 0))
            _moved = _wm_before - _wm_after
            if _moved > 0:
                _unified_diag["flush_pages_moved"] += _moved
            _sample_watermarks(self)
            return _r
        _MEA.alloc = _diag_alloc
        if _orig_mea_alloc_extend:
            _MEA.alloc_extend = _diag_alloc_extend
        if _orig_mea_alloc_decode:
            _MEA.alloc_decode = _diag_alloc_decode
        _MEA.free = _diag_free
        _MEA._compact_pending = _diag_compact
        _MEA._flush = _diag_flush
        _MEA._flush_peer_for_alloc = _diag_flush_peer

        # P2: Hook _maybe_zero_pages to trace zero_pages kernel calls
        _orig_mea_zero = _MEA._maybe_zero_pages
        def _diag_zero(self, phys_pages):
            try:
                _zc = getattr(self, '_zero_pages_on_alloc', 'N/A')
                _kvc = type(self._kvcache).__name__ if hasattr(self, '_kvcache') else 'N/A'
                _pp_info = f"numel={phys_pages.numel()} min={phys_pages.min().item()} max={phys_pages.max().item()}" if phys_pages.numel() > 0 else "empty"
                # Print kvcache internal state for MLA pool
                _kv = self._kvcache if hasattr(self, '_kvcache') else None
                _extra = ""
                if _kv is not None and hasattr(_kv, '_num_pages'):
                    _np = _kv._num_pages
                    _pb = _kv._page_bytes
                    _raw_numel = _kv._unified_buffer._raw.numel() if hasattr(_kv, '_unified_buffer') else 'N/A'
                    _raw_dev = _kv._unified_buffer._raw.device if hasattr(_kv, '_unified_buffer') else 'N/A'
                    _max_page = phys_pages.max().item() if phys_pages.numel() > 0 else 0
                    _oob = "OOB!" if _max_page >= _np else "ok"
                    _byte_offset = _max_page * _pb
                    _extra = f" kvcache_num_pages={_np} page_bytes={_pb} raw_numel={_raw_numel} raw_dev={_raw_dev} max_offset={_byte_offset}({_oob})"
                logger.info(f"[P2-ZERO] enter _maybe_zero_pages: _zero_pages_on_alloc={_zc} kvcache={_kvc} phys_pages[{_pp_info}]{_extra}")
            except Exception as _e:
                logger.warning(f"[P2-ZERO] diag failed: {_e}")
            return _orig_mea_zero(self, phys_pages)
        _MEA._maybe_zero_pages = _diag_zero
        sglang_hook.C_SchedulerHook._unified_diag = _unified_diag
        logger.info("Patched MultiEndedAllocator with observation counters (Phase 2: watermark+v2p)")
    except Exception as e:
        logger.warning(f"Failed to patch MultiEndedAllocator: {e}")

    # Patch support_triton to return False in CPU simulation mode.
    # In unified memory mode, allocation.py uses Triton kernels for
    # write_req_to_token_pool — these crash on NPU in HiSim simulation.
    # Forcing False makes it use the Python fallback path.
    # Must patch at the import source (sglang.srt.mem_cache.common) because
    # allocation.py does `from ...common import support_triton` which binds
    # the function reference at import time.
    try:
        import sglang.srt.utils.common as _common
        _common.support_triton = lambda backend: False
        import sglang.srt.mem_cache.common as _mc_common
        _mc_common.support_triton = lambda backend: False
        import sglang.srt.mem_cache.allocation as _alloc_mod
        _alloc_mod.support_triton = lambda backend: False
        logger.info("Patched support_triton to False (allocation.py + common modules)")
    except Exception as e:
        logger.warning(f"Failed to patch support_triton: {e}")

    # Patch zero_pages to add CPU fallback.
    # The original zero_pages always launches a Triton kernel, but the unified
    # pool buffer (_raw) is on CPU in HiSim (SGLANG_USE_CPU_ENGINE=1).
    # alloc_bind_inplace already has an is_cuda check with CPU fallback;
    # zero_pages does not — this patches it to match.
    try:
        from sglang.kernels.ops.kvcache.zero_pages import _zero_pages_kernel, _BLOCK
        import sglang.kernels.ops.kvcache.zero_pages as _zp_mod
        import torch as _torch

        _orig_zero_pages = _zp_mod.zero_pages

        def _zero_pages_cpu_fallback(raw, pages, num_pages, page_bytes):
            m = int(pages.numel())
            if m == 0:
                return
            assert raw.dtype == _torch.uint8
            assert page_bytes % 8 == 0
            page_words = page_bytes // 8
            words = raw[: num_pages * page_bytes].view(_torch.int64)
            pg_list = pages.to(_torch.int64).tolist()
            for pg in pg_list:
                words[pg * page_words : (pg + 1) * page_words].zero_()

        def _zero_pages_patched(raw, pages, num_pages, page_bytes):
            if not raw.is_cuda:
                _zero_pages_cpu_fallback(raw, pages, num_pages, page_bytes)
                return
            _orig_zero_pages(raw, pages, num_pages, page_bytes)

        _zp_mod.zero_pages = _zero_pages_patched
        # Also patch the imported reference in unified_memory_pool.py
        # (from ... import zero_pages binds at import time)
        import sglang.srt.mem_cache.unified_memory_pool as _ump
        _ump.zero_pages = _zero_pages_patched
        logger.info("Patched zero_pages with CPU fallback (is_cuda check) [module + unified_memory_pool]")
    except Exception as e:
        logger.warning(f"Failed to patch zero_pages: {e}")

    # Patch get_available_gpu_memory for CPU device to NOT divide by NUMA nodes.
    # The original code divides psutil.virtual_memory().available by n_numa_node,
    # limiting the budget to one NUMA node (~212GB on a 2TB/8-node machine).
    # For HiSim simulation, we simulate large HBM/DRAM caches on CPU — the buffer
    # is real memory but not performance-critical (mock forward). Using the full
    # available memory lets us simulate realistic cache sizes (e.g. 1M tokens).
    try:
        import sglang.srt.utils as _utils_mod
        _orig_get_avail = _utils_mod.get_available_gpu_memory

        def _patched_get_available(device, gpu_id, distributed=False, empty_cache=True, cpu_group=None):
            if device == "cpu":
                import psutil
                total_free = psutil.virtual_memory().available
                # Return in GB (matching original get_available_gpu_memory),
                # but without dividing by NUMA nodes — use full system memory.
                _free_gb = total_free / (1 << 30)
                logger.info(
                    f"HiSim CPU memory budget: {_free_gb:.1f} GB "
                    f"(full system, not divided by NUMA nodes)"
                )
                return _free_gb
            return _orig_get_avail(device, gpu_id, distributed=distributed,
                                   empty_cache=empty_cache, cpu_group=cpu_group)

        _utils_mod.get_available_gpu_memory = _patched_get_available
        # Also patch the import-bound reference in kv_cache_configurator,
        # which does `from sglang.srt.utils import get_available_gpu_memory`
        # at module load time — binding the original function before our patch.
        import sglang.srt.mem_cache.kv_cache_configurator as _cfg_mod
        _cfg_mod.get_available_gpu_memory = _patched_get_available
        logger.info("Patched get_available_gpu_memory: CPU uses full system memory (no NUMA division)")
    except Exception as e:
        logger.warning(f"Failed to patch get_available_gpu_memory: {e}")

    # HiSim MLA Capacity Simulation Mode
    # K3 uses MLA (kv_lora_rank=512, qk_rope_head_dim=64, entry_bytes=5760 B/token)
    # But use_mla_backend=True triggers real MLA NPU kernels that crash in simulation.
    # This patch keeps use_mla_backend=False (MHA infrastructure, no MLA kernels)
    # but overrides model_config head_num/head_dim so that MHASubPoolSpec's
    # entry_bytes() and view creation both use MLA-correct dimensions:
    #   head_num=1, head_dim=288 → entry_bytes = 5×1×288×2(K+V)×2(bf16) = 5760
    # The forward pass is mocked (never reads KV buffer), so view shapes are
    # irrelevant for computation — only byte-level capacity matters.
    try:
        from sglang.srt.configs.model_config import ModelConfig
        _orig_mc_post_init = ModelConfig.post_init if hasattr(ModelConfig, 'post_init') else None

        # Patch at import time: when ModelConfig is instantiated for K3,
        # override head_num and head_dim for correct MLA sizing.
        _orig_init = ModelConfig.__init__
        def _hisim_mc_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            # Only override for K3 (detected by kv_lora_rank presence)
            if getattr(self, 'kv_lora_rank', None) is not None and not getattr(self, '_hisim_overridden', False):
                self.head_num = 1
                # head_dim = (kv_lora_rank + qk_rope_head_dim) / 2 = (512+64)/2 = 288
                # Because MHA entry_bytes = layer_num × head_num × head_dim × 2(K+V) × itemsize
                # We need: 5 × 1 × head_dim × 2 × 2 = 5760 → head_dim = 288
                self.head_dim = 288
                self._hisim_overridden = True
                logger.info(f"HiSim MLA capacity mode: head_num=1, head_dim=288 (entry_bytes=5760)")
        ModelConfig.__init__ = _hisim_mc_init
        logger.info("Patched ModelConfig.__init__ for HiSim MLA capacity simulation")
    except Exception as e:
        logger.warning(f"Failed to patch ModelConfig: {e}")

    # HiSim Mamba State Mock: Patch MambaPool data operations to no-ops.
    # The forward pass is mocked, so actual Mamba state data is irrelevant.
    # But mamba_component.py calls copy_from/load_cpu_copy/get_cpu_copy during
    # checkpoint donate/match/load — these trigger NPU fused kernels that crash.
    # Two Pools mode uses _MockMambaPool (catch-all no-op), but One Pool uses
    # UnifiedMambaPool (real MambaPool). This patch makes the data operations
    # no-ops while keeping mamba_allocator (MultiEndedAllocator) fully functional
    # — alloc/free/compact/flush_peer still work, tracking the dynamic KV/Mamba split.
    try:
        from sglang.srt.mem_cache.memory_pool import MambaPool
        _orig_copy_from = MambaPool.copy_from
        _orig_get_cpu_copy = MambaPool.get_cpu_copy
        _orig_load_cpu_copy = MambaPool.load_cpu_copy

        def _noop_copy_from(self, src_indices, dst_indices):
            logger.debug(f"MambaPool.copy_from: no-op (src={src_indices}, dst={dst_indices})")
        def _noop_get_cpu_copy(self, indices):
            return None  # HiSim doesn't need CPU copies of Mamba state
        def _noop_load_cpu_copy(self, mamba_cache_cpu, indices):
            pass  # No-op: loading CPU copy back

        MambaPool.copy_from = _noop_copy_from
        MambaPool.get_cpu_copy = _noop_get_cpu_copy
        MambaPool.load_cpu_copy = _noop_load_cpu_copy
        logger.info("Patched MambaPool.copy_from/get_cpu_copy/load_cpu_copy to no-ops (NPU kernel bypass)")
    except Exception as e:
        logger.warning(f"Failed to patch MambaPool: {e}")

    # Patch _handle_page_major_kv_layout to allow torch_native backend.
    # sglang requires Triton for page-major-kv-layout (unified memory),
    # but HiSim mocks the forward pass — Triton attention backend
    # executes real NPU kernels (compute_position_kernel) that crash.
    # Allowing torch_native lets HiSim skip Triton kernel execution.
    try:
        import sglang.srt.server_args as _sa_mod
        _orig_handle_pml = _sa_mod.ServerArgs._handle_page_major_kv_layout
        def _patched_handle_pml(self):
            if getattr(self, 'enable_page_major_kv_layout', False):
                # Relax: allow torch_native alongside triton
                # The original asserts backends <= allowed_full;
                # we add torch_native to allowed_full.
                _orig_handle_pml(self)
            # If not page_major, original is a no-op
        # Instead of wrapping, just monkey-patch the assert by adding torch_native
        # to the allowed set. The simplest approach: override the whole method.
        def _relaxed_handle_pml(self):
            if not getattr(self, 'enable_page_major_kv_layout', False):
                return
            # Original logic but with relaxed backend checks — skip the
            # assertion that requires Triton. HiSim's forward is mocked,
            # so the attention backend is never actually used for computation.
            # Just ensure the mamba_backend is set.
            if self.mamba_backend is None:
                self.mamba_backend = "triton"
        _sa_mod.ServerArgs._handle_page_major_kv_layout = _relaxed_handle_pml
        logger.info("Patched _handle_page_major_kv_layout to allow torch_native (skip Triton assertion)")
    except Exception as e:
        logger.warning(f"Failed to patch _handle_page_major_kv_layout: {e}")

    # Patch _handle_unified_memory_pool to bypass the mutual-exclusion assert
    # that prevents enable_unified_memory + enable_hierarchical_cache together.
    # We temporarily disable hierarchical_cache to pass the assert, then restore.
    try:
        _orig_handle_ump = _sa_mod.ServerArgs._handle_unified_memory_pool
        def _patched_handle_ump(self):
            _hier = getattr(self, 'enable_hierarchical_cache', False)
            _lmc = getattr(self, 'enable_lmcache', False)
            if _hier:
                self.enable_hierarchical_cache = False
            if _lmc:
                self.enable_lmcache = False
            _orig_handle_ump(self)
            if _hier:
                self.enable_hierarchical_cache = _hier
            if _lmc:
                self.enable_lmcache = _lmc
        _sa_mod.ServerArgs._handle_unified_memory_pool = _patched_handle_ump
        logger.info("Patched _handle_unified_memory_pool: bypass unified+hierarchical mutual exclusion")
    except Exception as e:
        logger.warning(f"Failed to patch _handle_unified_memory_pool: {e}")

    # Patch HybridCacheController backup/prefetch thread funcs to no-op.
    # C_HiCacheController.hook overrides these on HiCacheController class,
    # but HybridCacheController has its own methods that shadow them.
    # Without this patch, prepare_for_decode calls the real backup_thread_func
    # which accesses NPU → SIGABRT in unified memory mode.
    try:
        import sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller as _hcc_mod
        _hcc_mod.HybridCacheController.backup_thread_func = lambda self, *a, **kw: None
        _hcc_mod.HybridCacheController.prefetch_thread_func = lambda self, *a, **kw: None
        logger.info("Patched HybridCacheController.backup_thread_func + prefetch_thread_func to no-op")
    except Exception as e:
        logger.warning(f"Failed to patch HybridCacheController thread funcs: {e}")

    # Log init_unified_mamba_pools parameters to debug tensor overflow
    try:
        import sglang.srt.mem_cache.unified_memory_pool as _ump_mod
        _orig_init_ump = _ump_mod.init_unified_mamba_pools
        def _logged_init_ump(**kwargs):
            _mtnt = kwargs.get('max_total_num_tokens', 'N/A')
            _mmcs = kwargs.get('max_mamba_cache_size', 'N/A')
            _mnr = kwargs.get('max_num_reqs', 'N/A')
            _umlb = kwargs.get('use_mla_backend', 'N/A')
            _klr = kwargs.get('kv_lora_rank', 'N/A')
            logger.info(f"init_unified_mamba_pools params: max_total_num_tokens={_mtnt}, max_mamba_cache_size={_mmcs}, max_num_reqs={_mnr}, use_mla_backend={_umlb}, kv_lora_rank={_klr}")
            result = _orig_init_ump(**kwargs)
            return result
        _ump_mod.init_unified_mamba_pools = _logged_init_ump
        logger.info("Patched init_unified_mamba_pools with parameter logging")
    except Exception as e:
        logger.warning(f"Failed to patch init_unified_mamba_pools: {e}")

# =============================================================================
# Install HiSim hooks (must be before sglang import)
# =============================================================================
logger.info("Installing class hooks...")
try:
    hisim_hook.install_class_hooks(
        [
            sglang_hook.C_SchedulerHook,
            sglang_hook.C_ModelRunnerHook,
            sglang_hook.C_TokenizerManagerHook,
            sglang_hook.C_StorageBackendFactory,
            sglang_hook.C_HiCacheController,
            sglang_hook.C_HiRadixCacheHook,
        ]
    )
    logger.info("Class hooks installed successfully")
except Exception as e:
    logger.error(f"ERROR installing class hooks: {e}")
    import traceback
    traceback.print_exc()

# Manually apply C_HiRadixCacheHook to UnifiedRadixCache if it was already
# imported (by kimi_linear_config/_MambaStrategy patches above). The class
# creation hook mechanism only fires on NEW class creation, so if the class
# was imported before install_class_hooks, the hook wasn't applied.
try:
    import sglang.srt.mem_cache.unified_radix_cache as _urc_mod
    if hasattr(_urc_mod, 'UnifiedRadixCache'):
        sglang_hook.C_HiRadixCacheHook.hook(_urc_mod.UnifiedRadixCache)
        logger.info("Manually applied C_HiRadixCacheHook to UnifiedRadixCache")
except Exception as e:
    logger.warning(f"Failed to manually apply C_HiRadixCacheHook: {e}")



# Ref: https://github.com/sgl-project/sglang/blob/v0.5.6.post2/python/sglang/launch_server.py
if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree
    from sglang.version import __version__ as sglang_version
    from hisim.simulation.sglang.version import COMPATIBLE_VERSIONS

    if sglang_version not in COMPATIBLE_VERSIONS:
        logger.warning(
            f"Current SGLang version {sglang_version} is not in the compatible versions "
            f"{COMPATIBLE_VERSIONS}, so errors may occur."
        )

    # CPU simulation: patch health check to always return 200.
    # The real /health fires a health_generate roundtrip through the
    # scheduler+detokenizer, which the CPU simulation cannot always serve
    # (mocked forward). Remove the stock route and register ours so FastAPI
    # does not shadow it (routes match in registration order).
    if _force_cpu or (not _is_cuda and not _is_npu):
        try:
            from sglang.srt.entrypoints.http_server import app as _fastapi_app

            _fastapi_app.router.routes = [
                r
                for r in _fastapi_app.router.routes
                if getattr(r, "path", None) != "/health"
            ]

            @_fastapi_app.get("/health")
            async def _mock_health():
                return {"status": "ok"}

            logger.info("Patched /health to always return 200 (CPU simulation)")
        except Exception as e:
            logger.warning(f"could not patch /health: {e}")

    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(g)

    g = parser.add_argument_group("simulation")
    SimulationArgs.add_cli_args(g)

    raw_args = parser.parse_args(sys.argv[1:])

    # K3 (kimi_linear) relies on the extra_buffer mamba radix strategy: the KDA
    # checkpoint donation is built on ping-pong track buffers, and the HiSim
    # MockReqToTokenPool mirrors that strategy. sglang 0.5.19's MambaComponent
    # reads enable_mamba_extra_buffer from ServerArgs (not from the pool), so if
    # the cache resolves to "no_buffer" while the mock allocates ping-pong
    # buffers the two disagree and prepare_for_caching_req takes the
    # mamba_pool.copy_from path. Force the strategy for K3 unless the user set
    # it explicitly.
    try:
        import json as _json
        _mpath = getattr(raw_args, "model_path", None)
        _is_k3 = False
        if _mpath and os.path.exists(os.path.join(_mpath, "config.json")):
            with open(os.path.join(_mpath, "config.json")) as _cf:
                _mc = _json.load(_cf)
            _mt = _mc.get("model_type", "")
            _tc = _mc.get("text_config")
            _tcmt = _tc.get("model_type", "") if isinstance(_tc, dict) else ""
            _is_k3 = _mt in ("kimi_k3", "kimi_linear") or _tcmt == "kimi_linear"
        if _is_k3 and getattr(raw_args, "mamba_radix_cache_strategy", None) in (
            None,
            "auto",
        ):
            raw_args.mamba_radix_cache_strategy = "extra_buffer"
            logger.info(
                "K3 detected: forcing --mamba-radix-cache-strategy=extra_buffer"
            )
    except Exception as _e:
        logger.warning(f"could not force mamba strategy for K3: {_e}")

    server_args = ServerArgs.from_cli_args(raw_args)
    simulation_args = SimulationArgs.from_cli_args(raw_args)

    config_path = os.getenv("HISIM_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        logger.info(f"Using config from {config_path}")
    elif simulation_args.config_path:
        os.environ["HISIM_CONFIG_PATH"] = simulation_args.config_path
    else:
        config_path = "/tmp/hisim/config.json"
        logger.info(f"Export config to {config_path}")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(simulation_args.to_dict(), f)
        os.environ["HISIM_CONFIG_PATH"] = config_path

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
