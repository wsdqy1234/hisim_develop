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
        logger.info("Patched kimi_linear_config to force K3 hybrid_ssm detection")
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
    # The detokenizer subprocess may not be running, causing /health to return 503.
    # In simulation mode we don't need the detokenizer — skip the health check.
    if _force_cpu or (not _is_cuda and not _is_npu):
        try:
            from sglang.srt.entrypoints.http_server import app as _fastapi_app
            import fastapi

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
