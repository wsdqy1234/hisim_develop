"""kvcm_py_optimizer: Python binding for the KVCacheManager optimizer (L3 simulation backend).

This package is built independently (see ../README.md) and provides the
``kvcm_py_optimizer`` extension module plus a bundled optimizer startup config.
hisim imports it optionally to enable L3 multi-level cache simulation.
"""
from ._native import *  # noqa: F401,F403

from pathlib import Path

__version__ = "0.1.0"


def get_config_path() -> str:
    """Return the absolute path of the bundled optimizer startup config JSON."""
    return str(Path(__file__).resolve().parent / "optimizer_startup_config_load.json")
