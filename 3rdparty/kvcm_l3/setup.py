from setuptools import setup

setup(
    name="kvcm_py_optimizer",
    version="0.1.0",
    description="Python binding of the KVCacheManager optimizer for hisim L3 multi-level cache simulation",
    packages=["kvcm_py_optimizer"],
    package_data={"kvcm_py_optimizer": ["*.so", "*.json"]},
    include_package_data=True,
    python_requires=">=3.9",
)
