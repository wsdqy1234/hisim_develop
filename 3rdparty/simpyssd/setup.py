from setuptools import setup

setup(
    name="simpyssd",
    version="0.1.0",
    description="Lightweight Python SSD timing simulator for HiSim L3 cache prefetch simulation",
    packages=["simpyssd"],
    package_data={"simpyssd": ["configs/*.cfg"]},
    include_package_data=True,
    python_requires=">=3.9",
)
