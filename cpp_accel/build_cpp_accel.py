from __future__ import annotations

import sys
from pathlib import Path

from setuptools import Extension, setup

try:
    import pybind11
except Exception as exc:
    raise SystemExit(
        "pybind11 is required for C++ acceleration build. Install with: pip install pybind11"
    ) from exc

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "cpp_accel" / "cpp_accel.cpp"

extra_compile_args = ["-O3", "-std=c++17"]
if sys.platform == "darwin":
    extra_compile_args.append("-stdlib=libc++")

ext_modules = [
    Extension(
        "cpp_accel_impl",
        [str(SRC)],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="cpp_accel_impl",
    version="0.1.0",
    description="Optional C++ acceleration for aix_contest",
    ext_modules=ext_modules,
)
