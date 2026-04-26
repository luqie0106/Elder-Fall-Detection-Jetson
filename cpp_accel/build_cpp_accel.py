from __future__ import annotations

import os
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


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


extra_compile_args = ["-O3", "-std=c++17"]
extra_link_args: list[str] = []
define_macros: list[tuple[str, str]] = []

if sys.platform == "darwin":
    extra_compile_args.append("-stdlib=libc++")

enable_openmp = _env_flag("FALL_CPP_ACCEL_OMP", sys.platform.startswith("linux"))
if enable_openmp:
    if sys.platform.startswith("linux"):
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")
        define_macros.append(("CPP_ACCEL_USE_OPENMP", "1"))
    elif sys.platform == "win32":
        extra_compile_args.append("/openmp")
        define_macros.append(("CPP_ACCEL_USE_OPENMP", "1"))
    elif sys.platform == "darwin":
        # macOS 需要单独安装 libomp；默认不启用，除非显式设置 FALL_CPP_ACCEL_OMP=1。
        extra_compile_args.extend(["-Xpreprocessor", "-fopenmp"])
        extra_link_args.append("-lomp")
        define_macros.append(("CPP_ACCEL_USE_OPENMP", "1"))

print(
    f"[cpp-build] OpenMP {'enabled' if any(k == 'CPP_ACCEL_USE_OPENMP' for k, _ in define_macros) else 'disabled'} "
    f"(FALL_CPP_ACCEL_OMP={os.getenv('FALL_CPP_ACCEL_OMP', '') or 'auto'})"
)

ext_modules = [
    Extension(
        "cpp_accel_impl",
        [str(SRC)],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )
]

setup(
    name="cpp_accel_impl",
    version="0.1.0",
    description="Optional C++ acceleration for aix_contest",
    ext_modules=ext_modules,
)
