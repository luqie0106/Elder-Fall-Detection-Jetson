from __future__ import annotations

import importlib
from types import ModuleType


def load_cpp_accel() -> tuple[ModuleType | None, str]:
    """
    Try importing the optional C++ acceleration module.
    Returns (module_or_none, message).
    """
    try:
        module = importlib.import_module("cpp_accel_impl")
        return module, "ok"
    except Exception as exc:
        return None, str(exc)
