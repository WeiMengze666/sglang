"""Load dependency-free experiment tools without importing SGLang's API."""

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "_expertplex_tools"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / "python/sglang/srt/expertplex")]
    sys.modules[PACKAGE] = package


def module(name):
    return importlib.import_module(f"{PACKAGE}.{name}")
