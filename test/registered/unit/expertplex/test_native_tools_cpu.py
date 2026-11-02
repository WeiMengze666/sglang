"""CPU-only test for locating the JIT build tool in a symlinked environment."""
import importlib.util
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / "python/sglang/srt/expertplex/native.py"
SPEC = importlib.util.spec_from_file_location("_expertplex_native_tools", PATH)
native = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = native
SPEC.loader.exec_module(native)


class NativeBuildToolTests(unittest.TestCase):
    def test_finds_ninja_next_to_python_when_path_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory) / "bin"
            bin_dir.mkdir()
            ninja = bin_dir / "ninja"
            ninja.write_text("#!/bin/sh\nexit 0\n")
            ninja.chmod(ninja.stat().st_mode | stat.S_IXUSR)
            previous = os.environ.get("PATH")
            try:
                os.environ["PATH"] = ""
                with patch.object(sys, "executable", str(bin_dir / "python")):
                    native._ensure_ninja_on_path()
                self.assertTrue(os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir))
            finally:
                if previous is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = previous


if __name__ == "__main__":
    unittest.main()
