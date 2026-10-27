import hashlib
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/expertplex"))
fingerprint = importlib.import_module("bench_communication").package_fingerprint
validate = importlib.import_module("plot_components").validate_deepep_build


class DependencyFingerprintTests(unittest.TestCase):
    def test_wrapper_and_binary_changes_are_both_tracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "_C.so").write_bytes(b"binary")
            (root / "__init__.py").write_text("api = 1")
            first = fingerprint(root)
            self.assertEqual(first["_C.so"], hashlib.sha256(b"binary").hexdigest())
            validate({"ranks": [[{"deepep_installed_files_sha256": first}]]})
            (root / "__init__.py").write_text("api = 2")
            second = fingerprint(root)
            self.assertNotEqual(first, second)
            with self.assertRaises(ValueError):
                validate(
                    {
                        "ranks": [
                            [{"deepep_installed_files_sha256": f}]
                            for f in (first, second)
                        ]
                    }
                )

    def test_missing_or_malformed_build_is_rejected(self):
        for files in (None, {}, {"api.py": "a" * 64}, {"_C.so": "z" * 64}):
            with self.subTest(files=files), self.assertRaises(ValueError):
                validate({"ranks": [[{"deepep_installed_files_sha256": files}]]})
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            fingerprint(tmp)


if __name__ == "__main__":
    unittest.main()
