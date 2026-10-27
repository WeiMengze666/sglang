import importlib
import io
import sys
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/expertplex"))
services = importlib.import_module("check_independent_services")


class RecorderHTTPTests(unittest.TestCase):
    def test_calibration_preserves_native_lengths_and_rejects_truncation(self):
        tokenizer = SimpleNamespace(encode=lambda text: list(range(len(text))))
        rows = [SimpleNamespace(prompt="abcd", prompt_len=4, output_len=3)]
        self.assertEqual(
            services.routing_requests(rows, tokenizer, 7), [([0, 1, 2, 3], 3)]
        )
        for invalid, cap in (
            ([], 7),
            (rows, 6),
            ([SimpleNamespace(prompt="abcd", prompt_len=3, output_len=3)], 7),
        ):
            with self.assertRaises(ValueError):
                services.routing_requests(invalid, tokenizer, cap)

    def test_native_text_response_and_http_failures(self):
        for action in ("start", "stop", "dump"):
            text = f"{action.title()} recording the expert distribution.\n"
            with patch.object(
                services.urllib.request,
                "urlopen",
                return_value=io.BytesIO(text.encode()),
            ) as call:
                self.assertEqual(
                    services.recorder_request("http://localhost:1/", action), text
                )
                request = call.call_args.args[0]
                self.assertEqual(request.get_method(), "POST")
                self.assertTrue(
                    request.full_url.endswith(f"/{action}_expert_distribution_record")
                )
        with (
            patch.object(
                services.urllib.request,
                "urlopen",
                side_effect=urllib.error.HTTPError("url", 500, "failed", {}, None),
            ),
            self.assertRaises(urllib.error.HTTPError),
        ):
            services.recorder_request("http://localhost:1", "start")


if __name__ == "__main__":
    unittest.main()
