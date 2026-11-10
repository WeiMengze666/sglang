"""Profiler export must not extend request timing or repeat an automatic stop."""

import asyncio
import contextlib
import io
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

from sglang.benchmark import serving
from sglang.benchmark.datasets import DatasetRow
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _MetricsCaptured(Exception):
    pass


class TestBenchServingProfile(CustomTestCase):
    def _run_profile(self, pd_separated, **profile_options):
        clock = 0.0
        profiler_running = False
        measured = {}
        automatic_stop = bool(profile_options)

        async def profile_endpoint(api_url):
            nonlocal clock, profiler_running
            if api_url.endswith("/start_profile"):
                self.assertFalse(profiler_running)
                profiler_running = True
            else:
                self.assertTrue(profiler_running, "Profiler already auto-stopped")
                profiler_running = False
                clock += 5.0  # Synchronous trace export after requests finish.
            return serving.RequestFuncOutput(success=True)

        async def generate(request_func_input, pbar):
            nonlocal clock, profiler_running
            self.assertTrue(profiler_running)
            clock += 2.0
            if automatic_stop:
                profiler_running = False
            return serving.RequestFuncOutput(success=True)

        def capture_metrics(**kwargs):
            measured.update(kwargs)
            raise _MetricsCaptured

        settings = Namespace(
            dataset_name="custom",
            warmup_requests=0,
            plot_throughput=False,
            profile_steps=None,
            profile_num_steps=None,
            profile_by_stage=False,
        )
        for name, value in profile_options.items():
            setattr(settings, name, value)
        with (
            patch.object(serving, "args", settings, create=True),
            patch.dict(serving.ASYNC_REQUEST_FUNCS, {"vllm": generate}),
            patch.object(serving, "async_request_profile", profile_endpoint),
            patch.object(serving, "time", SimpleNamespace(perf_counter=lambda: clock)),
            patch.object(serving, "calculate_metrics", capture_metrics),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(_MetricsCaptured),
        ):
            asyncio.run(
                serving.benchmark(
                    backend="vllm",
                    api_url="http://worker/generate",
                    base_url="http://worker",
                    model_id="test-model",
                    tokenizer=None,
                    input_requests=[
                        DatasetRow(prompt="hi", prompt_len=1, output_len=2)
                    ],
                    request_rate=float("inf"),
                    max_concurrency=None,
                    disable_tqdm=True,
                    lora_names=[],
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body={},
                    profile=True,
                    pd_separated=pd_separated,
                    warmup_requests=0,
                    profile_decode_url=["http://worker"],
                )
            )
        self.assertFalse(profiler_running)
        self.assertEqual(measured["dur_s"], 2.0)
        self.assertEqual(len(measured["outputs"]), 1)

    def test_manual_export_is_outside_request_duration(self):
        for pd_separated in (False, True):
            with self.subTest(pd_separated=pd_separated):
                self._run_profile(pd_separated)

    def test_automatic_stop_is_not_repeated(self):
        for pd_separated in (False, True):
            for option in ("profile_steps", "profile_num_steps", "profile_by_stage"):
                with self.subTest(pd_separated=pd_separated, option=option):
                    self._run_profile(pd_separated, **{option: 1})


if __name__ == "__main__":
    unittest.main()
