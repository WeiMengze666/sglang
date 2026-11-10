"""Check bounded diagnostic records with repeated epochs and full buffers."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

CSRC = Path(__file__).resolve().parents[4] / "python/sglang/srt/expertplex/csrc"

DRIVER = r"""
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#define __device__
#define __forceinline__ inline
using U64 = uint64_t;
struct OperationTraceRecord;
struct SlotState { U64 gate_tiles, activation_tiles, down_tiles; };
struct Context {
  OperationTraceRecord *operation_trace = nullptr;
  int operation_trace_capacity = 0;
  U64 *operation_trace_counts = nullptr;
};
#include "operation_trace.cuh"

int main(int argc, char **argv) {
  assert(argc == 2);
  if (std::strcmp(argv[1], "disabled") == 0) {
    Context disabled{};
    operation_begin(disabled, 0, 1, 0, 0, 10);
    operation_finish(disabled, 0, 1, 20, nullptr);
    return 0;
  }
  assert(std::strcmp(argv[1], "bounded") == 0);
  OperationTraceRecord records[4]{};
  U64 counts[2]{};
  Context context{records, 2, counts};
  SlotState tiles{3, 4, 5};
  operation_finish(context, 0, 1, 20, nullptr);
  assert(counts[0] == 0 && records[0].finish_ns == 0);
  operation_begin(context, 0, 7, 2, 0, 100);
  operation_finish(context, 0, 7, 110, &tiles);
  operation_begin(context, 1, 1, 3, 1, 200);
  operation_finish(context, 1, 1, 220, nullptr);
  // Reusing a slot epoch must preserve both admissions for causal analysis.
  operation_begin(context, 0, 7, 4, 1, 300);
  operation_finish(context, 0, 7, 330, nullptr);
  assert(counts[0] == 2 && counts[1] == 1);
  assert(records[0].epoch == 7 && records[0].layer == 2);
  assert(records[0].start_ns == 100 && records[0].finish_ns == 110);
  assert(records[0].gate_tiles == 3 && records[0].activation_tiles == 4);
  assert(records[0].down_tiles == 5);
  assert(records[1].epoch == 7 && records[1].layer == 4);
  assert(records[1].finish_ns == 330);
  assert(records[2].epoch == 1 && records[2].phase == 1);
  assert(records[2].finish_ns == 220);
  const auto first = records[0];
  const auto second = records[1];
  for (U64 epoch = 8; epoch < 10000; ++epoch) {
    operation_begin(context, 0, epoch, 5, 0, 400);
    operation_finish(context, 0, epoch, 450, &tiles);
  }
  // Overflow counts remain visible, while the retained prefix and other slot
  // remain intact. A circular-buffer rewrite would silently lose this prefix.
  assert(counts[0] == 9994 && counts[1] == 1);
  assert(std::memcmp(&first, &records[0], sizeof(first)) == 0);
  assert(std::memcmp(&second, &records[1], sizeof(second)) == 0);
  assert(records[2].finish_ns == 220 && records[3].finish_ns == 0);
}
"""


class TestOperationTrace(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required")
        directory = tempfile.TemporaryDirectory(prefix="expertplex-operation-trace-")
        cls.addClassCleanup(directory.cleanup)
        source = Path(directory.name) / "trace.cc"
        cls.executable = Path(directory.name) / "trace"
        source.write_text(DRIVER)
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O1",
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all",
                "-I",
                str(CSRC),
                str(source),
                "-o",
                str(cls.executable),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_disabled_trace_needs_no_storage(self):
        subprocess.run([str(self.executable), "disabled"], check=True, timeout=10)

    def test_repeated_epochs_and_overflow_preserve_prefix(self):
        subprocess.run([str(self.executable), "bounded"], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
