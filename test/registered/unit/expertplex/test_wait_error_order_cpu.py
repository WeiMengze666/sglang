"""A completed failed epoch must not return OK after an earlier error read.

Compile the production wait helper with a deterministic memory-access boundary.
The executor publishes ERROR then DONE between the consumer's first two loads.
GPU protocol tests separately cover CUDA visibility and graph replay.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

CSRC = Path(__file__).resolve().parents[4] / "python/sglang/srt/expertplex/csrc"

BOUNDARY = r"""
#include <cstddef>
#include <cstdio>
#define __device__
#define __forceinline__ inline
using U64 = unsigned long long;
CONTROL_DECLARATIONS
struct DeviceBuffers { U64 *control; };
U64 control[kControlWords]{};
int loads = 0;
U64 ticks = 0;
U64 system_load(U64 *address) {
  const U64 observed = *address;
  if (++loads == 1) {
    // A legal ordered publication after the consumer's first load.
    control[ERROR] = BAD_LAYER;
    control[DONE] = control[PENDING];
  }
  return observed;
}
U64 uncached_load_u64(U64 *address) { return *address; }
U64 now_ns() { return ++ticks; }
void set_error(U64 *words, U64 error) { words[ERROR] = error; }
void __threadfence_system() {}
void __nanosleep(unsigned) {}
WAIT_HELPER
int main() {
  control[PENDING] = 7;
  int status = -1;
  wait_slot_body(DeviceBuffers{control}, 0, 0, 100, &status);
  if (status != BAD_LAYER) {
    std::fprintf(stderr, "Completed failed epoch returned status %d\n", status);
    return 1;
  }
  return 0;
}
"""


class TestWaitErrorOrder(CustomTestCase):
    def test_failure_published_between_control_loads(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required")
        source = (CSRC / "expertplex.cu").read_text()
        declarations = source[
            source.index("constexpr int kControlWords") : source.index("enum Stage")
        ]
        start = source.index("__device__ __forceinline__ void wait_slot_body(")
        end = source.index("__global__ void wait_slot(")
        helper = source[start:end]
        driver = BOUNDARY.replace("CONTROL_DECLARATIONS", declarations).replace(
            "WAIT_HELPER", helper
        )
        with tempfile.TemporaryDirectory(prefix="expertplex-wait-error-") as directory:
            path = Path(directory) / "wait.cc"
            executable = Path(directory) / "wait"
            path.write_text(driver)
            subprocess.run(
                [compiler, "-std=c++17", "-O2", str(path), "-o", str(executable)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            result = subprocess.run(
                [str(executable)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
