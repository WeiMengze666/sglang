"""Compile the actual APK budget arithmetic without CUDA and check Eq. 4."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

CSRC = Path(__file__).resolve().parents[3] / "python/sglang/srt/expertplex/csrc"

DRIVER = r"""
#include "sm_budget.h"
#include <cassert>
#include <climits>
#include <cstdint>
#include <limits>
#include <string>

using expertplex::adaptive_decode_budget;

template <int Cluster> void compare_equation() {
  // These exact binary denominators keep the small integer products exactly
  // representable in FP32. The oracle uses integer arithmetic, independent
  // of the production float, ceil and saturation implementation.
  for (int q = 1; q <= 148; ++q)
    for (uint64_t x = 0; x <= 256; ++x)
      for (int expected : {1, 2, 8, 32, 128}) {
        const uint64_t denominator = expected * Cluster;
        const uint64_t rounded =
            (q * x + denominator - 1) / denominator * Cluster;
        for (int cap : {Cluster, 146, 148}) {
          const int want = rounded >= uint64_t(cap) ? cap : int(rounded);
          assert(adaptive_decode_budget<Cluster>(q, x, float(expected), cap)
                 == want);
        }
      }
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const std::string mode(argv[1]);
  if (mode == "equation") {
    compare_equation<1>();
    compare_equation<2>();
    compare_equation<4>();
  } else if (mode == "saturation") {
    // All inputs satisfy the executor's positive/finite host validation.
    const float smallest = std::numeric_limits<float>::min();
    const uint64_t largest = std::numeric_limits<uint64_t>::max();
    assert(adaptive_decode_budget<2>(INT_MAX, 2, 1.f, 146) == 146);
    assert(adaptive_decode_budget<2>(96, 1, smallest, 146) == 146);
    assert(adaptive_decode_budget<2>(INT_MAX, largest, smallest, 146) == 146);
    assert(adaptive_decode_budget<2>(INT_MAX, 1, 1.f, INT_MAX) == INT_MAX);
    assert(adaptive_decode_budget<1>(INT_MAX, largest, smallest, 147) == 147);
    assert(adaptive_decode_budget<2>(INT_MAX, 0, smallest, 146) == 0);
  } else if (mode == "rounding") {
    assert(adaptive_decode_budget<2>(3, 1, 1.f, 146) == 4);
    assert(adaptive_decode_budget<2>(3, 1, 1.5f, 146) == 2);
    assert(adaptive_decode_budget<2>(3, 1, 0.5f, 146) == 6);
    assert(adaptive_decode_budget<2>(145, 1, 1.f, 145) == 145);
    assert(adaptive_decode_budget<2>(1, 1,
           std::numeric_limits<float>::max(), 146) == 2);
    assert(adaptive_decode_budget<2>(96, 8, 8.f, 146) == 96);
    assert(adaptive_decode_budget<2>(96, 16, 8.f, 146) == 146);
  } else {
    return 2;
  }
}
"""


class TestAdaptiveSMBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required")
        cls.directory = tempfile.TemporaryDirectory(prefix="expertplex-sm-budget-")
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        source = root / "budget.cc"
        source.write_text(DRIVER)
        cls.executable = root / "budget"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fsanitize=undefined,float-cast-overflow",
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

    def run_case(self, mode):
        subprocess.run(
            [str(self.executable), mode], check=True, capture_output=True, text=True
        )

    def test_matches_integer_equation_for_normal_budgets(self):
        self.run_case("equation")

    def test_saturates_before_float_to_integer_conversion(self):
        self.run_case("saturation")

    def test_cluster_rounding_cap_and_zero(self):
        self.run_case("rounding")


if __name__ == "__main__":
    unittest.main()
