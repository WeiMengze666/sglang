# Validation status, 2026-10-13

Base repository: `WeiMengze666/sglang`, `main` at
`d8f852234850ac01a36adbb1c337a0acc0ff3912`.

Development host: macOS arm64, Python 3.12. No PyTorch, NVIDIA GPU, CUDA Toolkit,
model checkpoint, or running SGLang service was available.

## Completed locally

| Check | Result |
| --- | --- |
| Adapter / selective loader / unsupported-mode CPU contracts | 10 passed |
| Rendezvous generation, checkpoint identity, slot ownership, joint SLO contracts | 9 passed |
| Model/configuration, memory, tile policy, profile fit/search CPU contracts | 13 passed |
| Python syntax and all experiment CLI help entry points | Passed |
| Coder-480B metadata configuration and launcher dry run | Passed |
| Benchmark CLI flags checked against the pinned serving parser | Passed |
| Ruff 0.15.1 `F401,F821,F823,UP037` | Passed |
| Ruff format, isort 7.0.0, clang-format 20.1.7 | Passed |
| `git diff --check` | Passed |

Run the dependency-free checks with:

```bash
python examples/expertplex/check_cpu.py
```

The adapter tests execute extracted production methods with test doubles. They
verify that attention ranks do not construct/read routed expert parameters and
that control flow selects the remote path. They are not an import/initialization
test of a fully installed SGLang environment.

The optimizer's artificial unit-test coefficients verify algebra and filtering
only. No artificial profile is supplied as a B300 performance measurement.

The Coder-480B 1P3D4E capacity calculation is 217.96875 GiB routed expert weights
plus 9.06250096 GiB native buffers per expert GPU. With a 16 GiB planning reserve,
the estimate is 243.03125096 GiB per expert GPU. This is a static byte calculation,
not observed HBM usage, and does not prove the deployment starts.

## Pending B300 acceptance

1. Compile the SM103 extension with the repository's CUDA/PyTorch toolchain.
2. Run `check_gpu.py`, including repeated epochs, concurrent P/D, numerical
   comparison, timeout, and all ordered eight-GPU IPC/atomic pairs.
3. Run Qwen3-30B-A3B and then Coder-480B BF16 end-to-end. Validate actual HBM,
   Mooncake KV handoff, long prompts, cancellation, worker failure and cleanup.
4. Compare intermediate/logit/greedy-token results against ordinary BF16 SGLang
   and evaluate coding-task accuracy. Native fused rounding can differ.
5. Measure tile preemption and idle-resource reallocation with a GPU timeline.
   The code's scheduling boundary does not establish a measured latency bound.
6. Collect real attention/APK/routing/NVLink profiles, tune policy/layout, then
   run joint TTFT/TPOT goodput sweeps against the baselines with equal resources.

CUDA compilation, GPU primitives, full-model functional correctness, token
alignment, task accuracy, preemption latency and goodput are all **NOT RUN**.
The current kernel is a correctness-oriented WMMA implementation. CUDA Graph,
TBO/SBO and the paper's optimized TMA/cluster pipeline remain unimplemented.
