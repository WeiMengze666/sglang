# Validation status, 2026-10-15 — work in progress

Base: `WeiMengze666/sglang` at `d8f852234850ac01a36adbb1c337a0acc0ff3912`.
Target: one 8×B300 SM103 node, Qwen3-Coder-480B-A35B-Instruct BF16, 1P3D4E.
Original paper: arXiv:2607.18002v2; see [mechanism-by-mechanism audit](PAPER_CONFORMANCE.md).

## Environment

Remote tests run in the authorized `/home/alayadb/mengze/expertplex-b300` workspace.
8×B300, 148 SM/GPU, 275040 MiB/GPU, NVLink peer atomics on all pairs,
driver 580.178.04, CUDA 13.0.88, PyTorch 2.14.1+cu130, Triton 3.8.0,
sgl-deep-gemm 0.2.2, flashinfer-python 0.7.1, sglang-kernel 0.4.9.
Dependencies were installed in an isolated workspace venv; the original conda env was preserved.

## Observed results

| Scope | Evidence / result |
|---|---|
| CPU adapter, ownership, SLO, config and planner contracts | 36 passed; syntax and CLI checks passed |
| CUDA IPC / native atomics | All 56 ordered pairs passed |
| Device-generated epochs / CUDA graph | 20 replays with changing inputs/routes passed |
| Full 62-layer 480B expert residency, synthetic BF16 | Four experts and four attention clients passed; initial masked version used about 227.65 GiB/E |
| COMM stream / TBO client protocol | Each client passed 12 bitwise serial-vs-overlap and 20 graph output comparisons |
| TMA/TCGEN05 single CTA | Small and Coder H=6144/I=2560 shapes passed numerical/concurrent/graph/timeout tests |
| Contiguous prefill layout | Small-shape numerical/concurrent/graph/timeout tests passed |
| Two-CTA DSMEM / TMA multicast | Small and Coder shapes passed the same regression suite |
| Native BF16 gate/up output rounding correction | Coder shape passed at weight std=0.02; earlier results used a different documented rounding contract |
| Full tiny checkpoint 1P3D4E HTTP + KV | 8 requests × 32 output tokens completed with Mooncake INTRA_NODE_NVLINK |
| Strict greedy token alignment vs Triton | Failed for some prompts; raw outputs and logprob differences retained |
| Ordinary SGLang TRTLLM vs Triton | Also differs on some greedy sequences; this does not excuse additional candidate errors |
| Full tiny TBO + decode graph | 24 requests × 128 tokens completed; decode workers logged actual graph replay |
| Unchunked P input bound | 512 tokens accepted, 513 tokens rejected with HTTP 400 at capacity 512 |
| Native 2-SM MMA / transposed operands / TMA store | Coder shape passed numerical/concurrent/padding/graph/timeout regression |
| Real Coder weights / coding accuracy / paper goodput | Not validated; checkpoint download incomplete |

The first default RDMA KV attempt failed with unregistered GPU memory. The repository's
single-node NVLink configuration fixed transfer. The launcher now selects that path by default.
The first TRTLLM JIT compile hit the 300 s watchdog; it completed compilation and the cached
restart passed generation. These failures are not counted as successful serving measurements.

The std=0.05 large-shape reference-kernel numerical test failed its elementwise threshold.
Later std=0.02 passes do not erase that failure. No tolerance was loosened to label it a pass.
The full-model synthetic checkpoint is only a functional test, not pretrained-model accuracy.

## Performance evidence and current gap

Single-rank synthetic routing, H=6144/I=2560, 40 local experts, top-8 over 160 experts:
2048-token prefill on the early single-CTA APK measured about 5.75 ms expert-side time.
The two-CTA/vector-store version measured about 3.11 ms expert-side time, while native
DeepGEMM GEMM1 + activation + GEMM2 measured about 0.65 ms. These are distinct code revisions,
not a serving throughput gain. The native baseline excludes dispatch/combine. GPU clocks
were approximately 2032 MHz during the later run. Further isolated repetitions are needed.

No 1.5× serving gain has been demonstrated. The remaining native execution overhead is
material, and paper preemption bounds, all baselines/ablations, measured optimizer profiles,
real-model TBO and end-to-end acceptance remain outstanding.

## Reproducibility

Run CPU checks with `python examples/expertplex/check_cpu.py`.
`check_gpu.py --packed` exercises contiguous mailboxes; `check_node.py` validates all-layer
expert deployment and TBO client lifecycle; `check_e2e.py` saves actual generated tokens and
logprobs. `bench_native.py` reports communication and expert timestamps separately;
`bench_deepgemm.py` measures the native compute baseline. New native reports record a source
SHA256. Earlier files are identified by the stage names in this record and are retained as
historical evidence, not silently regenerated.

Joint P90 goodput requires at least 90% of **all offered requests** to meet both TTFT and
TPOT SLOs. Failed requests remain in the denominator. Kernel speed and synthetic routing
cannot establish this metric.

Subsequent M128/native-2SM/TMA-store iterations reduced P expert time to about 2.28 ms,
still materially slower than the native baseline. Removing per-thread parameter copies
reduced CUDA-reported local storage from 1456 bytes to zero, without a corresponding major speedup.
The diagnostic preemption runner releases pre-staged P/D payloads about 10.016 us apart;
traces include producer handoff and epilogue retirement. These are not yet a complete
measurement of system-word-to-cluster preemption propagation.

The larger deterministic tiny fixture (H=2048, I=768, 2 layers, 8 experts) also
completed 24 requests × 128 tokens on native DP4/EP4 DeepEP-auto + DeepGEMM BF16
with TBO. The earlier H=256 fixture is unsupported by DeepEP low-latency mode;
both startup logs are retained. This DP4 run is a correctness reference, not an
equal-8-GPU performance baseline.

Nsight Compute application-range replay with explicit driver profiler markers
reached the profiling range, but the server denied hardware counter access with
`ERR_NVGPUCTRPERM`. No hardware-counter profile is available. Software tile traces
show uneven work distribution and substantial gaps between claims; queue contention
is an investigation hypothesis, not yet a measured causal conclusion.
