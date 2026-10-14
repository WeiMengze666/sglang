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
| CPU adapter, ownership, SLO, config and planner contracts | 37 passed; syntax and CLI checks passed |
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

An eight-shard generation-tagged task queue subsequently reduced the same 2048-token
prefill expert time from about 2.26 ms (read-only stop polling) through 1.96 ms
(bounded single-head CAS retries) to 0.99 ms. Decode at 128 tokens measured about
0.96 ms. Dispatch/combine remain additional costs; these are not serving gains.
The shard version passed eight rounds of numerical, concurrent, padding, graph and
timeout checks. Direct native DeepGEMM BF16 arithmetic comparisons had relative L2
errors at most 3.66e-5 and at least 99.9918% exactly equal output elements in these
cases, using the unchanged existing tolerances and an explicit FP32 weighted combine.
This isolates expert arithmetic; it does not reproduce DeepEP's distributed reduction
order. The H=2048 full-model comparison against native DP4/EP4 completed but did not
pass strict greedy-token equality. That end-to-end discrepancy remains open.

## Independent native operation templates, 19:15–19:20 UTC checkpoint

The default SM100 executor now schedules GateUp GEMM, activation, and Down GEMM
as separate operations inside the persistent kernel. GateUp/Down use native
DeepGEMM's transposed STSM/TMA epilogue and nine loading stages for two-CTA
clusters. The earlier fused version remains the explicitly named `sm100_fused`
ablation. Activation tiles cover 32 tokens × 128 columns per CTA; an earlier
128-token activation tile exceeded 200 us and is retained as a failed latency
design in `preemption-nativeops-01.json`.

Attention-side metadata is now prepared locally before publishing it to E;
payload stores and output pulls use 16-byte vectors. The additional BF16 GateUp
buffer raises the default mailbox estimate to 2,085,619,332 bytes per E
(about 1.943 GiB, excluding descriptors/runtime allocations).

Eight-round Coder-shape native-reference regression passed after the native
epilogue change (`native-epilogue-02.json`). P2048 expert time remains about
1.16 ms and total dispatch/compute/pull about 1.40 ms; this still exceeds the
native compute baseline (~0.65 ms). D128 expert time is about 0.99 ms.
These are single-E microbenchmarks, not equal-resource serving comparisons.

The complete 62-layer 480B-geometry HTTP deployment then passed 8×32 and
24×128 output-token checks with concurrency 4 and 12 respectively, using
1P3D4E, TBO, decode graphs and NVLink KV transfer. E allocations were
227815 MiB/GPU; attention GPUs used about 33060 MiB each. Weights are explicitly
synthetic small-uniform dummy values and logits are nearly uniform: these runs
test capacity, protocol and finite generation, not pretrained accuracy or
realistic routing performance. Raw token/logprob reports are compressed under
`results/b300-20261015/full480-tbo-e2e-*.json.gz`.

Software tile traces with the 32-token activation tile measured maxima of
52.896 us (P GateUp), 25.632 us (P Down), and 21.888 us (activation) in one run.
The attention-clock publication-to-admission-ack roundtrip upper bound for D
was 28–63 us. It includes ack visibility/polling, is not a cross-GPU clock
subtraction, and does not alone prove the paper's complete preemption bound.
Concurrent P8192/D128 still showed about 4× D slowdown with the tested q=32;
low-interference acceptance remains open.

Nsight Systems confirmed native DeepGEMM's two-CTA, 128×128×64, nine-stage
geometry. Hardware counters remain unavailable. Green Context allocations
were tested directly: requested 112/32, 96/48, and 80/64 returned those counts
(144 total); requesting 116/32 instead returned 120/28. Baselines must use
actual returned counts, not assume the requested partition was granted.

## Baseline and sharing checkpoint, 19:59 UTC

All four eight-GPU baseline layouts completed 24 requests × 128 generated
tokens with concurrency 12 on the same deterministic tiny BF16 checkpoint:
DP8/EP8 colocated, chunked prefill, PDD (two DP4/EP4 instances), and TP8/EP8
PDMux. These are functional tests, not full-480B performance measurements.
PDD needed disjoint derived control-port blocks. PDMux needed independent
P/D DeepEP buffers, CPU-group descriptor bootstrap, correct MoE extend
metadata, and split-prefill padding/postprocessing only at the appropriate
layer boundaries. PDMux uses measured Green Context counts and passes those
counts to DeepGEMM. Failed startup/generation attempts are retained.

DeepEP initialization required explicitly selecting active InfiniBand HCAs;
the tested list and NVSHMEM environment are in the baseline summaries. Setting
the remote transport to `none` alone did not fix initialization. No shared
DeepEP installation or machine-wide GPU setting was modified.

The sharing harness now measures **individual GateUp/Down grouped GEMMs**,
as required by §7.3. A control GPU publishes P8192/D128 ready signals 10 us
apart, and CUDA stream memory waits avoid E-side polling CTAs. Exclusive,
priority-stream and actual Green Context partitions have completed both
operations. A separate-process, task-private MPS daemon completed GateUp at
100:100, 75:25, 50:50 and 25:75 active-thread percentages; client PIDs were
verified through the daemon. Completion-ack times are control-clock upper
bounds. The earlier `sharing-native-01` measures whole MoE and is supplementary,
not the paper's grouped-GEMM experiment. APK grouped-GEMM-only comparison and
full Pareto/latency-protection acceptance remain pending.

Tile-prefix lookup and 32 queue shards passed the eight-round Coder numerical,
concurrency, graph and timeout suite. However, 32 shards measured P2048 expert
time ~1.17 ms and D128 ~1.00 ms, slower than the preceding eight-shard prefix
version (~1.08/~0.92 ms). This experiment is retained, not called a speedup.
Nsight Systems also confirmed native D128 uses M=128, while the APK currently
uses M=32 for D: that template mismatch is being corrected.

ShareGPT and official LooGLE longdep_qa were downloaded at pinned revisions
with SHA256 records. With the common PDD per-rank KV cap of 24576 tokens,
only 339 LooGLE records have at least two input/output tokens and fit the cap.
The attempted 1000-record preparation explicitly failed; the subsequent
workload contains 256 distinct eligible records. Context/question formatting
is documented in `prepare_workloads.py`, because the paper does not publish
the exact preprocessing. This is not a claim of identical original workloads.

Raw reports, failed baseline logs and provenance are under
`results/b300-20261015/checkpoint-1959/`. No real-weight accuracy or 1.5×
end-to-end goodput claim follows from this checkpoint.

## 2026-10-15 checkpoint: full serving pilot and exact operator fixtures

M128 now matches both native GateUp/Down template heights; the eight-round
native-reference, padding, concurrency, graph and timeout regression passed.
The §7.3 APK fixture uses identical native DeepGEMM weights and inputs.
GateUp and Down each matched the native output bit-for-bit in every tested
P/D budget (8, 16, 32, 48, 64, 96). Nevertheless, isolated GateUp P8192/D128
was about 1.85 ms/0.54 ms, versus native ~1.03 ms/0.42 ms. Down was about
1.06 ms/0.34 ms versus ~0.55 ms/0.23 ms. Low scheduling overhead is NOT met.
Native L2 traversal matching passed numerical tests but showed no clear gain.
MPS Down also completed the four real daemon-controlled partition cases.

The full 62-layer Coder BF16 synthetic serving pilot used 64 identical
ShareGPT requests (seed 42), 24576 common sequence cap, aggregate decode KV
98304, Triton attention, TBO, and fixed 1s/50ms TTFT/TPOT SLOs:

| Offered req/s | Colocated joint attainment | ExpertPlex joint attainment |
| --- | --- | --- |
| 0.5 | 64/64 | 64/64 |
| 1 | 61/64 | 45/64 |
| 2 | 28/64 | 11/64 |

ExpertPlex currently loses this pilot. The maximum *tested* feasible arrival
rates are 1 and 0.5 respectively; this coarse grid does not locate either
boundary exactly. The 1 req/s P90 TPOT was 47.72 ms versus 56.66 ms; P90 TTFT
was 246.38 ms versus 355.30 ms. These are diagnostic synthetic-weight pilots,
not trained-model performance or a claimed 1.5x gain. A DeepEP build overlapped
part of the baseline pilot; final measurements must use an isolated machine.

DeepEP v1.2.1 was independently built from official commit
`9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`; the existing DeepEP 2.1 package was
not replaced. Both normal and low-latency eight-GPU identity fixtures passed.
Eager timings contain host submission gaps and rank skew. Subsequent graph
timings include uncached dynamic routing, with benchmark-only NCCL all-reduces
outside the measured intervals; echo preparation is excluded. This remains an
8-GPU NVLink adaptation, not the original 16-GPU hierarchical RDMA experiment.
The first graph attempt used a host-blocking NCCL barrier and failed capture;
its log is retained, and the working version uses preallocated all-reduces.

The fixed-history tiny-model comparison scored the P/D-supported top-128
logprobs: 24/24 top-1 choices matched, but a length-258 outlier reached 0.6283
absolute logprob difference. This is unresolved numerical evidence, not an
accuracy pass. Full-vocabulary `token_ids_logprob` for the first P/D token is
not carried by the pinned SGLang transfer metadata. A separate serial,
cache-flushed comparison is being run. Reports distinguish top-k mass coverage,
TV lower bounds and full-vocabulary KL; missing probabilities are never zeroed.

Completed reports and failed attempts are retained under
`results/b300-20261015/checkpoint-2049/`.
