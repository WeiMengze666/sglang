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
| CPU adapter, ownership, SLO, config and planner contracts | 42 passed; syntax and CLI checks passed |
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


## Native preprocessing checkpoint, 21:18 UTC

The opt-in `prefill_preprocess` path moves normal-mode EP Scatter1, Scatter2
and Gather into the persistent E kernel. Attention pushes unique token rows
and routing metadata into fixed receive buffers. Gather materializes BF16 per
E rank before the attention-side pull/reduction, matching the native normal
MoE gather boundary. D keeps the direct masked path. No E receiver kernel is
introduced. Descriptor version is 5; each invocation retains independent
immutable operation tile counts in a 48-byte scheduler state.

The first preprocessing test failed with zero outputs after a padding batch:
reusing a stage's tile count during transition let a stale claimant reserve
phantom tasks. Independent stage bounds and generation rechecks fixed this.
The failing report is retained. Coder H6144/I2560 tests then passed with 2E
(top-2 and top-8) and 4E/top-8, each with 35 numeric checks, 20 graph replays,
partial/all padding, changing input/routes, and concurrent P/D. The 4E run
verified exact data-operation task counts for all 31 P invocations per rank;
maximum relative L2 difference from native DeepGEMM was 2.07e-5. This is
operator evidence, not pretrained model accuracy.

The data-operation tiles are startup constants (256 route counters, 32
scatter routes, 4 gather tokens per CTA). Producer-handoff-to-retirement traces
include pipeline waits and exclude A-to-CTA0 propagation. The observed gather
maximum reached 149 us and scatter maximum 78.2 us in this diagnostic; these
are not the paper's reported 25.3 us bound and are not a full reallocation
bound proof. Further kernel tuning and controlled-size sweeps remain required.
The separate 1P3D4E group-versus-peer communication test passed 12 overlapping
and 20 graph comparisons per attention client bit-for-bit.

The grouped-communication/q96/expected-footprint-8 64-request ShareGPT pilot
reached joint attainment 61/64 at 1 rps, 17/64 at 2 rps and 13/64 at 3 rps.
This improves the earlier candidate pilot but does not establish 1.5x.
**Fairness limitation found during audit:** default SGLang dummy initialization
is shard-shape-dependent, while E workers used per-expert seeds. Consequently
these legacy full-size synthetic pilots do not compare identical weights and
cannot establish an acceptance speedup. A new explicit canonical initializer
uses full HF tensor names and shapes before standard TP/EP slicing, with the
same expert generator on E workers; its subsequent validation is recorded below.
The tiny checkpoint comparisons and paired native/APK microbenchmarks already
use identical weights and are unaffected by this limitation.

The cache-flushed serial tiny-model fixed-history comparison retained 24/24
identical top-1 choices; maximum top-128 common logprob difference decreased
from the cache-reuse outlier 0.6283 to 0.03163. This does not resolve the
cache-reuse anomaly or prove strict full-vocabulary accuracy. The one-GPU
attention profile pilot also completed, but its sparse unconstrained fit has
negative decode coefficients and needs a broader grid and held-out validation.

Raw evidence, including failed attempts and compressed traces, is retained in
`results/b300-20261015/checkpoint-2118/`. Runtime IPC handle/lock files are
excluded from evidence archives; the accidentally archived obsolete handles
in checkpoint-2049 were removed.

## Canonical weights and preprocessing checkpoint, 21:35 UTC

The canonical BF16 initializer produced bitwise-identical sampled expert tensors
on GPUs 0 and 4, including fused gate/up views and global expert/layer endpoints.
The tiny synthetic colocated and 1P3D4E deployments both completed 24 requests ×
128 tokens. Under the same 24 cache-flushed fixed histories, 23/24 top-1 choices
matched and the maximum common top-128 logprob difference was 0.016279. The one
top-1 difference had a 0.015625 reference margin. This is functional and numerical
evidence, not strict token alignment or trained-model accuracy. Startup protocol
v2 additionally rejects A/E initializer or seed mismatches before importing IPC.
The identity guard is CPU-tested; these GPU runs preceded that guard-only change.

The E preprocessing path also passed the existing tiny BF16 checkpoint's full
24×128 generation check and 24/24 fixed-history top-1 comparisons, with maximum
common logprob difference 0.031322. Full-480B serving with legacy weights reached
64/64, 62/64, 32/64 and 23/64 joint attainment at 0.5, 1, 1.5 and 2 rps. Because
the older baseline used different synthetic weights, this remains diagnostic.
New full-size comparisons using canonical weights are in progress.

Caching route metadata in warp lanes preserved all 35 four-E/top-8 numerical
checks and exact preprocessing task counts. The maximum observed Gather interval
fell to 94.656 us and Scatter2 to 58.688 us in this run; GateUp reached 55.072 us.
These include pipeline waiting, and trace retirement now includes stage commit.
They are neither isolated instruction timings nor a proof of the paper's complete
25.3 us reallocation bound. Raw reports/traces are in `checkpoint-2135/` below the
same results directory; `checkpoint-2118/` retains the earlier implementation.

## Planner corrections, 21:49 UTC

The search now filters memory using the candidate overlap strategy, including
two mailboxes for TBO, and counts each balanced microbatch's partial tiles
independently. `--overlap auto` compares both sequential and TBO candidates.
`--write-config` exports the selected placement, overlap, q, and expected decode
footprint instead of the original input configuration. CPU checks verify that
a memory-infeasible TBO candidate cannot win and that partial tiles do not merge
across mailboxes. Batch predictions remain separate from runtime capacity caps.
TP1 attention/EP-only placement and IID routing remain explicit limitations.

The 64-request canonical ShareGPT pilot completed: colocated joint attainment
was 64/64, 54/64, 17/64, 21/64 at 0.5, 1, 1.5, 2 rps; ExpertPlex was 64/64,
61/64, 33/64, 22/64. The maximum feasible *tested* rates are 0.5 and 1 rps.
This coarse grid and sample size do not establish a 2x boundary ratio or the
requested 1.5x acceptance. Other baselines, finer grids, and disjoint held-out
requests are still required. Both runs use canonical random weights, not the
pretrained Coder checkpoint. Evidence is retained in `checkpoint-2149/`.

## Canonical four-baseline pilot and validation failures, 22:07 UTC archive

`checkpoint-2207` extends the canonical same-weight ShareGPT pilot with ChunkedPrefill
and PDD. At 1 req/s, joint attainment was colocated 54/64, chunked (2048) 27/64,
PDD 58/64, and ExpertPlex 61/64. PDD and ExpertPlex both pass the same tested
arrival rate; this does not demonstrate a 1.5× gain. The plot retains failed
requests in the denominator and labels synthetic weights and n=64.

The full-480B PDMux attempt failed during decode graph capture: DeepEP LL's
160-expert dispatch uses 80 cooperative CTAs, exceeding the original 32–64 SM
decode Green Contexts. Its startup logs are retained. The revised partition
grid reserves at least 80 decode SMs and remains pending full-model validation.

Strict component-plot validation also detected that MPS partitions generated
different expert-load vectors despite equal CUDA RNG seeds. CUDA RNG execution
geometry depends on visible SM count. The old MPS partition results cannot be
used as matched-input comparisons. Sharing fixtures now generate weights in
fixed CPU per-expert chunks and generate inputs/routing on CPU before copying
them to the selected device. Native, APK and MPS must be rerun with this same
workload identity; old and new initialization schemes must not be combined.

The first pipelined-store experiment failed compilation after moving completion
logic into a helper (`gate` was out of scope). Its complete failure report is
preserved. The corrected build passed the eight-round Coder-shape GPU numerical,
native-reference, concurrency, graph and timeout suite; four-E and all-client
validation is still in progress. No performance improvement is assumed yet.

## Store retirement and dynamic metadata regression, 22:16 UTC

`checkpoint-2216` preserves the first failing multi-client pipeline run: prefill
serial/overlap outputs differed in 7% of elements (maximum absolute difference
1.90625). Decode clients passed. Store=0/raw=1, store=1/raw=0 and store=0/raw=0
isolation runs passed. Dynamic expert row counts and packed offsets now use
explicit L2 loads rather than potentially retaining earlier invocation metadata
in an SM's L1. After this change, store=1/raw=1 passed 30 rounds at both P128
and P257, with 120 exact overlap comparisons plus 20 exact graph comparisons
per attention client; store=0/P257 also passed. The pre-fix sources and failed
report are retained for reproduction. These regressions narrow the fault;
they are not a hardware-counter demonstration of a particular cache event.

Both store policies then passed 35 native-reference cases with two concurrent
raw prefill mailboxes, four E ranks, H6144/I2560, top-8, dynamic/padded routing
and 20 CUDA Graph replays. The changed control-plane identity guards were also
exercised by the node runs. CPU coverage is now 45 tests (including three
checkpoint-readiness cases), all passing.

`SGLANG_EXPERTPLEX_PIPELINED_STORES=1` delays tile completion publication until
its asynchronous global stores retire, retaining a pending tile and explicitly
draining at phase changes or lack of a successor. It remains an experimental
ablation, disabled by default. Matched-input A/B tests found it **slower** than
store=0: isolated GateUp P was roughly 2.35 ms versus 1.96 ms, and Down P roughly
1.61 ms versus 1.19 ms. The remaining serving queue therefore selects store=0;
no speedup is claimed for this change. Task-ring reuse still waits for delayed
retirement and is a candidate pipeline bottleneck requiring another controlled
experiment. Original low-overhead APK performance remains unfulfilled.

## Publication scope

Raw checkpoints 22:07 onward remain on the authorized server and in the user's
local workspace. Automatic approval review rejected publishing an unfiltered
raw artifact bundle because it could contain request/output payloads. The
repository publishes aggregate serving curves and allowlisted numerical test
summaries with source-file hashes, plus reconstruction scripts and dataset
revisions/seeds. `export_evidence.py` omits request text, token histories,
logprobs, IPC handles, environment, host paths and logs; its exclusion contract
is CPU-tested. Raw evidence is not silently discarded, but these later raw
archives are not part of the pushed repository.

The matched CPU-fixture native/APK/MPS runs are complete in checkpoint-2220.
Their GateUp and Down Pareto plots retain unfavorable results: APK is dominated
by Green Context at several points. All six operation intervals were measured
for 128–2048 rows per expert. Scatter2 medians reach about 101 us and Gather
about 223 us. These producer-handoff-to-retirement diagnostics include pipeline
waiting and cannot establish the paper's 25.3 us preemption bound. The original
native tile geometry is retained; shrinking tiles to obtain a lower interval
would not satisfy the paper's performance-preserving requirement.
