# Native Qwen2 shared-expert overlap

The Qwen2 adapter uses SGLang's existing shared-expert MLP, sigmoid gate,
top-k router and shared-expert CUDA stream. Routed experts use the same
ExpertPlex dispatcher as Qwen3. Shared weights stay on the attention workers;
routed weights stay on the expert workers. Shared-expert fusion into the
routed pool is disabled for this placement.

This adapter is for unquantized BF16 `Qwen2MoeForCausalLM` checkpoints with a
single shared MLP and routed experts in every layer. The main reproduction
model, Qwen3-Coder-480B-A35B-Instruct, has no shared expert and does not use SBO.

## Independent service configuration

Use the existing `--enable-expertplex` activation on every service. Set
`--enable-single-batch-overlap` on the expert service and all of its P/D
clients; the bootstrap configuration records `overlap="sbo"`. Each client
retains one mailbox. Configure the pool with
`--expertplex-prefill-count`, `--expertplex-decode-count` and
`--expertplex-expert-count`, then start the corresponding independent native
SGLang attention services, as in the normal ExpertPlex P/D deployment.

For an explicit topology JSON, `configure.py --overlap sbo` writes the same
setting. Attention services must still pass `--enable-single-batch-overlap`;
a disagreement with the pool configuration is an error. Keep prefill CUDA
graphs disabled for this adapter. Native full decode graphs are supported by
the tested path. Shared-expert TBO and prefill graph SBO remain outside the
validated scope.

When ExpertPlex is disabled, Qwen2 keeps its existing native shared-overlap
environment switch and native MoE backend. ExpertPlex SBO does not replace
SGLang's attention implementation, request scheduler or KV allocation policy.

## Validation boundary

The auxiliary two-layer BF16 Qwen2 fixture has passed native/reference-repeat
and ExpertPlex SBO-off/on full-vocabulary comparisons: 16 forced histories,
32 positions per history, vocabulary 1024. A separate native full decode-graph
trace attributed every shared and remote branch by actual CUDA graph clone
and node IDs. Both layers showed overlap with SBO enabled and none with it
disabled. These measurements used one active decode service; they do not
establish main-model performance, multi-client shared-expert concurrency or
optimizer calibration.

The publication-only integration passed a fresh GPU regression on the
published SGLang base (experiment 213): 98 CPU checks and four exact
512-position full-vocabulary captures. Do not interpret these results as completed
paper reproduction or a 1.5x goodput result. The placement optimizer rejects
shared-expert models until their separate shared-branch and overlap profiles
have passed calibration; it must not omit shared compute from its estimates.
