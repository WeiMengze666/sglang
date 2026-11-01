# ExpertPlex 单机复现记录

本记录对应 B300 单机、Qwen3-Coder-480B-A35B-Instruct BF16、1P+3D+4E。服务由原生 SGLang P/D attention 服务和独立 Expert 服务组成，保留 DeepEP 路径。严格数值测试使用 16 条固定长上下文轨迹、32 个 decode step 和全词表 logprob。

## 严格正确性

四个 per-client overlap policy 均已完成独立 GPU 运行和 CPU 终端审计。每个点都检查 native/repeat、三路 D、KV admission、compiled replay、缓存/PTX 元数据、清理状态和逐 token 全词表 bit-exact 对齐。

| P policy | D policy | job | 全词表 | P TBO proof | D TBO proof | 清理 |
| --- | --- | ---: | --- | --- | --- | --- |
| none | none | 320 | 512 positions × 2560 copies，vocab 151936，bit-exact | skip | skip | passed |
| none | tbo | 313 | 同一 raw digest，bit-exact | skip | 3 D workers passed | passed |
| tbo | tbo | 315 | 同一 raw digest，bit-exact | 43 events / 6 cuts | 3 D workers passed | corrected audit，children empty |
| tbo | none | 322 | 同一 raw digest，bit-exact | 22 events / 6 cuts | skip | corrected audit，children empty |

共同 raw digest 为 `a6cf60e894e61191ccdf4d342c52eac241ac87924b68a3f4575e0fe72da189fc`。315 和 322 的原始 runner 在所有阶段通过后因 cleanup returncode 断言触发 gate failure；独立 CPU 审计确认 owned children 为空并通过其余门禁，因此保留原始失败和修正审计两份证据。

CPU preflight 对每个候选均为 34 tests，且没有初始化 CUDA 或导入 torch。policy、编译准入、launch/configure 和 bootstrap contract 已在提交 `d5144d5a36` 中发布。

## Goodput 屏幕

公平屏幕固定 ShareGPT-24576、256 请求、seed 42/43、TTFT 1000 ms、TPOT 50 ms。已完成的最强原生屏幕为 DeepEP128 colocated，seed42 在 10 rps 达到 4.9972 observed SLO goodput。ExpertPlex NONE 为 seed42 0.9638 rps、seed43 1.0706 rps；TBO 为 seed42 0.4954 rps、seed43 0.4299 rps。该屏幕没有达到 1.5×，也没有把有限到达率结果解释成单 kernel slowdown。

当前 decode coalescing 改动仍需要完整主模型 grouped witness、取消/错误清理、全词表和 on/off serving 对照，组件级证据不足以启用它或宣称性能收益。跨节点 RDMA、完整 optimizer held-out 校准和论文级 APK 端到端收益仍未通过验收。

## 后续候选验证（2026-11-01）

在不改变默认 1P+3D+4E 的前提下，追加了两项独立候选验证：

- job334 将 P=TBO、D=NONE 的 decode SM budget 调到 128、expected tiles 调到 16。四个到达率的 observed SLO goodput 为 0.5040/0.9586/1.3148/0.7065 rps（0.5/1/1.5/2 rps），清理通过；与 job332 的 1.3160 rps（1.5 rps）基本持平，因此没有合入或宣称收益。
- sm100_fused 在同一 Qwen3-Coder BF16 形状上完成了服务启动和 full-vocabulary capture，但严格比较出现明显 logprob 差异及少数 top-1 不一致，故拒绝进入性能屏幕，也不作为默认后端。
- 1P+2D+5E 的探索受当前 DeepEP build 限制：Python 配置表补充后，编译的 legacy/intranode.cu 仍以 Unsupported ranks 拒绝 rank 5。该布局没有严格 reference 或性能结论。

因此当前公开代码保留经过验证的 1P+3D+4E 路径和原生 SGLang/DeepEP 兼容性；相对最强原生屏幕的约 1.5× goodput 目标仍未达成，以上负结果不被包装成性能通过。

- 关闭 SGLANG_EXPERTPLEX_PIPELINED_STORES 的 job369 在 512 positions、2560 copies、151936 vocab 上保持 bit-exact，但没有产生要求的两个 fresh nonempty TBO child replay；因此该消融没有进入 serving 或性能路径。
- 默认 ExpertPlex 使用 4 个 E ranks（每卡 40 个专家），最强 colocated DeepEP128 基线使用 8 个 E ranks（每卡 20 个专家）。这个布局差异解释了 E 卡长期满载的容量边界；它是当前比较的系统级约束，不能从有限 goodput 屏幕推出单 kernel slowdown。

### Logical E8 capacity experiment（未发布候选）

为检验上述 4E/8E 结构差异，候选 worktree 允许 E rank 与 P/D attention 物理重叠，但默认配置和公开源码保持不变。job387 在 P=NONE、D=NONE 下完成了 512 positions × 2560 copies、151936 全词表逐 token bit-exact 对齐，清理通过。

端到端候选 job391 使用自动 KV profiling，在 E/P/D 重叠卡上因 P/D 初始化 FlashInfer 8 GiB workspace 时 OOM；job395 仅降低节点 `max_prefill_tokens` 仍触发自动 KV 池扩张，同样在 benchmark 前 OOM。job397 改为显式 aggregate KV cap 393216，使四服务启动并完成了 0.5 和 1.0 rps 的各 256 请求，但两点 joint-SLO 均为 0/256：0.5 rps 的 p90 TTFT/TPOT 为 1239/312 ms，1.0 rps 为 63264/312 ms。该候选随后通过 gpuq 终止并确认无残留进程。它没有性能收益，未进入公开源码或默认布局。

### 3P+1D 长请求容量边界（CPU-only）

论文长请求的 3P+1D+4E TBO 配置在当前 B300 BF16 Coder 内存门禁下于 GPU 启动前被拒绝（`configure.py` RC=2：expert weights and mailboxes exceed the configured GPU memory budget）。无 TBO 且较低 prefill 容量的几何配置可以通过，因此该结果只标记论文场景的容量边界，不把所有 3P+1D 配置判为不可用。

### LooGLE 长上下文单机对照（有限 p90 证据）

冻结的 `loogle-24576-256.jsonl`（256 条，输入中位数 16736 token）在 TTFT 1 s、TPOT 50 ms、90% joint-SLO 下测试默认 1P+3D+4E。ExpertPlex 在 0.25 rps 完成 256 请求，joint-SLO 209/256（81.64%），observed SLO goodput 0.2116758 rps，p90 TTFT 1834.7 ms、p90 TPOT 23.3 ms，未通过 p90 gate。

同一 workload 的 native DP8/EP8 baseline 在显式 aggregate KV cap 393216 后完成 256 请求，joint-SLO 99/256（38.67%），observed SLO goodput 0.1002781 rps，p90 gate 同样未通过。两者的有限低速率 observed ratio 为 2.111×，但由于双方均未达到 90% joint-SLO，不能作为最终 P90 goodput 验收；该结果只说明长上下文下的容量/TTFT边界。

3P+1D+4E 论文长请求布局的 LooGLE screen 也在最低 0.25 rps 失败：256 请求中仅 137 个同时满足 TTFT/TPOT（53.52%），observed goodput 0.1387596 rps，p90 TTFT 1849.5 ms、p90 TPOT 26.3 ms。该布局随后清理退出，未进入更高负载或公开默认配置。


### Split tile 严格诊断（job428，未发布优化）

为区分 FlashInfer 内部 KV split 与外层 TBO query cut，使用独立 gpuq job428 将 `SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE` 从默认 4096 改为 2048。native/repeat 阶段使用 8 GiB workspace，all-decode 阶段使用 16 GiB workspace；两阶段均完成 512 positions × 151936 vocab 的 native/repeat exact gate。

all-decode 三个 D worker 均完成 512 positions，但在 `history_tokens=12287` 边界出现全词表差异：D1 的首个 `max_abs_logprob_difference` 为 0.656248，随后 12288 为 0.4375、12289 为 0.765629；top-1 仍相同。该配置因此被拒绝，未修改默认 split=4096，也未进入性能屏幕。已验证的 native-fa2-aligned + 默认 4096 路径 job282 仍是长上下文严格 gate 的权威证据。

job428 之后用显式 PYTHONPATH 和 `CUDA_VISIBLE_DEVICES=999` 在发布树运行 `unittest discover`，259 项 ExpertPlex CPU 合约通过；没有 GPU 服务残留。当前缓存前缀、短尾和更广 query-partition gate 仍未全部通过，因此严格总验收与 1.5× goodput 目标保持未完成。

### Split2048 workspace follow-up（jobs 418/421/423/425/427，未发布优化）

job418 的 native 参考在 warmup capture 申请约 7.25 GiB 的 batch_prefill_tmp_v，而候选只提供 4 GiB FlashInfer workspace，因而在 ExpertPlex 阶段前退出；这是 workspace 容量失败，不是 query-partition 数值结论。8 GiB/16 GiB workspace 的后续候选分别在 all-D harness、native graph capture 或同一边界再次失败；job428 是唯一到达三路 D 的 successor，已在上一节记录为 history_tokens=12287 全词表偏差。默认 split4096 和生产路径未改变。

### APK propagation/receipt diagnostics（已完成历史组件证据）

已有 warp-propagation 组件门对 raw 与 routed 两种路径都完成了严格 trace 检查：每条路径均覆盖 148 CTAs、cluster size 2、无截断 trace，并记录了 participating-warp receipt 与 conservative publication-to-receipt 上界。raw 路径为 485248 tiles、8026 phase switches、15 次 arrival，观测上界范围约 25.95–725.66 us；routed 路径为 427312 tiles、5114 phase switches、15 次 arrival，观测上界范围约 50.88–1778.53 us。policy-propagation 诊断进一步覆盖 routed/raw、D=128/1024 的四组实验，每组 25 个 policy versions，所有 decode-arrival recipient coverage 均完整，missing recipient 数为 0。

这些结果使用同一 GPU 时钟和保守上界，明确排除跨设备时钟相减、universal all-warp/one-tile bound 和 serving goodput；它们作为组件证据保留，不提升全局验收状态。

### Logical-E8 duplicate-rank reuse（jobs492/494，严格通过但性能拒绝）

为检验 E 容量边界，隔离候选将 8 个逻辑 E rank 映射到四张物理 E 卡 `[4,4,5,5,6,6,7,7]`，P/D attention 仍在 0–3。job492 在 16 条固定轨迹、32 steps、512 positions × 151936 vocab 下通过 native EP8 对照和全词表严格门，清理通过。随后 job494 使用相同 ShareGPT-24576/256/seed42、0.5/1/1.5/2 rps、TTFT 1 s、TPOT 50 ms、90% joint SLO 进行 serving 屏幕；四个点均为 0/256 joint-SLO，0 observed goodput。最低速率 p90 TTFT 约 590 ms，但 p90 TPOT 约 178 ms，故候选从性能和生产路径拒绝。该结果保留为严格数值和负性能证据，不改变默认 1P+3D+4E。

### APK tile/preemption strict component gate（job489）

为补充论文 §4.3 的 E 侧 tile 调度门，在不改变生产路径的情况下运行了独立 synthetic BF16 SM100 trace。配置为 148 CTAs、cluster size 2、prefill rows 512/2048/8192、decode rows 128，并开启 check-single-pending-compute。严格 gate 记录 215408 条 trace、0 个截断 CTA、107704 个决策、5654 次 phase switch，且每个 CTA 未观察到 ready 的 GEMM 数不超过 1；gpuq cleanup 返回 0 且无残留子进程。

该结果只证明 E 侧 tile/单 pending compute 组件门。它不证明完整 attention publication 到 E admission 的跨设备传播 bound、完整模型逐 token 对齐或 serving goodput，生产代码和全局验收状态保持不变。

### Cached-prefix / query-partition 严格门（job445，FA4 诊断后端）

job442 先确认了配置边界：确定性推理使用 FlashInfer 时 SGLang 自动关闭 radix cache，因而不能用于缓存复用门。job445 保持默认 1P+3D+4E、split=4096、native-fa2 对齐覆盖、逐 token 全词表和三路 D/TBO/清理门禁，仅将 native 与 P/D attention backend 改为 B300 确定性兼容的 FA4；sampling backend 未改变，生产默认 FlashInfer 未修改。

16 条顺序请求共享前 4097 个 token、尾部长度为 512 或 1024。每条请求的 `cached_tokens` 均为 4608；native 与 repeat、三路 D 的 512 positions × 151936 vocab 全词表 logprob 均严格 bit-exact，fresh TBO replay 为 29/29/30，gpuq 清理返回 0 且无残留 children。该结果关闭了“缓存门因配置导致未命中”的严格诊断缺口，但 FA4 只作为正确性诊断后端，没有替换默认 FlashInfer，也没有 goodput 或 1.5× 性能含义。

### Decode workspace reuse 候选（job449/451/454）

为降低每层 ExpertPlex decode 的 allocator bookkeeping，候选仅复用精确行数的 `row_map` 与 FP32 combine buffer，并保留逐次 BF16 输出转换。job449 的初版还复用了 BF16 输出，严格门在第二个 decode step 出现 hidden-state alias；该版本被拒绝。job451 修复 alias 后通过 FA4 严格门：native/repeat/all-D 全词表 exact、512 positions × 151936 vocab、TBO/缓存/清理均通过。

随后 job454 在默认 FlashInfer 上做四次 ABBA paired serving（同一 ShareGPT-24576、256 请求、seed42、TTFT 1 s、TPOT 50 ms）：1/1.5/2 rps 的 baseline observed goodput 分别为 0.9605/1.1221/0.5123 与 0.9641/1.0605/0.4303，候选分别为 0.9593/0.9967/0.3895 与 0.9647/1.1094/0.4645。结果没有稳定收益，候选不合入默认路径，也不改变约 1.5× goodput 未通过的结论。

### sm100_fused + prefill preprocess 隔离诊断（jobs 459–478）

为区分早期 fused mismatch 是否来自 direct routed-row prefill，隔离候选允许 `sm100_fused` 使用 expert-side prefill preprocess，并修正 fused pipeline 的 `GATE -> DOWN` retire 转移。短 prompt job463 和长 prompt job474 的单 ExpertPlex 服务均通过 512 positions × 151936 vocab 全词表 exact；长 prompt 仍没有产生 TBO replay 元数据。job478 在显式跳过 TBO proof 的 all-D 诊断中，首个位置 exact，但从第二个 decode position 开始出现全词表差异，因此该候选没有进入性能屏幕或默认代码。

这些结果只保留为隔离诊断：默认 `sm100`、FlashInfer、1P+3D+4E 和已通过的 P=TBO/D=NONE 路径保持不变。

### Decode activation fixed-tile strict and fair screen (jobs 507/510)

为验证 decode activation tile 由默认几何改为固定 32x256 BF16 tile 的候选，在隔离 worktree 中仅保留该 SM100 pipeline 改动，生产树的默认路径未改变。job507 使用默认 1P+3D+4E、native EP4 reference、16 条固定 prompt、每条 32 个 decode step，并对 512 个位置逐 token 比较完整 151936 词表 logprob。native 与 ExpertPlex 均通过，all_full_vocab_logprobs_exact 为 true，top-1 全部相同，cleanup_errors 为空。该结果是严格数值和组件证据，不是服务性能结果。

严格门通过后，job510 运行同几何的公平服务屏幕：ShareGPT-24576、256 requests、seed42、TTFT 1000 ms、TPOT 50 ms、0.5/1/1.5/2 rps，仅测试 P=TBO、D=NONE。四个点均完成 256 请求；联合 SLO 通过数和 observed SLO goodput 分别为 248/256、0.489005 rps，155/256、0.587118 rps，32/256、0.173245 rps，10/256、0.066743 rps。0.5 rps 的 p90 TTFT/TPOT 为 228.12/43.68 ms，满足 90% joint-SLO；1.0、1.5、2.0 rps 的 p90 TPOT 分别为 57.16、71.76、88.19 ms，均未通过。该候选未达到约 1.5x goodput，未合入生产代码；默认 ExpertPlex 与最强原生基线的性能验收仍保持未通过。

### Long cross-batch strict gate with 8 GiB workspace (job514)

对前一轮长上下文记录进行 provenance 审计后，确认 job262 在 all-D warmup 阶段因 FlashInfer 需要 5435817984 bytes 而 4 GiB workspace 不足退出，没有产生 all-D cross-batch 数值比较；排队中的 job268 也从未执行。随后只提交一次 8 GiB workspace successor job514，保持默认 split4096、deterministic torch compile、三路 D、全词表和 cleanup gates。

job514 的 native/repeat 阶段均通过。三路 D 都完成了 16 条混合 12K/8K/4K 历史的 full-vocabulary capture，但在 history_tokens=12287 处首次出现非零差异；首个报告的 max-absolute logprob difference 为 1.25，所有非零差异中的最大值为 3.3437271118164062。首个差异位置 top-1 仍相同，完整 151936 词表 exact gate 失败；三个 D worker 在同一边界出现该行为，cleanup 无残留进程。该结果确认严格长跨批次/query-partition 门未通过，也没有证明 workspace-only 修复或性能收益。

### FA4 边界正确性候选与公平屏幕（jobs 525/531）

job525 在 FA4 attention backend 下重新运行长混合上下文 strict gate。native/repeat 和三路 D 均通过 512 positions、151936 全词表 exact、compiled replay/batch gate 与 cleanup；默认 FlashInfer 在 12287 边界的差异不再出现。该结果证明 FA4 是一个可验证的边界正确性候选，但尚未证明它适合默认性能路径。

原生 launcher 已支持 fa4，而 run_serving_experiment.py 的实验 CLI 未暴露该选项；提交 8e9cb61231 补齐 fa3/fa4 choices。随后 job531 使用相同 ShareGPT-24576、256 请求、seed42、TTFT 1 s、TPOT 50 ms 和 0.5/1/1.5/2 rps 屏幕。P=TBO、D=NONE 的联合 SLO 通过数为 255/251/234/73，observed SLO goodput 为 0.503800/0.960556/1.296823/0.515997 rps；P=NONE、D=TBO 为 247/154/35/9 和 0.487159/0.585248/0.188880/0.059472 rps。FA4 在 1.5 rps 的 1.296823 rps 仍未达到相对最强原生基线约 1.5x，故不替换默认 FlashInfer，也不改变全局性能验收未通过结论。
