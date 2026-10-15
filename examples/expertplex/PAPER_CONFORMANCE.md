# ExpertPlex 原论文对照与验收记录

依据：用户工作空间原始 PDF，ExpertPlex, arXiv:2607.18002v2，2026-07-21，§1–9。
原文：https://arxiv.org/html/2607.18002v2 。本地原始 PDF 已逐节核对。
PDF SHA256：`8321f7095ccdde149178e16fa3cdbfa60a7a87fd42920483acb21e0cac2a363a`。
本文件记录复现要求及当前差距，不表示全部要求已经实现。

## 可追踪的实现路径

以下路径相对仓库根目录，按请求生命周期对应原文；函数名比随提交变化的行号更稳定。

1. **权重分离（§3）**：`python/sglang/srt/expertplex/integration.py::attention_weights_iterator`
   过滤 routed expert 权重；`worker.py::load_expert_weights` 按全局 expert ID 对 E rank
   取模分片，保留所有层。`worker.py::expert_main` 先校验 checkpoint 指纹与容量，创建
   `OwnedBuffers`、`PersistentExecutor`，再发布 IPC handles。随机权重比较另外使用
   `synthetic_weights.py` 的全局张量种子，不能把各分片独立初始化当成同一个模型。
2. **attention 发起通信（§5.2）**：`models/qwen3_moe.py` 的 ExpertPlex 分支调用
   `expertplex/client.py::begin/finish`。`begin` 在 COMM stream 等待 COMPUTE 的路由结果，
   调用 `csrc/group_communication.cuh::RemoteGroup` 的 dispatch/combine 接口。
   raw P 路径由 `group_dispatch_raw` 写入最终 raw buffers，`group_publish` 在 payload
   fence 后发布 READY。D 使用 routed rows 路径。WaitDone 在 attention GPU 上等待，
   `finish` 向 COMPUTE stream 加入完成 event；E 不启动配对的接收通信 kernel。
3. **长驻执行与传播（§4.2–4.3）**：`csrc/sm100_pipeline.cuh::executor` 的 CTA0 调用
   `update_policy`，经 system-scope READY、device-scope policy、cluster DSMEM Task、
   CTA 内 `task_full` mbarrier 传播。`claim` 是 cluster leader 的唯一 tile 领取入口；
   `initialize` 和 `commit` 推进 Scatter1 → Scatter2 → GateUp → ACT → Down → Gather。
   Task ring 的 metadata-consumed barrier 与 output-retired barrier 分离。phase 切换
   在流水模式下通过每 CTA 单调 64-bit 退役前缀等待旧输出，避免环形 barrier parity 的 ABA；
   这条实现路径和数值通过本身不能证明原文的“至多当前 tile”上界，
   尤其需要检查预取深度、传播延迟和所有 warp/CTA 的收敛。
4. **SM 自适应（Eq.4）**：`update_policy` 从实际 decode GateUp row tiles 计算 footprint，
   按期望 footprint 缩放 q，向 cluster 倍数取整并限制 Qmax。`claim` 在首选 phase 无可领取
   工作时尝试另一 phase，实现空闲资源借用。CPU 对照是
   `optimizer.py::adaptive_sm_allocation`；CPU 公式对照不替代 GPU 调度轨迹验收。
5. **离线选择（Eq.1–3、Algorithm 1）**：`optimizer.py::search` 枚举单节点布局，调用
   `estimate_memory` 过滤容量，再为已有 profile 的互补 SM 预算二分可行 P/D batch。
   `expected_rank_footprint` 保留各 client/mailbox 的独立 tile padding，
   `_iteration_latency` 计算 sequential 或 TBO critical path。它当前要求 attention TP=1、
   E-only EP，并没有原文的完整 parallelism、跨节点 placement、SBO 搜索空间。

原论文的 SM90 FP8 tile 参数不能直接作为 SM103 BF16 的实现参数。当前 B300 后端使用
所固定 DeepGEMM BF16 原生几何的双 CTA/2SM MMA、每 CTA M128/N128/K64、9 个 TMA stage。
这是明确的硬件与精度适配；不能据此宣称复现了原论文 2.2–25.3 μs 数值区间。

## 固定实验条件

- 用户指定适配：单机 8×B300、Qwen3-Coder-480B-A35B-Instruct、BF16、默认 1P+3D+4E。
- 论文原实验是 H800、MiniMax-M2.7 / GLM-5.1-FP8；不能把模型、精度及硬件改变后的结果冒充原论文数值复现。
- 真实权重未完整时，随机权重只能验证计算路径、容量及合成负载；不能验收预训练模型精度或真实路由分布下的性能。
- 原生 WMMA 后端是通信与功能参考实现，**尚不满足论文 APK 的原生流水线要求**。

## 机制与代码审计

| 原文 | 必须保留的机制 | 当前证据或差距 | 验收方式 |
|---|---|---|---|
| §3 | P、D attention 独占 GPU；E 共享完整专家权重 | 62 层 480B 随机权重完成完整服务及 Poisson sweep；最新小 checkpoint 的 24 个固定历史位置 top-1 全部对齐，但 logprob 非 bitwise、非完整词表 | 同权重基线逐 token / logprob 对照，真实 KV 移交 |
| §4.2、§7.1 | 一个长驻 kernel 融合两个 grouped GEMM、激活、数据处理、cluster scheduler | 默认 SM100 长驻融合 TMA/TCGEN05；当前性能明显落后于原生 DeepGEMM，不能验收调度开销 | 对照原生 BF16 GEMM，并记录融合前后开销 |
| §7.1、§7.6 Fig.15 | MoE 预处理/后处理也是 APK 中的 tile operation，包含 EP Scatter 两阶段与 EP Gather | `prefill_preprocess` 的 4E/top-8 数值、并发、graph 和完整随机权重服务通过；六种操作已有 tile interval 数据，但不构成完整传播/抢占上界证明 | 原生 BF16 per-rank gather 舍入边界对照；完整负载的各操作抢占间隔测试 |
| §4.2 | 启动前固定 operation templates、buffer、TMA descriptors；shape 在启动时编译为常量 | buffers / TMA descriptors 已固定；Coder/30B 的 H/I 已做编译期特化，小调试模型保留通用形状 | 多层、动态 batch、图 replay 的地址与描述符检查 |
| §4.2 | 保留原生 CTA/cluster shape、TMA multicast、warp specialization | 默认双 CTA、2SM MMA、9-stage TMA、原生 STSM/TMA epilogue 已通过回归；GateUp/ACT/Down 独立调度。Nsight Systems 确认对应 native GEMM 几何；性能仍落后 | 原生算子形状及流水线代码审计，数值测试 |
| §4.3 Fig.4 | system P → CTA0 → device p_i → cluster DSMEM → warp broadcast → mbarrier | 默认后端已实现该传播链；GPU 定时发布和 tile trace 已采集，尚未证明完整上界 | 可控时刻注入 urgent decode，记录各级 epoch/timestamp |
| §4.3 | 决策在 check epoch 内固定；观察切换后不得再领取旧 phase tile | leader 是唯一领取者并向所有成员分发同一 Task；CAS 竞争不会被当成 phase 空闲；trace validator 检查成员决策与阶段退役，但完整传播上界仍待验证 | 有预取、多 warp、多 CTA 的长时间压力测试 |
| §4.3 | 不在每个 tile 后 drain pipeline；无需 checkpoint、restore 或重算 | SM100 双 accumulator，延迟 store 退休与 Task ring 元数据复用已分离；跨 mailbox 回归通过，配对汇总仍慢约 0.73%。后期论文实验显式开启，库默认仍关闭 | pipeline 轨迹、吞吐及 phase 切换边界检查 |
| §5.2 | attention push 到最终 dispatch buffer；payload 可见后发布 ready | NVLink IPC 路径已验证全部 56 个有向 GPU 对 | 多发送方、重复 epoch、容量边界、故障路径 |
| §5.2 | E 无通信接收 kernel、无预留 polling SM；attention pull 最终结果 | 默认服务路径已实现，E 侧预处理直接读取最终 raw buffer；通信微基准的 identity echo 仅为计时外诊断 | kernel trace、并发 P/D 数值验证 |
| §5.2 Fig.5 | 单线程 WaitDone 位于 COMM stream；COMPUTE stream 可继续 TBO/SBO | 独立 COMM stream 和双 mailbox 已实现；client 的 TBO/graph 对照通过，小 checkpoint 完整 SGLang TBO/图回放通过，真实模型待验收 | CUDA timeline 及双 microbatch 生命周期检查 |
| §5.3 | 跨节点 P 同 local rank 转发，节点内 NVLink multicast/gather；无 P 节点走低优先级 IB VL | 单节点 MoE 路径为 NVLink；多节点 MoE RDMA/层次转发未实现 | 需要多节点与 IB 硬件；单机测试不能替代 |
| §6.1–6.3 Eq.1–3 | 联合枚举 attention/MoE 并行、布局、overlap、q；内存过滤；SLO 二分批量；critical path 估计 | 当前优化器只覆盖部分布局与简化模型，需补齐可执行策略 | profile 拟合留出集误差、候选可执行性、预测/实测对比 |
| §6.4 Eq.4 | q 随当前/预期 decode tile footprint 缩放，按 cluster 向上取整，Qmax 保证 P 进展；单 phase 借用全部资源 | CPU 对照及默认 SM100 调度器均使用实际 masked/packed tile footprint；仍需完整策略质量与饥饿上界验收 | 稀疏度变化、空闲借用、P/D 饥饿测试 |

## 论文实验逐项对应

| 原文实验 | 复现要求 |
|---|---|
| §7.1–7.2 Fig.7–10 | ShareGPT / LooGLE 长度分布，Poisson 到达，所有方法按 PDD KV 容量使用相同的 sampled-sequence 长度上限；报告 TTFT、TPOT、joint attainment 曲线及 P90 goodput |
| §7.1 基线 | SGLang-Colocated、ChunkedPrefill、PDD、基于 Green Context 的 PDMux；均固定版本、精度、模型、总 GPU 数及调优规则 |
| §7.1 指标 | goodput 是至少 90% 请求同时满足 TTFT 与 TPOT SLO 的最高到达率；不是成功请求吞吐或 kernel 加速比 |
| §7.3 Fig.11–12 | 8192 P tokens、128 D tokens、top-8，decode 在 prefill 后 10μs 就绪；对比 exclusive、priority stream、MPS、Green Context、APK，报告 Pareto 与 slowdown |
| §7.4 Fig.13 | 对比原生 DeepGEMM：P 扫 per-expert batch、D 扫 active experts；调度开销单独报告 |
| §7.5 Fig.14 | dispatch/combine 分别对比 DeepEP v1 normal / low-latency；原论文 16 GPU，单节点适配必须明确标记 |
| §7.6 Fig.15 | 扫 token 数并测所有 MoE 操作的抢占间隔，验证与总任务长度无关；不得只测 host launch 时间 |

论文 MiniMax ShareGPT 使用 TTFT=1s、TPOT=50ms，LooGLE 使用 10s/100ms；
Coder BF16 的 SLO 必须在对比前固定，不能在看到结果后改阈值以得到 1.5×。
论文短请求默认 1P3D4E，长请求使用 3P1D4E；用户的 1P3D4E 保留为默认，其他布局作为明确标记的实验配置。

## 已取得证据（不等同于完整验收）

2026-10-15：8×B300，4E 随机 BF16 权重覆盖全部 62 层，初始 masked 版本每个 E 进程实测占用约 227.65 GiB（包含缓冲区，并非纯权重）；
1P+3D 并发逐层 MoE 测试通过。参考 CUDA kernel 的 20 次变化输入/路由的 graph replay 通过。
标准差 0.05 的大维度随机权重曾超出元素级误差阈值，日志保留；不能用后续标准差 0.02 的通过结果消除该失败记录。
新增 SM100 后端使用预建 TMA descriptors、TCGEN05 BF16、4-stage 加载及双 accumulator
流水线、producer/MMA/epilogue warp 分工，已通过 H=6144/I=2560 数值、并发、图重放与超时测试。
随后双 CTA DSMEM/multicast 也通过小形状与 Coder 专家形状回归；尚未测得抢占上界。
新增 COMM stream 与双 mailbox 路径：4 个 attention 客户端同时访问 4E，各自 12 次重叠结果对照及
20 次 graph 输出对照均与串行路径 bitwise 一致。这是 client 协议测试，完整 SGLang TBO 随后完成 24×128 token，并确认实际图回放。
Triton 小模型基线已完成 8 个请求、每个 32 个输出 token；分离部署用 INTRA_NODE_NVLINK 后也完成 8×32 token；严格对齐未全部通过。
尚无真实模型完整精度或相对 SGLang 基线 1.5× 的验收证据。后续已完成随机权重 HTTP goodput 诊断，结果落后基线，见 VALIDATION.md 的完整记录。

2026-10-15 后续修正：prefill 使用连续 routed rows，decode 独立 masked 容量；
当时默认 Coder mailbox 数据分配估算降至 1.374 GiB/E；后续为保证 TMA tile 对齐，改为 128-row 对齐后为 1.444 GiB/E（均不含 TMA descriptors/运行时开销）。
已发现融合 gate/up 缺少原生 BF16 GEMM 输出舍入边界，修正并保留此前测试记录；
该修正后结果必须重新验收。原生 DeepGEMM 同路由计算基线已采集，早期 APK 有明显性能差距。

2026-10-15 补充：独立 TP1 attention 的 TBO 元数据准备条件已修正；支持 SGLang 的 `-1` padding route，
部分/全部 padding 与变化输入图回放通过。P 无 chunking 的输入边界实测 512 token 成功、513 token 返回 400。
已增加可选逐 CTA tile trace，以及预先写入 payload 后由 GPU 定时发布 P/D 的诊断；
实测发布间隔约 10.016μs。trace 的 producer handoff→epilogue retirement 包含流水等待，
CTA0 admission 时间不等于 attention publication 时间，当前不能把它称为完整抢占上界证明。
TMA 全局 store 的完成通知使用不带 `.read` 的 bulk wait，避免仅等待共享内存读取结束便发布 DONE。

## 对现有 SGLang / DeepEP 优化的保留审计

本适配应扩展现有 serving 路径。当前 Qwen3 模型仍使用原生 attention、RMSNorm、RoPE、
router/TopK、LM head 和 sampling；scheduler、KV allocator、radix cache、Mooncake KV 移交
以及 CUDA Graph runner 也继续使用 SGLang。TBO 使用已有 OperationsStrategy / YieldOperation，
在 MoE 边界插入异步 dispatch / combine，而不重新实现调度器。

| 路径 | 当前处理与边界 |
|---|---|
| CPU/GPU scheduler overlap | 已移除适配器原先的强制关闭约束；forward 提交串行，跨层及跨迭代的 COMM 操作由 stream/event 排序，由真实模型并发回归单独验证。同步捕获 CPU 请求历史的数值诊断显式关闭此优化，不能用于性能测试。 |
| Decode CUDA Graph | 使用现有 runner；启动器默认开启，`--no-cuda-graph` 仅用于消融/调试。 |
| Sampling | 普通基线、ExpertPlex 与实验总入口均默认使用 FlashInfer；PyTorch 仍可显式用于数值诊断。 |
| FlashInfer autotune | 删除无条件禁用参数；保留显式调试开关，普通启动遵循 SGLang 行为。 |
| KV / prefix cache | 没有改写 allocator 或 radix cache，也没有默认关闭缓存；容量受论文跨方法相同 KV 预算约束。 |
| Attention backend | 保留 SGLang backend 选择；Triton 的固定数值对照不能代表 FlashInfer/FA4 等路径的最佳性能。 |
| Prefill graph / torch.compile | 当前 ExpertPlex 的 opaque Python/native 通信接口尚未完成该路径集成。这是实现差距，不应写成论文必然冲突。原生 DeepEP normal 路径自身也在参数解析中关闭 prefill graph。 |
| DeepGEMM | 复用原生 TMA、TCGEN05、双 CTA multicast、STSM/TMA epilogue 和 BF16 舍入边界；新增的是长驻 operation 选择与合作阶段切换。负收益的替代实现不能替换已验证路径。 |
| DeepEP | 普通 SGLang 基线继续调用原生 normal / low-latency dispatcher；ExpertPlex 的 attention-initiated M-to-N 不能直接套用要求 E 侧 receiver kernel 的原始 API。新的协议仍需继续对照 DeepEP 的路由布局、向量搬运和归并实现，不能据仅有功能通过宣称保留了全部性能优化。 |
| Chunked prefill / SBO / TP | chunked prefill 有显式选项；SBO 与 attention TP>1 等当前尚未适配，不能以“论文冲突”代替实现与验证。默认整请求 P 是论文实验配置，不是 SGLang 能力被全局删除。 |

基线启动器也应保留原生 overlap 与 autotune；PDMux 继续遵循其独立调度循环的约束。
恢复这些默认值之后，旧的固定参数性能结果仍保留，但不能当成新配置的验收结果。
