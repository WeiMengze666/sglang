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
