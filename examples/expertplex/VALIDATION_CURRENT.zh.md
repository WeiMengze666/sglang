# ExpertPlex 单机当前验收矩阵

本文档只记录可复核的门槛状态和 artifact 路径，不包含原始请求、score 或 trace 数据。实验根目录为 `/home/axis/mengze/expertplex`，主 checkout 为 `sglang` 的 `feat/expertplex-b300`。

## 2026-11-10 新机器恢复

新机器运行源码为 `ac1ded02567f8e50a337e2c23df36a44649b00d1`，使用真实
Qwen3-Coder-480B-A35B-Instruct BF16 权重、PyTorch 2.14.1/cu130。
默认独立 1P+3D+4E、P=TBO/D=NONE、FA4、FlashInfer sampling 和 Decode graph
`[16]` 已完成 12 次并发请求的服务冒烟测试，自有进程清理后无残留。
请求输出上限为 64 token，测试检查非空返回，模型质量与 goodput 验收仍待完成。
证据位于 `evidence/20261110-recovery/main-service-smoke-v3/report.json`。

逐 token 全词表比较按用户更新后的要求用于差异定位，允许浮点舍入和计算顺序
引起的差异。验收要求算法正确、模型级质量不下降，以及固定数据和
TTFT 1s / TPOT 50ms / 90% joint SLO 下约 1.5× goodput。
完整 HumanEval+ / MBPP+ 双系统对照已提交，结果待评分。
ShareGPT 原始数据及重新采样的 1000 请求文件已恢复，二者 SHA256 均与历史记录一致。

## 2026-11-02 历史机器结果

以下任务编号、数值诊断和性能数据来自旧机器。新机器的端到端性能需要重新测量。

| 门槛 | 当前状态 | 证据 |
|---|---|---|
| 默认 1P+3D+4E 独立原生 P/D/E | 通过功能与严格诊断 | `artifacts/20261102-recovery-compiled-long-default-v1/all-decode/report.json` |
| 默认 FlashInfer 长跨批次/query partition 全词表 | 通过 | job552；16 条混合 12K/8K/4K 历史，512 positions，151936 vocab，三路 D，raw digest `a6cf60e894e61191ccdf4d342c52eac241ac87924b68a3f4575e0fe72da189fc` |
| APK universal one-live-tile 组件门 | 通过 | job548；74/74 clusters、2490 phase switches、live tile ≤1、E-side decision→warp 22.592µs、cleanup 无残留 |
| CPU ExpertPlex contracts | 通过 | 261 tests，CUDA disabled |
| 最强 native fair baseline | 已测 | colocated DeepEP128，seed42，observed SLO goodput 4.99723 rps |
| ExpertPlex 最佳 fair screen | 未达到目标 | job551 FA4 为 1.32318 rps；job555 默认 FlashInfer 为 1.30608 rps；均未达到约 1.5× |
| optimizer held-out TP/SBO calibration | 未通过 | 现有严格 3% instrumentation/monotonicity gate 拒绝，保持 fail-closed |
| §7.3–7.6 完整 Pareto | 未完成 | 已有 isolated/component evidence，尚缺论文要求的完整对照 Pareto |
| 多节点 §5.3 RDMA | 按用户范围暂缓 | 当前仅单节点 |

性能 fair screen 固定 ShareGPT-24576、256 requests、seed42、TTFT 1000ms、TPOT 50ms 和 90% joint-SLO。不能把 strict numerical pass、组件 pass 或有限到达率结果提升为论文完整性能结论。

已排除的候选包括 q/SM budget、E8 duplicate-rank、coalescing、DecodeM=32、FA4/FlashInfer backend/fraction 的重复盲测；DecodeM=32 在 job559 的全词表比较出现大幅 logprob 与 top-1 mismatch，未进入性能路径。后续代码或 GPU 实验必须有新的可证伪假设，并经 gpuq 与 cleanup gate。

## 2026-11-02 环境恢复与 D 容量候选

重建环境的 gpuq shell 未默认激活 conda `bin`，导致 job566 在 CUDA 启动前找不到 Ninja；源代码已在 commit `2eb27d8201` 中让 `native.load_extension()` 自动发现解释器或 `sys.prefix/bin/ninja`，并新增 CPU 回归测试。修复后 CPU 合同测试为 261，optimizer/unit 测试为 59；`PATH=/usr/bin` 的实际发现检查也通过。

job569、job572 和 job576 均完成并清理成功，其完整配置只在 0.5 rps 达到 p90 joint-SLO。但配置审计发现，它们不是相对 job555 的单变量 D 容量实验：三者均省略 D 的 `overlap=none` 覆盖，继承全局 TBO，同时修改 P 服务并发上限、D graph bucket 等参数。此前“扩大/缩小 D mailbox 导致性能恶化”和“cap32 容量整体拒绝”的因果结论撤回；观测到的 SLO 失败与原始记录保留。

| 实验 | P/D overlap | P max running | D max running | D extend chunk | D max prefill tokens | D graph max batch |
|---|---|---:|---:|---:|---:|---:|
| job555 | TBO/NONE | 16 | 16 | 16 | 16 | 16 |
| job569 | TBO/TBO | 32 | 32 | 32 | 32 | 32 |
| job572 | TBO/TBO | 32 | 32 | 16 | 32 | 32 |
| job576 | TBO/TBO | 8 | 8 | 8 | 8 | 8 |

上表根据各实验冻结的 `run/node.json` 和 `run/service/{prefill-0,decode-1}.log` 中实际 `server_args` 核验。job569 的 goodput 为 `[0.488163, 0.488895, 0.114876, 0.061253]`，job572 为 `[0.492103, 0.473900, 0.140294, 0.060501]`，job576 为 `[0.490654, 0.487353, 0.086525, 0.049752]` rps，对应到达率 `[0.5, 1.0, 1.5, 2.0]`。这些结果只能评价各自的完整配置，不能独立归因于 mailbox 容量或 D extend chunk。默认 cap16 配置保留，后续诊断须先比较冻结配置与实际启动参数。

## 本地电脑发布

已遵照用户要求从本地 Mac 推送 GitHub；`feat/expertplex-b300` 的 `abcaa6d564b64b1231ca574b205fa16753345044` 已通过 `git ls-remote` 核验。仅推送源码、测试和文档，未上传原始实验数据。本地电脑的 HEAD、未暂存差异和暂存区均保持原样；后续已验证修改继续从本地电脑发布。
