# ExpertPlex 单机当前验收矩阵

本文档只记录可复核的门槛状态和 artifact 路径，不包含原始请求、score 或 trace 数据。实验根目录为 `/home/axis/mengze/expertplex`，主 checkout 为 `sglang` 的 `feat/expertplex-b300`。

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
