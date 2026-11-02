# RemoteGroup 的 BF16 输出与客户端选择

客户端现在自动为 RemoteGroup 的 masked decode，以及启用 expert 侧 raw preprocessing 的 prefill 分配 BF16 输出。两种布局都会完整写回所有输出元素，包括 padding 和错误行，因此不再分配 FP32 中间输出后启动一次类型转换。

packed prefill 保留预清零的 FP32 累加及最终 BF16 转换；没有 RemoteGroup 的旧通信路径也保留原行为。`RemoteGroup.combine(output, ids, weights, row_map)` 继续同时接受符合布局约束的 FP32 或 BF16 输出。BF16 输出须位于 attention GPU，连续、二维、形状匹配且按 16 字节对齐；packed prefill 传入 BF16 仍会被拒绝。

FP32 按原路由顺序累加的计算不变，直接输出路径只在最终向量存储时执行一次 BF16 round-to-nearest-even。原 COMM stream 依赖、完成事件、pending 身份、mailbox 租约及 IPC 映射生命周期均保持不变。本次客户端选择与已验证的实现逐字节一致；native.py 和 C++ kernel 没有修改，native source fingerprint 仍为 `795670ad1e17d81ad951d53035a069944c33ba81dfafb80774f647ae6ea68dec`。

## 已验证范围

B300 的组件实验验证了 4 个 E peer、H=6144、2 个 mailbox：原 FP32 combine 加实际 CUDA BF16 转换，与直接 BF16 输出的全部有效元素逐位相等。覆盖 masked decode、raw prefill、保留的 FP32/packed 接口、空输入、padding、对齐和错误状态，以及图重放、双 pending、逆序 finish 和清理。组件采用合成 echo 数据，不执行完整 MoE；不能将这部分单独视为模型验证。

主模型数值实验使用 Qwen3-Coder-480B-A35B-Instruct BF16、单机 1P+3D+4E、独立原生 SGLang P/D/E 服务。对固定 16 条轨迹，沿锁定的 32-token 强制条件历史逐步对齐，并在三个 D 上重复验证；共 1536 个位置，每个位置的 151936 个 pre-forcing FP32 log-probability 均与锁定的原生 SGLang 参考逐位一致，共 233373696 个 FP32 词，零差异。参考来自已核验的原生 FA4 服务 cold、保留前缀缓存的 warm 和独立 fresh-repeat；本轮复用这些已锁定数据，没有重新启动 Native。

该验证配置明确为原生 deterministic/batch-invariant 模式、vendored FA4、PyTorch sampler、CPU overlap 开启、P=TBO、D=NONE、无 torch.compile、D full CUDA graph。保留请求内 KV；P 开启 radix 前缀共享并实际覆盖 miss/hit，D 使用正常 PD 的前缀索引关闭配置。所有原始前景轨迹都保留了完整词表、实际 GPU 输入、不可变调用身份、canonical token commit、终止多发一步、graph、实际 backend 元数据、PID 创建时间和清理门槛。

为保证首条前景请求确实进入多请求 decode，新增数值 cohort 在每个 D 上使用一条普通 greedy、最多 1024 token / 300 秒的有限背景请求；无采样强制、重试、扩展预算或调度器 hold。实际 ForwardBatch RID 与同调用的 canonical commit 证明了背景同批次，而非使用 SSE 心跳代替。全部前景 case 都观察到实际 batched host pipeline；六次仅针对自有 RID 的中止成功，进程清理无残留。背景请求改变了验证工作负载，因此这项结果仅证明这一配置和固定轨迹的严格数值通过。

该结果不证明关闭 deterministic、FlashInfer sampler 或其他未测配置也逐位一致，不是生产 B 模式的完整验收，也不构成端到端吞吐、goodput、约 1.5× 提升或论文全部复现完成的证据。性能仍须在相同输入与 SLO 下，对充分调优的最强原生 SGLang 基线独立测量；本次不改变性能基线。
