# RemoteGroup BF16 输出接口的组件验证

本次只为 RemoteGroup.combine(output, ids, weights, row_map) 增加可选 BF16 输出接口。默认客户端继续分配 FP32 输出并执行原来的最终类型转换；没有切换默认行为。

BF16 输出适用于 masked decode 和启用 expert 侧 raw preprocessing 的 prefill。输出必须是所在 attention GPU 上的连续二维张量，形状与原 FP32 接口一致，数据指针按 16 字节对齐。packed prefill 继续使用原来预清零的 FP32 累加接口，传入 BF16 会被拒绝。调用方仍须保持原有 COMM stream、事件及 IPC 映射生命周期。

新路径保留 FP32 累加、路由顺序和错误处理，只在最终向量存储时执行一次 BF16 round-to-nearest-even。调用方可以显式分配 BF16 输出后传给该接口；本变更不自动选择该路径。

## 已完成的组件范围

B300 的 gpuq 实验 650 验证了同一 C++ 翻译单元：

- 4 个 E peer、H=6144、2 个 mailbox；对比原 FP32 combine 后的实际 CUDA BF16 转换与直接 BF16 输出，全部有效元素逐位相等。
- masked decode 与 raw prefill 两类布局，以及保留的 FP32/packed 接口。
- 图重放、空输入、路由与 padding、BF16 对齐检查、错误状态、双 pending 和逆序 finish；进程清理无残留。

该实验使用通信/输出转换组件的合成 echo 数据，local experts=8、top-k=8、I=64，不执行完整 MoE 计算。图重放是这个组件的图重放，不等同于完整模型的 serving graph 验收。

验证时的 native source fingerprint 为 795670ad1e17d81ad951d53035a069944c33ba81dfafb80774f647ae6ea68dec。本次接口切片保留相同的 native.py 与 C++ 输入，客户端默认路径未修改。私有复核记录位于实验工作区的 artifacts/20261102-group-combine-bf16-component，原始数据不随源码发布。

这些证据不代表 Qwen3-Coder-480B 的完整逐 token 全词表正确性通过，也不代表端到端性能提升或论文复现完成。切换默认客户端行为仍须独立验证。
