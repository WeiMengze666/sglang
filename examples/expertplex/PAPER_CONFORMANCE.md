# ExpertPlex 原论文对照与验收记录

## 验收口径更新（2026-11-11）

按用户更新后的要求，逐 token 全词表与逐位比较用于定位数值差异。最终验收要求
模型推理算法实现正确，并达到固定数据及
TTFT 1s / TPOT 50ms / 90% joint SLO 下的 goodput 目标。浮点舍入和计算顺序带来的
差异需要解释来源，历史逐位诊断结果按其原始条件保留。

最新优先级是先收敛性能，允许一定质量偏差。性能迭代只从已有 benchmark 输出
检查是否可读，不重复生成、对齐或评分模型质量；达到性能目标后集中验收质量。
算子越界、非有限输出、协议错误等实现问题仍需修复。模型质量对照最终固定权重、
数据、提示模板、生成参数及评分实现，记录逐题结果与差异。
Coder 主模型已完成一次 HumanEval+ 和 MBPP+ 配对评分，以及选定差异题目的重复
诊断，条件和结果见 [README](README.zh.md#2026-11-10-主模型配对质量检查)。
完整模型质量和约 1.5× goodput 验收尚未完成。

新 B300 已完成此版本的 CUDA 编译，以及 grouped/per-peer packed IPC 的 FP32 和
最终 BF16 检查。PyTorch 2.12/cu130、2.14.1/cu130 两个环境均运行实际 GPU 测试，
各 5 项通过。6144×2560 专家形状的 SM100 scatter/GEMM/activation/gather、混合 P/D
和 20 次 graph replay 共 35 项满足原有算子容差，最大绝对误差为 0.001953125。
该算子对照在 CPU 按 expert 顺序加权相加，生产 EP gather 的 route 顺序与 FMA
边界需继续结合模型级评测核对。此记录覆盖组件验证，主模型服务与质量评测继续进行。

依据：用户工作空间原始 PDF，ExpertPlex, arXiv:2607.18002v2，2026-07-21，§1–9。
原文：https://arxiv.org/html/2607.18002v2 。本地原始 PDF 已逐节核对。
PDF SHA256：`8321f7095ccdde149178e16fa3cdbfa60a7a87fd42920483acb21e0cac2a363a`。
本文件记录复现要求及当前差距，不表示全部要求已经实现。

当前用户确认的验收范围为单节点 8×B300。§5.3 跨节点 MoE RDMA 转发与
InfiniBand 流量隔离暂缓；保留这些原文要求以供后续多节点验证，不以单机结果代替。

1P+3D+4E 按物理 GPU 角色计数，P/D/E 三类服务分别通过原生 SGLang 入口启动。
三个 D GPU 可由独立 D 前端或一个原生 attention-DP3 前端管理，两种配置均纳入
主性能验收。性能报告记录 D 组织方式及实际启动参数，并与公平调优后的最强
原生基线比较。

历史数值证据：更换机器前，真实 Coder BF16 权重在确定性模式下通过固定历史、并发请求的
全词表逐位对照，覆盖三路 D 同时活动、实际 decode 批处理和非空 TBO 重放完成。
三路 D 重复使用同一组参考轨迹；这不代表更多独立提示，也不是完整模型质量评估。
数值捕获用于正确性检查，不作为性能证据，也不能替代新环境复验；调度重叠等运行配置以具体验证记录为准。
APK 的算子开销、完整传播上界、
优化器搜索与预测质量、SBO、attention TP 和编译路径的运行覆盖仍有缺口；
尚未达到相对最强原生基线的端到端 1.5× P90 goodput 要求。

辅助阅读：[HaoyangPing 的 ExpertPlex 解读](https://pinghaoyang.com.cn/aigc/posts/expertplex/)。
实现约束仍以以上原始 PDF 为准，尤其是 tile 提交后的协作切换、单边最终缓冲区访问、
TBO/SBO 的关键路径，以及按联合 TTFT/TPOT SLO 定义的 P90 goodput。

## 2026-11-18 原生 SGLang 基线复测

在 8×B300 上用 Coder-480B BF16 交替运行两种原生 DP 请求分配策略。每轮重新
启动服务，先测 2.5 再测 2.75 requests/s。每个到达率使用相同的 256 个
ShareGPT 请求，seed 42，预热 8 个请求。下表按启动顺序列出同时满足
TTFT ≤ 1s 和 TPOT ≤ 50ms 的请求数。每格至少需要 231 个请求达标。

| 轮次 | DP 请求分配策略 | 2.5 requests/s | 2.75 requests/s |
| --- | --- | ---: | ---: |
| 1 | `round_robin` | 231 / 256 | 223 / 256 |
| 2 | `total_requests` | 223 / 256 | 224 / 256 |
| 3 | `total_requests` | 220 / 256 | 225 / 256 |
| 4 | `round_robin` | 228 / 256 | 227 / 256 |

八个 rank 的实际配置均为 TP8/EP8/attention-DP8、DP LM head、FA4、
FlashInfer A2A 和 `flashinfer_trtllm_routed`。原生调度重叠开启，TBO 关闭，
请求上限为 128，每个 DP rank 的 prefill chunk 为 6,144 tokens，decode
使用 full CUDA Graph，最大 batch 为 32。四轮的 2,048 个请求均成功完成；
逐请求复算联合 SLO，并核对了输入和输出长度序列。

2.5 requests/s 在四轮中有一轮刚好达标，2.75 requests/s 四轮均低于门槛。
这次复测保留此前原生基线在 2.5 requests/s 的达标记录，也显示了边界附近的
运行间波动。ExpertPlex 的 1.5× 验收仍需双方在相同负载下的容量对照。

### 1P+3D+4E 主模型的解码调度与 TBO

以下两组对照沿用相同请求和联合 SLO。P 使用单批，三张 D 卡组成原生
attention-DP3 服务。每行是一轮 256 请求/到达率的结果。请求分配组使用同一
源码。解码重叠组在修复 TBO 子批的路由和 scatter 后启用 TBO，并与先前完成的
同配置单批解码比较。

| 对照组与设置 | 3 requests/s 达标数 | 3.75 requests/s 达标数 | 3.75 requests/s 的 TPOT P90 |
| --- | ---: | ---: | ---: |
| 请求分配，D 单批，`round_robin` | 222 | 169 | 52.77 ms |
| 请求分配，D 单批，`total_requests` | 232 | 169 | 53.31 ms |
| 解码重叠，`round_robin`，D 单批 | 221 | 160 | 52.88 ms |
| 解码重叠，`round_robin`，D 开启 TBO | 12 | 8 | 79.89 ms |

在 3.75 requests/s，按在途请求数分配将 TTFT 超时从 33 个减至 1 个，
仍有 87 个请求超过 TPOT 门槛。3 requests/s 有一次达标；与原生已通过的
2.5 requests/s 相比，这两个离散负载点之比为 1.20。容量边界仍待复测。

GPUq 154 的客户端流式记录还能区分第一个 token 间隔和持续解码。第一个间隔从
客户端收到首 token 算起。后续间隔先在每个有至少两个间隔的请求内取中位数，
再在请求之间取中位数。

| 到达率 (requests/s) | 首个间隔中位数 (ms) | 后续间隔中位数 (ms) | TPOT 超时请求数 | 超时请求中超过 75 ms 的间隔位置 |
| ---: | ---: | ---: | ---: | --- |
| 3.00 | 87.23 | 46.04 | 24 | 24 个，均在首个间隔 |
| 3.75 | 93.38 | 48.84 | 87 | 82 个，均在首个间隔 |

原生客户端在一个流式事件携带多个新 token 时均分该事件的间隔。每档有
52,900 个已记录间隔。237/256 个请求的间隔数等于输出 token 数减一，
其余 19 个共少 26 个间隔。表中统计只用实际记录。达标数仍按每个完整请求的
TTFT、总延迟和输出长度计算。

较早的 GPUq 72 单变量对照也复现了首间隔现象。在 3.75 requests/s 下，关闭 D 的
原生 CPU 调度重叠把首间隔中位数从 94.76 降至 53.45 ms，但后续间隔中位数从
49.01 升至 54.60 ms，联合达标数从 140 降至 36。性能配置继续开启调度重叠。
验收后续改动时还需比较从发出请求到收到第二个 token 的绝对耗时，以及端到端
延迟，避免仅凭首间隔的变化判断加速。

解码 TBO 的功能验证覆盖三张 B300 上 117 次 rank-local scatter 的 CUDA Graph
重放，以及两档到达率的 512 个主模型请求。该配置显著增加逐 token 延迟；
当前性能实验继续使用 D 单批。

GPUq 162 在 GPUq 154 的配置上只更改了专家内核的解码行 tile 查找。对于使用
交错 SwiGLU 权重、每个专家容量至多为一个行 tile 的 decode mailbox，APK 在
接纳时将非空专家的 ID 和有效行数写入现有前缀数组占用的空间。每次领取 tile
即可直接索引。容量更大、prefill 或独立激活的路径继续使用前缀索引。
四个 E 进程均加载了候选库，两个负载点的 512 个请求全部完成，输入长度和输出
长度与对照逐请求一致。

| 到达率 (requests/s) | 原索引达标数 | 新索引达标数 | 原 TPOT P90 | 新 TPOT P90 |
| ---: | ---: | ---: | ---: | ---: |
| 3.00 | 232 / 256 | 235 / 256 | 49.36 ms | 49.04 ms |
| 3.75 | 169 / 256 | 200 / 256 | 53.31 ms | 51.96 ms |

在 3.75 requests/s，35 个请求从超时变为达标，4 个从达标变为超时。候选的
TTFT P90 为 547.33 ms，仍有 54 个请求超过 TPOT 门槛，距离 231 个联合达标请求
还差 31 个。这是一组分别启动的对照，容量结论仍需复测。

在这 54 个 TPOT 超时请求中，35 个只生成了 2–32 个 token，16 个生成了 33–128 个，
3 个生成了 129–512 个。按原请求的 TTFT、输出长度和后续间隔重算，将首个流式间隔
超过该请求后续间隔中位数的部分扣除，联合达标数会从 200 变成 234。六个缺少后续
间隔的请求按原值计算。这是对固定轨迹的敏感性分析，实际调度改动还需验证发出请求
到收到第二个 token 的绝对时延、完整请求时延和相同负载下的联合达标数。

GPUq 164 对原索引单独采样。只取同一负载前 32 个请求，在三个 D rank 各采集六次
forward，每卡覆盖 372 层。负责向 E 分发工作的 D rank 在 `group_wait` 中累计
163.59 ms，在 reduce-scatter 中累计 4.66 ms。其余两个 rank 的 reduce-scatter
分别累计 191.48 和 191.46 ms。参与同一 collective 的 rank 可以在不同时间进入，
这些跨卡时长不能相加为请求延迟。该次诊断用 237,088 条匹配的 A/E rank 操作
核验了分发路径。正式 goodput 使用关闭采样的 GPUq 154/162。

按论文公式 2 的 MoE 形式，在每张 E 卡上用解码行 tile 数拟合该卡从接纳到完成的
时间。训练和留出集按完整 forward 的先后顺序平分，只保留本卡没有同时执行 P 的
非空 D 操作，并在训练集覆盖的 tile 数范围内评估。GPUq 69 的 1,309 个留出
forward 中，四卡各自的平均绝对误差为 4.4–4.8 µs，使用该卡训练集平均时延预测的
误差为 157.4–159.4 µs。GPUq 164 的 457 个留出 forward 中，对应误差为
5.0–5.3 µs 和 100.3–104.5 µs。两组分别拟合、分别检验。这支持在给定执行路径上
使用行 tile 数预测 E 的解码时间。P/D 争用、指定 SM 预算的性能和完整请求的排队
时间仍需各自标定。[论文 §6.2](https://arxiv.org/html/2607.18002v2#S6.SS2)

## 2026-11-17 性能证据更新

以下汇总维护恢复后完成的实验。主模型为 Coder-480B BF16，运行时基于
`ac1ded0256` 和已验证的融合实现，使用 Torch 2.14.1+cu130。
原始请求、轨迹和实验凭据保存在工作区 evidence 中。

### 主模型容量扫描

GPUq 5、7 使用同一固定 ShareGPT 样本、seed 42、24,576-token 序列上限，
每个到达率 256 个请求，预热 8 个请求。联合门槛为 TTFT ≤ 1s、TPOT ≤ 50ms，
至少 90% 请求同时满足两项。两组服务均完成全部请求及自有进程清理。

原生配置采用 TP8/EP8/attention-DP8、DP LM head、FlashInfer A2A 与
`flashinfer_trtllm_routed` BF16、FA4 和 FlashInfer sampling，关闭 TBO，保留
CPU 调度重叠。全局 chunked-prefill 上限为 49,152，每个 DP rank 为 6,144。
ExpertPlex 采用物理 1P+3D+4E、可选的单组 attention-DP3、`sm100_swiglu`，
P/D 均关闭 TBO，保留 CPU 调度重叠，D 全局请求上限为 48、每 rank 为 16。
三路独立 D 实验还用于分析 attention 组织方式对结果的影响。

| 配置 | 到达率 (requests/s) | 联合达标请求 | TTFT P90 (ms) | TPOT P90 (ms) |
| --- | ---: | ---: | ---: | ---: |
| 原生 SGLang | 2 | 245 / 256 | 488.19 | 38.93 |
| 原生 SGLang | 2.5 | 233 / 256 | 533.88 | 48.72 |
| 原生 SGLang | 3 | 210 / 256 | 540.63 | 59.16 |
| ExpertPlex，attention-DP3 | 2 | 245 / 256 | 461.17 | 46.84 |
| ExpertPlex，attention-DP3 | 3 | 204 / 256 | 516.22 | 51.35 |
| ExpertPlex，attention-DP3 | 3.5 | 132 / 256 | 899.17 | 53.75 |
| ExpertPlex，attention-DP3 | 4 | 91 / 256 | 5114.56 | 55.27 |

最高已测可行到达率分别为 2.5 和 2 requests/s，有限测试点比值为 0.8。
精细边界、重复运行、留出负载和更强原生配置仍需验证。
较早 64 请求试跑的 3 requests/s 达标结果在此次 256 请求扫描中未复现。
完整复现与约 1.5× goodput 目标仍待完成。

### 专家执行与逐层计时

GPUq 53 对照原生 DeepGEMM 和融合 APK，使用相同合成 BF16 权重、路由与输入，
H=6144、I=2560、40 个本地专家、148 SM、每种条件 50 次 CPU 间隔提交。
计时取同一控制 GPU 的 publication-to-ACK 区间，覆盖隔离 E 执行。
主模型 attention、传输与完整服务延迟需分别测量。

| D 活跃专家数 | 每活跃专家行数 | 原生 D (μs) | APK D (μs) | APK 延迟增幅 |
| --- | ---: | ---: | ---: | ---: |
| 22 | 4 | 362.032 | 406.880 | 12.39% |
| 29 | 4 | 461.712 | 511.376 | 10.76% |

配对 P 条件为每专家 128 行，原生约 665 μs，APK 约 748 μs，延迟增幅约 12.41%。
这些结果确认当前 APK 相对原生专家计算仍有开销，尚需分离队列、流水线和数据操作的贡献。

GPUq 8 的 A/E 有界记录器完整关联 707,048 个 rank 操作。
在同一 A GPU 时钟上，2,601 个完整 62 层 D 周期从首层 MoE dispatch 到末层 E ACK
平均为 40.424 ms，其中 dispatch 至 wait 为 2.914 ms、MoE 等待为 26.028 ms、
层间间隔为 11.483 ms。层间间隔包含 combine、attention、router 和 DP collective。
该范围覆盖启动、图捕获、预热、健康检查及 benchmark，首层 attention 和末层
combine、norm、LM head、sampling 位于其范围之外。完整 forward 仍需独立计时。

同次 3 requests/s 的记录器关闭/开启实验分别有 202/256、195/256 请求联合达标，
后续性能比较继续使用关闭记录器的配置。D 的 group wait 平均为 419.80 μs，
同次调用各 E 本地持续时间最大值平均为 397.85 μs；两者具有各自的区间边界，
通信、准入和轮询开销需用匹配区间继续分解。

同一记录中，645,048 个 D rank 操作的 row-tile footprint 中位数为 23，P90 为 29。
配置为 `q=96`、期望 footprint 为 8、上限为 146 SM；将每个 D 操作的 footprint
代入 Eq.4，名义预算在 71.28% 的操作中触及上限，按 D 持续时间加权为 89.67%，
按与 P 重叠的时长加权为 97.32%。记录中同时准入的 D 操作最多一个。
该推算使用包含启动、图捕获、预热和健康检查的完整记录；实际策略传播、空闲资源借用
及 SM 占用仍需单独测量。期望 footprint 的后续标定需要 benchmark 区间和配对实验。

GPUq 94 随后固定生产配置的 `q=96`、`Qmax=0.99` 和 148 CTA，将预期 decode
footprint 从 8 改为 benchmark 窗口中位数 22。两组采用相同 BF16 权重、路由、
输入和 `sm100_swiglu` 实现，H=6144、I=2560、40 个本地专家。P 为 1024 行，
每个条件采样 50 次，计时使用同一控制 GPU 的 publication-to-ACK 区间。
六次运行通过组件结果、配置、样本和自有进程清理检查。

| D 活跃专家数 | 每专家行数 | P/D 并发时 P 加速比 | P/D 并发时 D 加速比 |
| --- | ---: | ---: | ---: |
| 22 | 4 | 1.00898 | 0.70618 |
| 29 | 4 | 1.00401 | 0.89075 |
| 22 | 17 | 1.00541 | 0.70650 |

加速比为预期 footprint 8 的延迟除以 22 的延迟。D 延迟增加约 12%–42%，P
收益不足 1%，因此保留预期值 8，暂不将该探测配置推进主模型。22 取自已测
分布中位数，离线优化器的最优参数仍需标定。这些组件结果尚不能确定端到端 goodput。

GPUq 97 完成等待 epoch 预取候选的后续检查，累计通过 9 项协议检查和两个
后端共 38 项边界、并发与图重放检查。固定合成输入的 12/24/48 行传输实验中，
原实现 combine 延迟分别为 61.392/59.424/59.488 μs，候选为
62.368/61.440/61.488 μs，每项各 30 个无 profiler 的 CUDA Graph 样本。
候选回退约 1.6%–3.4%，继续保留原等待实现。该计时在 E echo 完成后开始等待，
没有包含 E GEMM；模型级收益仍需主模型测量。

另一个独立修复调整了公共等待函数的控制字读取顺序，先 acquire DONE，再读取
ERROR。旧顺序可在两次读取之间遇到 E 发布错误及完成，从而把失败 epoch 返回为
成功。确定性 CPU 回归编译实际等待函数，在该交错下旧版本失败、修复版本通过。
GPUq 99 的修复版本通过 16 对真实 E 错误发布及失败后图重放检查、9 项协议检查，
以及两个后端共 38 项成功路径检查。GPU 基线的 16 对检查也全部通过，实际 GPU
运行尚未触发这个短暂交错。此修复的证据覆盖协议正确性，性能收益尚未测得。

GPUq 101 检查了按最多 8 条 route 预取远端向量、再按原顺序执行 FP32 FMA 的
combine 候选。基线与候选分别通过 72 项路由、dtype、尾部、padding、重复 expert、
权重与图重放检查；候选另通过 9 项协议检查和两个后端共 38 项 full-MoE 检查。
固定 H=6144、top-8、四个 E echo 的传输配对中，每个条件采样 30 次 CUDA Graph。

| 行数 | 原实现 combine (μs) | 预取候选 combine (μs) | 组件加速比 |
| ---: | ---: | ---: | ---: |
| 12 | 61.408 | 51.200 | 1.19937 |
| 24 | 59.392 | 51.200 | 1.16000 |
| 48 | 61.376 | 51.200 | 1.19875 |

输入张量逐项相同，计时包含 wait 和 combine，E echo 在计时前完成。该实验覆盖
通信组件，E GEMM、P 并发和主模型服务的表现由单独实验测量。生成代码的 BF16
输出实例寄存器数从 40 增至 92，FP32 从 48 增至 88，STACK 和 LOCAL 均为 0。
sender-packed P 与 masked D 共用 kernel 实例，P 路径也受该寄存器分配影响。

GPUq 108/113 将 packed 与 masked combine 按模板参数分别实例化，保留各自的
累加顺序及舍入边界。候选通过 72 项 masked 路由检查、4 项 packed IPC 子场景
及 9 项协议检查。以下配对结果使用相同输入、H=6144、top-8、四个 E echo，
每项采集 30 个 CUDA Graph 样本。

| packed 行数 | 共用实例 combine (μs) | 分离实例 combine (μs) | 组件加速比 |
| ---: | ---: | ---: | ---: |
| 128 | 85.856 | 61.440 | 1.39740 |
| 512 | 192.512 | 120.832 | 1.59322 |
| 2048 | 612.368 | 337.920 | 1.81217 |

配对输入与输出张量检查通过。packed FP32 实例的寄存器数从 88 降至 54，
STACK 和 LOCAL 均为 0。计时覆盖 wait 与 FP32 combine，E echo 已提前完成，
最终 BF16 转换、E GEMM 及并发服务开销另行测量。masked 12/24/48 行的 combine
仍约为 51.2 μs。下文 GPUq 135 的主模型组合对照给出了相同传输路径的服务指标。

GPUq 126 对公开源码组合重新执行了 72 项 masked 路由检查及 packed combine 的
5 项测试，全部通过，自有进程完成清理。native 源码指纹为
`bd3ab36f6edaf2e1faffc0353cd0434d21e24337e2d9a1d1d6e04bb758761775`。
公开实现保留原有的 65536 条 attention trace 上限。路由边界回归入口为
`test/manual/expertplex/check_combine_routes.py`，需在至少六张可见 GPU 的分配中运行。
`--check-cpu` 仅检查 36 组精确 dyadic 参考数据。

GPUq 104 随后验证默认的独立 1P+3D+4E 服务，固定 P=TBO、D=none、四个 A 服务
均为 TP1、每服务请求上限 16、D graph buckets 为 1/2/4/8/12/16，保留 CPU
调度重叠、FA4、FlashInfer sampling 和关闭记录器的配置。候选将 route 向量读取
提前至原顺序累加之前。实际四个 A 服务参数、输入/输出长度、完整请求数与自有进程清理均通过检查。

| 配置 | 到达率 (requests/s) | 联合达标请求 | TTFT P90 (ms) | TPOT P90 (ms) |
| --- | ---: | ---: | ---: | ---: |
| 原实现，独立三路 D | 3 | 31 / 256 | 1487.47 | 65.08 |
| 预取候选，独立三路 D | 3 | 43 / 256 | 4028.88 | 63.73 |
| 原实现，独立三路 D | 3.75 | 12 / 256 | 12721.92 | 72.32 |
| 预取候选，独立三路 D | 3.75 | 20 / 256 | 15081.86 | 69.38 |

这两个到达率均未达到 90% 联合门槛。候选的 TPOT 下降、TTFT 上升，prefill
延迟变化的原因仍需单变量实验定位。此前 attention-DP3 的 P/D 均无 TBO，其结果
与本轮拓扑和 overlap 设置不同。完整复现与约 1.5× goodput 目标仍待完成。

GPUq 104 的候选 2 requests/s 点排除在配对性能验收之外。排查启动延迟时读取的
CPU 栈实际落在该点的正式计时区间；原始数据及时间记录已保留。3 和 3.75
requests/s 在采样结束后开始。GPUq 106 已完成同一候选源码的 P=TBO/none
单变量对照，D 固定为 none，重新测量 2 和 3 requests/s，全程关闭诊断采样。
已检查原有 benchmark 的少量输出文本可读，完整模型质量仍按约定留待性能目标后验收。

### 独立服务的 prefill overlap 对照

GPUq 106 复用上述 route 预取候选，固定默认独立 1P+3D+4E、每 A 请求上限
16、D CUDA Graph、CPU 调度重叠及同一 ShareGPT 样本。仅改变 P 的 TBO
开关。四个到达率测试均完成 256 个请求，实际服务参数、请求长度及自有进程清理
通过检查。

| P overlap | 到达率 (requests/s) | 联合达标请求 | TTFT P90 (ms) | TPOT P90 (ms) |
| --- | ---: | ---: | ---: | ---: |
| TBO | 2 | 210 / 256 | 992.50 | 49.51 |
| none | 2 | 236 / 256 | 482.50 | 49.47 |
| TBO | 3 | 46 / 256 | 1438.14 | 64.36 |
| none | 3 | 37 / 256 | 1467.16 | 65.35 |

P=none 在 2 requests/s 达到 92.19% 联合达标率，TTFT 超限请求为 0，TPOT
超限请求为 20。该配置最高已测可行到达率为 2 requests/s。3 requests/s 时
仍有 205 个请求的 TPOT 超限，其中 19 个同时违反 TTFT；P-TBO 开关对照确认
当前仍需优化 decode 路径。此结果为单次配对扫描，重复性、最强原生基线和
约 1.5× goodput 验收继续进行。

### Dispatch 融合对照

GPUq 110 将 masked dispatch 的 clear、payload copy 和 publish 合为一个
cooperative kernel，用两次 grid barrier 保持阶段顺序，并按驻留容量选择原路径
回退。候选通过 72 项路由检查、9 项协议检查、两个 MoE 后端共 38 项边界检查，
以及 0/1/2048 行的图重放检查。GPU trace 确认小形状执行融合路径，2048 行
执行原有三个 kernel。配对输入与最终输出逐项相同。

| 行数 | 原实现 dispatch (μs) | 融合候选 dispatch (μs) |
| ---: | ---: | ---: |
| 12 | 55.232 | 65.568 |
| 24 | 55.328 | 67.520 |
| 48 | 61.440 | 71.680 |

每项为 H=6144、top-8、四个 E echo 下 30 次关闭 profiler 的 CUDA Graph
采样中位数。候选耗时增加约 16.7%–22.0%，继续使用原 dispatch。此结果覆盖
attention 侧通信组件，主模型收益仍需独立测量。

GPUq 11 的动态 Decode TMA 范围候选通过 58 项算子检查，三个 D 条件的加速比为
1.02164、0.99234、0.98566，收益混合，因此继续保留已验证的固定范围实现。
GPUq 64 的分片完成计数候选也通过 58 项算子检查。D 活跃专家数及每专家行数
分别为 22×4、29×4、22×17，每种条件采样 50 次，使用同一控制 GPU 的
publication-to-ACK 区间。独立 D 加速比分别为 0.97985、1.00398、0.99101，
P/D 并发时 D 加速比为 0.99073、0.99474、0.99350。三组条件均未显示稳定收益，
继续使用已验证的全局完成计数实现。该候选保留为组件实验记录。
GPUq 60 的融合 DP padding 路由掩码通过 3 个测试方法、19 个形状/类型/图状态检查，
覆盖有效路由逐位保留、输入无修改、非连续张量和固定地址下动态计数图重放。
原生 `mask_topk_padding` 入口也通过输出检查。每 rank 行数 1/4/8/16 的配对
CUDA Graph 微基准中，原实现约 12–14 μs，融合实现约 4.6–4.7 μs，加速比
为 3.112/2.617/2.619/2.636。

GPUq 62 完成同配置、关闭记录器的原始/融合掩码主模型配对比较。
固定 3 requests/s、256 请求下，联合达标数为 202/256 和 220/256，
TPOT P90 为 51.70 ms 和 50.50 ms，TTFT P90 为 545.11 ms 和 509.56 ms。
实际 P/D 配置均关闭 TBO、保留 CPU 调度重叠，输入/输出长度数组一致；
两组自有进程均完成清理。融合路径的联合达标率为 85.94%，仍低于 90% 门槛。
该单次固定到达率比较需要后续重复和容量扫描。

GPUq 69 的完整 forward 诊断复用了融合掩码配置。关闭/开启记录器时，3 requests/s
的联合达标数分别为 223/256、220/256，TPOT P90 为 50.21/50.46 ms。
四个 A rank 的原生 CUDA event 记录均完整结束，A/E 完整关联 718,208 个 rank
操作。benchmark 进程窗口内，三个 D rank 的非空 decode forward P90 分别为
48.42、48.70、48.63 ms。该窗口包含预热和健康检查；forward 按批次计时，TPOT
按请求计算，两者需分别报告。

按 D 组首个 rank 的原生 forward 耗时分组，可以进一步定位这次采样的尾延迟。
每个 forward 按首末 epoch 和本地时间戳关联到 62 层 A/E 记录。下表只纳入
benchmark 进程窗口中的非空 decode，窗口含该进程的预热。各列均为组内均值。

| 原生 forward 耗时分位 | forward 数 | 完整 forward (ms) | 等待 E 返回 (ms) | 相邻 MoE 调用间隔 (ms) | 与 P 工作重叠的层数（共 62 层） |
| --- | ---: | ---: | ---: | ---: | ---: |
| P50 至 P90 以下 | 1,047 | 46.79 | 33.10 | 10.09 | 11.00 |
| P90 及以上 | 262 | 49.30 | 35.37 | 10.31 | 34.11 |

完整 forward 的组间差为 2.51 ms，其中等待 E 返回的时间增加了 2.27 ms。
在同一 E rank、模型层和本地 GEMM 行 tile 数内比较，按 D 被接纳时已有 P 在运行的
样本分布加权，E 接纳至完成的均值为 494.54 μs，另一组为 449.83 μs。
该比较覆盖 3,047 个两组各有至少五次观测的组合，分别有 45,960 和 347,732 个
rank 操作。权重路由、每个 tile 的有效行数及后续 P 到达仍可变化。
此处测量属于 GPUq 69 的 3 requests/s 诊断，当前 3.75 requests/s 的瓶颈与
优化收益仍需在对应配置下测量。

同一窗口还可按层比较四个 E rank。2,617 次 forward 对应 162,254 组四卡记录。
每张卡使用本地时钟，分别计算接纳至完成的耗时。在同一层、同一次调用内，
耗时最长的 rank 同时也是行 tile 数最多的 rank 的比例为 98.78%（并列耗时均分）。
按该层并列最多的 rank 数计算，均匀随机选卡时的对应比例为 30.06%。
四卡最长耗时与四卡均值的差，在中间组为 47.53 μs，在最慢十分位为 47.92 μs。
因此这两组之间观察到的等待差异，主要还需从四卡共同耗时、准入和 P 并发等方向分解。

用前 1,308 次 forward 为每层选出平均行 tile 数最多的 rank，在后 1,309 次
forward 中，该 rank 的本地耗时平均高出四卡均值 17.49 μs。逐次选取耗时最长的
rank 则高出 42.20 μs。每层确有可重复的路由偏斜，具体批次也影响哪张卡最慢。
这些本地耗时差未计入各卡准入时差，专家重排对端到端性能的影响仍需实测。

GPUq 70 仅改变 D 服务的原生请求分配策略，保留上述容量、图和 CPU 调度重叠配置。
每个到达率各 256 个请求，两组实际启动配置、请求长度及自有进程清理均已核验。

| 原生 D 策略 | 到达率 (requests/s) | 联合达标请求 | TTFT P90 (ms) | TPOT P90 (ms) |
| --- | ---: | ---: | ---: | ---: |
| round_robin | 3 | 219 / 256 | 522.76 | 50.57 |
| total_requests | 3 | 227 / 256 | 467.67 | 50.59 |
| round_robin | 3.75 | 141 / 256 | 2424.11 | 53.75 |
| total_requests | 3.75 | 156 / 256 | 474.22 | 53.22 |

`total_requests` 改善了本次配对的 TTFT，两个到达率均未达到 90% 联合门槛。
GPUq 81 随后对原生基线执行了同类策略配对，结果见下表。

GPUq 72 的单变量配对验证了 grouped-D 的两种原生 CPU 调度循环。每个到达率
各 256 个请求，实际 P/D 参数、全部请求长度和自有进程清理均已核验。
在 3 requests/s 下，重叠开启/关闭的联合达标数为 222/256、52/256，TPOT P90
为 50.43/54.24ms，平均 TPOT 为 46.20/51.18ms。在 3.75 requests/s 下，联合
达标数为 140/256、36/256，TPOT P90 为 53.41/56.64ms。适配器现支持两种循环，
普通调度循环用于显式消融，性能配置采用默认的 CPU 调度重叠。

GPUq 79 使用原生 DP3 通信组测量 BF16、H=6144、每 rank 行数 1/2/4/8/12/16
的输出边界。三 rank 的 eager/graph 共 24 个条件通过动态输入输出检查，各采样
30 次、每次 32 个操作。CUDA Graph 中，原生 AllReduce 加本地切片相对
ReduceScatter 加速 1.38–1.94×，每次节省 7.88–13.77 μs；eager 模式更慢。
该组件测量排除了 E 并发、attention 和 norm 融合。

GPUq 86 随后完成同一源码、可选 attention-DP3 的主模型配对。仅将 D 的原生
`boundary_reduction` 从 `rs+rsv` 改为 `ar`，保留 round-robin、CPU 调度重叠、
P/D 无 TBO、请求上限和 graph buckets。四组各 256 个请求全部完成，实际各 rank
配置、输入/输出长度与自有进程清理均已核验。

| D 边界通信 | 到达率 (requests/s) | 联合达标请求 | TTFT P90 (ms) | TPOT P90 (ms) |
| --- | ---: | ---: | ---: | ---: |
| rs+rsv | 3 | 221 / 256 | 506.45 | 50.36 |
| ar | 3 | 117 / 256 | 2472.72 | 56.03 |
| rs+rsv | 3.75 | 143 / 256 | 2442.03 | 53.22 |
| ar | 3.75 | 147 / 256 | 2239.42 | 53.15 |

四组均低于固定的 90% 联合门槛。3 requests/s 下 `ar` 明显回退，组件加速尚未
转化为服务收益，性能配置继续采用 `rs+rsv`。两组 D 日志记录的 decode batch
均使用 CUDA Graph。此单次配对尚不能解释服务回退的具体机制，仍需请求时序和
完整 forward 的匹配诊断。该结果尚未达到完整 goodput 目标。

GPUq 81 的原生 TP8/EP8/attention-DP8 基线保留 FlashInfer BF16 A2A/runner、
FA4、DP LM head、CPU 调度重叠及原有容量。每个到达率各 256 个请求，实际八个
rank 配置、四组请求长度、固定 SLO 和自有进程清理均已核验。

| 原生基线策略 | 到达率 (requests/s) | 联合达标请求 | TTFT P90 (ms) | TPOT P90 (ms) |
| --- | ---: | ---: | ---: | ---: |
| round_robin | 2.5 | 213 / 256 | 571.77 | 58.38 |
| total_requests | 2.5 | 231 / 256 | 571.27 | 49.71 |
| round_robin | 3 | 206 / 256 | 548.25 | 57.75 |
| total_requests | 3 | 189 / 256 | 568.17 | 59.87 |

本次 `total_requests` 在 2.5 requests/s 达到 90% 联合门槛，3 requests/s 下
两种策略均未通过。此前 GPUq 5 的 `round_robin` 在 2.5 requests/s 达标数为
233/256，本次为 213/256，显示重复测试的必要性。已测最高可行到达率仍为
2.5 requests/s；最强基线容量、完整复现和约 1.5× goodput 尚待验收。

## 可追踪的实现路径

以下路径相对仓库根目录，按请求生命周期对应原文；函数名比随提交变化的行号更稳定。

1. **权重分离（§3）**：`python/sglang/srt/expertplex/integration.py::attention_weights_iterator`
   过滤 routed expert 权重；`worker.py::load_expert_weights` 按全局 expert ID 对 E rank
   取模分片，保留所有层。`worker.py::expert_main` 先校验 checkpoint 指纹与容量，创建
   `OwnedBuffers`、`PersistentExecutor`，再发布 IPC handles。随机权重比较另外使用
   `synthetic_weights.py` 的全局张量种子，不能把各分片独立初始化当成同一个模型。
2. **attention 发起通信（§5.2）**：原生 MoE 工厂选择 `layers/moe/ep_moe/expertplex.py`，
   由 `layers/moe/token_dispatcher/expertplex.py` 实现 `BaseDispatcher`，调用
   `expertplex/client.py::begin/finish`。Qwen3 forward 不直接调用通信 client；TBO
   使用原生 Qwen3 P/D operation 顺序和完整的 dispatch A/B、experts、combine A/B
   接口。`begin` 在 COMM stream 等待 COMPUTE 的路由结果，
   调用 `csrc/group_communication.cuh::RemoteGroup` 的 dispatch/combine 接口。
   当前性能实验采用 sender-packed P（`prefill_preprocess=False`）。attention 端的
   `group_count` / `group_pack` 计算每个专家的行号和 offsets，`group_dispatch_rows`
   直接写入 E 的 `b.input`。`attention_metadata()` 仅替换 counts/offsets 指针，
   保留 IPC 映射的 input/output 地址。`group_publish` 将最终元数据复制到 E，
   在各写入线程的 system fence 和 CTA 同步之后发布 READY。E 使用这些 offsets
   执行 GEMM。packed combine 从 E 的 `b.output` 拉取结果，先完成每个 EP rank
   的 top-k FP32 累加及 BF16 RN，再按 rank 顺序做 FP32 累加和最终 BF16 转换。
   raw P 路径（`prefill_preprocess=True`）由 `group_dispatch_raw` 写入固定的
   transport 接收张量 `raw_input` 及路由元数据，`group_publish` 在 payload fence 后
   发布 READY。APK 随后执行 Scatter1 计数、计算 expert offsets，再由 Scatter2
   将行写入 expert-packed GEMM 输入 `b.input`；Gather 生成 per-rank BF16
   `raw_output`，attention 再 pull 该结果。D 使用直接写 GEMM 输入的 routed rows 路径。
   WaitDone 在 attention GPU 上等待，`finish` 向 COMPUTE stream 加入完成 event。
   E 不启动配对的独立接收通信 kernel，也不使用中间 receiver-ring 的 drain/credit-return
   协议，但 APK 内仍有本地数据整理；mailbox 仍按 READY/DONE 轮次防止覆盖在途操作。
3. **长驻执行与传播（§4.2–4.3）**：`csrc/sm100_pipeline.cuh::executor` 的 CTA0 调用
   `update_policy`，经 system-scope READY、device-scope policy、cluster DSMEM Task、
   CTA 内 `task_full` mbarrier 传播。`claim` 是 cluster leader 的唯一 tile 领取入口；
   `initialize` 和 `commit` 推进 Scatter1 → Scatter2 → GateUp → ACT → Down → Gather。
   Task ring 的 metadata-consumed barrier 与 output-retired barrier 分离。phase 切换
   在流水模式下通过每 CTA 单调 64-bit 退役前缀等待旧输出，避免环形 barrier parity 的 ABA；
   这条实现路径和数值通过本身不能证明原文的“至多当前 tile”上界，
   尤其需要检查预取深度、传播延迟和所有 warp/CTA 的收敛。
   可选 `DecisionRecord` 单独记录 leader 获取策略前的时间、退休前缀、领取与交付序号；
   `bench_preemption.py::validate_decision_trace` 将其与 tile trace 关联，检查真实工作
   没有越过已作出的 phase 切换决策。这仍不覆盖此前 attention 发布到 CTA0 的传播。
   B300 双卡组件测试使用 `bench_preemption.py --max-decode-sm-fraction 0.99`，
   executor 报告 148 个 CTA、每 cluster 两个 CTA、decode 上限 146 个 CTA。
   在四次受控 decode 到达中，六阶段 `sm100` 路径的 74 个 cluster 都领取了 decode 工作。
   从决策到全部 warp 收到任务的实测上界最大为 32.064 µs，从发布起算的保守上界为
   73.056 µs；3604 次 CTA 切换中每个 CTA 至多有一个尚在计算的 tile。
   服务使用的 `sm100_swiglu` 路径产生了正确的组件输出，2920 次 CTA 切换也满足单 tile
   条件，但四次到达均缺少 cluster 73 的 decode 工作记录。该 cluster 位于为 prefill
   保留的范围；当前记录无法判断其中各 warp 收到策略的时刻。融合路径的全 cluster
   传播上界仍待验证。这组测试使用合成输入，不计入模型 goodput。
4. **SM 自适应（Eq.4）**：`update_policy` 从实际 decode GateUp row tiles 计算 footprint，
   按期望 footprint 缩放 q，向 cluster 倍数取整并限制 Qmax。每 phase 的 runnable 位图仅跳过空闲 mailbox；队列 CAS 仍按 tile 领取。`claim` 在首选 phase 无可领取
   工作时尝试另一 phase，实现空闲资源借用。CPU 对照是
   `optimizer.py::adaptive_sm_allocation`；CPU 公式对照不替代 GPU 调度轨迹验收。
5. **离线选择（Eq.1–3、Algorithm 1）**：`optimizer.py::search` 枚举单节点布局，调用
   `estimate_memory` 过滤容量，再为已有 profile 的互补 SM 预算二分可行 P/D batch。
   `expected_rank_footprint` 保留各 client/mailbox 的独立 tile padding，
   `_iteration_latency` 为 P/D 独立计算 sequential 或 TBO critical path；auto 枚举四种
   phase 组合，导出的原生服务配置保留策略和均衡 local batch 请求上限，不改变 mailbox
   容量。attention TP 宽度需有包含 collective 的独立实测 profile；缺少 DP/gather、
   SBO 的 profile 时仍拒绝搜索。当前没有专家内 TP、跨节点 placement 的完整搜索空间。
   CPU 算法/配置回归不等于模型预测误差或端到端 goodput 验收。

原论文的 SM90 FP8 tile 参数不能直接作为 SM103 BF16 的实现参数。当前 B300 后端使用
所固定 DeepGEMM BF16 原生几何的双 CTA/2SM MMA、每 CTA M128/N128/K64、9 个 TMA stage。
这是明确的硬件与精度适配；不能据此宣称复现了原论文 2.2–25.3 μs 数值区间。

## 固定实验条件

- 用户指定适配：单机 8×B300、Qwen3-Coder-480B-A35B-Instruct、BF16、默认 1P+3D+4E。
- 论文原实验是 H800、MiniMax-M2.7 / GLM-5.1-FP8；不能把模型、精度及硬件改变后的结果冒充原论文数值复现。
- 真实权重未完整时，随机权重只能验证计算路径、容量及合成负载；不能验收预训练模型精度或真实路由分布下的性能。
- 原生 WMMA 后端是通信与功能参考实现，**尚不满足论文 APK 的原生流水线要求**。

## 原生编译开关对照与启动恢复

2026-11-17，GPUq 123 的未编译组与 GPUq 128 的编译组完成独立三路 D
对照。两组使用相同的 Coder BF16、256 个固定请求、P/D overlap NONE 和
TTFT 1s / TPOT 50ms / 90% joint SLO。实际服务参数、输入输出长度及退出清理已核验。
编译组完成原生 Inductor 调优、CUDA graph 捕获和 PD warmup。

| Decode 配置 | 到达率 rps | joint 达标数 / 256 | P90 TTFT ms | P90 TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| 编译关闭 | 2 | 243 | 449.07 | 47.70 |
| 编译开启 | 2 | 236 | 470.42 | 48.54 |
| 编译关闭 | 3 | 52 | 839.81 | 64.81 |
| 编译开启 | 3 | 50 | 1213.48 | 63.42 |

两组最高已测通过点均为 2 rps。这次单次配对没有得到容量提升，编译保留为显式选项。
GPUq 123 的编译组先前因 600 秒冷启动期限退出，GPUq 128 仅重跑该组，启动期限
增至 2400 秒，复用已完成的未编译结果。请求 SLO 保持不变。

GPUq 129 的原生 attention-DP3 对照在正式请求前退出。D 服务初始化时，
Mooncake 的 P2P RPC 占用了端口 16000，随后路由器绑定同一端口失败。
GPUq 130 因前置对照未完成而退出，尚未加载模型。这两个任务没有性能结果。
辅助启动器现复用原生 socket 工具，在 E/P/D 初始化期间预留路由器端口，启动
路由器前释放。3 项真实 socket 回归覆盖初始化期间占用保护、交接、异常释放及
端口已被占用时提前失败，旧实现失败、修复后通过。原有 5 项启动器检查通过。
GPUq 134 已完成恢复后的 DP3 编译对照，两个服务组均正常结束且清理完成。固定
同一模型、数据、SLO、P/D NONE、D 全局容量 48、每副本 graph 上限 16、原生
`rs+rsv` 归约、round-robin 分配和 CPU scheduler overlap，仅改变 D 编译开关。

| DP3 Decode 配置 | 到达率 rps | joint 达标数 / 256 | P90 TTFT ms | P90 TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| 编译关闭 | 3 | 223 | 550.01 | 50.07 |
| 编译开启 | 3 | 222 | 568.38 | 50.25 |
| 编译关闭 | 3.75 | 161 | 2117.40 | 52.67 |
| 编译开启 | 3.75 | 148 | 2314.59 | 53.05 |

四组均未达到 90% joint SLO；本实验只测了 3/3.75 rps，不能据此将整体容量记为零。
普通 DP3 编译路径功能通过，未显示容量收益，默认关闭。准确公开准入/参数转发源码
通过 10 项编译与 24 项 DP CPU 检查；保留已有 CPU overlap 开关，不收紧原生选项。

GPUq 135 将 packed combine 专用实例用于同一 DP3 服务，分别关闭和开启 D 编译。
GPUq 142 复用 135 的未编译对照，仅将全局通信槽位从 4 减至 2。三个配置的 P/D
均使用 NONE，D 按 round-robin 分配。两次实验核验了服务实际参数和逐请求长度。

| packed combine 配置 | 到达率 rps | joint 达标数 / 256 | P90 TTFT ms | P90 TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| 4 槽，关闭编译 | 3 | 222 | 519.50 | 50.31 |
| 4 槽，关闭编译 | 3.75 | 169 | 1970.27 | 52.77 |
| 4 槽，开启编译 | 3 | 219 | 494.93 | 50.76 |
| 4 槽，开启编译 | 3.75 | 143 | 2293.28 | 53.39 |
| 2 槽，关闭编译 | 3 | 225 | 513.04 | 49.72 |
| 2 槽，关闭编译 | 3.75 | 164 | 1949.83 | 52.74 |

3 rps 的双槽位配置达到两项各自的 P90 阈值，但同时达标的只有 225 个请求，
尚缺 6 个才能达到 90% 联合门槛。3.75 rps 时，134 和 135 各自的编译开关对照加上 142 的双槽位实验，
共五组记录均有 33 个请求的 TTFT 超过 1 秒，因此保持实测 TTFT 时，联合达标数
最多为 223。五组的这 33 个请求在客户端返回数组中的下标都满足 `index % 3 == 2`
（下标从零开始）。请求所在的实际 GPU rank 仍需服务端关联记录确认。

DP3 的双批次重叠另行验证。GPUq 137 通过掩码、固定地址 graph replay、空行及
子批次偏移四项 GPU 检查。GPUq 138 的不重叠对照跑完，开启重叠时，D 在图捕获
期间重复写入原生子批次状态的 `topk_output`。候选修复先在局部变量中完成掩码，
再将结果写入状态一次。原生选择专家至 dispatch 的 CPU 回归在旧实现复现了断言，
修复后六种场景通过。该候选的主模型验证另行记录。

## 机制与代码审计

| 原文 | 必须保留的机制 | 当前证据或差距 | 验收方式 |
|---|---|---|---|
| §3 | P、D attention 独占 GPU；E 共享完整专家权重 | 原生独立 P/D/E 服务已接入；真实 Coder BF16 的确定性固定历史检查通过三路 D 并发全词表逐位对照，实际使用原生 KV 移交；仅限已验证轨迹与模式 | 继续扩大独立提示、序列长度和运行模式覆盖；保留失败记录 |
| §4.2、§7.1 | 一个长驻 kernel 融合两个 grouped GEMM、激活、数据处理、cluster scheduler | 默认 SM100 长驻融合 TMA/TCGEN05；当前性能明显落后于原生 DeepGEMM，不能验收调度开销 | 对照原生 BF16 GEMM，并记录融合前后开销 |
| §7.1、§7.6 Fig.15 | MoE 预处理/后处理也是 APK 中的 tile operation，包含 EP Scatter 两阶段与 EP Gather | `prefill_preprocess` 的 4E/top-8 数值、并发、graph 和完整随机权重服务通过；六种操作已有 tile interval 数据，但不构成完整传播/抢占上界证明 | 原生 BF16 per-rank gather 舍入边界对照；完整负载的各操作抢占间隔测试 |
| §4.2 | 启动前固定 operation templates、buffer、TMA descriptors；shape 在启动时编译为常量 | buffers / TMA descriptors 已固定；Coder/30B 的 H/I 已做编译期特化，小调试模型保留通用形状 | 多层、动态 batch、图 replay 的地址与描述符检查 |
| §4.2 | 保留原生 CTA/cluster shape、TMA multicast、warp specialization | 默认双 CTA、2SM MMA、9-stage TMA、原生 STSM/TMA epilogue 已通过回归；GateUp/ACT/Down 独立调度。Nsight Systems 确认对应 native GEMM 几何；性能仍落后 | 原生算子形状及流水线代码审计，数值测试 |
| §4.3 Fig.4 | system P → CTA0 → device p_i → cluster DSMEM → warp broadcast → mbarrier | 默认后端已实现该传播链；GPU 定时发布和 tile trace 已采集，尚未证明完整上界 | 可控时刻注入 urgent decode，记录各级 epoch/timestamp |
| §4.3 | 决策在 check epoch 内固定；观察切换后不得再领取旧 phase tile | leader 是唯一领取者并向所有成员分发同一 Task；CAS 竞争不会被当成 phase 空闲；trace validator 检查成员决策与阶段退役，但完整传播上界仍待验证 | 有预取、多 warp、多 CTA 的长时间压力测试 |
| §4.3 | 不在每个 tile 后 drain pipeline；无需 checkpoint、restore 或重算 | SM100 双 accumulator，延迟 store 退休与 Task ring 元数据复用已分离；跨 mailbox 回归通过，配对汇总仍慢约 0.73%。完整 SM100 后端默认开启，可显式关闭做消融 | pipeline 轨迹、吞吐及 phase 切换边界检查 |
| §5.2 | attention push 到最终 dispatch buffer；payload 可见后发布 ready | 当前性能配置的 sender-packed P 和 masked D 直接写入 E 的 GEMM 输入；P 的 counts/offsets 在 attention 端生成并在 READY 前发布，D 使用固定 expert stride。NVLink IPC 已验证全部 56 个有向 GPU 对。raw P 的本地 Scatter/Gather 使用另一路径 | 保持两条路径各自的 epoch、容量、错误与并发回归；补齐完整服务传播时延验收 |
| §5.2 | E 无独立通信接收 kernel、无预留 polling SM；attention pull 最终结果 | sender-packed P 从 E 的 GEMM 输出拉取，attention 端保留 per-rank BF16 gather 舍入。raw P 在 APK 内执行 Scatter1/Scatter2/Gather，attention 拉取其 BF16 输出。两条路径均使用 attention-side WaitDone | kernel trace、并发 P/D 验证；通信微基准的 identity echo 仅为计时外诊断 |
| §5.2 Fig.5 | 单线程 WaitDone 位于 COMM stream；COMPUTE stream 可继续 TBO/SBO | 独立 COMM stream 和双 mailbox 已实现；真实模型三路 D 已验证非空双 microbatch 图重放及所有 E 的 epoch 完成，Qwen2 SBO 已复用原生 shared 分支并通过辅助模型全词表对齐及实际 Decode 图节点重叠验证；prefill 图/shared TBO/标定仍未验收 | CUDA timeline 的计算/通信重叠量与 SBO 验证 |
| §5.3 | 跨节点 P 同 local rank 转发，节点内 NVLink multicast/gather；无 P 节点走低优先级 IB VL | 单节点 MoE 路径为 NVLink；多节点 MoE RDMA/层次转发未实现 | 需要多节点与 IB 硬件；单机测试不能替代 |
| §6.1–6.3 Eq.1–3 | 联合枚举 attention/MoE 并行、布局、overlap、q；内存过滤；SLO 二分批量；critical path 估计 | 当前优化器只覆盖部分布局与简化模型，需补齐可执行策略 | profile 拟合留出集误差、候选可执行性、预测/实测对比 |
| §6.4 Eq.4 | q 随当前/预期 decode tile footprint 缩放，按 cluster 向上取整，Qmax 保证 P 进展；单 phase 借用全部资源 | CPU 对照及默认 SM100 调度器均使用实际 masked/packed tile footprint；仍需完整策略质量与饥饿上界验收 | 稀疏度变化、空闲借用、P/D 饥饿测试 |

论文 §7.1、§7.6 本身包含 APK 内的 MoE 预处理/后处理和 EP Scatter/Gather，不能仅因
存在本地 scatter 就判定与论文不一致，也不能据此把两种“最终 buffer”视为已经等同。
2026-11-18 的源码核对确认，sender-packed 路径的 offset 由 attention 生成，payload
写入 APK GEMM 使用的同一个 IPC 映射地址。每个 mailbox 的单一 attention 写入者
由 bootstrap lease 管理，TBO 子批次使用不同 slot。GPUq 126 的公开源码指纹与
当前 native 源码一致，packed combine 的实际 grouped/per-peer IPC 及 FP32/最终
BF16 检查已通过。GPUq 128 的主模型服务配置也明确使用 sender-packed P。
这覆盖了该路径的缓冲区所有权和已记录的组件/服务验证。§7.1、§7.6 的 APK 内
Scatter/Gather 操作仍按 raw 路径独立验收，完整抢占传播、优化器标定及端到端
goodput 继续保留为未完成项。

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

## 早期证据与修复历史（不等同于当前完整验收）

以下按当时状态保留早期观察，不能覆盖上方当前验收边界；后续通过也不抹去早期
失败。实验原始记录单独保存，不随源码发布。

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
以及 CUDA Graph runner 也继续使用 SGLang。TBO 使用原生 Qwen3 OperationsStrategy / YieldOperation，
通过标准 MoE dispatcher 提交异步 dispatch / combine。启用开关与角色配置属于原生
ServerArgs；通信 client 由 Runtime 资源容器持有，并接入 scheduler 退出流程。
`sglang/srt/disaggregation/expertplex.py` 提供原生独立 MoE 入口，仅管理 E ranks；
P 和 D 各自运行原生 PD server，通过 `--expertplex-bootstrap-addr` 发现 MoE pool。
HTTP 仅用于启动、独占槽位登记与退出释放，每层 activation 仍走 CUDA IPC/NVLink。
布局数量由 E 的三个 count 参数或显式 topology 配置指定；当前不是在线 E 扩缩容。
`examples/expertplex/launch.py` 只作为可选实验编排，不是部署的必要入口。
`check_independent_services.py` 检查独立启动、并发 P/D、单个 D 的正常退出/重连与进程清理；
该生命周期检查不证明相对原生 SGLang 的数值对齐或性能提升。

| 路径 | 当前处理与边界 |
|---|---|
| CPU/GPU scheduler overlap | 已移除适配器原先的强制关闭约束；forward 提交串行，跨层及跨迭代的 COMM 操作由 stream/event 排序，由真实模型并发回归单独验证。同步捕获 CPU 请求历史的数值诊断显式关闭此优化，不能用于性能测试。 |
| Decode CUDA Graph | 使用现有 runner；启动器默认开启，`--no-cuda-graph` 仅用于消融/调试。 |
| Sampling | 普通基线、ExpertPlex 与实验总入口均默认使用 FlashInfer；PyTorch 仍可显式用于数值诊断。 |
| FlashInfer autotune | 删除无条件禁用参数；保留显式调试开关，普通启动遵循 SGLang 行为。 |
| KV / prefix cache | 沿用原生 allocator、radix cache 和显存容量 profiling；启动器默认不传 `--max-total-tokens`。§7.1 统一的是按 PDD 容量限制的采样序列长度，并非各布局的总 KV pool 容量。显式固定总 KV 的实验仅作为额外控制实验。 |
| Attention backend | 保留 SGLang backend 选择；Triton 的固定数值对照不能代表 FlashInfer/FA4 等路径的最佳性能。 |
| Prefill graph | 已接入原生 breakable prefill graph，D 保留原生 full decode graph。P graph 默认关闭，须使用 `overlap=none`；辅助入口拒绝与 TBO 同时启用，原生参数解析也会禁用该组合。应分别验证同模式数值对齐和实际 prefill 图执行，不能把捕获成功当作请求已经走图。跨 eager/graph 的逐位比较须另做确定性模式验证。 |
| torch.compile | TP1 decode 的公开入口保留 full graph 与 none/TBO，使用原生 bucket 对齐和编译调优，允许用户选择确定性及精度选项。普通参数集成通过 8 项 CPU 回归，GPUq 123/128 完成独立 1P+3D+4E、P/D none 的主模型编译开关对照，尚无容量收益。attention-DP3 扩展连同启动参数转发通过 10 项回归及已有 24 项 DP 检查，GPUq 134 完成普通 DP3 主模型编译开关对照，公开入口已接入，但 3/3.75 rps 均未达到联合 SLO 门槛。逐位比较继续用于定位问题，模型质量按当前口径集中评估。 |
| DeepGEMM | 复用原生 TMA、TCGEN05、双 CTA multicast、STSM/TMA epilogue 和 BF16 舍入边界；新增的是长驻 operation 选择与合作阶段切换。负收益的替代实现不能替换已验证路径。 |
| DeepEP | Decode combine 保持原生 low-latency 的 top-k 顺序 FP32 FMA 与最终 BF16 舍入；P 保持 normal 的每 rank BF16 gather 边界。普通 SGLang 基线继续调用原生 normal / low-latency dispatcher；ExpertPlex 的 attention-initiated M-to-N 不能直接套用要求 E 侧 receiver kernel 的原始 API。新的协议仍需继续对照 DeepEP 的路由布局、向量搬运和归并实现，不能据仅有功能通过宣称保留了全部性能优化。 |
| Chunked prefill / SBO / TP | chunked prefill 有显式选项；Qwen2 SBO 已通过辅助模型数值与 Decode 图重叠验证；prefill 图、shared TBO、共享分支标定及 attention TP>1 的完整验收仍待完成。默认整请求 P 是论文实验配置，不是 SGLang 能力被全局删除。 |

基线启动器也应保留原生 overlap 与 autotune；PDMux 继续遵循其独立调度循环的约束。
恢复这些默认值之后，旧的固定参数性能结果仍保留，但不能当成新配置的验收结果。


## 2026-11-09 本地开发历史记录

以下为当日硬件恢复前的记录。其后的 GPU 验证见本文件开头和 2026-11-17 更新。

当前 B300 不可用。本轮 CPU 回归共 108 项，107 项通过，实际 CUDA IPC 归并测试
1 项因缺少 GPU 跳过。已在原生参数生成链路修复 P/D 独立请求上限、显存比例和
exact Decode graph bank 的转发，并将显式全局 Decode KV cap 按 attention-DP
rank 分配。优化器能够搜索并导出 P=TBO、D=NONE；导出的请求上限遵循当前
compact P 的原生 admission 约束。

packed Prefill 的 attention-side combine 已补每个 EP rank 的 BF16 RN 边界，
位置在该 rank 的 top-k FP32 累加完成之后、跨 rank FP32 累加之前。CPU 算术反例
通过；实际 `group` / `per-peer` IPC、raw-versus-packed、CUDA Graph、并发和主模型
逐 token 全词表验证等待 B300。`test_packed_combine.py` 已包含实际 CUDA 回归入口。
默认 transport 选择沿用既有配置。

SM100 的 Eq.4 预算计算已在整数转换前饱和，并与 WMMA 共用 `sm_budget.h`。
主机 C++/UBSan 回归覆盖普通公式值与极值，实际 CUDA 编译及调度行为等待验证。
这两项 CUDA 改动保存在开发提交中，GPU 验收后再发布。

后续硬件验证仍需覆盖全部操作和参与 warp 的抢占传播、严格数值范围、布局模型的
实测预测误差，以及同数据、TTFT 1s / TPOT 50ms / 90% joint SLO 下的公平基线。
完整论文复现与约 1.5× goodput 尚未验收。
