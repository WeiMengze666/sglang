# ExpertPlex 单节点 B300 复现实验

这是基于论文机制独立编写的 SGLang 实验后端。主模型为
**Qwen/Qwen3-Coder-480B-A35B-Instruct，原始 BF16 safetensors**，调试模型为
**Qwen/Qwen3-30B-A3B，原始 BF16 safetensors**。默认布局是 **1P + 3D + 4E**。

已在 8×B300 编译并测试 CUDA IPC、原生 TMA/TCGEN05、双 CTA cluster、
连续 prefill 布局及 COMM stream 协议。小型 BF16 checkpoint 的 1P3D4E HTTP/KV
生成已打通，但严格 token 对齐未全部通过。Coder 完整真实权重、论文所有实验及
1.5× goodput **尚未验收**。逐项证据见 [论文对照](PAPER_CONFORMANCE.md) 和
[验证记录](VALIDATION.md)。

## 版本和范围

- 仓库：<https://github.com/WeiMengze666/sglang>
- 工作分支：`feat/expertplex-b300`
- 基线：`d8f852234850ac01a36adbb1c337a0acc0ff3912`，2026-10-13 的 `main`
- 论文：[ExpertPlex, arXiv:2607.18002v2](https://arxiv.org/html/2607.18002v2)
- 平台：Linux，单节点 8 张 NVIDIA B300 / SM103，完整 NVLink peer access 和 peer atomics
- 权重：本地、不可变、未量化的 Qwen3 MoE BF16 checkpoint
- attention 进程：每进程 TP=EP=PP=DP=1，独占一张 GPU
- expert 进程：每张 E 卡一个长期运行的 CUDA executor，P/D 共享一份专家权重

论文原实验使用 H800/SM90 与 FP8；本适配使用 B300/SM103 的 BF16 TMA/TCGEN05。
默认 `sm100` 后端采用双 CTA cluster、producer/MMA/epilogue warp 分工、4-stage TMA
与双 accumulator 流水线。CTA0 读取系统 ready word，cluster leader 读取 device policy，
通过 DSMEM 分发 tile 决策，mbarrier 向后续 warp 交接；共享输入使用 TMA multicast。
`sm100_cta1` 保留单 CTA 消融后端，`wmma` 保留功能参考。它们不能混作论文性能结果。

当前仍缺完整的原生流水线性能校准、抢占上界测量、跨栈策略覆盖与跨节点 RDMA MoE。
TBO 的双 mailbox client/graph 路径已通过；小 checkpoint 的 24×128-token 完整服务测试
已确认 decode 实际使用 CUDA Graph，尚无真实 Coder 权重下的吞吐/精度验收。
推测解码、LoRA、EPLB、CPU offload、量化、多节点、动态 P/D 切换均未支持。

## 权重与显存

[官方模型配置](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/config.json)
为 62 层、160 experts、top-8、hidden=6144、expert intermediate=2560。
[官方权重索引](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/model.safetensors.index.json)
列出的总权重大小为 960,309,750,784 bytes，专家命名为
`model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight`。

BF16 routed expert 权重按 `expert_id % E_count` 分片，每张 E 卡持有所有层的对应专家。
默认每卡 40 experts。attention 卡仅保留 embedding、attention、router、norm 和 lm_head。

| 默认配置的每卡项目 | GiB |
| --- | ---: |
| E 卡 routed expert 权重 | 217.969 |
| E 卡全部 mailbox 和中间缓冲区（连续 P / masked D） | 1.374 |
| E 卡计入 16 GiB 预留后的容量估算 | 235.342 |
| A 卡 attention/router/embedding/head 权重 | 22.483 |
| A 卡 65,536-token BF16 KV pool | 15.5 |

[B300 标称 288 GB](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)
约为 268.22 GiB，配置默认按 268 GiB 计算。上述是容量估算，不是显存实测。
worker 会使用 `cudaMemGetInfo` 对专家权重、固定 mailbox 和至少 8 GiB 运行时预留做
启动检查。规划器默认预留 16 GiB；GPU 测试仍需记录实际 allocated/reserved HBM。

buffer 不按层复制。P 为按专家连续打包的 routed rows，专家起点按 64 行对齐；
D 为 `[local_experts, max_decode_tokens, H/I]` masked 布局。每个 mailbox 各有
BF16 input、activation、output；TBO 分配两套。P 的最坏情况行数按
`max_prefill_tokens * min(top_k, local_experts) + 63 * local_experts` 向上对齐，
而非为所有专家预留最大 prefill 长度。`configure.py` 和优化器计算同一分配公式。
attention 直接写远端最终 buffer，没有额外完整本地 packing buffer。

235B/2E 也可以配置：235B 每张 E 卡约 211.5 GiB 专家权重，具体 P/D 比例应测量。
E 数控制计算和带宽容量，最少能装下权重的 E 数不必然获得最高 goodput。

## 数据路径与代码位置

```text
SGLang P attention GPU 0 ── dispatch / pull ─┐
SGLang D attention GPU 1 ── dispatch / pull ─┤
SGLang D attention GPU 2 ── dispatch / pull ─┼─ shared E GPUs 4,5,6,7
SGLang D attention GPU 3 ── dispatch / pull ─┘
           P ── existing SGLang Mooncake KV transfer ── D
```

1. `Qwen3MoeSparseMoeBlock` 保留 router，选择标准 top-k ID/weight；不构造本地
   `FusedMoE` 权重。loader 在 `safe_open.get_tensor()` 之前过滤 routed expert 名称。
2. attention 侧 CUDA dispatch 使用 NVLink peer stores 将 token 直接写入 E 卡最终输入
   buffer，生成本地 row map，然后以 system-scope release 发布 ready epoch。
3. E 卡长期运行的 kernel 通过 generation-tagged queue 认领 tile，融合 gate/up GEMM、
   SiLU 和 down GEMM。stage 之间使用完成计数器，最后一个 tile 发布下一 stage。
4. 每个 CTA 在完整 tile 提交后重新选择阶段。竞争时按论文 Eq.4 的当前 decode
   `sum(ceil(expert_rows/32))` 相对离线期望调整 decode CTA 预算；上限保留 prefill
   进展。一个阶段没有可认领任务时，其他阶段可以借用空闲 CTA。动态 shared memory
   限制每个 SM 只驻留一个 executor CTA，CTA 预算才具有 SM 预算含义。
5. E 卡发布 done epoch；attention 使用 system-scope acquire 等待并拉取最终专家输出，
   乘 routing weight，以 FP32 累加后转 BF16。E 卡不运行 receiver dispatch/combine kernel。
6. KV handoff 使用本仓库原有 SGLang PD/Mooncake 协议和 router，保留请求、KV 和取消
   处理。论文未给出独立的 KV 迁移协议，不在这里另造一个协议。

主要文件：

- `python/sglang/srt/expertplex/csrc/expertplex.cu`：IPC、GPU queue、tile 调度与 BF16 GEMM
- `python/sglang/srt/expertplex/native.py`：惰性 JIT 编译和显式 buffer 生命周期
- `python/sglang/srt/expertplex/{worker,client,rendezvous}.py`：加载、通信与启动控制面
- `python/sglang/srt/expertplex/{config,optimizer,integration}.py`：配置、Eq.1–4 与接入校验
- `python/sglang/srt/models/qwen3_moe.py`、`model_loader/loader.py`：attention-only 接入
- `examples/expertplex/`：配置、启动、验证、基准工具

每个 attention client 独占一个 mailbox，TBO 使用两个；每个 mailbox 只允许一个未完成调用。运行 ID、配置
digest、checkpoint 配置及 shard size/mtime fingerprint、单 client 所有权检查防止误用
旧 IPC 句柄。fingerprint 不是对全部权重内容的密码学哈希；运行中必须保持 checkpoint
不变。超时或协议错误后该 client 被标记失败，不覆盖或复用未完成 buffer；重启整个
部署并使用新的 rendezvous 目录。launcher 只清理自己启动的进程组，先退出 attention，
再释放 expert allocations。

## 安装与生成配置

在 B300 Linux 环境使用本分支配套依赖，遵循仓库 Dockerfile/`python/pyproject.toml`。
此基线锁定 PyTorch 2.14.1 和 CUDA 13 相关依赖。原生扩展需要可用的 `nvcc`、
C++ 编译器、`ninja`；JIT 显式编译 `sm_103a`，并使用 `sgl-deep-gemm 0.2.2` 的官方头文件。不要把旧版 SGLang wheel 与本分支源码
混用。KV handoff 需要 Mooncake；router 需要当前仓库配套 `sglang_router`。

```bash
python -m pip install -e ./python
# 按仓库 sgl-model-gateway 的安装说明安装/构建同版本 router。
python -m sglang_router.launch_router --help
nvcc --version
nvidia-smi topo -m
```

将官方 BF16 checkpoint 下载到本地目录，并记录 Hugging Face revision。不要使用 FP8、
GPTQ、AWQ 或 ModelOpt 权重代替。原始 480B 权重约需 960 GB 磁盘空间，另留缓存空间。

CPU 上也能检查配置和启动命令：

```bash
python examples/expertplex/configure.py \
  --model-path /data/models/Qwen3-Coder-480B-A35B-Instruct \
  --prefill 1 --decode 3 --experts 4 \
  --rendezvous-dir /tmp/expertplex-coder480b-run001 \
  --output /tmp/expertplex-coder480b.json

python examples/expertplex/launch.py \
  --config /tmp/expertplex-coder480b.json --dry-run
```

若模型目录已存在，configure 会校验真实 `config.json`、打印权重/mailbox 估算，并拒绝
明显超出容量的配置。`decode_sm_budget=32` 和 `expected_decode_tiles=128` 是初始调度
参数，**没有做过 B300 校准，不能解释为优化器已找到最优配置**。

## GPU 验收顺序

先执行纯 CPU 检查：

```bash
python examples/expertplex/check_cpu.py
```

再在两张空闲 B300 上编译和验证原生执行器：

```bash
python examples/expertplex/check_gpu.py \
  --attention-device 0 --expert-device 1 \
  --full-mesh --output /tmp/expertplex-native.json
```

它通过独立 spawned 进程导出 IPC allocation，检查两层权重、多次 epoch、极端路由、
inactive experts、P/D 同时进展、empty-input 分支和独立 allocation 的有界超时。
`--full-mesh` 检查 8 卡间全部 56 个有向 peer/atomic/IPC 映射。默认容差为
`atol=0.01, rtol=0.03, relative_l2<=0.03`，测试报告保存每项误差；这些是初始测试阈值，
不能替代真实模型精度验收。gate/up 的 FP32 accumulator 先舍入到 BF16，再执行 SiLU/mul，保持原生 BF16
GEMM 输出边界；down 也先舍入到 BF16，再进行 FP32 weighted combine。

随后先用 Qwen3-30B-A3B 跑完整服务，再换 480B。每次重启换一个 rendezvous 目录。
所有进程必须使用完全相同的 CUDA device ordering；默认不设置 `CUDA_VISIBLE_DEVICES`，
或统一设为 `0,1,2,3,4,5,6,7`，由 `--base-gpu-id` 选择 attention 卡。

```bash
python examples/expertplex/launch.py \
  --config /tmp/expertplex-coder480b.json \
  --kv-tokens 65536 --log-dir /tmp/expertplex-coder480b-logs
```

服务默认仅监听本机，router 为 `http://127.0.0.1:31000`。单节点 KV 默认使用 Mooncake
`INTRA_NODE_NVLINK`；显式传 `--kv-transport rdma --ib-device ...` 才选择 RDMA。
MoE 数据走 CUDA IPC/NVLink。P 默认关闭 chunking，并向请求验证器发布硬输入长度上限；
`--chunked-prefill` 仅用于消融。TBO 配置可同时传 `--cuda-graph` 启用 decode 图回放。

检查完整模型功能与串行 baseline 对照：

```bash
python examples/expertplex/check_e2e.py \
  --model-path /data/models/Qwen3-Coder-480B-A35B-Instruct \
  --output /tmp/expertplex-e2e.json
```

`check_e2e.py` 检查并发、多轮生成、有限 logprob 和输出长度，保存 top-5/logprob 信息。
先在普通 SGLang BF16 baseline 上记录 reference，再关闭 baseline、运行 ExpertPlex，
使用 `--reference /tmp/reference.json` 比较逐 token 共同前缀。可用 `--require-exact-tokens`
将不一致设为失败。数值近似与 token exact 是不同验收项；还需要 coder 任务数据集的
准确率，以及长 prompt、取消、worker 故障和长时间运行测试。脚本没有自动宣称这些通过。

## Goodput 测量

使用相同 checkpoint、输入/输出分布、随机种子、SLO、KV 容量上限和 8 卡总预算，串行
比较 ordinary SGLang colocated、chunked prefill、instance-level PDD 与本实现。480B BF16
的 PDD 容量要先单独审查；每个 phase 完整复制约 960 GB 权重，4+4 卡通常只剩有限 KV
空间。不能因为某 baseline OOM 就用不同请求长度悄悄比较。

`bench_sweep.py` 调用本仓库 `sglang.benchmark.serving` 的 Poisson arrival generator，
保留逐请求 `successes/ttfts/latencies/output_lens`，计算同一个请求同时满足 TTFT、TPOT
的比例，失败请求保留在分母中。分别比较两个 P90 百分位不等价于联合 90% attainment。
报告最高**已测试**且联合 attainment≥90% 的 offered rate，以及观测 SLO goodput。

以下 SLO 仅为 480B 初始实验参数，**并非论文 MiniMax/GLM 的同模型复现 SLO**：

```bash
python examples/expertplex/bench_sweep.py \
  --model-path /data/models/Qwen3-Coder-480B-A35B-Instruct \
  --label expertplex-coder480b-1p3d4e \
  --rates 0.25 0.5 1 2 --num-prompts 1000 \
  --input-len 1024 --output-len 256 \
  --ttft-slo-ms 2000 --tpot-slo-ms 100 \
  --output-dir /tmp/expertplex-goodput-seed42
```

ShareGPT 使用 `--dataset-name sharegpt --dataset-path /path/to/sharegpt.json`。
长文本可使用同一份 `custom` 数据集，按本仓库 custom dataset 格式准备 LooGLE 样本；
所有方法使用相同截断规则。正式结果需要足够样本、多个种子、SLO 边界附近更密集的
rate sweep，并记录 GPU、驱动、CUDA、commit、模型 revision、配置、功耗和 HBM。
原生 smoke latency 或拟合模型预测不等于端到端 goodput。

## 跨栈规划器

`optimizer.py` 实现 Eq.3 tile footprint、Eq.4 在线预算的 CPU 对照、容量过滤、
Eq.2 拟合和 Algorithm 1 的 batch/SLO 搜索。它采用实际 modulo expert placement 和独立
client mailbox 的 tile 数，不假设不同 attention client 的零散 tile 可以合并。
概率 profile 使用 IID Binomial 的期望整数 tile 数，避免对每个非零激活概率直接向上
取整而错误地把小 decode batch 的所有专家都视为已激活；实际路由相关性仍需实测校准。

```bash
python -m sglang.srt.expertplex.optimizer \
  --model-path /data/models/Qwen3-Coder-480B-A35B-Instruct \
  --prefill-count 1 --decode-count 3 --expert-count 4 \
  --kv-tokens-per-attention-gpu 65536
```

不提供 `--profiles` 时仅估算容量，不产生任何 latency/goodput 数字。提供 profile 时，
schema 要求 `units="seconds_per_layer"`、非空 `provenance`、真实 `total_sms`、
`tile_rows={"prefill":64,"decode":32}`、实测 `nvlink_bytes_per_second`，以及 prefill/decode 各自的
`attention` 系数 `{alpha,beta,gamma,delta}`、`moe_by_sm_count` 和逐专家激活概率
`expert_probabilities`（总和为 top-k）。`fit_coefficients` 接受实测
`{x,sequence_length,latency_s}` 样本；MoE 的 x 必须是 tile footprint。只有实际测量过的
SM 预算进入搜索。测试里的人工系数仅验证代数，不作为 B300 profiles 发布。

当前关键路径模型是近似，未计入完整排队、KV 迁移和服务 CPU 开销。`--overlap tbo`
需要实测 TBO profiles；client 路径支持该选项，完整服务验收仍待完成。
profile 搜索结果要在 B300 上用同配置的端到端 sweep 重新验收。
