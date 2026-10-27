# ExpertPlex 单节点 B300 复现实验

这是集成在 SGLang Runtime 内的实验性 ExpertPlex 后端，通过原生
`--enable-expertplex` 启用，默认关闭。主模型为
**Qwen/Qwen3-Coder-480B-A35B-Instruct，原始 BF16 safetensors**，调试模型为
**Qwen/Qwen3-30B-A3B，原始 BF16 safetensors**。默认布局是 **1P + 3D + 4E**。

代码包含 CUDA IPC、原生 TMA/TCGEN05、双 CTA cluster、连续 prefill 布局及
COMM stream 协议。功能、有限轨迹数值检查和性能验收需分别判断；论文完整复现及
1.5× goodput **尚未验收**。历史实验记录中的权重、环境和结果应按各自日期解释，
不能代替当前配置的复验。机制清单见 [论文对照](PAPER_CONFORMANCE.md)。

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
默认 `sm100` 后端采用双 CTA cluster、producer/MMA/epilogue warp 分工、9-stage TMA
与双 accumulator 流水线；GateUp、激活、Down 是独立调度的 operation templates，
GEMM 直接复用原生 DeepGEMM 的 STSM/TMA epilogue。
`configure.py` 默认启用 `--prefill-preprocess`，将 normal 模式的两阶段 scatter 和 BF16 gather 也纳入同一个长驻调度器；
已通过 Coder 专家形状的 4E/top-8 算子和小模型完整服务测试；`--no-prefill-preprocess` 保留 A 侧重排消融。CTA0 读取系统 ready word，cluster leader 读取 device policy，
通过 DSMEM 分发 tile 决策，mbarrier 向后续 warp 交接；共享输入使用 TMA multicast。
`sm100_cta1` 保留单 CTA 消融后端，`sm100_fused` 保留早期 4-stage 融合 gate/up/激活后端，
`wmma` 保留功能参考。它们不能混作论文性能结果。

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
| E 卡全部 mailbox 和中间缓冲区（默认单 mailbox，P2048 / D256，E 侧预处理） | 1.989 |
| E 卡计入 16 GiB 预留后的容量估算 | 235.958 |
| A 卡 attention/router/embedding/head 权重 | 22.483 |
| A 卡 65,536-token BF16 KV pool | 15.5 |

[B300 标称 288 GB](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)
约为 268.22 GiB，配置默认按 268 GiB 计算。上述是容量估算，不是显存实测。
worker 会使用 `cudaMemGetInfo` 对专家权重、固定 mailbox 和至少 8 GiB 运行时预留做
启动检查。规划器默认预留 16 GiB；GPU 测试仍需记录实际 allocated/reserved HBM。

buffer 不按层复制。P 为按专家连续打包的 routed rows，专家起点按 128 行对齐；
D 为 `[local_experts, max_decode_tokens, H/I]` masked 布局。每个 mailbox 各有
BF16 input、gate/up、activation、output；TBO 分配两套。P 的最坏情况行数按
`max_prefill_tokens * min(top_k, local_experts) + 127 * local_experts` 向上对齐，
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

1. `Qwen3MoeSparseMoeBlock` 保留原生 forward、router 和 TopK。MoE 工厂在开启
   ExpertPlex 时选择 `ExpertPlexMoE`，通过 `BaseDispatcher` 接口提交远端计算，
   保留 dispatch/combine hooks。attention 进程不分配本地专家参数；loader 在
   `safe_open.get_tensor()` 之前过滤 routed expert 名称。TBO 复用 Qwen3 原生 P/D
   operation 顺序，包括 dispatch A/B、experts、combine A/B；双 mailbox 对应原生子批次。
2. attention 侧 CUDA dispatch 使用 NVLink peer stores 将 token 直接写入 E 卡最终输入
   buffer，生成本地 row map，然后以 system-scope release 发布 ready epoch。
3. E 卡长期运行的 kernel 通过 generation-tagged queue 认领 tile，融合 gate/up GEMM、
   SiLU 和 down GEMM。stage 之间使用完成计数器，最后一个 tile 发布下一 stage。
4. 每个 CTA 在完整 tile 提交后重新选择阶段。竞争时按论文 Eq.4 的当前 decode
   `sum(ceil(expert_rows/128))` 相对离线期望调整 decode CTA 预算；上限保留 prefill
   进展。一个阶段没有可认领任务时，其他阶段可以借用空闲 CTA。动态 shared memory
   限制每个 SM 只驻留一个 executor CTA，CTA 预算才具有 SM 预算含义。
5. E 卡发布 done epoch；attention 使用 system-scope acquire 等待并拉取最终专家输出，
   乘 routing weight，以 FP32 累加后转 BF16。E 卡不运行 receiver dispatch/combine kernel。
6. KV handoff 使用本仓库原有 SGLang PD/Mooncake 协议和 router，保留请求、KV 和取消
   处理。论文未给出独立的 KV 迁移协议，不在这里另造一个协议。

主要文件：

- `python/sglang/srt/expertplex/csrc/expertplex.cu`：IPC、GPU queue、tile 调度与 BF16 GEMM
- `python/sglang/srt/expertplex/native.py`：惰性 JIT 编译和显式 buffer 生命周期
- `python/sglang/srt/disaggregation/expertplex.py`：原生独立 MoE 服务入口
- `python/sglang/srt/expertplex/{bootstrap,worker,client,rendezvous}.py`：服务发现、登记、加载与通信
- `python/sglang/srt/expertplex/{config,optimizer,integration}.py`：配置、Eq.1–4 与接入校验
- `python/sglang/srt/arg_groups/fields/disagg.py`：原生 ServerArgs / Runtime 配置
- `python/sglang/srt/layers/moe/ep_moe/expertplex.py`：由原生 MoE 工厂选择的远端专家层
- `python/sglang/srt/layers/moe/token_dispatcher/expertplex.py`：原生 dispatcher / TBO 接口
- `python/sglang/srt/models/qwen3_moe.py`、`model_loader/loader.py`：原生模型及分离权重加载
- `examples/expertplex/`：配置、进程编排、验证、基准工具，不包含另一套 serving engine

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

可先只读检查下载的 shard header、BF16 dtype 和文件长度；缺少索引、分片或 payload 时返回非零：

```bash
python examples/expertplex/check_checkpoint.py \
  --model-path /data/models/Qwen3-Coder-480B-A35B-Instruct \
  --output /tmp/expertplex-checkpoint-readiness.json
```

此检查不计算全部权重内容的哈希，部署期间仍须保持文件不变。

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

## 原生 SGLang 独立服务入口

Prefill attention、Decode attention 和 MoE pool 分别通过原生 `sglang.launch_server`
启动。MoE 仅启动自己的 expert ranks，不启动 P、D 或 router。P/D 使用
`--expertplex-bootstrap-addr` 连接 MoE，按原生 `--base-gpu-id` 和
`--disaggregation-mode` 选择对应槽位，不再需要传 client ID 或共享 JSON 文件。

以下是同一台 8×B300 上的 1P+3D+4E 示例。所有进程需使用相同的完整
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。在共享 gpuq 环境，须在**一个队列作业**中
启动这些独立进程、完成请求测试并清理所有子进程，不能把长期运行的三个角色分别排队，
否则第一项占住队列后其余角色无法启动。`gpuq` 约束不改变服务自身独立启动的接口。

共同环境（各服务均须继承）：

```bash
export MODEL=/data/models/Qwen3-Coder-480B-A35B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
export MC_INTRANODE_NVLINK=true
```

MoE 服务：三个 count 分别是 P GPU 数、D GPU 数、E GPU 数；默认是 1/3/4。
默认按 P→D→E 连续放置，合计可以小于八卡。这里 E ranks 使用 GPU 4–7。

```bash
python -m sglang.launch_server --model-path "$MODEL" --dtype bfloat16 \
  --enable-expertplex --expertplex-role expert \
  --expertplex-prefill-count 1 --expertplex-decode-count 3 --expertplex-expert-count 4 \
  --host 127.0.0.1 --port 31010 \
  --max-prefill-tokens 24576 --max-running-requests 256 --enable-two-batch-overlap
```

MoE 服务可直接通过 `--expertplex-decode-sm-budget`、
`--expertplex-expected-decode-tiles` 和 `--expertplex-max-decode-sm-fraction`
设置 Eq.4 的预期 decode 预算、每张 E 的预期 row-tile 数与竞争时占比上限。
不传 JSON 时默认分别为 `96`、`8`、`0.875`；传 JSON 时默认沿用文件中的值，
显式 CLI 参数覆盖对应字段。实际配置由 bootstrap 发布给 P/D，不能在 attention
服务上重复设置。这些是可调参数，不表示该组合已经达到最优 goodput。

Prefill attention（GPU 0）：

```bash
python -m sglang.launch_server --model-path "$MODEL" --dtype bfloat16 \
  --enable-expertplex --expertplex-bootstrap-addr http://127.0.0.1:31010 \
  --disaggregation-mode prefill --base-gpu-id 0 --port 30000 --nccl-port 20000 \
  --disaggregation-bootstrap-port 19000 --context-length 24576 \
  --max-prefill-tokens 24576 --max-running-requests 256 --chunked-prefill-size -1 \
  --mem-fraction-static 0.9 --enable-two-batch-overlap \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
  --attention-backend flashinfer --sampling-backend flashinfer
```

Decode attention：以下是 GPU 1 实例；另外两个实例分别使用
`base-gpu-id/port/nccl-port/bootstrap-port` = `2/30002/20002/19002`、
`3/30003/20003/19003`，各自执行一条启动命令。

```bash
python -m sglang.launch_server --model-path "$MODEL" --dtype bfloat16 \
  --enable-expertplex --expertplex-bootstrap-addr http://127.0.0.1:31010 \
  --disaggregation-mode decode --base-gpu-id 1 --port 30001 --nccl-port 20001 \
  --disaggregation-bootstrap-port 19001 --context-length 24576 \
  --max-prefill-tokens 256 --max-running-requests 256 --chunked-prefill-size 256 \
  --mem-fraction-static 0.9 --enable-two-batch-overlap \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 32 --attention-backend flashinfer --sampling-backend flashinfer
```

最后按原生 PD 方式启动 router，连接上述 P/D HTTP 地址：

```bash
python -m sglang_router.launch_router --pd-disaggregation \
  --host 127.0.0.1 --port 31000 \
  --prefill http://127.0.0.1:30000 19000 \
  --decode http://127.0.0.1:30001 \
  --decode http://127.0.0.1:30002 \
  --decode http://127.0.0.1:30003 \
  --prefill-policy round_robin --decode-policy power_of_two
```

MoE bootstrap 地址不能作为生成请求的 worker 地址。`GET /health` 和 `/get_server_info` 可检查 MoE readiness、
服务代次和已登记的客户端。P/D 可以独立先后接入，不要求所有客户端同时启动。

正常关闭 attention 时，scheduler 的资源清理路径会同步 GPU、关闭 IPC 映射并释放
登记槽位；其余 attention 和 E 服务继续运行。异常退出的槽位保持隔离，避免复用仍有
在途写入的 mailbox。E 重启后的旧客户端必须重启，不能继续使用旧代次的 IPC handles。
停止部署时先排空/退出 P/D，再停止 MoE。预留槽位的 phase/容量与 E 数量在 E 启动时
确定；修改该布局需要重启 E，当前不宣称支持在线扩缩 expert ranks。

需要非连续 GPU 放置或更多 APK 参数时，可给 MoE 使用 `--expertplex-config`。
P/D 仍可通过地址发现其配置。兼容的旧文件式配置需每次使用新的 rendezvous 目录。
`examples/expertplex/launch.py` 保留为实验编排辅助工具，内部也使用独立 bootstrap；
它不是运行 ExpertPlex 的必要入口。

KV pool 默认沿用 SGLang 显存 profiling；以上命令不限制总 KV token 数。固定容量
仅用于显式的控制实验。原生 scheduler overlap、decode CUDA Graph、FlashInfer
attention/sampling/autotune 和 Mooncake KV 移交继续使用。缓存行为遵循原生 PD 默认值
（例如 decode radix cache 默认关闭），可以在已支持的原生选项范围内调整。
尚未适配的 prefill graph、`torch.compile`、SBO、attention TP>1 等组合仍会明确拒绝。

控制面位于 `sglang/srt/disaggregation/expertplex.py` 和
`sglang/srt/expertplex/bootstrap.py`；APK、MoE client/worker 等计算后端保留在
`sglang/srt/expertplex`。详细边界见
[优化保留审计](PAPER_CONFORMANCE.md#对现有-sglang--deepep-优化的保留审计)。

## GPU 验收顺序

共享机器上的以下 GPU 命令均须放进 `gpuq` 作业，作业退出前清理本作业的所有服务。
独立服务生命周期检查可使用小型 BF16 checkpoint：

```bash
python examples/expertplex/check_independent_services.py \
  --model-path /path/to/tiny-bf16-checkpoint \
  --output-dir /path/to/new-lifecycle-run
```

它检查 D 先于 P 启动、三个 D 同时请求、单独退出/重新接入 D、重启前后 token/logprob
一致性及自有进程清理。`--kv-tokens auto --mem-fraction-static 0.9` 可验证原生 KV sizing；
较大的 checkpoint 可设置 `--startup-timeout 3600`。这项检查不替代同权重原生基线数值与性能对比。

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
  --kv-tokens auto --log-dir /tmp/expertplex-coder480b-logs
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

使用相同 checkpoint、输入/输出分布、随机种子、SLO、采样序列长度上限和 8 卡总预算，串行
比较 ordinary SGLang colocated、chunked prefill、instance-level PDD 与本实现。480B BF16
的 PDD 容量要先单独审查；每个 phase 完整复制约 960 GB 权重，4+4 卡通常只剩有限 KV
空间。不能因为某 baseline OOM 就用不同请求长度悄悄比较。

`bench_sweep.py` 调用本仓库 `sglang.benchmark.serving` 的 Poisson arrival generator，
保留逐请求 `successes/ttfts/latencies/output_lens`，计算同一个请求同时满足 TTFT、TPOT
的比例，失败请求保留在分母中。分别比较两个 P90 百分位不等价于联合 90% attainment。
报告最高**已测试**且联合 attainment≥90% 的 offered rate，以及观测 SLO goodput。

本次 ShareGPT 对比在测量前固定 TTFT=1s、TPOT=50ms；LooGLE 固定为 10s/100ms。
这沿用论文的阈值，但模型和 BF16/B300 平台已经改变，仍须标为适配实验。
所有方法都使用过滤后的同一数据文件，不能在看到结果后调整阈值：

```bash
python examples/expertplex/bench_sweep.py \
  --model-path /data/models/Qwen3-Coder-480B-A35B-Instruct \
  --label expertplex-coder480b-1p3d4e \
  --rates 0.75 1 1.25 1.5 --num-prompts 256 \
  --dataset-name custom --dataset-path /path/to/sharegpt-holdout-256.jsonl \
  --ttft-slo-ms 1000 --tpot-slo-ms 50 \
  --output-dir /tmp/expertplex-goodput-seed42
```

ShareGPT 使用 `--dataset-name sharegpt --dataset-path /path/to/sharegpt.json`。
长文本可使用同一份 `custom` 数据集，按本仓库 custom dataset 格式准备 LooGLE 样本；
所有方法使用相同的长度过滤规则，不截断已有样本。正式结果需要足够样本、多个种子、SLO 边界附近更密集的
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
`tile_rows={"prefill":128,"decode":128}`、实测 `nvlink_bytes_per_second`，以及 prefill/decode 各自的
`attention` 系数 `{alpha,beta,gamma,delta}`、`moe_by_sm_count` 和逐专家激活概率
`expert_probabilities`（总和为 top-k）。`fit_coefficients` 接受实测
`{x,sequence_length,latency_s}` 样本；MoE 的 x 必须是 tile footprint。只有实际测量过的
SM 预算进入搜索。测试里的人工系数仅验证代数，不作为 B300 profiles 发布。

profile 还必须声明 `prefill_preprocess`，并与运行配置一致（旧 schema 缺省为 false）。
启用 E 侧预处理时，通信量包含发往每张 E 的原始 BF16 hidden、top-k ID/weight 和返回的
BF16 partial output；不能继续套用 A 侧展开 top-k 后的 routed-row 通信量。
`profile_operations.py` 的 GEMM/ACT 数据没有包含 E 侧 Scatter/Gather，不能直接当作
`prefill_preprocess=true` 的完整 MoE profile；完整 critical path 仍需独立测量。

`profile_full_moe.py` 为 raw-prefill 路径测量完整 E 层，包含 Scatter1/Scatter2、
GateUp、ACT、Down 和 Gather。它复用 `bench_preemption.py` 的 GPU 发布与独立
attention/E 进程，但使用 `--trace-capacity 0` 关闭 trace，以 E 的 admission 到
retirement 计时；传输、attention、KV 迁移和服务排队不在该区间内。例如在一个
GPUQ 作业内运行：

```bash
python examples/expertplex/profile_full_moe.py \
  --sms 32 64 84 116 --prefill-rows 128 512 2048 8192 \
  --decode-rows 1 8 32 128 --iterations 5 --output-dir /path/to/new-moe-profile
```

结果保留实际 per-expert 行数、整数 tile footprint、重复测量及拟合残差。
这是单 E rank 的合成 BF16 权重和随机 top-k 负载，不能当作主模型的实测路由概率。
拟合包含完整 E 层操作仍不代表端到端模型已校准；需再检查留出点和并发服务误差。
可同时传入 `--holdout-prefill-rows 64 256 1024 4096 16384` 和
`--holdout-decode-rows 2 16 64 256`。训练与留出 token 数必须分别不相交；
工具先保存训练系数，再采集留出点，不使用留出数据重新拟合。报告分别列出
非正预测、相对误差和超出训练 footprint 范围的样本，并拒绝混用不同运行时代码。

当前关键路径模型是近似，未计入完整排队、KV 迁移和服务 CPU 开销。`--overlap auto`
同时比较 sequential/TBO，按候选策略计算双 mailbox 内存及独立 microbatch padding。
TBO 假设 token 均衡切分，实际序列边界会影响预测，需要实测校准。
`--write-config` 在有 profile 时导出搜索胜出的布局、overlap、q 和预期 decode footprint；
预测 batch 仍在 JSON 中单独报告，容量上限不等同于服务调度器的实际 batch。
当前并行策略仍限于 TP1 attention + EP 专家，未覆盖论文的全部跨栈候选。
profile 搜索结果要在 B300 上用同配置的端到端 sweep 重新验收。


## 实测检查点（2026-10-15，仍未验收）

完整 480B **随机权重**，ShareGPT 64 请求、1P3D4E、相同 24576 长度上限，
固定 TTFT=1s / TPOT=50ms，第一轮 1 req/s 的 joint attainment：
共置基线 61/64，ExpertPlex 45/64。**审计发现这些早期布局使用不同的随机初始化方式，
只能作为诊断，不能作为速度比验收。** 当前没有达到 1.5×；保留全部原始结果。
新增 `--canonical-synthetic-seed 1234` 在完整 HF 张量上统一初始化，再由正常 loader 分片；
E worker 使用相同的全局专家名称。公平对照的 64 请求 pilot 已完成四个基线：
1 req/s 时共置 54/64、Chunk2048 27/64、PDD 58/64、PDMux 初始预算 12/64、
ExpertPlex 61/64。PDD 和 ExpertPlex 都通过同一已测到达率，不能据此声称 1.5×。
PDMux 推进预算及更多边界点仍在调优，这些 pilot 不代替真实权重或独立留出集验收。
这一服务测试配置使用 P24576 / D256、双 mailbox，包含 raw hidden/路由 metadata 的固定 buffer 估算为 **18.420 GiB/E**，
不能套用上方 P2048 单 mailbox 的容量数字。

`run_serving_experiment.py` 会记录启动参数、环境、完整逐请求结果，并自动生成独立的
rendezvous。`bench_apk_sharing.py` 对 GateUp/Down 分别执行 §7.3 对照；
`profile_operations.py` 扫 §7.4 的 per-expert rows / active experts；
`bench_sharing.py` 与 `bench_apk_sharing.py` 的 `--decode-expert-rows` 可指定活跃
decode 专家的行数，配合 `--decode-active-experts` 检查部分 tile 和多 tile 的边界。
`bench_apk_sharing.py --profile-range` 在 E GPU 上标记完整 executor 生命周期，
用于 Nsight Compute `--replay-mode app-range`。必须包含长驻 kernel 的启动与停止，
避免普通 kernel replay 打断它与 attention publisher 的并发依赖。该区间包含
预热、空闲和 isolated/concurrent 工作，不是纯计算时延；绘图工具拒绝把其计时
放入正常 sharing 对照。
`bench_communication.py --cuda-graph` 用 GPU 同步隔离 dispatch 与 combine，
支持单独构建的固定 DeepEP v1。
DeepEP low-latency 模式会在初始化 NVSHMEM 前按整个 `--rows` 扫描的最大值设置
`NVSHMEM_QP_DEPTH`（至少 `2 * (max_rows + 1)`，向上取整到二次幂），
保留更大的用户配置，并将实际深度写入结果元数据。
`--group-transfers` 测试预建整组 attention 通信模板；
服务端在启用 E 侧预处理时自动使用整组通信；A 侧预处理消融可通过
`SGLANG_EXPERTPLEX_GROUP_COMM=1` 显式选择整组通信。
原始 hidden 通信诊断使用 `--group-transfers --prefill-preprocess --mode normal`，
其 E 侧 identity echo 在 dispatch/combine 计时区间外，只用于验证数据与计时边界；
该诊断不能作为服务路径无接收 kernel 的证明。
`check_teacher_forcing.py --top-k 128 --flush-between` 比较同一历史、无前缀缓存复用的
logprob；top-k 子集不能当作完整词表的 TV/KL。所有脚本的输出均明确实验范围。
完整词表超过 8192 时只允许显式 top-k 诊断。`run_serving_experiment.py --numerical-check`
在全部计时 sweep 之后记录生成及固定历史 top-128 分数；下一次同权重实验可使用
`--numerical-reference-dir <前次输出>/numerical` 对照，原始 token/logprob 保留本地。

上述逐历史单 token 请求重新执行 prefill，不单独覆盖连续 decode。另用
`check_incremental.py` 固定整条参考 token 轨迹，在强制下一 token **之前**保存模型的
FP32 log-softmax；第 0 步来自 prefill，后续步骤使用保留的 KV，发生 retraction 则报错。
它通过 SGLang 的诊断 custom logit processor 完成，不改模型计算路径。启动隔离诊断
服务时，给 `launch.py` 或 `launch_baseline.py` 添加 `--diagnostic-logit-processor`。
该开关默认关闭；只在可信本机测试服务上开启。随后在服务器本机执行：

```bash
python examples/expertplex/check_incremental.py \
  --base-url http://127.0.0.1:31000 --model-path /path/to/model \
  --histories /path/to/reference/numerical/generation.json \
  --steps 32 --top-k 128 --capture-dir /path/to/new-incremental-capture \
  --output /path/to/incremental.json
```

捕获目录必须尚不存在，且服务进程与诊断进程共享该绝对路径。后续部署用相同
`--histories`，再加 `--reference /path/to/incremental.json` 比较每步未强制的分布。
`--top-k 0` 保存完整词表；top-k 子集仍不等于全词表精度。所有采样 rank 的记录均保留，
副本分布不一致、历史不一致或缺步均拒绝生成成功报告。返回的 HTTP logprob 已被
强制采样改变，仅用于校验 token 轨迹，数值比较只使用独立保存的未强制分布。
CPU 拷贝和逐步文件写入会改变执行时序，因此该运行不能用于 TTFT/TPOT/吞吐测量。
对于未透传 `sampling_params.custom_params` 的路由器版本，可在单节点 TP1 P/D
诊断中令 `--base-url` 指向一个 D HTTP 端口，并指定 `--prefill-url` 和
`--bootstrap-port`。工具并发向两端提交相同 bootstrap room 的请求，仍使用服务的
真实 KV 传输；报告标记 `transport=direct-pd`。这条诊断路径不覆盖路由器调度，
不支持 DP-aware worker 的 rank 注入，也不用于服务性能对比。
本工具的 GPU/P-D/图回放路径需在实际服务器上验证，CPU 合约测试不替代该验收。

### 同布局的完整词表增量对齐

`check_numerical_alignment.py` 将上述流程包装为两个顺序运行的诊断。
原生参考使用 EP4、attention DP4、DP LM head，以及与 ExpertPlex 一致的
modulo expert placement。ExpertPlex 端分别启动 MoE、P、D 服务，使用真实 KV
传输。它保留默认 1P+3D+4E 拓扑，但只启动其中一个 D，其余 D 槽位空闲。
四条代码/问答提示各检查 32 个位置，比较强制采样前的完整词表 log-softmax。
这是有限样本上的算术一致性检查，不等于任务准确率、任意并发形态或吞吐验收。

在同一个 `gpuq` 作业内依次执行（每个输出目录必须不存在）：

```bash
python examples/expertplex/check_numerical_alignment.py \
  --variant native --model-path /path/to/model --output /path/to/alignment/native
python examples/expertplex/check_numerical_alignment.py \
  --variant expertplex --model-path /path/to/model \
  --reference /path/to/alignment/native --output /path/to/alignment/expertplex
```

模型必须支持 Qwen3 MoE 路径，专家数能被 4 整除，参考与被测服务必须使用同一份
BF16 权重。工具默认使用 33000 附近端口，可用 `--port-base` 调整；大模型可设置
`--startup-timeout`。该诊断显式限制 KV 池并关闭 CPU/GPU 调度重叠以捕获分数；
这些设置只属于诊断，不改变正常服务的自动 KV 容量、overlap 和图回放配置。
完整词表结果较大，应保留在本地实验目录中。工具结束时清理它启动的服务进程。
默认两端均开启 TBO；检查关闭 TBO 的路径时，两条命令都添加 `--disable-tbo`，
并使用新的参考/输出目录。工具在启动 GPU 服务前拒绝 TBO 模式不同、模型路径不同
或未成功结束的参考。改变 overlap 会改变原生执行形态，不能混用参考分数。

边界回归可在 native 命令上增加 `--prompt-lengths 1 127 128 129 511 512 513 2047`，
并在两条命令上都设置 `--max-prefill-tokens 2048 --steps 8`。提示使用确定性的 token
序列，ExpertPlex 重放参考中的完整历史；两端采集前均清空 prefix cache。更长的测试可
显式增加 `--context-length`。工具在启动前检查参考的分数步数、历史长度与容量，
不会将不同长度的分数集合混作同一个对齐测试。

并发轨迹检查可在两条命令中增加 `--concurrency 16 --require-decode-batch-size 2`，
并给 native 提供至少 16 个不同的 `--prompt-lengths`。工具在每波请求前统一清缓存，
并发发送后按原始顺序比对；禁止在其他请求存活时逐请求清缓存。每条轨迹至少一个
decode 位置的实际 sampling batch 必须达到指定大小，否则测试失败。该记录证明
实际发生了批处理，仍需结合 TBO operation/trace 证据判断非空 microbatch 的执行，
不能仅凭 TBO 启动开关验收。参考与被测端的请求并发数必须一致。

### 存储流水线诊断

`bench_preemption.py` 同时检查完整 trace 的 cluster 决策一致性和阶段切换顺序：
每个实际任务必须包含全部 CTA，epoch/slot/phase/stage 必须一致，进入新阶段前
旧阶段的全部 CTA 必须退休。缺失、重复、截断或未退休记录会使检查失败。
该检查允许同阶段 tile 的正常流水重叠，不把 handoff→retirement 时间当作完整
“system-scope 发布→资源重新分配”的上界。

启用 trace 时，额外保存 `*.trace.decisions.pt`：leader 在读取策略前记录 E 侧时间戳、
退休前缀和 phase，随后关联 tile 领取与实际交付序号。校验器要求每次真实交付都能找到
唯一的决策记录；允许中间插入一次 control flush，但不能插入旧 phase 的真实工作。
报告中的 observation→handoff 包含仲裁和阶段 drain，不包含更早的 A→E 可见性传播；
`old_tiles_in_flight_per_cta` 是未退役输出的数量，不等于正在执行的 Tensor Core tile 数。
同时保存 `*.trace.progress.pt`：记录 epilogue 观察到 accumulator 就绪的时刻
（GEMM 完成的上界）与输出提交时刻，并与最终退役关联。非 GEMM 操作不填写
accumulator 时刻；数据计算、异步 store 和未退役输出分开计数，不能用 pending store
数量直接推断未完成 GEMM 数量。这些 E 侧证据仍不包含完整 A→E 传播上界。
trace 会扰动时序，不应拿启用 trace 的延迟当作正常服务性能。
`--check-single-pending-compute` 要求每次已观察到的 phase 决策处，每 CTA 未完成的
GEMM 与未提交的数据/激活 tile 合计不超过一；没有发生切换也不能通过此检查。
该检查不把已提交但尚未退役的异步输出误算成执行中的 GEMM，也不认证完整传播上界。

`bench_preemption.py --prefill-preprocess` 进一步覆盖原始 P 数据路径的 Scatter1、
Scatter2 和 Gather。该单 E 诊断将逻辑 expert world 中 rank 0 的路由映射为本地 ID，
其余路由设为 padding；它检查该 shard 的调度与算术，不替代完整多 E 通信验证。


调度器保留每次 check epoch 内有界的 CAS 重试；竞争失败但仍有任务时重新检查策略，
不会因此误判该阶段空闲并借给另一阶段。只有确认无可领取任务后才尝试空闲资源借用。
流水 store 路径以每 CTA 的 64 位单调退休序号确认阶段交接，避免元数据提前复用后
仅凭环形屏障奇偶位误认完成。该等待只发生在阶段切换，同阶段 tile 继续流水执行。

完整 `sm100` 后端默认启用延迟 store 退休，以保留原生 epilogue pipeline；
`SGLANG_EXPERTPLEX_PIPELINED_STORES=0` 仅用于消融。其他参考后端保持各自的执行方式。
阶段切换仍等待旧阶段输出退休。每个 phase 的 runnable 位图用于跳过空闲 mailbox，
实际 tile 所有权仍由带 generation 的队列 CAS 决定；没有改成整批领取或减少阶段检查。
这不代表已完成原论文抢占上界或低开销验收，历史记录保留于 `VALIDATION.md`。
组件图由 `plot_components.py` 根据实际 JSON 数据绘制，并拒绝混用不同初始化算法或专家负载。

### 全尺寸实验参数

下面复原后期 1P3D4E 的容量和调度参数。q=96、期望 footprint=8 是被测配置，
不表示完整跨栈优化器已求得全局最优。真实权重模式要求模型先完成下载；若使用
`--synthetic-weights`，服务运行也必须选择 synthetic 并设置相同 canonical seed。

```bash
python examples/expertplex/configure.py \
  --model-path /data/models/Qwen3-Coder-480B-A35B-Instruct \
  --prefill 1 --decode 3 --experts 4 --overlap tbo \
  --max-prefill-tokens 24576 --max-decode-tokens 256 \
  --decode-sm-budget 96 --expected-decode-tiles 8 \
  --max-decode-sm-fraction 0.875 \
  --rendezvous-dir /tmp/expertplex-full480b \
  --output /tmp/expertplex-full480b.json

SGLANG_EXPERTPLEX_PIPELINED_STORES=1 \
python examples/expertplex/run_serving_experiment.py \
  --variant expertplex --config /tmp/expertplex-full480b.json \
  --model-path /data/models/Qwen3-Coder-480B-A35B-Instruct \
  --tokenizer-path /data/models/Qwen3-Coder-480B-A35B-Instruct \
  --weight-mode checkpoint \
  --dataset-path /path/to/sharegpt-holdout-256.jsonl \
  --requests 256 --rates 0.75 1 1.25 1.5 \
  --sequence-cap 24576 --decode-kv-tokens auto --mem-fraction-static 0.95 \
  --ttft-slo-ms 1000 --tpot-slo-ms 50 \
  --output-dir /tmp/expertplex-full480b-evaluation
```

三个启动入口默认使用 `auto`：不向原生 SGLang 传入 `--max-total-tokens`，由现有显存
profiling 分配 KV pool；也不再硬编码不同的 `mem_fraction_static`。可以给所有方案显式
传入相同的 `--mem-fraction-static`（上例为 0.95），保留各布局实际腾出的 KV 空间。
预算包含权重和 KV，剩余空间用于运行时分配；仍需验证峰值负载不会 OOM。
不能仅根据 ExpertPlex 的少量 A 权重选择比例；PDD 每个 phase 复制完整专家权重，
统一预算还必须容纳 PDD 权重及满足采样长度要求的 KV。

论文 §7.1 限制的是**采样请求的序列长度**，并不要求**总 KV pool 容量**相同。
`--sequence-cap` 应在 PDD 实测容量允许范围内冻结，并对所有方案使用相同值。
本目录先前将相同总 KV 写为论文要求，这是错误解读；旧数据只属于固定容量控制实验。
若需复跑这种控制实验，可显式设置 `--decode-kv-tokens 196608`：共置 DP8 每 rank
24576，ExpertPlex 的 3D 每 rank 65536。该选项为全部 D attention 的总容量；
`launch.py --kv-tokens` 则是每个 P/D 实例的容量。不同容量策略、显存预算的结果不能混算。
`run_serving_experiment.py` 会复制 node/PDMux 配置到本次输出目录，并记录其哈希；
后续调参不会覆盖该次实验的配置。

`layout-capacity.json` 记录当前固定 buffer 的布局估算：在 P24576/D256、总 KV196608 下，
1P3D4E+TBO 计入 16 GiB 预留约需 252.39 GiB/E；3P1D4E+TBO 约需 283.14 GiB/E，
超过配置的 268 GiB。3P1D4E 关闭 TBO 的估算约 258.55 GiB/E，仅作为需单独验证的
容量适配；不能把无 TBO 结果称为论文完整 overlap 配置。

也可显式设置 `--prefill-mailbox-tokens 12288` 保留 3P1D4E 的 TBO：请求/batch 上限
仍为 24576，每个 P 的两个 mailbox 分别容纳一个 microbatch。启动器强制
`--tbo-token-distribution-threshold 0.5`，并校验半批及退化 sequence-boundary split
的容量界限；client 在 dispatch 前拒绝超过单 mailbox 容量的调用。该配置估算为
259.50 GiB/E（含 16 GiB 预留）。3P1D4E 小 BF16 checkpoint 的完整/半批容量对照
已通过：两种容量各 14 个边界/并发请求，6 个固定输入的 96 个生成 token 完全一致。
完整 480B 随机权重已完成同样本 LooGLE 三档：0.1/0.25/0.5 req/s 分别为
32/32、32/32、26/32 联合 SLO 通过；E 卡运行中一次快照约 244.13 GiB/卡，
不是峰值。真实权重尚未验收。默认容量不变。

单 D 布局的 router 使用 `round_robin`；多 D 使用 `power_of_two`。启动器等待
`/readiness` 中全部 attention workers 健康后才报告 Ready，避免异步注册阶段的 503。

后续 `run_serving_experiment.py` 默认每档预热 8 个请求，随后清空前缀缓存再计时；
所有被比较方法采用相同数量。此前 pilot/LooGLE 使用 1 个预热请求，无法保证多 P
都经过相同 shape 的预热。可用 `--warmup-requests 1` 复原旧条件；汇总程序拒绝混合
不同预热数量的结果。8 请求预热也不代表穷尽所有动态 batch/sequence shape。

Tile trace 在 exporter 停止后同时写入 `.trace.pt` 与 `.trace.metadata.json`。
metadata 的每 CTA 最后工作序号用于判断截断，包括容量末尾是未记录的 flush 控制任务的情况；
缺少 metadata、容量不一致或序号越界时，`bench_preemption.py` 将拒绝验收该 trace。
旧版只保存 tensor 的 trace 可用于历史分析，但不能直接通过新版完整性检查。

普通服务默认保留 SGLang 的 CPU/GPU scheduler overlap、TBO（由 node config 决定）
与 FlashInfer autotune，并默认开启 decode CUDA Graph。`--disable-overlap-schedule`、
`--disable-flashinfer-autotune` 和 `--no-cuda-graph` 是显式消融开关。
`--diagnostic-logit-processor` 会关闭 CPU/GPU scheduler overlap，以便同步读取 CPU 请求历史；
该数值捕获配置不能用于吞吐/延迟比较。普通基线同样保留 overlap/autotune，PDMux 使用自身调度循环。

路由分布采集复用 SGLang 的 `--expert-distribution-recorder-mode stat`、
`--expert-distribution-recorder-buffer-size -1` 与既有
`/start_expert_distribution_record`、`/stop_expert_distribution_record`、
`/dump_expert_distribution_record` 接口。每个 P/D 服务设置不同的
`SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR`，不要开启 EPLB 改变采集中的专家映射。
`summarize_routing.py` 接受 `--prefill-records`、`--decode-records`、`--model-path`、
`--provenance` 和 `--output`，读取原生 stat dump 的 logical counts，保留逐层计数，
并输出按 token 加权的各 phase 专家概率。stat 已在 TP group 内归约，不要重复传入
同一 group 的副本。采集会改变执行开销，必须与性能验收分开运行。
`check_independent_services.py --record-routing` 只检查接口及重启兼容性；
其少量诊断请求不属于真实数据集的优化器校准。
进一步指定 `--routing-workload /path/to/custom.jsonl --routing-requests 64
--routing-seed 42 --routing-sequence-cap 24576 --kv-tokens auto` 可采集固定数据集。
工具复用原生 custom loader，清除生命周期探测的记录，再逐请求运行真实 KV 移交；
保存数据集及有序 token 请求的散列，并要求 P/D 计数分别精确覆盖输入 token 与
输出长度减一。此模式清空 prefix cache，采集仍不作为性能实验。

严格数值诊断的 `check_numerical_alignment.py --expert-count N` 用于匹配不同
E 数量的 native DP/EP reference 与 ExpertPlex 分片。两边的 expert count、
TBO 和 torch.compile 模式必须一致，不能复用不同分片布局的 reference。
启动 native reference 前会检查已安装 DeepEP 是否支持该 EP 数量；未提供配置的
拓扑不能以其他分片数的结果代替。没有 tokenizer 的小型测试 checkpoint 可使用
`--prompt-lengths 31 127 255 --prompt-token-ids 11 12 13 14` 构造确定的输入。

`launch_baseline.py` 与 `run_serving_experiment.py` 也默认使用 FlashInfer sampling；
数值诊断可显式指定 `--sampling-backend pytorch`。实验总入口默认使用 FlashInfer attention，
保留 `--attention-backend triton` 用于既有固定条件诊断。汇总工具检查 sampling backend
和 serving defaults 版本，拒绝将恢复原生优化后的实验与旧默认配置混合计算加速比。

数值路径继续对齐原生 SGLang/DeepEP：P 使用 contiguous CUDA SiLU，D 使用 masked
Triton 对应的 FP32 运算顺序，最终激活舍入为 BF16。Decode combine 按 top-k 顺序
使用同一个 FP32 累加器，最后才转 BF16；不能先按 E rank 求部分和再相加。
Prefill 则保留原生 normal 路径的每 rank BF16 gather 边界。

```bash
python examples/expertplex/check_activation.py --output /tmp/expertplex-activation.json
```

该检查将生产 helper 与原生两个激活 kernel 逐位比较。升级 Triton、SGLang kernel
或编译参数后需要重跑。通信诊断支持 `--routing-weights nonuniform` 和
`--capture-dir PATH`，保存双方同输入/路由下的输出供交叉比较；`cancellation` 是包含
负权重的算术压力测试，不代表模型 router。组件基准使用新的 native-activation-v2
身份，拒绝与旧版 masked decode 使用 contiguous 激活的完整 MoE 结果混合。

原生 DP-attention 基线可以通过 `launch_baseline.py --enable-dp-lm-head` 或
`run_serving_experiment.py --enable-dp-lm-head` 启用 SGLang 的 DP LM head，避免跨 DP
组的 logits 通信。它增加每个 attention rank 的 LM head 权重占用，应在独立调优集
测量后冻结。严格逐位诊断还需要一致的专家布局与 LM head GEMM 形状；仅在诊断时
对齐布局的结果，不能替代对未经该修改的性能基线进行比较。

严格增量验收在 `check_incremental.py` 上同时指定 `--reference REF.json`、
`--top-k 0` 和 `--require-exact-logprobs`。任一位置的任一词表项不同，或参考记录
缺少完整词表，都会保存失败报告并返回非零状态；默认诊断模式仅记录差异。
