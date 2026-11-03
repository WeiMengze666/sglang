# 可选的原生 Attention DP Decode 组

默认拓扑仍是三个独立的 TP1 Decode 服务。这个可选拓扑把 GPU 1–3 交给一个
原生 SGLang Decode 前端，使用原生 attention DP controller、gather、MoE 输出归约
和 DP LM head；GPU 0 的 Prefill 与 GPU 4–7 的 Expert 服务仍独立启动。
它不引入新的调度器或服务运行时，也不表示已经找到更优的 goodput 配置。

## 拓扑与容量

示例配置见 [JSON](configs/qwen3-coder-480b-bf16-native-attention-dp3.json)。
`ClientConfig.tp_size` 表示连续的物理 GPU 跨度，`attention_dp_size` 表示其中的
attention DP 副本数，有效 attention TP 为二者之商。本例是物理跨度 3、attention DP 3、
有效 attention TP 1。attention/router 权重和 KV pool 在三个 D GPU 上各自保留；
显存估计使用有效 TP1，不能按物理跨度 3 把这些权重再除以三。

| 配置 | 全局值 | 每个 D 副本 |
| --- | ---: | ---: |
| 最大 running requests / D mailbox token 容量 | 48 | 16 |
| Decode 启动参数 `--chunked-prefill-size` | 48 | 原生解析为 16 |
| `--max-prefill-tokens` | — | 16 |
| Decode CUDA graph 最大 bucket | — | 16 |

P 的 `--max-running-requests 16` 单独显式指定。不要把 D 的全局请求上限 48
直接用于 P，或把 D 的原始 chunk 参数改成 16：原生 attention DP hook 还会再除以三。
查询 D 的 server info 时，`chunked_prefill_size` 是解析后的 16。

默认 `attention_dp_size=1` 不写入序列化 JSON，已有 attention TP 组的含义保持不变。
当前新模式只接受 Decode 的有效 TP1、CPU overlap 开启、D overlap NONE、完整 Decode
CUDA graphs 且不开启 `torch.compile`；新的 TBO/SBO、TP×DP 混合分片组合会明确拒绝。
没有单独实测 DP/gather profile 时，placement optimizer 也会拒绝把该模式当作已测候选。

## 独立启动 P、D、E

在同一队列作业内启动下列独立服务，所有进程继承相同的完整 GPU 可见列表。
先修改 JSON 中的模型目录；每次新部署使用新的 `rendezvous_dir`。

```bash
export MODEL=/data/models/Qwen3-Coder-480B-A35B-Instruct
export CONFIG="$PWD/examples/expertplex/configs/qwen3-coder-480b-bf16-native-attention-dp3.json"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
export MC_INTRANODE_NVLINK=true
```

E 只启动 GPU 4–7 的 Expert workers；JSON 决定其 attention 客户端布局。
不要在 E 命令上设置 `--tp-size 3`。

```bash
python -m sglang.launch_server --model-path "$MODEL" --dtype bfloat16 \
  --load-format safetensors --enable-expertplex --expertplex-role expert \
  --expertplex-config "$CONFIG" --host 127.0.0.1 --port 31200
```

P 占 GPU 0，独立保留 16 个 running requests，使用原生 TBO：

```bash
python -m sglang.launch_server --model-path "$MODEL" --dtype bfloat16 \
  --load-format safetensors --enable-expertplex \
  --expertplex-bootstrap-addr http://127.0.0.1:31200 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --base-gpu-id 0 --tp-size 1 --dp-size 1 --ep-size 1 \
  --host 127.0.0.1 --port 30000 --nccl-port 20000 \
  --disaggregation-bootstrap-port 19000 --context-length 24576 \
  --max-prefill-tokens 49152 --max-running-requests 16 --chunked-prefill-size -1 \
  --mem-fraction-static 0.90 --enable-two-batch-overlap \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --cuda-graph-bs-decode 16 --attention-backend fa4 \
  --enable-deterministic-inference --sampling-backend pytorch --random-seed 42
```

D 占 GPU 1–3。这里的 `--tp-size 3` 是物理进程组跨度，attention 各副本仍为 TP1；
`--dp-size 1` 与 `--attn-dp-size 3` 分别指定外层 DP 和 attention DP，不能交换。
保留默认 CPU overlap，不加 `--disable-overlap-schedule`：

```bash
python -m sglang.launch_server --model-path "$MODEL" --dtype bfloat16 \
  --load-format safetensors --enable-expertplex \
  --expertplex-bootstrap-addr http://127.0.0.1:31200 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --base-gpu-id 1 --tp-size 3 --attn-dp-size 3 --dp-size 1 --ep-size 1 \
  --enable-dp-lm-head --host 127.0.0.1 --port 30001 --nccl-port 20001 \
  --disaggregation-bootstrap-port 19001 --context-length 24576 \
  --max-prefill-tokens 16 --max-running-requests 48 --chunked-prefill-size 48 \
  --mem-fraction-static 0.95 \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --cuda-graph-bs-decode 16 --attention-backend fa4 \
  --enable-deterministic-inference --sampling-backend pytorch --random-seed 42
```

按原生 PD 路由登记 P 和共享 D 的 HTTP 地址；D 只登记一次：

```bash
python -m sglang_router.launch_router --pd-disaggregation \
  --host 127.0.0.1 --port 31000 \
  --prefill http://127.0.0.1:30000 19000 \
  --decode http://127.0.0.1:30001 \
  --prefill-policy round_robin --decode-policy round_robin
```

此命令只要求 router 转发到共享前端，不假定安装的 router 能感知其内部 attention DP
副本。先等待 P/D 的 `/ready` 成功，并确认 E 的 `/health` 或 `/get_server_info`
返回 ready，再请求路由；停止时先排空并关闭 P/D，最后关闭 E。
E 的 `/get_server_info` 可以核对当前服务代次、布局和已登记客户端。KV pool 继续由
SGLang 按剩余显存自动估计；上述配置没有人为固定 KV token 数量。

## 原生 padding 与远端 MoE

原生 MAX_LEN gather 会在每个 DP segment 内产生 padding holes，不能用一个总有效
行数形成连续前缀 mask。新模式在原生 Decode graph registry 中保留一个固定地址的
三元素 int32 raw count 向量，每次 replay 根据该批次的原始 CPU counts 更新。
原生的 padded count buffers 保持原用途。各 MoE 层只把无效行的 route 改为 ID `-1`
和 weight `0`；有效 ID/FP32 weight 的位模式保持不变。SUM_LEN 则要求实际紧凑行数
与三个 raw counts 的和完全相等，缺失或不合法 counts 会报错。

原生 gather 后只有物理 TP rank 0 发布远端操作，followers 返回零 contribution，
再经过原生 MoE 输出归约/分发。DP1 与非 ExpertPlex 的原生 attention DP 不新增该
raw-count slot 或 mask 运算。Eager 路径的临时 count 分配不进入 graph replay 分支。
这些 CPU 接口约束与真实 GPU collective、图重放正确性是不同的验证层次。

## 验证范围

[注册 CPU 测试](../../test/registered/unit/expertplex/test_attention_dp.py)覆盖配置、
原生参数解析、DP1/native DP8 兼容边界、动态 counts、同 bucket 固定地址、owner idle、
非前缀 padding holes、有效 route 位模式和实际 dispatcher 的单 owner/follower-zero。
CPU collective stand-in 不证明 NCCL 或真实服务吞吐。

本节 P/D 命令显式指定唯一 Decode graph bank `[16]`，对应当前主模型数值检查的
bank 范围；`--cuda-graph-max-bs-decode 16` 会生成多个 bank，不能用它替代。
Graph1 等额外 bank 不由这一数值检查验收。

这里给出的主模型数值配置是 BF16 Qwen3-Coder、FA4、原生 deterministic inference、
CPU overlap、P TBO、D NONE、Decode full graphs、无 `torch.compile`。其验收需要新的
原生参考/自重复、逐 token 全词表严格对齐、三路共同进展、idle 副本、真实输入和提交
以及客户端/Expert 生命周期检查；是否通过须以对应实测报告为准。本文不把该范围扩展为
非确定性模式、编译模式、论文全部细节或相对最强原生基线的 1.5× goodput 结论。
