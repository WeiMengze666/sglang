"""Dependency-free configuration for the single-node ExpertPlex experiment.

The model adapter deliberately accepts only unquantized Qwen3 MoE checkpoints.
GPU memory is expressed in GiB; NVIDIA's advertised 288 GB is about 268 GiB.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _positive_integer(name: str, value: Any, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(
            f"{name} must be {'nonnegative' if allow_zero else 'positive'}"
        )
    return value


@dataclass(frozen=True)
class ClientConfig:
    """One independent SGLang attention process; its ID is its mailbox slot."""

    id: int
    phase: str
    device: int
    port: int

    def __post_init__(self) -> None:
        _positive_integer("client id", self.id, allow_zero=True)
        _positive_integer("client device", self.device, allow_zero=True)
        _positive_integer("client port", self.port)
        if self.port > 65535:
            raise ValueError("client port must be at most 65535")
        if self.phase not in ("prefill", "decode"):
            raise ValueError("client phase must be 'prefill' or 'decode'")


@dataclass(frozen=True)
class ModelSpec:
    """The dimensions required by the Qwen3 MoE adapter and memory planner."""

    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    architecture: str = "Qwen3MoeForCausalLM"
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "num_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
        ):
            _positive_integer(name, getattr(self, name))
        if self.architecture != "Qwen3MoeForCausalLM":
            raise ValueError("ExpertPlex currently supports Qwen3MoeForCausalLM only")
        if self.dtype not in ("bfloat16", "bf16"):
            raise ValueError("ExpertPlex currently supports BF16 checkpoints only")
        if self.top_k > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSpec:
        # Transformers 5 serializes Qwen3MoeConfig's num_experts alias as
        # num_local_experts. This still denotes the global model expert count,
        # before SGLang's EP partition, despite the historical field name.
        data = dict(data)
        if "num_local_experts" in data:
            if (
                "num_experts" in data
                and data["num_experts"] != data["num_local_experts"]
            ):
                raise ValueError("conflicting num_experts and num_local_experts")
            data["num_experts"] = data["num_local_experts"]
        if data.get("model_type") != "qwen3_moe":
            raise ValueError("model_type must be qwen3_moe")
        if data.get("quantization_config") is not None:
            raise ValueError(
                "quantized checkpoints are not supported; use the BF16 model"
            )
        architectures = data.get("architectures", [])
        if architectures != ["Qwen3MoeForCausalLM"]:
            raise ValueError("architectures must contain only Qwen3MoeForCausalLM")
        if data.get("mlp_only_layers", []):
            raise ValueError("dense MLP layers are not supported")
        if data.get("decoder_sparse_step", 1) != 1:
            raise ValueError("every layer must be an MoE layer")
        if data.get("attention_bias", False):
            raise ValueError("attention_bias is not supported")
        if data.get("qkv_bias", False):
            raise ValueError("qkv_bias is not supported")
        if data.get("shared_expert_intermediate_size", 0):
            raise ValueError("shared experts are not supported by this model adapter")
        if not data.get("use_qk_norm", True):
            raise ValueError("the Qwen3 adapter requires query/key normalization")
        if data.get("tie_word_embeddings", False):
            raise ValueError("tied embeddings are not supported by this model adapter")
        return cls(
            hidden_size=data["hidden_size"],
            intermediate_size=data["moe_intermediate_size"],
            num_experts=data["num_experts"],
            top_k=data["num_experts_per_tok"],
            num_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data["head_dim"],
            vocab_size=data["vocab_size"],
            architecture=architectures[0],
            dtype=data.get("dtype", data.get("torch_dtype", "")),
        )

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> ModelSpec:
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise ValueError(f"model_path must be an existing local directory: {path}")
        with (path / "config.json").open(encoding="utf-8") as source:
            return cls.from_dict(json.load(source))

    @property
    def expert_weight_bytes(self) -> int:
        # gate, up and down projections; BF16, without biases.
        return (
            3
            * self.hidden_size
            * self.intermediate_size
            * self.num_experts
            * self.num_layers
            * 2
        )

    @property
    def attention_weight_bytes(self) -> int:
        qkv = (
            self.hidden_size
            * self.head_dim
            * (self.num_attention_heads + 2 * self.num_key_value_heads)
        )
        output = self.num_attention_heads * self.head_dim * self.hidden_size
        router = self.hidden_size * self.num_experts
        norms = 2 * self.hidden_size + 2 * self.head_dim
        embeddings = 2 * self.vocab_size * self.hidden_size
        return 2 * (
            (qkv + output + router + norms) * self.num_layers
            + embeddings
            + self.hidden_size
        )

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_key_value_heads * self.head_dim * self.num_layers * 2


@dataclass(frozen=True)
class NodeConfig:
    model_path: str
    clients: tuple[ClientConfig, ...]
    expert_devices: tuple[int, ...]
    max_prefill_tokens: int = 2048
    max_decode_tokens: int = 256
    decode_sm_budget: int = 32
    expected_decode_tiles: int = 128
    max_decode_sm_fraction: float = 0.75
    startup_timeout_s: float = 600.0
    operation_timeout_s: float = 120.0
    rendezvous_dir: str = "/tmp/sglang-expertplex"
    node_gpu_count: int = 8
    cta_count: int = 0
    gpu_memory_gib: float = 268.0
    weight_mode: str = "checkpoint"
    synthetic_seed: int = 1234
    overlap: str = "none"
    kernel: str = "sm100"

    def __post_init__(self) -> None:
        object.__setattr__(self, "clients", tuple(self.clients))
        object.__setattr__(self, "expert_devices", tuple(self.expert_devices))
        self.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeConfig:
        values = dict(data)
        values["clients"] = tuple(
            client if isinstance(client, ClientConfig) else ClientConfig(**client)
            for client in values["clients"]
        )
        values["expert_devices"] = tuple(values["expert_devices"])
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> NodeConfig:
        with Path(path).expanduser().open(encoding="utf-8") as source:
            return cls.from_dict(json.load(source))

    def validate(self, model_spec: ModelSpec | None = None) -> None:
        if self.kernel not in ("sm100", "sm100_cta1", "wmma"):
            raise ValueError("kernel must be sm100, sm100_cta1 or wmma")
        if self.overlap not in ("none", "tbo"):
            raise ValueError("overlap must be none or tbo")
        if self.weight_mode not in ("checkpoint", "synthetic"):
            raise ValueError("weight_mode must be checkpoint or synthetic")
        _positive_integer("synthetic_seed", self.synthetic_seed, allow_zero=True)
        if self.node_gpu_count != 8:
            raise ValueError("this implementation supports exactly one 8-GPU node")
        if not self.clients or not self.expert_devices:
            raise ValueError("attention clients and expert devices are required")
        ids = sorted(client.id for client in self.clients)
        if ids != list(range(len(self.clients))):
            raise ValueError(
                "client IDs must be unique and contiguous, starting at zero"
            )
        if {client.phase for client in self.clients} != {"prefill", "decode"}:
            raise ValueError("at least one prefill and one decode client are required")
        devices = [client.device for client in self.clients] + list(self.expert_devices)
        for device in devices:
            _positive_integer("device", device, allow_zero=True)
        if sorted(devices) != list(range(self.node_gpu_count)):
            raise ValueError(
                "attention and expert devices must form a disjoint full partition of ranks 0..7"
            )
        if len({client.port for client in self.clients}) != len(self.clients):
            raise ValueError("attention client ports must be unique")
        for name in (
            "max_prefill_tokens",
            "max_decode_tokens",
            "decode_sm_budget",
            "expected_decode_tiles",
        ):
            _positive_integer(name, getattr(self, name))
        _positive_integer("cta_count", self.cta_count, allow_zero=True)
        for name in ("startup_timeout_s", "operation_timeout_s", "gpu_memory_gib"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            not isinstance(self.max_decode_sm_fraction, (float, int))
            or isinstance(self.max_decode_sm_fraction, bool)
            or not 0 < self.max_decode_sm_fraction < 1
        ):
            raise ValueError(
                "max_decode_sm_fraction must be strictly between zero and one"
            )
        for name in ("model_path", "rendezvous_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "://" in value:
                raise ValueError(f"{name} must be a nonempty local filesystem path")
        if model_spec is not None and model_spec.num_experts % len(self.expert_devices):
            raise ValueError(
                "the number of expert ranks must divide the model expert count"
            )

    def client(self, client_id: int) -> ClientConfig:
        for client in self.clients:
            if client.id == client_id:
                return client
        raise ValueError(f"unknown attention client ID: {client_id}")

    @property
    def mailboxes_per_client(self) -> int:
        return 2 if self.overlap == "tbo" else 1

    @property
    def slot_count(self) -> int:
        return len(self.clients) * self.mailboxes_per_client

    def buffer_layout(self) -> tuple[list[int], list[int]]:
        """Mailbox order matches client_id * mailboxes_per_client + index."""
        capacities, packed = [], []
        for client in sorted(self.clients, key=lambda c: c.id):
            for _ in range(self.mailboxes_per_client):
                capacities.append(
                    self.max_prefill_tokens
                    if client.phase == "prefill"
                    else self.max_decode_tokens
                )
                packed.append(int(client.phase == "prefill"))
        return capacities, packed

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expert_devices"] = list(self.expert_devices)
        return data

    def write_json(self, path: str | Path) -> None:
        Path(path).expanduser().write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )


def load_config(path: str | Path) -> NodeConfig:
    return NodeConfig.from_json(path)


def default_config(
    model_path: str,
    *,
    prefill_count: int = 1,
    decode_count: int = 3,
    expert_count: int = 4,
    **overrides: Any,
) -> NodeConfig:
    """Create a contiguous placement (the default is the paper's 1P3D4E)."""
    for name, count in (
        ("prefill_count", prefill_count),
        ("decode_count", decode_count),
        ("expert_count", expert_count),
    ):
        _positive_integer(name, count)
    attention_count = prefill_count + decode_count
    if attention_count + expert_count != 8:
        raise ValueError("prefill_count + decode_count + expert_count must equal 8")
    values: dict[str, Any] = {
        "model_path": model_path,
        "clients": tuple(
            ClientConfig(i, "prefill" if i < prefill_count else "decode", i, 30000 + i)
            for i in range(attention_count)
        ),
        "expert_devices": tuple(range(attention_count, 8)),
    }
    values.update(overrides)
    return NodeConfig(**values)
