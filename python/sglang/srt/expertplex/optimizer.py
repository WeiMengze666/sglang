"""CPU planner for ExpertPlex equations 1--4 (paper sections 6.1--6.4).

Dry runs estimate capacity only. A search requires measured per-layer latency
coefficients and NVLink bandwidth; no performance numbers are invented here.
The profile schema describes the critical path approximation explicitly, and
the result is a model prediction which still requires end-to-end GPU validation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import ModelSpec, NodeConfig, default_config
from .native import estimate_buffer_bytes

GIB = 1024**3


def tile_footprint(expert_rows: Iterable[int | float], tile_rows: int) -> int:
    """Equation 3: zero-load experts consume no tiles."""
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows <= 0:
        raise ValueError("tile_rows must be a positive integer")
    total = 0
    for rows in expert_rows:
        if (
            isinstance(rows, bool)
            or not isinstance(rows, (int, float))
            or not math.isfinite(rows)
            or rows < 0
        ):
            raise ValueError("expert row counts must be finite and nonnegative")
        total += math.ceil(rows / tile_rows)
    return total


@lru_cache(maxsize=32768)
def expected_expert_tiles(tokens: int, probability: float, tile_rows: int) -> float:
    """E[ceil(Binomial(tokens,p)/tile_rows)] for an IID routing approximation.

    ceil(E[rows]/tile_rows) falsely treats every nonzero-probability expert as
    active even for a single token. Sum actual integer-load probabilities instead.
    Expand from the binomial mode to avoid underflow at large token counts.
    """
    if tokens < 0 or tile_rows <= 0 or not 0 <= probability <= 1:
        raise ValueError("Invalid expected-tile inputs")
    if not tokens or probability == 0:
        return 0.0
    if probability == 1:
        return float(math.ceil(tokens / tile_rows))
    mode = math.floor((tokens + 1) * probability)
    mass = math.exp(
        math.lgamma(tokens + 1)
        - math.lgamma(mode + 1)
        - math.lgamma(tokens - mode + 1)
        + mode * math.log(probability)
        + (tokens - mode) * math.log1p(-probability)
    )
    result = mass * math.ceil(mode / tile_rows)
    weight = mass
    for value in range(mode, 0, -1):
        weight *= value * (1 - probability) / ((tokens - value + 1) * probability)
        result += weight * math.ceil((value - 1) / tile_rows)
        if weight < 1e-16:
            break
    weight = mass
    for value in range(mode, tokens):
        weight *= (tokens - value) * probability / ((value + 1) * (1 - probability))
        result += weight * math.ceil((value + 1) / tile_rows)
        if weight < 1e-16:
            break
    return result


@dataclass(frozen=True)
class SmAllocation:
    prefill_sms: int
    decode_sms: int


def adaptive_sm_allocation(
    *,
    total_sms: int,
    decode_sm_budget: int,
    expected_decode_tiles: int,
    current_decode_tiles: int,
    prefill_ready: bool,
    decode_ready: bool,
    cluster_size: int = 1,
    max_decode_sm_fraction: float = 0.75,
) -> SmAllocation:
    """Equation 4, with one cluster retained for prefill under contention."""
    for name, value in (
        ("total_sms", total_sms),
        ("decode_sm_budget", decode_sm_budget),
        ("expected_decode_tiles", expected_decode_tiles),
        ("cluster_size", cluster_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(current_decode_tiles, bool)
        or not isinstance(current_decode_tiles, int)
        or current_decode_tiles < 0
    ):
        raise ValueError("current_decode_tiles must be a nonnegative integer")
    if not 0 < max_decode_sm_fraction < 1:
        raise ValueError("max_decode_sm_fraction must be between zero and one")
    if total_sms % cluster_size:
        raise ValueError("total_sms must be divisible by cluster_size")
    if not prefill_ready and not decode_ready:
        return SmAllocation(0, 0)
    if not decode_ready:
        return SmAllocation(total_sms, 0)
    if not prefill_ready:
        return SmAllocation(0, total_sms)
    if total_sms < 2 * cluster_size:
        raise ValueError("contention requires at least two CTA clusters")
    q_max = min(
        total_sms - cluster_size,
        max(
            cluster_size,
            math.floor(total_sms * max_decode_sm_fraction / cluster_size)
            * cluster_size,
        ),
    )
    scaled = (
        math.ceil(
            decode_sm_budget
            * current_decode_tiles
            / expected_decode_tiles
            / cluster_size
        )
        * cluster_size
    )
    decode_sms = min(q_max, max(cluster_size, scaled))
    return SmAllocation(total_sms - decode_sms, decode_sms)


@dataclass(frozen=True)
class Layout:
    prefill_count: int
    decode_count: int
    expert_count: int

    @property
    def name(self) -> str:
        return f"{self.prefill_count}P{self.decode_count}D{self.expert_count}E"

    def config(self, model_path: str, **overrides: Any) -> NodeConfig:
        return default_config(model_path, **asdict(self), **overrides)


def enumerate_layouts(num_experts: int, node_gpu_count: int = 8) -> list[Layout]:
    if node_gpu_count != 8 or num_experts <= 0:
        raise ValueError(
            "placement enumeration requires one 8-GPU node and a positive expert count"
        )
    return [
        Layout(p, 8 - p - e, e)
        for e in range(1, 7)
        if num_experts % e == 0
        for p in range(1, 8 - e)
        if 8 - p - e > 0
    ]


@dataclass(frozen=True)
class MemoryEstimate:
    attention_weights_bytes: int
    attention_kv_bytes: int
    attention_buffers_bytes: int
    expert_weights_bytes: int
    expert_buffers_bytes: int
    reserve_bytes_per_gpu: int
    available_bytes_per_gpu: int

    @property
    def attention_total_bytes(self) -> int:
        return (
            self.attention_weights_bytes
            + self.attention_kv_bytes
            + self.attention_buffers_bytes
            + self.reserve_bytes_per_gpu
        )

    @property
    def expert_total_bytes(self) -> int:
        return (
            self.expert_weights_bytes
            + self.expert_buffers_bytes
            + self.reserve_bytes_per_gpu
        )

    @property
    def fits(self) -> bool:
        return (
            max(self.attention_total_bytes, self.expert_total_bytes)
            <= self.available_bytes_per_gpu
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            attention_total_gib=self.attention_total_bytes / GIB,
            expert_total_gib=self.expert_total_bytes / GIB,
            fits=self.fits,
            note="Capacity estimate only; verify allocated/reserved HBM after GPU startup.",
        )
        return result


def estimate_memory(
    config: NodeConfig,
    model: ModelSpec,
    *,
    kv_tokens_per_attention_gpu: int,
    reserve_gib: float = 16.0,
) -> MemoryEstimate:
    """Count this backend's fixed buffers, not idealized evenly routed buffers.

    Prefill reserves packed routed rows including alignment slack; decode
    reserves its own masked capacity. TBO gets two mailboxes per client.
    Buffers are reused across layers; expert weights retain all layers.
    """
    config.validate(model)
    if (
        isinstance(kv_tokens_per_attention_gpu, bool)
        or not isinstance(kv_tokens_per_attention_gpu, int)
        or kv_tokens_per_attention_gpu <= 0
    ):
        raise ValueError("kv_tokens_per_attention_gpu must be a positive integer")
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("reserve_gib must be finite and nonnegative")
    local_experts = model.num_experts // len(config.expert_devices)
    slots = config.slot_count
    rows = max(config.max_prefill_tokens, config.max_decode_tokens)
    capacities, packed = config.buffer_layout()
    expert_buffers = estimate_buffer_bytes(
        slots,
        local_experts,
        rows,
        model.hidden_size,
        model.intermediate_size,
        capacities=capacities,
        packed=packed,
        top_k=model.top_k,
    )["total"]
    # Dispatch writes to peer final buffers and combine reads those buffers
    # directly. Only hidden/output and routing tensors are local to attention;
    # additional transformer temporaries belong in the explicit reserve.
    attention_buffers = (
        config.mailboxes_per_client * rows * (8 * model.hidden_size + 12 * model.top_k)
        + config.mailboxes_per_client
        * len(config.expert_devices)
        * config.slot_count
        * local_experts
        * 8
    )
    return MemoryEstimate(
        attention_weights_bytes=model.attention_weight_bytes,
        attention_kv_bytes=model.kv_bytes_per_token * kv_tokens_per_attention_gpu,
        attention_buffers_bytes=attention_buffers,
        expert_weights_bytes=model.expert_weight_bytes // len(config.expert_devices),
        expert_buffers_bytes=expert_buffers,
        reserve_bytes_per_gpu=math.ceil(reserve_gib * GIB),
        available_bytes_per_gpu=math.floor(config.gpu_memory_gib * GIB),
    )


@dataclass(frozen=True)
class LatencyCoefficients:
    """Equation 2 coefficients; latency is seconds per transformer layer."""

    alpha: float
    beta: float
    gamma: float = 0.0
    delta: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("latency coefficients must be finite")

    def predict(self, x: float, sequence_length: float = 1.0) -> float:
        if (
            not math.isfinite(x)
            or x < 0
            or not math.isfinite(sequence_length)
            or sequence_length <= 0
        ):
            raise ValueError(
                "prediction inputs must be finite, x >= 0 and sequence_length > 0"
            )
        value = (
            self.alpha
            + self.beta * x
            + self.gamma * x * sequence_length
            + self.delta * x * sequence_length**2
        )
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "profile extrapolation predicted nonpositive/nonfinite latency"
            )
        return value


def fit_coefficients(
    samples: Sequence[dict[str, float]], *, component: str
) -> LatencyCoefficients:
    """Least squares over measured samples; reject underdetermined profiles.

    MoE uses s=1 in Eq2, so beta/gamma/delta are not identifiable separately;
    fit alpha+beta*x directly. Attention fits all four terms.
    """
    if component not in ("attention", "moe"):
        raise ValueError("component must be attention or moe")
    width = 4 if component == "attention" else 2
    if len(samples) < width:
        raise ValueError(f"at least {width} measured samples are required")
    design: list[list[float]] = []
    times: list[float] = []
    for sample in samples:
        x, s, latency = (
            float(sample["x"]),
            float(sample.get("sequence_length", 1)),
            float(sample["latency_s"]),
        )
        if (
            not all(math.isfinite(v) for v in (x, s, latency))
            or x < 0
            or s <= 0
            or latency <= 0
        ):
            raise ValueError(
                "measurements require finite x>=0, sequence_length>0, latency_s>0"
            )
        design.append([1.0, x, x * s, x * s * s] if width == 4 else [1.0, x])
        times.append(latency)
    # Scale columns to reduce conditioning problems for long sequence lengths.
    scales = [max(abs(row[j]) for row in design) or 1 for j in range(width)]
    design = [[row[j] / scales[j] for j in range(width)] for row in design]
    matrix = [
        [sum(row[i] * row[j] for row in design) for j in range(width)]
        + [sum(row[i] * time for row, time in zip(design, times))]
        for i in range(width)
    ]
    for column in range(width):
        pivot = max(range(column, width), key=lambda index: abs(matrix[index][column]))
        if abs(matrix[pivot][column]) < 1e-12:
            raise ValueError(
                "measurements are rank deficient; vary batch size and sequence length"
            )
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        factor = matrix[column][column]
        matrix[column] = [value / factor for value in matrix[column]]
        for row in range(width):
            if row == column:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                left - factor * right
                for left, right in zip(matrix[row], matrix[column])
            ]
    coefficients = [matrix[i][-1] / scales[i] for i in range(width)]
    return LatencyCoefficients(*coefficients)


@dataclass(frozen=True)
class MeasuredProfiles:
    """Measured profile database: no default coefficients or bandwidth."""

    provenance: str
    total_sms: int
    tile_rows: int | dict[str, int]
    nvlink_bytes_per_second: float
    attention: dict[str, LatencyCoefficients]
    moe: dict[tuple[str, int], LatencyCoefficients]
    expert_probabilities: dict[str, tuple[float, ...]]

    @classmethod
    def from_dict(cls, data: dict[str, Any], model: ModelSpec) -> MeasuredProfiles:
        if data.get("units") != "seconds_per_layer":
            raise ValueError("profile units must be seconds_per_layer")
        provenance = data.get("provenance", "")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("measured profile provenance is required")
        for name in ("total_sms",):
            if (
                isinstance(data.get(name), bool)
                or not isinstance(data.get(name), int)
                or data[name] <= 0
            ):
                raise ValueError(f"profile {name} must be a positive integer")
        tile_rows = data.get("tile_rows")
        phase_tiles = (
            tile_rows
            if isinstance(tile_rows, dict)
            else {p: tile_rows for p in ("prefill", "decode")}
        )
        if set(phase_tiles) != {"prefill", "decode"} or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in phase_tiles.values()
        ):
            raise ValueError(
                "tile_rows must be positive, or map prefill/decode to positive tile heights"
            )
        bandwidth = data.get("nvlink_bytes_per_second", 0)
        if (
            not isinstance(bandwidth, (int, float))
            or not math.isfinite(bandwidth)
            or bandwidth <= 0
        ):
            raise ValueError("measured NVLink bandwidth is required")
        attention: dict[str, LatencyCoefficients] = {}
        moe: dict[tuple[str, int], LatencyCoefficients] = {}
        probabilities: dict[str, tuple[float, ...]] = {}
        for phase in ("prefill", "decode"):
            phase_data = data[phase]
            attention[phase] = LatencyCoefficients(**phase_data["attention"])
            for budget, coefficients in phase_data["moe_by_sm_count"].items():
                sm_count = int(budget)
                if not 0 < sm_count <= data["total_sms"]:
                    raise ValueError("profile SM counts must be within GPU capacity")
                moe[phase, sm_count] = LatencyCoefficients(**coefficients)
            routing = tuple(phase_data["expert_probabilities"])
            if (
                len(routing) != model.num_experts
                or any(
                    not isinstance(p, (float, int))
                    or not math.isfinite(p)
                    or not 0 <= p <= 1
                    for p in routing
                )
                or not math.isclose(
                    sum(routing), model.top_k, rel_tol=1e-6, abs_tol=1e-6
                )
            ):
                raise ValueError(
                    "measured expert probabilities must be in [0,1], one per expert, and sum to top_k"
                )
            probabilities[phase] = routing
        return cls(
            provenance,
            data["total_sms"],
            data["tile_rows"],
            bandwidth,
            attention,
            moe,
            probabilities,
        )

    @classmethod
    def from_json(cls, path: str | Path, model: ModelSpec) -> MeasuredProfiles:
        with Path(path).open(encoding="utf-8") as source:
            return cls.from_dict(json.load(source), model)


@dataclass(frozen=True)
class SearchResult:
    layout: Layout
    decode_sm_budget: int
    prefill_batch: int
    decode_batch: int
    predicted_prefill_latency_s: float
    predicted_decode_latency_s: float
    predicted_goodput_requests_per_second: float
    profile_provenance: str
    overlap: str


def _iteration_latency(
    model: ModelSpec,
    profiles: MeasuredProfiles,
    layout: Layout,
    phase: str,
    batch: int,
    input_length: int,
    context_length: int,
    sm_count: int,
    overlap: str,
) -> float:
    attention_count = (
        layout.prefill_count if phase == "prefill" else layout.decode_count
    )
    local_batch = math.ceil(batch / attention_count)
    sequence = input_length if phase == "prefill" else context_length
    attention = profiles.attention[phase].predict(local_batch, sequence)
    tokens = batch * input_length if phase == "prefill" else batch
    # Mailboxes belong to independent attention clients. Their partial tiles
    # cannot be merged into a single global-batch tile without changing the
    # runtime. Sum each client's tiles on every expert rank, then take max rank.
    client_batches = [
        batch // attention_count + (i < batch % attention_count)
        for i in range(attention_count)
    ]
    probabilities = profiles.expert_probabilities[phase]
    footprint = max(
        sum(
            sum(
                expected_expert_tiles(
                    client_batch * (input_length if phase == "prefill" else 1),
                    probability,
                    profiles.tile_rows[phase]
                    if isinstance(profiles.tile_rows, dict)
                    else profiles.tile_rows,
                )
                for probability in probabilities[rank :: layout.expert_count]
            )
            for client_batch in client_batches
            if client_batch
        )
        for rank in range(layout.expert_count)
    )
    moe = profiles.moe[phase, sm_count].predict(footprint)
    # NVLink path is bidirectional activation dispatch + expert-result pull.
    # This is a shared-fabric byte model; topology/protocol overhead belongs in
    # measured bandwidth and per-layer APK profile, not a synthetic constant.
    communication = (
        2
        * tokens
        * model.top_k
        * model.hidden_size
        * 2
        / profiles.nvlink_bytes_per_second
    )
    layer = (
        max(attention, moe + communication)
        if overlap == "tbo"
        else attention + moe + communication
    )
    return layer * model.num_layers


def search(
    config: NodeConfig,
    model: ModelSpec,
    profiles: MeasuredProfiles,
    *,
    input_length: int,
    average_output_length: int,
    context_length: int,
    prefill_slo_s: float,
    decode_slo_s: float,
    kv_tokens_per_attention_gpu: int,
    reserve_gib: float = 16.0,
    overlap: str = "sequential",
) -> SearchResult | None:
    """Algorithm 1, using only explicitly profiled contention SM budgets.

    Routing probabilities are an expectation, not a worst-case distribution.
    TBO is an analytical option for a measured TBO setup; selecting it here
    does not enable communication overlap in the runtime.
    """
    if overlap not in ("sequential", "tbo"):
        raise ValueError("overlap must be sequential or tbo")
    for name, value in (
        ("input_length", input_length),
        ("average_output_length", average_output_length),
        ("context_length", context_length),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("prefill_slo_s", prefill_slo_s),
        ("decode_slo_s", decode_slo_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for phase, sequence in (("prefill", input_length), ("decode", context_length)):
        coefficients = profiles.attention[phase]
        if (
            coefficients.beta
            + coefficients.gamma * sequence
            + coefficients.delta * sequence**2
            < 0
        ):
            raise ValueError(
                "attention profile must be monotone in batch size for binary search"
            )
        if any(
            c.beta + c.gamma + c.delta < 0
            for (p, _), c in profiles.moe.items()
            if p == phase
        ):
            raise ValueError(
                "MoE profile must be monotone in tile count for binary search"
            )
    best: SearchResult | None = None
    for layout in enumerate_layouts(model.num_experts):
        candidate_config = layout.config(
            config.model_path,
            **{
                key: value
                for key, value in config.to_dict().items()
                if key not in ("model_path", "clients", "expert_devices")
            },
        )
        if not estimate_memory(
            candidate_config,
            model,
            kv_tokens_per_attention_gpu=kv_tokens_per_attention_gpu,
            reserve_gib=reserve_gib,
        ).fits:
            continue
        budgets = sorted(
            sm
            for (phase, sm) in profiles.moe
            if phase == "decode"
            and ("prefill", profiles.total_sms - sm) in profiles.moe
            and sm <= profiles.total_sms * config.max_decode_sm_fraction
        )
        for q in budgets:
            batches: dict[str, int] = {}
            latencies: dict[str, float] = {}
            for phase, count, maximum, slo, sms in (
                (
                    "prefill",
                    layout.prefill_count,
                    min(config.max_prefill_tokens, kv_tokens_per_attention_gpu)
                    // input_length,
                    prefill_slo_s,
                    profiles.total_sms - q,
                ),
                (
                    "decode",
                    layout.decode_count,
                    min(
                        config.max_decode_tokens,
                        kv_tokens_per_attention_gpu // context_length,
                    ),
                    decode_slo_s,
                    q,
                ),
            ):
                low, high = 0, count * maximum
                while low < high:
                    middle = (low + high + 1) // 2
                    latency = _iteration_latency(
                        model,
                        profiles,
                        layout,
                        phase,
                        middle,
                        input_length,
                        context_length,
                        sms,
                        overlap,
                    )
                    if latency <= slo:
                        low = middle
                    else:
                        high = middle - 1
                batches[phase] = low
                if low:
                    latencies[phase] = _iteration_latency(
                        model,
                        profiles,
                        layout,
                        phase,
                        low,
                        input_length,
                        context_length,
                        sms,
                        overlap,
                    )
            if not all(batches.values()):
                continue
            goodput = min(
                batches["prefill"] / latencies["prefill"],
                batches["decode"] / (latencies["decode"] * average_output_length),
            )
            result = SearchResult(
                layout,
                q,
                batches["prefill"],
                batches["decode"],
                latencies["prefill"],
                latencies["decode"],
                goodput,
                profiles.provenance,
                overlap,
            )
            if best is None or goodput > best.predicted_goodput_requests_per_second:
                best = result
    return best


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--write-config", type=Path)
    parser.add_argument("--prefill-count", type=int, default=1)
    parser.add_argument("--decode-count", type=int, default=3)
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--max-prefill-tokens", type=int, default=2048)
    parser.add_argument("--max-decode-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-gib", type=float, default=268.0)
    parser.add_argument("--kv-tokens-per-attention-gpu", type=int, default=65536)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--average-output-length", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--prefill-slo-s", type=float, default=1.0)
    parser.add_argument("--decode-slo-s", type=float, default=0.05)
    parser.add_argument(
        "--overlap", choices=("sequential", "tbo"), default="sequential"
    )
    args = parser.parse_args(argv)
    model = ModelSpec.from_model_path(args.model_path)
    config = default_config(
        args.model_path,
        prefill_count=args.prefill_count,
        decode_count=args.decode_count,
        expert_count=args.expert_count,
        max_prefill_tokens=args.max_prefill_tokens,
        max_decode_tokens=args.max_decode_tokens,
        gpu_memory_gib=args.gpu_memory_gib,
    )
    memory = estimate_memory(
        config,
        model,
        kv_tokens_per_attention_gpu=args.kv_tokens_per_attention_gpu,
        reserve_gib=args.reserve_gib,
    )
    if not memory.fits:
        parser.error(
            "selected placement exceeds configured HBM capacity; lower fixed mailbox rows or choose more expert GPUs"
        )
    if args.write_config:
        config.write_json(args.write_config)
    output: dict[str, Any] = {"config": config.to_dict(), "memory": memory.to_dict()}
    if args.profiles:
        profiles = MeasuredProfiles.from_json(args.profiles, model)
        result = search(
            config,
            model,
            profiles,
            input_length=args.input_length,
            average_output_length=args.average_output_length,
            context_length=args.context_length,
            prefill_slo_s=args.prefill_slo_s,
            decode_slo_s=args.decode_slo_s,
            kv_tokens_per_attention_gpu=args.kv_tokens_per_attention_gpu,
            reserve_gib=args.reserve_gib,
            overlap=args.overlap,
        )
        output["prediction"] = asdict(result) if result is not None else None
        output["prediction_note"] = (
            "Profile-backed estimate only; this is not measured end-to-end goodput."
        )
    else:
        output["prediction_note"] = (
            "No measured profiles supplied: no latency or goodput prediction made."
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
