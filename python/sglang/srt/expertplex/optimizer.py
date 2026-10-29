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
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

from .component_profile import ComponentProfiles, ProfileCoverageError
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
        prefill_preprocess=config.prefill_preprocess,
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
class AttentionDomain:
    """Interpolation support from training shapes, clipped by profiled capacity.

    The convex hull bounds input shapes, not prediction error. Heldout checks
    must independently validate the frozen fit; they must not expand this hull.
    Sequence length is fixed during batch search, so intersecting the hull and
    memory limits gives one contiguous interval of supported local batches.
    """

    phase: str
    training_points: tuple[tuple[int, int], ...]  # (local batch, sequence)
    max_phase_tokens: int
    max_kv_tokens: int
    kv_reserve_per_sequence: int

    def __post_init__(self):
        if self.phase not in ("prefill", "decode"):
            raise ValueError("Unknown attention profile phase")
        if any(
            type(v) is not int or v <= 0
            for v in (self.max_phase_tokens, self.max_kv_tokens)
        ) or (
            type(self.kv_reserve_per_sequence) is not int
            or self.kv_reserve_per_sequence < 0
        ):
            raise ValueError("Attention profile capacity must be explicit integers")
        if (
            len(self.training_points) < 3
            or any(
                len(point) != 2 or any(type(v) is not int or v <= 0 for v in point)
                for point in self.training_points
            )
            or len(set(self.training_points)) != len(self.training_points)
        ):
            raise ValueError("Need unique positive integer attention training shapes")
        for batch, sequence in self.training_points:
            if batch > self.capacity_batch(sequence):
                raise ValueError("Attention training shape exceeds profiled capacity")
        if len(self.hull()) < 3:
            raise ValueError("Attention training shapes must span batch and sequence")

    @classmethod
    def from_dict(cls, data, phase):
        if not isinstance(data, dict) or set(data) != {
            "training_points",
            "max_phase_tokens",
            "max_kv_tokens",
            "kv_reserve_per_sequence",
        }:
            raise ValueError(
                "Measured attention_domain with training shapes and capacity is required"
            )
        points = data["training_points"]
        if not isinstance(points, (list, tuple)) or any(
            not isinstance(p, (list, tuple)) for p in points
        ):
            raise ValueError("Invalid attention training shapes")
        return cls(
            phase,
            tuple(tuple(p) for p in points),
            data["max_phase_tokens"],
            data["max_kv_tokens"],
            data["kv_reserve_per_sequence"],
        )

    def capacity_batch(self, sequence):
        return min(
            self.max_phase_tokens // (sequence if self.phase == "prefill" else 1),
            self.max_kv_tokens // (sequence + self.kv_reserve_per_sequence),
        )

    @lru_cache(maxsize=32)
    def hull(self):
        points = sorted((s, b) for b, s in self.training_points)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        def half(values):
            result = []
            for point in values:
                while len(result) >= 2 and cross(result[-2], result[-1], point) <= 0:
                    result.pop()
                result.append(point)
            return result

        return tuple(half(points)[:-1] + half(reversed(points))[:-1])

    def batch_bounds(self, sequence):
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("Attention sequence length must be a positive integer")
        hull = self.hull()
        if not min(p[0] for p in hull) <= sequence <= max(p[0] for p in hull):
            # A batch search cannot repair missing sequence coverage.
            raise ProfileCoverageError(
                "Sequence is outside attention training coverage"
            )
        intersections = []
        for a, b in zip(hull, hull[1:] + hull[:1]):
            if min(a[0], b[0]) <= sequence <= max(a[0], b[0]):
                if a[0] == b[0]:
                    intersections.extend((Fraction(a[1]), Fraction(b[1])))
                else:
                    intersections.append(
                        Fraction(a[1])
                        + Fraction((sequence - a[0]) * (b[1] - a[1]), b[0] - a[0])
                    )
        if not intersections:
            raise ProfileCoverageError("No attention shape support at sequence")
        return (
            math.ceil(min(intersections)),
            min(math.floor(max(intersections)), self.capacity_batch(sequence)),
        )

    def validate(self, batch, sequence):
        if type(batch) is not int or batch <= 0:
            raise ValueError("Attention local batch must be a positive integer")
        lower, upper = self.batch_bounds(sequence)
        if not lower <= batch <= upper:
            raise ProfileCoverageError(
                "Batch is outside measured attention shape/capacity support",
                below=batch < lower,
                above=batch > upper,
            )


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
    prefill_preprocess: bool = False
    probabilities_by_layer: dict[str, tuple[tuple[float, ...], ...]] | None = None
    components_by_world: dict[int, ComponentProfiles] | None = None
    attention_domains: dict[str, AttentionDomain] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], model: ModelSpec) -> MeasuredProfiles:
        preprocess = data.get("prefill_preprocess", False)
        if not isinstance(preprocess, bool):
            raise ValueError("profile prefill_preprocess must be boolean")
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
        components = {}
        for key, value in data.get("moe_components_by_expert_count", {}).items():
            world = int(key)
            if str(world) != str(key) or world in components:
                raise ValueError("Ambiguous expert world size")
            component = ComponentProfiles.from_dict(value)
            g = component.geometry
            if (
                g.world != world
                or g.experts * world != model.num_experts
                or g.hidden != model.hidden_size
                or g.intermediate != model.intermediate_size
                or g.top_k != model.top_k
                or not preprocess
                or max(s for budgets in component.models.values() for s in budgets)
                > data["total_sms"]
            ):
                raise ValueError(
                    "Component geometry differs from checkpoint/transport/GPU"
                )
            components[world] = component
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
        if any(
            phase_tiles[phase] != getattr(c.geometry, phase + "_gemm_tile_rows")
            or data["total_sms"] % c.geometry.cluster_size
            for c in components.values()
            for phase in ("prefill", "decode")
        ):
            raise ValueError("Tile/SM metadata differs from component geometry")
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
        by_layer = {}
        domains = {}
        for phase in ("prefill", "decode"):
            phase_data = data[phase]
            domains[phase] = AttentionDomain.from_dict(
                phase_data.get("attention_domain"), phase
            )
            attention[phase] = LatencyCoefficients(**phase_data["attention"])
            legacy = phase_data.get("moe_by_sm_count", {})
            if not legacy and not components:
                raise ValueError("Measured MoE profiles are required")
            if legacy and components:
                raise ValueError("Do not mix aggregate and component MoE profiles")
            for budget, coefficients in legacy.items():
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
            if "probabilities_by_layer" in phase_data:
                layers = tuple(
                    tuple(row) for row in phase_data["probabilities_by_layer"]
                )
                if len(layers) != model.num_layers or any(
                    len(row) != model.num_experts
                    or any(
                        isinstance(p, bool)
                        or not isinstance(p, (float, int))
                        or not math.isfinite(p)
                        or not 0 <= p <= 1
                        for p in row
                    )
                    or not math.isclose(
                        sum(row), model.top_k, rel_tol=1e-6, abs_tol=1e-6
                    )
                    for row in layers
                ):
                    raise ValueError(
                        "Need valid routing probabilities for every model layer"
                    )
                if any(
                    not math.isclose(
                        sum(row[e] for row in layers) / model.num_layers,
                        routing[e],
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                    for e in range(model.num_experts)
                ):
                    raise ValueError(
                        "Phase routing probabilities must match the layer mean"
                    )
                by_layer[phase] = layers
        return cls(
            provenance,
            data["total_sms"],
            data["tile_rows"],
            bandwidth,
            attention,
            moe,
            probabilities,
            preprocess,
            by_layer or None,
            components or None,
            domains,
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
    expected_decode_tiles: int
    component_runtime_source_sha256: str | None = None
    prefill_sm_bracket: tuple[int, int] | None = None
    decode_sm_bracket: tuple[int, int] | None = None

    def config(self, source: NodeConfig) -> NodeConfig:
        """Export the selected placement and contention policy, not the input."""
        overrides = {
            key: value
            for key, value in source.to_dict().items()
            if key not in ("model_path", "clients", "expert_devices")
        }
        overrides.update(
            overlap="tbo" if self.overlap == "tbo" else "none",
            decode_sm_budget=self.decode_sm_budget,
            expected_decode_tiles=self.expected_decode_tiles,
        )
        if self.overlap != "tbo":
            overrides["prefill_mailbox_tokens"] = None
        return self.layout.config(source.model_path, **overrides)


def expected_rank_footprints(profiles, layout, phase, batch, input_length, overlap):
    """Maximum expected EP-rank footprint per measured layer.

    The analytical TBO case assumes balanced token microbatches. Actual prefill
    sequence-boundary splits can differ; this remains a profile-model assumption.
    A legacy phase-average profile returns one representative layer. IID routing
    and max-of-expectations remain approximations, even with per-layer profiles.
    """
    attention_count = (
        layout.prefill_count if phase == "prefill" else layout.decode_count
    )
    mailbox_tokens = []
    for i in range(attention_count):
        tokens = (batch // attention_count + (i < batch % attention_count)) * (
            input_length if phase == "prefill" else 1
        )
        mailbox_tokens.extend(
            (tokens // 2, tokens - tokens // 2) if overlap == "tbo" else (tokens,)
        )
    layers = (profiles.probabilities_by_layer or {}).get(
        phase, (profiles.expert_probabilities[phase],)
    )
    tile_rows = (
        profiles.tile_rows[phase]
        if isinstance(profiles.tile_rows, dict)
        else profiles.tile_rows
    )
    return tuple(
        max(
            sum(
                expected_expert_tiles(tokens, probability, tile_rows)
                for tokens in mailbox_tokens
                if tokens
                for probability in probabilities[rank :: layout.expert_count]
            )
            for rank in range(layout.expert_count)
        )
        for probabilities in layers
    )


def expected_rank_footprint(profiles, layout, phase, batch, input_length, overlap):
    """Layer-average expected footprint for the adaptive SM policy's denominator."""
    values = expected_rank_footprints(
        profiles, layout, phase, batch, input_length, overlap
    )
    return sum(values) / len(values)


def nvlink_payload_bytes(
    model: ModelSpec,
    layout: Layout,
    phase: str,
    tokens: int,
    *,
    prefill_preprocess: bool,
) -> int:
    """Payload upper bound for single-node transport, excluding controls.

    Raw P sends IDs/weights and pulls one BF16 partial vector per E. Its hidden
    vector is only sent to ranks with selected experts; this conservative bound
    assumes every E is selected. Packed P and masked D send and retrieve one
    vector per valid route (the bound assumes no padding). This is not measured
    link utilization or a cross-node deduplication model.
    """
    if phase not in ("prefill", "decode") or tokens < 0:
        raise ValueError("Invalid phase or token count")
    if phase == "prefill" and prefill_preprocess:
        return tokens * layout.expert_count * (4 * model.hidden_size + 8 * model.top_k)
    return 4 * tokens * model.top_k * model.hidden_size


def component_layer_latencies(
    profiles, layout, phase, batch, input_length, sms, overlap
):
    """Per-layer max of per-rank sums of separately executed client operations.

    Routing remains the existing IID-marginal approximation. In particular, a
    positive expected footprint charges one operation intercept; this does not
    infer the exact probability of a rank receiving no route from marginals.
    Raw preprocessing/gather work uses the client's tokens, not routed tokens.
    """
    component = profiles.components_by_world[layout.expert_count]
    geometry = component.geometry
    component.bracket(phase, sms)  # Fail before any workload-dependent search.
    if geometry.world != layout.expert_count:
        raise ValueError("Component profile has a different expert placement")
    clients = layout.prefill_count if phase == "prefill" else layout.decode_count
    mailboxes = []
    for i in range(clients):
        tokens = (batch // clients + (i < batch % clients)) * (
            input_length if phase == "prefill" else 1
        )
        mailboxes.extend(
            (tokens // 2, tokens - tokens // 2) if overlap == "tbo" else (tokens,)
        )
    layers = (profiles.probabilities_by_layer or {}).get(
        phase, (profiles.expert_probabilities[phase],)
    )
    result, errors = [], []
    for probabilities in layers:
        if len(probabilities) != geometry.world * geometry.experts:
            raise ValueError("Routing and measured expert geometry differ")
        ranks = [0.0] * geometry.world
        for tokens in mailboxes:
            if not tokens:
                continue
            data = geometry.features(phase, tokens, [0] * geometry.experts)
            for rank in range(geometry.world):
                local = probabilities[rank :: geometry.world]
                tile = getattr(geometry, phase + "_gemm_tile_rows")
                gemm = sum(expected_expert_tiles(tokens, p, tile) for p in local)
                activation = sum(
                    expected_expert_tiles(tokens, p, geometry.activation_tile_rows)
                    for p in local
                )
                footprint = dict(data, **{"1": gemm, "2": gemm, "3": activation})
                try:
                    ranks[rank] += sum(
                        component.predict_footprints(phase, sms, footprint).values()
                    )
                except ProfileCoverageError as error:
                    errors.append(error)
        result.append(max(ranks))
    if errors:
        raise ProfileCoverageError(
            "Client/rank workload is outside component profile coverage",
            below=any(e.below for e in errors),
            above=any(e.above for e in errors),
        )
    return tuple(result)


def max_feasible_batch(maximum, slo, latency):
    """Binary search a monotone latency within a contiguous measured domain.

    A below-coverage probe moves right, an above-coverage probe moves left.
    Only an actually evaluated in-domain SLO pass can become the result.
    Nonnegative component fits and monotone footprints provide the contract.
    """
    if type(maximum) is not int or maximum < 0 or not math.isfinite(slo) or slo <= 0:
        raise ValueError("Expected a nonnegative capacity and positive finite SLO")
    lo, hi, best = 1, maximum, 0
    while lo <= hi:
        middle = (lo + hi) // 2
        try:
            value = latency(middle)
        except ProfileCoverageError as error:
            if error.below and error.above:
                return best  # The component domains have no common batch.
            if error.below:
                lo = middle + 1
            elif error.above:
                hi = middle - 1
            else:
                raise  # Missing SM/geometry coverage is not a large batch.
            continue
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid predicted latency")
        if value <= slo:
            best, lo = middle, middle + 1
        else:
            hi = middle - 1
    return best


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
    prefill_preprocess: bool = False,
) -> float:
    attention_count = (
        layout.prefill_count if phase == "prefill" else layout.decode_count
    )
    local_batch = math.ceil(batch / attention_count)
    sequence = input_length if phase == "prefill" else context_length
    if not profiles.attention_domains or phase not in profiles.attention_domains:
        raise ProfileCoverageError("Measured attention shape coverage is required")
    profiles.attention_domains[phase].validate(local_batch, sequence)
    attention = profiles.attention[phase].predict(local_batch, sequence)
    tokens = batch * input_length if phase == "prefill" else batch
    # Mailboxes belong to independent attention clients. Their partial tiles
    # cannot be merged into a single global-batch tile without changing the
    # runtime. Sum each client's tiles on every expert rank, then take max rank.
    if profiles.components_by_world:
        moe = component_layer_latencies(
            profiles, layout, phase, batch, input_length, sm_count, overlap
        )
    else:
        footprints = expected_rank_footprints(
            profiles, layout, phase, batch, input_length, overlap
        )
        moe = [profiles.moe[phase, sm_count].predict(x) for x in footprints]
    # Match the selected transport's actual duplication boundary. Bandwidth
    # must be measured for that path; launch/control overhead is not inferred
    # from payload bytes and remains a limitation of this approximation.
    communication = (
        nvlink_payload_bytes(
            model, layout, phase, tokens, prefill_preprocess=prefill_preprocess
        )
        / profiles.nvlink_bytes_per_second
    )
    layers = [
        max(attention, latency + communication)
        if overlap == "tbo"
        else attention + latency + communication
        for latency in moe
    ]
    return sum(layers) / len(layers) * model.num_layers


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
    if profiles.prefill_preprocess != config.prefill_preprocess:
        raise ValueError(
            "profile prefill_preprocess differs from the selected transport; "
            "measure both E preprocessing and its raw-token communication path"
        )
    if profiles.components_by_world and (
        config.cta_count not in (0, profiles.total_sms)
        or any(client.tp_size != 1 for client in config.clients)
    ):
        raise ValueError(
            "Component search requires matching total SMs and TP1 attention profiles"
        )
    if overlap == "auto":
        candidates = [
            search(
                config,
                model,
                profiles,
                input_length=input_length,
                average_output_length=average_output_length,
                context_length=context_length,
                prefill_slo_s=prefill_slo_s,
                decode_slo_s=decode_slo_s,
                kv_tokens_per_attention_gpu=kv_tokens_per_attention_gpu,
                reserve_gib=reserve_gib,
                overlap=choice,
            )
            for choice in ("sequential", "tbo")
        ]
        return max(
            (c for c in candidates if c is not None),
            key=lambda c: c.predicted_goodput_requests_per_second,
            default=None,
        )
    if overlap not in ("sequential", "tbo"):
        raise ValueError("overlap must be sequential, tbo or auto")
    # FitsMemory must account for the strategy being evaluated: TBO reserves
    # two complete mailboxes even when the input config uses sequential mode.
    config = replace(
        config,
        overlap="tbo" if overlap == "tbo" else "none",
        prefill_mailbox_tokens=config.prefill_mailbox_tokens
        if overlap == "tbo"
        else None,
    )
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
        if not profiles.attention_domains or phase not in profiles.attention_domains:
            raise ProfileCoverageError("Measured attention shape coverage is required")
        profiles.attention_domains[phase].batch_bounds(sequence)
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
        if profiles.components_by_world:
            if layout.expert_count not in profiles.components_by_world:
                raise ProfileCoverageError(
                    f"Missing component profiles for memory-feasible {layout.name}; cannot declare a global optimum"
                )
            component = profiles.components_by_world[layout.expert_count]
            cluster = component.geometry.cluster_size
            geometry = component.geometry
            if (
                config.kernel != "sm100"
                or cluster != 2
                or geometry.prefill_gemm_tile_rows != 128
                or geometry.decode_gemm_tile_rows != 128
                or geometry.activation_tile_rows != 32
            ):
                raise ValueError(
                    "Component planner currently describes the complete two-CTA SM100 pipeline"
                )
            budgets = []
            for q in range(cluster, profiles.total_sms, cluster):
                if q > profiles.total_sms * config.max_decode_sm_fraction:
                    continue
                try:
                    component.bracket("prefill", profiles.total_sms - q)
                    component.bracket("decode", q)
                except ProfileCoverageError:
                    continue
                budgets.append(q)
        else:
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
                low = max_feasible_batch(
                    count * maximum,
                    slo,
                    lambda batch: _iteration_latency(
                        model,
                        profiles,
                        layout,
                        phase,
                        batch,
                        input_length,
                        context_length,
                        sms,
                        overlap,
                        config.prefill_preprocess,
                    ),
                )
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
                        config.prefill_preprocess,
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
                max(
                    1,
                    math.ceil(
                        expected_rank_footprint(
                            profiles,
                            layout,
                            "decode",
                            batches["decode"],
                            input_length,
                            overlap,
                        )
                    ),
                ),
                component_runtime_source_sha256=component.runtime_source_sha256
                if profiles.components_by_world
                else None,
                prefill_sm_bracket=component.bracket("prefill", profiles.total_sms - q)
                if profiles.components_by_world
                else (profiles.total_sms - q, profiles.total_sms - q),
                decode_sm_bracket=component.bracket("decode", q)
                if profiles.components_by_world
                else (q, q),
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
        "--overlap", choices=("sequential", "tbo", "auto"), default="auto"
    )
    parser.add_argument(
        "--prefill-preprocess", action=argparse.BooleanOptionalAction, default=True
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
        overlap="tbo" if args.overlap == "tbo" else "none",
        prefill_preprocess=args.prefill_preprocess,
    )
    memory = estimate_memory(
        config,
        model,
        kv_tokens_per_attention_gpu=args.kv_tokens_per_attention_gpu,
        reserve_gib=args.reserve_gib,
    )
    if not memory.fits and not args.profiles:
        parser.error(
            "selected placement exceeds configured HBM capacity; lower fixed mailbox rows or choose more expert GPUs"
        )
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
        if result is not None:
            config = result.config(config)
            output["config"] = config.to_dict()
            output["memory"] = estimate_memory(
                config,
                model,
                kv_tokens_per_attention_gpu=args.kv_tokens_per_attention_gpu,
                reserve_gib=args.reserve_gib,
            ).to_dict()
        elif args.write_config:
            parser.error("No feasible measured-profile candidate; no config written")
        output["prediction_note"] = (
            "Profile-backed estimate only; this is not measured end-to-end goodput."
        )
    else:
        output["prediction_note"] = (
            "No measured profiles supplied: no latency or goodput prediction made."
        )
    if args.write_config:
        config.write_json(args.write_config)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
