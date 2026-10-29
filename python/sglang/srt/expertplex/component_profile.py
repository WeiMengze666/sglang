"""Measured per-operation APK latency, with explicit geometry and coverage.

This is an isolated-rank compute model. It does not model attention, transport,
cross-phase contention, or TBO overlap. Its predictions are not serving results.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence


STAGES = {"prefill": ("5", "6", "1", "3", "2", "7"), "decode": ("1", "3", "2")}


class ProfileCoverageError(ValueError):
    """A request is outside the measured domain; never silently extrapolate."""

    def __init__(self, message, *, below=False, above=False):
        super().__init__(message)
        self.below = below
        self.above = above


@dataclass(frozen=True)
class ComponentGeometry:
    hidden: int
    intermediate: int
    experts: int
    world: int
    top_k: int
    cluster_size: int
    prefill_gemm_tile_rows: int
    decode_gemm_tile_rows: int
    activation_tile_rows: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError("Geometry must contain positive integers")
        if self.cluster_size not in (1, 2) or self.top_k > self.experts * self.world:
            raise ValueError("Invalid cluster size or global expert count")

    def features(self, phase: str, tokens: int, counts: Sequence[int]):
        if phase not in STAGES:
            raise ValueError("Expected prefill or decode")
        if (
            type(tokens) is not int
            or tokens <= 0
            or len(counts) != self.experts
            or any(type(n) is not int or not 0 <= n <= tokens for n in counts)
            or sum(counts) > tokens * self.top_k
        ):
            raise ValueError("Expected valid local expert counts for one mailbox")
        tile = getattr(self, phase + "_gemm_tile_rows")
        gemm = sum((n + tile - 1) // tile for n in counts)
        act = sum(
            (n + self.activation_tile_rows - 1) // self.activation_tile_rows
            for n in counts
        )
        return {
            "1": gemm,
            "2": gemm,
            "3": act,
            "5": (tokens * self.top_k + 256 * self.cluster_size - 1)
            // (256 * self.cluster_size),
            "6": (tokens * self.top_k + 32 * self.cluster_size - 1)
            // (32 * self.cluster_size),
            "7": (tokens + 4 * self.cluster_size - 1) // (4 * self.cluster_size),
            "gap": 0,
        }


@dataclass(frozen=True)
class ComponentLine:
    alpha: float
    beta: float
    minimum: float
    maximum: float

    def __post_init__(self):
        if (
            any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v < 0
                for v in vars(self).values()
            )
            or self.minimum > self.maximum
        ):
            raise ValueError("Invalid nonnegative component fit or coverage")

    def predict(self, x: float, *, gap: bool = False):
        if (
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x < 0
        ):
            raise ValueError("Footprint must be finite and nonnegative")
        # A skipped operation incurs no operation intercept. The whole-mailbox
        # initialization/completion gap still applies to a zero-routed rank.
        if x == 0 and not gap:
            return 0.0
        if not self.minimum <= x <= self.maximum:
            raise ProfileCoverageError(
                f"Footprint {x} outside [{self.minimum}, {self.maximum}]",
                below=x < self.minimum,
                above=x > self.maximum,
            )
        value = self.alpha + self.beta * x
        if not math.isfinite(value):
            raise ValueError("Nonfinite component prediction")
        return value


@dataclass(frozen=True)
class ComponentProfiles:
    geometry: ComponentGeometry
    models: Mapping[str, Mapping[int, Mapping[str, ComponentLine]]]
    provenance: str = ""
    runtime_source_sha256: str = ""

    @classmethod
    def from_dict(cls, data):
        if data.get("units") != "seconds_per_layer":
            raise ValueError("Component profile units must be seconds_per_layer")
        if (
            not isinstance(data.get("provenance"), str)
            or not data["provenance"].strip()
        ):
            raise ValueError("Measured component provenance is required")
        fingerprint = data.get("runtime_source_sha256", "")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)
        ):
            raise ValueError("Component runtime source fingerprint is required")
        models = {}
        for phase, budgets in data["models"].items():
            models[phase] = {}
            for key, values in budgets.items():
                sms = int(key)
                if str(sms) != str(key) or sms in models[phase]:
                    raise ValueError("Ambiguous SM budget")
                models[phase][sms] = {}
                for stage, line in values.items():
                    if line.get("gamma", 0) != 0 or line.get("delta", 0) != 0:
                        raise ValueError("Component fits use Eq2 at s=1: alpha+beta*x")
                    models[phase][sms][stage] = ComponentLine(
                        line["alpha"], line["beta"], *line["x_range"]
                    )
        return cls(
            ComponentGeometry(**data["geometry"]),
            models,
            data["provenance"],
            fingerprint,
        )

    def __post_init__(self):
        if set(self.models) != set(STAGES):
            raise ValueError("Both phase profiles are required")
        for phase, budgets in self.models.items():
            if not budgets:
                raise ValueError("Empty SM profile")
            for sms, components in budgets.items():
                if type(sms) is not int or sms <= 0 or sms % self.geometry.cluster_size:
                    raise ValueError("SM budgets must contain complete CTA clusters")
                if set(components) != set(STAGES[phase]) | {"gap"}:
                    raise ValueError("Missing or extra operation profile")
                if any(not isinstance(c, ComponentLine) for c in components.values()):
                    raise ValueError("Expected validated component coefficients")
                gap = components["gap"]
                if gap.beta != 0 or gap.minimum != 0 or gap.maximum != 0:
                    raise ValueError(
                        "Transition gap must be a separately fitted constant"
                    )
        object.__setattr__(
            self,
            "models",
            MappingProxyType(
                {
                    p: MappingProxyType(
                        {s: MappingProxyType(dict(c)) for s, c in budgets.items()}
                    )
                    for p, budgets in self.models.items()
                }
            ),
        )

    def bracket(self, phase: str, sms: int):
        if (
            phase not in STAGES
            or type(sms) is not int
            or sms <= 0
            or sms % self.geometry.cluster_size
        ):
            raise ValueError("Expected valid phase and complete SM clusters")
        budgets = sorted(self.models[phase])
        i = bisect.bisect_left(budgets, sms)
        if i < len(budgets) and budgets[i] == sms:
            return sms, sms
        if i == 0 or i == len(budgets):
            raise ProfileCoverageError("SM extrapolation is forbidden")
        return budgets[i - 1], budgets[i]

    def predict_components(
        self, phase: str, sms: int, tokens: int, counts: Sequence[int]
    ):
        footprint = self.geometry.features(phase, tokens, counts)
        return self.predict_footprints(phase, sms, footprint)

    def predict_footprints(self, phase: str, sms: int, footprint):
        """Also accept expected tile counts from the planner's routing model.

        Fractional expected footprints retain that model's IID approximation;
        their evaluation is not validation of a production routing distribution.
        """
        lo, hi = self.bracket(phase, sms)
        weight = 0 if lo == hi else (1 / sms - 1 / lo) / (1 / hi - 1 / lo)
        predictions = {}
        errors = []
        for stage in (*STAGES[phase], "gap"):
            sides = []
            for budget in (lo, hi):
                try:
                    sides.append(
                        self.models[phase][budget][stage].predict(
                            footprint[stage], gap=stage == "gap"
                        )
                    )
                except ProfileCoverageError as error:
                    errors.append(error)
            if len(sides) == 2:
                predictions[stage] = (1 - weight) * sides[0] + weight * sides[1]
        if errors:
            raise ProfileCoverageError(
                "Operation footprints exceed measured coverage",
                below=any(e.below for e in errors),
                above=any(e.above for e in errors),
            )
        return predictions

    def rank_latencies(self, phase: str, sms: int, mailboxes):
        """Sum separate client operations per E rank, retaining each intercept.

        Each mailbox is (tokens, counts_by_rank). Zero-token clients are omitted
        by the caller. Do not merge tiles across independently executed clients.
        The caller composes these ranks with the actual attention/transport DAG.
        """
        total = [0.0] * self.geometry.world
        for tokens, counts_by_rank in mailboxes:
            if len(counts_by_rank) != self.geometry.world:
                raise ValueError("Routing must cover every expert rank")
            if sum(sum(c) for c in counts_by_rank) != tokens * self.geometry.top_k:
                raise ValueError("Global routes must equal tokens times top_k")
            for rank, counts in enumerate(counts_by_rank):
                total[rank] += sum(
                    self.predict_components(phase, sms, tokens, counts).values()
                )
        return tuple(total)
