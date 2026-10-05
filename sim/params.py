"""Tier 0 parameter space: the physical/visual knobs that can differ between "sim" and hidden "real".

Pure Python (no Newton import) so the offline gate can test it. The Newton world builder reads a
ParamSet; the agent edits a Randomization (per-parameter sampling ranges) via ConfigDiff.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

GRAVITY = 9.81


@dataclass(frozen=True)
class PhysParam:
    name: str
    nominal: float
    low: float  # plausible physical bound (hidden values are drawn inside it)
    high: float
    unit: str
    kind: str  # "dynamics" | "perception"
    hint: str  # what a failure video would look like if this is off


PARAM_SPACE: dict[str, PhysParam] = {
    p.name: p
    for p in [
        PhysParam("object_mu", 0.8, 0.2, 1.2, "-", "dynamics", "cube slides farther/shorter than planned"),
        PhysParam("table_mu", 0.8, 0.2, 1.2, "-", "dynamics", "all objects slide differently on this surface"),
        PhysParam("object_density", 500.0, 200.0, 1500.0, "kg/m^3", "dynamics", "cube reacts sluggishly to contact"),
        PhysParam("object_half_size", 0.025, 0.018, 0.035, "m", "dynamics", "gripper closes early/late"),
        PhysParam("actuator_gain", 1.0, 0.7, 1.3, "x", "dynamics", "arm moves faster/slower than commanded"),
        PhysParam("restitution", 0.0, 0.0, 0.4, "-", "dynamics", "cube bounces on contact"),
        PhysParam("camera_dx", 0.0, -0.03, 0.03, "m", "perception", "consistent lateral miss of the target"),
        PhysParam("camera_dz", 0.0, -0.03, 0.03, "m", "perception", "approach too high/low"),
        PhysParam("camera_pitch_deg", 0.0, -5.0, 5.0, "deg", "perception", "distance misjudged, depth error grows with range"),
        PhysParam("light_intensity", 1.0, 0.4, 1.6, "x", "perception", "object darker/brighter than in sim"),
    ]
}


def effective_friction(object_mu: float, table_mu: float) -> float:
    """Newton XPBD combines contact friction as the arithmetic mean of the two shapes."""
    return 0.5 * (object_mu + table_mu)


@dataclass
class ParamSet:
    values: dict[str, float]

    @classmethod
    def nominal(cls) -> ParamSet:
        return cls({k: p.nominal for k, p in PARAM_SPACE.items()})

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def with_(self, **changes: float) -> ParamSet:
        unknown = set(changes) - set(PARAM_SPACE)
        if unknown:
            raise KeyError(f"unknown params: {sorted(unknown)}")
        return ParamSet({**self.values, **changes})

    def diff(self, other: ParamSet, rel_tol: float = 1e-9) -> dict[str, tuple[float, float]]:
        """Params whose values differ: name -> (self, other)."""
        out = {}
        for k in PARAM_SPACE:
            a, b = self.values[k], other.values[k]
            if abs(a - b) > rel_tol * max(1.0, abs(a), abs(b)):
                out[k] = (a, b)
        return out


def sample_hidden(rng: random.Random, n_perturbed: int, min_shift: float = 0.3) -> ParamSet:
    """Hidden "real" world: nominal with n params moved by >= min_shift of their half-range."""
    if not 0 < n_perturbed <= len(PARAM_SPACE):
        raise ValueError("n_perturbed out of range")
    chosen = rng.sample(sorted(PARAM_SPACE), n_perturbed)
    changes = {}
    for name in chosen:
        p = PARAM_SPACE[name]
        up_room, down_room = p.high - p.nominal, p.nominal - p.low
        go_up = down_room <= 0 or (up_room > 0 and rng.random() < 0.5)
        room = up_room if go_up else down_room
        mag = rng.uniform(min_shift, 1.0) * room
        changes[name] = p.nominal + mag if go_up else p.nominal - mag
    return ParamSet.nominal().with_(**changes)


@dataclass
class Randomization:
    """Per-parameter sampling range used when training a policy in sim. Point range = fixed value."""

    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def none(cls) -> Randomization:
        return cls({k: (p.nominal, p.nominal) for k, p in PARAM_SPACE.items()})

    @classmethod
    def full(cls) -> Randomization:
        """Baseline A: randomize every parameter over its whole plausible range."""
        return cls({k: (p.low, p.high) for k, p in PARAM_SPACE.items()})

    def sample(self, rng: random.Random) -> ParamSet:
        return ParamSet({k: rng.uniform(lo, hi) for k, (lo, hi) in self.ranges.items()})

    def apply(self, diff: ConfigDiff) -> Randomization:
        new = dict(self.ranges)
        for name, (lo, hi) in diff.set_ranges.items():
            p = PARAM_SPACE[name]
            lo, hi = max(p.low, min(lo, hi)), min(p.high, max(lo, hi))
            new[name] = (lo, hi)
        return Randomization(new)


@dataclass
class ConfigDiff:
    """What the agent changes in the simulator config for the next iteration."""

    set_ranges: dict[str, tuple[float, float]]
    rationale: str = ""

    def validate(self) -> None:
        unknown = set(self.set_ranges) - set(PARAM_SPACE)
        if unknown:
            raise KeyError(f"unknown params: {sorted(unknown)}")
