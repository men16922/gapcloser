"""Tier 0 task: push a cube so it stops on a target line. Analytic env (offline, deterministic).

The Newton env (later) implements the same `rollout` contract; this one is the fast reference used
by the gate and by the agent-loop tests. Physics: Coulomb sliding, stop distance v^2 / (2 mu g).
Perception: the target distance is *observed* through a camera whose pose error biases it.

Distractors on purpose: density, size, restitution and light do not change the outcome here, so a
diagnoser that blames them is measurably wrong.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from sim.params import GRAVITY, ParamSet, Randomization, effective_friction

SUCCESS_TOL = 0.03  # m
TARGET_RANGE = (0.2, 0.6)  # m
FRAME_DT = 1.0 / 30.0  # camera frame period for tracking
TRACK_FRAMES = 6  # first frames of cube position seen by the tracker


@dataclass(frozen=True)
class Policy:
    """v = sqrt(c * observed_distance). Training picks c; c = 2*mu_eff*g/gain^2 is exact for one world."""

    c: float

    def command(self, observed_distance: float) -> float:
        return math.sqrt(max(0.0, self.c * observed_distance))


@dataclass
class Trial:
    target: float
    observed: float
    command: float
    slide: float
    track: list[float] = field(default_factory=list)  # cube y at frames 0..TRACK_FRAMES (camera tracking)

    @property
    def success(self) -> bool:
        return abs(self.slide - self.target) <= SUCCESS_TOL


@dataclass
class Rollout:
    trials: list[Trial]

    @property
    def success_rate(self) -> float:
        return sum(t.success for t in self.trials) / len(self.trials)

    def to_json(self) -> dict:
        return {
            "success_rate": self.success_rate,
            "trials": [
                {"target": t.target, "observed": t.observed, "command": t.command, "slide": t.slide, "success": t.success,
                 "track": t.track}
                for t in self.trials
            ],
        }


def observe(target: float, p: ParamSet) -> float:
    """Camera pose error: lateral offset adds a bias, pitch error scales perceived depth."""
    return target * (1.0 + math.tan(math.radians(p["camera_pitch_deg"]))) + p["camera_dx"]


def slide_distance(command: float, p: ParamSet) -> float:
    v = command * p["actuator_gain"]
    mu = effective_friction(p["object_mu"], p["table_mu"])
    return v * v / (2.0 * mu * GRAVITY)


def analytic_track(command: float, p: ParamSet) -> list[float]:
    v = command * p["actuator_gain"]
    a = effective_friction(p["object_mu"], p["table_mu"]) * GRAVITY
    t_stop = v / a if a > 0 else float("inf")
    out = []
    for k in range(TRACK_FRAMES + 1):
        t = min(k * FRAME_DT, t_stop)
        out.append(v * t - 0.5 * a * t * t)
    return out


def eval_targets(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(*TARGET_RANGE) for _ in range(n)]


class AnalyticPushEnv:
    def rollout(self, params: ParamSet, policy: Policy, targets: list[float]) -> Rollout:
        trials = []
        for d in targets:
            obs = observe(d, params)
            cmd = policy.command(obs)
            trials.append(Trial(d, obs, cmd, slide_distance(cmd, params), analytic_track(cmd, params)))
        return Rollout(trials)


class GridTrainer:
    """Stand-in for policy training: pick c maximizing mean success over randomized sim worlds.

    Deterministic for a given seed. Counts calls so the eval can compare training budgets.
    """

    def __init__(self, env: AnalyticPushEnv, n_worlds: int = 64, n_targets: int = 8, seed: int = 0):
        self.env, self.n_worlds, self.n_targets, self.seed = env, n_worlds, n_targets, seed
        self.calls = 0

    def train(self, rand: Randomization) -> Policy:
        self.calls += 1
        rng = random.Random(self.seed)
        worlds = [rand.sample(rng) for _ in range(self.n_worlds)]
        targets = eval_targets(self.n_targets, self.seed + 1)
        scored = []
        for i in range(321):
            c = 2.0 * 1.01**i  # 2.0..~48.6 covers 2*mu_eff*g/gain^2 over the whole PARAM_SPACE
            pol = Policy(c)
            scored.append((sum(self.env.rollout(w, pol, targets).success_rate for w in worlds) / len(worlds), c))
        best = max(s for s, _ in scored)
        # middle of the best plateau, not its first edge (an edge biases every push short)
        plateau = [c for s, c in scored if s >= best - 1e-12]
        return Policy(plateau[len(plateau) // 2])
