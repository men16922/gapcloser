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
    tipped: bool = False  # cube rolled over instead of sliding (outside the sliding model)

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
                 "track": t.track, "tipped": t.tipped}
                for t in self.trials
            ],
        }


PATCH_OFF = 0.999  # patch_y0 at or beyond this means the table has no second friction region
FULL_TRACK_FRAMES = 75  # whole slide at 30 fps (2.5 s), for position-resolved evidence


def observe(target: float, p: ParamSet) -> float:
    """Camera error: lateral offset adds a bias, pitch error scales perceived depth, lens distortion
    adds an error growing with the square of range."""
    return target * (1.0 + math.tan(math.radians(p["camera_pitch_deg"]))) + p["camera_dx"] + p["lens_k"] * target * target


def unobserve(observed: float, p: ParamSet) -> float:
    """Inverse of `observe`: the true distance a policy trained in this sim believes it is looking at."""
    s, k = 1.0 + math.tan(math.radians(p["camera_pitch_deg"])), p["lens_k"]
    r = observed - p["camera_dx"]
    if abs(k) < 1e-9:
        return r / s
    disc = s * s + 4.0 * k * r
    return (-s + math.sqrt(disc)) / (2.0 * k) if disc >= 0 else -s / (2.0 * k)


def frictions(p: ParamSet) -> tuple[float, float | None, float]:
    """(mu_eff near, mu_eff beyond the patch or None, patch start)."""
    mu1 = effective_friction(p["object_mu"], p["table_mu"])
    if p["patch_y0"] >= PATCH_OFF:
        return mu1, None, float("inf")
    return mu1, p["patch_mu"], p["patch_y0"]


def slide_distance(command: float, p: ParamSet) -> float:
    v = command * p["actuator_gain"]
    mu1, mu2, y0 = frictions(p)
    d1 = v * v / (2.0 * mu1 * GRAVITY)
    if mu2 is None or d1 <= y0:
        return d1
    return y0 + (v * v - 2.0 * mu1 * GRAVITY * y0) / (2.0 * mu2 * GRAVITY)


def launch_command(target: float, p: ParamSet) -> float:
    """Exact inverse of slide_distance: the command that stops the cube at `target` in world p."""
    mu1, mu2, y0 = frictions(p)
    target = max(0.0, target)
    v2 = 2.0 * mu1 * GRAVITY * target if (mu2 is None or target <= y0) else 2.0 * GRAVITY * (mu1 * y0 + mu2 * (target - y0))
    return math.sqrt(v2) / p["actuator_gain"]


def analytic_track(command: float, p: ParamSet, frames: int = FULL_TRACK_FRAMES) -> list[float]:
    """Cube position at each camera frame (piecewise-constant deceleration across the patch)."""
    v0 = command * p["actuator_gain"]
    mu1, mu2, y0 = frictions(p)
    a1 = mu1 * GRAVITY
    d1 = v0 * v0 / (2.0 * a1)
    if mu2 is None or d1 <= y0:
        t_stop = v0 / a1
        return [v0 * min(k * FRAME_DT, t_stop) - 0.5 * a1 * min(k * FRAME_DT, t_stop) ** 2 for k in range(frames + 1)]
    a2 = mu2 * GRAVITY
    t1 = (v0 - math.sqrt(v0 * v0 - 2.0 * a1 * y0)) / a1  # reaches the patch
    v1 = v0 - a1 * t1
    t_stop = t1 + v1 / a2
    out = []
    for k in range(frames + 1):
        t = min(k * FRAME_DT, t_stop)
        out.append(v0 * t - 0.5 * a1 * t * t if t <= t1 else y0 + v1 * (t - t1) - 0.5 * a2 * (t - t1) ** 2)
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

    def push(self, params: ParamSet, commands: list[float]) -> Rollout:
        """Raw pushes with chosen commands (probe experiments); target/observed are NaN."""
        nan = float("nan")
        return Rollout([Trial(nan, nan, c, slide_distance(c, params), analytic_track(c, params)) for c in commands])


def _success_rate(p: ParamSet, policy: Policy, targets: list[float]) -> float:
    """AnalyticPushEnv.rollout(...).success_rate without building tracks (policy search hot path)."""
    return sum(abs(slide_distance(policy.command(observe(d, p)), p) - d) <= SUCCESS_TOL for d in targets) / len(targets)


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
            scored.append((sum(_success_rate(w, pol, targets) for w in worlds) / len(worlds), c))
        best = max(s for s, _ in scored)
        # middle of the best plateau, not its first edge (an edge biases every push short)
        plateau = [c for s, c in scored if s >= best - 1e-12]
        return Policy(plateau[len(plateau) // 2])


@dataclass(frozen=True)
class TablePolicy:
    """Command lookup over the perceived distance (linear interpolation). Produced by InverseTrainer:
    a policy that is exactly as good as the simulator it was trained in, including position-dependent
    friction and lens distortion."""

    observed: tuple[float, ...]
    commands: tuple[float, ...]

    @property
    def c(self) -> float:
        """Equivalent constant of the closed-form policy v = sqrt(c * d) (for display/back-compat)."""
        mid = len(self.observed) // 2
        return self.commands[mid] ** 2 / self.observed[mid]

    def command(self, observed_distance: float) -> float:
        xs, ys = self.observed, self.commands
        if observed_distance <= xs[0]:
            return ys[0] * math.sqrt(max(0.0, observed_distance) / xs[0])
        if observed_distance >= xs[-1]:
            return ys[-1] * math.sqrt(observed_distance / xs[-1])
        i = min(int((observed_distance - xs[0]) / (xs[1] - xs[0])), len(xs) - 2)
        u = (observed_distance - xs[i]) / (xs[i + 1] - xs[i])
        return ys[i] + u * (ys[i + 1] - ys[i])


class InverseTrainer:
    """Policy training by inverting the (randomized) simulator: for each perceived distance, the median
    over sampled sim worlds of the command that lands the cube on the distance that world believes it sees.
    Deterministic for a given seed; counts calls like GridTrainer."""

    GRID = tuple(round(0.05 + 0.005 * i, 4) for i in range(171))  # 0.05..0.90 m perceived

    def __init__(self, n_worlds: int = 31, seed: int = 0):
        self.n_worlds, self.seed = n_worlds, seed
        self.calls = 0

    def train(self, rand: Randomization) -> TablePolicy:
        self.calls += 1
        rng = random.Random(self.seed)
        fixed = all(hi <= lo for lo, hi in rand.ranges.values())
        worlds = [rand.sample(rng) for _ in range(1 if fixed else self.n_worlds)]
        cmds = []
        for o in self.GRID:
            per = sorted(launch_command(unobserve(o, w), w) for w in worlds)
            cmds.append(per[len(per) // 2])
        return TablePolicy(self.GRID, tuple(cmds))
