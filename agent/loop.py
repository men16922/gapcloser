"""Tether loop: train in sim -> run in "real" -> diagnose the gap -> edit sim config -> repeat.

The loop never sees hidden params: "real" is only reachable through RealWorld.rollout (measured
success, never LLM-estimated). Diagnoser/Planner are pluggable; the heuristic ones here are the
offline reference. The Nemotron/Cosmos implementations satisfy the same Protocols.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from sim.params import PARAM_SPACE, ConfigDiff, ParamSet, Randomization, effective_friction
from sim.params import GRAVITY
from sim.push_task import FRAME_DT, SUCCESS_TOL, TRACK_FRAMES, AnalyticPushEnv, GridTrainer, Policy, Rollout


class RealWorld:
    """Black box for the agent. Hidden params stay private."""

    def __init__(self, env: AnalyticPushEnv, hidden: ParamSet):
        self._env, self._hidden = env, hidden
        self.trials_used = 0  # every real push counts: policy evaluations and probe experiments

    def rollout(self, policy: Policy, targets: list[float]) -> Rollout:
        self.trials_used += len(targets)
        return self._env.rollout(self._hidden, policy, targets)

    def push(self, commands: list[float]) -> Rollout:
        """Probe experiment: raw pushes with chosen commands (no target)."""
        self.trials_used += len(commands)
        return self._env.push(self._hidden, commands)


@dataclass
class Suspect:
    name: str
    direction: str  # "up" | "down" | "unknown"
    confidence: float
    estimate: float | None = None


@dataclass
class Diagnosis:
    summary: str
    scale_ratio: float  # real slide / sim slide (slope)
    offset: float  # real - scaled sim intercept, m
    suspects: list[Suspect] = field(default_factory=list)
    reasoning: str = ""  # natural-language rationale (LLM diagnosers)
    model: str = ""  # model id that produced it, empty for rule-based
    trace: list = field(default_factory=list)  # tool calls of agentic diagnosers (tool, args, result)


class Diagnoser(Protocol):
    def diagnose(self, real: Rollout, sim: Rollout, sim_params: ParamSet) -> Diagnosis: ...


class Planner(Protocol):
    def plan(self, diag: Diagnosis, rand: Randomization, sim_params: ParamSet) -> ConfigDiff: ...


def sliding(ro: Rollout) -> Rollout:
    """Trials where the cube slid (tipped-over cubes are outside the sliding model). Falls back to all
    trials when fewer than 3 remain, so fits stay defined."""
    keep = [t for t in ro.trials if not getattr(t, "tipped", False)]
    return Rollout(keep) if len(keep) >= 3 else ro


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return (my / mx if mx else 1.0), 0.0
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return a, my - a * mx


class HeuristicDiagnoser:
    """Numbers-only diagnoser: fits slide vs target. Cannot tell friction from gain from pitch (all
    scale the slide) -- that confound is exactly what visual evidence (Cosmos/Nemotron) should break."""

    def diagnose(self, real: Rollout, sim: Rollout, sim_params: ParamSet) -> Diagnosis:
        real, sim = _paired_sliding(real, sim)
        targets = [t.target for t in real.trials]
        a_r, b_r = _fit_line(targets, [t.slide for t in real.trials])
        a_s, b_s = _fit_line(targets, [t.slide for t in sim.trials])
        ratio = a_r / a_s if a_s else 1.0
        offset = b_r - b_s * ratio
        suspects: list[Suspect] = []
        if abs(ratio - 1.0) > 0.02:
            longer = ratio > 1.0
            suspects += [
                Suspect("object_mu", "down" if longer else "up", 0.3),
                Suspect("table_mu", "down" if longer else "up", 0.3),
                Suspect("actuator_gain", "up" if longer else "down", 0.2),
                Suspect("camera_pitch_deg", "unknown", 0.2),
            ]
        if abs(offset) > SUCCESS_TOL / 3:
            suspects.append(Suspect("camera_dx", "up" if offset > 0 else "down", 0.6, sim_params["camera_dx"] + offset / ratio))
        summary = f"slide scale x{ratio:.3f}, offset {offset:+.3f} m"
        return Diagnosis(summary, ratio, offset, sorted(suspects, key=lambda s: -s.confidence))


def _paired_sliding(real: Rollout, sim: Rollout) -> tuple[Rollout, Rollout]:
    """Drop trial pairs where either cube tipped, keeping real/sim aligned by target."""
    pairs = [(r, s) for r, s in zip(real.trials, sim.trials)
             if not getattr(r, "tipped", False) and not getattr(s, "tipped", False)]
    if len(pairs) < 3:
        return real, sim
    return Rollout([r for r, _ in pairs]), Rollout([s for _, s in pairs])


def fit_launch(track: list[float], dt: float = FRAME_DT) -> tuple[float, float] | None:
    """Least-squares fit of y = v0*t - a*t^2/2 over the first frames (launch phase) while the cube moves."""
    track = track[:TRACK_FRAMES + 1]
    pts = [(k * dt, y) for k, y in enumerate(track) if k == 0 or y - track[k - 1] > 1e-4]
    if len(pts) < 3:
        return None
    # normal equations for basis (t, -t^2/2)
    s11 = sum(t * t for t, _ in pts)
    s12 = sum(-0.5 * t**3 for t, _ in pts)
    s22 = sum(0.25 * t**4 for t, _ in pts)
    b1 = sum(t * y for t, y in pts)
    b2 = sum(-0.5 * t * t * y for t, y in pts)
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-18:
        return None
    return (b1 * s22 - b2 * s12) / det, (s11 * b2 - s12 * b1) / det


def _launch_stats(ro: Rollout) -> tuple[float, float] | None:
    """Mean (launch speed per unit command, deceleration) over trials with a usable track."""
    fits = [(fit_launch(t.track), t.command) for t in ro.trials if t.command > 0]
    fits = [(f, c) for f, c in fits if f is not None and f[1] > 0]
    if not fits:
        return None
    return sum(f[0] / c for f, c in fits) / len(fits), sum(f[1] for f, _ in fits) / len(fits)


class TrajectoryDiagnoser:
    """Uses what a camera sees, not just where the cube ended up:
    launch speed / command -> actuator gain; deceleration / g -> effective friction;
    observed target vs the known target line -> camera offset and pitch.
    This breaks the friction-vs-gain-vs-pitch confound the numbers-only diagnoser cannot."""

    def __init__(self, rel_tol: float = 0.03):
        self.rel_tol = rel_tol

    def diagnose(self, real: Rollout, sim: Rollout, sim_params: ParamSet) -> Diagnosis:
        suspects: list[Suspect] = []
        notes = []
        n_tipped = sum(getattr(t, "tipped", False) for t in real.trials)
        if n_tipped:
            notes.append(f"{n_tipped} cube(s) tipped over, excluded from fits")
        real, sim = _paired_sliding(real, sim)
        real_fit, sim_fit = _launch_stats(real), _launch_stats(sim)
        if real_fit and sim_fit:
            # relative to the sim's own tracked motion, so engine-specific biases cancel
            g_ratio, a_ratio = real_fit[0] / sim_fit[0], real_fit[1] / sim_fit[1]
            gain = sim_params["actuator_gain"] * g_ratio
            mu = effective_friction(sim_params["object_mu"], sim_params["table_mu"]) * a_ratio
            notes.append(f"launch speed x{g_ratio:.2f} of sim, deceleration x{a_ratio:.2f}")
            if abs(g_ratio - 1) > self.rel_tol:
                suspects.append(Suspect("actuator_gain", "up" if g_ratio > 1 else "down", 0.8, gain))
            if abs(a_ratio - 1) > self.rel_tol:
                d = "up" if a_ratio > 1 else "down"
                suspects += [Suspect("object_mu", d, 0.6, mu), Suspect("table_mu", d, 0.6, mu)]
        targets = [t.target for t in real.trials]
        k, dx = _fit_line(targets, [t.observed for t in real.trials])
        pitch = math.degrees(math.atan(k - 1.0))
        if abs(pitch - sim_params["camera_pitch_deg"]) > 0.5:
            suspects.append(Suspect("camera_pitch_deg", "up" if pitch > sim_params["camera_pitch_deg"] else "down", 0.7, pitch))
        if abs(dx - sim_params["camera_dx"]) > 0.004:
            suspects.append(Suspect("camera_dx", "up" if dx > sim_params["camera_dx"] else "down", 0.7, dx))
        notes.append(f"perceived target = {k:.3f}*true {dx:+.3f} m")
        a_r, b_r = _fit_line(targets, [t.slide for t in real.trials])
        a_s, b_s = _fit_line(targets, [t.slide for t in sim.trials])
        ratio = a_r / a_s if a_s else 1.0
        return Diagnosis("; ".join(notes), ratio, b_r - b_s * ratio, sorted(suspects, key=lambda s: -s.confidence))


class HeuristicPlanner:
    """Moves the top dynamics suspect so the sim's slide scale matches real, plus camera_dx if suspected."""

    def __init__(self, half_width_frac: float = 0.05):
        self.hw = half_width_frac

    def _band(self, name: str, center: float) -> tuple[float, float]:
        p = PARAM_SPACE[name]
        half = self.hw * (p.high - p.low)
        return (center - half, center + half)

    def plan(self, diag: Diagnosis, rand: Randomization, sim_params: ParamSet) -> ConfigDiff:
        ranges: dict[str, tuple[float, float]] = {}
        for s in diag.suspects:  # direct estimates (trajectory/visual evidence) win
            if s.estimate is not None:
                ranges[s.name] = self._band(s.name, s.estimate)
        names = [s.name for s in diag.suspects if s.name not in ranges]
        if "object_mu" in names or "table_mu" in names:
            # slide ∝ 1/mu_eff and only the mean matters -> set both to the matching mean
            mu = effective_friction(sim_params["object_mu"], sim_params["table_mu"]) / diag.scale_ratio
            ranges["object_mu"] = self._band("object_mu", mu)
            ranges["table_mu"] = self._band("table_mu", mu)
        return ConfigDiff(ranges, rationale=diag.summary)


def center(rand: Randomization) -> ParamSet:
    return ParamSet({k: 0.5 * (lo + hi) for k, (lo, hi) in rand.ranges.items()})


@dataclass
class IterationLog:
    iteration: int
    real_success: float
    sim_success: float
    diagnosis: dict | None
    diff: dict | None


@dataclass
class LoopResult:
    final_success: float
    iterations: int
    train_calls: int
    final_randomization: dict[str, tuple[float, float]]
    log: list[IterationLog]

    def to_json(self) -> dict:
        return asdict(self)


def run_loop(
    real: RealWorld,
    trainer: GridTrainer,
    diagnoser: Diagnoser,
    planner: Planner,
    targets: list[float],
    rand: Randomization | None = None,
    max_iter: int = 5,
    goal: float = 0.9,
    log_path: Path | None = None,
    sim_env=None,
    emit: Callable[[dict], None] | None = None,
) -> LoopResult:
    """emit(event) receives one dict per agent step: train, measure, diagnose, plan, done."""
    env = sim_env or AnalyticPushEnv()
    rand = rand or Randomization.none()
    send = emit or (lambda e: None)
    logs: list[IterationLog] = []
    real_ro = None
    for it in range(max_iter):
        policy = trainer.train(rand)
        send({"type": "train", "iter": it, "policy_c": policy.c, "ranges": rand.ranges})
        sim_p = center(rand)
        real_ro = real.rollout(policy, targets)
        sim_ro = env.rollout(sim_p, policy, targets)
        ev = {"type": "measure", "iter": it, "real": real_ro.to_json(), "sim": sim_ro.to_json(),
              "sim_params": sim_p.values, "policy_c": policy.c}
        if hasattr(policy, "commands"):  # TablePolicy: the lookup itself, so recorders can replay exact commands
            ev["policy_table"] = {"observed": list(policy.observed), "commands": [round(c, 5) for c in policy.commands]}
        send(ev)
        if real_ro.success_rate >= goal or it == max_iter - 1:
            logs.append(IterationLog(it, real_ro.success_rate, sim_ro.success_rate, None, None))
            send({"type": "done", "iter": it, "final_success": real_ro.success_rate, "reached_goal": real_ro.success_rate >= goal})
            break
        diag = diagnoser.diagnose(real_ro, sim_ro, sim_p)
        send({"type": "diagnose", "iter": it, "agent": diag.model or type(diagnoser).__name__, **asdict(diag)})
        diff = planner.plan(diag, rand, sim_p)
        diff.validate()
        send({"type": "plan", "iter": it, "agent": type(planner).__name__, **asdict(diff)})
        logs.append(IterationLog(it, real_ro.success_rate, sim_ro.success_rate, asdict(diag), asdict(diff)))
        rand = rand.apply(diff)
    result = LoopResult(real_ro.success_rate, len(logs), trainer.calls, rand.ranges, logs)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(result.to_json(), indent=2))
    return result
