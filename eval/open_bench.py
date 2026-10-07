"""Gap-Bench: open-world comparison. Hidden worlds include faults outside the rule-based diagnoser's map.

Tiers (same seed -> same worlds):
  closed    two of the original 10 params changed (incl. distractors)
  open      one open fault (friction patch or lens distortion) + one closed param
  compound  friction patch + lens distortion + one closed param
Methods (all train with InverseTrainer, so the policy is exactly as good as its simulator):
  nominal, full_dr (all 13 params randomized), rule (TrajectoryDiagnoser), sysid (passive structure search with the
  same least-squares fitter, no LLM, no probes), agent (Nemotron tool agent; optional, --llm).
Every success rate is measured by rollout in the hidden world; real trials (evaluations + probes) are counted.
Run: .venv/bin/python -m eval.open_bench --worlds 6 [--env newton] [--llm tokenfactory]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field

from agent.loop import Diagnosis, HeuristicPlanner, RealWorld, Suspect, TrajectoryDiagnoser, run_loop
from agent.tool_agent import ToolAgentDiagnoser, Workbench, from_params, to_params
from sim.params import CLOSED_PARAMS, OPEN_PARAMS, PARAM_SPACE, ParamSet, Randomization, effective_friction, sample_hidden
from sim.push_task import AnalyticPushEnv, InverseTrainer, eval_targets

TIERS = ("closed", "open", "compound")


def sample_open_world(rng: random.Random, tier: str) -> ParamSet:
    """Hidden world for a tier. Patch friction stays <= 0.95 (above ~1 the cube tips instead of sliding)."""
    if tier == "closed":
        return sample_hidden(rng, 2)
    base = sample_hidden(rng, 1)
    mu = effective_friction(base["object_mu"], base["table_mu"])
    ch = {}
    kinds = [rng.choice(["patch", "lens"])] if tier == "open" else ["patch", "lens"]
    if "patch" in kinds:
        sign = rng.choice([-1, 1]) if mu < 0.9 else -1
        ch["patch_y0"] = round(rng.uniform(0.25, 0.5), 3)
        ch["patch_mu"] = round(min(0.95, max(0.25, mu + sign * rng.uniform(0.25, 0.45))), 3)
    if "lens" in kinds:
        ch["lens_k"] = round(rng.choice([-1, 1]) * rng.uniform(0.12, 0.3), 3)
    return base.with_(**ch)


STRUCTURES = [  # passive sysID: simplest first
    ["mu_eff", "actuator_gain", "camera_dx", "camera_pitch_deg"],
    ["mu_eff", "actuator_gain", "camera_dx", "camera_pitch_deg", "lens_k"],
    ["mu_eff", "actuator_gain", "camera_dx", "camera_pitch_deg", "patch_y0", "patch_mu"],
    ["mu_eff", "actuator_gain", "camera_dx", "camera_pitch_deg", "patch_y0", "patch_mu", "lens_k"],
]


class SysIdDiagnoser:
    """Baseline S3: black-box system identification over a fixed model library (no LLM, no probes): fit each
    structure by least squares, take the first whose stop residual < 1 cm and perception residual < 0.5 cm,
    else the best."""

    def __init__(self):
        self.fallback = TrajectoryDiagnoser()

    def diagnose(self, real, sim, sim_params: ParamSet) -> Diagnosis:
        wb = Workbench(real, sim_params, None, 0)
        start = from_params(sim_params)
        fits = []
        for free in STRUCTURES:
            f = wb.fit_hypothesis(dict(start), free)
            fits.append((f["stop_residual_rms_m"] + f["perception_residual_rms_m"], f))
            if f["stop_residual_rms_m"] < 0.01 and f["perception_residual_rms_m"] < 0.005:
                break
        best = fits[-1][1] if fits[-1][1]["stop_residual_rms_m"] < 0.01 else min(fits, key=lambda x: x[0])[1]
        new = to_params(best["fitted_model"], sim_params)
        base = self.fallback.diagnose(real, sim, sim_params)
        sus = [Suspect(n, "up" if b > a else "down", 0.8, b) for n, (a, b) in sim_params.diff(new, rel_tol=1e-3).items()]
        return Diagnosis(f"sysid: {len(fits)} structures fitted, stop rms {best['stop_residual_rms_m'] * 100:.1f} cm",
                         base.scale_ratio, base.offset, sus, model="sysid")


@dataclass
class Row:
    tier: str
    world: int
    hidden: dict[str, float]
    success: dict[str, float]
    real_trials: dict[str, int]
    iterations: dict[str, int] = field(default_factory=dict)


def make_env(name: str):
    if name == "newton":
        from sim.newton_push import NewtonPushEnv

        return NewtonPushEnv()
    return AnalyticPushEnv()


def run(n_worlds: int = 6, seed: int = 0, env_name: str = "analytic", tiers=TIERS, agent_llm=None, max_iter: int = 4,
        n_targets: int = 20, log=print, agent_only: bool = False) -> list[Row]:
    env = make_env(env_name)
    targets = eval_targets(n_targets, seed + 2000)
    rows = []
    all_open = CLOSED_PARAMS + OPEN_PARAMS
    for tier in tiers:
        rng = random.Random(f"{seed}-{tier}")
        for w in range(n_worlds):
            hidden = sample_open_world(rng, tier)
            diff = {k: b for k, (_, b) in ParamSet.nominal().diff(hidden).items()}
            success, trials, iters = {}, {}, {}
            for name, rand in (("nominal", Randomization.none()), ("full_dr", Randomization.full(all_open))):
                real = RealWorld(env, hidden)
                success[name] = real.rollout(InverseTrainer().train(rand), targets).success_rate
                trials[name], iters[name] = real.trials_used, 1
            methods = {} if agent_only else {"rule": lambda r: TrajectoryDiagnoser(), "sysid": lambda r: SysIdDiagnoser()}
            if agent_llm is not None:
                methods["agent"] = lambda r: ToolAgentDiagnoser(agent_llm, r)
            for name, make in methods.items():
                real = RealWorld(env, hidden)
                t0 = time.perf_counter()
                res = run_loop(real, InverseTrainer(), make(real), HeuristicPlanner(half_width_frac=0.01), targets,
                               sim_env=env, max_iter=max_iter)
                success[name], trials[name], iters[name] = res.final_success, real.trials_used, res.iterations
                if log:
                    log(f"  [{tier} w{w}] {name:7s} {res.final_success:4.0%}  trials {real.trials_used:3d}  iters {res.iterations}  "
                        f"{time.perf_counter() - t0:5.1f}s")
            rows.append(Row(tier, w, diff, success, trials, iters))
            if log:
                log(f"[{tier} w{w}] {json.dumps({k: round(v, 3) for k, v in diff.items()})} -> "
                    + " ".join(f"{m} {s:.0%}" for m, s in success.items()))
    return rows


def ci95(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def summarize(rows: list[Row]) -> dict:
    out = {}
    methods = list(rows[0].success)
    for tier in dict.fromkeys(r.tier for r in rows):
        rs = [r for r in rows if r.tier == tier]
        out[tier] = {m: {"mean": ci95([r.success[m] for r in rs])[0], "ci95": ci95([r.success[m] for r in rs])[1],
                         "real_trials": sum(r.real_trials[m] for r in rs) / len(rs)} for m in methods}
    return out


def format_table(rows: list[Row]) -> str:
    s = summarize(rows)
    methods = list(rows[0].success)
    lines = ["| tier | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for tier, ms in s.items():
        lines.append(f"| {tier} (n={sum(r.tier == tier for r in rows)}) | "
                     + " | ".join(f"{v['mean']:.0%} ±{v['ci95']:.0%} ({v['real_trials']:.0f} trials)" for v in ms.values()) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=6, help="worlds per tier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env", choices=["analytic", "newton"], default="analytic")
    ap.add_argument("--llm", choices=["none", "local", "tokenfactory"], default="none")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--model", default=None, help="model hint for the agent (e.g. 'nemotron nano', 'nemotron ultra', 'lightning')")
    ap.add_argument("--agent-only", action="store_true", help="skip the rule/sysID loops (model comparisons)")
    ap.add_argument("--out", default=None, help="write rows + summary JSON here")
    a = ap.parse_args()
    llm = None
    if a.llm != "none":
        from agent.llm import make_llm

        llm = make_llm(a.llm)
        if a.model:
            llm.hints["diagnose"] = a.model
    rows = run(a.worlds, a.seed, a.env, tuple(a.tiers.split(",")), llm, agent_only=a.agent_only)
    print(format_table(rows))
    if llm is not None:
        u = llm.usage
        print(f"\nLLM usage: {u.calls} calls, {u.prompt_tokens} prompt + {u.completion_tokens} completion tokens, models {u.by_model}")
    if a.out:
        from pathlib import Path

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"rows": [asdict(r) for r in rows], "summary": summarize(rows),
                                           "usage": asdict(llm.usage) if llm else None}, indent=1))


if __name__ == "__main__":
    main()
