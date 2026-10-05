"""Tier 0 comparison: same hidden worlds, same eval targets, three methods.

  A  full_dr   : train once with every param randomized over its full range
  B  nominal   : train once at nominal (no randomization) -- the "sim-perfect" policy
  G1 gapcloser_outcome : run_loop, diagnoser sees final positions only (HeuristicDiagnoser)
  G2 gapcloser_traj    : run_loop, diagnoser also sees the tracked cube motion (TrajectoryDiagnoser)

Success rates are measured by rollout in the hidden world, never estimated.
Run: .venv/bin/python -m eval.compare --worlds 10
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from agent.loop import HeuristicDiagnoser, HeuristicPlanner, RealWorld, TrajectoryDiagnoser, run_loop
from sim.params import PARAM_SPACE, ParamSet, Randomization, sample_hidden
from sim.push_task import AnalyticPushEnv, GridTrainer, eval_targets

METHODS = ("full_dr", "nominal", "gapcloser_outcome", "gapcloser_traj")
DIAGNOSERS = {"gapcloser_outcome": HeuristicDiagnoser, "gapcloser_traj": TrajectoryDiagnoser}


# hidden changes that can show up in the measurements at all (the rest are distractors by design)
IDENTIFIABLE = {"object_mu", "table_mu", "actuator_gain", "camera_dx", "camera_pitch_deg"}
FRICTION = {"object_mu", "table_mu"}  # only their mean is observable


@dataclass
class Row:
    world: int
    hidden_diff: dict[str, float]
    success: dict[str, float]
    train_calls: dict[str, int]
    diag_precision: dict[str, float | None] = None
    diag_recall: dict[str, float | None] = None


def diagnosis_scores(first_diag: dict | None, hidden: set[str]) -> tuple[float | None, float | None]:
    """Precision/recall of the first diagnosis vs the identifiable hidden changes.
    object_mu and table_mu count as one physical quantity (their mean)."""
    truth = hidden & IDENTIFIABLE
    if first_diag is None:
        return None, None
    norm = lambda n: "friction" if n in FRICTION else n  # noqa: E731
    named = {norm(s["name"]) for s in first_diag["suspects"]}
    t = {norm(n) for n in truth}
    hit = named & t
    precision = len(hit) / len(named) if named else (1.0 if not t else 0.0)
    recall = len(hit) / len(t) if t else None
    return precision, recall


def make_env(name: str):
    if name == "newton":
        from sim.newton_push import NewtonPushEnv

        return NewtonPushEnv()
    return AnalyticPushEnv()


def run(n_worlds: int = 10, n_perturbed: int = 2, n_targets: int = 20, seed: int = 0, env_name: str = "analytic",
        extra_diagnosers: dict | None = None) -> list[Row]:
    env = make_env(env_name)
    diagnosers = {**DIAGNOSERS, **(extra_diagnosers or {})}
    targets = eval_targets(n_targets, seed + 1000)
    rng = random.Random(seed)
    rows = []
    for w in range(n_worlds):
        hidden = sample_hidden(rng, n_perturbed)
        real = RealWorld(env, hidden)
        success, calls, prec, rec = {}, {}, {}, {}
        surrogate = AnalyticPushEnv()  # policy search always uses the fast surrogate
        for name, rand in (("full_dr", Randomization.full()), ("nominal", Randomization.none())):
            tr = GridTrainer(surrogate, seed=seed)
            success[name] = real.rollout(tr.train(rand), targets).success_rate
            calls[name] = tr.calls
        diff = {k: b for k, (_, b) in ParamSet.nominal().diff(hidden).items()}
        for name, diag in diagnosers.items():
            tr = GridTrainer(surrogate, seed=seed)
            res = run_loop(real, tr, diag(), HeuristicPlanner(), targets, sim_env=env)
            success[name], calls[name] = res.final_success, res.train_calls
            first = next((lg.diagnosis for lg in res.log if lg.diagnosis), None)
            prec[name], rec[name] = diagnosis_scores(first, set(diff))
        rows.append(Row(w, diff, success, calls, prec, rec))
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rows: list[Row]) -> dict[str, dict[str, float]]:
    out = {}
    for m in rows[0].success:
        out[m] = {
            "mean_success": sum(r.success[m] for r in rows) / len(rows),
            "mean_train_calls": sum(r.train_calls[m] for r in rows) / len(rows),
        }
        if m in rows[0].diag_precision:
            out[m]["diag_precision"] = _mean(r.diag_precision[m] for r in rows)
            out[m]["diag_recall"] = _mean(r.diag_recall[m] for r in rows)
    return out


def format_table(rows: list[Row]) -> str:
    methods = list(rows[0].success)
    lines = ["| world | hidden change | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for r in rows:
        hd = ", ".join(f"{k}={v:.3g}{PARAM_SPACE[k].unit if PARAM_SPACE[k].unit != '-' else ''}" for k, v in r.hidden_diff.items())
        lines.append(f"| {r.world} | {hd} | " + " | ".join(f"{r.success[m]:.0%}" for m in methods) + " |")
    s = summarize(rows)
    lines.append("| **mean** | | " + " | ".join(f"**{s[m]['mean_success']:.0%}** ({s[m]['mean_train_calls']:.1f} trains)" for m in methods) + " |")
    lines.append("| diagnosis P / R | | " + " | ".join(
        (f"{s[m]['diag_precision']:.2f} / {s[m]['diag_recall']:.2f}" if s[m].get("diag_precision") is not None else "—")
        for m in methods) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=10)
    ap.add_argument("--perturbed", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--env", choices=["analytic", "newton"], default="analytic")
    ap.add_argument("--llm", choices=["none", "local", "tokenfactory"], default="none")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    extra = {}
    if a.llm != "none":
        from agent.llm import make_llm
        from agent.llm_diagnoser import LLMDiagnoser

        llm = make_llm(a.llm)
        if a.model:
            llm.hints["diagnose"] = a.model
        extra["gapcloser_llm"] = lambda: LLMDiagnoser(llm)
    rows = run(a.worlds, a.perturbed, seed=a.seed, env_name=a.env, extra_diagnosers=extra)
    print(json.dumps(summarize(rows), indent=2) if a.json else format_table(rows))


if __name__ == "__main__":
    main()
