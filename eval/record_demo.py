"""Record demo runs for the dashboard: events + Newton clips + baselines + hidden truth.

Every success rate here is measured by rolling out in NVIDIA Newton (sim and hidden "real").
Policy search uses the analytic surrogate (matches Newton within ~5 mm, see tests).
Run: .venv/bin/python -m eval.record_demo   ->  runs/demo/bundle.json + clips/*.webp
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from agent.loop import HeuristicDiagnoser, HeuristicPlanner, RealWorld, TrajectoryDiagnoser, run_loop
from eval.compare import run as compare_run
from eval.compare import summarize
from sim.newton_push import NewtonPushEnv, render_trial_arm as render_trial
from sim.params import PARAM_SPACE, ParamSet, Randomization
from sim.push_task import SUCCESS_TOL, AnalyticPushEnv, GridTrainer, eval_targets

SCENARIOS = [
    {"id": "slippery-cube", "title": "Slippery cube", "hidden": {"object_mu": 0.25, "restitution": 0.15}},
    {"id": "camera-offset", "title": "Shifted camera", "hidden": {"camera_dx": -0.028, "light_intensity": 0.6}},
    {"id": "sticky-table", "title": "Sticky table", "hidden": {"table_mu": 1.05, "light_intensity": 1.4}},
    {"id": "weak-motor", "title": "Weak motor", "hidden": {"actuator_gain": 0.76, "restitution": 0.15}},
    {"id": "tipping-edge", "title": "Tipping edge", "hidden": {"table_mu": 1.18}},
]
CLIP_TARGET = 0.45


def record_scenario(sc: dict, out: Path, targets: list[float], diagnoser=None) -> dict:
    hidden = ParamSet.nominal().with_(**sc["hidden"])
    newton_env = NewtonPushEnv()
    real = RealWorld(newton_env, hidden)
    events: list[dict] = []
    t0 = time.perf_counter()

    def emit(e: dict) -> None:
        e = {"t": round(time.perf_counter() - t0, 3), **e}
        if e["type"] == "measure":
            it = e["iter"]
            cmd = _cmd(e["policy_c"], CLIP_TARGET, hidden)
            real_clip = f"clips/{sc['id']}-it{it}-real.webp"
            sim_clip = f"clips/{sc['id']}-it{it}-sim.webp"
            sim_p = ParamSet(e["sim_params"])
            e["clip"] = {
                "target": CLIP_TARGET,
                "real": real_clip,
                "real_slide": render_trial(hidden, cmd, CLIP_TARGET, out / real_clip),
                "sim": sim_clip,
                "sim_slide": render_trial(sim_p, _cmd(e["policy_c"], CLIP_TARGET, sim_p), CLIP_TARGET, out / sim_clip),
            }
        events.append(e)

    res = run_loop(real, GridTrainer(AnalyticPushEnv()), diagnoser or TrajectoryDiagnoser(), HeuristicPlanner(), targets,
                   sim_env=newton_env, emit=emit)
    outcome_only = run_loop(real, GridTrainer(AnalyticPushEnv()), HeuristicDiagnoser(), HeuristicPlanner(), targets,
                            sim_env=newton_env)
    baselines = {"outcome_only": outcome_only.final_success}
    for name, rand in (("full_dr", Randomization.full()), ("nominal", Randomization.none())):
        baselines[name] = real.rollout(GridTrainer(AnalyticPushEnv()).train(rand), targets).success_rate
    truth = {k: {"nominal": a, "hidden": b} for k, (a, b) in ParamSet.nominal().diff(hidden).items()}
    return {**sc, "events": events, "final_success": res.final_success, "iterations": res.iterations,
            "baselines": baselines, "truth": truth}


def rerender_clips(out: Path) -> None:
    """Re-render the sim/real clips of an existing bundle from its stored inputs (deterministic)."""
    path = out / "bundle.json"
    bundle = json.loads(path.read_text())
    for run in bundle["runs"]:
        hidden = ParamSet.nominal().with_(**run["hidden"])
        for e in run["events"]:
            clip = e.get("clip")
            if not clip:
                continue
            sim_p = ParamSet(e["sim_params"])
            clip["real_slide"] = render_trial(hidden, _cmd(e["policy_c"], clip["target"], hidden), clip["target"], out / clip["real"])
            clip["sim_slide"] = render_trial(sim_p, _cmd(e["policy_c"], clip["target"], sim_p), clip["target"], out / clip["sim"])
        print(f"re-rendered {run['id']}")
    bundle["stack"]["clips"] = "Franka FR3 (kinematic, IK) strikes; cube physics in NVIDIA Newton"
    path.write_text(json.dumps(bundle, indent=1))


def extract_frames(clip: Path, out_dir: Path, n: int = 3) -> list[Path]:
    """First, middle and last frame of an animated clip as PNGs (input for a multimodal diagnoser)."""
    from PIL import Image

    im = Image.open(clip)
    count = getattr(im, "n_frames", 1)
    idx = sorted({0, count // 2, count - 1})[:n]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in idx:
        im.seek(i)
        p = out_dir / f"{clip.stem}-f{i}.png"
        im.convert("RGB").save(p)
        paths.append(p)
    return paths


def _cmd(c: float, target: float, p: ParamSet) -> float:
    from sim.push_task import Policy, observe

    return Policy(c).command(observe(target, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/demo"))
    ap.add_argument("--llm", choices=["none", "local", "tokenfactory"], default="none",
                    help="diagnose with Nemotron (local Ollama or Token Factory); none = TrajectoryDiagnoser")
    ap.add_argument("--model", default=None, help="model hint override for the diagnose role")
    ap.add_argument("--bench-only", action="store_true", help="keep recorded scenarios, recompute only the benchmark")
    ap.add_argument("--clips-only", action="store_true", help="re-render every clip of the existing bundle (no LLM, no reruns)")
    ap.add_argument("--only", nargs="+", default=None, help="re-record only these scenario ids, keep the rest and the benchmark")
    a = ap.parse_args()
    if a.clips_only:
        rerender_clips(a.out)
        return
    llm = None
    if a.llm != "none":
        from agent.llm import make_llm

        llm = make_llm(a.llm)
        if a.model:
            llm.hints["diagnose"] = a.model
        print(f"diagnoser: Nemotron via {a.llm} -> {llm.model_for('diagnose')}")
    a.out.mkdir(parents=True, exist_ok=True)
    targets = eval_targets(20, 1000)
    runs = []
    old = json.loads((a.out / "bundle.json").read_text()) if (a.bench_only or a.only) else None
    todo = [] if a.bench_only else [sc for sc in SCENARIOS if not a.only or sc["id"] in a.only]
    for sc in todo:
        t = time.perf_counter()
        diag = None
        if llm is not None:
            from agent.llm_diagnoser import LLMDiagnoser

            diag = LLMDiagnoser(llm)
        r = record_scenario(sc, a.out, targets, diag)
        print(f"{sc['id']:15s} final {r['final_success']:.0%} in {r['iterations']} iters "
              f"(outcome-only {r['baselines']['outcome_only']:.0%}, full_dr {r['baselines']['full_dr']:.0%}, nominal {r['baselines']['nominal']:.0%})  {time.perf_counter() - t:.1f}s")
        runs.append(r)
    if a.only:
        fresh = {r["id"]: r for r in runs}
        old["runs"] = [fresh.get(r["id"], r) for r in old["runs"]]
        if llm is not None:
            old["stack"]["llm_usage_last"] = vars(llm.usage)
        old["stack"]["clips"] = "Franka FR3 (kinematic, IK) strikes; cube physics in NVIDIA Newton"
        (a.out / "bundle.json").write_text(json.dumps(old, indent=1))
        print(f"updated {a.out / 'bundle.json'} ({', '.join(fresh)})")
        return
    extra = {}
    if llm is not None:
        from agent.llm_diagnoser import LLMDiagnoser

        extra["gapcloser_llm"] = lambda: LLMDiagnoser(llm)
    rows = compare_run(n_worlds=10, seed=0, env_name="newton", extra_diagnosers=extra)
    if old is not None:
        runs = old["runs"]
    bundle = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": {"name": "Push-to-line", "success_tol_m": SUCCESS_TOL, "n_targets": len(targets), "targets": targets},
        "params": {k: {"nominal": p.nominal, "low": p.low, "high": p.high, "unit": p.unit, "kind": p.kind, "hint": p.hint}
                   for k, p in PARAM_SPACE.items()},
        "stack": {"physics": "NVIDIA Newton 1.6 (Warp, CPU)",
                  "diagnoser": (f"Nemotron ({llm.model_for('diagnose')}) via {a.llm}" if llm else "TrajectoryDiagnoser (cube tracking, offline)"),
                  "planner": "HeuristicPlanner", "llm_provider": a.llm,
                  "clips": "Franka FR3 (kinematic, IK) strikes; cube physics in NVIDIA Newton",
                  "llm_usage": (vars(llm.usage) if llm else None)},
        "runs": runs,
        "benchmark": {"summary": summarize(rows),
                      "env": "newton", "rows": [{"world": r.world, "hidden": r.hidden_diff, "success": r.success, "train_calls": r.train_calls,
                                     "diag_precision": r.diag_precision, "diag_recall": r.diag_recall} for r in rows]},
    }
    (a.out / "bundle.json").write_text(json.dumps(bundle, indent=1))
    print(f"wrote {a.out / 'bundle.json'}")


if __name__ == "__main__":
    random.seed(0)
    main()
