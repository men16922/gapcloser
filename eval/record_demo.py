"""Record demo runs for the dashboard: events + Newton clips + baselines + hidden truth.

Every success rate here is measured by rolling out in NVIDIA Newton (sim and hidden "real").
Policy search uses the analytic surrogate (matches Newton within ~5 mm, see tests).
Run: .venv/bin/python -m eval.record_demo   ->  runs/demo/bundle.json + clips/*.webp + replay/*.json
     .venv/bin/python -m eval.record_demo --replays-only   (3D viewer replays for the existing bundle, no LLM)
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
from sim.replay import record_pair, write_replay
from sim.push_task import SUCCESS_TOL, AnalyticPushEnv, GridTrainer, InverseTrainer, eval_targets

SCENARIOS = [
    {"id": "slippery-cube", "title": "Slippery cube", "hidden": {"object_mu": 0.25, "restitution": 0.15}},
    {"id": "camera-offset", "title": "Shifted camera", "hidden": {"camera_dx": -0.028, "light_intensity": 0.6}},
    {"id": "sticky-table", "title": "Sticky table", "hidden": {"table_mu": 1.05, "light_intensity": 1.4}},
    {"id": "weak-motor", "title": "Weak motor", "hidden": {"actuator_gain": 0.76, "restitution": 0.15}},
    {"id": "tipping-edge", "title": "Tipping edge", "hidden": {"table_mu": 1.18}},
]
# open-world faults outside the rule-based diagnoser's map; trained with InverseTrainer, diagnosed by the tool agent
OPEN_SCENARIOS = [
    {"id": "wet-strip", "title": "Wet strip", "open": True, "hidden": {"patch_y0": 0.35, "patch_mu": 0.4}},
    {"id": "rough-strip", "title": "Rough strip", "open": True, "hidden": {"table_mu": 0.5, "patch_y0": 0.3, "patch_mu": 0.9}},
    {"id": "fisheye", "title": "Lens distortion", "open": True, "hidden": {"lens_k": -0.3, "light_intensity": 0.7}},
    {"id": "three-faults", "title": "Three faults", "open": True,
     "hidden": {"actuator_gain": 0.85, "patch_y0": 0.4, "patch_mu": 0.5, "lens_k": -0.15}},
]
CLIP_TARGET = 0.45


def record_scenario(sc: dict, out: Path, targets: list[float], diagnoser=None, *, env=None, render: bool = True,
                    on_event=None, max_iter: int = 5, make_diagnoser=None) -> dict:
    """Run the agent on one hidden world, render clips, compute baselines.

    on_event(event) is called as each agent step happens (the live server streams these); events are
    also collected in the returned run dict. env defaults to NVIDIA Newton; render=False skips clips."""
    hidden = ParamSet.nominal().with_(**sc["hidden"])
    world_env = env or NewtonPushEnv()
    real = RealWorld(world_env, hidden)
    events: list[dict] = []
    truth = {k: {"nominal": a, "hidden": b} for k, (a, b) in ParamSet.nominal().diff(hidden).items()}
    send = on_event or (lambda e: None)
    t0 = time.perf_counter()
    send({"type": "start", "t": 0.0, "id": sc["id"], "title": sc.get("title", sc["id"]), "hidden": sc["hidden"],
          "truth": truth, "max_iter": max_iter, "open": bool(sc.get("open"))})

    def emit(e: dict) -> None:
        e = {"t": round(time.perf_counter() - t0, 3), **e}
        if e["type"] == "measure" and render:
            it = e["iter"]
            cmd = _cmd(e, CLIP_TARGET, hidden)
            real_clip = f"clips/{sc['id']}-it{it}-real.webp"
            sim_clip = f"clips/{sc['id']}-it{it}-sim.webp"
            sim_p = ParamSet(e["sim_params"])
            e["clip"] = {
                "target": CLIP_TARGET,
                "real": real_clip,
                "real_slide": render_trial(hidden, cmd, CLIP_TARGET, out / real_clip),
                "sim": sim_clip,
                "sim_slide": render_trial(sim_p, _cmd(e, CLIP_TARGET, sim_p), CLIP_TARGET, out / sim_clip),
            }
            e["clip"]["replay"] = save_replay(out, sc["id"], it, e, CLIP_TARGET, sim_p, hidden)
        events.append(e)
        send(e)

    diag = make_diagnoser(real) if make_diagnoser else (diagnoser or TrajectoryDiagnoser())
    if sc.get("open"):
        from eval.open_bench import SysIdDiagnoser
        from sim.params import CLOSED_PARAMS, OPEN_PARAMS

        res = run_loop(real, InverseTrainer(), diag, HeuristicPlanner(half_width_frac=0.01), targets,
                       sim_env=world_env, emit=emit, max_iter=max_iter)
        baselines = {"real_trials": real.trials_used}
        for name, d in (("rule", TrajectoryDiagnoser()), ("sysid", SysIdDiagnoser())):
            r = RealWorld(world_env, hidden)
            baselines[name] = run_loop(r, InverseTrainer(), d, HeuristicPlanner(half_width_frac=0.01), targets,
                                       sim_env=world_env, max_iter=max_iter).final_success
            baselines[name + "_trials"] = r.trials_used
        for name, rand in (("full_dr", Randomization.full(CLOSED_PARAMS + OPEN_PARAMS)), ("nominal", Randomization.none())):
            baselines[name] = real.rollout(InverseTrainer().train(rand), targets).success_rate
    else:
        res = run_loop(real, GridTrainer(AnalyticPushEnv()), diag, HeuristicPlanner(), targets,
                       sim_env=world_env, emit=emit, max_iter=max_iter)
        outcome_only = run_loop(real, GridTrainer(AnalyticPushEnv()), HeuristicDiagnoser(), HeuristicPlanner(), targets,
                                sim_env=world_env, max_iter=max_iter)
        baselines = {"outcome_only": outcome_only.final_success}
        for name, rand in (("full_dr", Randomization.full()), ("nominal", Randomization.none())):
            baselines[name] = real.rollout(GridTrainer(AnalyticPushEnv()).train(rand), targets).success_rate
    run = {**sc, "events": events, "final_success": res.final_success, "iterations": res.iterations,
           "baselines": baselines, "truth": truth}
    send({"type": "end", "t": round(time.perf_counter() - t0, 3), "run": {k: v for k, v in run.items() if k != "events"}})
    return run


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
            clip["real_slide"] = render_trial(hidden, _cmd(e, clip["target"], hidden), clip["target"], out / clip["real"])
            clip["sim_slide"] = render_trial(sim_p, _cmd(e, clip["target"], sim_p), clip["target"], out / clip["sim"])
        print(f"re-rendered {run['id']}")
    bundle["stack"]["clips"] = "Franka FR3 (kinematic, IK) strikes; cube physics in NVIDIA Newton"
    path.write_text(json.dumps(bundle, indent=1))


def save_replay(out: Path, run_id: str, it: int, ev: dict, target: float, sim_p: ParamSet, hidden: ParamSet) -> str:
    """Per-frame Newton poses (cube + Franka links) for the dashboard's 3D viewer; returns the bundle-relative path."""
    rel = f"replay/{run_id}-it{it}.json"
    data = record_pair(sim_p, _cmd(ev, target, sim_p), hidden, _cmd(ev, target, hidden), target, SUCCESS_TOL)
    write_replay(data, out / rel)
    return rel


def rerender_replays(out: Path) -> None:
    """Record 3D-viewer replays for every clip of an existing bundle (deterministic, no LLM, no reruns)."""
    path = out / "bundle.json"
    bundle = json.loads(path.read_text())
    for run in bundle["runs"]:
        hidden = ParamSet.nominal().with_(**run["hidden"])
        for e in run["events"]:
            clip = e.get("clip")
            if not clip:
                continue
            clip["replay"] = save_replay(out, run["id"], e["iter"], e, clip["target"], ParamSet(e["sim_params"]), hidden)
            rp = json.loads((out / clip["replay"]).read_text())["trials"]
            drift = max(abs(rp["real"]["slide"] - clip["real_slide"]), abs(rp["sim"]["slide"] - clip["sim_slide"]))
            print(f"  {clip['replay']}  real {rp['real']['slide']:.3f} m  sim {rp['sim']['slide']:.3f} m  (clip drift {drift * 1000:.2f} mm)")
        print(f"replays {run['id']}")
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


def _cmd(ev: dict, target: float, p: ParamSet) -> float:
    """Command the policy of a measure event issues for `target` in world p (TablePolicy when recorded)."""
    from sim.push_task import Policy, TablePolicy, observe

    tab = ev.get("policy_table")
    pol = TablePolicy(tuple(tab["observed"]), tuple(tab["commands"])) if tab else Policy(ev["policy_c"])
    return pol.command(observe(target, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/demo"))
    ap.add_argument("--llm", choices=["none", "local", "tokenfactory"], default="none",
                    help="diagnose with Nemotron (local Ollama or Token Factory); none = TrajectoryDiagnoser")
    ap.add_argument("--model", default=None, help="model hint override for the diagnose role")
    ap.add_argument("--bench-only", action="store_true", help="keep recorded scenarios, recompute only the benchmark")
    ap.add_argument("--clips-only", action="store_true", help="re-render every clip of the existing bundle (no LLM, no reruns)")
    ap.add_argument("--replays-only", action="store_true", help="record 3D-viewer replays for the existing bundle (no LLM, no reruns)")
    ap.add_argument("--only", nargs="+", default=None, help="re-record only these scenario ids, keep the rest and the benchmark")
    ap.add_argument("--open-only", action="store_true",
                    help="record the open-world scenarios (tool agent with --llm) into the existing bundle, keep the rest")
    ap.add_argument("--attach-bench", type=Path, default=None, help="embed a Gap-Bench JSON (eval.open_bench --out) in the bundle")
    a = ap.parse_args()
    if a.attach_bench and not a.open_only:
        path = a.out / "bundle.json"
        bundle = json.loads(path.read_text())
        bundle["open_benchmark"] = json.loads(a.attach_bench.read_text())
        path.write_text(json.dumps(bundle, indent=1))
        print(f"attached {a.attach_bench} to {path}")
        return
    if a.clips_only:
        rerender_clips(a.out)
        rerender_replays(a.out)
        return
    if a.replays_only:
        rerender_replays(a.out)
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
    if a.open_only:
        from agent.tool_agent import ToolAgentDiagnoser

        path = a.out / "bundle.json"
        bundle = json.loads(path.read_text())
        todo = [sc for sc in OPEN_SCENARIOS if not a.only or sc["id"] in a.only]
        for sc in todo:
            t = time.perf_counter()
            mk = (lambda real: ToolAgentDiagnoser(llm, real)) if llm is not None else None
            r = record_scenario(sc, a.out, targets, make_diagnoser=mk, max_iter=4)
            b = r["baselines"]
            print(f"{sc['id']:15s} final {r['final_success']:.0%} in {r['iterations']} iters, {b['real_trials']} real trials "
                  f"(rule {b['rule']:.0%}, sysid {b['sysid']:.0%}, full_dr {b['full_dr']:.0%}, nominal {b['nominal']:.0%})  "
                  f"{time.perf_counter() - t:.1f}s")
            runs.append(r)
        fresh = {r["id"]: r for r in runs}
        kept = [fresh.pop(r["id"], r) for r in bundle["runs"]]
        bundle["runs"] = kept + list(fresh.values())
        bundle["params"] = {k: {"nominal": p.nominal, "low": p.low, "high": p.high, "unit": p.unit, "kind": p.kind, "hint": p.hint}
                            for k, p in PARAM_SPACE.items()}
        if llm is not None:
            bundle["stack"]["open_agent"] = f"Nemotron tool agent ({llm.model_for('diagnose')}) via {a.llm}"
            bundle["stack"]["llm_usage_open"] = vars(llm.usage)
        if a.attach_bench:
            bundle["open_benchmark"] = json.loads(a.attach_bench.read_text())
        path.write_text(json.dumps(bundle, indent=1))
        print(f"updated {path} ({', '.join(r['id'] for r in runs)})")
        return
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
