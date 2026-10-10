"""Cross-engine check: Tether calibrates from data made by a different engine (MuJoCo) and is scored there.

Each hidden world is a MuJoCo table (contact-driven launch, soft contacts) with a random friction, friction region,
actuator gain and camera pitch, plus one of four conditions:

  in-menu      only effects the Studio's fitter has fields for (but a different engine makes the data)
  speed        friction that changes with sliding speed (off-menu)
  two regions  a second friction region further out (off-menu: the fitter has one)
  slope        the table tilts along the push axis (off-menu, but indistinguishable from a friction change)

For every world the robot writes the same first-day log as the Studio's samples (a nominal-sim policy aiming at 14
targets plus 4 probe pushes), the Studio analyses it offline (no LLM), and then:

  - flagged: does the analysis say the model leaves the data unexplained?
  - hold-out: stop error on 12 new pushes in the hidden world, calibrated sim vs the current sim
  - success: a policy trained in each simulator, run in the hidden world on 24 targets (+-3 cm):
      current sim, domain randomization around the current sim at 4 widths (best width reported),
      domain randomization over the hidden worlds' own distribution (an oracle no user has), and Tether's ranges
  - coverage: hidden values inside the 90% intervals (in-menu fields only; MuJoCo's effective friction differs from
    its parameter by a few %, see sim/mujoco_push.py, so this is reported but not the headline)

Run: .venv/bin/python -m eval.cross_engine [--worlds 24]   (CPU, a few minutes; writes runs/proof/cross_engine.json)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from sim.mujoco_push import MujocoPushEnv
from sim.params import PARAM_SPACE, ParamSet
from sim.push_task import SUCCESS_TOL, TablePolicy, eval_targets, launch_command, unobserve
from studio.session import Push, Session

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "proof" / "cross_engine.json"
CONDITIONS = ("in-menu", "speed", "two regions", "slope")
DR_WIDTHS = (0.25, 0.5, 0.75, 1.0)  # fraction of each parameter's plausible half-range around the current sim
GRID = tuple(round(0.05 + 0.005 * i, 4) for i in range(171))
PROBES = [1.5, 2.0, 2.5, 2.9]
NOISE = {"stop": 0.003, "track": 0.0015, "perceived": 0.002}


def hidden_world(rng: random.Random) -> dict:
    mu = rng.uniform(0.35, 0.8)
    return {"object_mu": mu, "table_mu": mu, "actuator_gain": rng.uniform(0.8, 1.15), "patch_y0": rng.uniform(0.25, 0.45),
            "patch_mu": mu * rng.uniform(0.45, 0.75), "camera_pitch_deg": rng.uniform(-2.5, 2.5)}


def effects_for(cond: str, h: dict, rng: random.Random) -> dict:
    sign = rng.choice((-1.0, 1.0))
    if cond == "speed":
        return {"speed_slope": sign * rng.uniform(0.1, 0.2)}
    if cond == "two regions":
        return {"region2": (h["patch_y0"] + rng.uniform(0.12, 0.2), min(1.0, h["patch_mu"] * rng.uniform(1.5, 1.9)))}
    if cond == "slope":
        return {"slope_deg": sign * rng.uniform(1.5, 3.0)}
    return {}


def train_on(worlds: list[ParamSet]) -> TablePolicy:
    """The Studio's policy trainer for a simulator: per perceived distance, the median command over its worlds."""
    cmds = []
    for o in GRID:
        per = sorted(launch_command(unobserve(o, w), w) for w in worlds)
        cmds.append(per[len(per) // 2])
    return TablePolicy(GRID, tuple(cmds))


def dr_worlds(width: float, rng: random.Random, n: int = 31) -> list[ParamSet]:
    nom = ParamSet.nominal()
    out = []
    for _ in range(n):
        ch = {}
        for k in ("object_mu", "actuator_gain", "camera_pitch_deg"):
            p = PARAM_SPACE[k]
            ch[k] = rng.uniform(p.nominal - width * (p.nominal - p.low), p.nominal + width * (p.high - p.nominal))
        ch["table_mu"] = ch["object_mu"]
        out.append(nom.with_(**ch))
    return out


def oracle_worlds(rng: random.Random, n: int = 31) -> list[ParamSet]:
    return [ParamSet.nominal().with_(**hidden_world(rng)) for _ in range(n)]


def first_day_log(env: MujocoPushEnv, hidden: ParamSet, seed: int) -> Session:
    rng = random.Random(seed)
    pol = train_on([ParamSet.nominal()])
    pushes = []
    for t in env.rollout(hidden, pol, eval_targets(14, 77)).trials + env.push(hidden, PROBES).trials:
        aimed = t.target == t.target
        pushes.append(Push(round(t.slide + rng.gauss(0, NOISE["stop"]), 4), t.command, None,
                           round(t.target, 4) if aimed else None,
                           round(t.observed + rng.gauss(0, NOISE["perceived"]), 4) if aimed else None,
                           t.tipped, [round(y + rng.gauss(0, NOISE["track"]), 4) for y in t.track]))
    return Session("cross-engine", "log", pushes)


def success(env: MujocoPushEnv, hidden: ParamSet, pol, targets: list[float]) -> float:
    ro = env.rollout(hidden, pol, targets)
    return sum(abs(t.slide - t.target) <= SUCCESS_TOL for t in ro.trials) / len(targets)


def one(i: int, cond: str, h: dict, eff: dict, policies: dict) -> dict:
    from agent.tool_agent import to_params
    from studio.fit import fit_base
    from sim.push_task import slide_distance
    from studio import structure
    from studio.pipeline import analyze

    env = MujocoPushEnv(eff)
    hidden = ParamSet.nominal().with_(**h)
    s = first_day_log(env, hidden, seed=1000 + i)
    cal = analyze(s, None).calibration
    base = fit_base(s)
    ens = [to_params(m, base) for m in (cal.ensemble or [cal.model])]
    tether = train_on(ens)
    targets = eval_targets(24, 99 + i)
    out = {"world": i, "condition": cond, "hidden": h, "effects": {k: v for k, v in eff.items()},
           "structure": cal.structure, "stop_rms_m": cal.residuals["stop_residual_rms_m"],
           "size_flag": bool(cal.residuals["unexplained"])}
    st = structure.check(s, cal.params)
    out.update(flagged=bool(st["findings"]), findings=[f["kind"] for f in st["findings"]], structure_tests=st["tests"],
               tipped=sum(p.tipped for p in s.pushes))
    # hold-out pushes in the hidden world: how well each simulator predicts where the cube stops
    hold = [0.9 + 0.2 * k for k in range(12)]
    ro = env.push(hidden, hold).trials
    calp = cal.params
    err = lambda p: math.sqrt(statistics.fmean((slide_distance(t.command, p) - t.slide) ** 2 for t in ro if not t.tipped))  # noqa: E731
    out["holdout_rms_m"] = {"tether": round(err(calp), 4), "current": round(err(base), 4)}
    out["success"] = {name: round(success(env, hidden, pol, targets), 3) for name, pol in policies.items()}
    out["success"]["tether"] = round(success(env, hidden, tether, targets), 3)
    # ceiling: a policy trained on the hidden world's exact parameters in the analytic sim (the engine gap remains)
    out["success"]["exact params"] = round(success(env, hidden, train_on([hidden]), targets), 3)
    truth = {**h, "mu_eff": h["object_mu"]}
    out["coverage"] = {k: bool(lo <= truth[k] <= hi) for k, (lo, hi) in cal.intervals.items() if k in truth}
    out["intervals"] = {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in cal.intervals.items()}
    return out


def ci95(xs: list[float]) -> list[float]:
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return [round(m, 3), round(m, 3)]
    h = 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))
    return [round(m - h, 3), round(m + h, 3)]


def summarize(rows: list[dict]) -> dict:
    out = {}
    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        names = list(rs[0]["success"])
        mean = {n: round(statistics.fmean(r["success"][n] for r in rs), 3) for n in names}
        dr = [n for n in names if n.startswith("dr ") and n != "dr oracle"]
        best_dr = max(dr, key=lambda n: mean[n])
        cov = [v for r in rs for v in r["coverage"].values()]
        out[cond] = {
            "worlds": len(rs),
            "success_mean": mean, "success_ci95": {n: ci95([r["success"][n] for r in rs]) for n in names},
            "best_dr_width": best_dr,
            "tether_beats_best_dr": sum(r["success"]["tether"] > r["success"][best_dr] for r in rs),
            "tether_ties_best_dr": sum(r["success"]["tether"] == r["success"][best_dr] for r in rs),
            "flagged": sum(r["flagged"] for r in rs), "size_flagged": sum(r["size_flag"] for r in rs),
            "findings": {k: sum(k in r["findings"] for r in rs) for k in ("speed", "position")},
            "holdout_rms_m": {k: round(statistics.median(r["holdout_rms_m"][k] for r in rs), 4) for k in ("tether", "current")},
            "coverage": round(sum(cov) / len(cov), 3) if cov else None, "values": len(cov),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=24)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    a = ap.parse_args()
    t0 = time.time()
    rng = random.Random(11)
    policies = {"current": train_on([ParamSet.nominal()])}
    for w in DR_WIDTHS:
        policies[f"dr {int(w * 100)}%"] = train_on(dr_worlds(w, random.Random(5)))
    policies["dr oracle"] = train_on(oracle_worlds(random.Random(6)))
    conds = [c for c in CONDITIONS if c in a.conditions.split(",")]
    rows = []
    for i in range(a.worlds):
        h = hidden_world(rng)
        erng = random.Random(100 + i)
        for cond in conds:
            r = one(i, cond, h, effects_for(cond, h, erng), policies)
            rows.append(r)
            print(f"world {i:2d} {cond:11s} flagged {str(r['flagged']):5s} rms {r['stop_rms_m']:.4f} "
                  f"holdout {r['holdout_rms_m']['tether']:.3f}/{r['holdout_rms_m']['current']:.3f} "
                  f"success " + " ".join(f"{k} {v:.2f}" for k, v in r["success"].items()), flush=True)
    summary = summarize(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"engine": "MuJoCo " + __import__("mujoco").__version__, "seconds": round(time.time() - t0, 1),
                               "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
