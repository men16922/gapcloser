"""Does the next-experiment card save real pushes? Active vs passive data collection on hidden worlds.

Every method starts from the same 4 short pushes (stops < ~0.3 m, the far table unmeasured) and adds 2 real
pushes per round, then recalibrates with the Studio (offline structure search + bootstrap):
  suggested   the Studio's two top next-experiment pushes (query by committee)
  random      two uniformly random commands over the robot's command range
  sweep       a fixed long-to-short ladder, the protocol an engineer would script without feedback
After each round a policy is trained in the calibrated sim and evaluated in the hidden world (40 targets,
0.2-0.6 m); a world is "closed" at the first round with >= 95% real success. Reported: real pushes needed.

Run: .venv/bin/python -m tether.eval.studio_bench --worlds 20 [--env newton]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

from tether.sim.params import ParamSet, Randomization, effective_friction
from tether.sim.push_task import AnalyticPushEnv, InverseTrainer, eval_targets
from tether.studio import design
from tether.studio.fit import calibrate
from tether.studio.session import Push, Session

CMD_RANGE = (0.8, 3.6)
SWEEP = (3.4, 2.4, 3.0, 2.0, 2.7, 1.6, 3.2, 2.2)
MAX_ROUNDS = 6
TARGET = 0.95


def hidden_world(rng: random.Random) -> ParamSet:
    """Dry table with a strip of different friction somewhere in the far half, and a possibly weak actuator."""
    mu = rng.uniform(0.45, 0.8)
    sign = rng.choice([-1, 1]) if mu < 0.75 else -1
    return ParamSet.nominal().with_(object_mu=mu, table_mu=mu, actuator_gain=rng.uniform(0.85, 1.1),
                                    patch_y0=rng.uniform(0.33, 0.48), patch_mu=min(0.95, max(0.25, mu + sign * rng.uniform(0.2, 0.4))))


def start_commands(h: ParamSet) -> list[float]:
    """4 short pushes: commands chosen (by the bench, from the truth) so stops land in 0.12-0.28 m."""
    import math

    mu, g = effective_friction(h["object_mu"], h["table_mu"]), h["actuator_gain"]
    return [math.sqrt(2 * mu * 9.81 * d) / g for d in (0.12, 0.17, 0.22, 0.28)]


def push(env, h: ParamSet, cmds: list[float], rng: random.Random) -> list[Push]:
    ro = env.push(h, cmds)
    return [Push(t.slide + rng.gauss(0, 0.003), t.command, track=[y + rng.gauss(0, 0.0015) for y in t.track], tipped=t.tipped)
            for t in ro.trials]


def real_success(env, h: ParamSet, cal_params: ParamSet, targets: list[float]) -> float:
    pol = InverseTrainer().train(Randomization({k: (cal_params[k], cal_params[k]) for k in cal_params.values}))
    return env.rollout(h, pol, targets).success_rate


def run_world(env, h: ParamSet, method: str, seed: int, targets: list[float]) -> dict:
    rng = random.Random(seed)
    pushes = push(env, h, start_commands(h), rng)
    hist = []
    sweep = list(SWEEP)
    for rnd in range(MAX_ROUNDS + 1):
        s = Session("bench", "log", pushes)
        cal = calibrate(s, n_boot=10)
        succ = real_success(env, h, cal.params, targets)
        hist.append({"pushes": len(pushes), "success": round(succ, 3), "structure": cal.structure})
        if succ >= TARGET or rnd == MAX_ROUNDS:
            break
        if method == "suggested":
            nxt = design.suggest(s, cal, k=2)
            cmds = [x["command"] for x in nxt["suggestions"]][:2]
            while len(cmds) < 2:
                cmds.append(rng.uniform(*CMD_RANGE))
        elif method == "random":
            cmds = [rng.uniform(*CMD_RANGE) for _ in range(2)]
        else:
            cmds = [sweep.pop(0), sweep.pop(0)]
        pushes += push(env, h, cmds, rng)
    closed = hist[-1]["success"] >= TARGET
    return {"method": method, "closed": closed, "pushes": hist[-1]["pushes"] if closed else None, "history": hist}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=20)
    ap.add_argument("--env", default="analytic", choices=["analytic", "newton"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.env == "newton":
        from tether.sim.newton_push import NewtonPushEnv

        env = NewtonPushEnv()
    else:
        env = AnalyticPushEnv()
    targets = eval_targets(40, 4242)
    rows = []
    t0 = time.time()
    for w in range(a.worlds):
        h = hidden_world(random.Random(f"studio-bench-{w}"))
        for m in ("suggested", "random", "sweep"):
            r = run_world(env, h, m, 1000 + w, targets)
            r["world"] = w
            rows.append(r)
        print(f"world {w}: " + "  ".join(f"{r['method']} {r['pushes']}" for r in rows[-3:]), flush=True)
    print(f"\n{a.worlds} worlds, {a.env}, {time.time() - t0:.0f} s. Real pushes to reach {TARGET:.0%} success "
          f"(start: 4 short pushes; +2 per round, max {4 + 2 * MAX_ROUNDS}):")
    summary = {}
    for m in ("suggested", "random", "sweep"):
        rs = [r for r in rows if r["method"] == m]
        done = [r["pushes"] for r in rs if r["closed"]]
        # unresolved worlds count as the cap + 2 (they needed more than the bench allowed)
        cost = [r["pushes"] if r["closed"] else 4 + 2 * MAX_ROUNDS + 2 for r in rs]
        summary[m] = {"closed": len(done), "worlds": len(rs), "median_pushes": statistics.median(cost),
                      "mean_pushes": round(statistics.mean(cost), 2)}
        print(f"  {m:9s} closed {len(done):2d}/{len(rs)}  median {statistics.median(cost):4.1f}  mean {statistics.mean(cost):5.2f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"env": a.env, "worlds": a.worlds, "summary": summary, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
