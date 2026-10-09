"""Sample sessions for the Studio, generated in NVIDIA Newton from a world whose physics the analysis never sees.

  lab-bench   robot log: a policy trained in the nominal sim aims at targets on a bench with a weak actuator, a
              slick strip far out and a pitched camera (the sort of log a robot writes on its first real day)
  short-reach robot log that only pushes short: the far table is unmeasured (shows the next-experiment logic)

Ground truth is written next to each sample (`*.truth.json`) and only shown after the diagnosis.
Run: .venv/bin/python -m studio.samples [--env analytic]   (Newton by default, CPU, ~1 min)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from sim.params import ParamSet, Randomization
from sim.push_task import AnalyticPushEnv, InverseTrainer, eval_targets
from studio.session import Push, Session, to_csv

OUT = Path(__file__).resolve().parent / "samples"

SAMPLES = {
    "lab-bench": {
        "title": "Lab bench, first day on the real robot",
        "hidden": {"object_mu": 0.7, "table_mu": 0.7, "actuator_gain": 0.88, "patch_y0": 0.38, "patch_mu": 0.45,
                   "camera_pitch_deg": 2.0},
        "targets": 16, "probes": [1.6, 2.2, 2.8, 3.2],
    },
    "short-reach": {
        "title": "Short pushes only (far table unmeasured)",
        "hidden": {"object_mu": 0.6, "table_mu": 0.6, "patch_y0": 0.33, "patch_mu": 0.4},
        "targets": 0, "probes": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.75, 1.85],
    },
}
TRACK_NOISE = 0.0015  # m, camera tracking jitter added to the simulated measurements
STOP_NOISE = 0.003


def make(name: str, env) -> tuple[Session, dict]:
    spec = SAMPLES[name]
    hidden = ParamSet.nominal().with_(**spec["hidden"])
    rng = random.Random(name)
    pushes: list[Push] = []
    if spec["targets"]:
        pol = InverseTrainer().train(Randomization.none())
        ro = env.rollout(hidden, pol, eval_targets(spec["targets"], 77))
        for t in ro.trials:
            pushes.append(Push(t.slide + rng.gauss(0, STOP_NOISE), t.command, None, t.target, t.observed, t.tipped,
                               [y + rng.gauss(0, TRACK_NOISE) for y in t.track]))
    if spec["probes"]:
        for t in env.push(hidden, spec["probes"]).trials:
            pushes.append(Push(t.slide + rng.gauss(0, STOP_NOISE), t.command, None, None, None, t.tipped,
                               [y + rng.gauss(0, TRACK_NOISE) for y in t.track]))
    for p in pushes:
        p.stop, p.track = round(p.stop, 4), [round(y, 4) for y in p.track]
    s = Session(spec["title"], "sample", pushes)
    return s, {"name": name, "hidden": spec["hidden"], "env": type(env).__name__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="newton", choices=["newton", "analytic"])
    a = ap.parse_args()
    if a.env == "newton":
        from sim.newton_push import NewtonPushEnv

        env = NewtonPushEnv()
    else:
        env = AnalyticPushEnv()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SAMPLES:
        s, truth = make(name, env)
        (OUT / f"{name}.csv").write_text(to_csv(s))
        (OUT / f"{name}.truth.json").write_text(json.dumps({**truth, "title": s.name}, indent=2))
        print(name, len(s.pushes), "pushes, reach", [round(x, 3) for x in s.coverage()])


if __name__ == "__main__":
    main()
