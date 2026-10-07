"""Render a small labeled set of Newton push clips (slid vs tipped) for the Cosmos Reason 2 spike.

Ground truth comes from the physics itself: the same params/command are simulated headless with
sim.newton_push._build/_simulate and the cube tilt is tracked every frame with tilt_deg; label = tipped if the peak tilt exceeds TIP_DEG=30
(a cube that rolls twice lands at ~0 deg final tilt, which the repo's final-only flag misses).
Run with the repo venv:  .venv/bin/python spike/cosmos/render_set.py [--holdout]
Writes spike/cosmos/clips/*.webp + labels.json (or clips_holdout/ with --holdout).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.newton_push import TIP_DEG, _build, _simulate, render_trial, render_trial_arm, tilt_deg  # noqa: E402
from sim.params import ParamSet  # noqa: E402

OUT = Path(__file__).parent / "clips"
TARGET = 0.45

# (name, param changes, command). table_mu=1.18 tips at higher launch speeds; nominal slides.
CASES = [
    ("nom-c20", {}, 2.0),
    ("nom-c26", {}, 2.6),
    ("nom-c30", {}, 3.0),
    ("nom-c34", {}, 3.4),
    ("nom-c10", {}, 1.0),  # short slide / stall
    ("slip-c20", {"table_mu": 0.3, "object_mu": 0.3}, 2.0),
    ("tip-c24", {"table_mu": 1.18}, 2.4),
    ("tip-c28", {"table_mu": 1.18}, 2.8),
    ("tip-c30", {"table_mu": 1.18}, 3.0),
    ("tip-c32", {"table_mu": 1.18}, 3.2),
    ("tip-c36", {"table_mu": 1.18}, 3.6),
    ("tip-c40", {"table_mu": 1.18}, 4.0),
    ("bounce-c30", {"restitution": 0.4}, 3.0),
]

# Held-out set (not used while choosing the prompt): other friction/size/light/camera combos.
HOLDOUT = [
    ("h-sticky105-c30", {"table_mu": 1.05}, 3.0),
    ("h-sticky105-c36", {"table_mu": 1.05}, 3.6),
    ("h-sticky110-c26", {"table_mu": 1.10}, 2.6),
    ("h-mu12-c34", {"table_mu": 1.2, "object_mu": 1.2}, 3.4),
    ("h-dark-tip-c32", {"table_mu": 1.18, "light_intensity": 0.6}, 3.2),
    ("h-bright-tip-c34", {"table_mu": 1.18, "light_intensity": 1.4}, 3.4),
    ("h-big-tip-c32", {"table_mu": 1.18, "object_half_size": 0.033}, 3.2),
    ("h-small-tip-c30", {"table_mu": 1.18, "object_half_size": 0.02}, 3.0),
    ("h-camdx-tip-c30", {"table_mu": 1.18, "camera_dx": 0.02}, 3.0),
    ("h-dark-slide-c30", {"light_intensity": 0.6}, 3.0),
    ("h-big-slide-c28", {"object_half_size": 0.033}, 2.8),
    ("h-small-slide-c24", {"object_half_size": 0.02}, 2.4),
    ("h-heavy-slide-c30", {"object_density": 1500.0}, 3.0),
    ("h-slip-c30", {"object_mu": 0.25}, 3.0),
    ("h-weak-slide-c30", {"actuator_gain": 0.76}, 3.0),
    ("h-camdx-slide-c30", {"camera_dx": -0.028}, 3.0),
]


def truth(params: ParamSet, command: float) -> dict:
    model, state, cubes, _ = _build(params, [command], [TARGET])
    peak = [0.0]

    def cb(s):
        peak[0] = max(peak[0], float(tilt_deg(s.body_q.numpy()[cubes][:, 3:7])[0]))

    final = _simulate(model, state, cb)
    q = final.body_q.numpy()[cubes]
    tilt = float(tilt_deg(q[:, 3:7])[0])
    return {"final_tilt_deg": tilt, "peak_tilt_deg": peak[0], "slide": float(q[0, 1]),
            "label": "tipped" if max(tilt, peak[0]) > TIP_DEG else "slid"}


def main() -> None:
    global OUT
    cases = CASES
    if "--holdout" in sys.argv:
        OUT, cases = OUT.parent / "clips_holdout", HOLDOUT
    OUT.mkdir(parents=True, exist_ok=True)
    labels = []
    for name, changes, cmd in cases:
        p = ParamSet.nominal().with_(**changes)
        gt = truth(p, cmd)
        for view, fn in (("side", render_trial), ("arm", render_trial_arm)):
            path = OUT / f"{name}-{view}.webp"
            if not path.exists():
                fn(p, cmd, TARGET, path)
            labels.append({"clip": path.name, "view": view, "params": changes, "command": cmd, **gt})
        print(name, gt, flush=True)
    (OUT / "labels.json").write_text(json.dumps(labels, indent=1))


if __name__ == "__main__":
    main()
