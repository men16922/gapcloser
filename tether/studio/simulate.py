"""Simulated data for the Studio: when filming a real table is not practical, the visitor defines the "real"
table and NVIDIA Newton produces the measurements, either a phone-style video (rendered by Newton's
SensorTiledCamera) or a robot push log. The Studio then diagnoses that data without seeing the settings,
and the visitor's own settings are the ground truth revealed at the end.

World (all optional except mu_eff):
  mu_eff            sliding friction of the object on the table                     0.20 - 1.00
  patch_y0/patch_mu a region of different friction starting patch_y0 m out          0.15 - 0.80 / 0.20 - 0.95
  actuator_gain     robot logs: real launch speed per unit command                  0.70 - 1.30
  camera_pitch_deg  robot logs: tilt of the robot's camera (distance error)          -5 - 5
Pushes: how many (3-12) and how far the longest one slides (0.15-0.9 m).
"""

from __future__ import annotations

import random
from pathlib import Path

from tether.sim.params import ParamSet, Randomization
from tether.sim.push_task import InverseTrainer, launch_command
from tether.studio.session import Push, Session, SessionError

BOUNDS = {"mu_eff": (0.2, 1.0), "patch_y0": (0.15, 0.8), "patch_mu": (0.2, 0.95), "actuator_gain": (0.7, 1.3),
          "camera_pitch_deg": (-5.0, 5.0)}
MAX_PUSHES = 12
REACH = (0.15, 0.9)
NOISE = {"stop": 0.003, "track": 0.0015, "perceived": 0.002}  # robot-log measurement noise, as in the samples


def validate(world: dict, kind: str) -> dict:
    out = {}
    for k, v in (world or {}).items():
        if v is None or k not in BOUNDS:
            continue
        lo, hi = BOUNDS[k]
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise SessionError(f"{k} must be a number") from None
        if not lo <= x <= hi:
            raise SessionError(f"{k} must be between {lo} and {hi} (got {x})")
        out[k] = x
    if "mu_eff" not in out:
        raise SessionError("set the table friction (mu_eff)")
    if ("patch_y0" in out) != ("patch_mu" in out):
        raise SessionError("a friction region needs both a start (patch_y0) and a friction (patch_mu)")
    if kind == "video":
        for k in ("actuator_gain", "camera_pitch_deg"):
            if k in out:
                raise SessionError(f"{k} only applies to robot logs (a hand flick has no command or robot camera)")
    return out


def params_of(world: dict) -> ParamSet:
    ch = {"object_mu": world["mu_eff"], "table_mu": world["mu_eff"]}
    for k in ("patch_y0", "patch_mu", "actuator_gain", "camera_pitch_deg"):
        if k in world:
            ch[k] = world[k]
    return ParamSet.nominal().with_(**ch)


def flick_speeds(world: dict, n: int, reach: float, seed: int = 0) -> list[float]:
    """Hand-flick launch speeds whose slides spread from ~0.1 m to `reach` in this world (shuffled, like a person)."""
    if not 3 <= n <= MAX_PUSHES:
        raise SessionError(f"between 3 and {MAX_PUSHES} pushes")
    if not REACH[0] <= reach <= REACH[1]:
        raise SessionError(f"the longest push must slide {REACH[0]}-{REACH[1]} m")
    p = params_of(world)
    rng = random.Random(seed)
    dists = [0.1 + (reach - 0.1) * i / (n - 1) for i in range(n)]
    speeds = [round(launch_command(d * rng.uniform(0.95, 1.05), p.with_(actuator_gain=1.0)), 3) for d in dists]
    rng.shuffle(speeds)
    return speeds


def render_video(world: dict, speeds: list[float], out: Path, seed: int = 11, domain: str = "robot") -> dict:
    from tether.studio.video_sample import project, render, sheet_corners_world

    meta = render(out, speeds, seed, world=world, show_strip=False, domain=domain)
    meta["sheet_corners_px"] = [[round(float(x), 1), round(float(y), 1)] for x, y in project(sheet_corners_world())]
    return meta


def robot_log(world: dict, n: int, reach: float, seed: int = 0, commands: list[float] | None = None) -> Session:
    """A robot's first day: a policy trained in the nominal simulator aims at targets up to `reach` (NVIDIA Newton
    plays the real table), or explicit probe commands when given."""
    from tether.sim.newton_push import NewtonPushEnv

    hidden = params_of(world)
    env = NewtonPushEnv()
    rng = random.Random(seed)
    pushes = []
    if commands:
        trials = env.push(hidden, list(commands)).trials
    else:
        if not 3 <= n <= MAX_PUSHES * 2:
            raise SessionError(f"between 3 and {MAX_PUSHES * 2} pushes")
        lo = 0.1
        targets = [lo + (max(reach, 0.2) - lo) * x for x in sorted(rng.random() for _ in range(n))]
        pol = InverseTrainer().train(Randomization.none())
        trials = env.rollout(hidden, pol, targets).trials
    for t in trials:
        aimed = t.target == t.target
        pushes.append(Push(round(t.slide + rng.gauss(0, NOISE["stop"]), 4), t.command, None,
                           round(t.target, 4) if aimed else None,
                           round(t.observed + rng.gauss(0, NOISE["perceived"]), 4) if aimed else None,
                           t.tipped, [round(y + rng.gauss(0, NOISE["track"]), 4) for y in t.track],
                           "added" if commands else "log"))
    return Session("Simulated robot log", "log", pushes)


def truth(world: dict) -> dict:
    model = {"mu_eff": world["mu_eff"], "patch_y0": world.get("patch_y0"), "patch_mu": world.get("patch_mu")}
    for k in ("actuator_gain", "camera_pitch_deg"):
        if k in world:
            model[k] = world[k]
    return {"source": "your settings, simulated in NVIDIA Newton", "model": model}
