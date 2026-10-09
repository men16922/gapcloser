"""Retrain and test: does training on Tether's measured physics give a better policy than the alternatives?

Three training conditions, same learner, same budget, all in parallel NVIDIA Newton worlds:

  current   your current simulator, no randomization (what you had before Tether)
  wide      wide domain randomization: friction 0.2-1.0 and (for commanded actuators) gain 0.7-1.3 over a uniform
            surface, the usual answer when nobody knows the real physics
  tether    worlds drawn from Tether's bootstrap ensemble: the measured friction, region and actuator, each varied
            only as much as the data allows (the Isaac Lab ranges Tether exports)

Learner: a command table over the perceived target distance, learned by trial in the training worlds. Each
iteration pushes every training world at every table point in one Newton model and moves each command by the
median over worlds of the slide error (iterative learning control; slide grows with speed squared). No gradients,
no access to the hidden world. After every iteration the policy is also run in the hidden "real" world (only
when the session has a known truth: samples and simulated sessions) to draw the real success curve.
"""

from __future__ import annotations

import math
import random
import time

from agent.tool_agent import to_params
from sim.params import ParamSet
from sim.push_task import SUCCESS_TOL, TARGET_RANGE, eval_targets, observe, unobserve
from studio.fit import Calibration, fit_base
from studio.session import Session

GRID = tuple(round(TARGET_RANGE[0] - 0.05 + i * 0.05, 3) for i in range(11))  # perceived 0.15 .. 0.65 m
N_WORLDS = 16
ITERS = 10
LR = 0.5
CONDITIONS = {"current": "Your current simulator", "wide": "Wide domain randomization", "tether": "Tether's measured ranges"}


class TablePolicy:
    def __init__(self, cmds: list[float]):
        self.cmds = list(cmds)

    def command(self, o: float) -> float:
        g, c = GRID, self.cmds
        if o <= g[0]:
            return c[0] * math.sqrt(max(o, 0.0) / g[0])
        if o >= g[-1]:
            return c[-1] * math.sqrt(o / g[-1])
        i = min(int((o - g[0]) / (g[1] - g[0])), len(g) - 2)
        u = (o - g[i]) / (g[i + 1] - g[i])
        return c[i] + u * (c[i + 1] - c[i])


def training_worlds(cond: str, session: Session, cal: Calibration, n: int | None = None, seed: int = 0) -> list[ParamSet]:
    n = n or N_WORLDS
    rng = random.Random(seed)
    base = fit_base(session)
    video = not session.has_commands
    if cond == "current":
        return [base] * n
    if cond == "wide":
        out = []
        for _ in range(n):
            mu = rng.uniform(0.2, 1.0)
            out.append(base.with_(object_mu=mu, table_mu=mu, actuator_gain=1.0 if video else rng.uniform(0.7, 1.3)))
        return out
    ens = cal.ensemble or [cal.model]
    out = [to_params(rng.choice(ens), base) for _ in range(n)]
    return [w.with_(actuator_gain=1.0) for w in out] if video else out


def _push(worlds: list[ParamSet], cmds: list[float]):
    from sim.newton_push import push_worlds

    return push_worlds(worlds, cmds)


def train(session: Session, cal: Calibration, hidden: ParamSet | None, iters: int | None = None, on_iter=None) -> dict:
    """Train a policy under each condition; return learning curves (training worlds and, when known, the hidden
    world), final policies and a few Newton tracks per iteration for the parallel-worlds view."""
    t0 = time.time()
    iters = iters or ITERS
    video = not session.has_commands
    if hidden is not None and video:
        hidden = hidden.with_(actuator_gain=1.0)
    real_targets = eval_targets(24, 99)
    init = math.sqrt(2 * 0.5 * 9.81)  # same neutral start for every condition: friction 0.5, gain 1
    out = {"grid_m": list(GRID), "iterations": iters, "n_worlds": len(training_worlds("current", session, cal)), "tolerance_m": SUCCESS_TOL,
           "target_range_m": list(TARGET_RANGE), "hidden_known": hidden is not None, "conditions": {}}
    for cond, label in CONDITIONS.items():
        worlds = training_worlds(cond, session, cal)
        pol = TablePolicy([init * math.sqrt(g) for g in GRID])
        curve_train, curve_real, lanes, history = [], [], [], []
        for it in range(iters + 1):
            # every world tries every table point: the target it aims at is what that world's camera reports there
            history.append([round(c, 5) for c in pol.cmds])
            ws, cmds, tgts, ks = [], [], [], []
            for w in worlds:
                for k, o in enumerate(GRID):
                    ws.append(w)
                    cmds.append(pol.cmds[k])
                    tgts.append(unobserve(o, w))
                    ks.append(k)
            trials = _push(ws, cmds)
            ok = [abs(tr.slide - t) <= SUCCESS_TOL for tr, t in zip(trials, tgts) if TARGET_RANGE[0] <= t <= TARGET_RANGE[1]]
            curve_train.append(round(sum(ok) / max(1, len(ok)), 3))
            real = None
            if hidden is not None:
                obs = [observe(t, hidden) for t in real_targets]
                rt = _push([hidden] * len(real_targets), [pol.command(o) for o in obs])
                real = round(sum(abs(tr.slide - t) <= SUCCESS_TOL for tr, t in zip(rt, real_targets)) / len(real_targets), 3)
            curve_real.append(real)
            # a few lanes for the parallel-worlds view: 8 worlds at the middle table point (0.40 m)
            mid = GRID.index(0.4) if 0.4 in GRID else len(GRID) // 2
            lanes.append([{"target": round(tgts[i], 4), "track": trials[i].track[::2], "slide": round(trials[i].slide, 4),
                           "mu": round((ws[i]["object_mu"] + ws[i]["table_mu"]) / 2, 3)}
                          for i in range(len(trials)) if ks[i] == mid][:8])
            if on_iter:
                on_iter({"condition": cond, "iteration": it, "train": curve_train[-1], "real": real})
            if it == iters:
                break
            # move each command by the median slide error over worlds (slide ~ speed^2)
            for k in range(len(GRID)):
                ratios = sorted(t / max(tr.slide, 1e-3) for tr, t, kk in zip(trials, tgts, ks) if kk == k)
                r = ratios[len(ratios) // 2]
                pol.cmds[k] *= r ** (0.5 * LR)  # noqa: module-level LR read at call time
        distinct = len({(w["object_mu"], w["table_mu"], w["actuator_gain"], w["patch_y0"], w["patch_mu"]) for w in worlds})
        out["conditions"][cond] = {"label": label, "train": curve_train, "real": curve_real, "final_real": curve_real[-1],
                                   "policy": [round(c, 4) for c in pol.cmds], "lanes": lanes, "history": history, "distinct_worlds": distinct,
                                   "mu_range": [round(min((w["object_mu"] + w["table_mu"]) / 2 for w in worlds), 3),
                                                round(max((w["object_mu"] + w["table_mu"]) / 2 for w in worlds), 3)]}
    out["seconds"] = round(time.time() - t0, 1)
    out["engine"] = "NVIDIA Newton (XPBD, CPU), parallel worlds"
    return out


def hidden_params(truth_model: dict | None, session: Session) -> ParamSet | None:
    """The hidden "real" world as a ParamSet, from a revealed truth model (Studio frame)."""
    if not truth_model:
        return None
    m = {"mu_eff": truth_model["mu_eff"], "actuator_gain": truth_model.get("actuator_gain", 1.0),
         "patch_y0": truth_model.get("patch_y0"), "patch_mu": truth_model.get("patch_mu"),
         "camera_dx": truth_model.get("camera_dx", 0.0), "camera_pitch_deg": truth_model.get("camera_pitch_deg", 0.0),
         "lens_k": truth_model.get("lens_k", 0.0)}
    return to_params(m, fit_base(session))
