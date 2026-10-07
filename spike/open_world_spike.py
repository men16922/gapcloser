"""Spike: do open-world faults break the rule-based agent, and does a correct sim fix them?

Analytic model with three faults outside the 10-param space:
  patch  - table friction changes to mu2 beyond y0 (wet/rough strip)
  dead   - actuator deadzone: v = gain * max(0, cmd - d0)
  sat    - actuator saturation: v = min(vmax, gain * cmd)
Policy = SimInversePolicy: per target, bisect the command so the *sim* slide equals the perceived target.
Compares: nominal sim, rule-fit sim (TrajectoryDiagnoser-style single gain + single mu_eff from tracks),
and oracle sim (true fault model).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

G = 9.81
TOL = 0.03
DT = 1 / 30


@dataclass(frozen=True)
class World:
    mu: float = 0.8           # mu_eff on the near table
    gain: float = 1.0
    patch_y0: float = 9.9     # friction patch start (m); 9.9 = none
    patch_mu: float = 0.8
    dead: float = 0.0         # command deadzone
    vmax: float = 99.0        # launch speed cap


def launch(w: World, cmd: float) -> float:
    return min(w.vmax, w.gain * max(0.0, cmd - w.dead))


def slide(w: World, cmd: float) -> float:
    v = launch(w, cmd)
    a1 = w.mu * G
    d1 = v * v / (2 * a1)
    if d1 <= w.patch_y0:
        return d1
    v2sq = v * v - 2 * a1 * w.patch_y0
    return w.patch_y0 + v2sq / (2 * w.patch_mu * G)


def track(w: World, cmd: float, n: int = 75) -> list[float]:
    """Full camera track (30 fps) of the cube position."""
    y, v, out = 0.0, launch(w, cmd), [0.0]
    sub = 20
    for _ in range(n):
        for _ in range(sub):
            a = (w.patch_mu if y >= w.patch_y0 else w.mu) * G
            if v <= 0:
                break
            v = max(0.0, v - a * DT / sub)
            y += v * DT / sub
        out.append(y)
    return out


def inverse_cmd(w: World, target: float) -> float:
    lo, hi = 0.0, 10.0
    if slide(w, hi) < target:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if slide(w, mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def success(real: World, sim: World, targets) -> float:
    return sum(abs(slide(real, inverse_cmd(sim, t)) - t) <= TOL for t in targets) / len(targets)


def rule_fit(real: World, sim: World, targets) -> World:
    """TrajectoryDiagnoser-style: launch speed from first 6 frames, deceleration from early motion,
    averaged over trials, mapped to one gain and one mu."""
    gs, decs = [], []
    for t in targets:
        cmd = inverse_cmd(sim, t)
        tr = track(real, cmd)[:7]
        v0 = (tr[1] - tr[0]) / DT + 0.5 * real.mu * G * DT  # good enough early estimate
        # least squares y = v0 t - a t^2/2 over first 6 frames
        pts = [(k * DT, y) for k, y in enumerate(tr)]
        s11 = sum(t_ * t_ for t_, _ in pts); s12 = sum(-0.5 * t_**3 for t_, _ in pts); s22 = sum(0.25 * t_**4 for t_, _ in pts)
        b1 = sum(t_ * y for t_, y in pts); b2 = sum(-0.5 * t_ * t_ * y for t_, y in pts)
        det = s11 * s22 - s12 * s12
        v0, a = (b1 * s22 - b2 * s12) / det, (s11 * b2 - s12 * b1) / det
        if cmd > 0 and v0 > 0:
            gs.append(v0 / cmd); decs.append(a / G)
    return replace(sim, gain=sum(gs) / len(gs), mu=sum(decs) / len(decs))


def main():
    rng = random.Random(0)
    targets = [rng.uniform(0.2, 0.6) for _ in range(20)]
    nominal = World()
    cases = {
        "closed: gain 0.8": World(gain=0.8),
        "closed: mu 0.5": World(mu=0.5),
        "patch: mu 0.8 -> 0.4 beyond 0.35 m": World(patch_y0=0.35, patch_mu=0.4),
        "patch: mu 0.8 -> 1.2 beyond 0.30 m": World(patch_y0=0.30, patch_mu=1.2),
        "deadzone 0.25": World(dead=0.25),
        "saturation vmax 2.2": World(vmax=2.2),
        "compound: gain 0.85 + patch 0.5@0.4 + dead 0.15": World(gain=0.85, patch_y0=0.4, patch_mu=0.5, dead=0.15),
    }
    print(f"{'world':52s} nominal  rule-fit  oracle")
    for name, real in cases.items():
        fit = rule_fit(real, nominal, targets)
        fit2 = rule_fit(real, fit, targets)  # second iteration
        print(f"{name:52s} {success(real, nominal, targets):6.0%}  {success(real, fit, targets):4.0%}/{success(real, fit2, targets):4.0%}  "
              f"{success(real, real, targets):5.0%}   fit gain {fit2.gain:.2f} mu {fit2.mu:.2f}")


if __name__ == "__main__":
    main()
