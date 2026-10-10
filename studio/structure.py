"""What the calibrated model still leaves unexplained, as structure rather than size.

A residual threshold ("stops off by more than 1 cm") cannot tell missing physics from launch-to-launch scatter: a
real arm's launch speed varies by 1-2%, which alone moves a 40 cm stop by about a centimetre. Missing physics
leaves a pattern instead. Along every tracked slide, the local deceleration is measured (7-frame Savitzky-Golay
fits) and the same measure of the calibrated model's own predicted slide is subtracted. What remains is tested for two patterns:

  speed     deceleration that changes with sliding speed (friction depends on speed: wet, greasy, rubbery)
  position  deceleration that changes again somewhere along the table (a friction region the model lacks)

Both enter one regression with standard errors clustered by push (samples within one slide are correlated, and
each push may sit a little above or below the model as a whole: that is scatter, not structure). A pattern is
reported only when it is large in physical terms and clearly beyond the scatter.
"""

from __future__ import annotations

import math

import numpy as np

from agent.tool_agent import at_start
from sim.params import GRAVITY, ParamSet
from sim.push_task import FRAME_DT, analytic_track, launch_command
from studio.session import Session

SG_ACC = np.array([5.0, 0.0, -3.0, -4.0, -3.0, 0.0, 5.0]) / (42.0 * FRAME_DT * FRAME_DT)
SG_VEL = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) / (28.0 * FRAME_DT)
MIN_SPEED = 0.25  # m/s: slower windows reach the stop (or measure noise at rest)
T_CRIT = 4.0  # |t| for a pattern to count (several candidate positions are scanned)
MIN_SPEED_EFFECT = 0.03  # friction change per m/s worth reporting
MIN_STEP = 0.04  # friction step worth reporting


def samples(session: Session, p: ParamSet) -> tuple[np.ndarray, ...]:
    """(push index, table position, speed, deceleration residual in g) for every usable window. The model side is
    the calibrated model's own track for a push that stops where this one did, filtered the same way, so the
    smoothing across a friction boundary cancels instead of looking like structure."""
    rows = []
    for i, d in enumerate(session.pushes):
        if d.tipped or len(d.track) < 9:
            continue
        tr = np.asarray(d.track, dtype=float)
        q = at_start(p, d.start)
        model = np.asarray(analytic_track(launch_command(max(d.stop, 1e-3), q), q, len(tr) - 1))
        for k in range(3, len(tr) - 3):
            w = tr[k - 3:k + 4]
            v = float(SG_VEL @ w)
            if v < MIN_SPEED or (tr[k + 3] - tr[k + 2]) / FRAME_DT < 0.12:
                continue
            a = -float(SG_ACC @ w) / GRAVITY
            a_model = -float(SG_ACC @ model[k - 3:k + 4]) / GRAVITY
            rows.append((i, tr[k] + d.start, v, a - a_model))
    if not rows:
        return tuple(np.zeros(0) for _ in range(4))
    return tuple(np.array(c) for c in zip(*rows))


def _ols_cluster(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    groups = np.unique(g)
    meat = sum(np.outer(X[g == k].T @ u[g == k], X[g == k].T @ u[g == k]) for k in groups)
    n_g = len(groups)
    cov = xtx_inv @ meat @ xtx_inv * (n_g / max(1, n_g - 1))
    return beta, np.sqrt(np.maximum(np.diag(cov), 1e-18))


def check(session: Session, p: ParamSet) -> dict:
    """Patterns left in the deceleration after the calibrated model `p`. Empty `findings` means only scatter."""
    g, y, v, e = samples(session, p)
    out = {"windows": int(len(e)), "pushes": int(len(np.unique(g))) if len(g) else 0, "findings": [], "tests": {}}
    if out["pushes"] < 6 or len(e) < 40:
        out["note"] = "too few tracked slides to look for patterns"
        return out
    out["scatter_g"] = round(float(np.std(e)), 4)
    # candidate positions for an extra friction change: where at least 3 pushes slide on both sides
    reach = np.array([y[g == k].max() for k in np.unique(g)])
    lo, hi = float(np.min(y)) + 0.05, float(np.quantile(reach, 0.75)) - 0.02
    cands = [round(c, 3) for c in np.arange(lo, hi, 0.02)] if hi > lo else []
    best = None
    for c in cands:
        step = (y >= c).astype(float)
        if step.sum() < 20 or len(np.unique(g[step > 0])) < 3:
            continue
        X = np.column_stack([np.ones_like(e), v - 1.0, step])
        beta, se = _ols_cluster(X, e, g)
        if best is None or abs(beta[2] / se[2]) > abs(best[1][2] / best[2][2]):
            best = (c, beta, se)
    X = np.column_stack([np.ones_like(e), v - 1.0])
    beta, se = _ols_cluster(X, e, g)
    speed_t, speed_b = float(beta[1] / se[1]), float(beta[1])
    if best is not None:  # with a step in the model, the speed term is tested net of it
        speed_t, speed_b = float(best[1][1] / best[2][1]), float(best[1][1])
    out["tests"]["speed"] = {"per_mps": round(speed_b, 4), "t": round(speed_t, 1)}
    if best is not None:
        c, b, s = best
        out["tests"]["position"] = {"from_m": float(c), "step": round(float(b[2]), 4), "t": round(float(b[2] / s[2]), 1)}
    speed_found = abs(speed_t) > T_CRIT and abs(speed_b) >= MIN_SPEED_EFFECT
    if speed_found:
        trend = "rises as the object slows" if speed_b < 0 else "falls as the object slows"
        out["findings"].append({"kind": "speed", "per_mps": round(speed_b, 3), "t": round(speed_t, 1),
                                "message": f"friction {trend} ({speed_b:+.2f} per m/s): it depends on sliding speed, "
                                           "which the model has no field for"})
    pos = out["tests"].get("position")
    # a speed effect also leaves steps along the table (a fitted region partly stands in for it): only a step that
    # appears without one is reported as its own finding
    if pos and not speed_found and abs(pos["t"]) > T_CRIT and abs(pos["step"]) >= MIN_STEP:
        out["findings"].append({"kind": "position", "from_m": pos["from_m"], "step": pos["step"], "t": pos["t"],
                                "message": f"deceleration still changes along the table (strongest near {pos['from_m']:.2f} m, "
                                           f"{pos['step']:+.2f}): the friction map needs a change the model does not have"})
    return out


def describe(result: dict) -> str:
    if result.get("note"):
        return result["note"]
    if not result["findings"]:
        return f"no pattern left: the remaining error is scatter (±{result.get('scatter_g', 0):.2f} g per window)"
    return "; ".join(f["message"] for f in result["findings"])


if __name__ == "__main__":  # quick look at the samples
    import sys
    from pathlib import Path

    from studio.pipeline import analyze
    from studio.session import load

    for name in sys.argv[1:] or ["lab-bench", "press-line", "brake-log"]:
        path = Path(__file__).parent / "samples" / f"{name}.csv"
        s = load(path.read_text(), path.name, name)
        cal = analyze(s, None).calibration
        r = check(s, cal.params)
        print(name, r["tests"], "->", describe(r))
    _ = math
