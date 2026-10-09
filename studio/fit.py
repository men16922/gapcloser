"""Calibration of a Session: model structure, fitted values, bootstrap intervals, predicted success.

The structure (which effects exist) comes from the Nemotron agent when one is configured, or from the
offline search below (the simplest structure whose residuals are within noise of the best one). Numbers
always come from the same least-squares fitter the agent uses (agent.tool_agent.Workbench), and the
uncertainty from a bootstrap over pushes (resample, refit) — never from a language model.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from itertools import combinations

from agent.tool_agent import FIELDS, Workbench, from_params, to_params
from sim.params import GRAVITY, PARAM_SPACE, ParamSet, Randomization
from sim.push_task import SUCCESS_TOL, TARGET_RANGE, InverseTrainer, Rollout, frictions, observe, slide_distance
from studio.session import Session

UNITS = {"mu_eff": "", "actuator_gain": "x", "patch_y0": "m", "patch_mu": "", "camera_dx": "m",
         "camera_pitch_deg": "deg", "lens_k": "1/m"}
LABELS = {"mu_eff": "Cube-table friction", "actuator_gain": "Actuator gain", "patch_y0": "Friction region starts at",
          "patch_mu": "Friction in that region", "camera_dx": "Camera offset", "camera_pitch_deg": "Camera pitch error",
          "lens_k": "Lens distortion"}
N_BOOT = 30
IDENT_MARGIN = 0.03  # a friction region needs this much measured slide inside it to be identifiable
EVAL_TARGETS = tuple(TARGET_RANGE[0] + (TARGET_RANGE[1] - TARGET_RANGE[0]) * (i + 0.5) / 40 for i in range(40))


@dataclass
class Calibration:
    structure: list[str]
    model: dict
    residuals: dict
    intervals: dict[str, tuple[float, float]]
    ensemble: list[dict]
    candidates: list[dict]
    predicted: dict
    gap: list[dict]
    curves: dict
    base: ParamSet = field(repr=False, default_factory=ParamSet.nominal)
    chosen_by: str = "offline structure search"

    @property
    def params(self) -> ParamSet:
        return to_params(self.model, self.base)

    def to_json(self) -> dict:
        return {"structure": self.structure, "model": self.model, "residuals": self.residuals,
                "intervals": {k: list(v) for k, v in self.intervals.items()}, "candidates": self.candidates,
                "predicted": self.predicted, "gap": self.gap, "curves": self.curves, "chosen_by": self.chosen_by,
                "ensemble_size": len(self.ensemble)}


def fit_base(session: Session) -> ParamSet:
    """Hand or phone pushes have no command: the measured launch speed stands in for it, so gain is 1."""
    return session.sim if session.has_commands else session.sim.with_(actuator_gain=1.0)


def library(session: Session) -> list[list[str]]:
    """Identifiable structures for this data, simplest first."""
    core = ["mu_eff"]
    if session.has_commands and session.has_tracks:
        core.append("actuator_gain")  # stops alone only pin gain^2 / mu; launch motion separates them
    if session.has_targets:
        core += ["camera_dx", "camera_pitch_deg"]
    extras = [["patch_y0", "patch_mu"]] if len([p for p in session.pushes if not p.tipped]) >= 6 else []
    if session.has_targets:
        extras.append(["lens_k"])
    out = []
    for r in range(len(extras) + 1):
        for combo in combinations(extras, r):
            out.append(core + [f for group in combo for f in group])
    return out


def _wb(ro: Rollout, base: ParamSet) -> Workbench:
    return Workbench(ro, base, None, 0)


def _fit(wb: Workbench, start: dict, free: list[str]) -> dict:
    return wb.fit_hypothesis(dict(start), list(free))


def search(session: Session) -> list[dict]:
    base = fit_base(session)
    wb, start = _wb(session.rollout(), base), from_params(base)
    out = []
    for free in library(session):
        f = _fit(wb, start, free)
        out.append({"free": free, "model": f["fitted_model"], "stop_rms_m": f["stop_residual_rms_m"],
                    "perception_rms_m": f["perception_residual_rms_m"], "launch_rms_mps": f["launch_speed_residual_rms_mps"],
                    "n_params": len(free)})
    best_s = min(c["stop_rms_m"] for c in out)
    best_p = min(c["perception_rms_m"] for c in out)
    ok = [c for c in out if c["stop_rms_m"] <= 1.25 * best_s + 0.002 and c["perception_rms_m"] <= 1.25 * best_p + 0.001]
    pick = min(ok, key=lambda c: (c["n_params"], c["stop_rms_m"]))
    for c in out:
        c["chosen"] = c is pick
    return out


def bootstrap(session: Session, model: dict, free: list[str], n: int = N_BOOT, seed: int = 0) -> list[dict]:
    base = fit_base(session)
    pushes = [p for p in session.pushes if not p.tipped]
    rng = random.Random(seed)
    ens = []
    for _ in range(n):
        sample = [pushes[rng.randrange(len(pushes))] for _ in pushes]
        ro = Rollout([p.to_trial() for p in sample])
        ens.append(_fit(_wb(ro, base), model, free)["fitted_model"])
    return ens


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    i = int(k)
    return xs[i] if i + 1 >= len(xs) else xs[i] + (k - i) * (xs[i + 1] - xs[i])


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def expected_success(world: ParamSet, policy, sigma: float, targets=EVAL_TARGETS) -> float:
    """Success of `policy` in `world` with stop noise sigma (the measured residual scatter)."""
    tot = 0.0
    for d in targets:
        e = slide_distance(policy.command(observe(d, world)), world) - d
        tot += _phi((SUCCESS_TOL - e) / sigma) - _phi((-SUCCESS_TOL - e) / sigma)
    return tot / len(targets)


def success_by_target(world: ParamSet, policy, sigma: float, n_bins: int = 8) -> list[float]:
    lo, hi = TARGET_RANGE
    out = []
    for b in range(n_bins):
        ts = [lo + (hi - lo) * (b + (j + 0.5) / 5) / n_bins for j in range(5)]
        out.append(round(expected_success(world, policy, sigma, ts), 3))
    return out


UNMEASURED_SHIFT = 0.3  # what-if: friction beyond the measured stretch may differ by this fraction


def whatif_worlds(session: Session, cal_params: ParamSet) -> list[ParamSet]:
    """Plausible worlds the data cannot rule out: past the farthest measured stop, the table might change
    friction (by +-30 %). Only added when the measurements do not reach the far end of the target range."""
    reach = session.coverage()[1]
    if reach >= TARGET_RANGE[1] - 0.02:
        return []
    mu_far = mu_at(cal_params, reach)
    y0 = max(reach + 0.02, 0.15)
    return [cal_params.with_(patch_y0=min(y0, 0.99), patch_mu=min(1.2, max(0.2, mu_far * (1 + s))))
            for s in (-UNMEASURED_SHIFT, UNMEASURED_SHIFT)]


def predict(session: Session, cal_params: ParamSet, ensemble: list[dict], sigma: float) -> dict:
    """Model-predicted success over the target range of a policy trained in the current sim vs in the
    calibrated sim, evaluated across the plausible real worlds: the bootstrap ensemble plus what-if worlds
    for any stretch of table the data never reached (so extrapolation is not reported as certainty)."""
    base = fit_base(session)
    worlds = [to_params(m, base) for m in ensemble] or [cal_params]
    whatif = whatif_worlds(session, cal_params)
    worlds += whatif * max(1, len(worlds) // 6)
    pol_before = InverseTrainer().train(_fixed(session.sim if session.has_commands else base))
    pol_after = InverseTrainer().train(_fixed(cal_params))
    out = {}
    for name, pol in (("before", pol_before), ("after", pol_after)):
        s = [expected_success(w, pol, sigma) for w in worlds]
        by = [success_by_target(w, pol, sigma) for w in worlds]
        out[name] = {"mean": round(sum(s) / len(s), 3), "lo": round(_pct(s, 0.05), 3), "hi": round(_pct(s, 0.95), 3),
                     "by_target": [round(sum(col) / len(col), 3) for col in zip(*by)]}
    out["noise_sigma_m"] = round(sigma, 4)
    out["target_range_m"] = list(TARGET_RANGE)
    out["tolerance_m"] = SUCCESS_TOL
    out["measured_reach_m"] = round(session.coverage()[1], 3)
    out["unmeasured_whatif"] = bool(whatif)
    return out


def _fixed(p: ParamSet) -> Randomization:
    return Randomization({k: (p[k], p[k]) for k in PARAM_SPACE})


def intervals(ensemble: list[dict], free: list[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for f in free:
        xs = [m[f] for m in ensemble if m.get(f) is not None]
        if xs:
            out[f] = (round(_pct(xs, 0.05), 4), round(_pct(xs, 0.95), 4))
    return out


def gap_table(session: Session, model: dict, iv: dict, free: list[str]) -> list[dict]:
    cur = from_params(fit_base(session))
    rows = []
    for f in FIELDS:
        new, old = model.get(f), cur.get(f)
        if f not in free and new == old:
            continue
        if new is None:
            continue
        lo, hi = iv.get(f, (new, new))
        ref = old if old is not None else (PARAM_SPACE["patch_y0"].nominal if f == "patch_y0" else None)
        changed = ref is None or not (lo <= ref <= hi)
        rel = None if ref in (None, 0) or f.startswith("patch") else (new - ref) / abs(ref)
        rows.append({"field": f, "label": LABELS[f], "unit": UNITS[f], "sim": old, "real": new, "lo": lo, "hi": hi,
                     "change": None if rel is None else round(rel, 3), "significant": bool(changed)})
    return rows


def mu_at(p: ParamSet, y: float) -> float:
    mu1, mu2, y0 = frictions(p)
    return mu1 if mu2 is None or y < y0 else mu2


def curves(session: Session, cal: ParamSet, ensemble: list[dict]) -> dict:
    """Plot data: stop vs launch speed (measured, model, ensemble band) and friction along the push axis."""
    base = fit_base(session)
    gain = cal["actuator_gain"]
    pts = [{"x": round((p.command if p.command is not None else p.launch_speed) * (gain if session.has_commands else 1.0), 4),
            "y": round(p.stop, 4), "tipped": p.tipped, "origin": p.origin} for p in session.pushes]
    xs = [p["x"] for p in pts if not p["tipped"]]
    vmax = max(max(xs, default=1.0) * 1.3, math.sqrt(2 * mu_at(cal, 0) * GRAVITY * 0.75))
    grid = [round(vmax * (i + 1) / 40, 4) for i in range(40)]
    worlds = [to_params(m, base) for m in ensemble]

    def stop_at(p: ParamSet, v: float) -> float:
        return slide_distance(v / p["actuator_gain"], p)

    model_y = [round(stop_at(cal, v), 4) for v in grid]
    band = [(round(_pct([stop_at(w, v) for w in worlds], 0.05), 4), round(_pct([stop_at(w, v) for w in worlds], 0.95), 4))
            if worlds else (y, y) for v, y in zip(grid, model_y)]
    sim_y = [round(stop_at(base, v), 4) for v in grid]
    wb = _wb(session.rollout(), base)
    prof = [b for b in wb.decel_profile() if "decel_g" in b]
    ys = [round(0.01 * i, 2) for i in range(0, 91)]
    span = [tuple(float(x) for x in re.match(r"(-?[\d.]+)-(-?[\d.]+)", b["y_m"]).groups()) for b in prof]
    fric = {"measured": [{"y0": a, "y1": c, "mu": b["decel_g"], "n": b["n"]} for (a, c), b in zip(span, prof)],
            "model": [round(mu_at(cal, y), 4) for y in ys],
            "model_lo": [round(_pct([mu_at(w, y) for w in worlds], 0.05), 4) if worlds else None for y in ys],
            "model_hi": [round(_pct([mu_at(w, y) for w in worlds], 0.95), 4) if worlds else None for y in ys],
            "sim": [round(mu_at(base, y), 4) for y in ys], "y": ys}
    perc = [{"true": round(p.target, 4), "perceived": round(p.perceived, 4)} for p in session.pushes
            if p.target is not None and p.perceived is not None]
    return {"stops": {"points": pts, "grid": grid, "model": model_y, "band": band, "sim": sim_y,
                      "x_label": "launch speed (m/s)"},
            "friction": fric, "perception": perc}


def calibrate(session: Session, structure: list[str] | None = None, start: dict | None = None,
              chosen_by: str | None = None, n_boot: int = N_BOOT) -> Calibration:
    """Fit, quantify, predict. `structure` (free fields) from the agent skips the offline search."""
    base = fit_base(session)
    wb = _wb(session.rollout(), base)
    if structure is None:
        cands = search(session)
        pick = next(c for c in cands if c["chosen"])
        structure, model = pick["free"], pick["model"]
        chosen_by = chosen_by or "offline structure search (simplest model within noise of the best)"
    else:
        cands = []
        model = _fit(wb, start or from_params(base), structure)["fitted_model"]
    reach = session.coverage()[1]
    if model.get("patch_y0") is not None and model["patch_y0"] > reach - IDENT_MARGIN:
        # a friction region that starts where no push reached is fitted to noise: drop it, the what-if worlds
        # and the next-experiment cards cover that stretch of table instead
        structure = [f for f in structure if f not in ("patch_y0", "patch_mu")]
        model = _fit(wb, {**model, "patch_y0": None, "patch_mu": None}, structure)["fitted_model"]
        chosen_by = (chosen_by or "Nemotron agent") + f"; friction region past {reach:.2f} m dropped (no push measured it)"
    res = wb.residuals(to_params(model, base))
    ens = bootstrap(session, model, structure, n_boot) if n_boot else []
    iv = intervals(ens, structure)
    cal = to_params(model, base)
    sigma = max(res["stop_residual_rms_m"], 0.003)
    return Calibration(structure, model, res, iv, ens, cands, predict(session, cal, ens, sigma),
                       gap_table(session, model, iv, structure), curves(session, cal, ens), base,
                       chosen_by or "Nemotron agent")
