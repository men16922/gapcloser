"""Next experiment: which pushes would settle what the data cannot.

Query by committee. The committee is the bootstrap ensemble of the chosen model plus the best fit of
every other plausible structure (within 2x of the best residual). A push is worth running where the
committee disagrees about the stop *and* the stop falls where the robot has to work (target range).
Structural disagreement (does a friction region exist out there?) dominates exactly where the data
does not reach, which is the case the agent's probe_real tool exists for.
"""

from __future__ import annotations

import math

from agent.tool_agent import to_params
from sim.params import GRAVITY, ParamSet
from sim.push_task import TARGET_RANGE, slide_distance
from studio.fit import Calibration, fit_base, mu_at, whatif_worlds
from studio.session import Session


def _stop(p: ParamSet, v: float) -> float:
    return slide_distance(v / p["actuator_gain"], p)


def committee(session: Session, cal: Calibration) -> list[ParamSet]:
    base = fit_base(session)
    members = [to_params(m, base) for m in cal.ensemble] or [cal.params]
    best = min((c["stop_rms_m"] for c in cal.candidates), default=None)
    for c in cal.candidates:
        if not c.get("chosen") and best is not None and c["stop_rms_m"] <= 2.0 * best + 0.01:
            members += [to_params(c["model"], base)] * max(1, len(cal.ensemble) // 4)
    members += whatif_worlds(session, cal.params) * max(1, len(cal.ensemble) // 6)
    return members


def suggest(session: Session, cal: Calibration, k: int = 3, extra_requests: list[float] | None = None) -> dict:
    """Ranked pushes. `extra_requests` are launch commands the agent asked for (probe_real while offline)."""
    members = committee(session, cal)
    p = cal.params
    lo_t, hi_t = TARGET_RANGE
    v_hi = math.sqrt(2 * max(mu_at(p, 0), 0.2) * GRAVITY * (hi_t + 0.12))
    grid = [v_hi * (i + 1) / 60 for i in range(60)]
    cov_lo, cov_hi = session.coverage()
    rows = []
    for v in grid:
        stops = [_stop(m, v) for m in members]
        mean = sum(stops) / len(stops)
        sd = math.sqrt(sum((s - mean) ** 2 for s in stops) / len(stops))
        rel = 1.0 if lo_t - 0.03 <= mean <= hi_t + 0.05 else 0.25
        novelty = 1.0 + (0.5 if mean > cov_hi + 0.02 else 0.0)
        rows.append({"launch_speed_mps": round(v, 3), "predicted_stop_m": round(mean, 3), "disagreement_m": round(sd, 4),
                     "score": sd * rel * novelty, "beyond_data": mean > cov_hi + 0.02})
    picked = []
    for r in sorted((r for r in rows if r["predicted_stop_m"] <= hi_t + 0.1), key=lambda r: -r["score"]):
        if all(abs(r["predicted_stop_m"] - q["predicted_stop_m"]) >= 0.08 for q in picked):
            picked.append(r)
        if len(picked) == k:
            break
    gain = p["actuator_gain"] if session.has_commands else 1.0
    for r in picked:
        r["command"] = round(r["launch_speed_mps"] / gain, 3) if session.has_commands else None
        why = f"models disagree by ±{r['disagreement_m'] * 100:.1f} cm here"
        if r["beyond_data"]:
            why += f"; your pushes stop by {cov_hi:.2f} m, so the table beyond is unmeasured"
        r["why"] = why
    max_sd = max((r["disagreement_m"] for r in rows if lo_t <= r["predicted_stop_m"] <= hi_t), default=0.0)
    settled = max_sd < 0.006 and cov_hi >= hi_t - 0.02
    agent = [{"command": round(c, 3), "predicted_stop_m": round(_stop(p, c * gain), 3)} for c in (extra_requests or [])]
    for r in agent:  # the agent may ask for pushes that leave the table; keep them visible but marked
        r["useful"] = r["predicted_stop_m"] <= hi_t + 0.15
    return {"settled": settled, "max_disagreement_in_range_m": round(max_sd, 4), "coverage_m": [round(cov_lo, 3), round(cov_hi, 3)],
            "suggestions": [] if settled else picked, "agent_requests": agent, "committee_size": len(members),
            "curve": [{"v": r["launch_speed_mps"], "stop": r["predicted_stop_m"], "sd": r["disagreement_m"]} for r in rows]}
