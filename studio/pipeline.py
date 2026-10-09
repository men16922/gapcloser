"""Studio pipeline: session -> (Nemotron agent picks the structure) -> calibration -> next experiment -> exports.

Without an LLM the offline structure search picks the structure. With one, the Nemotron tool agent works on
the user's recorded data exactly as it does in the benchmark (decel_profile, perception_check, fit/test
hypotheses), except that probe_real cannot touch a robot: requested pushes become "next experiment" cards.
Either way the numbers and their intervals come from the least-squares fitter and the bootstrap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent.tool_agent import FIELDS, ToolAgentDiagnoser, from_params
from sim.push_task import AnalyticPushEnv, Rollout, Trial
from studio import design, export
from studio.fit import Calibration, calibrate, fit_base, search
from studio.session import Session

AGENT_STEPS = 8


def misfit(c: dict) -> float:
    """Residuals in units of their noise thresholds (stop 1 cm, launch 0.08 m/s, perception 3 mm); <= 1 is explained."""
    return max(c["stop_rms_m"] / 0.01, c.get("launch_rms_mps", 0.0) / 0.08, c.get("perception_rms_m", 0.0) / 0.003)


class OfflineRobot:
    """Stands in for RealWorld in a recorded session: probe pushes are queued for the human."""

    offline = True

    def push(self, commands):  # pragma: no cover - Workbench never calls it when offline is set
        raise RuntimeError("offline session")


@dataclass
class Result:
    session: Session
    calibration: Calibration
    next_experiment: dict
    agent: dict | None = None
    seconds: float = 0.0
    events: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"session": self.session.to_json(), "calibration": self.calibration.to_json(),
                "next_experiment": self.next_experiment, "agent": self.agent, "seconds": round(self.seconds, 2),
                "exports": self.exports()}

    def exports(self) -> dict:
        out = {"newton": export.newton_snippet(self.session, self.calibration),
               "isaaclab": export.isaaclab_snippet(self.session, self.calibration),
               "markdown": export.markdown(self.session, self.calibration, self.next_experiment, self.agent)}
        if (self.session.meta or {}).get("domain") == "driving":
            out["carla"] = export.carla_snippet(self.session, self.calibration)
        return out


def agent_context(session: Session) -> dict:
    lo, hi = session.coverage()
    ctx = {"data_source": "recorded user data (offline): probe_real queues pushes for the human instead of running them",
           "real_track_coverage_m": [round(lo, 3), round(hi, 3)], "probe_budget": 6}
    if not session.has_commands:
        ctx["commands_are"] = ("measured launch speeds in m/s (hand or phone-recorded pushes): actuator_gain is 1 by "
                               "construction, do not fit it")
    if not session.has_targets:
        ctx["perception_data"] = "none: the robot was not aiming at targets, keep camera fields at the current values"
        ctx["real_success_rate"] = None
        ctx["sim_predicted_success_rate"] = None
    return ctx


def _sim_rollout(session: Session) -> Rollout:
    base = fit_base(session)
    env = AnalyticPushEnv()
    sim = env.push(base, [p.to_trial().command for p in session.pushes])
    trials = []
    for p, t in zip(session.pushes, sim.trials):
        tr = p.to_trial()
        trials.append(Trial(tr.target, tr.observed, t.command, t.slide, t.track))
    return Rollout(trials)


def run_agent(session: Session, llm, on_step=None) -> dict:
    base = fit_base(session)
    agent = ToolAgentDiagnoser(llm, OfflineRobot(), max_steps=AGENT_STEPS, probe_budget=6, on_step=on_step,
                               context=agent_context(session))
    diag = agent.diagnose(session.rollout(), _sim_rollout(session), base)
    hist = agent.history[-1] if agent.history else {}
    wb = agent.last_workbench
    commit = next((s for s in reversed(diag.trace) if s.get("tool") == "commit"), None)
    fits = [s for s in diag.trace if s.get("tool") == "fit_hypothesis"]
    out = {"model_name": diag.model, "error": hist.get("error") or "", "steps": len(diag.trace), "trace": diag.trace,
           "requested_commands": list(wb.requested) if wb else [], "explanation": diag.reasoning, "committed": None,
           "structure": None}
    if commit and not out["error"]:
        model = {k: v for k, v in (commit["args"].get("model") or {}).items() if k in FIELDS}
        out["committed"] = model
        # the structure is the free set of the agent's last fit whose fitted model matches the commit (else the
        # fields that differ from the current sim)
        cur = from_params(base)
        free = None
        for s in reversed(fits):
            if isinstance(s.get("result"), dict) and s["result"].get("fitted_model"):
                fm = s["result"]["fitted_model"]
                if all(abs((fm.get(k) or 0) - (model.get(k) or 0)) < 0.02 for k in ("mu_eff", "patch_mu", "patch_y0")):
                    free = [f for f in s["args"].get("free", []) if f in FIELDS]
                    break
        if not free:
            free = [k for k in FIELDS if model.get(k) is not None and model.get(k) != cur.get(k)] or ["mu_eff"]
        if not session.has_commands:
            free = [f for f in free if f != "actuator_gain"]
        if not session.has_targets:
            free = [f for f in free if f not in ("camera_dx", "camera_pitch_deg", "lens_k")]
        if any(f in free for f in ("patch_y0", "patch_mu")):
            free = list(dict.fromkeys(free + ["patch_y0", "patch_mu"]))
        out["structure"] = free or ["mu_eff"]
    return out


def analyze(session: Session, llm=None, on_event=None, n_boot: int | None = None) -> Result:
    """Full pipeline. on_event(dict) receives: stage, agent_step*, calibration, next, done."""
    t0 = time.perf_counter()
    events: list[dict] = []

    def emit(e: dict) -> None:
        events.append(e)
        if on_event:
            on_event(e)

    emit({"type": "stage", "stage": "data", "pushes": len(session.pushes), "coverage_m": [round(x, 3) for x in session.coverage()],
          "notes": session.notes})
    agent = None
    if llm is not None:
        emit({"type": "stage", "stage": "agent"})
        agent = run_agent(session, llm, on_step=lambda s: emit({"type": "agent_step", "step": s}))
    emit({"type": "stage", "stage": "fit"})
    kw = {} if n_boot is None else {"n_boot": n_boot}
    if agent and agent["structure"]:
        cal = calibrate(session, agent["structure"], {**from_params(fit_base(session)), **agent["committed"]},
                        chosen_by=f"Nemotron agent ({agent['model_name'] or 'llm'})", **kw)
        # cross-check: never ship a model that leaves evidence unexplained when a library structure explains it
        cands = search(session)
        alt = min(cands, key=lambda c: (round(misfit(c), 1), c["n_params"]))
        mine = misfit({"stop_rms_m": cal.residuals["stop_residual_rms_m"], "launch_rms_mps": cal.residuals["launch_speed_residual_rms_mps"],
                       "perception_rms_m": cal.residuals["perception_residual_rms_m"]})
        check = {"agent_misfit": round(mine, 2), "search_misfit": round(misfit(alt), 2), "search_structure": alt["free"],
                 "unexplained": cal.residuals.get("unexplained", []), "adopted": "agent"}
        if mine > 1.0 and misfit(alt) < 0.7 * mine:
            cal = calibrate(session, alt["free"], alt["model"], chosen_by="cross-check (agent model overruled)", **kw)
            cal.candidates = cands
            check["adopted"] = "search"
            check["reason"] = ("the agent's model leaves " + "; ".join(check["unexplained"] or ["residuals"]) +
                               f" — the library structure {'+'.join(alt['free'])} explains the data")
        agent["cross_check"] = check
        emit({"type": "cross_check", "cross_check": check})
    else:
        cal = calibrate(session, **kw)
        if agent and agent.get("error"):
            cal.chosen_by += f" — agent fell back: {agent['error'][:80]}"
    emit({"type": "calibration", "calibration": cal.to_json()})
    nxt = design.suggest(session, cal, extra_requests=(agent or {}).get("requested_commands"))
    emit({"type": "next", "next_experiment": nxt})
    res = Result(session, cal, nxt, agent, time.perf_counter() - t0, events)
    emit({"type": "done", "seconds": round(res.seconds, 2)})
    return res
