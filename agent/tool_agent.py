"""Nemotron as a tool-using scientist: inspect evidence, hypothesize a simulator model, fit it, run
probe experiments on the real robot when the data cannot decide, then commit.

Division of labor (found in spike/tool_agent_spike.py): the LLM chooses the *structure* of the model
(which effects exist: global friction, actuator gain, a table region with different friction, camera
offset/pitch, lens distortion) and which experiments to run; `fit_hypothesis` fits the numbers by
least squares against every real measurement. Fitting uses the analytic surrogate (matches NVIDIA Newton
within ~1 cm); real data comes from whatever env RealWorld wraps (Newton in the benchmark).

Satisfies the Diagnoser protocol; falls back to TrajectoryDiagnoser on any LLM failure.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from agent.llm import LLM
from agent.loop import Diagnosis, RealWorld, Suspect, TrajectoryDiagnoser
from sim.params import GRAVITY, PARAM_SPACE, ParamSet, effective_friction
from sim.push_task import FRAME_DT, PATCH_OFF, SUCCESS_TOL, TARGET_RANGE, Rollout, analytic_track, observe, slide_distance

FIELDS = ("mu_eff", "actuator_gain", "patch_y0", "patch_mu", "camera_dx", "camera_pitch_deg", "lens_k")

MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "mu_eff": {"type": "number", "description": "effective cube-table friction (deceleration = mu_eff * g)"},
        "actuator_gain": {"type": "number", "description": "launch speed = actuator_gain * command"},
        "patch_y0": {"type": ["number", "null"], "description": "start (m) of a table region with different friction; null = none"},
        "patch_mu": {"type": ["number", "null"], "description": "effective friction beyond patch_y0; null = none"},
        "camera_dx": {"type": "number", "description": "perception offset (m): perceived = true*(1+tan(pitch)) + dx + lens_k*true^2"},
        "camera_pitch_deg": {"type": "number", "description": "camera pitch error (deg)"},
        "lens_k": {"type": "number", "description": "radial lens distortion (1/m), 0 = none"},
    },
    "required": list(FIELDS),
}


def _tool(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties or {},
                                                            "required": required or []}}}


TOOLS = [
    _tool("decel_profile", "Deceleration of the real cube (in g) measured from camera tracks, binned by cube position along the push axis."),
    _tool("perception_check", "Perceived vs known true target positions of the real trials (camera calibration check)."),
    _tool("fit_hypothesis", "Free (sim only): least-squares fit of the chosen free fields of a model structure to all real data "
          "(stop positions, launch motion, perception). Returns the fitted model and residuals. Prefer this over hand-tuning.",
          {"model": MODEL_SCHEMA, "free": {"type": "array", "items": {"type": "string", "enum": list(FIELDS)}}}, ["model", "free"]),
    _tool("test_hypothesis", "Free (sim only): replay every real command in a candidate model; returns stop-distance and perception residuals.",
          {"model": MODEL_SCHEMA}, ["model"]),
    _tool("probe_real", "Costs real-robot trials (budget limited): push the real cube with chosen commands (0.5-4.5) and return stops "
          "and launch motion. Use only when existing data cannot separate hypotheses or does not cover the target range.",
          {"commands": {"type": "array", "items": {"type": "number"}, "maxItems": 6}}, ["commands"]),
    _tool("commit", "Final answer: the simulator model to retrain the policy on.",
          {"model": MODEL_SCHEMA, "explanation": {"type": "string", "description": "under 60 words"}}, ["model", "explanation"]),
]

SYSTEM = """You are GapCloser, an agent that fixes a robot simulator so a policy trained in it works on the real robot.
Task: a robot pushes a cube (launch speed = actuator_gain * command); it slides to a stop; success = stop within 3 cm of a target line.
The policy perceives the target through a camera and inverts the simulator to pick each command, so a correct simulator
(dynamics AND perception) means success. The current simulator is wrong. Work like a scientist:
1. inspect evidence with decel_profile and perception_check;
2. form hypotheses about the model STRUCTURE (global friction, actuator gain, a table region with different friction,
   camera offset/pitch, lens distortion) and fit each candidate with fit_hypothesis (free); prefer the simplest structure
   whose residuals are small (stops within ~1 cm);
3. targets span 0.2-0.6 m: if the real tracks do not cover that whole range, the model is unverified there, so probe_real
   with commands that reach it (real trials are expensive) and refit;
4. commit. Keep tool arguments numeric and brief; do not explain at length between calls."""


@dataclass
class Datum:
    command: float
    stop: float
    track: list[float]
    target: float | None = None
    observed: float | None = None


def to_params(model: dict, base: ParamSet) -> ParamSet:
    """Agent model -> simulator ParamSet (object_mu = table_mu = mu_eff; no patch = patch_y0 at nominal)."""
    def clamp(name, v):
        p = PARAM_SPACE[name]
        return min(p.high, max(p.low, float(v)))

    mu = clamp("object_mu", model.get("mu_eff") if model.get("mu_eff") is not None else effective_friction(base["object_mu"], base["table_mu"]))
    ch = {"object_mu": mu, "table_mu": mu}
    for f in ("actuator_gain", "camera_dx", "camera_pitch_deg", "lens_k"):
        if model.get(f) is not None:
            ch[f] = clamp(f, model[f])
    y0, pm = model.get("patch_y0"), model.get("patch_mu")
    if y0 is not None and pm is not None and y0 < PATCH_OFF:
        ch["patch_y0"], ch["patch_mu"] = clamp("patch_y0", y0), clamp("patch_mu", pm)
    else:
        ch["patch_y0"], ch["patch_mu"] = PARAM_SPACE["patch_y0"].nominal, PARAM_SPACE["patch_mu"].nominal
    return base.with_(**ch)


def from_params(p: ParamSet) -> dict:
    has_patch = p["patch_y0"] < PATCH_OFF
    return {"mu_eff": round(effective_friction(p["object_mu"], p["table_mu"]), 4), "actuator_gain": round(p["actuator_gain"], 4),
            "patch_y0": round(p["patch_y0"], 4) if has_patch else None, "patch_mu": round(p["patch_mu"], 4) if has_patch else None,
            "camera_dx": round(p["camera_dx"], 4), "camera_pitch_deg": round(p["camera_pitch_deg"], 3), "lens_k": round(p["lens_k"], 4)}


class Workbench:
    """The tools' state: real measurements so far and the probe budget. Pure functions of data + surrogate."""

    def __init__(self, real_ro: Rollout, base: ParamSet, real: RealWorld | None, probe_budget: int):
        self.base, self.real, self.budget = base, real, probe_budget
        self.data = [Datum(t.command, t.slide, t.track, t.target, t.observed) for t in real_ro.trials if not t.tipped]
        self.tipped = sum(t.tipped for t in real_ro.trials)
        self.probes_used = 0

    def residuals(self, p: ParamSet) -> dict:
        stop = [slide_distance(d.command, p) - d.stop for d in self.data]
        launch = [(analytic_track(d.command, p, 1)[1] - d.track[1]) / FRAME_DT for d in self.data if len(d.track) > 1]
        perc = [observe(d.target, p) - d.observed for d in self.data if d.target is not None and d.target == d.target]
        rms = lambda xs: round(math.sqrt(sum(x * x for x in xs) / len(xs)), 4) if xs else 0.0  # noqa: E731
        worst = sorted(((round(d.command, 2), round(e, 3)) for d, e in zip(self.data, stop)), key=lambda x: -abs(x[1]))[:3]
        return {"stop_residual_rms_m": rms(stop), "worst_stop_residuals_cmd_m": worst,
                "launch_speed_residual_rms_mps": rms(launch), "perception_residual_rms_m": rms(perc)}

    def loss(self, p: ParamSet) -> float:
        e = sum((slide_distance(d.command, p) - d.stop) ** 2 for d in self.data)
        e += sum(((analytic_track(d.command, p, 1)[1] - d.track[1]) / FRAME_DT * 0.05) ** 2 for d in self.data if len(d.track) > 1)
        e += sum((observe(d.target, p) - d.observed) ** 2 for d in self.data if d.target is not None and d.target == d.target)
        return e

    # --- tools -------------------------------------------------------------------------------
    def decel_profile(self) -> list[dict]:
        bins: dict[float, list[float]] = {}
        for d in self.data:
            tr = d.track
            for k in range(1, len(tr) - 1):
                v1, v2 = (tr[k] - tr[k - 1]) / FRAME_DT, (tr[k + 1] - tr[k]) / FRAME_DT
                if v2 > 0.05:
                    bins.setdefault(round(int(tr[k] / 0.05) * 0.05, 2), []).append((v1 - v2) / FRAME_DT / GRAVITY)
        out = [{"y_m": f"{b:.2f}-{b + 0.05:.2f}", "decel_g": round(sum(v) / len(v), 3), "n": len(v)} for b, v in sorted(bins.items())]
        return out or [{"note": "no usable tracks"}]

    def perception_check(self) -> list[dict]:
        rows = [d for d in self.data if d.target is not None and d.target == d.target]
        return [{"true_m": round(d.target, 3), "perceived_m": round(d.observed, 3)} for d in rows[::2]]

    def test_hypothesis(self, model: dict) -> dict:
        return self.residuals(to_params(model, self.base))

    def fit_hypothesis(self, model: dict, free: list[str]) -> dict:
        from scipy.optimize import minimize

        free = [f for f in dict.fromkeys(free or []) if f in FIELDS]
        start = {**from_params(self.base), **{k: v for k, v in model.items() if k in FIELDS}}
        if ("patch_y0" in free or "patch_mu" in free) and (start.get("patch_y0") is None or start.get("patch_mu") is None):
            start["patch_y0"] = start.get("patch_y0") or 0.35
            start["patch_mu"] = start.get("patch_mu") or start["mu_eff"]
            free = list(dict.fromkeys(free + ["patch_y0", "patch_mu"]))
        if not free:
            return {"fitted_model": from_params(to_params(start, self.base)), **self.test_hypothesis(start)}

        def build(x):
            return to_params({**start, **dict(zip(free, x))}, self.base)

        x0 = [float(start[f]) for f in free]
        starts = [x0]
        if "patch_y0" in free:  # loss is non-convex in where the region begins
            i = free.index("patch_y0")
            starts = [x0[:i] + [y] + x0[i + 1:] for y in (0.2, 0.3, 0.4, 0.5)]
        best = min((minimize(lambda x: self.loss(build(x)), st, method="Nelder-Mead",
                             options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 3000}) for st in starts), key=lambda r: r.fun)
        fitted = from_params(build(best.x))
        return {"fitted_model": fitted, **self.test_hypothesis(fitted)}

    def probe_real(self, commands: list[float]) -> dict:
        cmds = [min(4.5, max(0.3, float(c))) for c in (commands or [])][: max(0, self.budget - self.probes_used)]
        if not cmds or self.real is None:
            return {"error": "probe budget exhausted", "probe_budget_left": self.budget - self.probes_used}
        self.probes_used += len(cmds)
        ro = self.real.push(cmds)
        out = []
        for t in ro.trials:
            if not t.tipped:
                self.data.append(Datum(t.command, t.slide, t.track))
            out.append({"command": round(t.command, 3), "stop_m": round(t.slide, 3), "tipped": t.tipped,
                        "first_frame_speed_mps": round((t.track[1] - t.track[0]) / FRAME_DT, 3) if len(t.track) > 1 else None})
        return {"results": out, "probe_budget_left": self.budget - self.probes_used}

    def call(self, name: str, args: dict):
        if name == "decel_profile":
            return self.decel_profile()
        if name == "perception_check":
            return self.perception_check()
        if name == "fit_hypothesis":
            return self.fit_hypothesis(args.get("model") or {}, args.get("free") or [])
        if name == "test_hypothesis":
            return self.test_hypothesis(args.get("model") or {})
        if name == "probe_real":
            return self.probe_real(args.get("commands") or [])
        return {"error": f"unknown tool {name}"}


class ToolAgentDiagnoser:
    """`real` is needed for probe experiments; without it probe_real reports an exhausted budget."""

    def __init__(self, llm: LLM, real: RealWorld | None = None, role: str = "diagnose", max_steps: int = 10,
                 probe_budget: int = 12, fallback=None, on_step=None):
        self.llm, self.real, self.role = llm, real, role
        self.on_step = on_step  # called with each trace entry as it happens (the live server streams these)
        self.max_steps, self.probe_budget = max_steps, probe_budget
        self.fallback = fallback or TrajectoryDiagnoser()
        self.history: list[dict] = []

    def diagnose(self, real: Rollout, sim: Rollout, sim_params: ParamSet) -> Diagnosis:
        wb = Workbench(real, sim_params, self.real, self.probe_budget)
        stops = [t.slide for t in real.trials if not t.tipped] or [0.0]
        summary = {
            "target_range_m": list(TARGET_RANGE), "success_tolerance_m": SUCCESS_TOL,
            "real_success_rate": round(real.success_rate, 3), "sim_predicted_success_rate": round(sim.success_rate, 3),
            "real_cubes_tipped_over": wb.tipped, "real_track_coverage_m": [0.0, round(max(stops), 3)],
            "sim_current_model": from_params(sim_params), "probe_budget": self.probe_budget,
            "trials": [{"target_m": round(r.target, 3), "perceived_m": round(r.observed, 3), "command": round(r.command, 3),
                        "real_stop_m": round(r.slide, 3), "sim_predicted_stop_m": round(s.slide, 3)}
                       for r, s in list(zip(real.trials, sim.trials))[::2]],
        }
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Iteration evidence:\n" + json.dumps(summary)}]
        trace: list[dict] = []

        def log(step: dict) -> None:
            trace.append(step)
            if self.on_step is not None:
                self.on_step(step)

        committed, model, err = None, "", ""
        try:
            for step in range(self.max_steps):
                last = step == self.max_steps - 1
                if last:
                    messages.append({"role": "user", "content": "Step budget exhausted: call commit now with your best model."})
                turn = self.llm.chat(self.role, messages, TOOLS, "commit" if last else None)
                model = turn.model
                messages.append(turn.message())
                if not turn.tool_calls:
                    log({"tool": "text", "text": turn.content[:300]})
                    messages.append({"role": "user", "content": "Use the tools; finish with commit."})
                    continue
                for c in turn.tool_calls:
                    try:
                        args = json.loads(c["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if c["name"] == "commit":
                        committed = args
                        log({"tool": "commit", "args": args})
                        break
                    out = wb.call(c["name"], args)
                    log({"tool": c["name"], "args": args, "result": out})
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(out)})
                if committed:
                    break
            if not committed or not isinstance(committed.get("model"), dict):
                raise ValueError("agent did not commit a model")
        except Exception as e:  # noqa: BLE001 - any LLM failure falls back, the loop must not stall
            err = f"{type(e).__name__}: {e}"
        base = self.fallback.diagnose(real, sim, sim_params)
        self.history.append({"model": model, "error": err, "steps": len(trace), "probes": wb.probes_used})
        if err:
            base.summary = f"[fallback: {err[:80]}] " + base.summary
            base.model = f"{model or 'llm'} → fallback {type(self.fallback).__name__}"
            base.trace = trace
            return base
        new = to_params(committed["model"], sim_params)
        suspects = []
        for name, (cur, est) in sim_params.diff(new, rel_tol=1e-3).items():
            suspects.append(Suspect(name, "up" if est > cur else "down", 0.8, est))
        res = wb.residuals(new)
        summ = (f"tool agent: {len(trace)} steps, {wb.probes_used} probe pushes; committed model stop residual "
                f"{res['stop_residual_rms_m'] * 100:.1f} cm rms")
        return Diagnosis(summ, base.scale_ratio, base.offset, suspects, reasoning=str(committed.get("explanation", ""))[:400],
                         model=model, trace=trace)
