"""Nemotron-backed diagnoser: reads the measured evidence and names which sim parameters are off.

The LLM gets ratios measured from rollouts (end positions, tracked launch speed and deceleration,
perceived vs known target positions), the current sim values and the physics model. It returns
JSON (reasoning + suspects with estimates). Output is validated against PARAM_SPACE; anything
malformed or out of range falls back to TrajectoryDiagnoser, so the loop never stalls on the LLM.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Callable

from agent.llm import LLM
from agent.loop import Diagnosis, Suspect, TrajectoryDiagnoser, _fit_line, _launch_stats
from sim.params import PARAM_SPACE, ParamSet, effective_friction
from sim.push_task import SUCCESS_TOL, Rollout

SYSTEM = """You are the diagnosis module of Tether, an agent that closes the sim-to-real gap of a robot push task.
A cube is pushed with launch speed v = command * actuator_gain and slides to rest on a table.
Physics: deceleration a = mu_eff * g with mu_eff = (object_mu + table_mu) / 2, stop distance = v^2 / (2 a).
The policy computes its command from the target distance it perceives through the camera:
perceived = true_target * (1 + tan(camera_pitch_deg)) + camera_dx.
Real and sim run the identical commands. You see ratios real/sim of measured quantities.
Rules: name only parameters the evidence supports; give a numeric estimate of the REAL value when the
evidence determines it (e.g. real actuator_gain = sim actuator_gain * launch_speed_ratio; real mu_eff =
sim mu_eff * deceleration_ratio and object_mu/table_mu cannot be separated, so give both the same
estimate); parameters that cannot change these measurements (density, size, restitution, light,
camera_dz) must not be blamed. Keep "reasoning" under 60 words. Answer with JSON only, no other text."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "suspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": sorted(PARAM_SPACE)},
                    "direction": {"type": "string", "enum": ["up", "down", "unknown"]},
                    "confidence": {"type": "number"},
                    "estimate": {"type": ["number", "null"]},
                },
                "required": ["name", "direction", "confidence", "estimate"],
            },
        },
    },
    "required": ["reasoning", "suspects"],
}


def build_evidence(real: Rollout, sim: Rollout, sim_params: ParamSet) -> dict:
    targets = [t.target for t in real.trials]
    a_r, b_r = _fit_line(targets, [t.slide for t in real.trials])
    a_s, b_s = _fit_line(targets, [t.slide for t in sim.trials])
    k, dx = _fit_line(targets, [t.observed for t in real.trials])
    ev = {
        "outcome": {
            "real_success_rate": round(real.success_rate, 3),
            "real_trials_cube_tipped_over": sum(t.tipped for t in real.trials),
            "sim_predicted_success_rate": round(sim.success_rate, 3),
            "success_tolerance_m": SUCCESS_TOL,
            "stop_distance_scale_real_over_sim": round(a_r / a_s if a_s else 1.0, 4),
            "stop_distance_offset_m": round(b_r - b_s * (a_r / a_s if a_s else 1.0), 4),
        },
        "perception": {
            "fit_perceived_vs_true_target": {"slope": round(k, 4), "offset_m": round(dx, 4)},
        },
        "sim_current": {n: round(v, 4) for n, v in sim_params.values.items()},
        "sim_mu_eff": round(effective_friction(sim_params["object_mu"], sim_params["table_mu"]), 4),
        "bounds": {n: [p.low, p.high] for n, p in PARAM_SPACE.items()},
        "sample_trials": [
            {"target_m": round(r.target, 3), "command": round(r.command, 3),
             "real_stop_m": round(r.slide, 3), "sim_stop_m": round(s.slide, 3)}
            for r, s in list(zip(real.trials, sim.trials))[:6]
        ],
    }
    rf, sf = _launch_stats(real), _launch_stats(sim)
    if rf and sf:
        ev["tracking"] = {
            "launch_speed_ratio_real_over_sim": round(rf[0] / sf[0], 4),
            "deceleration_ratio_real_over_sim": round(rf[1] / sf[1], 4),
        }
    return ev


def parse_response(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def validate_suspects(raw: list, sim_params: ParamSet | None = None, rel_tol: float = 0.02) -> list[Suspect]:
    """Keep known params, clamp to bounds, one entry per name (highest confidence), and drop
    estimates that equal the current sim value (they imply no change)."""
    best: dict[str, Suspect] = {}
    for s in raw or []:
        if not isinstance(s, dict):
            continue
        name = s.get("name")
        if name not in PARAM_SPACE:
            continue
        p = PARAM_SPACE[name]
        est = s.get("estimate")
        if isinstance(est, (int, float)):
            est = min(p.high, max(p.low, float(est)))
        else:
            est = None
        conf = s.get("confidence", 0.5)
        conf = min(1.0, max(0.0, float(conf))) if isinstance(conf, (int, float)) else 0.5
        d = s.get("direction") if s.get("direction") in ("up", "down", "unknown") else "unknown"
        if est is None and conf < 0.5:  # vague, low-confidence mentions are noise, not a diagnosis
            continue
        if est is not None and sim_params is not None:
            cur = sim_params[name]
            if abs(est - cur) <= rel_tol * max(abs(cur), (p.high - p.low) * 0.1):
                continue
            d = "up" if est > cur else "down"
        if name not in best or conf > best[name].confidence:
            best[name] = Suspect(name, d, conf, est)
    return sorted(best.values(), key=lambda x: -x.confidence)


def image_part(path: Path) -> dict:
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[path.suffix.lower()]
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()}}


class LLMDiagnoser:
    """frames: optional callable returning still images of the current real attempt (e.g. first/mid/last
    Newton frames). When it returns images, the request goes to the multimodal `vision` role."""

    def __init__(self, llm: LLM, fallback=None, role: str = "diagnose",
                 frames: Callable[[], list[Path]] | None = None, vision_role: str = "vision", retries: int = 1):
        self.llm, self.role, self.vision_role = llm, role, vision_role
        self.retries = retries
        self.fallback = fallback or TrajectoryDiagnoser()
        self.frames = frames
        self.history: list[dict] = []

    def diagnose(self, real: Rollout, sim: Rollout, sim_params: ParamSet) -> Diagnosis:
        evidence = build_evidence(real, sim, sim_params)
        text = ("Evidence from this iteration:\n" + json.dumps(evidence, indent=1)
                + "\n\nReturn JSON: {\"reasoning\": short explanation, \"suspects\": [{name, direction, confidence 0-1, estimate or null}]}")
        images = list(self.frames()) if self.frames else []
        role = self.vision_role if images else self.role
        content = ([{"type": "text", "text": text + "\nThe images are camera frames of one real push (start, middle, end)."}]
                   + [image_part(Path(p)) for p in images]) if images else text
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        model, err, attempts = "", "", 0
        for attempts in range(1, self.retries + 2):
            try:
                r = self.llm.complete(role, messages, SCHEMA)
                model = r.model
                data = parse_response(r.text)
                suspects = validate_suspects(data.get("suspects"), sim_params)
                reasoning = str(data.get("reasoning", "")).strip()
                if not suspects and real.success_rate < 0.9:
                    raise ValueError("no valid suspects while the gap is still open")
                err = ""
                break
            except Exception as e:  # noqa: BLE001 - any LLM failure falls back, the loop must not stall
                err = f"{type(e).__name__}: {e}"
        base = self.fallback.diagnose(real, sim, sim_params)
        self.history.append({"evidence": evidence, "model": model, "error": err, "attempts": attempts})
        if err:
            base.summary = f"[fallback: {err[:80]}] " + base.summary
            base.model = f"{model or 'llm'} → fallback {type(self.fallback).__name__}"
            return base
        return Diagnosis(base.summary, base.scale_ratio, base.offset, suspects, reasoning=reasoning, model=model)
