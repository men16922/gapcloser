"""Ask GapCloser: questions about the Studio and about the visitor's current result, answered by Nemotron.

The model gets a short product guide and a compact summary of the session (data, calibration with intervals,
predicted success, next experiment, the agent's steps and the cross-check, the revealed truth if any). It is told
to use only those numbers, to say so when the answer is not in them, and to reply in the visitor's language.
"""

from __future__ import annotations

import json

MAX_TURNS = 12  # conversation turns sent to the model (older ones are dropped)
MAX_CHARS = 1200  # per user message

GUIDE = """GapCloser Studio fits a robot simulator to the real world.
Problem: a policy trained in simulation fails on the real robot because the simulator's physics differ from reality
(a slicker or stickier table, a region of different friction such as a spill or a mat, a weaker motor, a tilted camera).
Flow: 1 Data (a phone video of an object flicked across a table with an A4 sheet for scale, a robot push log, or a table
simulated in NVIDIA Newton from settings the visitor chooses) -> 2 Measure (the sheet gives scale and the camera pose;
each push's launch speed and slide in metres) -> 3 Diagnose (an NVIDIA Nemotron agent picks which effects exist and asks
for experiments; numbers come from least-squares fitting, never from the language model; a cross-check overrules the
agent if its model leaves evidence unexplained) -> 4 Calibrated sim (each value with a 90% interval from a bootstrap
over pushes that also redraws systematic video error; friction painted on the visitor's frame; ghost boxes replay each
push in the old and the calibrated simulator; predicted success of a policy trained in each; a range when part of the
table was never measured) -> 5 Export (NVIDIA Newton snippet, Isaac Lab EventTermCfg whose randomization ranges are the
intervals, Markdown report, JSON).
Terms: mu / friction 0.2 = icy, 1.0 = rubbery; mu_eff = friction of the object-table pair; patch_y0 = where a region of
different friction starts (m from the launch point); patch_mu = friction inside it; actuator_gain = real launch speed per
unit command; camera_pitch_deg = camera tilt error; lens_k = lens distortion; next experiment = pushes where the plausible
models disagree most inside the 0.2-0.6 m target range."""

SYSTEM = """You are the help assistant inside GapCloser Studio. Answer the visitor's question in {lang}.
Use only the guide and the session summary below for facts and numbers; if the answer is not there, say so plainly and
suggest what to do in the Studio. Explain like to a smart newcomer: short sentences, concrete numbers with units, no
jargon without a one-line explanation. Keep answers under 120 words unless asked for detail. Do not invent results,
and do not describe buttons, steps or features that are not in the guide (values cannot be typed in; results are used via Export).

GUIDE:
{guide}

SESSION (JSON, may be empty):
{session}"""

LANGS = {"en": "English", "ko": "Korean (한국어)"}


def _r(x, d=3):
    return None if x is None else round(float(x), d)


def session_summary(state: dict | None, truth: dict | None = None, step: str | None = None) -> dict:
    """Compact, model-friendly view of a Studio session (state = StudioSession.public())."""
    if not state:
        return {"step": step, "note": "no data loaded yet"}
    sess = state.get("session") or {}
    pushes = sess.get("pushes") or []
    out = {"step": step, "name": state.get("name"), "kind": state.get("kind"), "simulated": state.get("simulated"),
           "pushes": len(pushes), "notes": sess.get("notes"),
           "farthest_stop_m": _r(max((p.get("start", 0) + p["stop"] for p in pushes), default=None))}
    res = state.get("result")
    if res:
        cal = res["calibration"]
        out["calibration"] = {
            "chosen_by": cal.get("chosen_by"), "structure": cal.get("structure"),
            "differences": [{"what": g["label"], "current_sim": g["sim"], "measured": g["real"], "interval90": [g["lo"], g["hi"]],
                             "unit": g["unit"], "differs": g["significant"]} for g in cal.get("gap", [])],
            "stop_residual_cm": _r(cal["residuals"]["stop_residual_rms_m"] * 100, 2),
            "unexplained": cal["residuals"].get("unexplained"),
        }
        p = cal["predicted"]
        out["predicted_success"] = {"current_sim": {k: p["before"].get(k) for k in ("median", "lo", "hi")},
                                    "calibrated_sim": {k: p["after"].get(k) for k in ("median", "lo", "hi")},
                                    "by_target_calibrated": p["after"].get("by_target"),
                                    "target_range_m": p.get("target_range_m"), "measured_reach_m": p.get("measured_reach_m"),
                                    "unmeasured_table_whatif": p.get("unmeasured_whatif")}
        n = res.get("next_experiment") or {}
        out["next_experiment"] = {"settled": n.get("settled"), "max_disagreement_cm": _r((n.get("max_disagreement_in_range_m") or 0) * 100, 1),
                                  "suggestions": [{"stop_near_m": s["predicted_stop_m"], "launch_mps": s["launch_speed_mps"],
                                                   "command": s.get("command"), "why": s["why"]} for s in n.get("suggestions", [])]}
        ag = res.get("agent")
        if ag:
            out["agent"] = {"model": ag.get("model_name"), "explanation": ag.get("explanation"),
                            "steps": [s.get("tool") for s in ag.get("trace", [])], "cross_check": ag.get("cross_check"),
                            "requested_pushes": ag.get("requested_commands")}
    if truth:
        out["revealed_truth"] = truth.get("model")
    return out


def build_messages(history: list[dict], lang: str, summary: dict) -> list[dict]:
    lang_name = LANGS.get(lang, LANGS["en"])
    sys = SYSTEM.format(lang=lang_name, guide=GUIDE, session=json.dumps(summary, ensure_ascii=False, default=str)[:12000])
    msgs = [{"role": "system", "content": sys}]
    for m in history[-MAX_TURNS:]:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            msgs.append({"role": m["role"], "content": m["content"][:MAX_CHARS]})
    return msgs


def answer(llm, history: list[dict], lang: str, summary: dict) -> str:
    r = llm.complete("diagnose", build_messages(history, lang, summary))
    text = (r.text or "").strip()
    if "</think>" in text:  # some reasoning models inline their thinking
        text = text.split("</think>", 1)[1].strip()
    return text or ("I could not produce an answer this time. Please ask again." if lang != "ko"
                    else "이번에는 답을 만들지 못했습니다. 다시 질문해 주세요.")
