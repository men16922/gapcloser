"""Ask Tether: questions about the Studio and about the visitor's current result, answered by Nemotron.

The model gets a short product guide and a compact summary of the session (data, calibration with intervals,
predicted success, next experiment, the agent's steps and the cross-check, the revealed truth if any). It is told
to use only those numbers, to say so when the answer is not in them, and to reply in the visitor's language.
"""

from __future__ import annotations

import json

MAX_TURNS = 12  # conversation turns sent to the model (older ones are dropped)
MAX_CHARS = 1200  # per user message

GUIDE = """Tether Studio fits a simulator to the real world in three domains with the same physics (an object launched
toward a line slides to a stop; part of the surface may differ; the actuator may under-deliver; a camera judges distance):
robot manipulation (an arm pushes a box to a line on a table), autonomous vehicles (a car brakes to stop at a stop line;
wet or icy section of road; speed control; front camera range) and factory inspection (a pneumatic pusher slides a part
along a rail to the inspection camera; oily section; pusher pressure; inspection camera). The session summary gives the
domain and is already in that domain's units (driving: full-size metres and km/h). Driving exports also include CARLA.
Problem: a policy trained in simulation fails on the real robot because the simulator's physics differ from reality
(a slicker or stickier table, a region of different friction such as a spill or a mat, a weaker motor, a tilted camera).
Flow: 1 Data (a phone video of an object flicked across a table with an A4 sheet for scale, a robot push log, or a table
simulated in NVIDIA Newton from settings the visitor chooses) -> 2 Measure (the sheet gives scale and the camera pose;
each push's launch speed and slide in metres) -> 3 Diagnose (an NVIDIA Nemotron agent picks which effects exist and asks
for experiments; numbers come from least-squares fitting, never from the language model; a cross-check overrules the
agent if its model leaves evidence unexplained) -> 4 Calibrated sim (each value with a 90% interval from a bootstrap
over pushes that also redraws systematic video error; friction painted on the visitor's frame; ghost boxes replay each
push in the old and the calibrated simulator; predicted success of a policy trained in each; a range when part of the
table was never measured) -> 5 Retrain (the same learner trains a policy three ways in parallel NVIDIA Newton worlds:
the current simulator, wide domain randomization, and worlds drawn from Tether's bootstrap; when the hidden world is
known it scores each policy there, and Newton can render the three policies side by side) -> 6 Export (first a replay
of every measured run in NVIDIA Newton with the exported physics vs the current simulator; then NVIDIA Newton snippet,
Isaac Lab EventTermCfg whose randomization ranges are the intervals, CARLA for driving, Markdown report, JSON).
Terms: mu / friction 0.2 = icy, 1.0 = rubbery; mu_eff = friction of the object-table pair; patch_y0 = where a region of
different friction starts (m from the launch point); patch_mu = friction inside it; actuator_gain = real launch speed per
unit command; camera_pitch_deg = camera tilt error; lens_k = lens distortion; next experiment = pushes where the plausible
models disagree most inside the 0.2-0.6 m target range."""

SYSTEM = """You are the help assistant inside Tether Studio. Answer the visitor's question in {lang}.
Use only the guide and the session summary below for facts and numbers; if the answer is not there, say so plainly and
suggest what to do in the Studio. Explain like to a smart newcomer: short sentences, concrete numbers with units, no
jargon without a one-line explanation. Keep answers under 120 words unless asked for detail. Do not invent results,
and do not describe buttons, steps or features that are not in the guide (values cannot be typed in; results are used via Export).

GUIDE:
{guide}

SESSION (JSON, may be empty):
{session}"""

LANGS = {"en": "English", "ko": "Korean (한국어, formal 합니다체 endings, concise; never 해요체)"}


def _r(x, d=3):
    return None if x is None else round(float(x), d)


def session_summary(state: dict | None, truth: dict | None = None, step: str | None = None) -> dict:
    """Compact, model-friendly view of a Studio session (state = StudioSession.public()), in the domain's units."""
    if not state:
        return {"step": step, "note": "no data loaded yet"}
    from tether.studio import domains as D

    d = D.get(state.get("domain"))
    out = _summary(state, truth, step)
    out["domain"] = {"name": d.name["en"], "words": d.words["en"], "success_means": d.tolerance_label["en"]}
    for g in (out.get("calibration") or {}).get("differences", []):  # every domain names the fields its own way
        g["what"] = D.FIELD_LABELS[d.id].get(g.get("field"), g["what"])
    if d.scale == 1.0:
        return out
    k, kv = d.scale, d.speed_scale * 3.6

    def L(x):
        return None if x is None else round(x * k, 2)

    out["farthest_stop_m"] = L(out.get("farthest_stop_m"))
    out["notes"] = [D.scale_text(d, n) for n in (out.get("notes") or [])]
    cal = out.get("calibration")
    if cal:
        for g in cal["differences"]:
            if g["unit"] == "m":
                g["current_sim"], g["measured"] = L(g["current_sim"]), L(g["measured"])
                g["interval90"] = [L(x) for x in g["interval90"]]
        cal["stop_residual_m"] = round(cal.pop("stop_residual_cm") / 100 * k, 3)
    ps = out.get("predicted_success")
    if ps:
        ps["target_range_m"] = [L(x) for x in ps.get("target_range_m") or []]
        ps["measured_reach_m"] = L(ps.get("measured_reach_m"))
    nx = out.get("next_experiment")
    if nx:
        nx["max_disagreement_m"] = round((nx.pop("max_disagreement_cm") or 0) / 100 * k, 2)
        for sg in nx["suggestions"]:
            sg["stop_near_m"] = L(sg["stop_near_m"])
            sg["launch_kmh"] = round(sg.pop("launch_mps") * kv, 1)
            if sg.get("command") is not None:
                sg["speed_setpoint_kmh"] = round(sg.pop("command") * kv, 1)
            sg["why"] = D.scale_text(d, sg["why"])
    ag = out.get("agent")
    if ag and ag.get("explanation"):
        ag["explanation"] = D.scale_text(d, ag["explanation"])
    if out.get("revealed_truth"):
        out["revealed_truth"] = D.model_to_domain(d, out["revealed_truth"])
    nr = out.get("newton_replay")
    if nr:
        nr["rms_stop_error_exported_m"], nr["rms_stop_error_current_sim_m"] = L(nr["rms_stop_error_exported_m"]), L(nr["rms_stop_error_current_sim_m"])
    return out


def _summary(state: dict, truth: dict | None, step: str | None) -> dict:
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
            "differences": [{"what": g["label"], "field": g["field"], "current_sim": g["sim"], "measured": g["real"], "interval90": [g["lo"], g["hi"]],
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
        tr = res.get("training")
        if tr:
            out["retrain"] = {"hidden_world_known": tr.get("hidden_known"), "iterations": tr.get("iterations"), "worlds": tr.get("n_worlds"),
                              "success": {k: {"hidden_world": c.get("final_real"), "training_worlds": (c.get("train") or [None])[-1]}
                                          for k, c in tr.get("conditions", {}).items()}}
        v = res.get("verify")
        if v:
            out["newton_replay"] = {"rms_stop_error_exported_m": v.get("rms_calibrated_m"), "rms_stop_error_current_sim_m": v.get("rms_current_m"),
                                    "runs": len(v.get("pushes", []))}
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
