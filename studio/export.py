"""Exports of a calibration: what the engineer takes back to their simulator.

  json      gapcloser.calibration/v1: values, 90% intervals, residuals, structure, next experiments
  newton    NVIDIA Newton (ModelBuilder.ShapeConfig) material values + the measured friction region
  isaaclab  Isaac Lab EventTermCfg domain randomization over the measured intervals (calibrated DR)
  markdown  a one-page report

Notes on friction: GapCloser measures *sliding* (dynamic) friction of the pair. Newton XPBD and PhysX
("average" combine mode, the Isaac Lab default) both use the mean of the two materials, so setting both
the object and the table material to mu_eff reproduces the measured pair. Static friction is not observed
by a sliding test; the exports keep it equal to the dynamic value and say so.
"""

from __future__ import annotations

import json
from datetime import date

from studio.fit import LABELS, UNITS, Calibration
from studio.session import Session

SCHEMA = "gapcloser.calibration/v1"


def _iv(cal: Calibration, f: str) -> tuple[float, float]:
    v = cal.model[f]
    return tuple(cal.intervals.get(f, (v, v)))  # type: ignore[return-value]


def to_json(session: Session, cal: Calibration, nxt: dict | None = None, agent: dict | None = None) -> dict:
    return {"schema": SCHEMA, "created": date.today().isoformat(), "session": {"name": session.name, "source": session.source,
            "pushes": len(session.pushes), "tipped": sum(p.tipped for p in session.pushes), "notes": session.notes},
            "calibration": cal.to_json(), "next_experiment": nxt, "agent": agent}


def newton_snippet(session: Session, cal: Calibration) -> str:
    m = cal.model
    mu = m["mu_eff"]
    lo, hi = _iv(cal, "mu_eff")
    lines = [
        f"# GapCloser calibration for '{session.name}' ({len(session.pushes)} pushes, {date.today().isoformat()})",
        "# NVIDIA Newton: XPBD averages the two materials' mu, so set both to the measured pair value.",
        "import newton",
        "",
        f"MU_EFF = {mu:.4f}  # 90% interval {lo:.4f} - {hi:.4f} (bootstrap over pushes)",
        "object_cfg = newton.ModelBuilder.ShapeConfig(mu=MU_EFF)",
        "table_cfg = newton.ModelBuilder.ShapeConfig(mu=MU_EFF)",
        "# builder.add_shape_box(body, hx=..., hy=..., hz=..., cfg=object_cfg)",
        "# builder.add_ground_plane(cfg=table_cfg)",
    ]
    if session.has_commands and "actuator_gain" in cal.structure:
        g = m["actuator_gain"]
        glo, ghi = _iv(cal, "actuator_gain")
        lines += ["", f"ACTUATOR_GAIN = {g:.4f}  # real launch speed / commanded ({glo:.4f} - {ghi:.4f}); scale commanded velocities"]
    if m.get("patch_y0") is not None:
        y0, pm = m["patch_y0"], m["patch_mu"]
        (ylo, yhi), (plo, phi) = _iv(cal, "patch_y0"), _iv(cal, "patch_mu")
        lines += [
            "",
            "# Measured friction region along the push axis (+y from the launch point).",
            f"FRICTION_REGION = {{'y_start_m': {y0:.4f}, 'mu_eff': {pm:.4f}}}  # y {ylo:.3f}-{yhi:.3f}, mu {plo:.3f}-{phi:.3f}",
            "# Two table bodies leave a seam that trips sliding boxes; GapCloser instead switches the object's own mu",
            "# when it crosses y_start with a Warp kernel and ground mu 0 (see sim/newton_push.py: _patch_setup).",
        ]
    if session.has_targets:
        lines += ["", "# Perception (target distance as the robot's camera reports it):",
                  f"CAMERA = {{'dx_m': {m['camera_dx']:.4f}, 'pitch_error_deg': {m['camera_pitch_deg']:.3f}, 'lens_k_per_m': {m['lens_k']:.4f}}}",
                  "# perceived = true * (1 + tan(pitch)) + dx + lens_k * true**2"]
    return "\n".join(lines) + "\n"


def isaaclab_snippet(session: Session, cal: Calibration) -> str:
    lo, hi = _iv(cal, "mu_eff")
    lines = [
        f'"""GapCloser calibrated domain randomization for \'{session.name}\' ({date.today().isoformat()}).',
        "Ranges are 90% bootstrap intervals of the measured values, not hand-picked bounds: train on what the real",
        'table can plausibly be, nothing wider. PhysX "average" friction combine (the default) matches the measurement."""',
        "from isaaclab.envs import mdp",
        "from isaaclab.managers import EventTermCfg as EventTerm",
        "from isaaclab.managers import SceneEntityCfg",
        "from isaaclab.utils import configclass",
        "",
        "",
        "@configclass",
        "class CalibratedEventCfg:",
        "    # sliding friction of the object-table pair; static friction is not observed by a sliding test",
        "    object_material = EventTerm(",
        "        func=mdp.randomize_rigid_body_material,",
        '        mode="startup",',
        "        params={",
        '            "asset_cfg": SceneEntityCfg("object"),',
        f'            "static_friction_range": ({lo:.4f}, {hi:.4f}),',
        f'            "dynamic_friction_range": ({lo:.4f}, {hi:.4f}),',
        '            "restitution_range": (0.0, 0.0),',
        '            "num_buckets": 64,',
        '            "make_consistent": True,',
        "        },",
        "    )",
    ]
    if session.has_commands and "actuator_gain" in cal.structure:
        glo, ghi = _iv(cal, "actuator_gain")
        lines += ["", f"# Actuator: real launch speed is {cal.model['actuator_gain']:.3f}x the command ({glo:.3f}-{ghi:.3f}).",
                  "# Apply it as the action scale of the push primitive (e.g. JointVelocityActionCfg(scale=...)),",
                  "# or randomize the scale over that interval in your action term."]
    if cal.model.get("patch_y0") is not None:
        lines += ["", f"# Friction region: from y = {cal.model['patch_y0']:.3f} m the pair slides at mu {cal.model['patch_mu']:.3f}",
                  "# (90% " + "{:.3f}-{:.3f}".format(*_iv(cal, "patch_mu")) + "). Model it as a second table prim with",
                  "# sim.RigidBodyMaterialCfg(static_friction=..., dynamic_friction=...) flush with the first."]
    return "\n".join(lines) + "\n"


def _fmt(v, unit: str) -> str:
    if v is None:
        return "none"
    return f"{v:.3f}{(' ' + unit) if unit and unit not in ('-', 'x') else ('x' if unit == 'x' else '')}"


def markdown(session: Session, cal: Calibration, nxt: dict | None = None, agent: dict | None = None) -> str:
    pr = cal.predicted
    out = [f"# Sim2Real calibration: {session.name}", "",
           f"{len(session.pushes)} pushes ({session.source}), structure chosen by {cal.chosen_by}. "
           f"Stop residual {cal.residuals['stop_residual_rms_m'] * 100:.1f} cm rms.", "",
           "## What differs from your simulator", "", "| Parameter | Sim | Real (measured) | 90% interval | |", "|---|---|---|---|---|"]
    for g in cal.gap:
        flag = "changed" if g["significant"] else "within noise"
        out.append(f"| {g['label']} | {_fmt(g['sim'], g['unit'])} | {_fmt(g['real'], g['unit'])} | "
                   f"{_fmt(g['lo'], g['unit'])} – {_fmt(g['hi'], g['unit'])} | {flag} |")
    out += ["", "## Predicted effect (model rollouts, not measured)", "",
            f"Success within {pr['tolerance_m'] * 100:.0f} cm over targets {pr['target_range_m'][0]:.1f}–{pr['target_range_m'][1]:.1f} m, "
            f"for a policy trained in each simulator:", "",
            f"- current sim: **{pr['before']['median'] * 100:.0f}%** (90%: {pr['before']['lo'] * 100:.0f}–{pr['before']['hi'] * 100:.0f}%)",
            (f"- calibrated sim: **{pr['after']['lo'] * 100:.0f}–{pr['after']['hi'] * 100:.0f}%**, depending on the unmeasured table"
             if pr.get("unmeasured_whatif") else
             f"- calibrated sim: **{pr['after']['median'] * 100:.0f}%** (90%: {pr['after']['lo'] * 100:.0f}–{pr['after']['hi'] * 100:.0f}%)")]
    if pr.get("unmeasured_whatif"):
        out.append(f"- your pushes reach {pr['measured_reach_m']:.2f} m; beyond that the prediction includes ±30% friction what-ifs")
    if nxt:
        out += ["", "## Next experiment", ""]
        if nxt["settled"]:
            out.append("The data settles the model over the whole target range; no further pushes needed.")
        for s in nxt["suggestions"]:
            how = f"command {s['command']:.2f}" if s.get("command") is not None else f"launch ~{s['launch_speed_mps']:.2f} m/s"
            out.append(f"- push so it stops near **{s['predicted_stop_m']:.2f} m** ({how}): {s['why']}")
    if agent and agent.get("explanation"):
        out += ["", "## Agent's explanation", "", f"> {agent['explanation']}"]
    if session.notes:
        out += ["", "## Data notes", ""] + [f"- {n}" for n in session.notes]
    out += ["", "Generated by GapCloser Studio (NVIDIA Newton physics, Nemotron agent). Intervals: bootstrap over pushes."]
    return "\n".join(out) + "\n"


def write_all(out_dir, session: Session, cal: Calibration, nxt: dict | None = None, agent: dict | None = None) -> dict:
    from pathlib import Path

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    files = {"calibration.json": json.dumps(to_json(session, cal, nxt, agent), indent=2),
             "newton_calibration.py": newton_snippet(session, cal), "isaaclab_events.py": isaaclab_snippet(session, cal),
             "report.md": markdown(session, cal, nxt, agent)}
    for name, text in files.items():
        (d / name).write_text(text)
    return {k: str(d / k) for k in files}


__all__ = ["to_json", "newton_snippet", "isaaclab_snippet", "markdown", "write_all", "LABELS", "UNITS"]
