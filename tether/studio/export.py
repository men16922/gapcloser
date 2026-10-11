"""Exports of a calibration: what the engineer takes back to their simulator.

  json      tether.calibration/v1: values, 90% intervals, residuals, structure, next experiments
  newton    NVIDIA Newton (ModelBuilder.ShapeConfig) material values + the measured friction region
  isaaclab  Isaac Lab EventTermCfg domain randomization over the measured intervals (calibrated DR)
  carla     driving domain: CARLA 0.9.16 wheel friction and a friction trigger for the measured wet/icy section
  markdown  a one-page report

Every export is written in the session's domain units (studio.domains): the driving domain reports a full-size
car and road even though the fit runs at the base scale (Froude similarity keeps friction unchanged).

Notes on friction: Tether measures *sliding* (dynamic) friction of the pair. Newton XPBD and PhysX
("average" combine mode, the Isaac Lab default) both use the mean of the two materials, so setting both
the object and the table material to mu_eff reproduces the measured pair. Static friction is not observed
by a sliding test; the exports keep it equal to the dynamic value and say so.
"""

from __future__ import annotations

import json
from datetime import date

from tether.studio import domains as D
from tether.studio.fit import LABELS, UNITS, Calibration
from tether.studio.session import Session

SCHEMA = "tether.calibration/v1"


def _iv(cal: Calibration, f: str, d: D.Domain | None = None) -> tuple[float, float]:
    v = cal.model[f]
    iv = tuple(cal.intervals.get(f, (v, v)))
    return tuple(D.interval_to_domain(d, f, iv)) if d is not None else iv  # type: ignore[return-value]


def _m(cal: Calibration, d: D.Domain) -> dict:
    return D.model_to_domain(d, cal.model)


def to_json(session: Session, cal: Calibration, nxt: dict | None = None, agent: dict | None = None) -> dict:
    d = D.of_session(session)
    return {"schema": SCHEMA, "created": date.today().isoformat(),
            "domain": {"id": d.id, "name": d.name["en"], "length_scale": d.scale,
                       "model_in_domain_units": D.model_to_domain(d, cal.model)}, "session": {"name": session.name, "source": session.source,
            "pushes": len(session.pushes), "tipped": sum(p.tipped for p in session.pushes), "notes": session.notes},
            "calibration": cal.to_json(), "next_experiment": nxt, "agent": agent}


def newton_snippet(session: Session, cal: Calibration) -> str:
    d = D.of_session(session)
    w = d.words["en"]
    m = _m(cal, d)
    mu = m["mu_eff"]
    lo, hi = _iv(cal, "mu_eff", d)
    lines = [
        f"# Tether calibration for '{session.name}' ({d.name['en']}, {len(session.pushes)} runs, {date.today().isoformat()})",
        "# NVIDIA Newton: XPBD averages the two materials' mu, so set both to the measured pair value.",
        "import newton",
        "",
        f"MU_EFF = {mu:.4f}  # {w['object']}-{w['surface']} sliding friction, 90% interval {lo:.4f} - {hi:.4f} (bootstrap)",
        "object_cfg = newton.ModelBuilder.ShapeConfig(mu=MU_EFF)",
        "surface_cfg = newton.ModelBuilder.ShapeConfig(mu=MU_EFF)",
        f"# builder.add_shape_box(body, hx=..., hy=..., hz=..., cfg=object_cfg)   # the {w['object']}",
        f"# builder.add_ground_plane(cfg=surface_cfg)                             # the {w['surface']}",
    ]
    if d.scale != 1.0:
        lines.insert(2, f"# Lengths are full scale ({d.name['en'].lower()}); friction does not depend on scale.")
    if session.has_commands and "actuator_gain" in cal.structure:
        g = m["actuator_gain"]
        glo, ghi = _iv(cal, "actuator_gain", d)
        lines += ["", f"ACTUATOR_GAIN = {g:.4f}  # {w['actuator']}: actual speed / commanded ({glo:.4f} - {ghi:.4f}); scale commanded velocities"]
    if m.get("patch_y0") is not None:
        y0, pm = m["patch_y0"], m["patch_mu"]
        (ylo, yhi), (plo, phi) = _iv(cal, "patch_y0", d), _iv(cal, "patch_mu", d)
        lines += [
            "",
            f"# Measured {w['region']} along the travel axis (+y from the launch point).",
            f"FRICTION_REGION = {{'y_start_m': {y0:.4f}, 'mu_eff': {pm:.4f}}}  # y {ylo:.3f}-{yhi:.3f}, mu {plo:.3f}-{phi:.3f}",
            "# Two surface bodies leave a seam that trips sliding objects; Tether instead switches the object's own mu",
            "# when it crosses y_start with a Warp kernel and ground mu 0 (see tether/sim/newton_push.py: _patch_setup).",
        ]
    if session.has_targets:
        lines += ["", f"# Perception ({w['target']} distance as the {w['camera']} reports it):",
                  f"CAMERA = {{'dx_m': {m['camera_dx']:.4f}, 'pitch_error_deg': {m['camera_pitch_deg']:.3f}, 'lens_k_per_m': {m['lens_k']:.5f}}}",
                  "# perceived = true * (1 + tan(pitch)) + dx + lens_k * true**2"]
    return "\n".join(lines) + "\n"


def isaaclab_snippet(session: Session, cal: Calibration) -> str:
    d = D.of_session(session)
    w = d.words["en"]
    lo, hi = _iv(cal, "mu_eff", d)
    lines = [
        f'"""Tether calibrated domain randomization for \'{session.name}\' ({date.today().isoformat()}).',
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
        f"    # sliding friction of the {w['object']}-{w['surface']} pair; static friction is not observed by a sliding test",
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
        g = cal.model["actuator_gain"]
        lines += ["", "",
                  f"# {w['actuator'].capitalize()}: the real launch speed is {g:.3f}x the command (90% {glo:.3f}-{ghi:.3f}). It belongs",
                  "# in the push action's scale (joint names: your robot's).",
                  f"ACTUATOR_GAIN = {g:.4f}",
                  "push_action = mdp.JointVelocityActionCfg(asset_name=\"robot\", joint_names=[\".*\"], scale=ACTUATOR_GAIN)"]
    if cal.model.get("patch_y0") is not None:
        m = _m(cal, d)
        mu1, mu2 = m["mu_eff"], m["patch_mu"]
        combine = "min" if mu2 < mu1 else "max"
        ylo, yhi = _iv(cal, "patch_y0", d)
        lines += ["", "",
                  f"# {w['region'].capitalize()}: from y = {m['patch_y0']:.3f} m (90% {ylo:.3f}-{yhi:.3f}) the pair slides at mu {mu2:.3f}",
                  "# (90% " + "{:.3f}-{:.3f}".format(*_iv(cal, "patch_mu", d)) + f"). A static {w['surface']} prim whose top is flush with the first;",
                  "# shorten the first one so it ends at REGION_Y0 (overlapping coplanar surfaces leave PhysX two materials to pick from).",
                  f"# Its combine mode '{combine}' outranks PhysX's default 'average', so the pair takes the region's value.",
                  "import isaaclab.sim as sim_utils",
                  "from isaaclab.assets import AssetBaseCfg",
                  "",
                  f"REGION_Y0 = {m['patch_y0']:.4f}  # m along the push axis from the launch point",
                  "SURFACE_TOP_Z = 0.0  # set to your table top height",
                  "calibrated_region = AssetBaseCfg(",
                  "    prim_path=\"{ENV_REGEX_NS}/CalibratedRegion\",",
                  "    spawn=sim_utils.CuboidCfg(",
                  f"        size=(0.4, {max(1.0, 2.0 * d.scale):.1f}, 0.002),  # across the path, along it from REGION_Y0, thin",
                  "        collision_props=sim_utils.CollisionPropertiesCfg(),",
                  f"        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction={mu2:.4f}, dynamic_friction={mu2:.4f},",
                  f"                                                        friction_combine_mode=\"{combine}\"),",
                  "    ),",
                  f"    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, REGION_Y0 + {max(1.0, 2.0 * d.scale) / 2:.1f}, SURFACE_TOP_Z - 0.001)),",
                  ")"]
    return "\n".join(lines) + "\n"


def carla_snippet(session: Session, cal: Calibration) -> str:
    """Driving domain: the measured road as CARLA physics. CARLA's tire_friction is a per-vehicle scalar, not mu
    itself, so the snippet scales your vehicle's own value by measured / simulated friction, and checks the
    result with a braking test whose expected distance follows from the measured mu."""
    d = D.of_session(session)
    m = _m(cal, d)
    from tether.sim.params import effective_friction

    mu_sim = effective_friction(cal.base["object_mu"], cal.base["table_mu"])
    lo, hi = _iv(cal, "mu_eff", d)
    v = 50 / 3.6
    lines = [
        f'"""Tether calibration for CARLA 0.9.16: \'{session.name}\' ({len(session.pushes)} braking runs, {date.today().isoformat()}).',
        "tire_friction in CARLA is a per-vehicle scalar, not mu; scale your vehicle's value by measured/simulated mu,",
        'then verify with the braking check at the bottom. Values are 90% bootstrap intervals of the measurement."""',
        "import carla",
        "",
        f"MU_SIM = {mu_sim:.3f}     # tire-road friction your simulator assumed",
        f"MU_ROAD = {m['mu_eff']:.4f}   # measured, 90% {lo:.4f} - {hi:.4f}",
        "",
        'client = carla.Client("localhost", 2000)',
        "world = client.get_world()",
        "ego = world.get_actors().filter(\"vehicle.*\")[0]  # your ego vehicle",
        "pc = ego.get_physics_control()",
        "tire0 = [w.tire_friction for w in pc.wheels]",
        "wheels = pc.wheels",
        "for w, t in zip(wheels, tire0):",
        "    w.tire_friction = t * MU_ROAD / MU_SIM",
        "pc.wheels = wheels",
        "ego.apply_physics_control(pc)",
    ]
    if session.has_commands and "actuator_gain" in cal.structure:
        g = m["actuator_gain"]
        glo, ghi = _iv(cal, "actuator_gain", d)
        lines += ["", f"SPEED_GAIN = {g:.4f}  # actual / commanded approach speed ({glo:.4f} - {ghi:.4f}): divide your speed set-points by it,",
                  "# or randomize it over that interval in your scenario runner."]
    if m.get("patch_y0") is not None:
        (ylo, yhi), (plo, phi) = _iv(cal, "patch_y0", d), _iv(cal, "patch_mu", d)
        lines += [
            "",
            f"# Wet or icy section from {m['patch_y0']:.1f} m after the brake point (90% {ylo:.1f}-{yhi:.1f} m),",
            f"# friction {m['patch_mu']:.3f} (90% {plo:.3f}-{phi:.3f}). CARLA's friction trigger sets the tire friction of",
            "# wheels inside its box; extents are in centimetres.",
            f"Y_START, MU_REGION, LENGTH_M, HALF_WIDTH_M = {m['patch_y0']:.2f}, {m['patch_mu']:.4f}, 60.0, 1.75",
            "brake_point = ego.get_transform()  # place the ego at your brake point first",
            "fwd = brake_point.get_forward_vector()",
            "center = brake_point.location + fwd * (Y_START + LENGTH_M / 2)",
            'bp = world.get_blueprint_library().find("static.trigger.friction")',
            'bp.set_attribute("friction", str(tire0[0] * MU_REGION / MU_SIM))',
            'bp.set_attribute("extent_x", str(LENGTH_M / 2 * 100))',
            'bp.set_attribute("extent_y", str(HALF_WIDTH_M * 100))',
            'bp.set_attribute("extent_z", str(200.0))',
            "world.spawn_actor(bp, carla.Transform(center, brake_point.rotation))",
        ]
    if session.has_targets:
        lines += ["", "# Front camera: distance to the stop line as the camera reports it",
                  f"CAMERA = {{'range_offset_m': {m['camera_dx']:.3f}, 'pitch_error_deg': {m['camera_pitch_deg']:.3f}}}",
                  "# apply the pitch error to your camera rig's transform (carla.Rotation(pitch=...)) to reproduce the bias"]
    stop = v * v / (2 * m["mu_eff"] * 9.81)
    lines += ["", f"# Braking check: a full stop from 50 km/h on the dry section should take about {stop:.1f} m",
              f"# (v^2 / (2 mu g) with mu {m['mu_eff']:.3f}); if your CARLA vehicle differs, adjust tire_friction until it matches."]
    return "\n".join(lines) + "\n"


def _fmt(v, unit: str) -> str:
    if v is None:
        return "none"
    return f"{v:.3f}{(' ' + unit) if unit and unit not in ('-', 'x') else ('x' if unit == 'x' else '')}"


def _len(d: D.Domain, base_m: float, dec: int = 2) -> str:
    v = base_m * d.scale
    return f"{v * 100:.0f} cm" if d.scale == 1.0 and abs(v) < 0.1 else f"{v:.{dec if abs(v) < 10 else 1}f} m"


def _spd(d: D.Domain, base_mps: float) -> str:
    return f"{base_mps:.2f} m/s" if d.scale == 1.0 else f"{base_mps * d.speed_scale * 3.6:.0f} km/h"


def markdown(session: Session, cal: Calibration, nxt: dict | None = None, agent: dict | None = None) -> str:
    d = D.of_session(session)
    w = d.words["en"]
    labels = D.FIELD_LABELS[d.id]
    pr = cal.predicted
    out = [f"# Sim2Real calibration: {session.name}", "", f"Domain: **{d.name['en']}**. {d.blurb['en']}", "",
           f"{len(session.pushes)} runs ({session.source}), structure chosen by {cal.chosen_by}. "
           f"Stop residual {_len(d, cal.residuals['stop_residual_rms_m'])} rms.", "",
           "## What differs from your simulator", "", "| Parameter | Sim | Real (measured) | 90% interval | |", "|---|---|---|---|---|"]
    for g in cal.gap:
        flag = "changed" if g["significant"] else "within noise"
        f = g["field"]
        k = d.scale if f in ("patch_y0", "camera_dx") else (1 / d.scale if f == "lens_k" else 1.0)
        sc = (lambda x: None if x is None else x * k)
        out.append(f"| {labels.get(f, g['label'])} | {_fmt(sc(g['sim']), g['unit'])} | {_fmt(sc(g['real']), g['unit'])} | "
                   f"{_fmt(sc(g['lo']), g['unit'])} – {_fmt(sc(g['hi']), g['unit'])} | {flag} |")
    t0, t1 = pr["target_range_m"]
    out += ["", "## Predicted effect (model rollouts, not measured)", "",
            f"Success = stopping within {_len(d, pr['tolerance_m'])} of the {w['target']}, over targets {_len(d, t0, 1)}–{_len(d, t1, 1)}, "
            f"for a policy trained in each simulator:", "",
            f"- current sim: **{pr['before']['median'] * 100:.0f}%** (90%: {pr['before']['lo'] * 100:.0f}–{pr['before']['hi'] * 100:.0f}%)",
            (f"- calibrated sim: **{pr['after']['lo'] * 100:.0f}–{pr['after']['hi'] * 100:.0f}%**, depending on the unmeasured {w['surface']}"
             if pr.get("unmeasured_whatif") else
             f"- calibrated sim: **{pr['after']['median'] * 100:.0f}%** (90%: {pr['after']['lo'] * 100:.0f}–{pr['after']['hi'] * 100:.0f}%)")]
    if pr.get("unmeasured_whatif"):
        out.append(f"- your runs reach {_len(d, pr['measured_reach_m'])}; beyond that the prediction includes ±30% friction what-ifs")
    if nxt:
        out += ["", "## Next experiment", ""]
        if nxt["settled"]:
            out.append("The data settles the model over the whole target range; no further runs needed.")
        for sg in nxt["suggestions"]:
            if sg.get("command") is not None:
                how = f"command {sg['command']:.2f}" if d.scale == 1.0 else f"speed set-point {_spd(d, sg['command'])}"
            else:
                how = f"launch ~{_spd(d, sg['launch_speed_mps'])}"
            out.append(f"- run so the {w['object']} stops near **{_len(d, sg['predicted_stop_m'])}** ({how}): {D.scale_text(d, sg['why'])}")
    if agent and agent.get("explanation"):
        out += ["", "## Agent's explanation", "", f"> {D.scale_text(d, agent['explanation'])}"]
    if session.notes:
        out += ["", "## Data notes", ""] + [f"- {D.scale_text(d, n)}" for n in session.notes]
    if d.scale != 1.0:
        out += ["", f"Lengths are full scale. The fit runs on a Froude-scaled model (lengths 1:{d.scale:.0f}, speeds 1:{d.speed_scale:.0f}); "
                "friction is scale-free, so stop = v² / (2 μ g) holds at both scales."]
    out += ["", "Generated by Tether Studio (NVIDIA Newton physics, Nemotron agent). Intervals: bootstrap over runs."]
    return "\n".join(out) + "\n"


def write_all(out_dir, session: Session, cal: Calibration, nxt: dict | None = None, agent: dict | None = None) -> dict:
    from pathlib import Path

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    files = {"calibration.json": json.dumps(to_json(session, cal, nxt, agent), indent=2),
             "newton_calibration.py": newton_snippet(session, cal), "isaaclab_events.py": isaaclab_snippet(session, cal),
             "report.md": markdown(session, cal, nxt, agent)}
    if D.of_session(session).id == "driving":
        files["carla_calibration.py"] = carla_snippet(session, cal)
    for name, text in files.items():
        (d / name).write_text(text)
    return {k: str(d / k) for k in files}


__all__ = ["to_json", "newton_snippet", "isaaclab_snippet", "carla_snippet", "markdown", "write_all", "LABELS", "UNITS"]
