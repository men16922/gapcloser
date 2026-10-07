"""Record Newton push trials as per-frame poses for the browser 3D viewer.

Same physics as `sim.newton_push.render_trial_arm` (kinematic IK Franka FR3 strikes, the cube slides in
NVIDIA Newton under the given ParamSet), without the camera: every frame stores the cube pose and every
Franka link transform. `record_pair` records the agent's sim and the hidden "real" world for one command
so the viewer can overlay them (real solid, sim as a ghost).

JSON schema "gapcloser.replay/1" (positions in m, Z up, quaternions xyzw, 4 decimals, FPS frames/s):
  {schema, fps, strike_frame, scene: {target, tol, arm_base, table, marker}, links: [names],
   trials: {real|sim: {command, launch_speed, half_size, params, slide, tipped,
                       cube: [[x,y,z,qx,qy,qz,qw], ...],          # one per frame
                       arm:  [[link0 x,y,z,qx,qy,qz,qw, link1 ...], ...]}}}  # trailing still frames trimmed
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from sim.newton_push import (ARM_BASE_Y, FPS, MAX_SECONDS, STRIKE_FRAME, SUBSTEPS, TIP_DEG, _arm_targets, _patch_setup,
                              _solve_arm, tilt_deg)
from sim.push_task import frictions
from sim.params import ParamSet

SCHEMA = "gapcloser.replay/1"
TABLE = {"size": [1.4, 1.8], "center": [0.0, 0.1]}  # visual extent only; Newton uses an infinite ground plane
MARKER = {"hx": 0.12, "hy": 0.004}
SHOWN_PARAMS = ("object_mu", "table_mu", "actuator_gain", "object_half_size", "object_density", "restitution")


def _r(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64), 4).tolist()


@functools.lru_cache(maxsize=8)
def _arm(hs: float, n_frames: int):
    return _solve_arm(_arm_targets(hs, n_frames))


def record_trial(params: ParamSet, command: float, target: float) -> dict:
    """One push in Newton; returns per-frame cube pose + Franka link transforms (see module doc)."""
    import warp as wp

    import newton

    wp.config.quiet = True
    hs = params["object_half_size"]
    n_frames = int(MAX_SECONDS * FPS) + STRIKE_FRAME
    arm_poses, n_arm, arm_builder = _arm(hs, n_frames)

    # scene construction mirrors render_trial_arm so the cube's motion is identical to the clips
    b = newton.ModelBuilder()
    arm_start = b.body_count
    shape_start = b.shape_count
    b.add_builder(arm_builder)
    for i in range(shape_start, b.shape_count):
        b.shape_flags[i] = int(b.shape_flags[i]) & ~int(newton.ShapeFlags.COLLIDE_SHAPES) & ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
    cfg = newton.ModelBuilder.ShapeConfig(mu=params["object_mu"], density=params["object_density"], restitution=params["restitution"])
    cube = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, hs), q=wp.quat_identity()))
    b.add_shape_box(cube, hx=hs, hy=hs, hz=hs, cfg=cfg)
    ground = b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=params["table_mu"]))
    model = b.finalize()
    labels = list(getattr(arm_builder, "body_label", None) or arm_builder.body_key)
    arm_ids = np.arange(arm_start, arm_start + n_arm)
    inv_m = model.body_inv_mass.numpy(); inv_m[arm_ids] = 0.0; model.body_inv_mass.assign(inv_m)
    inv_i = model.body_inv_inertia.numpy(); inv_i[arm_ids] = 0.0; model.body_inv_inertia.assign(inv_i)

    solver = newton.solvers.SolverXPBD(model, iterations=10)
    s0, s1, control = model.state(), model.state(), model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    dt = 1.0 / FPS / SUBSTEPS

    def set_arm(state, f):
        q = state.body_q.numpy(); q[arm_ids] = arm_poses[f]; state.body_q.assign(q)
        qd = state.body_qd.numpy(); qd[arm_ids] = 0.0; state.body_qd.assign(qd)

    patch = _patch_setup(model, params, [cube], ground)
    cube_frames, arm_frames = [], []
    for f in range(n_frames):
        if f == STRIKE_FRAME:
            qd = s0.body_qd.numpy(); qd[cube, :] = 0.0; qd[cube, 1] = command * params["actuator_gain"]; s0.body_qd.assign(qd)
        for _ in range(SUBSTEPS):
            set_arm(s0, f)
            if patch is not None:
                patch(s0)
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        set_arm(s0, f)
        cube_frames.append(s0.body_q.numpy()[cube].copy())
        arm_frames.append(arm_poses[f].reshape(-1))
        if f > STRIKE_FRAME + 20 and np.abs(s0.body_qd.numpy()[cube]).max() < 1e-3:
            break
    cube_q = np.asarray(cube_frames)
    arm = np.round(np.asarray(arm_frames, dtype=np.float64), 4)
    last = len(arm)
    while last > 1 and np.array_equal(arm[last - 1], arm[last - 2]):
        last -= 1
    return {
        "links": labels,
        "command": round(float(command), 4),
        "launch_speed": round(float(command * params["actuator_gain"]), 4),
        "half_size": round(float(hs), 4),
        "patch": _patch_info(params),
        "params": {k: round(float(params[k]), 4) for k in SHOWN_PARAMS},
        "slide": round(float(cube_q[-1, 1]), 4),
        "tipped": bool(tilt_deg(cube_q[:, 3:7]).max() > TIP_DEG),  # peak over frames: a double roll ends upright
        "cube": _r(cube_q),
        "arm": arm[:last].tolist(),
    }


def _patch_info(params: ParamSet) -> dict | None:
    """Table region with different friction in this world (drawn as a strip by the viewer), or None."""
    mu1, mu2, y0 = frictions(params)
    return None if mu2 is None else {"y0": round(float(y0), 4), "mu_near": round(mu1, 3), "mu_far": round(mu2, 3)}


def record_pair(sim_params: ParamSet, sim_command: float, real_params: ParamSet, real_command: float,
                target: float, tol: float = 0.03) -> dict:
    """Sim (agent's model) and real (hidden physics) trial for the same policy and target."""
    trials = {"real": record_trial(real_params, real_command, target), "sim": record_trial(sim_params, sim_command, target)}
    links = trials["real"].pop("links")
    trials["sim"].pop("links")
    return {
        "schema": SCHEMA, "fps": FPS, "strike_frame": STRIKE_FRAME,
        "scene": {"target": round(float(target), 4), "tol": tol, "arm_base": [0.0, ARM_BASE_Y, 0.0],
                  "table": TABLE, "marker": MARKER},
        "links": links,
        "trials": trials,
    }


def write_replay(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path
