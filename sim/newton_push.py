"""Newton implementation of the Tier 0 push task (same rollout contract as AnalyticPushEnv).

One Newton world per target: a cube on a table gets an initial velocity (policy command x actuator
gain) and slides to rest. Object/table friction, density, size and restitution come from ParamSet;
perception params bias what the policy observes (`observe`) and move the render camera.
`render_trial` produces a camera clip for the dashboard. Newton is imported lazily.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from sim.params import ParamSet
from sim.push_task import TRACK_FRAMES, Policy, Rollout, Trial, observe

FPS, SUBSTEPS = 30, 10
MAX_SECONDS = 2.5
W, H = 480, 270


def _build(params: ParamSet, commands: list[float], targets: list[float]):
    import warp as wp

    import newton

    wp.config.quiet = True
    b = newton.ModelBuilder()
    hs = params["object_half_size"]
    cubes = []
    for cmd, tgt in zip(commands, targets):
        b.begin_world()
        cfg = newton.ModelBuilder.ShapeConfig(
            mu=params["object_mu"], density=params["object_density"], restitution=params["restitution"]
        )
        body = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, hs), q=wp.quat_identity()))
        b.add_shape_box(body, hx=hs, hy=hs, hz=hs, cfg=cfg, color=(0.85, 0.85, 0.85))
        marker = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
        b.add_shape_box(-1, xform=wp.transform(p=wp.vec3(0.0, tgt, 0.0005), q=wp.quat_identity()),
                        hx=0.12, hy=0.004, hz=0.0005, cfg=marker, color=(0.46, 0.73, 0.0))
        cubes.append(body)
        b.end_world()
    ground = b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=params["table_mu"]), color=(0.2, 0.2, 0.2))
    model = b.finalize()
    state = model.state()
    qd = state.body_qd.numpy()
    for body, cmd in zip(cubes, commands):
        qd[body, :] = 0.0
        qd[body, 1] = cmd * params["actuator_gain"]
    state.body_qd.assign(qd)
    return model, state, cubes, ground


def _simulate(model, state, frame_cb=None):
    import newton

    solver = newton.solvers.SolverXPBD(model, iterations=10)
    s0, s1, control = state, model.state(), model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    dt = 1.0 / FPS / SUBSTEPS
    for f in range(int(MAX_SECONDS * FPS)):
        for _ in range(SUBSTEPS):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        if frame_cb is not None:
            frame_cb(s0)
        if f > 3 and np.abs(s0.body_qd.numpy()).max() < 1e-3:
            break
    return s0


class NewtonPushEnv:
    def rollout(self, params: ParamSet, policy: Policy, targets: list[float]) -> Rollout:
        observed = [observe(d, params) for d in targets]
        commands = [policy.command(o) for o in observed]
        model, state, cubes, _ = _build(params, commands, targets)
        tracks = [state.body_q.numpy()[cubes, 1].copy()]

        def track(s):
            if len(tracks) <= TRACK_FRAMES:
                tracks.append(s.body_q.numpy()[cubes, 1].copy())

        final = _simulate(model, state, track)
        q = final.body_q.numpy()[cubes]
        ys = q[:, 1]
        tilt = tilt_deg(q[:, 3:7])
        tr = np.asarray(tracks)  # (frames, worlds)
        return Rollout([Trial(d, o, c, float(y), [float(v) for v in tr[:, i]], bool(tilt[i] > TIP_DEG))
                        for i, (d, o, c, y) in enumerate(zip(targets, observed, commands, ys))])


TIP_DEG = 30.0


def tilt_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    """Smallest rotation (deg) that maps the cube's resting pose to its current pose, modulo the
    cube's 90° symmetries: a cube that rolled exactly onto another face reads ~90°, so use the
    angle of whichever body axis is closest to world Z."""
    x, y, z, w = quat_xyzw.T
    # world-frame images of the body axes' z components (third row of the rotation matrix)
    zx = 2 * (x * z - w * y)
    zy = 2 * (y * z + w * x)
    zz = 1 - 2 * (x * x + y * y)
    best = np.max(np.abs(np.stack([zx, zy, zz])), axis=0)
    rolled_face = np.abs(zz) < 0.7  # resting on a different face than it started on
    ang = np.degrees(np.arccos(np.clip(best, -1, 1)))
    return np.where(rolled_face, np.maximum(ang, 90.0), ang)


def _camera_quat(pitch_deg: float):
    """Camera looks along world -X with Z up, pitched down by pitch_deg."""
    import warp as wp

    base = wp.quatf(0.5, 0.5, 0.5, 0.5)
    tilt = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), -math.radians(pitch_deg))
    return base * tilt


def render_trial(params: ParamSet, command: float, target: float, out_path: Path) -> float:
    """Render one push as an animated WebP from a side camera. Returns the measured slide (m)."""
    import warp as wp
    from PIL import Image

    from newton.sensors import SensorTiledCamera

    model, state, cubes, ground = _build(params, [command], [target])
    cam = SensorTiledCamera(model=model)
    cam.default_render_config.enable_shadows = True
    cam.default_render_config.enable_textures = True
    cam.utils.create_default_light(enable_shadows=True)
    cam.utils.assign_checkerboard_material(shape_indices=np.asarray([ground], dtype=np.uint32))
    rays = cam.utils.compute_camera_rays_pinhole(W, H, camera_fovs=math.radians(44.0))
    color = cam.utils.create_color_image_output(W, H, 1)
    # perception error moves the real camera; the sim camera (nominal) stays put
    pos = wp.vec3f(0.85, 0.27 + params["camera_dx"], 0.30 + params["camera_dz"])
    xf = wp.array([[wp.transformf(pos, _camera_quat(17.0 + params["camera_pitch_deg"]))]], dtype=wp.transformf)
    exposure = params["light_intensity"]
    frames: list[Image.Image] = []

    def grab(s):
        model.bvh_refit_shapes(s)
        cam.update(s, xf, rays, color_image=color, clear_data=SensorTiledCamera.GRAY_CLEAR_DATA)
        rgb = cam.utils.to_rgba_from_color(color).numpy()[0, ..., :3].astype(np.float32)
        frames.append(Image.fromarray(np.clip(rgb * exposure, 0, 255).astype(np.uint8)))

    grab(state)
    final = _simulate(model, state, grab)
    for _ in range(FPS // 2):  # hold the last frame so the outcome is readable
        frames.append(frames[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0, quality=70)
    return float(final.body_q.numpy()[cubes[0], 1])


# ---------------------------------------------------------------------------------------------
# Robot-arm rendering: a Franka FR3 strikes the cube. The arm is kinematic (IK-driven, no contact)
# and only visualizes the push; the cube's launch speed and sliding are the same physics as above.
ARM_BASE_Y = -0.62
TCP_INDEX = 11  # fr3_hand_tcp
STRIKE_FRAME = 16
HOME_Q = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785, 0.04, 0.04]


def _arm_targets(hs: float, n_frames: int) -> list[tuple[float, float, float]]:
    """TCP path: hover behind the cube, descend, strike through the launch point, retract."""
    hover, ready = (0.0, -0.16, 0.20), (0.0, -hs - 0.035, 0.035)
    strike, after = (0.0, -hs + 0.015, 0.035), (0.0, -0.10, 0.16)
    lerp = lambda a, b, u: tuple(x + (y - x) * u for x, y in zip(a, b))  # noqa: E731
    ease = lambda u: 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, u)))  # noqa: E731
    out = []
    for f in range(n_frames):
        if f < STRIKE_FRAME - 4:
            out.append(lerp(hover, ready, ease(f / (STRIKE_FRAME - 4))))
        elif f < STRIKE_FRAME:
            out.append(ready)
        elif f < STRIKE_FRAME + 3:
            out.append(lerp(ready, strike, ease((f - STRIKE_FRAME + 1) / 3)))
        else:
            out.append(lerp(strike, after, ease((f - STRIKE_FRAME - 2) / 18)))
    return out


def _solve_arm(targets: list[tuple[float, float, float]]):
    """IK per frame on an arm-only model; returns per-frame body transforms (n_frames, n_bodies, 7)."""
    import warp as wp

    import newton
    import newton.ik as ik

    b = newton.ModelBuilder()
    b.add_urdf(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
               xform=wp.transform(wp.vec3(0.0, ARM_BASE_Y, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2)),
               floating=False)
    m = b.finalize()
    m.joint_q.assign(np.asarray(HOME_Q, dtype=np.float32))
    st = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, st)
    down = st.body_q.numpy()[TCP_INDEX][3:7]  # keep the home orientation (gripper pointing down)
    pos_obj = ik.IKObjectivePosition(link_index=TCP_INDEX, link_offset=wp.vec3(0.0, 0.0, 0.0),
                                     target_positions=wp.array([wp.vec3(*targets[0])], dtype=wp.vec3))
    rot_obj = ik.IKObjectiveRotation(link_index=TCP_INDEX, link_offset_rotation=wp.quat_identity(),
                                     target_rotations=wp.array([wp.vec4(*down)], dtype=wp.vec4))
    lim = ik.IKObjectiveJointLimit(joint_limit_lower=m.joint_limit_lower, joint_limit_upper=m.joint_limit_upper, weight=10.0)
    solver = ik.IKSolver(model=m, n_problems=1, objectives=[pos_obj, rot_obj, lim], lambda_initial=0.1,
                         jacobian_mode=ik.IKJacobianType.ANALYTIC)
    jq = wp.array(np.asarray([HOME_Q], dtype=np.float32), dtype=wp.float32)
    poses = []
    for tgt in targets:
        pos_obj.set_target_position(0, wp.vec3(*tgt))
        solver.step(jq, jq, iterations=24)
        m.joint_q.assign(jq.numpy().reshape(-1))
        newton.eval_fk(m, m.joint_q, m.joint_qd, st)
        poses.append(st.body_q.numpy().copy())
    return np.asarray(poses), m.body_count, b


def render_trial_arm(params: ParamSet, command: float, target: float, out_path: Path) -> float:
    """Like render_trial, with a Franka arm striking the cube. Returns the measured slide (m)."""
    import warp as wp
    from PIL import Image

    import newton
    from newton.sensors import SensorTiledCamera

    wp.config.quiet = True
    hs = params["object_half_size"]
    n_frames = int(MAX_SECONDS * FPS) + STRIKE_FRAME
    arm_poses, n_arm, arm_builder = _solve_arm(_arm_targets(hs, n_frames))

    b = newton.ModelBuilder()
    arm_start = b.body_count
    shape_start = b.shape_count
    b.add_builder(arm_builder)
    for i in range(shape_start, b.shape_count):  # arm is visual only: no contacts
        b.shape_flags[i] = int(b.shape_flags[i]) & ~int(newton.ShapeFlags.COLLIDE_SHAPES) & ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
    cfg = newton.ModelBuilder.ShapeConfig(mu=params["object_mu"], density=params["object_density"], restitution=params["restitution"])
    cube = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, hs), q=wp.quat_identity()))
    b.add_shape_box(cube, hx=hs, hy=hs, hz=hs, cfg=cfg, color=(0.85, 0.85, 0.85))
    marker = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
    b.add_shape_box(-1, xform=wp.transform(p=wp.vec3(0.0, target, 0.0005), q=wp.quat_identity()),
                    hx=0.12, hy=0.004, hz=0.0005, cfg=marker, color=(0.46, 0.73, 0.0))
    ground = b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=params["table_mu"]), color=(0.2, 0.2, 0.2))
    model = b.finalize()
    arm_ids = np.arange(arm_start, arm_start + n_arm)
    inv_m = model.body_inv_mass.numpy(); inv_m[arm_ids] = 0.0; model.body_inv_mass.assign(inv_m)
    inv_i = model.body_inv_inertia.numpy(); inv_i[arm_ids] = 0.0; model.body_inv_inertia.assign(inv_i)

    solver = newton.solvers.SolverXPBD(model, iterations=10)
    s0, s1, control = model.state(), model.state(), model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    cam = SensorTiledCamera(model=model)
    cam.default_render_config.enable_shadows = True
    cam.default_render_config.enable_textures = True
    cam.utils.create_default_light(enable_shadows=True)
    cam.utils.assign_checkerboard_material(shape_indices=np.asarray([ground], dtype=np.uint32))
    rays = cam.utils.compute_camera_rays_pinhole(W, H, camera_fovs=math.radians(46.0))
    color = cam.utils.create_color_image_output(W, H, 1)
    pos = wp.vec3f(1.55, -0.02 + params["camera_dx"], 0.78 + params["camera_dz"])
    xf = wp.array([[wp.transformf(pos, _camera_quat(24.0 + params["camera_pitch_deg"]))]], dtype=wp.transformf)
    exposure = params["light_intensity"]
    frames: list[Image.Image] = []
    dt = 1.0 / FPS / SUBSTEPS

    def set_arm(state, f):
        q = state.body_q.numpy(); q[arm_ids] = arm_poses[f]; state.body_q.assign(q)
        qd = state.body_qd.numpy(); qd[arm_ids] = 0.0; state.body_qd.assign(qd)

    for f in range(n_frames):
        if f == STRIKE_FRAME:
            qd = s0.body_qd.numpy(); qd[cube, :] = 0.0; qd[cube, 1] = command * params["actuator_gain"]; s0.body_qd.assign(qd)
        for _ in range(SUBSTEPS):
            set_arm(s0, f)
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        set_arm(s0, f)
        model.bvh_refit_shapes(s0)
        cam.update(s0, xf, rays, color_image=color, clear_data=SensorTiledCamera.GRAY_CLEAR_DATA)
        rgb = cam.utils.to_rgba_from_color(color).numpy()[0, ..., :3].astype(np.float32)
        frames.append(Image.fromarray(np.clip(rgb * exposure, 0, 255).astype(np.uint8)))
        if f > STRIKE_FRAME + 20 and np.abs(s0.body_qd.numpy()[cube]).max() < 1e-3:
            break
    for _ in range(FPS // 2):
        frames.append(frames[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0, quality=70)
    return float(s0.body_q.numpy()[cube, 1])
