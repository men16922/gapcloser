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
        ys = final.body_q.numpy()[cubes, 1]
        tr = np.asarray(tracks)  # (frames, worlds)
        return Rollout([Trial(d, o, c, float(y), [float(v) for v in tr[:, i]])
                        for i, (d, o, c, y) in enumerate(zip(targets, observed, commands, ys))])


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
