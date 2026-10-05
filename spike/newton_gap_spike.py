"""Spike: can NVIDIA Newton on Mac CPU (1) reproduce a sim-vs-"real" gap and (2) render camera video?

Same push (initial velocity) on a cube in two worlds that differ only in hidden friction.
Outputs: final displacement per world + GIF/PNG frames from SensorTiledCamera.
Run: .venv/bin/python spike/newton_gap_spike.py
"""

import math
import time
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image

import newton
from newton.sensors import SensorTiledCamera

OUT = Path(__file__).parent / "out"
WORLDS = {"sim": 0.8, "real_hidden": 0.3}  # friction mu per world
PUSH_SPEED = 2.5  # m/s along +y
FPS, SECONDS, SUBSTEPS = 30, 2.0, 10
W, H = 320, 240


def build():
    builder = newton.ModelBuilder()
    cube_bodies = []
    for mu in WORLDS.values():
        builder.begin_world()
        cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=500.0)
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -1.0, 0.1), q=wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg, color=(0.85, 0.2, 0.2))
        cube_bodies.append(body)
        builder.end_world()
    ground = builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=1.0), color=(0.6, 0.6, 0.6))
    return builder.finalize(), cube_bodies, ground


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, cube_bodies, ground = build()
    solver = newton.solvers.SolverXPBD(model, iterations=10)
    s0, s1, control = model.state(), model.state(), model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    # initial push: set linear velocity of every cube (try both spatial layouts, verified by displacement)
    qd = s0.body_qd.numpy()
    for b in cube_bodies:
        qd[b, :] = 0.0
        qd[b, 1] = PUSH_SPEED
    s0.body_qd.assign(qd)
    start = s0.body_q.numpy()[cube_bodies, :3].copy()

    cam = SensorTiledCamera(model=model)
    cam.utils.create_default_light(enable_shadows=True)
    cam.utils.assign_checkerboard_material(shape_indices=np.asarray([ground], dtype=np.uint32))
    rays = cam.utils.compute_camera_rays_pinhole(W, H, camera_fovs=math.radians(50.0))
    color = cam.utils.create_color_image_output(W, H, 1)
    n_worlds = len(WORLDS)
    cam_xf = wp.array(
        [[wp.transformf(wp.vec3f(3.0, 0.0, 0.8), wp.quatf(0.5, 0.5, 0.5, 0.5))] * n_worlds],
        dtype=wp.transformf,
    )

    dt = 1.0 / FPS / SUBSTEPS
    frames = {k: [] for k in WORLDS}
    t0 = time.perf_counter()
    for _ in range(int(FPS * SECONDS)):
        for _ in range(SUBSTEPS):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        model.bvh_refit_shapes(s0)
        cam.update(s0, cam_xf, rays, color_image=color, clear_data=SensorTiledCamera.GRAY_CLEAR_DATA)
        rgba = cam.utils.to_rgba_from_color(color).numpy()  # (worlds*cams, H, W, 4)
        for i, k in enumerate(WORLDS):
            frames[k].append(Image.fromarray(rgba[i, ..., :3].astype(np.uint8)))
    elapsed = time.perf_counter() - t0

    end = s0.body_q.numpy()[cube_bodies, :3]
    print(f"sim+render {int(FPS * SECONDS)} frames x {n_worlds} worlds in {elapsed:.1f}s on {wp.get_device()}")
    for (k, mu), a, b in zip(WORLDS.items(), start, end):
        print(f"  {k:12s} mu={mu:.2f}  slide={b[1] - a[1]:+.3f} m  z={b[2]:.3f}")
    for k, fs in frames.items():
        fs[0].save(OUT / f"{k}.gif", save_all=True, append_images=fs[1:], duration=1000 // FPS, loop=0)
        fs[-1].save(OUT / f"{k}_last.png")
    side = Image.new("RGB", (W * 2, H))
    side.paste(frames["sim"][-1], (0, 0))
    side.paste(frames["real_hidden"][-1], (W, 0))
    side.save(OUT / "compare_last.png")
    print(f"  wrote {OUT}/{{sim,real_hidden}}.gif, compare_last.png")


if __name__ == "__main__":
    main()
