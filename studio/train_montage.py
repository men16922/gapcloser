"""Parallel training worlds, rendered: NVIDIA Newton's tiled camera draws 16 training worlds at once (4 x 4), each
with its own physics drawn from Tether's measured ranges, every world aiming at the same target with the policy as it
was at a given training iteration. Three episodes (first, early, final iteration) make the montage: the cars or
parts scatter around the line at first and gather on it as the policy learns.

Run: .venv/bin/python -m studio.train_montage brake-log  ->  video/out/montage-brake-log.mp4
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from studio.fonts import font

from sim.push_task import SUCCESS_TOL, frictions, observe, unobserve
from studio.fit import calibrate
from studio.session import load
from studio.train import GRID, TablePolicy, hidden_params, train, training_worlds

ROOT = Path(__file__).resolve().parent.parent
TILE_W, TILE_H, COLS = 448, 252, 4
TARGET = 0.45


def render_episode(worlds, cmds, targets, domain: str, frames_dir: Path, start_index: int) -> tuple[int, list[bool]]:
    import warp as wp

    import newton
    from newton.sensors import SensorTiledCamera
    from sim.newton_push import _simulate, _worlds_mu_kernel
    from studio.video_sample import FOV_V, HALF, LOOK, _scenery, look_at_quat

    wp.config.quiet = True
    look = LOOK.get(domain, LOOK["robot"])
    b = newton.ModelBuilder()
    vis = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False, density=0.0)
    bodies = []
    for w, t in zip(worlds, targets):
        b.begin_world()
        body = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, HALF), q=wp.quat_identity()))
        b.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, density=400.0), color=look["object"])
        _scenery(b, wp, vis, domain, body, t)
        if domain == "robot":
            b.add_shape_box(-1, xform=wp.transform(p=wp.vec3(0.0, t, 0.0005), q=wp.quat_identity()), hx=0.07, hy=0.004, hz=0.0005,
                            cfg=vis, color=(0.46, 0.73, 0.0))
        bodies.append(body)
        b.end_world()
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0), color=look["ground"])
    model = b.finalize()
    state = model.state()
    qd = state.body_qd.numpy()
    for body, c, w in zip(bodies, cmds, worlds):
        qd[body, 1] = c * w["actuator_gain"]
    state.body_qd.assign(qd)
    shape_body = model.shape_body.numpy()
    shapes = [int(np.flatnonzero(shape_body == bd)[0]) for bd in bodies]
    fr = [frictions(w) for w in worlds]
    arrs = [wp.array(bodies, dtype=int), wp.array(shapes, dtype=int),
            wp.array([f[2] if f[1] is not None else 1e9 for f in fr], dtype=float),
            wp.array([2 * f[0] for f in fr], dtype=float), wp.array([2 * (f[1] if f[1] is not None else f[0]) for f in fr], dtype=float)]
    kern = _worlds_mu_kernel()

    def patch(s):
        wp.launch(kern, dim=len(bodies), inputs=[s.body_q, *arrs, model.shape_material_mu])

    cam = SensorTiledCamera(model=model)
    cam.default_render_config.enable_shadows = True
    cam.utils.create_default_light(enable_shadows=True)
    rays = cam.utils.compute_camera_rays_pinhole(TILE_W, TILE_H, camera_fovs=FOV_V)
    color = cam.utils.create_color_image_output(TILE_W, TILE_H, 1)
    pos, look_at = (0.40, 0.36, 0.20), (0.0, 0.40, 0.0)  # side view centred on the line: short, on it, past it read at a glance
    quat, _ = look_at_quat(pos, look_at)
    xf = wp.array([[wp.transformf(wp.vec3f(*pos), wp.quatf(*quat)) for _ in worlds]], dtype=wp.transformf)  # (cameras, worlds)
    n = [start_index]

    def grab(s, hold=1):
        model.bvh_refit_shapes(s)
        cam.update(s, xf, rays, color_image=color, clear_data=SensorTiledCamera.GRAY_CLEAR_DATA)
        rgb = cam.utils.to_rgba_from_color(color).numpy()[..., :3]  # (worlds x cameras, H, W, 3), world-major
        tiles = [Image.fromarray(rgb[i].astype(np.uint8)) for i in range(len(worlds))]
        for _ in range(hold):
            im = Image.new("RGB", (TILE_W * COLS, TILE_H * ((len(worlds) + COLS - 1) // COLS)))
            for i, t in enumerate(tiles):
                im.paste(t, ((i % COLS) * TILE_W, (i // COLS) * TILE_H))
            im.save(frames_dir / f"g{n[0]:05d}.png")
            n[0] += 1

    grab(state, hold=8)
    final = _simulate(model, state, grab, patch)
    grab(final, hold=20)
    ys = final.body_q.numpy()[bodies, 1]
    return n[0], [abs(float(y) - t) <= SUCCESS_TOL for y, t in zip(ys, targets)]


def montage(sample: str, out: Path) -> dict:
    rec = json.loads((ROOT / "runs/studio-demo" / f"{sample}.json").read_text())
    st = rec["stages"][-1]
    s = load(json.dumps(st["session"]), "x.json", st["session"]["name"])
    domain = rec["sample"].get("domain", "robot")
    cal = calibrate(s, st["result"]["calibration"]["structure"])
    res = train(s, cal, hidden_params(rec["truth"]["model"], s))
    hist = res["conditions"]["tether"]["history"]
    worlds = training_worlds("tether", s, cal)
    k = GRID.index(TARGET) if TARGET in GRID else min(range(len(GRID)), key=lambda i: abs(GRID[i] - TARGET))
    targets = [unobserve(GRID[k], w) for w in worlds]  # where each world's camera puts the line
    tmp = Path(tempfile.mkdtemp())
    episodes = []
    idx = 0
    for it in (0, 1, 3, len(hist) - 1):
        pol = TablePolicy(hist[it])
        cmds = [pol.command(observe(t, w)) for t, w in zip(targets, worlds)]
        a = idx
        idx, hits = render_episode(worlds, cmds, targets, domain, tmp, idx)
        episodes.append((it, a, idx, hits))
    fd = font("display", 54)
    fm = font("mono", 24)
    frames = tmp / "out"
    frames.mkdir()
    j = 0
    for it, a, z, hits in episodes:
        for i in range(a, z):
            g = Image.open(tmp / f"g{i:05d}.png")
            im = Image.new("RGB", (1920, 1080))
            im.paste(g.resize((1792, 1008)), (64, 64))
            d = ImageDraw.Draw(im)
            d.rectangle((64, 0, 1856, 62), fill=(0, 0, 0))
            d.text((64, 8), f"16 parallel training worlds, Tether's measured ranges · iteration {it}", font=fd, fill=(238, 238, 238))
            if i > z - 20:
                d.text((1300, 22), f"{sum(hits)}/16 on the line", font=fm, fill=(118, 185, 0) if sum(hits) > 12 else (226, 87, 76))
            d.rectangle((64, 1072, 1856, 1080), fill=(0, 0, 0))
            im.save(frames / f"f{j:05d}.png")
            j += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(frames / "f%05d.png"), "-r", "30", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "21", str(out)], check=True)
    return {"out": str(out), "episodes": [(it, sum(h)) for it, _, _, h in episodes]}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "brake-log"
    print(montage(name, ROOT / "video" / "out" / f"montage-{name}.mp4"))
