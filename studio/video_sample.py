"""A phone-style test video rendered by NVIDIA Newton (SensorTiledCamera), with ground truth.

Scene: a table with a sheet of A4 paper lying beside the push path (the scale reference the Studio asks
for), a wet strip further out (lower friction), an orange box flicked several times from roughly the same
spot. Camera: hand-held height and angle, oblique, like a phone on a tripod or held still.

The video is synthetic and labelled as such everywhere; it exists to test the tracking + calibration
pipeline end to end against known physics. A real phone video goes through exactly the same path.

Run: .venv/bin/python -m studio.video_sample   -> studio/samples/flick-video.mp4 (+ .truth.json)
"""

from __future__ import annotations

import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np

from sim.params import ParamSet
from sim.push_task import frictions

OUT = Path(__file__).resolve().parent / "samples"
W, H, FPS, SUBSTEPS = 960, 540, 30, 10
FOV_V = math.radians(50.0)
CAM_POS = (0.62, -0.42, 0.48)
CAM_LOOK = (0.02, 0.30, 0.0)
SHEET = {"center": (0.24, 0.16), "size_x": 0.210, "size_y": 0.297}  # A4, long side along the push axis
STRIP = (0.36, 0.30)  # (y start, effective friction beyond it)
MU_EFF = 0.55
HALF = 0.03  # box half size (6 cm box)
SPEEDS = (1.05, 1.3, 1.55, 1.75, 1.95, 2.15, 1.45)  # first video: hand flicks, m/s (mostly short)
SPEEDS_2 = (2.3, 2.45, 2.2, 2.55)  # second video: the longer pushes the Studio asks for


def look_at_quat(pos, target):
    """Warp quaternion (x, y, z, w) for a camera looking -Z with +Y up (Newton ray convention)."""
    p, t = np.asarray(pos, float), np.asarray(target, float)
    fwd = (t - p) / np.linalg.norm(t - p)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd], axis=1)  # columns = camera axes in world
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return (x, y, z, w), R


def project(points_world: np.ndarray) -> np.ndarray:
    """Pixel coordinates of world points in the rendered camera (for ground truth corners)."""
    _, R = look_at_quat(CAM_POS, CAM_LOOK)
    f = (H / 2) / math.tan(FOV_V / 2)
    pc = (points_world - np.asarray(CAM_POS)) @ R  # camera coords
    u = W / 2 + f * pc[:, 0] / -pc[:, 2]
    v = H / 2 - f * pc[:, 1] / -pc[:, 2]
    return np.stack([u, v], axis=1)


def sheet_corners_world() -> np.ndarray:
    cx, cy = SHEET["center"]
    hx, hy = SHEET["size_x"] / 2, SHEET["size_y"] / 2
    return np.array([[cx - hx, cy - hy, 0], [cx + hx, cy - hy, 0], [cx + hx, cy + hy, 0], [cx - hx, cy + hy, 0]], float)


def render(out: Path = OUT / "flick-video.mp4", speeds=SPEEDS, seed: int = 3, hard: bool = False) -> dict:
    """hard: a worse phone. Textured table, hand-held camera shake (1.5 mm, 0.15 deg per frame), motion blur
    (blend with the previous frame), exposure flicker, heavier compression. Used to test the tracker."""
    import warp as wp

    import newton
    from newton.sensors import SensorTiledCamera
    from sim.newton_push import _patch_setup, _simulate

    wp.config.quiet = True
    rng = random.Random(seed)
    params = ParamSet.nominal().with_(object_mu=MU_EFF, table_mu=MU_EFF, patch_y0=STRIP[0], patch_mu=STRIP[1],
                                      object_half_size=HALF)
    quat, _ = look_at_quat(CAM_POS, CAM_LOOK)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30" if hard else "20", str(out)],
                            stdin=subprocess.PIPE)
    truth = []
    frame_no = 0
    for i, v in enumerate(speeds):
        y_start = rng.uniform(-0.015, 0.015)
        b = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(mu=MU_EFF, density=400.0)
        body = b.add_body(xform=wp.transform(p=wp.vec3(0.0, y_start, HALF), q=wp.quat_identity()))
        b.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=cfg, color=(0.95, 0.42, 0.08))
        vis = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
        cx, cy = SHEET["center"]
        b.add_shape_box(-1, xform=wp.transform(p=wp.vec3(cx, cy, 0.0004), q=wp.quat_identity()),
                        hx=SHEET["size_x"] / 2, hy=SHEET["size_y"] / 2, hz=0.0004, cfg=vis, color=(0.96, 0.96, 0.94))
        b.add_shape_box(-1, xform=wp.transform(p=wp.vec3(0.0, (STRIP[0] + 1.2) / 2, 0.0002), q=wp.quat_identity()),
                        hx=0.16, hy=(1.2 - STRIP[0]) / 2, hz=0.0002, cfg=vis, color=(0.20, 0.24, 0.30))
        ground = b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=MU_EFF), color=(0.33, 0.30, 0.27))
        model = b.finalize()
        state = model.state()
        cam = SensorTiledCamera(model=model)
        cam.default_render_config.enable_shadows = True
        cam.utils.create_default_light(enable_shadows=True)
        rays = cam.utils.compute_camera_rays_pinhole(W, H, camera_fovs=FOV_V)
        color = cam.utils.create_color_image_output(W, H, 1)
        xf = wp.array([[wp.transformf(wp.vec3f(*CAM_POS), wp.quatf(*quat))]], dtype=wp.transformf)
        noise = np.random.default_rng(seed * 100 + i)
        if hard:
            cam.utils.assign_checkerboard_material(shape_indices=np.asarray([ground], dtype=np.uint32))
        prev = {"rgb": None}

        def grab(s, hold: int = 1):
            nonlocal frame_no
            model.bvh_refit_shapes(s)
            for _ in range(hold):  # sensor noise like a phone in indoor light
                pose = xf
                if hard:  # hand-held: the camera wanders a little every frame
                    pos = np.asarray(CAM_POS) + noise.normal(0, 0.0015, 3)
                    look = np.asarray(CAM_LOOK) + noise.normal(0, 0.0015, 3) + (pos - np.asarray(CAM_POS))
                    q, _ = look_at_quat(pos, look)
                    pose = wp.array([[wp.transformf(wp.vec3f(*pos), wp.quatf(*q))]], dtype=wp.transformf)
                cam.update(s, pose, rays, color_image=color, clear_data=SensorTiledCamera.GRAY_CLEAR_DATA)
                rgb = cam.utils.to_rgba_from_color(color).numpy()[0, ..., :3].astype(np.float32)
                if hard:
                    rgb = rgb * noise.uniform(0.95, 1.05)
                    if prev["rgb"] is not None:
                        rgb, prev["rgb"] = 0.6 * rgb + 0.4 * prev["rgb"], rgb
                    else:
                        prev["rgb"] = rgb
                img = np.clip(rgb + noise.normal(0, 4.0 if hard else 3.0, rgb.shape), 0, 255).astype(np.uint8)
                proc.stdin.write(img.tobytes())
                frame_no += 1

        grab(state, hold=12)  # box at rest before the flick
        first = frame_no
        qd = state.body_qd.numpy()
        qd[body, 1] = v
        state.body_qd.assign(qd)
        ys = []

        def cb(s):
            ys.append(float(s.body_q.numpy()[body, 1]))
            grab(s)

        final = _simulate(model, state, cb, _patch_setup(model, params, [body], ground))
        grab(final, hold=14)
        stop_y = float(final.body_q.numpy()[body, 1])
        truth.append({"push": i, "launch_frame": first, "launch_speed_mps": v, "start_y_m": round(y_start, 4),
                      "stop_y_m": round(stop_y, 4), "slide_m": round(stop_y - y_start, 4)})
    proc.stdin.close()
    proc.wait()
    corners = project(sheet_corners_world())
    mu1, mu2, y0 = frictions(params)
    meta = {"video": out.name, "fps": FPS, "size": [W, H], "synthetic": True, "renderer": "NVIDIA Newton SensorTiledCamera",
            "sheet": "a4", "sheet_corners_px": [[round(float(x), 1), round(float(y), 1)] for x, y in corners],
            "object_height_m": 2 * HALF, "hidden": {"mu_eff": MU_EFF, "patch_y0": y0, "patch_mu": mu2}, "pushes": truth,
            "camera": {"pos": CAM_POS, "look_at": CAM_LOOK, "fov_v_deg": math.degrees(FOV_V)}}
    out.with_suffix(".truth.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    if "--hard" in sys.argv:  # robustness check, not a Studio sample (written outside the repo's samples)
        m = render(Path(sys.argv[-1]), SPEEDS + SPEEDS_2, 5, hard=True)
        print("hard", len(m["pushes"]), "pushes ->", sys.argv[-1])
        sys.exit(0)
    for name, speeds, seed in (("flick-video", SPEEDS, 3), ("flick-video-2", SPEEDS_2, 4)):
        m = render(OUT / f"{name}.mp4", speeds, seed)
        print(name, json.dumps({k: m[k] for k in ("sheet_corners_px", "hidden")}), len(m["pushes"]), "pushes")
