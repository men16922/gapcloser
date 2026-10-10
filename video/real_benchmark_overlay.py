"""Real objects for the demo video: one EV-RealPhys clip (a mug pushed across a real table, 30 Hz) with Tether's tracked
path drawn on it (green) next to the motion-capture path (white), and beside it the friction Tether found for all
five objects against the tilt test measured separately (runs/proof/real_benchmark.json).

Data: EV-RealPhys (Kandukuri, Strecke, Stueckler 2023, MPI, CC BY-SA 4.0); `make real-benchmark` downloads it.
Run: .venv/bin/python -m video.real_benchmark_overlay [split/sequence]   -> video/out/real-benchmark.mp4 (1920x1080)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from eval.real_benchmark import DATA, load
from studio.fonts import font
from studio.video import track

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "video" / "out" / "real-benchmark.mp4"
SEQ = "val_sliding/000015"
W, H = 1920, 1080
BG, FG, MUTED, DIM, GREEN, AMBER = (0, 0, 0), (238, 238, 238), (163, 163, 163), (110, 110, 110), (118, 185, 0), (230, 162, 60)


def pixel_paths(d: dict, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(Tether's tracked pixels, motion capture projected into the image) per frame."""
    c = d["cam"][-1]
    K, R, t = np.asarray(c["cam_K"]).reshape(3, 3), np.asarray(c["cam_R_w2c"]).reshape(3, 3), np.asarray(c["cam_t_w2c"]) / 1000.0

    def proj(w):
        q = K @ (R @ w + t)
        return q[:2] / q[2]

    rest = d["com"][-1]
    x, y = proj(rest)
    size = float(np.linalg.norm(proj(rest + [0.05, 0, 0]) - proj(rest - [0.05, 0, 0])))
    uv, _ = track(frames[::-1], (x, y), int(np.clip(size * 1.1, 20, 220)))
    uv = np.asarray(uv)[::-1]
    pad = np.pad(uv, ((2, 2), (0, 0)), mode="edge")
    uv = np.stack([np.median(pad[i:i + 5], axis=0) for i in range(len(uv))])
    mocap = np.array([proj(p) for p in d["com"]])
    return uv, mocap


def render(seq: str = SEQ, out: Path = OUT) -> Path:
    import cv2

    path = DATA / "ev-realphys" / seq
    if not path.exists():
        raise SystemExit("EV-RealPhys not found: run `make real-benchmark` first")
    res = json.loads((ROOT / "runs" / "proof" / "real_benchmark.json").read_text())
    d = load(path)
    files = sorted((path / "rgb").glob("*.png"))
    frames = [cv2.imread(str(f)) for f in files][: d["n"]]
    uv, mocap = pixel_paths(d, frames)
    # Tether measures from release (the hand and ruler let go) to rest: draw its path over that stretch only
    from eval.real_benchmark import push_from, video_xy

    xy, _ = video_xy(d)
    _, seg = push_from(xy)
    rel = seg[0] if seg else 0
    fh, fw = frames[0].shape[:2]
    vw = 1120
    s = vw / fw
    vh = int(fh * s)
    ox, oy = 64, 190
    ft, fb, fm, fs = font("display", 56), font("body", 30), font("mono", 28), font("mono", 22)
    rows = sorted(res["objects"], key=lambda o: o["mu_tilt_test"])
    sm = res["summary"]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        n = len(frames)
        k_out = 0
        for i in range(2 * n + 45):  # half speed, then hold
            k = min(i // 2, n - 1)
            im = Image.new("RGB", (W, H), BG)
            fr = Image.fromarray(frames[k][:, :, ::-1]).resize((vw, vh))
            im.paste(fr, (ox, oy))
            dr = ImageDraw.Draw(im)
            dr.text((ox, 70), "Real objects: friction measured separately", font=ft, fill=FG)
            dr.text((ox, 142), "a mug pushed across a real table, 30 Hz video (EV-RealPhys benchmark)", font=fb, fill=MUTED)
            mc = [(ox + p[0] * s, oy + p[1] * s) for p in mocap[: k + 1]]
            for q in mc:
                dr.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=FG)
            tv = [(ox + p[0] * s, oy + p[1] * s) for p in uv[rel: k + 1]]
            if len(tv) > 1:
                dr.line(tv, fill=GREEN, width=5)
            if tv:
                q = tv[-1]
                dr.ellipse([q[0] - 14, q[1] - 14, q[0] + 14, q[1] + 14], outline=AMBER, width=4)
            dr.text((ox, oy + vh + 18), "green: Tether's tracker from release (RGB only)    white: motion capture", font=fs, fill=MUTED)
            # the five objects
            x0, y0 = 1250, 210
            dr.text((x0, y0), "object", font=fs, fill=DIM)
            dr.text((x0 + 300, y0), "tilt test", font=fs, fill=DIM)
            dr.text((x0 + 460, y0), "Tether", font=fs, fill=DIM)
            for j, o in enumerate(rows):
                y = y0 + 52 + j * 62
                ok = abs(o["video"]["mu"] - o["mu_tilt_test"]) <= 0.05
                dr.text((x0, y), o["object"], font=fm, fill=FG)
                dr.text((x0 + 300, y), f"{o['mu_tilt_test']:.3f}", font=fm, fill=FG)
                dr.text((x0 + 460, y), f"{o['video']['mu']:.3f}", font=fm, fill=GREEN if ok else AMBER)
            yb = y0 + 52 + len(rows) * 62 + 30
            dr.text((x0, yb), f"{sm['video']['within_0_05']} of {sm['video']['objects']} within ±0.05", font=fb, fill=GREEN)
            dr.text((x0, yb + 48), f"mean error {sm['video']['mean_abs_error']:.3f}", font=fb, fill=FG)
            dr.text((x0, yb + 92), f"paper's own estimator: {res['paper_estimator_on_real']['mean_abs_error']:.3f}", font=fb, fill=MUTED)
            dr.text((ox, H - 104), "EV-RealPhys: Kandukuri, Strecke, Stueckler 2023 (Max Planck Institute), CC BY-SA 4.0.", font=fs, fill=DIM)
            dr.text((ox, H - 70), "Tilt-test friction from the paper, never shown to Tether.", font=fs, fill=DIM)
            im.save(tmp / f"f{k_out:05d}.png")
            k_out += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(tmp / "f%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
    return out


if __name__ == "__main__":
    print(render(sys.argv[1] if len(sys.argv) > 1 else SEQ))
