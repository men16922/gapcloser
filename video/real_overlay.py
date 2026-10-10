"""A real slide with Tether's measurement drawn on it, for the demo video: one public iPhone slow-motion clip (IDPP real
split, Apache-2.0; `make real-check` downloads it), the tracked path growing as the object slides, and next to it the
distance still to go with the constant-deceleration (Coulomb) fit that Tether's model assumes.

Run: .venv/bin/python -m video.real_overlay [clip-name]   -> video/out/real-overlay.mp4 (1920x1080)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from eval.real_friction import read, rest_blob
from studio.fonts import font
from studio.video import track

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "video" / "out" / "real-overlay.mp4"
CLIP = "paper-box_kitchen-paper_c2_4"
W, H = 1920, 1080
BG, FG, MUTED, GREEN, AMBER = (0, 0, 0), (238, 238, 238), (163, 163, 163), (118, 185, 0), (230, 162, 60)


def measure(path: Path) -> dict:
    frames = read(path)
    x, y, size = rest_blob(frames)
    uv, _ = track(frames[::-1], (x, y), int(np.clip(size * 1.1, 20, 220)))
    uv = np.asarray(uv)[::-1]
    pad = np.pad(uv, ((2, 2), (0, 0)), mode="edge")
    uv = np.stack([np.median(pad[i:i + 5], axis=0) for i in range(len(uv))])
    speed = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    moving = np.flatnonzero(speed > 0.6)
    t_stop = int(moving[-1]) + 1
    sm = np.convolve(speed, np.ones(3) / 3, mode="same")
    a0 = int(np.argmax(sm[:t_stop])) + 2
    disp = uv - uv[-1]
    axis = disp[a0] / (np.linalg.norm(disp[a0]) + 1e-9)
    s = disp[a0:t_stop + 1] @ axis / size
    t = np.arange(len(s), dtype=float)
    A = np.stack([np.ones_like(t), t, 0.5 * t * t], axis=1)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    pred = A @ coef
    r2 = 1 - float(np.sum((s - pred) ** 2) / np.sum((s - s.mean()) ** 2))
    return {"frames": frames, "uv": uv, "a0": a0, "t_stop": t_stop, "s": s, "coef": coef, "r2": r2}


def render(name: str = CLIP, out: Path = OUT) -> Path:
    src = next((ROOT / "runs" / "real_data").rglob(f"{name}_rgb.mp4"), None)
    if src is None:
        raise SystemExit("IDPP clips not found: run `make real-check` first (downloads ~310 MB to runs/real_data)")
    m = measure(src)
    frames, uv, a0, t_stop, s, coef = m["frames"], m["uv"], m["a0"], m["t_stop"], m["s"], m["coef"]
    fh, fw = frames[0].shape[:2]
    scale = min(1100 / fw, 860 / fh)
    vw, vh = int(fw * scale), int(fh * scale)
    ox, oy = 80, 150
    cx, cy, cw, ch = 1260, 260, 580, 520  # chart box
    ft, fb, fs = font("display", 54), font("body", 30), font("mono", 22)
    t_all = np.arange(len(s), dtype=float)
    fit = np.stack([np.ones_like(t_all), t_all, 0.5 * t_all * t_all], axis=1) @ coef
    smax = float(max(s.max(), fit.max()))
    X = lambda k: cx + 40 + (cw - 60) * k / max(1, len(s) - 1)  # noqa: E731
    Y = lambda v: cy + ch - 50 - (ch - 90) * max(0.0, v) / smax  # noqa: E731
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        n = len(frames)
        for i in range(n + 30):  # hold the last frame for a second
            k = min(i, n - 1)
            im = Image.new("RGB", (W, H), BG)
            fr = Image.fromarray(frames[k][:, :, ::-1]).resize((vw, vh))
            im.paste(fr, (ox, oy))
            d = ImageDraw.Draw(im)
            d.text((ox, 60), "Real footage: an object sliding to a stop", font=ft, fill=FG)
            pts = [(ox + p[0] * scale, oy + p[1] * scale) for p in uv[a0:min(k, t_stop) + 1]]
            if len(pts) > 1:
                d.line(pts, fill=GREEN, width=5)
            if k >= a0:
                q = uv[min(k, len(uv) - 1)]
                r = 14
                d.ellipse([ox + q[0] * scale - r, oy + q[1] * scale - r, ox + q[0] * scale + r, oy + q[1] * scale + r], outline=AMBER, width=4)
            # chart: distance still to go (object sizes) vs frame, measured dots and the Coulomb fit
            d.rectangle([cx, cy, cx + cw, cy + ch], outline=(51, 51, 51), width=2)
            d.text((cx, cy - 46), "Distance still to go", font=fb, fill=FG)
            d.line([(X(j), Y(fit[j])) for j in range(len(fit))], fill=(110, 110, 110), width=3)
            upto = max(0, min(k, t_stop) - a0)
            for j in range(min(upto + 1, len(s))):
                d.ellipse([X(j) - 4, Y(s[j]) - 4, X(j) + 4, Y(s[j]) + 4], fill=GREEN)
            d.text((cx + 40, cy + ch - 38), "time →", font=fs, fill=MUTED)
            d.text((cx, cy + ch + 24), "green: tracked by Tether", font=fs, fill=GREEN)
            d.text((cx, cy + ch + 56), "grey: constant deceleration (dry Coulomb)", font=fs, fill=MUTED)
            d.text((cx, cy + ch + 88), f"R² {m['r2']:.4f}", font=fs, fill=FG)
            d.text((ox, H - 104), "IDPP real split (Ma et al. 2025, Apache-2.0): iPhone slow motion on kitchen paper.", font=fs, fill=MUTED)
            d.text((ox, H - 70), "Coulomb sliding is the best of three friction laws on 36 of 45 such slides.", font=fs, fill=MUTED)
            im.save(tmp / f"f{i:05d}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(tmp / "f%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
    return out


if __name__ == "__main__":
    print(render(sys.argv[1] if len(sys.argv) > 1 else CLIP))
