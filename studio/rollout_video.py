"""Before / after, in pictures: the policies from "Retrain and test" acting in the hidden real world, rendered by
NVIDIA Newton side by side (your current simulator | wide randomization | Tether's measured ranges).

Each column replays the same targets from the same start points; only the policy differs. The policy reads the
target through the hidden world's own camera and its command passes through the hidden actuator, so what you see
is what the success rate measured.

Run: .venv/bin/python -m studio.rollout_video stop-line  ->  video/out/rollout-stop-line.mp4
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from studio.fonts import font

from sim.push_task import SUCCESS_TOL, observe
from studio.session import load
from studio.train import TablePolicy, hidden_params

ROOT = Path(__file__).resolve().parent.parent
TARGETS = (0.28, 0.40, 0.50, 0.58)
COLS = (("current", "Your current simulator"), ("wide", "Wide domain randomization"), ("tether", "Tether's measured ranges"))
UNITS = {"robot": (1.0, "cm"), "factory": (1.0, "cm"), "driving": (25.0, "m")}


def rollout_video(sample: str, out: Path) -> dict:
    """From a recorded Studio run (runs/studio-demo/<sample>.json)."""
    rec = json.loads((ROOT / "runs/studio-demo" / f"{sample}.json").read_text())
    st = rec["stages"][-1]
    s = load(json.dumps(st["session"]), "x.json", st["session"]["name"])
    return render_rollout(s, rec["truth"]["model"], rec["training"], rec["sample"].get("domain", "robot"), out)


def render_rollout(s, truth_model: dict, training: dict, domain: str, out: Path) -> dict:
    """The three trained policies (training = studio.train.train output) acting in the hidden world."""
    from studio.video_sample import render

    rec = {"training": training, "truth": {"model": truth_model}}
    hidden = hidden_params(truth_model, s)
    if not s.has_commands:
        hidden = hidden.with_(actuator_gain=1.0)
    world = {"mu_eff": rec["truth"]["model"]["mu_eff"], "patch_y0": hidden["patch_y0"] if hidden["patch_y0"] < 1 else None,
             "patch_mu": hidden["patch_mu"]}
    tmp = Path(tempfile.mkdtemp())
    cols = []
    for cond, label in COLS:
        pol = TablePolicy(rec["training"]["conditions"][cond]["policy"])
        speeds = [pol.command(observe(t, hidden)) * hidden["actuator_gain"] for t in TARGETS]
        meta = render(tmp / f"{cond}.mp4", speeds, 7, world=world, show_strip=False, domain=domain, targets=list(TARGETS))
        hits = [abs(p["slide_m"] + p["start_y_m"] - t) <= SUCCESS_TOL for p, t in zip(meta["pushes"], TARGETS)]
        cols.append((cond, label, tmp / f"{cond}.mp4", meta, hits))
    # frames side by side with labels and per-run verdicts
    for cond, *_ in cols:
        (tmp / cond).mkdir()
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(tmp / f"{cond}.mp4"), "-vf", "crop=720:380:150:90,scale=620:-1", str(tmp / cond / "f%05d.png")], check=True)
    counts = {c: len(list((tmp / c).glob("*.png"))) for c, *_ in cols}
    n = max(counts.values()) + 30  # columns run at their own pace; a finished column holds its last frame
    scale, unit = UNITS.get(domain, (1.0, "cm"))
    fd = font("display", 34)
    fm = font("mono", 20)
    W, H = 1920, 1080
    frames = tmp / "out"
    frames.mkdir()
    rate = rec["training"]["conditions"]
    for i in range(n):
        im = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(im)
        for k, (cond, label, _, meta, hits) in enumerate(cols):
            x = 20 + k * 633
            fr = Image.open(tmp / cond / f"f{min(i, counts[cond] - 1) + 1:05d}.png").convert("RGB")
            im.paste(fr, (x, 300))
            col = (118, 185, 0) if cond == "tether" else (238, 238, 238)
            d.text((x, 220), label, font=fd, fill=col)
            d.text((x, 262), f"hidden-world success {round(rate[cond]['final_real'] * 100)}%", font=fm, fill=(153, 153, 153))
            # which push is on screen, and how it ended once it has stopped
            pi = max([j for j, p in enumerate(meta["pushes"]) if p["launch_frame"] <= i] or [0])
            p = meta["pushes"][pi]
            end = (p["slide_m"] + p["start_y_m"] - TARGETS[pi]) * scale
            nxt = meta["pushes"][pi + 1]["launch_frame"] if pi + 1 < len(meta["pushes"]) else counts[cond]
            done = i >= nxt - 14  # the push has come to rest (the renderer holds 14 frames after rest)
            txt = f"run {pi + 1}: target {TARGETS[pi] * scale:.2f} m" + (f"  ->  {end * (100 if unit == 'cm' else 1):+.1f} {unit}" if done else "")
            d.text((x, 300 + fr.height + 16), txt, font=fm, fill=((118, 185, 0) if hits[pi] else (226, 87, 76)) if done else (153, 153, 153))
        d.text((20, 60), "Same hidden world, same targets. Only the training differs. (0.6x)", font=font("display", 60), fill=(238, 238, 238))
        d.text((20, H - 60), "Rendered by NVIDIA Newton. Policies from Tether Studio's Retrain step; the hidden world never trained them.",
               font=fm, fill=(105, 113, 122))
        im.save(frames / f"f{i:05d}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "18", "-i", str(frames / "f%05d.png"), "-r", "30", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)  # 0.6x slow motion
    return {"out": str(out), "hits": {c: h for c, _, _, _, h in cols}}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "brake-log"
    print(rollout_video(name, ROOT / "video" / "out" / f"rollout-{name}.mp4"))
