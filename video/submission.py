"""Everything the submission uploads, in one folder: the film, a YouTube thumbnail, Devpost gallery stills (3:2), the
YouTube description with chapters, and the "About the project" text cut from docs/submission/DEVPOST.md.

Run after `make film`: .venv/bin/python -m video.submission   -> submission/ (gitignored)
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from video import film

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"
LIVE = "https://tether-454741001655.us-central1.run.app"
REPO = "https://github.com/men16922/tether"
CHAPTERS = {"cold": "The Sim2Real gap", "title": "Tether", "appInput": "Studio: bring a video or a log",
            "appAgent": "Nemotron on Nebius Token Factory investigates", "appResult": "Your simulator, corrected",
            "retrain": "Retraining in parallel NVIDIA Newton worlds", "appExport": "Checked in Newton, exported",
            "appAsk": "Ask Tether", "proofReal": "Real objects, independent friction", "proofEngine": "Worlds from a different engine",
            "honest": "The AI proposes, the data decides", "close": "Try it"}
# gallery stills: (scene, fraction of the scene) -> file name
STILLS = [("title", 0.8, "01-tether"), ("gap", 0.75, "02-the-gap"), ("appAgent", 0.85, "03-nemotron-investigates"),
          ("appResult", 0.4, "04-calibrated-simulator"), ("retrain", 0.95, "05-retrained-hidden-world"),
          ("appExport", 0.6, "06-newton-replay-export"), ("appAsk", 0.92, "07-ask-tether"),
          ("proofReal", 0.9, "08-real-objects"), ("proofEngine", 0.9, "09-different-engine")]


def starts() -> dict[str, tuple[float, float]]:
    plan = json.loads((film.WORK / "plan.json").read_text())
    out, t = {}, 0.0
    for name, _ in plan:
        d = len(list((film.WORK / "frames" / name).glob("f*.jpg"))) / film.FPS  # exact length as encoded
        out[name] = (t, d)
        t += d
    return out


def stamp(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def thumbnail() -> None:
    page = film.WORK / "stage.html"
    asyncio.run(film.capture(page, [("thumb", 2.0)], "thumb"))
    still = Image.open(film.WORK / "frames" / "thumb" / "f00045.jpg").convert("RGB")  # t = 1.5 s, past the fade-in
    still.resize((1280, 720), Image.LANCZOS).save(OUT / "youtube-thumbnail.png")
    padded(still).save(OUT / "gallery" / "00-cover.png")


def padded(im: Image.Image) -> Image.Image:
    """16:9 -> 3:2 on black (Devpost shows gallery images at 3:2)."""
    w, h = im.size
    canvas = Image.new("RGB", (w, w * 2 // 3), (0, 0, 0))
    canvas.paste(im, (0, (canvas.height - h) // 2))
    return canvas


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "gallery").mkdir(exist_ok=True)
    movie = film.OUT / "tether_film.mp4"
    if not movie.exists():
        raise SystemExit("run `make film` first")
    shutil.copyfile(movie, OUT / "tether_film.mp4")
    data = film.build_data()
    (film.WORK / "stage.html").write_text((Path(film.__file__).parent / "stage.html").read_text().replace("/*__DATA__*/null", json.dumps(data), 1))
    thumbnail()

    sc = starts()
    for name, frac, fname in STILLS:
        t0, d = sc[name]
        tmp = OUT / "gallery" / f"{fname}.tmp.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t0 + d * frac:.2f}", "-i", str(movie), "-frames:v", "1", str(tmp)], check=True)
        padded(Image.open(tmp).convert("RGB")).save(OUT / "gallery" / f"{fname}.png")
        tmp.unlink()

    total = sum(d for _, d in sc.values())
    chapters = "\n".join(f"{stamp(sc[n][0])} {label}" for n, label in CHAPTERS.items() if n in sc)
    (OUT / "youtube-description.txt").write_text(f"""Tether ties your simulator to the real world. From a phone video or the log a robot, car or production line
already writes, it finds which physics the simulator gets wrong, says how sure it is, checks the fix by replaying
every run in NVIDIA Newton, retrains the policy in parallel Newton worlds, and exports configs for NVIDIA Newton,
Isaac Lab and CARLA. The reasoning agent is NVIDIA Nemotron 3 Super running on Nebius Token Factory.

Try it: {LIVE}
Code (Apache-2.0): {REPO}

{chapters}

Built for the Nebius x NVIDIA Global AI Hackathon (Physical AI track).
Real-object footage: EV-RealPhys (Kandukuri, Strecke, Stueckler 2023, Max Planck Institute), CC BY-SA 4.0.
Narration: ElevenLabs. Music: generated for this film. Not affiliated with or endorsed by NVIDIA or Nebius.
""")
    devpost = (ROOT / "docs" / "submission" / "DEVPOST.md").read_text()
    about = devpost[devpost.index("## Inspiration"):devpost.index("## Built with")].rstrip() + "\n"
    (OUT / "about-the-project.md").write_text(about)
    shutil.copyfile(ROOT / "docs" / "submission" / "SUBMIT.md", OUT / "SUBMIT.md")
    print(f"submission/: film {total:.0f} s, thumbnail, {len(STILLS) + 1} gallery images, youtube-description.txt, about-the-project.md, SUBMIT.md")


if __name__ == "__main__":
    main()
