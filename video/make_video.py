"""Build the demo video locally: dashboard screenshots + Newton clips + synthesized narration.

Numbers in the narration are read from runs/demo/bundle.json so the voice matches the screen.
Needs: Google Chrome (headless screenshots), ffmpeg, macOS `say`. The live "New run" scene needs
`make serve` running on localhost:8000 (skipped otherwise).

Run: .venv/bin/python -m video.capture && .venv/bin/python -m video.make_video  ->  video/out/tether_demo.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "video" / "out"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H, FPS = 1920, 1080, 30
GREEN, BG, FG, MUTED, RED = (118, 185, 0), (0, 0, 0), (238, 238, 238), (153, 153, 153), (226, 87, 76)
FONT_D = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"
FONT_B = "/System/Library/Fonts/SFNS.ttf"  # has arrows; Avenir lacks "→"
FONT_M = "/System/Library/Fonts/Menlo.ttc"
VOICE = "Samantha"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@dataclass
class Scene:
    name: str
    narration: str
    image: Path | None = None  # still image (1920x1080)
    frames: list[Path] | None = None  # animated scene (PNG sequence at FPS)
    min_s: float = 4.0
    zoom: bool = True  # slow push-in on stills


# ---------- sources ----------
def screenshot(url: str, out: Path, budget_ms: int = 5000) -> Path:
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1.2",
                    f"--virtual-time-budget={budget_ms}", "--window-size=1600,900", f"--screenshot={out}", url],
                   check=True, capture_output=True)
    return out


def crop_zoom(src: Path, box: tuple[int, int, int, int], out: Path) -> Path:
    """Crop a region of a 1920x1080 screenshot and fit it to 16:9 (for legible close-ups)."""
    im = Image.open(src).convert("RGB")
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w / h < W / H:  # widen to 16:9 around the box
        nw = int(h * W / H)
        x0 = max(0, min(im.width - nw, x0 - (nw - w) // 2)); x1 = x0 + nw
    else:
        nh = int(w * H / W)
        y0 = max(0, min(im.height - nh, y0 - (nh - h) // 2)); y1 = y0 + nh
    im.crop((x0, y0, x1, y1)).resize((W, H), Image.LANCZOS).save(out)
    return out


def card(lines: list[tuple[str, str, int, tuple]], out: Path, footer: str = "") -> Path:
    """Text card: lines of (text, font path, size, color), left aligned on black."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = 330
    for text, fp, size, color in lines:
        d.text((160, y), text, font=font(fp, size), fill=color)
        y += int(size * 1.35)
    d.rectangle((160, 300, 260, 306), fill=GREEN)
    if footer:
        d.text((160, H - 140), footer, font=font(FONT_M, 26), fill=MUTED)
    im.save(out)
    return out


def clip_pair(sim: Path, real: Path, labels: tuple[str, str], results: tuple[str, str], title: str, out_dir: Path) -> list[Path]:
    """Side-by-side animated clips (sim | real) with labels, as a PNG sequence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = Image.open(sim), Image.open(real)
    na, nb = getattr(a, "n_frames", 1), getattr(b, "n_frames", 1)
    n = max(na, nb) + FPS  # hold the last frame for a second
    cw, ch = 860, 484
    fd, fm, ft = font(FONT_D, 44), font(FONT_M, 30), font(FONT_D, 64)
    paths = []
    for i in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((80, 90), title, font=ft, fill=FG)
        for k, (src, nf, x, lab, res) in enumerate(((a, na, 80, labels[0], results[0]), (b, nb, 980, labels[1], results[1]))):
            src.seek(min(i, nf - 1))
            im.paste(src.convert("RGB").resize((cw, ch), Image.LANCZOS), (x, 250))
            d.rectangle((x, 250, x + cw, 250 + ch), outline=(51, 51, 51), width=2)
            d.text((x, 200), lab, font=fd, fill=MUTED)
            if i >= min(n - FPS, nf + 4):
                ok = "on line" in res
                d.text((x, 760), res, font=fm, fill=GREEN if ok else RED)
        p = out_dir / f"f{i:04d}.png"
        im.save(p)
        paths.append(p)
    return paths


# ---------- rendering ----------
def tts(text: str, out: Path) -> float:
    subprocess.run(["say", "-v", VOICE, "-r", "178", "-o", str(out), text], check=True)
    return duration(out)


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       check=True, capture_output=True, text=True)
    return float(r.stdout.strip())


def render_scene(sc: Scene, work: Path) -> Path:
    audio = work / f"{sc.name}.aiff"
    dur = max(sc.min_s, tts(sc.narration, audio) + 0.7)
    out = work / f"{sc.name}.mp4"
    common = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
              "-af", "apad", "-t", f"{dur:.2f}"]
    if sc.frames:
        seq = work / f"{sc.name}_seq"
        seq.mkdir(exist_ok=True)
        frames = list(sc.frames)
        need = int(dur * FPS)
        frames += [frames[-1]] * max(0, need - len(frames))
        for i, f in enumerate(frames[:need]):
            dst = seq / f"f{i:05d}.png"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(f)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(seq / "f%05d.png"), "-i", str(audio), *common, str(out)]
    else:
        n = int(dur * FPS)
        vf = (f"scale={W * 2}:{H * 2},zoompan=z='min(1+0.00025*on,1.05)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS}"
              if sc.zoom else f"scale={W}:{H}")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(sc.image), "-i", str(audio), "-vf", vf, *common, str(out)]
    subprocess.run(cmd, check=True)
    return out


def server_up(url: str = "http://localhost:8000/api/status") -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


# ---------- storyboard ----------
def newton_clip(video: Path, out_dir: Path, title: str, seconds: float = 7.0, start: float = 0.0) -> list[Path]:
    """Frames of a Newton-rendered sample video, letterboxed on black with a title (PNG sequence)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-t", f"{seconds}", "-i", str(video),
                    "-vf", f"fps={FPS},scale=1600:-1", str(raw / "r%04d.png")], check=True)
    ft, fm = font(FONT_D, 64), font(FONT_M, 28)
    paths = []
    for i, f in enumerate(sorted(raw.glob("r*.png"))):
        im = Image.new("RGB", (W, H), BG)
        fr = Image.open(f).convert("RGB")
        im.paste(fr, ((W - fr.width) // 2, 170))
        d = ImageDraw.Draw(im)
        d.text((160, 70), title, font=ft, fill=FG)
        d.text((160, H - 70), "Synthetic road rendered by NVIDIA Newton (SensorTiledCamera); Tether never sees its physics.", font=fm, fill=MUTED)
        p = out_dir / f"f{i:04d}.png"
        im.save(p)
        paths.append(p)
    return paths


def video_frames(video: Path, out_dir: Path) -> list[Path]:
    """A finished 1920x1080 clip as a PNG sequence (for scenes made by studio.rollout_video)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", f"fps={FPS},scale={W}:{H}", str(out_dir / "f%04d.png")], check=True)
    return sorted(out_dir.glob("f*.png"))


def build_scenes(work: Path, caps: Path) -> list[Scene]:
    """v2 (Tether): overview -> Studio on the driving domain -> Newton replay -> exports -> other domains -> console."""
    bundle = json.loads((ROOT / "runs/demo/bundle.json").read_text())
    runs = {r["id"]: r for r in bundle["runs"]}
    wm = runs["weak-motor"]
    meas = [e for e in wm["events"] if e["type"] == "measure"]
    m0, mN = meas[0], meas[-1]
    short_cm = abs(m0["clip"]["real_slide"] - m0["clip"]["target"]) * 100
    model = bundle["stack"]["diagnoser"]
    stop = json.loads((ROOT / "runs/studio-demo/stop-line.json").read_text())
    press = json.loads((ROOT / "runs/studio-demo/press-line.json").read_text())
    v = stop["verify"]
    v_cal, v_cur = v["rms_calibrated_m"] * 25, v["rms_current_m"] * 25  # driving runs at 1:25 (Froude); full-size metres
    pp = press["stages"][-1]["result"]["calibration"]["predicted"]
    pc = lambda x: int(x * 100 + 0.5)  # noqa: E731

    clips = ROOT / "runs/demo"
    res = lambda m, w: f"{(m['clip'][w + '_slide'] - m['clip']['target']) * 100:+.1f} cm " + (  # noqa: E731
        "on line" if abs(m["clip"][w + "_slide"] - m["clip"]["target"]) <= 0.03 else ("short" if m["clip"][w + "_slide"] < m["clip"]["target"] else "overshoot"))
    pair0 = clip_pair(clips / m0["clip"]["sim"], clips / m0["clip"]["real"], ("SIMULATOR", "REAL WORLD (HIDDEN PHYSICS)"),
                      (res(m0, "sim"), res(m0, "real")), "Same policy, same push. Different physics.", work / "pair0")
    road = newton_clip(ROOT / "studio/samples/stop-line-video.mp4", work / "road", "Autonomous vehicles: braking to a stop line", 7.0, 1.0)
    c = lambda n: caps / f"{n}.png"  # noqa: E731
    zoom = lambda n, box: crop_zoom(c(n), box, work / f"{n}_zoom.png")  # noqa: E731

    return [
        Scene("01_title", "Tether. Tie your simulator to the real world. Built with NVIDIA Newton, and Nemotron on Nebius Token Factory.",
              image=card([("Tether", FONT_D, 150, FG), ("Tie your simulator to the real world", FONT_B, 56, FG),
                          ("NVIDIA Newton  ·  Nemotron 3 Super on Nebius Token Factory", FONT_B, 40, GREEN)], work / "title.png",
                         footer="Nebius × NVIDIA Global AI Hackathon · Physical AI track"), min_s=6),
        Scene("02_problem", f"A policy trained in simulation pushes a cube onto the green line. In simulation it lands. "
                            f"In the real world, with physics nobody wrote down, it stops {short_cm:.0f} centimeters short. "
                            "Robots, cars and production lines all hit this gap.", frames=pair0, min_s=9),
        Scene("03_overview", "Tether reads how things really slide and stop, from a phone video or the logs a robot, a car or a line already writes. "
                             "On seven braking runs, the current simulator puts the car three point seven meters short on a wet section. "
                             "Tether's calibrated simulator lands on the real stops.", image=c("home_hero")),
        Scene("04_road", "Pick a domain. Here, autonomous driving: a car braking to a stop line, filmed from the roadside. "
                         "A painted box of known size gives the scale and the camera pose.", frames=road, min_s=7),
        Scene("05_measure", "Tether recovers the camera from the box, tracks every run with parallax correction, "
                            "and measures each launch speed and stopping distance.", image=zoom("studio_tracked", (255, 40, 1920, 990)), zoom=False),
        Scene("06_agent", f"Nemotron 3 Super works as a tool-using agent. It profiles the deceleration along the road, sees friction drop after about eight and a half meters, "
                          "fits hypotheses, and asks for longer runs. The numbers come from least squares, never from the language model, "
                          "and a library search cross-checks the agent's choice.", image=zoom("studio_diag", (270, 0, 1265, 560)), zoom=False, min_s=12),
        Scene("07_results", "The calibrated simulator: friction painted on the road, a ninety percent interval for every value, "
                            "ghost cars replaying each run in the old and the new simulator, and the runs that would settle the rest.",
              image=zoom("studio_results", (270, 60, 1920, 1000)), zoom=False),
        Scene("08_truth", "Revealing the hidden physics: road friction, the start of the wet section, and its friction all fall inside the intervals.",
              image=zoom("studio_truth", (270, 120, 1920, 900)), zoom=False),
        Scene("09_verify", f"Before export, every run is replayed in NVIDIA Newton with the exported physics. "
                           f"About {v_cal * 100:.0f} centimeters of stopping error at full scale, against {v_cur:.1f} meters for the current simulator.",
              image=zoom("studio_verify", (270, 80, 1920, 1000)), zoom=False, min_s=9),
        Scene("09b_retrain", "Does it matter for learning? Tether retrains the policy three ways in parallel NVIDIA Newton worlds: on the current simulator, "
                             "on wide domain randomization, and on Tether's measured ranges. Scored in the hidden real world: forty two, zero, and one hundred percent.",
              image=zoom("studio_retrain", (270, 60, 1920, 1000)), zoom=False, min_s=11),
        Scene("09c_rollout", "Here are the three policies braking in the hidden world, rendered by Newton. Only the Tether-trained car stops on the line every time.",
              frames=video_frames(ROOT / "video/out/rollout-brake-log.mp4", work / "rollout"), min_s=8),
        Scene("10_export", "Then take it home: NVIDIA Newton materials, Isaac Lab randomization over the measured intervals, "
                           "and for driving, CARLA tire friction with a friction trigger on the wet section.", image=c("studio_carla")),
        Scene("11_domains", f"One physics covers three jobs. In factory inspection, a pneumatic pusher on an oily rail goes from "
                            f"{pc(pp['before']['median'])} to {pc(pp['after']['median'])} percent predicted hits on the inspection window.",
              image=zoom("factory_results", (270, 60, 1920, 1000)), zoom=False),
        Scene("12_console", "The agent console shows why an agent: on the hardest open-world faults, fixed rules reach fifty four percent, "
                            "the Nemotron agent ninety six.", image=c("console_bench")),
        Scene("13_stack", "NVIDIA Newton simulates and renders on a laptop CPU. Nemotron reasons on Nebius Token Factory for about a cent per diagnosis. "
                          "Days of hand-tuning a simulator become one session.",
              image=card([("How it works", FONT_D, 96, FG),
                          ("data → measure → diagnose → calibrate → verify → retrain in Newton → export", FONT_B, 40, FG),
                          ("Physics + rendering:  NVIDIA Newton (Warp, CPU)", FONT_M, 34, MUTED),
                          (f"Agent:  {model} on Nebius Token Factory", FONT_M, 34, MUTED),
                          ("Exports:  NVIDIA Newton · Isaac Lab · CARLA", FONT_M, 34, MUTED)], work / "stack.png"), min_s=8),
        Scene("14_end", "Tether.", image=card([("Tether", FONT_D, 150, FG), ("Sim-to-real, tethered.", FONT_B, 48, MUTED)],
                                                 work / "end.png", footer="Apache-2.0 · not affiliated with or endorsed by NVIDIA or Nebius"), min_s=4),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "tether_demo.mp4")
    ap.add_argument("--caps", type=Path, default=OUT / "caps", help="screens from `python -m video.capture`")
    a = ap.parse_args()
    for tool in ("ffmpeg", "ffprobe", "say"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found")
    work = OUT / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    parts = []
    if not (a.caps / "home_hero.png").exists():
        raise SystemExit(f"no captures in {a.caps}: run `python -m video.capture` with `make serve` up")
    for sc in build_scenes(work, a.caps):
        p = render_scene(sc, work)
        print(f"{sc.name:12s} {duration(p):5.1f}s")
        parts.append(p)
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(a.out)], check=True)
    print(f"wrote {a.out} ({duration(a.out):.1f}s, {a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
