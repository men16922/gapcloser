"""Build the demo video locally: dashboard screenshots + Newton clips + synthesized narration.

Numbers in the narration are read from runs/demo/bundle.json so the voice matches the screen.
Needs: Google Chrome (headless screenshots), ffmpeg, macOS `say`. The live "New run" scene needs
`make serve` running on localhost:8000 (skipped otherwise).

Run: .venv/bin/python -m video.make_video  ->  video/out/gapcloser_demo.mp4
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
def build_scenes(work: Path) -> list[Scene]:
    bundle = json.loads((ROOT / "runs/demo/bundle.json").read_text())
    runs = {r["id"]: r for r in bundle["runs"]}
    wm = runs["weak-motor"]
    ev = wm["events"]
    meas = [e for e in ev if e["type"] == "measure"]
    diag = next(e for e in ev if e["type"] == "diagnose")
    plan = next(e for e in ev if e["type"] == "plan")
    m0, mN = meas[0], meas[-1]
    n = len(m0["real"]["trials"])
    ok0 = sum(t["success"] for t in m0["real"]["trials"])
    okN = sum(t["success"] for t in mN["real"]["trials"])
    sim0 = sum(t["success"] for t in m0["sim"]["trials"])
    short_cm = abs(m0["clip"]["real_slide"] - m0["clip"]["target"]) * 100
    gain_est = next(s["estimate"] for s in diag["suspects"] if s["name"] == "actuator_gain")
    truth = wm["truth"]["actuator_gain"]["hidden"]
    sm = bundle["benchmark"]["summary"]
    pc = lambda x: int(x * 100 + 0.5)  # noqa: E731 - same rounding as the dashboard (JS Math.round)
    dr, nom = pc(sm["full_dr"]["mean_success"]), pc(sm["nominal"]["mean_success"])
    best = pc(max(sm[k]["mean_success"] for k in sm if k.startswith("gapcloser")))
    llm = pc(sm.get("gapcloser_llm", sm["gapcloser_traj"])["mean_success"])
    model = bundle["stack"]["diagnoser"]
    model_short = "Nemotron 3 Super" if "super" in model.lower() else "Nemotron"
    words = lambda x: f"{x:.2f}".replace(".", " point ")  # noqa: E731 - "0 point 76" reads better than "0.76"

    dash = (ROOT / "dashboard/dist/gapcloser.standalone.html").as_uri()
    shot = lambda name, h: screenshot(f"{dash}#{h}", work / f"{name}.png")  # noqa: E731
    n_ev = len(ev)
    i_diag = ev.index(diag) + 1
    i_plan = ev.index(plan) + 1

    s_measure = shot("measure", f"weak-motor.s{ev.index(m0) + 1}.i0")
    s_diag = shot("diag", f"weak-motor.s{i_diag}.i0")
    s_plan = shot("plan", f"weak-motor.s{i_plan}.i0")
    s_done = shot("done", f"weak-motor.s{n_ev}")
    s_reveal = shot("reveal", f"weak-motor.s{n_ev}.reveal")
    s_bench = shot("bench", "benchmark")

    clips = ROOT / "runs/demo"
    res = lambda m, w: f"{(m['clip'][w + '_slide'] - m['clip']['target']) * 100:+.1f} cm " + (  # noqa: E731
        "on line" if abs(m["clip"][w + "_slide"] - m["clip"]["target"]) <= 0.03 else ("short" if m["clip"][w + "_slide"] < m["clip"]["target"] else "overshoot"))
    pair0 = clip_pair(clips / m0["clip"]["sim"], clips / m0["clip"]["real"], ("SIMULATOR · WHAT THE AGENT EXPECTS", "REAL · HIDDEN PHYSICS"),
                      (res(m0, "sim"), res(m0, "real")), "Same policy, same push. Different physics.", work / "pair0")
    pairN = clip_pair(clips / mN["clip"]["sim"], clips / mN["clip"]["real"], ("SIMULATOR · AFTER ONE FIX", "REAL · HIDDEN PHYSICS"),
                      (res(mN, "sim"), res(mN, "real")), "After GapCloser", work / "pairN")

    scenes = [
        Scene("01_title", f"GapCloser. An agent that closes the sim-to-real gap, built with NVIDIA Newton and {model_short} on Nebius Token Factory.",
              image=card([("GapCloser", FONT_D, 150, FG), ("An agent that closes the Sim2Real gap", FONT_B, 56, FG),
                          ("NVIDIA Newton  ·  Nemotron on Nebius Token Factory", FONT_B, 40, GREEN)], work / "title.png",
                         footer="Nebius × NVIDIA Global AI Hackathon · Physical AI track"), min_s=6),
        Scene("02_problem", f"A robot arm pushes a cube onto the green line. In the simulator, the push lands on target. "
                            f"In the real world, with physics the agent cannot see, it stops {short_cm:.0f} centimeters short.", frames=pair0, min_s=9),
        Scene("03_measure", f"GapCloser trains a policy in its simulator, then measures it in the hidden world. "
                            f"{ok0} of {n} pushes land on target, while the simulator predicted {sim0} of {n}.", image=s_measure),
        Scene("04_diagnose", f"{model_short} reads the evidence. The tracked launch speed is only {pc(gain_est)} percent of the simulator's, "
                             "while the deceleration is unchanged. So the motor is weak. It is not the friction.",
              image=crop_zoom(s_diag, (70, 440, 1380, 1080), work / "diag_zoom.png"), zoom=False),
        Scene("05_plan", f"The planner turns that diagnosis into a simulator change: actuator gain, {words(gain_est)}.",
              image=crop_zoom(s_plan, (70, 440, 1380, 1080), work / "plan_zoom.png"), zoom=False),
        Scene("06_done", f"The agent retrains and measures again. {okN} of {n}. One fix closed the gap.", image=s_done),
        Scene("07_after", "Same commands, now landing on the line.", frames=pairN, min_s=5),
        Scene("08_reveal", f"Revealing the hidden truth: the real motor gain was {words(truth)}. "
                           f"An agent that only looks at where the cube stopped blames friction, and stays at "
                           f"{pc(wm['baselines']['outcome_only'])} percent.", image=s_reveal),
        Scene("09_bench", f"Across ten random hidden worlds, domain randomization reaches {dr} percent and the nominal simulator {nom}. "
                          f"GapCloser reaches up to {best} percent, {llm} with {model_short} as the diagnoser, and the first diagnosis names the true cause in every world.",
              image=s_bench),
    ]
    if server_up():
        scenes.append(Scene("10_live", "Anyone can try it. Hide up to three physical parameters, and watch the agent find them, live.",
                            image=screenshot("http://localhost:8000/#new", work / "live.png")))
    scenes += [
        Scene("11_stack", "NVIDIA Newton simulates and renders on a laptop CPU. Nemotron reasons on Nebius Token Factory, "
                          "at about a tenth of a cent per diagnosis. Days of manual simulator tuning become one agent loop.",
              image=card([("How it works", FONT_D, 96, FG),
                          ("Train in sim  →  measure in hidden world  →  diagnose  →  fix sim  →  repeat", FONT_B, 42, FG),
                          ("Physics + rendering:  NVIDIA Newton 1.6 (CPU)", FONT_M, 34, MUTED),
                          (f"Diagnosis:  {model}", FONT_M, 34, MUTED),
                          ("Cost:  ≈ $0.0013 per diagnosis", FONT_M, 34, MUTED)], work / "stack.png"), min_s=8),
        Scene("12_end", "GapCloser.", image=card([("GapCloser", FONT_D, 150, FG), ("Closing the Sim2Real gap, one agent loop at a time.", FONT_B, 48, MUTED)],
                                                 work / "end.png", footer="Apache-2.0 · not affiliated with or endorsed by NVIDIA or Nebius"), min_s=4),
    ]
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "gapcloser_demo.mp4")
    a = ap.parse_args()
    for tool in ("ffmpeg", "ffprobe", "say"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found")
    work = OUT / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    parts = []
    for sc in build_scenes(work):
        p = render_scene(sc, work)
        print(f"{sc.name:12s} {duration(p):5.1f}s")
        parts.append(p)
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(a.out)], check=True)
    print(f"wrote {a.out} ({duration(a.out):.1f}s, {a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
