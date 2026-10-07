"""CosmosEyes spike: turn a short Newton push clip into physical events with NVIDIA Cosmos Reason 2 (8B).

Runs fully local against a llama.cpp `llama-server` hosting the community GGUF + vision projector
(see README.md). No paid API is called.

    # one clip -> events JSON
    python spike/cosmos/classify.py spike/cosmos/clips/tip-c30-side.webp
    # evaluate on the labeled set (writes spike/cosmos/results/<tag>.jsonl and prints accuracy)
    python spike/cosmos/classify.py --labels spike/cosmos/clips/labels.json --mode frames --n 8 --crop track

Modes: frames = N frames as separate images (with timestamps in the text), grid = one contact
sheet image, video = the clip re-encoded as mp4 and sent as `video_url` (llama.cpp decodes it with
ffmpeg at --video-fps). Needs only Pillow, numpy, scipy, requests (the repo .venv has them).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image

SYSTEM = "You are a helpful assistant."

QUESTION = (
    "{media_intro} It shows a small light-grey cube on a dark checkered table. The cube is pushed and "
    "moves to the right across the table. Watch the cube's ORIENTATION frame by frame. If in any frame "
    "the cube is rotated, rolling onto another face, or seen as a tilted diamond, it TIPPED over. If it "
    "keeps the same upright square appearance while translating, it SLID. If it barely moves, it "
    "STALLED. If it leaves the table surface and comes back down, it BOUNCED.\n"
    "List only the moments when something physically changes (at most 4 events, not one per frame), "
    "then give the overall outcome. Output strict JSON on one line in exactly this schema:\n"
    '{{"events":[{{"t":<seconds>,"type":"slid|tipped|bounced|stalled"}}],"outcome":"slid|tipped|stalled"}}'
)
# Variant "tilt": per-frame perception only; event logic (first tilted frame -> "tipped") lives in code.
TILT_QUESTION = (
    "{media_intro} Each frame is a close-up crop centred on a small light-grey cube on a dark checkered "
    "table. For EACH frame decide whether the cube sits flat and upright (a face on the table, its "
    "outline an axis-aligned square/box) or is TILTED (rotated, balanced on an edge or corner, its "
    "outline a diamond or skewed box).\n"
    "Output strict JSON on one line: "
    '{{"frames":[{{"i":<frame number>,"tilted":true|false}}]}}'
)
THINK = (
    "\nAnswer the question using the following format:\n<think>\nYour reasoning.\n</think>\n\n"
    "Write your final answer immediately after the </think> tag."
)
NO_THINK = "\nReply with the JSON only."


# ------------------------------------------------------------------------------------------ frames
def load_frames(clip: Path) -> tuple[list[Image.Image], list[float]]:
    """All frames of an animated WebP/GIF plus their start time (s), honoring per-frame durations."""
    im = Image.open(clip)
    frames, times, t = [], [], 0.0
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        frames.append(im.convert("RGB"))
        times.append(t)
        t += (im.info.get("duration") or 33) / 1000.0
    return frames, times


def sample(frames, times, n: int, by_index: bool = False):
    """n frames evenly spaced in TIME (WebP merges identical frames, so index spacing is uneven),
    or evenly by INDEX (used inside the motion window, where every stored frame is a distinct pose)."""
    if by_index:
        picks = sorted({round(k * (len(frames) - 1) / max(1, n - 1)) for k in range(n)})
        return [frames[i] for i in picks], [times[i] for i in picks]
    total = times[-1]
    picks = []
    for k in range(n):
        target = total * k / max(1, n - 1)
        picks.append(int(np.argmin([abs(t - target) for t in times])))
    picks = sorted(set(picks))
    return [frames[i] for i in picks], [times[i] for i in picks]


# Camera presets copied from sim/newton_push.py (render_trial = "side", render_trial_arm = "arm"):
# (position, pitch-down deg, vertical fov deg). Used to crop to the task workspace, because the cube
# is only ~10 px wide in the 480x270 render.
CAMERAS = {"side": ((0.85, 0.27, 0.30), 17.0, 44.0), "arm": ((1.55, -0.02, 0.78), 24.0, 46.0)}
WORKSPACE = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (-0.06, 1.0) for z in (0.0, 0.07)])


def project(points: np.ndarray, view: str, W: int = 480, H: int = 270) -> np.ndarray:
    """World -> pixel for the Newton pinhole camera (looks along world -X, Z up, pitched down)."""
    from scipy.spatial.transform import Rotation

    pos, pitch, fov = CAMERAS[view]
    R = (Rotation.from_quat([0.5, 0.5, 0.5, 0.5]) * Rotation.from_rotvec([-np.radians(pitch), 0, 0])).as_matrix()
    pc = (points - np.asarray(pos)) @ R
    f = (H / 2) / np.tan(np.radians(fov) / 2)
    return np.stack([W / 2 + f * pc[:, 0] / -pc[:, 2], H / 2 - f * pc[:, 1] / -pc[:, 2]], 1)


def detect_view(frame: Image.Image) -> str:
    """'arm' if the big bright Franka arm is in frame (thousands of bright px), else 'side' (cube only)."""
    g = np.asarray(frame.convert("L"), dtype=np.float32)
    return "arm" if (g > 0.6 * g.max()).mean() > 0.01 else "side"  # relative: robust to light_intensity


def workspace_roi(view: str, size: tuple[int, int], pad: int = 14, aspect: float = 3.0):
    W, H = size
    px = project(WORKSPACE, view, W, H)
    x0, y0 = px.min(0) - pad
    x1, y1 = px.max(0) + pad
    w, h = x1 - x0, y1 - y0
    if not aspect:
        pass
    elif w / h < aspect:
        x0, x1 = x0 - (h * aspect - w) / 2, x1 + (h * aspect - w) / 2
    else:
        y0, y1 = y0 - (w / aspect - h) / 2, y1 + (w / aspect - h) / 2
    return max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))


def track_cube(frames: list[Image.Image], view: str) -> list[tuple[float, float, float]]:
    """Per-frame (cx, cy, size_px) of the cube: the small bright blob inside the projected workspace
    strip that is closest to the previous position (starts at the projected cube start pose)."""
    from scipy import ndimage

    W, H = frames[0].size
    x0, y0, x1, y1 = workspace_roi(view, (W, H), pad=6, aspect=0.0)
    prev = project(np.array([[0.0, 0.0, 0.025]]), view, W, H)[0]
    min_area = 8  # raised to ~40% of the cube's first-frame area (drops gripper fragments / marker)
    g0 = np.asarray(frames[0].convert("L"), dtype=np.float32)
    px, py = int(prev[0]), int(prev[1])
    thr = 0.65 * g0[max(0, py - 30):py + 30, max(0, px - 30):px + 30].max()  # adapts to light_intensity
    out = []
    for f in frames:
        g = np.asarray(f.convert("L"))[y0:y1, x0:x1] > thr
        lab, n = ndimage.label(g)
        best = None
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            area = int((lab[sl] == i).sum())
            if not min_area <= area <= 900 or sl[0].start == 0:  # touching the strip top = the arm reaching in
                continue
            c = np.array([(sl[1].start + sl[1].stop) / 2 + x0, (sl[0].start + sl[0].stop) / 2 + y0])
            size = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
            d = np.linalg.norm(c - prev)
            if d < 80 and (best is None or d < best[0]):
                best = (d, c, size, area)
        if best is not None:
            prev = best[1]
            if not out:
                min_area = max(8, int(0.4 * best[3]))
        out.append((float(prev[0]), float(prev[1]), float(best[2] if best else (out[-1][2] if out else 10))))
    return out


def prepare(clip: Path, n: int, crop: str, out_w: int = 672, tile: int = 256, motion_window: bool = False):
    """crop: none | workspace (projected task strip) | track (square window following the cube)."""
    frames, times = load_frames(clip)
    view = detect_view(frames[0])
    if crop == "workspace":
        roi = workspace_roi(view, frames[0].size)
        frames = [f.crop(roi) for f in frames]
        frames = [f.resize((out_w, int(out_w * f.height / f.width)), Image.LANCZOS) for f in frames]
    elif crop == "track":
        tr = track_cube(frames, view)
        if motion_window:  # keep only the frames where the cube moves (tipping happens mid-motion)
            c = np.asarray([t[:2] for t in tr])
            moving = np.nonzero(np.linalg.norm(np.diff(c, axis=0), axis=1) > 0.5)[0]
            if len(moving):
                a, b = max(0, moving[0]), min(len(frames), moving[-1] + 2)
                frames, times, tr = frames[a:b], times[a:b], tr[a:b]
        side = int(max(36, 4.0 * np.median([t[2] for t in tr])))
        frames = [f.crop((int(cx - side / 2), int(cy - side * 0.6), int(cx + side / 2), int(cy + side * 0.4)))
                  .resize((tile, tile), Image.LANCZOS) for f, (cx, cy, _) in zip(frames, tr)]
    return frames, times


def b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def grid(frames: list[Image.Image], cols: int = 4) -> Image.Image:
    w, h = frames[0].size
    rows = (len(frames) + cols - 1) // cols
    g = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
    for k, f in enumerate(frames):
        g.paste(f, ((k % cols) * w, (k // cols) * h))
    return g


def to_mp4(frames: list[Image.Image], times: list[float]) -> bytes:
    """Constant-30fps mp4 (frames repeated per their WebP duration)."""
    with tempfile.TemporaryDirectory() as d:
        k = 0
        for i, f in enumerate(frames):
            nxt = times[i + 1] if i + 1 < len(times) else times[i] + 0.5
            for _ in range(max(1, round((nxt - times[i]) * 30))):
                f.save(f"{d}/{k:04d}.png")
                k += 1
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", f"{d}/%04d.png",
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", f"{d}/clip.mp4"], check=True)
        return Path(f"{d}/clip.mp4").read_bytes()


# ------------------------------------------------------------------------------------------ model
@dataclass
class CosmosEyes:
    server: str = "http://127.0.0.1:8080"
    mode: str = "frames"  # frames | grid | video
    n: int = 8
    crop: str = "track"  # none | workspace | track
    think: bool = False  # <think> reasoning HURT accuracy here (loops, "upright" hallucinations)
    max_tokens: int = 2048
    seed: int = 0
    tile: int = 448  # px per tracked-crop frame
    temperature: float = 0.0  # greedy is best without <think>; use 0.6 / top_p 0.95 if you enable thinking
    motion_window: bool = False  # True = sample only while the cube moves (tested: worse, 85%/72%)
    prompt: str = "tilt"  # events (model writes the event list) | tilt (per-frame tilt; code builds events)

    def messages(self, clip: Path):
        frames, times = prepare(clip, self.n, self.crop, tile=self.tile, motion_window=self.motion_window)
        content = []
        if self.mode == "video":
            mp4 = to_mp4(frames, times)
            content.append({"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(mp4).decode()}})
            intro = f"This is a {times[-1]:.1f} s video of a robot push experiment rendered in a physics simulator."
        else:
            fs, ts = sample(frames, times, self.n, by_index=self.motion_window)
            self._times = ts
            if self.mode == "grid":
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(grid(fs))}})
                stamps = ", ".join(f"{t:.2f}s" for t in ts)
                intro = (f"This image is a contact sheet of {len(fs)} frames from a robot push experiment rendered in a "
                         f"physics simulator, read left-to-right, top-to-bottom, at times {stamps}.")
            else:
                for k, (f, t) in enumerate(zip(fs, ts)):
                    content.append({"type": "text", "text": f"Frame {k} (t={t:.2f}s):"})
                    content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(f)}})
                intro = f"These are {len(fs)} frames in time order from a robot push experiment rendered in a physics simulator."
        q = (TILT_QUESTION if self.prompt == "tilt" else QUESTION).format(media_intro=intro)
        q += THINK if self.think else NO_THINK
        content.append({"type": "text", "text": q})
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]

    def ask(self, clip: Path) -> dict:
        t0 = time.time()
        r = requests.post(f"{self.server}/v1/chat/completions", json={
            "messages": self.messages(clip), "temperature": self.temperature, "top_p": 0.95, "seed": self.seed,
            "max_tokens": self.max_tokens,
        }, timeout=600)
        r.raise_for_status()
        js = r.json()
        msg = js["choices"][0]["message"]
        text = (msg.get("reasoning_content") or "") + "\n</think>\n" + (msg.get("content") or "")
        out = parse_tilt(msg.get("content") or text, self._times) if self.prompt == "tilt" else parse(msg.get("content") or text)
        return {"latency_s": time.time() - t0, "raw": text, "usage": js.get("usage"), "timings": js.get("timings"), **out}

    def events(self, clip: Path) -> list[dict]:
        """The CosmosEyes contract: clip -> [{"t": s, "type": "slid|tipped|bounced|stalled"}]."""
        return self.ask(clip)["events"]


def parse(text: str) -> dict:
    after = text.split("</think>")[-1]
    for cand in reversed(re.findall(r"\{.*\}", after, flags=re.S) or re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            js = json.loads(cand)
            ev = [e for e in js.get("events", []) if isinstance(e, dict)]
            outcome = js.get("outcome") or outcome_of(ev)
            return {"events": ev, "outcome": outcome, "parsed": True}
        except (json.JSONDecodeError, AttributeError):
            continue
    low = after.lower()
    guess = "tipped" if "tipped" in low else "slid" if "slid" in low else None
    return {"events": [], "outcome": guess, "parsed": False}


def parse_tilt(text: str, times: list[float], min_tilted: int = 1) -> dict:
    """Per-frame tilt flags -> events. tipped at the first tilted frame (ignoring frame 0, the cube at rest)."""
    after = text.split("</think>")[-1]
    cands = re.findall(r"\{.*\}", after, flags=re.S) + re.findall(r"\[.*\]", after, flags=re.S)
    for cand in reversed(cands):
        try:
            js = json.loads(cand)
            fr = js["frames"] if isinstance(js, dict) else js  # the model sometimes returns the bare list
            tilted = [int(f["i"]) for f in fr if f.get("tilted") in (True, "true") and 0 < int(f["i"]) < len(times)]
            if len(tilted) >= min_tilted:
                ev = [{"t": round(times[0 if len(times) < 2 else 1], 2), "type": "slid"},
                      {"t": round(times[min(tilted)], 2), "type": "tipped"}]
                return {"events": ev, "outcome": "tipped", "parsed": True, "tilted_frames": tilted}
            return {"events": [{"t": round(times[min(1, len(times) - 1)], 2), "type": "slid"}], "outcome": "slid",
                    "parsed": True, "tilted_frames": []}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return {"events": [], "outcome": None, "parsed": False}


def outcome_of(events: list[dict]) -> str | None:
    types = [e.get("type") for e in events]
    if "tipped" in types:
        return "tipped"
    if "slid" in types:
        return "slid"
    return types[-1] if types else None


# ------------------------------------------------------------------------------------------ eval
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--mode", choices=["frames", "grid", "video"], default="frames")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--crop", choices=["none", "workspace", "track"], default="track")
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tile", type=int, default=448)
    ap.add_argument("--motion-window", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--prompt", choices=["events", "tilt"], default="tilt")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--view", choices=["side", "arm", "all"], default="all")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--rescore", type=Path, default=None, help="re-parse a results .jsonl (no model calls)")
    args = ap.parse_args()
    eyes = CosmosEyes(server=args.server, mode=args.mode, n=args.n, crop=args.crop, think=args.think,
                      tile=args.tile, temperature=args.temp, prompt=args.prompt,
                      motion_window=args.motion_window)

    if not args.labels:
        for c in args.clips:
            res = eyes.ask(c)
            print(json.dumps({"clip": str(c), "events": res["events"], "outcome": res["outcome"],
                              "latency_s": round(res["latency_s"], 2)}))
        return

    if args.rescore:  # re-parse stored raw outputs (after a parser fix) without calling the model
        rows = [json.loads(line) for line in args.rescore.open()]
        for r in rows:
            clip = args.labels.parent / r["clip"] if args.labels else Path(r["clip"])
            if args.prompt == "tilt":
                fr, ts = prepare(clip, args.n, args.crop, tile=args.tile, motion_window=args.motion_window)
                res = parse_tilt(r["raw"], sample(fr, ts, args.n, by_index=args.motion_window)[1])
            else:
                res = parse(r["raw"].split("\n</think>\n", 1)[-1])
            r.update(pred=res["outcome"], ok=res["outcome"] == r["label"], parsed=res["parsed"], events=res["events"])
        args.rescore.write_text("".join(json.dumps(r) + "\n" for r in rows))
        summarize(rows, args.rescore.stem + " (rescored)")
        return

    labels = json.loads(args.labels.read_text())
    if args.view != "all":
        labels = [x for x in labels if x["view"] == args.view]
    tag = args.tag or f"{args.prompt}-{args.mode}-n{args.n}-{args.crop}{args.tile if args.crop == 'track' else ''}-t{args.temp}-{'think' if args.think else 'nothink'}-{args.view}"
    out = Path(__file__).parent / "results" / f"{tag}.jsonl"
    out.parent.mkdir(exist_ok=True)
    rows = []
    with out.open("w") as fh:
        for x in labels:
            res = eyes.ask(args.labels.parent / x["clip"])
            ok = res["outcome"] == x["label"]
            row = {"clip": x["clip"], "view": x["view"], "label": x["label"], "pred": res["outcome"], "ok": ok,
                   "parsed": res["parsed"], "events": res["events"], "latency_s": round(res["latency_s"], 2),
                   "usage": res["usage"], "timings": res["timings"], "raw": res["raw"]}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            print(f"{x['clip']:24s} gt={x['label']:7s} pred={str(res['outcome']):7s} {'OK ' if ok else 'BAD'} "
                  f"{res['latency_s']:.1f}s", flush=True)
    summarize(rows, tag)


def summarize(rows: list[dict], tag: str) -> None:
    acc = np.mean([r["ok"] for r in rows])
    lat = [r["latency_s"] for r in rows]
    tip = [r for r in rows if r["label"] == "tipped"]
    slid = [r for r in rows if r["label"] == "slid"]
    print(f"[{tag}] acc={acc:.2f} ({sum(r['ok'] for r in rows)}/{len(rows)})  "
          f"tipped-recall={np.mean([r['ok'] for r in tip]) if tip else float('nan'):.2f}  "
          f"slid-recall={np.mean([r['ok'] for r in slid]) if slid else float('nan'):.2f}  "
          f"parsed={np.mean([r['parsed'] for r in rows]):.2f}  "
          f"latency median={np.median(lat):.1f}s max={max(lat):.1f}s")


if __name__ == "__main__":
    main()
