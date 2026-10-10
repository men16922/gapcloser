"""CosmosEyes: NVIDIA Cosmos Reason 2 (8B) as a local *second opinion* on what happened in a clip.

Built on NVIDIA Cosmos. The model (`nvidia/Cosmos-Reason2-8B`, NVIDIA Open Model License) runs locally in
llama.cpp's `llama-server` (community Q4_K_M GGUF + vision projector, see README "Cosmos eyes"). No paid API.

Best config from early tuning (25/26 tuning, 20/20 demo, 24/32 held-out): track the cube with a small blob
tracker, send 8 time-sampled 448 px close-up crops, ask *per frame* "is the cube tilted?" (no <think>,
temperature 0), and build events in code: `slid` at the first moving frame, `tipped` at the first tilted
frame. Tip recall on held-out clips was only 6/12 (false tips are rare), so this never replaces the Newton
`tipped` flag; it sits next to it.

    eyes = CosmosEyes()                       # http://localhost:8080
    if eyes.available():
        eyes.events(Path("runs/demo/clips/tipping-edge-it0-real.webp"))
        # -> [{"t": 0.17, "type": "slid"}, {"t": 0.69, "type": "tipped"}]

Importing this module is cheap (stdlib only); numpy / scipy / Pillow load on the first call. When the
server is down, `available()` is False and `events()` returns [] (`last_error` says why); it never raises.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_NAME = "nvidia/Cosmos-Reason2-8B (Q4_K_M, llama.cpp, local)"
ATTRIBUTION = "Built on NVIDIA Cosmos"

SYSTEM = "You are a helpful assistant."
TILT_QUESTION = (
    "These are {n} frames in time order from a robot push experiment rendered in a physics simulator. "
    "Each frame is a close-up crop centred on a small light-grey cube on a dark checkered "
    "table. For EACH frame decide whether the cube sits flat and upright (a face on the table, its "
    "outline an axis-aligned square/box) or is TILTED (rotated, balanced on an edge or corner, its "
    "outline a diamond or skewed box).\n"
    'Output strict JSON on one line: {{"frames":[{{"i":<frame number>,"tilted":true|false}}]}}'
    "\nReply with the JSON only."
)

# Newton camera presets (sim/newton_push.py: render_trial = "side", render_trial_arm = "arm"):
# (position, pitch-down deg, vertical fov deg). Used to seed the tracker at the cube's start pose.
CAMERAS = {"side": ((0.85, 0.27, 0.30), 17.0, 44.0), "arm": ((1.55, -0.02, 0.78), 24.0, 46.0)}
WORKSPACE = [[x, y, z] for x in (-0.05, 0.05) for y in (-0.06, 1.0) for z in (0.0, 0.07)]


# ------------------------------------------------------------------------------------------ events
def events_from_frames(tilted: list[bool], times: list[float]) -> list[dict]:
    """Per-frame tilt flags -> events. Frame 0 is the cube at rest and is ignored (a perspective-skewed
    rest pose is not a tip). `slid` at the first moving sample, `tipped` at the first tilted one."""
    if not times:
        return []
    first_move = round(times[min(1, len(times) - 1)], 2)
    hits = [i for i, f in enumerate(tilted) if f and 0 < i < len(times)]
    ev = [{"t": first_move, "type": "slid"}]
    if hits:
        ev.append({"t": round(times[hits[0]], 2), "type": "tipped"})
    return ev


def parse_frames(text: str, n: int) -> list[bool] | None:
    """Model reply -> n tilt flags, or None if no parseable answer. Accepts {"frames":[...]}, a bare
    list, and code fences; frames the model skipped count as not tilted."""
    after = (text or "").split("</think>")[-1]
    cands = re.findall(r"\{.*\}", after, flags=re.S) + re.findall(r"\[.*\]", after, flags=re.S)
    for cand in reversed(cands):
        try:
            js = json.loads(cand)
            fr = js["frames"] if isinstance(js, dict) else js
            flags = [False] * n
            for f in fr:
                i = int(f["i"])
                if 0 <= i < n:
                    flags[i] = f.get("tilted") in (True, "true")
            return flags
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return None


def outcome(events: list[dict]) -> str | None:
    types = [e.get("type") for e in events]
    return "tipped" if "tipped" in types else "slid" if "slid" in types else None


# ------------------------------------------------------------------------------------------ frames
def load_frames(clip: Path):
    """All frames of an animated WebP/GIF plus their start time (s), honoring per-frame durations."""
    from PIL import Image

    im = Image.open(clip)
    frames, times, t = [], [], 0.0
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        frames.append(im.convert("RGB"))
        times.append(t)
        t += (im.info.get("duration") or 33) / 1000.0
    return frames, times


def sample(frames, times, n: int):
    """n frames evenly spaced in TIME (WebP merges identical frames, so index spacing is uneven)."""
    total = times[-1]
    picks = sorted({min(range(len(times)), key=lambda i: abs(times[i] - total * k / max(1, n - 1))) for k in range(n)})
    return [frames[i] for i in picks], [times[i] for i in picks]


def project(points, view: str, W: int = 480, H: int = 270):
    """World -> pixel for the Newton pinhole camera (looks along world -X, Z up, pitched down)."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    pos, pitch, fov = CAMERAS[view]
    R = (Rotation.from_quat([0.5, 0.5, 0.5, 0.5]) * Rotation.from_rotvec([-np.radians(pitch), 0, 0])).as_matrix()
    pc = (np.asarray(points, dtype=float) - np.asarray(pos)) @ R
    f = (H / 2) / np.tan(np.radians(fov) / 2)
    return np.stack([W / 2 + f * pc[:, 0] / -pc[:, 2], H / 2 - f * pc[:, 1] / -pc[:, 2]], 1)


def detect_view(frame) -> str:
    """'arm' if the big bright Franka arm is in frame, else 'side' (cube only)."""
    import numpy as np

    g = np.asarray(frame.convert("L"), dtype=np.float32)
    return "arm" if (g > 0.6 * g.max()).mean() > 0.01 else "side"


def track_cube(frames, view: str) -> list[tuple[float, float, float]]:
    """Per-frame (cx, cy, size_px): the small bright blob in the projected workspace strip closest to the
    previous position, seeded at the projected cube start pose."""
    import numpy as np
    from scipy import ndimage

    W, H = frames[0].size
    px = project(WORKSPACE, view, W, H)
    (x0, y0), (x1, y1) = px.min(0) - 6, px.max(0) + 6
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    prev = project([[0.0, 0.0, 0.025]], view, W, H)[0]
    min_area = 8
    g0 = np.asarray(frames[0].convert("L"), dtype=np.float32)
    cx, cy = int(prev[0]), int(prev[1])
    thr = 0.65 * g0[max(0, cy - 30):cy + 30, max(0, cx - 30):cx + 30].max()
    out: list[tuple[float, float, float]] = []
    for f in frames:
        g = np.asarray(f.convert("L"))[y0:y1, x0:x1] > thr
        lab, _ = ndimage.label(g)
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


def tracked_crops(clip: Path, n: int = 8, tile: int = 448):
    """n time-sampled square close-ups that follow the cube, and their timestamps (s)."""
    import numpy as np
    from PIL import Image

    frames, times = load_frames(clip)
    tr = track_cube(frames, detect_view(frames[0]))
    side = int(max(36, 4.0 * float(np.median([t[2] for t in tr]))))
    crops = [f.crop((int(cx - side / 2), int(cy - side * 0.6), int(cx + side / 2), int(cy + side * 0.4)))
             .resize((tile, tile), Image.LANCZOS) for f, (cx, cy, _) in zip(frames, tr)]
    return sample(crops, times, n)


def _png_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------------------------------ client
class CosmosEyes:
    """Clip -> physical events from Cosmos Reason 2 on a local llama-server (OpenAI-compatible API)."""

    model_name = MODEL_NAME

    def __init__(self, base_url: str = "http://localhost:8080", n: int = 8, tile: int = 448, timeout: float = 120.0,
                 seed: int = 0):
        self.base_url = base_url.rstrip("/")
        self.n, self.tile, self.timeout, self.seed = n, tile, timeout, seed
        self.last: dict = {}
        self.last_error = ""

    # transport (tests replace these two)
    def _get(self, path: str, timeout: float) -> dict:
        with urllib.request.urlopen(self.base_url + path, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")

    def _post(self, path: str, body: dict, timeout: float) -> dict:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def available(self, timeout: float = 1.5) -> bool:
        """Quick health check: llama-server answers /health with status ok (False while loading or down)."""
        try:
            return self._get("/health", timeout).get("status") == "ok"
        except (OSError, ValueError, urllib.error.URLError) as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return False

    def messages(self, frames, times) -> list[dict]:
        content: list[dict] = []
        for k, (f, t) in enumerate(zip(frames, times)):
            content.append({"type": "text", "text": f"Frame {k} (t={t:.2f}s):"})
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64(f)}})
        content.append({"type": "text", "text": TILT_QUESTION.format(n=len(frames))})
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]

    def look(self, clip: Path) -> dict:
        """Full answer: {"events", "tilted", "times", "outcome", "seconds", "raw", "error"}. Never raises."""
        t0 = time.perf_counter()
        res: dict = {"events": [], "tilted": [], "times": [], "outcome": None, "raw": "", "error": ""}
        try:
            frames, times = tracked_crops(Path(clip), self.n, self.tile)
            res["times"] = [round(t, 3) for t in times]
            js = self._post("/v1/chat/completions", {
                "messages": self.messages(frames, times), "temperature": 0.0, "top_p": 0.95, "seed": self.seed,
                "max_tokens": 512}, self.timeout)
            res["raw"] = (js["choices"][0]["message"].get("content") or "")
            flags = parse_frames(res["raw"], len(times))
            if flags is None:
                res["error"] = "unparseable reply"
            else:
                res["tilted"] = flags
                res["events"] = events_from_frames(flags, times)
                res["outcome"] = outcome(res["events"])
        except Exception as e:  # noqa: BLE001 - eyes are optional; a dead server or odd clip must not stop a run
            res["error"] = f"{type(e).__name__}: {e}"[:200]
        res["seconds"] = round(time.perf_counter() - t0, 2)
        self.last, self.last_error = res, res["error"]
        return res

    def events(self, clip_path) -> list[dict]:
        """The contract: clip -> [{"t": s, "type": "slid|tipped"}]; [] when the server is down."""
        return self.look(Path(clip_path))["events"]
