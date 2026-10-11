"""Studio session state and the pure helpers around it: the sample catalog, request models, the in-memory session,
hidden truth for samples and simulated worlds, and the overlay geometry drawn on videos. server.studio_api holds the
routes and the store; everything here is independent of FastAPI's app object."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel, Field

from tether.sim.settings import setting
from tether.paths import REPO

ROOT = REPO
SAMPLE_DIR = ROOT / "tether" / "studio" / "samples"
MAX_UPLOAD = 120 * 1024 * 1024
MAX_SESSIONS = int(setting("MAX_SESSIONS", "60"))  # uploads live on disk (in memory on Cloud Run)
AGENT_TURNS = 8
CHAT_CAP = 30  # questions per session (each one LLM call against the shared budget)
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")

SAMPLES = [
    {"id": "flick", "kind": "video", "title": "Phone video: a box on a table", "parts": ["flick-video", "flick-video-2"],
     "blurb": "Seven hand flicks filmed from the side, an A4 sheet for scale. Something about this table is off. "
              "Synthetic: rendered by NVIDIA Newton from physics the Studio never sees.", "object_height_m": 0.06, "domain": "robot"},
    {"id": "lab-bench", "kind": "log", "file": "lab-bench.csv", "title": "Robot log: first day on the real bench",
     "blurb": "20 pushes (16 aimed at targets, 4 probes) with camera tracks, from a policy trained in the nominal sim.", "domain": "robot"},
    {"id": "short-reach", "kind": "log", "file": "short-reach.csv", "title": "Robot log: short pushes only",
     "blurb": "8 short pushes. The far half of the table was never measured: watch the Studio refuse to be certain.", "domain": "robot"},
    {"id": "stop-line", "kind": "video", "parts": ["stop-line-video", "stop-line-video-2"], "domain": "driving",
     "title": "Roadside camera: braking to a stop line",
     "blurb": "Seven braking runs filmed from beside the road, a painted 5.25 × 7.4 m box for scale. Part of the road is wet. "
              "Synthetic: rendered by NVIDIA Newton (Froude-scaled, friction is scale-free).", "object_height_m": 0.06},
    {"id": "brake-log", "kind": "log", "file": "brake-log.csv", "domain": "driving",
     "title": "Vehicle log: braking at the stop line",
     "blurb": "18 braking runs from the planner (speed set-point, stop position, front-camera range to the line). "
              "The car keeps stopping past the line on part of the road."},
    {"id": "press-line", "kind": "log", "file": "press-line.csv", "domain": "factory",
     "title": "Pusher log: parts missing the inspection window",
     "blurb": "18 pusher strokes aimed at the inspection position, with the line camera's tracks. Parts keep stopping off-position."},
]


class TrackRequest(BaseModel):
    corners: list[list[float]] = Field(..., min_length=4, max_length=4)
    sheet: str = "a4"
    object_px: list[float] | None = None
    object_height_m: float | None = Field(None, gt=0.005, lt=0.5)


class AnalyzeRequest(BaseModel):
    agent: bool = True


class SimulateRequest(BaseModel):
    kind: str = Field("video", pattern="^(video|log)$")
    domain: str = Field("robot", pattern="^(robot|driving|factory)$")
    world: dict[str, float | None]
    pushes: int = Field(7, ge=3, le=12)
    reach: float = Field(0.45, ge=0.15, le=0.9)
    seed: int = 0


class ChatRequest(BaseModel):
    sid: str | None = None
    messages: list[dict] = Field(..., min_length=1, max_length=40)
    lang: str = Field("en", pattern="^(en|ko)$")
    step: str | None = Field(None, max_length=20)


class MorePushesRequest(BaseModel):
    speeds: list[float] | None = Field(None, max_length=8)  # video: launch speeds (m/s)
    commands: list[float] | None = Field(None, max_length=8)  # robot log: commands


class StudioSession:
    def __init__(self, sid: str, name: str, kind: str, sample: str | None, folder: Path):
        self.id, self.name, self.kind, self.sample, self.dir = sid, name, kind, sample, folder
        self.session = None  # studio.session.Session (logs: at upload; videos: after tracking)
        self.videos: list[dict] = []  # {file, frame, width, height, fps, corners_hint, tracked, pushes, overlay}
        self.cams: dict[int, object] = {}  # video index -> to_px callable (overlay geometry)
        self.cal = None  # studio.fit.Calibration of the last analysis (for the Newton replay)
        self.training: dict | None = None  # retrain-and-test job: status, progress, result
        self.result: dict | None = None
        self.events: list[dict] = []
        self.done = True
        self.cond = threading.Condition()
        self.created = time.time()
        self.world: dict | None = None  # simulated session: the visitor's hidden physics (ground truth)
        self.origin_offset = 0.0  # simulated video: median release of take 1 from the simulator origin
        self.object_height: float | None = None
        self.truth_shown: dict | None = None  # set when the visitor reveals the hidden physics
        self.domain = "robot"  # studio.domains id: how the page names and scales everything

    @property
    def busy(self) -> bool:
        return not self.done or bool(self.training and self.training["status"] == "running")

    def push(self, e: dict) -> None:
        with self.cond:
            self.events.append(e)
            if e["type"] in ("end", "error"):
                self.done = True
            self.cond.notify_all()

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "sample": self.sample,
                "videos": [{k: v for k, v in x.items() if k not in ("path",)} for x in self.videos],
                "session": self.session.to_json() if self.session else None, "result": self.result,
                "busy": not self.done, "simulated": self.world is not None, "domain": self.domain,
                "world_keys": sorted(self.world) if self.world else []}


def sample(sid: str) -> dict:
    m = next((x for x in SAMPLES if x["id"] == sid), None)
    if m is None:
        raise HTTPException(404, "Unknown sample.")
    return m


def hint_for(part: str) -> list | None:
    t = SAMPLE_DIR / f"{part}.truth.json"
    return json.loads(t.read_text())["sheet_corners_px"] if t.exists() else None


def overlay(s: StudioSession, cal) -> dict:
    """Friction regions of the calibrated model drawn on each video's first frame."""
    out = {}
    m = cal.model
    for i, (to_px, box_px) in s.cams.items():
        regions = []
        y_end = 0.95
        if m.get("patch_y0") is not None:
            regions.append({"from": -0.15, "to": m["patch_y0"], "mu": m["mu_eff"]})
            regions.append({"from": m["patch_y0"], "to": y_end, "mu": m["patch_mu"]})
        else:
            regions.append({"from": -0.15, "to": y_end, "mu": m["mu_eff"]})
        for r in regions:
            r["poly"] = [to_px(r["from"], -0.11), to_px(r["to"], -0.11), to_px(r["to"], 0.11), to_px(r["from"], 0.11)]
        reach = s.session.coverage()[1] if s.session else 0.0
        for r in regions:  # label anchor: early in the region, inside the measured stretch when possible
            y = r["from"] + 0.35 * (min(r["to"], max(reach, r["from"] + 0.05)) - r["from"])
            r["label_px"] = to_px(max(y, 0.05), 0.075)  # beside the push path, not on it
        unmeasured = ([to_px(reach, -0.11), to_px(y_end, -0.11), to_px(y_end, 0.11), to_px(reach, 0.11)]
                      if reach < y_end - 0.02 else None)
        out[str(i)] = {"regions": regions, "reach_px": [to_px(reach, -0.13), to_px(reach, 0.13)], "reach_m": round(reach, 3),
                       "unmeasured_poly": unmeasured, "ghosts": ghosts(s.videos[i], cal, box_px)}
    return out


def ghosts(v: dict, cal, box_px) -> list[dict]:
    """Per push of this video: where the box would be, frame by frame, in your current sim and in the
    calibrated sim, given the measured launch (speed, place). Drawn over the real video as wireframes."""
    from tether.agent.tool_agent import at_start
    from tether.sim.push_task import analytic_track

    base = cal.base
    cur = s_session_sim(base)
    out = []
    for p in v.get("pushes", []):
        frames = max(20, p["rest_frame"] - p["release_frame"] + 24)
        row = {"push": p["push"], "release_frame": p["release_frame"], "frames": frames}
        for name, prm in (("calibrated", cal.params), ("current", cur)):
            tr = analytic_track(p["launch_speed_mps"] / prm["actuator_gain"], at_start(prm, p["start_m"]), frames)
            row[name] = [box_px(p["start_m"] + y, p.get("lateral_m", 0.0)) for y in tr]
        out.append(row)
    return out


def hidden_truth(s: StudioSession) -> dict | None:
    """The hidden physics of a sample or simulated session, in the Studio frame (None for uploads)."""
    if s.world is not None:
        from tether.studio.simulate import truth as sim_truth

        out = sim_truth(s.world)
        if s.kind == "video" and out["model"].get("patch_y0") is not None:
            out["model"]["patch_y0"] = round(out["model"]["patch_y0"] - s.origin_offset, 4)
            out["note"] = (f"region start measured from the median release point "
                           f"({s.origin_offset * 100:+.1f} cm from the simulator origin)")
        return out
    if not s.sample:
        return None
    m = sample(s.sample)
    part = m["parts"][0] if m["kind"] == "video" else m["file"].rsplit(".", 1)[0]
    t = json.loads((SAMPLE_DIR / f"{part}.truth.json").read_text())
    out = truth_view(t)
    if m["kind"] == "video" and out["model"].get("patch_y0") is not None:
        # Studio measures along the table from the median release point of the first take, not the simulator's
        # origin: express the hidden region start in that frame so the comparison is like for like
        starts = sorted(p["start_y_m"] for p in t["pushes"])
        o = starts[len(starts) // 2]
        out["model"]["patch_y0"] = round(out["model"]["patch_y0"] - o, 4)
        out["note"] = f"region start measured from the median release point ({o * 100:+.1f} cm from the simulator origin)"
    return out


def s_session_sim(base):
    """The visitor's current simulator as the ghost should replay it: hand pushes are speeds, so gain 1."""
    return base.with_(actuator_gain=1.0)


def truth_view(t: dict) -> dict:
    """Ground truth in the agent's model fields (mu_eff = mean of object and table mu, as in Newton XPBD)."""
    from tether.sim.params import effective_friction

    h = t["hidden"]
    out = {"source": t.get("env") or t.get("renderer", "NVIDIA Newton")}
    if "mu_eff" in h:
        out["model"] = {"mu_eff": h["mu_eff"], "patch_y0": h.get("patch_y0"), "patch_mu": h.get("patch_mu")}
    else:
        mu = effective_friction(h.get("object_mu", 0.8), h.get("table_mu", 0.8))
        out["model"] = {"mu_eff": mu, **{k: h[k] for k in ("actuator_gain", "patch_y0", "patch_mu", "camera_dx", "camera_pitch_deg", "lens_k")
                                         if k in h}}
    return out
