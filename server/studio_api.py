"""Studio API: calibrate a simulator from the visitor's own data.

  GET  /studio                                   the Studio page
  GET  /api/studio/samples                       sample sessions (robot logs, phone videos)
  POST /api/studio/sessions                      multipart: file (video | .csv | .json) or sample=<id>; name
  POST /api/studio/sessions/{sid}/videos         multipart: file or sample part -> add another video of the same table
  POST /api/studio/simulate                      {"kind": "video"|"log", "world": {...}, "pushes", "reach"} -> NVIDIA Newton makes the data
  POST /api/studio/sessions/{sid}/simulate-more  {"speeds"|"commands": [...]} -> run the next experiment in the same world
  POST /api/studio/sessions/{sid}/logs           multipart: file -> append more pushes to a robot-log session
  POST /api/studio/sessions/{sid}/videos/{i}/track   {"corners": [x,y]*4, "sheet": "a4", "object_px": [x,y]?, "object_height_m"?}
  POST /api/studio/sessions/{sid}/analyze        {"agent": true|false} -> streams on /events
  POST /api/studio/chat                          {"sid"?, "messages": [...], "lang": "en"|"ko", "step"?} -> {"answer"}
  GET  /api/studio/sessions/{sid}/events         Server-Sent Events: stage, agent_step*, calibration, next, done, overlay, result, end | error
  GET  /api/studio/sessions/{sid}                session, tracked videos, last result
  GET  /api/studio/sessions/{sid}/export/{fmt}   json | newton | isaaclab | markdown | csv
  GET  /api/studio/sessions/{sid}/truth          samples only, after an analysis: the hidden physics
  GET  /api/studio/files/{sid}/{name}            uploaded video and its first frame

Sessions live in memory and under runs/studio/<sid>/ (uploads, results). Uploads are capped in size and
duration; the Nemotron agent shares the server's LLM budget and has a per-analysis turn cap.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "studio" / "samples"
MAX_UPLOAD = 120 * 1024 * 1024
MAX_SESSIONS = 60
AGENT_TURNS = 8
CHAT_CAP = 30  # questions per session (each one LLM call against the shared budget)
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")

SAMPLES = [
    {"id": "flick", "kind": "video", "title": "Phone video: a box on a table", "parts": ["flick-video", "flick-video-2"],
     "blurb": "Seven hand flicks filmed from the side, an A4 sheet for scale. Something about this table is off. "
              "Synthetic: rendered by NVIDIA Newton from physics the Studio never sees.", "object_height_m": 0.06},
    {"id": "lab-bench", "kind": "log", "file": "lab-bench.csv", "title": "Robot log: first day on the real bench",
     "blurb": "20 pushes (16 aimed at targets, 4 probes) with camera tracks, from a policy trained in the nominal sim."},
    {"id": "short-reach", "kind": "log", "file": "short-reach.csv", "title": "Robot log: short pushes only",
     "blurb": "8 short pushes. The far half of the table was never measured: watch the Studio refuse to be certain."},
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
        self.result: dict | None = None
        self.events: list[dict] = []
        self.done = True
        self.cond = threading.Condition()
        self.created = time.time()
        self.world: dict | None = None  # simulated session: the visitor's hidden physics (ground truth)
        self.origin_offset = 0.0  # simulated video: median release of take 1 from the simulator origin
        self.object_height: float | None = None
        self.truth_shown: dict | None = None  # set when the visitor reveals the hidden physics

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
                "busy": not self.done, "simulated": self.world is not None,
                "world_keys": sorted(self.world) if self.world else []}


def mount_studio(app: FastAPI, llm, budget, data_dir: Path, budgeted_llm_cls) -> None:
    store = data_dir / "studio"
    store.mkdir(parents=True, exist_ok=True)
    sessions: dict[str, StudioSession] = {}
    agent_lock = threading.Lock()

    def get(sid: str) -> StudioSession:
        s = sessions.get(sid)
        if s is None:
            raise HTTPException(404, "Unknown session (sessions expire when the server restarts).")
        return s

    def evict() -> None:
        while len(sessions) >= MAX_SESSIONS:
            old = min(sessions.values(), key=lambda x: x.created)
            sessions.pop(old.id, None)
            shutil.rmtree(old.dir, ignore_errors=True)

    def add_video(s: StudioSession, src: Path, hint: list | None = None) -> dict:
        from studio.session import SessionError
        from studio.video import first_frame_jpeg

        i = len(s.videos)
        dst = s.dir / f"video{i}{src.suffix.lower()}"
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        try:
            jpg, info = first_frame_jpeg(dst)
        except SessionError as e:
            raise HTTPException(422, str(e)) from None
        (s.dir / f"frame{i}.jpg").write_bytes(jpg)
        v = {"index": i, "file": f"files/{s.id}/{dst.name}", "frame": f"files/{s.id}/frame{i}.jpg", "path": str(dst),
             "width": round(info["width"] / info["scale"]), "height": round(info["height"] / info["scale"]), "fps": info["fps"],
             "corners_hint": hint, "tracked": False, "pushes": [], "overlay": None}
        s.videos.append(v)
        return v

    async def save_upload(file: UploadFile, folder: Path) -> Path:
        name = Path(file.filename or "upload").name
        path = folder / ("upload" + Path(name).suffix.lower())
        size = 0
        with path.open("wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, f"File too large (max {MAX_UPLOAD >> 20} MB). Trim the video to the pushes.")
                f.write(chunk)
        return path

    def sample(sid: str) -> dict:
        m = next((x for x in SAMPLES if x["id"] == sid), None)
        if m is None:
            raise HTTPException(404, "Unknown sample.")
        return m

    def hint_for(part: str) -> list | None:
        t = SAMPLE_DIR / f"{part}.truth.json"
        return json.loads(t.read_text())["sheet_corners_px"] if t.exists() else None

    @app.get("/studio", response_class=HTMLResponse)
    def studio_page():
        for page in (ROOT / "dashboard" / "dist" / "studio.live.html", ROOT / "dashboard" / "studio.html"):
            if page.exists():
                return HTMLResponse(page.read_text())
        raise HTTPException(404, "Studio page not built. Run `make dashboard`.")

    @app.get("/api/studio/sample-frame/{sample_id}")
    def sample_frame(sample_id: str):
        from studio.video import first_frame_jpeg

        m = sample(sample_id)
        if m["kind"] != "video":
            raise HTTPException(404, "Not a video sample.")
        cache = store / f"_thumb-{m['id']}.jpg"
        if not cache.exists():
            cache.write_bytes(first_frame_jpeg(SAMPLE_DIR / f"{m['parts'][0]}.mp4")[0])
        return FileResponse(cache, media_type="image/jpeg")

    @app.get("/api/studio/samples")
    def samples():
        return [{k: v for k, v in x.items() if k in ("id", "kind", "title", "blurb", "parts")} for x in SAMPLES]

    @app.post("/api/studio/sessions")
    async def create(file: UploadFile | None = File(None), sample_id: str | None = Form(None, alias="sample"),
                     name: str | None = Form(None)):
        from studio.session import SessionError, load

        if file is None and not sample_id:
            raise HTTPException(422, "Upload a video or a log, or pick a sample.")
        evict()
        sid = uuid.uuid4().hex[:10]
        folder = store / sid
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if sample_id:
                m = sample(sample_id)
                s = StudioSession(sid, m["title"], m["kind"], m["id"], folder)
                if m["kind"] == "video":
                    add_video(s, SAMPLE_DIR / f"{m['parts'][0]}.mp4", hint_for(m["parts"][0]))
                else:
                    s.session = load((SAMPLE_DIR / m["file"]).read_text(), m["file"], m["title"])
            else:
                path = await save_upload(file, folder)
                fname = (file.filename or "upload").lower()
                title = (name or Path(file.filename or "upload").stem)[:60]
                if fname.endswith(VIDEO_EXT):
                    s = StudioSession(sid, title, "video", None, folder)
                    add_video(s, path)
                elif fname.endswith((".csv", ".json", ".txt")):
                    s = StudioSession(sid, title, "log", None, folder)
                    s.session = load(path.read_text(errors="replace"), fname, title)
                else:
                    raise HTTPException(415, "Upload a phone video (mp4/mov/webm) or a robot log (.csv/.json).")
        except SessionError as e:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(422, str(e)) from None
        except HTTPException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        sessions[sid] = s
        return s.public()

    sim_lock = threading.Lock()

    @app.post("/api/studio/simulate")
    def simulate(req: SimulateRequest):
        """Make the 'real' data in NVIDIA Newton from the visitor's own hidden physics."""
        from studio import simulate as sm
        from studio.session import SessionError

        try:
            world = sm.validate(req.world, req.kind)
            speeds = sm.flick_speeds(world, req.pushes, req.reach, req.seed) if req.kind == "video" else None
        except SessionError as e:
            raise HTTPException(422, str(e)) from None
        if not sim_lock.acquire(timeout=90):
            raise HTTPException(429, "Newton is busy with another simulation. Try again in a minute.")
        evict()
        sid = uuid.uuid4().hex[:10]
        folder = store / sid
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if req.kind == "video":
                s = StudioSession(sid, "Your simulated table (phone video)", "video", None, folder)
                meta = sm.render_video(world, speeds, folder / "video0.mp4", seed=req.seed + 11)
                starts = sorted(p["start_y_m"] for p in meta["pushes"])
                s.origin_offset = starts[len(starts) // 2]
                s.object_height = meta["object_height_m"]
                add_video(s, folder / "video0.mp4", meta["sheet_corners_px"])
            else:
                s = StudioSession(sid, "Your simulated robot log", "log", None, folder)
                s.session = sm.robot_log(world, req.pushes, req.reach, req.seed)
                s.session.name = s.name
        except SessionError as e:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(422, str(e)) from None
        finally:
            sim_lock.release()
        s.world = world
        sessions[sid] = s
        return s.public()

    @app.post("/api/studio/sessions/{sid}/simulate-more")
    def simulate_more(sid: str, req: MorePushesRequest):
        """Run the suggested next experiment in the same simulated world: a new video take or more log rows."""
        from studio import simulate as sm
        from studio.session import Session, SessionError

        s = get(sid)
        if s.world is None:
            raise HTTPException(422, "Only simulated sessions can run pushes in Newton; film or log them for real data.")
        if not sim_lock.acquire(timeout=90):
            raise HTTPException(429, "Newton is busy with another simulation. Try again in a minute.")
        try:
            if s.kind == "video":
                if not s.videos or not s.videos[0]["tracked"]:
                    raise HTTPException(422, "Track the first take before adding another.")
                speeds = [min(4.0, max(0.3, float(v))) for v in (req.speeds or [])]
                if not speeds:
                    raise HTTPException(422, "Give the launch speeds to run.")
                i = len(s.videos)
                meta = sm.render_video(s.world, speeds, s.dir / f"take{i}.mp4", seed=97 + i)
                v = add_video(s, s.dir / f"take{i}.mp4", meta["sheet_corners_px"])
                (s.dir / f"take{i}.mp4").unlink(missing_ok=True)
                out = s.public() | {"added": v["index"]}
            else:
                cmds = [min(4.5, max(0.3, float(c))) for c in (req.commands or [])]
                if not cmds:
                    raise HTTPException(422, "Give the commands to run.")
                extra = sm.robot_log(s.world, 0, 0.0, seed=len(s.session.pushes), commands=cmds)
                s.session = Session(s.session.name, "log", s.session.pushes + extra.pushes, s.session.sim,
                                    s.session.notes, s.session.meta)
                out = s.public() | {"added_pushes": len(extra.pushes)}
        except SessionError as e:
            raise HTTPException(422, str(e)) from None
        finally:
            sim_lock.release()
        s.result = None
        return out

    @app.post("/api/studio/sessions/{sid}/videos")
    async def more_video(sid: str, file: UploadFile | None = File(None), part: str | None = Form(None)):
        s = get(sid)
        if s.kind != "video":
            raise HTTPException(422, "This session is a robot log; start a new session for a video.")
        if not s.videos or not s.videos[0]["tracked"]:
            raise HTTPException(422, "Track the first video before adding another.")
        if part:
            m = sample(s.sample or "")
            if part not in m["parts"]:
                raise HTTPException(404, "Unknown sample part.")
            v = add_video(s, SAMPLE_DIR / f"{part}.mp4", hint_for(part))
        elif file is not None:
            path = await save_upload(file, s.dir)
            v = add_video(s, path, s.videos[0].get("corners"))
        else:
            raise HTTPException(422, "Upload a video.")
        return s.public() | {"added": v["index"]}

    @app.post("/api/studio/sessions/{sid}/logs")
    async def more_log(sid: str, file: UploadFile = File(...)):
        """Append the pushes of another log (e.g. the suggested next experiment) to a log session."""
        from studio.session import Session, SessionError, load

        s = get(sid)
        if s.kind != "log" or s.session is None:
            raise HTTPException(422, "Only robot-log sessions take more log rows; add a video to a video session.")
        path = await save_upload(file, s.dir)
        try:
            extra = load(path.read_text(errors="replace"), (file.filename or "more.csv").lower())
        except SessionError as e:
            raise HTTPException(422, str(e)) from None
        if extra.has_commands != s.session.has_commands:
            raise HTTPException(422, "The new rows must use the same columns as the first log (commands vs launch speeds).")
        for p in extra.pushes:
            p.origin = "added"
        s.session = Session(s.session.name, "log", s.session.pushes + extra.pushes, s.session.sim,
                            list(dict.fromkeys(s.session.notes + extra.notes)), s.session.meta)
        s.result = None
        return s.public() | {"added_pushes": len(extra.pushes)}

    @app.post("/api/studio/sessions/{sid}/videos/{i}/track")
    def track(sid: str, i: int, req: TrackRequest):
        from studio.session import Session, SessionError
        from studio.video import track_video

        s = get(sid)
        if not 0 <= i < len(s.videos):
            raise HTTPException(404, "Unknown video.")
        v = s.videos[i]
        prior = s.session.meta if (i > 0 and s.session is not None) else None
        height = req.object_height_m or s.object_height or next((x.get("object_height_m") for x in SAMPLES if x["id"] == s.sample), None)
        try:
            sess, dbg = track_video(Path(v["path"]), req.corners, req.sheet, s.name, req.object_px, height, prior=prior)
        except SessionError as e:
            raise HTTPException(422, str(e)) from None
        s.cams[i] = (dbg.pop("to_px"), dbg.pop("box_px"))
        for p in sess.pushes:
            p.origin = f"video{i}"
        v.update(tracked=True, corners=req.corners, sheet=req.sheet, pushes=dbg["pushes"], camera=dbg["camera"],
                 object_px=dbg["object_px"], axis_px=dbg["axis_px"], notes=sess.notes)
        if i == 0 or s.session is None:
            s.session = sess
        else:  # replace this video's pushes, keep the others
            keep = [p for p in s.session.pushes if p.origin != f"video{i}"]
            s.session = Session(s.session.name, "video", keep + sess.pushes, s.session.sim,
                                list(dict.fromkeys(s.session.notes + sess.notes)), s.session.meta)
        return s.public()

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
        from agent.tool_agent import at_start
        from sim.push_task import analytic_track

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

    def worker(s: StudioSession, use_agent: bool) -> None:
        from studio import export
        from studio.pipeline import analyze

        got_lock = False
        try:
            ag = None
            if use_agent and llm is not None:
                got_lock = agent_lock.acquire(timeout=120)
                if got_lock:
                    ag = budgeted_llm_cls(llm, budget, AGENT_TURNS)
                else:
                    s.push({"type": "stage", "stage": "note", "message": "agent busy with another session: offline search"})
            res = analyze(s.session, ag, on_event=s.push)
            doc = res.to_json()
            if s.cams:
                doc["overlay"] = overlay(s, res.calibration)
                s.push({"type": "overlay", "overlay": doc["overlay"]})
            doc["report_json"] = export.to_json(s.session, res.calibration, res.next_experiment, res.agent)
            s.result = doc
            (s.dir / "result.json").write_text(json.dumps(doc, indent=1, default=str))
            s.push({"type": "result", "result": doc})
        except Exception as e:  # noqa: BLE001
            s.push({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            if got_lock:
                agent_lock.release()
            if not s.done:
                s.push({"type": "end"})

    @app.post("/api/studio/sessions/{sid}/analyze")
    def analyze_route(sid: str, req: AnalyzeRequest):
        s = get(sid)
        if s.session is None:
            raise HTTPException(422, "Track the video first (click the sheet corners).")
        if not s.done:
            raise HTTPException(429, "This session is already being analysed.")
        with s.cond:
            s.events, s.done = [], False
        threading.Thread(target=worker, args=(s, req.agent), daemon=True).start()
        return {"ok": True, "agent": bool(req.agent and llm is not None)}

    chat_counts: dict[str, int] = {}

    @app.post("/api/studio/chat")
    def chat(req: ChatRequest):
        """Questions about the Studio and the visitor's current result, answered by Nemotron in EN or KO."""
        from studio import chat as ch

        if llm is None:
            raise HTTPException(503, "Ask needs a language model: start the server with GAPCLOSER_LLM=tokenfactory.")
        last = req.messages[-1]
        if last.get("role") != "user" or not str(last.get("content") or "").strip():
            raise HTTPException(422, "The last message must be your question.")
        key = req.sid or "-"
        if chat_counts.get(key, 0) >= CHAT_CAP:
            raise HTTPException(429, f"This session reached {CHAT_CAP} questions. Start a new session to ask more.")
        s = sessions.get(req.sid) if req.sid else None
        # the hidden physics is part of the context only after the visitor revealed it on the page
        shown = {"model": s.truth_shown} if s is not None and s.truth_shown else None
        summary = ch.session_summary(s.public() if s else None, shown, req.step)
        try:
            text = ch.answer(budgeted_llm_cls(llm, budget, 1), req.messages, req.lang, summary)
        except RuntimeError as e:  # budget exhausted
            raise HTTPException(429, str(e)) from None
        except Exception as e:  # noqa: BLE001 - provider hiccup: tell the visitor, keep the page alive
            raise HTTPException(502, f"The model did not answer ({type(e).__name__}). Please try again.") from None
        chat_counts[key] = chat_counts.get(key, 0) + 1
        return {"answer": text, "left": CHAT_CAP - chat_counts[key]}

    @app.get("/api/studio/sessions/{sid}/events")
    def events(sid: str):
        s = get(sid)

        def stream():
            i = 0
            while True:
                with s.cond:
                    while i >= len(s.events) and not s.done:
                        if not s.cond.wait(timeout=15):
                            break
                    batch, done = s.events[i:], s.done
                if not batch and not done:
                    yield ": keep-alive\n\n"
                    continue
                for e in batch:
                    yield f"data: {json.dumps(e, default=str)}\n\n"
                i += len(batch)
                if done and i >= len(s.events):
                    return

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/studio/sessions/{sid}")
    def state(sid: str):
        return get(sid).public()

    @app.get("/api/studio/sessions/{sid}/export/{fmt}")
    def export_route(sid: str, fmt: str):
        from studio.session import to_csv

        s = get(sid)
        if fmt == "csv":
            if s.session is None:
                raise HTTPException(422, "No data yet.")
            return PlainTextResponse(to_csv(s.session), media_type="text/csv",
                                     headers={"Content-Disposition": f'attachment; filename="{sid}-pushes.csv"'})
        if s.result is None:
            raise HTTPException(422, "Run the analysis first.")
        files = {"json": ("calibration.json", json.dumps(s.result["report_json"], indent=2), "application/json"),
                 "newton": ("newton_calibration.py", s.result["exports"]["newton"], "text/x-python"),
                 "isaaclab": ("isaaclab_events.py", s.result["exports"]["isaaclab"], "text/x-python"),
                 "markdown": ("report.md", s.result["exports"]["markdown"], "text/markdown")}
        if fmt not in files:
            raise HTTPException(404, "Formats: json, newton, isaaclab, markdown, csv.")
        name, text, mime = files[fmt]
        return PlainTextResponse(text, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/studio/sessions/{sid}/truth")
    def truth(sid: str):
        s = get(sid)
        if not s.sample and s.world is None:
            raise HTTPException(404, "Only samples and simulated tables have a known ground truth.")
        if s.result is None:
            raise HTTPException(409, "Run the analysis first: the hidden physics is revealed afterwards.")
        if s.world is not None:
            from studio.simulate import truth as sim_truth

            out = sim_truth(s.world)
            s.truth_shown = out["model"]
            if s.kind == "video" and out["model"].get("patch_y0") is not None:
                out["model"]["patch_y0"] = round(out["model"]["patch_y0"] - s.origin_offset, 4)
                out["note"] = (f"region start measured from the median release point "
                               f"({s.origin_offset * 100:+.1f} cm from the simulator origin)")
            return out
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
        s.truth_shown = out["model"]
        return out

    @app.get("/api/studio/files/{sid}/{name}")
    def files(sid: str, name: str):
        s = get(sid)
        path = (s.dir / Path(name).name).resolve()
        if path.parent != s.dir.resolve() or not path.exists() or path.name.startswith("upload"):
            raise HTTPException(404, "No such file.")
        return FileResponse(path)


def s_session_sim(base):
    """The visitor's current simulator as the ghost should replay it: hand pushes are speeds, so gain 1."""
    return base.with_(actuator_gain=1.0)


def truth_view(t: dict) -> dict:
    """Ground truth in the agent's model fields (mu_eff = mean of object and table mu, as in Newton XPBD)."""
    from sim.params import effective_friction

    h = t["hidden"]
    out = {"source": t.get("env") or t.get("renderer", "NVIDIA Newton")}
    if "mu_eff" in h:
        out["model"] = {"mu_eff": h["mu_eff"], "patch_y0": h.get("patch_y0"), "patch_mu": h.get("patch_mu")}
    else:
        mu = effective_friction(h.get("object_mu", 0.8), h.get("table_mu", 0.8))
        out["model"] = {"mu_eff": mu, **{k: h[k] for k in ("actuator_gain", "patch_y0", "patch_mu", "camera_dx", "camera_pitch_deg", "lens_k")
                                         if k in h}}
    return out
