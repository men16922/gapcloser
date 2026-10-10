"""Studio API: calibrate a simulator from the visitor's own data.

  GET  /studio                                   the Studio page
  GET  /api/studio/samples                       sample sessions (robot logs, phone videos)
  POST /api/studio/sessions                      multipart: file (video | .csv | .json) or sample=<id>; name
  POST /api/studio/sessions/{sid}/videos         multipart: file or sample part -> add another video of the same table
  POST /api/studio/simulate                      {"kind": "video"|"log", "domain", "world": {...}, "pushes", "reach"} -> NVIDIA Newton makes the data
  POST /api/studio/sessions/{sid}/simulate-more  {"speeds"|"commands": [...]} -> run the next experiment in the same world
  POST /api/studio/sessions/{sid}/logs           multipart: file -> append more pushes to a robot-log session
  POST /api/studio/sessions/{sid}/videos/{i}/track   {"corners": [x,y]*4, "sheet": "a4", "object_px": [x,y]?, "object_height_m"?}
  POST /api/studio/sessions/{sid}/analyze        {"agent": true|false} -> streams on /events
  POST /api/studio/sessions/{sid}/verify         replay every push in NVIDIA Newton with the exported and the current physics
  POST /api/studio/sessions/{sid}/train          retrain a policy (current sim / wide randomization / Tether ranges) in parallel Newton worlds
  GET  /api/studio/sessions/{sid}/train          its progress and learning curves
  POST /api/studio/chat                          {"sid"?, "messages": [...], "lang": "en"|"ko", "step"?} -> {"answer"}
  GET  /api/studio/sessions/{sid}/events         Server-Sent Events: stage, agent_step*, calibration, next, done, overlay, result, end | error
  GET  /api/studio/sessions/{sid}                session, tracked videos, last result
  GET  /api/studio/domains                       robot manipulation, autonomous vehicles, factory inspection: names, units, labels
  GET  /api/studio/sessions/{sid}/export/{fmt}   json | newton | isaaclab | markdown | csv | carla (driving)
  GET  /api/studio/sessions/{sid}/truth          samples only, after an analysis: the hidden physics
  GET  /api/studio/files/{sid}/{name}            uploaded video and its first frame

Domains (studio.domains): robot manipulation, autonomous vehicles, factory inspection. Everything on the wire is in
base units (the robot scale); the page and the exports convert to the session's domain.

Sessions live in memory and under runs/studio/<sid>/ (uploads, results). Uploads are capped in size and
duration; the Nemotron agent shares the server's LLM budget and has a per-analysis turn cap.
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from server.studio_session import (  # noqa: F401 - re-exported: tests and tools import them from here
    AGENT_TURNS, CHAT_CAP, MAX_SESSIONS, MAX_UPLOAD, ROOT, SAMPLE_DIR, SAMPLES, VIDEO_EXT, AnalyzeRequest, ChatRequest,
    MorePushesRequest, SimulateRequest, StudioSession, TrackRequest, hidden_truth, hint_for, overlay, sample, truth_view,
)


def mount_studio(app: FastAPI, llm, budget, data_dir: Path, budgeted_llm_cls) -> None:
    store = data_dir / "studio"
    store.mkdir(parents=True, exist_ok=True)
    sessions: dict[str, StudioSession] = {}
    store_lock = threading.Lock()  # guards sessions and chat_counts
    agent_lock = threading.Lock()
    sim_lock = threading.Lock()  # one NVIDIA Newton job at a time (CPU)
    app.state.studio_sessions = sessions  # for tests and diagnostics

    def get(sid: str) -> StudioSession:
        s = sessions.get(sid)
        if s is None:
            raise HTTPException(404, "Unknown session (sessions expire when the server restarts).")
        return s

    def evict() -> None:
        """Make room for one more session: drop the oldest idle ones (never one that is analysing or training)."""
        with store_lock:
            while len(sessions) >= MAX_SESSIONS:
                idle = [x for x in sessions.values() if not x.busy]
                if not idle:
                    raise HTTPException(429, "The Studio is busy with other visitors. Try again in a minute.")
                old = min(idle, key=lambda x: x.created)
                sessions.pop(old.id, None)
                shutil.rmtree(old.dir, ignore_errors=True)

    def register(s: StudioSession) -> dict:
        with store_lock:
            sessions[s.id] = s
        return s.public()

    def newton(fn, *a, **kw):
        """Run one NVIDIA Newton job under the shared lock; a failure becomes a readable 500, not a bare one."""
        if not sim_lock.acquire(timeout=90):
            raise HTTPException(429, "Newton is busy with another simulation. Try again in a minute.")
        try:
            return fn(*a, **kw)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"NVIDIA Newton run failed ({type(e).__name__}: {str(e)[:160]}).") from None
        finally:
            sim_lock.release()

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
        auto = False
        if hint is None:  # an upload: look for the sheet ourselves; the visitor checks and can drag the corners
            from studio.video import auto_sheet

            hint = auto_sheet(dst)
            auto = hint is not None
        v = {"index": i, "file": f"files/{s.id}/{dst.name}", "frame": f"files/{s.id}/frame{i}.jpg", "path": str(dst),
             "width": round(info["width"] / info["scale"]), "height": round(info["height"] / info["scale"]), "fps": info["fps"],
             "corners_hint": hint, "corners_auto": auto, "tracked": False, "pushes": [], "overlay": None}
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

    @app.get("/studio", response_class=HTMLResponse)
    def studio_page():
        from dashboard.build import with_shared

        for page in (ROOT / "dashboard" / "dist" / "studio.live.html", ROOT / "dashboard" / "studio.html"):
            if page.exists():
                return HTMLResponse(with_shared(page.read_text()))
        raise HTTPException(404, "Studio page not built. Run `make dashboard`.")

    @app.get("/api/studio/sample-frame/{sample_id}")
    def sample_frame(sample_id: str):
        from studio.video import first_frame_jpeg

        m = sample(sample_id)
        still = SAMPLE_DIR / f"{m['id']}-frame.jpg"  # log samples may ship a Newton-rendered still of their scene
        if m["kind"] != "video":
            if still.exists():
                return FileResponse(still, media_type="image/jpeg")
            raise HTTPException(404, "Not a video sample.")
        cache = store / f"_thumb-{m['id']}.jpg"
        if not cache.exists():
            cache.write_bytes(first_frame_jpeg(SAMPLE_DIR / f"{m['parts'][0]}.mp4")[0])
        return FileResponse(cache, media_type="image/jpeg")

    @app.get("/api/studio/samples")
    def samples():
        from studio.session import load

        def preview(x: dict) -> list | None:  # where each logged push stopped (base metres), for the card's thumbnail
            if x["kind"] != "log":
                return None
            sess = load((SAMPLE_DIR / x["file"]).read_text(), x["file"], x["title"])
            return [round(p.start + p.stop, 3) for p in sess.pushes if not p.tipped]

        return [{k: v for k, v in x.items() if k in ("id", "kind", "title", "blurb", "parts", "domain")} | {"stops": preview(x)} for x in SAMPLES]

    @app.get("/api/studio/domains")
    def domain_list():
        from studio.domains import page_data

        return list(page_data().values())

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
                s.domain = m.get("domain", "robot")
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
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return register(s)

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
                meta = sm.render_video(world, speeds, folder / "video0.mp4", seed=req.seed + 11, domain=req.domain)
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
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(500, f"NVIDIA Newton run failed ({type(e).__name__}).") from None
        finally:
            sim_lock.release()
        s.world = world
        s.domain = req.domain
        s.name = {"robot": s.name, "driving": "Your simulated road", "factory": "Your simulated line"}[req.domain] + \
            (" (roadside video)" if req.domain == "driving" and req.kind == "video" else
             " (vehicle log)" if req.domain == "driving" else
             " (line camera video)" if req.domain == "factory" and req.kind == "video" else
             " (pusher log)" if req.domain == "factory" else "")
        if s.session is not None:
            s.session.name = s.name
        return register(s)

    @app.post("/api/studio/sessions/{sid}/simulate-more")
    def simulate_more(sid: str, req: MorePushesRequest):
        """Run the suggested next experiment in the same simulated world: a new video take or more log rows."""
        from studio import simulate as sm
        from studio.session import Session, SessionError

        s = get(sid)
        if s.world is None:
            raise HTTPException(422, "Only simulated sessions can run pushes in Newton; film or log them for real data.")
        if s.busy:
            raise HTTPException(429, "This session is being analysed or retrained; wait for it to finish.")
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
                meta = sm.render_video(s.world, speeds, s.dir / f"take{i}.mp4", seed=97 + i, domain=s.domain)
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
        s.result, s.cal, s.training = None, None, None
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
        if s.busy:
            raise HTTPException(429, "This session is being analysed or retrained; wait for it to finish.")
        try:
            sess, dbg = track_video(Path(v["path"]), req.corners, req.sheet, s.name, req.object_px, height, prior=prior)
        except SessionError as e:
            raise HTTPException(422, str(e)) from None
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"Tracking failed ({type(e).__name__}). Check the sheet corners and try again.") from None
        s.result, s.cal, s.training = None, None, None  # the data changed: the old analysis no longer applies
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
            s.session.meta["domain"] = s.domain
            res = analyze(s.session, ag, on_event=s.push)
            doc = res.to_json()
            if s.cams:
                doc["overlay"] = overlay(s, res.calibration)
                s.push({"type": "overlay", "overlay": doc["overlay"]})
            doc["report_json"] = export.to_json(s.session, res.calibration, res.next_experiment, res.agent)
            s.cal = res.calibration
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
        with s.cond:
            if s.busy:
                raise HTTPException(429, "This session is already being analysed or retrained.")
            s.events, s.done = [], False
        threading.Thread(target=worker, args=(s, req.agent), daemon=True).start()
        return {"ok": True, "agent": bool(req.agent and llm is not None)}

    @app.post("/api/studio/sessions/{sid}/verify")
    def verify_route(sid: str):
        """Close the loop: the exported parameters, replayed push by push in NVIDIA Newton, against the measured stops."""
        from studio.verify import verify

        s = get(sid)
        if s.result is None or s.session is None or s.cal is None:
            raise HTTPException(422, "Run the analysis first.")
        out = newton(verify, s.session, s.cal)
        s.result["verify"] = out
        s.result["report_json"]["newton_replay"] = out
        return out

    chat_counts: dict[str, int] = {}

    @app.post("/api/studio/chat")
    def chat(req: ChatRequest, request: Request):
        """Questions about the Studio and the visitor's current result, answered by Nemotron in EN or KO."""
        from studio import chat as ch

        if llm is None:
            raise HTTPException(503, "Ask needs a language model: start the server with TETHER_LLM=tokenfactory.")
        last = req.messages[-1]
        if last.get("role") != "user" or not str(last.get("content") or "").strip():
            raise HTTPException(422, "The last message must be your question.")
        key = req.sid if req.sid in sessions else f"ip:{request.client.host if request.client else '-'}"
        with store_lock:
            if chat_counts.get(key, 0) >= CHAT_CAP:
                raise HTTPException(429, f"This session reached {CHAT_CAP} questions. Start a new session to ask more.")
            chat_counts[key] = chat_counts.get(key, 0) + 1
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
        return {"answer": text, "left": CHAT_CAP - chat_counts[key]}

    @app.get("/api/studio/sessions/{sid}/events")
    def events(sid: str):
        from server.sse import event_stream

        return event_stream(get(sid))

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
        if "carla" in s.result["exports"]:
            files["carla"] = ("carla_calibration.py", s.result["exports"]["carla"], "text/x-python")
        if fmt not in files:
            raise HTTPException(404, f"Formats: {', '.join(files)}, csv.")
        name, text, mime = files[fmt]
        return PlainTextResponse(text, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/studio/sessions/{sid}/truth")
    def truth(sid: str):
        s = get(sid)
        if not s.sample and s.world is None:
            raise HTTPException(404, "Only samples and simulated tables have a known ground truth.")
        if s.result is None:
            raise HTTPException(409, "Run the analysis first: the hidden physics is revealed afterwards.")
        out = hidden_truth(s)
        s.truth_shown = out["model"]
        return out

    @app.post("/api/studio/sessions/{sid}/train")
    def train_start(sid: str):
        """Retrain a policy on the current sim, wide randomization and Tether's measured ranges (parallel Newton
        worlds); poll GET for progress. The hidden world, when known, only scores the policies."""
        from studio.train import hidden_params, train

        s = get(sid)
        if s.cal is None:
            raise HTTPException(422, "Run the analysis first.")
        with s.cond:
            if s.training and s.training["status"] == "running":
                return s.training
            if not s.done:
                raise HTTPException(429, "Wait for the analysis to finish.")
            s.training = {"status": "running", "progress": [], "result": None}
        job = s.training
        h = hidden_truth(s)
        hidden = hidden_params(h["model"] if h else None, s.session)

        def run():
            if not sim_lock.acquire(timeout=600):
                job["status"], job["error"] = "error", "Newton is busy with other visitors. Try again in a minute."
                return
            try:
                job["result"] = train(s.session, s.cal, hidden, on_iter=job["progress"].append)
                job["status"] = "done"
            except Exception as e:  # noqa: BLE001
                job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
            finally:
                sim_lock.release()
            if job["status"] == "done" and s.result is not None:
                s.result["training"] = job["result"]

        threading.Thread(target=run, daemon=True).start()
        return job

    @app.post("/api/studio/sessions/{sid}/train/video")
    def train_video(sid: str):
        """Render the three trained policies acting in the hidden world, side by side (NVIDIA Newton, ~20 s)."""
        from studio.rollout_video import render_rollout

        s = get(sid)
        if not (s.training and s.training.get("result")):
            raise HTTPException(422, "Retrain first.")
        h = hidden_truth(s)
        if h is None:
            raise HTTPException(422, "Only samples and simulated data have a hidden world to show the policies in.")
        out = newton(render_rollout, s.session, h["model"], s.training["result"], s.domain, s.dir / "rollout.mp4")
        return {"file": f"files/{s.id}/rollout.mp4", "hits": out["hits"]}

    @app.get("/api/studio/sessions/{sid}/train")
    def train_state(sid: str):
        s = get(sid)
        return s.training or {"status": "idle", "progress": [], "result": None}

    @app.get("/api/studio/files/{sid}/{name}")
    def files(sid: str, name: str):
        s = get(sid)
        path = (s.dir / Path(name).name).resolve()
        if path.parent != s.dir.resolve() or not path.exists() or path.name.startswith("upload"):
            raise HTTPException(404, "No such file.")
        return FileResponse(path)

