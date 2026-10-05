"""GapCloser live server: the dashboard plus an API that runs the agent on a user-chosen hidden world.

  GET  /                       dashboard (live mode: loads recorded runs, can start new runs)
  GET  /api/bundle             recorded scenarios + benchmark (runs/demo/bundle.json)
  GET  /api/status             provider, model, budget, whether a run is in progress
  POST /api/runs               {"hidden": {param: value}, "title": str?} -> {"id": ...}
  GET  /api/runs/{id}/events   Server-Sent Events: start, train, measure, diagnose, plan, done, end
  GET  /clips/... , /live/...  rendered clips

Cost guards (public demo, Token Factory credits are finite): one run at a time, per-IP hourly limit,
a global cap on LLM calls (then the rule-based diagnoser takes over), bounded parameters, max 4 iterations.

Config (env): GAPCLOSER_LLM=local|tokenfactory|none, GAPCLOSER_MAX_LLM_CALLS (200), GAPCLOSER_RUNS_PER_HOUR (6),
GAPCLOSER_SERVER_ENV=newton|analytic, GAPCLOSER_RENDER=1|0, GAPCLOSER_DATA (runs/).
Run: make serve  ->  http://localhost:8000
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sim.params import PARAM_SPACE
from sim.push_task import eval_targets

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("GAPCLOSER_DATA", ROOT / "runs"))
DEMO = DATA / "demo"
LIVE = DATA / "live"
MAX_ITER = 4
MAX_HIDDEN = 3


class Budget:
    """Global LLM-call cap shared by all runs; past the cap, runs use the rule-based diagnoser."""

    def __init__(self, max_calls: int):
        self.max_calls, self.used = max_calls, 0
        self.lock = threading.Lock()

    def take(self) -> bool:
        with self.lock:
            if self.used >= self.max_calls:
                return False
            self.used += 1
            return True

    @property
    def left(self) -> int:
        return max(0, self.max_calls - self.used)


class BudgetedLLM:
    """Wraps an LLM; refuses (raises) once the budget is spent so LLMDiagnoser falls back cleanly."""

    def __init__(self, inner, budget: Budget):
        self.inner, self.budget = inner, budget
        self.usage = getattr(inner, "usage", None)

    def complete(self, role, messages, schema=None):
        if not self.budget.take():
            raise RuntimeError("LLM budget for this demo is used up")
        return self.inner.complete(role, messages, schema)


class RunRequest(BaseModel):
    hidden: dict[str, float] = Field(..., description="parameter -> hidden real value")
    title: str | None = Field(None, max_length=60)


class Run:
    def __init__(self, rid: str, title: str, hidden: dict[str, float]):
        self.id, self.title, self.hidden = rid, title, hidden
        self.events: list[dict] = []
        self.done = False
        self.cond = threading.Condition()

    def push(self, e: dict) -> None:
        with self.cond:
            self.events.append(e)
            if e["type"] in ("end", "error"):
                self.done = True
            self.cond.notify_all()


def create_app(llm=None, env_name: str | None = None, render: bool | None = None, max_llm_calls: int | None = None,
               runs_per_hour: int | None = None, data_dir: Path | None = None) -> FastAPI:
    env_name = env_name or os.environ.get("GAPCLOSER_SERVER_ENV", "newton")
    render = (os.environ.get("GAPCLOSER_RENDER", "1") == "1") if render is None else render
    if runs_per_hour is None:
        runs_per_hour = int(os.environ.get("GAPCLOSER_RUNS_PER_HOUR", "6"))
    if max_llm_calls is None:
        max_llm_calls = int(os.environ.get("GAPCLOSER_MAX_LLM_CALLS", "200"))
    budget = Budget(max_llm_calls)
    data = Path(data_dir) if data_dir else DATA
    demo, live = data / "demo", data / "live"
    live.mkdir(parents=True, exist_ok=True)

    provider = os.environ.get("GAPCLOSER_LLM", "local")
    if llm is None and provider != "none":
        try:
            from agent.llm import make_llm

            llm = make_llm(provider)
            llm.model_for("diagnose")  # fail fast if the provider is unreachable
        except Exception as e:  # noqa: BLE001 - the demo still works with the rule-based diagnoser
            print(f"[gapcloser] LLM unavailable ({e}); using TrajectoryDiagnoser")
            llm = None

    app = FastAPI(title="GapCloser", docs_url="/api/docs", openapi_url="/api/openapi.json")
    runs: dict[str, Run] = {}
    busy = threading.Lock()
    hits: dict[str, deque] = defaultdict(deque)
    targets = eval_targets(20, 1000)

    def model_name() -> str | None:
        try:
            return llm.model_for("diagnose") if llm is not None and hasattr(llm, "model_for") else (getattr(llm, "name", None))
        except Exception:  # noqa: BLE001
            return None

    def make_env():
        if env_name == "analytic":
            from sim.push_task import AnalyticPushEnv

            return AnalyticPushEnv()
        from sim.newton_push import NewtonPushEnv

        return NewtonPushEnv()

    def worker(run: Run) -> None:
        from agent.llm_diagnoser import LLMDiagnoser
        from agent.loop import TrajectoryDiagnoser
        from eval.record_demo import record_scenario

        try:
            diag = LLMDiagnoser(BudgetedLLM(llm, budget)) if llm is not None else TrajectoryDiagnoser()
            out = live / run.id
            out.mkdir(parents=True, exist_ok=True)

            def on_event(e: dict) -> None:
                if e.get("clip"):
                    for k in ("real", "sim"):
                        e["clip"][k] = f"live/{run.id}/{e['clip'][k]}"
                run.push(e)

            sc = {"id": run.id, "title": run.title, "hidden": run.hidden}
            def save_then_emit(e: dict) -> None:
                if e["type"] == "end":  # persist before the stream closes
                    (out / "run.json").write_text(json.dumps({**e["run"], "events": [x for x in run.events if x["type"] not in ("start",)]}, indent=1))
                on_event(e)

            record_scenario(sc, out, targets, diag, env=make_env(), render=render, on_event=save_then_emit,
                            max_iter=MAX_ITER)
        except Exception as e:  # noqa: BLE001
            run.push({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            busy.release()

    @app.get("/api/status")
    def status():
        return {"provider": provider if llm is not None else "none", "model": model_name(),
                "llm_calls_left": budget.left, "busy": busy.locked(), "env": env_name, "render": render,
                "max_iter": MAX_ITER, "max_hidden": MAX_HIDDEN, "runs_per_hour": runs_per_hour,
                "params": {k: {"nominal": p.nominal, "low": p.low, "high": p.high, "unit": p.unit, "kind": p.kind}
                           for k, p in PARAM_SPACE.items()}}

    @app.get("/api/bundle")
    def bundle():
        path = demo / "bundle.json"
        if not path.exists():
            raise HTTPException(404, "No recorded runs yet. Run `make demo` first.")
        return FileResponse(path, media_type="application/json")

    @app.post("/api/runs")
    def start_run(req: RunRequest, request: Request):
        hidden = {}
        for k, v in req.hidden.items():
            if k not in PARAM_SPACE:
                raise HTTPException(422, f"Unknown parameter '{k}'.")
            p = PARAM_SPACE[k]
            if not p.low <= v <= p.high:
                raise HTTPException(422, f"{k} must be between {p.low} and {p.high}.")
            if abs(v - p.nominal) > 1e-9:
                hidden[k] = float(v)
        if not hidden:
            raise HTTPException(422, "Change at least one parameter away from its nominal value.")
        if len(hidden) > MAX_HIDDEN:
            raise HTTPException(422, f"Change at most {MAX_HIDDEN} parameters per run.")
        ip = request.client.host if request.client else "?"
        now, q = time.time(), hits[ip]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= runs_per_hour:
            raise HTTPException(429, f"Limit of {runs_per_hour} runs per hour reached. Try again later.")
        if not busy.acquire(blocking=False):
            raise HTTPException(429, "Another run is in progress. Watch it or try again in a minute.")
        q.append(now)
        rid = "live-" + uuid.uuid4().hex[:8]
        run = Run(rid, req.title or "Custom world", hidden)
        runs[rid] = run
        threading.Thread(target=worker, args=(run,), daemon=True).start()
        return {"id": rid}

    @app.get("/api/runs/{rid}/events")
    def events(rid: str):
        run = runs.get(rid)
        if run is None:
            raise HTTPException(404, "Unknown run.")

        def stream():
            i = 0
            while True:
                with run.cond:
                    while i >= len(run.events) and not run.done:
                        if not run.cond.wait(timeout=15):
                            break
                    batch, done = run.events[i:], run.done
                if not batch and not done:
                    yield ": keep-alive\n\n"
                    continue
                for e in batch:
                    yield f"data: {json.dumps(e)}\n\n"
                i += len(batch)
                if done and i >= len(run.events):
                    return

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/runs")
    def list_runs():
        """Finished live runs (newest last, at most 10), in the same shape as recorded scenarios."""
        done = sorted(live.glob("*/run.json"), key=lambda p: p.stat().st_mtime)[-10:]
        out = []
        for p in done:
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    @app.get("/api/runs/{rid}")
    def run_json(rid: str):
        run = runs.get(rid)
        if run is None:
            raise HTTPException(404, "Unknown run.")
        return JSONResponse({"id": rid, "done": run.done, "events": run.events})

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = ROOT / "dashboard" / "dist" / "gapcloser.live.html"
        if not page.exists():
            raise HTTPException(404, "Dashboard not built. Run `make dashboard`.")
        return HTMLResponse(page.read_text())

    if (demo / "clips").exists():
        app.mount("/clips", StaticFiles(directory=demo / "clips"), name="clips")
    app.mount("/live", StaticFiles(directory=live), name="live")
    return app


app = create_app() if os.environ.get("GAPCLOSER_NO_AUTOAPP") != "1" else None
