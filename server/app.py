"""Tether live server: the dashboard plus an API that runs the agent on a user-chosen hidden world.

  GET  /                       dashboard (live mode: loads recorded runs, can start new runs)
  GET  /api/bundle             recorded scenarios + benchmark (runs/demo/bundle.json)
  GET  /api/status             provider, model, budget, whether a run is in progress, presets
  GET  /api/surprise           a random hidden world inside the same bounds the server accepts
  POST /api/runs               {"hidden": {param: value}, "title": str?} -> {"id": ..., "open": bool}
  GET  /api/runs/{id}/events   Server-Sent Events: start, train, measure, agent_step*, diagnose, plan, done, end
                               (agent_step = one tool call of the Nemotron agent, streamed as it happens)
  GET  /clips/... , /live/...  rendered clips
  GET  /replay/...             3D viewer replays (per-frame Newton poses)
  /studio, /api/studio/*       Tether Studio: calibrate from your own video or robot log (server/studio_api.py)

"Stump the agent": with an LLM configured, every live run takes the open-world path (inverse policy,
Nemotron tool agent that inspects evidence, fits model structures, probes the real robot, commits a
simulator). The visitor may hide open-world faults (a friction strip = patch_y0 + patch_mu, lens
distortion) the rule book has no parameter for. Without an LLM, closed worlds use the rule-based
TrajectoryDiagnoser as before and open worlds run the open path with it (it cannot model them).

Cost guards (public demo, Token Factory credits are finite): one run at a time, per-IP hourly limit,
a global cap on LLM calls and a per-run cap on agent chat turns (past either, the rule-based diagnoser
takes over), bounded parameters (at most 3 hidden faults), max 4 iterations.

Config (env): TETHER_LLM=local|tokenfactory|none, TETHER_MAX_LLM_CALLS (200),
TETHER_MAX_TURNS_PER_RUN (24), TETHER_RUNS_PER_HOUR (6),
TETHER_SERVER_ENV=newton|analytic, TETHER_RENDER=1|0, TETHER_DATA (runs/),
TETHER_EYES=none|cosmos (+ TETHER_EYES_URL, default http://localhost:8080; local Cosmos Reason 2).
Run: make serve  ->  http://localhost:8000
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sim.params import OPEN_PARAMS, PARAM_SPACE
from sim.push_task import PATCH_OFF, eval_targets
from sim.settings import setting

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(setting("DATA", ROOT / "runs"))
DEMO = DATA / "demo"
LIVE = DATA / "live"
MAX_ITER = 4
MAX_HIDDEN = 3  # hidden faults per run; a friction strip (patch_y0 + patch_mu) counts as one
PATCH_MU_MAX = 0.95  # stickier strips tip the cube over (outside every model, including the agent's)
AGENT_STEPS = 8  # tool-agent turns per diagnosis (the last one is forced to commit)
STRIP = ("patch_y0", "patch_mu")

# "Stump the agent" presets (same worlds as the recorded open-world scenarios)
PRESETS = [
    {"id": "wet-strip", "title": "Wet strip", "hidden": {"patch_y0": 0.35, "patch_mu": 0.4}},
    {"id": "rough-strip", "title": "Rough strip", "hidden": {"table_mu": 0.5, "patch_y0": 0.3, "patch_mu": 0.9}},
    {"id": "fisheye", "title": "Lens distortion", "hidden": {"lens_k": -0.3, "light_intensity": 0.7}},
    {"id": "three-faults", "title": "Three faults", "hidden": {"actuator_gain": 0.85, "patch_y0": 0.4, "patch_mu": 0.5, "lens_k": -0.15}},
]


def faults(hidden: dict[str, float]) -> list[str]:
    """Distinct hidden faults: each parameter is one, except the friction strip (patch_y0 + patch_mu)."""
    return list(dict.fromkeys("friction_strip" if k in STRIP else k for k in hidden))


def is_open(hidden: dict[str, float]) -> bool:
    return any(k in OPEN_PARAMS for k in hidden)


def validate_hidden(raw: dict[str, float]) -> dict[str, float]:
    """Hidden world as the server will run it, or ValueError with a message for the visitor."""
    hidden = {}
    for k, v in raw.items():
        if k not in PARAM_SPACE:
            raise ValueError(f"Unknown parameter '{k}'.")
        p = PARAM_SPACE[k]
        if not isinstance(v, (int, float)) or v != v or not p.low <= v <= p.high:
            raise ValueError(f"{k} must be between {p.low} and {p.high}.")
        if abs(v - p.nominal) > 1e-9:
            hidden[k] = float(v)
    if "patch_mu" in hidden and hidden.get("patch_y0", PATCH_OFF) >= PATCH_OFF:
        raise ValueError("A friction strip needs a start: set patch_y0 below 1.0 m together with patch_mu.")
    if "patch_y0" in hidden:  # the strip's friction always travels with it (nominal 0.8 = same as the table)
        hidden.setdefault("patch_mu", PARAM_SPACE["patch_mu"].nominal)
    if hidden.get("patch_mu", 0.0) > PATCH_MU_MAX:
        raise ValueError(f"patch_mu must be at most {PATCH_MU_MAX} (stickier strips tip the cube over).")
    if not hidden:
        raise ValueError("Change at least one parameter away from its nominal value.")
    if len(faults(hidden)) > MAX_HIDDEN:
        raise ValueError(f"Hide at most {MAX_HIDDEN} faults per run (a friction strip counts as one).")
    return hidden


def surprise(rng: random.Random) -> dict[str, float]:
    """A random hidden world: 1-3 faults, at least one open-world, all inside the accepted bounds."""
    def away(lo_band, hi_band):
        return round(rng.uniform(*(lo_band if rng.random() < 0.5 else hi_band)), 3)

    pool = {
        "friction_strip": lambda: {"patch_y0": round(rng.uniform(0.25, 0.55), 3),
                                   "patch_mu": round(rng.uniform(0.3, 0.6) if rng.random() < 0.7 else rng.uniform(0.9, PATCH_MU_MAX), 3)},
        "lens_k": lambda: {"lens_k": away((-0.3, -0.12), (0.12, 0.3))},
        "actuator_gain": lambda: {"actuator_gain": away((0.75, 0.9), (1.1, 1.25))},
        "table_mu": lambda: {"table_mu": round(rng.uniform(0.4, 0.65), 3)},
        "camera_dx": lambda: {"camera_dx": away((-0.03, -0.012), (0.012, 0.03))},
        "light_intensity": lambda: {"light_intensity": away((0.5, 0.75), (1.25, 1.5))},
    }
    first = rng.choice(["friction_strip", "lens_k"])
    rest = rng.sample(sorted(k for k in pool if k != first), rng.randint(0, MAX_HIDDEN - 1))
    hidden: dict[str, float] = {}
    for k in [first, *rest]:
        hidden.update(pool[k]())
    return validate_hidden(hidden)


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
    """Wraps an LLM; refuses (raises) once the global budget or this run's turn cap is spent, so the
    diagnosers fall back cleanly. Every complete() and every chat() turn counts as one call."""

    def __init__(self, inner, budget: Budget, run_cap: int | None = None):
        self.inner, self.budget, self.run_cap = inner, budget, run_cap
        self.run_used = 0
        self.usage = getattr(inner, "usage", None)

    @property
    def run_left(self) -> int:
        left = self.budget.left
        return left if self.run_cap is None else min(left, max(0, self.run_cap - self.run_used))

    def _take(self) -> None:
        if self.run_cap is not None and self.run_used >= self.run_cap:
            raise RuntimeError(f"agent turn cap for this run ({self.run_cap}) is used up")
        if not self.budget.take():
            raise RuntimeError("LLM budget for this demo is used up")
        self.run_used += 1

    def complete(self, role, messages, schema=None):
        self._take()
        return self.inner.complete(role, messages, schema)

    def chat(self, role, messages, tools, tool_choice=None):
        self._take()
        return self.inner.chat(role, messages, tools, tool_choice)


class CappedToolAgent:
    """ToolAgentDiagnoser whose per-diagnosis step budget shrinks to the turns this run has left, so the
    agent is asked to commit on its last affordable turn instead of being cut off mid-experiment."""

    def __init__(self, agent, llm: BudgetedLLM, steps: int = AGENT_STEPS):
        self.agent, self.llm, self.steps = agent, llm, steps

    @property
    def camera_events(self):  # forwarded so record_scenario can hand Cosmos eyes evidence to the inner agent
        return self.agent.camera_events

    @camera_events.setter
    def camera_events(self, fn):
        self.agent.camera_events = fn

    def diagnose(self, real, sim, sim_params):
        self.agent.max_steps = max(1, min(self.steps, self.llm.run_left))
        return self.agent.diagnose(real, sim, sim_params)


class RunRequest(BaseModel):
    hidden: dict[str, float] = Field(..., description="parameter -> hidden real value")
    title: str | None = Field(None, max_length=60)


class Run:
    def __init__(self, rid: str, title: str, hidden: dict[str, float], open_: bool = False):
        self.id, self.title, self.hidden, self.open = rid, title, hidden, open_
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
               runs_per_hour: int | None = None, data_dir: Path | None = None,
               max_turns_per_run: int | None = None) -> FastAPI:
    env_name = env_name or setting("SERVER_ENV", "newton")
    render = (setting("RENDER", "1") == "1") if render is None else render
    if runs_per_hour is None:
        runs_per_hour = int(setting("RUNS_PER_HOUR", "6"))
    if max_llm_calls is None:
        max_llm_calls = int(setting("MAX_LLM_CALLS", "200"))
    if max_turns_per_run is None:
        max_turns_per_run = int(setting("MAX_TURNS_PER_RUN", "24"))
    budget = Budget(max_llm_calls)
    data = Path(data_dir) if data_dir else DATA
    demo, live = data / "demo", data / "live"
    live.mkdir(parents=True, exist_ok=True)

    provider = setting("LLM", "local")
    if llm is None and provider != "none":
        try:
            from agent.llm import make_llm

            llm = make_llm(provider)
            llm.model_for("diagnose")  # fail fast if the provider is unreachable
        except Exception as e:  # noqa: BLE001 - the demo still works with the rule-based diagnoser
            print(f"[tether] LLM unavailable ({e}); using TrajectoryDiagnoser")
            llm = None

    eyes = None
    if setting("EYES", "none") == "cosmos":  # optional local Cosmos Reason 2 (llama-server)
        from agent.cosmos_eyes import CosmosEyes

        eyes = CosmosEyes(setting("EYES_URL", "http://localhost:8080"))
        if not eyes.available():
            print(f"[tether] Cosmos eyes unavailable ({eyes.last_error}); running without them")
            eyes = None

    app = FastAPI(title="Tether", docs_url="/api/docs", openapi_url="/api/openapi.json")
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
        from agent.loop import TrajectoryDiagnoser
        from agent.tool_agent import ToolAgentDiagnoser
        from eval.record_demo import record_scenario

        try:
            out = live / run.id
            out.mkdir(parents=True, exist_ok=True)
            t0, cur = time.perf_counter(), {"iter": 0}

            def on_step(step: dict) -> None:  # one tool call of the agent, as it happens
                run.push({"type": "agent_step", "t": round(time.perf_counter() - t0, 3), "iter": cur["iter"], "step": step})

            make_diag = None
            if run.open and llm is not None:
                bllm = BudgetedLLM(llm, budget, max_turns_per_run)
                make_diag = lambda real: CappedToolAgent(ToolAgentDiagnoser(bllm, real, on_step=on_step), bllm)  # noqa: E731

            def on_event(e: dict) -> None:
                if e["type"] == "measure":
                    cur["iter"] = e["iter"]
                if e.get("clip"):
                    for k in ("real", "sim", "replay"):
                        if e["clip"].get(k):
                            e["clip"][k] = f"live/{run.id}/{e['clip'][k]}"
                run.push(e)

            sc = {"id": run.id, "title": run.title, "hidden": run.hidden, "open": run.open}

            def save_then_emit(e: dict) -> None:
                if e["type"] == "end":  # persist before the stream closes
                    keep = [x for x in run.events if x["type"] not in ("start", "agent_step")]
                    (out / "run.json").write_text(json.dumps({**e["run"], "events": keep}, indent=1))
                on_event(e)

            record_scenario(sc, out, targets, TrajectoryDiagnoser(), env=make_env(), render=render,
                            on_event=save_then_emit, max_iter=MAX_ITER, make_diagnoser=make_diag,
                            eyes=eyes if render else None)
        except Exception as e:  # noqa: BLE001
            run.push({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            busy.release()

    @app.get("/api/status")
    def status():
        return {"provider": provider if llm is not None else "none", "model": model_name(),
                "llm_calls_left": budget.left, "busy": busy.locked(), "env": env_name, "render": render,
                "max_iter": MAX_ITER, "max_hidden": MAX_HIDDEN, "runs_per_hour": runs_per_hour,
                "agent": "tool" if llm is not None else "rule", "max_turns_per_run": max_turns_per_run,
                "patch_mu_max": PATCH_MU_MAX, "presets": PRESETS,
                "params": {k: {"nominal": p.nominal, "low": p.low, "high": p.high, "unit": p.unit, "kind": p.kind}
                           for k, p in PARAM_SPACE.items()}}

    @app.get("/api/bundle")
    def bundle():
        path = demo / "bundle.json"
        if not path.exists():
            raise HTTPException(404, "No recorded runs yet. Run `make demo` first.")
        return FileResponse(path, media_type="application/json")

    @app.get("/api/surprise")
    def surprise_world(seed: int | None = None):
        return {"title": "Surprise world", "hidden": surprise(random.Random(seed))}

    @app.post("/api/runs")
    def start_run(req: RunRequest, request: Request):
        try:
            hidden = validate_hidden(req.hidden)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
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
        run = Run(rid, req.title or "Custom world", hidden, open_=llm is not None or is_open(hidden))
        runs[rid] = run
        threading.Thread(target=worker, args=(run,), daemon=True).start()
        return {"id": rid, "open": run.open, "agent": "tool" if run.open and llm is not None else "rule"}

    @app.get("/api/runs/{rid}/events")
    def events(rid: str):
        run = runs.get(rid)
        if run is None:
            raise HTTPException(404, "Unknown run.")
        from server.sse import event_stream

        return event_stream(run)

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
    def home():
        """Overview: what Tether does, the route Console -> Studio -> Export, the three domains."""
        from dashboard.build import with_shared

        return HTMLResponse(with_shared((ROOT / "dashboard" / "home.html").read_text()))

    @app.get("/console", response_class=HTMLResponse)
    def console():
        page = ROOT / "dashboard" / "dist" / "console.live.html"
        if not page.exists():
            raise HTTPException(404, "Dashboard not built. Run `make dashboard`.")
        return HTMLResponse(page.read_text())

    from server.studio_api import mount_studio

    mount_studio(app, llm, budget, data, BudgetedLLM)

    if (demo / "clips").exists():
        app.mount("/clips", StaticFiles(directory=demo / "clips"), name="clips")
    if (demo / "replay").exists():  # per-frame Newton poses for the 3D viewer
        app.mount("/replay", StaticFiles(directory=demo / "replay"), name="replay")
    app.mount("/live", StaticFiles(directory=live), name="live")
    return app


app = create_app() if setting("NO_AUTOAPP") != "1" else None
