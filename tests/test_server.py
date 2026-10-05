import json
import os
from pathlib import Path

os.environ["GAPCLOSER_NO_AUTOAPP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from agent.llm import RecordedLLM  # noqa: E402
from server.app import create_app  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "nemotron_weak_motor.json"


def client(tmp_path, llm=None, **kw):
    (tmp_path / "demo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "demo" / "bundle.json").write_text(json.dumps({"runs": [], "benchmark": None}))
    os.environ["GAPCLOSER_LLM"] = "none"
    app = create_app(llm=llm, env_name="analytic", render=False, data_dir=tmp_path, **kw)
    return TestClient(app)


def read_events(c, rid):
    with c.stream("GET", f"/api/runs/{rid}/events") as r:
        body = "".join(r.iter_text())
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_status_and_bundle(tmp_path):
    c = client(tmp_path)
    st = c.get("/api/status").json()
    assert st["provider"] == "none" and st["max_hidden"] == 3 and "actuator_gain" in st["params"]
    assert c.get("/api/bundle").json() == {"runs": [], "benchmark": None}


def test_live_run_streams_agent_steps_and_closes_gap(tmp_path):
    c = client(tmp_path, llm=RecordedLLM.from_file(FIXTURE))
    rid = c.post("/api/runs", json={"hidden": {"actuator_gain": 0.76}}).json()["id"]
    ev = read_events(c, rid)
    types = [e["type"] for e in ev]
    assert types[0] == "start" and types[-1] == "end"
    assert {"train", "measure", "diagnose", "plan", "done"} <= set(types)
    diag = next(e for e in ev if e["type"] == "diagnose")
    assert diag["agent"].startswith("nemotron") and diag["suspects"][0]["name"] == "actuator_gain"
    end = ev[-1]["run"]
    assert end["final_success"] >= 0.9 and end["baselines"]["outcome_only"] < 0.5
    assert (tmp_path / "live" / rid / "run.json").exists()
    listed = c.get("/api/runs").json()
    assert listed[-1]["id"] == rid and listed[-1]["events"][0]["type"] == "train"


def test_rejects_bad_requests(tmp_path):
    c = client(tmp_path)
    assert c.post("/api/runs", json={"hidden": {"gravity": 1}}).status_code == 422
    assert c.post("/api/runs", json={"hidden": {"object_mu": 9}}).status_code == 422
    assert c.post("/api/runs", json={"hidden": {"object_mu": 0.8}}).status_code == 422  # nominal = no change
    four = {"object_mu": 0.3, "table_mu": 0.3, "actuator_gain": 0.9, "camera_dx": 0.01}
    assert c.post("/api/runs", json={"hidden": four}).status_code == 422


def test_rate_limit_per_ip(tmp_path):
    c = client(tmp_path, runs_per_hour=1)
    rid = c.post("/api/runs", json={"hidden": {"object_mu": 0.3}}).json()["id"]
    read_events(c, rid)  # let it finish so the busy lock is free
    r = c.post("/api/runs", json={"hidden": {"object_mu": 0.4}})
    assert r.status_code == 429 and "per hour" in r.json()["detail"]


def test_llm_budget_exhaustion_falls_back(tmp_path):
    c = client(tmp_path, llm=RecordedLLM.from_file(FIXTURE), max_llm_calls=0)
    rid = c.post("/api/runs", json={"hidden": {"actuator_gain": 0.76}}).json()["id"]
    ev = read_events(c, rid)
    diag = next(e for e in ev if e["type"] == "diagnose")
    assert "fallback" in diag["agent"]
    assert ev[-1]["run"]["final_success"] >= 0.9
