import json
import os
from pathlib import Path

os.environ["TETHER_NO_AUTOAPP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from agent.llm import RecordedLLM  # noqa: E402
from server.app import MAX_HIDDEN, PRESETS, create_app, faults, surprise, validate_hidden  # noqa: E402

TOOL_FIXTURE = Path(__file__).parent / "fixtures" / "tool_agent_wet_strip.json"
WET = {"patch_y0": 0.35, "patch_mu": 0.45}


def client(tmp_path, llm=None, **kw):
    (tmp_path / "demo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "demo" / "bundle.json").write_text(json.dumps({"runs": [], "benchmark": None}))
    os.environ["TETHER_LLM"] = "none"
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
    assert st["agent"] == "rule" and {p["title"] for p in st["presets"]} >= {"Wet strip", "Three faults"}
    assert c.get("/api/bundle").json() == {"runs": [], "benchmark": None}


def test_rule_based_live_run_streams_steps_and_closes_closed_gap(tmp_path):
    c = client(tmp_path)  # no LLM: closed worlds keep the rule-based path
    r = c.post("/api/runs", json={"hidden": {"actuator_gain": 0.76}}).json()
    assert r["open"] is False and r["agent"] == "rule"
    ev = read_events(c, r["id"])
    types = [e["type"] for e in ev]
    assert types[0] == "start" and types[-1] == "end"
    assert {"train", "measure", "diagnose", "plan", "done"} <= set(types)
    end = ev[-1]["run"]
    assert end["final_success"] >= 0.9 and "outcome_only" in end["baselines"]
    assert (tmp_path / "live" / r["id"] / "run.json").exists()
    listed = c.get("/api/runs").json()
    assert listed[-1]["id"] == r["id"] and listed[-1]["events"][0]["type"] == "train"


def test_open_world_run_streams_tool_agent_steps_and_closes_gap(tmp_path):
    llm = RecordedLLM.from_file(TOOL_FIXTURE)
    c = client(tmp_path, llm=llm)
    r = c.post("/api/runs", json={"hidden": WET, "title": "Wet strip"}).json()
    assert r["open"] is True and r["agent"] == "tool"
    ev = read_events(c, r["id"])
    assert ev[0]["type"] == "start" and ev[0]["open"] is True and ev[-1]["type"] == "end"
    steps = [e for e in ev if e["type"] == "agent_step"]
    assert [e["step"]["tool"] for e in steps] == ["decel_profile", "fit_hypothesis", "commit"]
    diag_i = next(i for i, e in enumerate(ev) if e["type"] == "diagnose")
    assert all(ev.index(e) < diag_i and e["iter"] == 0 for e in steps)  # streamed before the diagnosis lands
    diag = ev[diag_i]
    assert diag["agent"] == "nemotron-3-super-scripted"
    assert [t["tool"] for t in diag["trace"]] == ["decel_profile", "fit_hypothesis", "commit"]
    assert {s["name"] for s in diag["suspects"]} == {"patch_y0", "patch_mu"}
    run = ev[-1]["run"]
    assert run["open"] is True and run["title"] == "Wet strip"
    assert run["final_success"] >= 0.9 and run["baselines"]["rule"] < 0.9 and run["baselines"]["real_trials"] > 0
    assert llm.usage.calls == 3
    saved = json.loads((tmp_path / "live" / r["id"] / "run.json").read_text())
    assert saved["title"] == "Wet strip" and not any(e["type"] == "agent_step" for e in saved["events"])
    assert c.get("/api/status").json()["llm_calls_left"] == 197


def test_rejects_bad_requests(tmp_path):
    c = client(tmp_path)
    assert c.post("/api/runs", json={"hidden": {"gravity": 1}}).status_code == 422
    assert c.post("/api/runs", json={"hidden": {"object_mu": 9}}).status_code == 422
    assert c.post("/api/runs", json={"hidden": {"object_mu": 0.8}}).status_code == 422  # nominal = no change
    four = {"object_mu": 0.3, "table_mu": 0.3, "actuator_gain": 0.9, "camera_dx": 0.01}
    assert c.post("/api/runs", json={"hidden": four}).status_code == 422
    assert c.post("/api/runs", json={"hidden": {"patch_y0": 0.35, "patch_mu": 0.97}}).status_code == 422  # tips over
    assert c.post("/api/runs", json={"hidden": {"patch_y0": 0.1, "patch_mu": 0.4}}).status_code == 422  # below bound
    assert c.post("/api/runs", json={"hidden": {"lens_k": 0.5}}).status_code == 422
    r = c.post("/api/runs", json={"hidden": {"patch_mu": 0.4}})  # a strip needs a start
    assert r.status_code == 422 and "patch_y0" in r.json()["detail"]
    r = c.post("/api/runs", json={"hidden": {**WET, "lens_k": 0.2, "actuator_gain": 0.9, "table_mu": 0.5}})
    assert r.status_code == 422 and "faults" in r.json()["detail"]


def test_validation_counts_a_friction_strip_as_one_fault():
    three = validate_hidden({"actuator_gain": 0.85, "patch_y0": 0.4, "patch_mu": 0.5, "lens_k": -0.15})
    assert faults(three) == ["actuator_gain", "friction_strip", "lens_k"]
    assert validate_hidden({"patch_y0": 0.3}) == {"patch_y0": 0.3, "patch_mu": 0.8}  # strip friction travels with it
    assert validate_hidden({"patch_y0": 0.3, "patch_mu": 0.95})["patch_mu"] == 0.95
    for p in PRESETS:
        assert validate_hidden(p["hidden"]) == p["hidden"]


def test_surprise_worlds_are_valid_and_open(tmp_path):
    import random

    for seed in range(200):
        h = surprise(random.Random(seed))
        assert validate_hidden(h) == h and 1 <= len(faults(h)) <= MAX_HIDDEN
        assert "patch_y0" in h or "lens_k" in h
    c = client(tmp_path)
    a, b = c.get("/api/surprise?seed=3").json(), c.get("/api/surprise?seed=3").json()
    assert a == b and a["hidden"] == surprise(random.Random(3))


def test_rate_limit_per_ip(tmp_path):
    c = client(tmp_path, runs_per_hour=1)
    rid = c.post("/api/runs", json={"hidden": {"object_mu": 0.3}}).json()["id"]
    read_events(c, rid)  # let it finish so the busy lock is free
    r = c.post("/api/runs", json={"hidden": {"object_mu": 0.4}})
    assert r.status_code == 429 and "per hour" in r.json()["detail"]


def test_llm_budget_exhaustion_falls_back(tmp_path):
    llm = RecordedLLM.from_file(TOOL_FIXTURE)
    c = client(tmp_path, llm=llm, max_llm_calls=0)
    rid = c.post("/api/runs", json={"hidden": WET}).json()["id"]
    ev = read_events(c, rid)
    diags = [e for e in ev if e["type"] == "diagnose"]
    assert diags and all("fallback TrajectoryDiagnoser" in d["agent"] for d in diags)
    assert "budget" in diags[0]["summary"] and llm.usage.calls == 0
    assert ev[-1]["type"] == "end" and not any(e["type"] == "agent_step" for e in ev)


def test_per_run_turn_cap_stops_the_agent(tmp_path):
    llm = RecordedLLM.from_file(TOOL_FIXTURE)  # needs 3 turns; the cap allows 2
    c = client(tmp_path, llm=llm, max_turns_per_run=2)
    rid = c.post("/api/runs", json={"hidden": WET}).json()["id"]
    ev = read_events(c, rid)
    assert llm.usage.calls == 2
    diags = [e for e in ev if e["type"] == "diagnose"]
    assert "fallback" in diags[0]["agent"] and [t["tool"] for t in diags[0]["trace"]] == ["decel_profile", "fit_hypothesis"]
    assert all("fallback" in d["agent"] and "turn cap" in d["summary"] for d in diags[1:])
    assert c.get("/api/status").json()["llm_calls_left"] == 198
