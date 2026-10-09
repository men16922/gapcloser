import json
import os

import pytest

os.environ["GAPCLOSER_NO_AUTOAPP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from server.app import create_app  # noqa: E402
from server.studio_api import SAMPLE_DIR  # noqa: E402


def client(tmp_path):
    (tmp_path / "demo").mkdir(parents=True, exist_ok=True)
    os.environ["GAPCLOSER_LLM"] = "none"
    return TestClient(create_app(llm=None, env_name="analytic", render=False, data_dir=tmp_path))


def events(c, sid):
    with c.stream("GET", f"/api/studio/sessions/{sid}/events") as r:
        body = "".join(r.iter_text())
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_log_upload_analyze_export(tmp_path):
    c = client(tmp_path)
    csv = (SAMPLE_DIR / "lab-bench.csv").read_text()
    s = c.post("/api/studio/sessions", files={"file": ("bench.csv", csv, "text/csv")}).json()
    assert s["kind"] == "log" and len(s["session"]["pushes"]) == 20
    assert c.get(f"/api/studio/sessions/{s['id']}/export/newton").status_code == 422  # not analysed yet
    assert c.post(f"/api/studio/sessions/{s['id']}/analyze", json={"agent": True}).json()["agent"] is False  # no LLM
    ev = events(c, s["id"])
    kinds = [e["type"] for e in ev]
    assert kinds[-1] == "end" and "calibration" in kinds and "result" in kinds
    res = next(e["result"] for e in ev if e["type"] == "result")
    assert {g["field"] for g in res["calibration"]["gap"]} >= {"mu_eff", "actuator_gain", "patch_y0", "camera_pitch_deg"}
    nb = c.get(f"/api/studio/sessions/{s['id']}/export/newton")
    assert nb.status_code == 200 and "MU_EFF" in nb.text and "attachment" in nb.headers["content-disposition"]
    assert c.get(f"/api/studio/sessions/{s['id']}/export/isaaclab").text.count("EventTerm(") == 1
    assert c.get(f"/api/studio/sessions/{s['id']}/truth").status_code == 404  # uploads have no ground truth


def test_bad_upload_is_a_plain_422(tmp_path):
    c = client(tmp_path)
    r = c.post("/api/studio/sessions", files={"file": ("x.csv", "command\n2\n", "text/csv")})
    assert r.status_code == 422 and "stop" in r.json()["detail"]
    assert c.post("/api/studio/sessions", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415


def test_sample_truth_only_after_analysis(tmp_path):
    c = client(tmp_path)
    assert {x["id"] for x in c.get("/api/studio/samples").json()} == {"flick", "lab-bench", "short-reach"}
    s = c.post("/api/studio/sessions", data={"sample": "short-reach"}).json()
    assert c.get(f"/api/studio/sessions/{s['id']}/truth").status_code == 409
    c.post(f"/api/studio/sessions/{s['id']}/analyze", json={"agent": False})
    events(c, s["id"])
    t = c.get(f"/api/studio/sessions/{s['id']}/truth").json()
    assert t["model"]["patch_y0"] == 0.33


@pytest.mark.skipif(not (SAMPLE_DIR / "flick-video.mp4").exists(), reason="sample video not rendered")
def test_video_sample_track_append_analyze_overlay(tmp_path):
    pytest.importorskip("cv2")
    c = client(tmp_path)
    s = c.post("/api/studio/sessions", data={"sample": "flick"}).json()
    v = s["videos"][0]
    assert c.get("/" + v["frame"].replace("files/", "api/studio/files/", 1)).status_code == 200
    assert c.post(f"/api/studio/sessions/{s['id']}/analyze", json={}).status_code == 422  # not tracked yet
    t = c.post(f"/api/studio/sessions/{s['id']}/videos/0/track", json={"corners": v["corners_hint"]}).json()
    assert t["videos"][0]["tracked"] and len(t["session"]["pushes"]) == 7
    a = c.post(f"/api/studio/sessions/{s['id']}/videos", data={"part": "flick-video-2"}).json()
    assert a["added"] == 1
    t = c.post(f"/api/studio/sessions/{s['id']}/videos/1/track", json={"corners": a["videos"][1]["corners_hint"]}).json()
    assert len(t["session"]["pushes"]) == 11
    c.post(f"/api/studio/sessions/{s['id']}/analyze", json={"agent": False})
    ev = events(c, s["id"])
    res = next(e["result"] for e in ev if e["type"] == "result")
    m = res["calibration"]["model"]
    assert abs(m["mu_eff"] - 0.55) < 0.03 and abs(m["patch_y0"] - 0.36) < 0.04 and abs(m["patch_mu"] - 0.30) < 0.06
    ov = res["overlay"]["0"]["regions"]
    assert len(ov) == 2 and all(len(r["poly"]) == 4 for r in ov)
    assert c.get(f"/api/studio/files/{s['id']}/upload.mp4").status_code == 404


def test_log_session_takes_the_suggested_pushes(tmp_path):
    c = client(tmp_path)
    s = c.post("/api/studio/sessions", data={"sample": "short-reach"}).json()
    more = "command,stop\n2.3,0.52\n2.5,0.66\n"
    r = c.post(f"/api/studio/sessions/{s['id']}/logs", files={"file": ("more.csv", more, "text/csv")}).json()
    assert r["added_pushes"] == 2 and len(r["session"]["pushes"]) == 10
    bad = c.post(f"/api/studio/sessions/{s['id']}/logs", files={"file": ("x.csv", "launch_speed,stop\n2,0.3\n", "text/csv")})
    assert bad.status_code == 422


def test_simulated_robot_log_round_trip(tmp_path):
    pytest.importorskip("newton")
    c = client(tmp_path)
    bad = c.post("/api/studio/simulate", json={"kind": "video", "world": {"mu_eff": 0.5, "actuator_gain": 0.9}})
    assert bad.status_code == 422 and "robot logs" in bad.json()["detail"]
    assert c.post("/api/studio/simulate", json={"kind": "log", "world": {"mu_eff": 1.5}}).status_code == 422
    world = {"mu_eff": 0.6, "actuator_gain": 0.85}
    s = c.post("/api/studio/simulate", json={"kind": "log", "world": world, "pushes": 8, "reach": 0.5}).json()
    assert s["simulated"] and len(s["session"]["pushes"]) == 8
    c.post(f"/api/studio/sessions/{s['id']}/analyze", json={"agent": False})
    res = next(e["result"] for e in events(c, s["id"]) if e["type"] == "result")
    m = res["calibration"]["model"]
    assert abs(m["mu_eff"] - 0.6) < 0.04 and abs(m["actuator_gain"] - 0.85) < 0.03
    t = c.get(f"/api/studio/sessions/{s['id']}/truth").json()
    assert t["model"]["actuator_gain"] == 0.85
    more = c.post(f"/api/studio/sessions/{s['id']}/simulate-more", json={"commands": [3.0, 3.4]}).json()
    assert more["added_pushes"] == 2 and len(more["session"]["pushes"]) == 10
