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
    got = {x["id"]: x["domain"] for x in c.get("/api/studio/samples").json()}
    assert got == {"flick": "robot", "lab-bench": "robot", "short-reach": "robot", "stop-line": "driving", "brake-log": "driving", "press-line": "factory"}
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


def test_chat_is_grounded_in_the_session_and_answers_in_the_chosen_language(tmp_path):
    from agent.llm import RecordedLLM

    (tmp_path / "demo").mkdir(parents=True, exist_ok=True)
    llm = RecordedLLM({"diagnose": [{"text": "<think>x</think>마찰은 0.70입니다.", "model": "scripted"}]})
    c = TestClient(create_app(llm=llm, env_name="analytic", render=False, data_dir=tmp_path))
    s = c.post("/api/studio/sessions", data={"sample": "lab-bench"}).json()
    c.post(f"/api/studio/sessions/{s['id']}/analyze", json={"agent": False})
    events(c, s["id"])
    r = c.post("/api/studio/chat", json={"sid": s["id"], "lang": "ko", "step": "results",
                                         "messages": [{"role": "user", "content": "마찰이 얼마야?"}]}).json()
    assert r["answer"] == "마찰은 0.70입니다." and r["left"] == 29
    system = llm.calls[0][1][0]["content"]
    assert "Korean" in system and "Cube-table friction" in system and "revealed_truth" not in system
    assert c.post("/api/studio/chat", json={"messages": [{"role": "assistant", "content": "hi"}]}).status_code == 422


def test_chat_without_llm_says_how_to_enable_it(tmp_path):
    c = client(tmp_path)
    r = c.post("/api/studio/chat", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 503 and "GAPCLOSER_LLM" in r.json()["detail"]


def test_domains_label_the_same_physics_and_driving_exports_carla(tmp_path):
    c = client(tmp_path)
    doms = {d["id"]: d for d in c.get("/api/studio/domains").json()}
    assert set(doms) == {"robot", "driving", "factory"} and doms["driving"]["scale"] == 25.0
    assert doms["driving"]["name"]["ko"] == "자율주행 차량" and doms["factory"]["labels"]["en"]["mu_eff"] == "Part-rail friction"
    s = c.post("/api/studio/sessions", data={"sample": "press-line"}).json()
    assert s["domain"] == "factory"
    c.post(f"/api/studio/sessions/{s['id']}/analyze", json={"agent": False})
    events(c, s["id"])
    md = c.get(f"/api/studio/sessions/{s['id']}/export/markdown").text
    assert "Factory inspection" in md and "Part-rail friction" in md and "Oily section starts at" in md
    r = c.post("/api/studio/simulate", json={"kind": "log", "domain": "driving", "pushes": 8, "reach": 0.5,
                                            "world": {"mu_eff": 0.7, "patch_y0": 0.3, "patch_mu": 0.3}})
    s = r.json()
    assert r.status_code == 200 and s["domain"] == "driving" and s["name"].startswith("Your simulated road")
    c.post(f"/api/studio/sessions/{s['id']}/analyze", json={"agent": False})
    events(c, s["id"])
    carla = c.get(f"/api/studio/sessions/{s['id']}/export/carla").text
    compile(carla, "carla_calibration.py", "exec")
    assert "static.trigger.friction" in carla and "tire_friction" in carla and "Braking check" in carla
    md = c.get(f"/api/studio/sessions/{s['id']}/export/markdown").text
    assert "Tire-road friction" in md
    assert " m" in md and "Froude" in md


def test_newton_replay_of_the_export_beats_the_current_sim(tmp_path):
    pytest.importorskip("newton")
    c = client(tmp_path)
    s = c.post("/api/studio/sessions", data={"sample": "press-line"}).json()
    assert c.post(f"/api/studio/sessions/{s['id']}/verify").status_code == 422  # nothing calibrated yet
    c.post(f"/api/studio/sessions/{s['id']}/analyze", json={"agent": False})
    events(c, s["id"])
    v = c.post(f"/api/studio/sessions/{s['id']}/verify").json()
    assert len(v["pushes"]) == 18 and v["engine"].startswith("NVIDIA Newton")
    assert v["rms_calibrated_m"] < 0.015 and v["rms_current_m"] > 5 * v["rms_calibrated_m"]


def test_retraining_on_tether_ranges_beats_current_and_wide(tmp_path, monkeypatch):
    pytest.importorskip("newton")
    import time

    import studio.train as tr

    monkeypatch.setattr(tr, "N_WORLDS", 6)
    monkeypatch.setattr(tr, "ITERS", 4)
    monkeypatch.setattr(tr, "LR", 0.85)
    c = client(tmp_path)
    s = c.post("/api/studio/sessions", data={"sample": "press-line"}).json()
    assert c.post(f"/api/studio/sessions/{s['id']}/train").status_code == 422
    c.post(f"/api/studio/sessions/{s['id']}/analyze", json={"agent": False})
    events(c, s["id"])
    c.post(f"/api/studio/sessions/{s['id']}/train")
    for _ in range(240):
        st = c.get(f"/api/studio/sessions/{s['id']}/train").json()
        if st["status"] != "running":
            break
        time.sleep(0.5)
    res = st["result"]["conditions"]
    assert st["status"] == "done" and st["result"]["hidden_known"]
    assert res["tether"]["final_real"] >= 0.9 > res["current"]["final_real"] and res["tether"]["final_real"] > res["wide"]["final_real"]
    assert len(res["tether"]["real"]) == 5 and res["tether"]["lanes"][0]


def test_overview_console_and_studio_are_linked(tmp_path):
    c = client(tmp_path)
    home = c.get("/").text
    assert "Tether" in home and 'href="/console' in home and "/studio" in home and "sample=" in home
    assert c.get("/api/studio/sample-frame/press-line").headers["content-type"] == "image/jpeg"


def test_uploaded_video_gets_its_sheet_corners_found(tmp_path):
    pytest.importorskip("cv2")
    path = SAMPLE_DIR / "flick-video.mp4"
    if not path.exists():
        pytest.skip("sample video not rendered")
    c = client(tmp_path)
    s = c.post("/api/studio/sessions", files={"file": ("my-table.mp4", path.read_bytes(), "video/mp4")}).json()
    v = s["videos"][0]
    assert v["corners_auto"] and len(v["corners_hint"]) == 4
