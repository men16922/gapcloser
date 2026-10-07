"""3D viewer data: replay JSON schema (recorded + freshly simulated), Franka mesh asset, dashboard embedding."""

import base64
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "runs" / "demo"
FRANKA = ROOT / "dashboard" / "assets" / "franka_fr3.json"
FPS, STRIKE_FRAME, MAX_FRAMES = 30, 16, int(2.5 * 30) + 16  # mirrors sim.newton_push (imported lazily below)


def check_replay(d: dict) -> None:
    assert d["schema"] == "gapcloser.replay/1"
    assert d["fps"] == FPS and d["strike_frame"] == STRIKE_FRAME
    sc = d["scene"]
    assert 0.1 < sc["target"] < 0.8 and sc["tol"] > 0 and len(sc["arm_base"]) == 3
    assert len(sc["table"]["size"]) == 2 and sc["marker"]["hx"] > 0
    links = d["links"]
    assert len(links) >= 10 and "fr3/fr3_hand" in links
    assert set(d["trials"]) == {"real", "sim"}
    for t in d["trials"].values():
        n = len(t["cube"])
        # the cube is struck at STRIKE_FRAME, recording stops once it rests (>= 20 frames later) or at the cap
        assert STRIKE_FRAME + 21 <= n <= MAX_FRAMES
        assert all(len(f) == 7 for f in t["cube"])
        assert 1 <= len(t["arm"]) <= n and all(len(f) == 7 * len(links) for f in t["arm"])
        assert abs(t["cube"][-1][1] - t["slide"]) < 1e-3
        assert abs(t["cube"][0][1]) < 1e-3  # cube starts on the start line
        for q in (t["cube"][0][3:], t["cube"][-1][3:], t["arm"][-1][3:7]):
            assert abs(sum(x * x for x in q) - 1) < 2e-3  # unit quaternions (xyzw)
        assert {"object_mu", "table_mu", "actuator_gain"} <= set(t["params"])
        assert isinstance(t["tipped"], bool) and t["half_size"] > 0
        # 4-decimal rounding keeps the files compact
        assert all(round(x, 4) == x for x in t["cube"][-1])


def _recorded():
    if not (DEMO / "bundle.json").exists():
        return []
    bundle = json.loads((DEMO / "bundle.json").read_text())
    return [(r, e) for r in bundle["runs"] for e in r["events"] if e.get("clip")]


@pytest.mark.skipif(not _recorded(), reason="no recorded demo bundle")
def test_recorded_replays_match_schema_and_clips():
    pairs = _recorded()
    for run, e in pairs:
        rel = e["clip"].get("replay")
        assert rel == f"replay/{run['id']}-it{e['iter']}.json"
        d = json.loads((DEMO / rel).read_text())
        check_replay(d)
        assert d["scene"]["target"] == pytest.approx(e["clip"]["target"])
        # replays come from the same Newton scene as the clips: identical stop positions
        for k in ("real", "sim"):
            assert d["trials"][k]["slide"] == pytest.approx(e["clip"][f"{k}_slide"], abs=1e-3)
        assert (DEMO / rel).stat().st_size < 120_000


def test_franka_mesh_asset_is_small_and_decodes():
    data = json.loads(FRANKA.read_text())
    assert FRANKA.stat().st_size < 3_000_000
    assert data["schema"] == "gapcloser.franka_mesh/1"
    names = {lk["name"] for lk in data["links"]}
    assert {"fr3/fr3_link0", "fr3/fr3_link7", "fr3/fr3_hand", "fr3/fr3_leftfinger"} <= names
    tris = 0
    for lk in data["links"]:
        for p in lk["parts"]:
            v = base64.b64decode(p["v"])
            i = base64.b64decode(p["i"])
            nv = len(v) // 6  # uint16 xyz
            assert len(v) % 6 == 0 and nv > 0
            w = 4 if p["i32"] else 2
            idx = [int.from_bytes(i[k:k + w], "little") for k in range(0, len(i), w)]
            assert len(idx) % 3 == 0 and max(idx) < nv
            tris += len(idx) // 3
    assert tris == data["triangles"]


def test_dashboard_embeds_viewer_offline(tmp_path):
    from dashboard.build import build, build_live

    bundle = {"generated": "2026-10-07T00:00:00+00:00", "task": {"name": "Push-to-line", "success_tol_m": 0.03, "n_targets": 1, "targets": [0.45]},
              "params": {}, "stack": {"physics": "NVIDIA Newton", "diagnoser": "x"}, "benchmark": None,
              "runs": [{"id": "r", "title": "R", "events": [{"type": "measure", "iter": 0, "clip": {
                  "target": 0.45, "real": "clips/a.webp", "sim": "clips/a.webp", "real_slide": 0.3, "sim_slide": 0.45,
                  "replay": "replay/r-it0.json"}}]}]}
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "a.webp").write_bytes(b"RIFF0000WEBP")
    (tmp_path / "replay").mkdir()
    (tmp_path / "replay" / "r-it0.json").write_text(json.dumps({"schema": "gapcloser.replay/1", "marker": "</script>"}))
    (tmp_path / "bundle.json").write_text(json.dumps(bundle))
    out = build(tmp_path / "bundle.json", tmp_path / "dist" / "g.html")
    html = (tmp_path / "dist" / "g.standalone.html").read_text()
    assert out.exists()
    assert '"replay":{"schema":"gapcloser.replay/1"' in html  # replay inlined as an object, not a path
    assert "<\\/script>" in html and html.count("</script>") == html.count("<script")  # nothing closes a script early
    assert "WebGLRenderer" in html  # three.js vendored inline: works offline
    assert '"gapcloser.franka_mesh/1"' in html
    assert "/*__THREE__*/" not in html and '"__FRANKA__"' not in html and '"__BUNDLE__"' not in html
    live = build_live(tmp_path / "dist" / "live.html").read_text()
    assert "WebGLRenderer" in live and "window.BUNDLE = null" in live


def test_fresh_newton_replay_schema_and_frames():
    pytest.importorskip("newton")
    from sim.newton_push import MAX_SECONDS, STRIKE_FRAME as SF
    from sim.params import ParamSet
    from sim.replay import record_pair

    assert int(MAX_SECONDS * FPS) + SF == MAX_FRAMES and SF == STRIKE_FRAME
    nominal = ParamSet.nominal()
    weak = nominal.with_(actuator_gain=0.76)
    d = record_pair(nominal, 2.6, weak, 2.6, 0.45)
    check_replay(d)
    real, sim = d["trials"]["real"], d["trials"]["sim"]
    assert real["launch_speed"] == pytest.approx(2.6 * 0.76, abs=1e-3)
    assert real["slide"] < sim["slide"] - 0.05  # weak motor: the real cube stops short
    assert real["arm"] == sim["arm"]  # same cube size -> identical kinematic strike
