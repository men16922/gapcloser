import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from tether.agent.llm import RecordedLLM
from tether.agent.tool_agent import from_params
from tether.sim.params import ParamSet
from tether.sim.push_task import AnalyticPushEnv
from tether.studio import design, export
from tether.studio.fit import calibrate
from tether.studio.pipeline import analyze
from tether.studio.session import Push, Session, SessionError, load, resample, to_csv

SAMPLES = Path(__file__).resolve().parent.parent / "tether" / "studio" / "samples"
PATCH = ParamSet.nominal().with_(object_mu=0.6, table_mu=0.6, patch_y0=0.35, patch_mu=0.4, actuator_gain=0.9)


def session_from(hidden, cmds, seed=1, noise=0.003):
    rng = random.Random(seed)
    ro = AnalyticPushEnv().push(hidden, cmds)
    return Session("t", "log", [Push(t.slide + rng.gauss(0, noise), t.command, track=[y + rng.gauss(0, 0.0015) for y in t.track])
                                for t in ro.trials])


def test_csv_roundtrip_and_aliases():
    s = load("cmd,stop_m,track\n2.0,0.31,0 0.06 0.11\n2.5,0.48,\n", "log.csv")
    assert [p.command for p in s.pushes] == [2.0, 2.5] and s.pushes[0].track == [0.0, 0.06, 0.11]
    again = load(to_csv(s), "x.csv")
    assert [p.stop for p in again.pushes] == [0.31, 0.48]


@pytest.mark.parametrize("text,msg", [
    ("command\n2.0\n", "no stop distance"),
    ("command,stop\n2.0,31\n", "is it in metres"),
    ("command,stop\n2.0,0.3\n,0.4\n", "needs a command"),
    ('{"pushes": []}', "no pushes"),
    ("{bad json", "not valid JSON"),
])
def test_bad_uploads_get_plain_errors(text, msg):
    with pytest.raises(SessionError, match=msg):
        load(text, "f.json" if text.startswith("{") else "f.csv")


def test_resample_60fps_to_30fps():
    tr = [0.01 * k for k in range(61)]  # 1 s at 60 fps, 0.6 m/s
    out = resample(tr, 60.0)
    assert len(out) == 31 and math.isclose(out[30], 0.6, abs_tol=1e-9)


def test_calibration_recovers_a_friction_region_with_intervals_that_cover_truth():
    s = session_from(PATCH, [1.5 + 0.2 * i for i in range(12)])
    cal = calibrate(s, n_boot=20)
    assert set(cal.structure) == {"mu_eff", "actuator_gain", "patch_y0", "patch_mu"}
    truth = {"mu_eff": 0.6, "actuator_gain": 0.9, "patch_y0": 0.35, "patch_mu": 0.4}
    for k, v in truth.items():
        lo, hi = cal.intervals[k]
        assert lo - 0.02 <= v <= hi + 0.02, (k, v, lo, hi)
    assert cal.predicted["after"]["mean"] > 0.9 > cal.predicted["before"]["mean"]


def test_short_data_is_not_overconfident_and_asks_for_far_pushes():
    s = session_from(PATCH.with_(actuator_gain=1.0), [1.2 + 0.08 * i for i in range(8)])
    assert s.coverage()[1] < 0.33
    cal = calibrate(s, n_boot=15)
    nxt = design.suggest(s, cal)
    assert cal.predicted["unmeasured_whatif"] and cal.predicted["after"]["mean"] < 0.97
    assert not nxt["settled"] and all(x["beyond_data"] for x in nxt["suggestions"])
    assert max(x["predicted_stop_m"] for x in nxt["suggestions"]) > 0.45


def test_exports_are_valid_python_and_carry_the_numbers():
    s = session_from(PATCH, [1.5 + 0.2 * i for i in range(12)])
    cal = calibrate(s, n_boot=8)
    nb, il = export.newton_snippet(s, cal), export.isaaclab_snippet(s, cal)
    compile(nb, "newton.py", "exec")
    compile(il, "isaaclab.py", "exec")
    assert f"{cal.model['mu_eff']:.4f}" in nb and "FRICTION_REGION" in nb
    assert "randomize_rigid_body_material" in il and "dynamic_friction_range" in il
    lo, hi = cal.intervals["mu_eff"]
    assert f"({lo:.4f}, {hi:.4f})" in il
    md = export.markdown(s, cal, design.suggest(s, cal))
    assert "Friction region starts at" in md and "Predicted effect" in md
    doc = export.to_json(s, cal)
    assert doc["schema"] == "tether.calibration/v1"
    json.dumps(doc)


def test_launch_offset_keeps_a_friction_region_fixed_to_the_table():
    hidden = ParamSet.nominal().with_(object_mu=0.6, table_mu=0.6, patch_y0=0.35, patch_mu=0.35)
    env = AnalyticPushEnv()
    pushes = []
    for i, c in enumerate([1.6 + 0.15 * i for i in range(10)]):
        start = (-0.04, 0.0, 0.05)[i % 3]
        p = hidden.with_(patch_y0=0.35 - start)
        t = env.push(p, [c]).trials[0]
        pushes.append(Push(t.slide, None, c, track=t.track, origin="video", start=start))
    cal = calibrate(Session("v", "video", pushes), n_boot=0)
    assert abs(cal.model["patch_y0"] - 0.35) < 0.01 and abs(cal.model["patch_mu"] - 0.35) < 0.02


def test_plane_camera_recovers_pose_and_backprojects_with_parallax():
    from tether.studio.video import PlaneCamera
    from tether.studio.video_sample import CAM_POS, H, W, project, sheet_corners_world

    cam = PlaneCamera(project(sheet_corners_world()), (0.210, 0.297), (W, H))
    assert abs(cam.height_m - CAM_POS[2]) < 0.005
    # a box centre 3 cm above the table at two places: distance between them survives back-projection
    pts = np.array([[0.0, 0.0, 0.03], [0.0, 0.4, 0.03]])
    P = cam.ray_to_plane(project(pts), 0.03)
    assert abs(np.linalg.norm(P[1] - P[0]) - 0.4) < 0.003


@pytest.mark.skipif(not (SAMPLES / "flick-video.mp4").exists(), reason="sample video not rendered")
def test_sample_video_tracks_to_ground_truth():
    pytest.importorskip("cv2")
    from tether.studio.video import track_video

    truth = json.loads((SAMPLES / "flick-video.truth.json").read_text())
    s, dbg = track_video(SAMPLES / "flick-video.mp4", truth["sheet_corners_px"], object_height_m=0.06)
    assert len(s.pushes) == len(truth["pushes"])
    for p, g in zip(s.pushes, truth["pushes"]):
        assert abs(p.stop - g["slide_m"]) < 0.01 and abs(p.launch_speed - g["launch_speed_mps"]) < 0.06
    assert abs(dbg["camera"]["height_m"] - truth["camera"]["pos"][2]) < 0.01


def test_agent_structure_and_offline_probe_requests_reach_the_report():
    s = session_from(PATCH, [1.5 + 0.2 * i for i in range(12)])
    free = ["mu_eff", "actuator_gain", "patch_y0", "patch_mu"]
    start = {**from_params(ParamSet.nominal()), "patch_y0": 0.3, "patch_mu": 0.5}

    def call(name, args, i):
        return {"content": "", "model": "scripted", "tool_calls": [{"id": f"c{i}", "name": name, "arguments": json.dumps(args)}]}

    llm = RecordedLLM({"chat:diagnose": [call("decel_profile", {}, 0), call("fit_hypothesis", {"model": start, "free": free}, 1),
                                         call("probe_real", {"commands": [3.9, 4.2]}, 2),
                                         call("commit", {"model": {**start, "patch_y0": 0.35, "patch_mu": 0.4, "mu_eff": 0.6,
                                                                   "actuator_gain": 0.9}, "explanation": "slick strip"}, 3)]})
    events = []
    res = analyze(s, llm, on_event=events.append, n_boot=8)
    assert res.agent["structure"] == free and res.agent["requested_commands"] == [3.9, 4.2]
    assert res.calibration.chosen_by.startswith("Nemotron agent")
    assert [r["command"] for r in res.next_experiment["agent_requests"]] == [3.9, 4.2]
    assert [e["type"] for e in events][-1] == "done" and any(e["type"] == "agent_step" for e in events)
    probe = next(e["step"] for e in events if e["type"] == "agent_step" and e["step"]["tool"] == "probe_real")
    assert probe["result"]["queued_for_human"] == [3.9, 4.2]


def test_friction_region_past_the_measured_reach_is_not_invented():
    s = session_from(PATCH.with_(actuator_gain=1.0), [1.2 + 0.08 * i for i in range(8)])
    cal = calibrate(s, ["mu_eff", "actuator_gain", "patch_y0", "patch_mu"],
                    {**from_params(ParamSet.nominal()), "patch_y0": 0.45, "patch_mu": 0.3}, n_boot=0)
    assert "patch_y0" not in cal.structure and cal.model["patch_y0"] is None
    assert "dropped" in cal.chosen_by


def test_cross_check_overrules_an_agent_that_ignores_the_launch_speed():
    s = session_from(PATCH, [1.5 + 0.2 * i for i in range(12)])  # true gain 0.9
    free = ["mu_eff", "patch_y0", "patch_mu"]  # the agent forgot the actuator: stops fit by trading friction for gain

    def call(name, args, i):
        return {"content": "", "model": "scripted", "tool_calls": [{"id": f"c{i}", "name": name, "arguments": json.dumps(args)}]}

    start = {**from_params(ParamSet.nominal()), "patch_y0": 0.3, "patch_mu": 0.5}
    llm = RecordedLLM({"chat:diagnose": [call("fit_hypothesis", {"model": start, "free": free}, 0),
                                         call("commit", {"model": {**start, "mu_eff": 0.74}, "explanation": "x"}, 1)]})
    res = analyze(s, llm, n_boot=0)
    cc = res.agent["cross_check"]
    assert cc["adopted"] == "search" and "actuator_gain" in res.calibration.structure
    assert any("launch" in u for u in cc["unexplained"])
    assert abs(res.calibration.model["actuator_gain"] - 0.9) < 0.02


def test_domain_units_scale_lengths_and_speeds_but_not_friction():
    from tether.studio import domains as D

    d = D.get("driving")
    assert D.model_to_domain(d, {"mu_eff": 0.7, "patch_y0": 0.36, "camera_dx": 0.01, "lens_k": 0.5}) == \
        {"mu_eff": 0.7, "patch_y0": 9.0, "camera_dx": 0.25, "lens_k": 0.02}
    assert D.scale_text(d, "stop at 0.36 m, ±1.0 cm, launch 2.0 m/s") == "stop at 9.00 m, ±0.25 m, launch 36 km/h"
    assert D.scale_text(D.get("factory"), "0.36 m") == "0.36 m" and D.get(None).id == "robot"
    # Froude: the stop distance of the scaled run, scaled up, equals the full-size one (v^2 / (2 mu g))
    v, mu = 2.0, 0.7
    assert abs((v * d.speed_scale) ** 2 / (2 * mu * 9.81) - d.scale * v * v / (2 * mu * 9.81)) < 1e-9


def test_sheet_corners_are_found_automatically():
    pytest.importorskip("cv2")
    import json

    import numpy as np

    from tether.studio.video import auto_sheet

    path = SAMPLES / "stop-line-video.mp4"
    if not path.exists():
        pytest.skip("sample video not rendered")
    found = auto_sheet(path)
    truth = json.loads((SAMPLES / "stop-line-video.truth.json").read_text())["sheet_corners_px"]
    assert found is not None
    assert max(min(np.hypot(a - x, b - y) for x, y in found) for a, b in truth) < 2.0


def test_carla_export_runs_against_the_carla_api_shape():
    """CARLA is not installed here (it needs a GPU server), so the export runs against a stand-in `carla` module with
    the calls the snippet makes, named as in the CARLA 0.9.16 Python API and the friction-trigger tutorial
    (static.trigger.friction: friction, extent_x/y/z in centimetres; the trigger sets the wheels' friction while a
    vehicle is inside and restores it on exit)."""
    import sys
    import types

    from tether.studio import export
    from tether.studio.pipeline import analyze
    from tether.studio.session import load

    class V:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

        def __add__(self, o):
            return V(self.x + o.x, self.y + o.y, self.z + o.z)

        def __mul__(self, k):
            return V(self.x * k, self.y * k, self.z * k)

    rec = {"applied": None, "attrs": {}, "spawned": None}

    class Wheel:
        def __init__(self):
            self.tire_friction = 3.5  # a CARLA vehicle's own value: a scalar, not mu

    class PC:
        def __init__(self):
            self.wheels = [Wheel() for _ in range(4)]

    class Tf:
        def __init__(self, location=None, rotation=None):
            self.location, self.rotation = location or V(), rotation

        def get_forward_vector(self):
            return V(1.0, 0.0, 0.0)

    class Ego:
        def get_physics_control(self):
            return PC()

        def apply_physics_control(self, pc):
            rec["applied"] = [w.tire_friction for w in pc.wheels]

        def get_transform(self):
            return Tf(V(10.0, 0.0, 0.0), "rot")

    class BP:
        def set_attribute(self, k, v):
            assert isinstance(v, str)
            rec["attrs"][k] = float(v)

    class World:
        def get_actors(self):
            return types.SimpleNamespace(filter=lambda pat: [Ego()] if pat == "vehicle.*" else [])

        def get_blueprint_library(self):
            return types.SimpleNamespace(find=lambda name: BP() if name == "static.trigger.friction" else None)

        def spawn_actor(self, bp, tf):
            rec["spawned"] = tf

    fake = types.ModuleType("carla")
    fake.Client = lambda host, port: types.SimpleNamespace(get_world=World)
    fake.Transform = Tf
    sys.modules["carla"] = fake
    try:
        s = load((SAMPLES / "lab-bench.csv").read_text(), "lab-bench.csv", "t")
        s.meta["domain"] = "driving"
        res = analyze(s, None)
        exec(compile(export.carla_snippet(s, res.calibration), "carla_calibration.py", "exec"), {})
    finally:
        sys.modules.pop("carla", None)
    m = res.calibration.model
    assert rec["applied"] == pytest.approx([3.5 * m["mu_eff"] / 0.8] * 4, rel=1e-3)
    assert rec["attrs"]["friction"] == pytest.approx(3.5 * m["patch_mu"] / 0.8, rel=1e-3)
    assert rec["attrs"]["extent_x"] == 3000.0 and rec["attrs"]["extent_y"] == 175.0  # 30 m and 1.75 m in cm
    start = round(m["patch_y0"] * 25, 2)  # full-scale metres from the brake point
    assert rec["spawned"].location.x == pytest.approx(10.0 + start + 30.0, abs=0.01)



def test_real_table_validation_runs_end_to_end_on_a_clip_with_a_tilt_angle(tmp_path, monkeypatch):
    import math
    import shutil

    import tether.eval.prove_real as pr

    shutil.copyfile(SAMPLES / "flick-video.mp4", tmp_path / "table.mp4")
    (tmp_path / "truth.json").write_text(json.dumps({"table": {"tilt_kinetic_deg": math.degrees(math.atan(0.55)), "object_height_cm": 6.0},
                                                     "missing": {"tilt_kinetic_deg": 20.0}}))
    monkeypatch.setattr(pr, "OUT", tmp_path / "out.json")
    out = pr.main(tmp_path)
    row = next(r for r in out["rows"] if r["clip"] == "table")
    assert row["pushes"] == 7 and abs(row["error_vs_tilt"]) < 0.05 and row["within_tol"]
    assert next(r for r in out["rows"] if r["clip"] == "missing")["error"] == "video not found"
    assert out["pass"] is False  # one pair is not three


def test_trajectory_shape_pins_friction_when_the_launch_speed_is_mismeasured():
    """Real clocks jitter (EV-RealPhys groups 240 Hz poses with 60 Hz images): a launch speed read off a few frames
    can be 20% off. The shape of the whole slide (how long it takes to stop) still gives friction."""
    import random

    from tether.studio.fit import calibrate
    from tether.studio.session import Push, Session

    rng, mu, g = random.Random(4), 0.16, 9.81
    pushes = []
    for v0 in (0.9, 1.1, 1.3, 1.5, 1.7, 1.9):
        stop = v0 * v0 / (2 * mu * g)
        T = v0 / (mu * g)
        times = [k / 30 + rng.uniform(-0.01, 0.01) for k in range(int(T * 30) + 5)]
        track = [round(v0 * min(t, T) - 0.5 * mu * g * min(t, T) ** 2, 4) for t in times]
        track[0] = 0.0
        pushes.append(Push(round(stop, 4), None, round(v0 * rng.choice((0.8, 1.2)), 3), track=track, origin="video"))
    cal = calibrate(Session("jitter", "video", pushes), n_boot=0)
    assert abs(cal.model["mu_eff"] - mu) < 0.01, cal.model


def test_tracks_that_cannot_be_slides_are_flagged():
    from tether.studio.video import implausible

    clean = [0.0, 0.04, 0.075, 0.105, 0.13, 0.15, 0.165, 0.175, 0.18, 0.18]
    assert implausible(clean, 30.0, 1.25) is None
    assert implausible([0.0, 0.05, 0.1, 0.06, 0.08, 0.09], 30.0) == "runs backwards"
    assert implausible([0.0, 0.04, 0.4, 0.45, 0.9, 0.92, 0.93], 30.0) == "jumps"
    assert implausible([0.0, 0.03, 0.05, 0.06, 0.06, 0.06], 30.0, 2.4) == "stops too soon for its speed"


def test_a_friction_region_the_agent_adds_within_noise_is_left_out():
    """The short-reach log never reaches its slick region; an agent that fits one anyway (it lowers the stop error
    by under a millimetre) is overruled by the same within-noise rule the offline search uses."""
    from tether.agent.llm import RecordedLLM
    from tether.agent.tool_agent import from_params

    s = load((SAMPLES / "short-reach.csv").read_text(), "short-reach.csv", "sr")
    free = ["mu_eff", "actuator_gain", "patch_y0", "patch_mu"]
    start = {**from_params(ParamSet.nominal()), "patch_y0": 0.25, "patch_mu": 0.5}

    def call(name, args, i):
        return {"content": "", "model": "scripted", "tool_calls": [{"id": f"c{i}", "name": name, "arguments": json.dumps(args)}]}

    llm = RecordedLLM({"chat:diagnose": [call("fit_hypothesis", {"model": start, "free": free}, 0),
                                         call("commit", {"model": {**start, "mu_eff": 0.61, "actuator_gain": 1.0}, "explanation": "a region"}, 1)]})
    res = analyze(s, llm, n_boot=0)
    assert res.agent["cross_check"]["adopted"] == "agent without region"
    assert "patch_y0" not in res.calibration.structure and res.calibration.model["patch_y0"] is None
