"""CosmosEyes offline: event aggregation from per-frame answers, parsing, graceful degradation (mocked HTTP)."""

import json
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path

from agent.cosmos_eyes import CosmosEyes, events_from_frames, parse_frames
from agent.llm import RecordedLLM
from agent.tool_agent import ToolAgentDiagnoser
from sim.params import ParamSet, Randomization
from sim.push_task import AnalyticPushEnv, InverseTrainer, eval_targets

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "runs" / "demo"
TIP_CLIP = DEMO / "clips" / "tipping-edge-it0-real.webp"
TIMES = [0.0, 0.17, 0.33, 0.5, 0.69, 0.85, 1.02, 1.19]


def test_events_from_frames_slid_only_tipped_at_first_tilted_frame():
    assert events_from_frames([False] * 8, TIMES) == [{"t": 0.17, "type": "slid"}]
    flags = [False, False, False, False, True, True, False, True]
    assert events_from_frames(flags, TIMES) == [{"t": 0.17, "type": "slid"}, {"t": 0.69, "type": "tipped"}]


def test_frame_zero_tilt_is_ignored_and_empty_input_gives_no_events():
    assert events_from_frames([True] + [False] * 7, TIMES) == [{"t": 0.17, "type": "slid"}]
    assert events_from_frames([], []) == []


def test_parse_frames_accepts_object_bare_list_and_fences():
    obj = '{"frames":[{"i":0,"tilted":false},{"i":1,"tilted":true}]}'
    assert parse_frames(obj, 3) == [False, True, False]
    assert parse_frames('```json\n[{"i":2,"tilted":"true"}]\n```', 3) == [False, False, True]
    assert parse_frames('{"frames":[{"i":9,"tilted":true}]}', 3) == [False, False, False]  # out of range dropped
    assert parse_frames("the cube slid", 3) is None
    assert parse_frames("", 3) is None


def test_import_is_light():
    code = "import sys, agent.cosmos_eyes; print(sorted(m for m in ('numpy','scipy','PIL','requests') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


class DownEyes(CosmosEyes):
    def _get(self, path, timeout):
        raise urllib.error.URLError("connection refused")

    def _post(self, path, body, timeout):
        raise urllib.error.URLError("connection refused")


def test_server_down_degrades_gracefully():
    eyes = DownEyes()
    assert eyes.available() is False and "refused" in eyes.last_error
    assert eyes.events(TIP_CLIP) == []
    assert "URLError" in eyes.last["error"]
    assert eyes.events(Path("/nonexistent/clip.webp")) == []  # bad clip: no exception either


class ScriptedEyes(CosmosEyes):
    def __init__(self, reply, health="ok"):
        super().__init__()
        self.reply, self.health, self.bodies = reply, health, []

    def _get(self, path, timeout):
        return {"status": self.health}

    def _post(self, path, body, timeout):
        self.bodies.append(body)
        return {"choices": [{"message": {"content": self.reply}}]}


def test_tracked_clip_through_mocked_server_builds_events():
    reply = json.dumps({"frames": [{"i": i, "tilted": i >= 4} for i in range(8)]})
    eyes = ScriptedEyes(reply)
    assert eyes.available()
    assert not ScriptedEyes(reply, health="loading model").available()
    res = eyes.look(TIP_CLIP)
    assert res["error"] == ""
    assert len(res["times"]) == 8 and res["tilted"] == [False] * 4 + [True] * 4
    assert [e["type"] for e in res["events"]] == ["slid", "tipped"]
    assert res["events"][1]["t"] == round(res["times"][4], 2)
    body = eyes.bodies[0]
    assert body["temperature"] == 0.0
    imgs = [c for c in body["messages"][1]["content"] if c["type"] == "image_url"]
    assert len(imgs) == 8 and imgs[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_unparseable_reply_gives_no_events():
    res = ScriptedEyes("I think it slid.").look(TIP_CLIP)
    assert res["events"] == [] and res["error"] == "unparseable reply"


def _rollouts():
    targets = eval_targets(20, 1000)
    env = AnalyticPushEnv()
    pol = InverseTrainer().train(Randomization.none())
    return env.rollout(ParamSet.nominal().with_(actuator_gain=0.8), pol, targets), env.rollout(ParamSet.nominal(), pol, targets)


def _commit_llm():
    from agent.tool_agent import from_params

    commit = {"id": "c0", "name": "commit", "arguments": json.dumps({"model": from_params(ParamSet.nominal()), "explanation": "x"})}
    return RecordedLLM({"chat:diagnose": [{"content": "", "tool_calls": [commit], "model": "scripted"}]})


def test_tool_agent_sees_camera_events_only_when_provided():
    real, sim = _rollouts()
    cam = {"cosmos": [{"t": 0.17, "type": "slid"}, {"t": 0.69, "type": "tipped"}], "physics_tipped_count": 3}
    llm = _commit_llm()
    ToolAgentDiagnoser(llm, camera_events=lambda: cam).diagnose(real, sim, ParamSet.nominal())
    evidence = json.loads(llm.calls[0][1][1]["content"].split("\n", 1)[1])
    assert evidence["camera_events"] == cam
    llm = _commit_llm()
    ToolAgentDiagnoser(llm).diagnose(real, sim, ParamSet.nominal())
    assert "camera_events" not in llm.calls[0][1][1]["content"]


class FixedEyes:
    model_name = "fake-cosmos"

    def look(self, clip):
        return {"events": [{"t": 0.17, "type": "slid"}, {"t": 0.69, "type": "tipped"}], "seconds": 0.0,
                "times": [0.0, 0.17], "tilted": [False, True], "error": ""}


def test_eyes_only_annotates_bundle_with_agreement(tmp_path):
    from eval.record_demo import annotate_eyes

    bundle = json.loads((DEMO / "bundle.json").read_text())
    run = next(r for r in bundle["runs"] if r["id"] == "tipping-edge")
    ev = next(e for e in run["events"] if e.get("clip"))
    for k in ("real", "replay"):
        (tmp_path / ev["clip"][k]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DEMO / ev["clip"][k], tmp_path / ev["clip"][k])
    run["events"] = [ev]
    bundle["runs"] = [run]
    (tmp_path / "bundle.json").write_text(json.dumps(bundle))
    st = annotate_eyes(tmp_path, FixedEyes())
    assert st == {"clips": 1, "agree": 1, "tp": 1, "fn": 0, "fp": 0, "tn": 0, "errors": 0}
    clip = json.loads((tmp_path / "bundle.json").read_text())["runs"][0]["events"][0]["clip"]
    assert clip["physics"]["tipped"] is True and 0.4 < clip["physics"]["t_tip"] < 0.9
    assert clip["eyes"]["model"] == "fake-cosmos" and clip["eyes"]["frames"][1] == {"t": 0.17, "tilted": True}
