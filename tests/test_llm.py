import json
from pathlib import Path

import pytest

from agent.llm import LLMResponse, RecordedLLM, RecordingLLM, pick_model
from agent.llm_diagnoser import LLMDiagnoser, build_evidence, parse_response, validate_suspects
from agent.loop import HeuristicPlanner, RealWorld, run_loop
from sim.params import ParamSet
from sim.push_task import AnalyticPushEnv, GridTrainer, eval_targets

TARGETS = eval_targets(20, 1000)
FIXTURE = Path(__file__).parent / "fixtures" / "nemotron_weak_motor.json"


def test_pick_model_matches_all_words_and_prefers_exact():
    ids = ["nvidia/Nemotron-3-Super-120B", "nvidia/Nemotron-3-Nano-30B", "nemotron-3-nano:4b"]
    assert pick_model(ids, "nemotron super") == "nvidia/Nemotron-3-Super-120B"
    assert pick_model(ids, "nemotron nano 4b") == "nemotron-3-nano:4b"
    assert pick_model(ids, "nemotron-3-nano:4b") == "nemotron-3-nano:4b"
    assert pick_model(ids, "llama") is None


def test_parse_response_strips_thinking_and_prose():
    txt = '<think>gain looks low</think>Sure! {"reasoning": "r", "suspects": []} done'
    assert parse_response(txt) == {"reasoning": "r", "suspects": []}
    with pytest.raises(ValueError):
        parse_response("no json here")


def test_validate_suspects_filters_dedupes_clamps_and_drops_no_change():
    sim = ParamSet.nominal()
    raw = [
        {"name": "actuator_gain", "direction": "up", "confidence": 0.4, "estimate": 0.76},
        {"name": "actuator_gain", "direction": "down", "confidence": 0.9, "estimate": 0.76},
        {"name": "object_mu", "direction": "up", "confidence": 0.8, "estimate": 0.8},  # equals sim -> dropped
        {"name": "gravity", "direction": "up", "confidence": 1.0, "estimate": 3},  # unknown -> dropped
        {"name": "camera_dx", "direction": "sideways", "confidence": 7, "estimate": -1.0},  # clamped
    ]
    out = validate_suspects(raw, sim)
    assert [s.name for s in out] == ["camera_dx", "actuator_gain"]
    assert out[0].confidence == 1.0 and out[0].estimate == -0.03 and out[0].direction == "down"
    assert out[1].confidence == 0.9 and out[1].direction == "down"


def test_evidence_has_tracking_ratios_and_no_hidden_values():
    env = AnalyticPushEnv()
    hidden = ParamSet.nominal().with_(actuator_gain=0.76)
    pol = GridTrainer(env).train(__import__("sim.params", fromlist=["Randomization"]).Randomization.none())
    ev = build_evidence(env.rollout(hidden, pol, TARGETS), env.rollout(ParamSet.nominal(), pol, TARGETS), ParamSet.nominal())
    assert abs(ev["tracking"]["launch_speed_ratio_real_over_sim"] - 0.76) < 1e-3
    assert ev["sim_current"]["actuator_gain"] == 1.0  # the agent sees its own sim, never the hidden value


def test_recorded_nemotron_response_closes_weak_motor_gap():
    """Replays a real Nemotron 3 Nano response recorded from a local run (no network)."""
    llm = RecordedLLM.from_file(FIXTURE)
    env = AnalyticPushEnv()
    res = run_loop(RealWorld(env, ParamSet.nominal().with_(actuator_gain=0.76)), GridTrainer(env),
                   LLMDiagnoser(llm), HeuristicPlanner(), TARGETS)
    d = res.log[0].diagnosis
    assert res.final_success >= 0.9
    assert d["model"].startswith("nemotron")
    assert d["suspects"][0]["name"] == "actuator_gain"
    assert d["reasoning"]


def test_malformed_llm_output_falls_back_without_stalling():
    llm = RecordedLLM({"diagnose": [{"text": "I think it's the friction?", "model": "m"}] * 5})
    env = AnalyticPushEnv()
    diag = LLMDiagnoser(llm)
    res = run_loop(RealWorld(env, ParamSet.nominal().with_(actuator_gain=0.76)), GridTrainer(env), diag,
                   HeuristicPlanner(), TARGETS)
    assert res.final_success >= 0.9  # TrajectoryDiagnoser fallback still closes it
    assert "fallback" in res.log[0].diagnosis["model"]
    assert diag.history[0]["error"]


def test_recording_llm_writes_replayable_fixture(tmp_path):
    class Fake:
        def complete(self, role, messages, schema=None):
            return LLMResponse('{"reasoning": "x", "suspects": []}', "fake", 3, 4)

    path = tmp_path / "rec.json"
    rec = RecordingLLM(Fake(), path)
    rec.complete("diagnose", [{"role": "user", "content": "hi"}])
    data = json.loads(path.read_text())
    assert data["diagnose"][0]["model"] == "fake"
    assert RecordedLLM(data).complete("diagnose", []).completion_tokens == 4


def test_frames_switch_to_vision_role_with_image_parts(tmp_path):
    from PIL import Image

    frame = tmp_path / "f0.png"
    Image.new("RGB", (8, 8), (10, 200, 10)).save(frame)
    reply = {"text": '{"reasoning": "slow launch", "suspects": [{"name": "actuator_gain", "direction": "down", '
                     '"confidence": 0.9, "estimate": 0.76}]}', "model": "omni"}
    env = AnalyticPushEnv()
    for frames, role in ((lambda: [frame], "vision"), (None, "diagnose")):
        llm = RecordedLLM({"vision": [reply] * 3, "diagnose": [reply] * 3})
        run_loop(RealWorld(env, ParamSet.nominal().with_(actuator_gain=0.76)), GridTrainer(env),
                 LLMDiagnoser(llm, frames=frames), HeuristicPlanner(), TARGETS, max_iter=2)
        sent_role, messages = llm.calls[0]
        assert sent_role == role
        content = messages[1]["content"]
        has_image = isinstance(content, list) and any(p.get("type") == "image_url" for p in content)
        assert has_image == (frames is not None)
