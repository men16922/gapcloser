import json

from tether.agent.llm import RecordedLLM
from tether.agent.loop import HeuristicPlanner, RealWorld, run_loop
from tether.agent.tool_agent import TOOLS, ToolAgentDiagnoser, Workbench, from_params, to_params
from tether.sim.params import ParamSet, Randomization
from tether.sim.push_task import AnalyticPushEnv, InverseTrainer, eval_targets

TARGETS = eval_targets(20, 1000)
PATCH = ParamSet.nominal().with_(patch_y0=0.35, patch_mu=0.45)


def call(name, args, i=0):
    return {"id": f"c{i}", "name": name, "arguments": json.dumps(args)}


def turn(*calls):
    return {"content": "", "tool_calls": list(calls), "model": "scripted"}


def first_rollout(hidden):
    env = AnalyticPushEnv()
    pol = InverseTrainer().train(Randomization.none())
    return env.rollout(hidden, pol, TARGETS), env.rollout(ParamSet.nominal(), pol, TARGETS)


def test_model_roundtrip_keeps_patch_and_perception():
    p = ParamSet.nominal().with_(object_mu=0.6, table_mu=0.6, patch_y0=0.4, patch_mu=0.5, lens_k=0.1, actuator_gain=0.9)
    assert ParamSet.nominal().diff(to_params(from_params(p), ParamSet.nominal())) == ParamSet.nominal().diff(p)


def test_fit_hypothesis_recovers_a_friction_patch_the_rules_cannot_express():
    real, _ = first_rollout(PATCH)
    wb = Workbench(real, ParamSet.nominal(), None, 0)
    flat = wb.fit_hypothesis(from_params(ParamSet.nominal()), ["mu_eff", "actuator_gain"])
    patch = wb.fit_hypothesis({**from_params(ParamSet.nominal()), "patch_y0": 0.3, "patch_mu": 0.8},
                              ["mu_eff", "actuator_gain", "patch_y0", "patch_mu"])
    assert flat["stop_residual_rms_m"] > 0.02
    assert patch["stop_residual_rms_m"] < 0.003
    assert abs(patch["fitted_model"]["patch_y0"] - 0.35) < 0.01 and abs(patch["fitted_model"]["patch_mu"] - 0.45) < 0.02


def test_probe_counts_against_budget_and_real_trials():
    real_w = RealWorld(AnalyticPushEnv(), PATCH)
    ro, _ = first_rollout(PATCH)
    wb = Workbench(ro, ParamSet.nominal(), real_w, 3)
    out = wb.probe_real([3.0, 3.5, 4.0, 4.5])
    assert len(out["results"]) == 3 and out["probe_budget_left"] == 0 and real_w.trials_used == 3
    assert "error" in wb.probe_real([2.0])


def test_scripted_tool_agent_closes_patch_gap_in_the_loop():
    fit_args = {"model": {**from_params(ParamSet.nominal()), "patch_y0": 0.3, "patch_mu": 0.8},
                "free": ["mu_eff", "actuator_gain", "patch_y0", "patch_mu"]}
    commit = {"model": {**from_params(ParamSet.nominal()), "patch_y0": 0.35, "patch_mu": 0.45}, "explanation": "patch"}
    llm = RecordedLLM({"chat:diagnose": [turn(call("decel_profile", {})), turn(call("fit_hypothesis", fit_args, 1)),
                                         turn(call("commit", commit, 2))]})
    real = RealWorld(AnalyticPushEnv(), PATCH)
    res = run_loop(real, InverseTrainer(), ToolAgentDiagnoser(llm, real), HeuristicPlanner(half_width_frac=0.01), TARGETS)
    assert res.log[0].real_success < 0.7
    assert res.final_success >= 0.9
    d = res.log[0].diagnosis
    assert [t["tool"] for t in d["trace"]] == ["decel_profile", "fit_hypothesis", "commit"]
    assert {s["name"] for s in d["suspects"]} == {"patch_y0", "patch_mu"}


def test_tool_agent_falls_back_when_it_never_commits():
    llm = RecordedLLM({"chat:diagnose": [turn(call("decel_profile", {}))] * 3})
    real, sim = first_rollout(ParamSet.nominal().with_(actuator_gain=0.8))
    d = ToolAgentDiagnoser(llm, max_steps=3).diagnose(real, sim, ParamSet.nominal())
    assert d.model.endswith("fallback TrajectoryDiagnoser") and d.suspects[0].name == "actuator_gain"


def test_tool_schemas_are_well_formed():
    names = [t["function"]["name"] for t in TOOLS]
    assert names == ["decel_profile", "perception_check", "fit_hypothesis", "test_hypothesis", "probe_real", "commit"]
    json.dumps(TOOLS)
