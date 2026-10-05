import json

from agent.loop import HeuristicDiagnoser, HeuristicPlanner, RealWorld, TrajectoryDiagnoser, fit_launch, run_loop
from eval.compare import run, summarize
from sim.params import ParamSet, Randomization
from sim.push_task import AnalyticPushEnv, GridTrainer, analytic_track, eval_targets

TARGETS = eval_targets(20, 1000)


def test_nominal_policy_succeeds_in_nominal_world():
    env = AnalyticPushEnv()
    pol = GridTrainer(env).train(Randomization.none())
    assert env.rollout(ParamSet.nominal(), pol, TARGETS).success_rate == 1.0


def test_loop_closes_friction_gap_and_writes_log(tmp_path):
    env = AnalyticPushEnv()
    real = RealWorld(env, ParamSet.nominal().with_(object_mu=0.3))
    log = tmp_path / "run.json"
    res = run_loop(real, GridTrainer(env), HeuristicDiagnoser(), HeuristicPlanner(), TARGETS, log_path=log)
    assert res.log[0].real_success < 0.5  # gap exists before the loop acts
    assert res.final_success >= 0.9
    assert res.iterations >= 2
    data = json.loads(log.read_text())
    assert data["log"][0]["diagnosis"]["suspects"][0]["name"] in {"object_mu", "table_mu"}


def test_loop_closes_camera_offset_gap():
    env = AnalyticPushEnv()
    real = RealWorld(env, ParamSet.nominal().with_(camera_dx=-0.025))
    res = run_loop(real, GridTrainer(env), HeuristicDiagnoser(), HeuristicPlanner(), TARGETS)
    assert res.final_success >= 0.9


def test_loop_runs_three_iterations_when_goal_unreachable():
    env = AnalyticPushEnv()
    # actuator gain confound: friction compensation clamps at bounds -> heuristic cannot fully close
    real = RealWorld(env, ParamSet.nominal().with_(actuator_gain=0.72))
    res = run_loop(real, GridTrainer(env), HeuristicDiagnoser(), HeuristicPlanner(), TARGETS, max_iter=3)
    assert res.iterations == 3
    assert res.train_calls == 3


def test_compare_is_deterministic_and_gapcloser_beats_baselines():
    a = summarize(run(n_worlds=4, seed=0))
    b = summarize(run(n_worlds=4, seed=0))
    assert a == b
    for g in ("gapcloser_outcome", "gapcloser_traj"):
        assert a[g]["mean_success"] > a["full_dr"]["mean_success"]
        assert a[g]["mean_success"] > a["nominal"]["mean_success"]
    assert a["gapcloser_traj"]["mean_success"] >= a["gapcloser_outcome"]["mean_success"]


def test_fit_launch_recovers_speed_and_deceleration():
    p = ParamSet.nominal().with_(actuator_gain=1.2, object_mu=0.4)
    v0, a = fit_launch(analytic_track(2.0, p))
    assert abs(v0 - 2.4) < 1e-6
    assert abs(a - 0.6 * 9.81) < 1e-6


def test_trajectory_diagnoser_breaks_gain_confound():
    env = AnalyticPushEnv()
    real = RealWorld(env, ParamSet.nominal().with_(actuator_gain=0.76))
    outcome = run_loop(real, GridTrainer(env), HeuristicDiagnoser(), HeuristicPlanner(), TARGETS)
    traj = run_loop(real, GridTrainer(env), TrajectoryDiagnoser(), HeuristicPlanner(), TARGETS)
    assert outcome.final_success < 0.5
    assert traj.final_success >= 0.9
    first = traj.log[0].diagnosis["suspects"][0]
    assert first["name"] == "actuator_gain" and abs(first["estimate"] - 0.76) < 0.01


def test_tipped_trials_do_not_drive_friction_changes():
    """Tipped cubes stop short for reasons outside the sliding model; the diagnosers must ignore them."""
    from agent.loop import HeuristicDiagnoser, TrajectoryDiagnoser
    from sim.push_task import Rollout

    env = AnalyticPushEnv()
    pol = GridTrainer(env).train(Randomization.none())
    sim = env.rollout(ParamSet.nominal(), pol, TARGETS)
    real = env.rollout(ParamSet.nominal(), pol, TARGETS)
    for t in real.trials[:8]:  # 8 cubes tipped and stopped early
        t.tipped, t.slide = True, t.slide * 0.5
    real = Rollout(real.trials)
    for d in (HeuristicDiagnoser(), TrajectoryDiagnoser()):
        names = [s.name for s in d.diagnose(real, sim, ParamSet.nominal()).suspects]
        assert "object_mu" not in names and "table_mu" not in names
