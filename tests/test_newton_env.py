import pytest

pytest.importorskip("newton")

from agent.loop import HeuristicPlanner, RealWorld, TrajectoryDiagnoser, run_loop  # noqa: E402
from sim.newton_push import NewtonPushEnv  # noqa: E402
from sim.params import ParamSet, Randomization  # noqa: E402
from sim.push_task import AnalyticPushEnv, GridTrainer, eval_targets  # noqa: E402

TARGETS = eval_targets(20, 1000)


def test_newton_matches_analytic_surrogate_within_1cm():
    pol = GridTrainer(AnalyticPushEnv()).train(Randomization.none())
    for p in (ParamSet.nominal(), ParamSet.nominal().with_(object_mu=0.3, actuator_gain=1.2)):
        n = NewtonPushEnv().rollout(p, pol, TARGETS)
        a = AnalyticPushEnv().rollout(p, pol, TARGETS)
        assert max(abs(x.slide - y.slide) for x, y in zip(n.trials, a.trials)) < 0.01
        assert len(n.trials[0].track) == len(a.trials[0].track)


def test_newton_loop_closes_weak_motor_gap():
    env = NewtonPushEnv()
    real = RealWorld(env, ParamSet.nominal().with_(actuator_gain=0.76))
    res = run_loop(real, GridTrainer(AnalyticPushEnv()), TrajectoryDiagnoser(), HeuristicPlanner(), TARGETS, sim_env=env)
    assert res.log[0].real_success < 0.2
    assert res.final_success >= 0.9


def test_tipping_is_flagged_only_at_high_friction():
    from sim.push_task import Policy

    env = NewtonPushEnv()
    low = env.rollout(ParamSet.nominal(), Policy(2 * 0.8 * 9.81), TARGETS)
    high = env.rollout(ParamSet.nominal().with_(object_mu=1.0, table_mu=1.0), Policy(2 * 1.0 * 9.81), TARGETS)
    assert sum(t.tipped for t in low.trials) == 0
    assert sum(t.tipped for t in high.trials) >= 5


def test_extract_frames_from_rendered_clip(tmp_path):
    from eval.record_demo import extract_frames
    from sim.newton_push import render_trial

    clip = tmp_path / "c.webp"
    render_trial(ParamSet.nominal(), 2.5, 0.4, clip)
    frames = extract_frames(clip, tmp_path / "frames")
    assert 2 <= len(frames) <= 3 and all(f.exists() and f.suffix == ".png" for f in frames)


def test_newton_friction_patch_matches_analytic_without_tipping():
    from sim.push_task import slide_distance

    cmds = [1.5, 2.5, 3.0, 3.4]
    for p in (ParamSet.nominal().with_(patch_y0=0.35, patch_mu=0.3), ParamSet.nominal().with_(patch_y0=0.3, patch_mu=1.1)):
        ro = NewtonPushEnv().push(p, cmds)
        assert not any(t.tipped for t in ro.trials)
        assert max(abs(t.slide - slide_distance(c, p)) for t, c in zip(ro.trials, cmds)) < 0.015
