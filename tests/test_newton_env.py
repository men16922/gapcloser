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
