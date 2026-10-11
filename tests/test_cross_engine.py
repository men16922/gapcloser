"""The second engine (MuJoCo, contact-driven launch) and the pattern check that tells missing physics from scatter."""

import math
import random

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from tether.sim.mujoco_push import MujocoPushEnv  # noqa: E402
from tether.sim.params import GRAVITY, ParamSet  # noqa: E402
from tether.sim.push_task import FRAME_DT  # noqa: E402


def _fit_track(track, stop):
    tr = np.asarray(track)
    t = np.arange(len(tr)) * FRAME_DT
    k = max(3, int(np.argmax(tr >= stop - 1e-4)))
    v0, a = np.linalg.lstsq(np.c_[t[:k], -0.5 * t[:k] ** 2], tr[:k], rcond=None)[0]
    return v0, a


def test_contact_launch_and_sliding_follow_coulomb_within_a_few_percent():
    p = ParamSet.nominal().with_(object_mu=0.45, table_mu=0.45, actuator_gain=0.9)
    env = MujocoPushEnv()
    for cmd in (1.4, 2.2, 3.0):
        stop, track, tipped = env._slide(p, cmd)
        v0, a = _fit_track(track, stop)
        assert not tipped
        assert abs(v0 / (cmd * 0.9) - 1) < 0.05  # the paddle's push, not an assigned velocity
        assert abs(a / (0.45 * GRAVITY) - 1) < 0.06  # MuJoCo's soft contact: a few % off the parameter


def test_friction_region_and_off_menu_effects_change_the_slide():
    p = ParamSet.nominal().with_(object_mu=0.5, table_mu=0.5, actuator_gain=1.0)
    plain = MujocoPushEnv()._slide(p, 2.6)[0]
    assert MujocoPushEnv()._slide(p.with_(patch_y0=0.3, patch_mu=0.3), 2.6)[0] > plain + 0.05
    assert MujocoPushEnv({"speed_slope": -0.2})._slide(p, 2.6)[0] > plain + 0.03  # slicker while fast
    assert MujocoPushEnv({"slope_deg": 2.5})._slide(p, 2.6)[0] > plain + 0.02  # downhill


def _log(effects, seed=3):
    from tether.eval.cross_engine import first_day_log, hidden_world

    h = hidden_world(random.Random(seed))
    return first_day_log(MujocoPushEnv(effects), ParamSet.nominal().with_(**h), seed), h


def test_pattern_check_finds_speed_dependence_but_not_plain_scatter():
    from tether.studio import structure
    from tether.studio.fit import calibrate

    s, _ = _log({})
    cal = calibrate(s, n_boot=0)
    assert cal.residuals["patterns"]["findings"] == []  # MuJoCo's launch scatter is not structure
    s, _ = _log({"speed_slope": -0.2})
    found = structure.check(s, calibrate(s, n_boot=0).params)["findings"]
    assert [f["kind"] for f in found] == ["speed"] and found[0]["per_mps"] < 0
    assert math.isfinite(found[0]["t"])
