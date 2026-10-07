import random

import pytest

from sim.params import (CLOSED_PARAMS, OPEN_PARAMS, PARAM_SPACE, ConfigDiff, ParamSet, Randomization,
                        effective_friction, sample_hidden)


def test_param_space_has_ten_closed_and_three_open_params_with_valid_bounds():
    assert len(CLOSED_PARAMS) == 10 and set(OPEN_PARAMS) == {"patch_y0", "patch_mu", "lens_k"}
    for p in PARAM_SPACE.values():
        assert p.low <= p.nominal <= p.high
        assert p.kind in {"dynamics", "perception"}


def test_effective_friction_is_mean():
    assert effective_friction(0.3, 1.0) == pytest.approx(0.65)


def test_sample_hidden_is_deterministic_and_perturbs_exactly_n():
    a = sample_hidden(random.Random(7), 3)
    b = sample_hidden(random.Random(7), 3)
    assert a.values == b.values
    diff = ParamSet.nominal().diff(a)
    assert len(diff) == 3
    for name, (_, v) in diff.items():
        p = PARAM_SPACE[name]
        assert p.low <= v <= p.high


def test_sample_hidden_rejects_bad_n():
    with pytest.raises(ValueError):
        sample_hidden(random.Random(0), 0)


def test_with_rejects_unknown_param():
    with pytest.raises(KeyError):
        ParamSet.nominal().with_(gravity=1.0)


def test_randomization_apply_clamps_to_bounds_and_orders():
    r = Randomization.none().apply(ConfigDiff({"object_mu": (5.0, -1.0)}))
    lo, hi = r.ranges["object_mu"]
    assert (lo, hi) == (PARAM_SPACE["object_mu"].low, PARAM_SPACE["object_mu"].high)


def test_full_randomization_samples_inside_bounds():
    rng = random.Random(0)
    for _ in range(50):
        s = Randomization.full().sample(rng)
        for k, v in s.values.items():
            assert PARAM_SPACE[k].low <= v <= PARAM_SPACE[k].high


def test_config_diff_validate_rejects_unknown():
    with pytest.raises(KeyError):
        ConfigDiff({"nope": (0, 1)}).validate()


def test_old_configs_without_open_params_read_nominal():
    old = ParamSet({k: PARAM_SPACE[k].nominal for k in CLOSED_PARAMS})
    assert old["lens_k"] == 0.0 and old["patch_y0"] == PARAM_SPACE["patch_y0"].nominal
    assert ParamSet.nominal().diff(old) == {}


def test_closed_sampling_never_touches_open_params():
    for seed in range(20):
        h = sample_hidden(random.Random(seed), 3)
        assert not set(ParamSet.nominal().diff(h)) & set(OPEN_PARAMS)
