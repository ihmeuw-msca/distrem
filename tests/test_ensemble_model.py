import numpy as np
import pytest

from ensemble.distributions import distribution_dict
from ensemble.ensemble_model import EnsembleFitter, EnsembleModel

STD_NORMAL_DRAWS = distribution_dict["normal"](0, 1).rvs(100)

ENSEMBLE_RL_DRAWS = EnsembleModel(
    distributions=["normal", "gumbel"], weights=[0.7, 0.3], mean=0, variance=1
).rvs(size=100)

ENSEMBLE_POS_DRAWS = EnsembleModel(
    distributions=["exponential", "lognormal"],
    weights=[0.5, 0.5],
    mean=5,
    variance=1,
).rvs(size=100)

ENSEMBLE_POS_DRAWS2 = EnsembleModel(
    distributions=["exponential", "lognormal", "fisk"],
    weights=[0.3, 0.5, 0.2],
    mean=40,
    variance=5,
)
# ENSEMBLE_POS_DRAWS = EnsembleModel(
#     distributions=["exponential", "lognormal"],
#     weights=[0.5, 0.5],
#     mean=5,
#     variance=2,
# ).rvs(size=100)

DEFAULT_SETTINGS = ([0.5, 0.5], 1, 1)


def test_bad_weights():
    with pytest.raises(ValueError):
        EnsembleModel(["normal", "gumbel"], [1, 0.1], 1, 1)
    with pytest.raises(ValueError):
        EnsembleModel(["normal", "gumbel"], [0.3, 0.69], 1, 1)


def test_incompatible_dists():
    with pytest.raises(ValueError):
        EnsembleModel(["normal", "exponential"], *DEFAULT_SETTINGS)
    with pytest.raises(ValueError):
        EnsembleModel(["beta", "normal"], *DEFAULT_SETTINGS)
    with pytest.raises(ValueError):
        EnsembleModel(["beta", "exponential"], *DEFAULT_SETTINGS)


def test_incompatible_data():
    neg_data = np.linspace(-1, 1, 100)
    with pytest.raises(ValueError):
        EnsembleFitter(["exponential", "fisk"], "L2").fit(neg_data)
    with pytest.raises(ValueError):
        EnsembleFitter(["beta"], "L2").fit(neg_data)


def test_resulting_weights():
    model = EnsembleFitter(["normal"], "L2")
    res = model.fit(STD_NORMAL_DRAWS)
    assert np.isclose(np.sum(res.weights), 1)

    model1 = EnsembleFitter(["normal", "gumbel"], "L2")
    res1 = model1.fit(ENSEMBLE_RL_DRAWS)
    assert np.isclose(np.sum(res1.weights), 1)

    model2 = EnsembleFitter(["exponential", "lognormal", "fisk"], "KS")
    res2 = model2.fit(ENSEMBLE_POS_DRAWS)
    assert np.isclose(np.sum(res2.weights), 1)
