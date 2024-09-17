import numpy as np
import pytest
import scipy.stats as stats

from ensemble.model import EnsembleDistribution, EnsembleFitter

STD_NORMAL_DRAWS = stats.norm(loc=0, scale=1).rvs(100)

ENSEMBLE_RL_DRAWS = EnsembleDistribution(
    distributions=["normal", "gumbel"], weights=[0.7, 0.3], mean=0, variance=1
).rvs(size=100)

ENSEMBLE_POS_DRAWS = EnsembleDistribution(
    distributions=["exponential", "lognormal"],
    weights=[0.5, 0.5],
    mean=5,
    variance=1,
).rvs(size=100)

ENSEMBLE_POS_DRAWS2 = EnsembleDistribution(
    distributions=["exponential", "lognormal", "fisk"],
    weights=[0.3, 0.5, 0.2],
    mean=40,
    variance=5,
)


DEFAULT_SETTINGS = ([0.5, 0.5], 1, 1)


def test_bad_weights():
    with pytest.raises(ValueError):
        EnsembleDistribution(["normal", "gumbel"], [1, 0.1], 1, 1)
    with pytest.raises(ValueError):
        EnsembleDistribution(["normal", "gumbel"], [0.3, 0.69], 1, 1)


def test_incompatible_dists():
    with pytest.raises(ValueError):
        EnsembleDistribution(["normal", "exponential"], *DEFAULT_SETTINGS)
    with pytest.raises(ValueError):
        EnsembleDistribution(["beta", "normal"], *DEFAULT_SETTINGS)
    with pytest.raises(ValueError):
        EnsembleDistribution(["beta", "exponential"], *DEFAULT_SETTINGS)


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
