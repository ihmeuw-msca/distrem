import numpy as np
import pytest
import scipy.stats as stats

from ensemble.distributions import distribution_dict
from ensemble.model import EnsembleDistribution, EnsembleFitter

STD_NORMAL_DRAWS = stats.norm(loc=0, scale=1).rvs(100)

ENSEMBLE_RL_DRAWS = EnsembleDistribution(
    named_weights={"Normal": 0.7, "GumbelR": 0.3}, mean=0, variance=1
).rvs(size=100)

ENSEMBLE_POS_DRAWS = EnsembleDistribution(
    named_weights={"Exponential": 0.5, "LogNormal": 0.5},
    mean=5,
    variance=1,
).rvs(size=100)

ENSEMBLE_POS_DRAWS2 = EnsembleDistribution(
    named_weights={"Exponential": 0.3, "LogNormal": 0.5, "Fisk": 0.2},
    mean=40,
    variance=5,
)


DEFAULT_SETTINGS = (1, 1)


def test_bad_weights():
    with pytest.raises(ValueError):
        EnsembleDistribution({"Normal": 1, "GumbelR": 0.1}, *DEFAULT_SETTINGS)
    with pytest.raises(ValueError):
        EnsembleDistribution(
            {"Normal": 0.3, "GumbelR": 0.69}, *DEFAULT_SETTINGS
        )


def test_incompatible_dists():
    with pytest.raises(ValueError):
        EnsembleDistribution(
            {"Normal": 0.5, "Exponential": 0.5}, *DEFAULT_SETTINGS
        )
    with pytest.raises(ValueError):
        EnsembleDistribution({"Beta": 0.5, "Normal": 0.5}, *DEFAULT_SETTINGS)
    with pytest.raises(ValueError):
        EnsembleDistribution(
            {"Beta": 0.5, "Exponential": 0.5}, *DEFAULT_SETTINGS
        )


def test_incompatible_data():
    neg_data = np.linspace(-1, 1, 100)
    with pytest.raises(ValueError):
        EnsembleFitter(["Exponential", "Fisk"], "L2").fit(neg_data)
    with pytest.raises(ValueError):
        EnsembleFitter(["Beta"], "L2").fit(neg_data)


def test_resulting_weights():
    model = EnsembleFitter(["Normal"], "L2")
    res = model.fit(STD_NORMAL_DRAWS)
    assert np.isclose(np.sum(res.weights), 1)

    model1 = EnsembleFitter(["Normal", "GumbelR"], "L2")
    res1 = model1.fit(ENSEMBLE_RL_DRAWS)
    assert np.isclose(np.sum(res1.weights), 1)

    model2 = EnsembleFitter(["Exponential", "LogNormal", "Fisk"], "KS")
    res2 = model2.fit(ENSEMBLE_POS_DRAWS)
    assert np.isclose(np.sum(res2.weights), 1)


def test_bounds():
    with pytest.raises(ValueError):
        EnsembleDistribution({"Normal": 0.5, "GumbelR": 0.5}, 1, 1, lb=0)
    with pytest.raises(ValueError):
        EnsembleDistribution({"Normal": 0.5, "GumbelR": 0.5}, 1, 1, ub=0)
    with pytest.raises(ValueError):
        EnsembleDistribution({"Exponential": 0.5, "Gamma": 0.5}, 1, 1, ub=0)
    with pytest.raises(ValueError):
        EnsembleDistribution({"Exponential": 0.5, "Gamma": 0.5}, 1, 1, lb=4)


def test_from_obj():
    gamma1 = distribution_dict["Gamma"](7, 1, lb=3)
    # diff mean/var
    gamma2 = distribution_dict["Gamma"](6, 1, lb=3)
    logn1 = distribution_dict["LogNormal"](7, 1, lb=3)
    # diff bounds
    logn2 = distribution_dict["LogNormal"](7, 1, lb=2)

    # unweighted
    with pytest.raises(ValueError):
        EnsembleDistribution.from_objs([gamma1, logn1])

    # weighted
    gamma1._weight = 0.5
    logn1._weight = 0.5
    EnsembleDistribution.from_objs([gamma1, logn1])
    # weights that dont sum to 1
    gamma1._weight = 0.4
    logn1._weight = 0.5
    with pytest.raises(ValueError):
        EnsembleDistribution.from_objs([gamma1, logn1])

    # mean/var mismatch
    with pytest.raises(ValueError):
        EnsembleDistribution.from_objs([gamma2, logn1])
    with pytest.raises(ValueError):
        EnsembleDistribution.from_objs([gamma1, logn2])


def test_json():
    model0 = EnsembleDistribution(
        {"Normal": 0.5, "GumbelR": 0.5}, *DEFAULT_SETTINGS
    )
    model0.to_json("tests/test_read.json")
    model1 = EnsembleDistribution(
        {"Gamma": 0.2, "InvGamma": 0.8}, *DEFAULT_SETTINGS
    )
    model1.to_json("tests/test_read.json", appending=True)

    m1 = EnsembleDistribution.from_json("tests/test_read.json")[1]
    assert m1.stats_temp("mv") == DEFAULT_SETTINGS
    assert m1._distributions == ["Gamma", "InvGamma"]
    assert m1._weights == [0.2, 0.8]
