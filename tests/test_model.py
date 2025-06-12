import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
from cvxpy.error import SolverError

from distrem.distributions import distribution_dict
from distrem.model import EnsembleDistribution, EnsembleFitter, SDOptimizer

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
).rvs(size=100)


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
        EnsembleFitter(["Exponential", "Fisk"], "L1").fit(neg_data)
    with pytest.raises(ValueError):
        EnsembleFitter(["Beta"], "L1").fit(neg_data)


def test_resulting_weights():
    model = EnsembleFitter(["Normal"], "L1")
    res = model.fit(STD_NORMAL_DRAWS)
    assert np.isclose(np.sum(res.weights), 1)

    model1 = EnsembleFitter(["Normal", "GumbelR"], "L1")
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


def test_objective_funcs():
    bad_obj = EnsembleFitter(["Normal"], "not_an_obj_func")
    with pytest.raises(NotImplementedError):
        bad_obj.fit(STD_NORMAL_DRAWS)
    model_L1 = EnsembleFitter(["Fisk", "Gamma"], "L1")
    model_L1.fit(ENSEMBLE_POS_DRAWS)
    model_L1.fit(ENSEMBLE_POS_DRAWS2)

    model_sumsq = EnsembleFitter(["Fisk", "Gamma"], "sum_squares")
    model_sumsq.fit(ENSEMBLE_POS_DRAWS)
    model_sumsq.fit(ENSEMBLE_POS_DRAWS2)

    model_KS = EnsembleFitter(["Fisk", "Gamma"], "KS")
    model_KS.fit(ENSEMBLE_POS_DRAWS)
    model_KS.fit(ENSEMBLE_POS_DRAWS2)


def test_duplicates():
    model = EnsembleFitter(["Weibull", "Gamma"], "L1")
    with pytest.raises(SolverError):
        model.fit(np.array([2.7, 2.7]))


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
    assert m1.ensemble_stats("mv") == DEFAULT_SETTINGS
    assert m1._distributions == ["Gamma", "InvGamma"]
    assert m1._weights == [0.2, 0.8]


def test_restricted_moments():
    mean = 4
    variance = 1
    ex_bounded = EnsembleDistribution({"Gamma": 0.7, "Fisk": 0.3}, 4, 1, lb=2)
    bounded_rvs = ex_bounded.rvs(1000000)
    assert np.isclose(np.mean(bounded_rvs), mean, atol=1e-02)
    assert np.isclose(np.var(bounded_rvs, ddof=1), variance, atol=1e-02)


def test_tsh_weave():
    modelRL = EnsembleFitter(["Normal"], "sum_squares")
    # incomplete parameters throw errors
    with pytest.raises(ValueError):
        modelRL.fit(STD_NORMAL_DRAWS, tsh_pts=[0.5, 1])
    with pytest.raises(ValueError):
        modelRL.fit(STD_NORMAL_DRAWS, tsh_wts=[0.5, 0.5])

    modelPOS = EnsembleFitter(["Gamma", "LogNormal"], "sum_squares")
    # tsh out of range or wts dont sum to 1
    with pytest.raises(ValueError):
        modelPOS.fit(STD_NORMAL_DRAWS, tsh_pts=[-1, 1], tsh_wts=[0.5, 0.5])
    with pytest.raises(ValueError):
        modelPOS.fit(STD_NORMAL_DRAWS, tsh_pts=[0.5, 1], tsh_wts=[0.1, 1])
    with pytest.raises(ValueError):
        modelPOS.fit(STD_NORMAL_DRAWS, tsh_pts=[0.5, 1], tsh_wts=[0.1, 0.8])

    mod = EnsembleFitter(["Normal", "GumbelR"], "sum_squares")
    mod.fit(
        data=STD_NORMAL_DRAWS,
        tsh_pts=[-0.25, 0.33, 0.7],
        tsh_wts=[0.5, 0.3, 0.2],
    )


def test_expSD():
    correct_mean = 14
    target_sd = 7
    target_dist = EnsembleDistribution(
        named_weights={"Gamma": 0.6, "Weibull": 0.4},
        mean=correct_mean,
        variance=target_sd**2,
    )
    q0, q1, q2 = 23, 26, 27
    target_prev = target_dist.cdf([q0, q1, q2])
    prev0, prev1, prev2 = (
        target_prev[1] - target_prev[0],
        target_prev[2] - target_prev[1],
        1 - target_prev[2],
    )
    p_hat = [prev0, prev1, prev2]

    model = SDOptimizer(correct_mean, {"LogNormal": 0.2, "InvGamma": 0.8})
    df = pd.DataFrame(
        data={
            "weights": [0.1, 0.4, 0.5],
            "lb": [q0, q1, q2],
            "ub": [q1, q2, np.inf],
            "prev": p_hat,
        }
    )
    model.optimize_sd(df)

    p_hat.append(0.01)
    df_dupe = pd.DataFrame(
        data={
            "weights": [0.1, 0.4, 0.4, 0.1],
            "lb": [q0, q1, q2, q2],
            "ub": [q1, q2, np.inf, np.inf],
            "prev": p_hat,
        }
    )
    model.optimize_sd(df_dupe)

    df_wrong = pd.DataFrame(
        data={
            "weights": [0.2, 0.4, 0.4, 0.1],
            "lb": [q0, q1, q2, q2],
            "ub": [q1, q2, np.inf, np.inf],
            "prev": p_hat,
        }
    )
    with pytest.raises(ValueError):
        model.optimize_sd(df_wrong)
