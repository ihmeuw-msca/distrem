import numpy as np
import pytest

from ensemble.distributions import (
    Beta,
    Exponential,
    Fisk,
    Gamma,
    GumbelR,
    InvGamma,
    LogNormal,
    Normal,
    Weibull,
)

# from scipy.stats import expon


# @pytest.mark.parametrize("a, b, expected", [(1, 2, 3), (2, 3, 5)])
# def test_add(a, b, expected):
#     assert add(a, b) == expected
NEG_MEAN = -2
BETA_MEAN = 0.5
BETA_VARIANCE = 0.2
MEAN = 2
VARIANCE = 8


def test_exp():
    exp = Exponential(MEAN, VARIANCE)
    res = exp.stats(moments="mv")
    exp_var = MEAN**2
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], exp_var)


def test_gamma():
    gamma = Gamma(MEAN, VARIANCE)
    res = gamma.stats(moments="mv")
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], VARIANCE)


def test_invgamma():
    invgamma = InvGamma(MEAN, VARIANCE)
    res = invgamma.stats(moments="mv")
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], VARIANCE)


def test_fisk():
    fisk = Fisk(MEAN, VARIANCE)
    res = fisk.stats(moments="mv")
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], VARIANCE)


def test_gumbel():
    gumbel = GumbelR(MEAN, VARIANCE)
    res = gumbel.stats(moments="mv")
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], VARIANCE)

    gumbel = GumbelR(NEG_MEAN, VARIANCE)
    res = gumbel.stats(moments="mv")
    assert np.isclose(res[0], NEG_MEAN)
    assert np.isclose(res[1], VARIANCE)


def test_weibull():
    weibull = Weibull(MEAN, VARIANCE)
    res = weibull.stats(moments="mv")
    print("resulting mean and var: ", res)
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], VARIANCE)


def test_lognormal():
    lognormal = LogNormal(MEAN, VARIANCE)
    res = lognormal.stats(moments="mv")
    print("resulting mean and var: ", res)
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], VARIANCE)


def test_normal():
    norm = Normal(MEAN, VARIANCE)
    res = norm.stats(moments="mv")
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], VARIANCE)

    norm = Normal(NEG_MEAN, VARIANCE)
    res = norm.stats(moments="mv")
    assert np.isclose(res[0], NEG_MEAN)
    assert np.isclose(res[1], VARIANCE)


def test_beta():
    beta = Beta(BETA_MEAN, VARIANCE)
    res = beta.stats(moments="mv")
    print("resulting mean and var: ", res)
    assert np.isclose(res[0], BETA_MEAN)
    assert np.isclose(res[1], VARIANCE)


def test_diff_supports():
    # negative means for only positive RVs
    with pytest.raises(ValueError):
        Exponential(NEG_MEAN, VARIANCE)
    with pytest.raises(ValueError):
        Gamma(NEG_MEAN, VARIANCE)
    with pytest.raises(ValueError):
        InvGamma(NEG_MEAN, VARIANCE)
    with pytest.raises(ValueError):
        Fisk(NEG_MEAN, VARIANCE)

    # mean outside of 0 and 1 for Beta
    with pytest.raises(ValueError):
        Beta(NEG_MEAN, VARIANCE)
