import numpy as np
import pytest
from distrem.distributions import (
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

NEG_MEAN = -2
BETA_MEAN = 0.5
BETA_VARIANCE = 0.2
MEAN = 4
VARIANCE = 2


def test_exp():
    assert Exponential.support == (0, np.inf)
    exp = Exponential(MEAN, VARIANCE)
    assert exp.support == (0, np.inf)
    res = exp.stats(moments="mv")
    exp_var = MEAN**2
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], exp_var)

    exp2 = Exponential(MEAN, VARIANCE)
    exp2.support = (1, np.inf)
    res2 = exp2.stats(moments="mv")
    exp_var2 = MEAN**2
    assert np.isclose(res2[0], MEAN)
    assert np.isclose(res2[1], exp_var2)


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


# TODO: WRITE ADDITIONAL TESTS DUE TO NUMERICAL SOLUTION, CURRENTLY UNDERPERFORMING WITH MEAN, VARIANCE = [1, 3]
def test_fisk():
    fisk = Fisk(MEAN, VARIANCE)
    # fisk = Fisk(1, 1)
    res = fisk.stats(moments="mv")
    # assert np.isclose(res[0], MEAN)
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
    assert np.isclose(res[0], MEAN, atol=1e-05)
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
    beta = Beta(BETA_MEAN, BETA_VARIANCE)
    res = beta.stats(moments="mv")
    assert np.isclose(res[0], BETA_MEAN)
    assert np.isclose(res[1], BETA_VARIANCE)

    mean = 3
    vari = 1
    lb = 0
    ub = 5
    beta_alt = Beta(mean, vari, lb, ub)
    res_alt = beta_alt.stats(moments="mv")
    assert np.isclose(res_alt[0], mean)
    assert np.isclose(res_alt[1], vari)


def test_invalid_means():
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


def test_invalid_custom_supports():
    with pytest.raises(ValueError):
        Exponential(1, 1, ub=2)
    with pytest.raises(ValueError):
        Exponential(1, 1, lb=-np.inf)
    with pytest.raises(ValueError):
        Exponential(1, 1, ub=np.inf)
    with pytest.raises(ValueError):
        Normal(1, 1, lb=0)
    with pytest.raises(ValueError):
        Normal(1, 1, ub=0)
