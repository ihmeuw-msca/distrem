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
MEAN = 2
VARIANCE = 8


def test_exp():
    # x = np.linspace(0, 1, num=10)
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
    print("resulting mean and var: ", res)
    # assert False
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], VARIANCE)


def test_gumbel():
    gumbel = GumbelR(MEAN, VARIANCE)
    res = gumbel.stats(moments="mv")
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], VARIANCE)


def test_weibull():
    raise NotImplementedError


def test_lognormal():
    raise NotImplementedError


def test_normal():
    norm = Normal(MEAN, VARIANCE)
    res = norm.stats(moments="mv")
    assert np.isclose(res[0], MEAN)
    assert np.isclose(res[1], VARIANCE)


def test_beta():
    raise NotImplementedError
