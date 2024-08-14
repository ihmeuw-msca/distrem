import numpy as np
import scipy.optimize as opt
import scipy.stats as stats

from ensemble.distributions import distribution_dict
from ensemble.ensemble_model import EnsembleFitter


def ensemble_cdf(x, distributions, weights, mean, variance):
    my_objs = []
    for distribution in distributions:
        my_objs.append(distribution_dict[distribution](mean, variance))
    return sum(
        weight * distribution.cdf(x)
        for distribution, weight in zip(my_objs, weights)
    )


def ppf_to_solve(x, q):
    return ensemble_cdf(x, ["normal", "gumbel"], [0.7, 0.3], 0, 1) - q


def ppf_single(q):
    factor = 10.0
    left = -factor
    right = factor

    while ppf_to_solve(left, q) > 0:
        left, right = left * factor, left

    while ppf_to_solve(right, q) < 0:
        left, right = right, right * factor

    return opt.brentq(ppf_to_solve, left, right, args=q)


def ensemble_rvs(size):
    ppf_vec = np.vectorize(ppf_single, otypes="d")
    unif_samp = stats.uniform.rvs(size=size)
    return ppf_vec(unif_samp)


STD_NORMAL_DRAWS = distribution_dict["normal"](0, 1).rvs(100)
# TODO: REMOVE HARDCODED RVS FUNCTION IN TEST FILE AND TEST MORE VARIED DISTRIBUTIONS ONCE THIS IS IMPLEMENTED IN ENSEMBLE_MODEL.PY
ENSEMBLE_RAND_DRAWS = ensemble_rvs(100)


def test_1_dist():
    model = EnsembleFitter(["normal"], None)
    res = model.fit(STD_NORMAL_DRAWS)
    print(res.weights)
    assert np.isclose(res.weights[0], 1)
    wrong_model = EnsembleFitter(["normal", "gumbel"], None)
    res = wrong_model.fit(STD_NORMAL_DRAWS)
    print(res.weights)
    assert np.allclose(res.weights, [1, 0])


def test_2_dists():
    model = EnsembleFitter(["normal", "gumbel"], None)
    res = model.fit(ENSEMBLE_RAND_DRAWS)
    print(res.weights)
    assert np.allclose(res.weights, [0.7, 0.3])
