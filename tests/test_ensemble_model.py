import numpy as np

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

# ENSEMBLE_POS_DRAWS = EnsembleModel(
#     distributions=["exponential", "lognormal"],
#     weights=[0.5, 0.5],
#     mean=5,
#     variance=2,
# ).rvs(size=100)


def test_1_dist():
    model = EnsembleFitter(["normal"], None)
    res = model.fit(STD_NORMAL_DRAWS)
    print(res.weights)
    assert np.isclose(res.weights[0], 1)

    wrong_model = EnsembleFitter(["normal", "gumbel"], None)
    res = wrong_model.fit(STD_NORMAL_DRAWS)
    print(res.weights)
    assert np.allclose(res.weights, [1, 0])


def test_2_real_line_dists():
    model1 = EnsembleFitter(["normal", "gumbel"], None)
    res1 = model1.fit(ENSEMBLE_RL_DRAWS)
    print(res1.weights)
    assert np.allclose(res1.weights, [0.7, 0.3])


def test_2_positive_dists():
    model2 = EnsembleFitter(["exponential", "lognormal"], None)
    res2 = model2.fit(ENSEMBLE_POS_DRAWS)
    print(res2.weights)
    assert np.allclose(res2.weights, [0.5, 0.5])
