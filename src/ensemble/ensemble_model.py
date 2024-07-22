import numpy as np
import scipy.optimize
from numpy.typing import NDArray


class EnsembleResult:
    weights: NDArray
    ensemble_density: NDArray


class EnsembleFitter:
    def __init__(self, data, distributions, objective):
        pass

    def ensemble_obj(self, weights):
        # return data - F @ weights
        pass

    def ensemble_density(self, fitted_weights):
        # return F @ fitted_weights
        pass

    def fit(
        self,
    ):
        initial_guess = None
        minimizer_result = scipy.optimize.minimize(
            self.ensemble_func, initial_guess
        )

        res = EnsembleResult(
            weights=minimizer_result.x,
            ensemble_density=self.ensemble_density(minimizer_result.x),
        )

        return res
