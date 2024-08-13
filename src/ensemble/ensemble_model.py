from typing import List

import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.stats

from ensemble.distributions import distribution_dict


class EnsembleResult:
    weights: npt.NDArray
    ensemble_density: npt.NDArray

    def __init__(self, weights, ensemble_density) -> None:
        self.weights = weights
        self.ensemble_density = ensemble_density


# class EnsembleModel:
#     def __init__(self, distributions, weights):
#         self.distributions =


class EnsembleFitter:
    def __init__(self, distributions: List[str], objective):
        self.supports = set()
        for distribution in distributions:
            self.supports.add(distribution_dict[distribution].support)
        # TODO: HOW SHOULD WE TELL THE USER WHICH DISTRIBUTION IS THE "TROUBLEMAKER"?
        if len(self.supports) != 1:
            raise ValueError(
                "the provided list of distributions do not all have the same support: "
                + str(self.supports)
            )
        self.distributions = distributions

    def objective(self, vec):
        return scipy.linalg.norm(vec, 2)

    def ensemble_obj(self, weights):
        # return data - F @ weights
        pass

    def get_ensemble_density(
        self, pdfs: np.ndarray, fitted_weights: np.ndarray
    ):
        return pdfs @ fitted_weights

    def ensemble_func(self, weights: list, ecdf: np.ndarray, cdfs: np.ndarray):
        return self.objective(ecdf - cdfs @ weights)

    def fit(self, data: npt.ArrayLike) -> EnsembleResult:
        # TODO: SWITCH CASE STATEMENT FOR BOUNDS OF DATA NOT MATCHING THE ELEMENT OF SELF.SUPPORTS
        # sample stats, ecdf
        sample_mean = np.mean(data)
        sample_variance = np.var(data, ddof=1)
        ecdf = scipy.stats.ecdf(data).cdf.probabilities

        # may be able to condense into 1 line if ub and lb are not used elsewhere
        # support_lb = np.min(data)
        # support_ub = np.max(data)
        # support = np.linspace(support_lb, support_ub, len(data))
        equantiles = scipy.stats.ecdf(data).cdf.quantiles

        # fill matrix with cdf values over support of data
        num_distributions = len(self.distributions)
        cdfs = np.zeros((len(data), num_distributions))
        pdfs = np.zeros((len(data), num_distributions))
        for i in range(num_distributions):
            curr_dist = distribution_dict[self.distributions[i]](
                sample_mean, sample_variance
            )
            cdfs[:, i] = curr_dist.cdf(equantiles)
            pdfs[:, i] = curr_dist.pdf(equantiles)

        # initialize equal weights for all dists and optimize
        initial_guess = np.zeros(num_distributions) + 1 / num_distributions
        bounds = tuple((0, 1) for i in range(num_distributions))
        minimizer_result = scipy.optimize.minimize(
            fun=self.ensemble_func,
            x0=initial_guess,
            args=(ecdf, cdfs),
            bounds=bounds,
        )
        fitted_weights = minimizer_result.x

        # return minimizer_result.x

        res = EnsembleResult(
            weights=fitted_weights,
            ensemble_density=self.get_ensemble_density(cdfs, fitted_weights),
        )

        return res


### HELPER FUNCTIONS

### HELPER FUNCTIONS
