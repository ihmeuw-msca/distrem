import numpy as np
import scipy.optimize
import scipy.stats
from numpy.typing import NDArray

from ensemble.distributions import distribution_dict


class EnsembleResult:
    weights: NDArray
    ensemble_density: NDArray


class EnsembleFitter:
    def __init__(self, distributions, objective):
        # distribution_dict = {
        #     "exponential": Expone
        # }
        self.distributions = distributions

    def ensemble_obj(self, weights):
        # return data - F @ weights
        pass

    def ensemble_density(self, cdfs, fitted_weights):
        return cdfs @ fitted_weights

    def ensemble_func(self, weights, ecdf, cdfs):
        return ecdf - cdfs @ weights

    def fit(self, data):
        # sample stats, ecdf
        sample_mean = np.mean(data)
        sample_variance = np.var(data, ddof=1)
        ecdf = scipy.stats.ecdf(data)

        # may be able to condense into 1 line if ub and lb are not used elsewhere
        support_lb = np.min(data)
        support_ub = np.max(data)
        support = np.linspace(support_lb, support_ub, len(data))

        # fill matrix with cdf values over support of data
        num_distributions = len(self.distributions)
        cdfs = np.zeros(len(data), num_distributions)
        for i in range(num_distributions):
            curr_dist = distribution_dict[self.distributions[i]](
                sample_mean, sample_variance
            )
            cdfs[:, i] = curr_dist.cdf(support)

        # initialize equal weights for all dists and optimize
        initial_guess = np.zeros(num_distributions) + 1 / num_distributions
        minimizer_result = scipy.optimize.minimize(
            fun=self.ensemble_func,
            x0=initial_guess,
            args=(ecdf, cdfs),
        )

        res = EnsembleResult(
            weights=minimizer_result.x,
            ensemble_density=self.ensemble_density(cdfs, minimizer_result.x),
        )

        return res
