from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import scipy.linalg as linalg
import scipy.optimize as opt
import scipy.stats as stats

from ensemble.distributions import distribution_dict


class EnsembleModel:
    def __init__(self, distributions, weights, mean, variance):
        self.distributions = distributions
        self.weights = weights
        self.mean = mean
        self.variance = variance

    def pdf(self, x):
        raise NotImplementedError

    def cdf(self, q):
        raise NotImplementedError

    def ppf(self, p):
        raise NotImplementedError

    def rvs(self, size=1):
        raise NotImplementedError

    def stats_temp(self, moments="mv"):
        res_list = []
        if "m" in moments:
            res_list.append(self.mean)
        if "v" in moments:
            res_list.append(self.variance)

        res_list = [res[()] for res in res_list]
        if len(res_list) == 1:
            return res_list[0]
        else:
            return tuple(res_list)


class EnsembleResult:
    weights: Tuple[str, float]
    ensemble_model: EnsembleModel

    def __init__(self, weights, ensemble_model) -> None:
        self.weights = weights
        self.ensemble_model = ensemble_model


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
        self.objective = objective

    def objective_func(self, vec, objective):
        if objective is not None:
            raise NotImplementedError
        return linalg.norm(vec, 2)

    def ensemble_func(self, weights: list, ecdf: np.ndarray, cdfs: np.ndarray):
        return self.objective_func(ecdf - cdfs @ weights, self.objective)

    def fit(self, data: npt.ArrayLike) -> EnsembleResult:
        # TODO: SWITCH CASE STATEMENT FOR BOUNDS OF DATA NOT MATCHING THE ELEMENT OF SELF.SUPPORTS
        # sample stats, ecdf
        sample_mean = np.mean(data)
        sample_variance = np.var(data, ddof=1)
        ecdf = stats.ecdf(data).cdf.probabilities

        # may be able to condense into 1 line if ub and lb are not used elsewhere
        # support_lb = np.min(data)
        # support_ub = np.max(data)
        # support = np.linspace(support_lb, support_ub, len(data))
        equantiles = stats.ecdf(data).cdf.quantiles

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
        minimizer_result = opt.minimize(
            fun=self.ensemble_func,
            x0=initial_guess,
            args=(ecdf, cdfs),
            bounds=bounds,
        )
        fitted_weights = minimizer_result.x

        res = EnsembleResult(
            # weights=tuple(
            #     (distribution, weight)
            #     for distribution, weight in zip(
            #         self.distributions, fitted_weights
            #     )
            # ),
            # distributions=self.distributions,
            weights=fitted_weights,
            ensemble_model=EnsembleModel(None, None, None, None),
        )

        return res


### HELPER FUNCTIONS

### HELPER FUNCTIONS
