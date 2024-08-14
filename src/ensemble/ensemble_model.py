from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import scipy.linalg as linalg
import scipy.optimize as opt
import scipy.stats as stats

from ensemble.distributions import distribution_dict


class EnsembleModel:
    def __init__(
        self, distributions: List[str], weights: List[float], mean, variance
    ):
        self.supports = _check_supports_match(distributions)
        self.distributions = distributions
        self.my_objs = []
        for distribution in distributions:
            self.my_objs.append(distribution_dict[distribution](mean, variance))
        self.weights = weights
        self.mean = mean
        self.variance = variance

    def _ppf_to_solve(self, x, q):
        return (
            self.cdf(
                x, self.distributions, self.weights, self.mean, self.variance
            )
            - q
        )

    def _ppf_single(self, q):
        factor = 10.0
        some_distribution = self.supports.pop()
        left, right = some_distribution.support()

        if np.isinf(left):
            left = min(-factor, right)
            while self._ppf_to_solve(left, q) > 0:
                left, right = left * factor, left

        if np.isinf(right):
            right = max(factor, left)
            while self._ppf_to_solve(right, q) < 0:
                left, right = right, right * factor

        return opt.brentq(self._ppf_to_solve, left, right, args=q)

    def pdf(self, x):
        return sum(
            weight * distribution.pdf(x)
            for distribution, weight in zip(self.my_objs, self.weights)
        )

    def cdf(self, q):
        return sum(
            weight * distribution.cdf(q)
            for distribution, weight in zip(self.my_objs, self.weights)
        )

    def ppf(self, p):
        ppf_vec = np.vectorize(self._ppf_single, otypes="d")
        return ppf_vec(p)

    def rvs(self, size=1):
        unif_samp = stats.uniform.rvs(size=size)
        return self.ppf(unif_samp)

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
        self.supports = _check_supports_match(distributions)
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
            ensemble_model=EnsembleModel(
                self.distributions, fitted_weights, sample_mean, sample_variance
            ),
        )

        return res


### HELPER FUNCTIONS


def _check_supports_match(distributions):
    supports = set()
    for distribution in distributions:
        supports.add(distribution_dict[distribution].support)
    # TODO: HOW SHOULD WE TELL THE USER WHICH DISTRIBUTION IS THE "TROUBLEMAKER"?
    if len(supports) != 1:
        raise ValueError(
            "the provided list of distributions do not all have the same support: "
            + str(supports)
        )
    return supports
