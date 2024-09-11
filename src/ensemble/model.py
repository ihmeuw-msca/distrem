from typing import List, Tuple, Union

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
import scipy.stats as stats

from ensemble.distributions import distribution_dict


class EnsembleDistribution:
    """Ensemble distribution object that provides limited functionality similar
    to scipy's rv_continuous class both in implementation and features. Current
    features include: pdf, cdf, ppf, rvs (random draws), and stats (first 2
    moments) functions

    Parameters
    ----------

    distributions: List[str]
        names of distributions in ensemble
    weights: List[float]
        weight assigned to each distribution in ensemble
    mean: float
        desired mean of ensemble distribution
    varaince: float
        desired variance of ensemble distribution

    """

    def __init__(
        self,
        distributions: List[str],
        weights: List[float],
        mean: float,
        variance: float,
    ):
        _check_valid_ensemble(distributions, weights)
        self.support = _check_supports_match(distributions)

        self.distributions = distributions
        self.my_objs = []
        for distribution in distributions:
            self.my_objs.append(distribution_dict[distribution](mean, variance))
        self.weights = weights
        self.mean = mean
        self.variance = variance

    def _ppf_to_solve(self, x: float, p: float) -> float:
        """ensemble_CDF(x) - lower tail probability

        Parameters
        ----------
        x : float
            quantile
        p : float
            lower tail probability

        Returns
        -------
        float
            distance between ensemble CDF and lower tail probability

        """
        return self.cdf(x) - p

    def _ppf_single(self, p: float) -> float:
        """Finds value to minimize distance between ensemble CDF and lower tail
        probability

        Parameters
        ----------
        p : float
            lower tail probability

        Returns
        -------
        float
            value that minimizes distance between ensemble CDF and lower tail
            probability

        """
        factor = 10.0
        left, right = self.support

        if np.isinf(left):
            left = min(-factor, right)
            while self._ppf_to_solve(left, p) > 0:
                left, right = left * factor, left

        if np.isinf(right):
            right = max(factor, left)
            while self._ppf_to_solve(right, p) < 0:
                left, right = right, right * factor

        return opt.brentq(self._ppf_to_solve, left, right, args=p)

    def pdf(self, x: npt.ArrayLike) -> np.ndarray:
        """probability density function of ensemble distribution

        Parameters
        ----------
        x : npt.ArrayLike
            quantiles

        Returns
        -------
        np.ndarray
            ensemble PDF evaluated at quantile x

        """
        return sum(
            weight * distribution.pdf(x)
            for distribution, weight in zip(self.my_objs, self.weights)
        )

    def cdf(self, q: npt.ArrayLike) -> np.ndarray:
        """cumulative density function of ensemble distribution

        Parameters
        ----------
        q : npt.ArrayLike
            quantiles

        Returns
        -------
        np.ndarray
            ensemble CDF evaluated at quantile x

        """
        return sum(
            weight * distribution.cdf(q)
            for distribution, weight in zip(self.my_objs, self.weights)
        )

    def ppf(self, p: npt.ArrayLike) -> np.ndarray:
        """percent point function of ensemble distribution

        Parameters
        ----------
        p : npt.ArrayLike
            lower tail probability

        Returns
        -------
        np.ndarray
            quantile corresponding to lower tail probability p

        """
        ppf_vec = np.vectorize(self._ppf_single, otypes="d")
        return ppf_vec(p)

    def rvs(self, size: int = 1) -> np.ndarray:
        """random variates from ensemble distribution

        Parameters
        ----------
        size : int, optional
            number of draws to generate, by default 1

        Returns
        -------
        np.ndarray
            individual draws from ensemble distribution

        """

        # reference: https://github.com/scipy/scipy/blob/v1.14.0/scipy/stats/_distn_infrastructure.py#L994
        # relevant lines: 1026, 1048, 1938, 1941
        # summary:
        #   create ensemble cdf with at least 2 distributions/corresponding
        #     weights with shared mean and variance
        #   draw sample of size given by user from Unif(0, 1) representing lower
        #     tail probabilities
        #   give sample to vectorized ppf_single
        #   optimize (using Brent's method) with objective function
        #     ensemble_cdf(x) - p, where p is aforementioned Unif(0, 1) sample
        #   return quantiles which minimize the objective function (i.e. which
        #     values of x minimize ensemble_cdf(x) - q)
        dist_counts = np.random.multinomial(size, self.weights)
        samples = np.hstack(
            [
                distribution_dict[dist](self.mean, self.variance).rvs(
                    size=counts
                )
                for dist, counts in zip(self.distributions, dist_counts)
            ]
        )
        np.random.shuffle(samples)
        return samples

    def stats_temp(
        self, moments: str = "mv"
    ) -> Union[float, Tuple[float, float]]:
        """retrieves mean and/or variance of ensemble distribution based on
        characters passed into moments parameter

        Parameters
        ----------
        moments : str, optional
            m for mean, v for variance, by default "mv"

        Returns
        -------
        Union[float, Tuple[float, float]]
            mean, variance, or both

        """
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
    """Result from ensemble distribution fitting

    Parameters
    ----------

    weights: List[str]
        Weights of each distribution in the ensemble
    ensemble_model: EnsembleModel
        EnsembleModel object allowing user to get density, draws, etc...

    """

    weights: Tuple[str, float]
    ensemble_model: EnsembleDistribution

    def __init__(
        self, weights, ensemble_distribution: EnsembleDistribution
    ) -> None:
        self.weights = weights
        self.ensemble_distribution = ensemble_distribution


class EnsembleFitter:
    """Model to fit ensemble distributions composed of distributions of the
    user's choice with an objective function, also of the user's choice.
    Distributions that compose the ensemble are required to have the *exact*
    same supports

    Parameters
    ----------
    distributions: List[str]
        names of distributions in ensemble
    objective: str
        name of objective function for use in fitting ensemble

    """

    def __init__(self, distributions: List[str], objective: str):
        self.support = _check_supports_match(distributions)
        self.distributions = distributions
        self.objective = objective

    def _objective_func(self, vec: np.ndarray) -> float:
        """applies different penalties to vector of distances given by user

        Parameters
        ----------
        vec : np.ndarray
            distances, in this case, between empirical and ensemble CDFs
        objective : str
            name of objective function

        Returns
        -------
        float
            penalized distance metric between empirical and ensemble CDFs

        Raises
        ------
        NotImplementedError
            because the other ones havent been implemented yet lol
        """
        match self.objective:
            case "L1":
                return cp.norm(vec, 1)
            case "sum_squares":
                return cp.sum_squares(vec)
            case "KS":
                return cp.norm(vec, "inf")

    def _ensemble_func_temp(
        self,
        weights: List[float],
        pdfs,  # , ecdf: np.ndarray, cdfs: np.ndarray
    ) -> float:
        """

        Parameters
        ----------
        weights : List[float]
            _description_
        ecdf : np.ndarray
            _description_
        cdfs : np.ndarray
            _description_

        Returns
        -------
        float
            _description_
        """
        # return self._objective_func(ecdf - cdfs @ weights)
        return -1 * cp.sum(cp.log(pdfs @ weights))

    def fit(self, data: npt.ArrayLike) -> EnsembleResult:
        """fits weighted sum of CDFs corresponding to distributions in
        EnsembleModel object to empirical CDF of given data

        Parameters
        ----------
        data : npt.ArrayLike
            individual-level data (i.e. microdata)

        Returns
        -------
        EnsembleResult
            result of ensemble distribution fitting
        """
        if np.min(data) < self.support[0] or self.support[1] < np.max(data):
            raise ValueError(
                "data exceeds bounds of the support of your ensemble"
            )
        # sample stats, ecdf
        sample_mean = np.mean(data)
        sample_variance = np.var(data, ddof=1)
        ecdf = stats.ecdf(data).cdf.probabilities
        equantiles = stats.ecdf(data).cdf.quantiles
        print(len(equantiles), ", num of empirical quantiles")

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

        # CVXPY implementation
        w = cp.Variable(num_distributions)
        objective = cp.Minimize(self._objective_func(ecdf - cdfs @ w))
        constraints = [0 <= w, cp.sum(w) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        fitted_weights = w.value

        # ML implementation
        # w = cp.Variable(num_distributions)
        # objective = cp.Minimize(self._ensemble_func_temp(w, pdfs))
        # constraints = [0 <= w, cp.sum(w) == 1]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # fitted_weights = w.value

        res = EnsembleResult(
            weights=fitted_weights,
            ensemble_distribution=EnsembleDistribution(
                self.distributions, fitted_weights, sample_mean, sample_variance
            ),
        )

        return res


### HELPER FUNCTIONS


def _check_valid_ensemble(distributions: List[str], weights: List[float]):
    if len(distributions) != len(weights):
        raise ValueError(
            "there must be the same number of distributions as weights!"
        )
    if not np.isclose(np.sum(weights), 1):
        raise ValueError("weights must sum to 1")


def _check_supports_match(distributions: List[str]) -> Tuple[float, float]:
    """checks that supports of all distributions given are *exactly* the same

    Parameters
    ----------
    distributions : List[str]
        names of distributions

    Returns
    -------
    supports: Tuple[float, float]
        support of ensemble distributions given that all distributions in
        ensemble are compatible

    Raises
    ------
    ValueError
        upon giving distributions whose supports do not exactly match one
        another
    """
    supports = set()
    for distribution in distributions:
        supports.add(distribution_dict[distribution]().support())
    if len(supports) != 1:
        raise ValueError(
            "the provided list of distributions do not all have the same support: "
            + str(supports)
            + "please check the documentation for the supports of the distributions you've specified"
        )
    # note: if the return statement is reached, the `set()` named `supports`
    # will only ever have one support within it, which is popped out and returned
    return supports.pop()
