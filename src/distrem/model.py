import json
import warnings

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
import scipy.stats as stats

from distrem.distributions import Distribution, distribution_dict


class EnsembleDistribution:
    """Ensemble distribution object that provides limited functionality similar
    to scipy's rv_continuous class both in implementation and features. Current
    features include: pdf, cdf, ppf, rvs (random draws), and stats (first 2
    moments) functions

    Parameters
    ----------
    named_weights : dict[str, float]
        dictionary of distributions as keys w/corresponding weights as values
    mean : float
        desired mean of ensemble distribution
    variance : float
        desired variance of ensemble distribution
    lb : float, optional
        desired lower bound of ensemble distribution, by default None
    ub : float, optional
        desired upper bound of ensemble distribution, by default None

    """

    def __init__(
        self,
        named_weights: dict[str, float],
        mean: float,
        variance: float,
        lb: float = None,
        ub: float = None,
    ):
        self._distributions = list(named_weights.keys())
        self._weights = list(named_weights.values())
        self.lb = lb
        self.ub = ub

        _check_valid_ensemble(self._distributions, self._weights)
        self.support = _check_supports_match(self._distributions)

        self.named_weights = named_weights
        self.fitted_distributions = []
        for distribution in self.named_weights.keys():
            self.fitted_distributions.append(
                distribution_dict[distribution](
                    mean, variance, self.lb, self.ub
                )
            )
        self.mean = mean
        self.variance = variance
        if lb is not None and self.cdf(lb) > 0.05:
            warnings.warn(
                "Ensemble density less than the specified lower bound "
                + lb
                + " exceeds 0.05. Check for low sample size!"
            )
        if ub is not None and (1 - self.cdf(ub)) > 0.05:
            warnings.warn(
                "Ensemble density greater than the specified upper bound "
                + ub
                + " exceeds 0.05. Check for low sample size!"
            )

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

    def pdf(self, x: npt.ArrayLike) -> npt.NDArray:
        """probability density function of ensemble distribution

        Parameters
        ----------
        x : npt.ArrayLike
            quantiles

        Returns
        -------
        npt.NDArray
            ensemble PDF evaluated at quantile x

        """
        return sum(
            weight * distribution.pdf(x)
            for distribution, weight in zip(
                self.fitted_distributions, self._weights
            )
        )

    def cdf(self, q: npt.ArrayLike) -> npt.NDArray:
        """cumulative density function of ensemble distribution

        Parameters
        ----------
        q : npt.ArrayLike
            quantiles

        Returns
        -------
        npt.NDArray
            ensemble CDF evaluated at quantile x

        """
        return sum(
            weight * distribution.cdf(q)
            for distribution, weight in zip(
                self.fitted_distributions, self._weights
            )
        )

    def ppf(self, p: npt.ArrayLike, uncertainty: bool = True) -> npt.NDArray:
        """percent point function of ensemble distribution

        Parameters
        ----------
        p : npt.ArrayLike
            lower tail probability
        uncertainty : bool, optional
            return a 95% CI using the delta method about p

        Returns
        -------
        npt.NDArray
            quantile corresponding to lower tail probability p

        """
        ppf_vec = np.vectorize(self._ppf_single, otypes="d")
        return ppf_vec(p)

    def rvs(self, size: int = 1) -> npt.NDArray:
        """random variates from ensemble distribution

        Parameters
        ----------
        size : int, optional
            number of draws to generate, by default 1

        Returns
        -------
        npt.NDArray
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
        dist_counts = np.random.multinomial(size, self._weights)
        samples = np.hstack(
            [
                distribution_dict[dist](self.mean, self.variance).rvs(
                    size=counts
                )
                for dist, counts in zip(self._distributions, dist_counts)
            ]
        )
        np.random.shuffle(samples)
        return samples

    def ensemble_stats(
        self, moments: str = "mv"
    ) -> float | tuple[float, float]:
        """retrieves mean and/or variance of ensemble distribution based on
        characters passed into moments parameter

        Parameters
        ----------
        moments : str, optional
            m for mean, v for variance, by default "mv"

        Returns
        -------
        float | tuple[float, float]
            mean, variance, or both

        """
        res_list = []
        if "m" in moments:
            res_list.append(self.mean)
        if "v" in moments:
            res_list.append(self.variance)

        if len(res_list) == 1:
            return res_list[0]
        else:
            return tuple(res_list)

    def plot(self):
        """THIS IS A DEMONSTRATION FUNCTION. SEE DOCUMENTATION FOR MORE PRACTICAL PLOTS

        plots the PDF and CDF of an ensemble distribution

        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        scaling = 3 * np.sqrt(self.variance)
        lb = np.max([self.support[0], self.mean - scaling])
        ub = np.min([self.support[1], self.mean + scaling])
        support = np.linspace(lb, ub, 100)
        pdf = self.pdf(support)
        cdf = self.cdf(support)
        ax[0].plot(support, pdf)
        ax[0].set_xlabel("DATA VALUES (UNITS)")
        ax[0].set_ylabel("density")
        ax[0].set_title("ensemble PDF")

        ax[1].plot(support, cdf)
        ax[1].set_xlabel("DATA VALUES (UNITS)")
        ax[1].set_ylabel("density")
        ax[1].set_title("ensemble CDF")

        return fig

    def to_json(self, file_path: str, appending: bool = False) -> None:
        """serializes EnsembleDistribution object as a JSON file with the option
        to append instead of writing a new file

        Parameters
        ----------
        file_path : str
            path to file to write in
        appending : bool, optional
            option to append to existing file instead of overwrite,
            by default False
        """
        distribution_summary = {
            "named_weights": self.named_weights,
            "mean": self.mean,
            "variance": self.variance,
        }

        if appending:
            with open(file_path, "r") as outfile:
                existing = json.load(outfile)
            with open(file_path, "w") as outfile:
                existing.append(distribution_summary)
                json.dump(existing, outfile)
        else:
            with open(file_path, "w") as outfile:
                json.dump([distribution_summary], outfile)

    @classmethod
    def from_objs(
        cls, fitted_distributions: list[Distribution]
    ) -> "EnsembleDistribution":
        """generates ensemble distribution from Distribution objects also from
        the ensemble package. all parameters, such as mean, variance, upper
        bound, and lower bound, must match

        Parameters
        ----------
        fitted_distributions : list[Distribution]
            list of distribution objects from the ensemble package

        Returns
        -------
        EnsembleDistribution
            ensemble distribution object with parameters equal to individual
            input distributions

        Raises
        ------
        ValueError
            if parameters across individual distributions don't match
        ValueError
            if the weight of a distribution is not set

        """
        try:
            mean, variance, lb, ub = (
                fitted_distributions[0].mean,
                fitted_distributions[0].variance,
                fitted_distributions[0].lb,
                fitted_distributions[0].ub,
            )
            named_weights = {}
            for distribution in fitted_distributions:
                if (
                    distribution.mean != mean
                    or distribution.variance != variance
                    or distribution.lb != lb
                    or distribution.ub != ub
                ):
                    raise ValueError(
                        "means, variances, lower bounds, and upper bounds must match across all distributions\ncurrently, they are:"
                    )
                else:
                    curr_weight = distribution._weight
                    if curr_weight is None:
                        raise ValueError(
                            "_weight must be set in order to create an ensemble distribution from Distribution objects"
                        )
                    named_weights[type(distribution).__name__] = curr_weight
            return cls(named_weights, mean, variance, lb, ub)
        except ValueError as err:
            if "means, variances" in str(err):
                for distribution in fitted_distributions:
                    print(
                        type(distribution).__name__ + ":",
                        "mean =",
                        distribution.mean,
                        "variance =",
                        distribution.variance,
                        "lb =",
                        distribution.lb,
                        "ub =",
                        distribution.ub,
                    )
            raise

    @classmethod
    def from_json(cls, file_path: str) -> list["EnsembleDistribution"]:
        """deserializes JSON object into list of Ensemble Distribution objects

        Parameters
        ----------
        file_path : str
            path to file that JSON object is stored in

        Returns
        -------
        list["EnsembleDistribution"]
            list of EnsembleDistribution objects

        """
        with open(file_path, "r") as infile:
            distribution_summaries = json.load(infile)

        res = [None] * len(distribution_summaries)
        for i in range(len(distribution_summaries)):
            named_weights, mean, variance = (
                distribution_summaries[i]["named_weights"],
                distribution_summaries[i]["mean"],
                distribution_summaries[i]["variance"],
            )
            res[i] = cls(named_weights, mean, variance)

        return res


class EnsembleResult:
    """Result from ensemble distribution fitting

    Parameters
    ----------

    weights: list[str]
        Weights of each distribution in the ensemble
    ensemble_model: EnsembleModel
        EnsembleModel object allowing user to get density, draws, etc...

    """

    weights: tuple[str, float]
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
    distributions: list[str]
        names of distributions in ensemble
    objective: str
        name of objective function for use in fitting ensemble

    """

    def __init__(
        self,
        distributions: list[str],
        objective: str,
    ):
        self.support = _check_supports_match(distributions)
        self.distributions = distributions
        self.objective = objective

    def _objective_func(
        self,
        eprobabilities: npt.NDArray,
        cdfs: npt.NDArray,
        close_idx: npt.NDArray,
        w: npt.NDArray,
        tsh_wts: npt.NDArray,
    ) -> float:
        """applies different penalties to vector of distances given by user

        Parameters
        ----------
        d : npt.NDArray
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
            when input corresponds to unimplemented objective function

        """
        unwtd_d = eprobabilities[close_idx] - cdfs[close_idx] @ w
        d = cp.multiply(unwtd_d, tsh_wts)

        match self.objective:
            case "L1":
                return cp.norm(d, 1)
            case "sum_squares":
                return cp.sum_squares(d)
            case "KS":
                return cp.norm(d, "inf")
            case _:
                raise NotImplementedError(
                    "Your choice of objective function hasn't yet been implemented!"
                )

    def fit(
        self,
        data: npt.ArrayLike,
        tsh_pts: list[float] | None = None,
        tsh_wts: list[float] | None = None,
        lb: float | None = None,
        ub: float | None = None,
    ) -> EnsembleResult:
        """fits weighted sum of CDFs corresponding to distributions in
        EnsembleModel object to empirical CDF of given data

        Parameters
        ----------
        data : npt.ArrayLike
            individual-level data (i.e. microdata)
        tsh_pts : list[float] | None, optional
            threshold values at which fit should be close, by default None
        tsh_wts : list[float] | None, optional
            weights assigned to threshold values at which fit should be prioritized, by default None
        lb : float | None, optional
            lower allowable bound of data, by default None
        ub : float | None, optional
            upper allowable bound of data, by default None

        Returns
        -------
        EnsembleResult
            result of ensemble distribution fitting

        Raises
        ------
        ValueError
            if range of data exceeds bounds of the support for ensemble distribution
        ValueError
            if there are fewer than 2 observations provided

        """
        _check_data_bounds(data, self.support)
        _check_data_len(data)
        _warn_duplicates(data)

        # sample stats, ecdf
        sample_mean = np.mean(data)
        sample_variance = np.var(data, ddof=1)
        ecdf = stats.ecdf(data).cdf

        # reintroduce duplicates into scipy's ecdf for fitting only
        sorted_indices = np.argsort(data)
        equantiles = data[sorted_indices]
        eprobabilities = np.interp(
            equantiles, ecdf.quantiles, ecdf.probabilities
        )

        # given valid inputs, finds points on eCDF closest to threshold points
        close_idx = slice(None)
        if tsh_pts is not None and tsh_wts is not None:
            _check_tsh_wts(tsh_wts)
            _check_tsh_pts(tsh_pts, self.support)
            close_idx = [
                np.searchsorted(equantiles, tsh_pts[i], side="left")
                for i in range(len(tsh_pts))
            ]
        elif tsh_pts is None and tsh_wts is None:
            tsh_wts = np.ones((len(eprobabilities),))
        else:
            raise ValueError(
                "if you would like to use the tsh_pts and tsh_wts arguments, you must provide both"
            )

        # fill matrix with cdf values over support of data
        num_distributions = len(self.distributions)
        fitted_distributions = []
        cdfs = np.zeros((len(data), num_distributions))
        pdfs = np.zeros((len(data), num_distributions))
        for i in range(num_distributions):
            curr_dist = distribution_dict[self.distributions[i]](
                sample_mean, sample_variance, lb=lb, ub=ub
            )
            fitted_distributions.append(curr_dist)
            cdfs[:, i] = curr_dist.cdf(equantiles)
            pdfs[:, i] = curr_dist.pdf(equantiles)

        # CVXPY implementation
        w = cp.Variable(num_distributions)
        objective = cp.Minimize(
            self._objective_func(
                eprobabilities=eprobabilities,
                cdfs=cdfs,
                close_idx=close_idx,
                tsh_wts=np.array(tsh_wts),
                w=w,
            )
        )
        constraints = [0 <= w, cp.sum(w) == 1]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except cp.error.SolverError as e:
            raise cp.error.SolverError(
                f"{e}\nAdditional context for distrem: you have most likely supplied an array of all duplicate values causing the solver to fail"
            )

        # assign weights to each distribution object
        fitted_weights = w.value
        for i in range(len(fitted_weights)):
            fitted_distributions[i]._weight = fitted_weights[i]

        res = EnsembleResult(
            weights=fitted_weights,
            ensemble_distribution=EnsembleDistribution.from_objs(
                fitted_distributions
            ),
        )

        return res


class SDOptimizer:
    def __init__(self, mean: float, named_weights: dict[str, float]):
        self.mean = mean
        self.named_weights = named_weights

    def _objective(
        self,
        sd: float,
        weights: npt.ArrayLike,
        upper: npt.ArrayLike,
        lower: npt.ArrayLike,
        p_hat: npt.ArrayLike,
    ):
        ens = EnsembleDistribution(
            named_weights=self.named_weights,
            mean=self.mean,
            variance=sd**2,
        )
        return weights @ ((ens.cdf(upper) - ens.cdf(lower)) - p_hat) ** 2

    def optimize_sd(
        self,
        data,
        weights="weights",
        lb="lb",
        ub="ub",
        prev="prev",
    ):
        weights = np.array(data[weights])
        lb = np.array(data[lb])
        ub = np.array(data[ub])
        prev = np.array(data[prev])

        _check_prevalences(prev)
        weights, lb, ub, prev = _check_bounds(weights, lb, ub, prev)

        if np.any(np.isinf(lb)):
            inf_idx_lb = np.where(np.isinf(lb))
            z_score = stats.norm.ppf(prev[inf_idx_lb])
            # print(z_score, self.mean, ub[inf_idx_lb])
            sigma_init = np.abs((self.mean - ub[inf_idx_lb]) / z_score)
        elif np.any(np.isinf(ub)):
            inf_idx_ub = np.where(np.isinf(ub))
            z_score = stats.norm.ppf(prev[inf_idx_ub])
            sigma_init = np.abs((self.mean - lb[inf_idx_ub]) / z_score)

        res = opt.minimize_scalar(
            fun=lambda sd: self._objective(sd, weights, ub, lb, prev),
            bounds=(0, sigma_init * 1.5),
            method="bounded",
            options={"disp": True},
        )

        return res.x


####################
### HELPER FUNCTIONS
####################
def _warn_duplicates(data: npt.ArrayLike):
    if len(np.unique(data)) == 1:
        warnings.warn(
            "Your data contains all duplicate values. You may receive a message regarding solver failure."
        )


def _check_bounds(
    weights: npt.ArrayLike,
    lb: npt.ArrayLike,
    ub: npt.ArrayLike,
    p_hat: npt.ArrayLike,
):
    bounds = dict()
    if not np.isclose(np.sum(weights), 1):
        raise ValueError("weights must all sum to 1")
    for i in range(len(lb)):
        if lb[i] >= ub[i]:
            raise ValueError(
                f"provided lower bound {lb[i]} was greater than/equal to upper bound {ub[i]}. lower bound must be strictly less than upper bound"
            )
        bound_pair = (lb[i], ub[i])
        if bound_pair in bounds:
            bounds[bound_pair][0].append(weights[i])
            bounds[bound_pair][1].append(p_hat[i])
        else:
            bounds[bound_pair] = [[weights[i]], [p_hat[i]]]

    weights, lb, ub, p_hat = [], [], [], []
    for key, value in bounds.items():
        weight = np.array(value[0])
        prev = np.array(value[1])
        interval_wt_sum = np.sum(weight)

        weights.append(np.sum(weight))
        lb.append(key[0])
        ub.append(key[1])
        p_hat.append(np.sum(weight @ prev) / interval_wt_sum)

    return (
        np.array(weights),
        np.array(lb),
        np.array(ub),
        np.array(p_hat),
    )


def _check_prevalences(p_hat: npt.ArrayLike):
    if np.any(p_hat) < 0 or np.any(p_hat) > 1:
        raise ValueError("all prevalence values must be between [0, 1]")


def _check_data_bounds(data, support):
    if np.min(data) < support[0] or support[1] < np.max(data):
        raise ValueError("data exceeds bounds of the support of your ensemble")


def _check_data_len(data):
    if len(data) <= 1:
        raise ValueError(
            "you may only run this function with 2 or more data points"
        )


def _check_tsh_pts(tsh_pts, support):
    if np.any(tsh_pts) < support[0] or support[1] < np.any(tsh_pts):
        raise ValueError(
            "threshold weights must be within the support of chosen distributions"
        )


def _check_tsh_wts(tsh_wts):
    wt_sum = np.sum(tsh_wts)
    if not np.isclose(wt_sum, 1):
        raise ValueError(
            f"threshold weights must sum to 1, current sum is {wt_sum}"
        )


def _check_valid_ensemble(
    distributions: list[str], weights: list[float]
) -> None:
    """checks if ensemble distribution is valid

    Parameters
    ----------
    distributions : list[str]
        list of named distributions, as strings
    weights : list[float]
        list of weights, in order of provided distribution list

    Raises
    ------
    ValueError
        if there is a mismatch between num distributions and num weights
    ValueError
        if weights do not sum to 1

    """
    if len(distributions) != len(weights):
        raise ValueError(
            "there must be the same number of distributions as weights!"
        )
    if not np.isclose(np.sum(weights), 1):
        raise ValueError("weights must sum to 1")


def _check_supports_match(distributions: list[str]) -> tuple[float, float]:
    """checks that supports of all distributions given are *exactly* the same

    Parameters
    ----------
    distributions : list[str]
        names of distributions

    Returns
    -------
    supports: tuple[float, float]
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
        supports.add(distribution_dict[distribution].support)
    if len(supports) != 1:
        raise ValueError(
            "the provided list of distributions do not all have the same support: "
            + str(supports)
            + "please check the documentation for the supports of the distributions you've specified"
        )
    # note: if the return statement is reached, the `set()` named `supports`
    # will only ever have one support within it, which is popped out and returned
    return supports.pop()
