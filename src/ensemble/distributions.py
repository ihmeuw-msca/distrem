from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
import scipy.stats as stats
from scipy.special import gamma as gamma_func

# from scipy.special import gammainccinv, gammaincinv


class Distribution(ABC, metaclass=ABCMeta):
    """Abstract class for objects that fit scipy distributions given a certain
    mean/variance, and return a limited amount of the original functionality
    of the original scipy rv_continuous object.

    """

    def __init__(
        self, mean: float, variance: float, lb: float = None, ub: float = None
    ):
        self.mean = mean
        self.variance = variance
        self.lb = lb
        self.ub = ub
        # # some kind of dictionary with
        # #   key: the support (full real line, semi infinite, etc...)
        # #   value: function that gets called when distribution is initialized
        # self.support = None
        # self._support_setup()
        self.shifted_mean = None
        self._scipy_dist = None
        # ONLY for use when creating ensemble from pre-fitted distributions
        self._weight = None
        match (
            self.lb is not None and not np.isinf(self.lb),
            self.ub is not None and not np.isinf(self.ub),
            # np.isinf(self.lb) or np.isinf(self.ub),
            # not np.isinf(self.support[0]),
            # not np.isinf(self.support[1]),
        ):
            case (True, True):
                # print(np.isinf(self.support[0]))
                # print(np.isinf(self.support[1]))
                if np.isinf(self.support[0]) or np.isinf(self.support[1]):
                    raise ValueError(
                        "You may not change an infinite bound to be finite or"
                        + "set a bound to be infinite"
                    )
                if self.lb > self.mean or self.ub < mean:
                    raise ValueError(
                        "mean must be between upper and lower bounds"
                    )
                self.support = (self.lb, self.ub)
            case (True, False):
                if np.isinf(self.support[0]):
                    raise ValueError(
                        "You may not change an infinite bound to be finite or"
                        + "set a bound to be infinite"
                    )
                if self.lb > self.mean:
                    raise ValueError(
                        "mean must be between upper and lower bounds"
                    )
                self.support = (lb, self.support[1])
                # self.mean = self.mean - lb
                self.shifted_mean = self.mean - lb
            case (False, True):
                if np.isinf(self.support[1]):
                    raise ValueError(
                        "You may not change an infinite bound to be finite or"
                        + "set a bound to be infinite"
                    )
                if self.ub < mean:
                    raise ValueError(
                        "mean must be between upper and lower bounds"
                    )
                self.support = (self.support[0], ub)
            case _:
                if self.lb is not None and np.isinf(self.lb):
                    raise ValueError(
                        "You may not change an infinite bound to be finite or"
                        + "set a bound to be infinite"
                    )
                if self.ub is not None and np.isinf(self.ub):
                    raise ValueError(
                        "You may not change an infinite bound to be finite or"
                        + "set a bound to be infinite"
                    )
                pass

        csd_mean = self.mean if self.shifted_mean is None else self.shifted_mean
        self._create_scipy_dist(csd_mean)

    @abstractmethod
    def _create_scipy_dist(self, csd_mean: int) -> None:
        """Create scipy distribution from mean and variance"""

    @property
    @abstractmethod
    def support(self) -> Tuple[float, float]:
        """create tuple representing endpoints of support"""
        pass

    def _shift(self, x: float) -> float:
        if self.lb is not None:
            return x - self.lb
        return x

    # def validate_finite_bounds(self, b1, b2):
    #     if np.isinf(b1) or np.isinf(b2):
    #         raise ValueError(
    #             "you may not change an infinite bound to be finite or set a bound to be infinite"
    #         )

    def rvs(self, *args, **kwds):
        """defaults to scipy implementation for generating random variates

        Returns
        -------
        np.ndarray
            random variates from a given distribution/parameters
        """
        return self._shift(self._scipy_dist.rvs(*args, **kwds))

    def pdf(self, x: npt.ArrayLike) -> np.ndarray:
        """defaults to scipy implementation for probability density function

        Parameters
        ----------
        x : npt.ArrayLike
            quantiles

        Returns
        -------
        np.ndarray
            PDF evaluated at quantile x
        """
        return self._scipy_dist.pdf(self._shift(x))

    def cdf(self, q: npt.ArrayLike) -> np.ndarray:
        """defaults to scipy implementation for cumulative density function

        Parameters
        ----------
        q : npt.ArrayLike
            quantiles

        Returns
        -------
        np.ndarray
            CDF evaluated at quantile q
        """
        return self._scipy_dist.cdf(self._shift(q))

    def ppf(self, p: npt.ArrayLike) -> np.ndarray:
        """defaults to scipy implementation for percent point function

        Parameters
        ----------
        p : npt.ArrayLike
            lower tail probability

        Returns
        -------
        np.ndarray
            PPF evaluated at lower tail probability p
        """
        return self._shift(self._scipy_dist.ppf(p))

    def stats(self, moments: str) -> Union[float, Tuple[float, ...]]:
        """defaults to scipy implementation for obtaining moments

        Parameters
        ----------
        moments : str
            m for mean, v for variance, s for skewness, k for kurtosis

        Returns
        -------
        Union[float, Tuple[float, ...]]
            mean, variance, skewness, and/or kurtosis
        """
        # return self._scipy_dist.stats(moments=moments)
        res_list = []
        if "m" in moments:
            res_list.append(self._shift(self._scipy_dist.stats("m")))
        if "v" in moments:
            res_list.append(self._scipy_dist.stats("v"))

        # res_list = [res[()] for res in res_list]
        if len(res_list) == 1:
            return res_list[0]
        else:
            return tuple(res_list)


# analytic sol
class Exponential(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html"""

    support = (0, np.inf)

    def _create_scipy_dist(self, csd_mean) -> None:
        positive_support(self.mean)
        lambda_ = 1 / csd_mean
        self._scipy_dist = stats.expon(scale=1 / lambda_)


# analytic sol
class Gamma(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html#scipy.stats.gamma"""

    support = (0, np.inf)

    def _create_scipy_dist(self, csd_mean) -> None:
        strict_positive_support(self.mean)
        alpha = csd_mean**2 / self.variance
        beta = csd_mean / self.variance
        self._scipy_dist = stats.gamma(a=alpha, scale=1 / beta)


# analytic sol
class InvGamma(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html#scipy.stats.invgamma"""

    support = (0, np.inf)

    def _create_scipy_dist(self, csd_mean) -> None:
        strict_positive_support(self.mean)
        alpha = csd_mean**2 / self.variance + 2
        beta = csd_mean * (csd_mean**2 / self.variance + 1)
        self._scipy_dist = stats.invgamma(a=alpha, scale=beta)


# numerical sol
class Fisk(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisk.html#scipy.stats.fisk"""

    support = (0, np.inf)

    def _create_scipy_dist(self, csd_mean):
        positive_support(self.mean)

        optim_params = opt.minimize(
            fun=self._shape_scale,
            # start beta at 1.1 and solve for alpha
            x0=[csd_mean * 1.1 * np.sin(np.pi / 1.1) / np.pi, 1.1],
            args=(csd_mean, self.variance),
            # options={"disp": True},
        )
        alpha, beta = np.abs(optim_params.x)
        # parameterization notes: numpy's c is wikipedia's beta, numpy's scale is wikipedia's alpha
        # additional note: analytical solution doesn't work b/c dependent on derivative
        # print("from optim: ", alpha, beta)
        self._scipy_dist = stats.fisk(c=beta, scale=alpha)

    def _shape_scale(self, x: list, samp_mean: float, samp_var: float) -> None:
        alpha = x[0]
        beta = x[1]
        b = np.pi / beta
        mean_guess = alpha * b / np.sin(b)
        variance_guess = alpha**2 * (
            (2 * b / np.sin(2 * b)) - b**2 / np.sin(b) ** 2
        )
        return (mean_guess - samp_mean) ** 2 + (variance_guess - samp_var) ** 2


# analytic sol
class GumbelR(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html#scipy.stats.gumbel_r"""

    support = (-np.inf, np.inf)

    def _create_scipy_dist(self, csd_mean) -> None:
        loc = self.mean - np.sqrt(self.variance * 6) * np.euler_gamma / np.pi
        scale = np.sqrt(self.variance * 6) / np.pi
        self._scipy_dist = stats.gumbel_r(loc=loc, scale=scale)


# hopelessly broken (sort of)
class Weibull(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min"""

    # def support(self) -> Tuple[float, float]:
    #     return (0, np.inf)
    support = (0, np.inf)

    def _create_scipy_dist(self, csd_mean) -> None:
        positive_support(self.mean)

        # https://real-statistics.com/distribution-fitting/method-of-moments/method-of-moments-weibull/
        k = opt.root_scalar(self._func, x0=0.5, method="newton")
        lambda_ = csd_mean / gamma_func(1 + 1 / k.root)
        print("hi!", lambda_, k.root)

        # most likely a parameterization issue
        self._scipy_dist = stats.weibull_min(c=k.root, scale=lambda_)

    def _func(self, k: float) -> None:
        return (
            np.log(1 + (2 / k))
            - 2 * np.log(gamma_func(1 + (1 / k)))
            - np.log(self.variance + self.mean**2)
            + 2 * np.log(self.mean)
        )


# analytic sol (M.O.M. estimators)
class LogNormal(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm"""

    support = (0, np.inf)

    def _create_scipy_dist(self, csd_mean) -> None:
        strict_positive_support(self.mean)
        mu = np.log(csd_mean / np.sqrt(1 + (self.variance / csd_mean**2)))
        sigma = np.sqrt(np.log(1 + (self.variance / csd_mean**2)))
        # scipy multiplies in the argument passed to `scale` so in the exponentiated space,
        # you're essentially adding `mu` within the exponentiated expression within the
        # lognormal's PDF; hence, scale is with exponentiation instead of loc
        self._scipy_dist = stats.lognorm(scale=np.exp(mu), s=sigma)


# analytic sol
class Normal(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm"""

    support = (-np.inf, np.inf)

    def _create_scipy_dist(self, csd_mean) -> None:
        self._scipy_dist = stats.norm(
            loc=self.mean, scale=np.sqrt(self.variance)
        )


# analytic sol
class Beta(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html#scipy.stats.beta"""

    support = (0, 1)

    def __init__(
        self,
        mean: float,
        variance: float,
        lb: float = 0,
        ub: float = 1,
    ):
        self.width = np.abs(ub - lb)
        super().__init__(mean, variance, lb, ub)

    def _squeeze(self, x: float) -> float:
        """transform x to be within (0, 1)

        Parameters
        ----------
        x : float
            value within support

        Returns
        -------
        float
            transformed value within support
        """
        return (x - self.lb) / self.width

    def _stretch(self, x: float) -> float:
        """transform x from (0, 1) back to original bounds

        Parameters
        ----------
        x : float
            value within standard Beta support

        Returns
        -------
        float
            transformed value within original support
        """
        return (x + self.lb) * self.width

    # def support(self) -> Tuple[float, float]:
    #     return (self.lb, self.ub)

    def _create_scipy_dist(self, csd_mean) -> None:
        # TODO: what happens here if the mean and variance are shifted?
        if self.mean**2 <= self.variance:
            raise ValueError(
                "beta distributions do not exist for certain mean and variance "
                + "combinations. The supplied variance must be in between "
                + "(0, mean^2)"
            )
        if self.lb != 0 or self.ub != 1:
            mean = (self.mean - self.lb) / self.width
            var = self.variance / self.width
        else:
            mean = self.mean
            var = self.variance

        alpha = (mean**2 * (1 - mean) - mean * var) / var
        beta = (1 - mean) * (mean - mean**2 - var) / var
        print(alpha, beta)
        self._scipy_dist = stats.beta(a=alpha, b=beta)
        print(self._scipy_dist.stats("mv"))

    def rvs(self, *args, **kwds):
        """defaults to scipy implementation for generating random variates

        Returns
        -------
        np.ndarray
            random variates from a given distribution/parameters
        """
        return self._stretch(self._scipy_dist.rvs(*args, **kwds))

    def pdf(self, x: npt.ArrayLike) -> np.ndarray:
        """defaults to scipy implementation for probability density function

        Parameters
        ----------
        x : npt.ArrayLike
            quantiles

        Returns
        -------
        np.ndarray
            PDF evaluated at quantile x
        """
        return self._scipy_dist.pdf(self._squeeze(x))

    def cdf(self, q: npt.ArrayLike) -> np.ndarray:
        """defaults to scipy implementation for cumulative density function

        Parameters
        ----------
        q : npt.ArrayLike
            quantiles

        Returns
        -------
        np.ndarray
            CDF evaluated at quantile q
        """
        return self._scipy_dist.cdf(self._squeeze(q))

    def ppf(self, p: npt.ArrayLike) -> np.ndarray:
        """defaults to scipy implementation for percent point function

        Parameters
        ----------
        p : npt.ArrayLike
            lower tail probability

        Returns
        -------
        np.ndarray
            PPF evaluated at lower tail probability p
        """
        return self._stretch(self._scipy_dist.ppf(p))

    def stats(self, moments: str) -> Union[float, Tuple[float, ...]]:
        """defaults to scipy implementation for obtaining moments

        Parameters
        ----------
        moments : str
            m for mean, v for variance, s for skewness, k for kurtosis

        Returns
        -------
        Union[float, Tuple[float, ...]]
            mean, variance, skewness, and/or kurtosis
        """
        res_list = []
        if "m" in moments:
            res_list.append(self._stretch(self._scipy_dist.stats("m")))
        if "v" in moments:
            res_list.append(self._scipy_dist.stats("v") * self.width)

        # res_list = [res[()] for res in res_list]
        if len(res_list) == 1:
            return res_list[0]
        else:
            return tuple(res_list)


distribution_dict = {
    "Exponential": Exponential,
    "Gamma": Gamma,
    "InvGamma": InvGamma,
    "Fisk": Fisk,
    "GumbelR": GumbelR,
    "Weibull": Weibull,
    "LogNormal": LogNormal,
    "Normal": Normal,
    "Beta": Beta,
}


### HELPER FUNCTIONS
def positive_support(mean: float) -> None:
    if mean < 0:
        raise ValueError("This distribution is only supported on [0, np.inf)")


def strict_positive_support(mean: float) -> None:
    if mean <= 0:
        raise ValueError("This distribution is only supported on (0, np.inf)")


# def beta_bounds(mean: float) -> None:
#     if (mean < 0) or (mean > 1):
#         raise ValueError("This distribution is only supposrted on [0, 1]")
