from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
import scipy.stats as stats
from scipy.special import gamma as gamma_func

# from scipy.special import gammainccinv, gammaincinv


class Distribution(ABC):
    """Abstract class for objects that fit scipy distributions given a certain
    mean/variance, and return a limited amount of the original functionality
    of the original scipy rv_continuous object.

    """

    def __init__(
        self,
        mean: float = None,
        variance: float = None,
        lb: float = None,
        ub: float = None,
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
        self._scipy_dist = None
        if self.mean is not None and self.variance is not None:
            self._create_scipy_dist()

    @abstractmethod
    def _create_scipy_dist(self) -> None:
        """Create scipy distribution from mean and variance"""

    def support(self) -> Tuple[float, float]:
        """create tuple representing endpoints of support"""

    def rvs(self, *args, **kwds):
        """defaults to scipy implementation for generating random variates

        Returns
        -------
        np.ndarray
            random variates from a given distribution/parameters
        """
        return self._scipy_dist.rvs(*args, **kwds)

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
        return self._scipy_dist.pdf(x)

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
        return self._scipy_dist.cdf(q)

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
        return self._scipy_dist.ppf(p)

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
        return self._scipy_dist.stats(moments=moments)


# analytic sol
class Exponential(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html"""

    def support(self) -> Tuple[float, float]:
        return (0, np.inf)

    def _create_scipy_dist(self) -> None:
        positive_support(self.mean)
        lambda_ = 1 / self.mean
        self._scipy_dist = stats.expon(scale=1 / lambda_)


# analytic sol
class Gamma(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html#scipy.stats.gamma"""

    def support(self) -> Tuple[float, float]:
        return (0, np.inf)

    def _create_scipy_dist(self) -> None:
        strict_positive_support(self.mean)
        alpha = self.mean**2 / self.variance
        beta = self.mean / self.variance
        self._scipy_dist = stats.gamma(a=alpha, scale=1 / beta)


# analytic sol
class InvGamma(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html#scipy.stats.invgamma"""

    def support(self) -> Tuple[float, float]:
        return (0, np.inf)

    def _create_scipy_dist(self) -> None:
        strict_positive_support(self.mean)
        alpha = self.mean**2 / self.variance + 2
        beta = self.mean * (self.mean**2 / self.variance + 1)
        self._scipy_dist = stats.invgamma(a=alpha, scale=beta)


# numerical sol
class Fisk(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisk.html#scipy.stats.fisk"""

    def support(self) -> Tuple[float, float]:
        return (0, np.inf)

    def _create_scipy_dist(self):
        positive_support(self.mean)

        optim_params = opt.minimize(
            fun=self._shape_scale,
            # start beta at 1.1 and solve for alpha
            x0=[self.mean * 1.1 * np.sin(np.pi / 1.1) / np.pi, 1.1],
            args=(self.mean, self.variance),
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

    def support(self) -> Tuple[float, float]:
        return (-np.inf, np.inf)

    def _create_scipy_dist(self) -> None:
        loc = self.mean - np.sqrt(self.variance * 6) * np.euler_gamma / np.pi
        scale = np.sqrt(self.variance * 6) / np.pi
        self._scipy_dist = stats.gumbel_r(loc=loc, scale=scale)


# hopelessly broken (sort of)
class Weibull(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min"""

    def support(self) -> Tuple[float, float]:
        return (0, np.inf)

    def _create_scipy_dist(self) -> None:
        positive_support(self.mean)

        # https://real-statistics.com/distribution-fitting/method-of-moments/method-of-moments-weibull/
        k = opt.root_scalar(self._func, x0=0.5, method="newton")
        lambda_ = self.mean / gamma_func(1 + 1 / k.root)

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

    def support(self) -> Tuple[float, float]:
        return (0, np.inf)

    def _create_scipy_dist(self) -> None:
        strict_positive_support(self.mean)
        mu = np.log(self.mean / np.sqrt(1 + (self.variance / self.mean**2)))
        sigma = np.sqrt(np.log(1 + (self.variance / self.mean**2)))
        # scipy multiplies in the argument passed to `scale` so in the exponentiated space,
        # you're essentially adding `mu` within the exponentiated expression within the
        # lognormal's PDF; hence, scale is with exponentiation instead of loc
        self._scipy_dist = stats.lognorm(scale=np.exp(mu), s=sigma)


# analytic sol
class Normal(Distribution):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm"""

    def support(self) -> Tuple[float, float]:
        return (-np.inf, np.inf)

    def _create_empty_scipy_dist(self) -> None:
        self._scipy_dist = stats.norm

    def _create_scipy_dist(self) -> None:
        self._scipy_dist = stats.norm(
            loc=self.mean, scale=np.sqrt(self.variance)
        )


# analytic sol
class Beta(Distribution):
    # TODO: WANT TO BE ABLE TO PASS IN UPPER AND LOWER BOUNDS TO BE REFLECTED IN THE DIST
    # EX: MEAN 6, VAR 0.2, LB 5, UB 10
    # ADJ_MEAN = (MEAN - LB) / INTERVAL_WIDTH
    # ADJ_VAR = VAR / INTERVAL_WIDTH
    # INPUT ADJ MEAN & VAR INTO FUNCTION
    # JUST GET RVS TO WORK FOR NOW, WHEN YOU TAKE A SAMPLE OF SIZE 100,
    # JUST MULTIPLICATIVELY SCALE AND THEN LINERALY SHIFT THE DATA TO THE ORIGINAL BOUNDS
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html#scipy.stats.beta"""

    def support(self) -> Tuple[float, float]:
        return (0, 1)

    def _create_scipy_dist(self) -> None:
        beta_bounds(self.mean)
        alpha = (
            self.mean**2 * (1 - self.mean) - self.mean * self.variance
        ) / self.variance
        beta = (
            (1 - self.mean)
            * (self.mean - self.mean**2 - self.variance)
            / self.variance
        )
        print(alpha, beta)
        self._scipy_dist = stats.beta(a=alpha, b=beta)


class MSCABeta(Distribution):
    def _create_scipy_dist(self) -> None:
        self.width = self.ub - self.lb
        adj_mean = (self.mean - self.lb) / self.width
        adj_var = self.variance / self.width
        self._scipy_dist = Beta(adj_mean, adj_var)

    def support(self) -> Tuple[float, float]:
        """create tuple representing endpoints of support"""

    def rvs(self, *args, **kwds):
        """defaults to scipy implementation for generating random variates

        Returns
        -------
        np.ndarray
            random variates from a given distribution/parameters
        """
        return (self._scipy_dist.rvs(*args, **kwds) + self.lb) * self.width

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
        return (self._scipy_dist.pdf(x) + self.lb) * self.width

    # def cdf(self, q: npt.ArrayLike) -> np.ndarray:
    #     """defaults to scipy implementation for cumulative density function

    #     Parameters
    #     ----------
    #     q : npt.ArrayLike
    #         quantiles

    #     Returns
    #     -------
    #     np.ndarray
    #         CDF evaluated at quantile q
    #     """
    #     return self._scipy_dist.cdf(q)

    # def ppf(self, p: npt.ArrayLike) -> np.ndarray:
    #     """defaults to scipy implementation for percent point function

    #     Parameters
    #     ----------
    #     p : npt.ArrayLike
    #         lower tail probability

    #     Returns
    #     -------
    #     np.ndarray
    #         PPF evaluated at lower tail probability p
    #     """
    #     return self._scipy_dist.ppf(p)

    # def stats(self, moments: str) -> Union[float, Tuple[float, ...]]:
    #     """defaults to scipy implementation for obtaining moments

    #     Parameters
    #     ----------
    #     moments : str
    #         m for mean, v for variance, s for skewness, k for kurtosis

    #     Returns
    #     -------
    #     Union[float, Tuple[float, ...]]
    #         mean, variance, skewness, and/or kurtosis
    #     """
    #     return self._scipy_dist.stats(moments=moments)


distribution_dict = {
    "exponential": Exponential,
    "gamma": Gamma,
    "invgamma": InvGamma,
    "fisk": Fisk,
    "gumbel": GumbelR,
    "weibull": Weibull,
    "lognormal": LogNormal,
    "normal": Normal,
    "beta": Beta,
    "MSCAbeta": MSCABeta,
}


### HELPER FUNCTIONS
def positive_support(mean: float) -> None:
    if mean < 0:
        raise ValueError("This distribution is only supported on [0, np.inf)")


def strict_positive_support(mean: float) -> None:
    if mean <= 0:
        raise ValueError("This distribution is only supported on (0, np.inf)")


def beta_bounds(mean: float) -> None:
    if (mean < 0) or (mean > 1):
        raise ValueError("This distribution is only supposrted on [0, 1]")
