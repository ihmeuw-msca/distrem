from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.stats
from scipy.special import gamma as gamma_func

# from scipy.special import gammainccinv, gammaincinv


class Distribution(ABC):
    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance
        # # some kind of dictionary with
        # #   key: the support (full real line, semi infinite, etc...)
        # #   value: function that gets called when distribution is initialized
        # self.support = None
        # self._support_setup()
        self._scipy_dist = None
        self._create_scipy_dist()

    @abstractmethod
    def _create_scipy_dist(self) -> None:
        """Create scipy distribution from mean and variance"""

    def rvs(self, *args, **kwds):
        return self._scipy_dist.rvs(*args, **kwds)

    def pdf(self, x: npt.ArrayLike):
        return self._scipy_dist.pdf(x)

    def cdf(self, x: npt.ArrayLike):
        return self._scipy_dist.cdf(x)

    def ppf(self, x: npt.ArrayLike):
        return self._scipy_dist.ppf(x)

    def stats(self, moments: str):
        return self._scipy_dist.stats(moments=moments)


# analytic sol
class Exponential(Distribution):
    support = "positive"

    def _create_scipy_dist(self) -> None:
        positive_support(self.mean)
        lambda_ = 1 / self.mean
        self._scipy_dist = scipy.stats.expon(scale=1 / lambda_)


# analytic sol
class Gamma(Distribution):
    support = "strictly positive"

    def _create_scipy_dist(self) -> None:
        strict_positive_support(self.mean)
        alpha = self.mean**2 / self.variance
        beta = self.mean / self.variance
        self._scipy_dist = scipy.stats.gamma(a=alpha, scale=1 / beta)


# analytic sol
class InvGamma(Distribution):
    support = "strictly positive"

    def _create_scipy_dist(self) -> None:
        strict_positive_support(self.mean)
        alpha = self.mean**2 / self.variance + 2
        beta = self.mean * (self.mean**2 / self.variance + 1)
        self._scipy_dist = scipy.stats.invgamma(a=alpha, scale=beta)


# numerical sol
class Fisk(Distribution):
    support = "positive"

    def _create_scipy_dist(self):
        positive_support(self.mean)
        optim_params = scipy.optimize.minimize(
            fun=self._shape_scale,
            # start beta at 1.1 and solve for alpha
            x0=[self.mean * 1.1 * np.sin(np.pi / 1.1) / np.pi, 1.1],
            args=(self.mean, self.variance),
            # options={"disp": True},
        )
        alpha, beta = np.abs(optim_params.x)
        # parameterization notes: numpy's c is wikipedia's beta, numpy's scale is wikipedia's alpha
        # additional note: analytical solution doesn't work b/c dependent on derivative
        self._scipy_dist = scipy.stats.fisk(c=beta, scale=alpha)

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
    support = "real line"

    def _create_scipy_dist(self) -> None:
        loc = self.mean - np.sqrt(self.variance * 6) * np.euler_gamma / np.pi
        scale = np.sqrt(self.variance * 6) / np.pi
        self._scipy_dist = scipy.stats.gumbel_r(loc=loc, scale=scale)


# hopelessly broken
class Weibull(Distribution):
    support = "positive"

    def _create_scipy_dist(self) -> None:
        positive_support(self.mean)
        optim_params = scipy.optimize.minimize(
            fun=self._shape_scale,
            # ideally can invert gamma function for k, then use mean / sd as a guess for lambda
            x0=[self.mean / gamma_func(1 + 1 / 1.5), 1.5],
            args=(self.mean, self.variance),
            options={"disp": True},
        )
        lambda_, k = np.abs(optim_params.x)
        print("params from optim: ", lambda_, k)
        self._scipy_dist = scipy.stats.weibull_min(c=k, scale=lambda_)

    def _shape_scale(self, x: list, samp_mean: float, samp_var: float) -> float:
        # TODO: TAKE A LOOK AT JAX SINCE IT DOES AUTOMATIC DERIVATIVES
        lambda_ = x[0]
        k = x[1]
        mean_guess = lambda_ * gamma_func(1 + (1 / k))
        variance_guess = lambda_**2 * (
            gamma_func(1 + (2 / k) - gamma_func(1 + (1 / k)) ** 2)
        )
        return (mean_guess - samp_mean) ** 2 + (variance_guess - samp_var) ** 2


# analytic sol (M.O.M. estimators)
class LogNormal(Distribution):
    support = "strictly positive"

    def _create_scipy_dist(self) -> None:
        strict_positive_support(self.mean)
        mu = np.log(self.mean / np.sqrt(1 + (self.variance / self.mean**2)))
        sigma = np.sqrt(np.log(1 + (self.variance / self.mean**2)))
        # scipy multiplies in the argument passed to `scale` so in the exponentiated space,
        # you're essentially adding `mu` within the exponentiated expression within the
        # lognormal's PDF; hence, scale is with exponentiation instead of loc
        self._scipy_dist = scipy.stats.lognorm(scale=np.exp(mu), s=sigma)


# analytic sol
class Normal(Distribution):
    support = "real line"

    def _create_scipy_dist(self) -> None:
        self._scipy_dist = scipy.stats.norm(
            loc=self.mean, scale=np.sqrt(self.variance)
        )


# analytic sol
class Beta(Distribution):
    support = "bounded"

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
        self._scipy_dist = scipy.stats.beta(a=alpha, b=beta)


# exp, gamma, invgamma, llogis, gumbel, weibull, lognormal, normal, mgamma, mgumbel, beta

# TODO: change strings later on
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
}


### HELPER FUNCTIONS
# the following functions give a crude solution to negative means which surely mean the data is negative
# what about data that is negative, but still has a positive mean?
def positive_support(mean: float) -> None:
    if mean < 0:
        raise ValueError("This distribution is only supported on [0, np.inf)")


def strict_positive_support(mean: float) -> None:
    if mean <= 0:
        raise ValueError("This distribution is only supported on (0, np.inf)")


def beta_bounds(mean: float) -> None:
    if (mean < 0) or (mean > 1):
        raise ValueError("This distribution is only supposrted on [0, 1]")
