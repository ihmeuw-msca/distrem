from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize
import scipy.stats
from scipy.special import gamma as gamma_func

# from scipy.special import gammainccinv, gammaincinv


# distribution parent class to abstract away the diff scipy funcs
class Distribution(ABC):
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        self._scipy_dist = None
        self._create_scipy_dist()
        self.name = self._set_name()

    @abstractmethod
    def _create_scipy_dist(self) -> None:
        """Create scipy distribution from mean and variance"""

    def pdf(self, x):
        return self._scipy_dist.pdf(x)

    def ppf(self, x):
        return self._scipy_dist.ppf(x)

    def stats(self, moments):
        return self._scipy_dist.stats(moments=moments)


class Exponential(Distribution):
    def __init__(self, mean, variance):
        super().__init__(mean, variance)
        self._name = "exponential"

    def _create_scipy_dist(self) -> None:
        positive_support(self.mean)
        lambda_ = 1 / self.mean
        self._scipy_dist = scipy.stats.expon(scale=1 / lambda_)


class Gamma(Distribution):
    def _create_scipy_dist(self) -> None:
        strict_positive_support(self.mean)
        alpha = self.mean**2 / self.variance
        beta = self.mean / self.variance
        self._scipy_dist = scipy.stats.gamma(a=alpha, scale=1 / beta)


class InvGamma(Distribution):
    def _create_scipy_dist(self) -> None:
        strict_positive_support(self.mean)
        optim_params = scipy.optimize.minimize(
            fun=self._shape_scale,
            # a *good* friend told me that this is a good initial guess and it works so far???
            #   alpha = 3 is because alpha > 2 must be true due to variance formula
            #   beta = mean * (alpha - 1) after isolating beta from formula for mean
            x0=[3, self.mean * 2],
            args=(self.mean, self.variance),
        )
        shape, scale = np.abs(optim_params.x)
        self._scipy_dist = scipy.stats.invgamma(a=shape, scale=scale)

    def _shape_scale(self, x, samp_mean, samp_var) -> None:
        alpha = x[0]
        beta = x[1]
        mean_guess = beta / (alpha - 1)
        variance_guess = beta**2 / ((alpha - 1) ** 2 * (alpha - 2))
        return (mean_guess - samp_mean) ** 2 + (variance_guess - samp_var) ** 2


class Fisk(Distribution):
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
        self._scipy_dist = scipy.stats.fisk(c=beta, scale=alpha)

    def _shape_scale(self, x, samp_mean, samp_var) -> None:
        alpha = x[0]
        beta = x[1]
        b = np.pi / beta
        mean_guess = alpha * b / np.sin(b)
        variance_guess = alpha**2 * (
            (2 * b / np.sin(2 * b)) - b**2 / np.sin(b) ** 2
        )
        return (mean_guess - samp_mean) ** 2 + (variance_guess - samp_var) ** 2


class GumbelR(Distribution):
    def _create_scipy_dist(self) -> None:
        loc = self.mean - np.sqrt(self.variance * 6) * np.euler_gamma / np.pi
        scale = np.sqrt(self.variance * 6) / np.pi
        self._scipy_dist = scipy.stats.gumbel_r(loc=loc, scale=scale)


class Weibull(Distribution):
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

    def _shape_scale(self, x, samp_mean, samp_var) -> None:
        lambda_ = x[0]
        k = x[1]
        mean_guess = lambda_ * gamma_func(1 + (1 / k))
        variance_guess = lambda_**2 * (
            gamma_func(1 + (2 / k) - gamma_func(1 + (1 / k)) ** 2)
        )
        return (mean_guess - samp_mean) ** 2 + (variance_guess - samp_var) ** 2


class LogNormal(Distribution):
    def _create_scipy_dist(self) -> None:
        # using method of moments gets close, but not quite there
        loc = np.log(self.mean / np.sqrt(1 + (self.variance / self.mean**2)))
        scale = np.sqrt(np.log(1 + (self.variance / self.mean**2)))
        # loc = np.log(self.mean**2 / np.sqrt(self.mean**2 + self.variance))
        # scale = np.log(1 + self.variance / self.mean**2)
        self._scipy_dist = scipy.stats.lognorm(loc=loc, s=scale)


class Normal(Distribution):
    def _create_scipy_dist(self) -> None:
        self._scipy_dist = scipy.stats.norm(
            loc=self.mean, scale=np.sqrt(self.variance)
        )


class Beta(Distribution):
    def _create_scipy_dist(self) -> None:
        beta_bounds(self.mean)
        optim_params = scipy.optimize.minimize(
            fun=self._shape_scale,
            # trying something similar to invgamma, unsuccessful for variance
            x0=[2, self.mean * 2 - 2],
            args=(self.mean, self.variance),
            options={"disp": True},
        )
        alpha, beta = np.abs(optim_params.x)
        print("params from optim: ", alpha, beta)
        self._scipy_dist = scipy.stats.beta(a=alpha, b=beta)

    def _shape_scale(self, x, samp_mean, samp_var):
        alpha = x[0]
        beta = x[1]
        mean_guess = alpha / (alpha + beta)
        variance_guess = (
            alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        )
        return (mean_guess - samp_mean) ** 2 + (variance_guess - samp_var) ** 2


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
def positive_support(mean):
    if mean < 0:
        raise ValueError("This distribution is only supported on [0, np.inf)")


def strict_positive_support(mean):
    if mean <= 0:
        raise ValueError("This distribution is only supported on (0, np.inf)")


def beta_bounds(mean):
    if (mean < 0) or (mean > 1):
        raise ValueError("This distribution is only supposrted on [0, 1]")
