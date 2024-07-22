from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize
import scipy.stats


# distribution parent class to abstract away the diff scipy funcs
class Distribution(ABC):
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        self._scipy_dist = None
        self._create_scipy_dist()

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
    def _create_scipy_dist(self) -> None:
        lambda_ = 1 / self.mean
        self._scipy_dist = scipy.stats.expon(scale=1 / lambda_)


class Gamma(Distribution):
    def _create_scipy_dist(self) -> None:
        alpha = self.mean**2 / self.variance
        beta = self.mean / self.variance
        self._scipy_dist = scipy.stats.gamma(a=alpha, scale=1 / beta)


class InvGamma(Distribution):
    def _create_scipy_dist(self) -> None:
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
        optim_params = scipy.optimize.minimize(
            fun=self._shape_scale,
            x0=[2, self.mean * 2 / np.pi * np.sin(np.pi / 2)],
            args=(self.mean, self.variance),
        )
        shape, scale = np.abs(optim_params.x)
        print("parameters from optimizer: ", shape, scale)
        self._scipy_dist = scipy.stats.fisk(c=shape, scale=scale)

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
        raise NotImplementedError


class LogNormal(Distribution):
    def _create_scipy_dist(self) -> None:
        raise NotImplementedError


class Normal(Distribution):
    def _create_scipy_dist(self) -> None:
        self._scipy_dist = scipy.stats.norm(
            loc=self.mean, scale=np.sqrt(self.variance)
        )


class Beta(Distribution):
    def _create_scipy_dist(self) -> None:
        raise NotImplementedError


# exp, gamma, invgamma, llogis, gumbel, weibull, lognormal, normal, mgamma, mgumbel, beta


# distribution_dict = {"exponential": Exponential()}
