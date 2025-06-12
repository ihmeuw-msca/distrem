========
Concepts
========

Distributions
-------------

Each individual distribution in an ensemble is fit to the given mean and variance of the data. This
process typically involves using algebra to isolate the parameters of the distributions with the
sample mean and variance as given, and then solving for the 2 parameter system. You may look within
the :code:`create_scipy_dist()` function to find the equations used. The single exception is the
Fisk distribution, where the form of the PDF necessitates the use of numerical minimization

EnsembleDistribution
--------------------

PDF, CDF, PPF
^^^^^^^^^^^^^

Methods used for creating the PDF, CDF, and PPF of the EnsembleDistribution object are relatively
"off the shelf" so to speak, generally following the structure and methodology of scipy's
implementation `here <https://github.com/scipy/scipy/blob/v1.14.0/scipy/stats/_distn_infrastructure.py>`_.
In summary, the PDF and CDF can just be weighted linear combinations of the component distributions
while the PPF requires use use of Brent's algorithm to solve for the quantile corresponding to the
correct point in the PDF.

rvs
^^^

A.K.A. scipy's function to generate draws, was not implemented by solving for the PPF, as listed in
the source code above. Instead, since a linear combination of distributions is functionally
equivalent to sampling from individual distributions with probability of sampling from a
distribution dictated by a multinomial distribution, the latter method has been chosen here for
efficiency purposes.

ensemble_stats
^^^^^^^^^^^^^^

A getter function for the mean and variance supplied to the EnsembleDistribution object, does not
supply skewness and kurtosis like scipy's :code:`stats()`.

EnsembleFitter
--------------

The :code:`fit()` function performs fitting of ensemble distributions by minimizing the distances
of the eCDF of given microdata to the CDF of an ensemble distribution subject to some penalty.
Legacy code at IHME implements only the Kolmogorov-Smirnoff distance, but the sum of squares and L1
norm distance metrics have also been implemented as well.

tsh_points
^^^^^^^^^^

The threshold points feature has been implemented by only feeding a specific subset of points to the
optimization function instead of all of the data, as is done in the base case. If weights are
provided, then they are multiplied in before the optimization function is applied. The user may
choose to provide equal weights by feeding in a :code:`np.ones` array of the same length as :code:`tsh_points`
if they do not wish to apply heavier or lighter weights on specific parts of the data.

SDOptimizer
-----------

The SD optimizer performs runs bounded optimizations using scipy's :code:`scipy.optimize.minimize_scalar(method="bounded")`
with a lower bound of 0 (since SD cannot be negative) and an upper bound of 1.5 times an initial
guess of the SD using the z score of the normal distribution as briefly described in the first
paragraph of :ref:`this<SD Optimization>` page. The objective function minimized over is the
following:

.. math::
  \min\limits_{\sigma} \mathbf{w} \cdot \mathbf{\left((F(q_u; \sigma, \mu) - F(q_\ell; \sigma, \mu)) - \hat{p}\right)}^2

* :math:`\mu` is the mean
* :math:`\sigma` is the SD
* :math:`\mathbf{w}` is an array of weights which must sum to 1
* :math:`\mathbf{F}` is the CDF of the ensemble distribution function
* :math:`\mathbf{\hat{p}}` is an array of the observed prevalence values between specified upper and lower bound quantiles
* :math:`\mathbf{q}` are arrays of the quantiles corresponding to the upper and lower bound of the observed prevalence values, with the :math:`u` and :math:`\ell` subscripts denoting upper or lower, respectively

When the user provides multiple measures of prevalence for the same interval as demonstrated
:ref:`here <One Interval w/Multiple Prevalence Values>`, the values are aggregated as follows:

* the aggregate weight for the interval is the sum of the individual weight values
* the aggregate prevalence value for the interval is the weighted sum of the invididual prevalence values times the reciprocal of the sum of the weights

Please refer to the aforementioned demonstration for an example of the aggregation.
