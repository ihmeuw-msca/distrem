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

EnsembleModel
-------------

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