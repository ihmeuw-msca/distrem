=====================
Ensemble distribution
=====================

There are currently 8 named distributions that are available for use in this package.
:code:`distributions.py` has implementation details if desired.

1. Exponential
2. Gamma
3. Inverse Gamma
4. Fisk (aka Log-Logistic)
5. Gumbel
6. Log-Normal
7. Normal
8. Beta
9. Weibull

These distributions have "`supports <https://en.wikipedia.org/wiki/Support_(mathematics)#In_probability_and_measure_theory>`_"
that differ from each other. Distributions with differing supports **cannot** be part of the same
ensemble distribution. For example, the Normal distribution is supported on the entire real line,
but the Gamma is only supported on (0, :math:`\infty`), meaning they would be incompatible in an
ensemble distribution.

After creating an EnsembleDistribution object, you can use various functions akin to those from
scipy's :code:`rv_continuous` class. These functions are:

* :code:`pdf()`
* :code:`cdf()`
* :code:`ppf()`
* :code:`rvs()`
* :code:`stats_temp()`

Example: Normal/Gumbel ensemble
-------------------------------

When creating an ensemble distribution, you are required to provide the following:

* list of distributions
* list of weights
* mean
* variance

In code form, this looks like...

.. code-block:: python

    from distrem.model import EnsembleDistribution

    ensemble_ex = EnsembleDistribution(
        named_weights={"Normal": 0.7, "GumbelR": 0.3},
        mean=-4
        variance=5
    )

Now, to create 100 draws from this ensemble, or get its PDF, you can do the following...


.. code-block:: python

    # create 100 draws from ensemble
    ensemble_draws = enesmble_ex.rvs(size=100)
    # return pdf values at x values [-3, 0, 1]
    ensemble_pdf = ensemble_ex.pdf(x=[-3, 0, 1])