==============
Ensemble Model
==============

There are currently 8 named distributions that are available for use in this package. See
:code:`distributions.py` for implementation details if desired. In general, you do **not** have to
interact with this file to be able to perform the functions described in the documentation.

1. Exponential
2. Gamma
3. Inverse Gamma
4. Fisk (aka Log-Logistic)
5. Gumbel
6. Log-Normal
7. Normal
8. Beta
9. Weibull

These distributions have "supports" that differ from each other. A support, for our purposes, can be
thought of as the x values that are compatible with some given distribution. For example, the Normal
distribution is supported on the entire real line, so it can take negative x values, but the Gamma
is only supported on (0, :math:`\infty`), so it cannot take negative values. **Recall: you are not
permitted to use distributions with differing supports in the same ensemble.**

After creating an EnsembleModel object, you can use various functions akin to those from scipy's
:code:`rv_continuous` class. These functions are:

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

    from ensemble.model import EnsembleDistribution

    ensemble_ex = EnsembleDistribution(
        distributions=["Normal", "GumbelR"],
        weights=[0.7, 0.3],
        mean=-4
        variance=5
    )

Now, to create 100 draws from this ensemble, or get its PDF, you can do the following...


.. code-block:: python

    # create 100 draws from ensemble
    ensemble_draws = enesmble_ex.rvs(size=100)
    # return pdf values at x values [-3, 0, 1]
    ensemble_pdf = ensemble_ex.pdf(x=[-3, 0, 1])