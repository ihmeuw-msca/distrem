================
Ensemble Fitting
================

In order to fit an ensemble distribution to microdata, use the :code:`EnsembleFitter` object. The
object must be initialized with 2 things.

*A list of named distributions.* These distributions have "supports" that differ from each other. A
support, for our purposes, can be thought of as the x values that are compatible with some given
distribution. For example, the Normal distribution is supported on the entire real line, so it can
take negative x values, but the Gamma is only supported on (0, :math:`\infty`), so it cannot take
negative values. **Recall: you are not permitted to use distributions with differing supports in the
same ensemble.**

*A penalty function of choice.* In a nutshell, we are minimizing the distances between the empirical
cumulative distribution function (eCDF) and the CDF of the ensemble subject to said chosen penalty.
The penalties currently implemented are as follows:

* :code:`"L1"`: L1 norm
* :code:`sum_squares"`: sum of squares
* :code:`"KS"`: the Kolmogorov-Smirnoff distance, A.K.A. infinity norm

Finally, the function of interest for this use case is the :code:`fit()` function.

Example: Fitting an Ensemble
----------------------------

Suppose we have microdata for systolic blood pressure (SBP) from a certain population of young
people in Seattle. Since SBP must be positive, let's use all the distributions (except the
exponential) with a positive support to fit this data.

.. code-block:: python

    import scipy.stats as stats
    from distrem.model import EnsembleFitter

    SBP_vals = stats.norm(loc=120, scale=7).rvs(size=100)
    model = EnsembleFitter(
        distributions=["Gamma", "InvGamma", "Fisk", "LogNormal"],
        objective="L2"
    )
    res = model.fit(SBP_vals)

:code:`res` contains an array of fitted weights as well as an :code:`EnsembleDistribution` object
that has already been initialized with the distributions provided to :code:`model`. They can be
accessed as follows:

.. code-block:: python

    # fitted weights
    fitted_weights = res.weights

    # fitted ensemble
    fitted_ensemble = res.ensemble_distribution
