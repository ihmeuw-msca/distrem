================
Ensemble Fitting
================

Basics
------

The 2 main functionalities provided in this package are the following:

#. Fit an ensemble distribution to a set of microdata :ref:`Fitting to Microdata`
#. Optimize standard deviation of an ensemble distribution w/known mean, component distributions to 1 or more observed prevalence values.

Fitting to Microdata
--------------------

In order to fit an ensemble distribution to microdata, use the :code:`EnsembleFitter` object. The
object must be initialized with 2 things.

*A list of named distributions.* These distributions have "supports" that differ from each other. A
support, for our purposes, can be thought of as the x values that are compatible with some given
distribution. For example, the Normal distribution is supported on the entire real line, so it can
take negative x values, but the Gamma is only supported on (0, :math:`\infty`), so it cannot take
negative values. **Recall: you are not permitted to use distributions with differing supports in the
same ensemble.**

*A penalty function of choice.* In a nutshell, we are minimizing across all the distance values
between the empirical cumulative distribution function (eCDF) and the CDF of the ensemble subject to
the user's chosen penalty. The penalties currently implemented are as follows:

* :code:`"L1"`: L1 norm
* :code:`sum_squares"`: sum of squares
* :code:`"KS"`: the Kolmogorov-Smirnoff distance, A.K.A. infinity norm

You'll then call the the :code:`fit()` function after creating the model.

Example: Fitting an Ensemble
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Example: Fitting an Ensemble w/Thresholds
-----------------------------------------

Optional additional parameters may be provided to the :code:`fit()` function for specific use cases.

*Threshold points and weights* If you do not wish to minimize across all the distance
values between the eCDF and the ensemble's CDF, you may provide a subset of values at which the fit
should be prioritized. Example use cases include providing the left half of the microdata with
weights that are all equal to minimize only the distances of the left half of the distribution or
providing 2 important values 25 and 29, with weights of 0.3 and 0.7 to ensure a close fit at these
points.

