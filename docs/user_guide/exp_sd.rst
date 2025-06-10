===============
SD Optimization
===============

Another use case for ensemble distributions is to estimate a value for standard deviation given a
value for the mean and 1 or more prevalence values. The process is similar to using a z score, it's
associated x value, and an assumed mean to obtain the standard deviation of a normal distribution,
where the z score is found from the PPF function of the normal: (:math:`\sigma = \frac{\mu - x}{PPF(x)}`)

In the ensemble distribution case, there are 2 main differences. The first is that there is no well-
defined z-score function, hence the optimization, and the second is that there must be some assumed
list of component distributions. An example building from the SBP example in previous sections of
the user guide follows.

Imagine that a researcher obtains the following from previous literature, and would like to obtain
an estimate of the standard deviation of SBP in young males living in Seattle

* mean SBP for young adult males in Seattle is 117
* of this population, 0.5 have SBP between 120 and 140, 0.1 between 140 and 160, and 0.05 are above 160
* the distribution of SBP is modeled well by an ensemble distribution with compoenents Gamma 0.7 and Weibull 0.3

.. code-block:: python

    import pandas as pd
    from distrem.model import ExposureSDOptimizer

    model = ExposureSDOptimizer(
        mean=117,
        named_weights={"Gamma": 0.7, "Weibull": 0.3}
    )

    df = pd.DataFrame(
        data={
              "weights": [0.1, 0.4, 0.5],
              "lb": [120, 140, 160],
              "ub": [140, 160, np.inf],
              "prev": [0.5, 0.1, 0.05],
        }
    )

    model.optimize_sd(df)
