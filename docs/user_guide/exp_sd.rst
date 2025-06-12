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

Single Prevalence
-----------------

Imagine that a researcher obtains the following from previous literature, and would like to obtain
an estimate of the standard deviation of SBP in young males living in Seattle:

* mean SBP for young adult males in Seattle is 117
* of this population 0.05 have SBP above 160
* the distribution of SBP is modeled well by an ensemble distribution with components Gamma 0.7 and Weibull 0.3

.. code-block:: python

    import pandas as pd
    from distrem.model import SDOptimizer

    model = SDOptimizer(
        mean=117,
        named_weights={"Gamma": 0.7, "Weibull": 0.3}
    )

    df = pd.DataFrame(
        data={
              "weights": [1],
              "lb": [160],
              "ub": [np.inf],
              "prev": [0.05],
        }
    )

    sd = model.optimize_sd(df)

The value returned from the function will simply be the scalar value of SD that the optimizer
converges at.

Multiple Prevalence Values
--------------------------

The researcher obtains additional prevalence values. They now know the additional **bolded**
information.

* mean SBP for young adult males in Seattle is 117
* of this population, **0.5 have SBP between 120 and 140, 0.1 between 140 and 160,** and 0.05 are above 160
* the distribution of SBP is modeled well by an ensemble distribution with components Gamma 0.7 and Weibull 0.3

.. code-block:: python

    import pandas as pd
    from distrem.model import SDOptimizer

    model = SDOptimizer(
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

    sd = model.optimize_sd(df)

One Interval w/Multiple Prevalence Values
-----------------------------------------

The researcher discovers a prevalence result from a paper with a larger sample size for the :math:`[160, \infty)`
range that they'd like to include. They decide to incorporate the additional **bolded** information.

* mean SBP for young adult males in Seattle is 117
* of this population, 0.5 have SBP between 120 and 140, 0.1 between 140 and 160, and 0.05 are above 160
* **the higher sample size paper measures that in this population, 0.02 have SBP have SBP above 160**
* the distribution of SBP is modeled well by an ensemble distribution with components Gamma 0.7 and Weibull 0.3

.. code-block:: python

    import pandas as pd
    from distrem.model import SDOptimizer

    model = SDOptimizer(
        mean=117,
        named_weights={"Gamma": 0.7, "Weibull": 0.3}
    )

    df = pd.DataFrame(
        data={
              "weights": [0.1, 0.4, 0.2, 0.3],
              "lb": [120, 140, 160, 160],
              "ub": [140, 160, np.inf, np.inf],
              "prev": [0.5, 0.1, 0.05, 0.02],
        }
    )

    sd = model.optimize_sd(df)

For the interval :math:`[160, \infty)` here, after aggregation, the weight would be 0.5 and the
prevalence value would be 0.032. See this :ref:`page <SDOptimizer>` for aggregation details.
