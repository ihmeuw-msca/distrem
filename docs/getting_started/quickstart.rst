==========
Quickstart
==========

Example
-------

.. code-block:: python

    import scipy.stats
    from ensemble.ensemble_model import EnsembleModel, EnsembleFitter
    # creates an ensemble distribution composed of the normal and gumbel distributions both sharing
    # the same mean and variance; the normal distribution can be thought of as contributing a
    # quarter of the "height" of the density curve to the ensemble's density, and the gumbel as
    # contributing the remaining 3 quarters
    normal_gumbel = EnsembleModel(distributions=["normal", "gumbel"],
                                  weights=[0.25, 0.75],
                                  mean=4,
                                  variance=1)

    # fits an EnsembleModel object to standard normal draws. Here, the user has specified a
    # distribution (the gumbel) that is not reflected in the truth. The model typically correctly
    # identifies this and will give weights close to 1 for the normal, and 0 for the gumbel
    std_norm_draws = scipy.stats.norm.rvs(0, 1, size=100)
    model = EnsembleFitter(["normal", "gumbel"], "L2").fit(std_norm_draws)

    fitted_weights = model.weights
    fitted_distribution = model.ensemble_distribution

Using both the draws from the standard normal and the fitted :code:`EnsembleModel` object from
above, we can also plot the results with the help of the :code:`matplotlib` package. There are many
things that you may want to plot, but 2 useful plots that will be demonstrated below are a density
histogram overlaid with the ensemble PDF, as well as a comparison of the eCDF and the ensemble's
CDF.

Plotting
--------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)
    support = np.linspace(np.min(std_norm_draws), np.max(std_norm_draws), 1000)

    # plot histogram vs fitted PDF
    ax[0].hist(std_norm_draws, density=True, bins=30)
    ax[0].plot(support, fitted_distribution.pdf(support))
    ax[0].set_xlabel("DATA VALUES (UNITS)")
    ax[0].set_ylabel("density")
    ax[0].set_title("DATA histogram w/ensemble PDF Overlay")

    # plot eCDF vs fitted CDF
    stats.ecdf(std_norm_draws).cdf.plot(ax[1])
    ax[1].plot(support, fitted_distribution.cdf(support))
    ax[1].set_xlabel("DATA VALUES (UNITS)")
    ax[1].set_ylabel("density")
    ax[1].set_title("Empirical vs Ensemble CDF Comparison")