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
    fitted_model = model.ensemble_model
