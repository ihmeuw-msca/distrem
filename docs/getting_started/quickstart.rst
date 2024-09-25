==========
Quickstart
==========

Example
-------

.. code-block:: python

    import scipy.stats
    from ensemble.ensemble_model import EnsembleModel, EnsembleFitter
    # creates an ensemble distribution composed of the normal and gumbel distributions both sharing
    # the same mean and variance
    normal_gumbel = EnsembleModel(distributions=["normal", "gumbel"],
                                  weights=[0.25, 0.75],
                                  mean=4,
                                  variance=1)

    # fits an EnsembleModel object to standard normal draws. Here, the user has specified a
    # distribution (the gumbel) that is not reflected in the truth. Try on your own to see how the
    # model reflects this!
    std_norm_draws = scipy.stats.norm.rvs(0, 1, size=100)
    model = EnsembleFitter(["normal", "gumbel"], "L2").fit(std_norm_draws)

    fitted_weights = model.weights
    fitted_distribution = model.ensemble_distribution

    # default plotting function for a demo visualization
    normal_gumbel.plot()

**Please see** :ref:`Plotting` **for a practical guide on plotting with ensemble.**