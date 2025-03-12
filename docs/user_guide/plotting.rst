Plotting
========

Let's reuse the example from :ref:`Example: Fitting an Ensemble` with the SBP. There are many
things that you may want to plot, but 2 useful plots that will be demonstrated below are:

* a density histogram of the SBP overlaid with the ensemble PDF
* a comparison between the eCDF of the SBP data and the ensemble's CDF

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    # generate SBP data and fit model as before
    data = stats.norm(loc=120, scale=7).rvs(size=100)
    model = EnsembleFitter(
        distributions=["Gamma", "InvGamma", "Fisk", "LogNormal"],
        objective="L2"
    )
    res = model.fit(data)

    # set up matplotlib plotting
    fig, ax = plt.subplots(1, 2)
    support = np.linspace(np.min(data), np.max(data), 1000)

    # plot histogram w/fitted PDF
    ax[0].hist(data, density=True, bins=30)
    ax[0].plot(support, fitted_distribution.pdf(support))
    ax[0].set_xlabel("SBP (mm/Hg)")
    ax[0].set_ylabel("density")
    ax[0].set_title("SBP histogram w/ensemble PDF Overlay")

    # plot eCDF vs fitted CDF
    stats.ecdf(std_norm_draws).cdf.plot(ax[1])
    ax[1].plot(support, fitted_distribution.cdf(support))
    ax[1].set_xlabel("SBP (mm/Hg)")
    ax[1].set_ylabel("density")
    ax[1].set_title("Empirical vs Ensemble CDF Comparison")

**What is** :code:`support` **?:** You can think of :code:`support` as the x values (in the space of the
data)for which we will calculate corresponding y values of density for, whether that be the PDF or
CDF.