==========
Quickstart
==========

Example
-------

**Premise:** You have individual level fasting plasma glucose (FPG) data in units of mmol/L for a
population of 1000 patients.

**Role of** :code:`ensemble` **package:** fit a density curve composed of a weighted sum of named
probability distributions of your choice to the histogram of your FPG values

The following code chunk generates data under the above premise, then fits a density curve of an
ensemble distribution composed of the fisk and weibull distributions to the generated data. We'll
optimize the distance between the eCDF of the data and the ensemble distribution's CDF by
minimizing the KS statistic

.. code-block:: python

    import scipy.stats
    from ensemble.model import EnsembleDistribution, EnsembleFitter

    # "true" data under the above FPG premise
    true_model = EnsembleDistribution(
        distributions={"Gamma": 0.5, "Fisk": 0.1, "InvGamma": 0.05, "Weibull": 0.3, "LogNormal": 0.05},
        mean=4,
        variance=1
    )
    fpg_data = true_model.rvs(1000)

    # fitting the fisk weibull ensemble distribution
    fsk_wbl_fitter = EnsembleFitter(
        distributions=["Fisk", "Weibull"],
        objective="KS"
    )
    weights, fsk_wbl_dist = fsk_wbl_fitter.fit(fpg_data)

    # the 1st and 2nd moments of your fitted ensemble distributions are guaranteed to match those
    # of your inputted data by solving for appropriate values of the component distributions parameters
    fpg_mean, fpg_variance = fsk_wbl_dist.ensemble_stats("mv")
    # Output: fpg_mean == 4, fpg_variance == 1

    # default plotting function for a demo visualization
    fisk_wbl_dist.plot()

    # if a bound in your ensemble is finite (0, in this case), you may change it to ensure no
    # density is assigned below/above that bound, let's change the lower bound to 2
    weights, fsk_wbl_dist = fsk_wbl_fitter.fit(fpg_data, lb=2)

    # try and see how the default plot changes!
    fisk_wbl_dist.plot()


**Please see** :ref:`Plotting` **for a practical guide on plotting with ensemble.**