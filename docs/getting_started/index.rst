Getting started
===============

.. toctree::
   :hidden:

   installation
   quickstart

Welcome to ensemble!
--------------------

ensemble allows you to fit a weighted linear combination of distributions to individual-level data,
or create an ensemble distribution given mean, variance, the distributions you want to include, and
their respective weights.

**BEFORE YOU PROCEED**

* We define an ensemble distribution in this package to be a weighted sum of individual named distributions.
* Weights must sum to 1 in order to satisfy the property that a probability density function (PDF) must integrate to 1.
* Distributions with differing supports cannot be in the same ensemble.
* Current implementation forces mean and variance to be equivalent over each distribution, and forces ensemble mean and variance to match data, if relevant.

For installing the package please check :ref:`Installing ensemble`.
For a simple example please check :ref:`Quickstart`.