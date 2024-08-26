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

Note: We define an ensemble distribution in this package to be a weighted sum of individual named
distributions. This resulting ensemble distribution must have weights that sum to 1 in order to
satisfy the property that a probability density function (PDF) must integrate to 1.

For installing the package please check :ref:`Installing distrx`.
For a simple example please check :ref:`Quickstart`.