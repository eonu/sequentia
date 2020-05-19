.. _knn:

Dynamic Time Warping `k`-Nearest Neighbors Classifier (``KNNClassifier``)
=========================================================================

| Recall that the isolated sequences we are dealing with are represented as
  multivariate time series of different durations.
| Suppose that our sequences are all :math:`D`-dimensional. The main requirement of
  `k-Nearest Neighbor <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_
  (:math:`k`-NN) classifiers is that each example must have the same number of
  dimensions – and hence, be in the same feature space. This is indeed the case with
  our :math:`D`-dimensional sequences. However, we can't use :math:`k`-NN with simple
  distance metrics such as Euclidean distance because we are comparing sequences
  (which represent an ordered collection of points in :math:`D`-dimensional space)
  rather than individual points in :math:`D`-dimensional space.

One distance metric that allows us to compare multivariate sequences of different length
is `Dynamic Time Warping <https://en.wikipedia.org/wiki/Dynamic_time_warping>`_. Coupling this metric
with :math:`k`-NN creates a powerful classifier that assigns the class of a new
observation sequence by looking at the classes of observation sequences with similar patterns.

However, :math:`k`-NN classifiers suffer from the fact that they are non-parametric,
which means that when predicting the class for a new observation sequence,
we must look back at every observation sequence that was used to fit the model.
To speed up prediction times, we have chosen to use a constrained DTW algorithm that
sacrifices accuracy by calculating an approximate distance, but saves **a lot** of time.
This is the `FastDTW <https://pdfs.semanticscholar.org/05a2/0cde15e172fc82f32774dd0cf4fe5827cad2.pdf>`_
implementation, which has a `radius` parameter for controlling the imposed constraint on the distance calculation.

This approximate DTW :math:`k`-NN classifier is implemented by the :class:`~KNNClassifier` class.

Example
-------

.. literalinclude:: ../../_includes/examples/classifiers/knn_classifier.py
    :language: python
    :linenos:

For more elaborate examples, please have a look at the `example notebooks <https://nbviewer.jupyter.org/github/eonu/sequentia/tree/master/notebooks>`_.

API reference
-------------

.. autoclass:: sequentia.classifiers.knn.KNNClassifier
    :members: