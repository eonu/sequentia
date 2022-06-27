.. _knn:

Dynamic Time Warping `k`-Nearest Neighbors Classifier (``KNNClassifier``)
=========================================================================

The :math:`k`-nearest neighbors (:math:`k`-NN) classification algorithm is a very commonly used algorithm,
and perhaps one of the most intuitive ones too.

Before we discuss :math:`k`-NN with dynamic time warping for sequence classification,
let us recap :math:`k`-NN in the usual case of individual points :math:`\mathbf{x}\in\mathbb{R}^D`
in :math:`D`-dimensional Euclidean space.

Recap of general `k`-NN
-----------------------

Suppose we have a training dataset :math:`\mathcal{D}_\text{train}=\big\{(\mathbf{x}^{(n)},c^{(n)})\big\}_{n=1}^N`,
where :math:`N` is the number of training example-label pairs, :math:`\mathbf{x}^{(n)}\in\mathbb{R}^D` is a training example
and :math:`c^{(n)}\in\{1,\ldots,C\}` is its corresponding label.

When classifying a new test example :math:`\mathbf{x}^{(m)}\in\mathbb{R}^D`, a :math:`k`-NN classifier does the following:

1. Computes the distance from :math:`\mathbf{x}^{(m)}` to every training example in :math:`\mathcal{D}_\text{train}`, using a distance measure such as Euclidean distance, :math:`d(\mathbf{x}^{(m)},\mathbf{x}^{(n)})=\Vert\mathbf{x}^{(m)}-\mathbf{x}^{(n)}\Vert`.
2. Assigns :math:`c^{(m)}` as the most common label of the :math:`k` nearest training examples.

The most immediate observation one can make is that for every prediction,
you need to look through the entire training dataset. As you can imagine, the inability to summarize
the model with simpler parameters (e.g. weights of a neural network, or transition/emission/initial probabilities for a HMM),
limits the practical use of :math:`k`-NN classifiers – especially on large datasets.

While this classifier is conceptually simple, it can very often outperform more sophisticated
machine learning algorithms in various classification tasks, even when looking only at the nearest neighbor (:math:`k=1`).

The algorithm works on the intuition that if :math:`\mathbf{x}^{(m)}` has similar features to :math:`\mathbf{x}^{(n)}`,
then they should physically be close together (in :math:`D`-dimensional Euclidean space).

Extending `k`-NN to sequences
-----------------------------

We can try to extend this intuition to work with sequences. In general, Sequentia supports multivariate
observation sequences. These can be represented as an ordered sequence :math:`O=\mathbf{o}^{(1)},\ldots,\mathbf{o}^{(T)}`
of observations :math:`\mathbf{o}^{(t)}\in\mathbb{R}^D`. Indeed, the durations of any two observation sequences
:math:`O^{(n)}` and :math:`O^{(m)}` may differ.

When trying to apply the :math:`k`-NN intuition to observation sequences, we can say that
two sequences :math:`O^{(n)}` and :math:`O^{(m)}` which are similar to each other should have a small *'distance'*,
and if they are different, they should have a large *'distance'*.

But what sort of *'distance'* could this be? We need a measure that can compare any two sequences of different
length, and is small when the sequences are similar, and large if they are different. One such distance measure
that allows us to compare sequences of different length is `Dynamic Time Warping <https://en.wikipedia.org/wiki/Dynamic_time_warping>`_ (DTW).

Given sequence-label pairs :math:`\mathcal{D}_\text{train}=\big\{(O^{(n)},c^{(n)})\big\}_{n=1}^N`,
apart from the fact that we now compute DTW distances between sequences rather than Euclidean distances between points,
the rest of the :math:`k`-NN algorithm remains unchanged, and indeed :math:`k`-NN and DTW coupled together creates a powerful sequence classifier.

This :math:`k`-NN classifier with the DTW distance measure is implemented by the :class:`~KNNClassifier` class.

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