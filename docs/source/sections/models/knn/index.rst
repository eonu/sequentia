Dynamic Time Warping K-Nearest Neighbors
========================================

.. toctree::
   :maxdepth: 2

   regressor
   classifier

----

:math:`k`-nearest neigbors (:math:`k`-NN) is an algorithm for identifying the observations most similar to an example.

By combining :math:`k`-NN with dynamic time warping as a distance measure, :math:`k`-NN can be used as an effective method of sequence classification and regression.

Recap of general :math:`k`-NN
-----------------------------

Suppose we have a dataset :math:`\mathcal{D}_\text{train}=\big\{\mathbf{x}^{(n)}\big\}_{n=1}^N` with :math:`N` training examples, such that :math:`\mathbf{x}^{(n)}\in\mathbb{R}^D`.

When querying against a new example  :math:`\mathbf{x}'\in\mathbb{R}^D`, the :math:`k`-NN algorithm does the following:

1. | Computes the distance from :math:`\mathbf{x}'` to every training example in :math:`\mathcal{D}_\text{train}`, using a distance measure such as Euclidean distance :math:`d(\mathbf{x}',\mathbf{x}^{(n)})=\Vert\mathbf{x}'-\mathbf{x}^{(n)}\Vert`.
2. | Obtains a :math:`k`-neighborhood :math:`\mathcal{K}'` which is a set of the :math:`k` nearest training examples to :math:`\mathbf{x}'`.

Using the intuition that :math:`\mathbf{x}'` being physically close to the examples in :math:`\mathcal{K}'` suggests that they have similar properties, we can use the :math:`k`-neighborhood for inference (classification or regression) on :math:`\mathbf{x}'`.

Extending :math:`k`-NN to sequences
-----------------------------------

To apply :math:`k`-NN to sequences, we use a distance measure that quantifies similarity between sequences.

Sequentia supports multivariate sequences for :math:`k`-NN, which can be represented as an ordered sequence of observations :math:`O=\mathbf{o}^{(1)},\ldots,\mathbf{o}^{(T)}`, such that :math:`\mathbf{o}^{(t)}\in\mathbb{R}^D`.
Indeed the lengths of any two observation sequences :math:`O^{(n)}` and :math:`O^{(m)}` may differ.

To compare sequences :math:`O^{(n)}` and :math:`O^{(m)}` we can use the **dynamic time warping** (DTW) distance measure, which is a dynamic programming algorithm for finding the optimal alignment between two sequences, and can be generalized to higher dimensions.