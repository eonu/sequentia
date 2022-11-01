KNN Regressor
=============

The KNN Regressor is a regressor that uses the :math:`k`-NN algorithm with DTW as a distance measure to identify a :math:`k`-neighborhood of the most similar training sequences to the sequence being predicted.

To predict an output :math:`y'\in\mathbb{R}` for a sequence :math:`O'`, the :class:`.KNNRegressor` works by:

1. | Calculating the **DTW distance** between :math:`O'` and every training sequence.
   
2. | Forming a **k-neighborhood** :math:`\mathcal{K}'=\left\{O^{(1)},\ldots,O^{(k)}\right\}` of the :math:`k` nearest training sequences to :math:`O'`.

3. | Calculating a **distance weighting** :math:`w^{(1)},\ldots,w^{(k)}` for each sequence in :math:`\mathcal{K}'`. 
   | A uniform weighting of 1 is used by default, meaning that all sequences in :math:`\mathcal{K}'` have equal influence on the predicted output :math:`y'`. However, custom functions such as :math:`e^{-x}` (where :math:`x` is the DTW distance) can be specified to increase weight on training sequences that are more similar to :math:`O'`.

4. | Calculating :math:`y'` as the **distance weighted mean of the outputs** :math:`y^{(1)},\ldots,y^{(k)}` of sequences in :math:`\mathcal{K}'`.
   
   .. math::

      y' = \frac{\sum_{k=1}^Kw^{(k)}y^{(k)}}{\sum_{k=1}^Kw^{(k)}}

.. note::

   Using a value of :math:`k` greater than 1 is highly recommended for regression, to reduce variance.

API reference
-------------

Class
^^^^^

.. autosummary::

   ~sequentia.models.knn.regressor.KNNRegressor

Methods
^^^^^^^

.. autosummary::

   ~sequentia.models.knn.regressor.KNNRegressor.__init__
   ~sequentia.models.knn.regressor.KNNRegressor.compute_distance_matrix
   ~sequentia.models.knn.regressor.KNNRegressor.dtw
   ~sequentia.models.knn.regressor.KNNRegressor.fit
   ~sequentia.models.knn.regressor.KNNRegressor.load
   ~sequentia.models.knn.regressor.KNNRegressor.plot_dtw_histogram
   ~sequentia.models.knn.regressor.KNNRegressor.plot_warping_path_1d
   ~sequentia.models.knn.regressor.KNNRegressor.plot_weight_histogram
   ~sequentia.models.knn.regressor.KNNRegressor.predict
   ~sequentia.models.knn.regressor.KNNRegressor.query_neighbors
   ~sequentia.models.knn.regressor.KNNRegressor.save
   ~sequentia.models.knn.regressor.KNNRegressor.score

|

.. autoclass:: sequentia.models.knn.regressor.KNNRegressor
   :members:
   :inherited-members:
   :exclude-members: get_params, set_params

.. rubric:: References

.. [#dtw_multi] `Mohammad Shokoohi-Yekta, Jun Wang and Eamonn Keogh (2015). On the Non-Trivial Generalization of Dynamic Time Warping to the Multi-Dimensional Case (Extended). SDM 2015. <https://www.cs.ucr.edu/~eamonn/Multi-Dimensional_DTW_Journal.pdf>`_
