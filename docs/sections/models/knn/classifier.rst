KNN Classifier
==============

The KNN Classifier is a classifier that uses the :math:`k`-NN algorithm with DTW as a distance measure to identify a :math:`k`-neighborhood of the most similar training sequences to the sequence being classified.

To classify a sequence :math:`O'`, the :class:`.KNNClassifier` works by:

1. | Calculating the **DTW distance** between :math:`O'` and every training sequence.

2. | Forming a **k-neighborhood** :math:`\mathcal{K}'=\left\{O^{(1)},\ldots,O^{(k)}\right\}` of the :math:`k` nearest training sequences to :math:`O'`.

3. | Calculating a **distance weighting** for each sequence in :math:`\mathcal{K}'`.
   | A uniform weighting of 1 is used by default, meaning that all sequences in :math:`\mathcal{K}'` have equal influence on the predicted class. However, custom functions such as :math:`e^{-x}` (where :math:`x` is the DTW distance) can be specified to increase classification weight on training sequences that are more similar to :math:`O'`.

4. | Calculating a **score** for each of the unique classes corresponding to the sequences in :math:`\mathcal{K}'`.
   | The score for each class is calculated as the sum of the distance weightings of all sequences in :math:`\mathcal{K}'` belonging to that class.

5. | Selecting the class with the **highest score**.
   | If there is a tie between classes, a class is randomly selected between the tied classes.

API reference
-------------

Class
^^^^^

.. autosummary::

   ~sequentia.models.knn.classifier.KNNClassifier

Methods
^^^^^^^

.. autosummary::

   ~sequentia.models.knn.classifier.KNNClassifier.__init__
   ~sequentia.models.knn.classifier.KNNClassifier.compute_distance_matrix
   ~sequentia.models.knn.classifier.KNNClassifier.dtw
   ~sequentia.models.knn.classifier.KNNClassifier.fit
   ~sequentia.models.knn.classifier.KNNClassifier.fit_predict
   ~sequentia.models.knn.classifier.KNNClassifier.load
   ~sequentia.models.knn.classifier.KNNClassifier.plot_dtw_histogram
   ~sequentia.models.knn.classifier.KNNClassifier.plot_warping_path_1d
   ~sequentia.models.knn.classifier.KNNClassifier.plot_weight_histogram
   ~sequentia.models.knn.classifier.KNNClassifier.predict
   ~sequentia.models.knn.classifier.KNNClassifier.predict_proba
   ~sequentia.models.knn.classifier.KNNClassifier.predict_scores
   ~sequentia.models.knn.classifier.KNNClassifier.query_neighbors
   ~sequentia.models.knn.classifier.KNNClassifier.save
   ~sequentia.models.knn.classifier.KNNClassifier.score

|

.. autoclass:: sequentia.models.knn.classifier.KNNClassifier
   :members:
   :inherited-members:
   :exclude-members: get_params, set_params

.. rubric:: References

.. [#dtw_multi] `Mohammad Shokoohi-Yekta, Jun Wang and Eamonn Keogh (2015). On the Non-Trivial Generalization of Dynamic Time Warping to the Multi-Dimensional Case (Extended). SDM 2015. <https://www.cs.ucr.edu/~eamonn/Multi-Dimensional_DTW_Journal.pdf>`_
