Pipeline
========

Before fitting and using a model, it is common to apply a sequence of preprocessing steps to data.

Pipelines can be used to wrap preprocessing transformations as well as a model into a single estimator,
making it more convenient to reapply the transformations and make predictions on new data.

The :class:`.Pipeline` class implements this feature and is based on :class:`sklearn.pipeline.Pipeline`.

API reference
-------------

Class
^^^^^

.. autosummary::

   ~sequentia.pipeline.Pipeline

Methods
^^^^^^^

.. autosummary::

   ~sequentia.pipeline.Pipeline.__init__
   ~sequentia.pipeline.Pipeline.fit
   ~sequentia.pipeline.Pipeline.fit_predict
   ~sequentia.pipeline.Pipeline.fit_transform
   ~sequentia.pipeline.Pipeline.inverse_transform
   ~sequentia.pipeline.Pipeline.predict
   ~sequentia.pipeline.Pipeline.predict_proba
   ~sequentia.pipeline.Pipeline.score
   ~sequentia.pipeline.Pipeline.transform

|

.. autoclass:: sequentia.pipeline.Pipeline
   :members:
   :inherited-members:
   :exclude-members: decision_function, get_feature_names_out, get_params, set_params, set_output, predict_log_proba, score_samples, feature_names_in_
