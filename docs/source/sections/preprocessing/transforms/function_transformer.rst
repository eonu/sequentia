Function Transformer
====================

When preprocessing sequential data, it is often preferable to apply certain transformations
on each sequence independently rather than applying a single transformation to all of the data collectively.

For example, we might want to apply signal filters to each sequence independently.

:class:`.IndependentFunctionTransformer` allows for such transformations to be defined for arbitrary functions.

API reference
-------------

Class
^^^^^

.. autosummary::

   ~sequentia.preprocessing.transforms.IndependentFunctionTransformer

Methods
^^^^^^^

.. autosummary::

   ~sequentia.preprocessing.transforms.IndependentFunctionTransformer.__init__
   ~sequentia.preprocessing.transforms.IndependentFunctionTransformer.fit
   ~sequentia.preprocessing.transforms.IndependentFunctionTransformer.fit_transform
   ~sequentia.preprocessing.transforms.IndependentFunctionTransformer.inverse_transform
   ~sequentia.preprocessing.transforms.IndependentFunctionTransformer.transform

.. _definitions:

Definitions
^^^^^^^^^^^

.. autoclass:: sequentia.preprocessing.transforms.IndependentFunctionTransformer
   :members:
   :inherited-members:
   :exclude-members: get_params, set_params, get_feature_names_out, get_metadata_routing, set_fit_request, set_inverse_transform_request, set_output, set_transform_request
