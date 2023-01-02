Function Transformer
====================

When preprocessing sequential data, it is often preferable to apply certain transformations
on each sequence independently rather than applying a single transformation to all of the data collectively.

For example in speech recognition, suppose we have a dataset of MFCC features extracted from audio recordings of different speakers.
If we are not interested in speaker-focused tasks such as speaker recognition, and instead only want to classify recordings,
we need to be able to compare recordings to each other — especially if using algorithms such as :class:`.KNNClassifier` which rely on distance comparisons.

In this case, we might want to standardize the MFCCs for each recording individually, (i.e. centering and scaling by separate feature means and standard deviations for each recording) so that they are represented as deviations from zero,
which is a form that is better suited for comparison as it reduces speaker-specific nuances in the data due to differences in scale or location.

Another example would be signal filters, which should be applied to each sequence independently.

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

|

.. autoclass:: sequentia.preprocessing.transforms.IndependentFunctionTransformer
   :members:
   :inherited-members:
   :exclude-members: get_params, set_params, set_output
