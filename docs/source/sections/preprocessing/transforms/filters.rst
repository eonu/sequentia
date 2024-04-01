.. _filters:

Filters
=======

Filters are a common preprocessing method for reducing noise in signal processing.

:func:`.mean_filter` and :func:`.median_filter` can be applied to individual sequences.

.. seealso::
    Consider using :class:`.IndependentFunctionTransformer` to apply these filters to multiple sequences.

API reference
-------------

Methods
^^^^^^^

.. autosummary::

   ~sequentia.preprocessing.transforms.mean_filter
   ~sequentia.preprocessing.transforms.median_filter

|

.. autofunction:: sequentia.preprocessing.transforms.mean_filter
.. autofunction:: sequentia.preprocessing.transforms.median_filter
