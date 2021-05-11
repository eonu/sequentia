.. _combined:

Combined Preprocessing (``Compose``)
=======================================

The :class:`~Compose` class provides a way of efficiently applying multiple
preprocessing transformations to provided input observation sequences.

For further information, please see the `preprocessing tutorial notebook <https://nbviewer.jupyter.org/github/eonu/sequentia/blob/master/notebooks/2%20-%20Preprocessing%20%28Tutorial%29.ipynb#Combining-preprocessing-methods>`_.

API reference
-------------

.. autoclass:: sequentia.preprocessing.Compose
    :members: __call__, fit, fit_transform, summary