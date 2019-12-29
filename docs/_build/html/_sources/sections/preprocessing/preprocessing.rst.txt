.. _combined:

Combined Preprocessing (``Preprocess``)
=======================================

The :class:`~Preprocess` class provides a way of efficiently applying multiple
preprocessing transformations to provided input observation sequences.

For further information, please see the `preprocessing tutorial notebook <https://nbviewer.jupyter.org/github/eonu/sequentia/blob/master/notebooks/2%20-%20Preprocessing%20%28Tutorial%29.ipynb#Combining-preprocessing-methods-(Preprocess)>`_.

Example
-------

.. literalinclude:: ../../_includes/examples/preprocessing/preprocess.py
    :language: python
    :linenos:

API reference
-------------

.. autoclass:: sequentia.preprocessing.Preprocess
    :members: