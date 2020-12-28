.. _trim_zeros:

Zero Trimming (``TrimZeros``)
=============================

Many datasets consisting of sequential data often pad observation sequences with zeros
in order to ensure that the machine learning algorithms receive sequences of equal length.
Although this comes with the advantage of being able to represent the sequences in a matrix,
the added zeros may affect the performance of the machine learning algorithms.

As the algorithms implemented by Sequentia focus on supporting variable-length
sequences out of the box, zero padding is not necessary, and can be removed with this method.

.. warning::
   This preprocessing method does not only remove trailing zeros from the start or end of a sequence,
   but will also remove **any** zero-observations that occur anywhere in the sequence.

API reference
-------------

.. autoclass:: sequentia.preprocessing.TrimZeros
   :members: fit, fit_transform, transform, __call__