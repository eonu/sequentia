.. _trim_zeros:

Zero-trimming (``trim_zeros``)
==============================

Removes zero-observations from an observation sequence.

Many datasets consisting of sequential data often pad observation sequences with zeros
in order to ensure that the machine learning algorithms receive sequences of equal length.
Although this comes with the advantage of being able to represent the sequences in a matrix,
the added zeros may affect the performance of the machine learning algorithms.

As the algorithms implemented by Sequentia focus on supporting variable-length
sequences out of the box, zero padding is not necessary, and can be removed with this method.

**NOTE**: This preprocessing method does not only remove trailing zeros from the start
or end of a sequence, but will also remove **any** zero-observations that occur anywhere
in the sequence.

For further information, please see the `preprocessing tutorial notebook <https://nbviewer.jupyter.org/github/eonu/sequentia/blob/master/notebooks/2%20-%20Preprocessing%20%28Tutorial%29.ipynb#Zero-trimming-(trim_zeros)>`_.

Example
-------

.. literalinclude:: ../../_includes/examples/preprocessing/trim_zeros.py
    :language: python
    :linenos:

API reference
-------------

.. automodule:: sequentia.preprocessing
   :noindex:
.. autofunction:: trim_zeros