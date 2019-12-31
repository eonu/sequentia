.. _filtrate:

Filtering (``filtrate``)
=============================

Filtering removes or reduces some unwanted components (such as noise) from an observation sequence
according to some window size and one of two methods: **mean** and **median** filtering.

Suppose we have an observation sequence :math:`\mathbf{o}^{(1)}\mathbf{o}^{(2)}\ldots\mathbf{o}^{(T)}`
and we are filtering with a window size of :math:`n`. Filtering replaces every observation :math:`\mathbf{o}^{(t)}`
with either the mean or median of the set of observations consisting of itself and the next :math:`n-1` observations.

- For median filtering: :math:`\mathbf{o}^{(t)\prime}=\mathrm{med}\left[\mathbf{o}^{(t)}, \mathbf{o}^{(t+1)}, \ldots, \mathbf{o}^{(t+n-1)}\right]`
- For mean filtering: :math:`\mathbf{o}^{(t)\prime}=\mathrm{mean}\left[\mathbf{o}^{(t)}, \mathbf{o}^{(t+1)}, \ldots, \mathbf{o}^{(t+n-1)}\right]=\frac{1}{n}\sum_{i=0}^{n-1}\mathbf{o}^{(t+i)}`

**Note**: Towards the end of the sequence, there may be some observations that don't have enough
remaining subsequent observations to form another window of size :math:`n`. These observations are removed
entirely from the filtered observation sequence – specifically, the last :math:`n-1` observations.

For further information, please see the `preprocessing tutorial notebook <https://nbviewer.jupyter.org/github/eonu/sequentia/blob/master/notebooks/2%20-%20Preprocessing%20%28Tutorial%29.ipynb#Filtering-(filtrate)>`_.

Example
-------

.. literalinclude:: ../../_includes/examples/preprocessing/filtrate.py
    :language: python
    :linenos:

API reference
-------------

.. automodule:: sequentia.preprocessing
   :noindex:
.. autofunction:: filtrate