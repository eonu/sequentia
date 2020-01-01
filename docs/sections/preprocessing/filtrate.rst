.. _filtrate:

Filtering (``filtrate``)
=============================

Filtering removes or reduces some unwanted components (such as noise) from an observation sequence
according to some window size and one of two methods: **mean** and **median** filtering.

Suppose we have an observation sequence :math:`\mathbf{o}^{(1)}\mathbf{o}^{(2)}\ldots\mathbf{o}^{(T)}`
and we are filtering with a window size of :math:`n`. Filtering replaces every observation :math:`\mathbf{o}^{(t)}`
with either the mean or median of the window of observations of size :math:`n` containing :math:`\mathbf{o}^{(t)}` in its centre.

- For median filtering: :math:`\mathbf{o}^{(t)\prime}=\mathrm{med}\underbrace{\left[\ldots, \mathbf{o}^{(t-1)}, \mathbf{o}^{(t)}, \mathbf{o}^{(t+1)}, \ldots\right]}_n`
- For mean filtering: :math:`\mathbf{o}^{(t)\prime}=\mathrm{mean}\underbrace{\left[\ldots, \mathbf{o}^{(t-1)}, \mathbf{o}^{(t)}, \mathbf{o}^{(t+1)}, \ldots\right]}_n`

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