.. _normalize:

Normalization (``normalize``)
=============================

Normalizing centers an observation sequence about the mean of its observations – that is, given:

.. math::

    O=\begin{pmatrix}
        o_1^{(1)} & o_2^{(1)} & \cdots & o_D^{(1)} \\
        o_1^{(2)} & o_2^{(2)} & \cdots & o_D^{(2)} \\
        \vdots    & \vdots    & \ddots & \vdots    \\
        o_1^{(T)} & o_2^{(T)} & \cdots & o_D^{(T)}
    \end{pmatrix}
    \qquad
    \boldsymbol{\mu}=\begin{pmatrix}
        \overline{o_1} & \overline{o_2} & \cdots & \overline{o_D}
    \end{pmatrix}

Where :math:`\overline{o_d}` represents the mean of the :math:`d^\text{th}` feature of :math:`O`.

We subtract :math:`\boldsymbol{\mu}` from each observation, or row in :math:`O`. This centers the observations.

For further information, please see the `preprocessing tutorial notebook <https://nbviewer.jupyter.org/github/eonu/sequentia/blob/master/notebooks/2%20-%20Preprocessing%20%28Tutorial%29.ipynb#Normalization-(normalize)>`_.

Example
-------

.. literalinclude:: ../../_includes/examples/preprocessing/normalize.py
    :language: python
    :linenos:

API reference
-------------

.. automodule:: sequentia.preprocessing
.. autofunction:: normalize