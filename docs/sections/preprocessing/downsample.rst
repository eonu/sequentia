.. _downsample:

Downsampling (``downsample``)
=============================

Downsampling reduces the number of frames in an observation sequence according
to a specified downsample factor and one of two methods: **averaging** and **decimation**.

This is an especially helpful preprocessing method for speeding up classification times.

For further information, please see the `preprocessing tutorial notebook <https://nbviewer.jupyter.org/github/eonu/sequentia/blob/master/notebooks/2%20-%20Preprocessing%20%28Tutorial%29.ipynb#Downsampling-(downsample)>`_.

Example
-------

.. literalinclude:: ../../_includes/examples/preprocessing/downsample.py
    :language: python
    :linenos:

API reference
-------------

.. automodule:: sequentia.preprocessing
   :noindex:
.. autofunction:: downsample