.. _downsample:

Downsampling (``Downsample``)
=============================

Downsampling reduces the number of frames in an observation sequence according
to a specified downsample factor and one of two methods: **averaging** and **decimation**.

This is an especially helpful preprocessing method for speeding up classification times.

API reference
-------------

.. autoclass:: sequentia.preprocessing.Downsample
    :members: fit, fit_transform, transform, __call__