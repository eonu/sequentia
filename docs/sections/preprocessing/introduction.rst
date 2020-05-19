.. _preprocessing-introduction:

Introduction to Preprocessing
=============================

Sequentia provides a number of useful preprocessing methods for sequential data.

- :doc:`Length Equalizing <equalize>` (``Equalize``)
- :doc:`Zero Trimming <trim_zeros>` (``TrimZeros``)
- :doc:`Min-max Scaling <min_max_scale>` (``MinMaxScale``)
- :doc:`Centering <center>` (``Center``)
- :doc:`Standardizing <standardize>` (``Standardize``)
- :doc:`Downsampling <downsample>` (``Downsample``)
- :doc:`Filtering <filter>` (``Filter``)

Additionally, the provided ``Preprocess`` class makes it possible to :doc:`apply multiple transformations <preprocessing>`.

Each of the transformations follow a similar interface, based on the abstract :class:`Transform` class:

.. autoclass:: sequentia.preprocessing.Transform
    :members: fit, fit_transform, transform, __call__