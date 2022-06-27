.. _preprocessing-introduction:

Introduction to Preprocessing
=============================

Sequentia provides a number of useful preprocessing methods for sequential data.

- :doc:`Custom Transformations <custom>` (``Custom``)
- :doc:`Constant Trimming <trim_constants>` (``TrimConstants``)
- :doc:`Min-max Scaling <min_max_scale>` (``MinMaxScale``)
- :doc:`Centering <center>` (``Center``)
- :doc:`Standardizing <standardize>` (``Standardize``)
- :doc:`Downsampling <downsample>` (``Downsample``)
- :doc:`Filtering <filter>` (``Filter``)

Additionally, the provided ``Compose`` class makes it possible to :doc:`apply multiple transformations <compose>`.

Each of the transformations follow a similar interface, based on the abstract :class:`Transform` class:

.. autoclass:: sequentia.preprocessing.Transform
    :members: fit, is_fitted, unfit, fit_transform, transform, __call__