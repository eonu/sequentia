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

.. note::

    The existing preprocessing methods in :py:mod:`sequentia.preprocessing` are currently only
    applicable to lists of :class:`numpy:numpy.ndarray` objects, and therefore cannot be applied
    as transformations for :class:`torch:torch.Tensor` objects.

    This means that if you wish to use :class:`sequentia.classifiers.rnn.DeepGRU`, you must first
    apply the transformations on :class:`numpy:numpy.ndarray` objects then transform them into :class:`torch:torch.Tensor` objects.

Each of the transformations follow a similar interface, based on the abstract :class:`Transform` class:

.. autoclass:: sequentia.preprocessing.Transform
    :members: fit, is_fitted, unfit, fit_transform, transform, __call__