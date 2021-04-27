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

.. note::

    The existing preprocessing methods in :py:mod:`sequentia.preprocessing` are currently only
    applicable to lists of :class:`numpy:numpy.ndarray` objects, and therefore cannot be applied
    as transformations for :class:`torch:torch.Tensor` objects.

    Unfortunately this means that the preprocessing methods can only be used to preprocess data for
    :class:`sequentia.classifiers.knn.KNNClassifier` and :class:`sequentia.classifiers.hmm.HMMClassifier`,
    and not :class:`sequentia.classifiers.rnn.DeepGRU`.

Each of the transformations follow a similar interface, based on the abstract :class:`Transform` class:

.. autoclass:: sequentia.preprocessing.Transform
    :members: fit, fit_transform, transform, __call__