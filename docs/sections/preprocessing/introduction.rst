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

    Unfortunately this means that the preprocessing methods can only be used to preprocess data for
    :class:`sequentia.classifiers.knn.KNNClassifier` and :class:`sequentia.classifiers.hmm.HMMClassifier`,
    and not :class:`sequentia.classifiers.rnn.DeepGRU`.

    It is possible to attempt to use these transformations on :class:`torch:torch.Tensor` objects by
    bypassing validation when applying the transformation,

    .. code-block:: python

        x = torch.rand(5, 3)
        x = Center()(x, validate=False)

    but this likely will not work due to differences in :class:`numpy:numpy.ndarray` and :class:`torch:torch.Tensor`.

Each of the transformations follow a similar interface, based on the abstract :class:`Transform` class:

.. autoclass:: sequentia.preprocessing.Transform
    :memberS: fit, is_fitted, unfit, fit_transform, transform, __call__