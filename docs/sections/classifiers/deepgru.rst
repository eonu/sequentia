.. _deepgru:

DeepGRU: Gesture Recognition Utility (``DeepGRU``)
==================================================

DeepGRU is a neural network architecture created by Mehran Maghoumi and Joseph J. LaViola Jr,
originally designed to perform the task of gesture recognition, but is widely applicable to
general sequence classification tasks.

The architecture is essentially a recurrent neural network encoder combined with an attentional module
which learns to place more focus on sub-sequences which are more important for the classification.

Rather than the commonly used long short-term memory (LSTM) unit, the authors opt for the
gated recurrent unit (GRU), which has fewer parameters, and therefore makes the network faster to train.
Interestingly, the encoder network used in DeepGRU is not bidirectional, which is typically the standard
way to use recurrent neural networks in sequence classification and sequence-to-sequence modelling. The authors
found that a unidirectional one was sufficient, faster to train and had similar performance to a bidirectional one.

.. image:: /_static/deepgru.png
    :alt: DeepGRU
    :width: 80%
    :align: center

The :class:`DeepGRU` class is a PyTorch implementation of the DeepGRU architecture.

A utility function :meth:`collate_fn` is also provided, which is passed to a :class:`torch:torch.utils.data.DataLoader`
and specifies how batches should be formed from provided observation sequences.

.. note::

    The existing preprocessing methods in :py:mod:`sequentia.preprocessing` are currently only
    applicable to lists of :class:`numpy:numpy.ndarray` objects, and therefore cannot be applied
    as transformations for :class:`torch:torch.Tensor` objects.

    Unfortunately this means that the preprocessing methods can only be used to preprocess data for
    :class:`sequentia.classifiers.knn.KNNClassifier` and :class:`sequentia.classifiers.hmm.HMMClassifier`,
    and not :class:`sequentia.classifiers.rnn.DeepGRU`.

API Reference
-------------

.. autoclass:: sequentia.classifiers.rnn.DeepGRU
    :members:

Batching and collation
----------------------

.. autofunction:: sequentia.classifiers.rnn.collate_fn