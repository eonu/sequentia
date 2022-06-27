.. _custom:

Custom Transformations (``Custom``)
===================================

The :class:`~Custom` class allows you to specify your own transformations
that operate on a single observation sequence. This allows your own transformations
to be seamlessly combined with others provided by Sequentia, by using the :class:`~Compose` class.

API reference
-------------

.. autoclass:: sequentia.preprocessing.Custom
   :members: transform, __call__