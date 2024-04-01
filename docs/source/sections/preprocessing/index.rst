Preprocessing
=============

.. toctree::
    :titlesonly:

    transforms/index

----

Although :mod:`sklearn.preprocessing` is compatible with Sequentia, 
we also provide a lightweight preprocessing interface with additional features.

Transformations can be applied to all of the input sequences collectively — treated as a single array,
or on an individual basis by using the :class:`.IndependentFunctionTransformer`.

Additional transformations specific to sequences are also provided, such as :ref:`filters <filters>` for signal data.
