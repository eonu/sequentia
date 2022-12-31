Preprocessing
=============

.. toctree::
    :titlesonly:

    pipeline
    transforms/index

----

Sequentia provides an adapted version of the :mod:`sklearn.preprocessing` interface,
modified for sequential data support but also continuing to support most of the Scikit-Learn transformations out-of-the-box.

Transformations can be applied to all of the input sequences collectively — treated as a single array,
or on an individual basis by using the :class:`.IndependentFunctionTransformer`.

Transformation steps can be combined together with an estimator in a :class:`.Pipeline` which follows the Scikit-Learn interface.

Additional transformations specific to sequences are also provided, such as :ref:`filters <filters>` for signal data.
