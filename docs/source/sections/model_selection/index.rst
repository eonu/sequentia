Model Selection
===============

.. toctree::
    :titlesonly:

    searching.rst
    splitting.rst

----

For validating models and performing hyper-parameter selection, it is common
to use cross-validation methods such as those in :mod:`sklearn.model_selection`.

Although :mod:`sklearn.model_selection` is partially compatible with Sequentia, 
we define our own wrapped versions of certain classes and functions to allow 
support for sequences.

- :ref:`searching` defines methods for searching hyper-parameter spaces in different ways, such as :class:`sequentia.model_selection.GridSearchCV`.
- :ref:`splitting` defines methods for partitioning data into training/validation splits for cross-validation, such as :class:`sequentia.model_selection.KFold`.
