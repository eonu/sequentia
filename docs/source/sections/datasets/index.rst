Datasets
========

.. toctree::
    :titlesonly:

    digits
    gene_families

----

Sequentia provides a selection of sample sequential datasets for quick experimentation.

Each dataset follows the interface described below.

API reference
-------------

Class
^^^^^

.. autosummary::

    ~sequentia.datasets.base.SequentialDataset

Methods
^^^^^^^

.. autosummary::

   ~sequentia.datasets.base.SequentialDataset.__init__
   ~sequentia.datasets.base.SequentialDataset.copy
   ~sequentia.datasets.base.SequentialDataset.iter_by_class
   ~sequentia.datasets.base.SequentialDataset.load
   ~sequentia.datasets.base.SequentialDataset.save
   ~sequentia.datasets.base.SequentialDataset.split

Properties
^^^^^^^^^^

.. autosummary::

   ~sequentia.datasets.base.SequentialDataset.X
   ~sequentia.datasets.base.SequentialDataset.X_lengths
   ~sequentia.datasets.base.SequentialDataset.X_y
   ~sequentia.datasets.base.SequentialDataset.X_y_lengths
   ~sequentia.datasets.base.SequentialDataset.classes
   ~sequentia.datasets.base.SequentialDataset.idxs
   ~sequentia.datasets.base.SequentialDataset.lengths
   ~sequentia.datasets.base.SequentialDataset.y

|

.. autoclass:: sequentia.datasets.base.SequentialDataset
   :members:

