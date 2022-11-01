Datasets
========

.. toctree::
    :titlesonly:

    digits

----

Sequentia provides a selection of sample sequential datasets for quick experimentation.

Each dataset follows the interface described below.

API reference
-------------

Class
^^^^^

.. autosummary::

    ~sequentia.utils.SequentialDataset

Methods
^^^^^^^

.. autosummary::

   ~sequentia.utils.SequentialDataset.__init__
   ~sequentia.utils.SequentialDataset.iter_by_class
   ~sequentia.utils.SequentialDataset.load
   ~sequentia.utils.SequentialDataset.save
   ~sequentia.utils.SequentialDataset.split

Properties
^^^^^^^^^^

.. autosummary::

   ~sequentia.utils.SequentialDataset.X
   ~sequentia.utils.SequentialDataset.X_lengths
   ~sequentia.utils.SequentialDataset.X_y
   ~sequentia.utils.SequentialDataset.X_y_lengths
   ~sequentia.utils.SequentialDataset.classes
   ~sequentia.utils.SequentialDataset.idxs
   ~sequentia.utils.SequentialDataset.lengths
   ~sequentia.utils.SequentialDataset.y

|

.. autoclass:: sequentia.utils.SequentialDataset
   :members:

