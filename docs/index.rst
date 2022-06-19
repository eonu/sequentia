.. Sequentia documentation master file, created by
   sphinx-quickstart on Sat Dec 28 19:22:34 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: /_static/logo.png
    :alt: Sequentia
    :width: 125px
    :target: https://github.com/eonu/sequentia
    :align: center

About
=====

Sequentia is a package that provides various classification algorithms for sequential data, including classifiers based on hidden Markov models, dynamic time warping and recurrent neural networks.

Some examples of how Sequentia can be used in sequence classification include:

- determining a spoken word based on its audio signal or alternative representations such as MFCCs,
- identifying heart conditions such as arrhythmia from ECG signals,
- predicting motion intent for gesture control from sEMG signals,
- classifying hand-written characters according to their pen-tip trajectories.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Sequentia

   self
   changelog.rst

.. toctree::
   :maxdepth: 1
   :caption: Classifiers and Models

   sections/classifiers/knn.rst
   sections/classifiers/gmmhmm.rst
   sections/classifiers/deepgru.rst

.. toctree::
   :maxdepth: 1
   :caption: Preprocessing Methods

   sections/preprocessing/introduction.rst
   sections/preprocessing/custom.rst
   sections/preprocessing/trim_constants.rst
   sections/preprocessing/min_max_scale.rst
   sections/preprocessing/center.rst
   sections/preprocessing/standardize.rst
   sections/preprocessing/downsample.rst
   sections/preprocessing/filter.rst
   sections/preprocessing/compose.rst

.. toctree::
   :maxdepth: 1
   :caption: Datasets

   sections/datasets/introduction.rst
   sections/datasets/load_digits.rst
   sections/datasets/dataset.rst
   sections/datasets/torch_dataset.rst

Documentation Search and Index
==============================

* :ref:`search`
* :ref:`genindex`