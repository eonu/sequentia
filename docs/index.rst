.. Sequentia documentation master file, created by
   sphinx-quickstart on Sat Dec 28 19:22:34 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://i.ibb.co/42GkhfR/sequentia.png
    :alt: Sequentia
    :width: 275
    :target: https://github.com/eonu/sequentia

About
=====

Sequentia is a collection of machine learning algorithms for performing the classification of isolated sequences.

Each isolated sequence is generally modeled as a section of a longer multivariate time series
that represents the entire sequence. Naturally, this fits the description of many types of problems such as:

- isolated word utterance frequencies in speech audio signals,
- isolated hand-written character pen-tip trajectories,
- isolated hand or head gestures positions in a video or motion-capture recording.

Most modern machine learning algorithms won't work directly out of the box when applied to such
sequential data – mostly due to the fact that the dependencies between observations at different
time frames must be considered, and also because each isolated sequence generally has a different duration.

Sequentia offers some appropriate classification algorithms for these kinds of tasks.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Sequentia

   self
   changelog.rst

.. toctree::
   :maxdepth: 1
   :caption: Classifiers and Models

   sections/classifiers/hmm.rst
   sections/classifiers/knn.rst

.. toctree::
   :maxdepth: 1
   :caption: Preprocessing Methods

   sections/preprocessing/introduction.rst
   sections/preprocessing/equalize.rst
   sections/preprocessing/trim_zeros.rst
   sections/preprocessing/min_max_scale.rst
   sections/preprocessing/center.rst
   sections/preprocessing/standardize.rst
   sections/preprocessing/downsample.rst
   sections/preprocessing/filter.rst
   sections/preprocessing/preprocessing.rst

Documentation Search and Index
==============================

* :ref:`search`
* :ref:`genindex`