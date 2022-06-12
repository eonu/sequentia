.. Sequentia documentation master file, created by
   sphinx-quickstart on Sat Dec 28 19:22:34 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://i.ibb.co/42GkhfR/sequentia.png
    :alt: Sequentia
    :width: 275
    :target: https://github.com/eonu/sequentia
    :align: center

About
=====

Sequentia is a Python package that provides implementations of a selection of machine learning algorithms
for performing sequence classification. Examples of such sequences include:

- isolated word utterance frequencies in speech audio signals,
- isolated hand-written character pen-tip trajectories,
- isolated hand or head gestures positions in a video or motion-capture recording.

Most modern machine learning algorithms won't work directly out of the box when applied to such
sequential data – mostly due to the fact that the dependencies between observations at different
time frames must be considered, and also because each sequence generally has a different duration.

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

Documentation Search and Index
==============================

* :ref:`search`
* :ref:`genindex`