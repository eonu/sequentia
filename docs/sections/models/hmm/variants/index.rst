HMM Variants
============

.. toctree::
    :maxdepth: 2

    gaussian_mixture
    multinomial

----

Sequentia provides two types of HMMs that can be used with a HMM Classifier.

- | **Gaussian Mixture HMM**: 
  | For modelling univariate or multivariate numerical sequences with real-valued observations.
  | e.g. MFCCs, sEMG/ECG/EEG signals, position/rotation signals, word embeddings.
- | **Multinomial HMM**: 
  | For modelling univariate categorical sequences with observations from a finite set of symbols.
  | e.g. DNA sequences, part-of-speech tags.