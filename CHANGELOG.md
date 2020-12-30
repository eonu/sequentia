## [0.10.0](https://github.com/eonu/sequentia/releases/tag/v0.10.0)

#### Major changes

- Switch out [`pomegranate`](https://github.com/jmschrei/pomegranate) HMM backend to [`hmmlearn`](https://github.com/hmmlearn/hmmlearn). ([#105](https://github.com/eonu/sequentia/pull/105))
- Remove separate HMM and GMM-HMM implementations – only keep a single GMM-HMM implementation (in the `GMMHMM` class) and treat multivariate Gaussian emission HMM as a special case of GMM-HMM. ([#105](https://github.com/eonu/sequentia/pull/105))
- Support string and numeric labels by using label encodings (from [`sklearn.preprocessing.LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)). ([#105](https://github.com/eonu/sequentia/pull/105))
- Add support for Python v3.6, v3.7, v3.8, v3.9 and remove support for v3.5. ([#105](https://github.com/eonu/sequentia/pull/105))
- Switch from approximate DTW algorithm ([`fastdtw`](https://github.com/slaypni/fastdtw)) to exact implementation ([`dtaidistance`](https://github.com/wannesm/dtaidistance)) for `KNNClassifier`. ([#106](https://github.com/eonu/sequentia/pull/106))

#### Minor changes

- Switch to use duck-typing for iterables instead of requiring lists. ([#105](https://github.com/eonu/sequentia/pull/105))
- Rename 'strict left-right' HMM topology to 'linear'. ([#105](https://github.com/eonu/sequentia/pull/105))
- Switch `m2r` to `m2r2`, as `m2r` is no longer maintained. ([#105](https://github.com/eonu/sequentia/pull/105))
- Change `covariance` to `covariance_type`, to match `hmmlearn`. ([#105](https://github.com/eonu/sequentia/pull/105))
- Use `numpy.random.RandomState(seed=None)` as default instead of `numpy.random.RandomState(seed=0)`. ([#105](https://github.com/eonu/sequentia/pull/105))
- Switch `KNNClassifier` serialization from HDF5 to pickling. ([#106](https://github.com/eonu/sequentia/pull/106))
- Use [`intersphinx`](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html) for external documentation links, e.g. to `numpy`. ([#108](https://github.com/eonu/sequentia/pull/108))
- Change `MinMaxScale` bounds to floats. ([#112](https://github.com/eonu/sequentia/pull/112))
- Add `__repr__` function to `GMMHMM`, `HMMClassifier` and `KNNClassifier`. ([#120](https://github.com/eonu/sequentia/pull/120))
- Use feature-independent warping (DTWI). ([#121](https://github.com/eonu/sequentia/pull/121))
- Ensure minimum Sakoe-Chiba band width is 1. ([#126](https://github.com/eonu/sequentia/pull/126))

## [0.7.2](https://github.com/eonu/sequentia/releases/tag/v0.7.2)

#### Major changes

- Stop referring to sequences as temporal, as non-temporal sequences can also be used. ([#103](https://github.com/eonu/sequentia/pull/103))

## [0.7.1](https://github.com/eonu/sequentia/releases/tag/v0.7.1)

#### Major changes

- Fix deserialization for `KNNClassifier`. ([#93](https://github.com/eonu/sequentia/pull/93))
  - Sort HDF5 keys before loading as `numpy.ndarray`s.
  - Pass `weighting` function into deserialization constructor.

## [0.7.0](https://github.com/eonu/sequentia/releases/tag/v0.7.0)

#### Major changes

- Fix `pomegranate` version to v0.12.0. ([#79](https://github.com/eonu/sequentia/pull/79))
- Add serialization and deserialization support for all classifiers. ([#80](https://github.com/eonu/sequentia/pull/80))
  - `HMM`, `HMMClassifier`: Serialized in JSON format.
  - `KNNClassifier`: Serialized in [HDF5](https://support.hdfgroup.org/HDF5/doc/H5.intro.html) format.
- Finish preprocessing documentation and tests. ([#81](https://github.com/eonu/sequentia/pull/81))
- (_Internal_) Remove nested helper functions in `KNNClassifier.predict()`. ([#84](https://github.com/eonu/sequentia/pull/84))
- Add strict left-right HMM topology. ([#85](https://github.com/eonu/sequentia/pull/85))<br/>**Note**: This is the more traditional left-right HMM topology.
- Implement GMM-HMMs in the `GMMHMM` class. ([#87](https://github.com/eonu/sequentia/pull/87))
- Implement custom, uniform and frequency-based HMM priors. ([#88](https://github.com/eonu/sequentia/pull/88))
- Implement distance-weighted DTW-kNN predictions. ([#90](https://github.com/eonu/sequentia/pull/90))
- Rename `DTWKNN` to `KNNClassifer`. ([#91](https://github.com/eonu/sequentia/pull/91))

#### Minor changes
- (_Internal_) Simplify package imports. ([#82](https://github.com/eonu/sequentia/pull/82))
- (_Internal_) Add `Validator.func()` for validating callables. ([#90](https://github.com/eonu/sequentia/pull/90))

## [v0.7.0a1](https://github.com/eonu/sequentia/releases/tag/v0.7.0a1)

#### Major changes
- Clean up package imports. ([#77](https://github.com/eonu/sequentia/pull/77))
- Rework `preprocessing` module. ([#75](https://github.com/eonu/sequentia/pull/75))

#### Minor changes
- Fix typos and update preprocessing information in `README.md`. ([#76](https://github.com/eonu/sequentia/pull/76))

## [0.6.1](https://github.com/eonu/sequentia/releases/tag/v0.6.1)

#### Major changes
- Remove strict requirement of Numpy arrays being two-dimensional by using `numpy.atleast_2d` to convert one-dimensional arrays into 2D. ([#70](https://github.com/eonu/sequentia/pull/70))

#### Minor changes
- As the HMM classifier is not a true ensemble of HMMs (since each HMM doesn't really contribute to the classification), it is no longer referred to as an ensemble. ([#69](https://github.com/eonu/sequentia/pull/69))

## [0.6.0](https://github.com/eonu/sequentia/releases/tag/v0.6.0)

#### Major changes
- Add package tests and Travis CI support. ([#56](https://github.com/eonu/sequentia/pull/56))
- Remove Python v3.8+ support. ([#56](https://github.com/eonu/sequentia/pull/56))
- Rename `normalize` preprocessing method to `center`, since it just centers an observation sequence. ([#62](https://github.com/eonu/sequentia/pull/62))
- Add `standardize` preprocessing method for standardizing (standard scaling) an observation sequence. ([#63](https://github.com/eonu/sequentia/pull/63))
- Add `trim_zeros` preprocessing method for removing zero-observations from an observation sequence. ([#67](https://github.com/eonu/sequentia/pull/67))

#### Minor changes
- (_Internal_) Add `Validator.random_state` for validating random state objects and seeds. ([#56](https://github.com/eonu/sequentia/pull/56))
- (_Internal_) Internalize `Validator` and topology (`Topology`, `ErgodicTopology`, `LeftRightTopology`) classes. ([#57](https://github.com/eonu/sequentia/pull/57))
- (_Internal_) Use proper documentation format for topology classes. ([#58](https://github.com/eonu/sequentia/pull/58))

## [0.5.0](https://github.com/eonu/sequentia/releases/tag/v0.5.0)

#### Major changes
- Add `Preprocess.summary()` to display an ordered summary of preprocessing transformations. ([#54](https://github.com/eonu/sequentia/pull/54))
- Add mean and median filtering preprocessing methods. ([#48](https://github.com/eonu/sequentia/pull/48))
- Use median filtering and decimation downsampling by default. ([#52](https://github.com/eonu/sequentia/pull/52))
- Modify preprocessing boundary conditions ([#51](https://github.com/eonu/sequentia/pull/51)):
  - Use a bi-directional window for filtering to resolve boundary problems.
  - Modify downsampling method to downsample residual observations.

#### Minor changes

- Add supported topologies (left-right and ergodic) to feature list. ([#53](https://github.com/eonu/sequentia/pull/53))
- Add restrictions on preprocessing parameters: downsample factor and window size. ([#50](https://github.com/eonu/sequentia/pull/50))
- Allow `Preprocess` class to be used to apply preprocessing transformations to a single observation sequence. ([#49](https://github.com/eonu/sequentia/pull/49))

## [0.4.0](https://github.com/eonu/sequentia/releases/tag/v0.4.0)

#### Major changes
- Re-add `euclidean` metric as `DTWKNN` default. ([#43](https://github.com/eonu/sequentia/pull/43))

#### Minor changes
- Add explicit labels to `evaluate()` in `HMMClassifier` example. ([#44](https://github.com/eonu/sequentia/pull/44))

## [0.3.0](https://github.com/eonu/sequentia/releases/tag/v0.3.0)

#### Major changes
- Add proper documentation, hosted on [Read The Docs](https://sequentia.readthedocs.io/en/latest). ([#40](https://github.com/eonu/sequentia/pull/40), [#41](https://github.com/eonu/sequentia/pull/41))

## [0.2.0](https://github.com/eonu/sequentia/releases/tag/v0.2.0)

#### Major changes
- Add multi-processing support for `DTWKNN` predictions. ([#29](https://github.com/eonu/sequentia/pull/29))
- Rename the `fit_transform()` function in `Preprocess` to `transform()` since there is nothing being fitted. ([#35](https://github.com/eonu/sequentia/pull/35))
- (_Internal_) Modify package classifiers in `setup.py` ([#31](https://github.com/eonu/sequentia/pull/31)):
  - Set development status classifier to `Pre-Alpha`.
  - Add Python version classifiers for v3.5+.
  - Specify UNIX and macOS operating system classifiers.

#### Minor changes
- (_Internal_) Finish tutorial and example notebooks. ([#35](https://github.com/eonu/sequentia/pull/35))
- (_Internal_) Rename `examples` directory to `notebooks`. ([#32](https://github.com/eonu/sequentia/pull/32))
- (_Internal_) Host notebooks statically on [nbviewer](https://github.com/jupyter/nbviewer). ([#32](https://github.com/eonu/sequentia/pull/32))
- (_Internal_) Add reference to Pomegranate [paper](http://jmlr.org/papers/volume18/17-636/17-636.pdf) and [repository](https://github.com/jmschrei/pomegranate). ([#30](https://github.com/eonu/sequentia/pull/30))
- (_Internal_) Add badges to `README.md`. ([#28](https://github.com/eonu/sequentia/pull/28))

## [0.1.0](https://github.com/eonu/sequentia/releases/tag/v0.1.0)

#### Major changes
Nothing, initial release!