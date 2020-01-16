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