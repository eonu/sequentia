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
- Modify package classifiers in `setup.py` ([#31](https://github.com/eonu/sequentia/pull/31)):
  - Set development status classifier to `Pre-Alpha`.
  - Add Python version classifiers for v3.5+.
  - Specify UNIX and macOS operating system classifiers.

#### Minor changes
- Finish tutorial and example notebooks. ([#35](https://github.com/eonu/sequentia/pull/35))
- Rename `examples` directory to `notebooks`. ([#32](https://github.com/eonu/sequentia/pull/32))
- Host notebooks statically on [nbviewer](https://github.com/jupyter/nbviewer). ([#32](https://github.com/eonu/sequentia/pull/32))
- Add reference to Pomegranate [paper](http://jmlr.org/papers/volume18/17-636/17-636.pdf) and [repository](https://github.com/jmschrei/pomegranate). ([#30](https://github.com/eonu/sequentia/pull/30))
- Add badges to `README.md`. ([#28](https://github.com/eonu/sequentia/pull/28))

## [0.1.0](https://github.com/eonu/sequentia/releases/tag/v0.1.0)

#### Major changes

Nothing, initial release!