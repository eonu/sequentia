<p align="center">
  <h1 align="center">
    <img src="https://raw.githubusercontent.com/eonu/sequentia/master/docs/_static/images/logo.png" width="75px"><br/>
    Sequentia
  </h1>
</p>

<p align="center">
  <em>HMM and DTW-based sequence machine learning algorithms in Python following an sklearn-like interface.</em>
</p>

<p align="center">
  <div align="center">
    <a href="https://pypi.org/project/sequentia">
      <img src="https://img.shields.io/pypi/v/sequentia?logo=pypi&style=flat-square" alt="PyPI"/>
    </a>
    <a href="https://pypi.org/project/sequentia">
      <img src="https://img.shields.io/pypi/pyversions/sequentia?logo=python&style=flat-square" alt="PyPI - Python Version"/>
    </a>
    <a href="https://sequentia.readthedocs.io/en/latest">
      <img src="https://img.shields.io/readthedocs/sequentia.svg?logo=read-the-docs&style=flat-square" alt="Read The Docs - Documentation">
    </a>
    <a href="https://raw.githubusercontent.com/eonu/sequentia/master/LICENSE">
      <img src="https://img.shields.io/pypi/l/sequentia?style=flat-square" alt="PyPI - License"/>
    </a>
  </div>
</p>

<p align="center">
  <sup>
    <a href="#about">About</a> ·
    <a href="#build-status">Build Status</a> ·
    <a href="#features">Features</a> ·
    <a href="#documentation">Documentation</a> ·
    <a href="#examples">Examples</a> ·
    <a href="#acknowledgments">Acknowledgments</a> ·
    <a href="#references">References</a> ·
    <a href="#contributors">Contributors</a> ·
    <a href="#licensing">Licensing</a>
  </sup>
</p>

## About

Sequentia is a Python package that provides various classification and regression algorithms for sequential data, including methods based on hidden Markov models and dynamic time warping.

Some examples of how Sequentia can be used on sequence data include:

- determining a spoken word based on its audio signal or alternative representations such as MFCCs,
- predicting motion intent for gesture control from sEMG signals,
- classifying hand-written characters according to their pen-tip trajectories.

## Build Status

| `master` | `dev` |
| -------- | ------|
| [![CircleCI Build (Master)](https://img.shields.io/circleci/build/github/eonu/sequentia/master?logo=circleci&style=flat-square)](https://app.circleci.com/pipelines/github/eonu/sequentia?branch=master) | [![CircleCI Build (Development)](https://img.shields.io/circleci/build/github/eonu/sequentia/dev?logo=circleci&style=flat-square)](https://app.circleci.com/pipelines/github/eonu/sequentia?branch=master) |

## Features

### Models

The following models provided by Sequentia all support variable length sequences.

#### [Dynamic Time Warping + k-Nearest Neighbors](https://sequentia.readthedocs.io/en/latest/sections/models/knn/index.html) (via [`dtaidistance`](https://github.com/wannesm/dtaidistance))

- [x] Classification
- [x] Regression
- [x] Multivariate real-valued observations
- [x] Sakoe–Chiba band global warping constraint
- [x] Dependent and independent feature warping (DTWD/DTWI)
- [x] Custom distance-weighted predictions
- [x] Multi-processed predictions

#### [Hidden Markov Models](https://sequentia.readthedocs.io/en/latest/sections/models/hmm/index.html) (via [`hmmlearn`](https://github.com/hmmlearn/hmmlearn))

Parameter estimation with the Baum-Welch algorithm and prediction with the forward algorithm [[1]](#references)

- [x] Classification
- [x] Multivariate real-valued observations (Gaussian mixture model emissions)
- [x] Univariate categorical observations (discrete emissions)
- [x] Linear, left-right and ergodic topologies
- [x] Multi-processed predictions

### Scikit-Learn compatibility

Sequentia aims to follow the Scikit-Learn interface for estimators and transformations,
as well as to be largely compatible with three core Scikit-Learn modules to improve the ease of model development:
[`preprocessing`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing), [`model_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) and [`pipeline`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline).

While there are many other modules, maintaining full compatibility with Scikit-Learn is challenging and many of its features are inapplicable to sequential data, therefore we only focus on the relevant core modules.

Despite some deviation from the Scikit-Learn interface in order to accommodate sequences, the following features are currently compatible with Sequentia.

- [x] [`preprocessing`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
  - [x] [`FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer) — via an adapted class definition
  - [x] Function-based transformations (stateless)
  - [x] Class-based transformations (stateful)
- [ ] [`pipeline`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline)
  - [x] [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline) — via an adapted class definition
  - [ ] [`FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion)
- [ ] [`model_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)

## Installation

You can install Sequentia using `pip`.

### Stable

The latest stable version of Sequentia can be installed with the following command.

```console
pip install sequentia
```

#### C library compilation

For optimal performance when using any of the k-NN based models, it is important that `dtaidistance` C libraries are compiled correctly.

Please see the [`dtaidistance` installation guide](https://dtaidistance.readthedocs.io/en/latest/usage/installation.html) for troubleshooting if you run into C compilation issues, or if setting `use_c=True` on k-NN based models results in a warning.

You can use the following to check if the appropriate C libraries have been installed.

```python
from dtaidistance import dtw
dtw.try_import_c()
```

### Pre-release

Pre-release versions include new features which are in active development and may change unpredictably.

The latest pre-release version can be installed with the following command.

```console
pip install --pre sequentia
```

### Development

Please see the [contribution guidelines](/CONTRIBUTING.md) to see installation instructions for contributing to Sequentia.

## Documentation

Documentation for the package is available on [Read The Docs](https://sequentia.readthedocs.io/en/latest).

## Examples

Demonstration of classifying multivariate sequences with two features into two classes using the `KNNClassifier`.

This example also shows a typical preprocessing workflow, as well as compatibility with Scikit-Learn.

```python
import numpy as np

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

from sequentia.models import KNNClassifier
from sequentia.pipeline import Pipeline
from sequentia.preprocessing import IndependentFunctionTransformer, mean_filter

# Create input data
# - Sequentia expects sequences to be concatenated into a single array
# - Sequence lengths are provided separately and used to decode the sequences when needed
# - This avoids the need for complex structures such as lists of arrays with different lengths

# Sequences
X = np.array([
    # Sequence 1 - Length 3
    [1.2 , 7.91],
    [1.34, 6.6 ],
    [0.92, 8.08],
    # Sequence 2 - Length 5
    [2.11, 6.97],
    [1.83, 7.06],
    [1.54, 5.98],
    [0.86, 6.37],
    [1.21, 5.8 ],
    # Sequence 3 - Length 2
    [1.7 , 6.22],
    [2.01, 5.49]
])

# Sequence lengths
lengths = np.array([3, 5, 2])

# Sequence classes
y = np.array([0, 1, 1])

# Create a transformation pipeline that feeds into a KNNClassifier
# 1. Individually denoise each sequence by applying a mean filter for each feature
# 2. Individually standardize each sequence by subtracting the mean and dividing the s.d. for each feature
# 3. Reduce the dimensionality of the data to a single feature by using PCA
# 4. Pass the resulting transformed data into a KNNClassifier
pipeline = Pipeline([
    ('denoise', IndependentFunctionTransformer(mean_filter)),
    ('scale', IndependentFunctionTransformer(scale)),
    ('pca', PCA(n_components=1)),
    ('knn', KNNClassifier(k=1))
])

# Fit the pipeline to the data - lengths must be provided
pipeline.fit(X, y, lengths)

# Predict classes for the sequences and calculate accuracy - lengths must be provided
y_pred = pipeline.predict(X, lengths)
acc = pipeline.score(X, y, lengths)
```

## Acknowledgments

In earlier versions of the package, an approximate DTW implementation [`fastdtw`](https://github.com/slaypni/fastdtw) was used in hopes of speeding up k-NN predictions, as the authors of the original FastDTW paper [[2]](#references) claim that approximated DTW alignments can be computed in linear memory and time, compared to the O(N<sup>2</sup>) runtime complexity of the usual exact DTW implementation.

I was contacted by [Prof. Eamonn Keogh](https://www.cs.ucr.edu/~eamonn/) whose work makes the surprising revelation that FastDTW is generally slower than the exact DTW algorithm that it approximates [[3]](#references). Upon switching from the `fastdtw` package to [`dtaidistance`](https://github.com/wannesm/dtaidistance) (a very solid implementation of exact DTW with fast pure C compiled functions), DTW k-NN prediction times were indeed reduced drastically.

I would like to thank Prof. Eamonn Keogh for directly reaching out to me regarding this finding.

## References

<table>
  <tbody>
    <tr>
      <td>[1]</td>
      <td>
        <a href=https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf">Lawrence R. Rabiner. <b>"A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"</b> <em>Proceedings of the IEEE 77 (1989)</em>, no. 2, 257-86.</a>
      </td>
    </tr>
    <tr>
      <td>[2]</td>
      <td>
        <a href="https://pdfs.semanticscholar.org/05a2/0cde15e172fc82f32774dd0cf4fe5827cad2.pdf">Stan Salvador & Philip Chan. <b>"FastDTW: Toward accurate dynamic time warping in linear time and space."</b> <em>Intelligent Data Analysis 11.5 (2007)</em>, 561-580.</a>
      </td>
    </tr>
    <tr>
      <td>[3]</td>
      <td>
        <a href="https://arxiv.org/ftp/arxiv/papers/2003/2003.11246.pdf">Renjie Wu & Eamonn J. Keogh. <b>"FastDTW is approximate and Generally Slower than the Algorithm it Approximates"</b> <em>IEEE Transactions on Knowledge and Data Engineering (2020)</em>, 1–1.</a>
      </td>
    </tr>
  </tbody>
</table>

## Contributors

All contributions to this repository are greatly appreciated. Contribution guidelines can be found [here](/CONTRIBUTING.md).

<table>
	<thead>
		<tr>
			<th align="center">
        <a href="https://github.com/eonu">
          <img src="https://avatars0.githubusercontent.com/u/24795571?s=460&v=4" alt="eonu" width="60px">
          <br/><sub><b>eonu</b></sub>
        </a>
			</th>
      <th align="center">
        <a href="https://github.com/Prhmma">
          <img src="https://avatars0.githubusercontent.com/u/16954887?s=460&v=4" alt="Prhmma" width="60px">
          <br/><sub><b>Prhmma</b></sub>
        </a>
			</th>
      <th align="center">
        <a href="https://github.com/manisci">
          <img src="https://avatars.githubusercontent.com/u/30268711?v=4" alt="manisci" width="60px">
          <br/><sub><b>manisci</b></sub>
        </a>
      </th>
      <th align="center">
        <a href="https://github.com/jonnor">
          <img src="https://avatars.githubusercontent.com/u/45185?v=4" alt="jonnor" width="60px">
          <br/><sub><b>jonnor</b></sub>
        </a>
      </th>
			<!-- Add more <th></th> blocks for more contributors -->
		</tr>
	</thead>
</table>

## Licensing

Sequentia is released under the [MIT](https://opensource.org/licenses/MIT) license.

Certain parts of the source code are heavily adapted from [Scikit-Learn](scikit-learn.org/).
Such files contain copy of [their license](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).

---

<p align="center">
  <b>Sequentia</b> &copy; 2019-2023, Edwin Onuonga - Released under the <a href="https://opensource.org/licenses/MIT">MIT</a> license.<br/>
  <em>Authored and maintained by Edwin Onuonga.</em>
</p>