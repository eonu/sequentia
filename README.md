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
    <a href="#contributors">Contributors</a>
  </sup>
</p>

## About

Sequentia is a Python package that provides various classification and regression algorithms for sequential data, including methods based on hidden Markov models and dynamic time warping.

Some examples of how Sequentia can be used on sequence data include:

- determining a spoken word based on its audio signal or alternative representations such as MFCCs,
- predicting motion intent for gesture control from sEMG signals,
- classifying hand-written characters according to their pen-tip trajectories,
- predicting the gene family that a DNA sequence belongs to.

## Build Status

| `master` | `dev` |
| -------- | ------|
| [![CircleCI Build (Master)](https://img.shields.io/circleci/build/github/eonu/sequentia/master?logo=circleci&style=flat-square)](https://app.circleci.com/pipelines/github/eonu/sequentia?branch=master) | [![CircleCI Build (Development)](https://img.shields.io/circleci/build/github/eonu/sequentia/dev?logo=circleci&style=flat-square)](https://app.circleci.com/pipelines/github/eonu/sequentia?branch=master) |

## Features

The following models provided by Sequentia all support variable length sequences.

- [x] [Dynamic Time Warping + k-Nearest Neighbors](https://sequentia.readthedocs.io/en/latest/sections/classifiers/knn.html) (via [`dtaidistance`](https://github.com/wannesm/dtaidistance))
  - [x] Classification
  - [x] Regression
  - [x] Multivariate real-valued observations
  - [x] Sakoe–Chiba band global warping constraint
  - [x] Dependent and independent feature warping (DTWD/DTWI)
  - [x] Custom distance-weighted predictions
  - [x] Multi-processed predictions
- [x] [Hidden Markov Models](https://sequentia.readthedocs.io/en/latest/sections/classifiers/gmmhmm.html) (via [`hmmlearn`](https://github.com/hmmlearn/hmmlearn))<br/><em>Parameter estimation with the Baum-Welch algorithm and prediction with the forward algorithm</em> [[1]](#references)
  - [x] Classification
  - [x] Multivariate real-valued observations (Gaussian mixture model emissions)
  - [x] Univariate categorical observations (discrete emissions)
  - [x] Linear, left-right and ergodic topologies
  - [x] Multi-processed predictions

  <p align="center">
    <img src="https://raw.githubusercontent.com/eonu/sequentia/master/docs/_static/images/classifier.png" width="80%"/><br/>
    HMM Sequence Classifier
  </p>

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

This example demonstrates multivariate sequences classified into classes `0`/`1` using the `KNNClassifier`.

```python
import numpy as np
from sequentia.models import KNNClassifier

# Generate training sequences and labels
X = [
  np.array([[1., 0., 5., 3., 7., 2., 2., 4., 9., 8., 7.],
            [3., 8., 4., 0., 7., 1., 1., 3., 4., 2., 9.]]).T,
  np.array([[2., 1., 4., 6., 5., 8.],
            [5., 3., 9., 0., 8., 2.]]).T,
  np.array([[5., 8., 0., 3., 1., 0., 2., 7., 9.],
            [0., 2., 7., 1., 2., 9., 5., 8., 1.]]).T
]
y = [0, 1, 1]

# Sequentia expects a concatenated array of sequences (and their corresponding lengths)
X, lengths = np.vstack(X), [len(x) for x in X]

# Create and fit the classifier
clf = KNNClassifier(k=1).fit(X, y, lengths)

# Make a prediction for a new observation sequence
x_new = np.array([[0., 3., 2., 7., 9., 1., 1.],
                  [2., 5., 7., 4., 2., 0., 8.]]).T
y_new = clf.predict(x_new)
```

## Acknowledgments

In earlier versions of the package, an approximate DTW implementation [`fastdtw`](https://github.com/slaypni/fastdtw) was used in hopes of speeding up k-NN predictions, as the authors of the original FastDTW paper [[2]](#references) claim that approximated DTW alignments can be computed in linear memory and time, compared to the O(N<sup>2</sup>) runtime complexity of the usual exact DTW implementation.

I was contacted by [Prof. Eamonn Keogh](https://www.cs.ucr.edu/~eamonn/) whose work [[3]](#references) makes the surprising revelation that FastDTW is generally slower than the exact DTW algorithm that it approximates. Upon switching from the `fastdtw` package to [`dtaidistance`](https://github.com/wannesm/dtaidistance) (a very solid implementation of exact DTW with fast pure C compiled functions), DTW k-NN prediction times were indeed reduced drastically.

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

---

<p align="center">
  <b>Sequentia</b> &copy; 2019-2023, Edwin Onuonga - Released under the <a href="https://opensource.org/licenses/MIT">MIT</a> License.<br/>
  <em>Authored and maintained by Edwin Onuonga.</em>
</p>