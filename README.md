<p align="center">
  <h1 align="center">
    <img src="/docs/_static/logo.png" width="125px" alt="Logo created by DALL·E Mini"><br/>
    Sequentia
  </h1>
</p>

<p align="center">
  <em>A machine learning interface for sequence classification algorithms in Python.</em>
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
    <a href="#tutorials-and-examples">Tutorials and Examples</a> ·
    <a href="#acknowledgments">Acknowledgments</a> ·
    <a href="#references">References</a> ·
    <a href="#contributors">Contributors</a>
  </sup>
</p>

## About

Sequentia is a Python package that provides various classification algorithms for sequential data, including classifiers based on hidden Markov models and dynamic time warping.

Some examples of how Sequentia can be used in sequence classification include:

- determining a spoken word based on its audio signal or alternative representations such as MFCCs,
- identifying heart conditions such as arrhythmia from ECG signals,
- predicting motion intent for gesture control from sEMG signals,
- classifying hand-written characters according to their pen-tip trajectories.

## Build Status

| `master` | `dev` |
| -------- | ------|
| [![CircleCI Build (Master)](https://img.shields.io/circleci/build/github/eonu/sequentia/master?logo=circleci&style=flat-square)](https://app.circleci.com/pipelines/github/eonu/sequentia?branch=master) | [![CircleCI Build (Development)](https://img.shields.io/circleci/build/github/eonu/sequentia/dev?logo=circleci&style=flat-square)](https://app.circleci.com/pipelines/github/eonu/sequentia?branch=master) |

## Features

Sequentia provides the following algorithms, all supporting multivariate sequences with different durations.

### Classification algorithms

- [x] [Hidden Markov Models](https://sequentia.readthedocs.io/en/latest/sections/classifiers/gmmhmm.html) (via [`hmmlearn`](https://github.com/hmmlearn/hmmlearn))<br/><em>Parameter estimation with the Baum-Welch algorithm and prediction with the forward algorithm</em> [[1]](#references)
  - [x] Gaussian Mixture Model emissions
  - [x] Linear, left-right and ergodic topologies
  - [x] Multi-processed predictions
- [x] [Dynamic Time Warping k-Nearest Neighbors](https://sequentia.readthedocs.io/en/latest/sections/classifiers/knn.html) (via [`dtaidistance`](https://github.com/wannesm/dtaidistance))
  - [x] Sakoe–Chiba band global warping constraint
  - [x] Dependent and independent feature warping (DTWD & DTWI)
  - [x] Custom distance-weighted predictions
  - [x] Multi-processed predictions

<p align="center">
  <img src="/docs/_static/classifier.png" width="80%"/><br/>
  Example of a classification algorithm (HMM sequence classifier)
</p>

### Preprocessing methods

- [x] Centering, standardization and min-max scaling
- [x] Decimation and mean downsampling
- [x] Mean and median filtering
- [ ] Dimensionality reduction (<em>soon!</em>)

## Installation

You can install Sequentia using `pip`.

```console
pip install sequentia
```

<details>
<summary>
    <b>Click here for installation instructions for contributing to Sequentia or running the notebooks.</b>
</summary>
<p>

If you intend to help contribute to Sequentia, you will need some additional dependencies for running tests, notebooks and generating documentation.

Depending on what you intend to do, you can specify the following extras.

- For running tests in the `/lib/test` directory:

  ```console
  pip install sequentia[test]
  ```
- For generating Sphinx documentation in the `/docs` directory:

  ```console
  pip install sequentia[docs]
  ```
- For running notebooks in the `/notebooks` directory:

  ```console
  pip install sequentia[notebooks]
  ```
- **A full development suite which installs all of the above extras:**

  ```console
  pip install sequentia[dev]
  ```

</p>
</details>

## Documentation

Documentation for the package is available on [Read The Docs](https://sequentia.readthedocs.io/en/latest).

## Tutorials and Examples

For detailed tutorials and examples on the usage of Sequentia, [see the notebooks here](https://nbviewer.jupyter.org/github/eonu/sequentia/tree/master/notebooks/).

Below are some basic examples of how univariate and multivariate sequences can be used in Sequentia.

### Univariate sequences

```python
import numpy as np, sequentia as seq

# Generate training observation sequences and labels
X, y = [
  np.array([1, 0, 5, 3, 7, 2, 2, 4, 9, 8, 7]),
  np.array([2, 1, 4, 6, 5, 8]),
  np.array([5, 8, 0, 3, 1, 0, 2, 7, 9])
], ['good', 'good', 'bad']

# Create and fit the classifier
clf = seq.KNNClassifier(k=1, classes=('good', 'bad')).fit(X, y)

# Make a prediction for a new observation sequence
x_new = np.array([0, 3, 2, 7, 9, 1, 1])
y_new = clf.predict(x_new)
```

### Multivariate sequences

```python
import numpy as np, sequentia as seq

# Generate training observation sequences and labels
X, y = [
  np.array([[1, 0, 5, 3, 7, 2, 2, 4, 9, 8, 7],
            [3, 8, 4, 0, 7, 1, 1, 3, 4, 2, 9]]).T,
  np.array([[2, 1, 4, 6, 5, 8],
            [5, 3, 9, 0, 8, 2]]).T,
  np.array([[5, 8, 0, 3, 1, 0, 2, 7, 9],
            [0, 2, 7, 1, 2, 9, 5, 8, 1]]).T
], ['good', 'good', 'bad']

# Create and fit the classifier
clf = seq.KNNClassifier(k=1, classes=('good', 'bad')).fit(X, y)

# Make a prediction for a new observation sequence
x_new = np.array([[0, 3, 2, 7, 9, 1, 1],
                  [2, 5, 7, 4, 2, 0, 8]]).T
y_new = clf.predict(x_new)
```

## Acknowledgments

In earlier versions of the package (<0.10.0), an approximate dynamic time warping algorithm implementation ([`fastdtw`](https://github.com/slaypni/fastdtw)) was used in hopes of speeding up k-NN predictions, as the authors of the original FastDTW paper [[2]](#references) claim that approximated DTW alignments can be computed in linear memory and time - compared to the O(N^2) runtime complexity of the usual exact DTW implementation.

However, I was recently contacted by [Prof. Eamonn Keogh](https://www.cs.ucr.edu/~eamonn/) (at _University of California, Riverside_), whose recent work [[3]](#references) makes the surprising revelation that FastDTW is generally slower than the exact DTW algorithm that it approximates. Upon switching from the `fastdtw` package to [`dtaidistance`](https://github.com/wannesm/dtaidistance) (a very solid implementation of exact DTW with fast pure C compiled functions), DTW k-NN prediction times were indeed reduced drastically.

I would like to thank Prof. Eamonn Keogh for directly reaching out to me regarding this finding!

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
			<!-- Add more <th></th> blocks for more contributors -->
		</tr>
	</thead>
</table>

---

<p align="center">
  <b>Sequentia</b> &copy; 2019-2023, Edwin Onuonga - Released under the <a href="https://opensource.org/licenses/MIT">MIT</a> License.<br/>
  <em>Authored and maintained by Edwin Onuonga.</em>
</p>