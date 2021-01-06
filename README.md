<p align="center">
  <h1 align="center">
    <img src="https://i.ibb.co/42GkhfR/sequentia.png" width="275px" alt="Sequentia">
  </h1>
</p>

<p align="center">
  <em>A machine learning interface for isolated sequence classification algorithms in Python.</em>
</p>

<p align="center">
  <div align="center">
    <a href="https://pypi.org/project/sequentia">
      <img src="https://img.shields.io/pypi/v/sequentia?style=flat-square" alt="PyPI"/>
    </a>
    <a href="https://pypi.org/project/sequentia">
      <img src="https://img.shields.io/pypi/pyversions/sequentia?style=flat-square" alt="PyPI - Python Version"/>
    </a>
    <a href="https://raw.githubusercontent.com/eonu/sequentia/master/LICENSE">
      <img src="https://img.shields.io/pypi/l/sequentia?style=flat-square" alt="PyPI - License"/>
    </a>
    <a href="https://sequentia.readthedocs.io/en/latest">
      <img src="https://readthedocs.org/projects/sequentia/badge/?version=latest&style=flat-square" alt="Read The Docs - Documentation">
    </a>
    <a href="https://travis-ci.org/eonu/sequentia">
      <img src="https://img.shields.io/travis/eonu/sequentia?logo=travis&style=flat-square" alt="Travis - Build">
    </a>
  </div>
</p>

## Introduction

Sequential data is a commonly-observed, yet difficult-to-handle form of data. These can range from time series (sequences of observations occurring through time) to non-temporal sequences such as DNA nucleotides.

Time series data such as audio signals, stock prices and electro-cardiogram signals are often of particular interest to machine learning practitioners and researchers, as changing patterns over time naturally provide many interesting opportunities and challenges for machine learning prediction and statistical inference.

**Sequentia is a Python package that specifically aims to tackle classification problems for isolated sequences by providing implementations of a number of classification algorithms**.

Examples of such classification problems include:

- classifying a spoken word based on its audio signal (or some other representation such as MFCCs),
- classifying a hand-written character according to its pen-tip trajectory,
- classifying a hand or head gesture in a motion-capture recording,
- classifying the sentiment of a phrase or sentence in natural language.

Compared to the classification of fixed-size inputs (e.g. a vector, or images), sequence classification problems face two major hurdles:

1. the sequences are generally of different duration to each other,
2. the observations within a given sequence (may) have temporal dependencies on previous observations which occured earlier within the same sequence, and these dependencies may be arbitrarily long.

Sequentia aims to provide interpretable out-of-the-box machine learning algorithms suitable for these tasks, which require minimal configuration.

In recent times, variants of the Recurrent Neural Network (particularly LSTMs and GRUs) have generally proven to be the most successful in modelling long-term dependencies in sequences. However, the design of RNN architectures is very opiniated and requires much configuration and engineering, and is therefore not included as part of the package.

## Features

The following algorithms provided within Sequentia support the use of multivariate observation sequences with different durations.

### Classification algorithms

- [x] Hidden Markov Models (via [`hmmlearn`](https://github.com/hmmlearn/hmmlearn))<br/><em>Learning with the Baum-Welch algorithm</em> [[1]](#references)
  - [x] Gaussian Mixture Model emissions
  - [x] Linear, left-right and ergodic topologies
  - [x] Multi-processed predictions
- [x] Dynamic Time Warping k-Nearest Neighbors (via [`dtaidistance`](https://github.com/wannesm/dtaidistance))
  - [x] Sakoe–Chiba band global warping constraint
  - [x] Dependent and independent feature warping (DTWD & DTWI)
  - [x] Custom distance-weighted predictions
  - [x] Multi-processed predictions

<p align="center">
  <img src="/docs/_static/classifier.svg" width="60%"/><br/>
  Example of a classification algorithm: <em>a multi-class HMM sequence classifier</em>
</p>

### Preprocessing methods

- [x] Centering, standardization and min-max scaling
- [x] Decimation and mean downsampling
- [x] Mean and median filtering

## Installation

```console
pip install sequentia
```

## Documentation

Documentation for the package is available on [Read The Docs](https://sequentia.readthedocs.io/en/latest).

## Tutorials and examples

For tutorials and examples on the usage of Sequentia, [look at the notebooks here](https://nbviewer.jupyter.org/github/eonu/sequentia/tree/master/notebooks/).

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
        <a href=https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf">Lawrence R. Rabiner. <b>"A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"</b> <em>Proceedings of the IEEE 77 (1989)</em>, no. 2, pp. 257-86.</a>
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
        <img src="https://avatars0.githubusercontent.com/u/24795571?s=460&v=4" alt="Edwin Onuonga" width="60px">
        <br/><sub><b>Edwin Onuonga</b></sub>
        </a>
        <br/>
        <a href="mailto:ed@eonu.net">✉️</a>
        <a href="https://eonu.net">🌍</a>
			</th>
			<!-- Add more <th></th> blocks for more contributors -->
		</tr>
	</thead>
</table>

---

<p align="center">
  <b>Sequentia</b> &copy; 2019-2022, Edwin Onuonga - Released under the <a href="https://opensource.org/licenses/MIT">MIT</a> License.<br/>
  <em>Authored and maintained by Edwin Onuonga.</em>
</p>