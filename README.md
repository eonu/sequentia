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

Sequential data is one of the most commonly observed forms of data. These can range from time series (sequences of observations occurring through time) to non-temporal sequences such as DNA nucleotides. Time series such as audio signals and stock prices are often of particular interest as changing patterns over time naturally provide many interesting opportunities and challenges for machine learning.

This library specifically aims to tackle classification problems for isolated sequences by creating an interface to a number of classification algorithms.

Despite these types of sequences sounding very specific, you probably observe some of them on a regular basis!

**Some examples of classification problems for isolated sequences include classifying**:

- a word utterance by its speech audio signal,
- a hand-written character according to its pen-tip trajectory,
- a hand or head gesture in a video or motion-capture recording.

## Features

Sequentia offers the use of multivariate observation sequences with varying durations using the following methods:

### Classification algorithms

- [x] Hidden Markov Models (via [Pomegranate](https://github.com/jmschrei/pomegranate) [[1]](#references))<br/>Learning with the Baum-Welch algorithm <a href="#references">[2]</a>
  - [x] Multivariate Gaussian emissions
  - [x] Gaussian Mixture Model emissions (full and diagonal covariances)
  - [x] Left-right and ergodic topologies
- [x] Approximate Dynamic Time Warping k-Nearest Neighbors (implemented with [FastDTW](https://github.com/slaypni/fastdtw) [[3]](#references))
  - [x] Custom distance-weighted predictions
  - [x] Multi-processed predictions

<p align="center">
  <img src="https://i.ibb.co/jVD2S4b/classifier.png" width="60%"/><br/>
  Example of a classification algorithm: a multi-class HMM isolated sequence classifier
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

## References

<table>
  <tbody>
    <tr>
      <td>[1]</td>
      <td>
        <a href="http://jmlr.org/papers/volume18/17-636/17-636.pdf">Jacob Schreiber. <b>"pomegranate: Fast and Flexible Probabilistic Modeling in Python."</b> Journal of Machine Learning Research 18 (2018), (164):1-6.</a>
      </td>
    </tr>
    <tr>
      <td>[2]</td>
      <td>
        <a href=https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf">Lawrence R. Rabiner. <b>"A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"</b> Proceedings of the IEEE 77 (1989), no. 2, pp. 257-86.</a>
      </td>
    </tr>
    <tr>
      <td>[3]</td>
      <td>
        <a href="https://pdfs.semanticscholar.org/05a2/0cde15e172fc82f32774dd0cf4fe5827cad2.pdf">Stan Salvador, and Philip Chan. <b>"FastDTW: Toward accurate dynamic time warping in linear time and space."</b> Intelligent Data Analysis 11.5 (2007), 561-580.</a>
      </td>
    </tr>
  </tbody>
</table>

# Contributors

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
  <b>Sequentia</b> &copy; 2019-2021, Edwin Onuonga - Released under the <a href="https://opensource.org/licenses/MIT">MIT</a> License.<br/>
  <em>Authored and maintained by Edwin Onuonga.</em>
</p>