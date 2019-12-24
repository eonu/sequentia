<div align="center">
  <a href="https://pypi.org/project/sequentia/">
    <img src="https://img.shields.io/pypi/v/sequentia?style=flat-square" alt="PyPI"/>
  </a>
  <a href="https://pypi.org/project/sequentia/">
    <img src="https://img.shields.io/pypi/pyversions/sequentia?style=flat-square" alt="PyPI - Python Version"/>
  </a>
  <a href="https://raw.githubusercontent.com/eonu/sequentia/master/LICENSE">
    <img src="https://img.shields.io/pypi/l/sequentia?style=flat-square" alt="PyPI - License"/>
  </a>
</div>

# Sequentia

_A machine learning interface for isolated temporal sequence classification algorithms in Python._

## Introduction

<img src="https://i.ibb.co/gPymgs4/classifier.png" width="40%" align="right"></img>

Temporal sequences are sequences of observations that occur over time. Changing patterns over time naturally provide many interesting opportunities and challenges for machine learning.

This library specifically aims to tackle classification problems for isolated temporal sequences by creating an interface to a number of classification algorithms.

Despite these types of sequences sounding very specific, you probably observe some of them on a regular basis!

**Some examples of classification problems for isolated temporal sequences include classifying**:

- isolated word utterances in speech audio signals,
- isolated hand-written characters according to their pen-tip trajectories,
- isolated hand or head gestures in a video or motion-capture recording.

## Features

Sequentia offers the use of **multivariate observation sequences with differing durations** in conjunction with the following algorithms and methods.

### Classication algorithms

- [x] Ensemble Hidden Markov Models (via [Pomegranate](https://github.com/jmschrei/pomegranate) [[1]](#references))
  - [x] Multivariate Gaussian emission distributions
  - [ ] Gaussian Mixture Model emission distributions (soon!)
- [x] Approximate Dynamic Time Warping k-Nearest Neighbors (implemented with [FastDTW](https://github.com/slaypni/fastdtw) [[2]](#references))
- [ ] Long Short-Term Memory Networks (soon!)

### Preprocessing methods

- [x] Normalization (centering observation sequences)
- [x] Downsampling (by decimation and averaging)
- [x] Discrete (Fast) Fourier Transform

### Parallelization

- [x] Multi-processing for DTW k-NN predictions

> **Disclaimer**: The package currently remains largely untested and is still in its early pre-alpha stages – _use with caution_!

## Installation

```
pip install sequentia
```

## Tutorials and examples

For tutorials and examples on the usage of Sequentia, [look at the notebooks here](https://nbviewer.jupyter.org/github/eonu/sequentia/tree/master/notebooks/)!

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
  <b>Sequentia</b> &copy; 2019-2020, Edwin Onuonga - Released under the <a href="https://opensource.org/licenses/MIT">MIT</a> License.<br/>
  <em>Authored and maintained by Edwin Onuonga.</em>
</p>