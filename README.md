# Sequentia

A generalized interface for isolated temporal sequence classification algorithms.

## Introduction

<img src="https://i.ibb.co/gPymgs4/classifier.png" width="35%" align="right"></img>

Temporal sequences are sequences of observations that occur over time. Changing patterns over time naturally provide many interesting opportunities and challenges for machine learning.

This library specifically aims to tackle classification problems for isolated temporal sequences. Despite these types of sequences sounding very specific, you probably observe some of them on a regular basis!

Some examples of classification problems for isolated temporal sequences include classifying individual:

- word utterances in speech audio signals,
- hand-written characters according to their pen-tip trajectories,
- hand or head gestures in a video or motion-capture recording.

## Features

**Classication algorithms**:

- [x] Ensemble Hidden Markov Models
- [x] Dynamic Time Warping k-Nearest Neighbors (implemented with [FastDTW](https://github.com/slaypni/fastdtw))
- [ ] Long Short-Term Memory Networks

**Preprocessing methods**:

- [x] Sequence Normalization
- [x] Downsampling (by decimation and averaging)
- [x] Discrete (Fast) Fourier Transform

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