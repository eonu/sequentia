# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Configuration values for Sequentia classes and functions."""

import enum

__all__ = ["TopologyMode", "CovarianceMode", "TransitionMode", "PriorMode"]


class TopologyMode(enum.StrEnum):
    """Topology types for :ref:`hmms`."""

    ERGODIC = "ergodic"
    """All states have a non-zero probability of transitioning to any
    state."""

    LEFT_RIGHT = "left-right"
    """States are arranged in a way such that any state may only
    transition to itself or any state ahead of it, but not to any
    previous state."""

    LINEAR = "linear"
    """Same as :py:enum:mem:`+TopologyMode.LEFT_RIGHT`,
    but states are only permitted to transition to the next state."""


class CovarianceMode(enum.StrEnum):
    """Covariance matrix types for
    :class:`~sequentia.models.hmm.variants.gaussian_mixture.GaussianMixtureHMM`.
    """

    FULL = "full"
    """All values are fully learnable independently for each component."""

    DIAGONAL = "diag"
    """Only values along the diagonal may be learned independently
    for each component."""

    SPHERICAL = "spherical"
    """Same as :py:enum:mem:`+CovarianceMode.DIAGONAL`,
    with a single value shared along the diagonal for each component."""

    TIED = "tied"
    """Same as :py:enum:mem:`+CovarianceMode.FULL`,
    with all components sharing the same single covariance matrix."""


class TransitionMode(enum.StrEnum):
    """Initial state and transition probability types for :ref:`hmms`."""

    UNIFORM = "uniform"
    """Equal probability of starting in or transitioning to each state
    according to the topology."""

    RANDOM = "random"
    """Random probability of starting in or transitioning to each state
    according to the topology. State probabilities are sampled from a
    Dirichlet distribution with unit concentration parameters."""


class PriorMode(enum.StrEnum):
    """Prior probability types for
    :class:`~sequentia.models.hmm.classifier.HMMClassifier`.
    """

    UNIFORM = "uniform"
    """Equal probability for each class."""

    FREQUENCY = "frequency"
    """Inverse count of the occurrences of the class in the training data."""


try:
    # add enum documentation for Sphinx
    import enum_tools.documentation

    TopologyMode = enum_tools.documentation.document_enum(TopologyMode)
    CovarianceMode = enum_tools.documentation.document_enum(CovarianceMode)
    TransitionMode = enum_tools.documentation.document_enum(TransitionMode)
    PriorMode = enum_tools.documentation.document_enum(PriorMode)
except ImportError:
    pass
