# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Scikit-Learn compatible HMM and DTW based
sequence machine learning algorithms in Python.
"""

import sklearn

from sequentia import (
    datasets,
    enums,
    model_selection,
    models,
    preprocessing,
    version,
)

__all__ = [
    "datasets",
    "enums",
    "model_selection",
    "models",
    "preprocessing",
    "version",
]

sklearn.set_config(enable_metadata_routing=True)
