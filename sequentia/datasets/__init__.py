# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Utilities for creating and loading sample sequential datasets."""

from sequentia.datasets import data
from sequentia.datasets.base import SequentialDataset
from sequentia.datasets.digits import load_digits
from sequentia.datasets.gene_families import load_gene_families

__all__ = ["data", "load_digits", "load_gene_families", "SequentialDataset"]
