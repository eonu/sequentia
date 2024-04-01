# Copyright (c) 2019-2025 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

import numpy as np
import numpy.typing as npt

__all__ = ["FloatArray", "IntArray", "Array"]

FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]
Array = FloatArray | IntArray
