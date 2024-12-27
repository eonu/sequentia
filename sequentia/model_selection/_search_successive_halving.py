# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""This file is an adapted version of the same file from the
sklearn.model_selection sub-package.

Below is the original license from Scikit-Learn, copied on 27th December 2024
from https://github.com/scikit-learn/scikit-learn/blob/main/COPYING.

---

BSD 3-Clause License

Copyright (c) 2007-2024 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from sklearn.model_selection import _search_successive_halving as _search

from sequentia.model_selection._search import BaseSearchCV

__all__ = ["HalvingGridSearchCV", "HalvingRandomSearchCV"]


class HalvingGridSearchCV(_search.HalvingGridSearchCV, BaseSearchCV):
    """Search over specified parameter values with successive halving.

    ``cv`` must be a valid splitting method from
    :mod:`sequentia.model_selection`.

    See Also
    --------
    :class:`sklearn.model_selection.HalvingGridSearchCV`
        :class:`.HalvingGridSearchCV` is a modified version
        of this class that supports sequences.
    """


class HalvingRandomSearchCV(_search.HalvingRandomSearchCV, BaseSearchCV):
    """Randomized search on hyper parameters with successive halving.

    ``cv`` must be a valid splitting method from
    :mod:`sequentia.model_selection`.

    See Also
    --------
    :class:`sklearn.model_selection.HalvingRandomSearchCV`
        :class:`.HalvingRandomSearchCV` is a modified version
        of this class that supports sequences.
    """
