# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

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
