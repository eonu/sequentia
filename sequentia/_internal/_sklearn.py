# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

import sklearn

__all__ = ["routing_enabled"]


def routing_enabled() -> bool:
    return sklearn.get_config()["enable_metadata_routing"]
