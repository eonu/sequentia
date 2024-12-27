# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

"""Sphinx configuration file for Sequentia documentation."""

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "sequentia"
copyright = "2019, Sequentia Developers"  # noqa: A001
author = "Edwin Onuonga (eonu)"
release = "2.5.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    # 'sphinx.ext.viewcode',
    "sphinx.ext.intersphinx",
    # "numpydoc",
    "enum_tools.autoenum",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "hmmlearn": ("https://hmmlearn.readthedocs.io/en/stable/", None),
}

napoleon_numpy_docstring = True
napoleon_use_admonition_for_examples = True
autodoc_members = True
autodoc_member_order = "groupwise"  # bysource, groupwise, alphabetical
autosummary_generate = True
numpydoc_show_class_members = False

# Set master document
master_doc = "index"

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pyramid"  # sphinx_rtd_theme

autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Custom stylesheets
def setup(app) -> None:  # noqa: ANN001, D103
    app.add_css_file("css/toc.css")
