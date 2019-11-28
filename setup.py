#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import print_function
from setuptools import setup

VERSION = '0.1.0.alpha'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "stateful",
    version = VERSION,
    author = "Edwin Onuonga",
    author_email = "ed@eonu.net",
    description = "State-based isolated temporal pattern recognition with multivariate Hidden Markov Models.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/eonu/stateful",
    project_urls = {
        "Bug Tracker": "https://github.com/eonu/stateful/issues",
        "Source Code": "https://github.com/eonu/stateful"
    },
    license = 'MIT',
    packages = ['stateful'],
    package_dir = {'': 'lib'},
    classifiers = [
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Natural Language :: English"
    ],
    install_requires = [
        'numpy>=1.17,<2',
        'pomegranate>=0.11,<1'
    ],
    python_requires='>=3.5'
)