#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import print_function
from setuptools import setup, find_packages

VERSION = '0.1.0'

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name = 'sequentia',
    version = VERSION,
    author = 'Edwin Onuonga',
    author_email = 'ed@eonu.net',
    description = 'A machine learning interface for isolated temporal sequence classification algorithms in Python.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/eonu/sequentia',
    project_urls = {
        'Bug Tracker': 'https://github.com/eonu/sequentia/issues',
        'Source Code': 'https://github.com/eonu/sequentia'
    },
    license = 'MIT',
    package_dir = {'': 'lib'},
    packages = find_packages(where='lib'),
    classifiers = [
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English'
    ],
    python_requires='>=3.5',
    install_requires = [
        'numpy>=1.17,<2',
        'pomegranate>=0.11,<1',
        'fastdtw>=0.3,<0.4',
        'scipy>=1.3,<2',
        'scikit-learn>=0.22,<1',
        'tqdm>=4.36,<5'
    ]
)