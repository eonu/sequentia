#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from setuptools import setup, find_packages
from requirements import *

VERSION = '0.12.0'

with open('README.md', 'r', encoding='utf8') as fh:
    long_description = fh.read()

setup(
    name = 'sequentia',
    version = VERSION,
    author = 'Edwin Onuonga',
    author_email = 'ed@eonu.net',
    description = 'A machine learning interface for isolated sequence classification algorithms in Python.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/eonu/sequentia',
    project_urls = {
        'Documentation': 'https://sequentia.readthedocs.io/en/latest',
        'Bug Tracker': 'https://github.com/eonu/sequentia/issues',
        'Source Code': 'https://github.com/eonu/sequentia'
    },
    license = 'MIT',
    package_dir = {'': 'lib'},
    packages = find_packages(where='lib'),
    classifiers = [
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English'
    ],
    python_requires = python_requires,
    setup_requires = setup_requires,
    install_requires = install_requires,
    extras_require = extras_require
)
