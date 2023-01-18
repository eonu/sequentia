#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
from setuptools import setup, find_packages
from pathlib import Path

with open('README.md', 'r', encoding='utf8') as fh:
    long_description = fh.read()

init = Path(__file__).parent / "lib" / "sequentia" / "__init__.py"
def load_meta(meta):
    with open(init, "r") as file:
        info = re.search(rf'^__{meta}__\s*=\s*[\'"]([^\'"]*)[\'"]', file.read(), re.MULTILINE).group(1)
        if not info:
            raise RuntimeError(f"Could not load {repr(meta)} metadata")
        return info

setup(
    name = load_meta("name"),
    version = load_meta("version"),
    author = load_meta("author"),
    author_email = load_meta("email"),
    description = 'HMM and DTW-based sequence machine learning algorithms in Python following an sklearn-like interface.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/eonu/sequentia',
    project_urls = {
        'Documentation': 'https://sequentia.readthedocs.io/en/latest',
        'Bug Tracker': 'https://github.com/eonu/sequentia/issues',
        'Source Code': 'https://github.com/eonu/sequentia',
    },
    license = 'MIT',
    package_dir = {'': 'lib'},
    packages = find_packages(where='lib'),
    package_data={
        'sequentia': [
            'datasets/data/digits.npz',
            'datasets/data/gene_familites.npz',
        ]
    },
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English',
    ],
    python_requires = '>=3.8',
    setup_requires = [
        'Cython>=0.28.5',
        'numpy>=1.18,<1.24',
        'scipy>=1.3',
    ],
    install_requires = [
        'numba>=0.56',
        'numpy>=1.18,<1.24',
        'hmmlearn>=0.2.8',
        'dtaidistance>=2.3.10', # [numpy]
        'scikit-learn>=1.0',
        'joblib>=0.14',
        'pydantic<1.9',
    ],
    extras_require = {
        'dev': [
            'sphinx',
            'numpydoc',
            'sphinx_rtd_theme',
            'sphinx-autodoc-typehints',
            'sphinx-autobuild',
            'm2r2',
            'mistune==0.8.4',
            'Jinja2',
            'pytest',
        ]
    }
)
