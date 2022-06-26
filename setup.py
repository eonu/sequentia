#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from setuptools import setup, find_packages

import platform
from pkg_resources import packaging

VERSION = '0.13.0'

with open('README.md', 'r', encoding='utf8') as fh:
    long_description = fh.read()

pkg_versions = {
    # setup dependencies (core)
    'Cython': '>=0.28.5',
    'numpy': '>=1.17,<2',
    'scipy': '>=1.3,<2',
    # install dependencies (core)
    'hmmlearn': '==0.2.7',
    'dtaidistance[numpy]': '>=2.2,<2.3',
    'scikit-learn': '>=0.22,<1',
    'tqdm': '>=4.36,<5',
    'joblib': '>=0.14,<1',
    'tslearn': '>=0.5,<0.6',
    # [docs]
    'sphinx': '>=5,<6',
    'numpydoc': '>=1.4,<1.5',
    'sphinx_rtd_theme': '>=1',
    'm2r2': '>=0.3,<0.4',
    'Jinja2': '<3.1',
    # [test]
    'pytest': '==5.3.2',
    # [notebooks]
    'jupyter': '==1.0.0',
    'requests': '==2.25.1',
    'matplotlib': '==3.3.3',
    'pandas': '==1.1.5',
    'seaborn': '==0.11.1',
    'librosa': '==0.8.0'
}

extra_pkgs = {
    'pytest': ['dev', 'test'],
    **{pkg:['dev', 'docs'] for pkg in ('sphinx', 'numpydoc', 'sphinx_rtd_theme', 'm2r2', 'Jinja2')},
    **{pkg:['dev', 'notebooks'] for pkg in (
        'jupyter', 'requests', 'matplotlib', 'pandas',
        'seaborn', 'tqdm', 'librosa'
    )},
}

def load_requires(*pkgs):
    return [pkg + pkg_versions[pkg] for pkg in pkgs]

def reverse_extra(extra):
    return [pkg + pkg_versions[pkg] for pkg, extras in extra_pkgs.items() if extra in extras]

python_requires = '>=3.6,<3.10'

setup_requires = load_requires('Cython', 'numpy', 'scipy')

install_requires = load_requires('numpy', 'hmmlearn', 'dtaidistance[numpy]', 'scipy', 'scikit-learn', 'tqdm', 'joblib', 'tslearn')
if packaging.version.parse(platform.python_version()) < packaging.version.parse('3.8'):
    install_requires.append('importlib_metadata') # Backports for importlib.metadata in Python <3.8

extras_require = {extra:reverse_extra(extra) for extra in ('dev', 'test', 'notebooks', 'docs')}

setup(
    name = 'sequentia',
    version = VERSION,
    author = 'Edwin Onuonga',
    author_email = 'ed@eonu.net',
    description = 'A machine learning interface for sequence classification algorithms in Python.',
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
