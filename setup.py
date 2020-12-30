#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import print_function
from setuptools import setup, find_packages

def fix_setuptools():
    """Work around bugs in setuptools.

    Some versions of setuptools are broken and raise SandboxViolation for normal
    operations in a virtualenv. We therefore disable the sandbox to avoid these
    issues.
    """
    try:
        from setuptools.sandbox import DirectorySandbox
           def violation(operation, *args, **_):
            print "SandboxViolation: %s" % (args,)

        DirectorySandbox._violation = violation
    except ImportError:
        pass
# Fix bugs in setuptools.
fix_setuptools()

VERSION = '0.10.0'

with open('README.md', 'r') as fh:
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
        'Development Status :: 2 - Pre-Alpha',
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
    python_requires='>=3.6,<3.10',
    setup_requires = [
        'Cython',
        'numpy>=1.17,<2',
        'scipy>=1.3,<2'
    ],
    install_requires = [
        'numpy>=1.17,<2',
        'hmmlearn==0.2.4',
        'dtaidistance[numpy]>=2.2,<2.3',
        'scipy>=1.3,<2',
        'scikit-learn>=0.22,<1',
        'tqdm>=4.36,<5',
        'joblib>=0.14,<1'
    ]
)