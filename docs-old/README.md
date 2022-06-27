# Documentation

This directory contains the Sphinx documentation for Sequentia.

## Viewing the documentation

You can view the documentation for different versions of Sequentia on [Read The Docs](https://sequentia.readthedocs.io/en/latest).

To view a local version of the documentation, you'll simply need a web browser. Once the documentation is built, you can access it via a `file://` URL to the `docs/_build` directory.

For example, `file://FULL-PATH-TO-SEQUENTIA/docs/_build/index.html`.

## Building the documentation

To build the documentation, you'll need to make sure you have the required dependencies installed.

Once you've cloned the repository, you can do this by running the following command in the repository root.

```console
pip install .[docs]
```

Once the dependencies are installed, you can build the documentation with the following command (from the `docs` directory).

```console
make html
```