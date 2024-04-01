# Contributing

As Sequentia is an open source library, any contributions from the community are greatly appreciated.
This document details the guidelines for making contributions to Sequentia.

## Reporting issues

Prior to reporting an issue, please ensure:

- [x] You have used the search utility provided on GitHub issues to look for similar issues.
- [x] You have checked the documentation (for the version of Sequentia you are using).
- [x] You are using the latest stable version of Sequentia (if possible).

## Making changes to Sequentia

- **Add tests**:
  Your pull request won't be accepted if it doesn't have any tests (if necessary).

- **Document any change in behaviour**:
  Make sure all relevant documentation is kept up-to-date.

- **Create topic branches**:
  Will not pull from your master branch!

- **One pull request per feature**:
  If you wish to add more than one new feature, make multiple pull requests.

- **Meaningful commit messages**:
  Each commit in your pull request should have a meaningful message.

- **De-clutter commit history**:
  If you had to make multiple intermediate commits while developing, please squash them before making your pull request.
  Or add a note on the PR specifying to squash and merge your changes when ready to be merged.

### Making pull requests

Please make new branches based on the current `dev` branch, and merge your PR back into `dev` (making sure you have fetched the latest changes).

### Installing dependencies

To install all dependencies and pre-commit hooks for development, ensure you have [Poetry](https://python-poetry.org/) (1.6.1+) installed and run:

```console
make
```

### Running tests

This repository relies on the use of [Tox](https://tox.wiki/en/4.11.3/) for running tests in virtual environments.

- Run **ALL tests** in a virtual environment:
  ```console
  # a.k.a. poetry run invoke tests.install tests.unit
  poetry run tox -e tests
  ```

### Linting and formatting

This repository relies on the use of:

- [Ruff](https://github.com/astral-sh/ruff) for linting and formatting Python source code,
- [Tox](https://tox.wiki/en/4.11.3/) for running linting and formatting in a virtual environment.

To lint the source code using Ruff and Pydoclint with Tox:

```console
# a.k.a poetry run invoke lint.install lint.check
poetry run tox -e lint
```

To format the source code and attempt to auto-fix any linting issues using Ruff with Tox:

```console
# a.k.a. poetry run invoke lint.install lint.format
poetry run tox -e format
```

Pre-commit hooks will prevent you from making a commit if linting fails or your code is not formatted correctly.

### Documentation

Package documentation is automatically produced from docstrings using [Sphinx](https://www.sphinx-doc.org/en/master/).
The package also uses [Tox](https://tox.wiki/en/4.11.3/) for building documentation inside a virtual environment.

To build package documentation and automatically serve the files as a HTTP server while watching for source code changes, run:

```console
# a.k.a. poetry run invoke docs.install docs.build
poetry run tox -e docs
```

This will start a server running on `localhost:8000` by default.

To only build the static documentation HTML files without serving them or watching for changes, run:

```console
# a.k.a. poetry run invoke docs.install docs.build --no-watch
poetry run tox -e docs -- --no-watch
```

## License

By contributing, you agree that your contributions will be licensed under the repository's [MIT License](/LICENSE).

---

<p align="center">
  <b>Sequentia</b> &copy; 2019-2025, Edwin Onuonga - Released under the <a href="https://opensource.org/licenses/MIT">MIT</a> license.<br/>
  <em>Authored and maintained by Edwin Onuonga.</em>
</p>
