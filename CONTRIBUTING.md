# Contributing

As Sequentia is an open source library, any contributions from the community are greatly appreciated. This document details the guidelines for making contributions to Sequentia.

## Reporting issues

Prior to reporting an issue, please ensure:

- [ ] You have used the search utility provided on GitHub issues to look for similar issues.
- [ ] You have checked the documentation (for the version of Sequentia you are using).
- [ ] You are using the latest version of Sequentia (if possible).

## Making changes to Sequentia

- **Add specs**: Your pull request won't be accepted if it doesn't have any specs.

- **Document any change in behaviour**: Make sure the README and all other relevant documentation is kept up-to-date.

- **Create topic branches**: Will not pull from your master branch!

- **One pull request per feature**: If you wish to add more than one new feature, please make multiple pull requests.

- **Meaningful commit messages**: Make sure each individual commit in your pull request has a meaningful message.

- **De-clutter commit history**: If you had to make multiple intermediate commits while developing, please squash them before making your pull request.

### Branch naming conventions

Branch names must be of the form `type/short-phrase-or-description`, where `type` is either a:

- `patch`: Making a change to an existing feature.
- `add`: Adding a new feature.
- `rm`: Removing an existing feature.

Branches should typically feature only one main change. If making multiple unrelated changes, please create separate branches and open separate pull requests.

### Making pull requests

Pull request titles must be of the form `[type:specifier] Pull request title`, where `type` is the same as the branch type (read above).

The `specifier` should be one of:

- `pkg`: Changes to any core package configuration.
- `lib`: Changes to any library code.
- `ci`: Changes to `.travis.yml`.
- `tests`: Changes to any test code.
- `git`: Changes to any Git-related code, such as `.gitignore`.
- `docs`: Changes to any documentation such as the Read The Docs documentation, `README.md`, `CONTRIBUTING.md`, `LICENSE` or `CHANGELOG.md`.

Continuous integration (Travis CI) builds must pass in order for your pull request to be merged.

## License

By contributing, you agree that your contributions will be licensed under the same [MIT License](/LICENSE) that covers this repository.