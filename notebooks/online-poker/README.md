# Neurotic apehex

[![Tagged Release][release-shield]](CHANGELOG.md)
[![Development Status][planning-status-shield]](ROADMAP.md)
[![Build Coverage][coverage-shield]][coverage-link]

[![Build Status][travis-shield]][travis-link]
[![Build Status][appveyor-shield]][appveyor-link]

> Can AI read a poker hand?! What about a poker table? :robot:

Starting from a computer screen (capture), the models are meant to locate the relevant informations, and extract the data. The project reuses well-known image classification models at its core.

* Free software: MIT
* Documentation: https://neurotic-apehex.readthedocs.io.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Features](#features)
  - [Card Recognition](#card-recognition)
  - [Table Segmentation](#table-segmentation)
  - [Screen Parsing](#screen-parsing)
- [Usage](#usage)
- [Development](#development)
  - [Future](#future)
  - [History](#history)
  - [Community](#community)
- [Credits](#credits)
- [License](#license)

## Features

At first the relevant areas will be **manually boxed** for the models so that they only receive relevant informations.

### Card Recognition

- Classify card / not-a-card
- Recognize the suit
- Recognize the value

### Table Segmentation

All of these should be either done automatically or manually.

#### Player

- Locate the player's frame
- Extract his user name
- Extract his stack

#### Bouton

- Locate the bouton on the table
- Extract the dealer's user name

#### Board

- Box the board
- Read the board cards

#### Current Player

- Find the current player
- Time his decision

#### Action Buttons

- Determine whether the user is playing on the table
- Find the user's action buttons

### Screen Parsing

Divide a random screen into boxes:
- poker windows:
  - open tables:
    - players
    - etc
  - chat
  - etc

## Usage

## Development

Contributions welcome! Read the [contribution guidelines](CONTRIBUTING.md) first.

### Future

See [ROADMAP](ROADMAP.md)

### History

See [CHANGELOG](CHANGELOG.md)

### Community

See [CODE OF CONDUCT](CODE_OF_CONDUCT.md)

## Credits

See [AUTHORS](AUTHORS.md)

This project was initially created with [Cookiecutter][cookiecutter] and the custom [cookiecutter-git][cookiecutter-git] :cookie:

## License

See [LICENSE](LICENSE)

[cookiecutter]: https://github.com/audreyr/cookiecutter
[cookiecutter-git]: https://github.com/apehex/cookiecutter-git

[appveyor-shield]: https://ci.appveyor.com/api/projects/status/github/apehex/neurotic-apehex?branch=master&svg=true
[appveyor-link]: https://ci.appveyor.com/project/apehex/neurotic-apehex/branch/master
[coverage-shield]: https://img.shields.io/badge/coverage-0%25-lightgrey.svg?longCache=true
[coverage-link]: https://codecov.io
[docs-shield]: https://readthedocs.org/projects/apehex/badge/?version=latest
[docs-link]: https://neurotic-apehex.readthedocs.io/en/latest/?badge=latest
[pypi-shield]: https://img.shields.io/pypi/v/neurotic-apehex.svg
[pypi-link]: https://pypi.python.org/pypi/neurotic-apehex
[pyup-shield]: https://pyup.io/repos/github/apehex/neurotic-apehex/shield.svg
[pyup-link]: https://pyup.io/repos/github/apehex/neurotic-apehex/
[release-shield]: https://img.shields.io/badge/release-v0-blue.svg?longCache=true
[travis-shield]: https://img.shields.io/travis/apehex/neurotic-apehex.svg
[travis-link]: https://travis-ci.org/apehex/neurotic-apehex

[planning-status-shield]: https://img.shields.io/badge/status-planning-lightgrey.svg?longCache=true
[pre-alpha-status-shield]: https://img.shields.io/badge/status-pre--alpha-red.svg?longCache=true
[alpha-status-shield]: https://img.shields.io/badge/status-alpha-yellow.svg?longCache=true
[beta-status-shield]: https://img.shields.io/badge/status-beta-brightgreen.svg?longCache=true
[stable-status-shield]: https://img.shields.io/badge/status-stable-blue.svg?longCache=true
[mature-status-shield]: https://img.shields.io/badge/status-mature-8A2BE2.svg?longCache=true
[inactive-status-shield]: https://img.shields.io/badge/status-inactive-lightgrey.svg?longCache=true