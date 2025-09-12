# sl-suite2p
The Sun lab reimplementation of the suite2p library that provides clean and modern code, expanded documentation, and a 
new pipeline to track cells across multiple days.


![PyPI - Version](https://img.shields.io/pypi/v/sl-suite2p)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sl-suite2p)
[![uv](https://tinyurl.com/uvbadge)](https://github.com/astral-sh/uv)
[![Ruff](https://tinyurl.com/ruffbadge)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/sl-suite2p)
![PyPI - Status](https://img.shields.io/pypi/status/sl-suite2p)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/sl-suite2p)

___

## Detailed Description

This library contains the refactored, re-documented and repackaged [suite2p](https://github.com/MouseLand/suite2p) 
source code, merged with the refactored, re-documented and repackaged multi-day cell tracking pipeline referenced 
[here](https://github.com/sprustonlab/multiday-suite2p-public).

The scope of the changes realized in this implementation is extensive and still ongoing. Overall, the goal of this 
project is to preserve the algorithmic core of the suite2p library while optimizing the documentation, typing, and 
implementation where possible. Once this refactoring is over, the project would transition to exploring further 
algorithmic and computation optimizations, such as adding GPU support and fine-tuning both single-day and multi-day 
pipelines. Currently, there are no plans to keep up with the existing suite2p implementation unless it receives an 
update that majorly expands or enhances its functionality relative to this project.

**Warning!** The resultant sl-suite2p is now largely ***incompatible*** with both 'source' pipelines due to extensive 
modifications to the APIs, CLIs, and configuration parameters in this project relative to both sources. Currently, 
there are no plans to make sl-suite2p compatible with the existing or future suite2p library implementations.

---

## Authorship Attribution

All original suite2p source code rights belong to the original authors and fall under the following copyright notice: 
**Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.**

For the original suite2p API and, importantly, algorithm documentation, see the original documentation available 
[here](https://suite2p.readthedocs.io/en/latest/settings.html).

All original multi-day pipeline code rights belong to the original authors of the 
[OSM Manuscript](https://www.nature.com/articles/s41586-024-08548-w).

All enhancements introduced in this library belong to the original authors and fall under the following copyright 
notice:
**Copyright © 2025 Cornell University, Authored by Ivan Kondratyev, Kushaan Gupta, and Elaine Wu.**

---

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Versioning](#versioning)
- [License](#license)
- [Acknowledgements](#Acknowledgments)
- 
___

## Features
- The library has been reworked to support the latest Python version (3.13). Support for older Python versions has 
  been deprecated and, moving forward, the library will only support the latest available Python version.
- All dependencies have been pinned to specific versions, eliminating compatibility issues.
- The library has been repackaged using modern toml-based standards.
- The high-level API has been revised to allow flexibly executing specific single-day and multi-day pipeline steps via 
  API or CLI calls. This allows experienced users to more easily parallelize pipeline steps locally or on remote compute
  servers and clusters.
- The source-code has been reformatted and refactored to improve readability and include explicit typing and improved 
  docstrings. This effort is still ongoing and, eventually, will cover the entire codebase.
- The source code has been augmented with many fixes and small optimizations to improve library performance and make it
  easier to maintain the source code in the future.
- Critically, the library now includes a similarly reimplemented multi-day cell tracking pipeline from the 
  [OSM manuscript](https://www.nature.com/articles/s41586-024-08548-w), enabling suite2p to track cells both 
  within-days and across-days.

---

## Dependencies

All software library dependencies are installed automatically as part of the library installation.

---

## Installation

### Source

Note, installation from source is ***highly discouraged*** for everyone who is not an active project developer.

1. Download this repository to your local machine using your preferred method, such as Git-cloning. Use one
   of the stable releases from [GitHub](https://github.com/Sun-Lab-NBB/suite2p/releases).
2. Unpack the downloaded zip and note the path to the binary wheel (`.whl`) file contained in the archive.
3. Run ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file, to install the 
   wheel into the active python environment.

### pip
Use the following command to install the library using pip: ```pip install sl-suite2p```.

### Optional performance enhancement

If this library is installed on an AMD64 (x64) system, use the optional `x64` installation specification to install 
additional dependencies that enhance library performance in some processing scenario, e.g.:
```pip install sl-suite2p[x64]```

---

## Usage

To learn about using this library to process your data, see detailed pipeline guides in the [notebooks](/notebooks)
folder. Note, both example notebooks are shipped with the source code of the library each time it is installed from 
pip or source. The notebook examples fully cover the single-day and multi-day pipeline API and are recommended for all
users, even those that intend to use the CLI commands to work with the library.

### CLI
All functions demonstrated in the example notebooks can also be called via the Command-Line Interface exposed by this 
library upon installation into a Python environment. See the API documentation for the list of available CLI commands 
and their arguments.

---

## API Documentation

See the [API documentation](https://sl-suite2p-api-docs.netlify.app/) for the detailed description of the methods and 
classes exposed by components of this library.

***Note!*** As the work on refactoring the library is still ongoing, the current version of the API documentation does 
not reflect the entirety of the public-facing API. Additional packages will be added to the API documentation as they 
are being refactored and redocumented.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). For the versions available, see the 
[tags on this repository](https://github.com/Sun-Lab-NBB/suite2p/tags).

**Note!** Since this project started as an extension of the original suite2p project, public versioning starts from 
version 1.0.0, to reflect that this library has evolved from the original suite2p version 0.

---

## License

This project is licensed under the GPL3 License: see the [LICENSE](../sl-suite2p/LICENSE) file for details.

___

## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- The authors and maintainers of the original [suite2p](https://github.com/MouseLand/suite2p) and 
[multi-day pipeline](https://github.com/sprustonlab/multiday-suite2p-public).
- The creators of all other projects used in our development automation pipelines and source code 
  [see pyproject.toml](../sl-suite2p/pyproject.toml).

---
