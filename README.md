# cindra

Provides pipelines for processing neural imaging data and tracking Regions of Interest across multiple recordings.

![PyPI - Version](https://img.shields.io/pypi/v/cindra)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cindra)
[![uv](https://tinyurl.com/uvbadge)](https://github.com/astral-sh/uv)
[![Ruff](https://tinyurl.com/ruffbadge)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/cindra)
![PyPI - Status](https://img.shields.io/pypi/status/cindra)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/cindra)

___

## Detailed Description

Cindra is a ground-up reimplementation of the [suite2p](https://github.com/MouseLand/suite2p) library, merged with
a similarly reimplemented multi-recording ROI tracking pipeline from the
[OSM manuscript](https://www.nature.com/articles/s41586-024-08548-w). The library maintains the algorithmic core of
these projects with extensive architecture, documentation, and implementation enhancements focused on improving memory
efficiency and runtime speed. Cindra offers CLI, GUI, and MCP server interfaces alongside the Python API to streamline
user interaction with the library.

___

## Authorship Attribution

The single-recording pipeline algorithms reimplemented in this library originate from the
[suite2p](https://github.com/MouseLand/suite2p) project. All original algorithm rights belong to the original authors
and fall under the following copyright notice:
**Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.**

For the original suite2p algorithm documentation, see the original documentation available
[here](https://suite2p.readthedocs.io/en/latest/settings.html).

The multi-recording ROI tracking pipeline algorithms reimplemented in this library originate from the
[OSM Manuscript](https://www.nature.com/articles/s41586-024-08548-w). All original algorithm rights belong to the
original authors.

The diffeomorphic registration algorithms reimplemented in this library originate from the
[pirt](https://github.com/almarklein/pirt) library. Copyright 2010-2017 Almar Klein, University of Twente.

All implementation details in this library, including the complete reimplementation of the above algorithms, the
codebase architecture, documentation, CLI, GUI, and MCP interfaces, belong to the original authors and fall under the
following copyright notice:
**Copyright © 2025 Cornell University, Authored by Ivan Kondratyev and Natalie Yeung.**

___

## Features

- Supports Python 3.14 with full type annotations and MyPy strict mode compliance.
- Reimplements the single-recording suite2p pipeline: TIFF binarization, rigid and nonrigid motion correction, ROI
  detection with PCA denoising, cell classification, fluorescence extraction, neuropil subtraction, and OASIS spike
  deconvolution.
- Implements a novel multi-recording ROI tracking pipeline: diffeomorphic demons registration to a common coordinate
  space, spatial clustering for cross-recording ROI matching, and template-based fluorescence extraction across
  recordings.
- Provides a configuration-driven architecture using YAML files, enabling flexible execution of individual pipeline
  phases via API or CLI for local and remote parallelization.
- Includes three interactive PySide6/PyQtGraph GUI viewers for inspecting ROI detection, registration quality, and
  multi-recording tracking results.
- Exposes two MCP servers for AI agent integration: a data processing server with 30 tools for pipeline orchestration
  and results querying, and a GUI server with 4 tools for viewer lifecycle management.
- Uses Numba JIT compilation with Intel TBB threading for parallelized frame-level computation.
- GPL-3.0-or-later License.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [API](#api)
  - [CLI Commands](#cli-commands)
  - [GUI Viewers](#gui-viewers)
  - [MCP Servers](#mcp-servers)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

___

## Dependencies

For users, all library dependencies are installed automatically by all supported installation
methods. For developers, see the [Developers](#developers) section for information on installing
additional development dependencies.

___

## Installation

### Source

***Note,*** installation from source is ***highly discouraged*** for anyone who is not an active
project developer.

1. Download this repository to the local machine using the preferred method, such as git-cloning.
   Use one of the [stable releases](https://github.com/Sun-Lab-NBB/cindra/tags) that
   include precompiled binary and source code distribution (sdist) wheels.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the
   appropriate decompression tool.
3. `cd` to the root directory of the prepared project distribution.
4. Run `pip install .` to install the project and its dependencies.

### pip

Use the following command to install the library and all of its dependencies via
[pip](https://pip.pypa.io/en/stable/): `pip install cindra`

___

## Usage

### API

The library exposes a high-level Python API for programmatic pipeline execution. The two primary entry points are
`run_single_recording_pipeline()` and `run_multi_recording_pipeline()`, which accept YAML configuration files generated
via `SingleRecordingConfiguration` and `MultiRecordingConfiguration` dataclasses respectively.

```python
from pathlib import Path
from cindra import (
    SingleRecordingConfiguration,
    run_single_recording_pipeline,
    run_multi_recording_pipeline,
    MultiRecordingConfiguration,
)

# Generate a default single-recording configuration and customize it.
config = SingleRecordingConfiguration()
config.file_io.data_path = Path("/path/to/tiff/directory")
config.file_io.output_path = Path("/path/to/output")
config.to_yaml(Path("/path/to/config.yaml"))

# Execute the full single-recording pipeline (binarize, process, combine).
run_single_recording_pipeline(configuration_path=Path("/path/to/config.yaml"))

# For multi-recording pipelines, configure and run similarly.
md_config = MultiRecordingConfiguration()
md_config.to_yaml(Path("/path/to/md_config.yaml"))
run_multi_recording_pipeline(configuration_path=Path("/path/to/md_config.yaml"))
```

***Note,*** raw imaging data directories must contain a `cindra_parameters.json` file specifying acquisition metadata
(frame rate, plane count, channel count) before pipeline execution. This file can be generated using the
`generate_acquisition_parameters_file` MCP tool or constructed manually.

### CLI Commands

This library provides the `cindra` and `cindra-gui` CLIs that expose the following commands:

#### cindra

| Command     | Description                                                                        |
|-------------|------------------------------------------------------------------------------------|
| `configure` | Generates default YAML configuration files for single or multi-recording pipelines |
| `run`       | Executes a pipeline using a YAML configuration file with optional CLI overrides    |
| `mcp`       | Starts the data processing MCP server for AI agent integration                     |

The `run` command supports executing individual pipeline phases (`--binarize`, `--process`, `--combine` for
single-recording; `--discover`, `--extract` for multi-recording), targeting specific planes (`--target-plane`) or
recordings (`--target-recording`), and controlling parallelism (`--workers`).

#### cindra-gui

| Command        | Description                                                    |
|----------------|----------------------------------------------------------------|
| `roi`          | Launches the ROI viewer for single or multi-recording datasets |
| `registration` | Launches the registration quality viewer                       |
| `tracking`     | Launches the multi-recording tracking quality viewer           |
| `mcp`          | Starts the GUI MCP server for viewer lifecycle management      |

Use `cindra --help`, `cindra COMMAND --help`, `cindra-gui --help`, or `cindra-gui COMMAND --help` for detailed usage
information.

### GUI Viewers

Cindra provides three interactive GUI viewers built with PySide6 and PyQtGraph. The viewers launch as separate
subprocesses to avoid loading Qt dependencies during headless pipeline execution.

| Viewer       | Command                   | Description                                                                                                                                                  |
|--------------|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ROI Viewer   | `cindra-gui roi`          | Displays detected ROIs overlaid on mean images with fluorescence traces, classification controls, and reclassification support                               |
| Registration | `cindra-gui registration` | Displays a dual-window viewer combining a binary movie player with rigid/nonrigid offset visualization and a principal component registration metrics viewer |
| Tracking     | `cindra-gui tracking`     | Displays multi-recording ROI tracking results with cross-recording template masks and matching confidence visualization                                      |

The ROI viewer supports both single-recording and multi-recording datasets. Pass the `--dataset` flag to view
multi-recording results for a specific dataset.

### MCP Servers

This library provides two MCP servers that expose neural imaging pipeline functionality for AI agent integration.

#### Data Processing Server

Start the data processing MCP server using the CLI:

```bash
cindra mcp
```

##### Available Tools

| Tool                                              | Description                                                         |
|---------------------------------------------------|---------------------------------------------------------------------|
| `generate_acquisition_parameters_file`            | Generates a `cindra_parameters.json` file for a recording directory |
| `validate_acquisition_parameters_file`            | Validates an existing acquisition parameters file                   |
| `validate_recording_readiness`                    | Validates that a recording is ready for pipeline processing         |
| `generate_config_file`                            | Generates a default YAML configuration file                         |
| `discover_recordings_tool`                        | Discovers all recordings under a root directory                     |
| `resolve_dataset_name_tool`                       | Constructs a qualified multi-recording dataset name                 |
| `read_config_file`                                | Reads and returns the contents of a configuration YAML file         |
| `validate_config_file`                            | Validates a configuration file and reports non-default parameters   |
| `get_recording_status_tool`                       | Gets the processing status of a single recording                    |
| `get_batch_status_overview_tool`                  | Gets an overview of batch processing status across recordings       |
| `prepare_single_recording_batch_tool`             | Prepares single-recording batch processing jobs without execution   |
| `prepare_multi_recording_batch_tool`              | Prepares multi-recording batch processing jobs without execution    |
| `reset_processing_phases_tool`                    | Resets completed processing phases for re-execution                 |
| `clean_processing_output_tool`                    | Deletes processing output artifacts for clean re-processing         |
| `execute_processing_jobs_tool`                    | Dispatches prepared processing jobs with saturating core allocation |
| `get_processing_jobs_status_tool`                 | Queries the status of active processing jobs                        |
| `get_active_execution_timing_tool`                | Gets execution timing metrics for active processing jobs            |
| `cancel_processing_jobs_tool`                     | Cancels currently running processing jobs                           |
| `execute_full_pipeline_tool`                      | Executes the full pipeline end-to-end in a single call              |
| `verify_single_recording_output_tool`             | Verifies completeness of single-recording pipeline output           |
| `verify_multi_recording_output_tool`              | Verifies completeness of multi-recording pipeline output            |
| `query_single_recording_metadata_tool`            | Queries recording metadata (planes, channels, frame count)          |
| `query_registration_quality_tool`                 | Queries registration quality metrics (rigid and nonrigid offsets)   |
| `query_detection_summary_tool`                    | Queries detection summary (ROI counts, classification statistics)   |
| `query_roi_statistics_tool`                       | Queries detailed ROI statistics for up to 500 ROIs                  |
| `query_traces_tool`                               | Queries fluorescence traces for up to 50 ROIs                       |
| `query_multi_recording_overview_tool`             | Queries multi-recording dataset overview                            |
| `query_multi_recording_registration_quality_tool` | Queries cross-recording registration quality metrics                |
| `query_multi_recording_tracking_summary_tool`     | Queries multi-recording ROI tracking summary statistics             |
| `query_cross_recording_traces_tool`               | Queries cross-recording fluorescence traces for tracked ROIs        |

#### GUI Lifecycle Server

Start the GUI MCP server using the CLI:

```bash
cindra-gui mcp
```

##### Available Tools

| Tool                      | Description                                    |
|---------------------------|------------------------------------------------|
| `launch_viewer_tool`      | Launches a GUI viewer as a managed subprocess  |
| `list_viewers_tool`       | Lists all active GUI viewer processes          |
| `close_viewer_tool`       | Closes a specific GUI viewer by its identifier |
| `query_viewer_state_tool` | Queries the current state of an active viewer  |

#### Client Registration

MCP server registration and Claude Code skill assets for this library are distributed through the
[cindra](https://github.com/Sun-Lab-NBB/cindra) marketplace. Install the marketplace plugins to automatically register
the MCP servers with compatible clients and make all associated skills available.

___

## API Documentation

See the [API documentation](https://cindra-api-docs.netlify.app/) for the detailed
description of the methods and classes exposed by components of this library.

___

## Developers

This section provides installation, dependency, and build-system instructions for the developers
that want to modify the source code of this library.

### Installing the Project

***Note,*** this installation method requires **mamba version 2.3.2 or above**. Currently, all
Sun lab automation pipelines require that mamba is installed through the
[miniforge3](https://github.com/conda-forge/miniforge) installer.

1. Download this repository to the local machine using the preferred method, such as git-cloning.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the
   appropriate decompression tool.
3. `cd` to the root directory of the prepared project distribution.
4. Install the core Sun lab development dependencies into the ***base*** mamba environment via the
   `mamba install tox uv tox-uv` command.
5. Use the `tox -e create` command to create the project-specific development environment followed
   by `tox -e install` command to install the project into that environment as a library.

### Additional Dependencies

In addition to installing the project and all user dependencies, install the following
dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version supported by
   the developed project. Currently, this library supports Python 3.14 only. It is recommended to
   use a tool like [pyenv](https://github.com/pyenv/pyenv) to install and manage the required
   versions.

### Development Automation

This project uses `tox` for development automation. The following tox environments are available:

| Environment    | Description                                                  |
|----------------|--------------------------------------------------------------|
| `lint`         | Runs ruff formatting, ruff linting, and mypy type checking   |
| `stubs`        | Generates py.typed marker and .pyi stub files                |
| `py314-test`   | Runs the test suite via pytest for Python 3.14               |
| `coverage`     | Aggregates test coverage into an HTML report                 |
| `docs`         | Builds the API documentation via Sphinx                      |
| `build`        | Builds sdist and wheel distributions                         |
| `upload`       | Uploads distributions to PyPI via twine                      |
| `install`      | Builds and installs the project into its mamba environment   |
| `uninstall`    | Uninstalls the project from its mamba environment            |
| `create`       | Creates the project's mamba development environment          |
| `remove`       | Removes the project's mamba development environment          |
| `provision`    | Recreates the mamba environment from scratch                 |
| `export`       | Exports the mamba environment as .yml and spec.txt files     |
| `import`       | Creates or updates the mamba environment from a .yml file    |

Run any environment using `tox -e ENVIRONMENT`. For example, `tox -e lint`.

***Note,*** all pull requests for this project have to successfully complete the `tox` task before
being merged. To expedite the task's runtime, use the `tox --parallel` command to run some tasks
in parallel.

### AI-Assisted Development

Claude Code skills and AI development assets for this project are distributed through two marketplaces:

- [cindra](https://github.com/Sun-Lab-NBB/cindra) marketplace: Provides MCP server registrations, pipeline-specific
  skills for single-recording and multi-recording processing, configuration, results inspection, visualization, and MCP
  environment setup. Install this marketplace to register the `cindra mcp` and `cindra-gui mcp` servers with
  compatible MCP clients and make all pipeline workflow skills available.
- [ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) marketplace: Provides shared development skills that enforce
  Sun Lab coding conventions (Python style, README style, commit messages, pyproject.toml, tox configuration) and
  general-purpose codebase exploration tools via the **automation** plugin.

Install both marketplaces to make all associated skills and development tools available to compatible AI coding agents.

### Automation Troubleshooting

Many packages used in `tox` automation pipelines (uv, mypy, ruff) and `tox` itself may experience
runtime failures. In most cases, this is related to their caching behavior. If an unintelligible
error is encountered with any of the automation components, deleting the corresponding cache
directories (`.tox`, `.ruff_cache`, `.mypy_cache`, etc.) manually or via a CLI command typically
resolves the issue.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the
[tags on this repository](https://github.com/Sun-Lab-NBB/cindra/tags) for the available project
releases.

___

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))
- Natalie Yeung

___

## License

This project is licensed under the GPL-3.0-or-later License: see the [LICENSE](LICENSE) file for details.

___

## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- The authors and maintainers of the original [suite2p](https://github.com/MouseLand/suite2p) and
  [multi-recording pipeline](https://github.com/sprustonlab/multi_recording-suite2p-public), whose algorithms were
  reimplemented in this library.
- Elaine Wu for contributing to the early reimplementation of the I/O module.
- Almar Klein, author of the original [pirt](https://github.com/almarklein/pirt) library, whose diffeomorphic
  registration algorithms were reimplemented to form the basis of the multi-recording registration module.
- The creators of all other dependencies and projects listed in the [pyproject.toml](pyproject.toml) file.

___
