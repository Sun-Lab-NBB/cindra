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
**Copyright © 2026 Sun (NeuroAI) lab, Authored by Ivan Kondratyev and Natalie Yeung.**

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
- Natively supports two-channel functional imaging with independent ROI detection, colocalization analysis, and
  fluorescence extraction per channel.
- Uses Numba JIT compilation with Intel TBB threading for parallelized frame-level computation.
- GPL-3.0-or-later License.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Input Data Format](#input-data-format)
  - [Configuration](#configuration)
  - [Data Structures](#data-structures)
  - [Single-Recording Pipeline](#single-recording-pipeline)
    - [Phase 1: Binarization](#phase-1-binarization)
    - [Phase 2: Processing](#phase-2-processing)
    - [Phase 3: Combination](#phase-3-combination)
  - [Multi-Recording Pipeline](#multi-recording-pipeline)
    - [Phase 1: Discovery](#phase-1-discovery)
    - [Phase 2: Multi-Recording Extraction](#phase-2-multi-recording-extraction)
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

On macOS, cindra uses Numba's OpenMP threading layer for parallel execution because `tbb4py` is not published for
Apple Silicon. The OpenMP runtime (`libomp.dylib`) is not shipped with macOS or Apple's clang toolchain and must be
provided separately. The recommended path is a conda environment with `llvm-openmp` from conda-forge:

`conda install -c conda-forge llvm-openmp` (or `mamba install -c conda-forge llvm-openmp`)

For `pip`-only installs, install libomp via [Homebrew](https://brew.sh/) (`brew install libomp`) and either symlink
`$(brew --prefix libomp)/lib/libomp.dylib` into the active virtual environment's `lib/` directory or set
`DYLD_LIBRARY_PATH` to include that path before importing cindra. Without a loadable `libomp.dylib`, importing
cindra fails with `ValueError: No threading layer could be loaded`. Linux and Windows installations require no
additional steps.

For users, all other library dependencies are installed automatically by all supported installation
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

### Input Data Format

Cindra processes two-photon (or one-photon) calcium imaging data stored as TIFF files. Before running any pipeline,
the raw data directory must be prepared with the correct structure.

#### TIFF Files

The pipeline expects a flat directory containing one or more `.tif` / `.tiff` files. For multi-plane or multichannel
acquisitions, frames must be interleaved in the following order within each TIFF file:
plane0_channel1, plane0_channel2, plane1_channel1, plane1_channel2, and so on, repeating for each time point. This
interleaving pattern continues seamlessly across TIFF file boundaries when a recording spans multiple files.

For MROI (multi-region of interest) line-scanning acquisitions, each raw TIFF frame must contain the full imaging strip
with all ROI regions arranged vertically. The interleaving order across planes and channels is the same as standard
acquisitions. During binarization, the pipeline uses the `roi_lines` field from `cindra_parameters.json` to slice each
frame into region-specific strips. Each ROI-plane combination becomes a separate virtual plane for processing.

#### Acquisition Parameters

Each raw data directory must contain a `cindra_parameters.json` file that describes how the data was acquired. This
file can be generated using the `generate_acquisition_parameters_file` [MCP tool](#mcp-servers) or constructed
manually. The required fields are:

| Field            | Type  | Description                                                 |
|------------------|-------|-------------------------------------------------------------|
| `frame_rate`     | float | Volume acquisition rate in Hz (frames per second per plane) |
| `plane_number`   | int   | Number of physical imaging planes                           |
| `channel_number` | int   | Number of channels per plane (1 or 2)                       |

For MROI (multi-region of interest) line-scanning recordings, additional fields describe the geometry of each acquired
region:

| Field               | Type                | Description                                                                           |
|---------------------|---------------------|---------------------------------------------------------------------------------------|
| `roi_number`        | int                 | Number of ROI regions acquired per plane (> 1 for MROI)                               |
| `roi_lines`         | list of list of int | Line indices in the raw frame belonging to each ROI region                            |
| `roi_x_coordinates` | list of int         | Horizontal pixel position of each ROI's top-left corner in the combined field of view |
| `roi_y_coordinates` | list of int         | Vertical pixel position of each ROI's top-left corner in the combined field of view   |

In MROI mode, each ROI-plane combination is treated as a separate virtual plane for processing. The pipeline uses
`roi_lines` to slice each raw frame into region-specific strips and uses `roi_x_coordinates` / `roi_y_coordinates` to
position the regions in the combined field of view during the combination phase.

#### Example Directory Structure

```
recording_2025_03_10/
├── scan_00001.tif
├── scan_00002.tif
├── scan_00003.tif
└── cindra_parameters.json
```

### Configuration

All pipeline behavior is controlled through YAML configuration files. Generate a default configuration using the CLI
or API, then modify it as needed before running the pipeline.

To generate a default single-recording configuration via the CLI:
`cindra configure --pipeline single-recording --output-path /path/to/output`

To generate a default multi-recording configuration:
`cindra configure --pipeline multi-recording --output-path /path/to/output`

Configuration files are structured as nested sections, each controlling a different aspect of the pipeline. See the
[API documentation](https://cindra-api-docs.netlify.app/) for the complete description of every configuration field,
including defaults and valid ranges.

#### Single-Recording Configuration Sections

| Section                   | Purpose                                                                                                                     |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `main`                    | General pipeline behavior: time constant (tau), channel roles, flyback planes, custom classifier path                       |
| `file_io`                 | Input TIFF directory, output directory, file exclusion patterns                                                             |
| `runtime`                 | Parallel worker count, progress bar display                                                                                 |
| `registration`            | Motion correction: reference frame selection, smoothing, offset limits, bidirectional phase correction, two-step refinement |
| `nonrigid_registration`   | Block-based nonrigid correction: block size, SNR threshold, maximum block offset                                            |
| `one_photon_registration` | One-photon specific preprocessing: spatial high-pass filtering, edge tapering                                               |
| `roi_detection`           | ROI detection: threshold scaling, temporal/spatial high-pass windows, PCA denoising, overlap limits, preclassification      |
| `signal_extraction`       | Fluorescence extraction: neuropil settings, batch size, classification threshold, overlap handling                          |
| `spike_deconvolution`     | OASIS deconvolution: baseline method and parameters, spike extraction toggle                                                |

#### Multi-Recording Configuration Sections

| Section                      | Purpose                                                                                                                  |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `recording_io`               | Recording directory paths, dataset naming, output location                                                               |
| `runtime`                    | Parallel worker count, progress bar display                                                                              |
| `roi_selection`              | ROI filtering: probability threshold, maximum size, MROI region margin                                                   |
| `diffeomorphic_registration` | Diffeomorphic demons: reference image type, speed factor, grid sampling, iterations per scale                            |
| `roi_tracking`               | Cross-recording clustering: Jaccard distance threshold, mask/pixel prevalence, spatial bin size, centroid distance limit |
| `signal_extraction`          | Same extraction parameters as single-recording (applied to tracked ROIs)                                                 |
| `spike_deconvolution`        | Same deconvolution parameters as single-recording                                                                        |

### Data Structures

This section describes the key data files produced by the pipelines. All per-plane data is stored under
`<output_path>/cindra/plane_<i>/`, and combined data at the `<output_path>/cindra/` root.

#### Binary Imaging Data

Registration writes corrected frames back to the same binary files created during binarization. There are no separate
"registered" binary files — `channel_1_data.bin` is overwritten in place with motion-corrected data.

| File                 | Format               | Description                                                                            |
|----------------------|----------------------|----------------------------------------------------------------------------------------|
| `channel_1_data.bin` | int16 (frames, h, w) | Channel 1 imaging frames (raw after binarization, motion-corrected after registration) |
| `channel_2_data.bin` | int16 (frames, h, w) | Channel 2 imaging frames (two-channel recordings only)                                 |

#### Per-Plane Registration Data

Stored under `plane_<i>/registration_data/`:

| File                                     | Format                       | Description                                                |
|------------------------------------------|------------------------------|------------------------------------------------------------|
| `reference_image.npy`                    | float32 (h, w)               | Alignment target computed from the most stable frames      |
| `rigid_y_offsets.npy`                    | int32 (frames,)              | Per-frame vertical translation from phase correlation      |
| `rigid_x_offsets.npy`                    | int32 (frames,)              | Per-frame horizontal translation from phase correlation    |
| `rigid_correlations.npy`                 | float32 (frames,)            | Phase correlation quality per frame                        |
| `bad_frames.npy`                         | bool (frames,)               | Flags frames with excessive motion                         |
| `nonrigid_y_offsets.npy`                 | float32 (frames, num_blocks) | Per-block vertical offsets (when nonrigid enabled)         |
| `nonrigid_x_offsets.npy`                 | float32 (frames, num_blocks) | Per-block horizontal offsets (when nonrigid enabled)       |
| `nonrigid_correlations.npy`              | float32 (frames, num_blocks) | Per-block correlation quality (when nonrigid enabled)      |
| `principal_component_projections.npy`    | float32 (frames, n_pcs)      | Frame projections onto principal components (when enabled) |
| `principal_component_extreme_images.npy` | float32 (2, n_pcs, h, w)     | Mean images of low/high projection frames per PC           |
| `principal_component_shift_metrics.npy`  | float32 (n_pcs, 3)           | Registration quality metrics per PC                        |

#### Per-Plane Detection Data

Stored under `plane_<i>/detection_data/`:

| File                      | Format         | Description                                        |
|---------------------------|----------------|----------------------------------------------------|
| `mean_image.npy`          | float32 (h, w) | Average of all registered frames                   |
| `enhanced_mean_image.npy` | float32 (h, w) | Background-subtracted and contrast-normalized mean |
| `maximum_projection.npy`  | float32 (h, w) | Maximum intensity projection across all frames     |
| `correlation_map.npy`     | float32 (h, w) | Pixel-wise correlation with neighboring pixels     |

Channel 2 variants (`mean_image_channel_2.npy`, etc.) are saved when both channels are functional.

#### Per-Plane ROI and Extraction Data

Stored under `plane_<i>/`:

| File                          | Format                   | Description                                                              |
|-------------------------------|--------------------------|--------------------------------------------------------------------------|
| `roi_masks.npz`               | variable-length arrays   | Per-ROI pixel coordinates, weights, centroids                            |
| `roi_statistics.npz`          | float arrays             | Per-ROI shape properties (compactness, solidity, aspect ratio, skewness) |
| `cell_fluorescence.npy`       | float32 (n_rois, frames) | Raw fluorescence time series per ROI                                     |
| `neuropil_fluorescence.npy`   | float32 (n_rois, frames) | Background fluorescence from surround masks                              |
| `subtracted_fluorescence.npy` | float32 (n_rois, frames) | Neuropil-corrected and baseline-subtracted traces                        |
| `spikes.npy`                  | float32 (n_rois, frames) | Inferred spike amplitudes from OASIS                                     |
| `cell_classification.npy`     | float32 (n_rois, 2)      | Column 0: is_cell label (1.0 or 0.0); column 1: classifier probability   |

Channel 2 variants (`cell_fluorescence_channel_2.npy`, etc.) are saved when both channels are functional.

#### Combined Data

Stored at `<output_path>/cindra/`:

| File                    | Description                                                                                |
|-------------------------|--------------------------------------------------------------------------------------------|
| `combined_metadata.npz` | Plane geometry (offsets, dimensions), sampling rate, tau, and paths to registered binaries |
| `roi_masks.npz`         | ROI masks with coordinates adjusted to the combined coordinate system                      |
| `roi_statistics.npz`    | ROI statistics tagged with source plane index                                              |
| `cell_fluorescence.npy` | Concatenated fluorescence traces across all planes                                         |
| `spikes.npy`            | Concatenated spike trains across all planes                                                |
| `detection_data/`       | Combined detection images (mean, enhanced mean, maximum projection, correlation map)       |

The same set of extraction files (`neuropil_fluorescence.npy`, `subtracted_fluorescence.npy`, `cell_classification.npy`,
and channel 2 variants) follows the same naming convention at the combined level.

#### Multi-Recording Data

Stored under `<output_path>/cindra/multi_recording/<dataset_name>/` per recording:

| File / Directory                         | Description                                               |
|------------------------------------------|-----------------------------------------------------------|
| `multi_recording_runtime_data.yaml`      | Per-recording runtime metadata and timing                 |
| `multi_recording_configuration.yaml`     | Shared multi-recording configuration                      |
| `registration_arrays/deform_field_y.npy` | Vertical deformation field component                      |
| `registration_arrays/deform_field_x.npy` | Horizontal deformation field component                    |
| `registration_arrays/transformed_*.npy`  | Reference images warped to the shared coordinate space    |
| `registration_deformed_masks.npz`        | Forward-transformed ROI masks in the shared space         |
| `tracking_template_masks.npz`            | Consensus template masks for tracked cells                |
| `roi_masks.npz`                          | Backward-transformed template masks in native coordinates |
| `roi_statistics.npz`                     | Shape statistics for backward-transformed templates       |
| `cell_fluorescence.npy`                  | Fluorescence traces for tracked ROIs in this recording    |
| `neuropil_fluorescence.npy`              | Background fluorescence from surround masks               |
| `subtracted_fluorescence.npy`            | Neuropil-corrected and baseline-subtracted traces         |
| `spikes.npy`                             | Spike trains for tracked ROIs in this recording           |
| `cell_colocalization.npy`                | Channel colocalization scores (dual-channel only)         |

### Single-Recording Pipeline

The single-recording pipeline processes a single calcium imaging session through three sequential phases: binarization,
processing, and combination. Phase 2 (processing) runs independently per imaging plane, enabling parallel execution
across planes.

#### Phase 1: Binarization

The binarization phase converts raw TIFF files into an internal memory-mapped binary format that the rest of the
pipeline reads from. During conversion, interleaved frames are separated by plane and channel, and a mean image is
computed for each plane. TIFF files are slow to read frame-by-frame due to file format overhead, and the binary format
provides instant random access to any frame through memory mapping — essential for reading frames out of order or in 
parallel.

Reads:

| File / Data              | Description                                         |
|--------------------------|-----------------------------------------------------|
| `*.tif` / `*.tiff`       | Raw TIFF imaging files in the data directory        |
| `cindra_parameters.json` | Acquisition metadata (frame rate, planes, channels) |

Produces:

| File / Data                                         | Description                                                             |
|-----------------------------------------------------|-------------------------------------------------------------------------|
| `configuration.yaml`                                | Pipeline configuration copy (output root)                               |
| `acquisition_parameters.yaml`                       | Acquisition metadata copy (output root)                                 |
| `plane_<i>/channel_1_data.bin`                      | Binary imaging data for channel 1                                       |
| `plane_<i>/channel_2_data.bin`                      | Binary imaging data for channel 2 (if two-channel)                      |
| `plane_<i>/runtime_data.yaml`                       | Per-plane scalar metadata: frame dimensions, frame count, sampling rate |
| `plane_<i>/detection_data/mean_image.npy`           | Per-plane temporal mean image computed during binarization              |
| `plane_<i>/detection_data/mean_image_channel_2.npy` | Channel 2 mean image (if two-channel)                                   |

**Run via CLI:** `cindra run --input-path config.yaml --binarize`

#### Phase 2: Processing

Phase 2 runs four steps sequentially on each imaging plane: registration, detection, extraction (with classification),
and spike deconvolution. Each plane is processed independently, so multiple planes can be processed in parallel by
running separate `cindra run --process --target-plane <index>` commands.

**Run via CLI:** `cindra run --input-path config.yaml --process`

##### Registration (Motion Correction)

Registration aligns every frame in the recording to a stable reference image, correcting for brain motion that occurs
during imaging. Even small motion artifacts corrupt downstream analysis — if a cell drifts by a few pixels between
frames, its fluorescence trace will mix with signals from neighboring cells or neuropil. Registration ensures that
each pixel corresponds to the same physical location across all frames.

The algorithm proceeds in two stages. Rigid registration shifts each frame as a whole using phase correlation, and
optional nonrigid registration corrects local deformations by dividing the frame into blocks and aligning each block
independently.

Reads:

| File / Data                    | Description                                         |
|--------------------------------|-----------------------------------------------------|
| `plane_<i>/channel_1_data.bin` | Raw binary imaging data from binarization           |
| `plane_<i>/channel_2_data.bin` | Channel 2 binary data (two-channel recordings only) |
| `plane_<i>/runtime_data.yaml`  | Mean image and frame dimensions from binarization   |

Produces:

| File / Data                                          | Description                                   |
|------------------------------------------------------|-----------------------------------------------|
| `plane_<i>/channel_1_data.bin` (overwritten)         | Motion-corrected frames written back in place |
| `plane_<i>/registration_data/reference_image.npy`    | Alignment target computed from stable frames  |
| `plane_<i>/registration_data/rigid_y_offsets.npy`    | Per-frame vertical translation offsets        |
| `plane_<i>/registration_data/rigid_x_offsets.npy`    | Per-frame horizontal translation offsets      |
| `plane_<i>/registration_data/bad_frames.npy`         | Boolean mask flagging excessive-motion frames |
| `plane_<i>/registration_data/nonrigid_*_offsets.npy` | Per-block deformation offsets (when enabled)  |

When the `registration_metric_principal_components` configuration parameter is set above zero, the registration step
also computes principal component projections of the registered movie. These projections capture the dominant spatial
patterns of residual variance after motion correction. A well-registered recording should show principal components
dominated by neural activity rather than motion artifacts. The projections are saved as
`principal_component_projections.npy`, `principal_component_extreme_images.npy`, and
`principal_component_shift_metrics.npy` under `registration_data/`, and can be inspected interactively using the
registration quality GUI viewer (`cindra-gui registration`).

##### ROI Detection

Detection identifies regions of interest (ROIs) — typically neuronal cell bodies — in the registered imaging data.
Locating individual neurons is the prerequisite for extracting their activity. The sparse detection approach identifies
sources based on their spatiotemporal fluorescence patterns rather than morphological templates, making it robust to
variations in cell shape and brightness.

The algorithm temporally bins frames to improve signal-to-noise ratio, optionally applies PCA denoising, then runs a
sparse iterative detection procedure that identifies compact fluorescent sources. Detected ROIs are filtered by a
lightweight preclassification step, and shape statistics (area, compactness, aspect ratio) are computed for each
surviving ROI.

Reads:

| File / Data                                   | Description                               |
|-----------------------------------------------|-------------------------------------------|
| `plane_<i>/channel_1_data.bin`                | Motion-corrected binary data              |
| `plane_<i>/registration_data/bad_frames.npy`  | Bad-frame mask from registration          |
| Valid pixel ranges (from `runtime_data.yaml`) | Usable frame region after border cropping |

Produces:

| File / Data                                        | Description                                                |
|----------------------------------------------------|------------------------------------------------------------|
| `plane_<i>/roi_masks.npz`                          | Per-ROI pixel coordinates, weights, and centroids          |
| `plane_<i>/roi_statistics.npz`                     | Per-ROI shape properties (area, compactness, aspect ratio) |
| `plane_<i>/detection_data/mean_image.npy`          | Average of all registered frames                           |
| `plane_<i>/detection_data/enhanced_mean_image.npy` | Background-subtracted and contrast-normalized mean         |
| `plane_<i>/detection_data/maximum_projection.npy`  | Maximum intensity projection across all frames             |
| `plane_<i>/detection_data/correlation_map.npy`     | Pixel-wise correlation with neighboring pixels             |

##### Signal Extraction and Classification

Extraction pulls raw fluorescence time series from each detected ROI. Raw pixel values include contributions from
out-of-focus neuropil that must be removed to isolate each cell's true activity. Classification separates real neurons
from blood vessels, dendrite fragments, and noise artifacts, saving the researcher from manually curating potentially
thousands of ROIs.

For each ROI, a weighted spatial mask is created from its detected pixels, and a surrounding neuropil mask captures
local background fluorescence. The raw ROI trace is corrected by subtracting a scaled neuropil signal, and a baseline
is estimated and removed to produce a delta-fluorescence (dF) trace. A logistic regression classifier then scores each
ROI based on its shape statistics and fluorescence skewness, assigning a probability that it represents a genuine cell
rather than an artifact.

Reads:

| File / Data                    | Description                                       |
|--------------------------------|---------------------------------------------------|
| `plane_<i>/channel_1_data.bin` | Motion-corrected binary data for trace extraction |
| `plane_<i>/channel_2_data.bin` | Channel 2 data (two-channel recordings only)      |
| `plane_<i>/roi_masks.npz`      | ROI pixel masks from detection                    |
| `plane_<i>/roi_statistics.npz` | ROI shape properties from detection               |

Produces:

| File / Data                             | Description                                       |
|-----------------------------------------|---------------------------------------------------|
| `plane_<i>/cell_fluorescence.npy`       | Raw fluorescence time series per ROI              |
| `plane_<i>/neuropil_fluorescence.npy`   | Background fluorescence from surround masks       |
| `plane_<i>/subtracted_fluorescence.npy` | Neuropil-corrected and baseline-subtracted traces |
| `plane_<i>/cell_classification.npy`     | Cell probability per ROI                          |
| `plane_<i>/cell_colocalization.npy`     | Channel colocalization scores (two-channel only)  |

##### Spike Deconvolution

Deconvolution infers the underlying spike activity from each ROI's neuropil-corrected fluorescence trace using the
OASIS algorithm. Calcium fluorescence is a smoothed, delayed version of the underlying neural spiking activity.
Deconvolution recovers spike timing at a resolution finer than the indicator's decay time, enabling analyses that
depend on precise temporal relationships between neurons.

OASIS models the calcium indicator as an AR(1) exponential decay process: each spike produces a rapid fluorescence
increase that decays with time constant tau. The algorithm estimates when spikes occurred and their relative amplitudes
while enforcing a non-negativity constraint (fluorescence can only increase from a spike).

Reads:

| File / Data                             | Description                               |
|-----------------------------------------|-------------------------------------------|
| `plane_<i>/subtracted_fluorescence.npy` | Neuropil-corrected traces from extraction |

Produces:

| File / Data            | Description                                |
|------------------------|--------------------------------------------|
| `plane_<i>/spikes.npy` | Inferred spike amplitude per ROI per frame |

#### Phase 3: Combination

The combination phase merges the per-plane processing results into a single unified dataset. Multi-plane recordings
produce independent results per plane, and this step creates a single coordinate system and dataset that represents the
entire recording volume. The combined dataset is also the required input for the multi-recording pipeline.

Plane images are tiled into combined images using computed spatial offsets, ROI coordinates are adjusted to the combined
coordinate system, and fluorescence arrays are concatenated across planes.

Reads:

| File / Data                       | Description                                                    |
|-----------------------------------|----------------------------------------------------------------|
| `plane_<i>/runtime_data.yaml`     | Per-plane metadata for each processed plane                    |
| `plane_<i>/roi_masks.npz`         | Per-plane ROI masks                                            |
| `plane_<i>/roi_statistics.npz`    | Per-plane ROI shape statistics                                 |
| `plane_<i>/cell_fluorescence.npy` | Per-plane fluorescence traces (and all other extraction files) |
| `plane_<i>/detection_data/*.npy`  | Per-plane detection images                                     |

Produces:

| File / Data             | Description                                                 |
|-------------------------|-------------------------------------------------------------|
| `combined_metadata.npz` | Plane geometry, sampling rate, tau, registered binary paths |
| `roi_masks.npz`         | ROI masks with plane-adjusted coordinates                   |
| `roi_statistics.npz`    | ROI statistics tagged with source plane index               |
| `cell_fluorescence.npy` | Concatenated fluorescence traces across all planes          |
| `spikes.npy`            | Concatenated spike trains across all planes                 |
| `detection_data/*.npy`  | Combined detection images tiled across planes               |

**Run via CLI:** `cindra run --input-path config.yaml --combine`

### Multi-Recording Pipeline

The multi-recording pipeline tracks ROIs across multiple recordings of the same specimen captured on different days.
It requires that each recording has already been processed through the full single-recording pipeline. The pipeline
runs in two phases: discovery (identifying which ROIs correspond to the same cell across recordings) and extraction
(pulling fluorescence traces for tracked ROIs from each recording).

#### Phase 1: Discovery

The discovery phase performs four sequential steps across all recordings simultaneously.

**Run via CLI:** `cindra run --input-path md_config.yaml --discover`

##### ROI Selection

The first step filters each recording's detected ROIs to retain only high-confidence cells suitable for cross-recording
tracking. Including low-confidence ROIs or artifacts in the tracking step would produce spurious cross-recording
matches, so strict filtering ensures that only reliably detected neurons enter the alignment and clustering stages.
ROIs are filtered by their classification probability, pixel count, and (for MROI acquisitions) distance from MROI
region borders.

Reads:

| File / Data               | Description                                  |
|---------------------------|----------------------------------------------|
| `combined_metadata.npz`   | Plane geometry and registered binary paths   |
| `roi_statistics.npz`      | ROI shape properties from each recording     |
| `cell_classification.npy` | Cell probability per ROI from each recording |

Produces:

| File / Data                                                   | Description                                |
|---------------------------------------------------------------|--------------------------------------------|
| Selected ROI indices (in `multi_recording_runtime_data.yaml`) | Per-recording lists of passing ROI indices |

##### Cross-Recording Registration

The second step aligns the reference images from all recordings into a shared visual coordinate space. The same neuron
appears at slightly different positions across recording days due to tissue changes, slight repositioning of the
specimen, or slow biological drift. Diffeomorphic registration brings all recordings into spatial correspondence so
that ROIs from different days can be compared by their pixel overlap.

The algorithm uses diffeomorphic demons registration, a nonlinear image registration method that iteratively computes a
smooth, invertible deformation field for each recording. It operates on a multiscale image pyramid, starting from coarse
alignment and progressively refining at finer scales. B-spline regularization ensures the deformation remains smooth 
and diffeomorphic (no folding or tearing).

Reads:

| File / Data            | Description                                                                       |
|------------------------|-----------------------------------------------------------------------------------|
| `detection_data/*.npy` | Reference images (mean, enhanced mean, or maximum projection) from each recording |
| `roi_masks.npz`        | Selected ROI pixel masks from each recording                                      |

Produces:

| File / Data                              | Description                                       |
|------------------------------------------|---------------------------------------------------|
| `registration_arrays/deform_field_y.npy` | Vertical deformation field component              |
| `registration_arrays/deform_field_x.npy` | Horizontal deformation field component            |
| `registration_arrays/transformed_*.npy`  | Reference images warped to the shared space       |
| `registration_deformed_masks.npz`        | Forward-transformed ROI masks in the shared space |

##### ROI Tracking

The third step clusters spatially overlapping ROIs across recordings to identify cells that appear in multiple
sessions. This is the core step that enables longitudinal analysis. By identifying the same neuron across days,
researchers can study how neural representations evolve over time — whether cells maintain stable tuning, remap, or
drop in and out of the active population.

The algorithm divides the shared coordinate space into spatial bins and performs hierarchical clustering within each bin
using the Jaccard distance between ROI pixel masks (1 minus the intersection-over-union). Only cross-recording pairs
within a maximum centroid distance are considered as candidates. Clusters that appear in a sufficient fraction of 
recordings (controlled by `mask_prevalence`) are accepted as tracked templates. Template masks are constructed from 
the consensus pixels that appear in at least `pixel_prevalence` percent of cluster members.

Reads:

| File / Data                       | Description                                                                      |
|-----------------------------------|----------------------------------------------------------------------------------|
| `registration_deformed_masks.npz` | Forward-transformed ROI masks from each recording in the shared coordinate space |

Produces:

| File / Data                   | Description                                                                  |
|-------------------------------|------------------------------------------------------------------------------|
| `tracking_template_masks.npz` | Consensus template masks with source ROI and recording metadata per template |

##### Template Projection

The fourth step projects the tracked template masks from the shared coordinate space back into each recording's native
coordinates by inverting the diffeomorphic deformation field. Fluorescence extraction must operate on the original 
registered binary data, which is in each recording's native coordinate space. The inverse projection ensures that 
template masks align precisely with the recorded pixel data. Full ROI statistics (shape metrics, spatial properties) 
are recomputed for each projected template in native coordinates.

Reads:

| File / Data                              | Description                                                 |
|------------------------------------------|-------------------------------------------------------------|
| `tracking_template_masks.npz`            | Consensus template masks in the shared coordinate space     |
| `registration_arrays/deform_field_*.npy` | Per-recording deformation fields for inverse transformation |

Produces:

| File / Data          | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `roi_masks.npz`      | Template masks projected to native coordinates per recording |
| `roi_statistics.npz` | Shape statistics for projected templates                     |

All Phase 1 results are persisted under `multi_recording/<dataset_name>/` within each recording's cindra output
directory, along with a `multi_recording_runtime_data.yaml` file and a copy of the multi-recording configuration.

#### Phase 2: Multi-Recording Extraction

The extraction phase pulls fluorescence traces from the tracked template ROIs in each recording. The discovery phase
identifies *which* cells are present across recordings; the extraction phase recovers *what those cells did* during
each recording session. The result is a set of aligned fluorescence traces for the same neurons across multiple days.

This step uses the same extraction pipeline as the single-recording phase: mask creation, fluorescence extraction,
neuropil correction, baseline subtraction, and optional spike deconvolution. It operates on the backward-projected
template masks instead of the originally detected ROIs. Since tracked ROIs are already confirmed cells, no
reclassification is performed.

Reads:

| File / Data                    | Description                                                 |
|--------------------------------|-------------------------------------------------------------|
| `roi_masks.npz`                | Backward-transformed template masks in native coordinates   |
| `plane_<i>/channel_1_data.bin` | Motion-corrected binary data from single-recording pipeline |
| `plane_<i>/channel_2_data.bin` | Channel 2 data (two-channel recordings only)                |

Produces:

| File / Data                   | Description                                            |
|-------------------------------|--------------------------------------------------------|
| `cell_fluorescence.npy`       | Fluorescence traces for tracked ROIs in this recording |
| `neuropil_fluorescence.npy`   | Background fluorescence from surround masks            |
| `subtracted_fluorescence.npy` | Neuropil-corrected and baseline-subtracted traces      |
| `spikes.npy`                  | Spike amplitudes for tracked ROIs (when enabled)       |

**Run via CLI:** `cindra run --input-path md_config.yaml --extract`

Each recording is extracted independently, enabling parallel execution across recordings by running separate
`cindra run --extract --target-recording <recording_id>` commands.

### API

The library exposes a high-level Python API for programmatic pipeline execution. The two primary entry points are
`run_single_recording_pipeline()` and `run_multi_recording_pipeline()`, which accept YAML configuration files and
support executing specific pipeline phases.

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

# Execute individual phases for finer control.
run_single_recording_pipeline(configuration_path=Path("/path/to/config.yaml"), binarize=True)
run_single_recording_pipeline(configuration_path=Path("/path/to/config.yaml"), process=True)
run_single_recording_pipeline(configuration_path=Path("/path/to/config.yaml"), combine=True)

# For multi-recording pipelines, configure and run similarly.
md_config = MultiRecordingConfiguration()
md_config.to_yaml(Path("/path/to/md_config.yaml"))
run_multi_recording_pipeline(configuration_path=Path("/path/to/md_config.yaml"))
```

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

| Environment  | Description                                                |
|--------------|------------------------------------------------------------|
| `lint`       | Runs ruff formatting, ruff linting, and mypy type checking |
| `stubs`      | Generates py.typed marker and .pyi stub files              |
| `py314-test` | Runs the test suite via pytest for Python 3.14             |
| `coverage`   | Aggregates test coverage into an HTML report               |
| `docs`       | Builds the API documentation via Sphinx                    |
| `build`      | Builds sdist and wheel distributions                       |
| `upload`     | Uploads distributions to PyPI via twine                    |
| `install`    | Builds and installs the project into its mamba environment |
| `uninstall`  | Uninstalls the project from its mamba environment          |
| `create`     | Creates the project's mamba development environment        |
| `remove`     | Removes the project's mamba development environment        |
| `provision`  | Recreates the mamba environment from scratch               |
| `export`     | Exports the mamba environment as .yml and spec.txt files   |
| `import`     | Creates or updates the mamba environment from a .yml file  |

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
