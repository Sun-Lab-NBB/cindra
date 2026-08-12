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

For the original suite2p algorithm documentation, see the
[original suite2p settings reference](https://suite2p.readthedocs.io/en/latest/settings.html).

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

- Supports Windows, Linux, and macOS.
- Supports Python 3.14 with full type annotations and MyPy type checking.
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
- Uses Numba JIT compilation with Intel TBB threading (OpenMP on macOS) for parallelized frame-level computation.
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
    - [Phase 2: Registration](#phase-2-registration)
    - [Phase 3: Processing](#phase-3-processing)
    - [Phase 4: Combination](#phase-4-combination)
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

On macOS, cindra uses Numba's OpenMP threading layer, because the Numba macOS wheel carries no TBB support. The
OpenMP runtime (`libomp.dylib`) ships with neither Numba nor macOS itself, so it is installed separately.

Run `cindra omp` to report the runtimes present on the host, and `sudo cindra omp --yes` to make one loadable. The
command finds runtimes installed by [Homebrew](https://brew.sh/) or MacPorts, present in the active conda environment,
or carried inside an installed Python package. Install one with `brew install libomp` when the command finds none.
Without a loadable runtime, processing fails once it reaches a parallelized stage. Linux and Windows run the TBB
threading layer, which needs no additional steps, so `cindra omp` errors when run on them.

For users, all other library dependencies are installed automatically by all supported installation methods. For
developers, see the [Developers](#developers) section for information on installing additional development dependencies.

___

## Installation

### Source

***Note,*** installation from source is ***highly discouraged*** for anyone who is not an active project developer.

1. Download this repository to the local machine using the preferred method, such as git-cloning. Use one of the
   [stable releases](https://github.com/Sun-Lab-NBB/cindra/tags) that include precompiled binary and source code
   distribution (sdist) wheels.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. `cd` to the root directory of the prepared project distribution.
4. Run `pip install .` to install the project and its dependencies.

### pip

Use the following command to install the library and all of its dependencies via [pip](https://pip.pypa.io/en/stable/):
`pip install cindra`

___

## Usage

### Input Data Format

Cindra processes two-photon (or one-photon) calcium imaging data stored as TIFF files. Before running any pipeline,
the raw data directory must be prepared with the correct structure.

#### TIFF Files

Every TIFF file in the directory must hold frames of the same shape. Binarization checks this before conversion and
fails with an error naming the offending files if any differ, because a foreign file, most commonly an anatomical
z-stack stored alongside the functional recording, would otherwise corrupt the interleave accounting. Exclude any such
file with the `file_io.ignored_file_names` configuration parameter, which matches on the file stem without its
extension.

The pipeline expects a flat directory containing one or more `.tif` / `.tiff` files. For multi-plane or multichannel
acquisitions, frames must be interleaved in the following order within each TIFF file: plane0_channel1, plane0_channel2,
plane1_channel1, plane1_channel2, and so on, repeating for each time point. This interleaving pattern continues
seamlessly across TIFF file boundaries when a recording spans multiple files.

One interleave cycle carries one frame of every plane on every channel, and binarization consumes whole cycles, so the
total frame count across every TIFF file should be a multiple of `plane_number * channel_number`. The frames of a final
incomplete cycle reach some planes and channels and not others, so binarization discards them and logs a warning naming
how many it dropped. Stopping an acquisition on a volume boundary is what keeps every frame it collected. A recording
whose TIFF files hold fewer frames than one whole cycle is rejected with an error before any binary is written.

For MROI (multi-region of interest) line-scanning acquisitions, each raw TIFF frame must contain the full imaging strip
with all ROI regions arranged vertically. The interleaving order across planes and channels is the same as standard
acquisitions. During binarization, the pipeline uses the `roi_lines` field from `cindra_parameters.json` to slice each
frame into region-specific strips. Each ROI-plane combination becomes a separate virtual plane for processing.

#### Acquisition Parameters

Each raw data directory must contain a `cindra_parameters.json` file that describes how the data was acquired. This
file can be generated using the `generate_acquisition_parameters_file_tool` [MCP tool](#mcp-servers) or constructed
manually. The required fields are:

| Field            | Type  | Description                                                           |
|------------------|-------|-----------------------------------------------------------------------|
| `frame_rate`     | float | Volume acquisition rate in Hz (rate across all planes, not per-plane) |
| `plane_number`   | int   | Number of physical imaging planes                                     |
| `channel_number` | int   | Number of channels per plane (1 or 2)                                 |

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

```text
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
| `runtime`                 | Progress bar display during long-running processing steps                                                                   |
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
| `runtime`                    | Progress bar display during long-running processing steps                                                                |
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
"registered" binary files. `channel_1_data.bin` is overwritten in place with motion-corrected data.

Each stage that writes frames into a binary guards the write with a marker file beside it, which it removes once every
frame is in place. Binarization writes `channel_1_data.bin.binarizing` and registration writes
`channel_1_data.bin.registering`, each suffix spelling the phase the way the reported job status spells it.
Binarization sizes the binary to its full frame count before writing its first frame, and registration rewrites the
binary it reads. An interrupted run of either stage therefore leaves a correctly sized file whose contents are
indeterminate, and the marker is the only record of that state. Both markers mean the same thing to the pipeline, and
the two names exist so that whoever finds one on disk reads which phase died. Registration refuses to run against a
binary carrying either marker. Re-run the binarization phase to rebuild the binary from its source TIFF files, which
also clears the marker.

Binarization consumes whole plane-and-channel interleave cycles and discards any frame past the last whole cycle. Every
plane binary of a recording the current code converted therefore holds the same number of frames, and the two channels
of one plane stay aligned frame for frame. A recording converted by an earlier version can hold planes, or the two
channels of one plane, at unequal lengths.

| File                 | Format               | Description                                                                            |
|----------------------|----------------------|----------------------------------------------------------------------------------------|
| `channel_1_data.bin` | int16 (frames, h, w) | Channel 1 imaging frames (raw after binarization, motion-corrected after registration) |
| `channel_2_data.bin` | int16 (frames, h, w) | Channel 2 imaging frames (two-channel recordings only)                                 |

#### Per-Plane Registration Data

Stored under `plane_<i>/registration_data/`:

| File                                     | Format                       | Description                                              |
|------------------------------------------|------------------------------|----------------------------------------------------------|
| `reference_image.npy`                    | float32 (h, w)               | Alignment target computed from the most stable frames    |
| `rigid_y_offsets.npy`                    | int32 (frames,)              | Per-frame vertical translation from phase correlation    |
| `rigid_x_offsets.npy`                    | int32 (frames,)              | Per-frame horizontal translation from phase correlation  |
| `rigid_correlations.npy`                 | float32 (frames,)            | Phase correlation quality per frame                      |
| `bad_frames.npy`                         | bool (frames,)               | Flags frames with excessive motion                       |
| `nonrigid_y_offsets.npy`                 | float32 (frames, num_blocks) | Per-block vertical offsets (when nonrigid enabled)       |
| `nonrigid_x_offsets.npy`                 | float32 (frames, num_blocks) | Per-block horizontal offsets (when nonrigid enabled)     |
| `nonrigid_correlations.npy`              | float32 (frames, num_blocks) | Per-block correlation quality (when nonrigid enabled)    |
| `principal_component_projections.npy`    | float32 (samples, n_pcs)     | Projections of subsampled frames onto PCs (when enabled) |
| `principal_component_extreme_images.npy` | float32 (2, n_pcs, h, w)     | Mean images of low/high projection frames per PC         |
| `principal_component_shift_metrics.npy`  | float32 (n_pcs, 3)           | Registration quality metrics per PC                      |

#### Per-Plane Detection Data

Stored under `plane_<i>/detection_data/`:

| File                      | Format         | Description                                        |
|---------------------------|----------------|----------------------------------------------------|
| `mean_image.npy`          | float32 (h, w) | Average of all registered frames                   |
| `enhanced_mean_image.npy` | float32 (h, w) | Background-subtracted and contrast-normalized mean |
| `maximum_projection.npy`  | float32 (h, w) | Maximum intensity projection across all frames     |
| `correlation_map.npy`     | float32 (h, w) | Pixel-wise correlation with neighboring pixels     |

The `mean_image_channel_2.npy` variant is written for any two-channel recording. The remaining channel 2 variants
(`enhanced_mean_image_channel_2.npy`, `maximum_projection_channel_2.npy`, `correlation_map_channel_2.npy`) are saved
only when both channels are functional.

#### Per-Plane ROI and Extraction Data

Stored under `plane_<i>/`:

| File                                  | Format                   | Description                                                                                                                          |
|---------------------------------------|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `roi_masks.npz`                       | variable-length arrays   | Per-ROI pixel coordinates, weights, centroids                                                                                        |
| `roi_statistics.npz`                  | float arrays             | Per-ROI shape properties (compactness, solidity, aspect ratio, skewness)                                                             |
| `cell_fluorescence.npy`               | float32 (n_rois, frames) | Raw fluorescence time series per ROI                                                                                                 |
| `neuropil_fluorescence.npy`           | float32 (n_rois, frames) | Background fluorescence from surround masks                                                                                          |
| `subtracted_fluorescence.npy`         | float32 (n_rois, frames) | Neuropil-corrected and baseline-subtracted traces                                                                                    |
| `spikes.npy`                          | float32 (n_rois, frames) | Inferred spike amplitudes from OASIS                                                                                                 |
| `cell_classification.npy`             | float32 (n_rois, 2)      | Column 0: is_cell label (1.0 or 0.0), column 1: classifier probability                                                               |
| `cell_colocalization.npy`             | float32 (n_rois, 2)      | Two-channel only: (flag, probability) if channel 2 is structural, or (matched channel 2 index, overlap score) if both are functional |
| `corrected_structural_mean_image.npy` | float32 (h, w)           | Bleed-through-corrected structural mean (intensity colocalization)                                                                   |

Channel 2 variants (`cell_fluorescence_channel_2.npy`, etc.) are saved when both channels are functional.

#### Combined Data

Stored at `<output_path>/cindra/`:

| File                    | Description                                                                           |
|-------------------------|---------------------------------------------------------------------------------------|
| `combined_metadata.npz` | Plane geometry, sampling rate, tau, combined and per-plane frame counts, binary paths |
| `roi_masks.npz`         | ROI masks with coordinates adjusted to the combined coordinate system                 |
| `roi_statistics.npz`    | ROI statistics tagged with source plane index                                         |
| `cell_fluorescence.npy` | Fluorescence traces across all planes, trimmed to the shortest plane's frame count    |
| `spikes.npy`            | Concatenated spike trains across all planes                                           |
| `detection_data/`       | Combined detection images (mean, enhanced mean, maximum projection, correlation map)  |

The same set of extraction files (`neuropil_fluorescence.npy`, `subtracted_fluorescence.npy`, `cell_classification.npy`,
and channel 2 variants) follows the same naming convention at the combined level.

#### Multi-Recording Data

Stored under `<recording_directory>/cindra/multi_recording/<dataset_name>/` per recording:

| File / Directory                         | Description                                               |
|------------------------------------------|-----------------------------------------------------------|
| `multi_recording_runtime_data.yaml`      | Per-recording runtime metadata and timing                 |
| `multi_recording_configuration.yaml`     | Shared configuration (main recording only)                |
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
| `cell_colocalization.npy`                | Channel colocalization scores (two-channel only)          |

### Single-Recording Pipeline

The single-recording pipeline processes a single calcium imaging session through four sequential phases: binarization,
registration, processing, and combination. Phase 2 (registration) and Phase 3 (processing) both run independently per
imaging plane, enabling parallel execution across planes. Each plane must be registered before it is processed, because
detection reads the valid pixel ranges that registration computes.

#### Phase 1: Binarization

The binarization phase converts raw TIFF files into an internal memory-mapped binary format that the rest of the
pipeline reads from. During conversion, interleaved frames are separated by plane and channel, and a mean image is
computed for each plane. TIFF files are slow to read frame-by-frame due to file format overhead, and the binary format
provides instant random access to any frame through memory mapping, which is essential for reading frames out of order
or in parallel.

Binarization has two outcomes: it skips the conversion, or it rebuilds every plane binary from the source TIFFs. The
conversion is skipped when every plane holds a channel 1 binary whose size matches the geometry recorded for that plane.
A rebuild follows a missing channel 1 binary, a `.binarizing` or `.registering` marker left beside either channel's
binary by an interrupted conversion or registration, or a binary on disk whose size disagrees with its plane's recorded
frame geometry. That last case is what a truncation outside the pipeline leaves behind. Enabling the
`file_io.repeat_binarization` configuration parameter rebuilds an otherwise valid recording. Only channel 1 has to
exist, so a recording whose channel 2 binaries were deleted is skipped rather than rebuilt, and that parameter is what
restores them. Re-running this phase is therefore the recovery path for both an interrupted conversion and an
interrupted registration, and it needs no configuration change.

A rebuild replaces every plane binary of the recording, so it first discards everything the pipeline measured from the
previous ones: each plane's registration and detection output, its extracted traces, and the recording's combined
dataset. That discard covers every plane directory the output directory holds, including one the declared plane count no
longer reaches, whose own binary the rebuild leaves alone. The rebuilt binaries hold raw frames again, so every plane
has to be registered and processed once more before the recording can be combined. The discard follows the resolution of
every source file and destination the conversion needs, so a rebuild that the source files reject, such as one whose
TIFF files disagree about their frame shape, leaves the previous results in place.

A recording converted by an earlier version can hold planes, or the two channels of one plane, at unequal lengths,
because the frames of its final incomplete cycle reached some planes and channels and not others. A plane whose two
channels received different counts holds one binary disagreeing with the frame count recorded for that plane, so this
phase reports it as malformed and rebuilds the whole recording without `file_io.repeat_binarization` being set. Every
other such recording, including every single-channel one, has each binary matching the count recorded for its own
plane, so this phase skips the conversion and leaves the plane lengths unequal. Rebuilding one of those at equal
lengths takes `file_io.repeat_binarization`, and either path costs a full reprocessing run.

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

#### Phase 2: Registration

Phase 2 motion-corrects one imaging plane and computes the principal components used to review the registration
quality. Each plane is registered independently, so multiple planes can be registered in parallel by running separate
`cindra run --register --target-plane <index>` commands. This phase is the prerequisite of Phase 3.

**Run via CLI:** `cindra run --input-path config.yaml --register`

##### Registration (Motion Correction)

Registration aligns every frame in the recording to a stable reference image, correcting for brain motion that occurs
during imaging. Even small motion artifacts corrupt downstream analysis. If a cell drifts by a few pixels between
frames, its fluorescence trace mixes with signals from neighboring cells or neuropil. Registration ensures that
each pixel corresponds to the same physical location across all frames.

The algorithm proceeds in two stages. Rigid registration shifts each frame as a whole using phase correlation, and
optional nonrigid registration corrects local deformations by dividing the frame into blocks and aligning each block
independently.

Reads:

| File / Data                    | Description                                         |
|--------------------------------|-----------------------------------------------------|
| `plane_<i>/channel_1_data.bin` | Raw binary imaging data from binarization           |
| `plane_<i>/channel_2_data.bin` | Channel 2 binary data (two-channel recordings only) |
| `plane_<i>/runtime_data.yaml`  | Frame dimensions, frame count, and sampling rate    |

Produces:

| File / Data                                          | Description                                   |
|------------------------------------------------------|-----------------------------------------------|
| `plane_<i>/channel_1_data.bin` (overwritten)         | Motion-corrected frames written back in place |
| `plane_<i>/registration_data/reference_image.npy`    | Alignment target computed from stable frames  |
| `plane_<i>/registration_data/rigid_y_offsets.npy`    | Per-frame vertical translation offsets        |
| `plane_<i>/registration_data/rigid_x_offsets.npy`    | Per-frame horizontal translation offsets      |
| `plane_<i>/registration_data/bad_frames.npy`         | Boolean mask flagging excessive-motion frames |
| `plane_<i>/registration_data/nonrigid_*_offsets.npy` | Per-block deformation offsets (when enabled)  |

When the `registration_metric_principal_components` configuration parameter is set above zero and the recording contains
at least 1500 frames, the registration step computes principal component projections of the registered movie. These
projections capture the dominant spatial patterns of residual variance after motion correction. A well-registered
recording should show principal components dominated by neural activity rather than motion artifacts. The projections
are saved as `principal_component_projections.npy`, `principal_component_extreme_images.npy`, and
`principal_component_shift_metrics.npy` under `registration_data/`, and can be inspected interactively using the
registration quality GUI viewer (`cindra-gui registration`).

#### Phase 3: Processing

Phase 3 runs three steps sequentially on each registered imaging plane: detection, extraction (with classification),
and spike deconvolution. Each plane is processed independently, so multiple planes can be processed in parallel by
running separate `cindra run --process --target-plane <index>` commands. Processing a plane that Phase 2 has not
registered raises an error rather than detecting ROIs on uncorrected data.

**Run via CLI:** `cindra run --input-path config.yaml --process`

##### ROI Detection

Detection identifies regions of interest (ROIs), typically neuronal cell bodies, in the registered imaging data.
Locating individual neurons is the prerequisite for extracting their activity. The sparse detection approach identifies
sources based on their spatiotemporal fluorescence patterns rather than morphological templates, making it robust to
variations in cell shape and brightness.

The algorithm temporally bins frames to improve signal-to-noise ratio, optionally applies PCA denoising, then runs a
sparse iterative detection procedure that identifies compact fluorescent sources. Detected ROIs are optionally filtered
by a lightweight preclassification step (when the preclassification threshold is above zero), and shape statistics
(area, compactness, aspect ratio) are computed for each surviving ROI.

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

#### Phase 4: Combination

The combination phase merges the per-plane processing results into a single unified dataset. Multi-plane recordings
produce independent results per plane, and this step creates a single coordinate system and dataset that represents the
entire recording volume. The combined dataset is also the required input for the multi-recording pipeline.

Plane images are tiled into combined images using computed spatial offsets, ROI coordinates are adjusted to the combined
coordinate system, and fluorescence arrays are concatenated across planes.

Binarization gives every plane of a recording converted in one run the same frame count, so the `frame_count` and
`plane_frame_counts` keys of `combined_metadata.npz` agree for a recording whose planes all completed. `frame_count` is
the number of frames the combined traces span, and each `plane_frame_counts` entry is the number its plane's own traces
and binaries span. That per-plane entry is what a consumer reading one plane directly needs without opening that plane's
`runtime_data.yaml`. The combined traces are trimmed to the shortest contributing plane rather than padded to the
longest, which keeps every combined frame backed by real data on every plane. Planes whose processing phase did not
complete contribute nothing and are excluded from the trim target.

`combined_metadata.npz` also doubles as the marker that downstream consumers, including the multi-recording pipeline,
check to decide whether the single-recording pipeline completed. It is therefore written after every array it describes
and published through an atomic write that renames it into place, so an interrupted run never leaves a
marker describing a payload that is not on disk.

Reads:

| File / Data                       | Description                                                    |
|-----------------------------------|----------------------------------------------------------------|
| `plane_<i>/runtime_data.yaml`     | Per-plane metadata for each processed plane                    |
| `plane_<i>/roi_masks.npz`         | Per-plane ROI masks                                            |
| `plane_<i>/roi_statistics.npz`    | Per-plane ROI shape statistics                                 |
| `plane_<i>/cell_fluorescence.npy` | Per-plane fluorescence traces (and all other extraction files) |
| `plane_<i>/detection_data/*.npy`  | Per-plane detection images                                     |

Produces:

| File / Data             | Description                                                                           |
|-------------------------|---------------------------------------------------------------------------------------|
| `combined_metadata.npz` | Plane geometry, sampling rate, tau, `frame_count`, `plane_frame_counts`, binary paths |
| `roi_masks.npz`         | ROI masks with plane-adjusted coordinates                                             |
| `roi_statistics.npz`    | ROI statistics tagged with source plane index                                         |
| `cell_fluorescence.npy` | Concatenated fluorescence traces across all planes                                    |
| `spikes.npy`            | Concatenated spike trains across all planes                                           |
| `detection_data/*.npy`  | Combined detection images tiled across planes                                         |

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
recordings. This is the core step that enables longitudinal analysis. By identifying the same neuron across days,
researchers can study how neural representations evolve over time, whether cells maintain stable tuning, remap, or
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
directory, along with a per-recording `multi_recording_runtime_data.yaml` file. A single copy of the multi-recording
configuration (`multi_recording_configuration.yaml`) is written once to the main (first) recording's directory.

#### Phase 2: Multi-Recording Extraction

The extraction phase pulls fluorescence traces from the tracked template ROIs in each recording. The discovery phase
identifies *which* cells are present across recordings. The extraction phase recovers *what those cells did* during
each recording. The result is a set of aligned fluorescence traces for the same neurons across multiple days.

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

# Execute the full single-recording pipeline (binarize, register, process, combine).
run_single_recording_pipeline(configuration_path=Path("/path/to/config.yaml"))

# Execute individual phases for finer control, each with its own worker allocation.
run_single_recording_pipeline(configuration_path=Path("/path/to/config.yaml"), binarize=True)
run_single_recording_pipeline(configuration_path=Path("/path/to/config.yaml"), register=True, registration_workers=8)
run_single_recording_pipeline(configuration_path=Path("/path/to/config.yaml"), process=True, processing_workers=10)
run_single_recording_pipeline(configuration_path=Path("/path/to/config.yaml"), combine=True)

# For multi-recording pipelines, configure and run similarly.
md_config = MultiRecordingConfiguration()
md_config.to_yaml(Path("/path/to/md_config.yaml"))
run_multi_recording_pipeline(configuration_path=Path("/path/to/md_config.yaml"))

# Multi-recording phases also take their own worker allocations.
run_multi_recording_pipeline(configuration_path=Path("/path/to/md_config.yaml"), discover=True, discovery_workers=30)
run_multi_recording_pipeline(configuration_path=Path("/path/to/md_config.yaml"), extract=True, extraction_workers=16)
```

Every phase takes its worker count as a direct API parameter rather than a configuration field, which keeps the
configuration file immutable and therefore safe to share between concurrently dispatched jobs. Omitting a worker
parameter gives the phase its measured default (`BINARIZATION_WORKERS`, `REGISTRATION_WORKERS`, `PROCESSING_WORKERS`,
`DISCOVERY_WORKERS`, `EXTRACTION_WORKERS`, all exported from `cindra`), and passing `-1` requests every available core.

External schedulers that need to enumerate a recording's jobs and their dependencies without driving the pipeline
themselves can read the phase model exported from `cindra.orchestration`. `SINGLE_RECORDING_PHASES` and
`MULTI_RECORDING_PHASES` describe the ordered phases, `resolve_single_recording_jobs()` and
`resolve_multi_recording_jobs()` expand them into a job universe of `(job_name, specifier)` pairs, and
`resolve_single_recording_prerequisites()` and `resolve_multi_recording_prerequisites()` return the jobs each job
depends on. This keeps a scheduler's view of the pipeline in step with the library rather than restating it.

A scheduler that also wants to know which of those jobs can run right now, where their inputs and outputs live, and
how much memory each one holds reads three further groups, all exported from `cindra`.

`resolve_single_recording_job_universe()` and `resolve_multi_recording_job_universe()` pair the phase model with the
inventory on disk, returning both the jobs a recording declares and the subset whose own inputs already exist.
`resolve_recording_planes()` and `resolve_dataset_recordings()` report the planes a recording holds and the recordings
a dataset spans without building a runtime context or creating a directory, and `is_recording_processed()`,
`is_plane_registered()`, and `is_dataset_discovered()` answer the same questions one at a time.

`resolve_output_path()`, `resolve_plane_path()`, and `resolve_dataset_path()` build the three roots the pipelines write
under a caller-supplied output root, and `resolve_array_path()` names a file inside one of them. The result arrays sit
directly in those roots and are named by `RecordingArrays`, while `DetectionImages`, `RegistrationArrays`, and
`MultiRecordingArrays` name files inside the `detection_data`, `registration_data`, and `registration_arrays`
subdirectories, whose names the same module exports. `resolve_plane_specifier()` and
`parse_plane_specifier()` convert between a plane index and the specifier its jobs and its directory both carry.

`estimate_single_recording_job_memory_mb()` and `estimate_multi_recording_job_memory_mb()` project the memory one job
holds from the shape of the data it will process, returning the figure in megabytes alongside a flag stating whether it
followed from the recording's own geometry. A job whose geometry cannot be read is charged a conservative allowance for its
stage rather than a floor, because understating is the failure that gets a job killed. Two of those allowances are
measured peaks and four are flat figures the module documents individually.

`prime_recording()` and `prime_dataset()` write the shared bootstrap every job reads and report that same inventory, so
a scheduler primes a recording and enumerates its jobs in one step.

### CLI Commands

This library provides the `cindra` and `cindra-gui` CLIs that expose the following commands:

#### cindra

| Command     | Description                                                                        |
|-------------|------------------------------------------------------------------------------------|
| `configure` | Generates default YAML configuration files for single or multi-recording pipelines |
| `run`       | Executes a pipeline using a YAML configuration file with optional CLI overrides    |
| `mcp`       | Starts the data processing MCP server for AI agent integration                     |
| `omp`       | Links the OpenMP runtime Numba loads on macOS, erroring on every other platform    |

The `run` command supports executing individual pipeline phases (`--binarize`, `--register`, `--process`, `--combine`
for single-recording, `--discover` and `--extract` for multi-recording), targeting specific planes (`--target-plane`) or
recordings (`--target-recording`), and allocating workers per phase (`--binarize-workers`, `--register-workers`,
`--process-workers`, `--discover-workers`, `--extract-workers`). Omitting a worker option gives that phase its measured
default allocation, passing `-1` requests every available core, and any positive value is used exactly. The combination
phase takes no worker option because it merges the per-plane result files with serial input and output.

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
| `generate_acquisition_parameters_file_tool`       | Generates a `cindra_parameters.json` file for a recording directory |
| `validate_acquisition_parameters_file_tool`       | Validates an existing acquisition parameters file                   |
| `validate_recording_readiness_tool`               | Validates that a recording is ready for pipeline processing         |
| `generate_config_file_tool`                       | Generates a default YAML configuration file                         |
| `discover_recordings_tool`                        | Discovers all recordings under a root directory                     |
| `resolve_dataset_name_tool`                       | Constructs a qualified multi-recording dataset name                 |
| `read_config_file_tool`                           | Reads and returns the contents of a configuration YAML file         |
| `validate_config_file_tool`                       | Validates a configuration file and reports non-default parameters   |
| `get_recording_status_tool`                       | Gets the processing status of a single recording                    |
| `get_batch_status_overview_tool`                  | Gets an overview of batch processing status across recordings       |
| `prepare_single_recording_batch_tool`             | Prepares single-recording batch processing jobs without execution   |
| `prepare_multi_recording_batch_tool`              | Prepares multi-recording batch processing jobs without execution    |
| `reset_processing_phases_tool`                    | Resets completed processing phases for re-execution                 |
| `clean_processing_output_tool`                    | Deletes processing output artifacts for clean re-processing         |
| `execute_processing_jobs_tool`                    | Dispatches prepared processing jobs with per-class core allocation  |
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
[cindra](https://github.com/Sun-Lab-NBB/cindra) marketplace as part of the **cindra** plugin. Install the plugin from
the marketplace to automatically register both MCP servers with compatible clients and make all associated skills
available.

___

## API Documentation

See the [API documentation](https://cindra-api-docs.netlify.app/) for the detailed description of the methods and
classes exposed by components of this library.

___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to modify
the source code of this library.

### Installing the Project

***Note,*** this installation method requires **mamba version 2.3.2 or above**. Currently, all cindra automation
pipelines require that mamba is installed through the [miniforge3](https://github.com/conda-forge/miniforge) installer.

1. Download this repository to the local machine using the preferred method, such as git-cloning.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. `cd` to the root directory of the prepared project distribution.
4. Install the core cindra development dependencies into the ***base*** mamba environment via the
   `mamba install tox uv tox-uv` command.
5. Use the `tox -e create` command to create the project-specific development environment followed by `tox -e install`
   command to install the project into that environment as a library.

### Additional Dependencies

In addition to installing the project and all user dependencies, install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version supported by the developed project.
   Currently, this library supports Python 3.14 only. It is recommended to use a tool like
   [pyenv](https://github.com/pyenv/pyenv) to install and manage the required versions.

### Development Automation

This project uses `tox` for development automation. The following tox environments are available:

| Environment  | Description                                                 |
|--------------|-------------------------------------------------------------|
| `lint`       | Runs ruff formatting, ruff linting, and mypy type checking  |
| `stubs`      | Generates py.typed marker and .pyi stub files               |
| `py314-test` | Runs the test suite via pytest for Python 3.14              |
| `coverage`   | Aggregates test coverage and applies the 100% coverage gate |
| `docs`       | Builds the API documentation via Sphinx                     |
| `build`      | Builds sdist and wheel distributions                        |
| `upload`     | Uploads distributions to PyPI via twine                     |
| `deploy`     | Uploads the built documentation to the Netlify site         |
| `install`    | Builds and installs the project into its mamba environment  |
| `uninstall`  | Uninstalls the project from its mamba environment           |
| `create`     | Creates the project's mamba development environment         |
| `remove`     | Removes the project's mamba development environment         |
| `provision`  | Recreates the mamba environment from scratch                |
| `export`     | Exports the mamba environment as a .yml file                |
| `import`     | Creates or updates the mamba environment from a .yml file   |

Run any environment using `tox -e ENVIRONMENT`. For example, `tox -e lint`.

***Note,*** all pull requests for this project have to successfully complete the `tox` task before being merged. To
expedite the task's runtime, use the `tox --parallel` command to run some tasks in parallel.

### AI-Assisted Development

Claude Code skills and AI development assets for this project are distributed through two marketplaces:

- [cindra](https://github.com/Sun-Lab-NBB/cindra) marketplace: Provides MCP server registrations, pipeline-specific
  skills for single-recording and multi-recording processing, configuration, results inspection, visualization, and MCP
  environment setup. Install this marketplace to register the `cindra mcp` and `cindra-gui mcp` servers with
  compatible MCP clients and make all pipeline workflow skills available.
- [ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) marketplace: Provides shared development skills that enforce
  cindra coding conventions (Python style, README style, commit messages, pyproject.toml, tox configuration) and
  general-purpose codebase exploration tools via the **automation** plugin.

Install both marketplaces to make all associated skills and development tools available to compatible AI coding agents.

### Automation Troubleshooting

Many packages used in `tox` automation pipelines (uv, mypy, ruff) and `tox` itself may experience runtime failures. In
most cases, this is related to their caching behavior. If an unintelligible error is encountered with any of the
automation components, deleting the corresponding cache directories (`.tox`, `.ruff_cache`, `.mypy_cache`, etc.)
manually or via a CLI command typically resolves the issue.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the
[tags on this repository](https://github.com/Sun-Lab-NBB/cindra/tags) for the available project releases.

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
  [multi-recording pipeline](https://github.com/sprustonlab/multiday-suite2p-public), whose algorithms were
  reimplemented in this library.
- Elaine Wu for contributing to the early reimplementation of the I/O module.
- Almar Klein, author of the original [pirt](https://github.com/almarklein/pirt) library, whose diffeomorphic
  registration algorithms were reimplemented to form the basis of the multi-recording registration module.
- The creators of all other dependencies and projects listed in the [pyproject.toml](pyproject.toml) file.

___
