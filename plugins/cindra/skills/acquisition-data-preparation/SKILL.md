---
name: acquisition-data-preparation
description: >-
  Guides agents through preparing raw imaging data for cindra processing. Covers creating and validating
  cindra_parameters.json acquisition parameter files, gathering acquisition metadata from users, converting data from
  common microscope formats into cindra-compatible TIFFs, and organizing data directories. Use when the user needs to
  prepare data for processing, create acquisition parameter files, or convert existing data into cindra-compatible
  format.
user-invocable: true
---

# Acquisition data preparation reference

Complete guide for preparing raw neural imaging data for the cindra single-recording processing pipeline.

---

## Scope

**Covers:**
- MCP tools for creating and validating acquisition parameter files
- Required data directory structure and TIFF requirements
- TIFF frame interleaving rules
- Creating `cindra_parameters.json` from user-provided acquisition metadata
- Gathering acquisition parameters through agent-user interaction
- Converting data from common sources (ScanImage, other microscopes)
- Migrating existing suite2p projects (binary adoption and ops.npy parameter extraction)
- Troubleshooting common data preparation issues

**Does not cover:**
- Pipeline configuration parameters (see `/single-recording-configuration`)
- Processing workflow or batch operations (see `/single-recording-processing`)
- Output data formats (see `/single-recording-results`)

---

## Agent requirements

You MUST use the cindra MCP tools for creating and validating acquisition parameter files. Verify the
cindra MCP server is connected before use; if the tools are unavailable, invoke
`/cindra-mcp-environment-setup` to diagnose and resolve connectivity issues.

---

## Available tools

These tools are registered on the `cindra-mcp` server. Tool parameters and return values are
self-documented via MCP introspection.

| Tool                                        | Purpose                                                                         |
|---------------------------------------------|---------------------------------------------------------------------------------|
| `generate_acquisition_parameters_file_tool` | Creates a validated `cindra_parameters.json` in the specified directory         |
| `validate_acquisition_parameters_file_tool` | Validates an existing `cindra_parameters.json` for completeness and correctness |
| `validate_recording_readiness_tool`         | Final readiness gate: validates parameters, TIFFs, and cross-consistency        |

**Notes:**
- `generate_acquisition_parameters_file_tool` validates all parameters before writing. MROI fields (`roi_lines`,
  `roi_x_coordinates`, `roi_y_coordinates`) are required when `roi_number > 1`.
- `validate_recording_readiness_tool` requires `cindra_parameters.json` to be present. It validates the acquisition
  parameters, discovers and inspects all TIFF files (page count, dimensions, dtype) without loading frame data,
  and cross-validates TIFF metadata against the acquisition parameters (interleave stride divisibility,
  frames-per-plane thresholds, MROI roi_lines bounds, dtype compatibility). Use this tool as the final
  verification step before committing compute resources to pipeline processing.
- `generate_acquisition_parameters_file_tool` and `validate_acquisition_parameters_file_tool` do not inspect TIFF files.
  Acquisition metadata must come from the user, experiment logs, microscope software output, or other external
  sources. Use `validate_recording_readiness_tool` for combined parameter + TIFF validation.

---

## Required data directory structure

Each recording must have a single directory containing:

```text
recording_directory/
  cindra_parameters.json     <-- acquisition metadata (created by MCP tool or manually)
  file_001.tif               <-- raw TIFF files (any .tif/.tiff extension)
  file_002.tif
  ...
```

The `cindra_parameters.json` file may also be in a subdirectory — the pipeline searches recursively. However, TIFF
files must be in the same directory as the JSON file (non-recursive TIFF scan).

---

## TIFF file requirements

### Supported formats

The pipeline reads standard multipage TIFF files (`.tif` or `.tiff` extension). All data is automatically converted
to int16 for processing. uint16 and int32 data is divided by 2 before conversion to fit the int16 range. All other
data types are cast directly to int16 without scaling.

### Non-TIFF source data

If the user's data is not already in multipage TIFF format, they must convert it before cindra can process it. Common
scenarios requiring conversion:

- **HDF5 / NWB files**: Extract imaging data arrays and write as multipage TIFFs using `tifffile.imwrite`.
- **Binary / raw files**: Read with numpy and write as multipage TIFFs.
- **Proprietary formats** (Nikon .nd2, Zeiss .czi, Leica .lif): Use the appropriate reader library (e.g.,
  `nd2`, `aicspylibczi`, `readlif`) to extract frames, then write as multipage TIFFs.
- **Single-frame TIFFs**: Already compatible — the pipeline concatenates all TIFFs in natural sort order.

When helping the user convert data, use web searches and documentation to determine the correct reader library and
approach for their specific format. Ensure the converted TIFFs follow the frame interleaving rules below.

### Frame interleaving

TIFF frames must follow a specific interleave pattern. Within each volume, frames cycle through channels first
(innermost), then planes, with a stride of `plane_number * channel_number`:

```text
Single plane, single channel:
  frame0, frame1, frame2, ...

2 planes, 1 channel:
  plane0, plane1, plane0, plane1, ...

1 plane, 2 channels:
  plane0-ch0, plane0-ch1, plane0-ch0, plane0-ch1, ...

2 planes, 2 channels:
  plane0-ch0, plane0-ch1, plane1-ch0, plane1-ch1, plane0-ch0, ...
```

The total frame count across all TIFF files should be evenly divisible by `plane_number * channel_number`.
If not, the pipeline silently drops the trailing frames that do not complete a full stride. This is not a
runtime error, but warn the user that incomplete final volumes will be discarded.

For MROI data, all ROIs share the same raw frames. Each ROI is extracted as a horizontal slice using `roi_lines`.

### Multiple TIFF files per recording

The pipeline loads all TIFF files in the data directory in natural sort order and concatenates them. Frames from
all files are treated as one continuous sequence following the interleave pattern. Use
`file_io.ignored_file_names` (see `/single-recording-configuration` Section 3) to exclude specific files.

---

## Acquisition parameters reference

### Required fields (all recordings)

| Field            | Type  | Description                                                                                                                                                                               |
|------------------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `frame_rate`     | float | Volume acquisition rate in Hz. For multi-plane recordings, this is the rate at which complete volumes are acquired, not the per-plane rate. Per-plane rate = `frame_rate / plane_number`. |
| `plane_number`   | int   | Number of Z-planes acquired per volume. 1 for single-plane imaging.                                                                                                                       |
| `channel_number` | int   | Number of channels per plane. Must be 1 or 2.                                                                                                                                             |

### MROI fields (required when roi_number > 1)

| Field               | Type            | Description                                                                                |
|---------------------|-----------------|--------------------------------------------------------------------------------------------|
| `roi_number`        | int             | Number of ROIs per plane. Defaults to 1 if omitted.                                        |
| `roi_lines`         | list[list[int]] | Row indices in the raw frame for each ROI. Outer list length must equal `roi_number`.      |
| `roi_x_coordinates` | list[int]       | X pixel offset for each ROI in the combined field of view. Length must equal `roi_number`. |
| `roi_y_coordinates` | list[int]       | Y pixel offset for each ROI in the combined field of view. Length must equal `roi_number`. |

For MROI data, each ROI is a horizontal band within the raw frame, defined by its row indices in `roi_lines`.
Each ROI x plane combination becomes a separate virtual plane for processing (total virtual planes =
`roi_number * plane_number`).

---

## Gathering acquisition parameters

When the user does not know their acquisition parameters, guide the interaction to determine them:

1. **Ask about the microscope and software**: What microscope was used? What acquisition software (ScanImage,
   Prairie View, Nikon Elements, etc.)? This determines where to find metadata.
2. **Ask about the experiment**: Single-plane or multi-plane (volumetric)? Single-channel or dual-channel? What
   was the approximate frame rate? What calcium indicator was used?
3. **Check for metadata files**: Many acquisition systems produce metadata files alongside the imaging data.
   Ask the user to look for log files, XML sidecars, ops files, or header files that contain acquisition
   parameters.
4. **Use web searches**: If the user identifies their microscope or software but doesn't know how to extract
   metadata, search for documentation on that system's data format and metadata storage.
5. **Verify consistency**: Confirm that the total frame count across TIFF files is divisible by
   `plane_number * channel_number`. Ask the user to check the total number of frames if needed.

---

## Data preparation workflows

### Workflow 1: Known acquisition parameters

When the user knows their acquisition metadata (frame rate, planes, channels):

1. **Confirm TIFF files** — Ask the user to verify TIFF files are present in the data directory.
2. **Verify divisibility** — Confirm `total_frames % (plane_number * channel_number) == 0`.
3. **Create parameters file** — Use `generate_acquisition_parameters_file_tool` with the known values.
4. **Validate** — Use `validate_acquisition_parameters_file_tool` to confirm the file is correct.
5. **Verify readiness** — Use `validate_recording_readiness_tool` to confirm the recording is ready for processing.

### Workflow 2: Unknown acquisition parameters

When the user has imaging data but is unsure about the acquisition configuration:

1. **Identify the data source** — Ask what microscope and acquisition software was used.
2. **Locate metadata** — Guide the user to find metadata files or headers specific to their system.
3. **Extract parameters** — Help the user read metadata using appropriate tools or libraries.
4. **Confirm with user** — Present the extracted parameters and ask the user to verify.
5. **Create parameters file** — Use `generate_acquisition_parameters_file_tool` with the confirmed values.
6. **Verify readiness** — Use `validate_recording_readiness_tool` to confirm the recording is ready for processing.

### Workflow 3: ScanImage recordings

ScanImage recordings typically save multipage TIFF files with metadata embedded in the TIFF headers. The key
metadata to extract:

| ScanImage metadata                | cindra field     |
|-----------------------------------|------------------|
| `SI.hRoiManager.scanVolumeRate`   | `frame_rate`     |
| `SI.hStackManager.numSlices`      | `plane_number`   |
| `SI.hChannels.channelSave` length | `channel_number` |

For MROI (multi-region) recordings, additional metadata is needed:

| ScanImage metadata                   | cindra field        |
|--------------------------------------|---------------------|
| Number of scan ROIs                  | `roi_number`        |
| Per-ROI line indices from scan field | `roi_lines`         |
| Per-ROI position in combined FOV     | `roi_x_coordinates` |
| Per-ROI position in combined FOV     | `roi_y_coordinates` |

ScanImage typically handles the frame interleaving correctly. Flyback frames (if included in the TIFF) should
be accounted for using `main.ignored_flyback_planes` in the pipeline configuration.

After creating the parameters file, use `validate_recording_readiness_tool` to verify TIFF dimensions, interleave
consistency, and MROI roi_lines bounds against actual frame data.

### Workflow 4: Migrating from suite2p

When the user has an existing suite2p output directory, cindra can adopt the data directly. Suite2p and cindra
use the same binary format (int16, memory-mapped, frames x height x width), so no data conversion is needed.

**Suite2p directory structure:**
```text
suite2p/
  plane0/
    ops.npy           # Processing parameters (contains acquisition metadata)
    data.bin           # Registered binary data (int16, frames x Ly x Lx)
    data_raw.bin       # Raw binary data (optional, pre-registration)
    stat.npy           # ROI statistics (array of dicts)
    iscell.npy         # Cell classification (N x 2 array)
    F.npy              # Fluorescence traces (N_cells x N_frames)
    Fneu.npy           # Neuropil traces (N_cells x N_frames)
    spks.npy           # Deconvolved spikes (N_cells x N_frames)
  plane1/
    ...
```

**Step 1: Extract acquisition parameters from `ops.npy`.**

Read the first plane's `ops.npy` file (`numpy.load(path, allow_pickle=True).item()`) and extract:

| suite2p ops key | cindra field     | Conversion                                                        |
|-----------------|------------------|-------------------------------------------------------------------|
| `fs`            | `frame_rate`     | `fs / nplanes` for multi-plane recordings. `fs` for single-plane. |
| `nplanes`       | `plane_number`   | Direct mapping.                                                   |
| `nchannels`     | `channel_number` | Direct mapping.                                                   |

**Critical: suite2p `fs` is the per-plane sampling rate.** cindra `frame_rate` is the volume rate. For
multi-plane recordings: `frame_rate = fs / nplanes`. For single-plane recordings the values are identical.

**Step 2: Create `cindra_parameters.json`.**

Use `generate_acquisition_parameters_file_tool` with the extracted parameters. Place it in the directory
containing the original raw TIFF files. If the user no longer has the raw TIFFs, place it alongside the
suite2p output directory.

**Step 3: Adopt suite2p binary files.**

Follow Workflow 6 (direct binary file adoption) to place suite2p's `data.bin` files into the cindra output
structure. Suite2p's binary format is directly compatible with cindra.

**Step 4: Process with cindra.**

Configure and run the cindra pipeline normally (see `/single-recording-configuration`). Cindra re-runs
registration, ROI detection, and extraction from scratch using its own algorithms. The suite2p binary files
serve only as the binarized input — all downstream processing is independent.

### Workflow 5: Non-TIFF source data

When the user's data is in a format other than multipage TIFF:

1. **Identify the format** — Ask the user what format their data is in (HDF5, NWB, .nd2, .czi, binary, etc.).
2. **Find the right reader** — Use web searches to identify the appropriate Python library for reading the format.
3. **Write a conversion script** — Help the user write a script that reads the source data and writes multipage
   TIFFs using `tifffile.imwrite`, ensuring the correct frame interleaving order.
4. **Verify output** — Confirm the converted TIFFs have the expected frame count and dimensions.
5. **Create parameters file** — Use `generate_acquisition_parameters_file_tool` with the acquisition metadata.
6. **Verify readiness** — Use `validate_recording_readiness_tool` to confirm TIFF data and parameters are consistent.

### Workflow 6: Direct binary file adoption (potentially unsafe)

When the user has pre-existing binary files (from suite2p, custom pipelines, or other sources) and wants to
skip TIFF-to-binary conversion entirely. **This workflow is potentially unsafe** because cindra cannot verify
that the binary files are correctly formatted. All metadata (frame count, dimensions, data type) must come
from the user — incorrect values will produce silent data corruption or pipeline crashes.

**Binary format requirements:**

cindra expects raw binary files with no header, containing contiguous int16 (signed 16-bit integer) values
laid out as `frames x height x width` in C-contiguous (row-major) order. Each plane must be a separate file.
The file size must exactly equal `frame_count * height * width * 2` bytes.

**Step 1: Gather binary file metadata from the user.**

You MUST ask the user to confirm all the following. Do not guess or infer these values:

- **Frame dimensions**: Height and width of each frame in pixels.
- **Frame count**: Total number of frames per plane in the binary file.
- **Data type**: Must be int16. If the source data uses a different type, the user must convert first.
- **Memory layout**: Must be C-contiguous (row-major), `frames x height x width`.
- **Acquisition parameters**: `frame_rate` (volume rate in Hz), `plane_number`, `channel_number`.

**Step 2: Create `cindra_parameters.json`.**

Use `generate_acquisition_parameters_file_tool` with the user-provided acquisition parameters.

**Step 3: Generate the cindra output bootstrap.**

Binarization only skips TIFF conversion when the cindra output bootstrap already exists alongside valid
binaries. Configure the pipeline and run `prepare_single_recording_batch_tool` first: it writes
`recording/cindra/configuration.yaml`, `recording/cindra/acquisition_parameters.yaml`, and each plane's
`recording/cindra/plane_N/runtime_data.yaml` (whose `registered_binary_path` points at
`plane_N/channel_1_data.bin`), and creates the `plane_N/` directories. Without this bootstrap, binarization
falls through to TIFF conversion, which fails when no raw TIFFs exist at `data_path`.

**Step 4: Place binary files in the cindra output structure.**

Copy or symlink each plane's binary file into the directories created by Step 3:

```text
recording/cindra/
  plane_0/channel_1_data.bin  →  source_plane_0.bin
  plane_1/channel_1_data.bin  →  source_plane_1.bin
  ...
```

For dual-channel recordings, cindra routes the functional channel into `channel_1_data.bin` and the structural
channel into `channel_2_data.bin` per plane. When adopting binaries directly, place the functional-channel data
in `channel_1_data.bin`, since the rest of the pipeline assumes channel 1 holds the functional channel.

**Step 5: Verify file sizes.**

For each binary file, confirm that the file size matches the expected value:
`frame_count * height * width * 2` bytes. A mismatch indicates incorrect dimensions, frame count, or data
type. Ask the user to re-check their metadata.

**Step 6: Run binarization.**

With the bootstrap (Step 3) and valid binaries (Step 4) in place, run binarization normally — cindra loads the
existing plane contexts, confirms each `registered_binary_path` exists, and skips TIFF conversion. If
binarization instead attempts conversion or reports missing binaries, the bootstrap or file placement is
incorrect; re-check Steps 3-4 and the format requirements above.

---

## Common issues and troubleshooting

### Frame count not divisible by plane_number * channel_number

**Causes and fixes:**
- **Incomplete final volume:** The recording was stopped mid-volume. Remove trailing incomplete frames from the
  last TIFF, or exclude the last TIFF via `file_io.ignored_file_names` in the pipeline configuration
  (see `/single-recording-configuration` Section 3).
- **Flyback frames included:** Some microscopes include flyback plane frames. Add these to
  `main.ignored_flyback_planes` in the pipeline configuration (the flyback planes are still part of the
  interleave pattern but are discarded during processing).
- **Wrong plane/channel count:** Re-examine the experiment metadata to confirm the actual values.

### MROI line index determination

For MROI recordings, `roi_lines` specifies which rows in the raw TIFF frame belong to each ROI. These indices
depend on the microscope configuration and are typically available from the acquisition software. Each inner list
contains the row indices (0-based) for one ROI. The pipeline extracts `frame[:, first_line:last_line+1, :]` for
each ROI.

---

## Related skills

| Skill                             | Relationship                                                          |
|-----------------------------------|-----------------------------------------------------------------------|
| `/cindra-mcp-environment-setup`   | Prerequisite: MCP server must be connected for data preparation tools |
| `/single-recording-configuration` | Next step: configure the pipeline using prepared data                 |
| `/single-recording-processing`    | Downstream: processing workflow that uses the prepared data           |
| `/single-recording-results`       | Downstream: output data format reference for processing results       |
| `/visualization`                  | Downstream: launch viewers to inspect data after processing           |

---

## Verification checklist

You MUST verify data preparation against this checklist before proceeding to pipeline configuration.

```text
Acquisition Data Preparation Compliance:
- [ ] cindra MCP server is connected (if not, invoke `/cindra-mcp-environment-setup`)
- [ ] TIFF files present in the data directory (.tif or .tiff extension)
- [ ] Total frame count is divisible by plane_number * channel_number
- [ ] `cindra_parameters.json` exists in the data directory (or a subdirectory)
- [ ] `validate_acquisition_parameters_file_tool` reports no errors
- [ ] `frame_rate` represents the volume rate (not per-plane rate)
- [ ] For MROI data: roi_lines, roi_x_coordinates, roi_y_coordinates are set correctly
- [ ] Review any warnings from validation (unrecognized fields, unused MROI fields)
- [ ] `validate_recording_readiness_tool` reports no errors (final readiness gate)
- [ ] Review readiness warnings (interleave remainder, low frame count, dtype cast, dimension mismatches)
```

**End point**: Data preparation is complete once all recordings pass the checklist above. If this skill was
invoked from another skill, return control to the caller. If invoked standalone, inform the user that the
data is ready for processing.
