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

You MUST use the cindra MCP tools for creating and validating acquisition parameter files. Verify the cindra MCP server
is connected before use. If the tools are unavailable, invoke `/cindra-mcp-environment-setup` to diagnose and resolve
connectivity issues.

---

## Available tools

These tools are registered on the `cindra-mcp` server. Tool parameters and return values are self-documented via MCP
introspection.

| Tool                                        | Purpose                                                                         |
|---------------------------------------------|---------------------------------------------------------------------------------|
| `generate_acquisition_parameters_file_tool` | Creates a validated `cindra_parameters.json` in the specified directory         |
| `validate_acquisition_parameters_file_tool` | Validates an existing `cindra_parameters.json` for completeness and correctness |
| `validate_recording_readiness_tool`         | Final readiness gate: validates parameters, TIFFs, and cross-consistency        |

**Notes:**
- `generate_acquisition_parameters_file_tool` and `validate_recording_readiness_tool` take the raw imaging directory as
  `raw_data_path`, which `prepare_single_recording_batch_tool` takes under `raw_data_paths` and
  `discover_recordings_tool` reports per candidate. It is not the `output_root` the status, results, and viewer tools
  take, and the file itself is `file_path` on the file validator.
- `generate_acquisition_parameters_file_tool` validates all parameters before writing. When `roi_number > 1` it
  requires `roi_x_coordinates`, `roi_y_coordinates`, and the region rows as either `roi_line_spans`, one inclusive
  `[first, last]` pair per region, or `roi_lines` enumerated in full. Prefer spans. Supplying both is rejected.
- `validate_recording_readiness_tool` accepts the raw imaging directory or any parent of it, because it resolves the
  directory holding `cindra_parameters.json` beneath the named path the same way the conversion does, and reports that
  directory as `raw_data_path`. It validates the acquisition parameters and inspects all TIFF files (page count,
  dimensions, dtype) without loading frame data. It cross-validates TIFF metadata against the acquisition parameters
  (interleave cycle remainder, frames-per-plane thresholds, MROI roi_lines bounds and block layout, dtype
  compatibility). Use it as the final verification step before committing compute resources to processing. Given a path
  whose subtree holds no `cindra_parameters.json`, it reports that the file is stored neither inside that path nor
  anywhere beneath it. It then asks for `generate_acquisition_parameters_file_tool` to create one inside the directory
  that directly holds the raw TIFF files, naming that directory or a parent of it as `raw_data_path`.
- `generate_acquisition_parameters_file_tool` and `validate_acquisition_parameters_file_tool` do not inspect TIFF files.
  Acquisition metadata must come from the user, experiment logs, microscope software output, or other external sources.
  Use `validate_recording_readiness_tool` for combined parameter + TIFF validation.
- `validate_acquisition_parameters_file_tool` and `validate_recording_readiness_tool` return a `success` flag that only
  reports whether the tool ran. Gate every downstream step on the separate `valid` field, which is False whenever the
  tool collects any validation errors and can be False while `success` is True.
- All three tools summarize `roi_lines` rather than echoing it, returning one entry per ROI holding `roi`,
  `line_count`, `span`, and a `contiguous` flag. That flag is true when the block covers every row of its span exactly
  once. The `span` a summary reports is the same inclusive pair `roi_line_spans` accepts, so it round-trips.

---

## Required data directory structure

Each recording must have a single directory containing:

```text
raw_data_path/
  cindra_parameters.json     <-- acquisition metadata (created by MCP tool or manually)
  file_001.tif               <-- raw TIFF files (any .tif/.tiff extension)
  file_002.tif
  ...
```

That directory is the `raw_data_path` into which `generate_acquisition_parameters_file_tool` writes, and the one
`validate_recording_readiness_tool` and the prepare tools resolve from a path naming it or a parent. The flat TIFF scan
runs against that resolved directory, so the TIFF files must sit beside `cindra_parameters.json`, and a JSON one level
down resolves identically on every surface. Only a path whose subtree carries no acceptable source file is rejected,
reported under `invalid_recordings` with no manifest and no tracker while naming the subdirectory holding the files.

---

## TIFF file requirements

### Supported formats

The pipeline reads standard multipage TIFF files (`.tif`, `.tiff`, `.TIF`, or `.TIFF` extension). All data is
automatically converted to int16 for processing. uint16 and int32 data is halved by floor division and then clipped to
the int16 range, so int32 magnitudes that still exceed that range after halving saturate at -32768 or 32767 instead of
wrapping. All other data types are cast directly to int16 without scaling.

### Non-TIFF source data

If the user's data is not already in multipage TIFF format, they must convert it before cindra can process it. Common
scenarios requiring conversion:

- **HDF5 / NWB files**: Extract imaging data arrays and write as multipage TIFFs using `tifffile.imwrite`.
- **Binary / raw files**: Read with numpy and write as multipage TIFFs.
- **Proprietary formats** (Nikon .nd2, Zeiss .czi, Leica .lif): Use the appropriate reader library (e.g., `nd2`,
  `aicspylibczi`, `readlif`) to extract frames, then write as multipage TIFFs.
- **Single-frame TIFFs**: Already compatible, since the pipeline concatenates all TIFFs in natural sort order.

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

Binarization keeps whole cycles of `plane_number * channel_number` frames and discards whatever the total frame count
leaves past the last whole cycle, logging a warning that names the discarded count. Those trailing frames reach some
planes and channels and not others, so every plane binary holds `total_frames // (plane_number * channel_number)`
frames and the two channels of one plane stay aligned frame for frame. Stopping an acquisition on a volume boundary is
what keeps every frame it collected. A recording holding fewer frames than one whole cycle fails binarization with an
error naming the count it holds, which usually means `plane_number` or `channel_number` disagrees with the
microscope's settings.

For MROI data, all ROIs share the same raw frames. Each ROI is extracted as a horizontal slice using `roi_lines`.

### Multiple TIFF files per recording

The pipeline loads all TIFF files in the data directory in natural sort order and concatenates them. Frames from all
files are treated as one continuous sequence following the interleave pattern. Use `file_io.ignored_file_names` (see
`/single-recording-configuration` Section 3) to exclude specific files.

### Excluding files that are not part of the recording

A raw mesoscope directory commonly holds an anatomical z-stack, for example `zstack.tiff`, alongside the imaging files.
Exclude it through `file_io.ignored_file_names` (see `/single-recording-configuration` Section 3).

`validate_recording_readiness_tool` does not read the pipeline configuration, so it inspects the z-stack too and reports
its shape as a warning rather than an error. The recording is still ready. Confirm the excluded stems are listed in the
configuration and proceed, and do not ask the user to delete or reshape the file.

---

## Acquisition parameters reference

### Required fields (all recordings)

| Field            | Type  | Description                                                         |
|------------------|-------|---------------------------------------------------------------------|
| `frame_rate`     | float | Volume acquisition rate in Hz.                                      |
| `plane_number`   | int   | Number of Z-planes acquired per volume. 1 for single-plane imaging. |
| `channel_number` | int   | Number of channels per plane. Must be 1 or 2.                       |

For multi-plane recordings, `frame_rate` is the rate at which complete volumes are acquired, and the per-plane rate is
`frame_rate / plane_number`.

### MROI fields (required when roi_number > 1)

| Field               | Type            | Description                                                                                |
|---------------------|-----------------|--------------------------------------------------------------------------------------------|
| `roi_number`        | int             | Number of ROIs per plane. Defaults to 1 if omitted.                                        |
| `roi_lines`         | list[list[int]] | Row indices in the raw frame for each ROI. Outer list length must equal `roi_number`.      |
| `roi_x_coordinates` | list[int]       | X pixel offset for each ROI in the combined field of view. Length must equal `roi_number`. |
| `roi_y_coordinates` | list[int]       | Y pixel offset for each ROI in the combined field of view. Length must equal `roi_number`. |

For MROI data, each ROI is a horizontal band within the raw frame, defined by its row indices in `roi_lines`. Each ROI x
plane combination becomes a separate virtual plane for processing (total virtual planes = `roi_number * plane_number`).

---

## Gathering acquisition parameters

When the user does not know their acquisition parameters, guide the interaction to determine them:

1. **Ask about the microscope and software**: What microscope was used? What acquisition software (ScanImage, Prairie
   View, Nikon Elements, etc.)? This determines where to find metadata.
2. **Ask about the experiment**: Single-plane or multi-plane (volumetric)? Single-channel or dual-channel? What was the
   approximate frame rate? What calcium indicator was used?
3. **Check for metadata files**: Many acquisition systems produce metadata files alongside the imaging data. Ask the
   user to look for log files, XML sidecars, ops files, or header files that contain acquisition parameters.
4. **Use web searches**: If the user identifies their microscope or software but doesn't know how to extract metadata,
   search for documentation on that system's data format and metadata storage.
5. **Verify consistency**: Compare the total frame count across TIFF files against `plane_number * channel_number`. Ask
   the user to check the total if needed, and report any remainder as the trailing frames binarization discards.

---

## Data preparation workflows

### Workflow 1: Known acquisition parameters

When the user knows their acquisition metadata (frame rate, planes, channels):

1. **Confirm TIFF files**. Ask the user to verify TIFF files are present in the data directory.
2. **Verify the cycle count**. Confirm `total_frames >= plane_number * channel_number`, since a shorter recording is
   rejected, and tell the user how many trailing frames `total_frames % (plane_number * channel_number)` discards.
3. **Create parameters file**. Use `generate_acquisition_parameters_file_tool` with the known values.
4. **Validate**. Use `validate_acquisition_parameters_file_tool` to confirm the file is correct.
5. **Verify readiness**. Use `validate_recording_readiness_tool` to confirm the recording is ready for processing.

### Workflow 2: Unknown acquisition parameters

When the user has imaging data but is unsure about the acquisition configuration:

1. **Extract the parameters**. Work through Gathering acquisition parameters above, helping the user read the metadata
   of their system with the appropriate tools or libraries, then present the values and ask the user to verify them.
2. **Create parameters file**. Use `generate_acquisition_parameters_file_tool` with the confirmed values.
3. **Verify readiness**. Use `validate_recording_readiness_tool` to confirm the recording is ready for processing.

### Workflow 3: ScanImage recordings

ScanImage recordings typically save multipage TIFFs with metadata embedded in the TIFF headers. Key metadata to extract:

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

ScanImage typically handles the frame interleaving correctly. Flyback frames (if included in the TIFF) should be
accounted for using `main.ignored_flyback_planes` in the pipeline configuration.

After creating the parameters file, run `validate_recording_readiness_tool` as the final gate (see Available tools).

### Workflow 4: Migrating from suite2p

When the user has an existing suite2p output directory, cindra can adopt the data directly. Suite2p and cindra use the
same binary format (int16, memory-mapped, frames x height x width), so no data conversion is needed.

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
| `fs`            | `frame_rate`     | `fs * nplanes` for multi-plane recordings. `fs` for single-plane. |
| `nplanes`       | `plane_number`   | Direct mapping.                                                   |
| `nchannels`     | `channel_number` | Direct mapping.                                                   |

**Critical: suite2p `fs` is the per-plane sampling rate.** cindra `frame_rate` is the volume rate. For multi-plane
recordings: `frame_rate = fs * nplanes`. For single-plane recordings the values are identical.

**Step 2: Create `cindra_parameters.json`.**

Use `generate_acquisition_parameters_file_tool` with the extracted parameters. Place it in the directory containing the
original raw TIFF files. If the user no longer has the raw TIFFs, place it alongside the suite2p output directory.

**Step 3: Adopt suite2p binary files.**

Follow Workflow 6 (direct binary file adoption) to place suite2p's `data.bin` files into the cindra output structure.

**Step 4: Process with cindra.**

Configure and run the cindra pipeline normally (see `/single-recording-configuration`). Cindra re-runs registration, ROI
detection, and extraction from scratch using its own algorithms. The suite2p binary files serve only as the binarized
input, and all downstream processing is independent.

### Workflow 5: Non-TIFF source data

When the user's data is in a format other than multipage TIFF:

1. **Convert the data**. Ask what format the data holds, then follow Non-TIFF source data above to find the reader
   library. Help the user write a conversion script that reads the source data and writes multipage TIFFs using
   `tifffile.imwrite`, ensuring the correct frame interleaving order.
2. **Verify output**. Confirm the converted TIFFs have the expected frame count and dimensions.
3. **Create parameters file**. Use `generate_acquisition_parameters_file_tool` with the acquisition metadata.
4. **Verify readiness**. Use `validate_recording_readiness_tool` to confirm TIFF data and parameters are consistent.

### Workflow 6: Direct binary file adoption (potentially unsafe)

When the user has pre-existing binary files (from suite2p, custom pipelines, or other sources) and wants to skip
TIFF-to-binary conversion entirely. **This workflow is potentially unsafe** because cindra cannot verify that the binary
files are correctly formatted. All metadata (frame count, dimensions, data type) must come from the user, and incorrect
values will produce silent data corruption or pipeline crashes.

**Binary format requirements:**

cindra expects raw binary files with no header, containing contiguous int16 (signed 16-bit integer) values laid out as
`frames x height x width` in C-contiguous (row-major) order. Each plane must be a separate file. The file size must
exactly equal `frame_count * height * width * 2` bytes.

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

Binarization only skips TIFF conversion when the cindra output bootstrap already exists alongside valid binaries.
Configure the pipeline and run `prepare_single_recording_batch_tool` first. It writes
`recording/cindra/configuration.yaml`, `recording/cindra/acquisition_parameters.yaml`, and each plane's
`recording/cindra/plane_N/runtime_data.yaml` (whose `registered_binary_path` points at `plane_N/channel_1_data.bin`),
and creates the `plane_N/` directories. Without this bootstrap, binarization aborts before it reaches TIFF conversion,
naming the missing `runtime_data.yaml` and asking for `prepare_single_recording_batch_tool` to be run first. With the
bootstrap in place but no binaries at the plane paths, binarization instead falls through to TIFF conversion, which
fails when no raw TIFFs exist at `data_path`.

**Step 4: Place binary files in the cindra output structure.**

Copy or symlink each plane's binary file into the directories created by Step 3:

```text
recording/cindra/
  plane_0/channel_1_data.bin  →  source_plane_0.bin
  plane_1/channel_1_data.bin  →  source_plane_1.bin
  ...
```

For dual-channel recordings, cindra routes the functional channel into `channel_1_data.bin` and the structural channel
into `channel_2_data.bin` per plane. When adopting binaries directly, place the functional-channel data in
`channel_1_data.bin`, since the rest of the pipeline assumes channel 1 holds the functional channel.

**Step 5: Verify file sizes and record the frame geometry.**

For each binary file, confirm that the file size matches the expected value: `frame_count * height * width * 2` bytes. A
mismatch indicates incorrect dimensions, frame count, or data type. Ask the user to re-check their metadata.

Then write the confirmed geometry into the `io:` section of each plane's `recording/cindra/plane_N/runtime_data.yaml`,
setting `frame_height`, `frame_width`, and `frame_count`. The bootstrap from Step 3 leaves all three at 0, because only
TIFF conversion populates them and binarization returns early without touching them once it finds valid binaries. Every
later stage reads the geometry from this file, so a plane left at 0 fails registration with "Unable to register plane
{index}. A plane must contain at least 50 frames to be processed, but the input plane contains only 0 frames."

**Step 6: Run binarization.**

With the bootstrap (Step 3) and valid binaries (Step 4) in place, run binarization normally. Cindra loads the existing
plane contexts and skips TIFF conversion when every plane passes three checks. Each `registered_binary_path` exists,
neither a `<binary>.binarizing` nor a `<binary>.registering` marker sits beside it, and the binary's size matches the
frame geometry recorded for its plane in Step 5. A two-channel recording is held to its `channel_2_data.bin` as well. A
marker, a size mismatch, or an absent second channel binary fails the run with a RuntimeError naming the affected
files. A missing `registered_binary_path`, or `file_io.repeat_binarization` being True, makes cindra convert from the
source TIFFs instead, which cannot succeed for adopted data because no raw TIFFs exist. Re-check Steps 3-5 and the
format requirements above in either case.

**Step 7: Run registration for every plane.**

Adopted data follows the standard phase order of binarization, registration, processing, and combination. Run the
registration phase for every plane before dispatching any processing job. Registration writes the reference image, the
motion offsets, and the bad-frame mask into each plane's `registration_data/` directory and the valid pixel ranges into
the plane's `runtime_data.yaml`, and processing reads all of them back before detecting ROIs. A plane that carries no
`registration_data/` fails at the start of processing with "Unable to process plane {index}. The plane must be
registered before ROI detection...", so a binarize-then-process dispatch stops at the first processing job.

---

## Common issues and troubleshooting

### Frame count leaves a remainder over plane_number * channel_number

**Causes and fixes:**
- **Incomplete final volume:** The recording was stopped mid-volume. This is not an error and needs no fix. Tell the
  user how many frames a run drops, since those frames hold real signal on the planes they reached.
- **Flyback frames included:** Some microscopes include flyback plane frames. Add these to `main.ignored_flyback_planes`
  in the pipeline configuration (the flyback planes are still part of the interleave pattern but are discarded during
  processing).
- **Wrong plane/channel count:** Re-examine the experiment metadata to confirm the actual values.
- **Fewer frames than one whole cycle:** Binarization rejects the recording with `Unable to resolve the TIFF conversion
  plan for the recording stored in {path}. The N frame(s) ... do not fill one M frame plane and channel interleave
  cycle, so no plane receives any frames.` Correct `plane_number` and `channel_number`, or acquire a longer recording.

### Frame shape differs between TIFF files

Binarization fails with `Unable to determine frame dimensions. Every TIFF file in the data directory must hold frames of
the same shape...`, naming the differing files and both shapes.

**Causes and fixes:**
- **Anatomical z-stack in the data directory:** the usual cause. Exclude its stem as above, then re-run binarization.
- **Mixed acquisitions in one directory:** two recordings with different fields of view were written to the same folder.
  Separate them into one directory per recording.
- **Genuinely ragged recording:** re-check the acquisition, because cindra cannot combine differently shaped frames into
  one plane binary.

### MROI line index determination

For MROI recordings, `roi_lines` specifies which rows in the raw TIFF frame belong to each ROI. These indices depend on
the microscope configuration and are typically available from the acquisition software. Each inner list contains the row
indices (0-based) for one ROI. The pipeline extracts `frame[:, first_line:last_line+1, :]` for each ROI.

The file always stores those rows in full, but authoring them that way is impractical. A three-region recording carries
over two thousand indices, and one dropped value still parses. Pass `roi_line_spans` instead, naming each region by its
inclusive first and last row, as in `[[0, 793], [916, 1683], [1806, 2277]]`. The tool expands the pairs on write, so
both forms produce identical files. Reversed bounds, a negative first row, an entry that is not two integers, and a pair
count other than `roi_number` are each rejected by name.

`validate_recording_readiness_tool` cross-checks the blocks. One reaching past the last row of the raw frame is an error
naming both numbers and the highest valid index, reported only when a TIFF was readable, while consecutive blocks are
compared with no TIFF required. A block starting at or before the previous block's last line warns of an overlap, and
one starting more than a line past it warns of a gap naming the unassigned row count. Both usually mean a transcription
slip, so confirm the values against the acquisition software. Both validators return `roi_lines` as one summary per ROI
and serve the indices through the optional `roi_line_slice` argument, a `[roi_index, start, stop]` triplet naming a
half-open range. A malformed triplet, an out-of-range ROI index, bounds outside `0 <= start < stop <= line_count`,
absent `roi_lines`, or a span past 2000 lines is rejected with `success` False.

---

## Related skills

| Skill                             | Relationship                                                               |
|-----------------------------------|----------------------------------------------------------------------------|
| `/cindra-pipeline`                | Overview: end-to-end phases, handoffs, and the single-vs-multi entry point |
| `/cindra-mcp-environment-setup`   | Prerequisite: MCP server must be connected for data preparation tools      |
| `/single-recording-configuration` | Next step: configure the pipeline using prepared data                      |
| `/single-recording-processing`    | Downstream: processing workflow that uses the prepared data                |
| `/single-recording-results`       | Downstream: output data format reference for processing results            |
| `/visualization`                  | Downstream: launch viewers to inspect data after processing                |

---

## Verification checklist

You MUST verify data preparation against this checklist before proceeding to pipeline configuration.

```text
Acquisition Data Preparation Compliance, tool-settled (run `validate_acquisition_parameters_file_tool` then
`validate_recording_readiness_tool`):
- [ ] cindra MCP server is connected (if not, invoke `/cindra-mcp-environment-setup`)
- [ ] `validate_acquisition_parameters_file_tool` reports no errors
- [ ] `validate_recording_readiness_tool` reports no errors (final readiness gate)

Acquisition Data Preparation Compliance, reader-judged:
- [ ] TIFF files present directly beside `cindra_parameters.json` in the imaging directory (.tif or .tiff extension)
- [ ] Total frame count holds at least one whole plane_number * channel_number cycle, with any remainder the run
      discards reported to the user
- [ ] `cindra_parameters.json` exists in that imaging directory, named as `raw_data_path` or resolved beneath it
- [ ] `frame_rate` represents the volume rate (not per-plane rate)
- [ ] For MROI data: roi_lines, roi_x_coordinates, roi_y_coordinates are set correctly
- [ ] For MROI data: every `roi_lines` summary reports `contiguous` true, or the block was sliced and confirmed
- [ ] Review any warnings from validation (unrecognized fields, unused MROI fields)
- [ ] Review readiness warnings (interleave remainder, low frame count, dtype cast, frame shapes, MROI gaps)
- [ ] Any differing-frame-shape warning names a file already listed in `file_io.ignored_file_names`
```

**End point**: Data preparation is complete once all recordings pass the checklist above. If this skill was invoked from
another skill, return control to the caller. If invoked standalone, inform the user that the data is ready for
processing.
