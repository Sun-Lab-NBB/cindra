---
name: single-recording-configuration
description: >-
  Complete reference for single-recording pipeline configuration parameters, input data format, and MCP configuration
  tools. Documents all 9 configuration sections, parameter meanings, default values, acquisition parameter handling,
  and available MCP tools for generating configurations and discovering recordings. Use when configuring single-recording
  processing or when the user asks about single-recording configuration parameters.
user-invocable: true
---

# Single-recording configuration reference

Complete parameter reference for the single-recording (within-recording) cindra processing pipeline.

---

## Scope

**Covers:**
- All 9 configuration sections and their parameters for the `SingleRecordingConfiguration` dataclass
- Default values, types, and descriptions for every parameter
- Input data format requirements (TIFF structure, frame ordering, acquisition parameters JSON)
- Pipeline-set parameters
- MCP tools for configuration generation and recording discovery
- Configuration compliance verification

**Does not cover:**
- Output data formats and file references (see `/single-recording-results`)
- Processing workflow, batch operations, or status monitoring (see `/single-recording-processing`)
- Multi-recording configuration (see `/multi-recording-configuration`)

---

## MCP configuration tools

These tools are registered on the `cindra-mcp` server (defined in `configuration_tools.py`). Agents MUST use these
tools for configuration generation and recording discovery.

### generate_config_file

Generates a default configuration YAML file with sensible defaults.

**Parameters:**
| Parameter       | Type                                            | Description                                    |
|-----------------|-------------------------------------------------|------------------------------------------------|
| `output_path`   | str                                             | Absolute path where the config file is saved   |
| `pipeline_type` | Literal["single-recording", "multi-recording"]  | Pipeline type to generate configuration for    |

**Returns:** `{"success": bool, "file_path": str, "pipeline_type": str}`

### discover_single_recording_candidates_tool

Searches recursively for `cindra_parameters.json` files to find recordings with raw data.

**Parameters:**
| Parameter        | Type | Description                              |
|------------------|------|------------------------------------------|
| `root_directory` | str  | Absolute path to the root search directory |

**Returns:** `{"recordings": list[str], "count": int}` — each recording path is the parent directory of a
discovered `cindra_parameters.json` file.

### discover_multi_recording_candidates_tool

Searches recursively for `combined_metadata.npz` files to find recordings with completed single-recording output.

**Parameters:**
| Parameter        | Type | Description                              |
|------------------|------|------------------------------------------|
| `root_directory` | str  | Absolute path to the root search directory |

**Returns:** `{"recordings": list[str], "count": int}` — each recording path is the recording root directory
(grandparent of the `combined_metadata.npz` file).

---

## Configuration overview

The single-recording pipeline uses `SingleRecordingConfiguration`, a dataclass with 9 nested sections. Default values are
optimized for GCaMP6f data from 2-Photon Random Access Mesoscope (2P-RAM).

All parameters are specified in the `SingleRecordingConfiguration` YAML file. The pipeline loads the fully resolved
configuration directly from the file without any runtime overrides.

---

## Input data format

### TIFF file requirements

The pipeline expects raw imaging data as TIFF files (`.tif` or `.tiff` extension) stored in the `data_path`
directory or its subdirectories. All TIFF files found are processed in alphabetical order unless excluded via
`file_io.ignored_file_names`.

### Frame ordering

TIFF frames must be interleaved by plane and channel. The interleave pattern repeats with a stride of
`plane_number × channel_number`:

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

For MROI data, all ROIs share the same raw frames. Each ROI is a horizontal slice of the full frame, extracted
using the `roi_lines` indices from the acquisition parameters.

### Acquisition parameters file

A `cindra_parameters.json` file **must** exist in the data directory (or a subdirectory). The pipeline searches
recursively for this file. It contains acquisition metadata that cannot be inferred from the TIFF data alone.

**Single-ROI example** (standard imaging):

```json
{
    "frame_rate": 10.0,
    "plane_number": 1,
    "channel_number": 1
}
```

**Multi-plane, two-channel example**:

```json
{
    "frame_rate": 30.0,
    "plane_number": 3,
    "channel_number": 2
}
```

**MROI example** (multi-ROI acquisition):

```json
{
    "frame_rate": 10.0,
    "plane_number": 1,
    "channel_number": 1,
    "roi_number": 2,
    "roi_lines": [[0, 1, 2, 3, 4], [100, 101, 102, 103, 104]],
    "roi_x_coordinates": [0, 512],
    "roi_y_coordinates": [0, 0]
}
```

**Field reference:**

| Field               | Required          | Type             | Description                                         |
|---------------------|-------------------|------------------|-----------------------------------------------------|
| `frame_rate`        | Always            | float            | Volume acquisition rate in Hz                       |
| `plane_number`      | Always            | int              | Number of imaging planes per volume                 |
| `channel_number`    | Always            | int              | Number of channels per plane (1 or 2)               |
| `roi_number`        | MROI only         | int              | Number of ROIs per plane (default: 1)               |
| `roi_lines`         | When roi_number>1 | list[list[int]]  | Row indices in the raw frame for each ROI           |
| `roi_x_coordinates` | When roi_number>1 | list[int]        | X pixel offset for each ROI in the combined view    |
| `roi_y_coordinates` | When roi_number>1 | list[int]        | Y pixel offset for each ROI in the combined view    |

For MROI data, each ROI × plane combination becomes a separate virtual plane for processing. The total virtual
plane count is `roi_number × plane_number`.

---

## Pipeline-set parameters

These parameters are set automatically by the pipeline and should not be manually configured:

| Parameter                       | Set by        | Value                                               |
|---------------------------------|---------------|-----------------------------------------------------|
| `file_io.data_path`             | `pipeline.py` | Recording's raw data path                           |
| `file_io.output_path`           | `pipeline.py` | Recording's processed output path                   |
| `runtime.parallel_workers`      | CLI/MCP       | Number of workers (or auto-detected from CPU count) |
| `runtime.display_progress_bars` | CLI/MCP       | Whether to show progress bars                       |

---

## Section 1: runtime

Runtime behavior settings shared between single-recording and multi-recording pipelines.

| Parameter               | Type | Default | Description                                                                 |
|-------------------------|------|---------|-----------------------------------------------------------------------------|
| `parallel_workers`      | int  | 20      | Maximum CPU worker count. 10-20 optimal per plane. -1/0 = all cores.        |
| `display_progress_bars` | bool | False   | Show progress bars. Disable for parallel processing.                        |

---

## Section 2: main

Global parameters that broadly define the processing configuration.

| Parameter                   | Type         | Default | Description                                                     |
|-----------------------------|--------------|---------|-----------------------------------------------------------------|
| `two_channels`              | bool         | False   | Whether imaging data contains two channels per plane.           |
| `first_channel_functional`  | bool         | True    | Use first channel for ROI detection and signal extraction.      |
| `second_channel_functional` | bool         | False   | Use second channel for ROI detection and signal extraction.     |
| `tau`                       | float        | 0.4     | Sensor decay timescale in seconds. GCaMP6f: 0.4, GCaMP6s: ~1.5. |
| `ignored_flyback_planes`    | tuple[int]   | ()      | Flyback plane indices to exclude from processing (0-based).     |
| `custom_classifier_path`    | Path or None | None    | Path to custom classifier file. None = use built-in classifier. |

When both `first_channel_functional` and `second_channel_functional` are True, the pipeline performs independent
ROI detection on both channels.

---

## Section 3: file_io

I/O parameters for input data location and output directories.

| Parameter             | Type          | Default | Description                                                 |
|-----------------------|---------------|---------|-------------------------------------------------------------|
| `data_path`           | Path or None  | None    | Root directory containing input TIFFs. **Set by pipeline.** |
| `output_path`         | Path or None  | None    | Output directory root. **Set by pipeline.**                 |
| `ignored_file_names`  | tuple[str]    | ()      | Exact filenames to skip when loading TIFFs.                 |
| `repeat_binarization` | bool          | False   | Re-run TIFF to binary conversion even if binaries exist.    |

---

## Section 4: registration

Rigid registration parameters for motion correction.

| Parameter                                  | Type  | Default | Description                                                        |
|--------------------------------------------|-------|---------|--------------------------------------------------------------------|
| `repeat_registration`                      | bool  | False   | Re-register data even if already registered.                       |
| `align_by_first_channel`                   | bool  | True    | Use first channel for alignment. False = use second channel.       |
| `reference_frame_count`                    | int   | 500     | Frames to compute reference image.                                 |
| `batch_size`                               | int   | 100     | Frames per registration batch.                                     |
| `maximum_offset_fraction`                  | float | 0.1     | Max offset as fraction of frame size (0.1 = 10%).                  |
| `spatial_smoothing_sigma`                  | float | 1.15    | Gaussian sigma (pixels) for phase correlation smoothing.           |
| `temporal_smoothing_sigma`                 | float | 0.0     | Gaussian sigma (frames) for temporal smoothing. 0 = disabled.      |
| `two_step_registration`                    | bool  | False   | Enable refinement registration (two-step).                         |
| `bad_frame_threshold`                      | float | 1.0     | Threshold for excluding bad frames. Lower = more excluded.         |
| `normalize_frames`                         | bool  | True    | Clip pixel intensities to 1st-99th percentile during registration. |
| `registration_metric_principal_components` | int   | 5       | PCs for registration quality metrics. 0 = disable metrics.         |
| `compute_bidirectional_phase_offset`       | bool  | False   | Compute bidirectional phase offset for 2P line scanning.           |
| `bidirectional_phase_offset_override`      | int   | 0       | Manual bidiphase offset override. 0 = auto-detect.                 |

---

## Section 5: one_photon_registration

Additional processing for 1-photon data registration.

| Parameter                 | Type  | Default | Description                                                            |
|---------------------------|-------|---------|------------------------------------------------------------------------|
| `enabled`                 | bool  | False   | Enable 1P preprocessing (high-pass filtering, tapering). False for 2P. |
| `spatial_highpass_window` | int   | 42      | Spatial high-pass filter window (pixels).                              |
| `pre_smoothing_sigma`     | float | 0.0     | Gaussian smoothing sigma before high-pass. 0 = disabled.               |
| `edge_taper_pixels`       | float | 40.0    | Edge pixels to taper during registration.                              |

---

## Section 6: nonrigid_registration

Nonrigid registration for local motion correction.

| Parameter                   | Type            | Default    | Description                                          |
|-----------------------------|-----------------|------------|------------------------------------------------------|
| `enabled`                   | bool            | True       | Enable nonrigid registration for non-uniform motion. |
| `block_size`                | tuple[int, int] | (128, 128) | Block dimensions (pixels). Power of 2/3 recommended. |
| `signal_to_noise_threshold` | float           | 1.2        | SNR threshold for accepting block offsets.            |
| `maximum_block_offset`      | float           | 5.0        | Max block offset (pixels) relative to rigid offset.  |

---

## Section 7: roi_detection

ROI detection parameters.

| Parameter                     | Type  | Default | Description                                                        |
|-------------------------------|-------|---------|--------------------------------------------------------------------|
| `enabled`                     | bool  | True    | Enable ROI detection and classification.                           |
| `preclassification_threshold` | float | 0.5     | Min classifier confidence to keep ROI. 0 = keep all.               |
| `threshold_scaling`           | float | 2.0     | Detection threshold scaling. Higher = more distinct ROIs needed.   |
| `spatial_highpass_window`     | int   | 25      | High-pass window for neuropil subtraction during detection.        |
| `maximum_overlap`             | float | 0.75    | Max allowed ROI overlap fraction. Higher overlap = discard.        |
| `temporal_highpass_window`    | int   | 100     | Running mean window (frames) for drift removal.                    |
| `maximum_iterations`          | int   | 50      | Iteration scaling for ROI extraction (actual limit = value × 250). |
| `maximum_binned_frames`       | int   | 5000    | Max binned frames for detection. More = more ROIs, slower.         |
| `denoise`                     | bool  | False   | PCA-based denoising of binned movie before detection.              |
| `crop_to_soma`                | bool  | True    | Crop dendrites from ROIs before computing classification features. |

---

## Section 8: signal_extraction

Fluorescence signal extraction from ROIs.

| Parameter                      | Type  | Default | Description                                                     |
|--------------------------------|-------|---------|-----------------------------------------------------------------|
| `extract_neuropil`             | bool  | True    | Extract neuropil activity. False = assume zero neuropil.        |
| `allow_overlap`                | bool  | False   | Include overlapping pixels in signal extraction.                |
| `minimum_neuropil_pixels`      | int   | 350     | Min neuropil region size (pixels).                              |
| `inner_neuropil_border_radius` | int   | 2       | Pixels between cell and neuropil region.                        |
| `cell_probability_percentile`  | int   | 50      | Percentile threshold for cell vs neuropil pixel classification. |
| `classification_threshold`     | float | 0.5     | Min classifier confidence for labeling ROI as a cell.           |
| `batch_size`                   | int   | 500     | Frames per extraction batch.                                    |
| `colocalization_threshold`     | float | 0.65    | Threshold for cross-channel ROI colocalization.                 |

---

## Section 9: spike_deconvolution

Spike inference from fluorescence traces.

| Parameter              | Type  | Default   | Description                                                       |
|------------------------|-------|-----------|-------------------------------------------------------------------|
| `extract_spikes`       | bool  | True      | Deconvolve spikes from fluorescence.                              |
| `neuropil_coefficient` | float | 0.7       | Neuropil scaling before subtraction.                              |
| `baseline_method`      | str   | "maximin" | Baseline method: "maximin", "constant", or "constant_percentile". |
| `baseline_window`      | float | 60.0      | Sliding window (seconds) for maximin baseline.                    |
| `baseline_sigma`       | float | 10.0      | Gaussian sigma (frames) for baseline computation.                 |
| `baseline_percentile`  | float | 8.0       | Percentile for constant_percentile baseline.                      |

---

## User-configurable vs auto-set parameters

### Parameters users should configure

| Parameter                     | When to change                                       |
|-------------------------------|------------------------------------------------------|
| `main.tau`                    | Different calcium indicator (GCaMP6s, GCaMP7f, etc.) |
| `main.two_channels`           | Recording has two channels                           |
| `main.ignored_flyback_planes` | Flyback planes present in the recording              |
| `file_io.ignored_file_names`  | Specific TIFFs to exclude from processing            |

### Parameters typically left at default

- All registration parameters (work well for 2P imaging)
- ROI detection parameters (tuned for GCaMP6f)
- Signal extraction parameters
- Spike deconvolution parameters

---

## Configuration file format

```yaml
runtime:
  parallel_workers: 20
  display_progress_bars: false

main:
  two_channels: false
  tau: 0.4
  ignored_flyback_planes: []

file_io:
  ignored_file_names: []

# Other sections use defaults...
```

---

## Configuration workflow

1. **Discover recordings** using `discover_single_recording_candidates_tool` to find directories with raw data.
2. **Generate a default configuration** using `generate_config_file` with `pipeline_type="single-recording"`.
3. **Review and modify** the generated YAML file, setting at minimum `main.tau` and `main.two_channels`.
4. **Verify** the configuration against the compliance checklist below.
5. **Hand off** to the processing workflow (see `/single-recording-processing`).

---

## Related skills

| Skill                              | Relationship                                                                      |
|------------------------------------|-----------------------------------------------------------------------------------|
| `/multi-recording-configuration`   | Companion configuration reference for the multi-recording pipeline                |
| `/single-recording-processing`     | Processing workflow that consumes this configuration                              |
| `/single-recording-results`        | Output data format reference for evaluating processing results                    |
| `/multi-recording-processing`      | Multi-recording workflow that requires single-recording processing to be complete |

---

## Verification checklist

You MUST verify configuration files against this checklist before starting single-recording processing.

```text
Single-Recording Configuration Compliance:
- [ ] Configuration file is valid YAML with correct section nesting
- [ ] `main.tau` matches the calcium indicator used (0.4 for GCaMP6f, ~1.5 for GCaMP6s)
- [ ] `main.two_channels` set correctly for the recording type
- [ ] `main.ignored_flyback_planes` lists correct flyback plane indices if applicable
- [ ] `file_io.ignored_file_names` lists any TIFFs to exclude
- [ ] No manually set values for pipeline-set parameters (data_path, output_path, etc.)
- [ ] `cindra_parameters.json` exists in the data directory with correct acquisition metadata
- [ ] TIFF frame ordering matches the plane/channel interleave pattern
```
