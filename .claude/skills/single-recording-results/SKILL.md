---
name: single-recording-results
description: >-
  Complete reference for single-recording pipeline output data formats. Documents every file, directory, array shape,
  dtype, and NPZ key produced by the pipeline, plus verification checklists for output completeness. Use when evaluating
  single-recording processing results or when the user asks about single-recording output data.
user-invocable: true
---

# Single-recording results data reference

Complete output data format documentation for the single-recording (within-recording) cindra processing pipeline.

---

## Scope

**Covers:**
- Complete output data reference: every file, directory, array shape, dtype, and NPZ key produced by the pipeline
- Directory structure for combined and per-plane results
- Processing phase and file creation timeline
- Data type conventions and memory-mapping guidance
- Multi-recording compatibility requirements
- Output completeness verification

**Does not cover:**
- Configuration parameters or input data format (see `/single-recording-configuration`)
- Processing workflow, batch operations, or status monitoring (see `/single-recording-processing`)
- Multi-recording output data formats (see `/multi-recording-results`)

---

## Available tools

Use these cindra MCP tools to query and verify single-recording output data programmatically. Prefer these over
manual file reads whenever possible.

### Verification tool

| Tool                                  | Purpose                                                                     |
|---------------------------------------|-----------------------------------------------------------------------------|
| `verify_single_recording_output_tool` | Verifies completeness of all expected output files and NPZ keys             |

### Query tools

| Tool                                         | Purpose                                                                               |
|----------------------------------------------|---------------------------------------------------------------------------------------|
| `query_single_recording_metadata_tool`       | Queries recording dimensions, frame count, sampling rate, ROI/cell counts, and timing |
| `query_detection_summary_tool`               | Queries detection image intensity statistics and estimated ROI diameter               |
| `query_registration_quality_tool`            | Queries per-plane registration offset summaries, correlations, bad frames, PC metrics |
| `query_single_recording_roi_statistics_tool` | Queries per-ROI spatial statistics and classification with sorting and filtering      |
| `query_single_recording_traces_tool`         | Queries fluorescence trace arrays for specific ROIs with optional downsampling        |

### Recommended query order

1. `query_single_recording_metadata_tool` — understand recording properties and processing status
2. `query_registration_quality_tool` — assess motion correction quality per plane
3. `query_detection_summary_tool` — review detection image quality and ROI diameter
4. `query_single_recording_roi_statistics_tool` — inspect ROI quality metrics and classification
5. `query_single_recording_traces_tool` — examine fluorescence activity for specific ROIs

---

## Output data reference

All results are saved under `{output_path}/cindra/`. The pipeline produces combined (multi-plane merged) data at
the root level and per-plane data in numbered subdirectories. Channel 2 files are only present for two-channel
recordings where both channels are functional.

### Directory structure

```text
cindra/
├── configuration.yaml                          # Saved pipeline configuration
├── acquisition_parameters.yaml                 # Saved acquisition metadata
├── combined_metadata.npz                       # Combined multi-plane metadata
├── detection_data/                             # Combined detection images
│   ├── mean_image.npy
│   ├── enhanced_mean_image.npy
│   ├── maximum_projection.npy
│   └── correlation_map.npy
├── roi_masks.npz                               # Combined ROI spatial data
├── roi_statistics.npz                          # Combined ROI shape statistics
├── cell_fluorescence.npy                       # Combined fluorescence traces
├── neuropil_fluorescence.npy
├── subtracted_fluorescence.npy
├── spikes.npy
├── cell_classification.npy
├── plane_0/                                    # Per-plane processing results
│   ├── runtime_data.yaml                       # Plane runtime metadata
│   ├── channel_1_data.bin                      # Registered binary data
│   ├── registration_data/                      # Registration arrays
│   │   ├── reference_image.npy
│   │   ├── bad_frames.npy
│   │   ├── rigid_y_offsets.npy
│   │   ├── rigid_x_offsets.npy
│   │   ├── rigid_correlations.npy
│   │   ├── nonrigid_y_offsets.npy
│   │   ├── nonrigid_x_offsets.npy
│   │   ├── nonrigid_correlations.npy
│   │   ├── principal_component_extreme_images.npy
│   │   ├── principal_component_projections.npy
│   │   └── principal_component_shift_metrics.npy
│   ├── detection_data/                         # Plane detection images
│   │   ├── mean_image.npy
│   │   ├── enhanced_mean_image.npy
│   │   ├── maximum_projection.npy
│   │   └── correlation_map.npy
│   ├── roi_masks.npz                           # Plane ROI spatial data
│   ├── roi_statistics.npz                      # Plane ROI shape statistics
│   ├── cell_fluorescence.npy
│   ├── neuropil_fluorescence.npy
│   ├── subtracted_fluorescence.npy
│   ├── spikes.npy
│   └── cell_classification.npy
├── plane_1/
└── ...
```

### Processing phase and file creation timeline

**Phase 1 — Binarize:** Creates `configuration.yaml`, `acquisition_parameters.yaml`, per-plane
`channel_1_data.bin` (and `channel_2_data.bin` if two-channel), and initial `runtime_data.yaml` per plane.

**Phase 2 — Process (per-plane):** Creates `registration_data/`, `detection_data/`, ROI `.npz` files,
fluorescence `.npy` traces, and updates `runtime_data.yaml` with processing results and timing.

**Phase 3 — Combine:** Creates `combined_metadata.npz`, combined `detection_data/`, and combined ROI and trace
files at the root level by merging all per-plane results.

---

### Combined metadata

**File:** `combined_metadata.npz`

| NPZ key                             | Dtype   | Shape | Description                                            |
|-------------------------------------|---------|-------|--------------------------------------------------------|
| `plane_count`                       | uint8   | (1,)  | Number of planes combined                              |
| `combined_height`                   | uint32  | (1,)  | Height of combined field of view in pixels             |
| `combined_width`                    | uint32  | (1,)  | Width of combined field of view in pixels              |
| `tau`                               | float32 | (1,)  | Calcium indicator timescale in seconds                 |
| `sampling_rate`                     | float32 | (1,)  | Per-plane sampling rate in Hz                          |
| `plane_heights`                     | uint16  | (N,)  | Per-plane frame heights                                |
| `plane_widths`                      | uint16  | (N,)  | Per-plane frame widths                                 |
| `plane_y_offsets`                   | int32   | (N,)  | Per-plane Y displacement for combined view             |
| `plane_x_offsets`                   | int32   | (N,)  | Per-plane X displacement for combined view             |
| `registered_binary_paths`           | str     | (N,)  | Relative paths to channel 1 registered binaries        |
| `registered_binary_paths_channel_2` | str     | (N,)  | Relative paths to channel 2 registered binaries (2-ch) |

---

### Detection images

Saved in `detection_data/` subdirectories at both the combined root and per-plane levels. All files are `.npy`
format, float32 dtype, with shape `(height, width)`.

**Channel 1 (always present):**

| File                      | Description                                               |
|---------------------------|-----------------------------------------------------------|
| `mean_image.npy`          | Temporal mean across all registered frames                |
| `enhanced_mean_image.npy` | High-pass filtered mean for enhanced ROI visibility       |
| `maximum_projection.npy`  | Maximum intensity projection across all frames            |
| `correlation_map.npy`     | Pixel-wise correlation map for identifying active regions |

**Channel 2 (two-channel only, same shape and dtype):**

| File                                | Description                            |
|-------------------------------------|----------------------------------------|
| `mean_image_channel_2.npy`          | Channel 2 temporal mean image          |
| `enhanced_mean_image_channel_2.npy` | Channel 2 high-pass filtered mean      |
| `maximum_projection_channel_2.npy`  | Channel 2 maximum intensity projection |
| `correlation_map_channel_2.npy`     | Channel 2 correlation map              |

---

### ROI spatial data (roi_masks.npz)

Saved at both the combined root and per-plane levels. Uses the `ROIMask` serialization format.

| NPZ key           | Dtype   | Shape           | Description                                           |
|-------------------|---------|-----------------|-------------------------------------------------------|
| `pixel_counts`    | uint32  | (num_rois,)     | Number of pixels in each ROI                          |
| `y_pixels`        | int32   | (total_pixels,) | Y-coordinates of all ROI pixels (concatenated)        |
| `x_pixels`        | int32   | (total_pixels,) | X-coordinates of all ROI pixels (concatenated)        |
| `pixel_weights`   | float32 | (total_pixels,) | Spatial filter weights for each pixel                 |
| `centroids`       | int32   | (num_rois, 2)   | ROI centroid coordinates (y, x)                       |
| `radius`          | float32 | (num_rois,)     | Fitted ellipse radius per ROI                         |
| `cluster_id`      | uint32  | (num_rois,)     | Multi-recording tracking cluster ID (0 = unclustered) |
| `recording_count` | uint16  | (num_rois,)     | Number of recordings ROI appears in                   |
| `frame_width`     | uint32  | (1,)            | Frame width in pixels                                 |

To reconstruct per-ROI pixel arrays, split the concatenated `y_pixels`, `x_pixels`, and `pixel_weights` arrays
using cumulative sums of `pixel_counts`.

Channel 2 data uses identical keys in `roi_masks_channel_2.npz`.

---

### ROI shape statistics (roi_statistics.npz)

Saved at both the combined root and per-plane levels. Companion file to `roi_masks.npz`.

| NPZ key                       | Dtype   | Shape       | Description                                            |
|-------------------------------|---------|-------------|--------------------------------------------------------|
| `footprints`                  | uint16  | (num_rois,) | Spatial scale (hop size) used during detection         |
| `compactness`                 | float32 | (num_rois,) | Ratio of actual to expected mean radius (1.0=circular) |
| `solidity`                    | float32 | (num_rois,) | Ratio of soma pixels to convex hull area               |
| `pixel_count`                 | uint32  | (num_rois,) | Total pixels in complete ROI                           |
| `aspect_ratio`                | float32 | (num_rois,) | Ellipse axis ratio indicating elongation               |
| `normalized_pixel_count`      | float32 | (num_rois,) | Pixel count normalized by expected ROI size (soma)     |
| `skewness`                    | float32 | (num_rois,) | Fluorescence skewness (NaN if unavailable)             |
| `plane_index`                 | int32   | (num_rois,) | Imaging plane index for each ROI                       |

**Optional variable-length arrays** (present only when the data exists):

| NPZ key                | Dtype  | Description                                 |
|------------------------|--------|---------------------------------------------|
| `soma_mask_counts`     | uint32 | Per-ROI pixel count for soma masks          |
| `soma_mask`            | bool   | Concatenated soma boolean masks             |
| `neuropil_mask_counts` | uint32 | Per-ROI pixel count for neuropil masks      |
| `neuropil_mask`        | int32  | Concatenated raveled neuropil pixel indices |
| `overlap_mask_counts`  | uint32 | Per-ROI pixel count for overlap masks       |
| `overlap_mask`         | bool   | Concatenated overlap boolean masks          |

Variable-length arrays use the same split-by-counts pattern as `roi_masks.npz`.

Channel 2 data uses identical keys in `roi_statistics_channel_2.npz`.

---

### Fluorescence traces and classification

Saved at both the combined root and per-plane levels. All files are `.npy` format, float32 dtype.

**Channel 1 (always present):**

| File                          | Shape              | Description                                                            |
|-------------------------------|--------------------|------------------------------------------------------------------------|
| `cell_fluorescence.npy`       | (num_rois, frames) | Raw somatic fluorescence traces                                        |
| `neuropil_fluorescence.npy`   | (num_rois, frames) | Neuropil fluorescence traces                                           |
| `subtracted_fluorescence.npy` | (num_rois, frames) | Neuropil-and-baseline-subtracted fluorescence                          |
| `spikes.npy`                  | (num_rois, frames) | Deconvolved spike estimates                                            |
| `cell_classification.npy`     | (num_rois, 2)      | Column 0: is_cell label (1.0 or 0.0), column 1: classifier probability |

If `spike_deconvolution.extract_spikes` is False, both `subtracted_fluorescence.npy` and `spikes.npy` are filled with
zeroes.

**Channel 2 (two-channel only, same shapes):**

| File                                    | Description                           |
|-----------------------------------------|---------------------------------------|
| `cell_fluorescence_channel_2.npy`       | Channel 2 raw somatic fluorescence    |
| `neuropil_fluorescence_channel_2.npy`   | Channel 2 neuropil fluorescence       |
| `subtracted_fluorescence_channel_2.npy` | Channel 2 subtracted fluorescence     |
| `spikes_channel_2.npy`                  | Channel 2 deconvolved spikes          |
| `cell_classification_channel_2.npy`     | Channel 2 classification results      |

**Optional colocalization files (combined root and per-plane):**

| File                                  | Shape           | Description                                                                                                    |
|---------------------------------------|-----------------|----------------------------------------------------------------------------------------------------------------|
| `cell_colocalization.npy`             | (num_rois, 2)   | Column 0: is_colocalized label (1.0 or 0.0), column 1: probability                                             |
| `corrected_structural_mean_image.npy` | (height, width) | Bleed-through-corrected structural channel mean (dual-channel recordings where only one channel is functional) |

---

### Per-plane registration data

Saved in `plane_N/registration_data/`. All files are `.npy` format.

| File                                     | Dtype   | Shape                              | Description                                                        |
|------------------------------------------|---------|------------------------------------|--------------------------------------------------------------------|
| `reference_image.npy`                    | float32 | (height, width)                    | Template image used for alignment                                  |
| `bad_frames.npy`                         | bool    | (num_frames,)                      | Frames flagged for excessive motion                                |
| `rigid_y_offsets.npy`                    | int32   | (num_frames,)                      | Rigid registration Y displacement per frame                        |
| `rigid_x_offsets.npy`                    | int32   | (num_frames,)                      | Rigid registration X displacement per frame                        |
| `rigid_correlations.npy`                 | float32 | (num_frames,)                      | Phase correlation quality per frame                                |
| `nonrigid_y_offsets.npy`                 | float32 | (num_frames, num_blocks)           | Nonrigid Y displacement per block per frame                        |
| `nonrigid_x_offsets.npy`                 | float32 | (num_frames, num_blocks)           | Nonrigid X displacement per block per frame                        |
| `nonrigid_correlations.npy`              | float32 | (num_frames, num_blocks)           | Nonrigid correlation quality per block per frame                   |
| `principal_component_extreme_images.npy` | float32 | (2, num_components, height, width) | Mean images at PC extremes (0=low, 1=high)                         |
| `principal_component_projections.npy`    | float32 | (num_frames, num_components)       | Frame projections onto principal components                        |
| `principal_component_shift_metrics.npy`  | float32 | (num_components, 3)                | Columns: mean rigid shift, mean nonrigid shift, max nonrigid shift |

---

### Per-plane binary data

| File                 | Format             | Description                                                          |
|----------------------|--------------------|----------------------------------------------------------------------|
| `channel_1_data.bin` | Contiguous float32 | Motion-corrected frames: `[frame0_row0_col0, frame0_row0_col1, ...]` |
| `channel_2_data.bin` | Contiguous float32 | Channel 2 motion-corrected frames (two-channel only)                 |

Binary files store frames as contiguous float32 arrays. Each frame has `height × width` values. Read with
`np.memmap(path, dtype=np.float32, mode='r', shape=(frame_count, height, width))` using dimensions from
`runtime_data.yaml`.

---

### Per-plane runtime metadata (runtime_data.yaml)

A YAML file containing scalar metadata from all processing stages. Key sections:

| Section        | Key fields                                                                                                                                                                                                                                                                                                                                                                                |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `io`           | `frame_height`, `frame_width`, `frame_count`, `sampling_rate`, `plane_index`                                                                                                                                                                                                                                                                                                              |
| `registration` | `valid_y_range`, `valid_x_range`, `bidirectional_phase_offset`, `normalization_minimum`, `normalization_maximum`                                                                                                                                                                                                                                                                          |
| `detection`    | `roi_diameter`, `aspect_ratio`                                                                                                                                                                                                                                                                                                                                                            |
| `timing`       | `binarization_time`, `registration_time`, `two_step_registration_time`, `registration_metrics_time`, `detection_time`, `extraction_time`, `classification_time`, `deconvolution_time`, `detection_time_channel_2`, `extraction_time_channel_2`, `classification_time_channel_2`, `deconvolution_time_channel_2`, `total_plane_time`, `date_processed`, `python_version`, `cindra_version` |

Array fields from registration, detection, and extraction are saved as separate `.npy` files (documented above)
and set to None in the YAML.

---

### Data type conventions

| Category            | Dtype   | Examples                                       |
|---------------------|---------|------------------------------------------------|
| Pixel coordinates   | int32   | y_pixels, x_pixels, centroids, motion offsets  |
| Images and traces   | float32 | mean_image, fluorescence, spikes, correlations |
| Counts / dimensions | uint32  | pixel_counts, frame_count, combined_height     |
| Small counts        | uint16  | plane_heights, plane_widths, recording_count   |
| Booleans            | bool    | bad_frames, soma_mask, overlap_mask            |
| Plane indices       | int32   | plane_index                                    |
| Plane counts        | uint8   | plane_count                                    |

All `.npy` files are saved with `allow_pickle=False`. Arrays support memory-mapped loading via
`np.load(path, mmap_mode='r+')` for efficient access to large datasets.

---

## Multi-recording compatibility requirements

For recordings intended for multi-recording processing, single-recording processing must complete all three phases 
(binarize, process, combine). The multi-recording pipeline locates single-recording output by searching for 
`combined_metadata.npz` within the recording directory tree. The pipeline always generates combined output and 
preserves registered binary files, so no special configuration is required.

---

## Related skills

| Skill                             | Relationship                                                             |
|-----------------------------------|--------------------------------------------------------------------------|
| `/single-recording-configuration` | Configuration parameter reference for the single-recording pipeline      |
| `/single-recording-processing`    | Processing workflow that produces this output                            |
| `/multi-recording-results`        | Companion output data reference for the multi-recording pipeline         |
| `/multi-recording-configuration`  | Multi-recording configuration requires these outputs as prerequisites    |
| `/visualization`                  | Launch viewers and query tools to visualize and inspect this output data |

---

## Verification checklist

Use `verify_single_recording_output_tool` to automate this verification. The tool checks all expected files and NPZ
keys and returns a completeness verdict with any missing items listed. Fall back to the manual checklist below only
if the MCP tool is unavailable. Replace N with the expected plane count from the acquisition parameters.

```text
Single-Recording Output Completeness:
Root-level files:
- [ ] `configuration.yaml` exists
- [ ] `acquisition_parameters.yaml` exists
- [ ] `combined_metadata.npz` exists and contains `plane_count`, `combined_height`, `combined_width` keys

Combined detection images (cindra/detection_data/):
- [ ] `mean_image.npy` exists
- [ ] `enhanced_mean_image.npy` exists
- [ ] `maximum_projection.npy` exists
- [ ] `correlation_map.npy` exists
- [ ] Channel 2 variants exist if `main.two_channels` is True and both channels are functional

Combined extraction data (cindra/):
- [ ] `roi_masks.npz` exists and contains `pixel_counts`, `y_pixels`, `x_pixels`, `pixel_weights` keys
- [ ] `roi_statistics.npz` exists and contains `footprints`, `compactness`, `plane_index` keys
- [ ] `cell_fluorescence.npy` exists with shape (num_rois, num_frames)
- [ ] `neuropil_fluorescence.npy` exists with shape matching cell_fluorescence
- [ ] `subtracted_fluorescence.npy` exists with shape matching cell_fluorescence
- [ ] `spikes.npy` exists with shape matching cell_fluorescence (if spike_deconvolution.extract_spikes is True)
- [ ] `cell_classification.npy` exists with shape (num_rois, 2)
- [ ] Channel 2 trace and classification files exist if both channels are functional

Per-plane directories (cindra/plane_0/ through cindra/plane_{N-1}/):
- [ ] Each expected plane directory exists
- [ ] Each plane contains `runtime_data.yaml` with non-zero `io.frame_count` and `io.sampling_rate`
- [ ] Each plane contains `channel_1_data.bin` (registered binary)
- [ ] Each plane contains `channel_2_data.bin` if `main.two_channels` is True

Per-plane registration data (plane_N/registration_data/):
- [ ] `reference_image.npy` exists
- [ ] `bad_frames.npy` exists
- [ ] `rigid_y_offsets.npy` and `rigid_x_offsets.npy` exist
- [ ] `rigid_correlations.npy` exists
- [ ] Nonrigid arrays exist if nonrigid_registration.enabled is True
- [ ] PC metric arrays exist if registration.registration_metric_principal_components > 0

Per-plane detection and extraction data (plane_N/):
- [ ] `detection_data/mean_image.npy` and `detection_data/enhanced_mean_image.npy` exist
- [ ] `roi_masks.npz` and `roi_statistics.npz` exist
- [ ] Fluorescence trace .npy files exist with consistent shapes across all traces
- [ ] `cell_classification.npy` exists with shape (num_rois, 2)

Multi-recording readiness (if multi-recording processing is planned):
- [ ] `combined_metadata.npz` contains `registered_binary_paths` key
- [ ] All registered binary files referenced in `registered_binary_paths` exist on disk
```
