# Single-recording output formats

Documents every file, array shape, dtype, and NPZ key the single-recording pipeline writes. This reference is loaded on
demand by `/single-recording-results`.

---

## Combined metadata

**File:** `combined_metadata.npz`

| NPZ key                             | Dtype   | Shape | Description                                            |
|-------------------------------------|---------|-------|--------------------------------------------------------|
| `plane_count`                       | uint32  | (1,)  | Number of planes combined                              |
| `frame_count`                       | uint32  | (1,)  | Frames the combined traces span                        |
| `plane_frame_counts`                | uint32  | (N,)  | Per-plane frame counts recorded during binarization    |
| `combined_height`                   | uint32  | (1,)  | Height of combined field of view in pixels             |
| `combined_width`                    | uint32  | (1,)  | Width of combined field of view in pixels              |
| `tau`                               | float32 | (1,)  | Calcium indicator timescale in seconds                 |
| `sampling_rate`                     | float32 | (1,)  | Per-plane sampling rate in Hz                          |
| `plane_heights`                     | uint16  | (N,)  | Per-plane frame heights                                |
| `plane_widths`                      | uint16  | (N,)  | Per-plane frame widths                                 |
| `plane_y_offsets`                   | int32   | (N,)  | Per-plane Y displacement for combined view             |
| `plane_x_offsets`                   | int32   | (N,)  | Per-plane X displacement for combined view             |
| `registered_binary_paths`           | str     | (N,)  | Relative paths to channel 1 registered binaries        |
| `registered_binary_paths_channel_2` | str     | (N,)  | Channel 2 paths, when both channels are functional     |

`frame_count` is the frame count of the shortest plane that contributed traces, which is what the combined traces were
trimmed to. `plane_frame_counts` holds each plane's own count, which binarization makes identical across the planes of
the recording, so every entry equals `frame_count` once every plane has completed.

---

## Detection images

Saved in `detection_data/` subdirectories at both the combined root and per-plane levels. All files are `.npy` format,
float32 dtype, with shape `(height, width)`.

**Channel 1 (always present):**

| File                      | Description                                               |
|---------------------------|-----------------------------------------------------------|
| `mean_image.npy`          | Mean of the binned frames in the valid range              |
| `enhanced_mean_image.npy` | High-pass filtered mean for enhanced ROI visibility       |
| `maximum_projection.npy`  | Per-pixel max of the temporally filtered binned movie     |
| `correlation_map.npy`     | Per-pixel maximum across the detection scale pyramid      |

**Channel 2 (two-channel only, same shape and dtype):**

| File                                | Description                            | Presence condition          |
|-------------------------------------|----------------------------------------|-----------------------------|
| `mean_image_channel_2.npy`          | Channel 2 temporal mean image          | Every two-channel recording |
| `enhanced_mean_image_channel_2.npy` | Channel 2 high-pass filtered mean      | Both channels functional    |
| `maximum_projection_channel_2.npy`  | Channel 2 maximum intensity projection | Both channels functional    |
| `correlation_map_channel_2.npy`     | Channel 2 correlation map              | Both channels functional    |

`mean_image_channel_2.npy` appears at both levels for every two-channel recording, because the register phase writes a
mean image for each channel it rewrites without testing whether either channel is functional. The other three come from
channel 2 detection, which runs only when both `main.first_channel_functional` and `main.second_channel_functional` are
True. The combined `maximum_projection_channel_2.npy` additionally requires at least one plane to hold a channel 1
maximum projection, which every plane that completed detection does.

---

## ROI spatial data (roi_masks.npz)

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

To reconstruct per-ROI pixel arrays, split the concatenated `y_pixels`, `x_pixels`, and `pixel_weights` arrays using
cumulative sums of `pixel_counts`.

Channel 2 data uses identical keys in `roi_masks_channel_2.npz`.

---

## ROI shape statistics (roi_statistics.npz)

Saved at both the combined root and per-plane levels. Companion file to `roi_masks.npz`.

| NPZ key                  | Dtype   | Shape       | Description                                            |
|--------------------------|---------|-------------|--------------------------------------------------------|
| `footprints`             | uint16  | (num_rois,) | Index of the detection scale the ROI was found at      |
| `compactness`            | float32 | (num_rois,) | Ratio of actual to expected mean radius (1.0=circular) |
| `solidity`               | float32 | (num_rois,) | Ratio of soma pixels to convex hull area               |
| `pixel_count`            | uint32  | (num_rois,) | Total pixels in complete ROI                           |
| `aspect_ratio`           | float32 | (num_rois,) | Ellipse axis ratio indicating elongation               |
| `normalized_pixel_count` | float32 | (num_rois,) | Pixel count normalized by expected ROI size (soma)     |
| `skewness`               | float32 | (num_rois,) | Fluorescence skewness (NaN if unavailable)             |
| `plane_index`            | int32   | (num_rois,) | Imaging plane index for each ROI                       |

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

## Fluorescence traces and classification

Saved at both the combined root and per-plane levels. All files are `.npy` format, float32 dtype.

At the combined root, `frames` is the frame count of the shortest plane that contributed traces, recorded as
`frame_count` in `combined_metadata.npz`. Combination trims every plane's traces to that count rather than padding the
shorter ones, and logs a warning naming the range. Planes that did not complete extraction contribute no rows and are
excluded from the trim target. At the per-plane level, `frames` is that plane's own `io.frame_count`.

**Channel 1 (always present):**

| File                          | Shape              | Description                                                            |
|-------------------------------|--------------------|------------------------------------------------------------------------|
| `cell_fluorescence.npy`       | (num_rois, frames) | Raw somatic fluorescence traces                                        |
| `neuropil_fluorescence.npy`   | (num_rois, frames) | Neuropil fluorescence traces                                           |
| `subtracted_fluorescence.npy` | (num_rois, frames) | Neuropil-and-baseline-subtracted fluorescence                          |
| `spikes.npy`                  | (num_rois, frames) | Deconvolved spike estimates                                            |
| `cell_classification.npy`     | (num_rois, 2)      | Column 0: is_cell label (1.0 or 0.0), column 1: classifier probability |

If `spike_deconvolution.extract_spikes` is False, both `subtracted_fluorescence.npy` and `spikes.npy` are filled with
zeroes. In that case `query_traces_tool` returns an all-zero trace with `success=true` rather than an error, so an
all-zero spike or corrected trace can mean deconvolution was disabled rather than the absence of activity.

**Channel 2 (two-channel only, same shapes):**

| File                                    | Description                        |
|-----------------------------------------|------------------------------------|
| `cell_fluorescence_channel_2.npy`       | Channel 2 raw somatic fluorescence |
| `neuropil_fluorescence_channel_2.npy`   | Channel 2 neuropil fluorescence    |
| `subtracted_fluorescence_channel_2.npy` | Channel 2 subtracted fluorescence  |
| `spikes_channel_2.npy`                  | Channel 2 deconvolved spikes       |
| `cell_classification_channel_2.npy`     | Channel 2 classification results   |

`cell_fluorescence_channel_2.npy` and `neuropil_fluorescence_channel_2.npy` are written in every `plane_N/` directory
for every two-channel recording. A structural (non-functional) channel 2 reuses the channel 1 masks for extraction, so
those two traces carry one row per channel 1 ROI. It also skips classification and spike deconvolution entirely, leaving
`subtracted_fluorescence_channel_2.npy`, `spikes_channel_2.npy`, and `cell_classification_channel_2.npy` absent at both
levels.

The combined root holds no channel 2 trace or classification file at all when channel 2 is structural. Combination gates
the whole channel 2 aggregation, including `roi_masks_channel_2.npz` and `roi_statistics_channel_2.npz`, on both
channels being functional, so the root lacks even the two trace files every plane directory carries. Read structural
channel 2 traces from `plane_N/` rather than from the root.

**Optional colocalization files (combined root and per-plane):**

| File                                  | Shape           | Description                                                     |
|---------------------------------------|-----------------|-----------------------------------------------------------------|
| `cell_colocalization.npy`             | (num_rois, 2)   | Channel-2 colocalization. Columns depend on the extraction path |
| `corrected_structural_mean_image.npy` | (height, width) | Bleed-through-corrected structural channel mean                 |

The `cell_colocalization.npy` column semantics depend on the extraction path. When one channel is structural,
intensity-based colocalization runs (and also writes `corrected_structural_mean_image.npy`): column 0 is the
is_colocalized label (1.0 or 0.0) and column 1 the probability. When both channels are functional, spatial
colocalization runs instead: column 0 is the matched channel-2 ROI index (-1 if unmatched) and column 1 the overlap
score. `query_roi_statistics_tool` surfaces this as a per-ROI `colocalization` pair plus top-level `colocalization_mode`
and `colocalization_columns`.

The metadata tool's `two_channels` flag means channel 2 is present AND functional, not merely that the recording is
two-channel. A recording with a structural (non-functional) channel 2 reports `two_channels=False` yet still writes
`cell_colocalization.npy`. Use the presence of `cell_colocalization.npy` (not `two_channels`) as the signal that
channel-2 colocalization was computed.

---

## Per-plane registration data

Saved in `plane_N/registration_data/`. All files are `.npy` format.

| File                                     | Dtype   | Shape                              | Description                                                       |
|------------------------------------------|---------|------------------------------------|-------------------------------------------------------------------|
| `reference_image.npy`                    | float32 | (height, width)                    | Template image used for alignment                                 |
| `bad_frames.npy`                         | bool    | (num_frames,)                      | Frames flagged for excessive motion                               |
| `rigid_y_offsets.npy`                    | int32   | (num_frames,)                      | Rigid registration Y displacement per frame                       |
| `rigid_x_offsets.npy`                    | int32   | (num_frames,)                      | Rigid registration X displacement per frame                       |
| `rigid_correlations.npy`                 | float32 | (num_frames,)                      | Phase correlation quality per frame                               |
| `nonrigid_y_offsets.npy`                 | float32 | (num_frames, num_blocks)           | Nonrigid Y displacement per block per frame                       |
| `nonrigid_x_offsets.npy`                 | float32 | (num_frames, num_blocks)           | Nonrigid X displacement per block per frame                       |
| `nonrigid_correlations.npy`              | float32 | (num_frames, num_blocks)           | Nonrigid correlation quality per block per frame                  |
| `principal_component_extreme_images.npy` | float32 | (2, num_components, height, width) | Mean images at PC extremes (0=low, 1=high)                        |
| `principal_component_projections.npy`    | float32 | (num_samples, num_components)      | Subsampled-frame projections onto principal components            |
| `principal_component_shift_metrics.npy`  | float32 | (num_components, 3)                | Columns: rigid magnitude, mean nonrigid shift, max nonrigid shift |

The principal-component arrays come from a cropped subsample of the registered movie, not from every frame.
`num_samples` is `min(frame_count, 2000)` when the plane holds fewer than 5000 frames or either frame dimension exceeds
700 pixels, and 5000 otherwise. The `height` and `width` of `principal_component_extreme_images.npy` likewise span the
registration valid range rather than the full frame.

---

## Per-plane binary data

| File                 | Format           | Description                                                          |
|----------------------|------------------|----------------------------------------------------------------------|
| `channel_1_data.bin` | Contiguous int16 | Motion-corrected frames: `[frame0_row0_col0, frame0_row0_col1, ...]` |
| `channel_2_data.bin` | Contiguous int16 | Channel 2 motion-corrected frames (two-channel only)                 |

Binary files store frames as contiguous int16 arrays. Each frame has `height × width` values. Read with
`np.memmap(path, dtype=np.int16, mode='r', shape=(frame_count, height, width))` using dimensions from
`runtime_data.yaml`. Read each plane's `frame_count` from its own `runtime_data.yaml`, which is the sole authority on
how many frames its binaries hold, because binarization discards the frames of an incomplete final interleave cycle.

---

## Per-plane runtime metadata (runtime_data.yaml)

A YAML file containing scalar metadata from all processing stages. Key sections:

| Section        | Key fields                                                                                                                                        |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `io`           | `frame_height`, `frame_width`, `frame_count`, `sampling_rate`, `plane_index`                                                                      |
| `registration` | `valid_y_range`, `valid_x_range`, `bidirectional_phase_offset`, `bidirectional_phase_corrected`, `normalization_minimum`, `normalization_maximum` |
| `detection`    | `roi_diameter`, `roi_diameter_channel_2`, `aspect_ratio`                                                                                          |
| `timing`       | Stage durations, phase totals, worker counts, and version stamps, itemized below.                                                                 |

The `timing` section stores every duration as an integer number of seconds:

- Stage durations: `binarization_time`, `registration_time`, `two_step_registration_time`, `registration_metrics_time`,
  `detection_time`, `extraction_time`, `classification_time`, `deconvolution_time`.
- Channel 2 stage durations: `detection_time_channel_2`, `extraction_time_channel_2`, `classification_time_channel_2`,
  `deconvolution_time_channel_2`.
- Phase totals: `total_registration_time` covers motion correction and the registration quality metrics computation.
  `total_processing_time` covers ROI detection, trace extraction, classification, and spike deconvolution.
- Worker counts: `registration_workers` and `processing_workers` record the allocation each stage used.
- Version stamps: `date_processed`, `python_version`, `cindra_version`.

`query_single_recording_metadata_tool` surfaces the phase totals and both worker counts in its `plane_timing` entries,
so the per-plane worker allocation is readable without opening `runtime_data.yaml`.

Array fields from registration, detection, and extraction are saved as separate `.npy` files (documented above) and set
to None in the YAML.

---

## Data type conventions

| Category            | Dtype   | Examples                                       |
|---------------------|---------|------------------------------------------------|
| Pixel coordinates   | int32   | y_pixels, x_pixels, centroids, rigid offsets   |
| Images and traces   | float32 | mean_image, fluorescence, spikes, correlations |
| Counts / dimensions | uint32  | pixel_counts, frame_count, combined_height     |
| Small counts        | uint16  | plane_heights, plane_widths, recording_count   |
| Booleans            | bool    | bad_frames, soma_mask, overlap_mask            |
| Plane indices       | int32   | plane_index                                    |
| Plane counts        | uint32  | plane_count                                    |

Extraction trace, classification, and colocalization `.npy` files and all `.npz` archives are saved with
`allow_pickle=False`. Detection and registration `.npy` files use NumPy save defaults but contain only numeric arrays
that load safely with `allow_pickle=False`. Arrays support memory-mapped loading via `np.load(path, mmap_mode='r')` for
efficient access to large datasets.
