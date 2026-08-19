---
name: multi-recording-results
description: >-
  Complete reference for multi-recording pipeline output data formats. Documents every file, directory, array shape,
  dtype, and NPZ key produced by the pipeline, plus verification checklists for output completeness. Use when evaluating
  multi-recording processing results or when the user asks about multi-recording output data.
user-invocable: true
---

# Multi-recording results data reference

Complete output data format documentation for the multi-recording (cross-recording) cindra ROI tracking pipeline.

---

## Scope

**Covers:**
- Complete output data reference: every file, directory, array shape, dtype, and NPZ key produced by the pipeline
- Directory structure for per-recording multi-recording output
- Processing phase and file creation timeline
- Data serialization formats (ROIMask, ROIStatistics)
- Data type conventions and memory-mapping guidance
- Output completeness verification

**Does not cover:**
- Configuration parameters or prerequisites (see `/multi-recording-configuration`)
- Processing workflow, batch operations, or status monitoring (see `/multi-recording-processing`)
- Single-recording output data formats (see `/single-recording-results`)

---

## Agent requirements

You MUST use the cindra MCP query and verification tools to inspect output data rather than reading output files
directly when a tool exists for the task. If MCP tools are not available, invoke `/cindra-mcp-environment-setup` to
diagnose and resolve connectivity issues.

---

## Available tools

Use these cindra MCP tools to query and verify multi-recording output data programmatically. Prefer these over manual
file reads whenever possible.

### Verification tool

| Tool                                 | Purpose                                                        |
|--------------------------------------|----------------------------------------------------------------|
| `verify_multi_recording_output_tool` | Verifies completeness of all expected output files per dataset |

### Query tools

| Tool                                              | Purpose                                                                               |
|---------------------------------------------------|---------------------------------------------------------------------------------------|
| `query_multi_recording_overview_tool`             | Queries dataset structure, per-recording mask counts, timing, and completion status   |
| `query_multi_recording_registration_quality_tool` | Queries deformation field statistics and transformed image availability per recording |
| `query_multi_recording_tracking_summary_tool`     | Queries template count, recording count distribution, and cluster statistics          |
| `query_roi_statistics_tool`                       | Queries per-ROI spatial statistics (use `dataset` parameter for multi-recording)      |
| `query_traces_tool`                               | Queries fluorescence traces for ROIs (use `dataset` parameter for multi-recording)    |
| `query_cross_recording_traces_tool`               | Queries fluorescence traces for specific ROIs across all recordings in a dataset      |

### Recommended query order

1. `query_multi_recording_overview_tool`: understand dataset composition and processing completeness
2. `query_multi_recording_registration_quality_tool`: review deformation field magnitudes and transformed image
   availability
3. `query_multi_recording_tracking_summary_tool`: review template counts, cluster IDs, and recording count distribution
4. `query_roi_statistics_tool` (with `dataset` parameter): inspect per-ROI spatial statistics and tracking metadata
5. `query_traces_tool` (with `dataset` parameter): examine tracked ROI fluorescence activity per recording
6. `query_cross_recording_traces_tool`: compare longitudinal activity patterns for the same ROIs across sessions

**Important:** Deformation field magnitude does not indicate registration quality. It only reflects how much the field
of view shifted between sessions. Similarly, an ROI appearing in fewer recordings does not indicate tracking failure.
ROIs can be active in some sessions and inactive in others. The only reliable way to assess cross-day registration
quality is visual inspection: confirm that backward-deformed templates overlap with the same structures across days. Use
`/visualization` for this.

### Query tool argument semantics

The `output_root` argument for the verify and query tools must be the pipeline output root, the parent of the `cindra/`
folder. This equals the `output_roots` entries passed to `prepare_multi_recording_batch_tool`, which are the output
roots the single-recording pipeline wrote into (the `output_roots` given to `prepare_single_recording_batch_tool`,
echoed back as each manifest entry's `output_root`). It is not the `raw_data_path` holding the recording's TIFF files,
because the query tools read pipeline output rather than raw imaging data. The tools resolve the `cindra/`
subdirectory automatically.

The per-recording entries these tools return name two further paths under distinct keys.
`verify_multi_recording_output_tool` and `query_multi_recording_overview_tool` both report `dataset_output_path`, the
recording's own directory inside the dataset tree (`{cindra_root}/multi_recording/{dataset}/`), and the overview also
reports `cindra_path`, that recording's single-recording cindra output directory. Neither is an `output_root`, so pass
neither one back into a tool taking `output_root`.

The ROI indices accepted by `query_traces_tool`, `query_roi_statistics_tool`, and `query_cross_recording_traces_tool`
are 0-based positional row indices into the per-recording arrays. They are not the tracking `cluster_id`, which is a
separate 1-based identity (0 = unclustered). Out-of-range indices are silently dropped, so a confidently "successful"
empty result from `query_roi_statistics_tool` or `query_cross_recording_traces_tool` can mean a wrong index rather than
missing data. `query_traces_tool` drops out-of-range indices silently too, but fails with "No valid ROI indices
provided" when every requested index is out of range.

`query_cross_recording_traces_tool` excludes recordings with incomplete extraction into a `skipped_recordings` list, so
"all recordings" really means all recordings with complete extraction. Surface `skipped_recordings` to the user when it
is present.

The `dataset` argument is required by `verify_multi_recording_output_tool`, `query_multi_recording_overview_tool`,
`query_multi_recording_registration_quality_tool`, `query_multi_recording_tracking_summary_tool`, and
`query_cross_recording_traces_tool`, and it is what switches `query_roi_statistics_tool` and `query_traces_tool` into
multi-recording mode. It resolves as an exact directory lookup (`{cindra_root}/multi_recording/{dataset}`) against a
name the pipeline lowercases at preparation time, so pass the value `resolve_dataset_name_tool` or
`prepare_multi_recording_batch_tool` returned rather than the user's original casing. `query_roi_statistics_tool` and
`query_traces_tool` additionally accept `recording_index` (0-based, default 0). It indexes the dataset's own
`dataset_output_paths` list, so index 0 is the main recording rather than the recording `output_root` names. Without
it, a per-recording query silently reports the main recording whichever recording you addressed, so iterate the index
across the dataset when comparing recordings.

`query_cross_recording_traces_tool` and `query_traces_tool` accept at most 50 ROI indices per call, and
`query_roi_statistics_tool` returns at most 500 ROIs, so batch a larger pull across several calls.
`query_multi_recording_tracking_summary_tool` reports at most 200 per-template entries, and sets `templates_truncated:
true` with `templates_shown: 200` when the dataset holds more. Read those two keys before summarizing tracking quality,
because the cluster statistics the tool returns alongside them cover every template while the `templates` list does
not.

`query_traces_tool` and `query_cross_recording_traces_tool` also take `trace_type` (`fluorescence`, `neuropil`,
`corrected`, or `spikes`), `downsample_factor` (1 for none, N for every Nth sample, values below 1 clamped to 1), and a
`start_frame`/`end_frame` window applied before downsampling. `query_roi_statistics_tool` takes `sort_by` (`skewness`,
`compactness`, `footprint`, `aspect_ratio`, `pixel_count`, `solidity`, or `normalized_pixel_count`) and `top_n`.

---

## Output data reference

All results are saved under the dataset output root, `{cindra_root}/multi_recording/{dataset_name}/`, within each
recording's cindra output directory. The pipeline produces per-recording output for every recording, plus a shared
configuration file in the main recording (first after natural sorting). Channel 2 masks, statistics, and traces are only
present for dual-channel recordings where both channels are functional. `transformed_mean_image_channel_2.npy` is the
one exception: it appears for any dual-channel recording, because binarization computes the channel-2 mean image even
when the second channel is structural.

### Directory structure

```text
{cindra_root}/multi_recording/{dataset_name}/
├── multi_recording_configuration.yaml              # Shared config (main recording only)
├── multi_recording_tracker.yaml                    # Processing tracker (main recording only)
├── multi_recording_runtime_data.yaml               # Per-recording runtime metadata
├── registration_arrays/                            # Diffeomorphic registration data
│   ├── deform_field_y.npy
│   ├── deform_field_x.npy
│   ├── transformed_mean_image.npy
│   ├── transformed_enhanced_mean_image.npy
│   └── transformed_maximum_projection.npy
├── registration_deformed_masks.npz                 # Forward-deformed ROI masks
├── tracking_template_masks.npz                     # Consensus template masks
├── roi_masks.npz                                   # Backward-transformed extraction masks
├── roi_statistics.npz                              # Backward-transformed shape statistics
├── cell_fluorescence.npy                           # Extracted fluorescence traces
├── neuropil_fluorescence.npy
├── subtracted_fluorescence.npy
├── spikes.npy
└── cell_colocalization.npy                         # Dual-channel only
```

### Processing phase and file creation timeline

**Phase 1 (Discovery):** Executed once across all recordings. `multi_recording_configuration.yaml`,
`multi_recording_tracker.yaml` (main recording only), and `multi_recording_runtime_data.yaml` for each recording
already exist before this phase runs, because `prepare_multi_recording_batch_tool` (or the CLI pipeline entry) writes
them while bootstrapping the dataset. The phase runs the following sub-steps:

1. **Context resolution:** Reloads the bootstrap the prepare step wrote. The discovery job is load-only here and fails
   outright if a recording's `multi_recording_runtime_data.yaml` is missing.
2. **ROI selection:** Filters single-recording ROIs by probability, size, and MROI margins. Updates
   `multi_recording_runtime_data.yaml` with selected ROI indices.
3. **Registration:** Computes diffeomorphic deformation fields, transforms reference images and selected ROI masks to
   shared visual space. Creates `registration_arrays/` and `registration_deformed_masks.npz`.
4. **Tracking:** Clusters deformed masks across recordings using Jaccard distance and hierarchical clustering. Creates
   `tracking_template_masks.npz`.
5. **Backward projection:** Applies inverse deformation to project template masks back to each recording's native
   coordinate system. Creates `roi_masks.npz` and `roi_statistics.npz`.

**Phase 2 (Extraction):** Executed independently per recording (parallelizable). Extracts fluorescence from registered
binary data using backward-transformed template masks. Creates `cell_fluorescence.npy`, `neuropil_fluorescence.npy`,
`subtracted_fluorescence.npy`, `spikes.npy`, and optionally `cell_colocalization.npy`.

---

### Registration arrays

Saved in `registration_arrays/` subdirectory. All files are `.npy` format, float32 dtype.

**Deformation fields:**

| File                 | Shape           | Description                                              |
|----------------------|-----------------|----------------------------------------------------------|
| `deform_field_y.npy` | (height, width) | Y-dimension displacement field for diffeomorphic warping |
| `deform_field_x.npy` | (height, width) | X-dimension displacement field for diffeomorphic warping |

**Channel 1 transformed images:**

| File                                  | Shape           | Description                                       |
|---------------------------------------|-----------------|---------------------------------------------------|
| `transformed_mean_image.npy`          | (height, width) | Mean image warped to shared visual space          |
| `transformed_enhanced_mean_image.npy` | (height, width) | Enhanced mean image warped to shared visual space |
| `transformed_maximum_projection.npy`  | (height, width) | Maximum projection warped to shared visual space  |

**Channel 2 transformed images (same shape and dtype). The mean image appears for any dual-channel recording, while the
enhanced mean and the maximum projection require both channels to be functional:**

| File                                            | Description                                     |
|-------------------------------------------------|-------------------------------------------------|
| `transformed_mean_image_channel_2.npy`          | Channel 2 mean image in shared visual space     |
| `transformed_enhanced_mean_image_channel_2.npy` | Channel 2 enhanced mean in shared visual space  |
| `transformed_maximum_projection_channel_2.npy`  | Channel 2 max projection in shared visual space |

---

### Registration deformed masks

**File:** `registration_deformed_masks.npz` (channel 1), `registration_deformed_masks_channel_2.npz` (channel 2)

Uses the `ROIMask.save_list()` serialization format. Contains the selected single-recording ROI masks after forward
deformation to the shared visual space.

| NPZ key           | Dtype   | Shape           | Description                                            |
|-------------------|---------|-----------------|--------------------------------------------------------|
| `pixel_counts`    | uint32  | (num_rois,)     | Number of pixels in each deformed ROI                  |
| `y_pixels`        | int32   | (total_pixels,) | Y-coordinates of all ROI pixels (concatenated)         |
| `x_pixels`        | int32   | (total_pixels,) | X-coordinates of all ROI pixels (concatenated)         |
| `pixel_weights`   | float32 | (total_pixels,) | Spatial filter weights for each pixel                  |
| `centroids`       | int32   | (num_rois, 2)   | ROI centroid coordinates (y, x) in shared visual space |
| `radius`          | float32 | (num_rois,)     | Fitted radius per ROI                                  |
| `cluster_id`      | uint32  | (num_rois,)     | Tracking cluster ID (0 = unclustered)                  |
| `recording_count` | uint16  | (num_rois,)     | Always 0 in this file, set only on template masks      |
| `frame_width`     | uint32  | (1,)            | Frame width in pixels                                  |

---

### Tracking template masks

**File:** `tracking_template_masks.npz` (channel 1), `tracking_template_masks_channel_2.npz` (channel 2)

Uses the same `ROIMask.save_list()` serialization format as deformed masks. Contains consensus template masks generated
by clustering deformed ROI masks across recordings. Each template represents an ROI reliably identified across multiple
recordings. `cluster_id` uniquely identifies each tracked ROI. `recording_count` records how many recordings contributed
to the template.

---

### Backward-transformed extraction data (roi_masks.npz, roi_statistics.npz)

Saved at the dataset output root. Uses the same `ROIStatistics.save_list()` serialization format as single-recording
output. Contains template masks projected back to the recording's native coordinate system via inverse deformation, with
full shape statistics computed for each ROI.

**roi_masks.npz** holds the same NPZ keys and dtypes as the tracking template masks (see above).

**roi_statistics.npz:**

| NPZ key                  | Dtype   | Shape       | Description                                            |
|--------------------------|---------|-------------|--------------------------------------------------------|
| `footprints`             | uint16  | (num_rois,) | Set to 0 for tracked ROIs (no meaningful hop size)     |
| `compactness`            | float32 | (num_rois,) | Ratio of actual to expected mean radius (1.0=circular) |
| `solidity`               | float32 | (num_rois,) | Ratio of soma pixels to convex hull area               |
| `pixel_count`            | uint32  | (num_rois,) | Total pixels in complete ROI                           |
| `aspect_ratio`           | float32 | (num_rois,) | Ellipse axis ratio indicating elongation               |
| `normalized_pixel_count` | float32 | (num_rois,) | Pixel count normalized by expected ROI size (soma)     |
| `skewness`               | float32 | (num_rois,) | Neuropil-corrected fluorescence skewness               |
| `plane_index`            | int32   | (num_rois,) | Always 0 for tracked ROIs (combined-view masks)        |
| `soma_mask`              | bool    | (n_pixels,) | Flattened soma masks (present when populated)          |
| `soma_mask_counts`       | uint32  | (num_rois,) | Per-ROI lengths indexing `soma_mask`                   |
| `overlap_mask`           | bool    | (n_pixels,) | Flattened overlap masks (present when populated)       |
| `overlap_mask_counts`    | uint32  | (num_rois,) | Per-ROI lengths indexing `overlap_mask`                |
| `neuropil_mask`          | int32   | (n_pixels,) | Flattened neuropil indices (present when populated)    |
| `neuropil_mask_counts`   | uint32  | (num_rois,) | Per-ROI lengths indexing `neuropil_mask`               |

The `soma_mask`, `overlap_mask`, and `neuropil_mask` data arrays (with their `_counts` companions) appear only when the
corresponding per-ROI data is populated. Otherwise the keys are absent.

Channel 2 uses identical keys in `roi_masks_channel_2.npz` and `roi_statistics_channel_2.npz`.

---

### Fluorescence traces

Saved at the dataset output root. All files are `.npy` format, float32 dtype.

`frames` is the frame count of the combined view over the recording's registered plane binaries, which is that of the
recording's shortest plane binary. Binarization gives every plane of the recording the same frame count, so this
matches that recording's single-recording combined `frame_count`.

**Channel 1 (always present):**

| File                          | Shape              | Description                                   |
|-------------------------------|--------------------|-----------------------------------------------|
| `cell_fluorescence.npy`       | (num_rois, frames) | Lambda-weighted somatic fluorescence traces   |
| `neuropil_fluorescence.npy`   | (num_rois, frames) | Neuropil fluorescence traces                  |
| `subtracted_fluorescence.npy` | (num_rois, frames) | Neuropil-and-baseline-subtracted fluorescence |
| `spikes.npy`                  | (num_rois, frames) | Deconvolved spike estimates                   |

**Channel 2 (both channels functional only, same shapes):**

| File                                    | Description                       |
|-----------------------------------------|-----------------------------------|
| `cell_fluorescence_channel_2.npy`       | Channel 2 somatic fluorescence    |
| `neuropil_fluorescence_channel_2.npy`   | Channel 2 neuropil fluorescence   |
| `subtracted_fluorescence_channel_2.npy` | Channel 2 subtracted fluorescence |
| `spikes_channel_2.npy`                  | Channel 2 deconvolved spikes      |

If `spike_deconvolution.extract_spikes` is False, `subtracted_fluorescence.npy` and `spikes.npy` are filled with zeroes.

**Optional colocalization file (both channels functional only):**

| File                      | Shape         | Description                                                                            |
|---------------------------|---------------|----------------------------------------------------------------------------------------|
| `cell_colocalization.npy` | (num_rois, 2) | Column 0: matched channel-2 ROI index (-1 if unmatched), column 1: pixel-overlap score |

Multi-recording dual-channel processing uses spatial colocalization (pixel overlap between channel-1 and channel-2
ROIs), so column 0 holds the matched channel-2 ROI index (-1 when unmatched), not a 1.0/0.0 label.

---

### Per-recording runtime metadata (multi_recording_runtime_data.yaml)

A YAML file containing scalar metadata from all processing stages. Array fields are set to None in the YAML and saved as
separate `.npy`/`.npz` files (documented above).

| Section        | Key fields                                                                        |
|----------------|-----------------------------------------------------------------------------------|
| `io`           | Recording identity, dataset membership, and selected ROI indices, itemized below. |
| `registration` | Deformation fields, transformed images, and deformed ROI masks, itemized below.   |
| `tracking`     | Consensus template masks and template diameters, itemized below.                  |
| `extraction`   | Backward-transformed ROI statistics and the extracted traces, itemized below.     |
| `timing`       | Stage durations, phase totals, and version stamps, itemized below.                |

The file's top level additionally carries `output_path`, the dataset output directory this metadata describes, and
`combined_data`, which is always null because the single-recording data it references is reloaded on demand.

- `io`: `recording_id`, `data_path`, `dataset_name`, `mroi_region_borders`, `dataset_output_paths`,
  `selected_roi_indices`, `selected_roi_indices_channel_2`.
- `registration`: `deform_field_y`, `deform_field_x`, `transformed_mean_image`, `transformed_enhanced_mean_image`,
  `transformed_maximum_projection` (and the `*_channel_2` variants), `deformed_roi_masks`,
  `deformed_roi_masks_channel_2`. All are array fields, set to None in the YAML because their data is saved separately
  as `.npy` files in `registration_arrays/` and `.npz` files at the dataset output root.
- `tracking`: `template_masks`, `template_masks_channel_2`, `template_diameter`, `template_diameter_channel_2`. The mask
  fields are saved as NPZ and set to None in the YAML.
- `extraction`: `roi_statistics`, `cell_fluorescence`, `neuropil_fluorescence`, `subtracted_fluorescence`, `spikes`,
  `cell_classification`, `cell_colocalization`, `corrected_structural_mean_image`, and the `*_channel_2` variants of
  the first six. Every field is an array or list field, set to None in the YAML because its data is saved separately as
  the `.npy` and `.npz` files at the dataset output root. `cell_classification` stays None for tracked ROIs, which are
  already known cells, and `corrected_structural_mean_image` stays None because multi-recording dual-channel
  processing uses spatial colocalization rather than the intensity-based path that computes it.
- `timing`: `registration_time`, `tracking_time`, `backward_transform_time`, `total_discovery_time`, `extraction_time`,
  `deconvolution_time`, `total_extraction_time`, `date_processed`, `python_version`, `cindra_version`.

---

### Data type conventions

| Category            | Dtype   | Examples                                 |
|---------------------|---------|------------------------------------------|
| Pixel coordinates   | int32   | y_pixels, x_pixels, centroids            |
| Images and traces   | float32 | transformed images, fluorescence, spikes |
| Counts / dimensions | uint32  | pixel_counts, cluster_id, frame_width    |
| Small counts        | uint16  | footprints, recording_count              |
| Plane indices       | int32   | plane_index                              |
| Deformation fields  | float32 | deform_field_y, deform_field_x           |

The fluorescence trace and colocalization `.npy` files are saved with `allow_pickle=False`. The
`registration_arrays/*.npy` files use NumPy save defaults but contain only numeric arrays that load safely with
`allow_pickle=False`. Arrays support memory-mapped loading via `np.load(path, mmap_mode='r')` for efficient access to
large datasets. NPZ archives do not support memory mapping and are always eagerly loaded.

---

## Related skills

| Skill                            | Relationship                                                               |
|----------------------------------|----------------------------------------------------------------------------|
| `/cindra-pipeline`               | Overview: end-to-end phases, handoffs, and the single-vs-multi entry point |
| `/cindra-mcp-environment-setup`  | Prerequisite: cindra MCP server for query and verification tools           |
| `/multi-recording-configuration` | Configuration parameter reference for the multi-recording pipeline         |
| `/multi-recording-processing`    | Processing workflow that produces this output                              |
| `/single-recording-results`      | Companion output data reference for the single-recording pipeline          |
| `/visualization`                 | Launch viewers and query tools to visualize and inspect this output data   |

---

## Verification checklist

Use `verify_multi_recording_output_tool` to automate the file-presence and NPZ-key parts of this verification. The tool
checks every channel 1 output file and the required NPZ keys across every recording in the dataset and returns a
completeness verdict with any missing items listed. It validates no array shape, and it checks no channel 2 file under
`registration_arrays/`, so run those two items by hand. Fall back to the whole manual checklist below only if the MCP
tool is unavailable.

Gate on `complete`, which is false whenever `missing` is non-empty. The `failed` count covers the required checks that
did not pass, so it always equals the length of `missing`. The tool also returns `optional_absent` when the dataset
holds none of an optional output, listing the channel 2 arrays and `cell_colocalization.npy` each recording lacks in
the same label form `missing` uses. That list is informational, so report it as expected for a single-channel dataset
and never treat it as an incomplete run.

```text
Multi-Recording Output Completeness:

Shared files (main recording only):
- [ ] `multi_recording_configuration.yaml` exists

Per-recording files (every recording):
- [ ] `multi_recording_runtime_data.yaml` exists with non-zero timing fields

Registration data (per recording):
- [ ] `registration_arrays/deform_field_y.npy` exists
- [ ] `registration_arrays/deform_field_x.npy` exists
- [ ] `registration_arrays/transformed_mean_image.npy` exists
- [ ] `registration_arrays/transformed_enhanced_mean_image.npy` exists
- [ ] `registration_arrays/transformed_maximum_projection.npy` exists
- [ ] `registration_deformed_masks.npz` exists and contains `pixel_counts`, `y_pixels`, `x_pixels` keys
- [ ] Channel 2 registration files exist if both channels are functional, except
      `registration_arrays/transformed_mean_image_channel_2.npy`, which exists for any dual-channel recording

Tracking data (per recording, identical across recordings):
- [ ] `tracking_template_masks.npz` exists and contains `pixel_counts`, `cluster_id`, `recording_count` keys
- [ ] Channel 2 tracking files exist if both channels are functional

Extraction data (per recording):
- [ ] `roi_masks.npz` exists with backward-transformed template masks
- [ ] `roi_statistics.npz` exists with shape statistics
- [ ] `cell_fluorescence.npy` exists with shape (num_rois, num_frames)
- [ ] `neuropil_fluorescence.npy` exists with shape matching cell_fluorescence
- [ ] `subtracted_fluorescence.npy` exists with shape matching cell_fluorescence
- [ ] `spikes.npy` exists with shape matching cell_fluorescence
- [ ] Channel 2 trace files exist if both channels are functional
- [ ] `cell_colocalization.npy` exists if both channels are functional, with shape (num_rois, 2)
- [ ] Fluorescence trace shapes are consistent across all per-recording files
```
