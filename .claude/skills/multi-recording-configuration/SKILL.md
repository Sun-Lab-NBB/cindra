---
name: multi-recording-configuration
description: >-
  Complete reference for multi-recording pipeline configuration parameters, prerequisites, and MCP configuration tools.
  Documents all 7 configuration sections, parameter meanings, default values, prerequisites from single-recording
  processing, and available MCP tools for generating configurations and discovering candidates. Use when configuring
  multi-recording processing or when the user asks about multi-recording configuration parameters.
user-invocable: true
---

# Multi-recording configuration reference

Complete parameter reference for the multi-recording (cross-recording) cindra ROI tracking pipeline.

---

## Scope

**Covers:**
- All 7 configuration sections and their parameters for the `MultiRecordingConfiguration` dataclass
- Default values, types, and descriptions for every parameter
- Prerequisites from single-recording processing
- Pipeline-set parameters
- MCP tools for configuration generation and recording discovery
- Registration and tracking tuning guidance
- Configuration compliance verification

**Does not cover:**
- Output data formats and file references (see `/multi-recording-results`)
- Processing workflow, batch operations, or status monitoring (see `/multi-recording-processing`)
- Single-recording configuration (see `/single-recording-configuration`)

---

## MCP configuration tools

These tools are registered on the `cindra-mcp` server. You MUST verify the MCP server is connected before
using these tools. If the tools are unavailable, invoke `/mcp-environment-setup` to diagnose and resolve
connectivity issues. Tool parameters and return values are self-documented via MCP introspection.

| Tool                                       | Purpose                                                                   |
|--------------------------------------------|---------------------------------------------------------------------------|
| `generate_config_file`                     | Generates a default configuration YAML for the specified pipeline type    |
| `discover_multi_recording_candidates_tool` | Finds recording directories with completed single-recording output        |
| `read_config_file`                         | Reads any YAML file as a raw dictionary (supports legacy and non-cindra)  |
| `validate_config_file`                     | Validates a cindra config against schema, reports errors and non-defaults |

---

## Configuration overview

The multi-recording pipeline uses `MultiRecordingConfiguration`, a dataclass with 7 nested sections. This
pipeline tracks ROIs across multiple recordings and extracts consistent fluorescence traces using consensus
template masks.

All parameters are specified in the `MultiRecordingConfiguration` YAML file. The pipeline loads the fully resolved
configuration directly from the file without any runtime overrides.

---

## Prerequisites from single-recording processing

Before multi-recording processing, all recordings must have completed single-recording processing through
all three phases (binarize, process, combine). The multi-recording pipeline locates single-recording output by
recursively searching each recording directory for a `combined_metadata.npz` file. The parent directory of this
file becomes the cindra root for that recording.

**Required single-recording outputs per recording:**

| File                      | Used for                                            |
|---------------------------|-----------------------------------------------------|
| `combined_metadata.npz`   | Recording discovery, frame dimensions, binary paths |
| `roi_masks.npz`           | Loading ROI spatial data for selected ROIs          |
| `roi_statistics.npz`      | ROI filtering by size and shape                     |
| `cell_classification.npy` | ROI filtering by classifier probability             |
| `detection_data/*.npy`    | Reference images for diffeomorphic registration     |
| `channel_1_data.bin`      | Fluorescence extraction from registered binary data |

No special single-recording configuration is required. The pipeline always generates combined output and preserves
registered binary files by default.

---

## Pipeline-set parameters

These parameters are set automatically by the pipeline and should not be manually configured:

| Parameter                            | Set by         | Value                                                     |
|--------------------------------------|----------------|-----------------------------------------------------------|
| `recording_io.recording_directories` | MCP batch tool | List of recording paths (from `recording_paths` argument) |
| `runtime.parallel_workers`           | CLI/MCP        | Number of workers (or auto-detected from CPU count)       |
| `runtime.display_progress_bars`      | CLI/MCP        | Whether to show progress bars                             |

---

## Section 1: runtime

Runtime behavior settings shared with the single-recording pipeline.

| Parameter               | Type | Default | Description                                                              |
|-------------------------|------|---------|--------------------------------------------------------------------------|
| `parallel_workers`      | int  | 20      | Maximum CPU worker count. 10-20 optimal per recording. -1/0 = all cores. |
| `display_progress_bars` | bool | False   | Show progress bars. Disable for parallel processing.                     |

---

## Section 2: recording_io

Controls input recording paths, output dataset naming, and ROI selection caching. The pipeline receives a list
of recording directories (populated by the MCP batch tool), natural-sorts them to determine the main recording,
and stores all multi-recording output under `multi_recording/{dataset_name}/` inside the main recording's cindra
output directory. ROI selection results are cached per dataset so that re-running the pipeline skips selection
unless `repeat_selection` is enabled.

| Parameter               | Type        | Default | Description                                                                                                  |
|-------------------------|-------------|---------|--------------------------------------------------------------------------------------------------------------|
| `recording_directories` | tuple[Path] | ()      | **Set by batch tool.** Absolute paths to recording roots. Natural-sorted; first = main recording.            |
| `dataset_name`          | str         | ""      | **REQUIRED.** Unique identifier for this dataset. Used for output folder: `multi_recording/{dataset_name}/`. |
| `repeat_selection`      | bool        | False   | Re-run ROI selection even if existing selections are found.                                                  |

### Important notes on `recording_io`

- `recording_directories` is populated by the MCP batch tool from the `recording_paths` argument.
- `dataset_name` **must be set by the user** — it identifies the output and must be unique per animal/experiment.
- The first recording (after natural sorting) becomes the "main recording" storing the shared configuration file.
- When `repeat_selection` is True, ROI selection is re-run using current criteria even if selections already exist.
  This allows updated single-recording results or modified selection criteria to be integrated.

---

## Section 3: roi_selection

Filters single-recording ROIs before cross-recording tracking. Each recording's detected ROIs are filtered by
classifier probability, spatial size, and distance from MROI region borders. Only ROIs passing all criteria
enter the tracking pipeline. This pre-filtering controls the quality/quantity tradeoff for the tracked population.

| Parameter                         | Type         | Default | Description                                                               |
|-----------------------------------|--------------|---------|---------------------------------------------------------------------------|
| `probability_threshold`           | float        | 0.85    | Min classifier probability from single-recording. Lower = keep more ROIs. |
| `maximum_size`                    | int          | 1000    | Max ROI size (pixels). ROIs larger are excluded.                          |
| `mroi_region_margin`              | int          | 30      | Min distance (pixels) from ROI centroid to MROI region border.            |
| `probability_threshold_channel_2` | float / None | None    | Channel 2 probability threshold. None = use channel 1 value.              |
| `maximum_size_channel_2`          | int / None   | None    | Channel 2 max size. None = use channel 1 value.                           |
| `mroi_region_margin_channel_2`    | int / None   | None    | Channel 2 MROI margin. None = use channel 1 value.                        |

Channel 2 parameters default to None, which causes the pipeline to fall back to the corresponding channel 1 value.
Set these independently when channel 2 ROIs have different classification or size characteristics.

### Tuning guidance

- **More tracked ROIs**: Lower `probability_threshold` (0.7–0.8) to include lower-confidence ROIs from
  single-recording detection. This feeds more candidates into tracking but may include noisier ROIs.
- **Fewer false positives**: Raise `probability_threshold` (0.9+) to only track high-confidence cells.
- **Large ROIs excluded**: Increase `maximum_size` (1500–2000) if legitimate cells exceed the default limit.
- **MROI edge artifacts**: Increase `mroi_region_margin` (40–60) to exclude more ROIs near region borders
  where registration distortion is highest. Set to 0 for non-MROI recordings.
- **After re-running single-recording processing**: Set `repeat_selection=True` to re-apply selection criteria
  against updated single-recording results, then set back to False.

---

## Section 4: diffeomorphic_registration

Aligns recordings to a common coordinate space using groupwise diffeomorphic Demons registration. At each scale
level of a coarse-to-fine pyramid, symmetric Demons forces are computed from every recording to all others and
averaged, producing a displacement field that is regularized via cubic B-spline fitting with injectivity constraints
to guarantee invertibility. Deformations are composed across scales to produce smooth, invertible forward and inverse
maps per recording. Forward maps warp ROI masks into the shared space for tracking; inverse maps transform tracked
template masks back to each recording's native space for signal extraction.

| Parameter              | Type  | Default         | Description                                                                   |
|------------------------|-------|-----------------|-------------------------------------------------------------------------------|
| `image_type`           | str   | "enhanced_mean" | Reference image type: "mean", "enhanced_mean", or "maximum_projection".       |
| `grid_sampling_factor` | float | 1.0             | B-spline grid refinement per scale. 0-1. Lower = finer grid at coarse scales. |
| `scale_sampling`       | int   | 30              | Iterations per scale level. 20-30 typical. Higher = better but slower.        |
| `speed_factor`         | float | 3.0             | Deformation strength. **Most important tuning parameter.** 1-5 typical.       |
| `repeat_registration`  | bool  | False           | Re-run registration even if existing data is found.                           |

### Tuning guidance

- **Minimal drift** (stable chronic windows): Lower `speed_factor` (1.5) and `scale_sampling` (20) for faster
  convergence when tissue displacement between recordings is small.
- **Moderate drift** (typical use case): Default `speed_factor` (3.0) and `scale_sampling` (30) work well for
  most longitudinal imaging preparations.
- **Significant drift** (challenging cases): Increase `speed_factor` (4.5) and `scale_sampling` (40) to allow
  larger deformations and more iterations per scale level.
- **Different reference image**: Change `image_type` to "mean" if enhanced mean images introduce artifacts, or
  "maximum_projection" for sparse labeling where bright pixels are more informative.
- **Finer spatial control**: Lower `grid_sampling_factor` (0.5–0.8) to use a denser B-spline grid at coarse
  scales, improving accuracy for spatially complex deformations at the cost of speed.

---

## Section 5: roi_tracking

Clusters ROI masks across registered recordings to identify cells tracked over time. After forward-deforming all
single-recording masks into the shared coordinate space, the field of view is partitioned into spatial bins with
overlap margins. Within each bin, ROI pairs from different recordings within `maximum_distance` are compared by
Jaccard distance and grouped via hierarchical clustering. For each cluster passing `mask_prevalence`, a consensus
template mask is built by retaining only pixels present in at least `pixel_prevalence` percent of member masks.
Templates with fewer than `minimum_size` non-overlapping pixels are discarded. Final templates are inverse-deformed
back to each recording's native coordinates for fluorescence extraction.

| Parameter          | Type            | Default    | Description                                                              |
|--------------------|-----------------|------------|--------------------------------------------------------------------------|
| `threshold`        | float           | 0.75       | Jaccard distance threshold for clustering. Lower = stricter matching.    |
| `mask_prevalence`  | int             | 50         | Min % of recordings that must contain the ROI. Higher = more reliable.   |
| `pixel_prevalence` | int             | 50         | Min % of recordings a pixel must appear in for template mask.            |
| `step_sizes`       | tuple[int, int] | (200, 200) | Spatial bin size [h, w] in pixels for partitioning the clustering space. |
| `bin_size`         | int             | 50         | Overlap margin (pixels) between adjacent bins for border ROI clustering. |
| `maximum_distance` | int             | 20         | Max centroid distance (pixels) between masks to consider same ROI.       |
| `minimum_size`     | int             | 25         | Min non-overlapping pixels for ROI-template assignment.                  |

### Tuning guidance

- **Strict tracking** (fewer, highly reliable ROIs): Lower `threshold` (0.65) for stricter Jaccard matching,
  raise `mask_prevalence` (70) and `pixel_prevalence` (60) to require more cross-recording consistency, and
  decrease `maximum_distance` (15) to tighten centroid proximity.
- **Lenient tracking** (more ROIs, some less reliable): Raise `threshold` (0.85) for more permissive matching,
  lower `mask_prevalence` (30) and `pixel_prevalence` (40) to accept ROIs present in fewer recordings, and
  increase `maximum_distance` (25) to tolerate larger centroid shifts.
- **Small ROIs lost**: Lower `minimum_size` (15–20) to retain templates with fewer non-overlapping pixels.
- **Dense labeling**: Decrease `step_sizes` (e.g., (150, 150)) and increase `bin_size` (60–80) to improve
  clustering accuracy in crowded fields by using smaller spatial bins with wider overlap margins.

---

## Section 6: signal_extraction

Fluorescence extraction from tracked ROIs. Shared with the single-recording pipeline.

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

The multi-recording pipeline always uses `allow_overlap=True` internally regardless of the configured value, since
multi-recording template masks are spatially distinct by construction. No reclassification is performed because tracked
ROIs are already known cells, so `classification_threshold` is not used during multi-recording extraction.

### Tuning guidance

See `/single-recording-configuration` Section 8 for full tuning guidance. The same recommendations apply here,
with two exceptions: `allow_overlap` is always True internally, and `classification_threshold` has no effect since
tracked ROIs skip reclassification.

---

## Section 7: spike_deconvolution

Spike inference from multi-recording fluorescence traces. Shared with the single-recording pipeline.

| Parameter              | Type  | Default   | Description                                                       |
|------------------------|-------|-----------|-------------------------------------------------------------------|
| `extract_spikes`       | bool  | True      | Deconvolve spikes from fluorescence.                              |
| `neuropil_coefficient` | float | 0.7       | Neuropil scaling before subtraction.                              |
| `baseline_method`      | str   | "maximin" | Baseline method: "maximin", "constant", or "constant_percentile". |
| `baseline_window`      | float | 60.0      | Sliding window (seconds) for maximin baseline.                    |
| `baseline_sigma`       | float | 10.0      | Gaussian sigma (frames) for baseline computation.                 |
| `baseline_percentile`  | float | 8.0       | Percentile for constant_percentile baseline.                      |

### Tuning guidance

See `/single-recording-configuration` Section 9 for full tuning guidance. The same recommendations apply here.

---

## User-configurable vs auto-set parameters

### Parameters users must configure

| Parameter                   | Why required                                         |
|-----------------------------|------------------------------------------------------|
| `recording_io.dataset_name` | Uniquely identifies output; cannot be auto-generated |

### Parameters users should consider

| Parameter                                 | When to change                                     |
|-------------------------------------------|----------------------------------------------------|
| `roi_selection.probability_threshold`     | Different quality/quantity tradeoff                |
| `roi_selection.mroi_region_margin`        | MROI (multi-region) imaging mode                   |
| `diffeomorphic_registration.speed_factor` | Different amounts of tissue drift                  |
| `roi_tracking.mask_prevalence`            | Different recording-to-recording consistency needs |

### Parameters typically left at default

- `diffeomorphic_registration.image_type`, `grid_sampling_factor`, `scale_sampling`
- `roi_tracking.step_sizes`, `bin_size`, `minimum_size`
- All signal_extraction parameters
- All spike_deconvolution parameters

---

## Configuration file format

### Minimal configuration (required fields only)

```yaml
recording_io:
  dataset_name: "animal1_learning_task"
```

### Typical configuration

```yaml
recording_io:
  dataset_name: "animal1_learning_task"

roi_selection:
  probability_threshold: 0.85
  maximum_size: 1000

diffeomorphic_registration:
  speed_factor: 3.0

roi_tracking:
  threshold: 0.75
  mask_prevalence: 50
```

### Full configuration with MROI region filtering

```yaml
runtime:
  parallel_workers: 20
  display_progress_bars: false

recording_io:
  dataset_name: "animal1_vr_navigation"

roi_selection:
  probability_threshold: 0.85
  maximum_size: 1000
  mroi_region_margin: 30

diffeomorphic_registration:
  image_type: "enhanced_mean"
  grid_sampling_factor: 1.0
  scale_sampling: 30
  speed_factor: 3.0

roi_tracking:
  threshold: 0.75
  mask_prevalence: 50
  pixel_prevalence: 50
  step_sizes: [200, 200]
  bin_size: 50
  maximum_distance: 20
  minimum_size: 25

signal_extraction:
  extract_neuropil: true
  allow_overlap: false
  minimum_neuropil_pixels: 350
  inner_neuropil_border_radius: 2
  cell_probability_percentile: 50

spike_deconvolution:
  extract_spikes: true
  neuropil_coefficient: 0.7
  baseline_method: "maximin"
  baseline_window: 60.0
  baseline_sigma: 10.0
  baseline_percentile: 8.0
```

---

## Configuration workflow

1. **Verify prerequisites** — all recordings must have completed single-recording processing (all 3 phases).
2. **Discover candidates** using `discover_multi_recording_candidates_tool` to find recordings with completed output.
3. **Generate a default configuration** using `generate_config_file` with `pipeline_type="multi-recording"`.
   Alternatively, use `read_config_file` to inspect an existing or legacy configuration for conversion.
4. **Set `dataset_name`** — this is the only required user parameter.
5. **Review and tune** registration and tracking parameters based on expected tissue drift.
6. **Validate** the configuration using `validate_config_file` to check for errors, warnings, and non-default
   parameters.
7. **Hand off** to the processing workflow (see `/multi-recording-processing`).

---

## Related skills

| Skill                             | Relationship                                                                      |
|-----------------------------------|-----------------------------------------------------------------------------------|
| `/mcp-environment-setup`          | Prerequisite: MCP server must be connected for configuration tools                |
| `/single-recording-processing`    | Prerequisite: single-recording processing must complete before multi-recording    |
| `/single-recording-results`       | Prerequisite: single-recording output files required as input for multi-recording |
| `/single-recording-configuration` | Companion configuration reference for the single-recording pipeline               |
| `/multi-recording-processing`     | Next step: processing workflow that consumes this configuration                   |
| `/multi-recording-results`        | Output data format reference for evaluating processing results                    |
| `/visualization`                  | Downstream: launch viewers to inspect multi-recording results after processing    |

---

## Verification checklist

You MUST verify configuration files against this checklist before starting multi-recording processing.
Use `validate_config_file` for automated validation of YAML structure, parameter constraints, and pipeline-set
parameter detection.

```text
Multi-Recording Configuration Compliance:
- [ ] cindra MCP server is connected (if not, invoke `/mcp-environment-setup`)
- [ ] `validate_config_file` reports no errors (run this first)
- [ ] `recording_io.dataset_name` is set to a unique, non-empty string
- [ ] `roi_selection.probability_threshold` is appropriate for the dataset (0.85 default)
- [ ] `diffeomorphic_registration.speed_factor` matches expected tissue drift (1-5 range)
- [ ] `roi_tracking.mask_prevalence` is set appropriately for the number of recordings
- [ ] Channel 2 roi_selection parameters are set if channel 2 ROIs have different characteristics
- [ ] Review any warnings from `validate_config_file` (pipeline-set parameters, unusual ranges)
```
