---
name: multi-day-config
description: >-
  Complete reference for multi-day pipeline configuration parameters. Documents all 7 configuration sections,
  parameter meanings, default values, and which parameters are set by the pipeline vs user-configurable.
---

# Multi-Day Configuration Reference

Complete parameter reference for the multi-day (cross-session) cindra cell tracking pipeline.

---

## Configuration Overview

The multi-day pipeline uses `MultiDayConfiguration`, a dataclass with 7 nested sections. This pipeline tracks cells
across multiple recording sessions and extracts consistent fluorescence traces.

### Prerequisites

Before multi-day processing:
- All sessions must have completed single-day processing
- Single-day processing must have used `output.combined: true`
- Single-day processing must have used `file_io.delete_bin: false`

### Parameter Resolution

All parameters are specified in the `MultiDayConfiguration` YAML file. The pipeline loads the fully resolved
configuration directly from the file without any runtime overrides.

---

## Pipeline-Set Parameters

These parameters are set automatically by the pipeline and should not be manually configured:

| Parameter | Set By | Value |
|-----------|--------|-------|
| `io.session_directories` | MCP batch tool | List of session paths (from `session_paths` argument) |
| `main.parallel_workers` | CLI/MCP | Number of workers (or auto-detected from CPU count) |
| `main.progress_bars` | CLI/MCP | Whether to show progress bars |
| `main.python_version` | `multi_day.py` | Current Python version |
| `main.cindra_version` | `multi_day.py` | Current cindra version |

---

## Section 1: main

Global parameters for multi-day processing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel_workers` | int | 20 | Numba parallel workers. 10-20 optimal per session. -1/0 = all cores. |
| `progress_bars` | bool | False | Show progress bars. Disable for parallel processing. |
| `python_version` | str | (auto) | (Internal) Python version used. |
| `cindra_version` | str | (auto) | (Internal) Library version used. |

---

## Section 2: io

Input/output parameters for session data and results.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_directories` | list[str] | [] | **Set by batch tool.** Absolute paths to session roots. Natural-sorted; first = main session. |
| `dataset_name` | str | "" | **REQUIRED.** Unique identifier for this dataset. Used for output folder: `multiday/{dataset_name}/`. |

### Important Notes on `io`

- `session_directories` is populated by the MCP batch tool from the `session_paths` argument
- `dataset_name` **must be set by the user** - it identifies the output and must be unique per animal/experiment
- The first session (after natural sorting) becomes the "main session" storing the tracker file

---

## Section 3: cell_selection

Parameters for filtering which single-day cells are tracked across days.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `probability_threshold` | float | 0.85 | Min classifier probability from single-day. Lower = keep more cells. |
| `maximum_size` | int | 1000 | Max cell size (pixels). Cells larger are excluded. |
| `mroi_region_margin` | int | 30 | Min distance (pixels) from cell center to MROI region border. Cells too close excluded. |

### Cell Selection Guidance

**Strict selection** (fewer, more reliable cells):
```yaml
cell_selection:
  probability_threshold: 0.9
  maximum_size: 800
```

**Lenient selection** (more cells, potentially noisier):
```yaml
cell_selection:
  probability_threshold: 0.7
  maximum_size: 1200
```

**Mesoscope multi-ROI mode** (required if using multiple imaging ROIs):
```yaml
cell_selection:
  mroi_region_margin: 30  # Exclude cells within 30px of region boundaries
```

---

## Section 4: registration

Cross-session alignment parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_type` | str | "enhanced" | Reference image type: "enhanced", "mean", or "max". |
| `grid_sampling_factor` | float | 1.0 | Grid scaling with deformation scale. 0-1. Lower = finer grid at high scales. |
| `scale_sampling` | int | 30 | Iterations per scale level. 20-30 typical. Higher = better but slower. |
| `speed_factor` | float | 3.0 | Deformation strength. **Most important tuning parameter.** 1-5 typical. |

### Registration Guidance

**Minimal drift** (stable chronic windows):
```yaml
registration:
  speed_factor: 1.5
  scale_sampling: 20
```

**Moderate drift** (typical use case):
```yaml
registration:
  speed_factor: 3.0    # Default
  scale_sampling: 30   # Default
```

**Significant drift** (challenging cases):
```yaml
registration:
  speed_factor: 4.5
  scale_sampling: 40
```

---

## Section 5: clustering

Cell tracking (clustering) across registered sessions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `criterion` | str | "distance" | Clustering criterion. Only "distance" currently supported. |
| `threshold` | float | 0.75 | Clustering threshold. Lower = stricter matching. |
| `mask_prevalence` | int | 50 | Min % of sessions that must contain the cell. Higher = more reliable. |
| `pixel_prevalence` | int | 50 | Min % of sessions a pixel must appear in for template mask. Higher = more stable. |
| `step_sizes` | list[int] | [200, 200] | Block size [height, width] for clustering. Affects memory usage. |
| `bin_size` | int | 50 | Extension (pixels) into neighboring blocks for border cells. |
| `maximum_distance` | int | 20 | Max pixel distance between masks to consider same cell. |
| `minimum_size` | int | 25 | Min non-overlapping pixels for cell-template assignment. |

### Clustering Guidance

**Strict tracking** (fewer, highly reliable cells):
```yaml
clustering:
  threshold: 0.65
  mask_prevalence: 70
  pixel_prevalence: 60
  maximum_distance: 15
```

**Lenient tracking** (more cells, some may be less reliable):
```yaml
clustering:
  threshold: 0.85
  mask_prevalence: 30
  pixel_prevalence: 40
  maximum_distance: 25
```

---

## Section 6: signal_extraction

Fluorescence extraction from tracked cells.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extract_neuropil` | bool | True | Extract neuropil activity. False = assume zero. |
| `allow_overlap` | bool | False | Use overlapping pixels for extraction. |
| `minimum_neuropil_pixels` | int | 350 | Min neuropil region size (pixels). |
| `inner_neuropil_border_radius` | int | 2 | Pixels between cell and neuropil region. |
| `lambda_percentile` | int | 50 | Lambda percentile threshold for neuropil mask exclusion. |

These parameters mirror single-day settings. Typically left at defaults for consistency.

---

## Section 7: spike_deconvolution

Spike inference from multi-day fluorescence.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extract_spikes` | bool | True | Deconvolve spikes from fluorescence. |
| `neuropil_coefficient` | float | 0.7 | Neuropil scaling before subtraction. |
| `baseline` | str | "maximin" | Baseline method: "maximin", "constant", "constant_percentile". |
| `baseline_window` | float | 60.0 | Sliding window (seconds) for maximin baseline. |
| `baseline_sigma` | float | 10.0 | Gaussian sigma (seconds) for baseline computation. |
| `baseline_percentile` | float | 8.0 | Percentile for constant_percentile baseline. |

These parameters mirror single-day settings. Typically left at defaults for consistency.

---

## User-Configurable vs Auto-Set Parameters

### Parameters Users MUST Configure

| Parameter | Why Required |
|-----------|--------------|
| `io.dataset_name` | Uniquely identifies output; cannot be auto-generated |

### Parameters Users SHOULD Consider

| Parameter | When to Change |
|-----------|----------------|
| `cell_selection.probability_threshold` | Different quality/quantity tradeoff |
| `cell_selection.mroi_region_margin` | Multi-ROI mesoscope mode |
| `registration.speed_factor` | Different amounts of tissue drift |
| `clustering.mask_prevalence` | Different session-to-session consistency needs |

### Parameters Typically Left at Default

- `registration.image_type`, `grid_sampling_factor`, `scale_sampling`
- `clustering.step_sizes`, `bin_size`, `minimum_size`
- All signal_extraction parameters
- All spike_deconvolution parameters

---

## Configuration File Format

### Minimal Configuration (Required Fields Only)

```yaml
io:
  dataset_name: "mouse1_learning_task"  # REQUIRED
```

### Typical Configuration

```yaml
io:
  dataset_name: "mouse1_learning_task"

cell_selection:
  probability_threshold: 0.85
  maximum_size: 1000

registration:
  speed_factor: 3.0

clustering:
  threshold: 0.75
  mask_prevalence: 50
```

### Full Configuration with MROI Region Filtering

```yaml
main:
  parallel_workers: 20
  progress_bars: false

io:
  dataset_name: "mouse1_vr_navigation"

cell_selection:
  probability_threshold: 0.85
  maximum_size: 1000
  mroi_region_margin: 30

registration:
  image_type: "enhanced"
  grid_sampling_factor: 1.0
  scale_sampling: 30
  speed_factor: 3.0

clustering:
  criterion: "distance"
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
  lambda_percentile: 50

spike_deconvolution:
  extract_spikes: true
  neuropil_coefficient: 0.7
  baseline: "maximin"
  baseline_window: 60.0
  baseline_sigma: 10.0
  baseline_percentile: 8.0
```

---

## Output Structure

Multi-day processing creates outputs in each session's directory:

```
{session_path}/multiday/{dataset_name}/
├── ops.npy                              # Processing parameters
├── multi_day_cindra_configuration.yaml    # Configuration snapshot
├── multiday_tracker.json                # Processing tracker (main session only)
├── template_cell_masks.npy              # Tracked cell mask templates
├── F.npy                                # Fluorescence traces (n_cells × n_frames)
├── Fneu.npy                             # Neuropil traces
└── spks.npy                             # Deconvolved spikes
```

The main session (first after natural sorting) additionally contains:
- `multiday_tracker.json` - Job tracking for the entire dataset
- `template_cell_masks.npy` - Master cell templates used across all sessions

---

## Relationship to Single-Day Configuration

Multi-day processing **requires** single-day outputs. Ensure single-day was configured with:

```yaml
# In single-day config:
file_io:
  delete_bin: false   # Keep registered binaries for multi-day extraction
output:
  combined: true      # Create combined folder used by multi-day
```

The multi-day pipeline reads from `{session}/cindra/combined/` for each session.
