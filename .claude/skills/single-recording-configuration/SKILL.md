---
name: single-recording-configuration
description: >-
  Complete reference for single-recording pipeline configuration parameters and MCP configuration tools. Documents all
  9 configuration sections, parameter meanings, default values, and available MCP tools for generating configurations
  and discovering recordings. Use when configuring single-recording processing or when the user asks about
  single-recording configuration parameters.
user-invocable: true
---

# Single-recording configuration reference

Complete parameter reference for the single-recording (within-recording) cindra processing pipeline.

---

## Scope

**Covers:**
- All 9 configuration sections and their parameters for the `SingleRecordingConfiguration` dataclass
- Default values, types, and descriptions for every parameter
- Per-section tuning guidance for common scenarios (more cells, noisy data, new sensors)
- Pipeline-set parameters
- MCP tools for configuration generation and recording discovery
- Configuration compliance verification

**Does not cover:**
- Input data format, TIFF requirements, and acquisition parameters (see `/acquisition-data-preparation`)
- Output data formats and file references (see `/single-recording-results`)
- Processing workflow, batch operations, or status monitoring (see `/single-recording-processing`)
- Multi-recording configuration (see `/multi-recording-configuration`)

---

## MCP configuration tools

These tools are registered on the `cindra-mcp` server. You MUST verify the MCP server is connected before
using these tools. If the tools are unavailable, invoke `/mcp-environment-setup` to diagnose and resolve
connectivity issues. Tool parameters and return values are self-documented via MCP introspection.

| Tool                                        | Purpose                                                                   |
|---------------------------------------------|---------------------------------------------------------------------------|
| `generate_config_file`                      | Generates a default configuration YAML for the specified pipeline type    |
| `discover_single_recording_candidates_tool` | Finds recording directories containing `cindra_parameters.json`           |
| `read_config_file`                          | Reads any YAML file as a raw dictionary (supports legacy and non-cindra)  |
| `validate_config_file`                      | Validates a cindra config against schema, reports errors and non-defaults |

---

## Configuration overview

The single-recording pipeline uses `SingleRecordingConfiguration`, a dataclass with 9 nested sections.
Default values are optimized for GCaMP6f data from 2-Photon Random Access Mesoscope (2P-RAM).

All parameters are specified in the `SingleRecordingConfiguration` YAML file. The pipeline loads the fully resolved
configuration directly from the file without any runtime overrides.

---

## Pipeline-set parameters

These parameters are set automatically by the pipeline and should not be manually configured:

| Parameter                       | Set by        | Value                                               |
|---------------------------------|---------------|-----------------------------------------------------|
| `file_io.data_path`             | batch tool    | Recording's session root path (not raw data subdir) |
| `file_io.output_path`           | batch tool    | Recording's processed output path                   |
| `runtime.parallel_workers`      | CLI/MCP       | Number of workers (or auto-detected from CPU count) |
| `runtime.display_progress_bars` | CLI/MCP       | Whether to show progress bars                       |

---

## Section 1: runtime

Runtime behavior settings shared between single-recording and multi-recording pipelines.

| Parameter               | Type | Default | Description                                                          |
|-------------------------|------|---------|----------------------------------------------------------------------|
| `parallel_workers`      | int  | 20      | Maximum CPU worker count. 10-20 optimal per plane. -1/0 = all cores. |
| `display_progress_bars` | bool | False   | Show progress bars. Disable for parallel processing.                 |

---

## Section 2: main

Global parameters that broadly define the processing configuration.

| Parameter                   | Type         | Default | Description                                                          |
|-----------------------------|--------------|---------|----------------------------------------------------------------------|
| `two_channels`              | bool         | False   | Whether imaging data contains two channels per plane.                |
| `first_channel_functional`  | bool         | True    | Use first channel for ROI detection and signal extraction.           |
| `second_channel_functional` | bool         | False   | Use second channel for ROI detection. Ignored if two_channels=False. |
| `tau`                       | float        | 0.4     | Sensor decay timescale in seconds. GCaMP6f: 0.4, GCaMP6s: ~1.5.      |
| `ignored_flyback_planes`    | tuple[int]   | ()      | Flyback plane indices (0-based). Binarized but never processed.      |
| `custom_classifier_path`    | Path or None | None    | Path to custom .npz classifier. None = use built-in classifier.      |

Channel functional flags require `two_channels=True` to take effect. When both are True, the pipeline performs
independent ROI detection on both channels.

### Tuning guidance

- **Different calcium indicator**: Set `tau` to the sensor's decay time constant. GCaMP6f ≈ 0.4, GCaMP6s ≈ 1.25,
  GCaMP7f ≈ 0.4, GCaMP8f ≈ 0.2. Incorrect `tau` degrades spike deconvolution and ROI detection.
- **Structural channel**: Set `two_channels=True`, keep only `first_channel_functional=True`. The second channel
  is stored for colocalization analysis but not used for ROI detection.
- **Custom classifier**: Provide `custom_classifier_path` when imaging non-standard cell types or preparations
  where the built-in classifier performs poorly. The `.npz` file must contain `training_labels`.

---

## Section 3: file_io

Controls input data ingestion and output directory paths. During binarization (the first processing step), the
pipeline reads raw multipage TIFF files from the data directory, splits them by imaging plane, and writes each
plane's frames into a contiguous binary file optimized for fast random access during processing.
This TIFF-to-binary conversion only runs once unless `repeat_binarization` is enabled.

| Parameter             | Type          | Default | Description                                                 |
|-----------------------|---------------|---------|-------------------------------------------------------------|
| `data_path`           | Path or None  | None    | Root directory containing input TIFFs. **Set by pipeline.** |
| `output_path`         | Path or None  | None    | Output directory root. **Set by pipeline.**                 |
| `ignored_file_names`  | tuple[str]    | ()      | File stems (without extension) to skip when loading TIFFs.  |
| `repeat_binarization` | bool          | False   | Re-run TIFF to binary conversion even if binaries exist.    |

---

## Section 4: registration

Corrects whole-frame translational motion by computing per-frame X/Y offsets via phase correlation against a
reference image built from frames sampled across the recording. Each frame is shifted to align with the reference.
Offsets are computed in batches and optionally smoothed temporally. Frames with outlier offsets are flagged as bad
and excluded from downstream ROI detection.

This section controls the rigid (global) component of motion correction. The recommended approach is to always
use both rigid registration (this section) and nonrigid registration (Section 6) together. Rigid registration
removes bulk translational motion, while nonrigid registration corrects the spatially non-uniform residual
deformations that a single global offset cannot capture. Disabling nonrigid registration is only appropriate when
processing speed is critical and the preparation is exceptionally stable.

| Parameter                                  | Type  | Default | Description                                                         |
|--------------------------------------------|-------|---------|---------------------------------------------------------------------|
| `repeat_registration`                      | bool  | False   | Re-register data even if already registered.                        |
| `align_by_first_channel`                   | bool  | True    | Use first channel for alignment. False = use second channel.        |
| `reference_frame_count`                    | int   | 500     | Frames sampled evenly across recording to compute reference image.  |
| `batch_size`                               | int   | 100     | Frames per registration batch.                                      |
| `maximum_offset_fraction`                  | float | 0.1     | Max offset as fraction of frame size (0.1 = 10%).                   |
| `spatial_smoothing_sigma`                  | float | 1.15    | Gaussian sigma (pixels) for phase correlation smoothing.            |
| `temporal_smoothing_sigma`                 | float | 0.0     | Gaussian sigma (frames) for temporal smoothing. 0 = disabled.       |
| `two_step_registration`                    | bool  | False   | Enable refinement registration (two-step).                          |
| `bad_frame_threshold`                      | float | 1.0     | Offset outlier threshold. Excluded frames are skipped, not removed. |
| `normalize_frames`                         | bool  | True    | Clip pixel intensities to 1st-99th percentile during registration.  |
| `registration_metric_principal_components` | int   | 5       | PCs for registration quality metrics. 0 = disable metrics.          |
| `compute_bidirectional_phase_offset`       | bool  | False   | Compute bidirectional phase offset for 2P line scanning.            |
| `bidirectional_phase_offset_override`      | int   | 0       | Manual bidiphase offset override. 0 = auto-detect.                  |

### Tuning guidance

- **High motion artifacts**: Increase `maximum_offset_fraction` (0.15–0.2) and enable `two_step_registration`
  for recordings with large, rapid animal movement. Always keep nonrigid registration enabled (Section 6) as
  large movements typically produce non-uniform deformations.
- **Noisy or low-SNR data**: Increase `spatial_smoothing_sigma` (1.5–2.0) to stabilize phase correlation.
- **Residual jitter after registration**: Enable `temporal_smoothing_sigma` (1.0–2.0 frames) for sub-pixel
  smoothing of offset traces.
- **Bidirectional scanning artifacts**: Enable `compute_bidirectional_phase_offset` for resonant scanners. Use
  `bidirectional_phase_offset_override` if auto-detection fails.
- **Too many frames excluded**: Increase `bad_frame_threshold` (1.5–2.0) to retain more frames.

---

## Section 5: one_photon_registration

Preprocessing applied before registration for 1-photon (widefield/miniscope) data. Applies spatial high-pass
filtering to remove diffuse background fluorescence and edge tapering to suppress border artifacts, both of which
degrade phase-correlation accuracy in 1P recordings.

| Parameter                 | Type  | Default | Description                                                            |
|---------------------------|-------|---------|------------------------------------------------------------------------|
| `enabled`                 | bool  | False   | Enable 1P preprocessing (high-pass filtering, tapering). False for 2P. |
| `spatial_highpass_window` | int   | 42      | Spatial high-pass filter window (pixels).                              |
| `pre_smoothing_sigma`     | float | 0.0     | Gaussian smoothing sigma before high-pass. 0 = disabled.               |
| `edge_taper_pixels`       | float | 40.0    | Edge pixels to taper during registration.                              |

Enable this section only for widefield or miniscope (1-photon) data. The preprocessing removes
background fluorescence that interferes with phase-correlation registration. Never enable for 2P data.

---

## Section 6: nonrigid_registration

Corrects spatially non-uniform motion by subdividing each frame into overlapping blocks, computing independent
X/Y offsets per block via phase correlation, and applying smooth local warping. Runs after rigid registration to
correct residual local deformations that a single global offset cannot capture.

| Parameter                   | Type            | Default    | Description                                          |
|-----------------------------|-----------------|------------|------------------------------------------------------|
| `enabled`                   | bool            | True       | Enable nonrigid registration for non-uniform motion. |
| `block_size`                | tuple[int, int] | (128, 128) | Block dimensions (pixels). Power of 2/3 recommended. |
| `signal_to_noise_threshold` | float           | 1.2        | SNR threshold for accepting block offsets.           |
| `maximum_block_offset`      | float           | 5.0        | Max block offset (pixels) relative to rigid offset.  |

The recommended approach is to always keep nonrigid registration enabled alongside rigid registration (Section 4).
Rigid registration removes bulk translational motion, while nonrigid registration corrects spatially varying
residual deformations. In practice, nearly all in vivo recordings benefit from both.

### Tuning guidance

- **Default (recommended)**: Keep `enabled=True`. The combination of rigid + nonrigid registration produces the
  best motion correction for virtually all in vivo neural imaging data.
- **Localized motion** (e.g., brain pulsation): Decrease `block_size` to (64, 64) for finer correction. Uses
  more memory.
- **Severe local motion**: Increase `maximum_block_offset` (8–10) to allow larger block displacements.
- **Speed-critical batch processing**: Disable nonrigid (`enabled=False`) only when processing speed is critical
  and the preparation is exceptionally stable with minimal tissue deformation.

---

## Section 7: roi_detection

Detects individual neurons (ROIs) from the registered movie. The movie is temporally binned, high-pass filtered
to remove neuropil background, then decomposed via iterative source extraction to identify spatial components
(cell masks) and their temporal activity. Each candidate ROI is classified as cell or artifact using a
pre-trained classifier based on morphological features (compactness, skewness, pixel count).

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

### Tuning guidance

This is the most impactful section for controlling ROI yield.

- **Need more cells detected**: Lower `threshold_scaling` (1.0–1.5) to accept weaker ROI signals. Lower
  `preclassification_threshold` (0.0–0.3) to retain more candidates for final classification. Increase
  `maximum_iterations` (80–100) for more extraction passes. Increase `maximum_binned_frames` (8000–10000)
  to capture more temporal structure, at the cost of speed.
- **Too many false positives**: Raise `threshold_scaling` (2.5–3.0) to require more distinct signals. Raise
  `preclassification_threshold` (0.6–0.8) to filter weak candidates early.
- **Noisy or low-SNR data**: Enable `denoise=True` for PCA denoising before detection. Consider lowering
  `threshold_scaling` (1.5) to compensate for reduced signal clarity.
- **Densely labeled tissue**: Raise `maximum_overlap` (0.85) to tolerate more spatial overlap between ROIs.
- **Dendrite contamination in classification**: Keep `crop_to_soma=True` (default). Set to False only if
  imaging dendrites as the target structure.

---

## Section 8: signal_extraction

Extracts fluorescence time series from detected ROIs. For each ROI, the raw signal is computed by averaging
pixel intensities within the cell mask across all frames. A neuropil signal is estimated from a surrounding
annular region (excluding other cells), and ROIs are given a final cell/non-cell classification using the full
feature set (compactness, skewness, pixel count).

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

### Tuning guidance

- **Neuropil contamination**: Increase `inner_neuropil_border_radius` (3–5) to widen the gap between cell and
  neuropil regions. Useful in densely labeled tissue.
- **Sparse labeling**: Lower `minimum_neuropil_pixels` (200) if few ROIs leave insufficient surround pixels.
- **Overlapping ROIs**: Set `allow_overlap=True` for densely packed cells where shared pixels are acceptable.
- **Final cell/non-cell split**: Adjust `classification_threshold` (0.3–0.7) to shift the cell/non-cell
  boundary. Lower values label more ROIs as cells.

---

## Section 9: spike_deconvolution

Infers neural spiking activity from fluorescence traces. Subtracts a scaled neuropil signal from the raw
fluorescence, estimates a slowly varying baseline using a sliding window or percentile method, computes ΔF/F,
and deconvolves the result to produce an estimated spike rate trace per ROI.

| Parameter              | Type  | Default   | Description                                                       |
|------------------------|-------|-----------|-------------------------------------------------------------------|
| `extract_spikes`       | bool  | True      | Deconvolve spikes from fluorescence.                              |
| `neuropil_coefficient` | float | 0.7       | Neuropil scaling before subtraction.                              |
| `baseline_method`      | str   | "maximin" | Baseline method: "maximin", "constant", or "constant_percentile". |
| `baseline_window`      | float | 60.0      | Sliding window (seconds) for maximin baseline.                    |
| `baseline_sigma`       | float | 10.0      | Gaussian sigma (frames) for baseline computation.                 |
| `baseline_percentile`  | float | 8.0       | Percentile for constant_percentile baseline.                      |

### Tuning guidance

- **Slow baseline drift** (long recordings): Use `baseline_method="constant_percentile"` with default
  `baseline_percentile=8.0` for stable baseline estimation across hours-long recordings.
- **Fast transients dominate**: Lower `baseline_window` (30–45 seconds) to track baseline more closely.
- **High neuropil contamination**: Increase `neuropil_coefficient` (0.8–0.9) for more aggressive subtraction.
  Decrease (0.5–0.6) if traces appear over-corrected (negative dips after transients).

---

## User-configurable vs auto-set parameters

### Parameters users should configure

| Parameter                     | When to change                                       |
|-------------------------------|------------------------------------------------------|
| `main.tau`                    | Different calcium indicator (GCaMP6s, GCaMP7f, etc.) |
| `main.two_channels`           | Recording has two channels                           |
| `main.ignored_flyback_planes` | Flyback planes present in the recording              |
| `file_io.ignored_file_names`  | Specific TIFFs to exclude (file stems, no extension) |

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

## Configuration lifecycle

Configuration files follow a two-tier lifecycle:

1. **Template configs** — De-novo configurations generated via `generate_config_file` or manually created.
   Templates can live anywhere (e.g., `/Data/CA1_GCaMP6f_SD.yaml`) and are reusable across recordings.
   Templates are never modified by the pipeline.

2. **Fine-tuned copies** — When `start_batch_processing_tool` runs, it loads the template, applies
   recording-specific overrides (`file_io.data_path`, `file_io.output_path`, `runtime.parallel_workers`),
   and saves the resolved copy as `_batch_config.yaml` inside each recording's output directory. These
   fine-tuned copies are what the pipeline actually executes against.

**Do NOT** create per-recording configuration files manually. Pass a single template path to the batch tool
and let it handle per-recording fine-tuning automatically.

---

## Configuration workflow

1. **Discover recordings** using `discover_single_recording_candidates_tool` to find directories with raw data.
2. **Verify data readiness** — use `validate_recording_readiness` on each discovered recording to confirm that
   raw data and acquisition parameters are ready. If any recording fails validation, invoke
   `/acquisition-data-preparation` to resolve before continuing.
3. **Generate a template configuration** using `generate_config_file` with `pipeline_type="single-recording"`.
   Save it at a user-chosen location (e.g., `/Data/CA1_GCaMP6f_SD.yaml`). Alternatively, use `read_config_file`
   to inspect an existing or legacy configuration for conversion.
4. **Review and modify** the template YAML file, setting at minimum `main.tau` and `main.two_channels`.
5. **Validate** the configuration using `validate_config_file` to check for errors, warnings, and non-default
   parameters.
6. **Configuration complete** — the validated template file is ready for use. This skill does not start
   processing. If invoked standalone, inform the user that the configuration is ready and they can proceed
   when ready. If invoked from another skill, return control to the caller.

---

## Related skills

| Skill                            | Relationship                                                                    |
|----------------------------------|---------------------------------------------------------------------------------|
| `/mcp-environment-setup`         | Prerequisite: MCP server must be connected for configuration tools              |
| `/acquisition-data-preparation`  | Prerequisite: raw data must be prepared before configuring the pipeline         |
| `/single-recording-processing`   | Next step: processing workflow that consumes this configuration                 |
| `/single-recording-results`      | Output data format reference for evaluating processing results                  |
| `/multi-recording-configuration` | Companion configuration reference for the multi-recording pipeline              |
| `/multi-recording-processing`    | Downstream: multi-recording workflow requires single-recording processing first |
| `/visualization`                 | Downstream: launch viewers to inspect results after processing                  |

---

## Verification checklist

You MUST verify configuration files against this checklist before starting single-recording processing.
Use `validate_config_file` for automated validation of YAML structure, parameter constraints, and pipeline-set
parameter detection.

```text
Single-Recording Configuration Compliance:
- [ ] cindra MCP server is connected (if not, invoke `/mcp-environment-setup`)
- [ ] `validate_config_file` reports no errors (run this first)
- [ ] `main.tau` matches the calcium indicator used (0.4 for GCaMP6f, ~1.5 for GCaMP6s)
- [ ] `main.two_channels` set correctly for the recording type
- [ ] `main.ignored_flyback_planes` lists correct flyback plane indices if applicable
- [ ] `file_io.ignored_file_names` lists any TIFFs to exclude
- [ ] Review any warnings from `validate_config_file` (pipeline-set parameters, channel consistency)
- [ ] Acquisition data prepared, `validate_recording_readiness` passed (if not, invoke `/acquisition-data-preparation`)
```
