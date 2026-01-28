---
name: single-day-config
description: >-
  Complete reference for single-day pipeline configuration parameters. Documents all 11 configuration sections,
  parameter meanings, default values, and which parameters are automatically overwritten during mesoscope processing.
---

# Single-Day Configuration Reference

Complete parameter reference for the single-day (within-session) sl-suite2p processing pipeline.

---

## Configuration Overview

The single-day pipeline uses `SingleDayS2PConfiguration`, a dataclass with 11 nested sections. Default values are
optimized for GCaMP6f data from 2-Photon Random Access Mesoscope (2P-RAM).

### Parameter Resolution Order

Parameters are resolved with the following precedence (highest to lowest):

1. **Runtime overrides** (`db` dictionary) - Passed programmatically
2. **User configuration** (`ops` from YAML file) - Loaded from config file
3. **Defaults** (`generate_default_ops()`) - Built-in defaults

Additionally, **mesoscope processing automatically overwrites** certain parameters from `suite2p_parameters.json`.

---

## Mesoscope Auto-Overwrite Parameters

**IMPORTANT**: When `file_io.mesoscan = True`, these parameters are **automatically overwritten** during the binarize
phase by reading `suite2p_parameters.json` from the data directory:

| Parameter          | Source Field in JSON | Description                                                                 |
|--------------------|----------------------|-----------------------------------------------------------------------------|
| `main.nplanes`     | `plane_number`       | Restructured to `nplanes × nrois` (each ROI×plane becomes a separate plane) |
| `main.fs`          | `frame_rate`         | Sampling rate per plane in Hz                                               |
| Internal: `lines`  | `roi_lines`          | Line indices for each ROI                                                   |
| Internal: `nrois`  | `roi_number`         | Number of imaging ROIs                                                      |
| Internal: `dx`     | `roi_x_coordinates`  | X-coordinates for each ROI                                                  |
| Internal: `dy`     | `roi_y_coordinates`  | Y-coordinates for each ROI                                                  |
| Internal: `iplane` | (computed)           | Plane index for each ROI×plane combination                                  |

**User-specified values for `nplanes` and `fs` are ignored when `mesoscan=True`.**

### Pipeline-Set Parameters

These parameters are set automatically by the pipeline and should not be manually configured:

| Parameter                 | Set By          | Value                                               |
|---------------------------|-----------------|-----------------------------------------------------|
| `file_io.data_path`       | `pipeline.py`   | Session's raw mesoscope data path                   |
| `file_io.save_path`       | `pipeline.py`   | Session's processed mesoscope data path             |
| `main.parallel_workers`   | CLI/MCP         | Number of workers (or auto-detected from CPU count) |
| `main.progress_bars`      | CLI/MCP         | Whether to show progress bars                       |
| `main.python_version`     | `single_day.py` | Current Python version                              |
| `main.sl_suite2p_version` | `single_day.py` | Current sl-suite2p version                          |

---

## Section 1: main

Global parameters that broadly define the processing configuration.

| Parameter            | Type      | Default | Description                                                                                       |
|----------------------|-----------|---------|---------------------------------------------------------------------------------------------------|
| `nplanes`            | int       | 1       | Number of imaging planes per TIFF. **Auto-overwritten for mesoscope.**                            |
| `nchannels`          | int       | 1       | Channels per plane (1 or 2). Channel frames interleaved as: plane1-ch1, plane1-ch2, plane2-ch1... |
| `functional_chan`    | int       | 1       | Channel for ROI extraction (1-indexed). Usually 1.                                                |
| `tau`                | float     | 0.4     | Sensor decay timescale in seconds. GCaMP6f: 0.4, GCaMP6s: ~1.5, GCaMP7f: ~0.5                     |
| `fs`                 | float     | 10.0014 | Per-plane sampling rate in Hz. **Auto-overwritten for mesoscope.**                                |
| `do_bidiphase`       | bool      | False   | Compute bidirectional phase offset (2P only).                                                     |
| `bidiphase`          | int       | 0       | Manual bidiphase offset. 0 = auto-detect from `nimg_init` frames.                                 |
| `bidi_corrected`     | bool      | False   | (Internal) Tracks if bidiphase correction applied.                                                |
| `frames_include`     | int       | -1      | Frames to process per plane. -1 = all, 0 = none.                                                  |
| `parallel_workers`   | int       | 20      | Numba parallel workers. 10-20 optimal per plane. -1/0 = all cores.                                |
| `progress_bars`      | bool      | False   | Show progress bars. Disable for parallel processing.                                              |
| `ignore_flyback`     | list[int] | []      | Flyback plane indices to exclude (0-based). e.g., [0] or [3].                                     |
| `python_version`     | str       | (auto)  | (Internal) Python version used.                                                                   |
| `sl_suite2p_version` | str       | (auto)  | (Internal) Library version used.                                                                  |

---

## Section 2: file_io

I/O parameters for input data location and output directories.

| Parameter            | Type      | Default | Description                                                                           |
|----------------------|-----------|---------|---------------------------------------------------------------------------------------|
| `data_path`          | str       | ""      | Root directory containing input TIFFs. Recursively searched. **Set by pipeline.**     |
| `save_path`          | str       | ""      | Output directory root. Pipeline creates `suite2p/` subdirectory. **Set by pipeline.** |
| `mesoscan`           | bool      | False   | **True for mesoscope data.** Triggers auto-overwrite of plane/ROI parameters.         |
| `delete_bin`         | bool      | False   | Delete registered .bin files after processing. **Must be False for multi-day.**       |
| `ignored_file_names` | list[str] | []      | Exact filenames to skip when loading TIFFs.                                           |

---

## Section 3: output

Parameters for GUI display. The pipeline always generates combined output.

| Parameter | Type  | Default  | Description                               |
|-----------|-------|----------|-------------------------------------------|
| `aspect`  | float | 0.666... | Pixel-to-micron ratio (X:Y) for GUI display. |

---

## Section 4: registration

Rigid registration parameters for motion correction.

| Parameter                      | Type  | Default | Description                                                                 |
|--------------------------------|-------|---------|-----------------------------------------------------------------------------|
| `do_registration`              | int   | 1       | 0=skip, 1=register if needed, 2=force re-register.                          |
| `align_by_chan`                | int   | 1       | Channel for alignment (1-indexed). Use structural channel if available.     |
| `nimg_init`                    | int   | 500     | Frames to compute reference image.                                          |
| `batch_size`                   | int   | 100     | Frames per registration batch. Low for fast drives, higher for slow drives. |
| `maxregshift`                  | float | 0.1     | Max shift as fraction of frame size (0.1 = 10%).                            |
| `smooth_sigma`                 | float | 1.15    | Gaussian sigma (pixels) for phase correlation smoothing.                    |
| `smooth_sigma_time`            | float | 0.0     | Gaussian sigma (frames) for temporal smoothing before correlation.          |
| `keep_movie_raw`               | bool  | False   | Keep unregistered binary file. Required for two-step registration.          |
| `two_step_registration`        | bool  | False   | Refinement registration. Requires `keep_movie_raw=True`.                    |
| `reg_tif`                      | bool  | False   | Write registered data to TIFF files (in addition to .bin).                  |
| `reg_tif_chan2`                | bool  | False   | Generate TIFFs for registered channel 2.                                    |
| `th_badframes`                 | float | 1.0     | Threshold for excluding bad frames during cropping. Lower = more excluded.  |
| `norm_frames`                  | bool  | True    | Normalize frames during shift detection.                                    |
| `force_ref_img`                | bool  | False   | Force use of pre-stored reference image.                                    |
| `pad_fft`                      | bool  | False   | Pad images during FFT to reduce edge effects.                               |
| `compute_registration_metrics` | int   | 1       | 0=skip, 1=if registering, 2=always compute metrics.                         |
| `reg_metric_n_pc`              | int   | 10      | Principal components for registration metrics.                              |

---

## Section 5: one_p_registration

Additional processing for 1-photon data registration.

| Parameter        | Type  | Default | Description                                                                |
|------------------|-------|---------|----------------------------------------------------------------------------|
| `one_p_reg`      | bool  | False   | Enable 1P preprocessing (high-pass filtering, tapering). **False for 2P.** |
| `spatial_hp_reg` | int   | 42      | Spatial high-pass filter window (pixels).                                  |
| `pre_smooth`     | float | 0.0     | Gaussian smoothing sigma before high-pass. 0 = disabled.                   |
| `spatial_taper`  | float | 40.0    | Edge pixels to ignore during registration.                                 |

---

## Section 6: non_rigid

Non-rigid registration for local motion correction.

| Parameter        | Type      | Default    | Description                                                  |
|------------------|-----------|------------|--------------------------------------------------------------|
| `nonrigid`       | bool      | True       | Enable non-rigid registration for non-uniform motion.        |
| `block_size`     | list[int] | [128, 128] | Block dimensions (pixels). Power of 2/3 recommended for FFT. |
| `snr_thresh`     | float     | 1.2        | SNR threshold for accepting block shifts.                    |
| `maxregshift_nr` | float     | 5.0        | Max block shift (pixels) relative to rigid shift.            |

---

## Section 7: roi_detection

Cell ROI detection parameters.

| Parameter           | Type  | Default | Description                                                                |
|---------------------|-------|---------|----------------------------------------------------------------------------|
| `preclassify`       | float | 0.5     | Min classifier confidence to keep ROI. 0 = keep all.                       |
| `roidetect`         | bool  | True    | Enable ROI detection and classification.                                   |
| `sparse_mode`       | bool  | True    | Sparse detection for sparse signals.                                       |
| `spatial_scale`     | int   | 0       | Spatial scale (pixels). 0 = auto-detect. Values scale by 6px (1→6, 2→12).  |
| `diameter`          | int   | 0       | Cell diameter (pixels). 0 = auto from spatial_scale. Required if sparse_mode=False. |
| `connected`         | bool  | True    | Require fully connected ROI regions.                                       |
| `threshold_scaling` | float | 2.0     | Detection threshold scaling. Higher = more distinct ROIs needed.           |
| `spatial_hp_detect` | int   | 25      | High-pass window for neuropil subtraction during detection.                |
| `max_overlap`       | float | 0.75    | Max allowed ROI overlap fraction. Higher overlap → discard.                |
| `high_pass`         | int   | 100     | Running mean window (frames) for drift removal.                            |
| `smooth_masks`      | bool  | True    | Smooth ROI masks in final pass.                                            |
| `max_iterations`    | int   | 50      | Max cell extraction iterations.                                            |
| `nbinned`           | int   | 5000    | Max binned frames for detection. More = more ROIs, slower.                 |
| `denoise`           | bool  | False   | Denoise binned movie before sparse detection. Requires `sparse_mode=True`. |

---

## Section 8: signal_extraction

Fluorescence signal extraction from ROIs.

| Parameter                      | Type | Default | Description                                              |
|--------------------------------|------|---------|----------------------------------------------------------|
| `extract_neuropil`             | bool | True    | Extract neuropil activity. False = assume zero neuropil. |
| `allow_overlap`                | bool | False   | Use overlapping pixels for extraction.                   |
| `minimum_neuropil_pixels`      | int  | 350     | Min neuropil region size (pixels).                       |
| `inner_neuropil_border_radius` | int  | 2       | Pixels between cell and neuropil region.                 |
| `lambda_percentile`            | int  | 50      | Lambda percentile threshold for neuropil mask exclusion. |

---

## Section 9: spike_deconvolution

Spike inference from fluorescence traces.

| Parameter              | Type  | Default   | Description                                                       |
|------------------------|-------|-----------|-------------------------------------------------------------------|
| `extract_spikes`       | bool  | True      | Deconvolve spikes from fluorescence.                              |
| `neuropil_coefficient` | float | 0.7       | Neuropil scaling before subtraction.                              |
| `baseline`             | str   | "maximin" | Baseline method: "maximin", "constant", or "constant_percentile". |
| `baseline_window`      | float | 60.0      | Sliding window (seconds) for maximin baseline.                    |
| `baseline_sigma`       | float | 10.0      | Gaussian sigma (seconds) for baseline computation.                |
| `baseline_percentile`  | float | 8.0       | Percentile for constant_percentile baseline.                      |

---

## Section 10: classification

ROI classification parameters.

| Parameter                | Type | Default | Description                                  |
|--------------------------|------|---------|----------------------------------------------|
| `soma_crop`              | bool | True    | Crop dendrites from ROIs for classification. |
| `use_builtin_classifier` | bool | False   | Use built-in classifier.                     |
| `classifier_path`        | str  | ""      | Path to custom classifier file.              |

---

## Section 11: channel2

Second channel processing.

| Parameter     | Type  | Default | Description                                                      |
|---------------|-------|---------|------------------------------------------------------------------|
| `chan2_thres` | float | 0.65    | Threshold for cross-channel ROI detection (ch1/ch2 pixel ratio). |

---

## User-Configurable vs Auto-Set Parameters

### Parameters Users Should Configure

| Parameter                    | When to Change                                       |
|------------------------------|------------------------------------------------------|
| `main.tau`                   | Different calcium indicator (GCaMP6s, GCaMP7f, etc.) |
| `main.ignore_flyback`        | Flyback planes present                               |
| `file_io.mesoscan`           | Processing mesoscope data                            |
| `file_io.ignored_file_names` | Specific TIFFs to exclude                            |
| `file_io.delete_bin`         | Set True only if no multi-day planned                |

### Parameters Typically Left at Default

- All registration parameters (work well for 2P mesoscope)
- ROI detection parameters (tuned for GCaMP6f)
- Signal extraction parameters
- Spike deconvolution parameters

---

## Configuration File Format

```yaml
main:
  nplanes: 1          # Overwritten by mesoscope
  nchannels: 1
  functional_chan: 1
  tau: 0.4
  fs: 10.0014         # Overwritten by mesoscope
  ignore_flyback: []

file_io:
  mesoscan: true      # Enable for mesoscope data
  delete_bin: false   # Keep for multi-day
  ignored_file_names: []

# Other sections use defaults...
```

---

## Multi-Day Compatibility Requirements

For sessions intended for multi-day processing:

```yaml
file_io:
  delete_bin: false   # REQUIRED: Keep registered binary files
```

This is the default, so no changes needed unless explicitly disabled. The pipeline always generates
combined output, which is required for multi-day processing.
