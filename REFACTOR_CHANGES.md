# Refactor Changes: Field Rename Mappings and Deprecation Decisions

This document provides a complete record of all changes made during Phase 1 of the IO/Config refactor. Use this as a
reference when updating the rest of the codebase to use the new field names.

---

## Class Renames

| Old Name | New Name |
|----------|----------|
| `SingleDayS2PConfiguration` | `SingleDayConfiguration` |
| `SingleDayS2PRuntimeData` | `SingleDayRuntimeData` |
| `OnePRegistration` | `OnePhotonRegistration` |
| `NonRigid` | `NonRigidRegistration` |

---

## Field Renames by Section

### Main

| Old Name | New Name | Type | Notes |
|----------|----------|------|-------|
| `nplanes` | `plane_count` | `int` | |
| `nchannels` | `two_channels` | `bool` | Changed from int (1 or 2) to bool |
| `functional_chan` | `first_channel_functional` | `bool` | Changed from int (1 or 2) to bool |
| N/A | `second_channel_functional` | `bool` | NEW: Enables independent ROI detection in channel 2 |
| N/A | `colocalization_threshold` | `float` | Moved from Channel2 section |
| `fs` | `sampling_rate` | `float` | |
| `do_bidiphase` | `compute_bidirectional_phase_offset` | `bool` | |
| `bidiphase` | `bidirectional_phase_offset` | `int` | |
| `progress_bars` | `display_progress_bars` | `bool` | |
| `ignore_flyback` | `ignored_flyback_planes` | `list[int]` | |
| `frames_include` | REMOVED | | Deprecated feature |

### FileIO

| Old Name | New Name | Type | Notes |
|----------|----------|------|-------|
| `mesoscan` | `input_format` | `InputFormat \| str` | Changed from bool to StrEnum |
| `look_one_level_down` | REMOVED | | Not used |
| `delete_bin` | REMOVED | | Binary files always kept |
| `data_path` | `data_path` | `Path \| None` | Changed from `list[str]` to `Path` |
| `save_path0` | `save_path` | `Path \| None` | Changed from `str` to `Path` |
| `tiff_list` | `ignored_file_names` | `list[str]` | Inverted semantics (ignore vs include) |

### Registration

| Old Name | New Name | Type | Notes |
|----------|----------|------|-------|
| `do_registration` | `repeat_registration` | `bool` | Clarified semantics |
| `align_by_chan` | `align_by_first_channel` | `bool` | Changed from int to bool |
| `nimg_init` | `reference_frame_count` | `int` | |
| `batch_size` | `batch_size` | `int` | Unchanged |
| `maxregshift` | `maximum_shift_fraction` | `float` | |
| `smooth_sigma` | `spatial_smoothing_sigma` | `float` | |
| `smooth_sigma_time` | `temporal_smoothing_sigma` | `float` | |
| `keep_movie_raw` | `keep_movie_raw` | `bool` | Unchanged |
| `two_step_registration` | `two_step_registration` | `bool` | Unchanged |
| `th_badframes` | `bad_frame_threshold` | `float` | |
| `norm_frames` | `normalize_frames` | `bool` | |
| `compute_registration_metrics` | `compute_registration_metrics` | `bool` | Unchanged |
| `reg_metric_n_pc` | `registration_metric_principal_components` | `int` | |
| `reg_tif` | REMOVED | | Can convert via BinaryFile post-hoc |
| `reg_tif_chan2` | REMOVED | | Can convert via BinaryFile post-hoc |
| `force_ref_img` | REMOVED | | Not used |
| `pad_fft` | REMOVED | | Always enabled (hardcoded) |

### OnePhotonRegistration (formerly OnePRegistration)

| Old Name | New Name | Type | Notes |
|----------|----------|------|-------|
| `one_p_reg` | `enabled` | `bool` | |
| `spatial_hp_reg` | `spatial_highpass_window` | `int` | |
| `pre_smooth` | `pre_smoothing_sigma` | `float` | |
| `spatial_taper` | `edge_taper_pixels` | `float` | |

### NonRigidRegistration (formerly NonRigid)

| Old Name | New Name | Type | Notes |
|----------|----------|------|-------|
| `nonrigid` | `enabled` | `bool` | |
| `block_size` | `block_size` | `list[int]` | Unchanged |
| `snr_thresh` | `signal_to_noise_threshold` | `float` | |
| `maxregshift_nr` | `maximum_block_shift` | `float` | |

### ROIDetection

| Old Name | New Name | Type | Notes |
|----------|----------|------|-------|
| `roidetect` | `enabled` | `bool` | |
| `sparse_mode` | REMOVED | | Legacy non-sparse detection removed |
| `connected` | REMOVED | | Legacy non-sparse detection removed |
| `smooth_masks` | REMOVED | | Legacy non-sparse detection removed |
| `preclassify` | `preclassification_threshold` | `float` | |
| `spatial_scale` | `spatial_scale` | `int` | Unchanged |
| `diameter` | `diameter` | `int` | Unchanged |
| `threshold_scaling` | `threshold_scaling` | `float` | Unchanged |
| `spatial_hp_detect` | `spatial_highpass_window` | `int` | |
| `max_overlap` | `maximum_overlap` | `float` | |
| `high_pass` | `temporal_highpass_window` | `int` | |
| `max_iterations` | `maximum_iterations` | `int` | |
| `nbinned` | `maximum_binned_frames` | `int` | |
| `denoise` | `denoise` | `bool` | Unchanged |

### SignalExtraction

| Old Name | New Name | Type | Notes |
|----------|----------|------|-------|
| `extract_neuropil` | `extract_neuropil` | `bool` | Unchanged |
| `allow_overlap` | `allow_overlap` | `bool` | Unchanged |
| `min_neuropil_pixels` | `minimum_neuropil_pixels` | `int` | |
| `inner_neuropil_radius` | `inner_neuropil_border_radius` | `int` | |
| `lam_percentile` | `cell_probability_percentile` | `int` | |

### SpikeDeconvolution

| Old Name | New Name | Type | Notes |
|----------|----------|------|-------|
| `spikedetect` | `extract_spikes` | `bool` | |
| `neucoeff` | `neuropil_coefficient` | `float` | |
| `baseline` | `baseline_method` | `BaselineMethod \| str` | Changed from str to StrEnum |
| `win_baseline` | `baseline_window` | `float` | |
| `sig_baseline` | `baseline_sigma` | `float` | |
| `prctile_baseline` | `baseline_percentile` | `float` | |

### Classification

| Old Name | New Name | Type | Notes |
|----------|----------|------|-------|
| `soma_crop` | `crop_to_soma` | `bool` | |
| `use_builtin_classifier` | `use_builtin_classifier` | `bool` | Unchanged |
| `classifier_path` | `custom_classifier_path` | `Path \| None` | Changed from `str` to `Path` |

---

## Runtime Data Field Renames

### IOData

| Old Name (ops key) | New Name | Type | Notes |
|--------------------|----------|------|-------|
| `Ly` | `frame_height` | `int` | |
| `Lx` | `frame_width` | `int` | |
| `nframes` | `frame_count` | `int` | |
| `reg_file` | `registered_binary_path` | `Path \| None` | |
| `raw_file` | `raw_binary_path` | `Path \| None` | |
| `reg_file_chan2` | `registered_binary_path_channel_2` | `Path \| None` | |
| `raw_file_chan2` | `raw_binary_path_channel_2` | `Path \| None` | |
| `save_path`, `ops_path` | `output_directory` | `Path \| None` | |
| `dy` | `mesoscope_y_offset` | `int \| None` | |
| `dx` | `mesoscope_x_offset` | `int \| None` | |
| `lines` | `mesoscope_lines` | `list[int]` | |
| `iplane` | `plane_index` | `int \| None` | Tracks all input types, not just mesoscope |

### RegistrationData

| Old Name (ops key) | New Name | Type | Notes |
|--------------------|----------|------|-------|
| `yrange` | `valid_y_range` | `tuple[int, int]` | |
| `xrange` | `valid_x_range` | `tuple[int, int]` | |
| `bidiphase` | `bidirectional_phase_offset` | `int` | |
| `bidi_corrected` | `bidirectional_phase_corrected` | `bool` | |
| `rmin` | `normalization_minimum` | `int` | |
| `rmax` | `normalization_maximum` | `int` | |
| `refImg` | `reference_image` | `NDArray[np.float32] \| None` | Saved as reference_image.npy |
| `yoff` | `rigid_y_offsets` | `NDArray[np.float32] \| None` | Saved as rigid_y_offsets.npy |
| `xoff` | `rigid_x_offsets` | `NDArray[np.float32] \| None` | Saved as rigid_x_offsets.npy |
| `corrXY` | `rigid_correlations` | `NDArray[np.float32] \| None` | Saved as rigid_correlations.npy |
| `yoff1` | `nonrigid_y_offsets` | `NDArray[np.float32] \| None` | Saved as nonrigid_y_offsets.npy |
| `xoff1` | `nonrigid_x_offsets` | `NDArray[np.float32] \| None` | Saved as nonrigid_x_offsets.npy |
| `corrXY1` | `nonrigid_correlations` | `NDArray[np.float32] \| None` | Saved as nonrigid_correlations.npy |

### DetectionData

| Old Name (ops key) | New Name | Type | Notes |
|--------------------|----------|------|-------|
| `spatial_scale_pixels` | `spatial_scale` | `float` | |
| `diameter` | `cell_diameter` | `int` | |
| `aspect` | `aspect_ratio` | `float` | Moved from Output section |
| `meanImg` | `mean_image` | `NDArray[np.float32] \| None` | Saved as mean_image.npy |
| `meanImgE` | `enhanced_mean_image` | `NDArray[np.float32] \| None` | Saved as enhanced_mean_image.npy |
| `max_proj` | `maximum_projection` | `NDArray[np.float32] \| None` | Saved as maximum_projection.npy |
| `Vcorr` | `correlation_map` | `NDArray[np.float32] \| None` | Saved as correlation_map.npy |
| `meanImg_chan2` | `mean_image_channel_2` | `NDArray[np.float32] \| None` | Saved as mean_image_channel_2.npy |
| N/A | `enhanced_mean_image_channel_2` | `NDArray[np.float32] \| None` | NEW: Saved as enhanced_mean_image_channel_2.npy |
| N/A | `maximum_projection_channel_2` | `NDArray[np.float32] \| None` | NEW: Saved as maximum_projection_channel_2.npy |
| N/A | `correlation_map_channel_2` | `NDArray[np.float32] \| None` | NEW: Saved as correlation_map_channel_2.npy |

### TimingData

| Old Name (ops key) | New Name | Type | Notes |
|--------------------|----------|------|-------|
| `timing['registration']` | `registration_time` | `float` | |
| `timing['two_step_registration']` | `two_step_registration_time` | `float` | |
| `timing['registration_metrics']` | `registration_metrics_time` | `float` | |
| `timing['detection']` | `detection_time` | `float` | |
| `timing['extraction']` | `extraction_time` | `float` | |
| `timing['classification']` | `classification_time` | `float` | |
| `timing['deconvolution']` | `deconvolution_time` | `float` | |
| N/A | `detection_time_channel_2` | `float` | NEW: For independent channel 2 detection |
| N/A | `extraction_time_channel_2` | `float` | NEW: For independent channel 2 extraction |
| N/A | `classification_time_channel_2` | `float` | NEW: For independent channel 2 classification |
| N/A | `deconvolution_time_channel_2` | `float` | NEW: For independent channel 2 deconvolution |
| `timing['total_plane_runtime']` | `total_plane_time` | `float` | |
| `date_processed` | `date_processed` | `str` | Unchanged |

---

## Deprecated/Removed Sections

### Channel2 Section (REMOVED)

The entire `Channel2` configuration section has been removed. The only parameter (`chan2_thres`) has been moved to
`Main.colocalization_threshold`.

### Output Section (REMOVED)

The `Output` section has been removed. The only parameter (`aspect_ratio`) has been moved to `DetectionData.aspect_ratio`
as it is computed during detection, not user-configurable.

---

## Deprecated Features

### Legacy Non-Sparse Detection

The legacy non-sparse (sourcery) detection algorithm has been removed. The pipeline now only supports sparse detection
which automatically estimates cell size and is recommended for all use cases.

**Files removed:**
- `src/sl_suite2p/detection/sourcery.py`

**Removed parameters:**
- `ROIDetection.sparse_mode` - Always True (sparse detection only)
- `ROIDetection.connected` - Non-sparse only parameter
- `ROIDetection.smooth_masks` - Non-sparse only parameter

### FFT Padding Option

FFT padding is now always enabled. The option to disable it has been removed.

**Files affected:**
- `src/sl_suite2p/registration/utils.py` - Hardcode `pad_fft=True`
- `src/sl_suite2p/registration/register.py` - Remove conditional

**Removed parameters:**
- `Registration.pad_fft` - Always True (hardcoded)

### TIFF Export During Registration

The ability to export registered frames as TIFFs during registration has been removed. Users can convert binary files
to TIFFs post-hoc using the `BinaryFile` class.

**Removed parameters:**
- `Registration.reg_tif`
- `Registration.reg_tif_chan2`

### Binary File Deletion

The option to delete binary files after processing has been removed. Binary files are always kept.

**Removed parameters:**
- `FileIO.delete_bin`

---

## New Enumerations

### InputFormat

```python
class InputFormat(StrEnum):
    TIFF = "tiff"      # Standard TIFF files
    MESOSCAN = "mesoscan"  # ScanImage Mesoscope TIFFs
```

Replaces the boolean `mesoscan` parameter with a type-safe enum.

### BaselineMethod

```python
class BaselineMethod(StrEnum):
    MAXIMIN = "maximin"  # Sliding window min/max filters
    CONSTANT = "constant"  # Global minimum
    CONSTANT_PERCENTILE = "constant_percentile"  # Low percentile
```

Replaces the string `baseline` parameter with a type-safe enum.

---

## Path Handling

All path parameters now use `Path | None` instead of `str`. Each dataclass with path fields implements `__post_init__`
to convert strings to `Path` objects when loading from YAML.

**Affected classes:**
- `FileIO`: `data_path`, `save_path`
- `Classification`: `custom_classifier_path`
- `IOData`: `registered_binary_path`, `raw_binary_path`, `registered_binary_path_channel_2`,
  `raw_binary_path_channel_2`, `output_directory`

---

## Pickle Disabled

All NumPy save/load operations now use `allow_pickle=False` to eliminate the security risk from pickle serialization.

**Pattern:**
```python
# Loading
np.load(path, allow_pickle=False)

# Saving
np.save(path, array, allow_pickle=False)
```

---

## Multi-Day Configuration Updates

The `multi_day.py` module now imports shared classes from `single_day.py` instead of duplicating them:

```python
from .single_day import SignalExtraction, SpikeDeconvolution
```

This ensures consistency between single-day and multi-day pipeline parameters.

---

## Search Patterns for Updating Codebase

Use these patterns to find code that needs updating:

### Configuration Access Patterns

```bash
# Find ops dictionary access
rg "ops\[" --type py

# Find specific legacy field names
rg "nplanes|nchannels|functional_chan" --type py
rg "mesoscan|delete_bin" --type py
rg "do_registration|align_by_chan" --type py
rg "sparse_mode|connected|smooth_masks" --type py
rg "pad_fft|reg_tif" --type py
```

### Runtime Data Access Patterns

```bash
# Find legacy runtime data keys
rg "ops\[.Ly.\]|ops\[.Lx.\]|ops\[.nframes.\]" --type py
rg "ops\[.reg_file.\]|ops\[.raw_file.\]" --type py
rg "ops\[.yoff.\]|ops\[.xoff.\]|ops\[.corrXY.\]" --type py
rg "ops\[.meanImg.\]|ops\[.meanImgE.\]" --type py
rg "ops\[.refImg.\]|ops\[.Vcorr.\]" --type py
```

### Pickle Usage

```bash
# Find pickle usage that needs updating
rg "allow_pickle" --type py
rg "np\.load\(" --type py
rg "np\.save\(" --type py
```
