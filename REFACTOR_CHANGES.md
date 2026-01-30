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
| `dy` | `mroi_y_offset` | `int \| None` | Renamed from mesoscope to mroi |
| `dx` | `mroi_x_offset` | `int \| None` | Renamed from mesoscope to mroi |
| `lines` | `mroi_lines` | `list[int]` | Renamed from mesoscope to mroi |
| `iplane` | `plane_index` | `int \| None` | Tracks all input types, not just mesoscope |

### RegistrationData

| Old Name (ops key) | New Name | Type | Notes |
|--------------------|----------|------|-------|
| `yrange` | `valid_y_range` | `list[int]` | Changed from tuple for YAML serialization |
| `xrange` | `valid_x_range` | `list[int]` | Changed from tuple for YAML serialization |
| `bidiphase` | `bidirectional_phase_offset` | `int` | |
| `bidi_corrected` | `bidirectional_phase_corrected` | `bool` | |
| `rmin` | `normalization_minimum` | `int` | |
| `rmax` | `normalization_maximum` | `int` | |
| `refImg` | `reference_image` | `NDArray[np.float32] \| None` | Saved in registration_data.npz |
| `yoff` | `rigid_y_offsets` | `NDArray[np.float32] \| None` | Saved in registration_data.npz |
| `xoff` | `rigid_x_offsets` | `NDArray[np.float32] \| None` | Saved in registration_data.npz |
| `corrXY` | `rigid_correlations` | `NDArray[np.float32] \| None` | Saved in registration_data.npz |
| `yoff1` | `nonrigid_y_offsets` | `NDArray[np.float32] \| None` | Saved in registration_data.npz |
| `xoff1` | `nonrigid_x_offsets` | `NDArray[np.float32] \| None` | Saved in registration_data.npz |
| `corrXY1` | `nonrigid_correlations` | `NDArray[np.float32] \| None` | Saved in registration_data.npz |

### DetectionData

| Old Name (ops key) | New Name | Type | Notes |
|--------------------|----------|------|-------|
| `spatial_scale_pixels` | `spatial_scale` | `float` | |
| `diameter` | `cell_diameter` | `int` | |
| `aspect` | `aspect_ratio` | `float` | Moved from Output section |
| `meanImg` | `mean_image` | `NDArray[np.float32] \| None` | Saved in detection_data.npz |
| `meanImgE` | `enhanced_mean_image` | `NDArray[np.float32] \| None` | Saved in detection_data.npz |
| `max_proj` | `maximum_projection` | `NDArray[np.float32] \| None` | Saved in detection_data.npz |
| `Vcorr` | `correlation_map` | `NDArray[np.float32] \| None` | Saved in detection_data.npz |
| `meanImg_chan2` | `mean_image_channel_2` | `NDArray[np.float32] \| None` | Saved in detection_data.npz |
| N/A | `enhanced_mean_image_channel_2` | `NDArray[np.float32] \| None` | NEW: Saved in detection_data.npz |
| N/A | `maximum_projection_channel_2` | `NDArray[np.float32] \| None` | NEW: Saved in detection_data.npz |
| N/A | `correlation_map_channel_2` | `NDArray[np.float32] \| None` | NEW: Saved in detection_data.npz |

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

### ExtractionData (NEW)

This dataclass stores extraction and classification results, supporting two-channel workflows.

| Field | Type | Notes |
|-------|------|-------|
| `roi_statistics` | `list[ROIStatistics] \| None` | Saved as roi_statistics.npz (pickle-free) |
| `cell_fluorescence` | `NDArray[np.float32] \| None` | Saved as F.npy |
| `neuropil_fluorescence` | `NDArray[np.float32] \| None` | Saved as Fneu.npy |
| `subtracted_fluorescence` | `NDArray[np.float32] \| None` | Saved as Fsub.npy |
| `spikes` | `NDArray[np.float32] \| None` | Saved as spks.npy |
| `cell_classification` | `NDArray[np.float32] \| None` | Saved as iscell.npy |
| `roi_statistics_channel_2` | `list[ROIStatistics] \| None` | Saved as roi_statistics_channel_2.npz |
| `cell_fluorescence_channel_2` | `NDArray[np.float32] \| None` | Saved as F_chan2.npy |
| `neuropil_fluorescence_channel_2` | `NDArray[np.float32] \| None` | Saved as Fneu_chan2.npy |
| `subtracted_fluorescence_channel_2` | `NDArray[np.float32] \| None` | Saved as Fsub_chan2.npy |
| `spikes_channel_2` | `NDArray[np.float32] \| None` | Saved as spks_chan2.npy |
| `cell_classification_channel_2` | `NDArray[np.float32] \| None` | Saved as iscell_chan2.npy |
| `cell_colocalization` | `NDArray[np.float32] \| None` | Saved as redcell.npy |

### ROIStatistics (NEW)

This dataclass replaces the legacy dictionary-based `stat.npy` format with a pickle-free structure.

**Core pixel data (required):**

| Field | Type | Legacy Name |
|-------|------|-------------|
| `y_pixels` | `NDArray[np.int32]` | `ypix` |
| `x_pixels` | `NDArray[np.int32]` | `xpix` |
| `pixel_weights` | `NDArray[np.float32]` | `lam` |
| `centroid` | `list[float]` | `med` |
| `footprint` | `int` | `footprint` |

**Shape statistics (required):**

| Field | Type | Legacy Name |
|-------|------|-------------|
| `mean_r_squared` | `float` | `mrs` |
| `mean_r_squared_baseline` | `float` | `mrs0` |
| `compactness` | `float` | `compact` |
| `solidity` | `float` | `solidity` |
| `pixel_count` | `int` | `npix` |
| `soma_pixel_count` | `int` | `npix_soma` |
| `soma_mask` | `NDArray[np.bool_]` | `soma_crop` |
| `overlap_mask` | `NDArray[np.bool_]` | `overlap` |
| `radius` | `float` | `radius` |
| `aspect_ratio` | `float` | `aspect_ratio` |
| `normalized_pixel_count` | `float` | `npix_norm` |
| `normalized_pixel_count_full` | `float` | `npix_norm_no_crop` |

**Optional extraction data:**

| Field | Type | Legacy Name |
|-------|------|-------------|
| `skewness` | `float \| None` | `skew` |
| `standard_deviation` | `float \| None` | `std` |
| `neuropil_mask` | `NDArray[np.bool_] \| None` | `neuropil_mask` |

**Optional multi-plane/multi-day data:**

| Field | Type | Legacy Name |
|-------|------|-------------|
| `plane_index` | `int \| None` | `iplane` |
| `cluster_id` | `int \| None` | `id` |
| `raveled_pixels` | `NDArray[np.int32] \| None` | `ipix` |
| `session_count` | `int \| None` | `num_sessions` |

**Serialization methods:**

- `save_list(roi_list, file_path)` - Saves to .npz with `allow_pickle=False`
- `load_list(file_path)` - Loads from .npz file

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

## Input Architecture Changes

### Unified Input Format

The input format has been simplified and standardized. All input data must now be in TIFF format accompanied by a
`suite2p_parameters.json` metadata file. This replaces the previous `InputFormat` enum approach.

**Key changes:**
- `InputFormat` enum has been **REMOVED** (no longer needed)
- `FileIO.input_format` field has been **REMOVED**
- All inputs are TIFF files + JSON metadata
- The JSON file describes acquisition parameters (frame rate, planes, channels, MROI geometry)

### AcquisitionParameters Dataclass

A new `AcquisitionParameters` dataclass stores acquisition metadata loaded from the input JSON file:

```python
@dataclass
class AcquisitionParameters(YamlConfig):
    frame_rate: float           # Volume acquisition rate in Hz
    plane_number: int = 1       # Physical planes per volume
    channel_number: int = 1     # Channels per plane (1 or 2)
    roi_number: int = 1         # ROIs per plane (1 for standard, >1 for MROI)
    roi_lines: list[list[int]]  # Line indices for each ROI (MROI only)
    roi_x_coordinates: list[int] # X offsets for each ROI (MROI only)
    roi_y_coordinates: list[int] # Y offsets for each ROI (MROI only)

    @property
    def is_mroi(self) -> bool: ...        # True if roi_number > 1

    @property
    def virtual_plane_count(self) -> int: ...  # roi_number * plane_number

    @classmethod
    def from_json(cls, path: Path) -> AcquisitionParameters: ...
```

### RuntimeContext Updated

`RuntimeContext` now includes `AcquisitionParameters` as a required field:

```python
@dataclass
class RuntimeContext:
    config: SingleDayConfiguration    # User configuration (immutable)
    acquisition: AcquisitionParameters # Acquisition metadata from JSON
    runtime: SingleDayRuntimeData     # Pipeline-computed data (mutable)
```

### MROI (Multi-ROI) Generalization

The "mesoscope" handling has been generalized to support any line-scanning microscope with MROI capability:

| Old Term | New Term | Description |
|----------|----------|-------------|
| Mesoscope data | MROI data | Multi-ROI line-scanning data |
| `mesoscan` format | `is_mroi` property | Whether data uses multiple ROIs |
| Mesoscope parameters | ROI geometry | `roi_lines`, `roi_x_coordinates`, `roi_y_coordinates` |

Virtual planes are computed as ROI × physical plane combinations. For 2 ROIs and 3 planes, there are 6 virtual planes
organized as: ROI 0 plane 0, ROI 0 plane 1, ROI 0 plane 2, ROI 1 plane 0, ROI 1 plane 1, ROI 1 plane 2.

### Data Discovery

The TIFF discovery process now uses a two-step approach:

1. **Recursively search** for `suite2p_parameters.json` in the data directory
2. **Non-recursively scan** the same directory as the JSON file for TIFF files

This ensures that TIFF files and their metadata are co-located and simplifies the discovery process.

### Required JSON Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `frame_rate` | float | Always | Volume acquisition rate in Hz |
| `plane_number` | int | Always | Number of physical imaging planes |
| `channel_number` | int | Always | Number of channels (1 or 2) |
| `roi_number` | int | Optional | Number of ROIs (defaults to 1) |
| `roi_lines` | list[list[int]] | MROI only | Line indices for each ROI |
| `roi_x_coordinates` | list[int] | MROI only | X offset for each ROI |
| `roi_y_coordinates` | list[int] | MROI only | Y offset for each ROI |

---

## New Enumerations

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

## Output File Structure Changes

### Plane Directory Naming

Plane directories now use underscore separators for readability:

| Old Pattern | New Pattern |
|-------------|-------------|
| `plane0/` | `plane_0/` |
| `plane1/` | `plane_1/` |

### Binary File Naming

Binary files now use a consistent `channel_N_` prefix pattern:

| Old Name | New Name |
|----------|----------|
| `data.bin` | `channel_1_data.bin` |
| `data_raw.bin` | `channel_1_data_raw.bin` |
| `data_chan2.bin` | `channel_2_data.bin` |
| `data_chan2_raw.bin` | `channel_2_data_raw.bin` |

---

## Pickle Disabled

All NumPy save/load operations now use `allow_pickle=False` to eliminate the security risk from pickle serialization.

**Pattern:**
```python
# Loading
np.load(path, allow_pickle=False)

# Saving
np.save(path, array, allow_pickle=False)

# Saving multiple arrays to .npz
np.savez(path, allow_pickle=False, **arrays_dict)
```

---

## Function Renames

### io/save.py

| Old Name | New Name | Notes |
|----------|----------|-------|
| `compute_dydx` | `compute_plane_offsets` | Computes y/x displacement for each plane |
| `combined` | `combine_planes` | Combines multi-plane data into unified output |

Both functions now accept `list[RuntimeContext]` instead of loading ops.npy from disk.

---

## Consolidated .npz Files

Arrays are now saved in consolidated .npz archives instead of individual .npy files:

| Archive File | Contents |
|--------------|----------|
| `registration_data.npz` | reference_image, rigid_y_offsets, rigid_x_offsets, rigid_correlations, nonrigid_y_offsets, nonrigid_x_offsets, nonrigid_correlations |
| `detection_data.npz` | mean_image, enhanced_mean_image, maximum_projection, correlation_map, mean_image_channel_2, enhanced_mean_image_channel_2, maximum_projection_channel_2, correlation_map_channel_2 |
| `roi_statistics.npz` | All ROIStatistics fields (concatenated with counts for variable-length arrays) |

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
