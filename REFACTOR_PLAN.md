# IO/Config Refactor: Replace ops Dictionary with YAML + .npy Architecture

## Summary

Replace the pickle-based `ops.npy` dictionary with structured dataclasses and YAML + `.npy` file persistence
throughout the single-day pipeline, achieving clear separation between user configuration (immutable) and runtime
data (computed by pipeline).

## Key Changes

### Current Architecture (ops dictionary)

- Single `ops: dict[str, Any]` mixes user config with runtime data
- Persisted as `ops.npy` pickle files (security concern)
- Functions pass entire dictionary, extract what they need
- Runtime data (images, offsets) embedded in same dict as config

### Target Architecture (RuntimeContext)

- `SingleDayConfiguration` - user config (YAML, human-readable)
- `AcquisitionParameters` - input metadata from JSON (describes recording setup)
- `SingleDayRuntimeData` - pipeline outputs (YAML + .npy for arrays)
- `RuntimeContext` - combines all three for function calls
- Clear separation: user config vs input metadata vs pipeline outputs
- No backward compatibility - clean break from legacy format

### Unified Input Format

All inputs are now standardized as:
- TIFF files containing the imaging data
- `suite2p_parameters.json` file containing acquisition metadata (frame rate, planes, channels, MROI geometry)

This replaces the previous `InputFormat` enum approach and allows easy extension to support new microscope types
by implementing converter algorithms that produce this standardized format.

---

## Phase 1: Configuration Dataclass Refactoring (COMPLETED)

All configuration and runtime dataclasses have been updated with descriptive field names.

**Files modified:**
- `src/sl_suite2p/configuration/single_day.py` - Complete field renames, new enums, Path types
- `src/sl_suite2p/configuration/multi_day.py` - Imports shared classes from single_day.py
- `src/sl_suite2p/configuration/__init__.py` - Updated exports

**Key changes:**
- Added `BaselineMethod` StrEnum for type-safe configuration
- Added `AcquisitionParameters` dataclass for input metadata
- Added `ExtractionData` dataclass for fluorescence, spikes, and classification data (two-channel support)
- Added `ROIStatistics` dataclass with pickle-free serialization (replaces dict-based stat.npy)
- Changed all path fields from `str` to `Path | None` with `__post_init__` conversion
- Changed tuple fields to list for YAML serialization compatibility (valid_y_range, valid_x_range, centroid)
- Consolidated registration arrays into single `registration_data.npz` file
- Consolidated detection arrays into single `detection_data.npz` file
- Added `allow_pickle=False` to all NumPy save/load/savez operations
- Added channel 2 detection and timing fields for independent dual-channel processing
- Renamed IOData fields: mesoscope_* → mroi_* (e.g., mroi_y_offset, mroi_x_offset, mroi_lines)
- Removed deprecated sections (Channel2, Output)
- Removed `InputFormat` enum (replaced by unified TIFF + JSON input architecture)
- Imported shared SignalExtraction and SpikeDeconvolution in multi_day.py

See `REFACTOR_CHANGES.md` for the complete field rename mappings and deprecation decisions.

---

## Phase 2: Update IO Module (COMPLETED)

**Files modified:**

- `src/sl_suite2p/io/tiff.py` - All IO functionality consolidated here
- `src/sl_suite2p/io/__init__.py` - Updated exports
- `src/sl_suite2p/io/utils.py` - **REMOVED** (merged into tiff.py)

**Key changes:**

1. Created `AcquisitionParameters` dataclass with `from_json()` method
2. Added `AcquisitionParameters` field to `RuntimeContext`
3. Created `initialize_plane_contexts()` replacing `initialize_plane_ops()`
   - Now accepts `config` + `acquisition` instead of ops dictionaries
   - Uses `acquisition.is_mroi` and `acquisition.virtual_plane_count` for plane setup
   - Computes ROI index and physical plane index for MROI data
4. Implemented optimized TIFF discovery with two-step approach:
   - `find_acquisition_parameters()` - recursively finds `suite2p_parameters.json`
   - `discover_tiff_files()` - non-recursive scan in the same directory as JSON
5. Created unified `convert_tiffs_to_binary()` replacing both `tiff_to_binary()` and `mesoscan_to_binary()`
   - Handles both standard TIFF and MROI data based on `acquisition.is_mroi`
   - Uses `RuntimeContext` instead of ops dictionaries
   - Updates runtime data with frame dimensions, counts, and mean images
6. Removed `InputFormat` enum (no longer needed with unified input format)
7. Removed `input_format` field from `FileIO` class
8. Merged all utility functions from `utils.py` into `tiff.py`
9. Removed legacy `tiff_to_binary()` and `mesoscan_to_binary()` functions

**New public API:**

```python
# Find acquisition parameters (recursively searches for JSON)
acquisition, data_dir = find_acquisition_parameters(config.file_io.data_path)

# Discover TIFF files (non-recursive in same directory as JSON)
tiff_files = discover_tiff_files(data_dir, config.file_io.ignored_file_names)

# Initialize plane contexts
contexts = initialize_plane_contexts(config, acquisition)

# Convert TIFFs to binary format
contexts = convert_tiffs_to_binary(contexts, tiff_files)
```

---

## Phase 3: Update Registration Module (PENDING)

**Files:**

- `src/sl_suite2p/registration/register.py`
- `src/sl_suite2p/registration/rigid.py`
- `src/sl_suite2p/registration/utils.py`
- `src/sl_suite2p/single_day.py`

1. Update `registration_wrapper()` to accept RuntimeContext
2. Update all functions to use new field names
3. Always enable FFT padding (`pad_fft=True` hardcoded)
4. Save ref_image.npy, mean_image.npy to separate files

---

## Phase 4: Update Detection/Extraction Modules (PENDING)

**Files:**

- `src/sl_suite2p/detection/detect.py`
- `src/sl_suite2p/detection/sparsedetect.py`
- `src/sl_suite2p/detection/sourcery.py` (remove or deprecate)
- `src/sl_suite2p/extraction/extract.py`

1. Remove sourcery detection code path (always use sparse)
2. Update `detection_wrapper()` to accept RuntimeContext
3. Update `extraction_wrapper()` to accept RuntimeContext
4. Read config values from `ctx.config`, write outputs to `ctx.runtime`

---

## Phase 5: Update Pipeline Orchestration (PENDING)

**File:** `src/sl_suite2p/single_day.py`

1. `resolve_ops()` → `resolve_pipeline()`: Creates directories, saves initial config YAML
2. `resolve_binaries()` → `binarize_session()`: Returns list of RuntimeContext
3. `process_plane()`: Loads RuntimeContext, processes, saves runtime_data.yaml
4. `combine_planes()`: Updated to read runtime_data.yaml from each plane

---

## Phase 6: Update Combined Folder Generation (PARTIALLY COMPLETE)

**File:** `src/sl_suite2p/io/save.py`

**Completed:**
- Renamed `compute_dydx()` → `compute_plane_offsets()`
- Renamed `combined()` → `combine_planes()`
- Refactored `combine_planes()` to accept `list[RuntimeContext]` instead of loading ops.npy
- Removed legacy `_compute_dydx_from_legacy_ops()` function
- Function now reads data from `ctx.runtime.extraction.*` instead of loading .npy files from disk

**Remaining:**
- Update callers in `single_day.py`, `gui/io.py`, `gui/reggui.py`, and `multiday/process.py`
- Save combined `runtime_data.yaml` with aggregated metadata

---

## Phase 7: Update Multiday Integration (PENDING)

**File:** `src/sl_suite2p/multiday/io.py`

Update `import_sessions()`:

```python
runtime = SingleDayRuntimeData.load(combined_folder / "runtime_data.yaml")
images = {
    "mean": runtime.detection.mean_image,
    "enhanced": runtime.detection.enhanced_mean_image,
    "max": runtime.detection.maximum_projection,
}
```

---

## Phase 8: Update GUI (PENDING)

**Files:**

- `src/sl_suite2p/gui/io.py`
- `src/sl_suite2p/gui/rungui.py`
- `src/sl_suite2p/gui/merge.py`
- Other GUI files as needed

1. Update GUI data loading to read `runtime_data.yaml` and image `.npy` files
2. Update GUI data saving to use new format
3. Replace ops dictionary access patterns with dataclass attribute access
4. Remove references to deprecated parameters

---

## File Structure Change

**Before:**

```
suite2p/
├── ops.npy (pickle)
├── plane0/
│   ├── ops.npy (pickle)
│   ├── data.bin
│   └── F.npy, Fneu.npy, stat.npy, ...
└── combined/
    ├── ops.npy (pickle)
    └── F.npy, Fneu.npy, stat.npy, ...
```

**After:**

```
suite2p/
├── configuration.yaml
├── plane_0/
│   ├── runtime_data.yaml
│   ├── registration_data.npz     # reference_image, rigid/nonrigid offsets and correlations
│   ├── detection_data.npz        # mean_image, enhanced_mean_image, max_proj, correlation_map (+chan2)
│   ├── roi_statistics.npz        # Pickle-free ROI statistics
│   ├── channel_1_data.bin
│   ├── F.npy, Fneu.npy, Fsub.npy, spks.npy, iscell.npy
│   └── redcell.npy (if two channels)
└── combined/
    ├── runtime_data.yaml
    ├── mean_image.npy
    ├── enhanced_mean_image.npy
    ├── maximum_projection.npy
    ├── correlation_map.npy
    └── F.npy, Fneu.npy, Fsub.npy, spks.npy, stat.npy, iscell.npy
```

---

## Critical Files to Modify

| File                                            | Changes                                               |
|-------------------------------------------------|-------------------------------------------------------|
| `src/sl_suite2p/configuration/single_day.py`    | Dataclass field renames (DONE)                        |
| `src/sl_suite2p/configuration/multi_day.py`     | Import shared classes from single_day (DONE)          |
| `src/sl_suite2p/single_day.py`                  | Update field references                               |
| `src/sl_suite2p/io/utils.py`                    | Update initialize_plane_ops → initialize_plane_runtime|
| `src/sl_suite2p/io/tiff.py`                     | Update tiff_to_binary, mesoscan_to_binary             |
| `src/sl_suite2p/io/save.py`                     | Update combined() to use new file format              |
| `src/sl_suite2p/registration/register.py`       | Update field names, always enable FFT padding         |
| `src/sl_suite2p/registration/rigid.py`          | Update field names                                    |
| `src/sl_suite2p/registration/utils.py`          | Hardcode pad_fft=True                                 |
| `src/sl_suite2p/detection/detect.py`            | Update field names, remove non-sparse code path       |
| `src/sl_suite2p/extraction/extract.py`          | Update field names                                    |
| `src/sl_suite2p/multiday/io.py`                 | Update import_sessions to read new format             |
| `src/sl_suite2p/cli.py`                         | Update CLI entry points                               |
| `src/sl_suite2p/mcp/server.py`                  | Update status functions                               |
| `src/sl_suite2p/gui/*.py`                       | Update all GUI files                                  |

---

## Verification Plan

1. **Unit tests:** Dataclass serialization/deserialization round-trips
2. **Integration tests:** Run pipeline on test data, verify outputs match baseline
3. **Multiday integration:** Ensure multiday pipeline reads new single-day outputs correctly
4. **GUI verification:** Test GUI loads and displays data correctly from new format

---

## Notes

- **No backward compatibility:** Clean break from legacy ops.npy format
- **Non-sparse detection removed:** Only sparse detection supported (sourcery.py deleted)
- **FFT padding:** Always enabled (hardcoded)
- **Two-channel support:** Expanded to support independent ROI detection in both channels
- **Consolidated .npz archives:** Registration and detection arrays stored in single .npz files
- **ROIStatistics dataclass:** Replaces dictionary-based stat.npy with typed, pickle-free structure
- **ExtractionData dataclass:** Stores fluorescence, spikes, classification with two-channel support
- **Pickle disabled:** All np.save/np.load/np.savez use allow_pickle=False
- **YAML-safe types:** All tuple fields changed to list for YAML serialization compatibility
