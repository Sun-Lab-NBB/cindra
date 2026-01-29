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

### Target Architecture (RuntimeData)

- `SingleDayConfiguration` - user config (YAML, human-readable)
- `SingleDayRuntimeData` - pipeline outputs (YAML + .npy for arrays)
- `RuntimeContext` - combines both for function calls
- Clear separation: what user sets vs what pipeline computes
- No backward compatibility - clean break from legacy format

---

## Phase 1: Configuration Dataclass Refactoring (COMPLETED)

All configuration and runtime dataclasses have been updated with descriptive field names.

**Files modified:**
- `src/sl_suite2p/configuration/single_day.py` - Complete field renames, new enums, Path types
- `src/sl_suite2p/configuration/multi_day.py` - Imports shared classes from single_day.py
- `src/sl_suite2p/configuration/__init__.py` - Updated exports

**Key changes:**
- Added `InputFormat` and `BaselineMethod` StrEnums for type-safe configuration
- Changed all path fields from `str` to `Path | None` with `__post_init__` conversion
- Added `allow_pickle=False` to all NumPy save/load operations
- Added channel 2 detection and timing fields for independent dual-channel processing
- Removed deprecated sections (Channel2, Output)
- Imported shared SignalExtraction and SpikeDeconvolution in multi_day.py

See `REFACTOR_CHANGES.md` for the complete field rename mappings and deprecation decisions.

---

## Phase 2: Update IO Module (PENDING)

**Files:**

- `src/sl_suite2p/io/utils.py`
- `src/sl_suite2p/io/tiff.py`

1. Create `initialize_plane_runtime()` replacing `initialize_plane_ops()`
2. Update `tiff_to_binary()` and `mesoscan_to_binary()` to use RuntimeContext
3. Save images as separate `.npy` files, paths in runtime data

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

## Phase 6: Update Combined Folder Generation (PENDING)

**File:** `src/sl_suite2p/io/save.py`

Update `combined()` function:

- Read from `runtime_data.yaml` instead of `ops.npy`
- Load images from `.npy` paths
- Save combined `runtime_data.yaml` + image `.npy` files
- Keep saving `stat.npy`, `F.npy`, etc. as before

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
├── plane0/
│   ├── runtime_data.yaml
│   ├── reference_image.npy
│   ├── mean_image.npy
│   ├── enhanced_mean_image.npy
│   ├── maximum_projection.npy
│   ├── correlation_map.npy
│   ├── rigid_y_offsets.npy, rigid_x_offsets.npy, rigid_correlations.npy
│   ├── nonrigid_y_offsets.npy, nonrigid_x_offsets.npy, nonrigid_correlations.npy
│   ├── data.bin
│   └── F.npy, Fneu.npy, stat.npy, ...
└── combined/
    ├── runtime_data.yaml
    ├── mean_image.npy
    ├── enhanced_mean_image.npy
    ├── maximum_projection.npy
    ├── correlation_map.npy
    └── F.npy, Fneu.npy, stat.npy, ...
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
- **Registration offsets:** Persist all offsets as .npy files to enable re-processing
- **Pickle disabled:** All np.save/np.load use allow_pickle=False
