# Data Architecture Reference

Comprehensive documentation of input data formats, session organization, and output structures for the sl-suite2p
neural imaging processing pipeline. Use this reference when debugging processing errors, understanding data
requirements, or answering questions about the pipeline's input/output formats.

---

## Input Data Overview

The sl-suite2p pipeline processes 2-photon calcium imaging data acquired using the Sun Lab's 2P-RAM Mesoscope system.
Input data is stored in compressed TIFF format with associated metadata files.

### Supported Session Types

Only mesoscope experiment sessions contain processable neural imaging data:

| Session Type           | Processable | Description                                                      |
|------------------------|-------------|------------------------------------------------------------------|
| `mesoscope experiment` | Yes         | Full VR experiment with brain activity recording                 |
| `lick training`        | No          | No mesoscope data acquired                                       |
| `run training`         | No          | No mesoscope data acquired                                       |
| `window checking`      | No          | No mesoscope data acquired                                       |

---

## Session Directory Structure

Sun Lab sessions follow a standardized directory structure:

```
{session_root}/
├── raw_data/
│   ├── session_data.yaml           # Session metadata (required)
│   ├── mesoscope_data/             # Raw imaging data
│   │   ├── frame_0001.tif          # Compressed TIFF frames
│   │   ├── frame_0002.tif
│   │   ├── ...
│   │   ├── suite2p_parameters.json # ScanImage acquisition parameters
│   │   └── frame_variant_metadata.npz  # Per-frame metadata
│   ├── behavior_data/              # Behavior log files
│   └── tracking_data/              # Processing tracker files
└── processed_data/
    ├── mesoscope_data/             # suite2p output (created by pipeline)
    │   └── suite2p/
    │       ├── plane0/
    │       ├── plane1/
    │       └── combined/
    └── multiday/                   # Multi-day output (created by pipeline)
        └── {dataset_name}/
```

### Key Files

| File                         | Location                   | Purpose                                    |
|------------------------------|----------------------------|--------------------------------------------|
| `session_data.yaml`          | `raw_data/`                | Session metadata and configuration         |
| `suite2p_parameters.json`    | `raw_data/mesoscope_data/` | ScanImage acquisition parameters           |
| `frame_variant_metadata.npz` | `raw_data/mesoscope_data/` | Per-frame timestamps and numbers           |
| `*.tif`                      | `raw_data/mesoscope_data/` | Compressed TIFF imaging frames             |

---

## TIFF Format Specification

### Compression

All TIFF files use LERC (Limited Error Raster Compression) with lossless settings:

| Parameter          | Value    | Description                                |
|--------------------|----------|--------------------------------------------|
| Compression        | LERC     | Limited Error Raster Compression           |
| Max Z Error        | 0.0      | Lossless (no pixel value loss)             |
| Bits Per Sample    | 16       | 16-bit unsigned integer pixel values       |
| Samples Per Pixel  | 1        | Grayscale (single channel)                 |

### Frame Organization

Each TIFF file contains multiple frames (pages) organized by imaging plane:

```
frame_0001.tif
├── Page 0: Plane 0, Volume 0
├── Page 1: Plane 1, Volume 0
├── Page 2: Plane 2, Volume 0
├── Page 3: Plane 3, Volume 0
├── Page 4: Plane 0, Volume 1
├── Page 5: Plane 1, Volume 1
└── ...
```

The number of planes is determined from `suite2p_parameters.json`.

---

## Metadata Files

### suite2p_parameters.json

Contains ScanImage acquisition parameters extracted during recording:

```json
{
    "frame_rate": 30.0,
    "plane_number": 4,
    "pixel_resolution_um": 1.5,
    "roi_group": {
        "rois": [
            {
                "center_xy": [256.0, 256.0],
                "size_xy": [512.0, 512.0],
                "rotation_degrees": 0.0
            }
        ]
    },
    "scan_zoom_factor": 1.0,
    "bidirectional": true
}
```

**Key Parameters:**

| Parameter             | Type  | Description                                     |
|-----------------------|-------|-------------------------------------------------|
| `frame_rate`          | float | Volumetric acquisition rate in Hz               |
| `plane_number`        | int   | Number of imaging planes per volume             |
| `pixel_resolution_um` | float | Spatial resolution in micrometers per pixel     |
| `roi_group.rois`      | list  | ROI definitions with center, size, and rotation |
| `bidirectional`       | bool  | Whether bidirectional scanning was used         |

### frame_variant_metadata.npz

Contains per-frame metadata that varies during acquisition:

```python
{
    "frameTimestamps_sec": np.ndarray,  # Frame acquisition timestamps
    "frameNumbers": np.ndarray,         # Sequential frame numbers
    "volumeNumbers": np.ndarray         # Volume indices (optional)
}
```

**Arrays:**

| Array                 | Type    | Shape        | Description                        |
|-----------------------|---------|--------------|------------------------------------|
| `frameTimestamps_sec` | float64 | (n_frames,)  | Timestamp of each frame in seconds |
| `frameNumbers`        | int64   | (n_frames,)  | 1-indexed frame number             |
| `volumeNumbers`       | int64   | (n_volumes,) | 1-indexed volume number (optional) |

---

## Single-Day Pipeline Outputs

### Per-Plane Outputs (suite2p/planeN/)

| File         | Type           | Shape              | Description                                        |
|--------------|----------------|--------------------|----------------------------------------------------|
| `data.bin`   | int16          | (n_frames, Ly, Lx) | Motion-corrected binary data                       |
| `ops.npy`    | dict           | -                  | Processing parameters and results                  |
| `stat.npy`   | list[dict]     | (n_rois,)          | ROI statistics (position, shape, pixels)           |
| `F.npy`      | float32        | (n_rois, n_frames) | Fluorescence traces                                |
| `Fneu.npy`   | float32        | (n_rois, n_frames) | Neuropil fluorescence traces                       |
| `spks.npy`   | float32        | (n_rois, n_frames) | Deconvolved spike estimates                        |
| `iscell.npy` | float32        | (n_rois, 2)        | Cell classification (is_cell, probability)         |

### Combined Outputs (suite2p/combined/)

Same file structure as per-plane, but with data merged across all planes:

- ROI indices offset to create unique identifiers
- Fluorescence traces concatenated across planes
- Statistics include plane index for each ROI

### ops.npy Contents

The ops dictionary contains both input parameters and processing results:

**Input Parameters:**
```python
{
    "data_path": str,       # Path to raw TIFF files
    "save_path": str,       # Path to save outputs
    "nplanes": int,         # Number of imaging planes
    "nchannels": int,       # Number of channels (typically 1)
    "fs": float,            # Sampling rate (Hz)
    "tau": float,           # Indicator decay time constant
    "diameter": int,        # Expected cell diameter in pixels
    ...
}
```

**Processing Results:**
```python
{
    "meanImg": np.ndarray,  # Mean image of plane
    "max_proj": np.ndarray, # Maximum projection
    "Vcorr": np.ndarray,    # Correlation image for detection
    "xrange": list,         # Valid x-range after registration
    "yrange": list,         # Valid y-range after registration
    "yoff": np.ndarray,     # Y registration offsets per frame
    "xoff": np.ndarray,     # X registration offsets per frame
    ...
}
```

---

## Multi-Day Pipeline Outputs

### Output Structure (multiday/{dataset_name}/)

| File                                | Type    | Shape               | Description                       |
|-------------------------------------|---------|---------------------|-----------------------------------|
| `ops.npy`                           | dict    | -                   | Multi-day processing parameters   |
| `multi_day_ss2p_configuration.yaml` | YAML    | -                   | Configuration snapshot            |
| `multiday_tracker.json`             | JSON    | -                   | Processing tracker (main session) |
| `template_cell_masks.npy`           | bool    | (n_cells, Ly, Lx)   | Tracked cell mask templates       |
| `F.npy`                             | float32 | (n_cells, n_frames) | Extracted fluorescence            |
| `Fneu.npy`                          | float32 | (n_cells, n_frames) | Extracted neuropil                |
| `spks.npy`                          | float32 | (n_cells, n_frames) | Deconvolved spikes                |

### Multi-Day ops.npy Contents

```python
{
    "dataset_name": str,           # Unique identifier for this dataset
    "session_ids": list[str],      # Ordered list of session identifiers
    "session_directories": list[str],  # Paths to session directories
    "main_session_idx": int,       # Index of main (first) session
    "registration_params": dict,   # Cross-session registration settings
    "clustering_params": dict,     # Cell clustering settings
    "extraction_params": dict,     # Fluorescence extraction settings
    ...
}
```

### multiday_tracker.json

Tracks processing status for all multi-day jobs:

```json
{
    "jobs": {
        "job_id_1": {
            "status": "succeeded",
            "started_at": "2024-01-15T10:30:00",
            "completed_at": "2024-01-15T11:45:00"
        },
        "job_id_2": {
            "status": "running",
            "started_at": "2024-01-15T11:46:00"
        }
    }
}
```

---

## Common Error Patterns

### Input Data Errors

| Error Pattern                                | Likely Cause                        | Resolution                             |
|----------------------------------------------|-------------------------------------|----------------------------------------|
| `FileNotFoundError: session_data.yaml`       | Invalid session directory structure | Verify session path is correct         |
| `FileNotFoundError: suite2p_parameters.json` | Missing acquisition parameters      | Re-export from ScanImage               |
| `KeyError: 'frame_rate'`                     | Malformed parameters file           | Check file integrity                   |
| `ValueError: TIFF pages`                     | Corrupted or truncated TIFF file    | Re-transfer imaging data               |
| `MemoryError`                                | Insufficient RAM for dataset size   | Reduce workers or process fewer planes |

### Processing Errors

| Error Pattern                               | Likely Cause                               | Resolution                                |
|---------------------------------------------|--------------------------------------------|-------------------------------------------|
| `No ROIs detected`                          | Poor signal or incorrect diameter          | Adjust detection parameters               |
| `Registration failed`                       | Excessive motion or low signal             | Check raw data quality                    |
| `Plane count mismatch`                      | Inconsistent ops.npy after binarize        | Re-run binarization                       |

### Multi-Day Errors

| Error Pattern                          | Likely Cause                           | Resolution                           |
|----------------------------------------|----------------------------------------|--------------------------------------|
| `No suite2p output found`              | Single-day processing incomplete       | Complete single-day pipeline first   |
| `Session IDs mismatch`                 | Configuration doesn't match sessions   | Verify session_directories in config |
| `Registration failed between sessions` | Too much drift between days            | Check FOV alignment                  |
| `No trackable cells found`             | Insufficient overlap in detected cells | Adjust clustering threshold          |

---

## Debugging Workflow

When processing errors occur:

1. **Identify the failing phase** from the status output (binarize, process, or combine)
2. **Check input files exist** in the session's `raw_data/mesoscope_data/` directory
3. **Verify session metadata** by loading `session_data.yaml`
4. **Check acquisition parameters** in `suite2p_parameters.json`
5. **Review frame metadata** in `frame_variant_metadata.npz` for timestamp gaps
6. **Examine partial outputs** in `processed_data/mesoscope_data/suite2p/`

### Verifying Prerequisites

Before single-day processing:
```
- [ ] session_data.yaml exists and is valid
- [ ] suite2p_parameters.json contains required fields
- [ ] TIFF files exist in mesoscope_data/
- [ ] frame_variant_metadata.npz has correct frame count
```

Before multi-day processing:
```
- [ ] All sessions have single-day processing complete
- [ ] suite2p/combined/ exists for each session
- [ ] Combined ops.npy has valid nplanes and cell counts
- [ ] Sessions are from the same animal and imaging region
```
