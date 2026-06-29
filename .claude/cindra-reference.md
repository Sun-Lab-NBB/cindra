### Core components

| Component                         | File                                            | Purpose                                                 |
|-----------------------------------|-------------------------------------------------|---------------------------------------------------------|
| `SingleRecordingConfiguration`    | `dataclasses/single_recording_configuration.py` | User-facing config with nested dataclasses              |
| `MultiRecordingConfiguration`     | `dataclasses/multi_recording_configuration.py`  | Multi-recording pipeline config                         |
| `AcquisitionParameters`           | `dataclasses/single_recording_configuration.py` | Per-recording acquisition metadata                      |
| `RuntimeContext`                  | `dataclasses/runtime_contexts.py`               | Single-recording config + acquisition + runtime data    |
| `MultiRecordingRuntimeContext`    | `dataclasses/runtime_contexts.py`               | Multi-recording config + runtime data                   |
| `SingleRecordingRuntimeData`      | `dataclasses/single_recording_data.py`          | IOData, RegistrationData, DetectionData, ExtractionData |
| `MultiRecordingRuntimeData`       | `dataclasses/multi_recording_data.py`           | Multi-recording IO, registration, tracking, timing data |
| `run_single_recording_pipeline`   | `pipelines/pipeline.py`                         | Execute single-recording three-phase workflow           |
| `run_multi_recording_pipeline`    | `pipelines/pipeline.py`                         | Execute multi-recording two-phase workflow              |
| `register_plane`                  | `registration/register.py`                      | Per-plane motion correction (rigid + optional nonrigid) |
| `DiffeomorphicDemonsRegistration` | `registration/diffeomorphic.py`                 | Cross-day diffeomorphic alignment algorithm             |
| `Deformation`                     | `registration/deformation.py`                   | Deformation field application and inversion             |
| `detect_plane_rois`               | `detection/detect.py`                           | ROI detection via sparse detection with PCA denoising   |
| `track_rois_across_recordings`    | `detection/tracking.py`                         | Multi-recording ROI tracking via spatial clustering     |
| `compute_roi_statistics`          | `detection/roi_statistics.py`                   | ROI property computation (skewness, compactness, etc.)  |
| `extract_traces`                  | `extraction/extract.py`                         | Fluorescence extraction and neuropil subtraction        |
| `apply_oasis_deconvolution`       | `extraction/deconvolve.py`                      | OASIS spike deconvolution                               |
| `create_masks`                    | `extraction/masks.py`                           | ROI mask creation with lambda weight computation        |
| `Classifier`                      | `classification/classify.py`                    | Cell vs. artifact classification                        |
| `BinaryFile`                      | `io/binary.py`                                  | Memory-mapped binary file access for imaging data       |
| `convert_tiffs_to_binary`         | `io/tiff.py`                                    | TIFF to internal binary format conversion               |
| `combine_planes`                  | `io/combine.py`                                 | Multi-plane result combination                          |
| `run_roi_viewer`                  | `gui/app.py`                                    | Single-recording ROI inspector GUI                      |
| `run_tracking_viewer`             | `gui/app.py`                                    | Multi-recording tracking quality GUI                    |
| `run_registration_viewer`         | `gui/app.py`                                    | Registration quality viewer (binary + PC viewer)        |

### CLI entry points

| Command      | Entry point                           | Purpose                                                  |
|--------------|---------------------------------------|----------------------------------------------------------|
| `cindra`     | `cindra.interface.cli:cindra_cli`     | Main CLI for configuration, pipeline execution, and MCP  |
| `cindra-gui` | `cindra.interface.gui_cli:cindra_gui` | GUI launcher (separate to avoid Qt during headless runs) |

**`cindra` commands:**

| Command            | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| `cindra configure` | Generate default config files for single or multi-recording pipeline |
| `cindra run`       | Execute pipeline with CLI overrides for config parameters            |
| `cindra mcp`       | Start MCP server (stdio, sse, or streamable-http transport)          |

**`cindra-gui` commands:**

| Command                   | Description                                                    |
|---------------------------|----------------------------------------------------------------|
| `cindra-gui roi`          | Launch ROI viewer (single or multi-recording via dataset flag) |
| `cindra-gui registration` | Launch registration quality viewer (binary + PC viewer combo)  |
| `cindra-gui tracking`     | Launch multi-recording tracking quality viewer                 |
| `cindra-gui mcp`          | Start GUI MCP server for viewer lifecycle management           |

### Dependencies

| Library                    | Purpose                                                       |
|----------------------------|---------------------------------------------------------------|
| `numpy`                    | Array operations, memory mapping, data storage                |
| `numba`                    | JIT compilation for registration, detection, extraction       |
| `scipy`                    | Signal processing, spatial algorithms, sparse matrices        |
| `scikit-learn`             | PCA denoising, clustering for ROI detection                   |
| `natsort`                  | Semantic file path sorting (1, 2, 10 vs 1, 10, 2)             |
| `tifffile`                 | TIFF file loading and metadata extraction                     |
| `imagecodecs`              | Image codec support for TIFF decompression                    |
| `matplotlib`               | Visualization support for GUI viewers                         |
| `pyside6`                  | Qt6 GUI framework for interactive viewers                     |
| `pyqtgraph`                | High-performance plotting for GUI image display               |
| `click`                    | CLI framework for command-line interfaces                     |
| `mcp`                      | FastMCP server for agentic AI tool integration                |
| `httpx`                    | HTTP client used by the MCP transport layer                   |
| `ataraxis-time`            | PrecisionTimer for pipeline step timing                       |
| `ataraxis-base-utilities`  | Console for unified message handling and error reporting      |
| `ataraxis-data-structures` | YamlConfig, ProcessingTracker, and data logging utilities     |
| `importlib_metadata`       | Runtime version introspection for the cindra package          |
| `tbb4py`                   | Intel TBB threading layer for Numba parallelization (non-Mac) |
| `intel-cmplr-lib-rt`       | Intel compiler runtime paired with `tbb4py` (non-Mac)         |

### Workflow guidance

**Modifying pipeline orchestration:**

1. Review `src/cindra/pipelines/pipeline.py` for job orchestration and ProcessingTracker integration
2. Review `src/cindra/pipelines/single_recording.py` for the three-phase single-recording workflow
3. Review `src/cindra/pipelines/multi_recording.py` for the two-phase multi-recording workflow
4. Maintain the job naming convention (`SingleRecordingJobNames`, `MultiRecordingJobNames`) for tracker consistency

**Modifying registration:**

1. Review `src/cindra/registration/register.py` for per-plane motion correction entry point
2. Understand the two-step registration refinement when enabled
3. Rigid registration uses phase correlation (`rigid.py`); nonrigid uses block-based deformation (`nonrigid.py`)
4. Cross-recording registration uses diffeomorphic demons (`diffeomorphic.py`) with multiscale pyramid (`pyramid.py`)

**Modifying detection:**

1. Review `src/cindra/detection/detect.py` for the sparse detection entry point
2. Understand the PCA denoising step and temporal binning strategy
3. ROI extension logic is in `detect_rois.py`; statistics computation in `roi_statistics.py`
4. Multi-recording tracking via spatial clustering is in `tracking.py`

**Modifying extraction:**

1. Review `src/cindra/extraction/extract.py` for the polymorphic dispatch pattern
2. Numba JIT functions use `@njit(cache=True, parallel=True)` with `prange` for frame parallelization
3. Mask creation and lambda weight computation is in `masks.py`
4. OASIS deconvolution and delta fluorescence computation is in `deconvolve.py`

**Modifying GUI viewers:**

1. Review `src/cindra/gui/app.py` for viewer entry points
2. Viewers use PySide6 + PyQtGraph with custom widgets in `widgets.py`
3. State management via `viewer_context.py` and `viewer_state.py`
4. The GUI CLI (`gui_cli.py`) is separate from the main CLI to avoid loading Qt during headless execution

**Adding or modifying MCP tools:**

1. Review the relevant tool module in `src/cindra/interface/` (acquisition, configuration, processing, or results)
2. Tools register via `@mcp.tool()` decorator on the shared `mcp` instance from `mcp_instance.py`
3. Batch processing tools use background manager threads with per-job worker threads
4. Return formatted strings for user-facing output; use JSON response mode

**Adding or modifying CLI commands:**

1. Review `src/cindra/interface/cli.py` for the main CLI Click group structure
2. Review `src/cindra/interface/gui_cli.py` for the GUI CLI structure
3. Follow existing patterns for Click option decorators and error handling
4. CLI writes configuration overrides to the config file before pipeline execution

**Important considerations:**

- The `console` is enabled in `src/cindra/__init__.py` — do not re-enable elsewhere
- The Numba threading layer is configured in `__init__.py` (TBB on non-Mac, OpenMP on macOS) after importing
  `numba.config` and before importing modules that compile `@njit` functions — do not move this
- The `# type: ignore[import-untyped]` comments on the scikit-learn, threadpoolctl, PyQtGraph, and yaml imports are
  expected (Numba is excluded via the `pyproject.toml` mypy override; tifffile imports carry no such comment)
- The `# pragma: no cover` annotations on `@njit` function bodies are intentional
- Use `console.error()` from ataraxis-base-utilities for all error handling (no bare `raise`)
