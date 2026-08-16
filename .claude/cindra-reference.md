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
| `RecordingArrays`                 | `layout.py`                                     | On-disk contract: names, markers, and path resolvers    |
| `resolve_recording_planes`        | `io/inventory.py`                               | Read-only recording and dataset on-disk inventory       |
| `resolve_single_recording_job_universe` | `orchestration/discovery.py`              | Declared job set and the subset whose inputs exist      |
| `estimate_single_recording_job_memory_mb` | `orchestration/footprints.py`           | Per-stage memory models and the two job estimators      |
| `size_single_recording_job`       | `orchestration/footprints.py`                   | Pairs a job's declared cores with its memory estimate   |
| `execute_single_recording_job`    | `orchestration/worker.py`                       | Per-job entry point against a caller-owned tracker      |
| `prime_recording`                 | `orchestration/worker.py`                       | Writes the shared bootstrap and reports the inventory   |
| `run_single_recording_pipeline`   | `orchestration/pipeline.py`                     | Execute single-recording four-phase workflow            |
| `run_multi_recording_pipeline`    | `orchestration/pipeline.py`                     | Execute multi-recording two-phase workflow              |
| `start_execution_session`         | `orchestration/execution.py`                    | Batch engine: admission, process-pool dispatch, budgets |
| `register_recording_plane`        | `pipelines/single_recording.py`                 | Per-plane registration stage entry point (phase 2)      |
| `register_plane`                  | `registration/register.py`                      | Per-plane motion correction (rigid + optional nonrigid) |
| `resolve_stage_workers`           | `orchestration/allocation.py`                   | Measured per-stage worker defaults and worker resolver  |
| `SINGLE_RECORDING_PHASES`         | `orchestration/jobs.py`                         | Phase model: job universe and prerequisite graph        |
| `resolve_openmp_runtime`          | `orchestration/openmp.py`                       | macOS OpenMP runtime discovery, linking, verification   |
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
| `cindra omp`       | Link the macOS OpenMP runtime Numba loads, erroring on other systems |

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
| `mcp`                      | MCPServer host for agentic AI tool integration                |
| `psutil`                   | Available host memory reads that size the processing class    |
| `ataraxis-time`            | PrecisionTimer for pipeline step timing                       |
| `ataraxis-base-utilities`  | Console for unified message handling and error reporting      |
| `ataraxis-data-structures` | YamlConfig, ProcessingTracker, and data logging utilities     |
| `threadpoolctl`            | BLAS thread confinement around scikit-learn and LAPACK fits   |
| `pyyaml`                   | YAML serialization for configuration and tracker files        |
| `tbb4py`                   | Intel TBB threading layer for Numba parallelization (non-Mac) |
| `intel-cmplr-lib-rt`       | SVML runtime held for Numba's vectorization path (non-Mac)    |

### Workflow guidance

**Modifying pipeline orchestration:**

1. Review `src/cindra/orchestration/worker.py` for per-job execution and ProcessingTracker integration, and
   `src/cindra/orchestration/pipeline.py` for the two sequential entry points
2. Review `src/cindra/orchestration/execution.py` for the batch engine that admits and dispatches queued jobs, whose
   two-pass dispatcher honors every reservation before releasing it over the capacity the first pass left unused
3. Review `src/cindra/pipelines/single_recording.py` for the four-phase single-recording stage entry points
4. Review `src/cindra/pipelines/multi_recording.py` for the two-phase multi-recording stage entry points
5. Job universes and prerequisite edges derive from the phase model in `src/cindra/orchestration/jobs.py`
   (`SINGLE_RECORDING_PHASES`, `MULTI_RECORDING_PHASES`). Add, remove, or reorder a phase there rather than at each call
   site, and the pipelines, the execution engine, and the MCP layer follow automatically
6. Maintain the job naming convention (`SingleRecordingJobNames`, `MultiRecordingJobNames`) for tracker consistency
7. Keep the dependency chain one-way. `jobs.py` imports `cindra.layout` alone, `allocation.py` and `discovery.py`
   import `jobs`, `footprints.py` imports `jobs` and `allocation`, `worker.py` imports `jobs` and `allocation`,
   `pipeline.py` imports `worker`, `execution.py` imports `pipeline`, `jobs`, and `allocation`, and no orchestration
   module imports `interface`.
   `openmp.py` carries no module-level side effect and its check runs only inside the two sequential entry points,
   so importing the package writes nothing and a console message never precedes the stdio MCP server's JSON-RPC
   stream

**Modifying registration:**

1. Review `src/cindra/registration/register.py` for per-plane motion correction entry point
2. Understand the two-step registration refinement when enabled
3. Rigid registration uses phase correlation (`rigid.py`), and nonrigid uses block-based deformation (`nonrigid.py`)
4. Cross-recording registration uses diffeomorphic demons (`diffeomorphic.py`) with multiscale pyramid (`pyramid.py`)
5. Registration rewrites its input binary in place under a `<binary>.registering` marker, the parallel of the
   `<binary>.binarizing` marker binarization writes while it fills that binary. Keep the create and clear pair around
   any new write loop, and confine BLAS fits with `threadpool_limits` as `metrics.py` does

**Modifying detection:**

1. Review `src/cindra/detection/detect.py` for the sparse detection entry point
2. Understand the PCA denoising step and temporal binning strategy
3. ROI extension logic is in `detect_rois.py`, and statistics computation is in `roi_statistics.py`
4. Multi-recording tracking via spatial clustering is in `tracking.py`

**Modifying extraction:**

1. Review `src/cindra/extraction/extract.py` for the polymorphic dispatch pattern
2. Numba JIT functions use `@njit(cache=True, parallel=True)` with `prange` for ROI parallelization
3. Mask creation and lambda weight computation are in `masks.py`
4. OASIS deconvolution and delta fluorescence computation are in `deconvolve.py`

**Modifying GUI viewers:**

1. Review `src/cindra/gui/app.py` for viewer entry points
2. Viewers use PySide6 + PyQtGraph with custom widgets in `widgets.py`
3. State management via `viewer_context.py` and `viewer_state.py`
4. The GUI CLI (`gui_cli.py`) is separate from the main CLI to avoid loading Qt during headless execution

**Adding or modifying MCP tools:**

1. Review the relevant tool module in `src/cindra/interface/` (acquisition, configuration, processing, or results)
2. Tools register via `@mcp.tool()` decorator on the shared `mcp` instance from `mcp_instance.py`
3. Batch processing tools call into `cindra.orchestration`, whose manager thread dispatches each job into a
   worker process pinned by `limit_worker_threads` and `initialize_worker_threads`
4. Return JSON-serializable dictionaries. `run_server` and `run_gui_server` enable JSON responses only when they
   start the streamable-http transport

**Adding or modifying CLI commands:**

1. Review `src/cindra/interface/cli.py` for the main CLI Click group structure
2. Review `src/cindra/interface/gui_cli.py` for the GUI CLI structure
3. Follow existing patterns for Click option decorators and error handling
4. CLI writes configuration overrides to the config file before pipeline execution

**Important considerations:**

- The `console` is enabled in `src/cindra/__init__.py`. Do not re-enable it elsewhere
- The Numba threading layer is configured in `__init__.py` (TBB on non-Mac, OpenMP on macOS) after importing
  `numba.config` and before importing modules that compile `@njit` functions. Do not move this. macOS runs OpenMP
  because the Numba macOS wheel ships no tbbpool extension, so the TBB layer is unavailable there whatever runtime is
  installed
- The macOS OpenMP layer loads `libomp.dylib` from the dynamic loader's default search path, and
  `cindra.orchestration.openmp` owns the discovery and linking that put it there. Both pipeline entry points call
  `verify_openmp_runtime()` before they dispatch a stage, so a macOS host carrying no loadable runtime aborts having
  done no work, while `cindra omp` reports the runtimes found on the host and `cindra omp --yes` links one. Numba
  raises its threading-layer error at the first parallelized call rather than at import, which is what the check
  replaces. Keep the check off the import path, because a message written there reaches the stdio MCP server's
  JSON-RPC stream before any CLI code can silence the console
- The `# type: ignore[import-untyped]` comments on the scikit-learn, threadpoolctl, and PyQtGraph imports are
  expected (Numba is excluded via the `pyproject.toml` mypy override, and the tifffile and yaml imports carry no such
  comment, because both ship types)
- The `# pragma: no cover` annotations on `@njit` function bodies are intentional
- The multiscale diffeomorphic registration crosses the boundary between original-image pixels and the working
  resolution of a pyramid level in three places, and it converts units at two of them. `ScaleSpacePyramid` scales
  every smoothing sigma by the level's entry in `_level_downsample_factors`. `_scale_grid_sampling` converts the
  knot spacing into working-resolution pixels, while `_regularize_deformation` keeps the injectivity factor on the
  original-pixel spacing, because that factor divides by `scale`, which is an original-pixel quantity.
  `Deformation.resize_field` leaves displacement magnitudes unscaled, which discounts each coarse level by its
  resolution ratio and weights it below the finer levels that follow it. That third choice is deliberate, and the
  method's `Notes` block records its reasoning. Do not report it as a unit-conversion defect, as an inconsistency
  with the other two conversions, or as a regression, and do not add a scaling variant of `resize_field` unless the
  user asks for one
- Binarization and registration both write frames into a plane binary, each guarding its own write with its own marker,
  `<binary>.binarizing` and `<binary>.registering`. `cindra.io` exports a create, clear, and path helper per phase plus
  `resolve_active_binary_marker`, which reports whichever marker sits beside a binary and is what every reader calls.
  `register_plane` refuses to run while either marker exists, and `binarize_recording` refuses a marked binary, a
  binary whose size disagrees with its plane's recorded frame geometry, and a two-channel plane holding no second
  channel binary. Preserve this protocol when modifying either stage, and keep `repeat_binarization` named as the
  remedy every refusal states, because a caller-requested rebuild is the recovery path
- Binarization consumes whole plane and channel interleave cycles, discarding the frames of an incomplete final cycle
  and rejecting a recording that holds fewer frames than one whole cycle. Keep both in `resolve_tiff_conversion_plan`,
  because the plan resolves before the conversion touches any binary or deletes any result the recording already holds
- `_clear_downstream_data` sweeps every plane directory the output root holds rather than the contiguous range the
  declared plane count spans. A directory that count no longer covers therefore loses the results measured from the
  frames the conversion replaces, while keeping the binary the conversion does not rewrite. Keep the sweep reading the
  directories off disk, because the declared count comes from a user-editable file
- Use `console.error()` from ataraxis-base-utilities for all error handling (no bare `raise`)
