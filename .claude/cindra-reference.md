### Architecture

- **Single-recording pipeline**: Four-phase workflow (binarize, register, process, combine). Phase 1 converts TIFFs to
  internal binary format and initializes RuntimeContext per plane. Phase 2 runs per-plane motion correction and
  registration-quality metrics (parallelizable across planes). Phase 3 runs per-plane detection, classification, and
  extraction, and requires the plane to be registered (parallelizable across planes). Phase 4 merges plane-specific
  results into a unified `combined_metadata.npz` dataset, trimming the combined traces to the shortest contributing
  plane and recording `frame_count` and `plane_frame_counts` alongside the geometry. That metadata file doubles as the
  pipeline-completion marker, so it is written after its payload arrays and published through `atomic_write`. Phase 1
  rejects a data directory whose TIFF files do not all hold frames of the same shape, naming
  `file_io.ignored_file_names` as the exclusion mechanism. It also consumes whole plane and channel interleave cycles,
  so every plane and channel of a recording holds the same frame count. The frames of an incomplete final cycle are
  discarded, and a recording short of one whole cycle is rejected. Keep both in `resolve_tiff_conversion_plan`, because
  the plan resolves before the conversion touches any binary or deletes any result the recording already holds. Phases 2
  and 3 carry a `plane_{index}` tracker specifier.
- **Multi-recording pipeline**: Two-phase workflow (discover, extract). Phase 1 selects ROIs from each recording,
  performs diffeomorphic demons registration to a common space, clusters ROIs across recordings via spatial overlap, and
  projects template masks back to individual recordings. Phase 2 extracts fluorescence traces and applies OASIS
  deconvolution for tracked ROI templates (parallelizable across recordings).
- **Self-driven orchestration**: `cindra.orchestration` owns the whole scheduling surface across nine modules that
  form a one-way dependency chain. `gpu.py` and `openmp.py` import nothing from the package, and `jobs.py` is the leaf
  above `cindra.layout`. It holds the job name enumerations and the phase model (`SINGLE_RECORDING_PHASES`,
  `MULTI_RECORDING_PHASES`, `PipelinePhase`, `PrerequisiteScope`). It also holds the resolvers that expand that model
  into a recording's job universe (`resolve_single_recording_jobs`, `resolve_multi_recording_jobs`,
  `resolve_pipeline_jobs`), the prerequisite graph (`resolve_single_recording_prerequisites`,
  `resolve_multi_recording_prerequisites`, `resolve_prerequisite_job_ids`, `validate_job_prerequisites`), the phase
  expansion (`resolve_downstream_phases`, `order_phases_by_execution`), and the prerequisite messages.
  `generate_job_ids` derives the identifier that tracks each of those jobs, which is what the `job_id` parameter of both
  pipeline entry points names. The plane specifier, the tracker filenames, and every other on-disk name live one layer
  below in `cindra.layout`, which `jobs.py` reads. `allocation.py` adds the stage worker defaults, the
  resource-class model, and the host core and memory budgets. `footprints.py` adds the per-stage memory models, the two
  estimators that report what one job holds, and the two sizers that pair each estimate with its stage's declared cores
  and its device memory as a `JobSizing`. It reads `allocation.py` for those cores alone. A job is sized from the data
  that exists when the sizing happens, so a single-recording model reads the acquisition alone while a multi-recording
  model reads the completed single-recording output that it processes. `discovery.py` pairs the job model with the
  on-disk inventory to report both the jobs a recording declares and the subset whose inputs exist. `worker.py` holds
  the per-job entry points every scheduler dispatches, along with the two priming entry points that write the shared
  bootstrap. `execution.py` holds the batch engine: `PendingJob`, `JobExecutionState`, the admission scan, the two-pass
  dispatcher, and the manager thread. `pipeline.py` holds the two sequential entry points. `gpu.py` holds the CUDA
  device discovery, the runtime verification, and `ALL_DEVICES_REQUEST`, the sentinel naming every device the host
  exposes. `allocation.py` reads its device count, `pipeline.py` reads its verification, and `execution.py` reads both
  the sentinel and the discovery. `openmp.py` carries no module-level side effect and its check runs only inside those
  two entry points, so importing the package writes nothing and a console message never precedes the stdio MCP server's
  JSON-RPC stream. Nothing below `orchestration` imports it, and no module inside it imports `interface`, so the MCP
  layer is a thin argument-validation and JSON-shaping wrapper over calls into the package. This mirrors the
  orchestration package of `ataraxis-video-system` and `ataraxis-communication-interface`, and its concurrency model
  follows `sollertia-forgery`.
- **Tracker-driven job state**: The transitions of a job the pipeline runs belong to the tracker's `run_job()` context
  manager rather than to a hand-rolled `start_job`/`complete_job`/`fail_job` sequence. The engine's
  `_fail_dispatched_job`, `_fail_pending_jobs`, and the `_pipeline_worker` fallback are the exceptions, because each
  records a terminal outcome for a job whose worker or pool died without reaching one, leaving no block to wrap. A
  remote invocation recovers its job's name and specifier through `tracker.resolve_job(job_id, universe)` rather than
  rebuilding the identifier map itself. Both pipeline entry points validate `target_plane` and `target_recording`
  against the resolved universe before aligning the tracker, so an out-of-range request reports the argument the caller
  passed instead of a job identifier the caller never saw.
- **Context pattern**: `RuntimeContext` combines configuration, acquisition parameters, and runtime data into a single
  object passed through pipeline steps. `MultiRecordingRuntimeContext` follows the same pattern, but carries only
  configuration and runtime data.
- **Configuration-driven execution**: Pipelines read all processing parameters from YAML files (YamlConfig subclasses).
  The CLI writes overrides to the config file before execution rather than passing arguments. Worker counts are the
  exception: they are explicit API parameters resolved through `cindra.orchestration`, which keeps the configuration
  file immutable and safe to share between concurrently dispatched jobs. The CUDA device that a registration job uses is
  the same kind of exception. It is an argument threaded through the library, never a configuration field, and `None`
  means the host CPU. A job takes `device: int | None`, a session takes `gpu_devices: list[int] | None`, and the
  planner substitutes the device models under the `gpu_registration` flag. `registration.gpu_batch_size` stays in the
  configuration, because it shapes the data a device-backed pass stages rather than selecting where the pass runs.
- **Worker sentinel contract**: One convention governs every argument that resolves a worker or concurrency
  allocation. `None` accepts the measured default for that stage or resource class, `-1` (`ALL_CORES_REQUEST`)
  requests every available core, and a positive integer is used exactly. Any other non-positive value is rejected.
  This holds for `resolve_stage_workers`, the `cindra run` worker options, the pipeline entry points, and the
  `workers_per_job` and `max_parallel_jobs` arguments of the execute MCP tools. For `max_parallel_jobs`, `-1` lifts
  the derived cap so that only the job count bounds concurrency. A stage-internal parameter that receives an
  already-resolved count, such as the `workers` argument of the stage entry points or `pca_denoise`'s
  `parallel_workers`, takes a positive integer alone and rejects every other value. Keep the source, the skills, and
  the README stating this identically.
- **ProcessingTracker**: File-based YAML pipeline state tracking with FileLock for multi-process coordination. Manages
  job states (SCHEDULED, RUNNING, SUCCEEDED, FAILED) for resumable batch processing.
- **Subprocess GUI isolation**: GUI viewers launch as separate subprocesses with state file exchange via temporary
  files, avoiding Qt dependency loading during headless pipeline execution. The `cindra-gui` CLI entry point is separate
  from `cindra` for this reason.
- **MCP tool organization**: Tools are split across four modules (`acquisition_tools`, `configuration_tools`,
  `processing_tools`, `results_tools`) imported at module level to trigger `@mcp.tool()` registration. Every tool names
  a filesystem path by the thing that path holds, and the vocabulary is identical across parameter names, response keys,
  docstrings, and messages. `raw_data_path` and `raw_data_paths` name a recording's raw imaging path, which resolves to
  the directory holding its TIFF files and its `cindra_parameters.json` file. `output_root` and `output_roots` name the
  pipeline output root that parents the `cindra` directory, and `root_directory` names a tree searched for recordings.
  `configuration_path`, `tracker_path`, `output_path`, and `file_path` name one specific file. `output_path` is the
  configuration file `generate_config_file_tool` writes, and `file_path` is the normalized path it returns. Three names
  reach the caller only as response keys. `recording_root` is the session-level root `discover_recordings_tool` reports
  beside each candidate's `raw_data_path` or `output_root`, `cindra_path` is the `cindra` output directory the results
  tools report, and `dataset_output_path` is one recording's directory inside a multi-recording dataset tree. The
  configuration fields `file_io.data_path` and `file_io.output_path` and the YAML field
  `recording_io.recording_directories` keep their own names, because they are on-disk schema rather than tool surface.
  The "Adding or modifying MCP tools" workflow below holds a new tool and a new response key to this vocabulary. The
  configuration module also modifies a configuration. `set_config_values_tool` writes the dotted `section.parameter`
  paths `validate_config_file_tool` reports under `non_default_parameters`, resolving every entry before applying any,
  so a rejected entry leaves the file byte-identical. It must not run against a configuration whose jobs are executing,
  because the pipeline reads its configuration from disk at dispatch. Processing uses a prepare-then-execute model:
  preparation tools create execution manifests (trackers, per-recording configurations, job lists) without starting
  computation, and execution tools dispatch jobs with prerequisite validation, per-class resource allocation, and
  automatic phase sequencing. Four planning tools sit ahead of both halves. `get_pipeline_job_universe_tool` reports
  every job a configuration declares and which of them can run right now. `size_pipeline_jobs_tool` reports the cores,
  memory, and device memory each of those jobs holds, substituting the device models for the registration jobs when
  `gpu_registration` is set. `check_threading_runtime_tool` reports whether the host carries the numeric threading layer
  the platform selects, and `check_gpu_runtime_tool` reports the CUDA devices the host exposes, so an agent gates a
  batch on a flag rather than on parsing a per-job tracker failure. The dispatch half lives in `cindra.orchestration`,
  so the execute, monitor, and cancel tools hold only argument validation and response shaping. The prepare tools stay
  in the interface layer, because building a manifest is a user-facing operation over paths and configuration files
  rather than part of the scheduling model. Every job class carries a per-job worker count from `cindra.orchestration`,
  and the combination class holds the single core its serial merge needs. Concurrency follows three separate terms. The
  binarization class carries a hard ceiling, because it decodes at the storage's rate rather than the host's core count.
  A wider batch finishes the same work more slowly while holding cores other work could use, so spare capacity never
  lifts it. The device-backed registration class carries a hard ceiling of its own, which is the count of the devices
  the host exposes, and a session reports that ceiling clamped to the devices it holds. The registration and processing
  classes carry soft reservations, which hold capacity back for the stages that wait on no other job and are released
  once nothing else can use the room. Every other class derives its concurrency from the session CPU budget alone.
  Memory bounds admission rather than concurrency, because the memory one job holds follows the recording it processes
  rather than the class that owns it. A class is elastic where its `maximum_workers_per_job` ceiling stands strictly
  above its `workers_per_job` default, which covers registration, processing, discovery, and extraction. Binarization,
  combination, and the device-backed registration class carry no ceiling, so each takes its default whatever the host
  holds free. An elastic class widens its jobs at dispatch as the queue drains, and only in a session that accepted the
  class defaults, because an explicit `workers_per_job` reaches every job unchanged. The free cores divide among the
  elastic classes holding queued work before the share divides among the jobs, so a full queue resolves to the class
  default while a queue holding one job resolves toward the ceiling. A class carrying no ceiling takes its default
  whatever the host holds free.
- **Process-isolated jobs**: The batch engine dispatches every job into a `ProcessPoolExecutor` sized to the
  concurrency the per-class caps allow, so admission remains the only thing bounding how many jobs run. Isolation buys
  two things a thread pool cannot. A job's BLAS width belongs to its process, so concurrent jobs at different widths no
  longer overwrite each other. A detection job that exhausts memory takes down its own worker rather than the whole
  batch, and the engine records a terminal outcome for every job the resulting `BrokenProcessPool` strands. The manager
  itself stays a thread of the dispatching process, since it only polls trackers and moves queue entries.
  The pool requests the `spawn` start method through `_POOL_START_METHOD` rather than accepting the host default, which
  is `forkserver` on Linux. Spawn is the only method available on every supported platform, so a Linux session gets the
  process semantics a macOS or Windows session gets anyway. It is also the only sound method here, because a forked
  worker inherits the parent's already-sized numeric backends, which leaves the thread pin inert and hands every
  concurrent job a host-wide backend pool.

### Key patterns

- **Numba parallelization**: Functions use `@njit(cache=True, parallel=True)` with `prange` over each kernel's outermost
  independent axis, which is frames in registration and ROIs in extraction. A parallel kernel carries no eager
  signature. A signature compiles the kernel when its module is imported, which starts the threading layer before
  `verify_openmp_runtime()` runs and fails a host with no runtime at `import cindra` rather than at the stage that needs
  it.
- **Thread budget confinement**: Two ataraxis assets and one third-party context manager divide the work, and each
  covers a moment the others cannot. `limit_worker_threads` from `ataraxis-data-structures` encloses the batch
  engine's worker pool for the session's whole lifetime, so every worker process imports its numeric backends at a
  width of one instead of sizing each of them to the whole host. `initialize_worker_threads` runs as that pool's
  initializer, which reaches the backends that read their variable the first time they are asked to work rather than
  while they are imported. Neither touches `NUMBA_NUM_THREADS`, so a worker still latches the host's full ceiling and
  each stage raises to its own budget through `numba.set_num_threads`. `threadpool_limits` then confines the
  scikit-learn and LAPACK fits inside the running worker, which is the only one of the three that acts on an
  already-loaded library. `pipelines/single_recording.py`, `pipelines/multi_recording.py`, and
  `registration/register.py` limit to the job's `workers`, and `detection/denoise.py` limits to 1 because its own block
  pool already spends the budget. The BLAS width these set is a property of the process rather than of the thread that
  asked for it, which is why the engine gives each job its own process.
- **Binary write integrity**: Binarization fills a plane binary sized to its full frame count, and registration rewrites
  that binary in place. Each phase guards its own write with its own marker, `<binary>.binarizing` and
  `<binary>.registering`, whose suffixes match the `binarizing` and `registering` job statuses the interface reports.
  Both markers mean the same thing to the pipeline, so the names serve the user who finds one on disk. `io/binary.py`
  defines a create and clear helper per phase over a private path resolver, and `cindra.io` exports the registration
  pair alongside `resolve_active_binary_marker`, the one question every reader asks, which returns whichever marker sits
  beside a binary or None. The binarization pair stays inside `cindra.io`, whose `tiff.py` is its only caller.
  `register_plane` refuses to run while either marker exists. `binarize_recording` refuses a marked binary, a binary
  whose size disagrees with its plane's recorded frame geometry, and a two-channel plane holding no second channel
  binary, naming `repeat_binarization` as the remedy in each message. The conversion drops the registration marker of
  the binary it unlinks, because that marker describes a file that no longer exists. `binarize_recording` resolves the
  conversion plan (`resolve_tiff_conversion_plan`) before it clears the outputs derived from the previous binaries, so a
  conversion that fails its input validation leaves the recording's results in place. Preserve this protocol when
  modifying either stage, and keep `repeat_binarization` named as the remedy every refusal states, because a
  caller-requested rebuild is the recovery path.
- **Memory efficiency**: Pre-allocates arrays with `np.empty` when overwritten immediately. Uses flattened mask arrays
  with offset indices to reduce per-ROI allocations. Memory maps registration arrays on demand via
  `memory_map_arrays()`. Results tools use lightweight NumPy/YAML reads for targeted queries without full data loading.
  Groupwise diffeomorphic registration visits each unordered image pair once and caches each image's gradient with the
  deformed image that produced it, keeping the working set linear rather than quadratic in group size.
- **Polymorphic dispatch**: `extract_traces()` checks `isinstance(context, RuntimeContext)` to route between
  single-recording and multi-recording extraction paths.
- **Channel 2 behavior**: Channel 2 data returns empty arrays (`[]`) instead of None when absent. Channel 1 data raises
  an error if missing.
- **Module-level constants**: Use inline `"""docstring"""` below the definition, not `# comment` above.
- **Property docstrings**: Single sentence, even if spanning multiple lines. Do not split into summary + extended
  description.
- **Error messages**: Follow the `"Unable to [action]..."` pattern using `console.error()` from ataraxis-base-utilities.

### Core components

| Component                                 | File                                            | Purpose                                                 |
|-------------------------------------------|-------------------------------------------------|---------------------------------------------------------|
| `SingleRecordingConfiguration`            | `dataclasses/single_recording_configuration.py` | User-facing config with nested dataclasses              |
| `MultiRecordingConfiguration`             | `dataclasses/multi_recording_configuration.py`  | Multi-recording pipeline config                         |
| `AcquisitionParameters`                   | `dataclasses/single_recording_configuration.py` | Per-recording acquisition metadata                      |
| `RuntimeContext`                          | `dataclasses/runtime_contexts.py`               | Single-recording config + acquisition + runtime data    |
| `MultiRecordingRuntimeContext`            | `dataclasses/runtime_contexts.py`               | Multi-recording config + runtime data                   |
| `SingleRecordingRuntimeData`              | `dataclasses/single_recording_data.py`          | IO, registration, detection, extraction, timing data    |
| `MultiRecordingRuntimeData`               | `dataclasses/multi_recording_data.py`           | IO, registration, tracking, extraction, timing data     |
| `RecordingArrays`                         | `layout.py`                                     | On-disk contract: names, markers, and path resolvers    |
| `resolve_recording_planes`                | `io/inventory.py`                               | Read-only recording and dataset on-disk inventory       |
| `resolve_single_recording_job_universe`   | `orchestration/discovery.py`                    | Declared job set and the subset whose inputs exist      |
| `estimate_single_recording_job_memory_mb` | `orchestration/footprints.py`                   | Per-stage memory models and the two job estimators      |
| `size_single_recording_job`               | `orchestration/footprints.py`                   | Pairs a job's declared cores with its memory estimate   |
| `execute_single_recording_job`            | `orchestration/worker.py`                       | Per-job entry point against a caller-owned tracker      |
| `prime_recording`                         | `orchestration/worker.py`                       | Writes the shared bootstrap and reports the inventory   |
| `run_single_recording_pipeline`           | `orchestration/pipeline.py`                     | Execute single-recording four-phase workflow            |
| `run_multi_recording_pipeline`            | `orchestration/pipeline.py`                     | Execute multi-recording two-phase workflow              |
| `start_execution_session`                 | `orchestration/execution.py`                    | Batch engine: admission, process-pool dispatch, budgets |
| `register_recording_plane`                | `pipelines/single_recording.py`                 | Per-plane registration stage entry point (phase 2)      |
| `register_plane`                          | `registration/register.py`                      | Per-plane motion correction (rigid + optional nonrigid) |
| `GpuRegistrationBackend`                  | `registration/gpu.py`                           | Per-plane motion correction on a CUDA device            |
| `resolve_stage_workers`                   | `orchestration/allocation.py`                   | Measured per-stage worker defaults and worker resolver  |
| `SINGLE_RECORDING_PHASES`                 | `orchestration/jobs.py`                         | Phase model: job universe and prerequisite graph        |
| `resolve_openmp_runtime`                  | `orchestration/openmp.py`                       | macOS OpenMP runtime discovery, linking, verification   |
| `resolve_gpu_devices`                     | `orchestration/gpu.py`                          | CUDA device discovery and runtime probing               |
| `DiffeomorphicDemonsRegistration`         | `registration/diffeomorphic.py`                 | Cross-day diffeomorphic alignment algorithm             |
| `Deformation`                             | `registration/deformation.py`                   | Deformation field application and inversion             |
| `detect_plane_rois`                       | `detection/detect.py`                           | ROI detection via sparse detection with PCA denoising   |
| `track_rois_across_recordings`            | `detection/tracking.py`                         | Multi-recording ROI tracking via spatial clustering     |
| `compute_roi_statistics`                  | `detection/roi_statistics.py`                   | ROI property computation (skewness, compactness, etc.)  |
| `extract_traces`                          | `extraction/extract.py`                         | Fluorescence extraction and neuropil subtraction        |
| `apply_oasis_deconvolution`               | `extraction/deconvolve.py`                      | OASIS spike deconvolution                               |
| `create_masks`                            | `extraction/masks.py`                           | ROI mask creation with lambda weight computation        |
| `Classifier`                              | `classification/classify.py`                    | Cell vs. artifact classification                        |
| `BinaryFile`                              | `io/binary.py`                                  | Memory-mapped binary file access for imaging data       |
| `convert_tiffs_to_binary`                 | `io/tiff.py`                                    | TIFF to internal binary format conversion               |
| `combine_planes`                          | `io/combine.py`                                 | Multi-plane result combination                          |
| `run_roi_viewer`                          | `gui/app.py`                                    | ROI inspector GUI, single or multi-recording            |
| `run_tracking_viewer`                     | `gui/app.py`                                    | Multi-recording tracking quality GUI                    |
| `run_registration_viewer`                 | `gui/app.py`                                    | Registration quality viewer (binary + PC viewer)        |

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
| `cindra gpu`       | Report the CUDA devices registration uses and why it reaches none    |

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
| `scikit-learn`             | PCA denoising, registration PCs, ROI classification           |
| `natsort`                  | Semantic file path sorting (1, 2, 10 vs 1, 10, 2)             |
| `tifffile`                 | TIFF file loading and metadata extraction                     |
| `imagecodecs`              | Image codec support for TIFF decompression                    |
| `matplotlib`               | Visualization support for GUI viewers                         |
| `pyside6`                  | Qt6 GUI framework for interactive viewers                     |
| `pyqtgraph`                | High-performance plotting for GUI image display               |
| `click`                    | CLI framework for command-line interfaces                     |
| `mcp`                      | MCPServer host for agentic AI tool integration                |
| `psutil`                   | Available host memory reads that bound job admission          |
| `ataraxis-time`            | PrecisionTimer for pipeline step timing                       |
| `ataraxis-base-utilities`  | Console for unified message handling and error reporting      |
| `ataraxis-data-structures` | YamlConfig, ProcessingTracker, atomic and marker I/O          |
| `threadpoolctl`            | BLAS thread confinement around scikit-learn and LAPACK fits   |
| `pyyaml`                   | YAML serialization for configuration and tracker files        |
| `tbb4py`                   | Intel TBB threading layer for Numba parallelization (non-Mac) |
| `intel-cmplr-lib-rt`       | SVML runtime held for Numba's vectorization path (non-Mac)    |
| `cupy-cuda13x`             | CUDA array and FFT runtime for device registration (non-Mac)  |

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
7. Keep the dependency chain one-way. `gpu.py` and `openmp.py` import nothing from the package, and `jobs.py` imports
   `cindra.layout` alone. `discovery.py` imports `jobs`, `allocation.py` imports `jobs` and `gpu`, and `footprints.py`
   imports `jobs` and `allocation`. `worker.py` imports `jobs`, `allocation`, and `gpu`, and `pipeline.py` imports
   `worker`, `jobs`, `gpu`, and `openmp`. `execution.py` imports `pipeline`, `jobs`, `allocation`, and `gpu`, and no
   orchestration module imports `interface`. `gpu.py` writes nothing at import, and `verify_gpu_runtime()` runs only
   where the caller named a registration device

**Modifying registration:**

1. Review `src/cindra/registration/register.py` for per-plane motion correction entry point
2. Understand the two-step registration refinement when enabled
3. Rigid registration uses phase correlation (`rigid.py`), and nonrigid uses block-based deformation (`nonrigid.py`)
4. Cross-recording registration uses diffeomorphic demons (`diffeomorphic.py`) with multiscale pyramid (`pyramid.py`)
5. Registration rewrites its input binary in place under a `<binary>.registering` marker, the parallel of the
   `<binary>.binarizing` marker binarization writes while it fills that binary. Keep the create and clear pair around
   any new write loop, and confine BLAS fits with `threadpool_limits` as `register.py` does
6. `register_plane` takes `device: int | None`. None registers the plane on the host CPU, and an index builds
   `GpuRegistrationBackend` against that CUDA device (`registration/gpu.py`). The alignment pass reads
   `registration.gpu_batch_size` for its device batch while that field is above zero and `registration.batch_size`
   otherwise, and the secondary channel pass reads the shared size. Keep the device out of the configuration, because
   the batch engine assigns one device per running job

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
5. Name every path parameter and every path-valued response key from the shared vocabulary the MCP tool organization
   entry above defines (`raw_data_path`, `output_root`, `root_directory`, and the file-path names). Do not invent a
   per-tool word for a concept another tool already names, because one word carrying two of these concepts leaves the
   caller unable to tell which path a tool wants

**Adding or modifying CLI commands:**

1. Review `src/cindra/interface/cli.py` for the main CLI Click group structure
2. Review `src/cindra/interface/gui_cli.py` for the GUI CLI structure
3. Follow existing patterns for Click option decorators and error handling
4. CLI writes configuration overrides to the config file before pipeline execution

**Modifying API documentation:**

1. `docs/source/api.rst` documents the export surface. A package section carries an `autodata` directive for every
   constant that package's `__all__` names and its own modules define, and for nothing else
2. A constant a package holds for its own use keeps its public name and reaches no directive, and one whose defining
   module the page documents in its own section renders there instead of a second time
3. See `/api-docs` for the shared Sphinx conventions this rule tightens

**Important considerations:**

- The `console` is enabled in `src/cindra/__init__.py`. Do not re-enable it elsewhere
- The Numba threading layer is configured in `__init__.py` (TBB on non-Mac, OpenMP on macOS) immediately after importing
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
- A registration job runs on a CUDA device only where the caller named one, so `verify_gpu_runtime(device=...)` runs in
  two places. `run_single_recording_pipeline` calls it before its first dispatch, so a host exposing no usable device
  aborts having done no work. `dispatch_single_recording_job` calls it inside the REGISTER branch of the tracker's
  `run_job()` block, so a refusal is recorded as that job's failure. `run_multi_recording_pipeline` calls it nowhere,
  because no multi-recording stage runs on a device. `cindra gpu` and `check_gpu_runtime_tool` report what the host
  exposes, and both name `GPU_REMEDY`, which is the `cupy-cuda13x[ctk]` or `cupy-cuda12x[ctk]` installation the driver's
  CUDA major version selects. The CuPy pin carries a `sys_platform != 'darwin'` marker, so `orchestration/gpu.py` guards
  its import, holds `cupy` at None on a host without it, and reports `RUNTIME_MISSING`. Keep that guard, because the
  stub and documentation builds run on hosts carrying no CUDA device
- The `# type: ignore[import-untyped]` comments on the scikit-learn, threadpoolctl, and PyQtGraph imports are expected.
  Numba and CuPy are excluded via `pyproject.toml` mypy overrides, and the tifffile and yaml imports carry no such
  comment, because tifffile ships a py.typed marker and yaml checks against the `types-pyyaml` stub. Do not remove these
  comments
- The `# pragma: no cover` annotations on `@njit` function bodies are intentional. Do not remove them
- The multiscale diffeomorphic registration crosses the boundary between original-image pixels and the working
  resolution of a pyramid level in three places, and it converts units at two of them. `ScaleSpacePyramid` scales
  every smoothing sigma by the level's entry in `_level_downsample_factors`. `_scale_grid_sampling` converts the
  knot spacing into working-resolution pixels, while `_regularize_deformation` keeps the injectivity factor on the
  original-pixel spacing, because that factor divides `scale` by that spacing and both are original-pixel quantities.
  `Deformation.resize_field` leaves displacement magnitudes unscaled, which discounts each coarse level by its
  resolution ratio and weights it below the finer levels that follow it. That third choice is deliberate, and the
  method's `Notes` block records its reasoning. Do not report it as a unit-conversion defect, as an inconsistency
  with the other two conversions, or as a regression, and do not add a scaling variant of `resize_field` unless the
  user asks for one
- `_clear_downstream_data` sweeps every plane directory the output root holds rather than the contiguous range the
  declared plane count spans. A directory that count no longer covers therefore loses the results measured from the
  frames the conversion replaces, while keeping the binary the conversion does not rewrite. Keep the sweep reading the
  directories off disk, because the declared count comes from a user-editable file
- The imaging directory is resolved, then scanned flat. `find_data_directory` in `io/context.py` searches the configured
  data path for `cindra_parameters.json` and returns the directory holding it, and `_collect_tiff_files` in `io/tiff.py`
  then globs that one directory without descending. A data path may therefore name the imaging directory or any parent
  of it, while the TIFF files must sit beside the parameters file. Every reader shares that resolution.
  `resolve_tiff_conversion_plan` calls it before converting, `_read_source_geometry` calls it before sizing, and the
  prepare tools' raw data gate calls it before writing a manifest. `validate_recording_readiness_tool` calls it before
  validating, and `_find_acquisition_parameters` calls it before loading the acquisition metadata, so all five reach the
  same verdict for the same path. Keep any new reader on `find_data_directory` rather than scanning a configured path
  directly, because a reader that scans it directly rejects a recording the conversion would have processed. Correct a
  claim of recursive TIFF discovery rather than implementing one, because a caller who assumes recursion hands the batch
  the wrong directory
- Use `console.error()` from ataraxis-base-utilities for all error handling (no bare `raise`)
