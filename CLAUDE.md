# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive understanding
of the codebase by invoking the `/explore-codebase` skill.

## Style guide compliance

You MUST invoke the appropriate style skill before performing ANY of the following tasks:

| Task                                          | Skill to invoke    |
|-----------------------------------------------|--------------------|
| Writing or modifying Python code              | `/python-style`    |
| Writing or modifying README files             | `/readme-style`    |
| Writing or modifying skill files or this file | `/skill-design`    |
| Writing or modifying pyproject.toml           | `/pyproject-style` |
| Writing or modifying tox.ini                  | `/tox-config`      |
| Writing or modifying Sphinx docs files        | `/api-docs`        |
| Creating or verifying project structure       | `/project-layout`  |
| Committing local changes                      | `/commit`          |

Each skill contains a verification checklist that you MUST complete before submitting any work. Failure to invoke the
appropriate skill results in style violations.

## Cross-referenced library verification

cindra depends on several `ataraxis-*` libraries. These libraries may be stored locally in the same parent directory as
this project, reachable as `../` from the repository root.

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g., `../ataraxis-time/`,
   `../ataraxis-base-utilities/`, `../ataraxis-data-structures/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or main branch on
   GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch, notify the user
   with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative reference for API usage,
   patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the actual library
state to prevent integration errors.

## Available skills

For cindra pipeline work, `/cindra-pipeline` is the end-to-end orchestration entry point that routes to the
phase-specific skills.

**Cindra plugin skills** (`plugins/cindra/skills/`):

| Skill                             | Description                                                      |
|-----------------------------------|------------------------------------------------------------------|
| `/cindra-pipeline`                | Orchestrates the end-to-end pipeline and opens a cindra session  |
| `/single-recording-processing`    | Orchestrates single-recording batch processing via MCP           |
| `/multi-recording-processing`     | Orchestrates multi-recording batch processing via MCP            |
| `/single-recording-configuration` | Documents single-recording configuration parameters and tools    |
| `/multi-recording-configuration`  | Documents multi-recording configuration parameters and tools     |
| `/single-recording-results`       | Documents single-recording output data formats and verification  |
| `/multi-recording-results`        | Documents multi-recording output data formats and verification   |
| `/acquisition-data-preparation`   | Prepares raw imaging data and acquisition parameter files        |
| `/visualization`                  | Launches and manages cindra GUI viewers for visual inspection    |
| `/cindra-mcp-environment-setup`   | Diagnoses and resolves MCP server connectivity issues            |

**Ataraxis automation plugin skills** (external, shared across projects):

| Skill                   | Description                                                       |
|-------------------------|-------------------------------------------------------------------|
| `/explore-codebase`     | Performs in-depth codebase exploration at session start           |
| `/explore-dependencies` | Explores installed ataraxis dependency APIs for a live snapshot   |
| `/python-style`         | Applies Python coding conventions (REQUIRED for Python work)      |
| `/readme-style`         | Applies README conventions (REQUIRED for README work)             |
| `/pyproject-style`      | Applies pyproject.toml conventions (REQUIRED for pyproject.toml)  |
| `/tox-config`           | Applies tox.ini conventions (REQUIRED for tox.ini work)           |
| `/api-docs`             | Applies Sphinx documentation conventions (REQUIRED for docs work) |
| `/project-layout`       | Applies project directory layout conventions                      |
| `/skill-design`         | Applies skill and CLAUDE.md conventions (REQUIRED for this file)  |
| `/audit-correctness`    | Audits source for active and latent bugs                          |
| `/audit-facts`          | Fact-checks documentation against authoritative source            |
| `/audit-performance`    | Audits source for algorithmic, allocation, and dtype costs        |
| `/audit-project`        | Orchestrates all four audits and merges their findings            |
| `/audit-style`          | Audits files against the applicable style checklists              |
| `/commit`               | Stages changes and creates a style-compliant commit               |
| `/pr`                   | Drafts a style-compliant pull request summary                     |
| `/release`              | Drafts style-compliant release notes from merged PRs              |

## MCP server

cindra provides two MCP servers that expose neural imaging pipeline tools for agentic work. When working with this
project or its dependencies, prefer using available MCP tools over direct code execution when appropriate.

**Servers:**

| Server       | CLI command      | Purpose                                            |
|--------------|------------------|----------------------------------------------------|
| `cindra-mcp` | `cindra mcp`     | Data processing, configuration, discovery, results |
| `cindra-gui` | `cindra-gui mcp` | GUI viewer lifecycle management and state queries  |

**Guidelines for MCP usage:**

1. **Discover available tools**: At the start of a session, check which MCP servers are connected and what tools they
   provide. Use these tools when they offer functionality relevant to the current task.

2. **Prefer MCP for runtime operations**: For operations like batch processing orchestration, configuration generation,
   recording discovery, and result querying, use MCP tools rather than writing and executing Python code directly. MCP
   tools provide consistent, tested interfaces with proper resource management.

3. **Use MCP for cross-library operations**: When dependency libraries (e.g., `ataraxis-data-structures`,
   `ataraxis-time`) provide MCP servers, explore and use their tools for interacting with those libraries.

4. **Fall back to code when necessary**: Use direct code execution when no MCP tool exists for the required
   functionality, the task requires custom logic, or you are writing or modifying library source code.

## Distribution model

This project ships through two channels that both have to be installed before any MCP tool resolves. The library, both
CLIs, and both MCP server implementations live under `src/cindra/` and reach the user through PyPI. The Claude Code
assets live under `plugins/cindra/` in this same repository and reach Claude Code through the marketplace declared in
`.claude-plugin/marketplace.json`. That plugin's `.claude-plugin/plugin.json` points at its `skills/` directory and
carries the `mcpServers` registrations for `cindra mcp` and `cindra-gui mcp`.

When modifying a skill, edit the SKILL.md under `plugins/cindra/skills/` and bump `version` in
`plugins/cindra/.claude-plugin/plugin.json` once per branch. When modifying an MCP tool, edit the matching tool module
under `src/cindra/interface/`.

## Project context

This is **cindra**, a reimplementation of the [suite2p](https://github.com/MouseLand/suite2p) neural imaging processing
library with expanded documentation, optimized algorithms, modern Python 3.14 support, and a novel multi-recording ROI
tracking pipeline based on the [OSM manuscript](https://www.nature.com/articles/s41586-024-08548-w). The library
provides CLI and MCP server interfaces for agentic processing, and interactive GUIs for visualization of pipeline
outputs.

### Key areas

| Directory                    | Purpose                                                         |
|------------------------------|-----------------------------------------------------------------|
| `src/cindra/`                | Main library source code                                        |
| `src/cindra/orchestration/`  | Job model, worker allocation, batch execution engine, pipelines |
| `src/cindra/classification/` | Cell type classification (distinguishing cells from artifacts)  |
| `src/cindra/dataclasses/`    | Configuration and runtime data structures (YamlConfig-based)    |
| `src/cindra/detection/`      | ROI detection, tracking, and statistics computation             |
| `src/cindra/extraction/`     | Fluorescence trace extraction, neuropil subtraction, OASIS      |
| `src/cindra/gui/`            | Interactive PySide6/PyQtGraph viewers for pipeline outputs      |
| `src/cindra/interface/`      | CLI, MCP servers, and tool modules for user-facing entry points |
| `src/cindra/io/`             | TIFF loading, binary file management, multi-plane combination   |
| `src/cindra/pipelines/`      | Stage entry points the pipelines dispatch                       |
| `src/cindra/registration/`   | Motion correction, diffeomorphic registration, deformation      |
| `tests/`                     | Test suite (mirrors source module structure)                    |
| `docs/`                      | Sphinx API documentation source                                 |

### Architecture

- **Single-recording pipeline**: Four-phase workflow (binarize, register, process, combine). Phase 1 converts TIFFs to
  internal binary format and initializes RuntimeContext per plane. Phase 2 runs per-plane motion correction and
  registration-quality metrics (parallelizable across planes). Phase 3 runs per-plane detection, classification, and
  extraction, and requires the plane to be registered (parallelizable across planes). Phase 4 merges plane-specific
  results into a unified `combined_metadata.npz` dataset, trimming the combined traces to the shortest contributing
  plane and recording `frame_count` and `plane_frame_counts` alongside the geometry. That metadata file doubles as the
  pipeline-completion marker, so it is written after its payload arrays and published through `atomic_write`.
  Phase 1 rejects a data directory whose TIFF files do not all hold frames of the same shape, naming
  `file_io.ignored_file_names` as the exclusion mechanism. Phases 2 and 3 carry a `plane_{index}` tracker specifier.
- **Multi-recording pipeline**: Two-phase workflow (discover, extract). Phase 1 selects ROIs from each recording,
  performs diffeomorphic demons registration to a common space, clusters ROIs across recordings via spatial overlap, and
  projects template masks back to individual recordings. Phase 2 extracts fluorescence traces and applies OASIS
  deconvolution for tracked ROI templates (parallelizable across recordings).
- **Self-driven orchestration**: `cindra.orchestration` owns the whole scheduling surface across seven modules that
  form a one-way dependency chain. `jobs.py` is the leaf above `cindra.layout`: it holds the job name enumerations,
  the phase model
  (`SINGLE_RECORDING_PHASES`, `MULTI_RECORDING_PHASES`, `PipelinePhase`, `PrerequisiteScope`), the resolvers that
  expand it into a recording's job universe (`resolve_single_recording_jobs`, `resolve_multi_recording_jobs`,
  `resolve_pipeline_jobs`), the prerequisite graph (`resolve_single_recording_prerequisites`,
  `resolve_multi_recording_prerequisites`, `resolve_prerequisite_job_ids`, `validate_job_prerequisites`), the phase
  expansion (`resolve_downstream_phases`, `order_phases_by_execution`), and the prerequisite messages. The plane
  specifier, the tracker filenames, and every other on-disk name live one layer below in `cindra.layout`, which
  `jobs.py` reads them from. `allocation.py` adds the measured stage worker defaults, the resource-class model, and
  the host core and memory budgets. `footprints.py` adds the per-stage memory models and the two estimators that
  report what one job holds. `discovery.py` pairs the job model with the on-disk inventory to report both the jobs a
  recording declares and the subset whose inputs exist. `worker.py` holds the per-job entry points every scheduler
  dispatches, along with the two priming entry points that write the shared bootstrap. `execution.py` holds the batch
  engine: `PendingJob`, `JobExecutionState`, the admission scan, the two-pass dispatcher, and the manager thread.
  `pipeline.py` holds the two sequential entry points. `openmp.py` carries no module-level side effect and its check
  runs only inside those two entry points, so importing the package writes nothing and a console message never
  precedes the stdio MCP server's JSON-RPC stream. Nothing below `orchestration` imports it, and
  no module inside it imports `interface`, so the MCP layer is a thin argument-validation and JSON-shaping wrapper
  over calls into the package. This mirrors the orchestration package of `ataraxis-video-system` and
  `ataraxis-communication-interface`, and its concurrency model follows `sollertia-forgery`.
- **Tracker-driven job state**: The transitions of a job the pipeline runs belong to the tracker's `run_job()`
  context manager rather than to a hand-rolled `start_job`/`complete_job`/`fail_job` sequence. The engine's
  `_fail_pending_jobs` is the one exception, because it records a terminal outcome for a job that never ran and
  therefore has no block to wrap. A remote
  invocation recovers its job's name and specifier through `tracker.resolve_job(job_id, universe)` rather than
  rebuilding the identifier map itself. Both pipeline entry points validate `target_plane` and `target_recording`
  against the resolved universe before aligning the tracker, so an out-of-range request reports the argument the
  caller passed instead of a job identifier the caller never saw.
- **Context pattern**: `RuntimeContext` combines configuration, acquisition parameters, and runtime data into a single
  object passed through pipeline steps. `MultiRecordingRuntimeContext` follows the same pattern, but carries only
  configuration and runtime data.
- **Configuration-driven execution**: Pipelines read all processing parameters from YAML files (YamlConfig subclasses).
  The CLI writes overrides to the config file before execution rather than passing arguments. Worker counts are the
  exception: they are explicit API parameters resolved through `cindra.orchestration`, which keeps the configuration file
  immutable and safe to share between concurrently dispatched jobs.
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
  `processing_tools`, `results_tools`) imported at module level to trigger `@mcp.tool()` registration. Processing uses a
  prepare-then-execute model: preparation tools create execution manifests (trackers, per-recording configurations, job
  lists) without starting computation, and execution tools dispatch jobs with prerequisite validation, per-class
  resource allocation, and automatic phase sequencing. The dispatch half lives in `cindra.orchestration`, so the
  execute, monitor, and cancel tools hold only argument validation and response shaping. The prepare tools stay in
  the interface layer, because building a manifest is a user-facing operation over paths and configuration files
  rather than part of the scheduling model. Every job class carries a measured per-job worker count from
  `cindra.orchestration`, and the combination class holds the single core its serial merge needs. Concurrency follows
  three separate terms. The binarization class carries a hard ceiling, because it decodes at the storage's rate rather
  than the host's core count and a wider batch finishes the same work more slowly while holding cores other work could
  use, so spare capacity never lifts it. The registration and processing classes carry soft reservations, which hold
  capacity back for the stages that wait on no other job and are released once nothing else can use the room. Every
  other class derives its concurrency from the session CPU budget alone. Memory bounds admission rather than
  concurrency, because the memory one job holds follows the recording it processes rather than the class it belongs
  to.
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

- **Numba parallelization**: The Numba threading layer is configured in `__init__.py` (TBB on non-Mac, OpenMP on macOS)
  immediately after importing `numba.config` and before importing any modules that compile `@njit` functions. Functions
  use `@njit(cache=True, parallel=True)` with `prange` over each kernel's outermost independent axis, which is frames in
  registration and ROIs in extraction. Numba is excluded from type checking via a `pyproject.toml` mypy override. The
  `# type: ignore[import-untyped]` comments apply to the scikit-learn, threadpoolctl, and PyQtGraph imports, and
  `# pragma: no cover` on JIT-compiled function bodies is expected. None of these should be removed.
- **Thread budget confinement**: Two ataraxis assets and one third-party context manager divide the work, and each
  covers a moment the others cannot. `limit_worker_threads` from `ataraxis-data-structures` encloses the batch
  engine's worker pool for the session's whole lifetime, so every worker process imports its numeric backends at a
  width of one instead of sizing each of them to the whole host. `initialize_worker_threads` runs as that pool's
  initializer, which reaches the backends that read their variable the first time they are asked to work rather than
  while they are imported. Neither touches `NUMBA_NUM_THREADS`, so a worker still latches the host's full ceiling and
  each stage raises to its own budget through `numba.set_num_threads`. `threadpool_limits` then confines the
  scikit-learn and LAPACK fits inside the running worker, which is the only one of the three that acts on an
  already-loaded library. `detection/detect.py` and `registration/metrics.py` limit to the job's `workers`, and
  `detection/denoise.py` limits to 1 because its own block pool already spends the budget. The BLAS width these set
  is a property of the process rather than of the thread that asked for it, which is why the engine gives each job
  its own process.
- **Registration integrity**: Registration rewrites the plane binary in place and guards the rewrite with a
  `<binary>.registering` marker (`create_registration_marker`, `clear_registration_marker`,
  `resolve_registration_marker_path`, exported from `cindra.io`). `register_plane` refuses to run while a marker exists,
  and `binarize_recording` treats a marked binary, or one whose size disagrees with its plane's recorded frame geometry,
  as invalid and rebuilds it from the source TIFFs without requiring `repeat_binarization`.
- **Memory efficiency**: Pre-allocates arrays with `np.empty` when overwritten immediately. Uses flattened mask arrays
  with offset indices to reduce per-ROI allocations. Memory maps registration arrays on demand via
  `memory_map_arrays()`. Results tools use lightweight NumPy/YAML reads for targeted queries without full data loading.
  Groupwise diffeomorphic registration visits each unordered image pair once and caches each image's gradient with the
  deformed image it derives from, keeping the working set linear rather than quadratic in group size.
- **Polymorphic dispatch**: `extract_traces()` checks `isinstance(context, RuntimeContext)` to route between
  single-recording and multi-recording extraction paths.
- **Channel 2 behavior**: Channel 2 data returns empty arrays (`[]`) instead of None when absent. Channel 1 data raises
  an error if missing.
- **Module-level constants**: Use inline `"""docstring"""` below the definition, not `# comment` above.
- **Property docstrings**: Single sentence, even if spanning multiple lines. Do not split into summary + extended
  description.
- **Error messages**: Follow the `"Unable to [action]..."` pattern using `console.error()` from ataraxis-base-utilities.

### Code standards

- MyPy type checking with full type annotations (`disallow_untyped_defs`, `warn_unused_ignores`)
- Google-style docstrings
- 120 character line limit
- Ruff for formatting and linting
- Python 3.14 only
- See `/python-style` for complete conventions

### Development commands

```bash
tox -e lint        # ruff format, ruff check, and mypy
tox -e stubs       # Generate the py.typed marker and the .pyi stub files
tox -e py314-test  # Run the pytest suite on Python 3.14
tox -e coverage    # Combine the test run's coverage data into an HTML report and apply the 100% gate
tox -e docs        # Build the Sphinx API documentation
tox -e deploy      # Upload the documentation built by 'docs' to the project's Netlify site
tox                # Full envlist: uninstall, export, lint, stubs, py314-test, coverage, docs, build, install
```

The `deploy` and `upload` tasks stay out of the envlist and are invoked manually for a release. The environment tasks
(`create`, `remove`, `provision`, `import`) target the `cindra_dev` mamba environment and are also manual.

### Testing

Tests use pytest with pytest-xdist for parallel execution (`-n logical --dist loadgroup`). Test files mirror the source
structure under `tests/` with a `_test.py` suffix, across `classification/`, `dataclasses/`, `detection/`,
`extraction/`, `io/`, `orchestration/`, `pipelines/`, and `registration/`.

The component map, CLI command reference, dependency table, and per-area workflow guidance live in an imported reference
file:

@.claude/cindra-reference.md
