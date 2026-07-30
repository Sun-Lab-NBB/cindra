# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive understanding
of the codebase by invoking the `/explore-codebase` skill.

This ensures you:
- Understand the project architecture before modifying code
- Follow existing patterns and conventions
- Do not introduce inconsistencies or break integrations

## Style guide compliance

You MUST invoke the appropriate style skill before performing ANY of the following tasks:

| Task                                   | Skill to invoke    |
|----------------------------------------|--------------------|
| Writing or modifying Python code       | `/python-style`    |
| Writing or modifying README files      | `/readme-style`    |
| Writing git commit messages            | `/commit`          |
| Writing or modifying skill files       | `/skill-design`    |
| Writing or modifying pyproject.toml    | `/pyproject-style` |
| Writing or modifying tox.ini           | `/tox-config`      |
| Writing or modifying Sphinx docs files | `/api-docs`        |

Each skill contains a verification checklist that you MUST complete before submitting any work. Failure to invoke the
appropriate skill results in style violations.

## Cross-referenced library verification

cindra depends on several `ataraxis-*` libraries. These libraries may be stored locally in the
same parent directory as this project (`/home/cyberaxolotl/Desktop/GitHubRepos/`).

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

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative reference for API
   usage, patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the actual library
state to prevent integration errors.

## Available skills

Skills are provided via Claude Code plugins, not the cindra pip package. The cindra plugin provides project-specific
skills (pipeline orchestration, data preparation, configuration, processing, results, visualization, MCP setup). The
ataraxis automation plugin provides shared workflow skills (style guides, commit, codebase exploration). For cindra
pipeline work, `/cindra-pipeline` is the end-to-end orchestration entry point that routes to the phase-specific skills.

**Ataraxis automation plugin skills:**

| Skill                   | Description                                                              |
|-------------------------|--------------------------------------------------------------------------|
| `/explore-codebase`     | Perform in-depth codebase exploration at session start                   |
| `/explore-dependencies` | Explore ataraxis dependency APIs for a live API snapshot                 |
| `/python-style`         | Apply cindra Python coding conventions (REQUIRED for all Python changes) |
| `/cpp-style`            | Apply cindra C++ coding conventions (not used by this Python-only repo)  |
| `/csharp-style`         | Apply cindra C# coding conventions (not used by this Python-only repo)   |
| `/readme-style`         | Apply cindra README conventions (REQUIRED for README changes)            |
| `/commit`               | Draft cindra style-compliant git commit messages                         |
| `/skill-design`         | Generate and verify skill files and CLAUDE.md project instructions       |
| `/project-layout`       | Apply cindra project directory layout conventions                        |
| `/pyproject-style`      | Apply cindra pyproject.toml conventions                                  |
| `/tox-config`           | Apply cindra tox.ini conventions                                         |
| `/api-docs`             | Apply cindra API documentation conventions                               |
| `/audit-style`          | Audit files for style compliance against the style skills                |
| `/audit-facts`          | Audit documentation for factual accuracy against source code             |
| `/pr`                   | Draft a style-compliant pull request summary                             |
| `/release`              | Draft style-compliant release notes from merged PRs                      |

**Cindra plugin skills:**

| Skill                             | Description                                                      |
|-----------------------------------|------------------------------------------------------------------|
| `/cindra-pipeline`                | End-to-end pipeline orchestration and session entry point        |
| `/single-recording-processing`    | Orchestrate single-recording batch processing via MCP            |
| `/multi-recording-processing`     | Orchestrate multi-recording batch processing via MCP             |
| `/single-recording-configuration` | Reference for single-recording pipeline configuration parameters |
| `/multi-recording-configuration`  | Reference for multi-recording pipeline configuration parameters  |
| `/single-recording-results`       | Reference for single-recording pipeline output data formats      |
| `/multi-recording-results`        | Reference for multi-recording pipeline output data formats       |
| `/acquisition-data-preparation`   | Guide for preparing raw imaging data for cindra processing       |
| `/visualization`                  | Launch and manage cindra GUI viewers for visual inspection       |
| `/cindra-mcp-environment-setup`   | Diagnose and resolve MCP server connectivity issues              |

## MCP server integration

The cindra Claude Code plugin registers two MCP servers that expose neural imaging pipeline tools for agentic AI
interaction. The plugin provides the server registrations and skills; the cindra pip package provides the server
implementations (`cindra mcp` and `cindra-gui mcp` CLI commands). Both must be installed for MCP tools to function.
When working with this project or its dependencies, prefer using available MCP tools over direct code execution when
appropriate.

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

## Project context

This is **cindra**, a reimplementation of the [suite2p](https://github.com/MouseLand/suite2p) neural imaging
processing library with expanded documentation, optimized algorithms, modern Python 3.14 support, and a novel
multi-recording ROI tracking pipeline based on the [OSM manuscript](https://www.nature.com/articles/s41586-024-08548-w).
The library provides CLI and MCP server interfaces for agentic processing, and interactive GUIs for visualization of
pipeline outputs.

### Key areas

| Directory                    | Purpose                                                         |
|------------------------------|-----------------------------------------------------------------|
| `src/cindra/`                | Main library source code                                        |
| `src/cindra/classification/` | Cell type classification (distinguishing cells from artifacts)  |
| `src/cindra/dataclasses/`    | Configuration and runtime data structures (YamlConfig-based)    |
| `src/cindra/detection/`      | ROI detection, tracking, and statistics computation             |
| `src/cindra/extraction/`     | Fluorescence trace extraction, neuropil subtraction, OASIS      |
| `src/cindra/gui/`            | Interactive PySide6/PyQtGraph viewers for pipeline outputs      |
| `src/cindra/interface/`      | CLI, MCP servers, and tool modules for user-facing entry points |
| `src/cindra/io/`             | TIFF loading, binary file management, multi-plane combination   |
| `src/cindra/pipelines/`      | High-level pipeline orchestration for single/multi-recording    |
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
  pipeline-completion marker, so it is written after its payload arrays and moved into place from a temporary name.
  Phase 1 rejects a data directory whose TIFF files do not all hold frames of the same shape, naming
  `file_io.ignored_file_names` as the exclusion mechanism. Phases 2 and 3 carry a `plane_{index}` tracker specifier.
- **Multi-recording pipeline**: Two-phase workflow (discover, extract). Phase 1 selects ROIs from each recording,
  performs diffeomorphic demons registration to a common space, clusters ROIs across recordings via spatial overlap,
  and projects template masks back to individual recordings. Phase 2 extracts fluorescence traces and applies OASIS
  deconvolution for tracked ROI templates (parallelizable across recordings).
- **Phase model**: `cindra.allocation` exports the pipeline phase model (`SINGLE_RECORDING_PHASES`,
  `MULTI_RECORDING_PHASES`, `PipelinePhase`, `PrerequisiteScope`) together with the resolvers that expand it into a
  recording's job universe (`resolve_single_recording_jobs`, `resolve_multi_recording_jobs`, `resolve_pipeline_jobs`),
  build the prerequisite graph (`resolve_single_recording_prerequisites`, `resolve_multi_recording_prerequisites`),
  expand a phase to its dependents (`resolve_downstream_phases`), and format the per-plane tracker specifier
  (`resolve_plane_specifier`, `PLANE_SPECIFIER_PREFIX`). The pipelines and the MCP layer read the model rather than
  restating the phase order, so a phase is added or reordered there and every consumer follows.
- **Context pattern**: `RuntimeContext` and `MultiRecordingRuntimeContext` combine configuration, acquisition
  parameters, and runtime data into single objects passed through pipeline steps.
- **Configuration-driven execution**: Pipelines read all processing parameters from YAML files (YamlConfig
  subclasses). The CLI writes overrides to the config file before execution rather than passing arguments. Worker
  counts are the exception: they are explicit API parameters resolved through `cindra.allocation`, which keeps the
  configuration file immutable and safe to share between concurrently dispatched jobs.
- **ProcessingTracker**: File-based YAML pipeline state tracking with FileLock for multi-process coordination. Manages
  job states (SCHEDULED, RUNNING, SUCCEEDED, FAILED) for resumable batch processing.
- **Subprocess GUI isolation**: GUI viewers launch as separate subprocesses with state file exchange via temporary
  files, avoiding Qt dependency loading during headless pipeline execution. The `cindra-gui` CLI entry point is
  separate from `cindra` for this reason.
- **MCP tool organization**: Tools are split across four modules (`acquisition_tools`, `configuration_tools`,
  `processing_tools`, `results_tools`) imported at module level to trigger `@mcp.tool()` registration.
  Processing uses a prepare-then-execute model: preparation tools create execution manifests (trackers,
  per-recording configurations, job lists) without starting computation, and execution tools dispatch jobs
  with prerequisite validation, per-class resource allocation, and automatic phase sequencing. Every job class
  carries a measured per-job worker count from `cindra.allocation`. The I/O-bound classes (binarize, combine) pair it
  with a fixed concurrency cap; the compute-bound classes (register, process, discover, extract) derive their
  concurrency from the session CPU budget, and the processing class additionally from available system memory.

### Key patterns

- **Numba parallelization**: The Numba threading layer is configured in `__init__.py` (TBB on non-Mac, OpenMP on
  macOS) immediately after importing `numba.config` and before importing any modules that compile `@njit` functions.
  Functions use `@njit(cache=True, parallel=True)` with `prange` for frame-level parallelization. Numba is excluded
  from type checking via a `pyproject.toml` mypy override; the `# type: ignore[import-untyped]` comments apply to the
  scikit-learn, threadpoolctl, PyQtGraph, and yaml imports, and `# pragma: no cover` on JIT-compiled function bodies
  is expected. None of these should be removed.
- **BLAS confinement**: Every site that dispatches a scikit-learn or LAPACK fit wraps it in `threadpool_limits`, so the
  BLAS thread count cannot multiply against the job's worker budget. `detection/detect.py` and
  `registration/metrics.py` limit to the job's `workers`; `detection/denoise.py` limits to 1, because its own block
  pool already spends the budget.
- **Registration integrity**: Registration rewrites the plane binary in place and guards the rewrite with a
  `<binary>.registering` marker (`create_registration_marker`, `clear_registration_marker`,
  `resolve_registration_marker_path`, exported from `cindra.io`). `register_plane` refuses to run while a marker
  exists, and `binarize_recording` treats a marked binary, or one whose size disagrees with its plane's recorded frame
  geometry, as invalid and rebuilds it from the source TIFFs without requiring `repeat_binarization`.
- **Memory efficiency**: Pre-allocates arrays with `np.empty` when overwritten immediately. Uses flattened mask arrays
  with offset indices to reduce per-ROI allocations. Memory maps registration arrays on demand via
  `memory_map_arrays()`. Results tools use lightweight NumPy/YAML reads for targeted queries without full data loading.
  Groupwise diffeomorphic registration visits each unordered image pair once and caches each image's gradient with the
  deformed image it derives from, keeping the working set linear rather than quadratic in group size.
- **Polymorphic dispatch**: `extract_traces()` checks `isinstance(context, RuntimeContext)` to route between
  single-recording and multi-recording extraction paths.
- **Channel 2 behavior**: Channel 2 data returns empty arrays (`[]`) instead of None when absent. Channel 1 data
  raises an error if missing.
- **Module-level constants**: Use inline `"""docstring"""` below the definition, not `# comment` above.
- **Property docstrings**: Single sentence, even if spanning multiple lines. Do not split into summary + extended
  description.
- **Error messages**: Follow the `"Unable to [action]..."` pattern using `console.error()` from
  ataraxis-base-utilities.

### Code standards

- MyPy type checking with full type annotations (`disallow_untyped_defs`, `warn_unused_ignores`)
- Google-style docstrings
- 120 character line limit
- Ruff for formatting and linting
- Python 3.14 only
- See `/python-style` for complete conventions

### Development commands

```bash
tox -e lint        # Format, lint, and type-check
tox -e stubs       # Generate .pyi stub files
tox -e py314-test  # Run tests for Python 3.14
tox -e coverage    # Aggregate coverage reports
tox -e docs        # Build Sphinx API documentation
tox                # Run full pipeline (uninstall -> export -> lint -> ... -> install)
```

### Testing

Tests use pytest with pytest-xdist for parallel execution (`-n logical --dist loadgroup`). Coverage is collected and
aggregated by the `coverage` tox environment. Test files mirror the source structure under `tests/` with a `_test.py`
suffix. Test directories: `classification/`, `dataclasses/`, `detection/`, `extraction/`, `io/`, `registration/`.

The detailed component map, CLI command reference, dependency table, and per-area workflow
guidance are maintained in an imported reference file to keep these instructions focused:

@.claude/cindra-reference.md
