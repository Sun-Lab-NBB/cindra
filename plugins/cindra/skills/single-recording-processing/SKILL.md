---
name: single-recording-processing
description: >-
  Orchestrates single-recording neural imaging batch processing via the cindra MCP server, dispatching to
  configuration, validation, and results skills as needed. Use when the user asks to process single-recording
  imaging data, run the single-recording batch pipeline, monitor batch jobs, or re-run a processing phase,
  or when invoking /single-recording-processing.
user-invocable: true
---

# Single-recording processing

Orchestrates the single-recording batch processing workflow: discover recordings, validate prerequisites,
prepare execution manifests, dispatch jobs, monitor progress, and hand off to downstream skills for output
verification.

---

## Scope

**Covers:**
- Batch processing workflow: discovery, validation, preparation, execution, monitoring, and completion
- MCP preparation tools (`prepare_single_recording_batch_tool`, `execute_full_pipeline_tool`)
- MCP execution tools (`execute_processing_jobs_tool`, `get_processing_jobs_status_tool`,
  `get_active_execution_timing_tool`, `cancel_processing_jobs_tool`)
- MCP management tools (`get_batch_status_overview_tool`, `reset_processing_phases_tool`,
  `clean_processing_output_tool`)
- Supporting tools for discovery, validation, and status checking
- Resource management and CPU allocation guidance
- Status formatting and progress monitoring
- Error routing to appropriate upstream skills

**Does not cover:**
- Configuration parameters, tuning guidance, or config file creation (see `/single-recording-configuration`)
- Output data formats, array shapes, dtypes, file references, or data interpretation (see `/single-recording-results`)
- Input data format, TIFF requirements, or acquisition parameters (see `/acquisition-data-preparation`)
- Multi-recording processing workflow (see `/multi-recording-processing`)
- MCP server connectivity or environment issues (see `/cindra-mcp-environment-setup`)
- Visual inspection of results (see `/visualization`)

**Handoff rules:** If the user asks about specific output files, array shapes, data interpretation, or processing
result verification, invoke `/single-recording-results`. If the user asks about parameter tuning or configuration
options, invoke `/single-recording-configuration`. This skill owns the processing workflow only — not the data
it produces or the parameters it consumes.

---

## Agent requirements

You MUST use the cindra MCP tools for all processing operations. Do not import cindra Python functions
directly or run processing via scripts or CLI commands. If MCP tools are not available, invoke
`/cindra-mcp-environment-setup` to diagnose and resolve connectivity issues.

---

## Available tools

### Preparation tools

| Tool                                  | Purpose                                                                 |
|---------------------------------------|-------------------------------------------------------------------------|
| `prepare_single_recording_batch_tool` | Prepares execution manifest without starting execution (idempotent)     |
| `execute_full_pipeline_tool`          | Convenience: prepares and executes all phases with automatic sequencing |

### Execution tools

| Tool                               | Purpose                                             |
|------------------------------------|-----------------------------------------------------|
| `execute_processing_jobs_tool`     | Dispatches prepared jobs for background execution   |
| `get_processing_jobs_status_tool`  | Returns per-job status of active execution session  |
| `get_active_execution_timing_tool` | Returns per-job timing and session-level throughput |
| `cancel_processing_jobs_tool`      | Cancels active execution, clears pending queues     |

### Management tools

| Tool                             | Purpose                                                         |
|----------------------------------|-----------------------------------------------------------------|
| `get_batch_status_overview_tool` | Bird's-eye view of all processing status under a root directory |
| `reset_processing_phases_tool`   | Selectively reset completed phases for re-runs                  |
| `clean_processing_output_tool`   | Delete output files for specific phases to reclaim disk space   |

### Supporting tools (used during workflow)

| Tool                                | Purpose                                             |
|-------------------------------------|-----------------------------------------------------|
| `discover_recordings_tool`          | Discovers single and multi-recording candidates     |
| `validate_recording_readiness_tool` | Validates raw data and parameters before processing |
| `get_recording_status_tool`         | Checks single and multi-recording processing status |

---

## Pipeline architecture

Three-phase sequential pipeline per recording:

```text
Phase 1: BINARIZE (I/O bound, up to 4 parallel)
├── Converts raw TIFFs to binary format
└── Determines plane count

Phase 2: PROCESS (CPU bound, parallel by plane)
├── Motion correction, ROI detection, signal extraction
└── Workers per plane via saturating allocation (see Resource management)

Phase 3: COMBINE (I/O bound, up to 4 parallel)
└── Merges all plane results into unified dataset
```

Batch processing across multiple recordings:

```text
BINARIZE: Up to 4 concurrent recordings (I/O bound, fixed concurrency)
PROCESS:  Parallel (recording-plane pairs up to core limit)
COMBINE:  Up to 4 concurrent recordings (I/O bound, fixed concurrency)
```

---

## Processing workflow

### Execution model

The processing workflow uses a **prepare-then-execute** model:

1. **Prepare** creates an execution manifest (tracker files, per-recording configurations, job lists) without
   starting any computation. This step is idempotent — calling it again on the same recordings returns the
   existing manifest.

2. **Execute** dispatches jobs from the manifest with prerequisite validation, resource allocation, and automatic
   phase sequencing. Only one execution session can be active at a time.

For simple cases, `execute_full_pipeline_tool` combines both steps into a single call with automatic phase
advancement. For fine-grained control (e.g., running only specific phases, custom resource allocation, or
selective re-runs), use `prepare_single_recording_batch_tool` followed by `execute_processing_jobs_tool`.

### Pre-processing checklist

```text
- [ ] Recordings discovered or paths provided
- [ ] Raw data validated (or existing binaries confirmed via get_recording_status_tool)
- [ ] Template configuration confirmed or created (one template can serve multiple recordings)
- [ ] Output directory confirmed with user
- [ ] CPU core allocation confirmed with user
- [ ] Recordings to process confirmed
```

**STOP**: If any checkbox is incomplete, do not proceed. Complete the missing steps first.

### Workflow steps

1. **Discover recordings** — Use `discover_recordings_tool` (check the `single_recording_candidates` list) or
   accept explicit paths from user.

2. **Validate raw data** — Use `validate_recording_readiness_tool` on each recording. Skip for recordings where
   `get_recording_status_tool` shows status `binarizing`, `processing`, `combining`, or `completed`. If validation
   fails, invoke `/acquisition-data-preparation` to resolve issues before continuing.

3. **Configure** — Ask the user if they have an existing template configuration file. If not,
   invoke `/single-recording-configuration` to create one. Template configs are reusable across
   recordings and live at user-chosen locations (e.g., `/Data/CA1_GCaMP6f_SD.yaml`). Do NOT create
   per-recording config copies — the prepare tool automatically saves resolved copies as
   `cindra/configuration.yaml` inside each recording's output directory, preserving the original
   template. Pass the same template path for all recordings that share parameters.

4. **Confirm output directory** — Ask the user where processed data should be written. Each
   recording requires an explicit output path; the pipeline does not auto-resolve output locations.
   Common patterns include writing output alongside the raw data (producing a `cindra/` subdirectory
   inside each recording) or writing to a separate root by mirroring the recording directory
   structure. `recording_output_paths` is a required parameter for both `prepare_single_recording_batch_tool`
   and `execute_full_pipeline_tool`.

5. **Confirm CPU allocation** — Present the resource allocation model and ask the user how many cores
   to use (see Resource management section).

6. **Execute** — Choose one of two approaches:

   **Simple (recommended for straightforward runs):**
   Call `execute_full_pipeline_tool` with `pipeline_type="single-recording"`, the confirmed recording
   paths, configuration path, output paths, and worker settings. This prepares and executes all
   phases automatically.

   **Fine-grained (for selective execution or re-runs):**
   a. Call `prepare_single_recording_batch_tool` with recording paths, configuration path, and
      output paths. This returns a manifest with job IDs and statuses.
   b. Select the jobs to execute from the manifest (e.g., only SCHEDULED jobs, only specific phases).
   c. Call `execute_processing_jobs_tool` with the selected job descriptors and worker settings. Each
      job descriptor needs `configuration_path`, `tracker_path`, `job_id`, and `pipeline_type` from
      the manifest.

7. **Monitor** — Use `get_processing_jobs_status_tool` to check progress. Optionally use
   `get_active_execution_timing_tool` for per-job timing and session throughput. These two tools
   reflect only the active in-process execution session and return `active: false` with empty jobs
   when no session is running. This drained state happens not only after an MCP server restart, a
   reconnect, or a batch dispatched by a prior process, but also after NORMAL completion: the
   manager clears session state on success AND on failure. So an all-zero, inactive status can mean
   "finished," not "nothing ran." Do not read it as failure. For final per-job outcomes, read
   persisted on-disk tracker state via `get_batch_status_overview_tool` for a whole-tree view,
   `get_recording_status_tool` per recording, or `verify_single_recording_output_tool` (all using
   the output directory, see the Output-directory path rule). Present status as a formatted table
   (see Status formatting section).

8. **Handle completion** — When all recordings finish, check for failures. A `success: true` return
   only means a tool ran, not that work is ready or done: gate decisions on the domain flag, not on
   `success`. For `verify_single_recording_output_tool`, gate on `complete` (false whenever `missing`
   is non-empty); for validate tools, gate on `valid`; for `execute_full_pipeline_tool`, gate on
   `started` (it returns `started: false` with a `next_step` when all phases are already complete).
   Checking `success` alone can advance on an unready or already-complete state. Route errors to the
   appropriate skill (see Error routing section). On success, invoke `/single-recording-results`
   to verify outputs, then `/visualization` for visual inspection.

#### Output-directory path rule

`get_recording_status_tool`, `verify_single_recording_output_tool`, and `clean_processing_output_tool`
all take the recording OUTPUT directory (the parent of the `cindra/` folder), which equals the
`recording_output_paths` / per-entry `output_path` the prepare tool returns — NOT the raw-data root.
This matters on a separate-output layout where output and raw-data roots differ:

- `get_recording_status_tool` and `clean_processing_output_tool` resolve `cindra/` directly under the
  given path with NO fallback. Feeding the raw-data root makes them report `not_started` or
  "directory not found" — a silent false negative.
- `verify_single_recording_output_tool` also recursively searches for `configuration.yaml`, so it may
  still pass via that fallback even when fed the wrong root. The two then disagree.

Always reuse the `output_path` captured from the prepare manifest for status, verify, and clean.

### Re-running specific phases

To re-run specific phases (e.g., after changing ROI detection parameters):

1. Use `reset_processing_phases_tool` to reset the target phases to SCHEDULED status. Downstream
   phases are automatically reset (e.g., resetting `processing` also resets `combination`).
2. Optionally modify the configuration file before re-execution.
3. Optionally use `clean_processing_output_tool` to delete output files from the reset phases.
4. Call `execute_processing_jobs_tool` with the reset jobs from the manifest.

Both `reset_processing_phases_tool` and `clean_processing_output_tool` require `pipeline_type="single-recording"`
and a `phases` list drawn from the valid single-recording phase names: `binarization`, `processing`,
`combination`. `reset_processing_phases_tool` also requires `tracker_path`; `clean_processing_output_tool`
requires `recording_path`.

---

## Resource management

The system uses saturating core allocation to distribute CPU cores across parallel compute-bound jobs.
I/O-bound jobs (binarize, combine) always use a fixed concurrency of 4 regardless of resource settings.

When both `workers_per_job` and `max_parallel_jobs` are set to `-1` (automatic), the allocator
runs the following algorithm for compute-bound jobs:

1. **Budget**: `cpu_count - 2` (2 cores reserved for system operations)
2. **Max parallel jobs**: `min(total_jobs, max(1, budget // 30))` (targets ~30 workers per job, with a floor of 1)
3. **Raw workers per job**: `budget // max_parallel_jobs`
4. **Round down** to the nearest multiple of 5
5. **Saturate**: If workers per job falls below 10 and parallelism > 1, reduce parallelism and
   recalculate until each job has at least 10 workers

| CPU Cores | Budget | Planes | Workers/Plane | Max Parallel | Total Utilized |
|-----------|--------|--------|---------------|--------------|----------------|
| 128       | 126    | 4      | 30            | 4            | 120            |
| 64        | 62     | 4      | 30            | 2            | 60             |
| 32        | 30     | 4      | 30            | 1            | 30             |
| 16        | 14     | 4      | 14 (→ 10)     | 1            | 10             |

Either or both values can be overridden explicitly via `workers_per_job` and `max_parallel_jobs`
parameters in `execute_processing_jobs_tool` or `execute_full_pipeline_tool`.

---

## Status formatting

When presenting batch status to the user, format as a table:

```text
**Single-Recording Batch Processing Status**

Current Phase: PROCESS
Summary: 10/30 recordings complete | 2 processing | 18 queued | 0 failed

| Recording                    | Binarize | Process | Combine | Status     |
|------------------------------|----------|---------|---------|------------|
| 2024-01-15-10-30-00-123456   | done     | 2/4     | pending | PROCESSING |
| 2024-01-15-11-45-00-234567   | done     | 4/4     | running | PROCESSING |
| 2024-01-16-09-00-00-111111   | done     | 4/4     | done    | SUCCEEDED  |
| 2024-01-16-10-15-00-222222   | pending  | 0/0     | pending | QUEUED     |
```

---

## Error routing

### Preparation errors

| Error Message                             | Resolution                               |
|-------------------------------------------|------------------------------------------|
| "At least one recording path is required" | Provide recording paths                  |
| "Configuration file not found"            | Invoke `/single-recording-configuration` |
| "No valid recording paths provided"       | Inspect `invalid_paths` in the response  |

### Execution errors

| Error Message                            | Resolution                                   |
|------------------------------------------|----------------------------------------------|
| "An execution session is already active" | Wait for current session or cancel first     |
| "Job ID not found in tracker"            | Re-prepare the batch to regenerate manifests |
| "Prerequisite ... has not succeeded"     | Execute prerequisite phases first            |

Prerequisite failures are returned inside the `invalid_jobs` list with a `reason` field (for example,
"Prerequisite BINARIZE job X has not succeeded."), not as a top-level `error`.

### Processing failure routing

When processing fails for some recordings, read the error messages and route to the appropriate skill:

| Error pattern                                     | Skill to invoke                   |
|---------------------------------------------------|-----------------------------------|
| Missing `cindra_parameters.json`, TIFF read error | `/acquisition-data-preparation`   |
| Invalid parameter values, wrong plane/channel     | `/acquisition-data-preparation`   |
| Configuration parameter issues                    | `/single-recording-configuration` |
| MCP tools unavailable, server connection errors   | `/cindra-mcp-environment-setup`   |

Wait for the current execution session to complete before starting retries.

---

## Related skills

| Skill                             | Relationship                                         |
|-----------------------------------|------------------------------------------------------|
| `/cindra-mcp-environment-setup`   | Prerequisite: MCP server connectivity                |
| `/acquisition-data-preparation`   | Input: raw data preparation and validation           |
| `/single-recording-configuration` | Configuration: parameter reference and file creation |
| `/single-recording-results`       | Output: verify and explain processing results        |
| `/multi-recording-processing`     | Downstream: cross-recording ROI tracking             |
| `/visualization`                  | Downstream: visual inspection of results             |

---

## Verification checklist

```text
Single-Recording Processing Workflow:
- [ ] MCP server connected (if not, invoke `/cindra-mcp-environment-setup`)
- [ ] Recordings discovered or explicit paths provided
- [ ] Raw data validated via `validate_recording_readiness_tool` (or existing binaries confirmed)
- [ ] Configuration file confirmed or created via `/single-recording-configuration`
- [ ] Output directory confirmed with user (required, no default)
- [ ] CPU core allocation confirmed with user
- [ ] Batch prepared or full pipeline executed
- [ ] Status monitored until all recordings complete or fail
- [ ] Failed recordings routed to appropriate skill (see Error routing)
- [ ] Successful recordings verified via `/single-recording-results`
```
