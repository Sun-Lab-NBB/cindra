---
name: single-recording-processing
description: >-
  Orchestrates single-recording neural imaging batch processing via the cindra MCP server, dispatching to
  acquisition-preparation, configuration, and results skills as needed. Use when the user asks to process
  single-recording imaging data, run the single-recording batch pipeline, monitor single-recording batch jobs,
  re-run a single-recording processing phase, or when invoking /single-recording-processing.
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
- Supporting tools for validation and status checking (recording discovery owned by `/single-recording-configuration`)
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
options, invoke `/single-recording-configuration`. This skill owns the processing workflow. The results skill owns
the data it produces and the configuration skill owns the parameters it consumes.

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

Four-phase sequential pipeline per recording:

```text
Phase 1: BINARIZE (phase name binarization, I/O bound, 4 cores per job, up to 4 concurrent jobs)
├── Converts raw TIFFs to binary format
└── Initializes the per-plane runtime data hierarchy, one job per recording with an empty specifier

Phase 2: REGISTER (phase name registration, per plane, 8 cores per job)
├── Motion correction plus the registration-quality metrics computation
└── One job per plane, specifier plane_{plane_index}

Phase 3: PROCESS (phase name processing, per plane, 10 cores per job)
├── ROI detection, trace extraction, classification, spike deconvolution
└── One job per plane, specifier plane_{plane_index}, requires that plane's registration job to succeed

Phase 4: COMBINE (phase name combination, I/O bound, 1 core per job, up to 4 concurrent jobs)
└── Merges all plane results into a unified combined_metadata.npz dataset
```

Batch processing across multiple recordings:

```text
BINARIZE: Up to 4 concurrent recordings, 4 cores each, fixed concurrency
REGISTER: Concurrency bounded by the session CPU budget, 8 cores per plane job
PROCESS:  Concurrency bounded by the CPU budget and available memory, 10 cores per plane job
COMBINE:  Up to 4 concurrent recordings, 1 core each, fixed concurrency
```

Every job is admitted as soon as its own prerequisites succeed on its own tracker, so the whole dependency graph can
be submitted in one call. One plane starts processing while its peers are still registering, and each recording
advances independently.

---

## Processing workflow

### Execution model

The processing workflow uses a **prepare-then-execute** model:

1. **Prepare** creates an execution manifest (tracker files, per-recording configurations, job lists) without
   starting any computation. This step is idempotent, so calling it again on the same recordings returns the
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

1. **Discover recordings**: Use `discover_recordings_tool` (check the `single_recording_candidates` list) or
   accept explicit paths from user.

2. **Validate raw data**: Use `validate_recording_readiness_tool` on each recording. Skip for recordings where
   `get_recording_status_tool` shows status `binarizing`, `registering`, `processing`, `combining`, or `completed`.
   If validation fails, invoke `/acquisition-data-preparation` to resolve issues before continuing.

3. **Configure**: Ask the user if they have an existing template configuration file. If not,
   invoke `/single-recording-configuration` to create one. Template configs are reusable across
   recordings and live at user-chosen locations (e.g., `/Data/CA1_GCaMP6f_SD.yaml`). Do NOT create
   per-recording config copies. The prepare tool automatically saves resolved copies as
   `cindra/configuration.yaml` inside each recording's output directory, preserving the original
   template. Pass the same template path for all recordings that share parameters.

4. **Confirm output directory**: Ask the user where processed data should be written. Each
   recording requires an explicit output path. The pipeline does not auto-resolve output locations.
   Common patterns include writing output alongside the raw data (producing a `cindra/` subdirectory
   inside each recording) or writing to a separate root by mirroring the recording directory
   structure. `recording_output_paths` is a required parameter for both `prepare_single_recording_batch_tool`
   and `execute_full_pipeline_tool`.

5. **Confirm CPU allocation**: Present the per-phase measured defaults and ask the user whether to
   accept them or override them (see Resource management section).

6. **Execute**: Choose one of two approaches:

   **Simple (recommended for straightforward runs):**
   Call `execute_full_pipeline_tool` with `pipeline_type="single-recording"`, the confirmed recording
   paths, configuration path, output paths, and worker settings. This prepares and executes all
   phases automatically.

   **Fine-grained (for selective execution or re-runs):**
   a. Call `prepare_single_recording_batch_tool` with recording paths, configuration path, and
      output paths. This returns a manifest with job IDs and statuses.
   b. Select the jobs to execute from the manifest. Each entry in `recordings` is keyed by the
      recording data path and carries `configuration_path`, `tracker_path`, `output_path`,
      `pipeline_type`, `binarize_job`, `register_jobs`, `process_jobs`, and `combine_job`. The two
      `*_jobs` keys hold one entry per plane, and each job dict holds `job_id`, `name`, `specifier`,
      and `status`. The response also carries `total_recordings`, `total_jobs`, and
      `migrated_recordings` when an existing tracker gained per-plane registration jobs. When
      selecting by phase, you MUST include `register_jobs`, because every processing job depends on
      the registration job carrying the same `plane_{index}` specifier.
   c. Call `execute_processing_jobs_tool` with the selected job descriptors and worker settings. Each
      job descriptor needs `configuration_path`, `tracker_path`, `job_id`, and `pipeline_type` from
      the manifest.

7. **Monitor**: Use `get_processing_jobs_status_tool` to check progress. Optionally use
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

8. **Handle completion**: When all recordings finish, check for failures. A `success: true` return
   only means a tool ran, not that work is ready or done: gate decisions on the domain flag, not on
   `success`. For `verify_single_recording_output_tool`, gate on `complete` (false whenever `missing`
   is non-empty). For validate tools, gate on `valid`. For `execute_full_pipeline_tool`, gate on
   `started` (it returns `started: false` with a `next_step` when all phases are already complete).
   Checking `success` alone can advance on an unready or already-complete state. Route errors to the
   appropriate skill (see Error routing section). On success, invoke `/single-recording-results`
   to verify outputs, then `/visualization` for visual inspection.

#### Output-directory path rule

`get_recording_status_tool`, `verify_single_recording_output_tool`, and `clean_processing_output_tool`
all take the recording OUTPUT directory (the parent of the `cindra/` folder), which equals the
`recording_output_paths` / per-entry `output_path` the prepare tool returns, NOT the raw-data root.
This matters on a separate-output layout where output and raw-data roots differ:

- `get_recording_status_tool` and `clean_processing_output_tool` resolve `cindra/` directly under the
  given path with NO fallback. Feeding the raw-data root makes them report `not_started` or
  "directory not found", a silent false negative.
- `verify_single_recording_output_tool` also recursively searches for `configuration.yaml`, so it may
  still pass via that fallback even when fed the wrong root. The two then disagree.

Always reuse the `output_path` captured from the prepare manifest for status, verify, and clean.

### Re-running specific phases

To re-run specific phases (e.g., after changing ROI detection parameters):

1. Use `reset_processing_phases_tool` to reset the target phases to SCHEDULED status. Downstream
   phases are automatically reset. Resetting `binarization` also resets `registration`, `processing`,
   and `combination`. Resetting `registration` also resets `processing` and `combination`. Resetting
   `processing` also resets `combination`. The returned `effective_phases` list is sorted
   alphabetically rather than in execution order.
2. Optionally modify the configuration file before re-execution.
3. Optionally use `clean_processing_output_tool` to delete output files from the reset phases.
4. Call `execute_processing_jobs_tool` with the reset jobs from the manifest.

Both `reset_processing_phases_tool` and `clean_processing_output_tool` require `pipeline_type="single-recording"`
and a `phases` list drawn from the valid single-recording phase names: `binarization`, `registration`,
`processing`, `combination`. `reset_processing_phases_tool` also requires `tracker_path`, and
`clean_processing_output_tool` requires `recording_path`.

Cleaning `registration` deletes the plane's `registration_data` directory, which carries the `bad_frames.npy` array
that detection reads. Registration rewrites the plane binary in place, so clean `binarization` to rebuild that
binary from the raw TIFFs.

**Recovering an interrupted registration.** Registration rewrites the plane binary in place and holds a
`<binary>.registering` marker for the duration. If the job is killed, the marker persists and every later registration
of that plane fails with "Unable to register plane {index}. A previous registration of the binary file ... was
interrupted". Cleaning or resetting `registration` does NOT clear it, because the marker sits beside the `.bin` rather
than inside `registration_data`.

Recover by resetting the `binarization` phase with `reset_processing_phases_tool` and re-dispatching. Binarization
detects the marker, rebuilds the binary from the raw TIFFs, and clears the marker. You do NOT need to set
`repeat_binarization`, and you do NOT need `clean_processing_output_tool`. Binarization also rebuilds automatically when
a binary's size disagrees with the frame geometry recorded for its plane, which is what an interrupted conversion leaves
behind. `repeat_binarization` remains necessary only to force a rebuild of binaries that are intact.

---

## Resource management

Each single-recording phase runs under its own resource class with a measured per-job worker count. Setting
`workers_per_job` and `max_parallel_jobs` to `-1` accepts those measured defaults. The session CPU budget is
`cpu_count - 2`, with 2 cores reserved for system operations, and the dispatcher holds the sum of the cores
committed by every class inside that budget.

| Phase    | Resource class | Cores per job | Concurrency cap                      |
|----------|----------------|---------------|--------------------------------------|
| BINARIZE | `binarization` | 4             | Fixed at 4                           |
| REGISTER | `registration` | 8             | Session CPU budget                   |
| PROCESS  | `processing`   | 10            | CPU budget and memory, 15 GB per job |
| COMBINE  | `combination`  | 1             | Fixed at 4                           |

The `binarization` and `combination` classes ignore both `workers_per_job` and `max_parallel_jobs`. The
`registration` and `processing` classes accept either parameter as an override of the measured default and of the
derived concurrency cap, via `execute_processing_jobs_tool` or `execute_full_pipeline_tool`.

The multi-recording discovery and extraction stages run under their own `discovery` and `extraction` classes, with
measured defaults of 30 and 16 cores per job. Both accept `workers_per_job` and `max_parallel_jobs` as overrides. See
`/multi-recording-processing` for their concurrency caps. Discovery's 30 is the saturating allocation the stage is
admitted at, because its cost grows with the square of the recording count.

Present the measured per-phase defaults when confirming CPU allocation with the user. A `workers_per_job` value of 30
overrides the processing default of 10 and lowers the processing concurrency to at most the CPU budget divided by 30, so
it reduces rather than raises the memory the class holds. The override that can exhaust memory is a positive
`max_parallel_jobs`, which replaces the derived cap outright and therefore discards the 15 GB per job memory bound.

---

## Status formatting

When presenting batch status to the user, format as a table:

```text
**Single-Recording Batch Processing Status**

Current Phase: PROCESS
Summary: 10/30 recordings complete | 2 processing | 18 queued | 0 failed

| Recording                  | Binarize | Register | Process | Combine | Status     |
|----------------------------|----------|----------|---------|---------|------------|
| 2024-01-15-10-30-00-123456 | done     | 4/4      | 2/4     | pending | PROCESSING |
| 2024-01-15-11-45-00-234567 | done     | 4/4      | 4/4     | running | PROCESSING |
| 2024-01-16-09-00-00-111111 | done     | 4/4      | 4/4     | done    | SUCCEEDED  |
| 2024-01-16-10-15-00-222222 | pending  | 0/0      | 0/0     | pending | QUEUED     |
```

The `jobs` mapping that `get_recording_status_tool` returns holds exactly four keys, `binarize`, `register`,
`process`, and `combine`. The `register` and `process` keys map each `plane_{index}` specifier to a lowercased status
string. You MUST give the table a column for each of the four keys.

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

Prerequisite failures are returned inside the `invalid_jobs` list with a `reason` field, not as a top-level `error`.
The message reads "Unable to execute job {job_id}. Its prerequisite '{phase}' job {prerequisite_id} has not succeeded
and is not part of this submission.", where `{phase}` is the tracker phase name, for example `registration` for a
processing job.

### Processing failure routing

When processing fails for some recordings, read the error messages and route to the appropriate skill:

| Error pattern                                     | Skill to invoke                   |
|---------------------------------------------------|-----------------------------------|
| Missing `cindra_parameters.json`, TIFF read error | `/acquisition-data-preparation`   |
| Invalid parameter values, wrong plane/channel     | `/acquisition-data-preparation`   |
| TIFF files hold frames of differing shapes        | `/acquisition-data-preparation`   |
| Registration of the binary file was interrupted   | Reset `binarization`, re-dispatch |
| Configuration parameter issues                    | `/single-recording-configuration` |
| MCP tools unavailable, server connection errors   | `/cindra-mcp-environment-setup`   |

Wait for the current execution session to complete before starting retries. `cancel_processing_jobs_tool` clears the
admission pool and every resource class queue but does NOT stop already-dispatched worker threads, and it clears the
session state immediately, so a new session can start while cancelled jobs still run. After cancelling, poll
`get_recording_status_tool` on the affected recordings until no job remains RUNNING before dispatching again.

---

## Related skills

| Skill                             | Relationship                                                               |
|-----------------------------------|----------------------------------------------------------------------------|
| `/cindra-pipeline`                | Overview: end-to-end phases, handoffs, and the single-vs-multi entry point |
| `/cindra-mcp-environment-setup`   | Prerequisite: MCP server connectivity                                      |
| `/acquisition-data-preparation`   | Input: raw data preparation and validation                                 |
| `/single-recording-configuration` | Configuration: parameter reference and file creation                       |
| `/single-recording-results`       | Output: verify and explain processing results                              |
| `/multi-recording-configuration`  | Downstream: configure cross-recording tracking                             |
| `/multi-recording-processing`     | Downstream: cross-recording ROI tracking                                   |
| `/visualization`                  | Downstream: visual inspection of results                                   |

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
