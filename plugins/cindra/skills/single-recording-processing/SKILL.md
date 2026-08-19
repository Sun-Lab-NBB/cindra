---
name: single-recording-processing
description: >-
  Orchestrates single-recording neural imaging batch processing via the cindra MCP server, dispatching to
  acquisition-preparation, configuration, and results skills as needed. Use when the user asks to process
  single-recording imaging data, run the single-recording batch pipeline, monitor single-recording batch jobs, re-run a
  single-recording processing phase, or when invoking /single-recording-processing.
user-invocable: true
---

# Single-recording processing

Orchestrates the single-recording batch processing workflow: discover recordings, validate prerequisites, prepare
execution manifests, dispatch jobs, monitor progress, and hand off to downstream skills for output verification.

---

## Scope

**Covers:**
- Batch processing workflow: discovery, validation, preparation, execution, monitoring, and completion
- MCP planning tools (`get_pipeline_job_universe_tool`, `size_pipeline_jobs_tool`, `check_threading_runtime_tool`)
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

**Handoff rules:** If the user asks about specific output files, array shapes, data interpretation, or processing result
verification, invoke `/single-recording-results`. If the user asks about parameter tuning or configuration options,
invoke `/single-recording-configuration`. This skill owns the processing workflow. The results skill owns the data it
produces and the configuration skill owns the parameters it consumes.

---

## Agent requirements

You MUST use the cindra MCP tools for all processing operations. Do not import cindra Python functions directly or run
processing via scripts or CLI commands. If MCP tools are not available, invoke `/cindra-mcp-environment-setup` to
diagnose and resolve connectivity issues.

---

## Available tools

### Preparation tools

| Tool                                  | Purpose                                                                 |
|---------------------------------------|-------------------------------------------------------------------------|
| `get_pipeline_job_universe_tool`      | Reports every job a recording declares and which can run right now      |
| `size_pipeline_jobs_tool`             | Reports the cores and memory every job holds, before preparing anything |
| `check_threading_runtime_tool`        | Reports whether the numeric threading layer this host needs is loadable |
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
Phase 1: BINARIZE (phase name binarization, I/O bound, 3 cores per job, up to 4 concurrent jobs)
├── Converts raw TIFFs to binary format
└── Fills the per-plane binaries and records their frame geometry, one job per recording with an empty specifier

Phase 2: REGISTER (phase name registration, per plane, 4 cores per job)
├── Motion correction plus the registration-quality metrics computation
└── One job per plane, specifier plane_{plane_index}

Phase 3: PROCESS (phase name processing, per plane, 10 cores per job)
├── ROI detection, trace extraction, classification, spike deconvolution
└── One job per plane, specifier plane_{plane_index}, requires that plane's registration job to succeed

Phase 4: COMBINE (phase name combination, serial merge, 1 core per job)
└── Merges all plane results into a unified combined_metadata.npz dataset
```

Phase 1 consumes whole plane-and-channel interleave cycles, so it discards the frames of an incomplete final cycle and
warns with their count, and it fails a recording holding fewer frames than one whole cycle. Every plane and channel of
the recording therefore holds the same frame count.

Phase 1 has three outcomes: it skips the conversion, it converts every plane binary, or it refuses the recording. A
conversion discards the results of every plane directory the output root holds, including one the declared plane count
no longer reaches, whose own binary the conversion leaves in place. A refusal deletes nothing.

Batch processing across multiple recordings:

```text
BINARIZE: Up to 4 concurrent recordings, 3 cores each, a hard ceiling spare capacity never lifts
REGISTER: 4 cores per plane job, 4 jobs reserved, released when nothing else can use the capacity
PROCESS:  10 cores per plane job, 5 jobs reserved, released when nothing else can use the capacity
COMBINE:  1 core each, bounded by the session CPU budget alone
```

Every job is admitted as soon as its own prerequisites succeed on its own tracker, so the whole dependency graph can be
submitted in one call. One plane starts processing while its peers are still registering, and each recording advances
independently.

Dispatch runs in two passes. The first honors every reservation, so the conversion jobs at the root of the chain keep
their share of the host while planes are still registering. The second releases the reservations over whatever capacity
the first left unused, so a deep queue runs at its full width rather than idling the host. Memory bounds dispatch
rather than concurrency: `execute_processing_jobs_tool` sizes every job it submits from the recording it will process,
and a job whose geometry cannot be read is not dispatched at all. It joins `invalid_jobs` with the reason "Unable to
size the job from its configuration", while `execute_full_pipeline_tool` drops that recording into
`unsizable_recordings` and runs the rest of the batch.

Every job runs in its own spawned process, so a job that exhausts memory takes down its own worker rather than the
batch, and the engine records a terminal outcome for every job the resulting pool failure strands.

---

## Processing workflow

### Execution model

The processing workflow uses a **prepare-then-execute** model:

1. **Prepare** creates an execution manifest (tracker files, per-recording configurations, job lists) without starting
   any computation. This step is idempotent, so calling it again on the same recordings returns the existing manifest.

2. **Execute** dispatches jobs from the manifest with prerequisite validation, resource allocation, and automatic phase
   sequencing. Only one execution session can be active at a time.

For simple cases, `execute_full_pipeline_tool` combines both steps into a single call with automatic phase advancement.
For fine-grained control (e.g., running only specific phases, custom resource allocation, or selective re-runs), use
`prepare_single_recording_batch_tool` followed by `execute_processing_jobs_tool`.

### Pre-processing checklist

```text
- [ ] Recordings discovered or paths provided
- [ ] Raw data validated (or existing binaries confirmed via get_recording_status_tool)
- [ ] Template configuration confirmed or created (one template can serve multiple recordings)
- [ ] Output directory confirmed with user
- [ ] Share of the machine to dedicate to processing confirmed with user
- [ ] Recordings to process confirmed
```

**STOP**: If any checkbox is incomplete, do not proceed. Complete the missing steps first.

### Workflow steps

1. **Discover recordings**: Use `discover_recordings_tool` (check the `single_recording_candidates` list) or accept
   explicit paths from user.

2. **Validate raw data**: Use `validate_recording_readiness_tool` on each recording. Skip for recordings where
   `get_recording_status_tool` shows status `binarizing`, `registering`, `processing`, `combining`, or `completed`. If
   validation fails, invoke `/acquisition-data-preparation` to resolve issues before continuing.

3. **Configure**: Ask the user if they have an existing template configuration file. If not, invoke
   `/single-recording-configuration` to create one. Template configs are reusable across recordings and live at
   user-chosen locations (e.g., `/Data/CA1_GCaMP6f_SD.yaml`). Do NOT create per-recording config copies. The prepare
   tool automatically saves resolved copies as `cindra/configuration.yaml` inside each recording's output directory,
   preserving the original template. Pass the same template path for all recordings that share parameters.

4. **Confirm output directory**: Ask the user where processed data should be written. Each recording requires an
   explicit output path. The pipeline does not auto-resolve output locations. Common patterns include writing output
   alongside the raw data (producing a `cindra/` subdirectory inside each recording) or writing to a separate root by
   mirroring the recording directory structure. `recording_output_paths` is a required parameter for both
   `prepare_single_recording_batch_tool` and `execute_full_pipeline_tool`.

5. **Confirm the machine budget**: The engine resolves every per-class worker count and concurrency cap itself, so
   do not ask the user to choose them. Ask only how much of the machine this run may take, because that is the one
   allocation question the engine cannot answer: it claims every core but two and the memory available when the session
   starts. Confirm that the host is free for the run, or take an explicit ceiling from the user and pass it as
   `max_parallel_jobs`. Report the resolved per-class allocation after dispatch, from the `resource_classes` mapping
   the execute tool returns, rather than predicting it beforehand.

6. **Execute**: Choose one of two approaches:

   **Simple (recommended for straightforward runs):**
   Call `execute_full_pipeline_tool` with `pipeline_type="single-recording"`, the confirmed recording paths,
   configuration path, output paths, and worker settings. This prepares and executes all phases automatically.

   **Fine-grained (for selective execution or re-runs):**
   a. Call `prepare_single_recording_batch_tool` with recording paths, configuration path, and output paths. This
      returns a manifest with job IDs and statuses.
   b. Select the jobs to execute from the manifest. Each entry in `recordings` is keyed by the recording data path and
      carries `configuration_path`, `tracker_path`, `output_path`, `pipeline_type`, `binarize_job`, `register_jobs`,
      `process_jobs`, and `combine_job`. The two `*_jobs` keys hold one entry per plane, and each job dict holds
      `job_id`, `name`, `specifier`, and `status`, plus `executor_id` for a recording whose tracker already existed. The
      response also carries `total_recordings`, `total_jobs`, `migrated_recordings` when an existing tracker gained
      per-plane registration jobs, and the rejection lists described under Partially accepted batches. When selecting
      by phase, you MUST include `register_jobs`, because every processing job depends on the registration job carrying
      the same `plane_{index}` specifier.
   c. Call `execute_processing_jobs_tool` with the selected job descriptors and worker settings. Each job descriptor
      needs `configuration_path`, `tracker_path`, `job_id`, and `pipeline_type` from the manifest.

7. **Monitor**: Use `get_processing_jobs_status_tool` to check progress. Optionally use
   `get_active_execution_timing_tool` for per-job timing and session throughput. These two tools reflect only the active
   in-process execution session and return `active: false` with empty jobs when no session is running. This drained
   state happens not only after an MCP server restart, a reconnect, or a batch dispatched by a prior process, but also
   after NORMAL completion: the manager clears session state on success AND on failure. So an all-zero, inactive status
   can mean "finished," not "nothing ran." Do not read it as failure. For final per-job outcomes, read persisted on-disk
   tracker state via `get_batch_status_overview_tool` for a whole-tree view, `get_recording_status_tool` per recording,
   or `verify_single_recording_output_tool` (all using the output directory, see the Output-directory path rule).
   Present status as a formatted table (see Status formatting section).

8. **Handle completion**: When all recordings finish, check for failures. A `success: true` return only means a tool
   ran, not that work is ready or done: gate decisions on the domain flag, not on `success`. For
   `verify_single_recording_output_tool`, gate on `complete` (false whenever `missing` is non-empty). For validate
   tools, gate on `valid`. For `execute_full_pipeline_tool`, gate on `started` (it returns `started: false` with a
   `next_step` when all phases are already complete). Checking `success` alone can advance on an unready or
   already-complete state. Route errors to the appropriate skill (see Error routing section). On success, invoke
   `/single-recording-results` to verify outputs, then `/visualization` for visual inspection.

### Output-directory path rule

`get_recording_status_tool`, `verify_single_recording_output_tool`, and `clean_processing_output_tool` all take the
recording OUTPUT directory (the parent of the `cindra/` folder), which equals the `recording_output_paths` / per-entry
`output_path` the prepare tool returns, NOT the raw-data root. This matters on a separate-output layout where output and
raw-data roots differ:

- `get_recording_status_tool` and `clean_processing_output_tool` resolve `cindra/` directly under the given path with NO
  fallback. Feeding the raw-data root makes them report `not_started` or "directory not found", a silent false negative.
- `verify_single_recording_output_tool` also recursively searches for `configuration.yaml`, so it may still pass via
  that fallback even when fed the wrong root. The two then disagree.

Always reuse the `output_path` captured from the prepare manifest for status, verify, and clean.

### Re-running specific phases

To re-run specific phases (e.g., after changing ROI detection parameters):

1. Use `reset_processing_phases_tool` to reset the target phases to SCHEDULED status. Downstream phases are
   automatically reset. Resetting `binarization` also resets `registration`, `processing`, and `combination`. Resetting
   `registration` also resets `processing` and `combination`. Resetting `processing` also resets `combination`. The
   returned `effective_phases` list is ordered by pipeline execution order.
2. Optionally modify the configuration file before re-execution.
3. Optionally use `clean_processing_output_tool` to delete output files from the reset phases.
4. Call `execute_processing_jobs_tool` with the reset jobs from the manifest.

Both `reset_processing_phases_tool` and `clean_processing_output_tool` require `pipeline_type="single-recording"` and a
`phases` list drawn from the valid single-recording phase names: `binarization`, `registration`, `processing`,
`combination`. `reset_processing_phases_tool` also requires `tracker_path`, and `clean_processing_output_tool` requires
`recording_path`.

Cleaning `registration` deletes the plane's `registration_data` directory, which carries the `bad_frames.npy` array
that detection reads. It does not undo registration, because the stage rewrote the plane binary in place and that
binary stays registered. Clean `binarization` to rebuild the binary from the raw TIFFs.

**Recovering an interrupted write.** Binarization fills the plane binary under a `<binary>.binarizing` marker and
registration rewrites it in place under a `<binary>.registering` marker, each held for the duration of that phase's
write. The suffix names the phase that died, matching the `binarizing` and `registering` statuses
`get_recording_status_tool` reports, and the recovery is the same for both. If the job is killed, the marker persists
and every later registration of that plane fails with "Unable to register plane {index}. A previous write of the binary
file ... was interrupted". Cleaning or resetting `registration` does NOT clear either marker, because they sit beside
the `.bin` rather than inside `registration_data`.

Binarization refuses a marked binary rather than rebuilding it, and the refusal names the affected files and
`repeat_binarization` as the remedy. Recover by enabling `file_io.repeat_binarization` in the recording's
configuration, resetting the `binarization` phase with `reset_processing_phases_tool`, and re-dispatching. The rebuild
clears every marker, and you do NOT need `clean_processing_output_tool`. The same recovery applies to a binary whose
size disagrees with the frame geometry recorded for its plane, which is what a truncation outside the pipeline leaves
behind, and to a two-channel plane holding no channel 2 binary. Every rebuild discards the recording's registration,
detection, extraction, and combined results, which the later phases recompute, so warn the user before dispatching and
budget a full reprocessing run rather than a conversion alone.

---

## Resource management

Each single-recording phase runs under its own resource class with a measured per-job worker count. Leaving
`workers_per_job` as None accepts the measured stage default and leaving `max_parallel_jobs` as None accepts the derived
concurrency cap. A positive value of either is used exactly. Setting `workers_per_job` to -1 gives every job the whole
session core budget, while setting `max_parallel_jobs` to -1 lifts the derived cap so that only the job count bounds
concurrency. The session CPU budget is `cpu_count - 2`, with 2 cores reserved for system operations, and the dispatcher
holds the sum of the cores committed by every class inside that budget. The one exception is that while nothing is
running the dispatcher admits a single job regardless, so a job whose worker count exceeds the whole budget still runs
rather than stalling the session.

| Phase    | Resource class | Cores per job | Concurrency                            |
|----------|----------------|---------------|----------------------------------------|
| BINARIZE | `binarization` | 3             | Hard ceiling of 4                      |
| REGISTER | `registration` | 4             | Session CPU budget, 4 jobs reserved    |
| PROCESS  | `processing`   | 10            | Session CPU budget, 5 jobs reserved    |
| COMBINE  | `combination`  | 1             | Session CPU budget                     |

Every cap but binarization's derives from the host as `min(max(1, budget // cores_per_job), max(1, job_count))`, so a
wider machine raises it without being asked, and the dispatcher then admits against the live core and memory budgets
rather than against the cap alone. The engine saturates the host it is given, so leave both parameters as None unless
the user asks for an override. Binarization's ceiling of 4 is the exception it never lifts, because the stage decodes at
the storage's rate rather than the host's core count, and that class alone ignores both `workers_per_job` and
`max_parallel_jobs`.

A reservation binds only in the dispatcher's first pass. The second pass releases it over whatever capacity the first
left unused, so a reserved class runs at its full derived width whenever no other queue can use the room.

Memory bounds dispatch separately from every class cap. Each job is estimated from the recording it will process, and
the dispatcher holds the sum of the running jobs' estimates inside the session memory budget, reported as
`memory_budget_mb`. That budget is the host's available memory sampled once when the session starts, and it is never
re-read, so memory another process frees mid-batch does not widen it and memory another process claims does not narrow
it. A batch that dispatches fewer jobs than the caps allow, on a host with idle cores, is memory-bound rather than
stalled.

The multi-recording discovery and extraction stages run under their own `discovery` and `extraction` classes, with
measured defaults of 2 and 16 cores per job. Both accept `workers_per_job` and `max_parallel_jobs` as overrides. See
`/multi-recording-processing` for their concurrency caps. Discovery's 2 covers the deformation pool alone, because the
stage has no parallel critical path.

Report the resolved allocation after dispatch, from the `resource_classes` mapping the execute tool returns, rather
than predicting it beforehand. When the user does ask for an override, a `workers_per_job` value of 30 overrides the
processing default of 10 and lowers the processing concurrency to at most the CPU budget divided by 30. That override
therefore reduces the memory the class holds. A positive `max_parallel_jobs` replaces the derived cap outright, but it
cannot exhaust memory on its own: the dispatcher still holds the running jobs' estimated memory inside the session
memory budget whatever cap a class carries.

### Planning before dispatch

`get_pipeline_job_universe_tool` answers which jobs can run right now. It reads the inventory the output directories
already hold, so it works before a tracker exists and returns `resolved: false` with an empty universe for a recording
carrying nothing rather than failing. Each entry carries a `ready` flag reporting that the job's own input exists. The
conversion job is ready once the acquisition parameters resolve, a registration job once its plane carries the channel
binary, a processing job once its plane carries the reference image, and the combination job once every plane carries
its traces. Use it to plan a selective re-run, and `get_recording_status_tool` to read recorded outcomes once a batch
has been prepared. The two answer different questions, because a job whose input exists may still have a prerequisite
that has not succeeded on the tracker.

`size_pipeline_jobs_tool` reports the cores and memory every job of a recording holds, reading its acquisition metadata
and one source file header, so it plans a batch before any tracker or output directory exists. Pass the recording's
configuration path and `pipeline_type="single-recording"`. The response lists each job's `name`, `specifier`, `cores`,
and `memory_mb`, plus `peak_memory_mb` for the single largest job and `total_memory_mb` for every job at once. Compare
`peak_memory_mb` against the host's free memory to learn whether the largest job fits at all, and `total_memory_mb` to
learn whether the whole batch could ever run concurrently. These are the same figures the execute tools charge against
the session memory budget, so a batch whose peak exceeds free memory admits its jobs serially rather than failing.

`check_threading_runtime_tool` reports whether the numeric threading layer this host needs is loadable, which is OpenMP
on macOS and TBB elsewhere. Gate a batch on its `ready` flag. A macOS host that is not ready aborts every job at the
pipeline entry point before any stage runs, because both entry points verify the runtime first. A non-macOS host missing
TBB instead fails at the job's first parallelized call. Either way the outcome surfaces as a per-job tracker
failure rather than as a tool error, so checking first replaces parsing those failures. The response carries a `remedy`
command when the host is not ready.

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

The `single_recording.jobs` mapping that `get_recording_status_tool` returns holds exactly four keys, `binarize`,
`register`, `process`, and `combine`. The `register` and `process` keys map each `plane_{index}` specifier to a
lowercased status string. You MUST give the table a column for each of the four keys.

---

## Error routing

### Preparation errors

| Error Message                             | Resolution                               |
|-------------------------------------------|------------------------------------------|
| "At least one recording path is required" | Provide recording paths                  |
| "Configuration file not found"            | Invoke `/single-recording-configuration` |
| "No valid recording paths provided"       | Inspect `invalid_paths` in the response  |

### Partially accepted batches

A batch that rejects some of its input still returns `success: true`, so you MUST read the rejection lists rather than
treat the absence of an `error` as full acceptance. Report every rejected entry to the user by name before proceeding,
because the batch runs without it.

| Key                    | Returned by                          | Meaning                                                  |
|------------------------|--------------------------------------|----------------------------------------------------------|
| `invalid_paths`        | both prepare and full-pipeline tools | A supplied path is not an existing directory             |
| `invalid_recordings`   | both prepare and full-pipeline tools | Preparation failed, such as a missing acquisition file   |
| `migrated_recordings`  | both prepare and full-pipeline tools | A tracker gained the missing per-plane registration jobs |
| `unsizable_recordings` | `execute_full_pipeline_tool`         | Sizing cannot measure the recording, so it is omitted    |
| `invalid_jobs`         | `execute_processing_jobs_tool`       | A job failed validation or sizing and was not dispatched |

A recording the sizing pass cannot measure is excluded from the batch rather than aborting it, so a run that reports
`started: true` may still cover fewer recordings than you submitted. `execute_full_pipeline_tool` returns no recording
count of its own, so name every entry of `invalid_paths`, `invalid_recordings`, `migrated_recordings`, and
`unsizable_recordings` to the user rather than looking for a total to compare. Only the prepare tools return
`total_recordings`. A migrated recording is not rejected, but its dispatched job set differs from the one it last
carried, so report it alongside the rejections.

These lists do not share one element shape. Every job is sized from the recording's raw acquisition geometry rather than
from a per-stage allowance, so a recording with unreadable raw data loses every one of its jobs rather than the
conversion job alone. See [tool-responses.md](references/tool-responses.md) for the element shapes, the return-key
reference of the planning, execution, management, and status tools, and the terminal messages the engine writes to a
tracker.

### Execution errors

| Error Message                            | Resolution                                   |
|------------------------------------------|----------------------------------------------|
| "An execution session is already active" | Wait for current session or cancel first     |
| "Job ID not found in tracker"            | Re-prepare the batch to regenerate manifests |
| "Prerequisite ... has not succeeded"     | Execute prerequisite phases first            |

Prerequisite failures are returned inside the `invalid_jobs` list with a `reason` field, not as a top-level `error`. The
message reads "Unable to execute job {job_id}. Its prerequisite '{phase}' job {prerequisite_id} has not succeeded and is
not part of this submission.", where `{phase}` is the tracker phase name, for example `registration` for a processing
job.

### Processing failure routing

When processing fails for some recordings, read the error messages and route to the appropriate skill:

| Error pattern                                     | Skill to invoke                                                 |
|---------------------------------------------------|-----------------------------------------------------------------|
| Missing `cindra_parameters.json`, TIFF read error | `/acquisition-data-preparation`                                 |
| Invalid parameter values, wrong plane/channel     | `/acquisition-data-preparation`                                 |
| TIFF files hold frames of differing shapes        | `/acquisition-data-preparation`                                 |
| TIFF frames fall short of one interleave cycle    | `/acquisition-data-preparation`                                 |
| Previous write of the binary file was interrupted | Enable `repeat_binarization`, reset `binarization`, re-dispatch |
| Configuration parameter issues                    | `/single-recording-configuration`                               |
| MCP tools unavailable, server connection errors   | `/cindra-mcp-environment-setup`                                 |

Wait for the current execution session to complete before starting retries. `cancel_processing_jobs_tool` clears the
admission pool and every resource class queue but does NOT stop already-dispatched worker processes, and it clears the
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
- [ ] Share of the machine to dedicate to processing confirmed with user
- [ ] Batch prepared or full pipeline executed
- [ ] Status monitored until all recordings complete or fail
- [ ] Failed recordings routed to appropriate skill (see Error routing)
- [ ] Successful recordings verified via `/single-recording-results`
```
