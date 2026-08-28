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
- The MCP preparation, execution, management, and supporting tools listed in the Available tools section
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
invoke `/single-recording-configuration`.

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
| `size_pipeline_jobs_tool`             | Reports the cores, memory, and device memory every job holds            |
| `check_threading_runtime_tool`        | Reports whether the numeric threading layer this host needs is loadable |
| `check_gpu_runtime_tool`              | Reports the CUDA devices this host exposes for registration             |
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

### Supporting tools

`/single-recording-configuration` owns `discover_recordings_tool`. Invoke that skill for its parameters and usage.

| Tool                                | Purpose                                                  |
|-------------------------------------|----------------------------------------------------------|
| `discover_recordings_tool`          | Discovers single and multi-recording candidates          |
| `validate_recording_readiness_tool` | Validates raw data and parameters before processing      |
| `get_recording_status_tool`         | Checks single and multi-recording processing status      |
| `set_config_values_tool`            | Writes configuration values, such as a phase repeat flag |

---

## Pipeline architecture

Four-phase sequential pipeline per recording:

```text
Phase 1 BINARIZE (binarization, one job per recording, empty specifier): converts raw TIFFs to per-plane binaries and
        records their frame geometry
Phase 2 REGISTER (registration, one job per plane, specifier plane_{index}): motion correction plus the
        registration-quality metrics
Phase 3 PROCESS  (processing, one job per plane, specifier plane_{index}): ROI detection, trace extraction,
        classification, and spike deconvolution, requiring that plane's registration job to succeed
Phase 4 COMBINE  (combination, one job per recording, serial merge): merges every plane result into
        combined_metadata.npz
```

Phase 1 consumes whole plane-and-channel interleave cycles, so it discards the frames of an incomplete final cycle and
warns with their count, and it fails a recording holding fewer frames than one whole cycle. Every plane and channel of
the recording therefore holds the same frame count. Phase 1 has three outcomes: it skips the conversion, it converts
every plane binary, or it refuses the recording. A conversion discards the results of every plane directory the output
root holds, including one the declared plane count no longer reaches, whose own binary it leaves in place. A refusal
deletes nothing.

Every job is admitted as soon as its own prerequisites succeed on its own tracker, so the whole dependency graph can be
submitted in one call. One plane starts processing while its peers are still registering, and each recording advances
independently. A job whose geometry cannot be read is not dispatched at all. It joins `invalid_jobs` with the reason
"Unable to size the job from its configuration", while `execute_full_pipeline_tool` drops that recording into
`unsizable_recordings` and runs the rest of the batch. Every job runs in its own spawned process, so a job that exhausts
memory takes down its own worker rather than the batch, and the engine records a terminal outcome for every job the
resulting pool failure strands.

---

## Processing workflow

### Execution model

The processing workflow uses a **prepare-then-execute** model. **Prepare** creates an execution manifest (tracker
files, per-recording configurations, job lists) without starting any computation, and is idempotent, so calling it
again on the same recordings returns the existing manifest. **Execute** dispatches jobs from that manifest with
prerequisite validation, resource allocation, and automatic phase sequencing, and only one execution session can be
active at a time. For simple cases, `execute_full_pipeline_tool` combines both steps into a single call with automatic
phase advancement. For fine-grained control (e.g., running only specific phases, custom resource allocation, or
selective re-runs), use `prepare_single_recording_batch_tool` followed by `execute_processing_jobs_tool`.

**STOP**: Steps 1 through 5 below are the entry conditions for dispatch. Complete every one of them, and confirm each
against the verification checklist at the end of this skill, before calling any execute tool.

### Workflow steps

1. **Discover recordings**: Use `discover_recordings_tool` or accept explicit paths from the user. Every entry of
   `single_recording_candidates` is an object carrying `recording_root` and `raw_data_path`. Take `raw_data_path`,
   which is the directory that directly holds the TIFF files and the `cindra_parameters.json` file, and use
   `recording_root` only when naming the session to the user.

2. **Validate raw data**: Use `validate_recording_readiness_tool` with `raw_data_path` set to that same directory. Skip
   for recordings where `get_recording_status_tool` shows status `binarizing`, `registering`, `processing`,
   `combining`, or `completed`. If validation fails, invoke `/acquisition-data-preparation` to resolve issues first.

3. **Configure**: Ask the user if they have an existing template configuration file. If not, invoke
   `/single-recording-configuration` to create one. Template configs are reusable across recordings and live at
   user-chosen locations (e.g., `/Data/CA1_GCaMP6f_SD.yaml`), so pass the same template path for every recording that
   shares parameters. Do NOT create per-recording config copies: the prepare tool saves resolved copies as
   `cindra/configuration.yaml` inside each recording's output root, preserving the original template.

4. **Confirm output directory**: Ask the user where processed data should be written. Each recording requires an
   explicit output root, the parent of its `cindra/` folder, because the pipeline does not auto-resolve output
   locations. Common patterns write output alongside the raw data or into a separate root mirroring the recording
   directory structure. `output_roots` is required by both `prepare_single_recording_batch_tool` and
   `execute_full_pipeline_tool`, and holds one entry per `raw_data_paths` entry, in the same order.

5. **Confirm the machine budget**: The engine resolves every per-class worker count and concurrency cap itself, so do
   not ask the user to choose them. Ask only how much of the machine this run may take, which is the one allocation
   question the engine cannot answer: it claims every core but two and the memory available when the session starts.
   Confirm that the host is free, or pass the user's explicit ceiling as `max_parallel_jobs`, which caps each resource
   class separately rather than the session as a whole.

6. **Execute**: Choose one of two approaches:

   **Simple (recommended for straightforward runs):**
   Call `execute_full_pipeline_tool` with `pipeline_type="single-recording"`, the confirmed `raw_data_paths`,
   `configuration_path`, `output_roots`, worker settings, and `gpu_devices` when registration runs on a CUDA device.
   This prepares and executes all phases automatically.

   **Fine-grained (for selective execution or re-runs):**
   a. Call `prepare_single_recording_batch_tool` with `raw_data_paths`, `configuration_path`, and `output_roots`. This
      returns a manifest with job IDs and statuses.
   b. Select the jobs to execute from the manifest. Each entry in `recordings` is keyed by the raw data path and
      carries `configuration_path`, `tracker_path`, `output_root`, `pipeline_type`, `binarize_job`, `register_jobs`,
      `process_jobs`, and `combine_job`. The two `*_jobs` keys hold one entry per plane, and each job dict holds
      `job_id`, `name`, `specifier`, and `status`, plus `executor_id` for a recording whose tracker already existed. A
      `job_id` derives from the job name and specifier alone, so only the `(tracker_path, job_id)` pair identifies a
      job across the batch. The response also carries `total_recordings`, `total_jobs`, `migrated_recordings`, and the
      rejection lists described under Partially accepted batches. When selecting by phase, you MUST include
      `register_jobs`, because every processing job depends on the registration job carrying the same `plane_{index}`
      specifier.
   c. Call `execute_processing_jobs_tool` with the selected job descriptors and worker settings. Each job descriptor
      needs `configuration_path`, `tracker_path`, `job_id`, and `pipeline_type` from the manifest.

7. **Monitor**: Use `get_processing_jobs_status_tool` to check progress, passing `summary_only=True` on a wide batch
   to omit the per-job `jobs` list and return the session fields and summary counts alone. Optionally use
   `get_active_execution_timing_tool` for per-job timing and session throughput. These two tools reflect only the active
   in-process execution session and return `active: false` with empty jobs when no session is running. That drained
   state follows an MCP server restart, a reconnect, or a batch dispatched by a prior process, and also NORMAL
   completion, because the manager clears session state on success AND on failure. An all-zero, inactive status can
   therefore mean "finished," not "nothing ran." For final per-job outcomes, read persisted on-disk tracker state via
   `get_batch_status_overview_tool` for a whole-tree view, `get_recording_status_tool` per recording, or
   `verify_single_recording_output_tool` (all using the output root, see the Output-root path rule). Present status as
   a formatted table (see Status formatting section).

8. **Handle completion**: When all recordings finish, check for failures. A `success: true` return only means a tool
   ran, so gate decisions on the domain flag instead. Gate on `complete` for `verify_single_recording_output_tool`,
   which is false whenever `missing` is non-empty, on `valid` for the validate tools, and on `started` for
   `execute_full_pipeline_tool`, which returns `started: false` with a `next_step` when all phases are already complete.
   Route errors to the appropriate skill (see Error routing section). On success, invoke `/single-recording-results` to
   verify outputs, then `/visualization` for visual inspection.

### Output-root path rule

`get_recording_status_tool`, `verify_single_recording_output_tool`, and `clean_processing_output_tool` all name their
argument `output_root` and take the parent of the `cindra/` folder. That path equals the `output_roots` entries passed
to the prepare tool and the per-entry `output_root` it returns, never the `raw_data_paths` entry the same recording is
keyed by. This matters on a separate-output layout where the two roots differ. `get_recording_status_tool` and
`clean_processing_output_tool` resolve `cindra/` directly under the given path with NO fallback, so feeding the raw-data
path makes them report `not_started` or "Output root not found", a silent false negative.
`verify_single_recording_output_tool` also searches recursively for `configuration.yaml`, so it may still pass through
that fallback when fed the wrong root, and the two then disagree. Always reuse the `output_root` captured from the
prepare manifest for status, verify, and clean.

### Re-running specific phases

A reset changes tracker state alone. Binarization and registration each read their own output before running and
return immediately when it exists, so you MUST set the governing repeat flag before re-dispatching either.

| Phase          | Governing flag                     | Behavior while the flag is false                   |
|----------------|------------------------------------|----------------------------------------------------|
| `binarization` | `file_io.repeat_binarization`      | The job reuses the existing plane binaries         |
| `registration` | `registration.repeat_registration` | The job returns without re-registering any plane   |
| `processing`   | none                               | The job always recomputes detection and extraction |
| `combination`  | none                               | The job always re-merges the per-plane results     |

To re-run specific phases (e.g., after changing ROI detection parameters):

1. Use `reset_processing_phases_tool` to reset the target phases to SCHEDULED status. Downstream phases are
   automatically reset. Resetting `binarization` also resets `registration`, `processing`, and `combination`. Resetting
   `registration` also resets `processing` and `combination`. Resetting `processing` also resets `combination`. The
   returned `effective_phases` list is ordered by pipeline execution order.
2. Read the `warnings` list the reset returns. It appears when a reset phase is governed by a repeat flag that is false
   while that phase's output already exists, and each entry names the dotted flag to set. Act on every warning before
   dispatching, and consult the table above too, because the list is empty when the configuration is unreadable.
3. Set each named flag with `set_config_values_tool`, passing `<output_root>/cindra/configuration.yaml` as `file_path`
   and the flag as a dotted `section.parameter` key, for example `{"registration.repeat_registration": true}`. Write
   every other parameter this re-run changes, such as `roi_detection.threshold_scaling`, in the same call, because a
   rejected entry leaves the whole file unchanged.
4. Optionally use `clean_processing_output_tool` to delete output files from the reset phases.
5. Call `execute_processing_jobs_tool` with the reset jobs from the manifest.

Both `reset_processing_phases_tool` and `clean_processing_output_tool` require `pipeline_type="single-recording"` and a
`phases` list drawn from the valid single-recording phase names: `binarization`, `registration`, `processing`,
`combination`. `reset_processing_phases_tool` also requires `tracker_path`, and `clean_processing_output_tool` requires
`output_root`.

Cleaning `registration` deletes the plane's `registration_data` directory, which carries the `bad_frames.npy` array
that detection reads. It does not undo registration, because the stage rewrote the plane binary in place and that
binary stays registered. Clean `binarization` to rebuild the binary from the raw TIFFs.

**Recovering an interrupted write.** Binarization fills the plane binary under a `<binary>.binarizing` marker and
registration rewrites it in place under a `<binary>.registering` marker, each held for that phase's write. The suffix
names the phase that died, matching the `binarizing` and `registering` statuses `get_recording_status_tool` reports. If
the job is killed, the marker persists and every later registration of that plane fails with "Unable to register plane
{index}. A previous write of the binary file ... was interrupted". Cleaning or resetting `registration` does NOT clear
either marker, because they sit beside the `.bin` rather than inside `registration_data`.

Binarization refuses a marked binary rather than rebuilding it, naming the affected files and `repeat_binarization` as
the remedy. Recover by setting `file_io.repeat_binarization` to true with `set_config_values_tool`, resetting the
`binarization` phase, and re-dispatching. The rebuild clears every marker, and you do NOT need
`clean_processing_output_tool`. The same recovery applies to a binary whose size disagrees with the frame geometry
recorded for its plane, which a truncation outside the pipeline leaves behind, and to a two-channel plane holding no
channel 2 binary. Every rebuild discards the recording's registration, detection, extraction, and combined results,
which the later phases recompute, so warn the user and budget a full reprocessing run rather than a conversion alone.

---

## Resource management

Each single-recording phase runs under its own resource class with a per-job worker count. Both `workers_per_job` and
`max_parallel_jobs` are single scalars applied to every resource class separately, so neither one is a session-wide
total. Leaving either as None accepts that class's default, and a positive value replaces that class's figure outright,
in every class at once. Setting `workers_per_job` to -1 gives every job the whole session core budget, while setting
`max_parallel_jobs` to -1 lifts the derived cap so that only the job count bounds concurrency. The session CPU budget is
`cpu_count - 2`, with 2 cores reserved for system operations, and the dispatcher holds the sum of the cores committed by
every class inside that budget. While nothing is running it admits a single job regardless, so a job whose worker count
exceeds the whole budget still runs rather than stalling the session.

| Phase    | Resource class     | Cores per job | Dispatch ceiling | Concurrency                         |
|----------|--------------------|---------------|------------------|-------------------------------------|
| BINARIZE | `binarization`     | 3             | None             | Hard ceiling of 4                   |
| REGISTER | `registration`     | 4             | 32               | Session CPU budget, 4 jobs reserved |
| REGISTER | `registration_gpu` | 2             | None             | One job per session CUDA device     |
| PROCESS  | `processing`       | 10            | 10               | Session CPU budget, 5 jobs reserved |
| COMBINE  | `combination`      | 1             | None             | Session CPU budget                  |

Every cap but the two hard ceilings derives from the host as `min(max(1, budget // cores_per_job), max(1, job_count))`,
so a wider machine raises it without being asked, and the dispatcher then admits against the live core and memory
budgets rather than against the cap alone. The engine saturates the host it is given, so leave both parameters as None
unless the user asks for an override. Binarization's ceiling of 4 never lifts, because the stage decodes at the
storage's rate rather than the host's core count. A registration job takes `registration_gpu` when the session passed
`gpu_devices`. `Cores per job` is the smallest width a class gives a job and `Dispatch ceiling` is the largest, so a
class whose ceiling stands higher widens each job it dispatches over the cores no running job holds. That widening
applies only where the session left `workers_per_job` as None, and `resource_classes` reports the smallest width. A
reservation binds only in the dispatcher's first pass, and the second pass releases it over whatever capacity the first
left unused, so a reserved class runs at its full derived width whenever no other queue can use the room.

Memory bounds dispatch separately from every class cap. Each job is estimated from the recording it will process, and
the dispatcher holds the sum of the running jobs' estimates inside the session memory budget, reported as
`memory_budget_mb`. That budget is the host's available memory sampled once when the session starts and never re-read,
so memory another process frees or claims mid-batch changes nothing. A batch that dispatches fewer jobs than the caps
allow, on a host with idle cores, is memory-bound rather than stalled.

See `/multi-recording-processing` for the `discovery` and `extraction` classes the multi-recording pipeline runs.

Report the resolved allocation after dispatch, from the `resource_classes` mapping the execute tool returns, rather
than predicting it beforehand. When the user does ask for an override, state its full reach first. A `workers_per_job`
of 30 gives 30 cores to every registration, processing, and combination job alike rather than raising processing alone,
and it lowers each of those classes' derived concurrency to at most the CPU budget divided by 30. A `max_parallel_jobs`
of 4 likewise permits 4 registration jobs AND 4 processing jobs AND 4 combination jobs at the same time, held together
only by the session core and memory budgets. Binarization and device-backed registration ignore both overrides, because
a hard ceiling fixes each one's allocation.

### Planning before dispatch

The two planning tools that read a configuration load it through the loader the pipeline uses, which rejects a file
whose `file_io.output_path` is None with "The output_path must be configured in the FileIO section of the
configuration, but it is currently None." A freshly generated template carries None there, so neither tool accepts one.
Plan against the per-recording configuration the prepare tool writes at `<output_root>/cindra/configuration.yaml`, or
set `file_io.data_path` and `file_io.output_path` on a copy of the template with `set_config_values_tool` first. Beyond
those two fields, neither tool needs a tracker or any pipeline output.

`get_pipeline_job_universe_tool` answers which jobs can run right now. It reads the inventory the output directories
already hold, returning `resolved: false` with an empty universe for a recording carrying nothing rather than failing.
Each entry carries a `ready` flag reporting that the job's own input exists. The conversion job is ready once the
acquisition parameters resolve, and a registration job once its plane carries the channel binary. A processing job is
ready once its plane carries the reference image, and the combination job once every plane carries its traces. Use it to
plan a selective re-run, and `get_recording_status_tool` to read recorded outcomes once a batch has been prepared,
because a job whose input exists may still have a prerequisite that has not succeeded on the tracker.

`size_pipeline_jobs_tool` reports the cores, memory, and device memory every job of a recording holds, reading its
acquisition metadata and one source file header. Pass the recording's configuration path and
`pipeline_type="single-recording"`, and set `gpu_registration=True` whenever the batch will pass `gpu_devices`, so the
registration jobs report their device figures. The response lists each job's `name`, `specifier`, `cores`, `memory_mb`,
and `device_memory_mb`, plus `peak_memory_mb` and `peak_device_memory_mb` for the single largest job and
`total_memory_mb` for every job at once. Compare `peak_memory_mb` against the host's free memory to learn whether the
largest job fits at all, and `total_memory_mb` to learn whether the whole batch could ever run concurrently. These are
the figures the execute tools charge against the session memory budget, so a batch whose peak exceeds free memory admits
its jobs serially rather than failing.

`check_threading_runtime_tool` reports whether the numeric threading layer this host needs is loadable, which is OpenMP
on macOS and TBB elsewhere. Gate a batch on its `ready` flag. A macOS host that is not ready aborts every job at the
pipeline entry point before any stage runs, while a non-macOS host missing TBB fails at the job's first parallelized
call. Either outcome surfaces as a per-job tracker failure rather than as a tool error, so checking first replaces
parsing those failures. The response carries a `remedy` command when the host is not ready.

`check_gpu_runtime_tool` reports the CUDA devices this host exposes. `execute_processing_jobs_tool` and
`execute_full_pipeline_tool` both take a `gpu_devices` list, so gate any batch that passes one on the `ready` flag and
read `devices` for the indices, since an index the host does not expose is rejected with `started: false`. Omitting
`gpu_devices` registers on the host CPU, and `[-1]` names every device the host exposes without naming an index.

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
| "At least one raw data path is required"  | Provide raw data paths                   |
| "Configuration file not found"            | Invoke `/single-recording-configuration` |
| "No valid raw data paths provided"        | Inspect `invalid_paths` in the response  |

### Partially accepted batches

A batch that rejects some of its input still returns `success: true`, so you MUST read the rejection lists rather than
treat the absence of an `error` as full acceptance. Report every rejected entry to the user by name before proceeding,
because the batch runs without it.

| Key                    | Returned by                          | Meaning                                             |
|------------------------|--------------------------------------|-----------------------------------------------------|
| `invalid_paths`        | both prepare and full-pipeline tools | A supplied path is not an existing directory        |
| `invalid_recordings`   | both prepare and full-pipeline tools | Preparation failed, such as no usable TIFF file     |
| `path_conflicts`       | both prepare and full-pipeline tools | Stored paths differ from the ones passed here       |
| `migrated_recordings`  | both prepare and full-pipeline tools | A tracker gained the missing register jobs          |
| `unsizable_recordings` | `execute_full_pipeline_tool`         | Sizing cannot measure it, so it is omitted          |
| `invalid_jobs`         | `execute_processing_jobs_tool`       | A job failed validation or sizing, so it is omitted |

A `raw_data_paths` entry may name the recording's imaging directory or any parent of it, because preparation resolves
the imaging directory the way the conversion does, by locating the `cindra_parameters.json` file beneath the path and
reading the directory that holds it. Only a path whose subtree carries no source file the conversion accepts lands in
`invalid_recordings`, with a reason that names the subdirectory holding TIFF files when one exists, and it receives no
manifest and no tracker.

A recording reported under `path_conflicts` still runs, but against the paths its existing configuration records rather
than the ones you passed, because preparation never reinitializes an existing tracker. Each entry names the `recording`,
the `field`, the `stored` and `passed` values, and the `resolution`, which is to remove that recording's `cindra/`
directory before preparing it again.

A recording the sizing pass cannot measure is excluded from the batch rather than aborting it, so a run that reports
`started: true` may still cover fewer recordings than you submitted. Only the prepare tools return `total_recordings`,
so name every entry of `invalid_paths`, `invalid_recordings`, `path_conflicts`, `migrated_recordings`, and
`unsizable_recordings` rather than looking for a total to compare. A migrated recording is not rejected, but its
dispatched job set differs from the one it last carried, so report it alongside the rejections.

These lists do not share one element shape. Every job is sized from the recording's raw acquisition geometry rather
than from a per-stage allowance, so a recording with unreadable raw data loses every one of its jobs rather than the
conversion job alone. See [tool-responses.md](references/tool-responses.md) for the element shapes, the return-key
reference of the planning, execution, management, and status tools, and the terminal messages the engine writes to a
tracker.

### Execution errors

| Error Message                            | Resolution                                   |
|------------------------------------------|----------------------------------------------|
| "An execution session is already active" | Wait for current session or cancel first     |
| "Job ID not found in tracker"            | Re-prepare the batch to regenerate manifests |
| "Prerequisite ... has not succeeded"     | Execute prerequisite phases first            |

Prerequisite failures arrive inside the `invalid_jobs` list with a `reason` field rather than as a top-level `error`,
reading "Unable to execute job {job_id}. Its prerequisite '{phase}' job {prerequisite_id} has not succeeded and is not
part of this submission.", where `{phase}` is the tracker phase name, for example `registration` for a processing job.

### Processing failure routing

When processing fails for some recordings, read the error messages and route to the appropriate skill:

| Error pattern                                     | Skill to invoke                                                 |
|---------------------------------------------------|-----------------------------------------------------------------|
| Missing `cindra_parameters.json`, TIFF read error | `/acquisition-data-preparation`                                 |
| Invalid parameter values, wrong plane/channel     | `/acquisition-data-preparation`                                 |
| TIFF files hold frames of differing shapes        | `/acquisition-data-preparation`                                 |
| TIFF frames fall short of one interleave cycle    | `/acquisition-data-preparation`                                 |
| Directory holds no TIFF the conversion accepts    | Re-prepare with the subdirectory the reason names               |
| Previous write of the binary file was interrupted | Set `repeat_binarization`, reset `binarization`, re-dispatch    |
| Configuration parameter issues                    | `/single-recording-configuration`                               |
| MCP tools unavailable, server connection errors   | `/cindra-mcp-environment-setup`                                 |

Wait for the current execution session to complete before starting retries. `cancel_processing_jobs_tool` clears the
admission pool and every resource class queue, leaves already-dispatched worker processes running, and clears the
session state immediately, so a new session can start while cancelled jobs still run. Poll `get_recording_status_tool`
afterwards until no job remains RUNNING before dispatching again.

---

## Related skills

| Skill                             | Relationship                                                               |
|-----------------------------------|----------------------------------------------------------------------------|
| `/cindra-pipeline`                | Overview: end-to-end phases, handoffs, and the single-vs-multi entry point |
| `/cindra-mcp-environment-setup`   | Prerequisite: MCP server connectivity                                      |
| `/cli-reference`                  | Reference: `cindra run`, the manual counterpart this workflow replaces     |
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
- [ ] Recordings to process confirmed with user
- [ ] Raw data validated via `validate_recording_readiness_tool` (or existing binaries confirmed)
- [ ] Configuration file confirmed or created via `/single-recording-configuration`
- [ ] Output root confirmed with user (required, no default)
- [ ] Share of the machine to dedicate to processing confirmed with user
- [ ] For a device-backed batch, `check_gpu_runtime_tool` gated on `ready` and its `devices` passed as `gpu_devices`
- [ ] For a phase re-run, reset `warnings` acted on and every governing repeat flag set via `set_config_values_tool`
- [ ] Batch prepared or full pipeline executed
- [ ] Every entry of `invalid_paths`, `invalid_recordings`, `path_conflicts`, `migrated_recordings`, and
      `unsizable_recordings` reported
- [ ] Status monitored until all recordings complete or fail
- [ ] Failed recordings routed to appropriate skill (see Error routing)
- [ ] Successful recordings verified via `/single-recording-results`
```
