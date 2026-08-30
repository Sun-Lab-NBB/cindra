# Multi-recording processing tool responses

Documents the keys the preparation, planning, execution, management, and status tools return, the element shape of each
rejection list, and the terminal messages the execution engine writes to a tracker. This reference is loaded on demand
by `/multi-recording-processing`.

---

## Rejection list element shapes

The rejection lists do not share one element shape, so a caller that formats them uniformly prints `[object Object]`
for some of them.

| Key                      | Returned by            | Element shape                               |
|--------------------------|------------------------|---------------------------------------------|
| `invalid_configurations` | prepare, full-pipeline | string, the reason with the offending entry |
| `path_conflicts`         | prepare, full-pipeline | object, five string fields (see below)      |
| `unsizable_datasets`     | full-pipeline          | object, `{"dataset": str, "error": str}`    |
| `invalid_jobs`           | execute                | object, `{"job_id": str, "reason": str}`    |

`invalid_jobs` uses `{"job": str, "reason": str}` instead, keyed on the stringified descriptor, when the descriptor is
missing one of the four required keys and therefore carries no usable `job_id`.

A `path_conflicts` element carries `dataset`, `field`, `stored`, `passed`, and `resolution`, where `field` is always
`recording_io.recording_directories`, `stored` and `passed` join their output roots with commas, and `resolution` names
the dataset output directory to remove. It names a dataset the batch still runs rather than one it rejects, because
preparation never reinitializes an existing tracker and the dataset keeps using the stored recording directories.

Each list is included only when non-empty, so their absence is the success signal and `success: true` alone never is.

---

## Per-tool return keys

### prepare_multi_recording_batch_tool

Returns `datasets` keyed by the lowercased dataset name, `total_datasets`, and `total_jobs`. Each dataset entry holds
`configuration_path`, `tracker_path`, `dataset_name`, `pipeline_type`, `discover_job`, and `extract_jobs`, and every
job entry additionally carries `executor_id` when the dataset's tracker already existed.

### get_pipeline_job_universe_tool

Returns `jobs` holding the `name`, `specifier`, and `ready` flag of every declared job, plus `total_jobs`, `ready_jobs`,
`dataset_name`, `recording_ids`, `pipeline_type`, and a `resolved` flag. The recording identifiers derive from the
configured directory paths rather than from what those directories hold, so a dataset whose recordings are entirely
unprocessed still returns `resolved: true` with the full universe and `ready: false` on every job. Every configuration
the tool accepts names at least two recording directories, so `resolved` is true whenever the call succeeds and carries
no information. A configuration naming none returns `success: false` with the loader's message, so gate on `ready`
alone.

### size_pipeline_jobs_tool

Returns `jobs` holding the `name`, `specifier`, `cores`, `memory_mb`, and `device_memory_mb` of every declared job,
plus `total_jobs`, `peak_memory_mb` and `peak_device_memory_mb` for the single largest job, `total_memory_mb` for every
job at once, and `pipeline_type`. Every device figure is zero here, because no multi-recording stage runs on a CUDA
device, and the `gpu_registration` argument leaves them at zero as well. Like the universe tool, it fails when the
dataset names fewer than two recording directories, because both load through the same loader. Unlike the universe tool,
it also fails when any recording carries no combined metadata archive, which the universe tool reports as `ready: false`
instead, and when any recording reports no regions in its combined trace array, which the universe tool does not read at
all and still reports as `ready: true`.

### check_threading_runtime_tool

Returns `ready`, `platform`, `required_layer` (`omp` on macOS, `tbb` elsewhere), and a `detail` sentence. A host that is
not ready also carries one of three `remedy` commands: `sudo cindra omp --yes` when macOS holds a runtime it has not
linked, `brew install libomp` when macOS holds none, and `pip install tbb4py` off macOS. The first needs elevated
privileges, so surface it to the user rather than running it. On macOS the report adds `discovered_runtimes`, holding
the single runtime the discovery would link, and `searched_paths`, holding the candidates examined. When no runtime was
found, `discovered_runtimes` is empty while `searched_paths` still lists every candidate the discovery examined, which
is the list to surface to the user. Both are empty only when the runtime already loads, because a host that already
loads one runs no discovery.

### execute_processing_jobs_tool and execute_full_pipeline_tool

| Key                | Meaning                                                                |
|--------------------|------------------------------------------------------------------------|
| `started`          | Whether a session was dispatched. Gate on this, not on `success`       |
| `total_jobs`       | Jobs admitted into the session                                         |
| `cpu_budget`       | Session core budget, which is the host core count minus 2              |
| `memory_budget_mb` | Session memory budget, sampled once at session start                   |
| `resource_classes` | Per class, its `workers_per_job`, `max_parallel_jobs`, and `job_count` |

`execute_full_pipeline_tool` returns `pipeline_type` on every outcome, including every argument rejection and every
failure, so a caller can attribute a response to the pipeline it asked for without tracking the request itself. It
additionally returns `phase_count` and a per-phase `phases` list on every outcome that reached the phase-grouping step,
which covers a dispatched session and the already-complete case. A response whose preparation step accepted no input
carries `pipeline_type`, `total_jobs`, and the rejection lists alone. It returns `started: false` with a `message` and a
`next_step` when every phase is already complete.

`execute_processing_jobs_tool` forwards `invalid_jobs` whenever validation rejected a submitted job, which covers the
response for a session whose `workers_per_job` or `max_parallel_jobs` override was itself rejected. Read that list on a
failure rather than assuming an override rejection means every submitted job was valid.

### get_processing_jobs_status_tool

Returns `active`, `jobs`, a `summary` counting pending, running, succeeded, and failed, plus `awaiting_prerequisites`
for the jobs still held in the admission pool. Its `resource_classes` mapping carries `pending` and `active` in place
of `job_count`. Once the session drains it returns `active: false`, empty `jobs`, a zero `summary`, and a `note`.

### get_active_execution_timing_tool

Returns `jobs` with per-job timing and a `session` summary holding `total_elapsed_seconds`, `completed_count`,
`failed_count`, `running_count`, and `pending_count`. The counts span the whole session rather than one phase, so
`pending_count` is the not-yet-started remainder of the batch. `throughput_jobs_per_hour` appears only once elapsed time
and completed count are both above zero, so it is absent until the first job of the batch succeeds. A batch covering
several datasets therefore reports it while other discovery jobs still run. Its absence is not an error.

### cancel_processing_jobs_tool

Returns `canceled`, a `message`, and a `final_state` holding `succeeded_jobs`, `failed_jobs`, and
`active_jobs_at_cancel`. Cancellation empties the queues and never stops a running job, so `active_jobs_at_cancel` is
the number of jobs still executing after the call returns. Poll `get_recording_status_tool` on the affected datasets
until those jobs leave RUNNING before starting a new session. With no active session it returns `canceled: false` plus
a `note`.

### reset_processing_phases_tool

Returns `reset`, the echoed `tracker_path`, `requested_phases`, `effective_phases` after downstream expansion in
pipeline execution order, and a `jobs` list. That list is a post-reset snapshot of **every** job of every valid phase,
not only the jobs the reset touched, so selecting from it dispatches jobs that were already succeeded. Select from the
prepare manifest instead. A `warnings` list joins them when a reset phase is governed by a repeat flag that is false
while that phase's output already exists. Each entry names the dotted configuration flag to set, and a caller must act
on every entry before dispatching the reset phase.

### clean_processing_output_tool

Returns `cleaned`, `output_root`, `deleted_files`, `deleted_dirs`, `total_deleted`, `requested_phases`, and
`effective_phases`, plus `cleared_selections` counting the dataset region selections a `discovery` clean cleared, and
`errors` when a deletion failed. Both extra keys are present only when non-empty. The `cleaned` flag reports that the
tool ran rather than that every deletion succeeded, so gate on an empty `errors` list.

### get_batch_status_overview_tool

Returns `root_directory`, a `single_recordings` list, a `multi_recordings` list, and a `summary` holding
`total_single_recordings`, `total_multi_recording_datasets`, `completed`, `failed`, `in_progress`, and `not_started`.
It also returns `permission_errors` when a directory in the scanned tree could not be read. Datasets are found by
scanning for multi-recording tracker files, so an unreadable directory hides a whole dataset rather than one file.
Surface this list whenever it is present.

---

## Terminal messages the engine writes

These come from the execution engine rather than from a pipeline stage, so they name no data or configuration problem
and no upstream skill resolves them.

| Message                                                                                                                                                                                      | Cause                              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|
| `Unable to execute job. A preceding pipeline phase failed.`                                                                                                                                  | Cascade abort of an extraction job |
| `Unable to execute job. The worker process pool was terminated, which happens when a job's process is killed by the host, most often for exhausting memory.`                                 | A worker process died              |
| `Unable to execute job. The worker process pool canceled the job before any worker started it, which happens when the pool shuts down while the job is still waiting inside it.`             | Queued when the pool shut down     |
| `Unable to execute job. Its prerequisite jobs never succeeded and no queued job can still satisfy them.`                                                                                     | Session drained while job pooled   |
| `Unable to execute job. The worker process raised {type} outside the job's own error handling, which leaves the job holding no terminal state of its own. The reported reason is '{error}'.` | Worker raised outside the job      |
| `Unable to complete job. Worker terminated without reaching a terminal state.`                                                                                                               | Worker exited recording nothing    |

A job aborted because its prerequisite phase is absent from the tracker records a message naming that phase and asking
for the prepare tool to be re-run, because no phase failed in that case.

The unreachable-prerequisite message is a backstop the manager writes when a session drains while a job still waits in
the admission pool. Submitting an extraction job whose discovery job is neither in the submission nor already succeeded
does not produce it, because `execute_processing_jobs_tool` rejects that job into `invalid_jobs` before the session
starts.

Extraction is the widest class in the library at 16 cores per job and holds whole-dataset trace arrays, so it is the
class most likely to meet a pool termination. Every job runs in its own spawned process, so the kill takes down that
job alone and the engine records a terminal outcome for every job the failure strands.
