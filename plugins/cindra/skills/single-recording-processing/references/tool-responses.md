# Single-recording processing tool responses

Documents the keys the planning, execution, management, and status tools return, the element shape of each rejection
list, and the terminal messages the execution engine writes to a tracker. The prepare tools return a manifest rather
than a flat key set, which `/single-recording-processing` documents in place. This reference is loaded on demand by
that skill.

---

## Rejection list element shapes

The five lists a single-recording caller can receive do not share one element shape. Three hold plain strings and two
hold objects, so a caller that formats them uniformly prints `[object Object]` for the object ones.

| Key                    | Returned by            | Element shape                                        |
|------------------------|------------------------|------------------------------------------------------|
| `invalid_paths`        | prepare, full-pipeline | string, the offending path as supplied               |
| `invalid_recordings`   | prepare, full-pipeline | string, `"{recording_path}: {error}"`                |
| `migrated_recordings`  | prepare, full-pipeline | string, the recording key whose tracker was migrated |
| `unsizable_recordings` | full-pipeline          | object, `{"recording": str, "error": str}`           |
| `invalid_jobs`         | execute                | object, `{"job_id": str, "reason": str}`             |

`invalid_jobs` uses `{"job": str, "reason": str}` instead, keyed on the stringified descriptor, when the descriptor is
missing one of the four required keys and therefore carries no usable `job_id`.

Read every list that is present. Each is included only when non-empty, so their absence is the success signal and
`success: true` alone never is.

---

## Per-tool return keys

### get_pipeline_job_universe_tool

Returns `jobs` holding the `name`, `specifier`, and `ready` flag of every declared job, plus `total_jobs`, `ready_jobs`,
`plane_count`, `pipeline_type`, and a `resolved` flag. A recording carrying nothing returns `resolved: false` with an
empty `jobs` list and `success: true`, because the resolver reports absence rather than failing.

### size_pipeline_jobs_tool

Returns `jobs` holding the `name`, `specifier`, `cores`, and `memory_mb` of every declared job, plus `total_jobs`,
`peak_memory_mb` for the single largest job, `total_memory_mb` for every job at once, and `pipeline_type`. Unlike the
universe tool, this one fails when the recording's raw imaging data cannot be read, because no stage of it could run.

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
| `total_jobs`       | Jobs the session holds, including those still awaiting prerequisites   |
| `cpu_budget`       | Session core budget, which is the host core count minus 2              |
| `memory_budget_mb` | Session memory budget, sampled once at session start                   |
| `resource_classes` | Per class, its `workers_per_job`, `max_parallel_jobs`, and `job_count` |

`execute_full_pipeline_tool` additionally returns `phase_count`, a per-phase `phases` list, and the preparation lists
it forwards, including `migrated_recordings`. It returns `started: false` with a `message` and a `next_step` when every
phase is already complete.

### get_processing_jobs_status_tool

Returns `active`, `jobs`, a `summary` counting pending, running, succeeded, and failed, plus `awaiting_prerequisites`
for the jobs still held in the admission pool. Its `resource_classes` mapping carries `pending` and `active` in place
of `job_count`. Once the session drains it returns `active: false`, empty `jobs`, a zero `summary`, and a `note`.

### get_active_execution_timing_tool

Returns `jobs` with per-job timing and a `session` summary holding `total_elapsed_seconds`, `completed_count`,
`failed_count`, `running_count`, and `pending_count`. The counts span the whole session rather than one phase, so
`pending_count` is the not-yet-started remainder of the batch. `throughput_jobs_per_hour` appears only once elapsed time
and completed count are both above zero, so it is absent until the first job of the batch succeeds, which can happen
while the rest of that same phase still runs. Its absence is not an error.

### cancel_processing_jobs_tool

Returns `canceled`, a `message`, and a `final_state` holding `succeeded_jobs`, `failed_jobs`, and
`active_jobs_at_cancel`. Cancellation empties the queues and never stops a running job, so `active_jobs_at_cancel` is
the number of jobs still executing after the call returns. Poll `get_recording_status_tool` on the affected recordings
until those jobs leave RUNNING before starting a new session. With no active session it returns `canceled: false` plus
a `note`.

### reset_processing_phases_tool

Returns `reset`, `requested_phases`, `effective_phases` after downstream expansion in pipeline execution order, and a
`jobs` list. That list is a post-reset snapshot of **every** job of every valid phase, not only the jobs the reset
touched, so selecting from it dispatches jobs that were already succeeded. Select from the prepare manifest instead.

### clean_processing_output_tool

Returns `cleaned`, `recording_path`, `deleted_files`, `deleted_dirs`, `total_deleted`, `requested_phases`, and
`effective_phases`, plus `errors` when a deletion failed. The `cleaned` flag reports that the tool ran rather than that
every deletion succeeded, so gate on an empty `errors` list.

### get_batch_status_overview_tool

Returns `root_directory`, a `single_recordings` list, a `multi_recordings` list, and a `summary` holding
`total_single_recordings`, `total_multi_recording_datasets`, `completed`, `failed`, `in_progress`, and `not_started`.
It also returns `permission_errors` when a directory in the scanned tree could not be read. The scan then covers less of
the tree than it appears to, so a recording missing from the overview may be unreadable rather than unprocessed. Surface
this list whenever it is present.

### get_recording_status_tool

Returns `recording_path`, a `single_recording` section, and a `multi_recording` section. Each section reports
`status: "not_started"` when no tracker exists. The single-recording section carries `tracker_path`, a synthesized
`status`, a `summary` of per-status counts, and a `jobs` mapping whose four keys are `binarize`, `register`, `process`,
and `combine`. The `register` and `process` keys map each `plane_{index}` specifier to a lowercased status string.

---

## Terminal messages the engine writes

These come from the execution engine rather than from a pipeline stage, so they name no data or configuration problem
and no upstream skill resolves them.

| Message                                                                                                                                                                                      | Cause                             |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| `Unable to execute job. A preceding pipeline phase failed.`                                                                                                                                  | Cascade abort of a downstream job |
| `Unable to execute job. The worker process pool was terminated, which happens when a job's process is killed by the host, most often for exhausting memory.`                                 | A worker process died             |
| `Unable to execute job. The worker process pool canceled the job before any worker started it, which happens when the pool shuts down while the job is still waiting inside it.`             | Queued when the pool shut down    |
| `Unable to execute job. Its prerequisite jobs never succeeded and no queued job can still satisfy them.`                                                                                     | Session drained while job pooled  |
| `Unable to execute job. The worker process raised {type} outside the job's own error handling, which leaves the job holding no terminal state of its own. The reported reason is '{error}'.` | Worker raised outside the job     |
| `Unable to complete job. Worker terminated without reaching a terminal state.`                                                                                                               | Worker exited recording nothing   |

A job aborted because its prerequisite phase is absent from the tracker records a message naming that phase and asking
for the prepare tool to be re-run, because no phase failed in that case.

The unreachable-prerequisite message is the ordinary outcome of submitting a job whose prerequisite was neither part of
the submission nor already succeeded, so treat it as a submission-shape problem rather than a data one.

Route a pool termination to memory. The engine records the same message for every job the pool stranded and cannot name
the process the host killed, so compare `size_pipeline_jobs_tool`'s `peak_memory_mb` against the host's free memory,
then reduce the batch or free memory before re-running.
