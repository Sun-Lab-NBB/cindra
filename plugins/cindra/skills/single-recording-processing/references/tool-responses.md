# Single-recording processing tool responses

Documents the planning that precedes a dispatch, the keys the planning, execution, management, and status tools return,
the element shape of each rejection list, and the terminal messages the execution engine writes to a tracker. The
prepare tools return a manifest rather than a flat key set, which `/single-recording-processing` documents in place.

---

## Rejection list element shapes

The six lists a single-recording caller can receive do not share one element shape. Three hold plain strings and three
hold objects, so a caller that formats them uniformly prints `[object Object]` for the object ones.

| Key                    | Returned by            | Element shape                                                      |
|------------------------|------------------------|--------------------------------------------------------------------|
| `invalid_paths`        | prepare, full-pipeline | string, the offending path as supplied                             |
| `invalid_recordings`   | prepare, full-pipeline | string, `"{raw_data_path}: {error}"`                               |
| `migrated_recordings`  | prepare, full-pipeline | string, the raw data path whose tracker was migrated               |
| `path_conflicts`       | prepare, full-pipeline | object, `{"recording", "field", "stored", "passed", "resolution"}` |
| `unsizable_recordings` | full-pipeline          | object, `{"recording": str, "error": str}`                         |
| `invalid_jobs`         | execute                | object, `{"job_id": str, "reason": str}`                           |

A `path_conflicts` entry holds five string values, and its `field` is `file_io.data_path` or `file_io.output_path`,
naming which of the two stored paths disagrees with the one passed. One entry is emitted per disagreeing field, so a
recording whose stored raw data path and output path both differ contributes two entries.

`invalid_jobs` uses `{"job": str, "reason": str}` instead, keyed on the stringified descriptor, when the descriptor is
missing one of the four required keys and therefore carries no usable `job_id`.

Read every list that is present. Each is included only when non-empty, so their absence is the success signal and
`success: true` alone never is.

---

## Planning before dispatch

The two planning tools that read a configuration load it through the loader the pipeline uses. That loader rejects a
file whose `file_io.output_path` is None, reporting "The output_path must be configured in the FileIO section of the
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
registration jobs report their device figures. `planned_roi_count` names the regions to plan for across every plane of
the recording, and None accepts the ceiling the detection iteration bound provides. A known count sizes the processing
and combination figures for the regions the recording will actually hold. The response lists each job's `name`,
`specifier`, `cores`, `memory_mb`, and `device_memory_mb`, plus `peak_memory_mb` and `peak_device_memory_mb` for the
single largest job and `total_memory_mb` for every job at once. Compare `peak_memory_mb` against the host's free memory
to learn whether the largest job fits at all, and `total_memory_mb` to learn whether the whole batch could ever run
concurrently. These are the figures the execute tools charge against the session memory budget, so a batch whose peak
exceeds free memory admits its jobs serially rather than failing.

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

## Per-tool return keys

### get_pipeline_job_universe_tool

Returns `jobs` holding the `name`, `specifier`, and `ready` flag of every declared job, plus `total_jobs`, `ready_jobs`,
`plane_count`, `pipeline_type`, and a `resolved` flag. A recording carrying nothing returns `resolved: false` with an
empty `jobs` list and `success: true`, because the resolver reports absence rather than failing.

### size_pipeline_jobs_tool

Returns `jobs` holding the `name`, `specifier`, `cores`, `memory_mb`, and `device_memory_mb` of every declared job, plus
`total_jobs`, `peak_memory_mb` and `peak_device_memory_mb` for the single largest job, `total_memory_mb` for every job
at once, and `pipeline_type`. No device total is reported, because the device count rather than a shared pool bounds the
jobs that hold a device at once. `gpu_registration=True` plans the registration jobs for a CUDA device, which reports 2
cores in place of 4, raises their `memory_mb` by the page-locked host buffers the device staging holds, and fills their
`device_memory_mb`. Every other job reports a `device_memory_mb` of zero. Unlike the universe tool, this one fails when
the recording's raw imaging data cannot be read, because no stage of it could run.

### check_threading_runtime_tool

Returns `ready`, `platform`, `required_layer` (`omp` on macOS, `tbb` elsewhere), and a `detail` sentence. A host that is
not ready also carries one of three `remedy` commands: `sudo cindra omp --yes` when macOS holds a runtime it has not
linked, `brew install libomp` when macOS holds none, and `pip install tbb4py` off macOS. The first needs elevated
privileges, so surface it to the user rather than running it. On macOS the report adds `discovered_runtimes`, holding
the single runtime the discovery would link, and `searched_paths`, holding the candidates examined. When no runtime was
found, `discovered_runtimes` is empty while `searched_paths` still lists every candidate the discovery examined, which
is the list to surface to the user. Both are empty only when the runtime already loads, because a host that already
loads one runs no discovery.

### check_gpu_runtime_tool

Returns `ready`, the `status` naming the outcome, a `detail` sentence, a `device_count`, and a `devices` list holding
the `index`, `name`, `total_memory_mb`, and `compute_capability` of every usable device. A host that is not ready
carries an empty `devices` list and a `remedy` naming the CuPy installation. macOS carries no remedy, because the CuPy
project publishes no wheel for it. The `index` values are what `gpu_devices` takes.

### execute_processing_jobs_tool and execute_full_pipeline_tool

| Key                | Meaning                                                                |
|--------------------|------------------------------------------------------------------------|
| `started`          | Whether a session was dispatched. Gate on this, not on `success`       |
| `total_jobs`       | Jobs the session holds, including those still awaiting prerequisites   |
| `cpu_budget`       | Session core budget, which is the host core count minus 2              |
| `memory_budget_mb` | Session memory budget, sampled once at session start                   |
| `gpu_devices`      | The CUDA device indices the session holds, empty for a host-CPU run    |
| `resource_classes` | Per class, its `workers_per_job`, `max_parallel_jobs`, and `job_count` |

The `gpu_devices` argument both tools take carries three cases. None registers on the host CPU, `[-1]` names every
device the host exposes, and an explicit list names those devices. An empty list is rejected, `-1` cannot be paired
with an index, and an index the host does not expose is rejected against the indices it does.

`execute_full_pipeline_tool` returns `pipeline_type` on every outcome, including every argument rejection and every
failure, so a caller can attribute a response to the pipeline it asked for without tracking the request itself. It
additionally returns `phase_count`, a per-phase `phases` list, and the preparation lists it forwards, including
`migrated_recordings` and `path_conflicts`, on every outcome reached after the arguments validate. It returns
`started: false` with a `message` and a `next_step` when every phase is already complete.

`execute_processing_jobs_tool` forwards `invalid_jobs` whenever validation rejected a submitted job, which covers the
response for a session whose `workers_per_job` or `max_parallel_jobs` override was itself rejected. Read that list on a
failure rather than assuming an override rejection means every submitted job was valid.

### get_processing_jobs_status_tool

Returns `active`, `jobs`, a `summary` counting pending, running, succeeded, and failed, plus `awaiting_prerequisites`
for the jobs still held in the admission pool. Its `resource_classes` mapping carries `pending` and `active` in place of
`job_count`. Each `jobs` entry carries the `tracker_path` that owns its `job_id`, because a `job_id` identifies a job
only within its own tracker. Passing `summary_only: true` omits the `jobs` list and returns the session fields and the
counts alone. Poll a wide batch that way, because the list grows with the job count while the counts it summarizes do
not. Once the session drains it returns `active: false`, empty `jobs`, a zero `summary`, and a `note`.

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

Returns `reset`, `tracker_path`, `requested_phases`, `effective_phases` after downstream expansion in pipeline
execution order, and a `jobs` list. That list is a post-reset snapshot of **every** job of every valid phase, not only
the jobs the reset touched, so selecting from it dispatches jobs that had already succeeded. Select from the prepare
manifest instead. A `warnings` list of sentences is present when a reset phase is governed by a repeat flag that is
false while that phase's output already exists on disk, and each sentence names the dotted flag to set with
`set_config_values_tool`. Act on every warning before dispatching the reset phase, because the stage otherwise returns
immediately and records success without redoing its work. The list is empty when the configuration cannot be read.

### clean_processing_output_tool

Returns `cleaned`, `output_root`, `deleted_files`, `deleted_dirs`, `total_deleted`, `requested_phases`, and
`effective_phases`, plus `errors` when a deletion failed. The `cleaned` flag reports that the tool ran rather than that
every deletion succeeded, so gate on an empty `errors` list.

### get_batch_status_overview_tool

Returns `root_directory`, a `single_recordings` list, a `multi_recordings` list, and a `summary` holding
`total_single_recordings`, `total_multi_recording_datasets`, `completed`, `failed`, `in_progress`, and `not_started`.
It also returns `permission_errors` when a directory in the scanned tree could not be read. The scan then covers less of
the tree than it appears to, so a recording missing from the overview may be unreadable rather than unprocessed. Surface
this list whenever it is present.

### get_recording_status_tool

Returns `output_root`, a `single_recording` section, and a `multi_recording` section. Each section reports
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
