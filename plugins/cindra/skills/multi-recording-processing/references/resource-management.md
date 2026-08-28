# Multi-recording resource management and planning

Documents the per-class core arithmetic the execution engine applies, the override semantics of `workers_per_job` and
`max_parallel_jobs`, and what each planning tool reports before a batch is dispatched. This reference is loaded on
demand by `/multi-recording-processing`.

---

## Per-class core allocation

Discovery's 2 cores cover the deformation pool alone. The stage has no parallel critical path, so quadrupling the
allocation shortens a twenty-recording dataset by two percent. Extraction's 16 follows the concurrency a compute node
sustains rather than a measured plateau, leaving room for the six to eight datasets a node extracts at once while still
reaching a sevenfold single-job speedup.

Both classes are elastic, so a session that leaves `workers_per_job` as None widens each job it dispatches over the
cores no running job holds, bounded below by the class default and above by the dispatch ceiling. Those free cores
divide among the elastic classes holding queued work before the share divides among the jobs, so a full queue resolves
to the class default while a queue holding one job resolves toward the ceiling. An explicit `workers_per_job` reaches
every job unchanged, and the `resource_classes` mapping reports the smallest width.

Each class caps its own concurrency at `min(max(1, budget // cores_per_job), max(1, job_count))`. The discover phase
contributes one job per dataset, and the extract phase contributes one job per recording. On a 128-core host, with a
budget of 126, discovery's budget-derived bound is 63 jobs, so its dataset count binds first in any realistic batch,
and extraction runs at most 7.

No cap here is a fixed number the host outgrows. Both derive from the session core budget, so a wider host raises them
on its own, and the dispatcher then admits against the live core and memory budgets rather than against the cap alone.
The engine saturates the host it is given, so leave both parameters as None unless the user asks for an override.

Extraction holds its 16 workers on a small host too. A 16-core machine still asks for 16 against a budget of 14 and
dispatches one job regardless, because the dispatcher always admits a single job even when the budget cannot cover it.

Memory bounds dispatch separately from both class caps. Each job is estimated from the dataset it will process, and the
dispatcher holds the sum of the running jobs' estimates inside the session memory budget, reported as
`memory_budget_mb`. That budget is the host's available memory sampled once when the session starts and never re-read.
Every job runs in its own spawned process, so a job the host kills for exhausting memory takes down its own worker
rather than the batch, and the engine records a terminal outcome for every job the failure strands.

---

## Override semantics

Both `workers_per_job` and `max_parallel_jobs` default to None and can be overridden in `execute_processing_jobs_tool`
or `execute_full_pipeline_tool`. A positive value of either is used exactly. Setting `workers_per_job` to -1 gives every
job the whole session core budget, while setting `max_parallel_jobs` to -1 lifts the derived cap so that only the job
count bounds concurrency. An override is a single scalar applied to every non-fixed class, so passing
`workers_per_job=20` sets both discovery and extraction to 20. `max_parallel_jobs` is a per-class cap rather than a
session ceiling, so `max_parallel_jobs=4` permits 4 discovery jobs and 4 extraction jobs at once, and the session CPU
and memory budgets remain the only terms bounding the classes in aggregate. Both execute tools return the session-level
`cpu_budget` and `memory_budget_mb`, and a `resource_classes` mapping keyed by class name, with `discovery` and
`extraction` entries carrying `workers_per_job`, `max_parallel_jobs` and `job_count`. `get_processing_jobs_status_tool`
returns the same mapping with `pending` and `active` in place of `job_count`, and adds a session-level
`awaiting_prerequisites` count of the jobs still held in the admission pool.

---

## Planning tool responses

`get_pipeline_job_universe_tool` and `size_pipeline_jobs_tool` both need a resolved per-dataset configuration rather
than a template. Each loads the file through the loader the pipeline uses, which rejects a configuration naming fewer
than two `recording_io.recording_directories` or holding an empty `recording_io.dataset_name`. A generated template
holds neither, so a template makes both return `success: false` carrying the loader's message rather than a plan. Run
them against the `multi_recording_configuration.yaml` that `prepare_multi_recording_batch_tool` writes into a dataset's
output directory, or fill both fields on a template first with `set_config_values_tool`. Preparation is idempotent and
starts no computation, so planning a multi-recording batch means preparing it first and dispatching second.

`get_pipeline_job_universe_tool` answers which jobs can run right now. It reads the inventory the output roots already
hold rather than any tracker. The discovery job is ready once every recording carries its single-recording output, and
an extraction job once discovery has written the template masks its recording projects. The recording
identifiers come from the configured directory paths rather than from what those directories hold, so a wholly
unprocessed dataset still reports `resolved: true` with the full universe and `ready: false` on every job. Every
configuration the tool accepts names at least two directories, so `resolved` is true whenever the call succeeds and
carries no information. Gate on `ready` alone to decide what to dispatch. Use it to plan a selective re-run, and
`get_recording_status_tool` to read recorded outcomes once a batch has been prepared.

`size_pipeline_jobs_tool` reports the cores and memory every job of a dataset holds, reading the completed
single-recording output that underlies the dataset. Pass the dataset's configuration path and
`pipeline_type="multi-recording"`. The response lists each job's `name`, `specifier`, `cores`, `memory_mb`, and a
`device_memory_mb` of zero, because no multi-recording stage runs on a CUDA device. It also carries `peak_memory_mb` for
the single largest job and `total_memory_mb` for every job at once. Read `peak_memory_mb` rather than assuming which
stage dominates, because discovery's clustering term grows with the square of the region count while extraction's trace
arrays grow with the frame count, so either stage leads depending on the dataset.

`check_threading_runtime_tool` reports whether the numeric threading layer this host needs is loadable, which is OpenMP
on macOS and TBB elsewhere. Gate a batch on its `ready` flag. A macOS host that is not ready aborts every job at the
pipeline entry point before any stage runs, while a non-macOS host missing TBB fails at the job's first parallelized
call. Neither outcome returns a tool error, so both surface only as per-job tracker failures.
