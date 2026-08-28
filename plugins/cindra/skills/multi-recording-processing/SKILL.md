---
name: multi-recording-processing
description: >-
  Orchestrates multi-recording neural imaging batch processing via the cindra MCP server, dispatching to configuration
  and results skills as needed. Use when the user asks to run multi-recording (cross-day ROI tracking) processing,
  process multiple recordings as a tracked dataset, monitor multi-recording batch jobs, re-run a multi-recording
  processing phase, or when invoking /multi-recording-processing.
user-invocable: true
---

# Multi-recording processing

Orchestrates the multi-recording batch processing workflow: verify prerequisites, organize recordings by dataset,
prepare execution manifests, dispatch jobs, monitor progress, and hand off to downstream skills for output verification.

---

## Scope

**Covers:**
- Batch processing workflow: prerequisite verification, dataset organization, preparation, execution, monitoring, and
  completion
- The MCP preparation, execution, management, and supporting tools listed in the Available tools section
- Dataset name resolution via `resolve_dataset_name_tool`
- Resource management and CPU allocation guidance
- Status formatting and progress monitoring
- Error routing to appropriate upstream skills

**Does not cover:**
- Configuration parameters, tuning guidance, or config file creation (see `/multi-recording-configuration`)
- Output data formats, array shapes, dtypes, file references, or data interpretation (see `/multi-recording-results`)
- Single-recording processing workflow or prerequisites (see `/single-recording-processing`)
- Input data format, TIFF requirements, or acquisition parameters (see `/acquisition-data-preparation`)
- MCP server connectivity or environment issues (see `/cindra-mcp-environment-setup`)
- Visual inspection of results (see `/visualization`)

**Handoff rules:** If the user asks about specific output files, array shapes, data interpretation, registration arrays,
tracking templates, or processing result verification, invoke `/multi-recording-results`. If the user asks about
parameter tuning, registration/tracking configuration, or ROI selection criteria, invoke
`/multi-recording-configuration`.

---

## Agent requirements

You MUST use the cindra MCP tools for all processing operations. Do not import cindra Python functions directly or run
processing via scripts or CLI commands. If MCP tools are not available, invoke `/cindra-mcp-environment-setup` to
diagnose and resolve connectivity issues.

---

## Prerequisites

An incomplete recording routes to the earliest missing step in the chain `/acquisition-data-preparation` →
`/single-recording-configuration` → `/single-recording-processing`. Workflow step 1 states the check itself.

---

## Available tools

### Preparation tools

| Tool                                 | Purpose                                                                   |
|--------------------------------------|---------------------------------------------------------------------------|
| `get_pipeline_job_universe_tool`     | Reports every job a dataset declares and which can run right now          |
| `size_pipeline_jobs_tool`            | Reports the cores and memory every job holds, before dispatching anything |
| `check_threading_runtime_tool`       | Reports whether the numeric threading layer this host needs is loadable   |
| `prepare_multi_recording_batch_tool` | Prepares execution manifest without starting execution (idempotent)       |
| `execute_full_pipeline_tool`         | Convenience: prepares and executes all phases with automatic sequencing   |

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

`/multi-recording-configuration` owns the first four. Invoke that skill for their parameters and usage guidance.

| Tool                        | Purpose                                                          |
|-----------------------------|------------------------------------------------------------------|
| `discover_recordings_tool`  | Discovers single and multi-recording candidates under a root dir |
| `generate_config_file_tool` | Generates default multi-recording configuration YAML             |
| `resolve_dataset_name_tool` | Constructs qualified dataset names from base name + specifier    |
| `set_config_values_tool`    | Writes new values into an existing configuration file            |
| `get_recording_status_tool` | Checks single and multi-recording processing status              |

---

## Pipeline architecture

Two-phase pipeline per dataset:

```text
Phase 1: DISCOVER (phase name: discovery, CPU bound, parallel by dataset)
├── Selects/filters ROIs from each recording's single-recording outputs
├── Registers all recordings to common reference frame
├── Clusters ROI masks across recordings
├── Generates template masks for tracked ROIs
├── Projects template masks back to each recording's coordinate system
└── One job per dataset, workers from the discovery resource class (2 per job by default, see Resource management)

Phase 2: EXTRACT (phase name: extraction, CPU bound, parallel by recording)
├── Applies template masks to extract fluorescence
├── Computes neuropil signals, spike deconvolution
└── One job per recording, workers from the extraction resource class (16 per job by default, see Resource management)
```

---

## Dataset name resolution

Each dataset in a batch needs a unique `dataset_name` for output directories and batch tracking. The
`resolve_dataset_name_tool` constructs qualified names by combining a shared base name with a batch-specific specifier:

```text
resolve_dataset_name_tool(
    dataset_name="learning_task",           # shared analysis name from user
    output_roots=["/data/animal_A/rec1", "/data/animal_A/rec2"],
    specifier=""                            # auto-derived from common parent → "animal_a"
)
→ { "dataset_name": "animal_a_learning_task", "specifier": "animal_a", "base_name": "learning_task" }
```

`output_roots` holds the pipeline output roots of the completed single-recording runs, each the parent of that
recording's `cindra/` directory. The tool is a pure string computation that writes no file. The name it returns
reaches a configuration only when you pass it as a dataset's `dataset_name` to `prepare_multi_recording_batch_tool`,
which writes the lowercased name into `recording_io.dataset_name` of the resolved per-dataset copy. To set the field on
a template directly, use `set_config_values_tool`, owned by `/multi-recording-configuration`.

**Specifier derivation strategies:**
- **Auto (default):** Derived from the deepest common parent directory of the output roots.
- **Explicit:** The user or agent provides a specifier directly (e.g., brain region, session group).
- **Semantic:** The agent determines the specifier by analyzing recording directory names or paths.

This enables batch bootstrapping: discover all recordings under a project directory, group them by common parent, and
call `resolve_dataset_name_tool` once per group to generate unique dataset names.

---

## Processing workflow

### Execution model

The processing workflow uses a **prepare-then-execute** model:

1. **Prepare** creates an execution manifest (tracker files, per-dataset configurations, job lists) without starting any
   computation. This step is idempotent, so calling it again on the same datasets returns the existing manifest.

2. **Execute** dispatches jobs from the manifest with prerequisite validation, resource allocation, and automatic phase
   sequencing. Only one execution session can be active at a time.

For simple cases, `execute_full_pipeline_tool` combines both steps into a single call with automatic phase advancement.
For fine-grained control (e.g., running only specific phases, custom resource allocation, or selective re-runs), use
`prepare_multi_recording_batch_tool` followed by `execute_processing_jobs_tool`.

**STOP**: Steps 1 through 5 below are the entry conditions for dispatch. Complete every one of them, and confirm each
against the verification checklist at the end of this skill, before calling any execute tool.

### Workflow steps

1. **Verify prerequisites.** Use `discover_recordings_tool`, documented in `/multi-recording-configuration`, to find
   eligible recordings and `get_recording_status_tool` to confirm each has single-recording status `completed`. Each
   `multi_recording_candidates` entry is an object, so read `candidate["output_root"]` for both calls and carry that
   value through the rest of this workflow. If any recording is incomplete, invoke `/single-recording-processing` (or
   upstream skills as needed).

2. **Organize into datasets.** Group output roots by common parent directory, user-provided grouping, or semantic
   analysis of their paths. Each group becomes one dataset in the batch.

3. **Resolve dataset names.** Ask the user for a shared base dataset name (e.g., "learning_task"). For each group, call
   `resolve_dataset_name_tool` with the base name and the group's `output_roots` to generate a unique qualified name.
   The specifier is derived automatically from the common parent directory, or the user can provide one explicitly.

4. **Configure.** Ask the user if they have an existing template configuration file. If not, invoke
   `/multi-recording-configuration` to create one. Template configs are reusable across datasets and live at user-chosen
   locations (e.g., `/Data/CA1_GCaMP6f_MD.yaml`). The template's `dataset_name` only needs to be a non-empty,
   filesystem-safe string to pass validation, because the prepare tool overwrites it per dataset as the Dataset name
   resolution section states. Do NOT create per-dataset config copies. The prepare tool automatically saves resolved
   copies as `multi_recording_configuration.yaml` inside each dataset's output directory, preserving the original
   template. Pass the same template path for multiple datasets that share parameters.

5. **Confirm the machine budget.** The engine resolves the per-class worker counts and concurrency caps itself, so do
   not ask the user to choose them and do not build an allocation table before dispatching. Ask only how much of the
   machine this run may take, because that is the one allocation question the engine cannot answer: it claims every core
   but two and the memory available when the session starts. Confirm that the host is free for the run, or take an
   explicit ceiling from the user and pass it as `max_parallel_jobs`, which the Resource management section covers.
   After dispatch, report the resolved allocation from the `resource_classes` mapping the execute tool returns, which
   for 2 datasets of 15 recordings on a 128-core host reads:

   ```text
   Resource class | Jobs | Workers/Job | Max Parallel | Total Cores
   ---------------|------|-------------|--------------|------------
   discovery      |    2 |           2 |            2 |           4
   extraction     |   30 |          16 |            7 |         112
   ```

6. **Execute.** Choose one of two approaches:

   **Simple (recommended for straightforward runs):**
   Call `execute_full_pipeline_tool` with `pipeline_type="multi-recording"` and `dataset_configurations` containing each
   dataset's `configuration_path`, `output_roots`, and `dataset_name`. This prepares and executes all phases
   automatically.

   **Fine-grained (for selective execution or re-runs):**
   a. Call `prepare_multi_recording_batch_tool` with the dataset configurations, each carrying the same three keys.
      This returns a manifest with job IDs and statuses, plus `path_conflicts` when a dataset was already prepared
      against output roots other than the ones you passed.
   b. Select the jobs to execute from the manifest (e.g., only SCHEDULED jobs, only specific phases).
   c. Call `execute_processing_jobs_tool` with the selected job descriptors and worker settings. Each job descriptor
      needs `configuration_path`, `tracker_path`, `job_id`, and `pipeline_type` from the manifest.

7. **Monitor.** Use `get_processing_jobs_status_tool` to check progress, passing `summary_only=True` to poll a batch of
   many extraction jobs. That flag drops the per-job `jobs` list and returns the session fields and summary counts
   alone. Optionally use `get_active_execution_timing_tool` for per-job timing and session throughput. These two tools
   reflect only the active in-process execution session and return `active: false` with empty jobs when no session is
   running. This drained state happens not only after an MCP server restart, a reconnect, or a batch dispatched by a
   prior process, but also after NORMAL completion: the manager clears session state on success AND on failure. So an
   all-zero, inactive status can mean "finished," not "nothing ran." Do not read it as failure. For final per-job
   outcomes, read persisted on-disk tracker state via `get_batch_status_overview_tool` for a whole-tree view,
   `get_recording_status_tool` per recording, or `verify_multi_recording_output_tool` (all using the output root, see
   the Output-root path rule). Present status as a formatted table (see Status formatting section).

8. **Handle completion.** When all datasets finish, check for failures. A `success: true` return only means a tool ran,
   not that work is ready or done: gate decisions on the domain flag, not on `success`. For
   `verify_multi_recording_output_tool`, gate on `complete` (false whenever `missing` is non-empty). For validate tools,
   gate on `valid`. For `execute_full_pipeline_tool`, gate on `started` (it returns `started: false` with a `next_step`
   when all phases are already complete). Checking `success` alone can advance on an unready or already-complete state.
   Route errors to the appropriate skill (see Error routing section). On success, invoke `/multi-recording-results` to
   verify outputs, then `/visualization` for visual inspection.

### Output-root path rule

`get_recording_status_tool`, `verify_multi_recording_output_tool`, and `clean_processing_output_tool` all name their
argument `output_root` and take the pipeline output root (the parent of the `cindra/` folder), which equals the
`output_roots` entries passed to the prepare tools. It is not the `raw_data_path` holding the recording's TIFF files,
and the distinction matters on a separate-output layout where the two roots differ:

- `get_recording_status_tool` and `clean_processing_output_tool` resolve `cindra/` directly under the given path with NO
  fallback. Feeding the raw-data path makes them report `not_started` or "output root not found", a silent false
  negative.
- `verify_multi_recording_output_tool` also recursively searches for `configuration.yaml`, so it may still pass via that
  fallback even when fed the wrong root. The two then disagree.

Always reuse the output root that single-recording processing used, its `output_root` entry in the
`prepare_single_recording_batch_tool` manifest, for status, verify, and clean. The multi-recording prepare manifest
exposes no `output_root` field: each dataset entry holds only `configuration_path`, `tracker_path`, `dataset_name`,
`pipeline_type`, `discover_job`, and `extract_jobs`. Every job entry additionally carries `executor_id` when the
dataset's tracker already existed.

### Re-running specific phases

1. Use `reset_processing_phases_tool` with `tracker_path`, `phases`, and `pipeline_type="multi-recording"` to reset the
   target phases to SCHEDULED status. Downstream phases are automatically reset (e.g., resetting `discovery` also resets
   `extraction`).
2. Optionally modify the configuration file with `set_config_values_tool` before re-execution.
3. Use `clean_processing_output_tool` to delete output files from the reset phases (requires `output_root`,
   `pipeline_type="multi-recording"`, and the lowercased `dataset` name).
4. Call `execute_processing_jobs_tool` with the reset jobs from the manifest.

**Discovery is not idempotent under a reset alone.** A reset returns the phase to SCHEDULED and deletes nothing, so
each substage re-reads the output the previous run left and returns early.

| Sub-stage           | Skips while                                                   | Forced by             |
|---------------------|---------------------------------------------------------------|-----------------------|
| ROI selection       | Channel 1 selections exist, and channel 2 when it has data    | `repeat_selection`    |
| Registration        | Every recording reports itself registered                     | `repeat_registration` |
| Tracking            | `tracking_template_masks.npz` exists in the first output path | `repeat_registration` |
| Template projection | Every recording carries `roi_statistics.npz`                  | `repeat_registration` |

Such a re-run reports SUCCEEDED while writing byte-identical output, so the new parameter never takes effect. Clean
the `discovery` phase, which deletes the template masks, or set `repeat_selection` in `recording_io` and
`repeat_registration` in `diffeomorphic_registration` with `set_config_values_tool`, all owned by
`/multi-recording-configuration`.

`reset_processing_phases_tool` detects the ROI-selection case itself. It returns a `warnings` list naming
`recording_io.repeat_selection` whenever it resets `discovery` while that flag is false and the dataset already carries
selected ROIs. Act on every warning before dispatching the reset phase, because the key is absent when nothing would
skip and its presence is the only signal that a reset alone accomplishes nothing.

---

## Resource management

Discovery and extraction run under separate resource classes, each carrying its own measured per-job worker count. The
session CPU budget is `cpu_count - 2`, with 2 cores reserved for system operations. The dispatcher holds the sum of the
cores committed by every class inside that budget, so the two classes interleave rather than each claiming the whole
budget.

| Phase    | Resource class | Cores per job | Dispatch ceiling | Concurrency cap    |
|----------|----------------|---------------|------------------|--------------------|
| DISCOVER | `discovery`    | 2             | 8                | Session CPU budget |
| EXTRACT  | `extraction`   | 16            | 32               | Session CPU budget |

`Cores per job` is the smallest width a class gives a job and `Dispatch ceiling` is the largest. Both classes are
elastic, so a session that leaves `workers_per_job` as None widens each job it dispatches over the cores no running job
holds, bounded by those two columns. Memory bounds dispatch separately from the core budget, estimated per job from the
dataset it will process and held inside the session memory budget the execute tools report as `memory_budget_mb`.

No cap here is a fixed number the host outgrows, because every one of them derives from the session core budget. The
engine saturates the host it is given, so leave `workers_per_job` and `max_parallel_jobs` as None unless the user asks
for an override. See [resource-management.md](references/resource-management.md) for the per-class arithmetic, the
override semantics of both parameters, and the `resource_classes` mapping the execute and status tools return.

### Planning before dispatch

Four tools answer what a batch will do before it runs. `get_pipeline_job_universe_tool` reports which jobs can run right
now, `size_pipeline_jobs_tool` reports the cores and memory each job holds, `check_threading_runtime_tool` reports
whether this host carries the numeric threading layer its platform needs, and `get_recording_status_tool` reads the
recorded outcomes once a batch has been prepared.

The first two need a resolved per-dataset configuration rather than a template, because the loader rejects a
configuration naming fewer than two `recording_io.recording_directories` or holding an empty
`recording_io.dataset_name`. Run them against the `multi_recording_configuration.yaml` that
`prepare_multi_recording_batch_tool` writes into a dataset's output directory, or fill both fields on a template first
with `set_config_values_tool`. Preparation is idempotent and starts no computation, so planning a multi-recording batch
means preparing it first and dispatching second.

Gate what you dispatch on the per-job `ready` flag of `get_pipeline_job_universe_tool` rather than on its `resolved`
flag, which carries no information here. Gate the batch itself on the `ready` flag of `check_threading_runtime_tool`.
See [resource-management.md](references/resource-management.md) for the keys each of these tools returns and how to
read them.

---

## Status formatting

When presenting batch status to the user, format as a table:

```text
**Multi-Recording Batch Processing Status**

Current Phase: EXTRACT
Summary: 1/2 datasets complete | 2/4 recordings extracted | 0 failed

| Dataset                | Discover | Extract Progress | Status     |
|------------------------|----------|------------------|------------|
| animal_a_learning_task | done     | 2/2              | COMPLETED  |
| animal_b_learning_task | done     | 0/2              | EXTRACTING |
```

---

## Error routing

### Preparation errors

| Error Message                                    | Resolution                               |
|--------------------------------------------------|------------------------------------------|
| "At least one dataset configuration is required" | Provide dataset configurations           |
| "Configuration not found"                        | Invoke `/multi-recording-configuration`  |
| "output_roots must be a list"                    | Pass `output_roots` as a list of strings |
| "Need at least 2 recordings"                     | Provide at least 2 output roots          |
| "Invalid recordings"                             | Verify output roots exist and are dirs   |
| "Empty dataset_name"                             | Resolve a name with the name tool first  |

### Partially accepted batches

A batch that rejects some of its input still returns `success: true`, so you MUST read the rejection lists rather than
treat the absence of an `error` as full acceptance. Report every rejected dataset to the user by name before proceeding,
because the batch runs without it.

| Key                      | Returned by                          | Meaning                                                     |
|--------------------------|--------------------------------------|-------------------------------------------------------------|
| `invalid_configurations` | both prepare and full-pipeline tools | A dataset entry was rejected, with its reason               |
| `path_conflicts`         | both prepare and full-pipeline tools | The dataset runs against stored roots, not the ones passed  |
| `unsizable_datasets`     | `execute_full_pipeline_tool`         | The sizing models cannot size the dataset, so it is omitted |
| `invalid_jobs`           | `execute_processing_jobs_tool`       | A job failed validation or sizing and was not dispatched    |

`path_conflicts` is the one entry that names a dataset the batch still runs. Preparation never reinitializes an existing
tracker, so a dataset prepared earlier keeps using its stored `recording_io.recording_directories` and the entry reports
the dataset, the stored value, the passed value, and the directory to remove to prepare it again. Report it before
dispatching, because the run otherwise covers recordings the user never requested.

A dataset the sizing pass cannot measure is excluded from the batch rather than aborting it, so a run that reports
`started: true` may still cover fewer datasets than you submitted. `execute_full_pipeline_tool` returns no dataset count
of its own, so name every entry of `invalid_configurations` and `unsizable_datasets` to the user rather than looking for
a total to compare. Only the prepare tool returns `total_datasets`.

These lists do not share one element shape. See [tool-responses.md](references/tool-responses.md) for the element
shapes, the full return-key reference of every processing tool, and the terminal messages the engine writes to a
tracker.

### Execution errors

| Error Message                            | Resolution                                   |
|------------------------------------------|----------------------------------------------|
| "An execution session is already active" | Wait for current session or cancel first     |
| "Job ID not found in tracker"            | Re-prepare the batch to regenerate manifests |
| "Prerequisite ... has not succeeded"     | Execute prerequisite phases first            |

Prerequisite failures are returned inside the `invalid_jobs` list with a `reason` field (for example, "Unable to execute
job {job_id}. Its prerequisite 'discovery' job {prerequisite_id} has not succeeded and is not part of this
submission."), not as a top-level `error`.

### Processing failure routing

When processing fails for some datasets/recordings, read the error messages and route:

| Error pattern                                      | Skill to invoke                  |
|----------------------------------------------------|----------------------------------|
| Missing cindra output, incomplete single-recording | `/single-recording-processing`   |
| Plane binary left marked by an interrupted write   | `/single-recording-processing`   |
| Missing raw data, no `cindra_parameters.json`      | `/acquisition-data-preparation`  |
| Configuration parameter issues, bad dataset name   | `/multi-recording-configuration` |
| Registration tuning needed (too much/little drift) | `/multi-recording-configuration` |
| No trackable ROIs found                            | `/multi-recording-configuration` |
| MCP tools unavailable, server connection errors    | `/cindra-mcp-environment-setup`  |

Wait for the current execution session to complete before starting retries. `cancel_processing_jobs_tool` empties the
admission pool and every class queue but never stops a job already dispatched, so poll `get_recording_status_tool` until
the previously RUNNING jobs leave RUNNING before starting a new session.

---

## Related skills

| Skill                             | Relationship                                                                |
|-----------------------------------|-----------------------------------------------------------------------------|
| `/cindra-pipeline`                | Overview: end-to-end phases, handoffs, and the single-vs-multi entry point  |
| `/cindra-mcp-environment-setup`   | Prerequisite: MCP server connectivity                                       |
| `/cli-reference`                  | Reference: `cindra run`, the manual counterpart this workflow replaces      |
| `/acquisition-data-preparation`   | Upstream: raw data preparation                                              |
| `/single-recording-configuration` | Prerequisite chain: configure recordings before single-recording processing |
| `/single-recording-processing`    | Prerequisite: all recordings must be single-recording complete              |
| `/multi-recording-configuration`  | Configuration: parameter reference and file creation                        |
| `/multi-recording-results`        | Output: verify and explain processing results                               |
| `/visualization`                  | Downstream: visual inspection of results                                    |

---

## Verification checklist

```text
Multi-Recording Processing Workflow:
- [ ] MCP server connected (if not, invoke `/cindra-mcp-environment-setup`)
- [ ] All recordings confirmed as single-recording complete (status: completed)
- [ ] Recordings grouped into datasets
- [ ] Dataset names resolved via `resolve_dataset_name_tool`
- [ ] Template configuration confirmed or created via `/multi-recording-configuration` (reusable across datasets)
- [ ] Share of the machine to dedicate to processing confirmed with user
- [ ] Batch prepared or full pipeline executed
- [ ] `path_conflicts` read from the prepare response and every named dataset reported to the user
- [ ] Every `warnings` entry from `reset_processing_phases_tool` acted on before dispatching the reset phase
- [ ] Parameter-change re-run cleaned the `discovery` phase or set the matching repeat flag
- [ ] Status monitored until all datasets complete or fail
- [ ] Failed datasets routed to appropriate skill (see Error routing)
- [ ] Successful datasets verified via `/multi-recording-results`
```
