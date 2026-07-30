---
name: multi-recording-processing
description: >-
  Orchestrates multi-recording neural imaging batch processing via the cindra MCP server, dispatching to
  configuration and results skills as needed. Use when the user asks to run multi-recording (cross-day ROI
  tracking) processing, process multiple recordings as a tracked dataset, monitor multi-recording batch jobs,
  re-run a multi-recording processing phase, or when invoking /multi-recording-processing.
user-invocable: true
---

# Multi-recording processing

Orchestrates the multi-recording batch processing workflow: verify prerequisites, organize recordings
by dataset, prepare execution manifests, dispatch jobs, monitor progress, and hand off to downstream
skills for output verification.

---

## Scope

**Covers:**
- Batch processing workflow: prerequisite verification, dataset organization, preparation, execution,
  monitoring, and completion
- MCP preparation tools (`prepare_multi_recording_batch_tool`, `execute_full_pipeline_tool`)
- MCP execution tools (`execute_processing_jobs_tool`, `get_processing_jobs_status_tool`,
  `get_active_execution_timing_tool`, `cancel_processing_jobs_tool`)
- MCP management tools (`get_batch_status_overview_tool`, `reset_processing_phases_tool`,
  `clean_processing_output_tool`)
- Dataset name resolution via `resolve_dataset_name_tool`
- Supporting tools for status checking (recording discovery owned by `/multi-recording-configuration`)
- Resource management and CPU allocation guidance
- Status formatting and progress monitoring
- Error routing to appropriate upstream skills

**Does not cover:**
- Configuration parameters, tuning guidance, or config file creation (see `/multi-recording-configuration`)
- Output data formats, array shapes, dtypes, file references, or data interpretation
  (see `/multi-recording-results`)
- Single-recording processing workflow or prerequisites (see `/single-recording-processing`)
- Input data format, TIFF requirements, or acquisition parameters (see `/acquisition-data-preparation`)
- MCP server connectivity or environment issues (see `/cindra-mcp-environment-setup`)
- Visual inspection of results (see `/visualization`)

**Handoff rules:** If the user asks about specific output files, array shapes, data interpretation,
registration arrays, tracking templates, or processing result verification, invoke `/multi-recording-results`.
If the user asks about parameter tuning, registration/tracking configuration, or ROI selection criteria, invoke
`/multi-recording-configuration`. This skill owns the processing workflow. The data it produces belongs to
`/multi-recording-results`, and the parameters it consumes belong to `/multi-recording-configuration`.

---

## Agent requirements

You MUST use the cindra MCP tools for all processing operations. Do not import cindra Python functions
directly or run processing via scripts or CLI commands. If MCP tools are not available, invoke
`/cindra-mcp-environment-setup` to diagnose and resolve connectivity issues.

---

## Prerequisites

All recordings must have completed single-recording processing (`get_recording_status_tool` returns
single-recording status `completed`). If any recording is incomplete, invoke the earliest missing step in the chain:
`/acquisition-data-preparation` → `/single-recording-configuration` → `/single-recording-processing`.

---

## Available tools

### Preparation tools

| Tool                                 | Purpose                                                                 |
|--------------------------------------|-------------------------------------------------------------------------|
| `prepare_multi_recording_batch_tool` | Prepares execution manifest without starting execution (idempotent)     |
| `execute_full_pipeline_tool`         | Convenience: prepares and executes all phases with automatic sequencing |

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

### Name resolution tools

| Tool                        | Purpose                                                       |
|-----------------------------|---------------------------------------------------------------|
| `resolve_dataset_name_tool` | Constructs qualified dataset names from base name + specifier |

### Cross-referenced tools

`/multi-recording-configuration` owns the two tools below. Invoke that skill for their parameters and
usage guidance.

| Tool                        | Purpose                                                          |
|-----------------------------|------------------------------------------------------------------|
| `discover_recordings_tool`  | Discovers single and multi-recording candidates under a root dir |
| `generate_config_file_tool` | Generates default multi-recording configuration YAML             |

### Supporting tools (used during workflow)

| Tool                        | Purpose                                             |
|-----------------------------|-----------------------------------------------------|
| `get_recording_status_tool` | Checks single and multi-recording processing status |

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
└── One job per dataset, workers from the discovery resource class (30 per job, see Resource management)

Phase 2: EXTRACT (phase name: extraction, CPU bound, parallel by recording)
├── Applies template masks to extract fluorescence
├── Computes neuropil signals, spike deconvolution
└── One job per recording, workers from the extraction resource class (16 per job, see Resource management)
```

Batch processing across multiple datasets:

```text
DISCOVER: Parallel across datasets (if cores allow)
EXTRACT:  Parallel across all recordings from all datasets
```

---

## Dataset name resolution

Each dataset in a batch needs a unique `dataset_name` for output directories and batch tracking. The
`resolve_dataset_name_tool` constructs qualified names by combining a shared base name with a
batch-specific specifier:

```text
resolve_dataset_name_tool(
    dataset_name="learning_task",           # shared analysis name from user
    recording_paths=["/data/animal_A/rec1", "/data/animal_A/rec2"],
    specifier=""                            # auto-derived from common parent → "animal_a"
)
→ { "dataset_name": "animal_a_learning_task", "specifier": "animal_a", "base_name": "learning_task" }
```

**Specifier derivation strategies:**
- **Auto (default):** Derived from the deepest common parent directory of the recording paths.
- **Explicit:** The user or agent provides a specifier directly (e.g., brain region, session group).
- **Semantic:** The agent determines the specifier by analyzing recording directory names or paths.

This enables batch bootstrapping: discover all recordings under a project directory, group them by
common parent, and call `resolve_dataset_name_tool` once per group to generate unique dataset names.

---

## Processing workflow

### Execution model

The processing workflow uses a **prepare-then-execute** model:

1. **Prepare** creates an execution manifest (tracker files, per-dataset configurations, job lists) without
   starting any computation. This step is idempotent, so calling it again on the same datasets returns the
   existing manifest.

2. **Execute** dispatches jobs from the manifest with prerequisite validation, resource allocation, and automatic
   phase sequencing. Only one execution session can be active at a time.

For simple cases, `execute_full_pipeline_tool` combines both steps into a single call with automatic phase
advancement. For fine-grained control (e.g., running only specific phases, custom resource allocation, or
selective re-runs), use `prepare_multi_recording_batch_tool` followed by `execute_processing_jobs_tool`.

### Pre-processing checklist

```text
- [ ] All recordings confirmed as single-recording complete (status: completed)
- [ ] Recordings grouped into datasets (by common parent, explicit grouping, or user instruction)
- [ ] Dataset names resolved via resolve_dataset_name_tool
- [ ] Template configuration confirmed or created (one template can serve multiple datasets)
- [ ] CPU core allocation confirmed with user
- [ ] Recordings per dataset confirmed
```

**STOP**: If any checkbox is incomplete, do not proceed. Complete the missing steps first.

### Workflow steps

1. **Verify prerequisites.** Use `discover_recordings_tool`, documented in
   `/multi-recording-configuration`, to find eligible recordings (check the `multi_recording_candidates`
   list) and `get_recording_status_tool` to confirm each has single-recording status `completed`. If any
   recording is incomplete, invoke `/single-recording-processing` (or upstream skills as needed).

2. **Organize into datasets.** Group recordings by common parent directory, user-provided grouping,
   or semantic analysis of recording paths. Each group becomes one dataset in the batch.

3. **Resolve dataset names.** Ask the user for a shared base dataset name (e.g., "learning_task").
   For each group, call `resolve_dataset_name_tool` with the base name and recording paths to
   generate a unique qualified name. The specifier is derived automatically from the common parent
   directory, or the user can provide one explicitly.

4. **Configure.** Ask the user if they have an existing template configuration file. If not,
   invoke `/multi-recording-configuration` to create one. Template configs are reusable across
   datasets and live at user-chosen locations (e.g., `/Data/CA1_GCaMP6f_MD.yaml`). The template's
   `dataset_name` only needs to be a non-empty, filesystem-safe string to pass validation. The prepare tool
   overwrites it with the qualified dataset name you pass per dataset (lowercased to a
   filesystem-safe key). Do NOT create per-dataset config copies. The prepare tool automatically
   saves resolved copies as `multi_recording_configuration.yaml` inside each dataset's output
   directory, preserving the original template. Pass the same template path for multiple datasets
   that share parameters.

5. **Confirm CPU allocation.** Read the per-phase defaults from the Resource management section.
   Discovery and extraction resolve independently, so present one row per class. The example below
   covers 2 datasets of 15 recordings each on a 128-core host, which is 2 discovery jobs plus 30
   extraction jobs against a budget of 126:

   ```text
   Resource class | Jobs | Workers/Job | Max Parallel | Total Cores
   ---------------|------|-------------|--------------|------------
   discovery      |    2 |          30 |            2 |          60
   extraction     |   30 |          16 |            7 |         112
   ```

   Ask the user to confirm or override. Both `workers_per_job` and `max_parallel_jobs` default to
   `-1` (automatic). Only pass explicit values if the user requests an override. Report the values
   returned in the `resource_classes` mapping of the execute tool response rather than recomputing
   them per phase.

6. **Execute.** Choose one of two approaches:

   **Simple (recommended for straightforward runs):**
   Call `execute_full_pipeline_tool` with `pipeline_type="multi-recording"` and
   `dataset_configurations` containing each dataset's `configuration_path`, `recording_paths`, and
   `dataset_name`. This prepares and executes all phases automatically.

   **Fine-grained (for selective execution or re-runs):**
   a. Call `prepare_multi_recording_batch_tool` with the dataset configurations. This returns a
      manifest with job IDs and statuses.
   b. Select the jobs to execute from the manifest (e.g., only SCHEDULED jobs, only specific phases).
   c. Call `execute_processing_jobs_tool` with the selected job descriptors and worker settings. Each
      job descriptor needs `configuration_path`, `tracker_path`, `job_id`, and `pipeline_type` from
      the manifest.

7. **Monitor.** Use `get_processing_jobs_status_tool` to check progress. Optionally use
   `get_active_execution_timing_tool` for per-job timing and session throughput. These two tools
   reflect only the active in-process execution session and return `active: false` with empty jobs
   when no session is running. This drained state happens not only after an MCP server restart, a
   reconnect, or a batch dispatched by a prior process, but also after NORMAL completion: the
   manager clears session state on success AND on failure. So an all-zero, inactive status can mean
   "finished," not "nothing ran." Do not read it as failure. For final per-job outcomes, read
   persisted on-disk tracker state via `get_batch_status_overview_tool` for a whole-tree view,
   `get_recording_status_tool` per recording, or `verify_multi_recording_output_tool` (all using the
   output directory, see the Output-directory path rule). Present status as a formatted table
   (see Status formatting section).

8. **Handle completion.** When all datasets finish, check for failures. A `success: true` return
   only means a tool ran, not that work is ready or done: gate decisions on the domain flag, not on
   `success`. For `verify_multi_recording_output_tool`, gate on `complete` (false whenever `missing`
   is non-empty). For validate tools, gate on `valid`. For `execute_full_pipeline_tool`, gate on
   `started` (it returns `started: false` with a `next_step` when all phases are already complete).
   Checking `success` alone can advance on an unready or already-complete state. Route errors to the
   appropriate skill (see Error routing section). On success, invoke `/multi-recording-results`
   to verify outputs, then `/visualization` for visual inspection.

### Output-directory path rule

`get_recording_status_tool`, `verify_multi_recording_output_tool`, and `clean_processing_output_tool`
all take the recording OUTPUT directory (the parent of the `cindra/` folder), which equals the
per-recording `output_path` used during single-recording processing, rather than the raw-data root. This
matters on a separate-output layout where output and raw-data roots differ:

- `get_recording_status_tool` and `clean_processing_output_tool` resolve `cindra/` directly under the
  given path with NO fallback. Feeding the raw-data root makes them report `not_started` or
  "directory not found", a silent false negative.
- `verify_multi_recording_output_tool` also recursively searches for `configuration.yaml`, so it may
  still pass via that fallback even when fed the wrong root. The two then disagree.

Always reuse the recording OUTPUT directory (the parent of `cindra/`) that single-recording processing
used, its `output_path` entry in the `prepare_single_recording_batch_tool` manifest, for status, verify,
and clean. The multi-recording prepare manifest exposes no `output_path` field: each dataset entry holds
only `configuration_path`, `tracker_path`, `dataset_name`, `pipeline_type`, `discover_job`, and
`extract_jobs`.

### Re-running specific phases

To re-run specific phases (e.g., after changing tracking parameters):

1. Use `reset_processing_phases_tool` with `tracker_path`, `phases`, and
   `pipeline_type="multi-recording"` to reset the target phases to SCHEDULED status. Downstream
   phases are automatically reset (e.g., resetting `discovery` also resets `extraction`).
2. Optionally modify the configuration file before re-execution.
3. Optionally use `clean_processing_output_tool` to delete output files from the reset phases
   (requires `pipeline_type="multi-recording"` plus the lowercased `dataset` name).
4. Call `execute_processing_jobs_tool` with the reset jobs from the manifest.

---

## Resource management

Discovery and extraction run under separate resource classes, each carrying its own measured per-job worker count. The
session CPU budget is `cpu_count - 2`, with 2 cores reserved for system operations, and the dispatcher holds the sum of
the cores committed by every class inside that budget, so the two classes interleave rather than each claiming the whole
budget.

| Phase    | Resource class | Cores per job | Concurrency cap    |
|----------|----------------|---------------|--------------------|
| DISCOVER | `discovery`    | 30            | Session CPU budget |
| EXTRACT  | `extraction`   | 16            | Session CPU budget |

Discovery's 30 is the saturating allocation the phase is admitted at, because its cost grows with the square of the
recording count. Extraction's 16 is measured: the phase reads each frame batch serially before the kernel runs, so it
plateaus above that width.

Each class caps its own concurrency at `min(budget // cores_per_job, job_count)`. The discover phase contributes
one job per dataset, and the extract phase contributes one job per recording. On a 128-core host, with a budget of
126, discovery therefore runs at most 4 jobs concurrently and extraction at most 7.

Neither count shrinks on a small host: a 16-core machine still asks for 30 discovery workers against a budget of 14 and
dispatches one job regardless, because the dispatcher always admits a single job even when the budget cannot cover it.

Both `workers_per_job` and `max_parallel_jobs` default to `-1` and can be overridden in `execute_processing_jobs_tool`
or `execute_full_pipeline_tool`. An override is a single scalar applied to every non-fixed class alike, so passing
`workers_per_job=20` sets discovery and extraction to 20 both. Both execute tools return a session-level `cpu_budget`
and a `resource_classes` mapping keyed by class name, with `discovery` and `extraction` entries carrying
`workers_per_job`, `max_parallel_jobs` and `job_count`. `get_processing_jobs_status_tool` returns the same mapping with
`pending` and `active` in place of `job_count`.

---

## Status formatting

When presenting batch status to the user, format as a table:

```text
**Multi-Recording Batch Processing Status**

Current Phase: EXTRACT
Summary: 1/2 datasets complete | 2/4 recordings extracted | 0 failed

| Dataset                | Discover | Extract Progress | Status     |
|------------------------|----------|------------------|------------|
| animal_A_learning_task | done     | 2/2              | SUCCEEDED  |
| animal_B_learning_task | done     | 0/2              | EXTRACTING |
```

---

## Error routing

### Preparation errors

| Error Message                                    | Resolution                              |
|--------------------------------------------------|-----------------------------------------|
| "At least one dataset configuration is required" | Provide dataset configurations          |
| "Configuration not found"                        | Invoke `/multi-recording-configuration` |
| "Need at least 2 recordings"                     | Provide at least 2 recording paths      |
| "Invalid recordings"                             | Verify paths exist and are directories  |

### Execution errors

| Error Message                            | Resolution                                   |
|------------------------------------------|----------------------------------------------|
| "An execution session is already active" | Wait for current session or cancel first     |
| "Job ID not found in tracker"            | Re-prepare the batch to regenerate manifests |
| "Prerequisite ... has not succeeded"     | Execute prerequisite phases first            |

Prerequisite failures are returned inside the `invalid_jobs` list with a `reason` field (for example,
"Unable to execute job {job_id}. Its prerequisite 'discovery' job {prerequisite_id} has not succeeded and
is not part of this submission."), not as a top-level `error`.

### Processing failure routing

When processing fails for some datasets/recordings, read the error messages and route:

| Error pattern                                      | Skill to invoke                  |
|----------------------------------------------------|----------------------------------|
| Missing cindra output, incomplete single-recording | `/single-recording-processing`   |
| Missing raw data, no `cindra_parameters.json`      | `/acquisition-data-preparation`  |
| Configuration parameter issues, bad dataset name   | `/multi-recording-configuration` |
| Registration tuning needed (too much/little drift) | `/multi-recording-configuration` |
| No trackable ROIs found                            | `/multi-recording-configuration` |
| MCP tools unavailable, server connection errors    | `/cindra-mcp-environment-setup`  |

Wait for the current execution session to complete before starting retries.

---

## Related skills

| Skill                             | Relationship                                                                |
|-----------------------------------|-----------------------------------------------------------------------------|
| `/cindra-pipeline`                | Overview: end-to-end phases, handoffs, and the single-vs-multi entry point  |
| `/cindra-mcp-environment-setup`   | Prerequisite: MCP server connectivity                                       |
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
- [ ] CPU core allocation confirmed with user
- [ ] Batch prepared or full pipeline executed
- [ ] Status monitored until all datasets complete or fail
- [ ] Failed datasets routed to appropriate skill (see Error routing)
- [ ] Successful datasets verified via `/multi-recording-results`
```
