---
name: multi-recording-processing
description: >-
  Orchestrates multi-recording neural imaging batch processing via the cindra MCP server.
  Dispatches to configuration, validation, and results skills as needed.
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
- Supporting tools for candidate discovery and status checking
- Resource management and CPU allocation guidance
- Status formatting and progress monitoring
- Error routing to appropriate upstream skills

**Does not cover:**
- Configuration parameters, tuning guidance, or config file creation (see `/multi-recording-configuration`)
- Output data formats, array shapes, dtypes, file references, or data interpretation
  (see `/multi-recording-results`)
- Single-recording processing workflow or prerequisites (see `/single-recording-processing`)
- Input data format, TIFF requirements, or acquisition parameters (see `/acquisition-data-preparation`)
- MCP server connectivity or environment issues (see `/mcp-environment-setup`)
- Visual inspection of results (see `/visualization`)

**Handoff rules:** If the user asks about specific output files, array shapes, data interpretation,
registration arrays, tracking templates, or processing result verification, invoke `/multi-recording-results`.
If the user asks about parameter tuning, registration/tracking configuration, or ROI selection criteria, invoke
`/multi-recording-configuration`. This skill owns the processing workflow only — not the data it produces or
the parameters it consumes.

---

## Prerequisites

All recordings must have completed single-recording processing (`get_single_recording_status` returns
status `completed`). If any recording is incomplete, invoke the earliest missing step in the chain:
`/acquisition-data-preparation` → `/single-recording-configuration` → `/single-recording-processing`.

---

## Agent requirements

You MUST use the cindra MCP tools for all processing operations. Do not import cindra Python functions
directly or run processing via scripts or CLI commands. If MCP tools are not available, invoke
`/mcp-environment-setup` to diagnose and resolve connectivity issues.

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

| Tool                             | Purpose                                                           |
|----------------------------------|-------------------------------------------------------------------|
| `get_batch_status_overview_tool` | Bird's-eye view of all processing status under a root directory   |
| `reset_processing_phases_tool`   | Selectively reset completed phases for re-runs                    |
| `clean_processing_output_tool`   | Delete output files for specific phases to reclaim disk space     |

### Configuration & name resolution tools

| Tool                                       | Purpose                                                            |
|--------------------------------------------|--------------------------------------------------------------------|
| `resolve_dataset_name_tool`                | Constructs qualified dataset names from base name + specifier      |
| `discover_multi_recording_candidates_tool` | Finds recordings with completed single-recording output            |
| `generate_config_file`                     | Generates default multi-recording configuration YAML               |

### Supporting tools (used during workflow)

| Tool                                       | Purpose                                                 |
|--------------------------------------------|---------------------------------------------------------|
| `get_single_recording_status`              | Verifies single-recording prerequisites                 |
| `get_multi_recording_status`               | Checks filesystem for multi-recording outputs           |

---

## Pipeline architecture

Two-phase pipeline per dataset:

```text
Phase 1: DISCOVER (Mixed parallelization)
├── Registers all recordings to common reference frame
├── Clusters ROI masks across recordings
├── Generates template masks for tracked ROIs
└── 20 workers per dataset, registration sequential

Phase 2: EXTRACT (CPU bound, parallel by recording)
├── Applies template masks to extract fluorescence
├── Computes neuropil signals, spike deconvolution
└── Workers per recording via saturating allocation (see Resource Management)
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
    specifier=""                            # auto-derived from common parent → "animal_A"
)
→ { "dataset_name": "animal_A_learning_task", "specifier": "animal_A", "base_name": "learning_task" }
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
   starting any computation. This step is idempotent — calling it again on the same datasets returns the
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

1. **Verify prerequisites** — Use `discover_multi_recording_candidates_tool` to find eligible
   recordings and `get_single_recording_status` to confirm each has status `completed`. If any
   recording is incomplete, invoke `/single-recording-processing` (or upstream skills as needed).

2. **Organize into datasets** — Group recordings by common parent directory, user-provided grouping,
   or semantic analysis of recording paths. Each group becomes one dataset in the batch.

3. **Resolve dataset names** — Ask the user for a shared base dataset name (e.g., "learning_task").
   For each group, call `resolve_dataset_name_tool` with the base name and recording paths to
   generate a unique qualified name. The specifier is derived automatically from the common parent
   directory, or the user can provide one explicitly.

4. **Configure** — Ask the user if they have an existing template configuration file. If not,
   invoke `/multi-recording-configuration` to create one. Template configs are reusable across
   datasets and live at user-chosen locations (e.g., `/Data/CA1_GCaMP6f_MD.yaml`). Set each
   configuration's `dataset_name` to the qualified name from step 3. Do NOT create per-dataset
   config copies — the prepare tool automatically saves resolved copies as
   `multi_recording_configuration.yaml` inside each dataset's output directory, preserving the
   original template. Pass the same template path for multiple datasets that share parameters.

5. **Confirm CPU allocation** — Compute the saturating allocation for both phases using the
   algorithm in the Resource Management section. Present the computed values to the user as a
   summary table before starting:

   ```text
   Phase     | Jobs | Workers/Job | Max Parallel | Total Cores
   ----------|------|-------------|--------------|------------
   Discover  |    2 |          60 |            2 |         120
   Extract   |   30 |          30 |            4 |         120
   ```

   Ask the user to confirm or override. Both `workers_per_job` and `max_parallel_jobs` default to
   `-1` (automatic). Only pass explicit values if the user requests an override.

6. **Execute** — Choose one of two approaches:

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

7. **Monitor** — Use `get_processing_jobs_status_tool` to check progress. Optionally use
   `get_active_execution_timing_tool` for per-job timing and session throughput. Present status as a
   formatted table (see Status Formatting section).

8. **Handle completion** — When all datasets finish, check for failures. Route errors to the
   appropriate skill (see Error Routing section). On success, invoke `/multi-recording-results`
   to verify outputs, then `/visualization` for visual inspection.

### Re-running specific phases

To re-run specific phases (e.g., after changing tracking parameters):

1. Use `reset_processing_phases_tool` to reset the target phases to SCHEDULED status. Downstream
   phases are automatically reset (e.g., resetting `discovery` also resets `extraction`).
2. Optionally modify the configuration file before re-execution.
3. Optionally use `clean_processing_output_tool` to delete output files from the reset phases
   (requires the `dataset` parameter for multi-recording).
4. Call `execute_processing_jobs_tool` with the reset jobs from the manifest.

---

## Resource management

The system uses saturating core allocation to distribute CPU cores across parallel compute-bound jobs.
When both `workers_per_job` and `max_parallel_jobs` are set to `-1` (automatic), the allocator
runs the following algorithm:

1. **Budget**: `cpu_count - 2` (2 cores reserved for system operations)
2. **Max parallel jobs**: `min(total_jobs, budget // 30)` (targets ~30 workers per job)
3. **Raw workers per job**: `budget // max_parallel_jobs`
4. **Round down** to the nearest multiple of 5
5. **Saturate**: If workers per job falls below 10 and parallelism > 1, reduce parallelism and
   recalculate until each job has at least 10 workers

| CPU Cores | Budget | Jobs | Workers/Job | Max Parallel | Total Utilized |
|-----------|--------|------|-------------|--------------|----------------|
| 128       | 126    | 4    | 30          | 4            | 120            |
| 64        | 62     | 4    | 30          | 2            | 60             |
| 32        | 30     | 4    | 30          | 1            | 30             |
| 16        | 14     | 4    | 14 (→ 10)   | 1            | 10             |

Both phases use this same allocation model independently. The discover phase treats each dataset
as a job; the extract phase treats each recording as a job. Both `workers_per_job` and
`max_parallel_jobs` default to `-1` (automatic) and can be overridden explicitly in
`execute_processing_jobs_tool` or `execute_full_pipeline_tool`.

---

## Status formatting

When presenting batch status to the user, format as a table:

```text
**Multi-Recording Batch Processing Status**

Current Phase: EXTRACT
Summary: 1/2 datasets complete | 2/4 recordings extracted | 0 failed

| Dataset                    | Discover | Extract Progress | Status     |
|----------------------------|----------|------------------|------------|
| animal_A_learning_task     | done     | 2/2              | SUCCEEDED  |
| animal_B_learning_task     | done     | 0/2              | EXTRACTING |
```

---

## Error routing

### Preparation errors

| Error Message                                     | Resolution                              |
|---------------------------------------------------|-----------------------------------------|
| "At least one dataset configuration is required"  | Provide dataset configurations          |
| "Configuration not found"                         | Invoke `/multi-recording-configuration` |
| "Need at least 2 recordings"                      | Provide at least 2 recording paths      |
| "Invalid recordings"                              | Verify paths exist and are directories  |

### Execution errors

| Error Message                                   | Resolution                                     |
|-------------------------------------------------|------------------------------------------------|
| "An execution session is already active"        | Wait for current session or cancel first       |
| "Job ID not found in tracker"                   | Re-prepare the batch to regenerate manifests   |
| "Prerequisites not satisfied"                   | Execute prerequisite phases first              |

### Processing failure routing

When processing fails for some datasets/recordings, read the error messages and route:

| Error pattern                                       | Skill to invoke                    |
|-----------------------------------------------------|------------------------------------|
| Missing cindra output, incomplete single-recording  | `/single-recording-processing`     |
| Missing raw data, no `cindra_parameters.json`       | `/acquisition-data-preparation`    |
| Configuration parameter issues, bad dataset name    | `/multi-recording-configuration`   |
| Registration tuning needed (too much/little drift)  | `/multi-recording-configuration`   |
| No trackable ROIs found                             | `/multi-recording-configuration`   |
| MCP tools unavailable, server connection errors     | `/mcp-environment-setup`           |

Wait for the current execution session to complete before starting retries.

---

## Related skills

| Skill                              | Role                                                           |
|------------------------------------|----------------------------------------------------------------|
| `/mcp-environment-setup`           | Prerequisite: MCP server connectivity                          |
| `/acquisition-data-preparation`    | Upstream: raw data preparation                                 |
| `/single-recording-processing`     | Prerequisite: all recordings must be single-recording complete |
| `/multi-recording-configuration`   | Configuration: parameter reference and file creation           |
| `/multi-recording-results`         | Output: verify and explain processing results                  |
| `/visualization`                   | Downstream: visual inspection of results                       |

---

## Verification checklist

```text
Multi-Recording Processing Workflow:
- [ ] MCP server connected (if not, invoke `/mcp-environment-setup`)
- [ ] All recordings confirmed as single-recording complete (status: completed)
- [ ] Recordings grouped into datasets
- [ ] Dataset names resolved via `resolve_dataset_name_tool`
- [ ] Configuration file confirmed or created per dataset via `/multi-recording-configuration`
- [ ] CPU core allocation confirmed with user
- [ ] Batch prepared or full pipeline executed
- [ ] Status monitored until all datasets complete or fail
- [ ] Failed datasets routed to appropriate skill (see Error Routing)
- [ ] Successful datasets verified via `/multi-recording-results`
```
