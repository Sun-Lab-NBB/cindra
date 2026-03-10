---
name: multi-recording-processing
description: >-
  Orchestrates multi-recording neural imaging batch processing via the cindra MCP server.
  Dispatches to configuration, validation, and results skills as needed.
---

# Multi-Recording Processing

Orchestrates the multi-recording batch processing workflow: verify prerequisites, organize recordings
by dataset, start batch processing, monitor progress, and hand off to downstream skills for output
verification.

---

## Prerequisites

All recordings must have completed single-recording processing (`get_single_recording_status` returns
status `combined`). If any recording is incomplete, invoke the earliest missing step in the chain:
`/acquisition-data-preparation` → `/single-recording-configuration` → `/single-recording-processing`.

---

## Agent Requirements

You MUST use the cindra MCP tools for all processing operations. Do not import cindra Python functions
directly or run processing via scripts or CLI commands. If MCP tools are not available, invoke
`/mcp-environment-setup` to diagnose and resolve connectivity issues.

---

## Available Tools

### Batch Execution Tools

| Tool                                               | Purpose                                    |
|----------------------------------------------------|--------------------------------------------|
| `start_multi_recording_batch_processing_tool`      | Starts batch processing (1+ datasets)      |
| `get_multi_recording_batch_processing_status_tool` | Returns in-memory status of running batch  |
| `cancel_multi_recording_batch_processing_tool`     | Cancels batch processing, clears queues    |

### Configuration & Name Resolution Tools

| Tool                                       | Purpose                                                            |
|--------------------------------------------|--------------------------------------------------------------------|
| `resolve_dataset_name_tool`                | Constructs qualified dataset names from base name + specifier      |
| `discover_multi_recording_candidates_tool` | Finds recordings with completed single-recording output            |
| `generate_config_file`                     | Generates default multi-recording configuration YAML               |

### Supporting Tools (used during workflow)

| Tool                                       | Purpose                                                 |
|--------------------------------------------|---------------------------------------------------------|
| `get_single_recording_status`              | Verifies single-recording prerequisites                 |
| `get_multi_recording_status`               | Checks filesystem for multi-recording outputs           |

---

## Pipeline Architecture

Two-phase pipeline per dataset:

```
Phase 1: DISCOVER (Mixed parallelization)
├── Registers all recordings to common reference frame
├── Clusters ROI masks across recordings
├── Generates template masks for tracked ROIs
└── 20 workers per dataset, registration sequential

Phase 2: EXTRACT (CPU bound, parallel by recording)
├── Applies template masks to extract fluorescence
├── Computes neuropil signals, spike deconvolution
└── Each recording uses up to 30 workers
```

Batch processing across multiple datasets:

```
DISCOVER: Parallel across datasets (if cores allow)
EXTRACT:  Parallel across all recordings from all datasets
```

---

## Dataset Name Resolution

Each dataset in a batch needs a unique `dataset_name` for output directories and batch tracking. The
`resolve_dataset_name_tool` constructs qualified names by combining a shared base name with a
batch-specific specifier:

```
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

## Processing Workflow

### Pre-Processing Checklist

```
- [ ] All recordings confirmed as single-recording complete (status: combined)
- [ ] Recordings grouped into datasets (by common parent, explicit grouping, or user instruction)
- [ ] Dataset names resolved via resolve_dataset_name_tool
- [ ] Configuration confirmed or created per dataset
- [ ] CPU core allocation confirmed with user
- [ ] Recordings per dataset confirmed
```

**STOP**: If any checkbox is incomplete, do not proceed. Complete the missing steps first.

### Workflow Steps

1. **Verify prerequisites** — Use `discover_multi_recording_candidates_tool` to find eligible
   recordings and `get_single_recording_status` to confirm each has status `combined`. If any
   recording is incomplete, invoke `/single-recording-processing` (or upstream skills as needed).

2. **Organize into datasets** — Group recordings by common parent directory, user-provided grouping,
   or semantic analysis of recording paths. Each group becomes one dataset in the batch.

3. **Resolve dataset names** — Ask the user for a shared base dataset name (e.g., "learning_task").
   For each group, call `resolve_dataset_name_tool` with the base name and recording paths to
   generate a unique qualified name. The specifier is derived automatically from the common parent
   directory, or the user can provide one explicitly.

4. **Configure** — Ask the user if they have existing configuration files per dataset. If not,
   invoke `/multi-recording-configuration` to create and customize them. Set each configuration's
   `dataset_name` to the qualified name from step 3. Do not proceed without confirmed configuration
   paths.

5. **Confirm CPU allocation** — Present the resource allocation model and ask the user how many
   cores to use (see Resource Management section).

6. **Start batch** — Call `start_multi_recording_batch_processing_tool` with the dataset
   configurations and worker settings.

7. **Monitor** — Use `get_multi_recording_batch_processing_status_tool` to check progress.
   Present status as a formatted table (see Status Formatting section).

8. **Handle completion** — When all datasets finish, check for failures. Route errors to the
   appropriate skill (see Error Routing section). On success, invoke `/multi-recording-results`
   to verify outputs, then `/visualization` for visual inspection.

---

## Resource Management

The system automatically calculates optimal resource allocation:

- **Workers per discover**: 20 cores (fixed, internal parallelization)
- **Workers per extract**: `min(cpu_count - 2, 30)` cores
- **Reserved cores**: 2 (for system operations)
- **Maximum job cores**: 30 (processing saturates beyond this)

| CPU Cores | Max Parallel Discovers | Max Parallel Extracts | Behavior                      |
|-----------|------------------------|-----------------------|-------------------------------|
| 32        | 1                      | 1                     | Sequential processing         |
| 64        | 3                      | 2                     | Multiple datasets in parallel |
| 96        | 4                      | 3                     | Higher parallelism            |
| 128       | 6                      | 4                     | Maximum parallelism           |

---

## Status Formatting

When presenting batch status to the user, format as a table:

```
**Multi-Recording Batch Processing Status**

Current Phase: EXTRACT
Summary: 1/2 datasets complete | 2/4 recordings extracted | 0 failed

| Dataset                    | Discover | Extract Progress | Status     |
|----------------------------|----------|------------------|------------|
| animal_A_learning_task     | done     | 2/2              | SUCCEEDED  |
| animal_B_learning_task     | done     | 0/2              | EXTRACTING |
```

---

## Error Routing

### Batch Start Errors

| Error Message                                     | Resolution                              |
|---------------------------------------------------|-----------------------------------------|
| "At least one dataset configuration is required"  | Provide dataset configurations          |
| "Configuration file not found"                    | Invoke `/multi-recording-configuration` |
| "Recording directory not found"                   | Verify path exists                      |
| "Batch processing already in progress"            | Wait for current batch or cancel first  |

### Processing Failure Routing

When processing fails for some datasets/recordings, read the error messages and route:

| Error pattern                                       | Skill to invoke                    |
|-----------------------------------------------------|------------------------------------|
| Missing cindra output, incomplete single-recording  | `/single-recording-processing`     |
| Missing raw data, no `cindra_parameters.json`       | `/acquisition-data-preparation`    |
| Configuration parameter issues, bad dataset name    | `/multi-recording-configuration`   |
| Registration tuning needed (too much/little drift)  | `/multi-recording-configuration`   |
| No trackable ROIs found                             | `/multi-recording-configuration`   |
| MCP tools unavailable, server connection errors     | `/mcp-environment-setup`           |

Wait for the current batch to complete before starting retries.

---

## Related Skills

| Skill                              | Role                                                           |
|------------------------------------|----------------------------------------------------------------|
| `/mcp-environment-setup`           | Prerequisite: MCP server connectivity                          |
| `/acquisition-data-preparation`    | Upstream: raw data preparation                                 |
| `/single-recording-processing`     | Prerequisite: all recordings must be single-recording complete |
| `/multi-recording-configuration`   | Configuration: parameter reference and file creation           |
| `/multi-recording-results`         | Output: verify and explain processing results                  |
| `/visualization`                   | Downstream: visual inspection of results                       |
