---
name: single-recording-processing
description: >-
  Orchestrates single-recording neural imaging batch processing via the cindra MCP server.
  Dispatches to configuration, validation, and results skills as needed.
---

# Single-Recording Processing

Orchestrates the single-recording batch processing workflow: discover recordings, validate prerequisites,
start batch processing, monitor progress, and hand off to downstream skills for output verification.

---

## Agent Requirements

You MUST use the cindra MCP tools for all processing operations. Do not import cindra Python functions
directly or run processing via scripts or CLI commands. If MCP tools are not available, invoke
`/mcp-environment-setup` to diagnose and resolve connectivity issues.

---

## Available Tools

### Batch Execution Tools

| Tool                               | Purpose                                                  |
|------------------------------------|----------------------------------------------------------|
| `start_batch_processing_tool`      | Starts batch single-recording processing (1+ recordings) |
| `get_batch_processing_status_tool` | Returns in-memory status of running batch                |
| `cancel_batch_processing_tool`     | Cancels batch processing, clears queues                  |

### Supporting Tools (used during workflow)

| Tool                                        | Purpose                                             |
|---------------------------------------------|-----------------------------------------------------|
| `discover_single_recording_candidates_tool` | Finds recordings under a root directory             |
| `validate_recording_readiness`              | Validates raw data and parameters before processing |
| `get_single_recording_status`               | Checks filesystem for processing outputs            |

---

## Pipeline Architecture

Three-phase sequential pipeline per recording:

```
Phase 1: BINARIZE (I/O bound, up to 3 parallel)
├── Converts raw TIFFs to binary format
└── Determines plane count

Phase 2: PROCESS (CPU bound, parallel by plane)
├── Motion correction, ROI detection, signal extraction
└── Up to 30 workers per plane

Phase 3: COMBINE (I/O bound, up to 3 parallel)
└── Merges all plane results into unified dataset
```

Batch processing across multiple recordings:

```
BINARIZE: Up to 3 concurrent recordings (I/O bound)
PROCESS:  Parallel (recording-plane pairs up to core limit)
COMBINE:  Up to 3 concurrent recordings (I/O bound)
```

---

## Processing Workflow

### Pre-Processing Checklist

```
- [ ] Recordings discovered or paths provided
- [ ] Raw data validated (or existing binaries confirmed via get_single_recording_status)
- [ ] Configuration confirmed or created
- [ ] CPU core allocation confirmed with user
- [ ] Recordings to process confirmed
```

**STOP**: If any checkbox is incomplete, do not proceed. Complete the missing steps first.

### Workflow Steps

1. **Discover recordings** — Use `discover_single_recording_candidates_tool` or accept explicit paths
   from user.

2. **Validate raw data** — Use `validate_recording_readiness` on each recording. Skip for recordings
   where `get_single_recording_status` shows status `binarized` or later. If validation fails, invoke
   `/acquisition-data-preparation` to resolve issues before continuing.

3. **Configure** — Ask the user if they have an existing configuration file. If not, invoke
   `/single-recording-configuration` to create and customize one. Do not proceed without a confirmed
   configuration path.

4. **Confirm CPU allocation** — Present the resource allocation model and ask the user how many cores
   to use (see Resource Management section).

5. **Start batch** — Call `start_batch_processing_tool` with the confirmed recording paths,
   configuration path, and worker settings.

6. **Monitor** — Use `get_batch_processing_status_tool` to check progress. Present status as a
   formatted table (see Status Formatting section).

7. **Handle completion** — When all recordings finish, check for failures. Route errors to the
   appropriate skill (see Error Routing section). On success, invoke `/single-recording-results`
   to verify outputs, then `/visualization` for visual inspection.

---

## Resource Management

The system automatically calculates optimal resource allocation:

- **Workers per plane**: `min(cpu_count - 2, 30)` cores
- **Reserved cores**: 2 (for system operations)
- **Maximum job cores**: 30 (processing saturates beyond this)

| CPU Cores | Max Parallel Planes | Workers/Plane | Behavior                         |
|-----------|---------------------|---------------|----------------------------------|
| 16        | 1                   | 12            | Sequential plane processing      |
| 32        | 1                   | 28            | Sequential, 28 workers per plane |
| 64        | 2                   | 30            | 2 concurrent planes              |
| 96        | 3                   | 30            | 3 concurrent planes              |
| 128       | 4                   | 30            | 4 concurrent planes              |

---

## Status Formatting

When presenting batch status to the user, format as a table:

```
**Single-Recording Batch Processing Status**

Current Phase: PROCESS
Summary: 10/30 recordings complete | 2 processing | 18 queued | 0 failed

| Recording                    | Binarize | Process | Combine | Status     |
|------------------------------|----------|---------|---------|------------|
| 2024-01-15-10-30-00-123456   | done     | 2/4     | pending | PROCESSING |
| 2024-01-15-11-45-00-234567   | done     | 4/4     | running | PROCESSING |
| 2024-01-16-09-00-00-111111   | done     | 4/4     | done    | SUCCEEDED  |
| 2024-01-16-10-15-00-222222   | pending  | 0/0     | pending | QUEUED     |
```

---

## Error Routing

### Batch Start Errors

| Error Message                             | Resolution                               |
|-------------------------------------------|------------------------------------------|
| "At least one recording path is required" | Provide recording paths                  |
| "Configuration file not found"            | Invoke `/single-recording-configuration` |
| "Recording directory not found"           | Verify path exists                       |
| "Batch processing already in progress"    | Wait for current batch or cancel first   |

### Processing Failure Routing

When processing fails for some recordings, read the error messages and route to the appropriate skill:

| Error pattern                                     | Skill to invoke                   |
|---------------------------------------------------|-----------------------------------|
| Missing `cindra_parameters.json`, TIFF read error | `/acquisition-data-preparation`   |
| Invalid parameter values, wrong plane/channel     | `/acquisition-data-preparation`   |
| Configuration parameter issues                    | `/single-recording-configuration` |
| MCP tools unavailable, server connection errors   | `/mcp-environment-setup`          |

Wait for the current batch to complete before starting retries.

---

## Related Skills

| Skill                              | Role                                                           |
|------------------------------------|----------------------------------------------------------------|
| `/mcp-environment-setup`           | Prerequisite: MCP server connectivity                          |
| `/acquisition-data-preparation`    | Input: raw data preparation and validation                     |
| `/single-recording-configuration`  | Configuration: parameter reference and file creation           |
| `/single-recording-results`        | Output: verify and explain processing results                  |
| `/multi-recording-processing`      | Downstream: cross-recording ROI tracking                       |
| `/visualization`                   | Downstream: visual inspection of results                       |
