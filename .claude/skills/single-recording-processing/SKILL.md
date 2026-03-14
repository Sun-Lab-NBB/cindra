---
name: single-recording-processing
description: >-
  Orchestrates single-recording neural imaging batch processing via the cindra MCP server.
  Dispatches to configuration, validation, and results skills as needed.
user-invocable: true
---

# Single-recording processing

Orchestrates the single-recording batch processing workflow: discover recordings, validate prerequisites,
start batch processing, monitor progress, and hand off to downstream skills for output verification.

---

## Scope

**Covers:**
- Batch processing workflow: discovery, validation, execution, monitoring, and completion
- MCP batch execution tools (`start_batch_processing_tool`, `get_batch_processing_status_tool`,
  `cancel_batch_processing_tool`)
- Supporting tools for discovery, validation, and status checking
- Resource management and CPU allocation guidance
- Status formatting and progress monitoring
- Error routing to appropriate upstream skills

**Does not cover:**
- Configuration parameters, tuning guidance, or config file creation (see `/single-recording-configuration`)
- Output data formats, array shapes, dtypes, file references, or data interpretation (see `/single-recording-results`)
- Input data format, TIFF requirements, or acquisition parameters (see `/acquisition-data-preparation`)
- Multi-recording processing workflow (see `/multi-recording-processing`)
- MCP server connectivity or environment issues (see `/mcp-environment-setup`)
- Visual inspection of results (see `/visualization`)

**Handoff rules:** If the user asks about specific output files, array shapes, data interpretation, or processing
result verification, invoke `/single-recording-results`. If the user asks about parameter tuning or configuration
options, invoke `/single-recording-configuration`. This skill owns the processing workflow only — not the data
it produces or the parameters it consumes.

---

## Agent requirements

You MUST use the cindra MCP tools for all processing operations. Do not import cindra Python functions
directly or run processing via scripts or CLI commands. If MCP tools are not available, invoke
`/mcp-environment-setup` to diagnose and resolve connectivity issues.

---

## Available tools

### Batch execution tools

| Tool                               | Purpose                                                  |
|------------------------------------|----------------------------------------------------------|
| `start_batch_processing_tool`      | Starts batch single-recording processing (1+ recordings) |
| `get_batch_processing_status_tool` | Returns disk-based status of running batch               |
| `cancel_batch_processing_tool`     | Cancels batch processing, clears queues                  |

### Supporting tools (used during workflow)

| Tool                                        | Purpose                                             |
|---------------------------------------------|-----------------------------------------------------|
| `discover_single_recording_candidates_tool` | Finds recordings under a root directory             |
| `validate_recording_readiness`              | Validates raw data and parameters before processing |
| `get_single_recording_status`               | Checks filesystem for processing outputs            |

---

## Pipeline architecture

Three-phase sequential pipeline per recording:

```text
Phase 1: BINARIZE (I/O bound, up to 3 parallel)
├── Converts raw TIFFs to binary format
└── Determines plane count

Phase 2: PROCESS (CPU bound, parallel by plane)
├── Motion correction, ROI detection, signal extraction
└── Up to cpu_count - 2 workers per plane

Phase 3: COMBINE (I/O bound, up to 3 parallel)
└── Merges all plane results into unified dataset
```

Batch processing across multiple recordings:

```text
BINARIZE: Up to 3 concurrent recordings (I/O bound)
PROCESS:  Parallel (recording-plane pairs up to core limit)
COMBINE:  Up to 3 concurrent recordings (I/O bound)
```

---

## Processing workflow

### Pre-processing checklist

```text
- [ ] Recordings discovered or paths provided
- [ ] Raw data validated (or existing binaries confirmed via get_single_recording_status)
- [ ] Template configuration confirmed or created (one template can serve multiple recordings)
- [ ] CPU core allocation confirmed with user
- [ ] Recordings to process confirmed
```

**STOP**: If any checkbox is incomplete, do not proceed. Complete the missing steps first.

### Workflow steps

1. **Discover recordings** — Use `discover_single_recording_candidates_tool` or accept explicit paths
   from user.

2. **Validate raw data** — Use `validate_recording_readiness` on each recording. Skip for recordings
   where `get_single_recording_status` shows status `binarized` or later. If validation fails, invoke
   `/acquisition-data-preparation` to resolve issues before continuing.

3. **Configure** — Ask the user if they have an existing template configuration file. If not,
   invoke `/single-recording-configuration` to create one. Template configs are reusable across
   recordings and live at user-chosen locations (e.g., `/Data/CA1_GCaMP6f_SD.yaml`). Do NOT create
   per-recording config copies — the batch tool automatically saves resolved copies as
   `cindra/configuration.yaml` inside each recording's output directory, preserving the original
   template. Pass the same template path for all recordings that share parameters.

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

## Resource management

The system automatically calculates optimal resource allocation:

- **Workers per plane**: Up to `cpu_count - 2` cores (reserved cores for system operations)
- **No per-job cap**: Workers are limited only by available CPU cores minus reserved
- **Parallel capacity**: Automatically calculated from `workers_per_plane` and available cores

---

## Status formatting

When presenting batch status to the user, format as a table:

```text
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

## Error routing

### Batch start errors

| Error Message                             | Resolution                               |
|-------------------------------------------|------------------------------------------|
| "At least one recording path is required" | Provide recording paths                  |
| "Configuration file not found"            | Invoke `/single-recording-configuration` |
| "Recording directory not found"           | Verify path exists                       |
| "Batch processing already in progress"    | Wait for current batch or cancel first   |

### Processing failure routing

When processing fails for some recordings, read the error messages and route to the appropriate skill:

| Error pattern                                     | Skill to invoke                   |
|---------------------------------------------------|-----------------------------------|
| Missing `cindra_parameters.json`, TIFF read error | `/acquisition-data-preparation`   |
| Invalid parameter values, wrong plane/channel     | `/acquisition-data-preparation`   |
| Configuration parameter issues                    | `/single-recording-configuration` |
| MCP tools unavailable, server connection errors   | `/mcp-environment-setup`          |

Wait for the current batch to complete before starting retries.

---

## Related skills

| Skill                              | Role                                                           |
|------------------------------------|----------------------------------------------------------------|
| `/mcp-environment-setup`           | Prerequisite: MCP server connectivity                          |
| `/acquisition-data-preparation`    | Input: raw data preparation and validation                     |
| `/single-recording-configuration`  | Configuration: parameter reference and file creation           |
| `/single-recording-results`        | Output: verify and explain processing results                  |
| `/multi-recording-processing`      | Downstream: cross-recording ROI tracking                       |
| `/visualization`                   | Downstream: visual inspection of results                       |

---

## Verification checklist

```text
Single-Recording Processing Workflow:
- [ ] MCP server connected (if not, invoke `/mcp-environment-setup`)
- [ ] Recordings discovered or explicit paths provided
- [ ] Raw data validated via `validate_recording_readiness` (or existing binaries confirmed)
- [ ] Configuration file confirmed or created via `/single-recording-configuration`
- [ ] CPU core allocation confirmed with user
- [ ] Batch started via `start_batch_processing_tool`
- [ ] Status monitored until all recordings complete or fail
- [ ] Failed recordings routed to appropriate skill (see Error Routing)
- [ ] Successful recordings verified via `/single-recording-results`
```
