---
name: multi-recording-processing
description: >-
  Guides AI agents through multi-recording (cross-recording) neural imaging data processing using the cindra MCP server.
  Covers ROI tracking across recordings, configuration, batch processing, and status monitoring for longitudinal studies.
---

# Multi-Recording Neural Imaging Processing

Guides AI agents through the workflow for tracking ROIs across multiple recordings and extracting consistent
fluorescence traces using the cindra MCP server tools.

---

## Prerequisites

**Before using this skill, ensure:**

1. All recordings have been processed with the single-recording pipeline
2. The `combine` step completed for each recording (cindra/combined/ exists)
3. Recordings are from the same animal and imaging region

If single-recording processing is not complete, the full prerequisite chain is:
`/acquisition-data-preparation` → `/single-recording-configuration` → `/single-recording-processing`.
Invoke the earliest missing step and work forward through the chain before returning here.

---

## Agent Requirements

**You MUST use the MCP tools provided by this library for all neural imaging data processing tasks.** The cindra
library provides an MCP server that exposes specialized tools for discovering recordings, executing pipelines, and
monitoring processing status. These tools are the only supported interface for agentic neural imaging data processing.

### Mandatory Tool Usage

- You MUST NOT import or call cindra Python functions directly (e.g., `from cindra.multi_recording import ...`)
- You MUST NOT attempt to run processing by executing Python scripts or CLI commands
- You MUST use the MCP tools listed in the "Available Tools" section below
- You MUST verify the MCP server is connected before attempting any processing operations

### Why MCP Tools Are Required

The MCP tools provide:

1. **Background processing** - Jobs run in separate threads, allowing parallel animal processing
2. **Intelligent batching** - Two-phase processing (discover → extract)
3. **Automatic queuing** - Recordings beyond parallel capacity are queued and started automatically
4. **Status tracking** - Real-time progress monitoring across all phases
5. **Error isolation** - Failures in one job don't crash the entire pipeline
6. **Resource management** - Automatic CPU core allocation and cleanup

Direct Python calls bypass these capabilities and will fail in agentic contexts.

---

## MCP Server Configuration

The MCP server must be running and connected for the tools to be available.

### Server Startup

The server is started via the CLI:

```bash
cindra mcp
```

Transport options:
- `cindra mcp` - Default stdio transport
- `cindra mcp -t sse` - Server-Sent Events transport
- `cindra mcp -t streamable-http` - Streamable HTTP transport

### Claude Code Configuration

Add to your `.mcp.json` file in the project root:

```json
{
  "mcpServers": {
    "cindra-mcp": {
      "type": "stdio",
      "command": "cindra",
      "args": ["mcp"]
    }
  }
}
```

### Verifying Connection

Before processing, verify the MCP tools are available by checking your tool list. If the cindra tools
(`discover_multi_recording_candidates_tool`, `start_multi_recording_batch_processing_tool`, etc.) are not
present, the server is not connected. Invoke `/mcp-environment-setup` to diagnose and resolve connectivity
issues.

---

## Available Tools

The MCP server exposes a unified API where all processing goes through the batch manager, even for single animals.

### Configuration Tool

| Tool                   | Purpose                                                   |
|------------------------|-----------------------------------------------------------|
| `generate_config_file` | Generates a default configuration YAML file (multi-recording)   |

### Recording Discovery and Status Tools

| Tool                     | Purpose                                            |
|--------------------------|----------------------------------------------------|
| `discover_multi_recording_candidates_tool` | Finds recordings under a root directory              |
| `get_multi_recording_status`   | Checks filesystem for multi-recording processing outputs |

### Batch Processing Tools

| Tool                                        | Purpose                                       |
|---------------------------------------------|-----------------------------------------------|
| `start_multi_recording_batch_processing_tool`      | Starts batch multi-recording processing (1+ animals)|
| `get_multi_recording_batch_processing_status_tool` | Returns in-memory status of running batch     |
| `cancel_multi_recording_batch_processing_tool`     | Cancels batch processing, clears queues       |

---

## Pipeline Architecture

The multi-recording pipeline tracks ROIs across multiple recordings and extracts their fluorescence consistently.

### Two-Phase Processing

```
Phase 1: DISCOVER (Mixed - internal parallelization)
├── Registers all recordings to common reference frame
├── Clusters ROI masks across recordings
├── Generates template masks for tracked ROIs
├── Uses 20 workers for deformation application
└── Registration computation is sequential per animal

Phase 2: EXTRACT (CPU bound, independent recordings)
├── Applies template masks to extract fluorescence
├── Computes neuropil signals
├── Performs spike deconvolution
├── Each recording uses up to 30 workers
└── All recordings can be extracted in parallel
```

### Batch Processing Architecture

When processing multiple animals:

```
Phase 1: DISCOVER (Parallel across animals if cores allow)
├── Animal 1: discover (registration + deformation)
├── Animal 2: discover (registration + deformation)
└── Animal N: discover
    [Can run multiple animals in parallel]

Phase 2: EXTRACT (Parallel across all recordings)
├── Animal 1, Recording 1 ─┐
├── Animal 1, Recording 2  │
├── Animal 2, Recording 1  ├── Parallel batch
├── Animal 2, Recording 2  │
└── ...                  ┘
```

### Output Structure

Results are saved to `{recording_path}/cindra/multi_recording/{dataset_name}/`:

```
multi_recording/
└── dataset_name/
    ├── multi_recording_runtime_data.yaml       # Runtime data (per-recording)
    ├── multi_recording_configuration.yaml      # Configuration snapshot (main recording only)
    ├── multi_recording_tracker.yaml            # Processing tracker (main recording only)
    ├── registration_arrays/                    # Deformation fields and transformed images
    ├── registration_deformed_masks.npz         # Forward-deformed ROI masks
    ├── tracking_template_masks.npz             # Tracked consensus ROI masks
    ├── roi_masks.npz                           # Extracted ROI spatial masks
    ├── roi_statistics.npz                      # Extracted ROI statistics
    ├── cell_fluorescence.npy                   # Fluorescence traces
    ├── neuropil_fluorescence.npy               # Neuropil traces
    └── spikes.npy                              # Deconvolved spikes
```

---

## Formatting status as a table

When presenting status to the user, format the data as a clear table:

```
**Multi-Recording Batch Processing Status**

Current Phase: EXTRACT
Summary: 1/2 animals complete | 2/4 recordings extracted | 0 failed

| Animal           | Discover | Extract Progress | Status     |
|------------------|----------|------------------|------------|
| animal1_dataset  | done     | 2/2              | SUCCEEDED  |
| animal2_dataset  | done     | 0/2              | EXTRACTING |
```

---

## Processing Workflow

### Pre-Processing Checklist

**You MUST complete this checklist before starting batch processing.** Do not skip any step.

```
- [ ] Verified all recordings have single-recording processing complete (cindra/combined/ exists)
- [ ] Organized recordings by animal/dataset
- [ ] Configuration file confirmed or created (see Configuration Guidance section)
- [ ] Asked about dataset name and MROI region margin if creating new config
- [ ] Asked user about CPU core allocation (see Resource Allocation section)
- [ ] Received user response confirming worker count
- [ ] Confirmed which recordings to process per animal
```

**STOP**: If any checkbox is incomplete, do not proceed. Complete the missing steps first.

### Workflow Steps

1. **Verify prerequisites** → Use `discover_multi_recording_candidates_tool` and `get_single_recording_status`
   to confirm all recordings have completed single-recording processing. If any recording is incomplete, invoke
   `/single-recording-processing` (or `/acquisition-data-preparation` if raw data is not yet prepared).
2. **Organize by animal** → Group recordings by animal/dataset
3. **Check configuration** → Ask user if they have an existing config (see Configuration Guidance)
4. **Create config if needed** → Generate default and ask about dataset name + MROI region margin
5. **Ask about CPU allocation** → Explain resource model and ask how many cores to use
6. **Start batch processing** → Call `start_multi_recording_batch_processing_tool`
7. **Monitor progress** → Check two-phase progress (discover → extract)
8. **Explain any errors** → Analyze failures when complete

---

## Configuration Guidance

**CRITICAL**: You MUST ask the user about configuration before processing. Never skip this step.

For complete parameter documentation, invoke `/multi-recording-configuration`. If the user asks about specific
parameters (registration speed, ROI tracking thresholds, mask prevalence, etc.) or needs help tuning
configuration values, transition to `/multi-recording-configuration` to provide the full parameter reference
before returning here to continue the processing workflow.

### Step 1: Ask About Existing Configuration

Before processing, always ask:

> Do you have an existing configuration file you'd like to use for this processing run?
> - If yes, provide the path to your `.yaml` configuration file
> - If no, I'll create a default configuration (optimized for GCaMP6f mesoscope data)

### Step 2: Configuration Handling

**If user has an existing configuration:**
1. Confirm the file path exists
2. Proceed to CPU core allocation

**If user needs a new configuration:**
1. Use `generate_config_file` with `pipeline_type: "multi-recording"` to create a default configuration file
2. Ask about required/optional parameters (see below)
3. Proceed to CPU core allocation

### Key Questions for New Configurations

The default configuration works well for GCaMP6f data. Ask these questions:

- **Required**: "What name should identify this multi-recording dataset?" (sets `io.dataset_name`)
- **Optional**: "Is this an MROI (multi-region) recording? If so, you can adjust the region border margin for ROI
  filtering" (sets `roi_selection.mroi_region_margin`)

### Pipeline-Set Parameters

These are set automatically by the MCP batch tool:
- `io.recording_directories` - Set from `recording_paths` argument
- `main.parallel_workers` - Set from worker arguments

See `/multi-recording-configuration` for complete parameter documentation including ROI selection, registration tuning, and
clustering options.

---

## Resource Management

### CPU Core Allocation

The system automatically calculates optimal resource allocation:

- **Workers per discover**: 20 cores (fixed, internal parallelization)
- **Workers per extract**: `min(cpu_count - 4, 30)` cores
- **Reserved cores**: 4 (for system operations)
- **Maximum job cores**: 30 (processing saturates beyond this)

### Allocation Table

| CPU Cores | Max Parallel Discovers | Max Parallel Extracts | Behavior                     |
|-----------|------------------------|-----------------------|------------------------------|
| 32        | 1                      | 1                     | Sequential processing        |
| 64        | 3                      | 2                     | Multiple animals in parallel |
| 96        | 4                      | 3                     | Higher parallelism           |
| 128       | 6                      | 4                     | Maximum parallelism          |

---

## Error Handling

### Common Errors

| Error Message                          | Cause                          | Resolution                        |
|----------------------------------------|--------------------------------|-----------------------------------|
| "No cindra output directory found"     | Single-recording processing incomplete| Run single-recording pipeline first     |
| "Configuration file not found"         | Invalid configuration_path     | Generate or verify configuration  |
| "Recording directory not found"          | Invalid path for recording       | Verify path exists                |
| "Batch processing already in progress" | Batch already running          | Wait for current batch to complete|

### Handling Failures

If processing fails for some animals/recordings:

1. Note which animals/recordings failed from the status output
2. Read the error messages in the output
3. Explain the errors to the user with root cause and resolution
4. Wait for the current batch to complete before starting retries

### Multi-Recording Specific Errors

| Error Pattern                          | Likely Cause                           | Resolution                           |
|----------------------------------------|----------------------------------------|--------------------------------------|
| `No cindra output found`               | Single-recording processing incomplete       | Complete single-recording pipeline first   |
| `Recording IDs mismatch`                 | Configuration doesn't match recordings   | Verify recording_directories in config |
| `Registration failed between recordings` | Too much drift between recordings            | Check FOV alignment                  |
| `No trackable ROIs found`              | Insufficient overlap in detected ROIs  | Adjust clustering threshold          |

### Error-to-Skill Routing

When errors suggest upstream issues, invoke the appropriate skill to resolve before retrying:

| Error pattern                                       | Upstream skill to invoke           |
|-----------------------------------------------------|------------------------------------|
| Missing cindra output, incomplete single-recording  | `/single-recording-processing`     |
| Missing raw data, no `cindra_parameters.json`       | `/acquisition-data-preparation`    |
| Configuration parameter issues, bad dataset name    | `/multi-recording-configuration`   |
| Registration tuning needed (too much/little drift)  | `/multi-recording-configuration`   |
| MCP tools unavailable, server connection errors     | `/mcp-environment-setup`           |

---

## Post-Processing Next Steps

After batch processing completes successfully:

1. **Verify outputs** — invoke `/multi-recording-results` to validate that all expected output files are
   present and correctly formatted for each animal.
2. **Inspect results** — invoke `/visualization` to launch viewers for visual inspection of cross-recording
   registration quality, tracked ROI masks, and extracted traces.

---

## Related skills

| Skill                              | Relationship                                                                             |
|------------------------------------|------------------------------------------------------------------------------------------|
| `/mcp-environment-setup`           | Prerequisite: MCP server must be connected for processing tools                          |
| `/acquisition-data-preparation`    | Upstream: raw data preparation and readiness validation for each recording               |
| `/single-recording-processing`     | Prerequisite: all recordings must have single-recording processing complete              |
| `/single-recording-results`        | Prerequisite: single-recording output files required as input                            |
| `/multi-recording-configuration`   | Configuration reference for all multi-recording pipeline parameters                      |
| `/multi-recording-results`         | Output data format reference for evaluating processing results                           |
| `/visualization`                   | Next step: launch viewers to inspect tracking quality and query results                  |

