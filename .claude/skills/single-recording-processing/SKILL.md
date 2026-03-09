---
name: single-recording-processing
description: >-
  Guides AI agents through single-recording (within-recording) neural imaging data processing using the cindra MCP server.
  Covers recording discovery, configuration, batch processing, and status monitoring for individual recordings.
---

# Single-Recording Neural Imaging Processing

Guides AI agents through the workflow for processing neural imaging data from individual recordings using the
cindra MCP server tools.

---

## Agent Requirements

**You MUST use the MCP tools provided by this library for all neural imaging data processing tasks.** The cindra
library provides an MCP server that exposes specialized tools for discovering recordings, executing pipelines, and
monitoring processing status. These tools are the only supported interface for agentic neural imaging data processing.

### Mandatory Tool Usage

- You MUST NOT import or call cindra Python functions directly (e.g., `from cindra.pipeline import ...`)
- You MUST NOT attempt to run processing by executing Python scripts or CLI commands
- You MUST use the MCP tools listed in the "Available Tools" section below
- You MUST verify the MCP server is connected before attempting any processing operations

### Why MCP Tools Are Required

The MCP tools provide:

1. **Background processing** - Jobs run in separate threads, allowing parallel recording processing
2. **Intelligent batching** - Three-phase processing (binarize → process → combine)
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
(`discover_single_recording_candidates_tool`, `start_batch_processing_tool`, etc.) are not present, the server
is not connected. Invoke `/mcp-environment-setup` to diagnose and resolve connectivity issues.

---

## Available Tools

The MCP server exposes a unified API where all processing goes through the batch manager, even for single recordings.

### Configuration Tool

| Tool                   | Purpose                                                    |
|------------------------|------------------------------------------------------------|
| `generate_config_file` | Generates a default configuration YAML file (single-recording)   |

### Recording Discovery, Readiness, and Status Tools

| Tool                     | Purpose                                             |
|--------------------------|-----------------------------------------------------|
| `discover_single_recording_candidates_tool` | Finds recordings under a root directory               |
| `validate_recording_readiness`             | Validates raw data + parameters before processing     |
| `get_single_recording_status`  | Checks filesystem for single-recording processing outputs |

### Batch Processing Tools

| Tool                               | Purpose                                            |
|------------------------------------|----------------------------------------------------|
| `start_batch_processing_tool`      | Starts batch single-recording processing (1+ recordings)   |
| `get_batch_processing_status_tool` | Returns in-memory status of running batch          |
| `cancel_batch_processing_tool`     | Cancels batch processing, clears queues            |

---

## Pipeline Architecture

The single-recording pipeline processes brain imaging data from a single recording through three sequential phases.

### Three-Phase Processing

```
Phase 1: BINARIZE (I/O bound)
├── Converts raw TIFFs to binary format
├── Determines plane count for the recording
└── Sequential - I/O limited

Phase 2: PROCESS (CPU bound)
├── Motion correction (registration)
├── ROI detection
├── Signal extraction (fluorescence traces)
├── Processes each plane independently
└── Parallel - up to 30 workers per plane

Phase 3: COMBINE (I/O bound)
├── Merges data from all planes
├── Creates unified dataset
└── Sequential - I/O limited
```

### Batch Processing Architecture

When processing multiple recordings:

```
Phase 1: BINARIZE (Sequential across recordings)
├── Recording 1: binarize → determines plane count
├── Recording 2: binarize → determines plane count
└── Recording N: binarize → determines plane count

Phase 2: PROCESS (Parallel across recording-plane pairs)
├── Recording 1, Plane 0 ─┐
├── Recording 1, Plane 1  │
├── Recording 2, Plane 0  ├── Parallel batch
├── Recording 2, Plane 1  │
└── ...                 ┘

Phase 3: COMBINE (Sequential across recordings)
├── Recording 1: combine
├── Recording 2: combine
└── Recording N: combine
```

### Output Structure

Results are saved to `{recording_path}/cindra/`:

```
cindra/
├── plane_0/                       # Per-plane processing results
│   ├── runtime_data.yaml          # Plane runtime configuration
│   ├── channel_1_data.bin         # Registered binary data
│   ├── registration_data/         # Registration outputs
│   ├── detection_data/            # Detection reference images
│   ├── roi_masks.npz              # ROI spatial masks
│   ├── roi_statistics.npz         # ROI morphological statistics
│   ├── cell_fluorescence.npy      # Fluorescence traces
│   ├── neuropil_fluorescence.npy  # Neuropil traces
│   ├── spikes.npy                 # Deconvolved spikes
│   └── cell_classification.npy    # Cell/non-cell classification
├── plane_1/
├── ...
├── combined_metadata.npz          # Merged plane metadata
├── detection_data/                # Combined detection reference images
├── roi_masks.npz                  # Combined ROI spatial masks
├── roi_statistics.npz             # Combined ROI statistics
├── cell_fluorescence.npy          # Combined fluorescence traces
├── neuropil_fluorescence.npy      # Combined neuropil traces
├── spikes.npy                     # Combined deconvolved spikes
└── cell_classification.npy        # Combined cell classification
```

---

## Formatting status as a table

When presenting status to the user, format the data as a clear table:

```
**Single-Recording Batch Processing Status**

Current Phase: PROCESS
Summary: 10/30 recordings complete | 2 processing | 18 queued | 0 failed

| Recording                      | Binarize | Process | Combine | Status     |
|------------------------------|----------|---------|---------|------------|
| 2024-01-15-10-30-00-123456   | done     | 2/4     | pending | PROCESSING |
| 2024-01-15-11-45-00-234567   | done     | 4/4     | running | PROCESSING |
| 2024-01-16-09-00-00-111111   | done     | 4/4     | done    | SUCCEEDED  |
| 2024-01-16-10-15-00-222222   | pending  | 0/0     | pending | QUEUED     |
```

---

## Processing Workflow

### Pre-Processing Checklist

**You MUST complete this checklist before starting batch processing.** Do not skip any step.

```
- [ ] Recording discovery complete (used discover_single_recording_candidates_tool or received explicit paths)
- [ ] Raw data validated for each recording (see Raw Data Validation section below)
- [ ] Configuration file confirmed or created (see Configuration Guidance section)
- [ ] Asked about exclusions if creating new config (flyback planes, ignored files)
- [ ] Asked user about CPU core allocation (see Resource Allocation section)
- [ ] Received user response confirming worker count
- [ ] Confirmed which recordings to process
```

**STOP**: If any checkbox is incomplete, do not proceed. Complete the missing steps first.

### Raw Data Validation

Before committing compute resources to batch processing, validate that each recording's raw data is ready. Use
`validate_recording_readiness` on each discovered recording directory to confirm that `cindra_parameters.json`
is present and valid, that TIFF files are readable with consistent dimensions, and that TIFF data is compatible
with the acquisition parameters.

**Exception**: Skip this step for recordings that already have valid binary files from a previous binarization run
(check via `get_single_recording_status`). Existing valid binaries mean the raw data was already successfully
ingested, so re-validation is unnecessary.

**If validation fails**: Invoke `/acquisition-data-preparation` to resolve the issues. Common problems include
missing `cindra_parameters.json` (needs to be created), incorrect acquisition parameters (needs correction),
frame count not divisible by the interleave stride (incomplete volume or wrong plane/channel count), or
inconsistent TIFF dimensions (corrupted or mismatched files). Do not proceed to processing until all recordings
pass readiness validation or have valid existing binaries.

### Workflow Steps

1. **Discover recordings** → Use `discover_single_recording_candidates_tool` to find all recording paths
2. **Validate raw data** → Use `validate_recording_readiness` on each recording (skip if valid binaries exist)
3. **Check configuration** → Ask user if they have an existing config (see Configuration Guidance)
4. **Create config if needed** → Generate default and ask about exclusions (flyback planes, ignored files)
5. **Ask about CPU allocation** → Explain resource model and ask how many cores to use
6. **Start batch processing** → Call `start_batch_processing_tool`
7. **Inform user** → Report batch status and explain three-phase processing
8. **Check status on request** → Display formatted status table
9. **Explain any errors** → Analyze and explain errors when processing completes

---

## Configuration Guidance

**CRITICAL**: You MUST ask the user about configuration before processing. Never skip this step.

For complete parameter documentation, invoke `/single-recording-configuration`. If the user asks about specific
parameters (tau, registration, ROI detection, etc.) or needs help tuning configuration values, transition to
`/single-recording-configuration` to provide the full parameter reference before returning here to continue
the processing workflow.

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
1. Use `generate_config_file` with `pipeline_type: "single-recording"` to create a default configuration file
2. Ask about exclusions (see below)
3. Proceed to CPU core allocation

### Key Questions for New Configurations

The default configuration works well for GCaMP6f data from 2P mesoscope. Only ask about exclusions:

- "Are there any flyback planes to exclude from processing? If so, provide the plane indices (0-based, e.g., [0] or [3])"
- "Are there any TIFF files that should be ignored? If so, provide the exact file names to skip"

### Auto-Overwritten Parameters (Mesoscope)

When `mesoscan=True`, these parameters are **automatically overwritten** from `cindra_parameters.json` (or
`suite2p_parameters.json` as a legacy fallback):
- `nplanes` - Set to `plane_number × roi_number`
- `fs` - Set to `frame_rate`
- ROI geometry (`lines`, `dx`, `dy`, `nrois`)

User-specified values for these parameters are ignored. See `/single-recording-configuration` for complete details.

### Multi-Recording Compatibility

If the user intends to run multi-recording processing on these recordings later, ensure:
- `file_io.delete_bin: false` (keep registered binary files)
- `output.combined: true` (merge plane results)

These are the defaults, so no changes needed unless the user explicitly disabled them.

---

## Resource Management

### CPU Core Allocation

The system automatically calculates optimal resource allocation:

- **Workers per plane**: `min(cpu_count - 4, 30)` cores
- **Reserved cores**: 4 (for system operations)
- **Maximum job cores**: 30 (processing saturates beyond this)

### Allocation Table

| CPU Cores | Max Parallel Planes | Workers/Plane | Behavior                         |
|-----------|---------------------|---------------|----------------------------------|
| 16        | 1                   | 12            | Sequential plane processing      |
| 32        | 1                   | 28            | Sequential, 28 workers per plane |
| 64        | 2                   | 30            | 2 concurrent planes              |
| 96        | 3                   | 30            | 3 concurrent planes              |
| 128       | 4                   | 30            | 4 concurrent planes              |

---

## Error Handling

### Common Errors

| Error Message                           | Cause                    | Resolution                        |
|-----------------------------------------|--------------------------|-----------------------------------|
| "At least one recording path is required" | Empty recording_paths list | Provide at least one recording path |
| "Configuration file not found"          | Invalid configuration_path | Generate or verify configuration  |
| "Recording directory not found"           | Invalid path for recording | Verify path exists                |
| "Batch processing already in progress"  | Batch already running    | Wait for current batch to complete|

### Handling Failures

If processing fails for some recordings:

1. Note which recordings failed from the status output
2. Read the error messages in the output
3. Explain the errors to the user with root cause and resolution
4. Wait for the current batch to complete before starting retries

### Error-to-Skill Routing

When errors suggest upstream issues, invoke the appropriate skill to resolve before retrying:

| Error pattern                                     | Upstream skill to invoke          |
|---------------------------------------------------|-----------------------------------|
| Missing `cindra_parameters.json`, TIFF read error | `/acquisition-data-preparation`   |
| Invalid parameter values, wrong plane/channel     | `/acquisition-data-preparation`   |
| Configuration parameter issues, bad tau/channels  | `/single-recording-configuration` |
| MCP tools unavailable, server connection errors   | `/mcp-environment-setup`          |

---

## Post-Processing Next Steps

After batch processing completes successfully:

1. **Verify outputs** — invoke `/single-recording-results` to validate that all expected output files are
   present and correctly formatted.
2. **Inspect results** — invoke `/visualization` to launch viewers for visual inspection of registration
   quality, detected ROIs, and extracted traces.
3. **Multi-recording processing** — if the user plans to track ROIs across recordings, invoke
   `/multi-recording-configuration` to set up the multi-recording pipeline, then `/multi-recording-processing`
   to execute it.

---

## Related skills

| Skill                              | Relationship                                                                             |
|------------------------------------|------------------------------------------------------------------------------------------|
| `/mcp-environment-setup`           | Prerequisite: MCP server must be connected for processing tools                          |
| `/acquisition-data-preparation`    | Prerequisite: raw data must be prepared before processing                                |
| `/single-recording-configuration`  | Configuration reference for all pipeline parameters                                      |
| `/single-recording-results`        | Output data format reference for evaluating processing results                           |
| `/multi-recording-processing`      | Next step: multi-recording processing for longitudinal ROI tracking                      |
| `/multi-recording-configuration`   | Downstream: multi-recording configuration requires single-recording output               |
| `/visualization`                   | Next step: launch viewers to inspect and query processing results                        |
