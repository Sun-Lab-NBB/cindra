---
name: multi-day-processing
description: >-
  Guides AI agents through multi-day (cross-session) neural imaging data processing using the cindra MCP server.
  Covers cell tracking across sessions, configuration, batch processing, and status monitoring for longitudinal studies.
---

# Multi-Day Neural Imaging Processing

Guides AI agents through the workflow for tracking cells across multiple recording sessions and extracting consistent
fluorescence traces using the cindra MCP server tools.

---

## Prerequisites

**Before using this skill, ensure:**

1. All sessions have been processed with the single-day pipeline
2. The `combine` step completed for each session (cindra/combined/ exists)
3. Sessions are from the same animal and imaging region

If single-day processing is not complete, use the `/single-day-processing` skill first.

---

## Agent Requirements

**You MUST use the MCP tools provided by this library for all neural imaging data processing tasks.** The cindra
library provides an MCP server that exposes specialized tools for discovering sessions, executing pipelines, and
monitoring processing status. These tools are the only supported interface for agentic neural imaging data processing.

### Mandatory Tool Usage

- You MUST NOT import or call cindra Python functions directly (e.g., `from cindra.multi_day import ...`)
- You MUST NOT attempt to run processing by executing Python scripts or CLI commands
- You MUST use the MCP tools listed in the "Available Tools" section below
- You MUST verify the MCP server is connected before attempting any processing operations

### Why MCP Tools Are Required

The MCP tools provide:

1. **Background processing** - Jobs run in separate threads, allowing parallel animal processing
2. **Intelligent batching** - Two-phase processing (discover → extract)
3. **Automatic queuing** - Sessions beyond parallel capacity are queued and started automatically
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
(`discover_sessions_tool`, `start_multiday_batch_processing_tool`, etc.) are not present, the server is not connected.

---

## Available Tools

The MCP server exposes a unified API where all processing goes through the batch manager, even for single animals.

### Configuration Tool

| Tool                   | Purpose                                                   |
|------------------------|-----------------------------------------------------------|
| `generate_config_file` | Generates a default configuration YAML file (multi-day)   |

### Session Discovery and Status Tools

| Tool                     | Purpose                                            |
|--------------------------|----------------------------------------------------|
| `discover_sessions_tool` | Finds sessions under a root directory              |
| `get_multi_day_status`   | Checks filesystem for multi-day processing outputs |

### Batch Processing Tools

| Tool                                        | Purpose                                       |
|---------------------------------------------|-----------------------------------------------|
| `start_multiday_batch_processing_tool`      | Starts batch multi-day processing (1+ animals)|
| `get_multiday_batch_processing_status_tool` | Returns in-memory status of running batch     |
| `cancel_multiday_batch_processing_tool`     | Cancels batch processing, clears queues       |

---

## Pipeline Architecture

The multi-day pipeline tracks cells across multiple recording sessions and extracts their fluorescence consistently.

### Two-Phase Processing

```
Phase 1: DISCOVER (Mixed - internal parallelization)
├── Registers all sessions to common reference frame
├── Clusters cell masks across sessions
├── Generates template masks for tracked cells
├── Uses 20 workers for deformation application
└── Registration computation is sequential per animal

Phase 2: EXTRACT (CPU bound, independent sessions)
├── Applies template masks to extract fluorescence
├── Computes neuropil signals
├── Performs spike deconvolution
├── Each session uses up to 30 workers
└── All sessions can be extracted in parallel
```

### Batch Processing Architecture

When processing multiple animals:

```
Phase 1: DISCOVER (Parallel across animals if cores allow)
├── Animal 1: discover (registration + deformation)
├── Animal 2: discover (registration + deformation)
└── Animal N: discover
    [Can run multiple animals in parallel]

Phase 2: EXTRACT (Parallel across all sessions)
├── Animal 1, Session 1 ─┐
├── Animal 1, Session 2  │
├── Animal 2, Session 1  ├── Parallel batch
├── Animal 2, Session 2  │
└── ...                  ┘
```

### Output Structure

Results are saved to `{session_path}/multiday/{dataset_name}/`:

```
multiday/
└── dataset_name/
    ├── ops.npy                              # Processing parameters
    ├── multi_day_cindra_configuration.yaml    # Configuration snapshot
    ├── multiday_tracker.json                # Processing tracker (main session only)
    ├── template_cell_masks.npy              # Tracked cell masks
    ├── F.npy                                # Fluorescence traces
    ├── Fneu.npy                             # Neuropil traces
    └── spks.npy                             # Deconvolved spikes
```

---

## Tool Input/Output Formats

### `generate_config_file`

**Input:**
```python
{
    "output_path": "/path/to/config.yaml",  # Required
    "pipeline_type": "multi-day"            # Required
}
```

**Output:**
```python
{
    "success": True,
    "file_path": "/path/to/config.yaml",
    "pipeline_type": "multi-day"
}
```

### `discover_sessions_tool`

**Input:**
```python
{
    "root_directory": "/path/to/data"  # Required
}
```

**Output:**
```python
{
    "sessions": [
        "/path/to/data/animal1/2024-01-15-10-30-00-123456",
        "/path/to/data/animal1/2024-01-16-09-00-00-234567",
        ...
    ],
    "count": 30,
    "skipped": [...],  # Optional
    "errors": [...]    # Optional
}
```

### `start_multiday_batch_processing_tool`

**Input:**
```python
{
    "animal_configurations": [  # List of animals to process
        {
            "configuration_path": "/path/to/animal1_config.yaml",
            "session_paths": ["/path/to/session1", "/path/to/session2"]
        },
        {
            "configuration_path": "/path/to/animal2_config.yaml",
            "session_paths": ["/path/to/session3", "/path/to/session4"]
        }
    ],
    "workers_per_discover": 20,   # Optional, workers for discover phase
    "workers_per_extract": -1     # Optional, -1 for automatic (max 30)
}
```

**Output:**
```python
{
    "started": True,
    "total_animals": 2,
    "total_sessions": 4,
    "workers_per_discover": 20,
    "workers_per_extract": 28,
    "message": "Multi-day batch processing started."
}
```

### `get_multiday_batch_processing_status_tool`

**Input:** None

**Output:**
```python
{
    "current_phase": "extract",  # "discover" or "extract"
    "animals": [
        {
            "animal_key": "animal1_dataset",
            "status": "EXTRACTING",
            "discover": "done",
            "extract_completed": 1,
            "extract_total": 2
        },
        ...
    ],
    "summary": {
        "total_animals": 2,
        "discover_completed": 2,
        "extract_completed": 2,
        "extract_total": 4,
        "failed": 0
    }
}
```

### `cancel_multiday_batch_processing_tool`

**Input:** None

**Output:**
```python
{
    "cancelled": True,
    "message": "Multi-day batch processing cancelled. Active jobs will complete but no new jobs will start.",
    "final_state": {
        "discover_completed": 1,
        "extract_completed": 2,
        "active_jobs_at_cancel": 1
    }
}
```

---

## Formatting Status as a Table

When presenting status to the user, format the data as a clear table:

```
**Multi-Day Batch Processing Status**

Current Phase: EXTRACT
Summary: 1/2 animals complete | 2/4 sessions extracted | 0 failed

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
- [ ] Verified all sessions have single-day processing complete (cindra/combined/ exists)
- [ ] Organized sessions by animal/dataset
- [ ] Configuration file confirmed or created (see Configuration Guidance section)
- [ ] Asked about dataset name and MROI region margin if creating new config
- [ ] Asked user about CPU core allocation (see Resource Allocation section)
- [ ] Received user response confirming worker count
- [ ] Confirmed which sessions to process per animal
```

**STOP**: If any checkbox is incomplete, do not proceed. Complete the missing steps first.

### Workflow Steps

1. **Verify prerequisites** → Ensure all sessions have single-day processing complete
2. **Organize by animal** → Group sessions by animal/dataset
3. **Check configuration** → Ask user if they have an existing config (see Configuration Guidance)
4. **Create config if needed** → Generate default and ask about dataset name + MROI region margin
5. **Ask about CPU allocation** → Explain resource model and ask how many cores to use
6. **Start batch processing** → Call `start_multiday_batch_processing_tool`
7. **Monitor progress** → Check two-phase progress (discover → extract)
8. **Explain any errors** → Analyze failures when complete

---

## Configuration Guidance

**CRITICAL**: You MUST ask the user about configuration before processing. Never skip this step.

For complete parameter documentation, invoke `/multi-day-config`.

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
1. Use `generate_config_file` with `pipeline_type: "multi-day"` to create a default configuration file
2. Ask about required/optional parameters (see below)
3. Proceed to CPU core allocation

### Key Questions for New Configurations

The default configuration works well for GCaMP6f data. Ask these questions:

- **Required**: "What name should identify this multi-day dataset?" (sets `io.dataset_name`)
- **Optional**: "Is this an MROI (multi-region) recording? If so, you can adjust the region border margin for cell
  filtering" (sets `cell_selection.mroi_region_margin`)

### Pipeline-Set Parameters

These are set automatically by the MCP batch tool:
- `io.session_directories` - Set from `session_paths` argument
- `main.parallel_workers` - Set from worker arguments

See `/multi-day-config` for complete parameter documentation including cell selection, registration tuning, and
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
| "No cindra output directory found"     | Single-day processing incomplete| Run single-day pipeline first     |
| "Configuration file not found"         | Invalid configuration_path     | Generate or verify configuration  |
| "Session directory not found"          | Invalid path for session       | Verify path exists                |
| "Batch processing already in progress" | Batch already running          | Wait for current batch to complete|

### Handling Failures

If processing fails for some animals/sessions:

1. Note which animals/sessions failed from the status output
2. Read the error messages in the output
3. Explain the errors to the user with root cause and resolution
4. Wait for the current batch to complete before starting retries

### Multi-Day Specific Errors

| Error Pattern                          | Likely Cause                           | Resolution                           |
|----------------------------------------|----------------------------------------|--------------------------------------|
| `No cindra output found`               | Single-day processing incomplete       | Complete single-day pipeline first   |
| `Session IDs mismatch`                 | Configuration doesn't match sessions   | Verify session_directories in config |
| `Registration failed between sessions` | Too much drift between days            | Check FOV alignment                  |
| `No trackable cells found`             | Insufficient overlap in detected cells | Adjust clustering threshold          |

