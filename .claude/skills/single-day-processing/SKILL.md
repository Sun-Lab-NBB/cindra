---
name: single-day-processing
description: >-
  Guides AI agents through single-day (within-session) neural imaging data processing using the cindra MCP server.
  Covers session discovery, configuration, batch processing, and status monitoring for individual recording sessions.
---

# Single-Day Neural Imaging Processing

Guides AI agents through the workflow for processing neural imaging data from individual recording sessions using the
cindra MCP server tools.

---

## Agent Requirements

**You MUST use the MCP tools provided by this library for all neural imaging data processing tasks.** The cindra
library provides an MCP server that exposes specialized tools for discovering sessions, executing pipelines, and
monitoring processing status. These tools are the only supported interface for agentic neural imaging data processing.

### Mandatory Tool Usage

- You MUST NOT import or call cindra Python functions directly (e.g., `from cindra.pipeline import ...`)
- You MUST NOT attempt to run processing by executing Python scripts or CLI commands
- You MUST use the MCP tools listed in the "Available Tools" section below
- You MUST verify the MCP server is connected before attempting any processing operations

### Why MCP Tools Are Required

The MCP tools provide:

1. **Background processing** - Jobs run in separate threads, allowing parallel session processing
2. **Intelligent batching** - Three-phase processing (binarize → process → combine)
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
(`discover_sessions_tool`, `start_batch_processing_tool`, etc.) are not present, the server is not connected.

---

## Available Tools

The MCP server exposes a unified API where all processing goes through the batch manager, even for single sessions.

### Configuration Tool

| Tool                   | Purpose                                                    |
|------------------------|------------------------------------------------------------|
| `generate_config_file` | Generates a default configuration YAML file (single-day)   |

### Session Discovery and Status Tools

| Tool                     | Purpose                                             |
|--------------------------|-----------------------------------------------------|
| `discover_sessions_tool` | Finds sessions under a root directory               |
| `get_single_day_status`  | Checks filesystem for single-day processing outputs |

### Batch Processing Tools

| Tool                               | Purpose                                            |
|------------------------------------|----------------------------------------------------|
| `start_batch_processing_tool`      | Starts batch single-day processing (1+ sessions)   |
| `get_batch_processing_status_tool` | Returns in-memory status of running batch          |
| `cancel_batch_processing_tool`     | Cancels batch processing, clears queues            |

---

## Pipeline Architecture

The single-day pipeline processes brain imaging data from a single recording session through three sequential phases.

### Three-Phase Processing

```
Phase 1: BINARIZE (I/O bound)
├── Converts raw TIFFs to binary format
├── Determines plane count for the session
└── Sequential - I/O limited

Phase 2: PROCESS (CPU bound)
├── Motion correction (registration)
├── ROI detection (cell identification)
├── Signal extraction (fluorescence traces)
├── Processes each plane independently
└── Parallel - up to 30 workers per plane

Phase 3: COMBINE (I/O bound)
├── Merges data from all planes
├── Creates unified dataset
└── Sequential - I/O limited
```

### Batch Processing Architecture

When processing multiple sessions:

```
Phase 1: BINARIZE (Sequential across sessions)
├── Session 1: binarize → determines plane count
├── Session 2: binarize → determines plane count
└── Session N: binarize → determines plane count

Phase 2: PROCESS (Parallel across session-plane pairs)
├── Session 1, Plane 0 ─┐
├── Session 1, Plane 1  │
├── Session 2, Plane 0  ├── Parallel batch
├── Session 2, Plane 1  │
└── ...                 ┘

Phase 3: COMBINE (Sequential across sessions)
├── Session 1: combine
├── Session 2: combine
└── Session N: combine
```

### Output Structure

Results are saved to `{session_path}/suite2p/`:

```
suite2p/
├── plane0/          # Per-plane processing results
│   ├── data.bin     # Registered binary data
│   ├── ops.npy      # Processing parameters
│   ├── stat.npy     # ROI statistics
│   ├── F.npy        # Fluorescence traces
│   ├── Fneu.npy     # Neuropil traces
│   └── spks.npy     # Deconvolved spikes
├── plane1/
├── ...
└── combined/        # Merged results from all planes
    ├── ops.npy
    ├── stat.npy
    ├── F.npy
    ├── Fneu.npy
    ├── spks.npy
    └── iscell.npy
```

---

## Tool Input/Output Formats

### `generate_config_file`

**Input:**
```python
{
    "output_path": "/path/to/config.yaml",  # Required
    "pipeline_type": "single-day"           # Required
}
```

**Output:**
```python
{
    "success": True,
    "file_path": "/path/to/config.yaml",
    "pipeline_type": "single-day"
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

### `start_batch_processing_tool`

**Input:**
```python
{
    "session_paths": ["/path/session1", "/path/session2", ...],  # Required, minimum 1
    "config_path": "/path/to/config.yaml",  # Required
    "workers_per_plane": -1,    # Optional, -1 for automatic (max 30)
    "max_parallel_planes": -1   # Optional, -1 for automatic
}
```

**Output:**
```python
{
    "started": True,
    "total_sessions": 30,
    "workers_per_plane": 28,
    "max_parallel_planes": 2,
    "message": "Batch processing started. Use get_batch_processing_status_tool to monitor progress."
}
```

### `get_batch_processing_status_tool`

**Input:** None

**Output:**
```python
{
    "current_phase": "process",  # "binarize", "process", or "combine"
    "sessions": [
        {
            "session_name": "2024-01-15-10-30-00-123456",
            "status": "PROCESSING",
            "binarize": "done",
            "process": "3/4",
            "combine": "pending"
        },
        ...
    ],
    "summary": {
        "total": 30,
        "binarize_completed": 30,
        "process_completed": 15,
        "combine_completed": 10,
        "failed": 0
    }
}
```

### `cancel_batch_processing_tool`

**Input:** None

**Output:**
```python
{
    "cancelled": True,
    "message": "Single-day batch processing cancelled. Active jobs will complete but no new jobs will start.",
    "final_state": {
        "binarize_completed": 5,
        "process_completed": 12,
        "combine_completed": 3,
        "active_jobs_at_cancel": 2
    }
}
```

---

## Formatting Status as a Table

When presenting status to the user, format the data as a clear table:

```
**Single-Day Batch Processing Status**

Current Phase: PROCESS
Summary: 10/30 sessions complete | 2 processing | 18 queued | 0 failed

| Session                      | Binarize | Process | Combine | Status     |
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
- [ ] Session discovery complete (used discover_sessions_tool or received explicit paths)
- [ ] Configuration file confirmed or created (see Configuration Guidance section)
- [ ] Asked about exclusions if creating new config (flyback planes, ignored files)
- [ ] Asked user about CPU core allocation (see Resource Allocation section)
- [ ] Received user response confirming worker count
- [ ] Confirmed which sessions to process
```

**STOP**: If any checkbox is incomplete, do not proceed. Complete the missing steps first.

### Workflow Steps

1. **Discover sessions** → Use `discover_sessions_tool` to find all session paths
2. **Check configuration** → Ask user if they have an existing config (see Configuration Guidance)
3. **Create config if needed** → Generate default and ask about exclusions (flyback planes, ignored files)
4. **Ask about CPU allocation** → Explain resource model and ask how many cores to use
5. **Start batch processing** → Call `start_batch_processing_tool`
6. **Inform user** → Report batch status and explain three-phase processing
7. **Check status on request** → Display formatted status table
8. **Explain any errors** → Analyze and explain errors when processing completes

---

## Configuration Guidance

**CRITICAL**: You MUST ask the user about configuration before processing. Never skip this step.

For complete parameter documentation, invoke `/single-day-config`.

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
1. Use `generate_config_file` with `pipeline_type: "single-day"` to create a default configuration file
2. Ask about exclusions (see below)
3. Proceed to CPU core allocation

### Key Questions for New Configurations

The default configuration works well for GCaMP6f data from 2P mesoscope. Only ask about exclusions:

- "Are there any flyback planes to exclude from processing? If so, provide the plane indices (0-based, e.g., [0] or [3])"
- "Are there any TIFF files that should be ignored? If so, provide the exact file names to skip"

### Auto-Overwritten Parameters (Mesoscope)

When `mesoscan=True`, these parameters are **automatically overwritten** from `suite2p_parameters.json`:
- `nplanes` - Set to `plane_number × roi_number`
- `fs` - Set to `frame_rate`
- ROI geometry (`lines`, `dx`, `dy`, `nrois`)

User-specified values for these parameters are ignored. See `/single-day-config` for complete details.

### Multi-Day Compatibility

If the user intends to run multi-day processing on these sessions later, ensure:
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
| "At least one session path is required" | Empty session_paths list | Provide at least one session path |
| "Configuration file not found"          | Invalid config_path      | Generate or verify configuration  |
| "Session directory not found"           | Invalid path for session | Verify path exists                |
| "Batch processing already in progress"  | Batch already running    | Wait for current batch to complete|

### Handling Failures

If processing fails for some sessions:

1. Note which sessions failed from the status output
2. Read the error messages in the output
3. Explain the errors to the user with root cause and resolution
4. Wait for the current batch to complete before starting retries
