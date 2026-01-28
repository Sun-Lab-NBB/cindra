---
name: suite2p-processing
description: >-
  Guides AI agents through neural imaging data processing workflows using this library's MCP server. Provides
  comprehensive documentation of data formats, pipeline architectures, and processing workflows for Sun Lab mesoscope
  experiments.
---

# Processing Neural Imaging Data

Guides AI agents through the workflow for processing neural imaging data from Sun Lab mesoscope experiments using the
sl-suite2p MCP server tools.

---

## Agent Requirements

**You MUST use the MCP tools provided by this library for all neural imaging data processing tasks.** The sl-suite2p
library provides an MCP server that exposes specialized tools for discovering sessions, executing pipelines, and
monitoring processing status. These tools are the only supported interface for agentic neural imaging data processing.

### Mandatory Tool Usage

- You MUST NOT import or call sl-suite2p Python functions directly (e.g., `from sl_suite2p.pipeline import ...`)
- You MUST NOT attempt to run processing by executing Python scripts or CLI commands
- You MUST use the MCP tools listed in the "Available Tools" section below
- You MUST verify the MCP server is connected before attempting any processing operations

### Why MCP Tools Are Required

The MCP tools provide:

1. **Background processing** - Jobs run in separate threads, allowing parallel session processing
2. **Intelligent batching** - Three-phase single-day processing and two-phase multi-day processing
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
ss2p mcp
```

Transport options:
- `ss2p mcp` - Default stdio transport
- `ss2p mcp -t sse` - Server-Sent Events transport
- `ss2p mcp -t streamable-http` - Streamable HTTP transport

### Claude Code Configuration

Add to your `.mcp.json` file in the project root:

```json
{
  "mcpServers": {
    "ss2p-mcp": {
      "type": "stdio",
      "command": "ss2p",
      "args": ["mcp"]
    }
  }
}
```

### Verifying Connection

Before processing, verify the MCP tools are available by checking your tool list. If the sl-suite2p tools
(`discover_sessions_tool`, `start_batch_processing_tool`, etc.) are not present, the server is not connected.

---

## Available Tools

The MCP server exposes tools for configuration, single-day processing, multi-day processing, and batch operations.

### Configuration Tools

| Tool                           | Purpose                                              |
|--------------------------------|------------------------------------------------------|
| `get_default_single_day_config`| Returns default single-day config as dictionary      |
| `get_default_multi_day_config` | Returns default multi-day config as dictionary       |
| `generate_config_file`         | Generates a configuration YAML file                  |

### Single-Day Tools

| Tool                      | Purpose                                                   |
|---------------------------|-----------------------------------------------------------|
| `run_single_day_pipeline` | Executes single-day processing for a single session       |
| `get_single_day_status`   | Gets processing status for a single session               |

### Multi-Day Tools

| Tool                      | Purpose                                                   |
|---------------------------|-----------------------------------------------------------|
| `run_multi_day_pipeline`  | Executes multi-day processing for a set of sessions       |
| `get_multi_day_status`    | Gets multi-day processing status for a session            |

### Single-Day Batch Processing Tools

| Tool                              | Purpose                                                      |
|-----------------------------------|--------------------------------------------------------------|
| `discover_sessions_tool`          | Finds sessions under a root directory                        |
| `start_batch_processing_tool`     | Starts batch single-day processing for multiple sessions     |
| `get_batch_processing_status_tool`| Returns status of single-day batch processing                |

### Multi-Day Batch Processing Tools

| Tool                                     | Purpose                                                 |
|------------------------------------------|---------------------------------------------------------|
| `start_multiday_batch_processing_tool`   | Starts batch multi-day processing for multiple animals  |
| `get_multiday_batch_processing_status_tool` | Returns status of multi-day batch processing         |

---

## Single-Day Pipeline Architecture

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

## Multi-Day Pipeline Architecture

The multi-day pipeline tracks cells across multiple recording sessions and extracts their fluorescence consistently.

### Prerequisites

- All sessions must have been processed with the single-day pipeline
- The `combine` step must have been run for each session

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
    ├── multi_day_ss2p_configuration.yaml    # Configuration snapshot
    ├── multiday_tracker.json                # Processing tracker (main session only)
    ├── template_cell_masks.npy              # Tracked cell masks
    ├── F.npy                                # Fluorescence traces
    ├── Fneu.npy                             # Neuropil traces
    └── spks.npy                             # Deconvolved spikes
```

---

## Tool Input/Output Formats

### `discover_sessions_tool`

**Input:**
```python
{
    "root_directory": "/path/to/data"  # Required, directory to search
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
    "skipped": [  # Optional, only present if sessions were filtered out
        "/path/to/data/animal1/2024-01-10-10-00-00-111111 (window checking)"
    ],
    "errors": [...]  # Optional, only present if errors occurred
}
```

### `start_batch_processing_tool`

**Input:**
```python
{
    "session_paths": ["/path/session1", "/path/session2", ...],  # Required, minimum 1
    "config_path": "/path/to/config.yaml",  # Required, configuration file
    "workers_per_plane": -1,  # Optional, -1 for automatic (max 30)
    "max_parallel_planes": -1  # Optional, -1 for automatic
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

**Input:** None (no parameters required)

**Output:**
```python
{
    "current_phase": "process",  # "binarize", "process", or "combine"
    "sessions": [
        {
            "session_name": "2024-01-15-10-30-00-123456",
            "status": "PROCESSING",  # QUEUED, PROCESSING, SUCCEEDED, FAILED
            "binarize": "done",      # "pending", "running", "done", "failed"
            "process": "3/4",        # Planes completed / total
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

### `start_multiday_batch_processing_tool`

**Input:**
```python
{
    "animal_configs": [  # List of animals to process
        {
            "config_path": "/path/to/animal1_config.yaml",
            "session_paths": ["/path/to/session1", "/path/to/session2"]
        },
        {
            "config_path": "/path/to/animal2_config.yaml",
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

**Input:** None (no parameters required)

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

---

## Formatting Status as a Table

When presenting status to the user, you MUST format the data as a clear table.

### Single-Day Batch Status Table

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

### Multi-Day Batch Status Table

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
- [ ] Session discovery complete (used discover_sessions_tool or received explicit paths)
- [ ] Configuration file exists and is valid for the pipeline type
- [ ] Asked user about CPU core allocation (see Resource Allocation section)
- [ ] Received user response confirming worker count
- [ ] Confirmed which sessions to process
```

**STOP**: If any checkbox is incomplete, do not proceed. Complete the missing steps first.

### Single-Day Batch Workflow

1. **Discover sessions** → Use `discover_sessions_tool` to find all session paths
2. **Generate configuration** → Use `generate_config_file` if needed
3. **Ask about CPU allocation** → Explain resource model and ask how many cores to use
4. **Start batch processing** → Call `start_batch_processing_tool`
5. **Inform user** → Report batch status and explain three-phase processing
6. **Check status on request** → Display formatted status table
7. **Explain any errors** → Analyze and explain errors when processing completes

### Multi-Day Batch Workflow

1. **Verify prerequisites** → Ensure all sessions have single-day processing complete
2. **Organize by animal** → Group sessions by animal/dataset
3. **Generate configurations** → Create multi-day config for each animal
4. **Start batch processing** → Call `start_multiday_batch_processing_tool`
5. **Monitor progress** → Check two-phase progress (discover → extract)
6. **Explain any errors** → Analyze failures when complete

---

## Resource Management

### CPU Core Allocation

The system automatically calculates optimal resource allocation:

- **Workers per plane/session**: `min(cpu_count - 4, 30)` cores
- **Reserved cores**: 4 (for system operations)
- **Maximum job cores**: 30 (processing saturates beyond this)

### Single-Day Batch Allocation

| CPU Cores | Max Parallel Planes | Workers/Plane | Behavior                         |
|-----------|---------------------|---------------|----------------------------------|
| 16        | 1                   | 12            | Sequential plane processing      |
| 32        | 1                   | 28            | Sequential, 28 workers per plane |
| 64        | 2                   | 30            | 2 concurrent planes              |
| 96        | 3                   | 30            | 3 concurrent planes              |
| 128       | 4                   | 30            | 4 concurrent planes              |

### Multi-Day Batch Allocation

| CPU Cores | Max Parallel Discovers | Max Parallel Extracts | Behavior                    |
|-----------|------------------------|----------------------|-----------------------------|
| 32        | 1                      | 1                    | Sequential processing       |
| 64        | 3                      | 2                    | Multiple animals in parallel |
| 96        | 4                      | 3                    | Higher parallelism          |
| 128       | 6                      | 4                    | Maximum parallelism         |

---

## Error Handling

### Common Errors

| Error Message                           | Cause                                  | Resolution                           |
|-----------------------------------------|----------------------------------------|--------------------------------------|
| "At least one session path is required" | Empty session_paths list               | Provide at least one session path    |
| "Configuration file not found"          | Invalid config_path                    | Generate or verify configuration     |
| "Session directory not found"           | Invalid path for session               | Verify path exists                   |
| "Batch processing already in progress"  | Batch already running                  | Wait for current batch to complete   |
| "No suite2p output directory found"     | Single-day processing not complete     | Run single-day pipeline first        |

### Handling Failures

If processing fails for some sessions:

1. Note which sessions failed from the status output
2. Read the error messages in the output
3. Consult DATA_ARCHITECTURE.md for data format issues
4. Explain the errors to the user with root cause and resolution
5. Wait for the current batch to complete before starting retries

---

## Data Architecture Reference

For detailed information about input data formats, TIFF structure, and session organization, refer to
[DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md).

Use this reference when:
- Answering user questions about input data requirements
- Debugging processing errors
- Understanding session directory structure
- Identifying missing prerequisite files
