"""This module provides the main MCP server implementation for sl-suite2p."""

import json
from typing import Any
import asyncio
from pathlib import Path
from dataclasses import fields

from mcp import types
import mcp.server.stdio
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel import Server, NotificationOptions

from ..version import version, python_version
from ..pipeline import process_multi_day, process_single_day
from ..configuration import (
    MultiDayS2PConfiguration,
    SingleDayS2PConfiguration,
    generate_default_ops,
    generate_default_multiday_ops,
)

# Minimum number of sessions required for multi-day processing.
_MINIMUM_SESSION_COUNT: int = 2


def _dataclass_to_schema(cls: type) -> dict[str, Any]:
    """Converts a dataclass to a simple schema dictionary."""
    schema = {}
    for field in fields(cls):
        field_type = str(field.type)
        default = field.default if field.default is not field.default_factory else None
        if default is None and field.default_factory is not field.default_factory:
            try:
                default = field.default_factory()
            except TypeError:
                default = "factory"

        schema[field.name] = {
            "type": field_type,
            "default": default,
        }

        if field.metadata and "description" in field.metadata:
            schema[field.name]["description"] = field.metadata["description"]

    return schema


def _get_single_day_schema() -> dict[str, Any]:
    """Generates the schema for single-day configuration."""
    config = SingleDayS2PConfiguration()
    schema = {}

    for field in fields(config):
        section = getattr(config, field.name)
        if hasattr(section, "__dataclass_fields__"):
            schema[field.name] = _dataclass_to_schema(type(section))
        else:
            schema[field.name] = {"type": str(type(section).__name__), "value": section}

    return schema


def _get_multi_day_schema() -> dict[str, Any]:
    """Generates the schema for multi-day configuration."""
    config = MultiDayS2PConfiguration()
    schema = {}

    for field in fields(config):
        section = getattr(config, field.name)
        if hasattr(section, "__dataclass_fields__"):
            schema[field.name] = _dataclass_to_schema(type(section))
        else:
            schema[field.name] = {"type": str(type(section).__name__), "value": section}

    return schema


# Create server instance
server = Server("sl-suite2p")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Lists all available tools."""
    return [
        # Configuration tools
        types.Tool(
            name="get_default_single_day_config",
            description="Returns the default single-day pipeline configuration as a dictionary.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="get_default_multi_day_config",
            description="Returns the default multi-day pipeline configuration as a dictionary.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="generate_config_file",
            description="Generates a configuration YAML file for the specified pipeline type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "The absolute path where the configuration file should be saved.",
                    },
                    "pipeline_type": {
                        "type": "string",
                        "enum": ["single-day", "multi-day"],
                        "description": "The type of pipeline configuration to generate.",
                    },
                },
                "required": ["output_path", "pipeline_type"],
            },
        ),
        # Single-day pipeline tools
        types.Tool(
            name="run_single_day_pipeline",
            description="Executes the single-day suite2p processing pipeline.",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_path": {
                        "type": "string",
                        "description": "The absolute path to the configuration YAML file.",
                    },
                    "session_path": {
                        "type": "string",
                        "description": "The absolute path to the session data directory.",
                    },
                    "binarize": {
                        "type": "boolean",
                        "description": "Run the binarization step (step 1).",
                        "default": False,
                    },
                    "process": {
                        "type": "boolean",
                        "description": "Run the processing step (step 2).",
                        "default": False,
                    },
                    "combine": {
                        "type": "boolean",
                        "description": "Run the combination step (step 3).",
                        "default": False,
                    },
                    "target_plane": {
                        "type": "integer",
                        "description": "Specific plane index to process (-1 for all).",
                        "default": -1,
                    },
                    "workers": {
                        "type": "integer",
                        "description": "Number of parallel workers (-1 for all cores).",
                        "default": -1,
                    },
                },
                "required": ["config_path", "session_path"],
            },
        ),
        types.Tool(
            name="get_single_day_status",
            description="Gets the processing status of a single-day session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_path": {
                        "type": "string",
                        "description": "The absolute path to the session data directory.",
                    },
                },
                "required": ["session_path"],
            },
        ),
        # Multi-day pipeline tools
        types.Tool(
            name="run_multi_day_pipeline",
            description="Executes the multi-day suite2p processing pipeline.",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_path": {
                        "type": "string",
                        "description": "The absolute path to the configuration YAML file.",
                    },
                    "session_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of absolute paths to session directories (minimum 2).",
                    },
                    "discover": {
                        "type": "boolean",
                        "description": "Run the discovery step (step 1).",
                        "default": False,
                    },
                    "extract": {
                        "type": "boolean",
                        "description": "Run the extraction step (step 2).",
                        "default": False,
                    },
                    "target_session": {
                        "type": "string",
                        "description": "Specific session ID for extraction (null for all).",
                        "default": None,
                    },
                    "workers": {
                        "type": "integer",
                        "description": "Number of parallel workers (-1 for all cores).",
                        "default": -1,
                    },
                },
                "required": ["config_path", "session_paths"],
            },
        ),
        types.Tool(
            name="get_multi_day_status",
            description="Gets the multi-day processing status for a session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_path": {
                        "type": "string",
                        "description": "The absolute path to a session directory.",
                    },
                },
                "required": ["session_path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handles tool invocations."""
    if name == "get_default_single_day_config":
        config: SingleDayS2PConfiguration = await asyncio.to_thread(generate_default_ops, as_dict=False)
        return config.to_ops()

    if name == "get_default_multi_day_config":
        config: MultiDayS2PConfiguration = await asyncio.to_thread(generate_default_multiday_ops, as_dict=False)
        return config.to_ops()

    if name == "generate_config_file":
        return await _generate_config_file(arguments["output_path"], arguments["pipeline_type"])

    if name == "run_single_day_pipeline":
        return await _run_single_day_pipeline(
            config_path=arguments["config_path"],
            session_path=arguments["session_path"],
            binarize=arguments.get("binarize", False),
            process=arguments.get("process", False),
            combine=arguments.get("combine", False),
            target_plane=arguments.get("target_plane", -1),
            workers=arguments.get("workers", -1),
        )

    if name == "get_single_day_status":
        return await _get_single_day_status(arguments["session_path"])

    if name == "run_multi_day_pipeline":
        return await _run_multi_day_pipeline(
            config_path=arguments["config_path"],
            session_paths=arguments["session_paths"],
            discover=arguments.get("discover", False),
            extract=arguments.get("extract", False),
            target_session=arguments.get("target_session"),
            workers=arguments.get("workers", -1),
        )

    if name == "get_multi_day_status":
        return await _get_multi_day_status(arguments["session_path"])

    msg = f"Unknown tool: {name}"
    raise ValueError(msg)


async def _generate_config_file(output_path: str, pipeline_type: str) -> dict[str, Any]:
    """Generates a configuration YAML file."""
    output = Path(output_path)

    if pipeline_type not in ("single-day", "multi-day"):
        return {
            "success": False,
            "error": f"Invalid pipeline_type '{pipeline_type}'. Must be 'single-day' or 'multi-day'.",
        }

    if not output.parent.exists():
        return {
            "success": False,
            "error": f"Parent directory does not exist: {output.parent}",
        }

    if output.suffix != ".yaml":
        output = output.with_suffix(".yaml")

    if pipeline_type == "single-day":
        config: SingleDayS2PConfiguration = await asyncio.to_thread(generate_default_ops, as_dict=False)
    else:
        config: MultiDayS2PConfiguration = await asyncio.to_thread(generate_default_multiday_ops, as_dict=False)

    await asyncio.to_thread(config.to_config, file_path=output)

    return {
        "success": True,
        "file_path": str(output),
        "pipeline_type": pipeline_type,
    }


async def _run_single_day_pipeline(
    config_path: str,
    session_path: str,
    binarize: bool,
    process: bool,
    combine: bool,
    target_plane: int,
    workers: int,
) -> dict[str, Any]:
    """Executes the single-day pipeline."""
    config = Path(config_path)
    session = Path(session_path)

    if not config.exists():
        return {"success": False, "error": f"Configuration file not found: {config_path}"}

    if config.suffix != ".yaml":
        return {"success": False, "error": f"Configuration file must be a .yaml file: {config_path}"}

    if not session.exists():
        return {"success": False, "error": f"Session directory not found: {session_path}"}

    if not session.is_dir():
        return {"success": False, "error": f"Session path must be a directory: {session_path}"}

    # If no steps specified, run all steps
    if not any([binarize, process, combine]):
        binarize = True
        process = True
        combine = True

    try:
        await asyncio.to_thread(
            process_single_day,
            configuration_path=config,
            session_path=session,
            binarize=binarize,
            process=process,
            combine=combine,
            target_plane=target_plane,
            workers=workers,
            progress_bars=False,
        )

        return {
            "success": True,
            "session_path": str(session),
            "steps_executed": {"binarize": binarize, "process": process, "combine": combine},
            "target_plane": target_plane if target_plane >= 0 else "all",
        }

    except Exception as e:
        return {"success": False, "error": str(e), "session_path": str(session)}


async def _get_single_day_status(session_path: str) -> dict[str, Any]:
    """Gets single-day processing status."""
    session = Path(session_path)

    if not session.exists():
        return {"success": False, "error": f"Session directory not found: {session_path}"}

    suite2p_path = session / "suite2p"
    if not suite2p_path.exists():
        return {
            "success": True,
            "session_path": str(session),
            "status": "not_started",
            "message": "No suite2p output directory found",
        }

    combined_path = suite2p_path / "combined"
    planes = [p for p in suite2p_path.iterdir() if p.is_dir() and p.name.startswith("plane")]

    status: dict[str, Any] = {
        "success": True,
        "session_path": str(session),
        "suite2p_path": str(suite2p_path),
        "planes_found": len(planes),
        "combined_exists": combined_path.exists(),
    }

    if combined_path.exists():
        status["combined_files"] = {
            "ops": (combined_path / "ops.npy").exists(),
            "stat": (combined_path / "stat.npy").exists(),
            "F": (combined_path / "F.npy").exists(),
            "Fneu": (combined_path / "Fneu.npy").exists(),
            "spks": (combined_path / "spks.npy").exists(),
            "iscell": (combined_path / "iscell.npy").exists(),
        }

    return status


async def _run_multi_day_pipeline(
    config_path: str,
    session_paths: list[str],
    discover: bool,
    extract: bool,
    target_session: str | None,
    workers: int,
) -> dict[str, Any]:
    """Executes the multi-day pipeline."""
    config = Path(config_path)

    if not config.exists():
        return {"success": False, "error": f"Configuration file not found: {config_path}"}

    if config.suffix != ".yaml":
        return {"success": False, "error": f"Configuration file must be a .yaml file: {config_path}"}

    if len(session_paths) < _MINIMUM_SESSION_COUNT:
        return {"success": False, "error": "At least two session paths are required for multi-day processing."}

    sessions = [Path(p) for p in session_paths]
    for session in sessions:
        if not session.exists():
            return {"success": False, "error": f"Session directory not found: {session}"}
        if not session.is_dir():
            return {"success": False, "error": f"Session path must be a directory: {session}"}

    # If no steps specified, run all steps
    if not any([discover, extract]):
        discover = True
        extract = True

    overrides = {"session_directories": session_paths}

    try:
        await asyncio.to_thread(
            process_multi_day,
            configuration_path=config,
            discover=discover,
            extract=extract,
            target_session=target_session,
            workers=workers,
            progress_bars=False,
            overrides=overrides,
        )

        return {
            "success": True,
            "session_count": len(session_paths),
            "steps_executed": {"discover": discover, "extract": extract},
            "target_session": target_session if target_session else "all",
        }

    except Exception as e:
        return {"success": False, "error": str(e), "session_count": len(session_paths)}


async def _get_multi_day_status(session_path: str) -> dict[str, Any]:
    """Gets multi-day processing status."""
    session = Path(session_path)

    if not session.exists():
        return {"success": False, "error": f"Session directory not found: {session_path}"}

    multiday_base = session / "multiday"
    if not multiday_base.exists():
        parent_multiday = session.parent / "multiday"
        if parent_multiday.exists():
            multiday_base = parent_multiday
        else:
            return {
                "success": True,
                "session_path": str(session),
                "status": "not_started",
                "message": "No multiday output directory found",
            }

    datasets = [d for d in multiday_base.iterdir() if d.is_dir()]

    if not datasets:
        return {
            "success": True,
            "session_path": str(session),
            "status": "not_started",
            "message": "No dataset folders found in multiday directory",
        }

    dataset_statuses = {}
    for dataset in datasets:
        dataset_status: dict[str, Any] = {
            "ops_exists": (dataset / "ops.npy").exists(),
            "config_exists": (dataset / "multi_day_ss2p_configuration.yaml").exists(),
            "tracker_exists": (dataset / "multiday_tracker.json").exists(),
            "template_masks_exists": (dataset / "template_cell_masks.npy").exists(),
            "F_exists": (dataset / "F.npy").exists(),
            "Fneu_exists": (dataset / "Fneu.npy").exists(),
            "spks_exists": (dataset / "spks.npy").exists(),
        }

        if dataset_status["F_exists"]:
            dataset_status["status"] = "completed"
        elif dataset_status["template_masks_exists"]:
            dataset_status["status"] = "discovery_completed"
        elif dataset_status["ops_exists"]:
            dataset_status["status"] = "initialized"
        else:
            dataset_status["status"] = "unknown"

        dataset_status["is_main_session"] = dataset_status["tracker_exists"]
        dataset_statuses[dataset.name] = dataset_status

    return {
        "success": True,
        "session_path": str(session),
        "multiday_path": str(multiday_base),
        "datasets": dataset_statuses,
    }


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """Lists all available resources."""
    return [
        types.Resource(
            uri="pipeline://version-info",
            name="Version Info",
            description="Returns sl-suite2p version and environment information.",
            mimeType="application/json",
        ),
        types.Resource(
            uri="pipeline://configuration/single-day-schema",
            name="Single-Day Configuration Schema",
            description="Returns the schema for single-day pipeline configuration parameters.",
            mimeType="application/json",
        ),
        types.Resource(
            uri="pipeline://configuration/multi-day-schema",
            name="Multi-Day Configuration Schema",
            description="Returns the schema for multi-day pipeline configuration parameters.",
            mimeType="application/json",
        ),
        types.Resource(
            uri="pipeline://help/single-day",
            name="Single-Day Pipeline Help",
            description="Returns help information for the single-day pipeline.",
            mimeType="text/markdown",
        ),
        types.Resource(
            uri="pipeline://help/multi-day",
            name="Multi-Day Pipeline Help",
            description="Returns help information for the multi-day pipeline.",
            mimeType="text/markdown",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Reads a resource by URI."""
    if uri == "pipeline://version-info":
        info = {
            "sl_suite2p_version": version,
            "python_version": python_version,
            "description": "Enhanced suite2p implementation with multi-day cell tracking",
        }
        return json.dumps(info, indent=2)

    if uri == "pipeline://configuration/single-day-schema":
        schema = _get_single_day_schema()
        return json.dumps(schema, indent=2, default=str)

    if uri == "pipeline://configuration/multi-day-schema":
        schema = _get_multi_day_schema()
        return json.dumps(schema, indent=2, default=str)

    if uri == "pipeline://help/single-day":
        return """# Single-Day Suite2p Pipeline

The single-day pipeline processes brain imaging data from a single recording session.

## Steps

1. **Binarize** (step 1): Converts raw TIFF files to binary format for efficient processing.
2. **Process** (step 2): For each imaging plane:
   - Motion correction (registration)
   - ROI detection (cell identification)
   - Signal extraction (fluorescence traces)
3. **Combine** (step 3): Merges data from all planes into a unified dataset.

## Usage

Use the `run_single_day_pipeline` tool with:
- `config_path`: Path to configuration YAML file
- `session_path`: Path to session data directory
- `binarize`, `process`, `combine`: Boolean flags to select steps
- `target_plane`: Specific plane index (-1 for all)
- `workers`: Number of parallel workers (-1 for all cores)

## Output

Results are saved to `{session_path}/suite2p/` with subdirectories for each plane
and a `combined/` directory with merged results.
"""

    if uri == "pipeline://help/multi-day":
        return """# Multi-Day Suite2p Pipeline

The multi-day pipeline tracks cells across multiple recording sessions and extracts
their fluorescence signals consistently.

## Prerequisites

- All sessions must have been processed with the single-day pipeline
- The `combine` step must have been run for each session

## Steps

1. **Discover** (step 1): Identifies cells that can be reliably tracked across sessions:
   - Registers all sessions to a common reference frame
   - Clusters cell masks across sessions
   - Generates template masks for tracked cells

2. **Extract** (step 2): For each session:
   - Applies template masks to extract fluorescence
   - Computes neuropil signals
   - Performs spike deconvolution

## Usage

Use the `run_multi_day_pipeline` tool with:
- `config_path`: Path to configuration YAML file
- `session_paths`: List of session directory paths (minimum 2)
- `discover`, `extract`: Boolean flags to select steps
- `target_session`: Specific session ID for extraction (None for all)
- `workers`: Number of parallel workers (-1 for all cores)

## Output

Results are saved to `{session_path}/multiday/{dataset_name}/` for each session.
The first session (after natural sorting) is the "main session" and stores the
processing tracker.
"""

    msg = f"Unknown resource: {uri}"
    raise ValueError(msg)


def create_server() -> Server:
    """Returns the configured sl-suite2p MCP server.

    Returns:
        The MCP Server instance with all tools and resources registered.
    """
    return server


async def run_server() -> None:
    """Runs the sl-suite2p MCP server using stdio transport.

    This function starts the MCP server and listens for requests via stdin/stdout.
    It is designed to be called from the CLI command `ss2p mcp serve`.
    """
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="sl-suite2p",
                server_version=version,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
