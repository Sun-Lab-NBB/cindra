"""Provides the MCP server for agentic neural imaging data processing.

Exposes tools that enable AI agents to discover sessions, execute pipelines, and monitor processing status for both
single-day and multi-day suite2p processing workflows.
"""

from __future__ import annotations

from os import cpu_count
from typing import Any, Literal
from pathlib import Path
from threading import Lock, Thread
import traceback
from dataclasses import field, dataclass

from natsort import natsorted
from ataraxis_time import PrecisionTimer, TimerPrecisions
from mcp.server.fastmcp import FastMCP

from ..io import resolve_multiday_contexts
from ..pipelines import process_multi_day, process_single_day
from ..dataclasses import (
    RuntimeContext,
    MultiDayConfiguration,
    SingleDayConfiguration,
)

# Initializes the MCP server with JSON response mode for structured output.
mcp = FastMCP(name="ss2p-mcp", json_response=True)

# CPU cores reserved for system operations.
_RESERVED_CORES: int = 4

# Maximum CPU cores any single job can use.
_MAXIMUM_JOB_CORES: int = 30

# Minimum number of sessions required for multi-day processing.
_MINIMUM_SESSION_COUNT: int = 2


@dataclass
class _SingleDayBatchState:
    """Tracks state for single-day batch processing operations."""

    sessions: list[Path] = field(default_factory=list)
    """All sessions to process."""
    config_path: Path | None = None
    """Shared configuration file path."""
    current_phase: str = "binarize"
    """Current processing phase: 'binarize', 'process', or 'combine'."""

    # Binarize phase tracking.
    binarize_queue: list[Path] = field(default_factory=list)
    """Sessions waiting to binarize."""
    binarize_active: dict[str, Thread] = field(default_factory=dict)
    """Currently binarizing (session_key -> thread)."""
    binarize_completed: set[str] = field(default_factory=set)
    """Sessions that finished binarizing."""
    binarize_failed: set[str] = field(default_factory=set)
    """Sessions that failed binarization."""
    plane_counts: dict[str, int] = field(default_factory=dict)
    """Session -> plane count (discovered during binarize)."""

    # Process phase tracking.
    process_queue: list[tuple[str, int]] = field(default_factory=list)
    """(session_key, plane_index) pairs to process."""
    process_active: dict[str, Thread] = field(default_factory=dict)
    """Currently processing (session_plane_key -> thread)."""
    process_completed: set[str] = field(default_factory=set)
    """Completed session_plane keys."""
    process_failed: set[str] = field(default_factory=set)
    """Failed session_plane keys."""

    # Combine phase tracking.
    combine_queue: list[Path] = field(default_factory=list)
    """Sessions waiting to combine."""
    combine_active: dict[str, Thread] = field(default_factory=dict)
    """Currently combining (session_key -> thread)."""
    combine_completed: set[str] = field(default_factory=set)
    """Sessions that finished combining."""
    combine_failed: set[str] = field(default_factory=set)
    """Sessions that failed combination."""

    # Configuration.
    workers_per_plane: int = 30
    """CPU cores per plane job."""
    max_parallel_planes: int = 1
    """Max concurrent plane jobs."""

    # Error tracking.
    errors: dict[str, list[str]] = field(default_factory=dict)
    """Session/plane key -> error messages."""

    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock."""
    manager_thread: Thread | None = None
    """Background manager thread."""


@dataclass
class _MultiDayBatchState:
    """Tracks state for multi-day batch processing operations."""

    animals: list[tuple[Path, list[Path]]] = field(default_factory=list)
    """(config_path, session_paths) per animal."""
    current_phase: str = "discover"
    """Current processing phase: 'discover' or 'extract'."""

    # Discover phase (per animal).
    discover_queue: list[str] = field(default_factory=list)
    """Animal keys waiting to discover."""
    discover_active: dict[str, Thread] = field(default_factory=dict)
    """Currently discovering (animal_key -> thread)."""
    discover_completed: set[str] = field(default_factory=set)
    """Animals that finished discovery."""
    discover_failed: set[str] = field(default_factory=set)
    """Animals that failed discovery."""

    # Extract phase (per session across all animals).
    extract_queue: list[tuple[str, str]] = field(default_factory=list)
    """(animal_key, session_id) pairs."""
    extract_active: dict[str, Thread] = field(default_factory=dict)
    """Currently extracting (animal_session_key -> thread)."""
    extract_completed: set[str] = field(default_factory=set)
    """Completed extractions."""
    extract_failed: set[str] = field(default_factory=set)
    """Failed extractions."""

    # Session IDs per animal (populated during discover).
    session_ids: dict[str, list[str]] = field(default_factory=dict)
    """animal_key -> list of session_ids."""

    # Configuration.
    workers_per_discover: int = 20
    """Workers for discover phase."""
    max_parallel_discovers: int = 1
    """Max concurrent discovers."""
    workers_per_extract: int = 30
    """Workers for extract phase."""
    max_parallel_extracts: int = 1
    """Max concurrent extractions."""

    # Error tracking.
    errors: dict[str, list[str]] = field(default_factory=dict)
    """Key -> error messages."""

    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock."""
    manager_thread: Thread | None = None
    """Background manager thread."""


# Module-level batch processing state.
_single_day_batch_state: _SingleDayBatchState | None = None
_multi_day_batch_state: _MultiDayBatchState | None = None


def _calculate_workers(requested_workers: int, max_workers: int = _MAXIMUM_JOB_CORES) -> int:
    """Calculates the number of CPU cores to allocate for a processing job.

    Args:
        requested_workers: The user-requested worker count. Set to -1 or less for automatic allocation.
        max_workers: The maximum number of workers to allocate.

    Returns:
        The number of CPU cores to use for the job.
    """
    if requested_workers > 0:
        return min(requested_workers, max_workers)

    available_cores = cpu_count()
    if available_cores is None:
        return _RESERVED_CORES

    return min(max(1, available_cores - _RESERVED_CORES), max_workers)


def _calculate_max_parallel(workers_per_job: int) -> int:
    """Calculates the maximum number of jobs that can run in parallel.

    Args:
        workers_per_job: The number of CPU cores allocated per job.

    Returns:
        The maximum number of parallel jobs.
    """
    available_cores = cpu_count()
    if available_cores is None:
        return 1

    return max(1, available_cores // workers_per_job)


def _get_session_key(session_path: Path) -> str:
    """Generates a unique key for a session path.

    Args:
        session_path: The path to the session directory.

    Returns:
        A string key for the session.
    """
    return str(session_path)


def _get_plane_key(session_path: Path, plane_index: int) -> str:
    """Generates a unique key for a session-plane combination.

    Args:
        session_path: The path to the session directory.
        plane_index: The plane index.

    Returns:
        A string key for the session-plane combination.
    """
    return f"{session_path}|plane_{plane_index}"


def _run_binarize_job(session_path: Path, config_path: Path) -> tuple[bool, int, str | None]:
    """Runs the binarize phase for a single session.

    Args:
        session_path: The path to the session directory.
        config_path: The path to the configuration file.

    Returns:
        A tuple containing success status, plane count (0 if failed), and error message if failed.
    """
    try:
        process_single_day(
            configuration_path=config_path,
            session_path=session_path,
            binarize=True,
            process=False,
            combine=False,
            progress_bars=False,
        )

        # Loads the configuration to find the save path, then counts planes via RuntimeContext.
        config = SingleDayConfiguration.from_yaml(file_path=config_path)
        root_path = config.file_io.save_path / "suite2p"
        contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]
        plane_count = len(contexts)
        return True, plane_count, None

    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        return False, 0, f"{type(error).__name__}: {error} ({location})"


def _run_process_job(session_path: Path, config_path: Path, plane_index: int, workers: int) -> tuple[bool, str | None]:
    """Runs the process phase for a single plane.

    Args:
        session_path: The path to the session directory.
        config_path: The path to the configuration file.
        plane_index: The plane index to process.
        workers: The number of workers to use.

    Returns:
        A tuple containing success status and error message if failed.
    """
    try:
        process_single_day(
            configuration_path=config_path,
            session_path=session_path,
            binarize=False,
            process=True,
            combine=False,
            target_plane=plane_index,
            workers=workers,
            progress_bars=False,
        )
        return True, None

    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        return False, f"{type(error).__name__}: {error} ({location})"


def _run_combine_job(session_path: Path, config_path: Path) -> tuple[bool, str | None]:
    """Runs the combine phase for a single session.

    Args:
        session_path: The path to the session directory.
        config_path: The path to the configuration file.

    Returns:
        A tuple containing success status and error message if failed.
    """
    try:
        process_single_day(
            configuration_path=config_path,
            session_path=session_path,
            binarize=False,
            process=False,
            combine=True,
            progress_bars=False,
        )
        return True, None

    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        return False, f"{type(error).__name__}: {error} ({location})"


def _binarize_worker(session_path: Path, config_path: Path) -> None:
    """Worker function for binarize phase.

    Args:
        session_path: The path to the session directory.
        config_path: The path to the configuration file.
    """
    session_key = _get_session_key(session_path)
    success: bool = False
    plane_count: int = 0
    error: str | None = None

    try:
        success, plane_count, error = _run_binarize_job(session_path=session_path, config_path=config_path)
    except Exception as e:
        frames = traceback.extract_tb(e.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        success, plane_count, error = False, 0, f"Worker crash: {type(e).__name__}: {e} ({location})"
    finally:
        if _single_day_batch_state is not None:
            with _single_day_batch_state.lock:
                _single_day_batch_state.binarize_active.pop(session_key, None)
                if success:
                    _single_day_batch_state.binarize_completed.add(session_key)
                    _single_day_batch_state.plane_counts[session_key] = plane_count
                else:
                    _single_day_batch_state.binarize_failed.add(session_key)
                    if error:
                        _single_day_batch_state.errors.setdefault(session_key, []).append(f"binarize: {error}")


def _process_worker(session_path: Path, config_path: Path, plane_index: int, workers: int) -> None:
    """Worker function for process phase.

    Args:
        session_path: The path to the session directory.
        config_path: The path to the configuration file.
        plane_index: The plane index to process.
        workers: The number of workers to use.
    """
    session_key = _get_session_key(session_path)
    plane_key = _get_plane_key(session_path, plane_index)
    success: bool = False
    error: str | None = None

    try:
        success, error = _run_process_job(
            session_path=session_path, config_path=config_path, plane_index=plane_index, workers=workers
        )
    except Exception as e:
        frames = traceback.extract_tb(e.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        success, error = False, f"Worker crash: {type(e).__name__}: {e} ({location})"
    finally:
        if _single_day_batch_state is not None:
            with _single_day_batch_state.lock:
                _single_day_batch_state.process_active.pop(plane_key, None)
                if success:
                    _single_day_batch_state.process_completed.add(plane_key)
                else:
                    _single_day_batch_state.process_failed.add(plane_key)
                    if error:
                        _single_day_batch_state.errors.setdefault(session_key, []).append(
                            f"process_plane_{plane_index}: {error}"
                        )


def _combine_worker(session_path: Path, config_path: Path) -> None:
    """Worker function for combine phase.

    Args:
        session_path: The path to the session directory.
        config_path: The path to the configuration file.
    """
    session_key = _get_session_key(session_path)
    success: bool = False
    error: str | None = None

    try:
        success, error = _run_combine_job(session_path=session_path, config_path=config_path)
    except Exception as e:
        frames = traceback.extract_tb(e.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        success, error = False, f"Worker crash: {type(e).__name__}: {e} ({location})"
    finally:
        if _single_day_batch_state is not None:
            with _single_day_batch_state.lock:
                _single_day_batch_state.combine_active.pop(session_key, None)
                if success:
                    _single_day_batch_state.combine_completed.add(session_key)
                else:
                    _single_day_batch_state.combine_failed.add(session_key)
                    if error:
                        _single_day_batch_state.errors.setdefault(session_key, []).append(f"combine: {error}")


def _single_day_batch_manager() -> None:
    """Manager thread for single-day batch processing.

    Orchestrates the three-phase processing: binarize → process → combine.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    if _single_day_batch_state is None or _single_day_batch_state.config_path is None:
        return

    config_path = _single_day_batch_state.config_path

    while True:
        with _single_day_batch_state.lock:
            # Phase 1: BINARIZE.
            if _single_day_batch_state.current_phase == "binarize":
                # Starts new binarize jobs (sequential - I/O bound).
                if not _single_day_batch_state.binarize_active and _single_day_batch_state.binarize_queue:
                    next_session = _single_day_batch_state.binarize_queue.pop(0)
                    session_key = _get_session_key(next_session)

                    thread = Thread(
                        target=_binarize_worker,
                        kwargs={"session_path": next_session, "config_path": config_path},
                        daemon=True,
                    )
                    thread.start()
                    _single_day_batch_state.binarize_active[session_key] = thread

                # Checks if binarize phase is complete.
                if not _single_day_batch_state.binarize_active and not _single_day_batch_state.binarize_queue:
                    # Builds process queue from completed binarizations (naturally sorted for deterministic order).
                    for session_key in natsorted(_single_day_batch_state.binarize_completed):
                        plane_count = _single_day_batch_state.plane_counts.get(session_key, 0)
                        for plane in range(plane_count):
                            _single_day_batch_state.process_queue.append((session_key, plane))

                    _single_day_batch_state.current_phase = "process"

            # Phase 2: PROCESS.
            elif _single_day_batch_state.current_phase == "process":
                # Starts new process jobs (parallel - CPU bound).
                while (
                    len(_single_day_batch_state.process_active) < _single_day_batch_state.max_parallel_planes
                    and _single_day_batch_state.process_queue
                ):
                    session_key, plane_index = _single_day_batch_state.process_queue.pop(0)
                    session_path = Path(session_key)
                    plane_key = _get_plane_key(session_path, plane_index)

                    thread = Thread(
                        target=_process_worker,
                        kwargs={
                            "session_path": session_path,
                            "config_path": config_path,
                            "plane_index": plane_index,
                            "workers": _single_day_batch_state.workers_per_plane,
                        },
                        daemon=True,
                    )
                    thread.start()
                    _single_day_batch_state.process_active[plane_key] = thread

                # Checks if process phase is complete.
                if not _single_day_batch_state.process_active and not _single_day_batch_state.process_queue:
                    # Builds combine queue from sessions with all planes processed (naturally sorted).
                    for session_key in natsorted(_single_day_batch_state.binarize_completed):
                        plane_count = _single_day_batch_state.plane_counts.get(session_key, 0)
                        all_planes_done = all(
                            _get_plane_key(Path(session_key), p) in _single_day_batch_state.process_completed
                            for p in range(plane_count)
                        )
                        any_plane_failed = any(
                            _get_plane_key(Path(session_key), p) in _single_day_batch_state.process_failed
                            for p in range(plane_count)
                        )

                        if all_planes_done and not any_plane_failed:
                            _single_day_batch_state.combine_queue.append(Path(session_key))

                    _single_day_batch_state.current_phase = "combine"

            # Phase 3: COMBINE.
            elif _single_day_batch_state.current_phase == "combine":
                # Starts new combine jobs (sequential - I/O bound).
                if not _single_day_batch_state.combine_active and _single_day_batch_state.combine_queue:
                    next_session = _single_day_batch_state.combine_queue.pop(0)
                    session_key = _get_session_key(next_session)

                    thread = Thread(
                        target=_combine_worker,
                        kwargs={"session_path": next_session, "config_path": config_path},
                        daemon=True,
                    )
                    thread.start()
                    _single_day_batch_state.combine_active[session_key] = thread

                # Checks if all processing is complete.
                if not _single_day_batch_state.combine_active and not _single_day_batch_state.combine_queue:
                    break

        # Sleeps briefly before checking again.
        timer.delay(delay=1000, allow_sleep=True)


def _run_discover_job(config_path: Path, session_paths: list[Path], workers: int) -> tuple[bool, list[str], str | None]:
    """Runs the discover phase for a single animal.

    Args:
        config_path: The path to the configuration file.
        session_paths: The list of session paths for this animal.
        workers: The number of workers to use.

    Returns:
        A tuple containing success status, list of session IDs, and error message if failed.
    """
    try:
        # Writes session directories into the configuration file before running the pipeline.
        configuration = MultiDayConfiguration.from_yaml(file_path=config_path)
        configuration.session_io.session_directories = list(session_paths)
        configuration.save(file_path=config_path)

        process_multi_day(
            configuration_path=config_path,
            discover=True,
            extract=False,
            workers=workers,
            progress_bars=False,
        )

        # Reloads the configuration and resolves contexts to extract session IDs.
        config = MultiDayConfiguration.from_yaml(file_path=config_path)
        contexts = resolve_multiday_contexts(configuration=config)
        session_ids = [ctx.runtime.io.session_id for ctx in contexts]
        return True, session_ids, None

    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        return False, [], f"{type(error).__name__}: {error} ({location})"


def _run_extract_job(
    config_path: Path, session_paths: list[Path], session_id: str, workers: int
) -> tuple[bool, str | None]:
    """Runs the extract phase for a single session.

    Args:
        config_path: The path to the configuration file.
        session_paths: The list of session paths for this animal.
        session_id: The session ID to extract.
        workers: The number of workers to use.

    Returns:
        A tuple containing success status and error message if failed.
    """
    try:
        # Writes session directories into the configuration file before running the pipeline.
        configuration = MultiDayConfiguration.from_yaml(file_path=config_path)
        configuration.session_io.session_directories = list(session_paths)
        configuration.save(file_path=config_path)

        process_multi_day(
            configuration_path=config_path,
            discover=False,
            extract=True,
            target_session=session_id,
            workers=workers,
            progress_bars=False,
        )
        return True, None

    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        return False, f"{type(error).__name__}: {error} ({location})"


def _discover_worker(animal_key: str, config_path: Path, session_paths: list[Path], workers: int) -> None:
    """Worker function for discover phase.

    Args:
        animal_key: The unique key for this animal.
        config_path: The path to the configuration file.
        session_paths: The list of session paths for this animal.
        workers: The number of workers to use.
    """
    success: bool = False
    session_ids: list[str] = []
    error: str | None = None

    try:
        success, session_ids, error = _run_discover_job(
            config_path=config_path, session_paths=session_paths, workers=workers
        )
    except Exception as e:
        frames = traceback.extract_tb(e.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        success, session_ids, error = False, [], f"Worker crash: {type(e).__name__}: {e} ({location})"
    finally:
        if _multi_day_batch_state is not None:
            with _multi_day_batch_state.lock:
                _multi_day_batch_state.discover_active.pop(animal_key, None)
                if success:
                    _multi_day_batch_state.discover_completed.add(animal_key)
                    _multi_day_batch_state.session_ids[animal_key] = session_ids
                else:
                    _multi_day_batch_state.discover_failed.add(animal_key)
                    if error:
                        _multi_day_batch_state.errors.setdefault(animal_key, []).append(f"discover: {error}")


def _extract_worker(
    animal_key: str, config_path: Path, session_paths: list[Path], session_id: str, workers: int
) -> None:
    """Worker function for extract phase.

    Args:
        animal_key: The unique key for this animal.
        config_path: The path to the configuration file.
        session_paths: The list of session paths for this animal.
        session_id: The session ID to extract.
        workers: The number of workers to use.
    """
    extract_key = f"{animal_key}|{session_id}"
    success: bool = False
    error: str | None = None

    try:
        success, error = _run_extract_job(
            config_path=config_path, session_paths=session_paths, session_id=session_id, workers=workers
        )
    except Exception as e:
        frames = traceback.extract_tb(e.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        success, error = False, f"Worker crash: {type(e).__name__}: {e} ({location})"
    finally:
        if _multi_day_batch_state is not None:
            with _multi_day_batch_state.lock:
                _multi_day_batch_state.extract_active.pop(extract_key, None)
                if success:
                    _multi_day_batch_state.extract_completed.add(extract_key)
                else:
                    _multi_day_batch_state.extract_failed.add(extract_key)
                    if error:
                        _multi_day_batch_state.errors.setdefault(animal_key, []).append(
                            f"extract_{session_id}: {error}"
                        )


def _multi_day_batch_manager() -> None:
    """Manager thread for multi-day batch processing.

    Orchestrates the two-phase processing: discover → extract.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    if _multi_day_batch_state is None:
        return

    # Builds animal key to config/sessions mapping.
    animal_configs: dict[str, tuple[Path, list[Path]]] = {}
    for config_path, session_paths in _multi_day_batch_state.animals:
        config = MultiDayConfiguration.from_yaml(file_path=config_path)
        animal_key = config.session_io.dataset_name
        animal_configs[animal_key] = (config_path, session_paths)

    while True:
        with _multi_day_batch_state.lock:
            # Phase 1: DISCOVER.
            if _multi_day_batch_state.current_phase == "discover":
                # Starts new discover jobs.
                while (
                    len(_multi_day_batch_state.discover_active) < _multi_day_batch_state.max_parallel_discovers
                    and _multi_day_batch_state.discover_queue
                ):
                    animal_key = _multi_day_batch_state.discover_queue.pop(0)
                    config_path, session_paths = animal_configs[animal_key]

                    thread = Thread(
                        target=_discover_worker,
                        kwargs={
                            "animal_key": animal_key,
                            "config_path": config_path,
                            "session_paths": session_paths,
                            "workers": _multi_day_batch_state.workers_per_discover,
                        },
                        daemon=True,
                    )
                    thread.start()
                    _multi_day_batch_state.discover_active[animal_key] = thread

                # Checks if discover phase is complete.
                if not _multi_day_batch_state.discover_active and not _multi_day_batch_state.discover_queue:
                    # Builds extract queue from completed discoveries (naturally sorted for deterministic order).
                    for animal_key in natsorted(_multi_day_batch_state.discover_completed):
                        for session_id in natsorted(_multi_day_batch_state.session_ids.get(animal_key, [])):
                            _multi_day_batch_state.extract_queue.append((animal_key, session_id))

                    _multi_day_batch_state.current_phase = "extract"

            # Phase 2: EXTRACT.
            elif _multi_day_batch_state.current_phase == "extract":
                # Starts new extract jobs.
                while (
                    len(_multi_day_batch_state.extract_active) < _multi_day_batch_state.max_parallel_extracts
                    and _multi_day_batch_state.extract_queue
                ):
                    animal_key, session_id = _multi_day_batch_state.extract_queue.pop(0)
                    extract_key = f"{animal_key}|{session_id}"
                    config_path, session_paths = animal_configs[animal_key]

                    thread = Thread(
                        target=_extract_worker,
                        kwargs={
                            "animal_key": animal_key,
                            "config_path": config_path,
                            "session_paths": session_paths,
                            "session_id": session_id,
                            "workers": _multi_day_batch_state.workers_per_extract,
                        },
                        daemon=True,
                    )
                    thread.start()
                    _multi_day_batch_state.extract_active[extract_key] = thread

                # Checks if all processing is complete.
                if not _multi_day_batch_state.extract_active and not _multi_day_batch_state.extract_queue:
                    break

        # Sleeps briefly before checking again.
        timer.delay(delay=1000, allow_sleep=True)


@mcp.tool()
def generate_config_file(output_path: str, pipeline_type: Literal["single-day", "multi-day"]) -> dict[str, Any]:
    """Generates a default configuration YAML file for the specified pipeline type.

    Creates a configuration file with sensible defaults that can be used directly or modified before processing.

    Args:
        output_path: The absolute path where the configuration file should be saved.
        pipeline_type: The type of pipeline configuration to generate ('single-day' or 'multi-day').
    """
    output = Path(output_path)

    if not output.parent.exists():
        return {"success": False, "error": f"Parent directory does not exist: {output.parent}"}

    if output.suffix != ".yaml":
        output = output.with_suffix(".yaml")

    if pipeline_type == "single-day":
        config: SingleDayConfiguration | MultiDayConfiguration = SingleDayConfiguration()
    else:
        config = MultiDayConfiguration()

    config.save(file_path=output)

    return {"success": True, "file_path": str(output), "pipeline_type": pipeline_type}


@mcp.tool()
def get_single_day_status(session_path: str) -> dict[str, Any]:
    """Gets the processing status of a single-day session.

    Args:
        session_path: The absolute path to the session data directory.
    """
    session = Path(session_path)

    if not session.exists():
        return {"success": False, "error": f"Session directory not found: {session_path}"}

    suite2p_path = session / "suite2p"
    if not suite2p_path.exists():
        # Searches recursively for the RuntimeContext configuration marker.
        matches = list(session.rglob("configuration.yaml"))
        if matches:
            suite2p_path = matches[0].parent

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
            "combined_metadata": (combined_path / "combined_metadata.npz").exists(),
            "stat": (combined_path / "stat.npy").exists(),
            "F": (combined_path / "F.npy").exists(),
            "Fneu": (combined_path / "Fneu.npy").exists(),
            "spks": (combined_path / "spks.npy").exists(),
            "iscell": (combined_path / "iscell.npy").exists(),
        }

    return status


@mcp.tool()
def get_multi_day_status(session_path: str) -> dict[str, Any]:
    """Gets the multi-day processing status for a session.

    Args:
        session_path: The absolute path to a session directory.
    """
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
            "runtime_exists": (dataset / "multiday_runtime_data.yaml").exists(),
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
        elif dataset_status["runtime_exists"]:
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


# Single-day batch processing tools.


@mcp.tool()
def discover_single_day_sessions_tool(root_directory: str) -> dict[str, Any]:
    """Discovers sessions containing raw neural imaging data that can be processed by the single-day pipeline.

    Searches recursively for suite2p_parameters.json files, which mark directories containing raw session data suitable
    for single-day processing. Returns the parent directory of each match as a session candidate path.

    Args:
        root_directory: The absolute path to the root directory to search.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    session_paths: list[str] = []
    errors: list[str] = []

    try:
        for marker_file in root_path.rglob("suite2p_parameters.json"):
            try:
                session_paths.append(str(marker_file.parent))
            except Exception as error:
                errors.append(f"{marker_file.parent}: {error}")
    except PermissionError as error:
        errors.append(f"Access denied during search: {error}")

    # Sorts paths for consistent output.
    session_paths.sort()

    result: dict[str, Any] = {"sessions": session_paths, "count": len(session_paths)}

    if errors:
        result["errors"] = errors

    return result


@mcp.tool()
def discover_multi_day_candidates_tool(root_directory: str) -> dict[str, Any]:
    """Discovers sessions with completed single-day processing that are candidates for multi-day cell tracking.

    Searches recursively for combined_metadata.npz files, which mark completed single-day suite2p outputs. Returns the
    grandparent directory paths (session root directories containing suite2p output).

    Args:
        root_directory: The absolute path to the root directory to search.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    session_paths: list[str] = []
    errors: list[str] = []

    try:
        for marker_file in root_path.rglob("combined_metadata.npz"):
            try:
                # The combined_metadata.npz lives in suite2p/combined/; grandparent is the suite2p output root,
                # and its parent is the session directory.
                session_root = str(marker_file.parent.parent.parent)
                if session_root not in session_paths:
                    session_paths.append(session_root)
            except Exception as error:
                errors.append(f"{marker_file}: {error}")
    except PermissionError as error:
        errors.append(f"Access denied during search: {error}")

    # Sorts paths for consistent output.
    session_paths.sort()

    result: dict[str, Any] = {"sessions": session_paths, "count": len(session_paths)}

    if errors:
        result["errors"] = errors

    return result


@mcp.tool()
def start_batch_processing_tool(
    session_paths: list[str],
    config_path: str,
    *,
    workers_per_plane: int = -1,
    max_parallel_planes: int = -1,
) -> dict[str, Any]:
    """Starts batch single-day processing for multiple sessions.

    Manages a three-phase batch: binarize (sequential), process (parallel), combine (sequential). Use
    get_batch_processing_status_tool to monitor progress.

    Args:
        session_paths: List of absolute paths to session root data directories.
        config_path: The absolute path to the configuration YAML file.
        workers_per_plane: CPU cores per plane job (-1 for automatic, max 30).
        max_parallel_planes: Max concurrent plane jobs (-1 for automatic).
    """
    global _single_day_batch_state

    if not session_paths:
        return {"error": "At least one session path is required"}

    config = Path(config_path)
    if not config.exists():
        return {"error": f"Configuration file not found: {config_path}"}

    if config.suffix != ".yaml":
        return {"error": f"Configuration file must be a .yaml file: {config_path}"}

    # Validates session paths.
    valid_paths: list[Path] = []
    invalid_paths: list[str] = []

    for session_path in session_paths:
        path = Path(session_path)
        if path.exists() and path.is_dir():
            valid_paths.append(path)
        else:
            invalid_paths.append(session_path)

    if not valid_paths:
        return {"error": "No valid session paths provided", "invalid_paths": invalid_paths}

    # Checks if batch processing is already active.
    if _single_day_batch_state is not None:
        with _single_day_batch_state.lock:
            active_count = (
                len(_single_day_batch_state.binarize_active)
                + len(_single_day_batch_state.process_active)
                + len(_single_day_batch_state.combine_active)
            )
            queue_count = (
                len(_single_day_batch_state.binarize_queue)
                + len(_single_day_batch_state.process_queue)
                + len(_single_day_batch_state.combine_queue)
            )
            if active_count > 0 or queue_count > 0:
                return {
                    "error": "Batch processing already in progress. Wait for current batch to complete.",
                    "active_count": active_count,
                    "queued_count": queue_count,
                }

    # Calculates resource allocation.
    actual_workers = _calculate_workers(requested_workers=workers_per_plane)
    actual_max_parallel = max_parallel_planes if max_parallel_planes > 0 else _calculate_max_parallel(actual_workers)

    # Initializes batch state.
    _single_day_batch_state = _SingleDayBatchState(
        sessions=list(valid_paths),
        config_path=config,
        current_phase="binarize",
        binarize_queue=list(valid_paths),
        workers_per_plane=actual_workers,
        max_parallel_planes=actual_max_parallel,
        lock=Lock(),
    )

    # Starts the batch manager thread.
    manager = Thread(target=_single_day_batch_manager, daemon=True)
    manager.start()
    _single_day_batch_state.manager_thread = manager

    result: dict[str, Any] = {
        "started": True,
        "total_sessions": len(valid_paths),
        "workers_per_plane": actual_workers,
        "max_parallel_planes": actual_max_parallel,
        "message": "Batch processing started. Use get_batch_processing_status_tool to monitor progress.",
    }

    if invalid_paths:
        result["invalid_paths"] = invalid_paths

    return result


@mcp.tool()
def get_batch_processing_status_tool() -> dict[str, Any]:
    """Returns the current status of single-day batch processing.

    Returns status for all sessions including phase progress (binarize, process, combine).
    """
    if _single_day_batch_state is None:
        return {
            "current_phase": "none",
            "sessions": [],
            "summary": {
                "total": 0,
                "binarize_completed": 0,
                "process_completed": 0,
                "combine_completed": 0,
                "failed": 0,
            },
        }

    with _single_day_batch_state.lock:
        sessions_status: list[dict[str, Any]] = []

        for session_path in _single_day_batch_state.sessions:
            session_key = _get_session_key(session_path)
            session_name = session_path.name

            # Determines binarize status.
            if session_key in _single_day_batch_state.binarize_completed:
                binarize_status = "done"
            elif session_key in _single_day_batch_state.binarize_failed:
                binarize_status = "failed"
            elif session_key in _single_day_batch_state.binarize_active:
                binarize_status = "running"
            elif session_path in _single_day_batch_state.binarize_queue:
                binarize_status = "pending"
            else:
                binarize_status = "pending"

            # Determines process status.
            plane_count = _single_day_batch_state.plane_counts.get(session_key, 0)
            if plane_count > 0:
                completed_planes = sum(
                    1
                    for p in range(plane_count)
                    if _get_plane_key(session_path, p) in _single_day_batch_state.process_completed
                )
                failed_planes = sum(
                    1
                    for p in range(plane_count)
                    if _get_plane_key(session_path, p) in _single_day_batch_state.process_failed
                )
                running_planes = sum(
                    1
                    for p in range(plane_count)
                    if _get_plane_key(session_path, p) in _single_day_batch_state.process_active
                )

                if failed_planes > 0:
                    process_status = f"{completed_planes}/{plane_count} (failed: {failed_planes})"
                elif running_planes > 0:
                    process_status = f"{completed_planes}/{plane_count} (running: {running_planes})"
                else:
                    process_status = f"{completed_planes}/{plane_count}"
            else:
                process_status = "0/0"

            # Determines combine status.
            if session_key in _single_day_batch_state.combine_completed:
                combine_status = "done"
            elif session_key in _single_day_batch_state.combine_failed:
                combine_status = "failed"
            elif session_key in _single_day_batch_state.combine_active:
                combine_status = "running"
            elif session_path in _single_day_batch_state.combine_queue:
                combine_status = "pending"
            else:
                combine_status = "pending"

            # Determines overall status.
            if session_key in _single_day_batch_state.combine_completed:
                overall_status = "SUCCEEDED"
            elif (
                session_key in _single_day_batch_state.binarize_failed
                or session_key in _single_day_batch_state.combine_failed
                or any(
                    _get_plane_key(session_path, p) in _single_day_batch_state.process_failed
                    for p in range(plane_count)
                )
            ):
                overall_status = "FAILED"
            elif (
                session_key in _single_day_batch_state.binarize_active
                or session_key in _single_day_batch_state.combine_active
                or any(
                    _get_plane_key(session_path, p) in _single_day_batch_state.process_active
                    for p in range(plane_count)
                )
            ):
                overall_status = "PROCESSING"
            else:
                overall_status = "QUEUED"

            session_status: dict[str, Any] = {
                "session_name": session_name,
                "status": overall_status,
                "binarize": binarize_status,
                "process": process_status,
                "combine": combine_status,
            }

            if session_key in _single_day_batch_state.errors:
                session_status["errors"] = _single_day_batch_state.errors[session_key]

            sessions_status.append(session_status)

        # Computes summary.
        total_failed = len(_single_day_batch_state.binarize_failed) + len(_single_day_batch_state.combine_failed)
        for session_key in _single_day_batch_state.binarize_completed:
            session_path = Path(session_key)
            plane_count = _single_day_batch_state.plane_counts.get(session_key, 0)
            if any(
                _get_plane_key(session_path, p) in _single_day_batch_state.process_failed for p in range(plane_count)
            ):
                total_failed += 1

        summary = {
            "total": len(_single_day_batch_state.sessions),
            "binarize_completed": len(_single_day_batch_state.binarize_completed),
            "process_completed": len(
                {_get_session_key(Path(key.split("|")[0])) for key in _single_day_batch_state.process_completed}
            ),
            "combine_completed": len(_single_day_batch_state.combine_completed),
            "failed": total_failed,
        }

        return {
            "current_phase": _single_day_batch_state.current_phase,
            "sessions": sessions_status,
            "summary": summary,
        }


@mcp.tool()
def cancel_batch_processing_tool() -> dict[str, Any]:
    """Cancels any running single-day batch processing.

    Clears all queues and resets the batch state. Active jobs will complete but no new jobs will start.

    Returns:
        A status message indicating whether cancellation was successful.
    """
    global _single_day_batch_state

    if _single_day_batch_state is None:
        return {"cancelled": False, "message": "No single-day batch processing is active."}

    with _single_day_batch_state.lock:
        active_count = (
            len(_single_day_batch_state.binarize_active)
            + len(_single_day_batch_state.process_active)
            + len(_single_day_batch_state.combine_active)
        )

        # Clears all queues to prevent new jobs from starting.
        _single_day_batch_state.binarize_queue.clear()
        _single_day_batch_state.process_queue.clear()
        _single_day_batch_state.combine_queue.clear()

        # Records final state before reset.
        final_state = {
            "binarize_completed": len(_single_day_batch_state.binarize_completed),
            "process_completed": len(_single_day_batch_state.process_completed),
            "combine_completed": len(_single_day_batch_state.combine_completed),
            "active_jobs_at_cancel": active_count,
        }

    # Resets batch state after releasing lock.
    _single_day_batch_state = None

    return {
        "cancelled": True,
        "message": "Single-day batch processing cancelled. Active jobs will complete but no new jobs will start.",
        "final_state": final_state,
    }


# Multi-day batch processing tools.


@mcp.tool()
def start_multiday_batch_processing_tool(
    animal_configs: list[dict[str, Any]],
    *,
    workers_per_discover: int = 20,
    workers_per_extract: int = -1,
) -> dict[str, Any]:
    """Starts batch multi-day processing for multiple animals.

    Manages a two-phase batch: discover (parallel by animal), extract (parallel by session). Use
    get_multiday_batch_processing_status_tool to monitor progress.

    Args:
        animal_configs: List of animal configurations, each with 'config_path' and 'session_paths'.
        workers_per_discover: Workers for discover phase (default 20).
        workers_per_extract: Workers for extract phase (-1 for automatic, max 30).
    """
    global _multi_day_batch_state

    if not animal_configs:
        return {"error": "At least one animal configuration is required"}

    # Validates animal configurations.
    valid_animals: list[tuple[Path, list[Path]]] = []
    invalid_configs: list[str] = []
    animal_keys: list[str] = []

    for animal_config in animal_configs:
        if "config_path" not in animal_config or "session_paths" not in animal_config:
            invalid_configs.append(f"Missing required keys: {animal_config}")
            continue

        config_path = Path(animal_config["config_path"])
        if not config_path.exists():
            invalid_configs.append(f"Config not found: {config_path}")
            continue

        session_paths = [Path(p) for p in animal_config["session_paths"]]
        if len(session_paths) < _MINIMUM_SESSION_COUNT:
            invalid_configs.append(f"Need at least 2 sessions: {config_path}")
            continue

        invalid_sessions = [str(p) for p in session_paths if not p.exists() or not p.is_dir()]
        if invalid_sessions:
            invalid_configs.append(f"Invalid sessions for {config_path}: {invalid_sessions}")
            continue

        # Extracts animal key from config.
        try:
            config = MultiDayConfiguration.from_yaml(file_path=config_path)
            animal_keys.append(config.session_io.dataset_name)
        except Exception as error:
            invalid_configs.append(f"Failed to load config {config_path}: {error}")
            continue

        valid_animals.append((config_path, session_paths))

    if not valid_animals:
        return {"error": "No valid animal configurations provided", "invalid_configs": invalid_configs}

    # Checks if batch processing is already active.
    if _multi_day_batch_state is not None:
        with _multi_day_batch_state.lock:
            active_count = len(_multi_day_batch_state.discover_active) + len(_multi_day_batch_state.extract_active)
            queue_count = len(_multi_day_batch_state.discover_queue) + len(_multi_day_batch_state.extract_queue)
            if active_count > 0 or queue_count > 0:
                return {
                    "error": "Multi-day batch processing already in progress.",
                    "active_count": active_count,
                    "queued_count": queue_count,
                }

    # Calculates resource allocation.
    actual_workers_discover = min(workers_per_discover, _MAXIMUM_JOB_CORES)
    actual_workers_extract = _calculate_workers(requested_workers=workers_per_extract)
    max_parallel_discovers = _calculate_max_parallel(actual_workers_discover)
    max_parallel_extracts = _calculate_max_parallel(actual_workers_extract)

    # Initializes batch state.
    _multi_day_batch_state = _MultiDayBatchState(
        animals=valid_animals,
        current_phase="discover",
        discover_queue=list(animal_keys),
        workers_per_discover=actual_workers_discover,
        max_parallel_discovers=max_parallel_discovers,
        workers_per_extract=actual_workers_extract,
        max_parallel_extracts=max_parallel_extracts,
        lock=Lock(),
    )

    # Starts the batch manager thread.
    manager = Thread(target=_multi_day_batch_manager, daemon=True)
    manager.start()
    _multi_day_batch_state.manager_thread = manager

    total_sessions = sum(len(sessions) for _, sessions in valid_animals)

    result: dict[str, Any] = {
        "started": True,
        "total_animals": len(valid_animals),
        "total_sessions": total_sessions,
        "workers_per_discover": actual_workers_discover,
        "workers_per_extract": actual_workers_extract,
        "message": "Multi-day batch processing started. Use get_multiday_batch_processing_status_tool to monitor.",
    }

    if invalid_configs:
        result["invalid_configs"] = invalid_configs

    return result


@mcp.tool()
def get_multiday_batch_processing_status_tool() -> dict[str, Any]:
    """Returns the current status of multi-day batch processing.

    Returns status for all animals including phase progress (discover, extract).
    """
    if _multi_day_batch_state is None:
        return {
            "current_phase": "none",
            "animals": [],
            "summary": {
                "total_animals": 0,
                "discover_completed": 0,
                "extract_completed": 0,
                "extract_total": 0,
                "failed": 0,
            },
        }

    with _multi_day_batch_state.lock:
        animals_status: list[dict[str, Any]] = []

        # Builds animal key to config mapping.
        animal_keys_ordered: list[str] = []
        for config_path, _ in _multi_day_batch_state.animals:
            try:
                config = MultiDayConfiguration.from_yaml(file_path=config_path)
                animal_keys_ordered.append(config.session_io.dataset_name)
            except Exception:
                continue

        for animal_key in animal_keys_ordered:
            # Determines discover status.
            if animal_key in _multi_day_batch_state.discover_completed:
                discover_status = "done"
            elif animal_key in _multi_day_batch_state.discover_failed:
                discover_status = "failed"
            elif animal_key in _multi_day_batch_state.discover_active:
                discover_status = "running"
            elif animal_key in _multi_day_batch_state.discover_queue:
                discover_status = "pending"
            else:
                discover_status = "pending"

            # Determines extract progress.
            session_ids = _multi_day_batch_state.session_ids.get(animal_key, [])
            extract_total = len(session_ids)
            extract_completed = sum(
                1 for sid in session_ids if f"{animal_key}|{sid}" in _multi_day_batch_state.extract_completed
            )
            extract_failed = sum(
                1 for sid in session_ids if f"{animal_key}|{sid}" in _multi_day_batch_state.extract_failed
            )

            # Determines overall status.
            if animal_key in _multi_day_batch_state.discover_failed:
                overall_status = "FAILED"
            elif extract_failed > 0:
                overall_status = "PARTIAL"
            elif extract_completed == extract_total and extract_total > 0:
                overall_status = "SUCCEEDED"
            elif animal_key in _multi_day_batch_state.discover_active or any(
                f"{animal_key}|{sid}" in _multi_day_batch_state.extract_active for sid in session_ids
            ):
                overall_status = "PROCESSING"
            else:
                overall_status = "QUEUED"

            animal_status: dict[str, Any] = {
                "animal_key": animal_key,
                "status": overall_status,
                "discover": discover_status,
                "extract_completed": extract_completed,
                "extract_total": extract_total,
            }

            if animal_key in _multi_day_batch_state.errors:
                animal_status["errors"] = _multi_day_batch_state.errors[animal_key]

            animals_status.append(animal_status)

        # Computes summary.
        total_extract_completed = len(_multi_day_batch_state.extract_completed)
        total_extract_total = sum(len(ids) for ids in _multi_day_batch_state.session_ids.values())
        total_failed = len(_multi_day_batch_state.discover_failed) + len(_multi_day_batch_state.extract_failed)

        summary = {
            "total_animals": len(_multi_day_batch_state.animals),
            "discover_completed": len(_multi_day_batch_state.discover_completed),
            "extract_completed": total_extract_completed,
            "extract_total": total_extract_total,
            "failed": total_failed,
        }

        return {
            "current_phase": _multi_day_batch_state.current_phase,
            "animals": animals_status,
            "summary": summary,
        }


@mcp.tool()
def cancel_multiday_batch_processing_tool() -> dict[str, Any]:
    """Cancels any running multi-day batch processing.

    Clears all queues and resets the batch state. Active jobs will complete but no new jobs will start.

    Returns:
        A status message indicating whether cancellation was successful.
    """
    global _multi_day_batch_state

    if _multi_day_batch_state is None:
        return {"cancelled": False, "message": "No multi-day batch processing is active."}

    with _multi_day_batch_state.lock:
        active_count = len(_multi_day_batch_state.discover_active) + len(_multi_day_batch_state.extract_active)

        # Clears all queues to prevent new jobs from starting.
        _multi_day_batch_state.discover_queue.clear()
        _multi_day_batch_state.extract_queue.clear()

        # Records final state before reset.
        final_state = {
            "discover_completed": len(_multi_day_batch_state.discover_completed),
            "extract_completed": len(_multi_day_batch_state.extract_completed),
            "active_jobs_at_cancel": active_count,
        }

    # Resets batch state after releasing lock.
    _multi_day_batch_state = None

    return {
        "cancelled": True,
        "message": "Multi-day batch processing cancelled. Active jobs will complete but no new jobs will start.",
        "final_state": final_state,
    }


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport type to use ('stdio', 'sse', or 'streamable-http').
    """
    mcp.run(transport=transport)
