"""Contains tests for the GUI viewer lifecycle MCP server."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from cindra.interface.gui_mcp_server import (
    _ViewerProcess,
    _viewer_registry,
    list_viewers_tool,
    close_viewer_tool,
    launch_viewer_tool,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Iterator

_VIEWER_COUNT: int = 30
"""The number of viewers the concurrency test registers before the tools contend over them."""

_LIST_SWEEPS: int = 50
"""The number of listing sweeps each reader thread performs while the closer threads mutate the registry."""

_READER_THREADS: int = 4
"""The number of threads listing viewers concurrently with the same number of closing threads."""


class _StubProcess:
    """Stands in for a viewer subprocess, reporting a liveness state the test controls."""

    def __init__(self, *, alive: bool) -> None:
        self._alive = alive

    def poll(self) -> int | None:
        """Returns None while the stub is alive and a zero exit code once it is not."""
        return None if self._alive else 0

    def terminate(self) -> None:
        """Marks the stub as exited."""
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002 - Mirrors the signature the tool calls.
        """Returns the exit code without blocking."""
        return 0

    def kill(self) -> None:
        """Marks the stub as exited."""
        self._alive = False


@pytest.fixture(autouse=True)
def _empty_registry() -> Iterator[None]:
    """Clears the module-global viewer registry around every test, so no viewer leaks between them."""
    _viewer_registry.clear()
    yield
    _viewer_registry.clear()


class TestLaunchViewer:
    """Tests the precondition the launch tool enforces before it spawns a viewer subprocess."""

    @pytest.mark.xdist_group(name="viewer_registry")
    def test_file_path_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a regular file reports a failure rather than a launch the viewer cannot complete."""
        recording_file = tmp_path / "combined_metadata.npz"
        recording_file.write_bytes(b"")

        result = launch_viewer_tool(viewer_type="roi", recording_path=str(recording_file))

        assert result["success"] is False
        assert "not a cindra output directory" in result["error"]
        assert _viewer_registry == {}

    @pytest.mark.xdist_group(name="viewer_registry")
    def test_missing_path_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that an absent path reports its own failure message."""
        result = launch_viewer_tool(viewer_type="roi", recording_path=str(tmp_path / "absent"))

        assert result["success"] is False
        assert "Path does not exist" in result["error"]
        assert _viewer_registry == {}


class TestViewerRegistryConcurrency:
    """Tests the registry under the overlapping tool calls the MCP server dispatches into its worker threads."""

    @pytest.mark.xdist_group(name="viewer_registry")
    def test_overlapping_tool_calls_raise_nothing(self, tmp_path: Path) -> None:
        """Verifies that listing and closing viewers from several threads leaves no exception and no stale entry."""
        for index in range(_VIEWER_COUNT):
            viewer_id = f"viewer_{index}"
            _viewer_registry[viewer_id] = _ViewerProcess(
                viewer_id=viewer_id,
                viewer_type="roi",
                recording_path=str(tmp_path),
                dataset=None,
                state_path=str(tmp_path / f"{viewer_id}.json"),
                process=_StubProcess(alive=index % 2 == 0),
            )

        failures: list[Exception] = []

        def _list_viewers() -> None:
            try:
                for _ in range(_LIST_SWEEPS):
                    list_viewers_tool()
            except Exception as error:
                failures.append(error)

        def _close_viewers() -> None:
            try:
                for index in range(_VIEWER_COUNT):
                    close_viewer_tool(viewer_id=f"viewer_{index}")
            except Exception as error:
                failures.append(error)

        threads = [threading.Thread(target=_list_viewers) for _ in range(_READER_THREADS)]
        threads.extend(threading.Thread(target=_close_viewers) for _ in range(_READER_THREADS))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert failures == []
        assert _viewer_registry == {}
