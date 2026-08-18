"""Contains tests for the OpenMP runtime discovery, linking, and reporting assets."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
from ataraxis_base_utilities import error_format

from cindra.orchestration import (
    OpenMpStatus,
    openmp as openmp_module,
    resolve_openmp_runtime,
)
from cindra.orchestration.openmp import (
    OpenMpSummary,
    _link_openmp_runtime,
    verify_openmp_runtime,
    _discover_openmp_runtime,
    _openmp_runtime_loadable,
    _resolve_candidate_paths,
    _verify_runtime_loadable,
)


def _make_runtime(directory: Path) -> Path:
    """Creates a stand-in OpenMP runtime file inside the given directory and returns its path."""
    runtime_path = directory / "libomp.dylib"
    runtime_path.write_bytes(b"")
    return runtime_path


def _summary(status: OpenMpStatus, **overrides: object) -> OpenMpSummary:
    """Builds a summary carrying the given status and field overrides."""
    fields: dict[str, object] = {
        "status": status,
        "unresolved_reason": "",
        "runtime_path": None,
        "link_path": None,
        "searched_paths": (),
        "loadable": False,
    }
    fields.update(overrides)
    return OpenMpSummary(**fields)  # type: ignore[arg-type]


class TestRuntimeProbe:
    """Tests the interpreter-level and subprocess-level OpenMP runtime probes."""

    def test_loadable_reports_true_when_the_loader_resolves_the_runtime(self, monkeypatch):
        """Verifies that the probe reports success when the dynamic loader resolves the runtime."""
        monkeypatch.setattr("cindra.orchestration.openmp.ctypes.CDLL", lambda name: object())
        assert _openmp_runtime_loadable()

    def test_loadable_reports_false_when_the_loader_fails(self, monkeypatch):
        """Verifies that the probe reports failure when the dynamic loader cannot find the runtime."""

        def _raise(name):
            raise OSError(name)

        monkeypatch.setattr("cindra.orchestration.openmp.ctypes.CDLL", _raise)
        assert not _openmp_runtime_loadable()

    @pytest.mark.parametrize(("return_code", "expected"), [(0, True), (1, False)])
    def test_verification_follows_the_fresh_interpreter_exit_code(self, monkeypatch, return_code, expected):
        """Verifies that the post-link verification reports the exit code of the fresh interpreter."""
        monkeypatch.setattr(
            "cindra.orchestration.openmp.subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=return_code),
        )
        assert _verify_runtime_loadable() is expected


class TestRuntimeDiscovery:
    """Tests the candidate path search that locates an installed OpenMP runtime."""

    def test_discovery_returns_the_first_existing_candidate(self, tmp_path):
        """Verifies that the search returns the earliest candidate that resolves to a file."""
        second = tmp_path / "second"
        second.mkdir()
        runtime_path = _make_runtime(second)
        candidates = (tmp_path / "missing" / "libomp.dylib", runtime_path)
        assert _discover_openmp_runtime(candidates=candidates) == runtime_path

    def test_discovery_returns_none_when_no_candidate_resolves(self, tmp_path):
        """Verifies that the search reports no result when no candidate resolves to a file."""
        assert _discover_openmp_runtime(candidates=(tmp_path / "libomp.dylib",)) is None


class TestCandidateResolution:
    """Tests the assembly of the runtime paths examined when no runtime is named explicitly."""

    def test_candidates_cover_every_package_manager_directory(self, monkeypatch):
        """Verifies that every package manager directory contributes a candidate ahead of the other sources."""
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr("cindra.orchestration.openmp.sysconfig.get_path", lambda name: "")
        candidates = _resolve_candidate_paths()
        assert candidates == tuple(
            directory / "libomp.dylib" for directory in openmp_module._PACKAGE_MANAGER_DIRECTORIES
        )

    def test_candidates_include_the_active_conda_environment(self, monkeypatch, tmp_path):
        """Verifies that an active conda environment contributes its lib directory as a candidate."""
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
        monkeypatch.setattr("cindra.orchestration.openmp.sysconfig.get_path", lambda name: "")
        assert tmp_path / "lib" / "libomp.dylib" in _resolve_candidate_paths()

    def test_candidates_include_runtimes_vendored_in_installed_distributions(self, monkeypatch, tmp_path):
        """Verifies that a runtime vendored inside an installed distribution contributes a trailing candidate."""
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        vendored_directory = tmp_path / "scikit_learn" / ".dylibs"
        vendored_directory.mkdir(parents=True)
        vendored_runtime = _make_runtime(vendored_directory)
        monkeypatch.setattr("cindra.orchestration.openmp.sysconfig.get_path", lambda name: str(tmp_path))
        assert _resolve_candidate_paths()[-1] == vendored_runtime


class TestRuntimeLinking:
    """Tests the symbolic link creation that places the runtime on the loader search path."""

    def test_linking_creates_the_link_and_its_parent_directory(self, tmp_path):
        """Verifies that linking creates the missing parent directory and points the link at the runtime."""
        runtime_path = _make_runtime(tmp_path)
        link_path = tmp_path / "lib" / "libomp.dylib"
        _link_openmp_runtime(runtime_path=runtime_path, link_path=link_path)
        assert link_path.is_symlink()
        assert link_path.resolve() == runtime_path

    def test_linking_replaces_an_existing_link(self, tmp_path):
        """Verifies that linking overwrites a link left behind by an earlier run."""
        runtime_path = _make_runtime(tmp_path)
        link_path = tmp_path / "lib" / "libomp.dylib"
        link_path.parent.mkdir()
        link_path.symlink_to(target=tmp_path / "stale.dylib")
        _link_openmp_runtime(runtime_path=runtime_path, link_path=link_path)
        assert link_path.resolve() == runtime_path

    def test_linking_errors_when_the_link_cannot_be_written(self, tmp_path, monkeypatch):
        """Verifies that a filesystem refusal is reported as an actionable runtime error."""
        runtime_path = _make_runtime(tmp_path)

        def _raise(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(Path, "mkdir", _raise)
        expected_message = (
            f"Unable to link the OpenMP runtime into {tmp_path / 'lib'}. Writing the link requires permission to "
            f"modify that directory, which usually means running the command through sudo. The loader reported: "
            f"read-only file system."
        )
        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            _link_openmp_runtime(runtime_path=runtime_path, link_path=tmp_path / "lib" / "libomp.dylib")


class TestSummaryDescription:
    """Tests the human-readable description produced for every request outcome."""

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (OpenMpStatus.AVAILABLE, "already loads"),
            (OpenMpStatus.UNRESOLVED, "no OpenMP runtime to link"),
            (OpenMpStatus.PREVIEWED, "dry run"),
        ],
    )
    def test_description_reports_each_unchanged_outcome(self, status, expected):
        """Verifies that every outcome leaving the host unchanged describes itself distinctly."""
        assert expected in _summary(status).describe()

    def test_description_reports_a_link_that_resolved_the_runtime(self):
        """Verifies that a successful link reports the runtime as loadable."""
        summary = _summary(OpenMpStatus.LINKED, loadable=True)
        assert "the runtime now loads" in summary.describe()

    def test_description_reports_a_link_that_left_the_runtime_unloadable(self):
        """Verifies that a link leaving the runtime unloadable names the fallback remedy."""
        summary = _summary(OpenMpStatus.LINKED, loadable=False)
        assert "DYLD_LIBRARY_PATH" in summary.describe()


class TestRuntimeResolution:
    """Tests the request that discovers a runtime and links it into the loader search path."""

    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_resolution_refuses_every_platform_that_runs_tbb(self, monkeypatch, platform):
        """Verifies that the platforms running the TBB threading layer refuse to resolve an OpenMP runtime."""
        monkeypatch.setattr("cindra.orchestration.openmp.sys.platform", platform)
        expected_message = (
            f"Unable to resolve an OpenMP runtime on the '{platform}' platform. Only macOS runs the OpenMP "
            f"threading layer. Every other platform runs TBB, which carries lower overhead on the flat prange loops "
            f"this library compiles, so an OpenMP runtime linked here would go unused."
        )
        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            resolve_openmp_runtime()

    def test_resolution_leaves_a_host_whose_runtime_loads_alone(self, monkeypatch):
        """Verifies that a host whose runtime already loads is reported as available and loadable."""
        monkeypatch.setattr("cindra.orchestration.openmp.sys.platform", "darwin")
        monkeypatch.setattr("cindra.orchestration.openmp._openmp_runtime_loadable", lambda: True)
        summary = resolve_openmp_runtime()
        assert summary.status == OpenMpStatus.AVAILABLE
        assert summary.loadable

    def test_resolution_reports_a_host_carrying_no_runtime(self, monkeypatch, tmp_path):
        """Verifies that a host with no discoverable runtime reports the paths it examined."""
        monkeypatch.setattr("cindra.orchestration.openmp.sys.platform", "darwin")
        monkeypatch.setattr("cindra.orchestration.openmp._openmp_runtime_loadable", lambda: False)
        monkeypatch.setattr("cindra.orchestration.openmp._PACKAGE_MANAGER_DIRECTORIES", (tmp_path,))
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setattr("cindra.orchestration.openmp.sysconfig.get_path", lambda name: "")
        summary = resolve_openmp_runtime()
        assert summary.status == OpenMpStatus.UNRESOLVED
        assert summary.searched_paths == (tmp_path / "libomp.dylib",)
        assert "brew install libomp" in summary.unresolved_reason

    def test_resolution_previews_the_link_without_creating_it(self, monkeypatch, tmp_path):
        """Verifies that a dry run resolves the link and leaves the filesystem untouched."""
        monkeypatch.setattr("cindra.orchestration.openmp.sys.platform", "darwin")
        monkeypatch.setattr("cindra.orchestration.openmp._openmp_runtime_loadable", lambda: False)
        runtime_path = _make_runtime(tmp_path)
        link_path = tmp_path / "lib" / "libomp.dylib"
        summary = resolve_openmp_runtime(runtime_path=runtime_path, link_path=link_path)
        assert summary.status == OpenMpStatus.PREVIEWED
        assert summary.runtime_path == runtime_path
        assert not link_path.parent.exists()

    def test_resolution_links_the_runtime_when_execution_is_requested(self, monkeypatch, tmp_path):
        """Verifies that an executed request creates the link and reports the verification outcome."""
        monkeypatch.setattr("cindra.orchestration.openmp.sys.platform", "darwin")
        monkeypatch.setattr("cindra.orchestration.openmp._openmp_runtime_loadable", lambda: False)
        monkeypatch.setattr("cindra.orchestration.openmp._verify_runtime_loadable", lambda: True)
        runtime_path = _make_runtime(tmp_path)
        link_path = tmp_path / "lib" / "libomp.dylib"
        summary = resolve_openmp_runtime(runtime_path=runtime_path, link_path=link_path, execute=True)
        assert summary.status == OpenMpStatus.LINKED
        assert summary.loadable
        assert link_path.resolve() == runtime_path

    def test_resolution_links_a_loadable_host_when_forced(self, monkeypatch, tmp_path):
        """Verifies that forcing the request bypasses the check that leaves a loadable host alone."""
        monkeypatch.setattr("cindra.orchestration.openmp.sys.platform", "darwin")
        monkeypatch.setattr("cindra.orchestration.openmp._openmp_runtime_loadable", lambda: True)
        runtime_path = _make_runtime(tmp_path)
        summary = resolve_openmp_runtime(runtime_path=runtime_path, link_path=tmp_path / "link.dylib", force=True)
        assert summary.status == OpenMpStatus.PREVIEWED

    def test_resolution_derives_the_link_path_from_the_loader_search_path(self, monkeypatch, tmp_path):
        """Verifies that an omitted link path resolves to the directory the loader searches by default."""
        monkeypatch.setattr("cindra.orchestration.openmp.sys.platform", "darwin")
        monkeypatch.setattr("cindra.orchestration.openmp._openmp_runtime_loadable", lambda: False)
        monkeypatch.setattr("cindra.orchestration.openmp._LINK_DIRECTORY", tmp_path)
        summary = resolve_openmp_runtime(runtime_path=_make_runtime(tmp_path))
        assert summary.link_path == tmp_path / "libomp.dylib"


class TestMissingRuntimeVerification:
    """Tests the pre-dispatch check that aborts a run whose OpenMP runtime cannot be loaded."""

    def test_check_passes_on_platforms_needing_no_runtime(self, monkeypatch):
        """Verifies that a platform resolving its threading layer without a runtime is admitted."""
        monkeypatch.setattr("cindra.orchestration.openmp.sys.platform", "linux")
        verify_openmp_runtime()

    def test_check_passes_when_the_runtime_loads(self, monkeypatch):
        """Verifies that a macOS host whose runtime loads is admitted."""
        monkeypatch.setattr("cindra.orchestration.openmp.sys.platform", "darwin")
        monkeypatch.setattr("cindra.orchestration.openmp._openmp_runtime_loadable", lambda: True)
        verify_openmp_runtime()

    def test_check_aborts_and_names_the_remedy_when_the_runtime_is_missing(self, monkeypatch):
        """Verifies that a macOS host carrying no loadable runtime aborts and is told which command resolves it."""
        monkeypatch.setattr("cindra.orchestration.openmp.sys.platform", "darwin")
        monkeypatch.setattr("cindra.orchestration.openmp._openmp_runtime_loadable", lambda: False)

        expected_message = (
            f"Unable to locate the OpenMP runtime ({openmp_module._OPENMP_LIBRARY_NAME}) that the Numba threading "
            f"layer loads on macOS. Processing fails once it reaches a parallelized stage until the runtime is "
            f"loadable. Run 'cindra omp' to report the runtimes found on this host, and 'cindra omp --yes' to link "
            f"one into {openmp_module._LINK_DIRECTORY}. Install one with 'brew install libomp' when the report finds "
            f"none."
        )
        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            verify_openmp_runtime()
