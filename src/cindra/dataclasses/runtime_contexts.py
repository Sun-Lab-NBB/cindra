"""Provides RuntimeContext classes that combine configuration and runtime data for pipelines."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from natsort import natsorted
from ataraxis_base_utilities import console, ensure_directory_exists

from .multi_day_data import MultiDayRuntimeData
from .single_day_data import SingleDayRuntimeData
from .multi_day_configuration import MultiDayConfiguration
from .single_day_configuration import AcquisitionParameters, SingleDayConfiguration


def _load_single_day_runtime(plane_directory: Path) -> SingleDayRuntimeData:
    """Loads a SingleDayRuntimeData instance and corrects stale paths if the dataset was relocated.

    When a dataset is moved between machines, the paths cached in the plane's runtime YAML no longer match the actual
    directory structure. This function detects the mismatch by comparing the cached output_path to the known-correct
    plane directory, computes a prefix substitution, relocates all cached paths and persists the corrected paths to
    disk.

    Args:
        plane_directory: The actual on-disk path to the plane directory (e.g., ``cindra/plane_0``).

    Returns:
        A fully-loaded SingleDayRuntimeData instance with all paths and arrays resolved against the correct location.
    """
    runtime = SingleDayRuntimeData.load(output_path=plane_directory)

    if runtime.output_path is not None and runtime.output_path != plane_directory:
        old_prefix, new_prefix = _compute_relocation_prefixes(old_path=runtime.output_path, new_path=plane_directory)
        _relocate_runtime_paths(runtime=runtime, old_prefix=old_prefix, new_prefix=new_prefix)

        # Persists corrected paths so future loads find correct paths without re-relocating.
        runtime.save(output_path=runtime.output_path)

        # Reloads the runtime from the corrected YAML so that arrays are resolved against the new paths.
        runtime = SingleDayRuntimeData.load(output_path=plane_directory)

    return runtime


def _compute_relocation_prefixes(old_path: Path, new_path: Path) -> tuple[Path, Path]:
    """Computes the old and new path prefixes for relocating a moved dataset.

    Walks both paths from the end to find the longest common suffix, then returns the diverging prefixes. The
    assumption is that the entire processed data hierarchy was moved intact, so only the leading prefix changed.

    Args:
        old_path: The cached path from the serialized YAML data.
        new_path: The actual resolved path on the current filesystem.

    Returns:
        A tuple of (old_prefix, new_prefix) that can be used to transform any cached path to its new location.
    """
    old_parts = old_path.parts
    new_parts = new_path.parts

    # Walks from the end to find the longest common suffix.
    common_suffix_length = 0
    for old_part, new_part in zip(reversed(old_parts), reversed(new_parts), strict=False):
        if old_part == new_part:
            common_suffix_length += 1
        else:
            break

    old_prefix = Path(*old_parts[: len(old_parts) - common_suffix_length]) if common_suffix_length > 0 else old_path
    new_prefix = Path(*new_parts[: len(new_parts) - common_suffix_length]) if common_suffix_length > 0 else new_path
    return old_prefix, new_prefix


def _relocate_cross_session_path(path: Path, old_prefix: Path, new_prefix: Path) -> Path:
    """Relocates a dataset output path from a different session that does not share the entry session's prefix.

    Multi-day datasets store output paths for all sessions in each session's runtime data. When the entry session's
    prefix is used for relocation, paths belonging to other sessions fail ``relative_to`` because they contain a
    different session-specific directory segment (e.g., a different timestamp-based session directory). This function
    handles that case by splitting the cross-session path at the same depth as the entry prefix, substituting the
    differing session-specific segments into the new prefix, and reattaching the trailing suffix.

    Args:
        path: The stale cross-session path to relocate.
        old_prefix: The entry session's old prefix computed by ``_compute_relocation_prefixes``.
        new_prefix: The entry session's new prefix computed by ``_compute_relocation_prefixes``.

    Returns:
        The relocated cross-session path with the correct new prefix and session-specific segments.

    Raises:
        ValueError: If the cross-session path has fewer segments than the entry session prefix, indicating an
            incompatible directory structure.
    """
    path_parts = path.parts
    old_prefix_parts = old_prefix.parts

    if len(path_parts) < len(old_prefix_parts):
        message = (
            f"Unable to relocate cross-session path {path}. The path has {len(path_parts)} segments but the entry "
            f"session prefix has {len(old_prefix_parts)} segments, indicating an incompatible directory structure."
        )
        console.error(message=message, error=ValueError)

    # Splits the cross-session path into a base (same depth as old_prefix) and a trailing suffix.
    cross_base_parts = path_parts[: len(old_prefix_parts)]
    suffix_parts = path_parts[len(old_prefix_parts) :]

    # Identifies directory segments that differ between the entry session's old prefix and the cross-session base,
    # then applies those differing segments to the new prefix at the corresponding positions. This preserves the
    # structural transformation (e.g., directory insertion or rename) while swapping in the correct session-specific
    # segments.
    relocated_prefix_parts = list(new_prefix.parts)
    for index, (old_part, cross_part) in enumerate(zip(old_prefix_parts, cross_base_parts, strict=False)):
        if old_part != cross_part and index < len(relocated_prefix_parts):
            relocated_prefix_parts[index] = cross_part

    relocated_prefix = Path(*relocated_prefix_parts)
    if suffix_parts:
        return relocated_prefix / Path(*suffix_parts)
    return relocated_prefix


def _relocate_runtime_paths(
    runtime: SingleDayRuntimeData | MultiDayRuntimeData, old_prefix: Path, new_prefix: Path
) -> None:
    """Applies a prefix substitution to all cached paths in a runtime data instance.

    Args:
        runtime: The runtime data instance whose paths will be updated in-place. Accepts either SingleDayRuntimeData
            or MultiDayRuntimeData instances.
        old_prefix: The stale path prefix to replace.
        new_prefix: The correct path prefix on the current filesystem.
    """
    if runtime.output_path is not None:
        runtime.output_path = new_prefix / runtime.output_path.relative_to(old_prefix)

    # Paths that already reside under new_prefix are skipped to prevent double-relocation. This occurs when some
    # paths in the YAML were updated independently (e.g., single-day pipeline re-ran after a directory rename) while
    # others remained stale.
    if isinstance(runtime, SingleDayRuntimeData):
        if runtime.io.registered_binary_path is not None and not runtime.io.registered_binary_path.is_relative_to(
            new_prefix
        ):
            runtime.io.registered_binary_path = new_prefix / runtime.io.registered_binary_path.relative_to(old_prefix)
        if (
            runtime.io.registered_binary_path_channel_2 is not None
            and not runtime.io.registered_binary_path_channel_2.is_relative_to(new_prefix)
        ):
            runtime.io.registered_binary_path_channel_2 = (
                new_prefix / runtime.io.registered_binary_path_channel_2.relative_to(old_prefix)
            )
        if runtime.io.output_path is not None and not runtime.io.output_path.is_relative_to(new_prefix):
            runtime.io.output_path = new_prefix / runtime.io.output_path.relative_to(old_prefix)
    else:
        if runtime.io.data_path is not None and not runtime.io.data_path.is_relative_to(new_prefix):
            runtime.io.data_path = new_prefix / runtime.io.data_path.relative_to(old_prefix)

        # Relocates dataset_output_paths with cross-session fallback. Multi-day datasets store paths for all
        # sessions, but the prefix is session-specific. Paths from other sessions fail relative_to and are handled
        # by _relocate_cross_session_path which substitutes the differing session-specific segments.
        relocated_paths: list[Path] = []
        for path in runtime.io.dataset_output_paths:
            if path.is_relative_to(new_prefix):
                relocated_paths.append(path)
                continue
            try:
                relocated_paths.append(new_prefix / path.relative_to(old_prefix))
            except ValueError:
                relocated_paths.append(
                    _relocate_cross_session_path(path=path, old_prefix=old_prefix, new_prefix=new_prefix)
                )
        runtime.io.dataset_output_paths = tuple(relocated_paths)


@dataclass
class RuntimeContext:
    """Combines configuration, acquisition parameters, and runtime data used in the single-day processing pipeline.

    Notes:
        This class provides a unified interface for pipeline functions to access user configuration (immutable),
        acquisition parameters (from input data), and runtime data (computed by pipeline). It replaces the legacy ops
        dictionary pattern with a type-safe structure.

        Each RuntimeContext instance represents a single plane (or virtual plane for MROI data). The configuration and
        acquisition fields are shared across all planes, while the runtime field contains plane-specific data.
    """

    configuration: SingleDayConfiguration
    """The user-defined processing configuration, which remains immutable during processing."""

    acquisition: AcquisitionParameters
    """The acquisition parameters loaded from the input data's JSON file. This describes the recording setup including
    frame rate, plane count, channel count, and MROI geometry if applicable."""

    runtime: SingleDayRuntimeData
    """The runtime data, which is computed and updated by pipeline stages."""

    def save_shared(self) -> None:
        """Saves shared configuration and acquisition parameters to the root output directory.

        This method derives the root path from self.configuration.file_io.output_path and creates the cindra subdirectory
        if it does not exist. It should be called once at pipeline initialization to save the static data shared
        across all planes.

        Raises:
            ValueError: If output_path is not configured in the configuration.
        """
        if self.configuration.file_io.output_path is None:
            message = (
                "Unable to save shared configuration data. The output_path must be configured in the FileIO section "
                "of the configuration, but it is currently None."
            )
            console.error(message=message, error=ValueError)

        root_path = self.configuration.file_io.output_path / "cindra"
        root_path.mkdir(parents=True, exist_ok=True)

        self.configuration.save(file_path=root_path / "configuration.yaml")
        self.acquisition.to_yaml(file_path=root_path / "acquisition_parameters.yaml")

    def save_runtime(self) -> None:
        """Saves this plane's runtime data to its output directory.

        This method uses self.runtime.io.output_path as the save location. This directory is set during plane
        initialization and is plane-specific (e.g., plane_0/).

        Raises:
            ValueError: If output_path is not set in the runtime IOData.
        """
        if self.runtime.io.output_path is None:
            message = (
                "Unable to save runtime data. The output_path must be set in the IOData section of the "
                "runtime data, but it is currently None."
            )
            console.error(message=message, error=ValueError)

        self.runtime.save(output_path=self.runtime.io.output_path)

    @classmethod
    def load(cls, root_path: Path, plane_index: int = -1) -> RuntimeContext | list[RuntimeContext]:
        """Loads one or more RuntimeContext instances from disk.

        Searches root_path recursively for configuration.yaml to discover the cindra output directory, then loads
        shared configuration, acquisition parameters, and plane-specific runtime data. If the dataset was moved to a
        different location, stale output_path values in each plane's runtime YAML are silently corrected so that
        array loading succeeds.

        Args:
            root_path: The path to the session's root processed data directory. The method searches
                recursively for configuration.yaml to locate the cindra output directory.
            plane_index: The index of the plane to load. Use -1 to load all available planes.

        Returns:
            A single RuntimeContext if plane_index >= 0, or a list of all RuntimeContext instances if plane_index
            is -1.

        Raises:
            FileNotFoundError: If no configuration.yaml is found, or if required files are missing.
            RuntimeError: If multiple configuration.yaml files are found under root_path.
        """
        # Discovers the cindra output directory within the root_path directory tree.
        matches = list(root_path.rglob("configuration.yaml"))

        if len(matches) == 0:
            message = (
                f"Unable to load RuntimeContext. No configuration.yaml file was found under {root_path}. "
                f"Ensure the single-day pipeline has been run for this session."
            )
            console.error(message=message, error=FileNotFoundError)

        if len(matches) > 1:
            message = (
                f"Unable to load RuntimeContext. Found {len(matches)} configuration.yaml files under "
                f"{root_path}, but expected exactly one."
            )
            console.error(message=message, error=RuntimeError)

        cindra_root = matches[0].parent

        config_path = cindra_root / "configuration.yaml"
        acquisition_path = cindra_root / "acquisition_parameters.yaml"

        if not acquisition_path.exists():
            message = (
                f"Unable to load RuntimeContext. Acquisition parameters file does not exist at the expected path: "
                f"{acquisition_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        config = SingleDayConfiguration.load(file_path=config_path)
        acquisition = AcquisitionParameters.from_yaml(file_path=acquisition_path)

        if plane_index == -1:
            # Loads all planes.
            plane_directories = natsorted([d for d in cindra_root.glob("plane_*") if d.is_dir()])
            contexts: list[RuntimeContext] = []

            for plane_directory in plane_directories:
                runtime = _load_single_day_runtime(plane_directory=plane_directory)
                contexts.append(cls(configuration=config, acquisition=acquisition, runtime=runtime))

            return contexts

        # Loads a specific plane.
        plane_path = cindra_root / f"plane_{plane_index}"
        if not plane_path.exists():
            message = (
                f"Unable to load RuntimeContext. Plane directory does not exist at the specified path: {plane_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        runtime = _load_single_day_runtime(plane_directory=plane_path)
        return cls(configuration=config, acquisition=acquisition, runtime=runtime)


@dataclass
class MultiDayRuntimeContext:
    """Combines configuration and runtime data used in the multi-day processing pipeline.

    Notes:
        This class provides a unified interface for multi-day pipeline functions to access user configuration
        (immutable) and per-session runtime data (computed during processing). It replaces the legacy ops dictionary
        pattern used in the original multi-day implementation with a type-safe structure.

        Each MultiDayRuntimeContext instance represents a single session. The configuration is shared across all
        session contexts, while the runtime field contains session-specific data. This mirrors the RuntimeContext
        pattern where each instance represents a single plane.
    """

    configuration: MultiDayConfiguration
    """The user-defined processing configuration, which remains immutable during processing."""

    runtime: MultiDayRuntimeData
    """The per-session runtime data, which is computed and updated by pipeline stages."""

    def save_shared(self) -> None:
        """Saves the shared configuration to the main session's output directory.

        This method saves the immutable configuration to the first session's multiday directory. It should be called
        once at the start of processing.

        Raises:
            ValueError: If output_path is not set in the runtime data.
        """
        if self.runtime.output_path is None:
            message = (
                "Unable to save configuration. The output_path must be set in the runtime data, "
                "but it is currently None."
            )
            console.error(message=message, error=ValueError)

        main_session_path = self.runtime.io.dataset_output_paths[0]
        ensure_directory_exists(main_session_path)

        self.configuration.save(file_path=main_session_path / "multiday_configuration.yaml")

    def save_runtime(self) -> None:
        """Saves this session's runtime data to its output directory.

        This method uses self.runtime.output_path as the save location. This directory is session-specific.

        Raises:
            ValueError: If output_path is not set in the runtime data.
        """
        if self.runtime.output_path is None:
            message = (
                "Unable to save runtime data. The output_path must be set in the runtime data, "
                "but it is currently None."
            )
            console.error(message=message, error=ValueError)

        self.runtime.save(output_path=self.runtime.output_path)

    @classmethod
    def load(cls, root_path: Path, session_index: int = -1) -> MultiDayRuntimeContext | list[MultiDayRuntimeContext]:
        """Loads one or more previously-saved MultiDayRuntimeContext instances from a session's data directory.

        Searches root_path recursively for a multiday_runtime_data.yaml file, loads that session's runtime data,
        then uses its stored dataset_output_paths to reconstruct the full dataset hierarchy. If the dataset was
        moved to a different location (e.g., transferred between machines), all cached absolute paths are
        automatically relocated to match the new directory structure.

        Args:
            root_path: The path to any dataset session's root processed data directory. The method searches
                recursively for the multiday_runtime_data.yaml file within this directory tree.
            session_index: The index of the session to load. Use -1 to load all available sessions.

        Returns:
            A single MultiDayRuntimeContext if session_index >= 0, or a list of all MultiDayRuntimeContext instances
            if session_index is -1.

        Raises:
            FileNotFoundError: If no multiday_runtime_data.yaml is found, or configuration files are missing.
            RuntimeError: If multiple multiday_runtime_data.yaml files are found under root_path.
            IndexError: If session_index is out of range.
        """
        # Discovers the multiday_runtime_data.yaml file within the root_path directory tree.
        matches = list(root_path.rglob("multiday_runtime_data.yaml"))

        if len(matches) == 0:
            message = (
                f"Unable to load MultiDayRuntimeContext. No multiday_runtime_data.yaml file was found under "
                f"{root_path}. Ensure the multi-day pipeline has been run for this session."
            )
            console.error(message=message, error=FileNotFoundError)

        if len(matches) > 1:
            message = (
                f"Unable to load MultiDayRuntimeContext. Found {len(matches)} multiday_runtime_data.yaml files "
                f"under {root_path}, but expected exactly one."
            )
            console.error(message=message, error=RuntimeError)

        resolved_output_path = matches[0].parent

        # Loads the entry-point session's runtime to access dataset_output_paths.
        entry_runtime = MultiDayRuntimeData.load(output_path=resolved_output_path)
        output_paths = entry_runtime.io.dataset_output_paths

        if not output_paths:
            message = (
                f"Unable to load MultiDayRuntimeContext. The runtime data at {resolved_output_path} does not "
                f"contain dataset_output_paths. Ensure the data was saved by resolve_multiday_contexts()."
            )
            console.error(message=message, error=FileNotFoundError)

        # Detects whether the dataset was moved by comparing the resolved path to the cached output_path. If they
        # differ, computes a prefix substitution, relocates and re-saves ALL sessions (both multi-day and underlying
        # single-day data) so future loads find correct paths.
        if entry_runtime.output_path is not None and entry_runtime.output_path != resolved_output_path:
            old_prefix, new_prefix = _compute_relocation_prefixes(
                old_path=entry_runtime.output_path, new_path=resolved_output_path
            )
            _relocate_runtime_paths(runtime=entry_runtime, old_prefix=old_prefix, new_prefix=new_prefix)
            output_paths = entry_runtime.io.dataset_output_paths

            # Persists corrected paths for every session's multi-day data and underlying single-day data. The entry
            # session has already been relocated in-place above. Other sessions are loaded (with stale YAML content),
            # relocated, and saved. Single-day plane data is relocated by calling _load_single_day_runtime() which
            # detects the path mismatch and persists corrected paths.
            entry_runtime.save(output_path=entry_runtime.output_path)
            if entry_runtime.io.data_path is not None:
                for plane_dir in entry_runtime.io.data_path.glob("plane_*"):
                    if plane_dir.is_dir() and (plane_dir / "runtime_data.yaml").exists():
                        _load_single_day_runtime(plane_directory=plane_dir)

            for session_output_path in output_paths:
                if session_output_path == resolved_output_path:
                    continue

                # Computes per-session relocation prefixes instead of reusing the entry session's prefix. Each
                # session has its own session-specific directory segment, so cross-session prefix substitution fails.
                other_runtime = MultiDayRuntimeData.load(output_path=session_output_path)
                if other_runtime.output_path is not None and other_runtime.output_path != session_output_path:
                    session_old_prefix, session_new_prefix = _compute_relocation_prefixes(
                        old_path=other_runtime.output_path, new_path=session_output_path
                    )
                    _relocate_runtime_paths(
                        runtime=other_runtime, old_prefix=session_old_prefix, new_prefix=session_new_prefix
                    )
                    other_runtime.save(output_path=session_output_path)

                if other_runtime.io.data_path is not None:
                    for plane_dir in other_runtime.io.data_path.glob("plane_*"):
                        if plane_dir.is_dir() and (plane_dir / "runtime_data.yaml").exists():
                            _load_single_day_runtime(plane_directory=plane_dir)

            # Reloads the entry runtime from the corrected YAML so that arrays are resolved against the new paths.
            # This will fail if the single-day data is unavailable, as CombinedData is required.
            entry_runtime = MultiDayRuntimeData.load(output_path=resolved_output_path)
            output_paths = entry_runtime.io.dataset_output_paths

        # Loads configuration from the first output path (the main session after natural sorting).
        config_path = output_paths[0] / "multiday_configuration.yaml"
        if not config_path.exists():
            message = (
                f"Unable to load MultiDayRuntimeContext. Configuration file does not exist at the expected "
                f"path: {config_path}."
            )
            console.error(message=message, error=FileNotFoundError)
        configuration = MultiDayConfiguration.load(file_path=config_path)

        if session_index == -1:
            # Loads all sessions. Reuses the already-loaded entry runtime to avoid redundant I/O.
            contexts: list[MultiDayRuntimeContext] = []
            for output_path in output_paths:
                if output_path == resolved_output_path:
                    contexts.append(cls(configuration=configuration, runtime=entry_runtime))
                else:
                    runtime = MultiDayRuntimeData.load(output_path=output_path)
                    contexts.append(cls(configuration=configuration, runtime=runtime))
            return contexts

        # Loads a specific session.
        if session_index < 0 or session_index >= len(output_paths):
            message = (
                f"Unable to load MultiDayRuntimeContext. Session index {session_index} is out of range. "
                f"Valid range is 0 to {len(output_paths) - 1}."
            )
            console.error(message=message, error=IndexError)

        # Reuses the already-loaded entry runtime if it matches the requested index.
        target_path = output_paths[session_index]
        if target_path == resolved_output_path:
            return cls(configuration=configuration, runtime=entry_runtime)

        runtime = MultiDayRuntimeData.load(output_path=target_path)
        return cls(configuration=configuration, runtime=runtime)
