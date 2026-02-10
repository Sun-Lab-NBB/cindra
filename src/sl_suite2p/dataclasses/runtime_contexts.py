"""Provides RuntimeContext classes that combine configuration and runtime data for pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from natsort import natsorted
from ataraxis_base_utilities import console, ensure_directory_exists

from .multi_day_data import MultiDayRuntimeData, find_suite2p_directory
from .single_day_data import SingleDayRuntimeData
from .multi_day_configuration import MultiDayConfiguration
from .single_day_configuration import AcquisitionParameters, SingleDayConfiguration

if TYPE_CHECKING:
    from pathlib import Path


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

        This method derives the root path from self.configuration.file_io.save_path and creates the suite2p subdirectory
        if it does not exist. It should be called once at pipeline initialization to save the static data shared
        across all planes.

        Raises:
            ValueError: If save_path is not configured in the configuration.
        """
        if self.configuration.file_io.save_path is None:
            message = (
                "Unable to save shared configuration data. The save_path must be configured in the FileIO section "
                "of the configuration, but it is currently None."
            )
            console.error(message=message, error=ValueError)

        root_path = self.configuration.file_io.save_path / "suite2p"
        root_path.mkdir(parents=True, exist_ok=True)

        self.configuration.save(file_path=root_path / "configuration.yaml")
        self.acquisition.to_yaml(file_path=root_path / "acquisition_parameters.yaml")

    def save_runtime(self) -> None:
        """Saves this plane's runtime data to its output directory.

        This method uses self.runtime.io.output_directory as the save location. This directory is set during plane
        initialization and is plane-specific (e.g., plane_0/).

        Raises:
            ValueError: If output_directory is not set in the runtime IOData.
        """
        if self.runtime.io.output_directory is None:
            message = (
                "Unable to save runtime data. The output_directory must be set in the IOData section of the "
                "runtime data, but it is currently None."
            )
            console.error(message=message, error=ValueError)

        self.runtime.save(output_path=self.runtime.io.output_directory)

    @classmethod
    def load(cls, root_path: Path, plane_index: int = -1) -> RuntimeContext | list[RuntimeContext]:
        """Loads one or more RuntimeContext instances from disk.

        This method loads the shared configuration and acquisition parameters from the root suite2p directory, then
        loads plane-specific runtime data from plane directories. Use plane_index=-1 to load all available planes,
        or specify a non-negative index to load a single plane.

        Args:
            root_path: The root suite2p output directory containing configuration.yaml and acquisition_parameters.yaml.
            plane_index: The index of the plane to load. Use -1 to load all available planes.

        Returns:
            A single RuntimeContext if plane_index >= 0, or a list of all RuntimeContext instances if plane_index
            is -1.

        Raises:
            FileNotFoundError: If the configuration files or specified plane directory do not exist.
        """
        config_path = root_path / "configuration.yaml"
        acquisition_path = root_path / "acquisition_parameters.yaml"

        if not config_path.exists():
            message = (
                f"Unable to load RuntimeContext. Configuration file does not exist at the specified path: "
                f"{config_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        if not acquisition_path.exists():
            message = (
                f"Unable to load RuntimeContext. Acquisition parameters file does not exist at the specified path: "
                f"{acquisition_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        config = SingleDayConfiguration.load(file_path=config_path)
        acquisition = AcquisitionParameters.from_yaml(file_path=acquisition_path)

        if plane_index == -1:
            # Loads all planes.
            plane_directories = natsorted([d for d in root_path.glob("plane_*") if d.is_dir()])
            contexts: list[RuntimeContext] = []

            for plane_directory in plane_directories:
                runtime = SingleDayRuntimeData.load(output_path=plane_directory)
                contexts.append(cls(configuration=config, acquisition=acquisition, runtime=runtime))

            return contexts

        # Loads a specific plane.
        plane_path = root_path / f"plane_{plane_index}"
        if not plane_path.exists():
            message = (
                f"Unable to load RuntimeContext. Plane directory does not exist at the specified path: {plane_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        runtime = SingleDayRuntimeData.load(output_path=plane_path)
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

        main_session_path = self._get_main_session_path()
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

    def _get_main_session_path(self) -> Path:
        """Returns the main session's multiday output path (first session after natural sorting).

        Discovers the suite2p directory for the main session and places the multiday output as a sibling of the
        suite2p directory (under the same parent).
        """
        main_session_directory = self.configuration.session_io.session_directories[0]
        suite2p_directory = find_suite2p_directory(session_directory=main_session_directory)
        return suite2p_directory.parent / "multiday" / self.configuration.session_io.dataset_name

    @classmethod
    def load(cls, root_path: Path, session_index: int = -1) -> MultiDayRuntimeContext | list[MultiDayRuntimeContext]:
        """Loads one or more MultiDayRuntimeContext instances from disk.

        This method loads the configuration from the main session's multiday directory, then loads session-specific
        runtime data from each session's multiday directory. Use session_index=-1 to load all available sessions,
        or specify a non-negative index to load a single session.

        Args:
            root_path: The main session's multiday output directory containing multiday_configuration.yaml.
            session_index: The index of the session to load. Use -1 to load all available sessions.

        Returns:
            A single MultiDayRuntimeContext if session_index >= 0, or a list of all MultiDayRuntimeContext instances
            if session_index is -1.

        Raises:
            FileNotFoundError: If the configuration file or specified session directory does not exist.
        """
        config_path = root_path / "multiday_configuration.yaml"

        if not config_path.exists():
            message = (
                f"Unable to load MultiDayRuntimeContext. Configuration file does not exist at the specified path: "
                f"{config_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        configuration = MultiDayConfiguration.load(file_path=config_path)

        # Resolves session directories and output paths from configuration. The multiday output directory is
        # a sibling of the suite2p directory (under the same parent), so the suite2p directory must be discovered
        # for each session.
        session_directories = natsorted(configuration.session_io.session_directories)
        dataset_name = configuration.session_io.dataset_name

        if session_index == -1:
            # Loads all sessions.
            contexts: list[MultiDayRuntimeContext] = []

            for session_dir in session_directories:
                suite2p_directory = find_suite2p_directory(session_directory=session_dir)
                output_path = suite2p_directory.parent / "multiday" / dataset_name
                runtime_path = output_path / "multiday_runtime_data.yaml"
                runtime = (
                    MultiDayRuntimeData.load(output_path=output_path)
                    if runtime_path.exists()
                    else MultiDayRuntimeData()
                )
                contexts.append(cls(configuration=configuration, runtime=runtime))

            return contexts

        # Loads a specific session.
        if session_index < 0 or session_index >= len(session_directories):
            message = (
                f"Unable to load MultiDayRuntimeContext. Session index {session_index} is out of range. "
                f"Valid range is 0 to {len(session_directories) - 1}."
            )
            console.error(message=message, error=IndexError)

        session_dir = session_directories[session_index]
        suite2p_directory = find_suite2p_directory(session_directory=session_dir)
        output_path = suite2p_directory.parent / "multiday" / dataset_name
        runtime_path = output_path / "multiday_runtime_data.yaml"
        runtime = MultiDayRuntimeData.load(output_path=output_path) if runtime_path.exists() else MultiDayRuntimeData()

        return cls(configuration=configuration, runtime=runtime)
