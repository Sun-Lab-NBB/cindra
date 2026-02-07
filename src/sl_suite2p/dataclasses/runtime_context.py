"""Provides the RuntimeContext class that combines configuration, acquisition parameters, and runtime data."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from natsort import natsorted
from ataraxis_base_utilities import console

from .single_day_data import SingleDayRuntimeData
from .single_day_configuration import AcquisitionParameters, SingleDayConfiguration

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class RuntimeContext:
    """Combines configuration, acquisition parameters, and runtime data for pipeline functions.

    Notes:
        This class provides a unified interface for pipeline functions to access user configuration (immutable),
        acquisition parameters (from input data), and runtime data (computed by pipeline). It replaces the legacy ops
        dictionary pattern with a type-safe structure.

        Each RuntimeContext instance represents a single plane (or virtual plane for MROI data). The config and
        acquisition fields are shared across all planes, while the runtime field contains plane-specific data.
    """

    config: SingleDayConfiguration
    """The user configuration, which remains immutable during processing."""

    acquisition: AcquisitionParameters
    """The acquisition parameters loaded from the input data's JSON file. This describes the recording setup including
    frame rate, plane count, channel count, and MROI geometry if applicable."""

    runtime: SingleDayRuntimeData
    """The runtime data, which is computed and updated by pipeline stages."""

    def save_shared(self) -> None:
        """Saves shared configuration and acquisition parameters to the root output directory.

        This method derives the root path from self.config.file_io.save_path and creates the suite2p subdirectory
        if it does not exist. It should be called once at pipeline initialization to save the static data shared
        across all planes.

        Raises:
            ValueError: If save_path is not configured in the configuration.
        """
        if self.config.file_io.save_path is None:
            message = (
                "Unable to save shared configuration data. The save_path must be configured in the FileIO section "
                "of the configuration, but it is currently None."
            )
            console.error(message=message, error=ValueError)

        root_path = self.config.file_io.save_path / "suite2p"
        root_path.mkdir(parents=True, exist_ok=True)

        self.config.save(file_path=root_path / "configuration.yaml")
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
        """Loads one or more instances from disk.

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
                contexts.append(cls(config=config, acquisition=acquisition, runtime=runtime))

            return contexts

        # Loads a specific plane.
        plane_path = root_path / f"plane_{plane_index}"
        if not plane_path.exists():
            message = (
                f"Unable to load RuntimeContext. Plane directory does not exist at the specified path: {plane_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        runtime = SingleDayRuntimeData.load(output_path=plane_path)
        return cls(config=config, acquisition=acquisition, runtime=runtime)
