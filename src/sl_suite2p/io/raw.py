"""This module provides tools for reading and writing image data stored in Thorlabs raw (.raw) files and their
associated XML (.xml) configuration files.
"""

import copy
from typing import TYPE_CHECKING, Any
from pathlib import Path

from tqdm import tqdm
import numpy as np
from ataraxis_time import PrecisionTimer
from quick_xmltodict import parse
from ataraxis_base_utilities import console

from .utils import set_yaml_path
from ..configuration import RuntimeData

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _RawFile:
    """Creates or opens a Thorlabs raw (.raw) file and its XML (.xml) companion configuration file for reading and/or
    writing image data.

    This class parses the data stored inside the XML (.xml) configuration file and exposes the parameters used to read
    Thorlabs raw (.raw) files. The class instance exposes all data recording parameters for the target Thorlabs raw
    (.raw) file as class attributes.

    Args:
        directory_path: The absolute path to the directory that stores the target .raw file and the .xml configuration
            file.

    Notes:
        The instance is statically configured to search for the '001.raw' file and any file with '.xml' extension.
        It will not work as expected unless both files are found under the input directory and are named according to
        expectation.

    Attributes:
        _raw_file_path: The absolute path to the target raw (.raw) file.
        _raw_file_size: The size (in bytes) of the target raw (.raw) file.
        _xml_file_path: The absolute path to the XML (.xml) configuration file for the target .raw file.
        z_planes: The number of z-planes in the recording.
        recorded_planes: The total number of recorded planes, including any flyback planes.
        height: The height (in pixels) of each frame stored inside the file.
        width: The width (in pixels) of each frame stored inside the file.
        channel: The number of channels in the recording (1 for single channel data, 2 for multichannel data).
        frame_rate: The frame rate at which the data was recorded.
        physical_width: The physical width (in micrometers) of each frame.
        physical_height: The physical height (in micrometers) of each frame.
        frame_number: The total number of frames stored in the target file.

    Raises:
        FileNotFoundError: If either the target .raw file or its .xml configuration file is not found.
    """

    def __init__(self, directory_path: Path) -> None:
        # Initializes variables to store the absolute file paths of the target .raw file and its .xml configuration
        # file.
        raw_file_path: Path | None = None
        xml_file_path: Path | None = None

        # Loops over the files in the input directory to search for the target .raw file and .xml file.
        for file in directory_path.iterdir():
            # Verifies the file is a valid file.
            if file.is_file():
                # Checks if the file is the main .raw file. If so, stores the absolute path to the target file.
                if file.name.lower().endswith("001.raw"):
                    raw_file_path = file

                # Checks if the file is the .xml file. If so, stores the absolute path to the target file.
                elif file.name.lower().endswith(".xml"):
                    xml_file_path = file

            # If both files are found, exits the loop early.
            if raw_file_path and xml_file_path:
                break
        # If the main .raw file was not found, raises a FileNotFoundError.
        if not raw_file_path:
            message = (
                f"Unable to convert Thorlabs RAW data to the Suite2p BinaryFile format. Unable to find the required "
                f"'001.raw' file inside the input directory: {directory_path}."
            )
            console.error(message=message, error=FileNotFoundError)
            raise FileNotFoundError(message)  # Fallback to appease mypy, should not be reachable

        # If the .xml configuration file was not found, raises a FileNotFoundError.
        if not xml_file_path:
            message = (
                f"Unable to convert Thorlabs RAW data to the Suite2p BinaryFile format. Unable to find the required "
                f".xml configuration file inside the input directory: {directory_path}."
            )
            console.error(message=message, error=FileNotFoundError)
            raise FileNotFoundError(message)  # Fallback to appease mypy, should not be reachable

        # If both target files are found, uses them to initialize and configure instance attributes.
        self._raw_file_path: Path = raw_file_path
        self._raw_file_size: int = self._raw_file_path.stat().st_size
        self._xml_file_path: Path = xml_file_path

        # Initializes the public class attributes with default values.
        self.z_planes: int = 1
        self.recorded_planes: int = 1
        self.height: int = 0
        self.width: int = 0
        self.channel: int = 0
        self.frame_rate: float = 0
        self.physical_width: float = 0
        self.physical_height: float = 0
        self.frame_number: int = 0

        # Parses the .xml configuration file and reassigns the attributes to store and expose the parsed data.
        with self._xml_file_path.open(encoding="utf-8") as xml_file:
            self._load_xml_config(raw_file_size=self._raw_file_size, xml_contents=parse(xml_file.read()))

        # Determines and stores the dimensions of the data in the .RAW file to _shape attribute.
        self._shape = self._find_shape()

    @property
    def path(self) -> Path:
        """Returns the absolute path to the target .raw file."""
        return self._raw_file_path

    @property
    def size(self) -> int:
        """Returns the size (in bytes) of the target .raw file."""
        return self._raw_file_size

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the dimensions of the data in the file as a tuple of up to four elements.

        If the recording uses multiple planes, the first element is the number of planes, followed by the number of
        frames at each plane. If the recording uses a single plane, the first element is the number of frames. The
        following elements are the height of each frame and the width of each frame, in this order.
        """
        return self._shape

    def _find_shape(self) -> tuple[int, ...]:
        """Calculates and returns the dimensions of the data in the target file as a tuple of up to four elements.

        If the recording contains multiple recorded planes, the shape includes the number of recorded planes as the
        first dimension. If the recording uses two functional channels, the number of frames is doubled.

        Returns:
            The dimensions of the data in the file as up to a tuple of four elements. If the recording uses multiple
            planes, the first element is the number of planes, followed by the number of frames at each plane. If the
            recording uses a single plane, the first element is the number of frames. The following elements are the
            height of each frame and the width of each frame, in this order.
        """
        # Initializes shape as a tuple using class attributes.
        shape: tuple[int, ...] = (self.frame_number, self.height, self.width)

        # If the recording uses two functional channels, adjusts the first dimension 'frame_number' by doubling it.
        if self.channel > 1:
            shape = (self.frame_number * 2, *shape[1:])

        # If there are multiple recorded planes, inserts the number of recorded planes as the first dimension of the
        # shape tuple.
        if self.recorded_planes > 1:
            shape = (self.recorded_planes, *shape)

        # Returns the shape.
        return shape

    def _load_xml_config(self, raw_file_size: int, xml_contents: dict[str, Any]) -> None:
        """Loads the recording parameters from the XML (.xml) configuration file associated with the target Thorlabs
        raw data.

        This method extracts relevant data from the XML (.xml) configuration file and overwrites the attributes of
        the _RawFile instance with the read parameters.

        Args:
            raw_file_size: The size (in bytes) of the target .raw file.
            xml_contents: The content of the XML (.xml) configuration file for the target raw file loaded into memory
                as a dictionary.
        """
        # Queries the configuration data from the input 'xml_file' dictionary.
        xml_data = xml_contents["ThorImageExperiment"]

        # Updates the class attributes with the data parsed from the 'xml_data' dictionary.
        self.height = int(xml_data["LSM"]["@pixelX"])
        self.width = int(xml_data["LSM"]["@pixelY"])
        self.channel = int(xml_data["LSM"]["@channel"])
        self.frame_rate = float(xml_data["LSM"]["@frameRate"])
        self.physical_width = float(xml_data["LSM"]["@widthUM"])
        self.physical_height = float(xml_data["LSM"]["@heightUM"])
        self.frame_number = int(xml_data["Streaming"]["@frames"])

        # If z-stack is enabled, calculates and updates the number of z-planes, recorded planes, and number of frames.
        if int(xml_data["Streaming"]["@zFastEnable"]) > 0:
            self.z_planes = int(xml_data["ZStage"]["@steps"])
            self.recorded_planes = int(xml_data["Streaming"]["@flybackFrames"]) + self.z_planes
            self.frame_number = int(self.frame_number / self.recorded_planes)

        # Updates the 'channel' attribute to 2 for multichannel recordings.
        if self.channel > 1:
            self.channel = 2

        # If the experiment was stopped mid-recording, estimates the number of frames using dimension data and the size
        # of the file.
        if xml_data["ExperimentStatus"]["@value"] == "Stopped":
            all_frames = int(raw_file_size / self.height / self.width / self.recorded_planes / self.channel / 2)
            self.frame_number = int(all_frames / self.recorded_planes)


def raw_to_binary(runtime_data: RuntimeData, override_runtime_parameters: bool = True) -> RuntimeData:
    """Reads the input data stored as Thorlabs .raw files and converts it to the suite2p plane binary (.bin) file(s).

    Args:
        runtime_data: A RuntimeData instance that stores the suite2p single-day configuration and runtime parameters.
        override_runtime_parameters: Determines whether to override certain configuration parameters with data loaded
                                     from the .xml configuration files stored together with Thorlabs .raw files.
    """
    # Instantiates and resets the run timer
    timer = PrecisionTimer("s")
    timer.reset()

    # Loads Thorlabs .raw files from the paths provided and converts them into _RawFile instances.
    raw_files = [_RawFile(path) for path in runtime_data.configuration.file_io.data_path]

    # Initializes the destination files and resolves paths and configuration for further .raw to .bin file conversion.
    plane_runtime_data_list = _initialize_destination_files(
        runtime_data=runtime_data, raw_files=raw_files, override_runtime_parameters=override_runtime_parameters
    )

    # Determines the number of frames across all .raw files. This is used for the progress bar visualization.
    total_frames = sum(raw_file.frame_number for raw_file in raw_files)

    # Creates the progress bar.
    progress_bar = tqdm(
        total=total_frames,
        desc="Converting Thorlabs raw frames to binary",
        unit="frames",
        disable=not runtime_data.configuration.main.progress_bars,
    )

    # Converts all the .raw files into .bin format.
    for raw_file in raw_files:
        # Performs the raw to binary conversion using the target raw file.
        _single_raw_to_binary(plane_runtime_data_list=plane_runtime_data_list, raw_file=raw_file)

        # Updates the progress bar with the number of frames processed in the target file.
        progress_bar.update(raw_file.frame_number)

    # Closes the progress bar when the binary conversion is over.
    progress_bar.close()

    # Creates a mean image based on the final number of frames for each plane.
    plane_number = runtime_data.configuration.main.nplanes
    for plane_index in range(plane_number):
        # Loads the RuntimeData for this plane
        plane_directory = Path(runtime_data.configuration.output.save_path).joinpath("suite2p", f"plane{plane_index}")
        plane_runtime = RuntimeData.from_yaml(file_path=plane_directory.joinpath("runtime_data.yaml"))

        plane_data = plane_runtime.data.file_io
        mean_image_path = plane_directory.joinpath("mean_image.npy")
        plane_data.mean_image = np.load(mean_image_path)
        plane_data.mean_image /= plane_data.nframes

        if runtime_data.configuration.main.nchannels > 1:
            mean_image_channel_2_path = plane_directory.joinpath("mean_image_channel_2.npy")
            plane_data.mean_image_channel_2 = np.load(mean_image_channel_2_path)
            plane_data.mean_image_channel_2 /= plane_data.nframes

        plane_runtime.save()

    # Returns the updated RuntimeData instance for the first plane.
    plane0_directory = Path(runtime_data.configuration.output.save_path).joinpath("suite2p", "plane0")

    return RuntimeData.from_yaml(file_path=plane0_directory.joinpath("runtime_data.yaml"))


def _initialize_destination_files(
    raw_files: list[_RawFile],
    runtime_data: RuntimeData,
    override_runtime_parameters: bool = True,
) -> list[RuntimeData]:
    """Creates the destination binary files and plane-specific RuntimeData instances for converting Thorlabs .raw
    files to suite2p binary format.

    Args:
        raw_files: The list of _RawFile instances to convert to suite2p binary format.
        runtime_data: A RuntimeData instance that stores the suite2p single-day configuration and runtime parameters.
        override_runtime_parameters: Determines whether to override the configuration parameters with the values
                                     extracted from the raw files. If True, overrides 'nplanes', 'nchannels', and 'fs'
                                     with values from the first raw file. Defaults to True.

    Returns:
        A list of RuntimeData instances, one for each plane.
    """
    # Loads the configuration data of all .RAW files to be processed
    configurations = [
        [
            file.channel,
            file.z_planes,
            file.height,
            file.width,
            file.frame_rate,
            file.physical_width,
            file.physical_height,
        ]
        for file in raw_files
    ]

    # Verifies that all _RawFile instances have the same attributes (the recording configuration is the same for
    # all input files)
    if any(configuration != configurations[0] for configuration in configurations):
        message = (
            "Unable to convert the input list of Thorlabs .raw files to Suite2P plane BinaryFiles. The recording "
            "configurations of the input .raw files do not match for at least two file instances, indicating that "
            "the files belong to separate recordings."
        )
        console.error(message=message, error=ValueError)

    # Queries the configuration from the first raw file.
    raw_file = raw_files[0]

    # If 'override_runtime_parameters' is set to True, configures runtime_data with the configuration values from the
    # first raw file.
    if override_runtime_parameters:
        runtime_data.configuration.main.nplanes = raw_file.z_planes
        if raw_file.channel > 1:
            runtime_data.configuration.main.nchannels = 2
        runtime_data.configuration.main.fs = raw_file.frame_rate

    # Queries the number of planes and channels from the configuration.
    plane_number = runtime_data.configuration.main.nplanes
    channel_number = runtime_data.configuration.main.nchannels

    # Initialize list to store plane-specific RuntimeData instances.
    plane_runtime_data_list = []

    # Loops over all available planes and iteratively sets up paths and creates initial files for each plane.
    for plane_index in range(plane_number):
        # Create a separate RuntimeData instance for this plane with deep copied configuration.
        plane_runtime_data = RuntimeData(configuration=copy.deepcopy(runtime_data.configuration))

        # Sets up the yaml_path for this plane and creates the directory hierarchy.
        plane_directory = set_yaml_path(plane_runtime_data, plane_index)

        # Gets reference to this plane's IOData.
        plane_io_data = plane_runtime_data.data.file_io

        # Creates file paths for the binary data file.
        plane_io_data.reg_file = plane_directory.joinpath("data.bin")
        plane_io_data.reg_file.touch()

        # If the data uses two functional channels, creates a second data file for the second channel.
        if channel_number > 1:
            plane_io_data.reg_file_channel_2 = plane_directory.joinpath("data_chan2.bin")
            plane_io_data.reg_file_channel_2.touch()

        plane_io_data.mean_image = np.zeros((raw_file.height, raw_file.width), np.float32)
        plane_io_data.nframes = 0

        # If the data uses two functional channels, initializes an array for the second channel's mean image.
        if channel_number > 1:
            plane_io_data.mean_image_channel_2 = np.zeros((raw_file.height, raw_file.width), np.float32)

        # Overrides the height and width properties with the dimensions of the processed recording.
        plane_io_data.height = raw_file.height
        plane_io_data.width = raw_file.width

        # If registration is disabled, sets the pixel ranges to span the full height and width of the frame.
        if not runtime_data.configuration.registration.do_registration:
            plane_io_data.height_range = [0, plane_io_data.height]
            plane_io_data.width_range = [0, plane_io_data.width]

        # Saves the plane-specific RuntimeData to the .yaml file in its respective directory.
        plane_runtime_data.save()
        plane_runtime_data_list.append(plane_runtime_data)

    return plane_runtime_data_list


def _single_raw_to_binary(plane_runtime_data_list: list[RuntimeData], raw_file: _RawFile) -> None:
    """Converts a single Thorlabs raw (.raw) file to suite2p binary (.bin) format for each plane and updates each
    plane's RuntimeData with the configuration data from the processed file.

    Args:
        plane_runtime_data_list: A list of all plane-specific RuntimeData instances that store single-day plane
                                 processing parameters.
        raw_file: The _RawFile object containing the raw data to convert to BinaryFile format.
    """
    # Extracts the batch size from the configuration (same across all planes)
    batch_size = int(plane_runtime_data_list[0].configuration.registration.batch_size)

    # Opens the raw file for reading in binary mode
    with raw_file.path.open(mode="rb") as file:
        # Calculates the appropriate chunk size based on data dimensions and batch size
        chunk_size = batch_size * raw_file.height * raw_file.width * raw_file.channel * raw_file.recorded_planes * 2

        # Reads the raw data in chunks. Loops until all frames from the target file are processed
        frame_chunk = file.read(chunk_size)
        while frame_chunk:
            # Converts the raw data chunk into a NumPy array
            frames = np.frombuffer(frame_chunk, dtype=np.int16)

            # Calculates the number of frames inside the chunk
            frame_number = int(len(frames) / raw_file.height / raw_file.width / raw_file.recorded_planes)

            reshaped_frames: NDArray[np.int16]  # Pre-assigns the variable type

            # If the data uses two functional channels, splits the data into two separate channels
            if raw_file.channel > 1:
                # Reshapes the data into (number of frames, height, width)
                # noinspection PyTypeChecker
                reshaped_frames = frames.reshape(
                    raw_file.recorded_planes * frame_number, raw_file.height, raw_file.width
                )

                # Separates the interleaved data into two channels (even indices for channel 1, odd indices for
                # channel 2)
                channel_1_frames = reshaped_frames[::2]
                channel_2_frames = reshaped_frames[1::2]

                # Reorganizes frames into two separate arrays for each plane
                reshaped_frames = np.array(
                    [
                        [
                            channel_1_frames[plane_index :: raw_file.recorded_planes],
                            channel_2_frames[plane_index :: raw_file.recorded_planes],
                        ]
                        for plane_index in range(raw_file.recorded_planes)
                    ]
                )

            # If there is only one channel, reshapes the data without processing channels
            else:
                # noinspection PyTypeChecker
                reshaped_frames = frames.reshape(
                    raw_file.recorded_planes, frame_number, raw_file.height, raw_file.width
                )

            # Loops over all available planes and iteratively writes the frames for each plane into the plane-specific
            # binary file(s)
            for z_plane_index in range(raw_file.z_planes):
                # Gets the RuntimeData for this specific plane
                plane_runtime_data = plane_runtime_data_list[z_plane_index]
                plane_io_data = plane_runtime_data.data.file_io

                # Extracts the set of frames to write to the current plane's binary file
                frames_to_write = reshaped_frames[z_plane_index]

                # If the processed data uses two functional channels, writes the frames to their respective channel's
                # memory-mapped binary file
                if raw_file.channel > 1:
                    # Opens the (functional) channel 1 memory-mapped binary file for writing (appending)
                    with Path(plane_io_data.reg_file).open(mode="ab") as channel_1_binary_file:
                        # Converts all frames to bytes and writes (appends) them to the (functional) channel 1
                        # memory-mapped binary file
                        channel_1_binary_file.write(frames_to_write[0].astype(np.int16).tobytes())

                    # Opens the (functional) channel 2 memory-mapped binary file for writing (appending)
                    with Path(plane_io_data.reg_file_channel_2).open(mode="ab") as channel_2_binary_file:
                        # Converts all frames to bytes and writes (appends) them to the (functional) channel 2
                        # memory-mapped binary file
                        channel_2_binary_file.write(frames_to_write[1].astype(np.int16).tobytes())

                    # Appends the data from all processed frames to the data arrays in the plane-specific IOData
                    plane_io_data.mean_image += frames_to_write[0].astype(np.float32).sum(axis=0)
                    plane_io_data.mean_image_channel_2 += frames_to_write[1].astype(np.float32).sum(axis=0)

                # If the processed data uses one functional channel, repeats the same steps above for only the first
                # channel
                else:
                    # Opens the (functional) channel 1 memory-mapped binary file for writing (appending)
                    with Path(plane_io_data.reg_file).open(mode="ab") as channel_1_binary_file:
                        # Converts all frames to bytes and writes (appends) them to the (functional) channel 1
                        # memory-mapped binary file
                        channel_1_binary_file.write(frames_to_write.astype(np.int16).tobytes())

                    # Appends the data from all processed frames to the mean image data array in the plane-specific
                    # IOData
                    plane_io_data.mean_image += frames_to_write.astype(np.float32).sum(axis=0)

            # Reads the next chunk of frames
            frame_chunk = file.read(chunk_size)

    # Loops over each plane's RuntimeData and adds descriptive information about the data to be processed (frames
    for plane_runtime_data in plane_runtime_data_list:
        plane_io_data = plane_runtime_data.data.file_io
        total_frames = int(
            raw_file.size / raw_file.height / raw_file.width / raw_file.recorded_planes / raw_file.channel / 2
        )
        plane_io_data.nframes += total_frames

    # Save each plane's updated RuntimeData to its respective yaml file.
    for plane_runtime_data in plane_runtime_data_list:
        plane_runtime_data.save()
