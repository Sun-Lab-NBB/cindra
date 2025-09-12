"""This module provides tools for reading and writing image data stored in Thorlabs raw (.raw) files and their
associated XML (.xml) configuration files.
"""

from typing import TYPE_CHECKING, Any
from pathlib import Path

from tqdm import tqdm
import numpy as np
from ataraxis_time import PrecisionTimer
from quick_xmltodict import parse
from ataraxis_base_utilities import console, ensure_directory_exists

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


def raw_to_binary(plane_ops_dictionary: dict[str, Any], override_ops_parameters: bool = True) -> dict[str, Any]:
    """Reads the input data stored as Thorlabs .raw files and converts it to the suite2p plane binary (.bin) file(s).

    Args:
        plane_ops_dictionary: The dictionary that stores the suite2p processing parameters.
        override_ops_parameters: Determines whether to override certain configuration parameters, such as the number of
            planes and channels, from 'ops' with data loaded from the .xml configuration files stored together with
            Thorlabs .raw files.

    Returns:
        The 'ops' dictionary of the first available plane to be processed augmented with additional descriptive
        parameters for the processed data. Specifically, the dictionary includes the following additional keys:
        "Ly", "Lx", "nframes", "meanImg", "meanImg_chan2".
    """
    # Instantiates and resets the run timer
    timer = PrecisionTimer("s")
    timer.reset()

    # Loads Thorlabs .raw files from the paths provided in ops["data_path"] and converts them into _RawFile instances.
    raw_files = [_RawFile(path) for path in plane_ops_dictionary["data_path"]]

    # Initializes the destination files and resolves paths and configuration for further .raw to .bin file conversion.
    ops_paths = _initialize_destination_files(
        ops=plane_ops_dictionary, raw_files=raw_files, override_ops_parameters=override_ops_parameters
    )

    # Determines the number of frames across all .raw files. This is used for the progress bar visualization.
    total_frames = sum(raw_file.frame_number for raw_file in raw_files)

    # Creates the progress bar.
    progress_bar = tqdm(
        total=total_frames,
        desc="Converting Thorlabs raw frames to binary",
        unit="frames",
        disable=not plane_ops_dictionary["progress_bars"],
    )

    # Converts all the .raw files into .bin format.
    for raw_file in raw_files:
        # Loads plane-specific ops files generated above
        plane_ops: list[dict[str, Any]] = [np.load(ops_index, allow_pickle=True)[()] for ops_index in ops_paths]

        # Performs the raw to binary conversion using the loaded plane-specific 'ops' dictionaries and the target raw
        # file.
        _single_raw_to_binary(plane_ops=plane_ops, raw_file=raw_file)

        # Updates the progress bar with the number of frames processed in the target file.
        progress_bar.update(raw_file.frame_number)

    # Closes the progress bar when the binary conversion is over.
    progress_bar.close()

    # Reloads the updated ops.npy files after conversion.
    plane_ops = [np.load(ops_index, allow_pickle=True)[()] for ops_index in ops_paths]

    # Creates a mean image based on the final number of frames.
    for plane_ops_dict in plane_ops:
        plane_ops_dict["meanImg"] /= plane_ops_dict["nframes"]
        np.save(plane_ops_dict["ops_path"], plane_ops_dict)

    # Returns the updated 'ops' dictionary for the first plane.
    return plane_ops[0]


def _initialize_destination_files(
    ops: dict[str, Any], raw_files: list[_RawFile], override_ops_parameters: bool = True
) -> list[Path]:
    """Prepares the environment for Thorlabs .raw to Suite2P binary (.bin) file conversion by setting up directories
    and generating the necessary metadata files.

    Args:
        ops: The dictionary that stores the suite2p processing parameters.
        raw_files: A list of the _RawFile instances, one for each Thorlabs .raw file to be processed.
        override_ops_parameters: Determines whether to override certain configuration parameters, such as the number of
            planes and channels, from 'ops' with data loaded from the .xml configuration files stored together with
            Thorlabs .raw files.

    Returns:
        A list of absolute paths to the generated ops.npy files, one for each plane to be processed with the single-day
        suite2p pipeline.

    Raises:
        ValueError: If the recording configuration used by all input raw files does not match.
    """
    # Loads the configuration data of all .RAW files to be processed.
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

    # If 'override_ops_parameters' is set to True, configures 'ops' with the configuration values from the first raw
    # file.
    if override_ops_parameters:
        ops["nplanes"] = raw_file.z_planes
        if raw_file.channel > 1:
            ops["nchannels"] = 2
        ops["fs"] = raw_file.frame_rate

    # Queries the number of planes and channels from the 'ops' dictionary. This is especially relevant if the function
    # is configured to use original 'ops' parameters instead of loading them from the processed .raw file configuration.
    plane_number = ops["nplanes"]
    channel_number = ops["nchannels"]

    # Initializes a list to store the absolute paths to the generated plane ops.npy files.
    ops_paths = []

    # Initializes a flag to limit certain processing steps to the first plane
    first_plane = True

    # Loops over all available planes and iteratively sets up paths and writes the frames for each plane into the
    # plane-specific binary file(s).
    for plane_index in range(plane_number):
        # Constructs the directory path for each plane's output directory.
        ops["save_path"] = ops["save_path0"].joinpath("suite2p", f"plane{plane_index}")

        # If it's the first plane, sets the 'fast_disk' path to 'save_path'.
        if first_plane:
            ops["fast_disk"] = ops["save_path"]
            first_plane = False

        # For all other planes, joins the 'fast_disk' path with the appropriate plane-specific subdirectory.
        else:
            ops["fast_disk"] = ops["fast_disk"].joinpath("suite2p", f"plane{plane_index}")

        # Creates file paths for ops.npy and the binary data file.
        ops["ops_path"] = ops["save_path"].joinpath("ops.npy")
        ops["reg_file"] = ops["fast_disk"].joinpath("data.bin")

        # Creates directories if they do not exist.
        ensure_directory_exists(ops["fast_disk"])
        ensure_directory_exists(ops["save_path"])

        # Creates the binary file to store the data for the first channel.
        ops["reg_file"].touch()

        # If the data uses two functional channels, creates a second data file for the second channel.
        if channel_number > 1:
            ops["reg_file_chan2"] = ops["fast_disk"].joinpath("data_chan2.bin")
            ops["reg_file_chan2"].touch()

        # Initializes arrays for the mean image and the frame data.
        ops["meanImg"] = np.zeros((raw_file.height, raw_file.width), np.float32)
        ops["nframes"] = 0

        # If the data uses two functional channels, initializes an array for the second channel's mean image.
        if channel_number > 1:
            ops["meanImg_chan2"] = np.zeros((raw_file.height, raw_file.width), np.float32)

        # Overrides the height and width properties of the 'ops' dictionary with the dimensions of the processed
        # recording.
        ops["Ly"] = raw_file.height
        ops["Lx"] = raw_file.width

        # Determines whether the current runtime is configured to perform motion registration.
        do_registration = ops["do_registration"]

        # If registration is disabled, sets the pixel ranges to span the full height and width of the frame. Pixels on
        # the edges of each frame are excluded during registration as they are typically unstable and should be
        # discarded anyway.
        if not do_registration:
            ops["yrange"] = np.array([0, ops["Ly"]])
            ops["xrange"] = np.array([0, ops["Lx"]])

        # Appends the absolute file path to the ops.npy file for the current plane to the 'ops_path' data array.
        ops_paths.append(ops["ops_path"])

        # Caches each 'ops' dictionary to disk as an ops.npy file. The file is cached into the plane-specific processing
        # subdirectory.
        np.save(ops["ops_path"], ops)

    # Returns the list of absolute file paths to the generated ops.npy files.
    return ops_paths


def _single_raw_to_binary(plane_ops: list[dict[str, Any]], raw_file: _RawFile) -> None:
    """Converts a single Thorlabs raw (.raw) file to suite2p binary (.bin) format for each plane and updates the
    'ops' dictionary for each plane with the configuration data from the processed file.

    Args:
        plane_ops: A list of dictionaries containing suite2p processing parameters for each recording plane.
        raw_file: The _RawFile object containing the raw data to convert to BinaryFile format.
    """
    # Extracts the batch size from the first 'ops' dictionary.
    batch_size = int(plane_ops[0]["batch_size"])

    # Opens the raw file for reading in binary mode.
    with raw_file.path.open(mode="rb") as file:
        # Calculates the appropriate chunk size based on data dimensions and batch size.
        chunk_size = batch_size * raw_file.height * raw_file.width * raw_file.channel * raw_file.recorded_planes * 2

        # Reads the raw data in chunks. Loops until all frames from the target file are processed.
        frame_chunk = file.read(chunk_size)
        while frame_chunk:
            # Converts the raw data chunk into a NumPy array.
            frames = np.frombuffer(frame_chunk, dtype=np.int16)

            # Calculates the number of frames inside the chunk.
            frame_number = int(len(frames) / raw_file.height / raw_file.width / raw_file.recorded_planes)

            reshaped_frames: NDArray[np.int16]  # Pre-assigns the variable type

            # If the data uses two functional channels, splits the data into two separate channels.
            if raw_file.channel > 1:
                # Reshapes the data into (number of frames, height, width).
                # noinspection PyTypeChecker
                reshaped_frames = frames.reshape(
                    raw_file.recorded_planes * frame_number, raw_file.height, raw_file.width
                )

                # Separates the interleaved data into two channels (even indices for channel 1, odd indices for channel
                # 2).
                channel_1_frames = reshaped_frames[::2]
                channel_2_frames = reshaped_frames[1::2]

                # Reorganizes frames into two separate arrays for each plane.
                reshaped_frames = np.array(
                    [
                        channel_1_frames[plane_index :: raw_file.recorded_planes],
                        channel_2_frames[plane_index :: raw_file.recorded_planes],
                    ]
                    for plane_index in range(raw_file.recorded_planes)
                )

            # If there is only one channel, reshapes the data without processing channels.
            else:
                # noinspection PyTypeChecker
                reshaped_frames = frames.reshape(
                    raw_file.recorded_planes, frame_number, raw_file.height, raw_file.width
                )

            # Loops over all available planes and iteratively writes the frames for each plane into the plane-specific
            # binary file(s).
            for z_plane_index in range(raw_file.z_planes):
                # Extracts the 'ops' dictionary for the current plane.
                ops = plane_ops[z_plane_index]

                # Extracts the set of frames to write to the current plane's binary file.
                frames_to_write = reshaped_frames[z_plane_index]

                # If the processed data uses two functional channels, writes the frames to their respective channel's
                # memory-mapped binary file.
                if raw_file.channel > 1:
                    # Opens the (functional) channel 1 memory-mapped binary file for writing (appending).
                    with Path(ops["reg_file"]).open(mode="ab") as channel_1_binary_file:
                        # Converts all frames to bytes and writes (appends) them to the (functional) channel 1
                        # memory-mapped binary file.
                        channel_1_binary_file.write(frames_to_write[0].astype(np.int16).tobytes())

                    # Opens the (functional) channel 2 memory-mapped binary file for writing (appending).
                    with Path(ops["reg_file_chan2"]).open(mode="ab") as channel_2_binary_file:
                        # Converts all frames to bytes and writes (appends) them to the (functional) channel 2
                        # memory-mapped binary file.
                        channel_2_binary_file.write(frames_to_write[1].astype(np.int16).tobytes())

                    # Appends the data from all processed frames to the data arrays in the plane-specific 'ops'
                    # dictionary, as this data is used during further processing.
                    ops["meanImg"] += frames_to_write[0].astype(np.float32).sum(axis=0)
                    ops["meanImg_chan2"] += frames_to_write[1].astype(np.float32).sum(axis=0)

                # If the processed data uses one functional channel, repeats the same steps above for only the first
                # channel.
                else:
                    # Opens the (functional) channel 1 memory-mapped binary file for writing (appending).
                    with Path(ops["reg_file"]).open(mode="ab") as channel_1_binary_file:
                        # Converts all frames to bytes and writes (appends) them to the (functional) channel 1
                        # memory-mapped binary file.
                        channel_1_binary_file.write(frames_to_write.astype(np.int16).tobytes())

                    # Appends the data from all processed frames to the mean image data array in the plane-specific
                    # 'ops' dictionary,
                    ops["meanImg"] += frames_to_write.astype(np.float32).sum(axis=0)

            # Reads the next chunk of frames.
            frame_chunk = file.read(chunk_size)

    # Loops over each loaded 'ops' dictionary and adds descriptive information about the data to be processed (frames).
    for ops in plane_ops:
        total_frames = int(
            raw_file.size / raw_file.height / raw_file.width / raw_file.recorded_planes / raw_file.channel / 2
        )
        ops["nframes"] += total_frames

        # Caches each 'ops' dictionary to disk as an ops.npy file. The file is cached into the plane-specific processing
        # subdirectory.
        np.save(ops["ops_path"], ops)
