"""Contains tests for methods provided by the save.py module."""

import scipy
import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
from sl_suite2p.io.save import compute_dydx, combined, save_matlab


@pytest.mark.parametrize(
    "plane_ops, expected_x, expected_y",
    [
        # dx/dy are provided in ops
        (
            [
                {"Ly": 256, "Lx": 256, "dx": 0, "dy": 0},
                {"Ly": 256, "Lx": 256, "dx": 300, "dy": 0},
                {"Ly": 256, "Lx": 256, "dx": 0, "dy": 300},
            ],
            np.array([0, 300, 0]),
            np.array([0, 0, 300]),
        ),
        # dx/dy are computed from the image's dimensions
        (
            [{"Ly": 200, "Lx": 200}] * 3,
            np.array([0, 200, 0]),
            np.array([0, 0, 200]),
        ),
    ],
)
def test_compute_dydx(plane_ops, expected_x, expected_y):
    """Verfies that the correct list of y-axis and x-axis displacement values for each plane are returned."""
    y_displacement, x_displacement = compute_dydx(plane_ops)

    assert (x_displacement == expected_x).all()
    assert (y_displacement == expected_y).all()


@pytest.mark.parametrize(
    "plane_ops, expected_x, expected_y",
    [
        # y recalculation when number of ROIs < number of planes
        (
            [
                {"Ly": 100, "Lx": 300, "dx": 0, "dy": 0},
                {"Ly": 100, "Lx": 300, "dx": 0, "dy": 0},
                {"Ly": 100, "Lx": 300, "dx": 300, "dy": 0},
                {"Ly": 100, "Lx": 300, "dx": 300, "dy": 0},
            ],
            np.array([0, 0, 300, 300]),
            np.array([0, 0, 100, 100]),
        ),
    ],
)
def test_compute_dydx_roi_recalculation(plane_ops, expected_x, expected_y):
    """Verifies that compute_dydx the y-displacement is recalculated while preserving the
    x-displacement when the number of ROIs is less than the number of planes."""
    y_displacement, x_displacement = compute_dydx(plane_ops)

    assert (x_displacement == expected_x).all()
    assert (y_displacement == expected_y).all()


@pytest.mark.parametrize(
    "plane_idx, nchannels, has_chan2, include_max_proj, frame_count, has_redcell, save_mat, has_stat, frame_count_mismatch",
    [
        (0, 2, True, False, 400, True, True, True, False),
        (1, 1, False, True, 500, False, False, True, True),
        (2, 2, True, True, 300, True, True, False, False),
    ],
)
def test_combined_multiple_planes(
    tmp_path,
    plane_idx,
    nchannels,
    has_chan2,
    include_max_proj,
    frame_count,
    has_redcell,
    save_mat,
    has_stat,
    frame_count_mismatch,
):
    """Verifies that the input data are stored in the correct format when saved as a .mat file."""
    save_directory = tmp_path.joinpath("save_dir")
    save_directory.mkdir()

    max_frame = 0
    plane_frame_counts = []

    # Creates two plane directories with multi-channel data and projections
    for plane_index in range(2):
        plane_dir = save_directory.joinpath(f"plane{plane_index}")
        plane_dir.mkdir()

        # Creates frame count mismatch to test padding
        current_frame_count = frame_count
        if frame_count_mismatch and plane_index == 1:
            current_frame_count = frame_count - 100

        plane_frame_counts.append(current_frame_count)
        max_frame = max(max_frame, current_frame_count)

        ops = {
            "Ly": 128,
            "Lx": 128,
            "nchannels": 2,
            "nframes": current_frame_count,
            "mean_image": np.random.rand(128, 128).astype(np.float32),
            "enhanced_mean_image": np.random.rand(128, 128).astype(np.float32),
            "Vcorr": np.random.rand(128, 128).astype(np.float32),
            "xrange": [0, 128],
            "yrange": [0, 128],
            "save_mat": save_mat,
        }

        if has_chan2 and plane_index == 0:
            ops["meanImg_chan2"] = np.random.rand(128, 128).astype(np.float32)
            ops["meanImg_chan2_corrected"] = np.random.rand(128, 128).astype(np.float32)
            ops["mean_image_channel_2"] = np.random.rand(128, 128).astype(np.float32)

        if include_max_proj and plane_index == 1:
            ops["max_proj"] = np.random.rand(128, 128).astype(np.float32)

        np.save(plane_dir.joinpath("ops.npy"), ops)

        if has_stat or plane_index == 0:
            stat = np.array(
                [
                    {"xpix": np.array([10, 11]), "ypix": np.array([20, 21]), "med": [25, 15], "iplane": plane_index},
                ]
            )
            np.save(plane_dir.joinpath("stat.npy"), stat)
            np.save(plane_dir.joinpath("F.npy"), np.random.rand(1, current_frame_count).astype(np.float32))
            np.save(plane_dir.joinpath("Fneu.npy"), np.random.rand(1, current_frame_count).astype(np.float32))
            np.save(plane_dir.joinpath("Fsub.npy"), np.random.rand(1, current_frame_count).astype(np.float32))
            np.save(plane_dir.joinpath("spks.npy"), np.random.rand(1, current_frame_count).astype(np.float32))
            np.save(plane_dir.joinpath("iscell.npy"), np.array([[True, 0.85]]))

            if has_redcell:
                np.save(plane_dir.joinpath("redcell.npy"), np.array([[plane_index, 0.1]]))

    roi_statistics, ops, cell_fluorescence, *_, has_red = combined(save_directory, save=False)

    if frame_count_mismatch:
        assert cell_fluorescence.shape[1] == max_frame
    else:
        assert cell_fluorescence.shape[1] == frame_count

    assert has_red == has_redcell

    # Verifies that multi-channel and projection data are added to ops
    for key in ["max_proj", "mean_image_channel_2"]:
        if key in ops:
            assert ops[key] is not None

    # Verifies that the corresponding output files are added to the save_directory when save=True
    combined(save_directory, save=True)
    assert save_directory.joinpath("combined").joinpath("F.npy").exists()

    # Checks that the redcell.npy exists if has_redcell is true
    if has_redcell:
        assert save_directory.joinpath("combined").joinpath("redcell.npy").exists()

    # Checks that the .mat file exists if save_mat is true
    if save_mat:
        assert save_directory.joinpath("combined").joinpath("Fall.mat").exists()


def test_save_matlab(tmp_path):
    """Verifies matlab saving functionality for multiple channels."""
    save_path = tmp_path.joinpath("output")
    save_path.mkdir()
    data_path = tmp_path.joinpath("input_data")

    ops = {
        "save_path": save_path,
        "data_path": [data_path],
    }

    roi_statistics = [{"ypix": np.array([1, 2]), "xpix": np.array([3, 4])}]
    cell_fluorescence = np.random.rand(1, 10).astype(np.float32)
    neuropil_fluorescence = np.random.rand(1, 10).astype(np.float32)
    spikes = np.random.rand(1, 10).astype(np.float32)
    is_cell = np.array([[1, 0.8]])
    red_cell = np.array([[0, 0.1]])
    cell_fluorescence_channel_2 = np.random.rand(1, 10).astype(np.float32)
    neuropil_fluorescence_channel_2 = np.random.rand(1, 10).astype(np.float32)

    save_matlab(
        ops=ops,
        roi_statistics=roi_statistics,
        cell_fluorescence=cell_fluorescence,
        neuropil_fluorescence=neuropil_fluorescence,
        spikes=spikes,
        is_cell=is_cell,
        red_cell=red_cell,
        cell_fluorescence_channel_2=cell_fluorescence_channel_2,
        neuropil_fluorescence_channel_2=neuropil_fluorescence_channel_2,
    )

    output_file = save_path.joinpath("Fall.mat")
    assert output_file.exists()

    mat_data = scipy.io.loadmat(output_file)

    # Checks that path objects are converted to strings
    ops_data = mat_data["ops"][0, 0]
    assert isinstance(ops_data["save_path"][0], str)
    assert all(isinstance(item[0], str) for item in ops_data["data_path"][0])

    # Checks that the second channel data is added to 'mat_dictionary'
    assert "F_chan2" in mat_data
    assert "Fneu_chan2" in mat_data
