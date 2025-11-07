"""Contains tests for methods provided by the save.py module."""

import scipy
import numpy as np
from sl_suite2p.io.save import compute_dydx, combined, save_matlab


def test_compute_dydx():
    """Verfies that the correct list of y-axis and x-axis displacement values for each plane are returned."""
    # Checks that the "dy" and "dx" values are used directly when specified in ops.
    plane_ops_displacement = [
        {"Ly": 256, "Lx": 256, "dx": 0, "dy": 0},
        {"Ly": 256, "Lx": 256, "dx": 300, "dy": 0},
        {"Ly": 256, "Lx": 256, "dx": 0, "dy": 300},
    ]
    y_displacement, x_displacement = compute_dydx(plane_ops=plane_ops_displacement)

    expected_x_displacement = np.array([0, 300, 0])
    expected_y_displacement = np.array([0, 0, 300])

    assert (x_displacement == expected_x_displacement).all()
    assert (y_displacement == expected_y_displacement).all()

    # Computes "dy" and "dx" based on the dimensions if they are not provided in the first 'ops' dictionary.
    plane_ops_dim = [
        {"Ly": 200, "Lx": 200},
        {"Ly": 200, "Lx": 200},
        {"Ly": 200, "Lx": 200},
    ]
    y_displacement, x_displacement = compute_dydx(plane_ops=plane_ops_dim)

    expected_x_displacement = np.array([0, 200, 0])
    expected_y_displacement = np.array([0, 0, 200])

    assert (x_displacement == expected_x_displacement).all()
    assert (y_displacement == expected_y_displacement).all()


def test_compute_dydx_roi_recalculation():
    """Verifies that compute_dydx the y-displacement is recalculated while preserving the
    x-displacement when the number of ROIs is less than the number of planes."""

    plane_ops = [
        {"Ly": 100, "Lx": 300, "dx": 0, "dy": 0},
        {"Ly": 100, "Lx": 300, "dx": 0, "dy": 0},
        {"Ly": 100, "Lx": 300, "dx": 300, "dy": 0},
        {"Ly": 100, "Lx": 300, "dx": 300, "dy": 0},
    ]

    y_disp, x_disp = compute_dydx(plane_ops=plane_ops)

    expected_x = np.array([0, 0, 300, 300])
    expected_y = np.array([0, 0, 100, 100])

    assert (x_disp == expected_x).all()
    assert (y_disp == expected_y).all()


def test_combined_multiple_planes(tmp_path):
    """Verifies that the input data are stored in the correct format when saved as a .mat file."""
    save_directory = tmp_path.joinpath("save_dir")
    save_directory.mkdir()

    # Creates two plane directories
    for plane_idx in range(2):
        plane_dir = save_directory.joinpath(f"plane{plane_idx}")
        plane_dir.mkdir()

        ops = {
            "Ly": 128,
            "Lx": 128,
            "nchannels": 1,
            "nframes": 500,
            "mean_image": np.random.rand(128, 128).astype(np.float32),
            "enhanced_mean_image": np.random.rand(128, 128).astype(np.float32),
            "Vcorr": np.random.rand(128, 128).astype(np.float32),
            "xrange": [0, 128],
            "yrange": [0, 128],
        }
        np.save(plane_dir.joinpath("ops.npy"), ops)

        stat = np.array(
            [
                {"xpix": np.array([10, 11]), "ypix": np.array([20, 21]), "med": [25, 15], "iplane": plane_idx},
            ]
        )
        np.save(plane_dir.joinpath("stat.npy"), stat)
        np.save(plane_dir.joinpath("F.npy"), np.random.rand(1, 500).astype(np.float32))
        np.save(plane_dir.joinpath("Fneu.npy"), np.random.rand(1, 500).astype(np.float32))
        np.save(plane_dir.joinpath("Fsub.npy"), np.random.rand(1, 500).astype(np.float32))
        np.save(plane_dir.joinpath("spks.npy"), np.random.rand(1, 500).astype(np.float32))
        np.save(plane_dir.joinpath("iscell.npy"), np.array([[True, 0.85]]))

    roi_statistics, ops, cell_fluorescence, _, _, _, _, _, _, _, has_red = combined(save_directory, save=False)
    assert ops["Lx"] == 256

    # Checks that the ROIs are combined
    assert cell_fluorescence.shape[0] == 2
    assert has_red is False

    # Verifies that the corresponding output files are saved to the save_directory when save=True
    combined(save_directory, save=True)
    assert save_directory.joinpath("combined").joinpath("F.npy").exists()


def test_save_matlab(tmp_path):
    """Verifies that the input data are stored in the correct format when saved as a .mat file."""
    save_path = tmp_path.joinpath("output")
    save_path.mkdir()
    data_path = tmp_path.joinpath("input_data")

    ops = {
        "save_path": save_path,
        "data_path": [data_path],
    }

    # Creates a basic ROI statistics dictionary
    roi_statistics = [{"ypix": np.array([1, 2]), "xpix": np.array([3, 4])}]
    cell_fluorescence = np.random.rand(1, 10).astype(np.float32)
    neuropil_fluorescence = np.random.rand(1, 10).astype(np.float32)
    spikes = np.random.rand(1, 10).astype(np.float32)
    is_cell = np.array([[1, 0.8]])
    red_cell = np.array([[0, 0.1]])

    save_matlab(
        ops=ops,
        roi_statistics=roi_statistics,
        cell_fluorescence=cell_fluorescence,
        neuropil_fluorescence=neuropil_fluorescence,
        spikes=spikes,
        is_cell=is_cell,
        red_cell=red_cell,
    )

    output_file = save_path.joinpath("Fall.mat")
    mat_data = scipy.io.loadmat(output_file)

    # Checks that roi_statistics a numpy array
    assert "stat" in mat_data

    # Checks that path objects are converted to strings
    ops_data = mat_data["ops"][0, 0]
    assert isinstance(ops_data["save_path"][0], str)
    assert all(isinstance(item[0], str) for item in ops_data["data_path"][0])
