import numpy as np
from numpy.typing import NDArray as NDArray

def compute_delta_fluorescence(
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    neuropil_coefficient: float,
    baseline_method: str,
    baseline_window: float,
    baseline_sigma: float,
    baseline_percentile: float,
    sampling_rate: float,
) -> NDArray[np.float32]: ...
def apply_oasis_deconvolution(
    cell_fluorescence: NDArray[np.float32], batch_size: int, time_constant: float, sampling_rate: float
) -> NDArray[np.float32]: ...
def _oasis_matrix(
    cell_fluorescence: NDArray[np.float32],
    pool_amplitude: NDArray[np.float32],
    pool_weight: NDArray[np.float32],
    pool_start_frame: NDArray[np.int32],
    pool_length: NDArray[np.float32],
    spike_trace: NDArray[np.float32],
    time_constant: float,
    sampling_rate: float,
) -> None: ...
