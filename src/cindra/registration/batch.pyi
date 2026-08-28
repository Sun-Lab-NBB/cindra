from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type RegistrationBlocks = tuple[
    list[NDArray[np.int32]], list[NDArray[np.int32]], tuple[int, int], tuple[int, int], NDArray[np.float32]
]

@dataclass(frozen=True, slots=True)
class ReferenceData:
    taper_mask: NDArray[np.float32]
    mean_offset: NDArray[np.float32]
    reference_kernel: NDArray[np.complex64]
    taper_mask_nonrigid: NDArray[np.float32] | None
    mean_offset_nonrigid: NDArray[np.float32] | None
    reference_kernel_nonrigid: NDArray[np.complex64] | None
    blocks: RegistrationBlocks | None

@dataclass(frozen=True, slots=True)
class BatchRegistrationResult:
    frames: NDArray[np.int16] | NDArray[np.float32]
    y_offsets: NDArray[np.int32]
    x_offsets: NDArray[np.int32]
    correlations: NDArray[np.float32]
    y_offsets_nonrigid: NDArray[np.float32] | None
    x_offsets_nonrigid: NDArray[np.float32] | None
    correlations_nonrigid: NDArray[np.float32] | None
    frame_sum: NDArray[np.float32] | None = ...
