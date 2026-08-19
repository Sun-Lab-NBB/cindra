import numpy as np
from numpy.typing import NDArray as NDArray

from .utils import NORMALIZATION_EPSILON as NORMALIZATION_EPSILON

def compute_bidirectional_phase_offset(frames: NDArray[np.float32], workers: int) -> int: ...
def apply_bidirectional_phase_correction(frames: NDArray[np.float32], bidirectional_phase_offset: int) -> None: ...
