import numpy as np
from numpy.typing import NDArray as NDArray

from .utils import (
    compute_spatial_taper_mask as compute_spatial_taper_mask,
    compute_registration_blocks as compute_registration_blocks,
)

def pca_denoise(
    frames: NDArray[np.float32], block_size: tuple[int, int], component_fraction: float, parallel_workers: int = 1
) -> None: ...
def _fit_and_reconstruct_block(block: NDArray[np.float32], num_components: int) -> NDArray[np.float32]: ...
