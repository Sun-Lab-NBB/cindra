import numpy as np
from numpy.typing import NDArray as NDArray

from .deformation import (
    zoom as zoom,
    diffuse as diffuse,
)

_DOWNSAMPLE_ZOOM_THRESHOLD: float
_MINIMUM_DOWNSAMPLE_DIMENSION: int

class ScaleSpacePyramid:
    _LEVEL_FACTOR: float
    _levels: list[NDArray[np.float32]]
    _level_scales: list[float]
    def __init__(self, data: NDArray[np.float32], min_scale: float) -> None: ...
    def _initialize_base_level(self, data: NDArray[np.float32], min_scale: float) -> None: ...
    def get_scale(self, scale: float) -> NDArray[np.float32]: ...
    def _add_level(self) -> None: ...
