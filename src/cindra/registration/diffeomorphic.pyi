import numpy as np
from numpy.typing import NDArray as NDArray

from .pyramid import ScaleSpacePyramid as ScaleSpacePyramid
from .deformation import Deformation as Deformation
from .spline_grid import SplineGrid as SplineGrid

_MINIMUM_GRID_DIMENSION: int

class DiffeomorphicDemonsRegistration:
    _images: list[NDArray[np.float32]]
    _speed_factor: float
    _scale_sampling: int
    _grid_sampling_factor: float
    _final_scale: float
    _final_grid_sampling: float
    _smooth_scale: bool
    _injective: bool
    _freeze_edges: bool
    _deformation_limit: float
    _noise_factor: float
    _deformations: dict[int, Deformation]
    _pyramids: list[ScaleSpacePyramid] | None
    _cache: dict[str, tuple[tuple[int, int, float], Deformation | NDArray[np.float32]]]
    _interpolation_order: int
    def __init__(
        self,
        images: list[NDArray[np.float32]],
        speed_factor: float = 3.0,
        scale_sampling: int = 30,
        grid_sampling_factor: float = 1.0,
        final_scale: float = 1.0,
        final_grid_sampling: float = 16.0,
        *,
        smooth_scale: bool = True,
        injective: bool = True,
        freeze_edges: bool = True,
        deformation_limit: float = 1.0,
        noise_factor: float = 1.0,
    ) -> None: ...
    def get_deformation(self, image_index: int) -> Deformation: ...
    def register(self, *, progress: bool = True) -> None: ...
    def _perform_iteration(self, level: int, iteration: int, scale: float) -> None: ...
    def _compute_groupwise_deformation(
        self, image_index: int, iteration_key: tuple[int, int, float]
    ) -> Deformation | None: ...
    def _compute_pairwise_deformation(
        self, source_index: int, target_index: int, iteration_key: tuple[int, int, float]
    ) -> Deformation: ...
    def _get_image_and_gradient(
        self, image_index: int, iteration_key: tuple[int, int, float]
    ) -> tuple[NDArray[np.float32], tuple[NDArray[np.float32], NDArray[np.float32]]]: ...
    def _get_deformed_image(self, image_index: int, scale: float) -> NDArray[np.float32]: ...
    def _apply_incremental_deformation(self, image_index: int, incremental_deformation: Deformation | None) -> None: ...
    def _regularize_deformation(
        self, scale: float, deformation: Deformation, image_shape: tuple[int, ...] | None = None
    ) -> Deformation: ...
    def _compute_grid_sampling(self, scale: float) -> float: ...
    def _get_cached(
        self, key: str, iteration_key: tuple[int, int, float]
    ) -> Deformation | NDArray[np.float32] | None: ...
    def _set_cached(
        self, key: str, iteration_key: tuple[int, int, float], data: Deformation | NDArray[np.float32]
    ) -> None: ...
