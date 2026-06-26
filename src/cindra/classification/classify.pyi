from pathlib import Path

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray

from ..dataclasses import ROIStatistics as ROIStatistics

_BUILTIN_CLASSIFIER_PATH: Path
_CLASSIFICATION_FEATURES: tuple[str, ...]
_PRECLASSIFICATION_FEATURES: tuple[str, ...]
_GRID_NODE_COUNT: int
_LOG_EPSILON: float

class Classifier:
    _classifier_path: Path
    _available_features: list[str]
    _training_features: dict[str, NDArray[np.float32]]
    _training_labels: NDArray[np.bool_]
    def __init__(self, classifier_path: Path, feature_names: tuple[str, ...] | None = None) -> None: ...
    def _extract_features(self, roi_statistics: list[ROIStatistics]) -> NDArray[np.float32]: ...
    def _get_training_features(self) -> NDArray[np.float32]: ...
    def _compute_log_probabilities(self, features: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def _predict_probabilities(self, roi_statistics: list[ROIStatistics]) -> NDArray[np.float32]: ...
    _probability_grid: Incomplete
    _grid_cell_probabilities: Incomplete
    _model: Incomplete
    def _fit_model(self) -> None: ...
    @staticmethod
    def create_training_dataset(
        file_path: Path,
        training_labels: NDArray[np.bool_],
        normalized_pixel_count: NDArray[np.float32],
        compactness: NDArray[np.float32],
        skewness: NDArray[np.float32],
    ) -> None: ...
    def classify(
        self, roi_statistics: list[ROIStatistics], probability_threshold: float = 0.5
    ) -> NDArray[np.float32]: ...

def classify(
    roi_statistics: list[ROIStatistics],
    classification_threshold: float = 0.5,
    custom_classifier_path: Path | None = None,
    *,
    preclassification: bool = False,
) -> NDArray[np.float32]: ...
