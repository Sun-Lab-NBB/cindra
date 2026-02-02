"""Provides ROI classification functionality for distinguishing cells from artifacts."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from ataraxis_base_utilities import console

from ..configuration import ROIStatistics, RuntimeContext

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Path to the built-in classifier bundled with sl-suite2p.
_BUILTIN_CLASSIFIER_PATH: Path = Path(__file__).parent / "classifier.npz"

# The names of the ROI features used for classification, in the order they appear in the feature matrix.
_FEATURE_NAMES: tuple[str, ...] = ("normalized_pixel_count", "compactness", "skewness")

# The number of grid nodes used for probability estimation during model fitting.
_GRID_NODE_COUNT: int = 100

# Small epsilon value added to probabilities to prevent log(0) errors.
_LOG_EPSILON: float = 1e-6


def _resolve_classifier_path(custom_classifier_path: Path | None = None) -> Path:
    """Resolves the classifier file path based on configuration settings.

    Args:
        custom_classifier_path: An optional path to a custom classifier file. If provided, this path is returned.
            Otherwise, the path to the built-in classifier filed bundled with the sl-suite2p release is returned.

    Returns:
        The resolved path to the classifier .npz file.
    """
    if custom_classifier_path is not None:
        return custom_classifier_path
    return _BUILTIN_CLASSIFIER_PATH.resolve()


class Classifier:
    """Provides logistic regression-based classification for identifying cell ROIs.

    This class loads classifier training data from .npz files, fits a logistic regression model, and uses it to
    predict whether detected ROIs represent real cells or artifacts based on their morphological features.

    Args:
        classifier_path: The path to a classifier .npz file containing training_labels and feature arrays.

    Notes:
        The classifier file format uses pickle-free npz serialization containing training_labels and feature arrays
        (normalized_pixel_count, compactness, skewness). The model is fitted on load, which takes approximately 10ms
        for the default training set.

    Attributes:
        _classifier_path: The path to the loaded classifier file.
        _available_features: The list of feature names available in the loaded classifier.
        _training_features: A dictionary mapping feature names to their training value arrays.
        _training_labels: The training labels array with shape (n_samples,).
        _probability_grid: The grid boundaries computed from sorted training statistics with shape
            (n_nodes, n_features). Used to map input feature values to grid intervals for probability lookup
            during classification.
        _grid_cell_probabilities: The Gaussian-smoothed probability that an ROI is a cell for each grid interval
            with shape (n_nodes - 1, n_features). Used to compute log probability ratios that serve as input
            features for the logistic regression model.
        _model: The fitted LogisticRegression model.
    """

    def __init__(self, classifier_path: Path) -> None:
        self._classifier_path: Path = classifier_path
        self._available_features: list[str] = []
        self._training_features: dict[str, "NDArray[np.float32]"] = {}
        self._training_labels: "NDArray[np.float32] | None" = None
        self._probability_grid: "NDArray[np.float32] | None" = None
        self._grid_cell_probabilities: "NDArray[np.float32] | None" = None
        self._model: LogisticRegression | None = None

        self._load(classifier_path=classifier_path)

    def _load(self, classifier_path: Path) -> None:
        """Loads training data from a .npz file and fits the classification model.

        Args:
            classifier_path: The path to the classifier .npz file containing training_labels and feature arrays.

        Raises:
            FileNotFoundError: If the classifier file does not exist at the specified path.
            ValueError: If the classifier file is missing required data or has an invalid format.
        """
        if not classifier_path.exists():
            message = (
                f"Unable to load the classification training data. The classifier file does not exist at the specified "
                f"path: {classifier_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        try:
            data = np.load(classifier_path, allow_pickle=False)
            self._classifier_path = classifier_path

            if "training_labels" not in data:
                message = f"Unable to load classifier. The file is missing training_labels: {classifier_path}."
                console.error(message=message, error=ValueError)

            self._training_labels = data["training_labels"].astype(np.float32)
            n_samples = len(self._training_labels)

            self._training_features = {}
            self._available_features = []

            for feature_name in _FEATURE_NAMES:
                if feature_name in data:
                    feature_array = data[feature_name].astype(np.float32)
                    if len(feature_array) == n_samples and not np.all(np.isnan(feature_array)):
                        self._training_features[feature_name] = feature_array
                        self._available_features.append(feature_name)

            if not self._available_features:
                message = f"Unable to load classifier. No valid features found in {classifier_path}."
                console.error(message=message, error=ValueError)

            self._fit_model()

        except (ValueError, KeyError, TypeError) as exception:
            message = (
                f"Unable to load classifier. The classifier file at {classifier_path} is corrupted or has an "
                f"invalid format. Original error: {exception}."
            )
            console.error(message=message, error=ValueError)

    def save(self, file_path: Path) -> None:
        """Saves the classifier training data to a .npz file.

        Args:
            file_path: The path where the classifier file will be saved. Should have .npz extension.

        Raises:
            ValueError: If no classifier has been loaded or the classifier has no training data.
        """
        if self._training_labels is None or not self._training_features:
            message = "Unable to save classifier. No classifier has been loaded or the classifier has no training data."
            console.error(message=message, error=ValueError)

        save_dict: dict[str, "NDArray[np.float32]"] = {"training_labels": self._training_labels}
        save_dict.update(self._training_features)

        np.savez(file_path, **save_dict)

    def classify(
        self,
        roi_statistics: list[ROIStatistics],
        probability_threshold: float = 0.5,
    ) -> "NDArray[np.float32]":
        """Classifies ROIs as cells or non-cells based on their morphological features.

        Args:
            roi_statistics: The list of ROIStatistics instances to classify.
            probability_threshold: The probability threshold above which an ROI is classified as a cell.
                Defaults to 0.5.

        Returns:
            An array of shape (n_rois, 2) where each row contains [is_cell, probability]. The is_cell value is 1.0
            if the ROI is classified as a cell (probability > threshold) and 0.0 otherwise.
        """
        probabilities = self._predict_probabilities(roi_statistics=roi_statistics)
        is_cell = (probabilities > probability_threshold).astype(np.float32)

        return np.stack([is_cell, probabilities], axis=1).astype(np.float32)

    def _predict_probabilities(self, roi_statistics: list[ROIStatistics]) -> "NDArray[np.float32]":
        """Predicts the probability that each ROI is a cell.

        Args:
            roi_statistics: The list of ROIStatistics instances to predict probabilities for.

        Returns:
            An array of shape (n_rois,) containing the probability that each ROI is a cell.
        """
        if len(roi_statistics) == 0:
            return np.array([], dtype=np.float32)

        features = self._extract_features(roi_statistics=roi_statistics)
        log_probabilities = self._compute_log_probabilities(features=features)
        predictions = self._model.predict_proba(log_probabilities)[:, 1]

        return predictions.astype(np.float32)

    def _extract_features(self, roi_statistics: list[ROIStatistics]) -> "NDArray[np.float32]":
        """Extracts classification features from ROIStatistics instances.

        Args:
            roi_statistics: The list of ROIStatistics instances to extract features from.

        Returns:
            An array of shape (n_rois, n_features) containing the extracted features.
        """
        n_rois = len(roi_statistics)
        n_features = len(self._available_features)

        features = np.zeros((n_rois, n_features), dtype=np.float32)

        for roi_index, roi in enumerate(roi_statistics):
            for feature_index, feature_name in enumerate(self._available_features):
                value = getattr(roi, feature_name)
                features[roi_index, feature_index] = np.nan if value is None else value

        return features

    def _get_training_statistics(self) -> "NDArray[np.float32]":
        """Assembles the training statistics matrix from individual feature arrays.

        Returns:
            An array of shape (n_samples, n_features) containing the training statistics.
        """
        feature_arrays = [self._training_features[name] for name in self._available_features]
        return np.column_stack(feature_arrays).astype(np.float32)

    def _compute_log_probabilities(self, features: "NDArray[np.float32]") -> "NDArray[np.float32]":
        """Computes log probability ratios for the given features.

        Args:
            features: An array of shape (n_samples, n_features) containing the feature values.

        Returns:
            An array of shape (n_samples, n_features) containing the log probability ratios.
        """
        if self._probability_grid is None or self._grid_cell_probabilities is None:
            message = "Unable to compute log probabilities. The model has not been fitted."
            console.error(message=message, error=ValueError)

        log_probabilities = np.zeros(features.shape, dtype=np.float32)

        for feature_index in range(features.shape[1]):
            feature_values = features[:, feature_index].copy()

            grid_min = self._probability_grid[0, feature_index]
            grid_max = self._probability_grid[-1, feature_index]
            feature_values[feature_values < grid_min] = grid_min
            feature_values[feature_values > grid_max] = grid_max
            feature_values[np.isnan(feature_values)] = grid_min

            bin_indices = np.digitize(feature_values, self._probability_grid[:, feature_index], right=True) - 1
            bin_indices = np.clip(bin_indices, a_min=0, a_max=self._grid_cell_probabilities.shape[0] - 1)

            probabilities = self._grid_cell_probabilities[bin_indices, feature_index]
            log_probabilities[:, feature_index] = (
                np.log(probabilities + _LOG_EPSILON) - np.log(1 - probabilities + _LOG_EPSILON)
            )

        return log_probabilities

    def _fit_model(self) -> None:
        """Fits the logistic regression model using the loaded training data."""
        if self._training_labels is None or not self._training_features:
            message = "Unable to fit model. No training data has been loaded."
            console.error(message=message, error=ValueError)

        training_statistics = self._get_training_statistics()
        n_samples, n_features = training_statistics.shape

        sorted_statistics = np.sort(training_statistics, axis=0)
        sort_indices = np.argsort(training_statistics, axis=0)
        grid_indices = np.linspace(start=0, stop=n_samples - 1, num=_GRID_NODE_COUNT).astype(np.int32)
        self._probability_grid = sorted_statistics[grid_indices, :].astype(np.float32)

        self._grid_cell_probabilities = np.zeros((_GRID_NODE_COUNT - 1, n_features), dtype=np.float32)
        for bin_index in range(_GRID_NODE_COUNT - 1):
            for feature_index in range(n_features):
                bin_start = grid_indices[bin_index]
                bin_end = grid_indices[bin_index + 1]
                sample_indices = sort_indices[bin_start:bin_end, feature_index]
                self._grid_cell_probabilities[bin_index, feature_index] = np.mean(self._training_labels[sample_indices])

        self._grid_cell_probabilities = gaussian_filter(self._grid_cell_probabilities, sigma=(2.0, 0)).astype(np.float32)

        log_probabilities = self._compute_log_probabilities(features=training_statistics)
        self._model = LogisticRegression(C=100.0, solver="liblinear")
        self._model.fit(log_probabilities, self._training_labels)


def classify(context: RuntimeContext) -> None:
    """Classifies ROIs as cells or non-cells using a pre-trained logistic regression model.

    This function loads the classifier specified in the configuration (or the built-in classifier if none is
    specified), classifies the ROI statistics stored in the runtime context, and updates the cell_classification
    field in-place.

    Args:
        context: The runtime context containing configuration, ROI statistics, and where classification results
            will be stored. The function reads from context.runtime.extraction.roi_statistics and writes to
            context.runtime.extraction.cell_classification.
    """
    roi_statistics = context.runtime.extraction.roi_statistics
    if roi_statistics is None or len(roi_statistics) == 0:
        context.runtime.extraction.cell_classification = np.zeros((0, 2), dtype=np.float32)
        return

    classifier_path = _resolve_classifier_path(
        custom_classifier_path=context.config.classification.custom_classifier_path
    )
    probability_threshold = context.config.roi_detection.preclassification_threshold

    classifier = Classifier(classifier_path=classifier_path)
    context.runtime.extraction.cell_classification = classifier.classify(
        roi_statistics=roi_statistics, probability_threshold=probability_threshold
    )
