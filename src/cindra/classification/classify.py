"""Provides ROI classification functionality for distinguishing cells from artifacts."""

from typing import TYPE_CHECKING
from pathlib import Path
from operator import attrgetter

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import ROIStatistics


_BUILTIN_CLASSIFIER_PATH: Path = Path(__file__).parent / "classifier.npz"
"""The path to the built-in classifier bundled with cindra."""

_CLASSIFICATION_FEATURES: tuple[str, ...] = ("normalized_pixel_count", "compactness", "skewness")
"""The names of the ROI features used for full classification (after signal extraction), in the order they appear in
the feature matrix."""

_PRECLASSIFICATION_FEATURES: tuple[str, ...] = ("normalized_pixel_count", "compactness")
"""The names of the ROI features used for preclassification (during detection, before signal extraction). This subset
excludes skewness which requires extracted fluorescence traces to compute."""

_GRID_NODE_COUNT: int = 100
"""The number of grid nodes used for probability estimation during model fitting."""

_LOG_EPSILON: float = 1e-6
"""The small epsilon value added to probabilities to prevent log(0) errors."""


class Classifier:
    """Provides logistic regression-based classification for identifying cell ROIs.

    This class loads classifier training data from the specified .npz file, fits a logistic regression model, and uses
    it to predict whether detected ROIs represent real cells or artifacts based on their morphological features.

    Notes:
        The classifier file format uses pickle-free npz serialization containing training_labels and feature arrays
        (normalized_pixel_count, compactness, skewness). The model is fitted on load, which takes approximately 10ms
        for the default training set.

    Args:
        classifier_path: The path to a classifier .npz file containing training_labels and feature arrays.
        feature_names: The tuple of feature names to use for classification. Only these features will be loaded from
            the classifier file and used for model fitting. If None, all available features in the classifier file
            are used.

    Attributes:
        _classifier_path: The path to the loaded classifier file.
        _available_features: The list of feature names used by the classifier.
        _training_features: A dictionary mapping feature names to their training value arrays.
        _training_labels: The boolean training labels array with shape (n_samples,).
        _probability_grid: The grid boundaries computed from sorted training statistics with shape
            (n_nodes, n_features). Used to map input feature values to grid intervals for probability lookup
            during classification.
        _grid_cell_probabilities: The Gaussian-smoothed probability that an ROI is a cell for each grid interval
            with shape (n_nodes - 1, n_features). Used to compute log probability ratios that serve as input
            features for the logistic regression model.
        _model: The fitted LogisticRegression model.
    """

    def __init__(self, classifier_path: Path, feature_names: tuple[str, ...] | None = None) -> None:
        if not classifier_path.exists():
            message = (
                f"Unable to load the classification training data. The classifier file does not exist at the "
                f"specified path: {classifier_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        try:
            # Loads the training data.
            data = np.load(classifier_path, allow_pickle=False)

            if "training_labels" not in data:
                message = (
                    f"Unable to load the classification training data. The classifier file at {classifier_path} is "
                    f"missing the 'training_labels' column."
                )
                console.error(message=message, error=ValueError)

            # Resolves the labels and the training dataset size.
            training_labels = data["training_labels"].astype(np.bool_)
            n_samples = len(training_labels)

            training_features: dict[str, NDArray[np.float32]] = {}
            available_features: list[str] = []

            # Determines which features to load. If feature_names is specified, only those features are used.
            # Otherwise, all available features in the classifier file are used.
            target_features = feature_names if feature_names is not None else _CLASSIFICATION_FEATURES

            # Loads the requested features from the classifier file. As long as the dataset contains at least one
            # valid feature, the class can train the model. This allows flexibly working with incomplete datasets
            # and extending the feature set in the future.
            for feature_name in target_features:
                if feature_name in data:
                    feature_array = data[feature_name].astype(np.float32)
                    if len(feature_array) == n_samples and not np.all(np.isnan(feature_array)):
                        training_features[feature_name] = feature_array
                        available_features.append(feature_name)

            if not available_features:
                message = (
                    f"Unable to load the classification training data. The classifier file at {classifier_path} "
                    f"does not contain any of the expected feature columns: {', '.join(target_features)}."
                )
                console.error(message=message, error=ValueError)

            # Sets instance attributes after all validation passes.
            self._classifier_path: Path = classifier_path
            self._available_features: list[str] = available_features
            self._training_features: dict[str, NDArray[np.float32]] = training_features
            self._training_labels: NDArray[np.bool_] = training_labels

            # Fits the logistic regression model using the validated training data.
            self._fit_model()

        except (ValueError, KeyError, TypeError) as exception:
            message = (
                f"Unable to load the classification training data. The classifier file at {classifier_path} is "
                f"corrupted or has an invalid format. Original loader error: {exception}."
            )
            console.error(message=message, error=ValueError)

    def _extract_features(self, roi_statistics: list[ROIStatistics]) -> NDArray[np.float32]:
        """Extracts classification features supported by the model from ROIStatistics instances.

        Args:
            roi_statistics: The list of ROIStatistics instances to extract features from.

        Returns:
            An array of shape (n_rois, n_features) containing the extracted features.
        """
        n_rois = len(roi_statistics)
        n_features = len(self._available_features)
        features = np.zeros((n_rois, n_features), dtype=np.float32)

        # Pre-creates attribute accessors to avoid repeated string lookups.
        getters = [attrgetter(name) for name in self._available_features]

        # Extracts feature values, using NaN for missing values.
        for roi_index, roi in enumerate(roi_statistics):
            for feature_index, getter in enumerate(getters):
                value = getter(roi)
                features[roi_index, feature_index] = np.nan if value is None else value

        return features

    def _get_training_features(self) -> NDArray[np.float32]:
        """Assembles the training feature matrix from individual feature arrays.

        Returns:
            An array of shape (n_samples, n_features) containing the training features.
        """
        feature_arrays = [self._training_features[name] for name in self._available_features]
        return np.column_stack(feature_arrays)

    def _compute_log_probabilities(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Computes log probability ratios for the given features.

        Args:
            features: An array of shape (n_samples, n_features) containing the feature values.

        Returns:
            An array of shape (n_samples, n_features) containing the log probability ratios.
        """
        log_probabilities = np.zeros(features.shape, dtype=np.float32)

        for feature_index in range(features.shape[1]):
            feature_values = features[:, feature_index].copy()

            # Clips feature values to the grid bounds and replaces NaN with the minimum grid value.
            grid_min = self._probability_grid[0, feature_index]
            grid_max = self._probability_grid[-1, feature_index]
            feature_values = np.clip(feature_values, grid_min, grid_max)
            feature_values[np.isnan(feature_values)] = grid_min

            # Maps each feature value to its corresponding grid bin index.
            bin_indices = np.digitize(feature_values, self._probability_grid[:, feature_index], right=True) - 1
            bin_indices = np.clip(bin_indices, a_min=0, a_max=self._grid_cell_probabilities.shape[0] - 1)

            # Looks up the pre-computed cell probability for each bin and converts to log-odds.
            probabilities = self._grid_cell_probabilities[bin_indices, feature_index]
            log_probabilities[:, feature_index] = np.log(probabilities + _LOG_EPSILON) - np.log(
                1 - probabilities + _LOG_EPSILON
            )

        return log_probabilities

    def _predict_probabilities(self, roi_statistics: list[ROIStatistics]) -> NDArray[np.float32]:
        """Predicts the probability that each ROI in the input list is a cell.

        Args:
            roi_statistics: The list of ROIStatistics instances that define the ROIs to predict probabilities for.

        Returns:
            An array of shape (n_rois,) containing the probability that each ROI is a cell.
        """
        features = self._extract_features(roi_statistics=roi_statistics)
        log_probabilities = self._compute_log_probabilities(features=features)
        predictions = self._model.predict_proba(log_probabilities)[:, 1]

        return predictions.astype(np.float32)

    def _fit_model(self) -> None:
        """Fits the logistic regression model using the loaded training data."""
        training_features = self._get_training_features()
        n_samples, n_features = training_features.shape

        # Sorts features and creates evenly-spaced grid boundaries for probability estimation.
        sorted_features = np.sort(training_features, axis=0)
        sort_indices = np.argsort(training_features, axis=0)
        grid_indices = np.linspace(start=0, stop=n_samples - 1, num=_GRID_NODE_COUNT).astype(np.intp)
        self._probability_grid = sorted_features[grid_indices, :]

        # Computes the fraction of cells (vs artifacts) in each grid bin for each feature.
        self._grid_cell_probabilities = np.zeros((_GRID_NODE_COUNT - 1, n_features), dtype=np.float32)
        bin_sizes = grid_indices[1:] - grid_indices[:-1]

        for feature_index in range(n_features):
            # Reorders labels by sorted feature values and computes cumulative sum.
            sorted_labels = self._training_labels[sort_indices[:, feature_index]].astype(np.float32)
            cumulative_sum = np.concatenate([[0], np.cumsum(sorted_labels)])

            # Computes bin sums using cumulative sum differences, then converts to means.
            bin_sums = cumulative_sum[grid_indices[1:]] - cumulative_sum[grid_indices[:-1]]
            self._grid_cell_probabilities[:, feature_index] = bin_sums / bin_sizes

        # Smooths the probability estimates across bins to reduce noise.
        self._grid_cell_probabilities = gaussian_filter(self._grid_cell_probabilities, sigma=(2.0, 0)).astype(
            np.float32
        )

        # Fits the logistic regression model using log-odds transformed features.
        log_probabilities = self._compute_log_probabilities(features=training_features)
        self._model = LogisticRegression(C=100.0, solver="liblinear")
        self._model.fit(log_probabilities, self._training_labels)

    @staticmethod
    def create_training_dataset(
        file_path: Path,
        training_labels: NDArray[np.bool_],
        normalized_pixel_count: NDArray[np.float32],
        compactness: NDArray[np.float32],
        skewness: NDArray[np.float32],
    ) -> None:
        """Creates a new classifier training dataset file from the provided labels and features.

        Args:
            file_path: The path where the classifier file will be saved. Should have .npz extension.
            training_labels: An array of binary labels (False for artifact, True for cell) with shape (n_samples,).
            normalized_pixel_count: An array of normalized pixel count values with shape (n_samples,).
            compactness: An array of compactness values with shape (n_samples,).
            skewness: An array of skewness values with shape (n_samples,).

        Raises:
            ValueError: If feature arrays have mismatched lengths.
        """
        n_samples = len(training_labels)

        # Validates feature array lengths.
        features = {
            "normalized_pixel_count": normalized_pixel_count,
            "compactness": compactness,
            "skewness": skewness,
        }
        for feature_name, feature_array in features.items():
            if len(feature_array) != n_samples:
                message = (
                    f"Unable to create the classifier training dataset file. The feature '{feature_name}' has "
                    f"{len(feature_array)} samples, but training_labels has {n_samples} samples."
                )
                console.error(message=message, error=ValueError)

        # Saves the training dataset.
        np.savez(
            file_path,
            training_labels=training_labels,
            normalized_pixel_count=normalized_pixel_count,
            compactness=compactness,
            skewness=skewness,
        )

    def classify(
        self,
        roi_statistics: list[ROIStatistics],
        probability_threshold: float = 0.5,
    ) -> NDArray[np.float32]:
        """Classifies the ROIs as cells or non-cells based on their morphological features.

        Args:
            roi_statistics: The list of ROIStatistics instances that store the features of the ROIs to classify.
            probability_threshold: The probability threshold above which an ROI is classified as a cell.

        Returns:
            An array of shape (n_rois, 2) where each row contains [is_cell, probability]. The is_cell value is 1.0
            if the ROI is classified as a cell (probability > threshold) and 0.0 otherwise.

        Raises:
            ValueError: If the input roi_statistics list is empty.
        """
        if not roi_statistics:
            message = "Unable to classify ROIs. The input roi_statistics list is empty."
            console.error(message=message, error=ValueError)

        probabilities = self._predict_probabilities(roi_statistics=roi_statistics)
        is_cell = (probabilities > probability_threshold).astype(np.float32)

        return np.stack([is_cell, probabilities], axis=1)


def classify(
    roi_statistics: list[ROIStatistics],
    classification_threshold: float = 0.5,
    custom_classifier_path: Path | None = None,
    preclassification: bool = False,
) -> NDArray[np.float32]:
    """Classifies detected ROIs as cells or non-cells using a logistic regression model.

    This function loads classifier training data from the specified file (or the built-in classifier if no custom path
    is provided), fits a logistic regression model, and uses it to classify the input ROIs based on their morphological
    features.

    Args:
        roi_statistics: The list of ROIStatistics instances containing the morphological features of the ROIs to
            classify. Must contain at least one ROI.
        classification_threshold: The probability threshold above which an ROI is classified as a cell. ROIs with
            probabilities above this threshold are labeled as cells (1.0), others as non-cells (0.0).
        custom_classifier_path: An optional path to a custom classifier .npz file. If None, the built-in classifier
            bundled with cindra is used.
        preclassification: Determines whether to use a 2-feature model (normalized_pixel_count, compactness) suitable
            for early filtering during detection before signal extraction. When False, uses the full 3-feature model
            that includes skewness computed from extracted fluorescence traces.

    Returns:
        An array of shape (n_rois, 2) where each row contains [is_cell, probability]. The is_cell value is 1.0 if the
        ROI is classified as a cell (probability > threshold) and 0.0 otherwise.

    Raises:
        ValueError: If the input roi_statistics list is empty.
    """
    if not roi_statistics:
        message = (
            "Unable to classify ROIs. No ROIs appear to have been detected. Classification requires detection to "
            "discover at least one valid ROI candidate."
        )
        console.error(message=message, error=ValueError)

    # Resolves the classifier dataset to use for training the model.
    classifier_path = custom_classifier_path if custom_classifier_path is not None else _BUILTIN_CLASSIFIER_PATH

    # Selects the feature set based on the classification mode. Preclassification uses only morphological features
    # available during detection, while full classification includes skewness from extracted fluorescence.
    feature_names = _PRECLASSIFICATION_FEATURES if preclassification else _CLASSIFICATION_FEATURES

    # Trains the logistic regression model (~10 ms) and uses it to classify the detected ROIs.
    classifier = Classifier(classifier_path=classifier_path, feature_names=feature_names)
    return classifier.classify(roi_statistics=roi_statistics, probability_threshold=classification_threshold)
