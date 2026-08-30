"""Contains tests for the classify module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ataraxis_base_utilities import error_format

from cindra.dataclasses import ROIMask, ROIStatistics
from cindra.classification.classify import Classifier, classify

if TYPE_CHECKING:
    from pathlib import Path


class TestClassifierDiscrimination:
    """Tests that the fitted model separates ROIs drawn from the two training populations."""

    def test_separable_training_data_separates_the_predicted_probabilities(self, tmp_path: Path) -> None:
        """Verifies that ROIs matching the cell and artifact populations receive opposite classifications."""
        path = tmp_path / "separable.npz"
        _create_separable_classifier_file(path=path)
        classifier = Classifier(classifier_path=path)

        # Each ROI sits at the center of one training population, so a model that learned anything from the labels
        # must assign them opposite classifications. A model returning a constant fails this regardless of the
        # constant it returns.
        cell = _make_roi(compactness=1.0, normalized_pixel_count=2.0, skewness=2.0)
        artifact = _make_roi(compactness=2.5, normalized_pixel_count=0.3, skewness=-0.5)

        result = classifier.classify(roi_statistics=[cell, artifact])

        assert result[0, 1] > 0.99
        assert result[1, 1] < 0.01
        assert result[0, 0] == 1.0
        assert result[1, 0] == 0.0

    def test_swapped_populations_swap_the_classifications(self, tmp_path: Path) -> None:
        """Verifies that the classification follows the labels rather than the direction of the feature values."""
        path = tmp_path / "separable.npz"
        _create_separable_classifier_file(path=path)
        # Inverting every label without touching the features must invert every prediction, which is what separates a
        # model that learned from the labels from one that reads a fixed direction out of the feature values.
        data = dict(np.load(path))
        inverted_path = tmp_path / "inverted.npz"
        np.savez(inverted_path, **{**data, "training_labels": ~data["training_labels"]})

        cell = _make_roi(compactness=1.0, normalized_pixel_count=2.0, skewness=2.0)
        artifact = _make_roi(compactness=2.5, normalized_pixel_count=0.3, skewness=-0.5)

        original = Classifier(classifier_path=path).classify(roi_statistics=[cell, artifact])
        inverted = Classifier(classifier_path=inverted_path).classify(roi_statistics=[cell, artifact])

        assert original[0, 0] == 1.0
        assert original[1, 0] == 0.0
        assert inverted[0, 0] == 0.0
        assert inverted[1, 0] == 1.0
        assert inverted[0, 1] < 0.01
        assert inverted[1, 1] > 0.99

    def test_excluded_feature_cannot_change_a_prediction(self, tmp_path: Path) -> None:
        """Verifies that preclassification ignores skewness while full classification is driven by it."""
        path = tmp_path / "separable.npz"
        _create_separable_classifier_file(path=path)

        # The two ROIs share every morphological feature and differ in skewness alone, which is the one feature
        # preclassification excludes because it requires extracted fluorescence traces.
        high_skewness = _make_roi(compactness=1.5, normalized_pixel_count=1.0, skewness=5.0)
        low_skewness = _make_roi(compactness=1.5, normalized_pixel_count=1.0, skewness=-5.0)

        preclassified = classify(
            roi_statistics=[high_skewness, low_skewness], custom_classifier_path=path, preclassification=True
        )
        fully_classified = classify(
            roi_statistics=[high_skewness, low_skewness], custom_classifier_path=path, preclassification=False
        )

        # Excluding the only feature the two ROIs differ in must make them indistinguishable, exactly rather than
        # approximately, because they reach the fitted model as the same feature vector.
        assert preclassified[0, 1] == preclassified[1, 1]

        # Including it separates them, which proves the identical pair above comes from the exclusion rather than from
        # a model that ignores skewness anyway.
        assert fully_classified[0, 1] > 0.99
        assert fully_classified[1, 1] < 0.01


class TestClassifier:
    """Tests the training-file loading and its rejections, plus the probability output the fitted model returns."""

    def test_representation_reports_the_loaded_file_and_its_fitted_feature_set(self, tmp_path: Path) -> None:
        """Verifies that the representation reports the path, the features kept, and the training sample count."""
        path = tmp_path / "test_classifier.npz"
        # The file holds 200 samples and all three feature columns, but the instance is restricted to two of them, so
        # the feature list is derived from the intersection rather than echoed from either input.
        _create_classifier_file(path=path, sample_count=200)
        classifier = Classifier(classifier_path=path, feature_names=("compactness", "skewness"))

        assert repr(classifier) == (
            f"Classifier(classifier_path={path}, features=['compactness', 'skewness'], training_samples=200)"
        )

    def test_loads_and_fits(self, tmp_path: Path) -> None:
        """Verifies that the classifier loads training data and fits the model."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path=path)
        classifier = Classifier(classifier_path=path)
        assert hasattr(classifier, "_model")
        assert classifier._model is not None

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Verifies that a nonexistent classifier file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Classifier(classifier_path=tmp_path / "nonexistent.npz")

    def test_missing_labels_raises(self, tmp_path: Path) -> None:
        """Verifies that a file without training_labels raises ValueError naming the missing column."""
        path = tmp_path / "bad_classifier.npz"
        np.savez(path, compactness=np.ones(200, dtype=np.float32))
        expected_message = (
            f"Unable to load the classification training data. The classifier file at {path} is missing the "
            f"'training_labels' column."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            Classifier(classifier_path=path)

    def test_no_valid_features_raises(self, tmp_path: Path) -> None:
        """Verifies that a file with labels but no valid features raises ValueError naming the expected columns."""
        path = tmp_path / "labels_only.npz"
        np.savez(path, training_labels=np.ones(200, dtype=np.bool_))
        expected_message = (
            f"Unable to load the classification training data. The classifier file at {path} does not contain any "
            f"of the expected feature columns: normalized_pixel_count, compactness, skewness."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            Classifier(classifier_path=path)

    def test_corrupted_file_raises(self, tmp_path: Path) -> None:
        """Verifies that an unreadable archive raises ValueError describing the file as corrupted."""
        path = tmp_path / "corrupted.npz"
        path.write_bytes(b"not an archive at all")
        with pytest.raises(ValueError, match=r"corrupted\s+or\s+has\s+an\s+invalid\s+format"):
            Classifier(classifier_path=path)

    def test_string_feature_column_raises(self, tmp_path: Path) -> None:
        """Verifies that a feature column stored as strings raises ValueError describing the file as corrupted."""
        # The cast to float32 is what rejects the column, so it has to sit inside the loader's handler. Outside it,
        # the reader receives the bare NumPy conversion error instead of a message naming the classifier file.
        path = tmp_path / "string_feature.npz"
        generator = np.random.default_rng(seed=7)
        np.savez(
            path,
            training_labels=generator.choice([True, False], size=200),
            normalized_pixel_count=np.array(["not a number"] * 200),
            compactness=generator.standard_normal(200).astype(np.float32) + 1.5,
        )
        with pytest.raises(ValueError, match=r"corrupted\s+or\s+has\s+an\s+invalid\s+format"):
            Classifier(classifier_path=path)

    def test_too_few_samples_raises(self, tmp_path: Path) -> None:
        """Verifies that a dataset smaller than the probability grid is rejected by its sample count."""
        path = tmp_path / "small_classifier.npz"
        # The probability grid samples 100 positions across the sorted training values, so a 50-sample dataset would
        # otherwise produce zero-width bins, NaN bin probabilities, and a model fit that fails for an unrelated reason.
        _create_classifier_file(path=path, sample_count=50)
        expected_message = (
            f"Unable to load the classification training data. The classifier file at {path} holds 50 training "
            f"samples, but fitting the classification model requires at least 100 samples."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            Classifier(classifier_path=path)

    def test_classify_output_shape(self, tmp_path: Path) -> None:
        """Verifies that classify returns the correct output shape."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path=path)
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi() for _ in range(5)]
        result = classifier.classify(roi_statistics=rois)
        assert result.shape == (5, 2)

    def test_classify_output_dtype(self, tmp_path: Path) -> None:
        """Verifies that classify returns float32 output."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path=path)
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi()]
        result = classifier.classify(roi_statistics=rois)
        assert result.dtype == np.float32

    def test_classify_probabilities_bounded(self, tmp_path: Path) -> None:
        """Verifies that classification probabilities are between 0 and 1."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path=path)
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi(compactness=compactness) for compactness in [1.0, 1.5, 2.0, 5.0]]
        result = classifier.classify(roi_statistics=rois)
        assert np.all(result[:, 1] >= 0)
        assert np.all(result[:, 1] <= 1)

    def test_classify_is_cell_binary(self, tmp_path: Path) -> None:
        """Verifies that the is_cell column contains only 0.0 or 1.0."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path=path)
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi() for _ in range(10)]
        result = classifier.classify(roi_statistics=rois)
        assert set(result[:, 0].tolist()).issubset({0.0, 1.0})

    def test_classify_threshold(self, tmp_path: Path) -> None:
        """Verifies that the probability threshold is respected."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path=path)
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi()]
        result_low = classifier.classify(roi_statistics=rois, probability_threshold=0.0)
        result_high = classifier.classify(roi_statistics=rois, probability_threshold=1.0)
        # Column 0 of the classification array holds the binary is-cell flag.
        assert result_low[0, 0] == 1.0
        assert result_high[0, 0] == 0.0

    def test_classify_empty_raises(self, tmp_path: Path) -> None:
        """Verifies that classifying an empty list raises ValueError."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path=path)
        classifier = Classifier(classifier_path=path)
        expected_message = "Unable to classify ROIs. The input roi_statistics list is empty."
        with pytest.raises(ValueError, match=error_format(expected_message)):
            classifier.classify(roi_statistics=[])

    def test_feature_subset(self, tmp_path: Path) -> None:
        """Verifies that specifying a feature subset uses only those features."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path=path)
        classifier = Classifier(classifier_path=path, feature_names=("normalized_pixel_count", "compactness"))
        assert len(classifier._available_features) == 2
        assert "skewness" not in classifier._available_features

    def test_handles_none_skewness(self, tmp_path: Path) -> None:
        """Verifies that an ROI carrying no skewness still scores a finite probability pair."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path=path)
        classifier = Classifier(classifier_path=path)
        roi = _make_roi()
        roi.skewness = None
        result = classifier.classify(roi_statistics=[roi])
        assert result.shape == (1, 2)
        assert np.isfinite(result).all()

    def test_all_nan_feature_excluded(self, tmp_path: Path) -> None:
        """Verifies that an all-NaN feature is filtered out while a valid feature remains available."""
        path = tmp_path / "nan_feature.npz"
        generator = np.random.default_rng(seed=42)
        sample_count = 200
        np.savez(
            path,
            training_labels=np.array([True, False] * (sample_count // 2), dtype=np.bool_),
            normalized_pixel_count=generator.standard_normal(sample_count).astype(np.float32) + 1.0,
            compactness=np.full(shape=sample_count, fill_value=np.nan, dtype=np.float32),
        )
        classifier = Classifier(classifier_path=path)
        assert "compactness" not in classifier._available_features
        assert classifier._available_features == ["normalized_pixel_count"]


class TestCreateTrainingDataset:
    """Tests the dataset file the helper writes, its Classifier round-trip, and the sample counts it enforces."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """Verifies that the training dataset file is created."""
        path = tmp_path / "training.npz"
        sample_count = 50
        Classifier.create_training_dataset(
            file_path=path,
            training_labels=np.ones(sample_count, dtype=np.bool_),
            normalized_pixel_count=np.ones(sample_count, dtype=np.float32),
            compactness=np.ones(sample_count, dtype=np.float32),
            skewness=np.zeros(sample_count, dtype=np.float32),
        )
        assert path.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Verifies that a saved dataset can be loaded by the Classifier."""
        path = tmp_path / "roundtrip.npz"
        generator = np.random.default_rng(seed=42)
        sample_count = 200
        Classifier.create_training_dataset(
            file_path=path,
            training_labels=generator.choice([True, False], size=sample_count),
            normalized_pixel_count=generator.standard_normal(sample_count).astype(np.float32) + 1.0,
            compactness=generator.standard_normal(sample_count).astype(np.float32) + 1.5,
            skewness=generator.standard_normal(sample_count).astype(np.float32),
        )
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi()]
        result = classifier.classify(roi_statistics=rois)
        assert result.shape == (1, 2)

    def test_mismatched_lengths_raises(self, tmp_path: Path) -> None:
        """Verifies that mismatched feature array lengths raise ValueError."""
        path = tmp_path / "bad_training.npz"
        expected_message = (
            "Unable to create the classifier training dataset file. The feature 'compactness' has 5 samples, but "
            "training_labels has 10 samples."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            Classifier.create_training_dataset(
                file_path=path,
                training_labels=np.ones(10, dtype=np.bool_),
                normalized_pixel_count=np.ones(10, dtype=np.float32),
                compactness=np.ones(5, dtype=np.float32),  # Mismatched.
                skewness=np.ones(10, dtype=np.float32),
            )


class TestClassifyFunction:
    """Tests the module-level classify function."""

    def test_builtin_classifier(self) -> None:
        """Verifies that the built-in classifier scores every ROI it receives as float32."""
        rois = [_make_roi() for _ in range(3)]
        result = classify(roi_statistics=rois)
        assert result.shape == (3, 2)
        assert result.dtype == np.float32

    def test_custom_classifier(self, tmp_path: Path) -> None:
        """Verifies that a custom classifier path is used."""
        path = tmp_path / "custom.npz"
        _create_classifier_file(path=path)
        rois = [_make_roi()]
        result = classify(roi_statistics=rois, custom_classifier_path=path)
        assert result.shape == (1, 2)

    def test_preclassification_mode(self) -> None:
        """Verifies that preclassification mode returns a classification result for each supplied ROI."""
        rois = [_make_roi()]
        result = classify(roi_statistics=rois, preclassification=True)
        assert result.shape == (1, 2)

    def test_empty_list_raises(self) -> None:
        """Verifies that an empty ROI list raises ValueError."""
        expected_message = (
            "Unable to classify ROIs. No ROIs appear to have been detected. Classification requires detection to "
            "discover at least one valid ROI candidate."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            classify(roi_statistics=[])

    def test_threshold_respected(self) -> None:
        """Verifies that the classification threshold is respected."""
        rois = [_make_roi()]
        result_low = classify(roi_statistics=rois, classification_threshold=0.0)
        result_high = classify(roi_statistics=rois, classification_threshold=1.0)
        assert result_low[0, 0] == 1.0
        assert result_high[0, 0] == 0.0


def _make_roi(
    compactness: float = 1.5,
    normalized_pixel_count: float = 1.0,
    skewness: float = 0.5,
) -> ROIStatistics:
    """Creates a minimal ROIStatistics instance with classification features."""
    mask = ROIMask(
        y_pixels=np.array([5, 5, 6, 6], dtype=np.int32),
        x_pixels=np.array([5, 6, 5, 6], dtype=np.int32),
        pixel_weights=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        centroid=(5, 5),
        frame_width=20,
    )
    roi = ROIStatistics(mask=mask)
    roi.compactness = compactness
    roi.normalized_pixel_count = normalized_pixel_count
    roi.skewness = skewness
    return roi


def _create_classifier_file(path: Path, sample_count: int = 200) -> None:
    """Creates a temporary classifier .npz file."""
    generator = np.random.default_rng(seed=42)
    labels = generator.choice([True, False], size=sample_count)
    np.savez(
        path,
        training_labels=labels,
        normalized_pixel_count=generator.standard_normal(sample_count).astype(np.float32) + 1.0,
        compactness=generator.standard_normal(sample_count).astype(np.float32) + 1.5,
        skewness=generator.standard_normal(sample_count).astype(np.float32),
    )


def _create_separable_classifier_file(path: Path, sample_count: int = 200) -> None:
    """Creates a classifier file whose labels follow its features, so a fitted model can discriminate at all.

    The cell half and the artifact half occupy disjoint ranges of every feature, which is what makes the fitted
    probability grid, the log-odds transform, and the logistic fit observable in the predictions.
    """
    generator = np.random.default_rng(seed=11)
    half = sample_count // 2
    labels = np.concatenate([np.ones(half, dtype=np.bool_), np.zeros(half, dtype=np.bool_)])
    np.savez(
        path,
        training_labels=labels,
        normalized_pixel_count=np.concatenate(
            [generator.normal(loc=2.0, scale=0.1, size=half), generator.normal(loc=0.3, scale=0.1, size=half)]
        ).astype(np.float32),
        compactness=np.concatenate(
            [generator.normal(loc=1.0, scale=0.05, size=half), generator.normal(loc=2.5, scale=0.05, size=half)]
        ).astype(np.float32),
        skewness=np.concatenate(
            [generator.normal(loc=2.0, scale=0.2, size=half), generator.normal(loc=-0.5, scale=0.2, size=half)]
        ).astype(np.float32),
    )
