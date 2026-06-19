"""Contains tests for the classify module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from cindra.dataclasses import ROIMask, ROIStatistics
from cindra.classification.classify import Classifier, classify

if TYPE_CHECKING:
    from pathlib import Path


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
    rng = np.random.default_rng(42)
    labels = rng.choice([True, False], size=sample_count)
    np.savez(
        path,
        training_labels=labels,
        normalized_pixel_count=rng.standard_normal(sample_count).astype(np.float32) + 1.0,
        compactness=rng.standard_normal(sample_count).astype(np.float32) + 1.5,
        skewness=rng.standard_normal(sample_count).astype(np.float32),
    )


class TestClassifier:
    """Tests the Classifier class."""

    def test_loads_and_fits(self, tmp_path: Path) -> None:
        """Verifies that the classifier loads training data and fits the model."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path)
        classifier = Classifier(classifier_path=path)
        assert hasattr(classifier, "_model")
        assert classifier._model is not None

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Verifies that a nonexistent classifier file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Classifier(classifier_path=tmp_path / "nonexistent.npz")

    def test_missing_labels_raises(self, tmp_path: Path) -> None:
        """Verifies that a file without training_labels raises ValueError."""
        path = tmp_path / "bad_classifier.npz"
        np.savez(path, compactness=np.ones(10, dtype=np.float32))
        with pytest.raises(ValueError, match="Unable to load the classification training data"):
            Classifier(classifier_path=path)

    def test_no_valid_features_raises(self, tmp_path: Path) -> None:
        """Verifies that a file with labels but no valid features raises ValueError."""
        path = tmp_path / "labels_only.npz"
        np.savez(path, training_labels=np.ones(10, dtype=np.bool_))
        with pytest.raises(ValueError, match="Unable to load the classification training data"):
            Classifier(classifier_path=path)

    def test_classify_output_shape(self, tmp_path: Path) -> None:
        """Verifies that classify returns the correct output shape."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path)
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi() for _ in range(5)]
        result = classifier.classify(roi_statistics=rois)
        assert result.shape == (5, 2)

    def test_classify_output_dtype(self, tmp_path: Path) -> None:
        """Verifies that classify returns float32 output."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path)
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi()]
        result = classifier.classify(roi_statistics=rois)
        assert result.dtype == np.float32

    def test_classify_probabilities_bounded(self, tmp_path: Path) -> None:
        """Verifies that classification probabilities are between 0 and 1."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path)
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi(compactness=c) for c in [1.0, 1.5, 2.0, 5.0]]
        result = classifier.classify(roi_statistics=rois)
        assert np.all(result[:, 1] >= 0)
        assert np.all(result[:, 1] <= 1)

    def test_classify_is_cell_binary(self, tmp_path: Path) -> None:
        """Verifies that the is_cell column contains only 0.0 or 1.0."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path)
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi() for _ in range(10)]
        result = classifier.classify(roi_statistics=rois)
        assert set(result[:, 0].tolist()).issubset({0.0, 1.0})

    def test_classify_threshold(self, tmp_path: Path) -> None:
        """Verifies that the probability threshold is respected."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path)
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi()]
        result_low = classifier.classify(roi_statistics=rois, probability_threshold=0.0)
        result_high = classifier.classify(roi_statistics=rois, probability_threshold=1.0)
        # With threshold=0, all ROIs should be classified as cells.
        assert result_low[0, 0] == 1.0
        # With threshold=1, no ROIs should be classified as cells.
        assert result_high[0, 0] == 0.0

    def test_classify_empty_raises(self, tmp_path: Path) -> None:
        """Verifies that classifying an empty list raises ValueError."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path)
        classifier = Classifier(classifier_path=path)
        with pytest.raises(ValueError, match="Unable to classify ROIs"):
            classifier.classify(roi_statistics=[])

    def test_feature_subset(self, tmp_path: Path) -> None:
        """Verifies that specifying a feature subset uses only those features."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path)
        classifier = Classifier(classifier_path=path, feature_names=("normalized_pixel_count", "compactness"))
        assert len(classifier._available_features) == 2
        assert "skewness" not in classifier._available_features

    def test_handles_none_skewness(self, tmp_path: Path) -> None:
        """Verifies that ROIs with None skewness are handled correctly."""
        path = tmp_path / "test_classifier.npz"
        _create_classifier_file(path)
        classifier = Classifier(classifier_path=path)
        roi = _make_roi()
        roi.skewness = None
        result = classifier.classify(roi_statistics=[roi])
        assert result.shape == (1, 2)
        assert np.isfinite(result).all()


class TestCreateTrainingDataset:
    """Tests Classifier.create_training_dataset."""

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
        rng = np.random.default_rng(42)
        sample_count = 200
        Classifier.create_training_dataset(
            file_path=path,
            training_labels=rng.choice([True, False], size=sample_count),
            normalized_pixel_count=rng.standard_normal(sample_count).astype(np.float32) + 1.0,
            compactness=rng.standard_normal(sample_count).astype(np.float32) + 1.5,
            skewness=rng.standard_normal(sample_count).astype(np.float32),
        )
        classifier = Classifier(classifier_path=path)
        rois = [_make_roi()]
        result = classifier.classify(roi_statistics=rois)
        assert result.shape == (1, 2)

    def test_mismatched_lengths_raises(self, tmp_path: Path) -> None:
        """Verifies that mismatched feature array lengths raise ValueError."""
        path = tmp_path / "bad_training.npz"
        with pytest.raises(ValueError, match="Unable to create the classifier training dataset"):
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
        """Verifies that the built-in classifier works."""
        rois = [_make_roi() for _ in range(3)]
        result = classify(roi_statistics=rois)
        assert result.shape == (3, 2)
        assert result.dtype == np.float32

    def test_custom_classifier(self, tmp_path: Path) -> None:
        """Verifies that a custom classifier path is used."""
        path = tmp_path / "custom.npz"
        _create_classifier_file(path)
        rois = [_make_roi()]
        result = classify(roi_statistics=rois, custom_classifier_path=path)
        assert result.shape == (1, 2)

    def test_preclassification_mode(self) -> None:
        """Verifies that preclassification mode uses a 2-feature model."""
        rois = [_make_roi()]
        result = classify(roi_statistics=rois, preclassification=True)
        assert result.shape == (1, 2)

    def test_empty_list_raises(self) -> None:
        """Verifies that an empty ROI list raises ValueError."""
        with pytest.raises(ValueError, match="Unable to classify ROIs"):
            classify(roi_statistics=[])

    def test_threshold_respected(self) -> None:
        """Verifies that the classification threshold is respected."""
        rois = [_make_roi()]
        result_low = classify(roi_statistics=rois, classification_threshold=0.0)
        result_high = classify(roi_statistics=rois, classification_threshold=1.0)
        assert result_low[0, 0] == 1.0
        assert result_high[0, 0] == 0.0
