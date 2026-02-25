"""Provides classifier training, loading, and cell probability computation for the GUI."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (
    QLabel,
    QDialog,
    QWidget,
    QGroupBox,
    QFileDialog,
    QGridLayout,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QAbstractItemView,
)
from ataraxis_base_utilities import LogLevel, console

from .styles import STYLE, label_font, label_font_bold
from .roi_overlays import rgb_masks, istat_transform
from ...classification import Classifier

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .viewer import MainWindow
    from .signals import GUISignals

_CLASSIFIER_COLOR_INDEX: int = 6
"""The index of the classifier probability color mode in the color button group."""

_CLASSIFICATION_FEATURES: tuple[str, ...] = ("normalized_pixel_count", "compactness", "skewness")
"""The feature names used by the classifier, matching ROIStatistics attribute names."""


@dataclass
class ClassifierControls:
    """Holds references to classifier section widgets.

    Attributes:
        classifier_label: Label displaying the current classifier name or status.
        add_to_class_button: Button for adding the current session data to the classifier.
    """

    classifier_label: QLabel
    add_to_class_button: QPushButton


def create_classifier_controls(
    owner: QWidget,  # noqa: ARG001
    signals: GUISignals,  # noqa: ARG001
) -> tuple[QGroupBox, ClassifierControls]:
    """Creates the classifier section inside a group box.

    Args:
        owner: The parent widget for ownership of created widgets.
        signals: The central signal bus for GUI events (reserved for future use).

    Returns:
        Tuple of (group box, classifier controls container).
    """
    group_box = QGroupBox("Classifier")
    group_box.setStyleSheet("QGroupBox { color: white; }")
    layout = QVBoxLayout(group_box)

    classifier_label = QLabel("<font color='white'>not loaded (using prob from cell_classification.npy)</font>")
    classifier_label.setFont(label_font())
    layout.addWidget(classifier_label)

    add_to_class_button = QPushButton(" add current data to classifier")
    add_to_class_button.setFont(label_font_bold())
    add_to_class_button.setStyleSheet(STYLE.button_inactive)
    layout.addWidget(add_to_class_button)

    return group_box, ClassifierControls(
        classifier_label=classifier_label,
        add_to_class_button=add_to_class_button,
    )


def load_classifier(parent: MainWindow) -> None:
    """Opens a file dialog to load a custom classifier file.

    Args:
        parent: The main GUI window.
    """
    name = QFileDialog.getOpenFileName(parent, "Open File")
    if name:
        _load_model(parent=parent, name=name[0])
        _class_activated(parent=parent)
    else:
        console.echo(message="No classifier file selected.", level=LogLevel.WARNING)


def load_cindra_classifier(parent: MainWindow) -> None:
    """Loads the built-in cindra classifier.

    Args:
        parent: The main GUI window.
    """
    _load_model(parent=parent, name=parent.classorig)
    _class_file(parent=parent)
    parent.saveDefault.setEnabled(True)


def load_default_classifier(parent: MainWindow) -> None:
    """Loads the user's default classifier.

    Args:
        parent: The main GUI window.
    """
    _load_model(parent=parent, name=parent.classuser)
    _class_activated(parent=parent)


def class_default(parent: MainWindow) -> None:
    """Saves the current classifier as the user's default after confirmation.

    Args:
        parent: The main GUI window.
    """
    result = QMessageBox.question(
        parent,
        "Default classifier",
        "Are you sure you want to overwrite your default classifier?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    if result == QMessageBox.StandardButton.Yes:
        shutil.copy(parent.classfile, parent.classuser)


def reset_default(parent: MainWindow) -> None:
    """Resets the user's default classifier to the built-in cindra version.

    Args:
        parent: The main GUI window.
    """
    result = QMessageBox.question(
        parent,
        "Default classifier",
        "Are you sure you want to reset the default classifier to the built-in cindra classifier?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    if result == QMessageBox.StandardButton.Yes:
        shutil.copy(parent.classorig, parent.classuser)


def load_list(parent: MainWindow) -> None:
    """Opens the classifier training file chooser dialog.

    Args:
        parent: The main GUI window.
    """
    chooser = ListChooser("classifier training files", parent)
    _ = chooser.exec()


def activate(parent: MainWindow, inactive: bool) -> None:
    """Applies the loaded classifier to compute cell probabilities and update masks.

    Args:
        parent: The main GUI window.
        inactive: When True, recomputes cell probabilities using the loaded model.
    """
    if inactive:
        result = parent.model.classify(roi_statistics=parent.context_data.roi_statistics)  # type: ignore[attr-defined, union-attr]
        parent.context_data.cell_classification_probabilities[:] = result[:, 1]  # type: ignore[union-attr]
    _class_masks(parent=parent)
    parent.update_plot()


def disable(parent: MainWindow) -> None:
    """Disables all classifier controls.

    Args:
        parent: The main GUI window.
    """
    parent.classbtn.setEnabled(False)  # type: ignore[attr-defined]
    parent.saveClass.setEnabled(False)  # type: ignore[attr-defined]
    parent.saveTrain.setEnabled(False)  # type: ignore[attr-defined]
    for btn in parent.classbtns.buttons():  # type: ignore[attr-defined]
        btn.setEnabled(False)


class ListChooser(QDialog):
    """Dialog for selecting training files to build a classifier.

    Allows the user to add cell_classification.npy files or text file lists, then build and
    apply a classifier from the selected training data.

    Args:
        text: Window title text.
        parent: The parent widget.
    """

    def __init__(self, text: str, parent: MainWindow | None = None) -> None:
        super().__init__(parent)
        self.setGeometry(300, 300, 500, 320)
        self.setWindowTitle(text)
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)

        load_cell_button = QPushButton("Load cell_classification.npy")
        load_cell_button.resize(200, 50)
        load_cell_button.clicked.connect(self._load_cell)
        layout.addWidget(load_cell_button, 0, 0, 1, 1)

        load_text_button = QPushButton("Load txt file list")
        load_text_button.clicked.connect(self._load_text)
        layout.addWidget(load_text_button, 0, 1, 1, 1)

        layout.addWidget(QLabel("(select multiple using ctrl)"), 1, 0, 1, 1)
        self.list = QListWidget(parent)
        layout.addWidget(self.list, 2, 0, 5, 4)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

        assert parent is not None

        save_button = QPushButton("build classifier")
        save_button.clicked.connect(lambda: self._build_classifier(parent))
        layout.addWidget(save_button, 8, 0, 1, 1)

        self.apply_button = QPushButton("load in GUI")
        self.apply_button.clicked.connect(lambda: self._apply_class(parent))
        self.apply_button.setEnabled(False)
        layout.addWidget(self.apply_button, 8, 1, 1, 1)

        self.save_as_default_button = QPushButton("save as default")
        self.save_as_default_button.clicked.connect(lambda: self._save_default(parent))
        self.save_as_default_button.setEnabled(False)
        layout.addWidget(self.save_as_default_button, 8, 2, 1, 1)

        done_button = QPushButton("close")
        done_button.clicked.connect(self._exit_list)
        layout.addWidget(done_button, 8, 3, 1, 1)

    def _load_cell(self) -> None:
        """Loads a cell_classification.npy file and adds it to the training file list."""
        name = QFileDialog.getOpenFileName(
            self,
            "Open cell_classification.npy file",
            filter="cell_classification.npy",
        )
        if name:
            try:
                classification = np.load(name[0])
                bad_file = True
                if classification.shape[0] > 0 and (classification[0, 0] == 0 or classification[0, 0] == 1):
                    bad_file = False
                    self.list.addItem(name[0])
                if bad_file:
                    QMessageBox.information(self, "Error", "cell_classification.npy should be 0/1")
            except OSError, RuntimeError, TypeError, NameError:
                QMessageBox.information(self, "Error", "cell_classification.npy should be 0/1")
        else:
            QMessageBox.information(self, "Error", "cell_classification.npy should be 0/1")

    def _load_text(self) -> None:
        """Loads a text file containing paths to training files."""
        name = QFileDialog.getOpenFileName(self, "Open *.txt file", filter="text file (*.txt)")
        if name:
            try:
                with Path(name[0]).open() as text_file:
                    file_content = text_file.read()
                file_lines = file_content.splitlines()
                for file_path in file_lines:
                    self.list.addItem(file_path)
            except OSError, RuntimeError, TypeError, NameError:
                QMessageBox.information(self, "Error", "not a text file")
                console.echo(message="Failed to load text file: invalid file format.", level=LogLevel.ERROR)

    def _build_classifier(self, parent: MainWindow) -> None:
        """Builds a classifier from the selected training files."""
        parent.trainfiles = []
        for item in self.list.selectedItems():
            parent.trainfiles.append(item.text())
        if not parent.trainfiles:
            for row in range(self.list.count()):
                parent.trainfiles.append(self.list.item(row).text())
        if parent.trainfiles:
            console.echo(message="Populating classifier from training files...")
            loaded = _load_data(parent=parent, trainfiles=parent.trainfiles)
            if loaded:
                QMessageBox.information(
                    parent,
                    "Classifier saved",
                    "Classifier built from valid files and saved.",
                )
                self.apply_button.setEnabled(True)
                self.save_as_default_button.setEnabled(True)
        else:
            QMessageBox.information(
                parent,
                "Incorrect files",
                "No valid datasets chosen to build classifier, classifier not built.",
            )

    def _apply_class(self, parent: MainWindow) -> None:
        """Loads the built classifier into the GUI and applies it."""
        try:
            parent.model = Classifier(classifier_path=Path(parent.classfile))
            activate(parent=parent, inactive=True)
        except (ValueError, FileNotFoundError, OSError) as error:
            console.echo(message=f"Failed to load classifier: {error}", level=LogLevel.ERROR)

    def _save_default(self, parent: MainWindow) -> None:
        """Saves the current classifier as the user's default."""
        result = QMessageBox.question(
            self,
            "Default classifier",
            "Are you sure you want to overwrite your default classifier?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if result == QMessageBox.StandardButton.Yes:
            shutil.copy(parent.classfile, parent.classuser)

    def _exit_list(self) -> None:
        """Closes the dialog."""
        self.accept()


def _class_file(parent: MainWindow) -> None:
    """Updates the classifier label text to show the current classifier name.

    Args:
        parent: The main GUI window.
    """
    if parent.classfile == parent.classuser:
        classifier_name = "default classifier"
    elif parent.classfile == parent.classorig:
        classifier_name = "cindra classifier"
    else:
        classifier_name = parent.classfile
    parent._classifier_controls.classifier_label.setText(f"<font color='white'>{classifier_name}</font>")


def _class_activated(parent: MainWindow) -> None:
    """Updates the GUI state after a classifier is loaded.

    Args:
        parent: The main GUI window.
    """
    _class_file(parent=parent)
    parent.saveDefault.setEnabled(True)
    parent._classifier_controls.add_to_class_button.setStyleSheet(STYLE.button_unpressed)
    parent._classifier_controls.add_to_class_button.setEnabled(True)


def _load_model(parent: MainWindow, name: str) -> None:
    """Loads a classifier model from an .npz file.

    Args:
        parent: The main GUI window.
        name: Path to the classifier .npz file.
    """
    console.echo(message=f"Loading classifier from: {name}")
    parent.classfile = name
    try:
        parent.model = Classifier(classifier_path=Path(name))
        activate(parent=parent, inactive=True)
    except (ValueError, FileNotFoundError, OSError) as error:
        console.echo(message=f"Failed to load classifier: {error}", level=LogLevel.ERROR)


def _save_model(
    name: str,
    training_labels: NDArray[np.bool_],
    feature_names: tuple[str, ...],
    feature_matrix: NDArray[np.float32],
) -> None:
    """Saves a classifier training dataset to an .npz file.

    Args:
        name: Path to save the classifier file.
        training_labels: Boolean training labels array with shape (n_samples,).
        feature_names: Tuple of feature names matching columns in feature_matrix.
        feature_matrix: Training feature values with shape (n_samples, n_features).
    """
    save_dict: dict[str, NDArray] = {"training_labels": training_labels}
    for feature_index, feature_name in enumerate(feature_names):
        save_dict[feature_name] = feature_matrix[:, feature_index]
    console.echo(message=f"Saving classifier to: {name}", level=LogLevel.SUCCESS)
    np.savez(name, **save_dict)  # type: ignore[arg-type]


def _load_data(
    parent: MainWindow,
    trainfiles: list[str],
) -> bool:
    """Loads training data from multiple cell_classification.npy files and builds a classifier.

    Loads stat.npy files adjacent to each cell_classification.npy and extracts classification
    features. The stat.npy files must contain dicts with the standard feature keys
    (normalized_pixel_count, compactness, skewness).

    Args:
        parent: The main GUI window.
        trainfiles: List of paths to cell_classification.npy training files.

    Returns:
        True if the classifier was successfully built and saved.
    """
    feature_count = len(_CLASSIFICATION_FEATURES)
    train_stats = np.zeros((0, feature_count), dtype=np.float32)
    train_labels = np.zeros((0,), dtype=np.bool_)
    trainfiles_good = []
    loaded = False
    if trainfiles is not None:
        for fname in trainfiles:
            bad_file = False
            try:
                classification_data = np.load(fname)
                ncells = classification_data.shape[0]
            except ValueError, OSError, RuntimeError, TypeError, NameError:
                console.echo(message=f"  {fname}: not a numpy array of booleans", level=LogLevel.WARNING)
                bad_file = True
            if not bad_file:
                base_path = Path(fname).parent
                stat_length = 0
                try:
                    stat = np.load(str(base_path / "stat.npy"), allow_pickle=True)
                    _ = stat[0]["y_pixels"]
                    stat_length = len(stat)
                except IndexError, KeyError, OSError, RuntimeError, TypeError, NameError:
                    console.echo(
                        message=f"  {base_path}: incorrect or missing stat.npy file",
                        level=LogLevel.WARNING,
                    )
                if stat_length != ncells:
                    console.echo(
                        message=f"  {base_path}: stat.npy length doesn't match cell_classification.npy",
                        level=LogLevel.WARNING,
                    )
                else:
                    console.echo(message=f"  {fname}: added to classifier", level=LogLevel.SUCCESS)
                    labels = classification_data[:, 0].astype(np.bool_)
                    stat_values = np.reshape(
                        np.array(
                            [stat[j][k] for j in range(len(stat)) for k in _CLASSIFICATION_FEATURES],
                        ),
                        (len(stat), -1),
                    ).astype(np.float32)
                    train_stats = np.concatenate((train_stats, stat_values), axis=0)
                    train_labels = np.concatenate((train_labels, labels), axis=0)
                    trainfiles_good.append(fname)
    if trainfiles_good:
        classfile, saved = _save_classifier(
            parent=parent,
            feature_matrix=train_stats,
            training_labels=train_labels,
        )
        if saved:
            parent.classfile = classfile
            loaded = True
        else:
            QMessageBox.information(
                parent,
                "Incorrect file path",
                "Incorrect save path for classifier, classifier not built.",
            )
    else:
        QMessageBox.information(
            parent,
            "Incorrect files",
            "No valid datasets chosen to build classifier, classifier not built.",
        )
    return loaded


def _add_to(parent: MainWindow) -> None:
    """Adds the current session data to the loaded classifier.

    Extracts classification features from the current session's ROI statistics, appends them to
    the existing classifier training data, saves the updated dataset, and reloads the classifier.

    Args:
        parent: The main GUI window.
    """
    console.echo(message="Adding current dataset to classifier...")
    classifier_name = "the default classifier" if parent.classfile == parent.classuser else parent.classfile
    result = QMessageBox.question(
        parent,
        "Default classifier",
        f"Current classifier is {classifier_name}. Add to this classifier?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    if result == QMessageBox.StandardButton.Yes:
        # Extracts features from the current session's ROI statistics.
        roi_statistics = parent.context_data.roi_statistics  # type: ignore[union-attr]
        feature_names = parent.model._available_features  # type: ignore[attr-defined]
        new_features = parent.model._extract_features(roi_statistics=roi_statistics)  # type: ignore[attr-defined]

        # Gets existing training data from the loaded model.
        existing_features = parent.model._get_training_features()  # type: ignore[attr-defined]
        existing_labels = parent.model._training_labels  # type: ignore[attr-defined]

        # Concatenates new session data with existing training data.
        combined_features = np.concatenate([existing_features, new_features], axis=0)
        combined_labels = np.concatenate(
            [
                existing_labels,
                parent.context_data.cell_classification_labels,  # type: ignore[union-attr]
            ],
            axis=0,
        )

        # Saves the combined dataset and reloads the classifier.
        _save_model(
            name=parent.classfile,
            training_labels=combined_labels,
            feature_names=tuple(feature_names),
            feature_matrix=combined_features,
        )
        try:
            parent.model = Classifier(classifier_path=Path(parent.classfile))
            activate(parent=parent, inactive=True)
            QMessageBox.information(
                parent,
                "Classifier saved and loaded",
                "Current dataset added to classifier, and cell probabilities computed and in GUI",
            )
        except (ValueError, FileNotFoundError, OSError) as error:
            console.echo(message=f"Failed to reload classifier: {error}", level=LogLevel.ERROR)


def _save_classifier(
    parent: MainWindow,
    feature_matrix: NDArray[np.float32],
    training_labels: NDArray[np.bool_],
) -> tuple[str, bool]:
    """Opens a save dialog and saves the classifier training dataset.

    Args:
        parent: The main GUI window.
        feature_matrix: Training feature values with shape (n_samples, n_features).
        training_labels: Boolean training labels array with shape (n_samples,).

    Returns:
        Tuple of (file_path, success) indicating where the dataset was saved and whether it
        succeeded.
    """
    dialog_result = QFileDialog.getSaveFileName(parent, "Classifier name (*.npz)")
    name = dialog_result[0]
    saved = False
    if name:
        try:
            _save_model(
                name=name,
                training_labels=training_labels,
                feature_names=_CLASSIFICATION_FEATURES,
                feature_matrix=feature_matrix,
            )
            saved = True
        except OSError, RuntimeError, TypeError, NameError, FileNotFoundError:
            console.echo(message="Failed to save classifier: incorrect filename.", level=LogLevel.ERROR)
    return name, saved


def _save_list(parent: MainWindow) -> None:
    """Saves the list of training file paths to a text file.

    Args:
        parent: The main GUI window.
    """
    name = QFileDialog.getSaveFileName(parent, "Save list of cell_classification.npy")
    if name:
        try:
            with Path(name[0]).open("w") as output_file:
                for file_path in parent.trainfiles:
                    output_file.write(file_path)
                    output_file.write("\n")
        except ValueError, OSError, RuntimeError, TypeError, NameError, FileNotFoundError:
            console.echo(message="Failed to save list: incorrect filename.", level=LogLevel.ERROR)


def _class_masks(parent: MainWindow) -> None:
    """Computes and applies the classifier probability color overlay.

    Args:
        parent: The main GUI window.
    """
    if parent.context_data is None or parent.color_arrays is None or parent.roi_maps is None:
        return
    istat = parent.context_data.cell_classification_probabilities.copy()
    parent.color_arrays.colorbar[_CLASSIFIER_COLOR_INDEX] = [
        float(istat.min()),
        float((istat.max() - istat.min()) / 2),
        float(istat.max()),
    ]
    istat = istat - istat.min()
    istat_max = istat.max()
    if istat_max > 0:
        istat = istat / istat_max
    color = istat_transform(istat=istat, colormap=parent.view_state.roi_colormap)
    parent.color_arrays.cols[_CLASSIFIER_COLOR_INDEX] = color
    parent.color_arrays.istat[_CLASSIFIER_COLOR_INDEX] = istat.flatten()
    rgb_masks(
        color_arrays=parent.color_arrays,
        roi_maps=parent.roi_maps,
        color=color,
        color_index=_CLASSIFIER_COLOR_INDEX,
    )
