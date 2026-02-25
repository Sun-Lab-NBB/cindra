"""Provides the manual ROI drawing editor and trace extraction window."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, cast
from pathlib import Path

import numpy as np
from scipy import stats
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from scipy.ndimage import rotate
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
    QLineEdit,
    QGridLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QButtonGroup,
)
from matplotlib.colors import hsv_to_rgb
from ataraxis_base_utilities import LogLevel, console

from ...io import BinaryFile
from .styles import STYLE, label_font_bold
from ...extraction import (
    create_masks,
    apply_oasis_deconvolution,
    compute_delta_fluorescence,
    extract_fluorescence_traces,
)
from ...detection.roi_statistics import compute_roi_statistics

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PySide6.QtGui import QKeyEvent, QCloseEvent

    from .viewer import MainWindow

# Percentile range used for image normalization (lower bound).
_IMAGE_PERCENTILE_LOW: int = 1

# Percentile range used for image normalization (upper bound).
_IMAGE_PERCENTILE_HIGH: int = 99

# Number of reference images available (mean, enhanced, correlation, max projection).
_VIEW_COUNT: int = 4

# Maximum fraction of field of view used as default ROI diameter.
_MAX_DIAMETER_FRACTION: float = 0.2

# Minimum ROI diameter in pixels.
_MIN_DIAMETER: int = 3

# Number of reference ROIs used for normalized pixel count computation.
_REFERENCE_ROI_COUNT: int = 100

# Pen width for ROI ellipse outlines.
_ROI_PEN_WIDTH: int = 3

# Default initial position offset for the ROI ellipse.
_ROI_POSITION_OFFSET: int = 5

# View index for the correlation map in the reference image selector.
_CORRELATION_MAP_VIEW_INDEX: int = 2

# Random number generator for consistent ROI color assignment.
_random_generator: np.random.Generator = np.random.default_rng()

# Fixed width for the save-and-quit button.
_SAVE_BUTTON_WIDTH: int = 100


def _extract_masks_and_traces(
    ops: dict,
    manual_statistics: list[dict],
    original_statistics: np.ndarray,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, dict, list[dict]]:
    """Extracts fluorescence traces from manually drawn ROI masks.

    Creates cell and neuropil masks for the manually drawn ROIs, extracts fluorescence
    traces from the registered binary, and computes activity statistics for classifier
    compatibility.

    Args:
        ops: Operations dictionary.
        manual_statistics: List of stat dictionaries for manually drawn ROIs.
        original_statistics: Array of stat dictionaries for existing detected ROIs.

    Returns:
        Tuple of (cell_fluorescence, neuropil_fluorescence, channel_2_fluorescence,
        channel_2_neuropil, spikes, ops, manual_stat) arrays and updated stat list.
    """
    start_time = time.time()

    # Concatenates manual and original stats for proper neuropil mask computation.
    all_statistics = manual_statistics.copy()
    for index in range(len(original_statistics)):
        all_statistics.append(original_statistics[index])

    compute_roi_statistics(
        all_statistics,  # type: ignore[arg-type]
        ops["frame_height"],
        ops["frame_width"],
        aspect=ops.get("aspect_ratio"),
        diameter=ops["cell_diameter"],
    )
    per_roi_masks = create_masks(
        roi_statistics=all_statistics,  # type: ignore[arg-type]
        height=ops["frame_height"],
        width=ops["frame_width"],
        neuropil=True,
        include_overlap=ops["allow_overlap"],
        cell_probability_percentile=ops["cell_probability_percentile"],
        inner_neuropil_border_radius=ops["inner_neuropil_border_radius"],
        minimum_neuropil_pixels=ops["minimum_neuropil_pixels"],
    )
    manual_stat = all_statistics[: len(manual_statistics)]

    # Unpacks per-ROI masks for manual ROIs into the separate formats expected by the extraction pipeline.
    manual_masks = per_roi_masks[: len(manual_statistics)]
    manual_cell_masks = tuple((indices, weights) for indices, weights, _ in manual_masks)
    manual_neuropil_masks = (
        tuple(neuropil for _, _, neuropil in manual_masks) if manual_masks[0][2] is not None else None
    )
    console.echo(
        message=f"Manual ROI masks: created in {time.time() - start_time:.2f} seconds.",
        level=LogLevel.SUCCESS,
    )

    # Extracts channel 1 fluorescence traces from the registered binary file.
    with BinaryFile(
        height=ops["frame_height"],
        width=ops["frame_width"],
        file_path=ops["registered_binary_path"],
    ) as binary:
        cell_fluorescence, neuropil_fluorescence = extract_fluorescence_traces(
            frames=binary,
            cell_masks=manual_cell_masks,
            neuropil_masks=manual_neuropil_masks,  # type: ignore[arg-type]
            batch_size=ops["batch_size"],
            channel_label="manual ROI channel 1",
        )

    # Extracts channel 2 fluorescence traces if a second channel binary exists.
    if "registered_binary_path_channel_2" in ops:
        with BinaryFile(
            height=ops["frame_height"],
            width=ops["frame_width"],
            file_path=ops["registered_binary_path_channel_2"],
        ) as binary:
            channel_2_fluorescence, channel_2_neuropil = extract_fluorescence_traces(
                frames=binary,
                cell_masks=manual_cell_masks,
                neuropil_masks=manual_neuropil_masks,  # type: ignore[arg-type]
                batch_size=ops["batch_size"],
                channel_label="manual ROI channel 2",
            )
    else:
        roi_count = len(manual_cell_masks)
        frame_count = cell_fluorescence.shape[1]
        channel_2_fluorescence = np.zeros((roi_count, frame_count), dtype=np.float32)
        channel_2_neuropil = np.zeros((roi_count, frame_count), dtype=np.float32)

    # Computes activity statistics for classifier compatibility.
    pixel_counts = np.array(
        [original_statistics[index]["pixel_count"] for index in range(len(original_statistics))],
        dtype=np.float32,
    )
    for index in range(len(manual_stat)):
        manual_stat[index]["normalized_pixel_count"] = manual_stat[index]["pixel_count"] / np.mean(
            pixel_counts[:_REFERENCE_ROI_COUNT],
        )
        manual_stat[index]["compactness"] = 1
        manual_stat[index]["footprint"] = 2
        manual_stat[index]["manual"] = 1
        if "iplane" in original_statistics[0]:
            manual_stat[index]["plane_index"] = original_statistics[0]["plane_index"]

    # Computes skewness and standard deviation from neuropil-corrected fluorescence.
    corrected_fluorescence = cell_fluorescence - ops["neuropil_coefficient"] * neuropil_fluorescence
    skewness = stats.skew(corrected_fluorescence, axis=1)
    standard_deviation = np.std(corrected_fluorescence, axis=1)

    for index in range(cell_fluorescence.shape[0]):
        manual_stat[index]["skewness"] = skewness[index]
        manual_stat[index]["standard_deviation"] = standard_deviation[index]
        manual_stat[index]["centroid"] = [
            np.mean(manual_stat[index]["y_pixels"]),
            np.mean(manual_stat[index]["x_pixels"]),
        ]

    delta_fluorescence = compute_delta_fluorescence(
        roi_fluorescence=cell_fluorescence,
        neuropil_fluorescence=neuropil_fluorescence,
        neuropil_coefficient=ops["neuropil_coefficient"],
        baseline_method=ops["baseline"],
        baseline_window=ops["baseline_window"],
        baseline_sigma=ops["baseline_sigma"],
        baseline_percentile=ops["baseline_percentile"],
        sampling_rate=ops["sampling_rate"],
    )
    spikes = apply_oasis_deconvolution(
        roi_fluorescence=delta_fluorescence,
        batch_size=ops["batch_size"],
        time_constant=ops["tau"],
        sampling_rate=ops["sampling_rate"],
    )

    return (
        cell_fluorescence,
        neuropil_fluorescence,
        channel_2_fluorescence,
        channel_2_neuropil,
        spikes,
        ops,
        manual_stat,
    )


class ROIDraw(QMainWindow):
    """Manual ROI drawing and trace extraction window.

    Provides an interactive editor for drawing elliptical ROIs on reference images,
    extracting fluorescence traces from registered binary data, and saving the results
    alongside the original detected ROIs.

    Args:
        parent: The main GUI window containing session data.
    """

    def __init__(self, parent: MainWindow) -> None:
        super().__init__(parent)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._parent = parent
        self.setGeometry(70, 70, 1400, 800)
        self.setWindowTitle("extract ROI activity")
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self._grid_layout = QGridLayout()
        central_widget.setLayout(self._grid_layout)

        self._plot_widget = pg.GraphicsLayoutWidget()
        self._grid_layout.addWidget(self._plot_widget, 3, 0, 13, 14)

        # Image panel for displaying reference images with ROI overlays.
        self._trace_plot = self._plot_widget.addPlot(row=0, col=1)
        self._trace_plot.setMouseEnabled(x=True, y=False)
        self._trace_plot.setMenuEnabled(False)
        self._trace_plot.scene().sigMouseMoved.connect(self._mouse_moved)

        self._image_view = self._plot_widget.addViewBox(
            name="plot1",
            lockAspect=True,
            row=0,
            col=0,
            invertY=True,
        )
        self._image_item = pg.ImageItem()
        self._image_view.addItem(self._image_item)

        self._plot_widget.scene().sigMouseClicked.connect(self._plot_clicked)

        add_label = QLabel("Add ROI: button / Alt+CLICK")
        add_label.setStyleSheet(STYLE.white_label)
        self._grid_layout.addWidget(add_label, 0, 0, 1, 4)
        remove_label = QLabel("Remove last clicked ROI: D")
        remove_label.setStyleSheet(STYLE.white_label)
        self._grid_layout.addWidget(remove_label, 1, 0, 1, 4)

        self._add_roi_button = QPushButton("add ROI")
        self._add_roi_button.setFont(label_font_bold())
        self._add_roi_button.clicked.connect(lambda: self._add_roi(position=None))
        self._add_roi_button.setEnabled(True)
        self._add_roi_button.setFixedWidth(STYLE.medium_edit_width)
        self._add_roi_button.setStyleSheet(STYLE.button_unpressed)
        self._grid_layout.addWidget(self._add_roi_button, 2, 0, 1, 1)
        diameter_label = QLabel("diameter:")
        diameter_label.setFont(label_font_bold())
        diameter_label.setStyleSheet(STYLE.white_label)
        diameter_label.setFixedWidth(STYLE.medium_edit_width)
        self._grid_layout.addWidget(diameter_label, 2, 1, 1, 1)
        self._diameter_edit = QLineEdit(self)
        self._diameter_edit.setValidator(QtGui.QIntValidator(0, 10000))
        self._diameter_edit.setText("12")
        self._diameter_edit.setFixedWidth(STYLE.small_edit_width)
        self._diameter_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._grid_layout.addWidget(self._diameter_edit, 2, 2, 1, 1)
        self._roi_list: list[_EllipseROI] = []
        self._cell_positions: list = []
        self._extracted: bool = False
        self._extract_button = QPushButton("extract ROIs")
        self._extract_button.setFont(label_font_bold())
        self._extract_button.setStyleSheet(STYLE.button_unpressed)
        self._extract_button.setCheckable(False)
        self._extract_button.clicked.connect(self._process_rois)
        self._grid_layout.addWidget(self._extract_button, 3, 0, 1, 3)
        self._grid_layout.addWidget(QLabel(""), 4, 0, 1, 3)
        self._grid_layout.setRowStretch(4, 1)

        self._save_gui: bool = False
        self._save_button = QPushButton("Save and Quit")
        self._save_button.setFont(label_font_bold())
        self._save_button.clicked.connect(self._close_and_save)
        self._save_button.setEnabled(False)
        self._save_button.setFixedWidth(_SAVE_BUTTON_WIDTH)
        self._save_button.setStyleSheet(STYLE.button_unpressed)
        self._grid_layout.addWidget(self._save_button, 0, 5, 1, 1)

        # View selection buttons for switching reference images.
        self._view_names = [
            "W: mean img",
            "E: mean img (enhanced)",
            "R: correlation map",
            "T: max projection",
        ]
        self._view_buttons = QButtonGroup(self)
        for button_index, name in enumerate(self._view_names):
            button = _ViewButton(button_index=button_index, text=f"&{name}", parent=self)
            self._view_buttons.addButton(button, button_index)
            self._grid_layout.addWidget(button, button_index, 4, 1, 1)
            button.setEnabled(True)
        self._view_buttons.button(0).setChecked(True)
        self._view_buttons.button(0).setStyleSheet(STYLE.button_pressed)

        self._grid_layout.addWidget(QLabel("neuropil"), 13, 13, 1, 1)

        self._frame_height: int = self._parent.ops["frame_height"]  # type: ignore[attr-defined]
        self._frame_width: int = self._parent.ops["frame_width"]  # type: ignore[attr-defined]
        self._cell_classification_labels = self._parent.cell_classification  # type: ignore[attr-defined]

        self._masked_images = self._normalize_images_with_masks()
        self._image_item.setImage(self._masked_images[:, :, :, 0])

        # Trace-related state initialized during extraction.
        self._roi_count: int = 0
        self._current_roi_index: int = 0
        self._scatter_items: list = []
        self._text_labels: list = []
        self._cell_fluorescence: NDArray | None = None
        self._neuropil_fluorescence: NDArray | None = None
        self._channel_2_fluorescence: NDArray | None = None
        self._channel_2_neuropil: NDArray | None = None
        self._spikes: NDArray | None = None
        self._new_stat: list[dict] | None = None
        self._frame_indices: NDArray | None = None
        self._y_minimum: float = 0.0
        self._y_maximum: float = 0.0

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Handles window close, prompting to save if needed."""
        console.echo(message="Closing manual ROI drawing GUI...")
        if not self._save_gui:
            self._check_save_prompt(event)

    def _check_save_prompt(self, event: QCloseEvent) -> None:
        """Prompts the user to save traces before closing.

        Args:
            event: The close event that triggered this prompt.
        """
        result = QMessageBox.question(
            self,
            "PROC",
            "Would you like to save traces before closing? "
            "(if you havent extracted the traces, click Cancel and extract!)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        )
        if result == QMessageBox.StandardButton.Yes:
            self._close_and_save()
        elif result == QMessageBox.StandardButton.Cancel:
            event.ignore()

    def _close_and_save(self) -> None:
        """Saves all ROI data and closes the editor window."""
        base_path = Path(self._parent.basename)  # type: ignore[attr-defined]
        console.echo(message="Saving original stat file...")
        np.save(base_path / "stat_orig.npy", self._parent.stat)  # type: ignore[attr-defined]

        console.echo(message=f"Number of manually drawn ROIs: {self._roi_count}")

        # Appends manual ROIs to existing stats and saves.
        console.echo(message="Saving new combined stat file...")
        assert self._new_stat is not None
        stat_all = self._new_stat.copy()
        for index in range(len(self._parent.stat)):  # type: ignore[attr-defined]
            stat_all.append(self._parent.stat[index])  # type: ignore[attr-defined]
        np.save(base_path / "stat.npy", stat_all)  # type: ignore[arg-type]
        existing_classification = np.concatenate(
            (
                self._parent.cell_classification[:, np.newaxis],  # type: ignore[attr-defined]
                self._parent.cell_classification_probabilities[:, np.newaxis],  # type: ignore[attr-defined]
            ),
            axis=1,
        )

        new_classification = np.ones((self._roi_count, 2))
        new_classification = np.concatenate((new_classification, existing_classification), axis=0)
        np.save(base_path / "cell_classification.npy", new_classification)

        # Saves fluorescence traces.
        assert self._cell_fluorescence is not None
        assert self._neuropil_fluorescence is not None
        assert self._spikes is not None
        combined_cell_fluorescence: NDArray = np.concatenate(
            (self._cell_fluorescence, self._parent.Fcell),  # type: ignore[attr-defined]
            axis=0,
        )
        combined_neuropil: NDArray = np.concatenate(
            (self._neuropil_fluorescence, self._parent.Fneu),  # type: ignore[attr-defined]
            axis=0,
        )
        combined_spikes: NDArray = np.concatenate(
            (self._spikes, self._parent.Spks),  # type: ignore[attr-defined]
            axis=0,
        )
        np.save(base_path / "F.npy", combined_cell_fluorescence)
        np.save(base_path / "Fneu.npy", combined_neuropil)
        np.save(base_path / "spks.npy", combined_spikes)

        if "registered_binary_path_channel_2" in self._parent.ops:  # type: ignore[attr-defined]
            channel_2_fluorescence = np.load(base_path / "F_chan2.npy")
            channel_2_neuropil = np.load(base_path / "Fneu_chan2.npy")
            red_original = np.load(base_path / "cell_colocalization.npy")
            assert self._channel_2_fluorescence is not None
            assert self._channel_2_neuropil is not None
            channel_2_fluorescence = np.concatenate(
                (self._channel_2_fluorescence, channel_2_fluorescence),
                axis=0,
            )
            channel_2_neuropil = np.concatenate(
                (self._channel_2_neuropil, channel_2_neuropil),
                axis=0,
            )
            new_colocalization = np.zeros((self._roi_count, 2))
            new_colocalization = np.concatenate((new_colocalization, red_original), axis=0)
            np.save(base_path / "F_chan2.npy", channel_2_fluorescence)
            np.save(base_path / "Fneu_chan2.npy", channel_2_neuropil)
            np.save(base_path / "cell_colocalization.npy", new_colocalization)

        console.echo(
            message=(
                f"Saved data shapes - Fcell: {np.shape(combined_cell_fluorescence)}, "
                f"Fneu: {np.shape(combined_neuropil)}, Spks: {np.shape(combined_spikes)}, "
                f"cell_classification: {np.shape(new_classification)}, stat: {np.shape(stat_all)}"  # type: ignore[arg-type]
            ),
        )

        # Reloads the session data after saving. Requires roi_editor modernization to use
        # ContextData instead of the old dict-based self._parent.ops / self._parent.stat API.
        self._save_gui = True
        self.close()

    def _normalize_images_with_masks(self) -> NDArray:
        """Creates normalized reference images with cell mask overlays.

        Returns:
            Array of shape (height, width, 3, 4) containing RGBA images for each view.
        """
        masked_images = np.zeros((self._frame_height, self._frame_width, 3, _VIEW_COUNT))
        valid_y = slice(
            self._parent.ops["valid_y_range"][0],  # type: ignore[attr-defined]
            self._parent.ops["valid_y_range"][1],  # type: ignore[attr-defined]
        )
        valid_x = slice(
            self._parent.ops["valid_x_range"][0],  # type: ignore[attr-defined]
            self._parent.ops["valid_x_range"][1],  # type: ignore[attr-defined]
        )

        for view_index in range(_VIEW_COUNT):
            reference_image = np.zeros((self._frame_height, self._frame_width), dtype=np.float32)
            if view_index == 0:
                reference_image[valid_y, valid_x] = self._parent.ops["mean_image"][valid_y, valid_x]  # type: ignore[attr-defined]
            elif view_index == 1:
                reference_image[valid_y, valid_x] = self._parent.ops["enhanced_mean_image"][valid_y, valid_x]  # type: ignore[attr-defined]
            elif view_index == _CORRELATION_MAP_VIEW_INDEX:
                reference_image[valid_y, valid_x] = self._parent.ops["correlation_map"]  # type: ignore[attr-defined]
            elif "max_proj" in self._parent.ops:  # type: ignore[attr-defined]
                reference_image[valid_y, valid_x] = self._parent.ops["maximum_projection"]  # type: ignore[attr-defined]

            low_percentile = np.percentile(reference_image, _IMAGE_PERCENTILE_LOW)
            high_percentile = np.percentile(reference_image, _IMAGE_PERCENTILE_HIGH)
            reference_image = (reference_image - low_percentile) / (high_percentile - low_percentile)
            reference_image = np.maximum(0, np.minimum(1, reference_image))
            masked_images[:, :, :, view_index] = self._create_cell_overlay(mean_image=reference_image)

        return masked_images

    def _create_cell_overlay(self, mean_image: NDArray) -> NDArray:
        """Creates an HSV-to-RGB overlay of detected cells on a reference image.

        Args:
            mean_image: Normalized grayscale reference image of shape (height, width).

        Returns:
            RGB image array of shape (height, width, 3).
        """
        hue = np.zeros_like(mean_image)
        saturation = np.zeros_like(mean_image)

        for roi_index in range(self._parent.cell_classification.shape[0]):  # type: ignore[attr-defined]
            if self._parent.cell_classification[roi_index] == 1:  # type: ignore[attr-defined]
                y_pixels = self._parent.stat[roi_index]["y_pixels"].flatten()  # type: ignore[attr-defined]
                x_pixels = self._parent.stat[roi_index]["x_pixels"].flatten()  # type: ignore[attr-defined]
                hue[y_pixels, x_pixels] = _random_generator.random()
                saturation[y_pixels, x_pixels] = 1

        hsv_image = np.concatenate(
            (hue[:, :, np.newaxis], saturation[:, :, np.newaxis], mean_image[:, :, np.newaxis]),
            axis=-1,
        )
        return hsv_to_rgb(hsv_image)

    def _mouse_moved(self, position: Any) -> None:
        """Tracks mouse position over the trace plot for neuron identification.

        Args:
            position: Scene position of the mouse cursor.
        """
        if self._extracted and self._trace_plot.sceneBoundingRect().contains(position):
            y_value = self._trace_plot.vb.mapSceneToView(position).y()
            self._current_neuron_index = self._roi_count - y_value + 1

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard shortcuts for view switching and ROI deletion."""
        if event.modifiers() not in (QtCore.Qt.KeyboardModifier.AltModifier, QtCore.Qt.KeyboardModifier.ShiftModifier):
            if event.key() == QtCore.Qt.Key.Key_D:
                self._roi_list[self._current_roi_index].remove(parent=self)
            elif event.key() == QtCore.Qt.Key.Key_W:
                self._view_buttons.button(0).setChecked(True)
                cast("_ViewButton", self._view_buttons.button(0)).press(parent=self, button_index=0)
            elif event.key() == QtCore.Qt.Key.Key_E:
                self._view_buttons.button(1).setChecked(True)
                cast("_ViewButton", self._view_buttons.button(1)).press(parent=self, button_index=1)
            elif event.key() == QtCore.Qt.Key.Key_R:
                self._view_buttons.button(2).setChecked(True)
                cast("_ViewButton", self._view_buttons.button(2)).press(parent=self, button_index=2)
            elif event.key() == QtCore.Qt.Key.Key_T:
                self._view_buttons.button(3).setChecked(True)
                cast("_ViewButton", self._view_buttons.button(3)).press(parent=self, button_index=3)

    def _add_roi(self, position: NDArray | None = None) -> None:
        """Adds a new elliptical ROI to the editor.

        Args:
            position: Optional array of [y, x, height, width] for initial placement.
        """
        self._current_roi_index = len(self._roi_list)
        self._roi_count = len(self._roi_list)
        self._roi_list.append(
            _EllipseROI(
                roi_index=self._roi_count,
                parent=self,
                position=position,
                diameter=int(self._diameter_edit.text()),
            ),
        )
        self._roi_list[-1].update_position(parent=self)
        self._roi_count += 1
        console.echo(message=f"{self._roi_count} cells added to manual ROI GUI.")
        self._save_button.setEnabled(False)

    def _plot_clicked(self, event: Any) -> None:
        """Handles click events on the image and trace panels.

        Alt-click adds a new ROI at the cursor position. Double-click resets the view.

        Args:
            event: The mouse click event.
        """
        items = self._plot_widget.scene().items(event.scenePos())
        for item in items:
            if item == self._image_item:
                scene_position = self._image_view.mapSceneToView(event.scenePos())
                click_x = scene_position.x()
                click_y = scene_position.y()
                if event.modifiers() == QtCore.Qt.KeyboardModifier.AltModifier:
                    diameter = int(self._diameter_edit.text())
                    self._add_roi(
                        position=np.array(
                            [
                                click_y - _ROI_POSITION_OFFSET,
                                click_x - _ROI_POSITION_OFFSET,
                                diameter,
                                diameter,
                            ]
                        ),
                    )
                if event.double():
                    self._image_view.setXRange(0, self._frame_width)
                    self._image_view.setYRange(0, self._frame_height)
            elif item == self._trace_plot:
                if event.double():
                    assert self._frame_indices is not None
                    self._trace_plot.setXRange(0, self._frame_indices.size)
                    self._trace_plot.setYRange(self._y_minimum, self._y_maximum)

    def _process_rois(self) -> None:
        """Extracts masks and traces for all drawn ROIs."""
        stat_list: list[dict] = []
        if self._extracted:
            for scatter_item, text_label in zip(self._scatter_items, self._text_labels, strict=False):
                self._image_view.removeItem(text_label)
                self._image_view.removeItem(scatter_item)
        self._scatter_items = []
        self._text_labels = []
        for roi_index in range(self._roi_count):
            ellipse = self._roi_list[roi_index].ellipse
            y_range = self._roi_list[roi_index].y_range
            x_range = self._roi_list[roi_index].x_range
            centroid = self._roi_list[roi_index].centroid
            assert x_range is not None
            assert y_range is not None
            x_grid, y_grid = np.meshgrid(x_range, y_range)
            y_pixels = y_grid[ellipse].flatten()
            x_pixels = x_grid[ellipse].flatten()
            pixel_weights = np.ones(y_pixels.shape)
            stat_list.append(
                {
                    "y_pixels": y_pixels,
                    "x_pixels": x_pixels,
                    "pixel_weights": pixel_weights,
                    "pixel_count": y_pixels.size,
                    "centroid": centroid,
                }
            )
            text_label = pg.TextItem(str(roi_index), self._roi_list[roi_index].color, anchor=(0, 0))
            text_label.setPos(x_pixels.mean(), y_pixels.mean())
            self._image_view.addItem(text_label)
            self._text_labels.append(text_label)
            scatter = pg.ScatterPlotItem(
                [x_pixels.mean()],
                [y_pixels.mean()],
                pen=self._roi_list[roi_index].color,
                symbol="+",
            )
            self._image_view.addItem(scatter)
            self._scatter_items.append(scatter)

        binary_path = Path(self._parent.ops["registered_binary_path"])  # type: ignore[attr-defined]
        if not binary_path.is_file():
            self._parent.ops["registered_binary_path"] = str(Path(self._parent.basename) / "data.bin")  # type: ignore[attr-defined]
        if "registered_binary_path_channel_2" in self._parent.ops:  # type: ignore[attr-defined]
            channel_2_path = Path(self._parent.ops["registered_binary_path_channel_2"])  # type: ignore[attr-defined]
            if not channel_2_path.is_file():
                self._parent.ops["registered_binary_path_channel_2"] = str(  # type: ignore[attr-defined]
                    Path(self._parent.basename) / "data_chan2.bin",  # type: ignore[attr-defined]
                )

        (
            cell_fluorescence,
            neuropil_fluorescence,
            channel_2_fluorescence,
            channel_2_neuropil,
            spikes,
            _,
            new_stat,
        ) = _extract_masks_and_traces(
            ops=self._parent.ops,  # type: ignore[attr-defined]
            manual_statistics=stat_list,
            original_statistics=self._parent.stat,  # type: ignore[attr-defined]
        )
        self._cell_fluorescence = cell_fluorescence
        self._neuropil_fluorescence = neuropil_fluorescence
        self._channel_2_fluorescence = channel_2_fluorescence
        self._channel_2_neuropil = channel_2_neuropil
        self._spikes = spikes
        self._plot_traces()
        self._extracted = True
        self._new_stat = new_stat
        self._save_button.setEnabled(True)

    def _plot_traces(self) -> None:
        """Renders fluorescence traces for all drawn ROIs in the trace panel."""
        assert self._cell_fluorescence is not None
        assert self._neuropil_fluorescence is not None
        self._frame_indices = np.arange(0, self._cell_fluorescence.shape[1], dtype=np.int32)
        self._trace_plot.clear()
        vertical_spacing = 1.0
        axis = self._trace_plot.getAxis("left")
        row = self._roi_count - 1
        tick_labels: list[tuple] = []
        for roi_index in range(self._roi_count):
            fluorescence = self._cell_fluorescence[roi_index, :]
            neuropil = self._neuropil_fluorescence[roi_index, :]
            f_max = fluorescence.max()
            f_min = fluorescence.min()
            normalized = (fluorescence - f_min) / (f_max - f_min)
            rgb = self._roi_list[roi_index].color
            self._trace_plot.plot(self._frame_indices, normalized + row * vertical_spacing, pen=rgb)
            normalized_neuropil = (neuropil - f_min) / (f_max - f_min)
            if self._roi_count == 1:
                self._trace_plot.plot(
                    self._frame_indices,
                    normalized_neuropil + row * vertical_spacing,
                    pen="r",
                )
            tick_labels.append((row * vertical_spacing + normalized.mean(), str(roi_index)))
            row -= 1
        self._y_maximum = (self._roi_count - 1) * vertical_spacing + 1
        self._y_minimum = 0.0
        axis.setTicks([tick_labels])
        self._trace_plot.setXRange(0, self._cell_fluorescence.shape[1])


class _ViewButton(QPushButton):
    """Button for switching between reference image views in the ROI editor.

    Args:
        button_index: Index of the view this button controls.
        text: Display text for the button.
        parent: The ROI editor window.
    """

    def __init__(self, button_index: int, text: str, parent: ROIDraw) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setStyleSheet(STYLE.button_unpressed)
        self.setFont(label_font_bold())
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent=parent, button_index=button_index))
        self.show()

    def press(self, parent: ROIDraw, button_index: int) -> None:
        """Switches the displayed reference image.

        Args:
            parent: The ROI editor window.
            button_index: Index of the view to display.
        """
        for index in range(len(parent._view_names)):
            if parent._view_buttons.button(index).isEnabled():
                parent._view_buttons.button(index).setStyleSheet(STYLE.button_unpressed)
        self.setStyleSheet(STYLE.button_pressed)
        parent._image_item.setImage(parent._masked_images[:, :, :, button_index])
        parent._plot_widget.show()
        parent.show()


class _EllipseROI:
    """Interactive elliptical ROI for manual cell boundary drawing.

    Creates a draggable, resizable, and rotatable ellipse overlay on the image panel.
    Tracks its pixel coordinates and mask for trace extraction.

    Args:
        roi_index: Sequential index of this ROI.
        parent: The ROI editor window.
        position: Optional array of [y, x, height, width] for initial placement.
        diameter: Default diameter if position is not specified.
        color: Optional RGB color tuple. Random if not provided.
        y_range: Optional pre-computed y coordinate range.
        x_range: Optional pre-computed x coordinate range.
    """

    def __init__(
        self,
        roi_index: int,
        parent: ROIDraw | None = None,
        position: NDArray | None = None,
        diameter: int | None = None,
        color: tuple | None = None,
        y_range: NDArray | None = None,
        x_range: NDArray | None = None,
    ) -> None:
        self._roi_index = roi_index
        self.x_range = x_range
        self.y_range = y_range
        self.ellipse: NDArray | None = None
        self.centroid: list[float] = [0.0, 0.0]

        if color is None:
            hsv_color = hsv_to_rgb(np.array([_random_generator.random() / 1.4 + 0.1, 1, 1]))
            self.color = tuple(255 * hsv_color)
        else:
            self.color = color

        assert parent is not None
        if position is None:
            view = parent._image_view.viewRange()
            center_x = (view[0][1] + view[0][0]) / 2
            center_y = (view[1][1] + view[1][0]) / 2
            if diameter is None:
                width = max(
                    _MIN_DIAMETER,
                    min((view[0][1] - view[0][0]) / 4, parent._frame_width * _MAX_DIAMETER_FRACTION),
                )
                height = max(
                    _MIN_DIAMETER,
                    min((view[1][1] - view[1][0]) / 4, parent._frame_height * _MAX_DIAMETER_FRACTION),
                )
            else:
                width = diameter
                height = diameter
            origin_x = center_x - width / 2
            origin_y = center_y - height / 2
        else:
            origin_y = position[0]
            origin_x = position[1]
            height = position[2]
            width = position[3]

        self._draw(parent=parent, origin_y=origin_y, origin_x=origin_x, height=height, width=width)
        self.update_position(parent=parent)
        self._pyqtgraph_roi.sigRegionChangeFinished.connect(lambda: self.update_position(parent=parent))
        self._pyqtgraph_roi.sigClicked.connect(lambda: self.update_position(parent=parent))
        self._pyqtgraph_roi.sigRemoveRequested.connect(lambda: self.remove(parent=parent))

    def _draw(
        self,
        parent: ROIDraw,
        origin_y: float,
        origin_x: float,
        height: float,
        width: float,
    ) -> None:
        """Creates the pyqtgraph EllipseROI and adds it to the image panel.

        Args:
            parent: The ROI editor window.
            origin_y: Y coordinate of the ROI origin.
            origin_x: X coordinate of the ROI origin.
            height: Height of the ellipse.
            width: Width of the ellipse.
        """
        pen = pg.mkPen(self.color, width=_ROI_PEN_WIDTH, style=QtCore.Qt.PenStyle.SolidLine)
        self._pyqtgraph_roi = pg.EllipseROI([origin_x, origin_y], [width, height], pen=pen, removable=True)
        self._pyqtgraph_roi.handleSize = 8
        self._pyqtgraph_roi.handlePen = pen
        self._pyqtgraph_roi.addScaleHandle([1, 0.5], [0.0, 0.5])
        self._pyqtgraph_roi.addScaleHandle([0.5, 0], [0.5, 1])
        self._pyqtgraph_roi.addRotateHandle([0.5, 1], [0.5, 0.5])
        self._pyqtgraph_roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        self.centroid = [origin_y, origin_x]
        parent._image_view.addItem(self._pyqtgraph_roi)

    def remove(self, parent: ROIDraw) -> None:
        """Removes this ROI from the editor and updates indices.

        Args:
            parent: The ROI editor window.
        """
        parent._image_view.removeItem(self._pyqtgraph_roi)
        for index in range(len(parent._roi_list)):
            if index > self._roi_index:
                parent._roi_list[index]._roi_index -= 1
        del parent._roi_list[self._roi_index]
        parent._current_roi_index = min(len(parent._roi_list) - 1, max(0, parent._current_roi_index))
        parent._roi_count -= 1
        parent._plot_widget.show()
        parent.show()

    def _rotate_ellipse(
        self,
        ellipse: NDArray,
        x_range: NDArray,
        y_range: NDArray,
        origin_x: float,
        origin_y: float,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Rotates the ellipse mask to match the ROI handle angle.

        Args:
            ellipse: Boolean mask of the ellipse shape.
            x_range: X coordinate range of the ellipse bounding box.
            y_range: Y coordinate range of the ellipse bounding box.
            origin_x: X coordinate of the ROI origin.
            origin_y: Y coordinate of the ROI origin.

        Returns:
            Tuple of (rotated_ellipse, x_range, y_range) after rotation.
        """
        ellipse = rotate(ellipse, angle=math.floor(self._pyqtgraph_roi.angle()), order=0)
        ellipse = np.flip(ellipse, axis=0)
        x_range = (np.arange(-1 * int(ellipse.shape[1] - 1), 1) + int(origin_x)).astype(np.int32)
        y_range = (np.arange(-1 * int(ellipse.shape[0] - 1), 1) + int(origin_y)).astype(np.int32)
        y_range += int(np.floor(ellipse.shape[0] / 2)) + 1
        return ellipse, x_range, y_range

    def update_position(self, parent: ROIDraw) -> None:
        """Recalculates the ellipse mask after the ROI is moved or resized.

        Args:
            parent: The ROI editor window.
        """
        parent._current_roi_index = self._roi_index
        handle_positions = self._pyqtgraph_roi.getSceneHandlePositions()
        size_x, size_y = self._pyqtgraph_roi.size()
        scene_position = parent._image_view.mapSceneToView(handle_positions[0][1])
        origin_y = scene_position.y()
        origin_x = scene_position.x()

        x_range = (np.arange(-1 * int(size_x), 1) + int(origin_x)).astype(np.int32)
        y_range = (np.arange(-1 * int(size_y), 1) + int(origin_y)).astype(np.int32)
        y_range += int(np.floor(size_y / 2)) + 1

        bounding_rect = self._pyqtgraph_roi.boundingRect()
        x_grid, y_grid = np.meshgrid(
            np.arange(0, x_range.size, 1),
            np.arange(0, y_range.size, 1),
        )
        ellipse = (
            (y_grid - bounding_rect.center().y()) ** 2 / (bounding_rect.plane_heights() / 2) ** 2
            + (x_grid - bounding_rect.center().x()) ** 2 / (bounding_rect.plane_widths() / 2) ** 2
        ) <= 1
        if self._pyqtgraph_roi.angle() not in (0, 180, -180):
            ellipse, x_range, y_range = self._rotate_ellipse(
                ellipse=ellipse,
                x_range=x_range,
                y_range=y_range,
                origin_x=origin_x,
                origin_y=origin_y,
            )
        # Clips the ellipse mask to the field of view boundaries.
        valid_x = np.logical_and(x_range >= 0, x_range < parent._frame_width)
        ellipse = ellipse[:, valid_x]
        x_range = x_range[valid_x]
        valid_y = np.logical_and(y_range >= 0, y_range < parent._frame_height)
        ellipse = ellipse[valid_y, :]
        y_range = y_range[valid_y]

        self.ellipse = ellipse
        self.x_range = x_range
        self.y_range = y_range
