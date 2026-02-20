"""Provides the ROI merge dialog and merge computation logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QDialog, QWidget, QLineEdit, QGridLayout, QMessageBox, QPushButton
from ataraxis_base_utilities import LogLevel, console

from .styles import PARAMETER_EDIT_WIDTH, merge_label_font
from .roi_geometry import circle
from .roi_overlays import (
    add_roi,
    remove_roi,
    redraw_masks,
    init_roi_maps,
    compute_colors,
    flip_for_class,
)
from ..extraction.deconvolve import apply_oasis_deconvolution
from ..detection.roi_statistics import compute_roi_statistics, compute_median_pixel_position

if TYPE_CHECKING:
    from .main_window import MainWindow
    from .context_data import ContextData

# Large sentinel distance used to initialize the distance matrix upper triangle.
_SENTINEL_DISTANCE: float = 1e6

# Small epsilon added to denominators to prevent division by zero.
_CORRELATION_EPSILON: float = 1e-3

# Default correlation threshold for automated merge suggestions.
_DEFAULT_CORRELATION_THRESHOLD: float = 0.8

# Default euclidean distance threshold for automated merge suggestions.
_DEFAULT_DISTANCE_THRESHOLD: float = 100.0

# Scatter plot pen width for merge suggestion visualization.
_SCATTER_PEN_WIDTH: int = 3


def do_merge(parent: MainWindow) -> None:
    """Prompts the user to confirm and then merges selected ROIs.

    Shows a confirmation dialog. If confirmed, merges the currently selected ROIs
    via ``merge_activity_masks``, records the merge, and updates the display.

    Args:
        parent: The main GUI window.
    """
    result = QMessageBox.question(
        parent,
        "Merge cells",
        "Do you want to merge selected cells?",
        QMessageBox.Yes | QMessageBox.No,
    )
    if result == QMessageBox.Yes:
        merge_activity_masks(parent=parent)
        parent.merged.append(list(parent.view_state.merge_indices))
        parent.update_plot()
        console.echo(message=f"Merged ROIs list: {parent.merged}")
        console.echo(message="ROIs merged successfully.", level=LogLevel.SUCCESS)


def merge_activity_masks(parent: MainWindow) -> None:
    """Merges selected ROIs into a single combined ROI.

    Combines pixel masks, fluorescence traces, and statistics from all selected ROIs
    into one new ROI appended to the end of the data arrays. Previously merged
    constituent ROIs are removed and remaining ROIs are updated to reference the new
    merged ROI.

    Args:
        parent: The main GUI window containing ROI data and display state.
    """
    if parent.context_data is None or parent.color_arrays is None or parent.roi_maps is None:
        return
    context = parent.context_data
    state = parent.view_state

    console.echo(message="Merging ROI activity... this may take some time.")
    panel_index = int(1 - context.cell_classification_labels[state.chosen_index])
    y_pixels = np.zeros((0,), dtype=np.int32)
    x_pixels = np.zeros((0,), dtype=np.int32)
    pixel_weights = np.zeros((0,), dtype=np.float32)
    footprints = np.array([])
    cell_fluorescence = np.zeros((0, context.frame_count), dtype=np.float32)
    neuropil_fluorescence = np.zeros((0, context.frame_count), dtype=np.float32)
    if context.has_red_channel:
        channel_2_fluorescence = np.zeros((0, context.frame_count), dtype=np.float32)
        channel_2_neuropil = np.zeros((0, context.frame_count), dtype=np.float32)
        if context.cell_fluorescence_channel_2 is None and context.save_path is not None:
            context.cell_fluorescence_channel_2 = np.load(
                str(context.save_path / "F_chan2.npy"),
            )
            context.neuropil_fluorescence_channel_2 = np.load(
                str(context.save_path / "Fneu_chan2.npy"),
            )

    probabilities = []
    merged_cells = []
    remove_merged = []
    for roi_index in np.array(state.merge_indices):
        roi = context.roi_statistics[roi_index]
        if roi.merged_roi_indices is not None and len(roi.merged_roi_indices) > 0:
            remove_merged.append(roi_index)
            merged_cells.extend(list(roi.merged_roi_indices))
        else:
            merged_cells.append(roi_index)
    merged_cells = np.unique(np.array(merged_cells))

    for roi_index in merged_cells:
        roi = context.roi_statistics[roi_index]
        y_pixels = np.append(y_pixels, roi.y_pixels)
        x_pixels = np.append(x_pixels, roi.x_pixels)
        pixel_weights = np.append(pixel_weights, roi.pixel_weights)
        footprints = np.append(footprints, roi.footprint)
        cell_fluorescence = np.append(
            cell_fluorescence, context.cell_fluorescence[roi_index, :][np.newaxis, :], axis=0,
        )
        neuropil_fluorescence = np.append(
            neuropil_fluorescence, context.neuropil_fluorescence[roi_index, :][np.newaxis, :], axis=0,
        )
        if context.has_red_channel and context.cell_fluorescence_channel_2 is not None:
            channel_2_fluorescence = np.append(
                channel_2_fluorescence,
                context.cell_fluorescence_channel_2[roi_index, :][np.newaxis, :],
                axis=0,
            )
            channel_2_neuropil = np.append(
                channel_2_neuropil,
                context.neuropil_fluorescence_channel_2[roi_index, :][np.newaxis, :],
                axis=0,
            )
        probabilities.append(context.cell_classification_probabilities[roi_index])

    probability_array = np.array(probabilities)
    mean_probability = probability_array.mean()

    # Removes overlapping pixels, keeping the first occurrence.
    combined_pixels = np.concatenate(
        (y_pixels[:, np.newaxis], x_pixels[:, np.newaxis]), axis=1,
    )
    _, unique_indices = np.unique(combined_pixels, return_index=True, axis=0)
    y_pixels = y_pixels[unique_indices]
    x_pixels = x_pixels[unique_indices]
    pixel_weights = pixel_weights[unique_indices]

    # Computes the merged ROI centroid.
    centroid = list(compute_median_pixel_position(y_pixels=y_pixels, x_pixels=x_pixels))

    # Normalizes pixel weights.
    normalized_weights = pixel_weights / pixel_weights.sum()

    # Computes mean activity across merged cells.
    mean_cell_fluorescence = cell_fluorescence.mean(axis=0)
    mean_neuropil_fluorescence = neuropil_fluorescence.mean(axis=0)
    if context.has_red_channel:
        mean_channel_2_fluorescence = channel_2_fluorescence.mean(axis=0)
        mean_channel_2_neuropil = channel_2_neuropil.mean(axis=0)

    corrected_fluorescence = (
        mean_cell_fluorescence - context.neuropil_coefficient * mean_neuropil_fluorescence
    )

    spikes = apply_oasis_deconvolution(
        roi_fluorescence=corrected_fluorescence[np.newaxis, :],
        batch_size=context.extraction_batch_size,
        time_constant=context.tau,
        sampling_rate=context.sampling_rate,
    )

    # Determines the plane index from the first constituent.
    plane_index = context.roi_statistics[merged_cells[0]].plane_index

    # Creates the merged ROIStatistics instance.
    from ..dataclasses.single_day_data import ROIStatistics  # noqa: PLC0415

    merged_roi = ROIStatistics(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        pixel_weights=normalized_weights,
        centroid=centroid,
        footprint=int(np.mean(footprints)),
        pixel_count=len(y_pixels),
        skewness=float(stats.skew(corrected_fluorescence)),
        standard_deviation=float(corrected_fluorescence.std()),
        merged_roi_indices=list(merged_cells),
        plane_index=plane_index,
    )

    # Computes shape statistics for the merged ROI.
    compute_roi_statistics(
        rois=[merged_roi],
        frame_height=context.frame_height,
        frame_width=context.frame_width,
        aspect=context.aspect_ratio if context.aspect_ratio > 0 else None,
        diameter=context.cell_diameter if context.cell_diameter > 0 else None,
        crop=context.crop_to_soma,
    )

    # Rescales pixel weights to preserve total flux from merged cells.
    merged_roi.pixel_weights = merged_roi.pixel_weights * merged_cells.size

    # Removes previously merged constituent ROIs from arrays (in reverse order to preserve indices).
    for constituent in sorted(remove_merged, reverse=True):
        remove_roi(
            roi_maps=parent.roi_maps,
            roi_statistics=context.roi_statistics,
            roi_index=constituent,
            panel=panel_index,
        )
        del context.roi_statistics[constituent]
        context.cell_fluorescence = np.delete(context.cell_fluorescence, constituent, 0)
        context.neuropil_fluorescence = np.delete(context.neuropil_fluorescence, constituent, 0)
        context.spikes = np.delete(context.spikes, constituent, 0)
        context.cell_classification_labels = np.delete(context.cell_classification_labels, constituent, 0)
        context.cell_classification_probabilities = np.delete(
            context.cell_classification_probabilities, constituent, 0,
        )
        context.cell_colocalization_probabilities = np.delete(
            context.cell_colocalization_probabilities, constituent, 0,
        )
        context.cell_colocalization_labels = np.delete(context.cell_colocalization_labels, constituent, 0)
        context.not_merged = np.delete(context.not_merged, constituent, 0)
        if context.has_red_channel:
            if context.cell_fluorescence_channel_2 is not None:
                context.cell_fluorescence_channel_2 = np.delete(
                    context.cell_fluorescence_channel_2, constituent, 0,
                )
            if context.neuropil_fluorescence_channel_2 is not None:
                context.neuropil_fluorescence_channel_2 = np.delete(
                    context.neuropil_fluorescence_channel_2, constituent, 0,
                )

    # Appends the merged ROI to all data arrays.
    context.roi_statistics.append(merged_roi)
    context.cell_fluorescence = np.concatenate(
        (context.cell_fluorescence, mean_cell_fluorescence[np.newaxis, :]), axis=0,
    )
    context.neuropil_fluorescence = np.concatenate(
        (context.neuropil_fluorescence, mean_neuropil_fluorescence[np.newaxis, :]), axis=0,
    )
    if context.has_red_channel:
        if context.cell_fluorescence_channel_2 is not None:
            context.cell_fluorescence_channel_2 = np.concatenate(
                (context.cell_fluorescence_channel_2, mean_channel_2_fluorescence[np.newaxis, :]),
                axis=0,
            )
        if context.neuropil_fluorescence_channel_2 is not None:
            context.neuropil_fluorescence_channel_2 = np.concatenate(
                (context.neuropil_fluorescence_channel_2, mean_channel_2_neuropil[np.newaxis, :]),
                axis=0,
            )
    context.spikes = np.concatenate((context.spikes, spikes), axis=0)
    classification_label = np.array([context.cell_classification_labels[state.chosen_index]], dtype=bool)
    context.cell_classification_labels = np.concatenate(
        (context.cell_classification_labels, classification_label), axis=0,
    )
    context.cell_classification_probabilities = np.append(
        context.cell_classification_probabilities, mean_probability,
    )
    context.cell_colocalization_probabilities = np.append(
        context.cell_colocalization_probabilities, -1.0,
    )
    context.cell_colocalization_labels = np.append(context.cell_colocalization_labels, False)
    context.not_merged = np.append(context.not_merged, False)

    # Computes circle overlay for the new merged ROI.
    y_circle, x_circle = circle(
        centroid=merged_roi.centroid,
        radius=merged_roi.radius,
    )
    valid = (
        (y_circle >= 0) & (x_circle >= 0)
        & (y_circle < context.frame_height) & (x_circle < context.frame_width)
    )
    merged_roi.circle_y_pixels = y_circle[valid]
    merged_roi.circle_x_pixels = x_circle[valid]

    # Recomputes all color arrays and ROI maps.
    parent.color_arrays = compute_colors(context=context, state=state)
    parent.roi_maps = init_roi_maps(context=context, color_arrays=parent.color_arrays)
    parent.mode_change(state.activity_mode)

    # Marks constituent ROIs as merged into the new ROI.
    merged_target_index = context.roi_count - 1
    for roi_index in merged_cells:
        if roi_index < context.roi_count:
            context.roi_statistics[roi_index].merged_into_roi_index = merged_target_index
            remove_roi(
                roi_maps=parent.roi_maps,
                roi_statistics=context.roi_statistics,
                roi_index=roi_index,
                panel=panel_index,
            )
    add_roi(
        roi_maps=parent.roi_maps,
        roi_statistics=context.roi_statistics,
        roi_index=merged_target_index,
        panel=panel_index,
    )
    redraw_masks(
        color_arrays=parent.color_arrays,
        roi_maps=parent.roi_maps,
        y_pixels=y_pixels,
        x_pixels=x_pixels,
    )


def apply(parent: MainWindow) -> None:
    """Applies the probability threshold to reclassify ROIs.

    Reads the threshold from the probability edit field, reclassifies all ROIs based
    on their cell probability exceeding the threshold, and saves the result.

    Args:
        parent: The main GUI window.
    """
    if parent.context_data is None or parent.color_arrays is None or parent.roi_maps is None:
        return
    threshold = float(parent.probedit.text())
    classification_labels = parent.context_data.cell_classification_probabilities > threshold
    flip_for_class(
        context=parent.context_data,
        color_arrays=parent.color_arrays,
        roi_maps=parent.roi_maps,
        new_classification_labels=classification_labels,
    )
    parent.update_plot()
    parent.save_cell_classification()


class MergeWindow(QDialog):
    """Dialog for automated ROI merge suggestions and interactive merging.

    Computes pairwise correlations between cell activity traces and suggests
    ROI pairs that exceed the correlation and distance thresholds. Provides
    buttons to cycle through suggestions and perform merges.

    Args:
        parent: The main GUI window.
    """

    def __init__(self, parent: MainWindow | None = None) -> None:
        super().__init__(parent)
        if parent is None or parent.context_data is None:
            return
        context = parent.context_data

        self.setGeometry(700, 300, 700, 700)
        self.setWindowTitle("Choose merge options")
        self._central_widget = QWidget(self)
        self._grid_layout = QGridLayout()
        self._grid_layout.setVerticalSpacing(2)
        self._grid_layout.setHorizontalSpacing(25)
        self._central_widget.setLayout(self._grid_layout)
        self._plot_widget = pg.GraphicsLayoutWidget()
        self._grid_layout.addWidget(self._plot_widget, 11, 0, 4, 4)
        self._scatter_plot = self._plot_widget.addPlot(row=0, col=0)
        self._scatter_plot.setMouseEnabled(x=False, y=False)
        self._scatter_plot.enableAutoRange(x=True, y=True)

        merge_keys = ["corr_thres", "dist_thres"]
        merge_labels = ["correlation threshold", "euclidean distance threshold"]
        self.ops: dict[str, float] = {
            "corr_thres": _DEFAULT_CORRELATION_THRESHOLD,
            "dist_thres": _DEFAULT_DISTANCE_THRESHOLD,
        }
        self._grid_layout.addWidget(
            QLabel("Press enter in a text box to update params"), 0, 0, 1, 2,
        )
        self._grid_layout.addWidget(
            QLabel("(Correlations use 'activity mode' and 'bin' from main GUI)"), 1, 0, 1, 2,
        )
        self._grid_layout.addWidget(
            QLabel(">>>>>>>>>>>> Parameters <<<<<<<<<<<"), 2, 0, 1, 2,
        )
        self._merge_button = QPushButton("merge selected ROIs", default=False, autoDefault=False)
        self._merge_button.clicked.connect(lambda: self._do_merge(parent))
        self._merge_button.setEnabled(False)
        self._grid_layout.addWidget(self._merge_button, 9, 0, 1, 1)

        self._suggest_button = QPushButton("next merge suggestion", default=False, autoDefault=False)
        self._suggest_button.clicked.connect(lambda: self._suggest_merge(parent))
        self._suggest_button.setEnabled(False)
        self._grid_layout.addWidget(self._suggest_button, 10, 0, 1, 1)

        self._merge_count_label = QLabel("= X possible merges found with these parameters")
        self._grid_layout.addWidget(self._merge_count_label, 7, 0, 1, 2)

        self._suggested_rois_label = QLabel("suggested ROIs to merge: ")
        self._grid_layout.addWidget(self._suggested_rois_label, 8, 0, 1, 2)

        self._edit_list: list[_ParameterEdit] = []
        self._key_list: list[str] = []
        row_offset = 1
        for key, label_text in zip(merge_keys, merge_labels, strict=False):
            label = QLabel(label_text)
            label.setFont(merge_label_font())
            self._grid_layout.addWidget(label, row_offset * 2 + 1, 0, 1, 2)
            edit = _ParameterEdit(key=key, parent=self)
            edit.set_text(ops=self.ops)
            edit.setFixedWidth(PARAMETER_EDIT_WIDTH)
            edit.returnPressed.connect(lambda: self._compute_merge_list(parent))
            self._grid_layout.addWidget(edit, row_offset * 2 + 2, 0, 1, 2)
            self._edit_list.append(edit)
            self._key_list.append(key)
            row_offset += 1

        console.echo(message="Creating merge window... this may take some time.")
        self._correlation_matrix = (
            np.matmul(parent.Fbin[context.cell_classification_labels], parent.Fbin[context.cell_classification_labels].T)
            / parent.Fbin.shape[-1]
        )
        self._correlation_matrix /= (
            np.matmul(
                parent.Fstd[context.cell_classification_labels][:, np.newaxis],
                parent.Fstd[context.cell_classification_labels][np.newaxis, :],
            )
            + _CORRELATION_EPSILON
        )
        self._correlation_matrix -= np.diag(np.diag(self._correlation_matrix))

        self._merge_list: list = []
        self._suggestion_index: int = 0
        self._unmerged: np.ndarray = np.array([], dtype=bool)
        self._compute_merge_list(parent)

    def _do_merge(self, parent: MainWindow) -> None:
        """Performs the merge and updates the correlation matrix.

        Args:
            parent: The main GUI window.
        """
        if parent.context_data is None:
            return
        context = parent.context_data
        state = parent.view_state

        merge_activity_masks(parent=parent)
        parent.merged.append(list(state.merge_indices))
        parent.update_plot()

        correlation_row = (
            np.matmul(parent.Fbin[context.cell_classification_labels], parent.Fbin[-1].T) / parent.Fbin.shape[-1]
        )
        correlation_row /= parent.Fstd[context.cell_classification_labels] * parent.Fstd[-1] + _CORRELATION_EPSILON
        correlation_row[-1] = 0
        self._correlation_matrix = np.concatenate(
            (self._correlation_matrix, correlation_row[np.newaxis, :-1]), axis=0,
        )
        self._correlation_matrix = np.concatenate(
            (self._correlation_matrix, correlation_row[:, np.newaxis]), axis=1,
        )
        for _ in state.merge_indices:
            self._correlation_matrix[state.merge_indices] = 0
            self._correlation_matrix[:, state.merge_indices] = 0

        state.chosen_index = context.roi_count - 1
        state.merge_indices = [state.chosen_index]
        console.echo(
            message=(
                f"ROIs merged: {context.roi_statistics[state.chosen_index].merged_roi_indices}"
            ),
            level=LogLevel.SUCCESS,
        )
        self._compute_merge_list(parent)

    def _compute_merge_list(self, parent: MainWindow) -> None:
        """Computes automated merge suggestions based on correlation and distance thresholds.

        Args:
            parent: The main GUI window.
        """
        if parent.context_data is None:
            return
        context = parent.context_data

        console.echo(message="Computing automated merge suggestions...")
        for index, key in enumerate(self._key_list):
            self.ops[key] = self._edit_list[index].get_text()
        candidate_groups: list = []
        cell_count = context.cell_count
        not_used = np.ones(cell_count, dtype=bool)
        cell_indices = np.where(context.cell_classification_labels)[0]
        for cell_index in range(cell_count):
            if not_used[cell_index]:
                correlated = [
                    i for i, correlation in enumerate(self._correlation_matrix[cell_index])
                    if correlation >= self.ops["corr_thres"]
                ]
                correlated.append(cell_index)
                if len(correlated) > 1:
                    for position, candidate in enumerate(correlated):
                        if not_used[candidate]:
                            correlated[position] = cell_indices[candidate]
                            roi = context.roi_statistics[correlated[position]]
                            if (
                                roi.merged_into_roi_index is not None
                                and roi.merged_into_roi_index >= 0
                            ):
                                correlated[position] = roi.merged_into_roi_index
                    correlated = np.unique(np.array(correlated))
                    if correlated.size > 1:
                        distances = _distance_matrix(context=context, roi_indices=correlated)
                        min_distances = distances.min(axis=1)
                        correlated = correlated[min_distances <= self.ops["dist_thres"]]
                        if correlated.size > 1:
                            for candidate in correlated:
                                not_used[context.cell_classification_labels[:candidate].sum()] = False
                            candidate_groups.append(correlated)
        self._set_merge_list(parent=parent, candidate_groups=candidate_groups)

    def _set_merge_list(self, parent: MainWindow, candidate_groups: list) -> None:
        """Updates the merge suggestion list and UI labels.

        Args:
            parent: The main GUI window.
            candidate_groups: List of arrays, each containing ROI indices to merge.
        """
        self._merge_count_label.setText(
            f"= {len(candidate_groups)} possible merges found with these parameters",
        )
        self._merge_list = candidate_groups
        self._suggestion_index = 0
        if self._merge_list:
            self._suggest_button.setEnabled(True)
            self._unmerged = np.ones(len(self._merge_list), dtype=bool)
            self._suggest_merge(parent=parent)

    def _suggest_merge(self, parent: MainWindow) -> None:
        """Displays the next merge suggestion in the scatter plot.

        Args:
            parent: The main GUI window.
        """
        if parent.context_data is None or parent.color_arrays is None:
            return
        context = parent.context_data
        state = parent.view_state

        state.chosen_index = self._merge_list[self._suggestion_index][0]
        state.merge_indices = list(self._merge_list[self._suggestion_index])
        if self._unmerged[self._suggestion_index]:
            self._suggested_rois_label.setText(f"suggested ROIs to merge: {state.merge_indices}")
            self._merge_button.setEnabled(True)
            self._scatter_plot.clear()
            reference_cell = state.merge_indices[0]
            label_parts = ""
            for roi_index in state.merge_indices[1:]:
                rgb = parent.color_arrays.cols[0, roi_index]
                pen = pg.mkPen(rgb, width=_SCATTER_PEN_WIDTH)
                scatter = pg.ScatterPlotItem(
                    parent.Fbin[reference_cell], parent.Fbin[roi_index], pen=pen,
                )
                self._scatter_plot.addItem(scatter)
                label_parts += f" {roi_index} "
            self._scatter_plot.setLabel("left", label_parts)
            self._scatter_plot.setLabel("bottom", str(reference_cell))
        else:
            # Sets to the merged ROI index.
            roi = context.roi_statistics[state.chosen_index]
            if roi.merged_into_roi_index is not None:
                state.chosen_index = roi.merged_into_roi_index
            state.merge_indices = [state.chosen_index]
            merged_roi = context.roi_statistics[state.chosen_index]
            self._suggested_rois_label.setText(
                f"ROIs merged: {list(merged_roi.merged_roi_indices or [])}",
            )
            self._merge_button.setEnabled(False)
            self._scatter_plot.clear()

        self._suggestion_index += 1
        if self._suggestion_index > len(self._merge_list) - 1:
            self._suggestion_index = 0
        parent.checkBoxz.setChecked(True)
        parent.update_plot()
        parent.win.show()
        parent.show()


class _ParameterEdit(QLineEdit):
    """Numeric input field for merge threshold parameters.

    Args:
        key: The parameter key this edit field controls.
        parent: The parent widget.
    """

    def __init__(self, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._key = key

    def get_text(self) -> float:
        """Returns the current text value as a float."""
        return float(self.text())

    def set_text(self, ops: dict[str, float]) -> None:
        """Sets the display text from the parameter dictionary.

        Args:
            ops: Dictionary mapping parameter keys to their current values.
        """
        self.setText(str(ops[self._key]))


def _distance_matrix(
    context: ContextData,
    roi_indices: np.ndarray,
) -> np.ndarray:
    """Computes a pairwise distance matrix between ROI pixel centroids.

    For each pair of ROIs, computes the mean Euclidean distance between all pairs
    of their constituent pixels. Only the upper triangle is computed; the lower
    triangle is filled with a large sentinel value.

    Args:
        context: The loaded data context containing ROI statistics.
        roi_indices: Array of ROI indices to compute distances between.

    Returns:
        Square distance matrix of shape (n, n) where n is the number of ROI indices.
    """
    count = len(roi_indices)
    distances = _SENTINEL_DISTANCE * np.ones((count, count))
    for row, roi_j in enumerate(roi_indices):
        for col, roi_k in enumerate(roi_indices):
            if row < col:
                distances[row, col] = (
                    (
                        (
                            context.roi_statistics[roi_j].y_pixels[np.newaxis, :]
                            - context.roi_statistics[roi_k].y_pixels[:, np.newaxis]
                        )
                        ** 2
                        + (
                            context.roi_statistics[roi_j].x_pixels[np.newaxis, :]
                            - context.roi_statistics[roi_k].x_pixels[:, np.newaxis]
                        )
                        ** 2
                    )
                    ** 0.5
                ).mean()
    return distances
