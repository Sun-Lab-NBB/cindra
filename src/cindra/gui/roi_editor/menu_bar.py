"""Provides menu bar construction functions for the ROI editor window."""

from __future__ import annotations

from typing import TYPE_CHECKING
from importlib.metadata import entry_points

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from . import (
    roi_draw,
    merge_dialog,
    context_loader,
    classifier_panel,
)

if TYPE_CHECKING:
    from .viewer import ROIEditor


def mainmenu(parent: ROIEditor) -> None:
    """Builds the main File menu bar with core actions.

    Args:
        parent: The ROI editor window.
    """
    main_menu = parent.menuBar()

    # Loads processed data.
    load_action = QAction("&Load processed data", parent)
    load_action.setShortcut("Ctrl+L")
    load_action.triggered.connect(lambda: context_loader.load_dialog(parent))
    parent.addAction(load_action)

    # Loads folder of processed data.
    load_folder_action = QAction("Load &Folder with planeX folders", parent)
    load_folder_action.setShortcut("Ctrl+F")
    load_folder_action.triggered.connect(lambda: context_loader.load_dialog_folder(parent))
    parent.addAction(load_folder_action)

    # Exports figure.
    export_action = QAction("Export as image (svg)", parent)
    export_action.triggered.connect(lambda: context_loader.export_fig(parent))
    export_action.setEnabled(True)
    parent.addAction(export_action)

    # Manual labelling.
    parent.manual = QAction("Manual labelling", parent)
    parent.manual.triggered.connect(lambda: manual_label(parent))
    parent.manual.setEnabled(False)

    # Assembles the File menu.
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(load_action)
    file_menu.addAction(load_folder_action)
    file_menu.addAction(export_action)
    file_menu.addAction(parent.manual)


def classifier(parent: ROIEditor) -> None:
    """Builds the Classifier menu with load, build, and save actions.

    Args:
        parent: The ROI editor window.
    """
    main_menu = parent.menuBar()
    parent.trainfiles = []
    parent.statlabels = None
    parent.loadMenu = QMenu("Load", parent)
    parent.loadClass = QAction("from file", parent)
    parent.loadClass.triggered.connect(lambda: classifier_panel.load_classifier(parent))
    parent.loadClass.setEnabled(False)
    parent.loadMenu.addAction(parent.loadClass)
    parent.loadUClass = QAction("default classifier", parent)
    parent.loadUClass.triggered.connect(lambda: classifier_panel.load_default_classifier(parent))
    parent.loadUClass.setEnabled(False)
    parent.loadMenu.addAction(parent.loadUClass)
    parent.loadSClass = QAction("built-in classifier", parent)
    parent.loadSClass.triggered.connect(lambda: classifier_panel.load_cindra_classifier(parent))
    parent.loadSClass.setEnabled(False)
    parent.loadMenu.addAction(parent.loadSClass)
    parent.loadTrain = QAction("Build", parent)
    parent.loadTrain.triggered.connect(lambda: classifier_panel.load_list(parent))
    parent.loadTrain.setEnabled(False)
    parent.saveDefault = QAction("Save loaded as default", parent)
    parent.saveDefault.triggered.connect(lambda: classifier_panel.class_default(parent))
    parent.saveDefault.setEnabled(False)
    parent.resetDefault = QAction("Reset default to built-in", parent)
    parent.resetDefault.triggered.connect(lambda: classifier_panel.reset_default(parent))
    parent.resetDefault.setEnabled(True)
    class_menu = main_menu.addMenu("&Classifier")
    class_menu.addMenu(parent.loadMenu)
    class_menu.addAction(parent.loadTrain)
    class_menu.addAction(parent.resetDefault)
    class_menu.addAction(parent.saveDefault)


def mergebar(parent: ROIEditor) -> None:
    """Builds the Merge ROIs menu with auto-suggest and save actions.

    Args:
        parent: The ROI editor window.
    """
    main_menu = parent.menuBar()
    merge_menu = main_menu.addMenu("&Merge ROIs")
    parent.sugMerge = QAction("Auto-suggest merges", parent)
    parent.sugMerge.triggered.connect(lambda: suggest_merge(parent))
    parent.sugMerge.setEnabled(False)
    parent.saveMerge = QAction("&Append merges to npy files", parent)
    parent.saveMerge.triggered.connect(lambda: context_loader.save_merge(parent))
    parent.saveMerge.setEnabled(False)
    merge_menu.addAction(parent.sugMerge)
    merge_menu.addAction(parent.saveMerge)


def plugins(parent: ROIEditor) -> None:
    """Builds the Plugins menu from installed entry points.

    Args:
        parent: The ROI editor window.
    """
    main_menu = parent.menuBar()
    parent.plugins = {}
    plugin_menu = main_menu.addMenu("&Plugins")
    for entry_pt in entry_points(group="cindra.plugin"):
        plugin_obj = entry_pt.load()
        parent.plugins[entry_pt.name] = plugin_obj(parent)
        action = QAction(
            parent.plugins[entry_pt.name].name,  # type: ignore[attr-defined]
            parent,
        )
        action.triggered.connect(
            parent.plugins[entry_pt.name].trigger,  # type: ignore[attr-defined]
        )
        plugin_menu.addAction(action)


def manual_label(parent: ROIEditor) -> None:
    """Opens the manual ROI labelling window.

    Args:
        parent: The ROI editor window.
    """
    window = roi_draw.ROIDraw(parent)
    window.show()


def suggest_merge(parent: ROIEditor) -> None:
    """Opens the auto-suggest merge dialog.

    Args:
        parent: The ROI editor window.
    """
    merge_window = merge_dialog.MergeWindow(parent)
    merge_window.show()
