"""Provides menu bar construction functions for the main GUI window."""

from __future__ import annotations

from typing import TYPE_CHECKING
from importlib.metadata import entry_points

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from . import (
    roi_editor,
    merge_dialog,
    context_loader,
    classifier_panel,
)

if TYPE_CHECKING:
    from .main_window import MainWindow


def mainmenu(parent: MainWindow) -> None:
    """Build the main File menu bar with core actions."""
    main_menu = parent.menuBar()

    # load processed data
    load_action = QAction("&Load processed data", parent)
    load_action.setShortcut("Ctrl+L")
    load_action.triggered.connect(lambda: context_loader.load_dialog(parent))
    parent.addAction(load_action)

    # load folder of processed data
    load_folder_action = QAction("Load &Folder with planeX folders", parent)
    load_folder_action.setShortcut("Ctrl+F")
    load_folder_action.triggered.connect(lambda: context_loader.load_dialog_folder(parent))
    parent.addAction(load_folder_action)

    # export figure
    export_action = QAction("Export as image (svg)", parent)
    export_action.triggered.connect(lambda: context_loader.export_fig(parent))
    export_action.setEnabled(True)
    parent.addAction(export_action)

    # manual labelling
    parent.manual = QAction("Manual labelling", parent)
    parent.manual.triggered.connect(lambda: manual_label(parent))
    parent.manual.setEnabled(False)

    # make mainmenu!
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(load_action)
    file_menu.addAction(load_folder_action)
    file_menu.addAction(export_action)
    file_menu.addAction(parent.manual)


def classifier(parent: MainWindow) -> None:
    """Build the Classifier menu with load, build, and save actions."""
    main_menu = parent.menuBar()
    # classifier menu
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
    parent.loadSClass.triggered.connect(lambda: classifier_panel.load_s2p_classifier(parent))
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


def mergebar(parent: MainWindow) -> None:
    """Build the Merge ROIs menu with auto-suggest and save actions."""
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


def plugins(parent: MainWindow) -> None:
    """Build the Plugins menu from installed entry points."""
    main_menu = parent.menuBar()
    parent.plugins = {}
    plugin_menu = main_menu.addMenu("&Plugins")
    for entry_pt in entry_points(group="suite2p.plugin"):
        plugin_obj = entry_pt.load()  # load the advertised class from entry_points
        parent.plugins[entry_pt.name] = plugin_obj(
            parent
        )  # initialize an object instance from the loaded class and keep it alive in parent; expose parent to plugin
        action = QAction(
            parent.plugins[entry_pt.name].name, parent
        )  # create plugin menu item with the name property of the loaded class
        action.triggered.connect(
            parent.plugins[entry_pt.name].trigger
        )  # attach class method "trigger" to plugin menu action
        plugin_menu.addAction(action)


def manual_label(parent: MainWindow) -> None:
    """Open the manual ROI labelling window."""
    window = roi_editor.ROIDraw(parent)
    window.show()


def suggest_merge(parent: MainWindow) -> None:
    """Open the auto-suggest merge dialog."""
    merge_window = merge_dialog.MergeWindow(parent)
    merge_window.show()
