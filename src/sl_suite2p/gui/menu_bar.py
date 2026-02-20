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
    pipeline_dialog,
    classifier_panel,
    registration_viewer,
    visualization_window,
)

if TYPE_CHECKING:
    from .main_window import MainWindow


def mainmenu(parent: MainWindow) -> None:
    """Build the main File menu bar with core actions."""
    main_menu = parent.menuBar()
    # --------------- MENU BAR --------------------------
    # run suite2p from scratch
    run_action = QAction("&Run suite2p", parent)
    run_action.setShortcut("Ctrl+R")
    run_action.triggered.connect(lambda: run_suite2p(parent))
    parent.addAction(run_action)

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

    # load a behavioral trace
    parent.loadBeh = QAction("Load behavior or stim trace (1D only)", parent)
    parent.loadBeh.triggered.connect(lambda: context_loader.load_behavior(parent))
    parent.loadBeh.setEnabled(False)
    parent.addAction(parent.loadBeh)

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
    file_menu.addAction(run_action)
    file_menu.addAction(load_action)
    file_menu.addAction(load_folder_action)
    file_menu.addAction(parent.loadBeh)
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


def visualizations(parent: MainWindow) -> None:
    """Build the Visualizations menu with cell visualization actions."""
    main_menu = parent.menuBar()
    vis_menu = main_menu.addMenu("&Visualizations")
    parent.visualizations = QAction("&Visualize selected cells", parent)
    parent.visualizations.triggered.connect(lambda: vis_window(parent))
    parent.visualizations.setEnabled(False)
    vis_menu.addAction(parent.visualizations)
    parent.visualizations.setShortcut("Ctrl+V")
    parent.custommask = QAction("Load custom hue for ROIs (*.npy)", parent)
    parent.custommask.triggered.connect(lambda: context_loader.load_custom_mask(parent))
    parent.custommask.setEnabled(False)
    vis_menu.addAction(parent.custommask)


def registration(parent: MainWindow) -> None:
    """Build the Registration menu with binary viewer and metrics actions."""
    main_menu = parent.menuBar()
    reg_menu = main_menu.addMenu("&Registration")
    parent.reg = QAction("View registered &binary", parent)
    parent.reg.triggered.connect(lambda: reg_window(parent))
    parent.reg.setShortcut("Ctrl+B")
    parent.reg.setEnabled(True)
    parent.regPC = QAction("View registration &Metrics", parent)
    parent.regPC.triggered.connect(lambda: _registration_metrics_window(parent))
    parent.regPC.setShortcut("Ctrl+M")
    parent.regPC.setEnabled(True)
    reg_menu.addAction(parent.reg)
    reg_menu.addAction(parent.regPC)


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


def run_suite2p(parent: MainWindow) -> None:
    """Open the suite2p pipeline run dialog."""
    window = pipeline_dialog.RunWindow(parent)
    window.show()


def manual_label(parent: MainWindow) -> None:
    """Open the manual ROI labelling window."""
    window = roi_editor.ROIDraw(parent)
    window.show()


def vis_window(parent: MainWindow) -> None:
    """Open the cell visualization window."""
    parent.VW = visualization_window.VisWindow(parent)
    parent.VW.show()


def reg_window(parent: MainWindow) -> None:
    """Open the registered binary viewer window."""
    window = registration_viewer.BinaryPlayer(parent)
    window.show()


def _registration_metrics_window(parent: MainWindow) -> None:
    """Open the registration metrics viewer window."""
    window = registration_viewer.PCViewer(parent)
    window.show()


def suggest_merge(parent: MainWindow) -> None:
    """Open the auto-suggest merge dialog."""
    merge_window = merge_dialog.MergeWindow(parent)
    merge_window.show()
