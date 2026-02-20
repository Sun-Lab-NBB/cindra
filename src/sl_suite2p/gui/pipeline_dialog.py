"""Provides the pipeline run configuration dialog and supporting helper widgets."""

import sys
import json
import shutil
from pathlib import Path
from datetime import UTC, datetime

import numpy as np
from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import (
    QLabel,
    QDialog,
    QWidget,
    QComboBox,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QGridLayout,
    QPushButton,
    QButtonGroup,
)
from ataraxis_base_utilities import LogLevel, console

from .styles import header_font
from ..dataclasses import generate_default_ops
from .context_loader import load_proc


class TextChooser(QDialog):
    """Prompt dialog for entering an HDF5 dataset key."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the HDF5 key chooser dialog."""
        super().__init__(parent)
        self.setGeometry(300, 300, 180, 100)
        self.setWindowTitle("h5 key")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        self.qedit = QLineEdit("data")
        layout.addWidget(QLabel("h5 key for data field"), 0, 0, 1, 3)
        layout.addWidget(self.qedit, 1, 0, 1, 2)
        done = QPushButton("OK")
        done.clicked.connect(self.exit_list)
        layout.addWidget(done, 2, 1, 1, 1)

    def exit_list(self) -> None:
        """Store the entered key and accept the dialog."""
        self.h5_key = self.qedit.text()
        self.accept()


class RunWindow(QDialog):
    """Pipeline run configuration dialog for suite2p processing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the run configuration dialog and load default ops."""
        super().__init__(parent)
        self.setGeometry(10, 10, 1500, 900)
        self.setWindowTitle("Choose run options (hold mouse over parameters to see descriptions)")
        self.parent = parent
        self.win = QWidget(self)
        self.layout = QGridLayout()
        self.layout.setVerticalSpacing(2)
        self.layout.setHorizontalSpacing(25)
        self.win.setLayout(self.layout)
        # initial ops values
        self.opsfile = parent.opsuser
        self.ops_path = Path.home() / ".suite2p" / "ops"
        try:
            self.reset_ops()
            console.echo(message="Loaded default ops", level=LogLevel.SUCCESS)
        except Exception as e:
            console.echo(message=f"ERROR: {e}", level=LogLevel.ERROR)
            console.echo(message="Could not load default ops, using built-in ops settings", level=LogLevel.WARNING)
            self.ops = generate_default_ops()

        # Removes any remaining ops files from the current directory.
        for ops_file in Path.cwd().glob("ops*.npy"):
            ops_file.unlink()
        for db_file in Path.cwd().glob("db*.npy"):
            db_file.unlink()

        self.data_path = ""
        self.save_path = ""
        self.opslist = []
        self.batch = False
        self.f = 0
        self.create_buttons()

    def reset_ops(self) -> None:
        """Reload ops from disk and merge with built-in defaults."""
        self.ops = np.load(self.opsfile, allow_pickle=True).item()
        ops0 = generate_default_ops()
        self.ops = {**ops0, **self.ops}
        if hasattr(self, "editlist"):
            for k in range(len(self.editlist)):
                self.editlist[k].set_text(self.ops)

    def create_buttons(self) -> None:
        """Build all parameter editing widgets, section labels, and action buttons."""
        self.intkeys = [
            "nplanes",
            "nchannels",
            "functional_chan",
            "align_by_chan",
            "nimg_init",
            "batch_size",
            "max_iterations",
            "nbinned",
            "inner_neuropil_border_radius",
            "minimum_neuropil_pixels",
            "spatial_scale",
            "do_registration",
        ]
        self.boolkeys = [
            "compute_bidirectional_phase_offset",
            "one_p_reg",
            "nonrigid",
            "roidetect",
            "extract_neuropil",
            "extract_spikes",
            "keep_movie_raw",
            "allow_overlap",
        ]
        self.stringkeys = []
        tifkeys = [
            "nplanes",
            "nchannels",
            "functional_chan",
            "tau",
            "fs",
            "compute_bidirectional_phase_offset",
            "bidirectional_phase_offset",
            "ignore_flyback",
        ]
        outkeys = [
            "preclassification_threshold",
            "aspect_ratio",
        ]
        regkeys = [
            "do_registration",
            "align_by_chan",
            "nimg_init",
            "batch_size",
            "smooth_sigma",
            "smooth_sigma_time",
            "maxregshift",
            "th_badframes",
            "keep_movie_raw",
            "two_step_registration",
        ]
        nrkeys = [
            ["nonrigid", "block_size", "snr_thresh", "maxregshiftNR"],
            ["one_p_reg", "spatial_hp_reg", "pre_smooth", "spatial_taper"],
        ]
        cellkeys = [
            "roidetect",
            "denoise",
            "spatial_scale",
            "threshold_scaling",
            "maximum_overlap",
            "maximum_iterations",
            "high_pass",
            "spatial_hp_detect",
        ]
        neudeconvkeys = [
            ["extract_neuropil", "allow_overlap", "inner_neuropil_border_radius", "minimum_neuropil_pixels"],
            ["crop_to_soma", "extract_spikes", "baseline_window", "baseline_sigma", "neuropil_coefficient"],
        ]
        keys = [tifkeys, outkeys, regkeys, nrkeys, cellkeys, neudeconvkeys]
        labels = [
            "Main settings",
            "Output settings",
            "Registration",
            ["Nonrigid", "1P"],
            "ROI Detection",
            ["Extraction/Neuropil", "Classify/Deconv"],
        ]
        tooltips = [
            "each tiff has this many planes in sequence",
            "each tiff has this many channels per plane",
            "this channel is used to extract functional ROIs (1-based)",
            "timescale of sensor in deconvolution (in seconds)",
            "sampling rate (per plane)",
            "whether or not to compute bidirectional phase offset of recording (from line scanning)",
            "set a fixed number (in pixels) for the bidirectional phase offset",
            "process each plane with a separate job on a computing cluster",
            "ignore flyback planes 0-indexed separated by a comma e.g. '0,10'; "
            "'-1' means no planes ignored so all planes processed",
            "apply ROI classifier before signal extraction with probability threshold (set to 0 to turn off)",
            "um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)",
            "if 1, registration is performed if it wasn't performed already",
            "when multi-channel, you can align by non-functional channel (1-based)",
            "# of subsampled frames for finding reference image",
            "number of frames per batch",
            "gaussian smoothing after phase corr: 1.15 good for 2P recordings, recommend 2-5 for 1P recordings",
            "gaussian smoothing in time, useful for low SNR data",
            "max allowed registration shift, as a fraction of frame max(width and height)",
            "this parameter determines which frames to exclude when determining cropped frame size "
            "- set it smaller to exclude more frames",
            "if 1, unregistered binary is kept in a separate file data_raw.bin",
            "run registration twice (useful if data is really noisy), *keep_movie_raw must be 1*",
            "whether to use nonrigid registration (splits FOV into blocks of size block_size)",
            "block size in number of pixels in Y and X (two numbers separated by a comma)",
            "if any nonrigid block is below this threshold, it gets smoothed until above this threshold. "
            "1.0 results in no smoothing",
            "maximum *pixel* shift allowed for nonrigid, relative to rigid",
            "whether to perform high-pass filtering and tapering for registration (necessary for 1P recordings)",
            "window for spatial high-pass filtering before registration",
            "whether to smooth before high-pass filtering before registration",
            "how much to ignore on edges (important for vignetted windows, "
            "for FFT padding do not set BELOW 3*smooth_sigma)",
            "if 1, run cell (ROI) detection",
            "if 1, run PCA denoising on binned movie to improve cell detection",
            "choose size of ROIs: 0 = multi-scale; 1 = 6 pixels, 2 = 12, 3 = 24, 4 = 48",
            "adjust the automatically determined threshold for finding ROIs by this scalar multiplier",
            "ROIs with greater than this overlap as a fraction of total pixels will be discarded",
            "maximum number of iterations for ROI detection",
            "temporal running mean subtraction with window of size 'high_pass' (use low values for 1P)",
            "spatial high-pass filter size (used to remove spatially-correlated neuropil)",
            "whether or not to extract neuropil; if 0, Fneu is set to 0",
            "allow shared pixels to be used for fluorescence extraction from overlapping ROIs "
            "(otherwise excluded from both ROIs)",
            "number of pixels between ROI and neuropil donut",
            "minimum number of pixels in the neuropil",
            "if 1, crop dendrites for cell classification stats like compactness",
            "if 1, run spike detection (deconvolution)",
            "window for maximin",
            "smoothing constant for gaussian filter",
            "neuropil coefficient",
        ]

        section_font = header_font()
        qlabel = QLabel("File paths")
        qlabel.setFont(section_font)
        self.layout.addWidget(qlabel, 0, 0, 1, 1)
        load_ops_button = QPushButton("Load ops file")
        load_ops_button.clicked.connect(self.load_ops)
        save_default_button = QPushButton("Save ops as default")
        save_default_button.clicked.connect(self.save_default_ops)
        revert_default_button = QPushButton("Revert default ops to built-in")
        revert_default_button.clicked.connect(self.revert_default_ops)
        save_ops_button = QPushButton("Save ops to file")
        save_ops_button.clicked.connect(self.save_ops)
        self.layout.addWidget(load_ops_button, 0, 4, 1, 2)
        self.layout.addWidget(save_default_button, 1, 4, 1, 2)
        self.layout.addWidget(revert_default_button, 2, 4, 1, 2)
        self.layout.addWidget(save_ops_button, 3, 4, 1, 2)
        self.layout.addWidget(QLabel(""), 4, 4, 1, 2)
        self.layout.addWidget(QLabel("Load example ops"), 5, 4, 1, 2)
        self.opsbtns = QButtonGroup(self)
        opsstr = ["1P imaging", "dendrites/axons"]
        self.opsname = ["1P", "dendrite"]
        for b in range(len(opsstr)):
            btn = OpsButton(b, opsstr[b], self)
            self.opsbtns.addButton(btn, b)
            self.layout.addWidget(btn, 6 + b, 4, 1, 2)
        self.keylist = []
        self.editlist = []
        tooltip_index = 0
        widget_index = 0
        for section_index, section_keys in enumerate(keys):
            row = 0
            if type(labels[section_index]) is list:
                section_labels = labels[section_index]
                sub_keys = section_keys
            else:
                section_labels = [labels[section_index]]
                sub_keys = [section_keys]
            for sub_section, label in enumerate(section_labels):
                qlabel = QLabel(label)
                qlabel.setFont(section_font)
                self.layout.addWidget(qlabel, row * 2, 2 * (section_index + 4), 1, 2)
                row += 1
                for key in sub_keys[sub_section]:
                    if self.ops[key] or (self.ops[key] == 0) or len(self.ops[key]) == 0:
                        qedit = LineEdit(widget_index, key, self)
                        qlabel = QLabel(key)
                        qlabel.setToolTip(tooltips[tooltip_index])
                        qedit.set_text(self.ops)
                        qedit.setToolTip(tooltips[tooltip_index])
                        qedit.setFixedWidth(90)
                        self.layout.addWidget(qlabel, row * 2 - 1, 2 * (section_index + 4), 1, 2)
                        self.layout.addWidget(qedit, row * 2, 2 * (section_index + 4), 1, 2)
                        self.keylist.append(key)
                        self.editlist.append(qedit)
                        widget_index += 1
                    row += 1
                    tooltip_index += 1

        # data_path
        key = "input_format"
        qlabel = QLabel(key)
        qlabel.setFont(section_font)
        qlabel.setToolTip("File format (selects which parser to use)")
        self.layout.addWidget(qlabel, 1, 0, 1, 1)
        self.inputformat = QComboBox()
        [self.inputformat.addItem(f) for f in ["tiff", "binary", "mesoscan"]]
        self.inputformat.currentTextChanged.connect(self.parse_inputformat)
        self.layout.addWidget(self.inputformat, 2, 0, 1, 1)

        key = "look_one_level_down"
        qlabel = QLabel(key)
        qlabel.setToolTip("(deprecated) files are now always searched recursively")
        self.layout.addWidget(qlabel, 3, 0, 1, 1)
        qedit = LineEdit(widget_index, key, self)
        qedit.set_text(self.ops)
        qedit.setFixedWidth(95)
        self.layout.addWidget(qedit, 4, 0, 1, 1)
        self.keylist.append(key)
        self.editlist.append(qedit)

        cw = 4
        self.btiff = QPushButton("Add directory to data_path")
        self.btiff.clicked.connect(self.get_folders)
        self.layout.addWidget(self.btiff, 5, 0, 1, cw)
        qlabel = QLabel("data_path")
        qlabel.setFont(section_font)
        self.layout.addWidget(qlabel, 6, 0, 1, 1)
        self.qdata = []
        for n in range(9):
            self.qdata.append(QLabel(""))
            self.layout.addWidget(self.qdata[n], n + 7, 0, 1, cw)

        self.bsave = QPushButton("Add save_path (default is data_path)")
        self.bsave.clicked.connect(self.save_folder)
        self.layout.addWidget(self.bsave, 16, 0, 1, cw)
        self.savelabel = QLabel("")
        self.layout.addWidget(self.savelabel, 17, 0, 1, cw)
        self.runButton = QPushButton("RUN SUITE2P")
        self.runButton.clicked.connect(self.run_suite2p)
        n0 = 22
        self.layout.addWidget(self.runButton, n0, 0, 1, 1)
        self.runButton.setEnabled(False)
        self.textEdit = QTextEdit()
        self.layout.addWidget(self.textEdit, n0 + 1, 0, 30, 2 * section_index)
        self.textEdit.setFixedHeight(300)
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdout_write)
        self.process.readyReadStandardError.connect(self.stderr_write)
        # disable the button when running the s2p process
        self.process.started.connect(self.started)
        self.process.finished.connect(self.finished)
        # stop process
        self.stopButton = QPushButton("STOP")
        self.stopButton.setEnabled(False)
        self.layout.addWidget(self.stopButton, n0, 1, 1, 1)
        self.stopButton.clicked.connect(self.stop)
        # cleanup button
        self.cleanButton = QPushButton("Add a clean-up *.py")
        self.cleanButton.setToolTip("will run at end of processing")
        self.cleanButton.setEnabled(True)
        self.layout.addWidget(self.cleanButton, n0, 2, 1, 2)
        self.cleanup = False
        self.cleanButton.clicked.connect(self.clean_script)
        self.cleanLabel = QLabel("")
        self.layout.addWidget(self.cleanLabel, n0, 4, 1, 12)
        self.listOps = QPushButton("save settings and\n add more (batch)")
        self.listOps.clicked.connect(self.add_batch)
        self.layout.addWidget(self.listOps, n0, 12, 1, 2)
        self.listOps.setEnabled(False)
        self.removeOps = QPushButton("remove last added")
        self.removeOps.clicked.connect(self.remove_ops)
        self.layout.addWidget(self.removeOps, n0, 14, 1, 2)
        self.removeOps.setEnabled(False)
        self.odata = []
        self.n_batch = 15
        for n in range(self.n_batch):
            self.odata.append(QLabel(""))
            self.layout.addWidget(self.odata[n], n0 + 1 + n, 12, 1, 4)

    def remove_ops(self) -> None:
        """Remove the most recently added batch ops entry."""
        count = len(self.opslist)
        if count == 1:
            self.batch = False
            self.opslist = []
            self.removeOps.setEnabled(False)
        else:
            del self.opslist[count - 1]
        self.odata[count - 1].setText("")
        self.odata[count - 1].setToolTip("")
        self.f = 0

    def add_batch(self) -> None:
        """Save current settings as a batch entry and reset file fields."""
        self.add_ops()
        count = len(self.opslist)
        self.odata[count].setText(self.datastr)
        self.odata[count].setToolTip(self.datastr)

        # clear file fields
        self.db = {}
        self.data_path = ""
        self.save_path = ""
        for n in range(self.n_batch):
            self.qdata[n].setText("")
        self.savelabel.setText("")

        # Enables all the file loaders again.
        self.btiff.setEnabled(True)
        self.bsave.setEnabled(True)
        # and enable the run button
        self.runButton.setEnabled(True)
        self.removeOps.setEnabled(True)
        self.listOps.setEnabled(False)

    def add_ops(self) -> None:
        """Compile current ops and database, then save to disk."""
        self.f = 0
        self.compile_ops_db()
        count = len(self.opslist)
        np.save(self.ops_path / f"ops{count}.npy", self.ops)
        np.save(self.ops_path / f"db{count}.npy", self.db)
        self.opslist.append(f"ops{count}.npy")

    def compile_ops_db(self) -> None:
        """Gather current widget values into the ops dictionary and build the database dict."""
        for k, key in enumerate(self.keylist):
            self.ops[key] = self.editlist[k].get_text(self.intkeys, self.boolkeys, self.stringkeys)
        self.db = {}
        self.db["data_path"] = self.data_path
        self.datastr = self.data_path

        # add save_path
        if len(self.save_path) == 0:
            self.save_path = self.db["data_path"]
        self.db["save_path"] = self.save_path
        self.db["input_format"] = self.inputformat.currentText()

    def run_suite2p(self) -> None:
        """Launch the suite2p pipeline as an external process."""
        if not self.opslist:
            self.add_ops()
        self.finish = True
        self.error = False
        ops_file = self.ops_path / "ops.npy"
        db_file = self.ops_path / "db.npy"
        shutil.copy(self.ops_path / f"ops{self.f}.npy", ops_file)
        shutil.copy(self.ops_path / f"db{self.f}.npy", db_file)
        self.db = np.load(db_file, allow_pickle=True).item()
        console.echo(message=f"Parameter overrides: {self.db}")
        console.echo(message="Running suite2p with command:")
        cmd = f"-u -W ignore -m suite2p --ops {ops_file} --db {db_file}"
        console.echo(message=f"python {cmd}")
        self.process.start(sys.executable, cmd.split(" "))

    def stop(self) -> None:
        """Terminate the running suite2p process."""
        self.finish = False
        self.logfile.close()
        self.process.kill()

    def started(self) -> None:
        """Handle process start by disabling controls and opening the log file."""
        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.cleanButton.setEnabled(False)
        save_folder = Path(self.db["save_path"]) / "suite2p"
        save_folder.mkdir(parents=True, exist_ok=True)
        self.logfile = (save_folder / "run.log").open(mode="a")
        timestamp = datetime.now(tz=UTC).strftime("%d/%m/%Y %H:%M:%S")
        self.logfile.write(f"\n >>>>> started run at {timestamp}")

    def finished(self) -> None:
        """Handle process completion, load results or continue batch processing."""
        self.logfile.close()
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        if self.finish and not self.error:
            self.cleanButton.setEnabled(True)
            if len(self.opslist) == 1:
                stat_path = Path(self.db["save_path"]) / "suite2p" / "plane0" / "stat.npy"
                self.parent.fname = str(stat_path)
                if stat_path.exists():
                    cursor.insertText("Opening in GUI (can close this window)\n")
                    load_proc(self.parent)
                else:
                    cursor.insertText("not opening plane in GUI (no ROIs)\n")
            else:
                remaining = len(self.opslist) - self.f - 1
                cursor.insertText(f"BATCH MODE: {remaining} more recordings remaining \n")
                self.f += 1
                if self.f < len(self.opslist):
                    self.run_suite2p()
        elif not self.error:
            cursor.insertText("Interrupted by user (not finished)\n")
        else:
            cursor.insertText("Interrupted by error (not finished)\n")

        # remove current ops from processing list
        if len(self.opslist) == 1:
            del self.opslist[0]

    def save_ops(self) -> None:
        """Save the current ops to a user-selected file."""
        name = QFileDialog.getSaveFileName(self, "Ops name (*.npy)")
        name = name[0]
        self.save_text()
        if name:
            np.save(name, self.ops)
            console.echo(message=f"Saved current settings to {name}", level=LogLevel.SUCCESS)

    def save_default_ops(self) -> None:
        """Persist the current GUI settings as the default ops file."""
        name = self.opsfile
        ops = self.ops.copy()
        self.ops = generate_default_ops()
        self.save_text()
        np.save(name, self.ops)
        self.ops = ops
        console.echo(message="Saved current settings in GUI as default ops", level=LogLevel.SUCCESS)

    def revert_default_ops(self) -> None:
        """Reset the default ops file to the built-in ops values."""
        name = self.opsfile
        self.ops = generate_default_ops()
        np.save(name, self.ops)
        self.load_ops(name)
        console.echo(message="Reverted default ops to built-in ops", level=LogLevel.SUCCESS)

    def save_text(self) -> None:
        """Read all widget values back into the ops dictionary."""
        for k in range(len(self.editlist)):
            key = self.keylist[k]
            self.ops[key] = self.editlist[k].get_text(self.intkeys, self.boolkeys, self.stringkeys)

    def load_ops(self, name: str | None = None) -> None:
        """Load ops from a npy or json file and populate the dialog widgets."""
        console.echo(message="Loading ops...")
        if not (isinstance(name, str) and len(name) > 0):
            name = QFileDialog.getOpenFileName(self, "Open ops file (npy or json)")
            name = name[0]

        if len(name) > 0:
            ext = Path(name).suffix
            try:
                if ext == ".npy":
                    ops = np.load(name, allow_pickle=True).item()
                elif ext == ".json":
                    with Path(name).open() as file:
                        ops = json.load(file)
                ops0 = generate_default_ops()
                ops = {**ops0, **ops}
                for key in ops:
                    if key not in {"data_path", "save_path", "cleanup"}:
                        if key in self.keylist:
                            self.editlist[self.keylist.index(key)].set_text(ops)
                        self.ops[key] = ops[key]
                if "input_format" not in self.ops:
                    self.ops["input_format"] = "tiff"
                if "data_path" in ops and len(ops["data_path"]) > 0:
                    self.data_path = ops["data_path"]
                    self.qdata[0].setText(self.data_path)
                    for n in range(1, 9):
                        self.qdata[n].setText("")
                    self.runButton.setEnabled(True)
                    self.btiff.setEnabled(True)
                    self.listOps.setEnabled(True)
                self.inputformat.currentTextChanged.connect(lambda x: x)
                self.inputformat.setCurrentText(self.ops["input_format"])
                self.inputformat.currentTextChanged.connect(self.parse_inputformat)
                if self.ops["input_format"] == "sbx":
                    self.runButton.setEnabled(True)
                    self.btiff.setEnabled(False)
                    self.listOps.setEnabled(True)

                if "save_path" in ops and len(ops["save_path"]) > 0:
                    self.save_path = ops["save_path"]
                    self.savelabel.setText(self.save_path)
                if "clean_script" in ops and len(ops["clean_script"]) > 0:
                    self.ops["clean_script"] = ops["clean_script"]
                    self.cleanLabel.setText(ops["clean_script"])

            except Exception as e:
                console.echo(message="Could not load ops file", level=LogLevel.ERROR)
                console.echo(message=f"Error details: {e}", level=LogLevel.ERROR)

    def load_db(self) -> None:
        """Load parameter overrides from disk."""
        console.echo(message="Loading parameter overrides...")

    def stdout_write(self) -> None:
        """Append standard output from the subprocess to the text area and log file."""
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        output = str(self.process.readAllStandardOutput(), "utf-8")
        cursor.insertText(output)
        self.textEdit.ensureCursorVisible()
        self.logfile.write(output)

    def stderr_write(self) -> None:
        """Append standard error from the subprocess to the text area and log file."""
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(">>>ERROR<<<\n")
        output = str(self.process.readAllStandardError(), "utf-8")
        cursor.insertText(output)
        self.textEdit.ensureCursorVisible()
        self.error = True
        self.logfile.write(">>>ERROR<<<\n")
        self.logfile.write(output)

    def clean_script(self) -> None:
        """Select a Python cleanup script to run after processing completes."""
        name = QFileDialog.getOpenFileName(self, "Open clean up file", filter="*.py")
        name = name[0]
        if name:
            self.cleanup = True
            self.cleanScript = name
            self.cleanLabel.setText(name)
            self.ops["clean_script"] = name

    def get_folders(self) -> None:
        """Prompt the user to select a directory and add it to the data path list."""
        name = QFileDialog.getExistingDirectory(self, "Add directory to data path")
        if len(name) > 0:
            self.data_path.append(name)
            self.qdata[len(self.data_path) - 1].setText(name)
            self.qdata[len(self.data_path) - 1].setToolTip(name)
            self.runButton.setEnabled(True)
            self.listOps.setEnabled(True)

    def get_h5py(self) -> None:
        """Open the HDF5 key chooser dialog and store the selected key."""
        chooser = TextChooser(self)
        result = chooser.exec()
        if result:
            self.h5_key = chooser.h5_key
        else:
            self.h5_key = "data"

    def parse_inputformat(self) -> None:
        """Handle input format selection changes and trigger format-specific dialogs."""
        inputformat = self.inputformat.currentText()
        console.echo(message=f"Input format: {inputformat}")
        if inputformat == "h5":
            # replace functionality of "old" button
            self.get_h5py()
        else:
            pass

    def save_folder(self) -> None:
        """Prompt the user to select a save directory for output data."""
        name = QFileDialog.getExistingDirectory(self, "Save folder for data")
        if len(name) > 0:
            self.save_path = name
            self.savelabel.setText(name)
            self.savelabel.setToolTip(name)


class LineEdit(QLineEdit):
    """Parameter line editor that converts between display text and typed ops values."""

    def __init__(self, _index: int, key: str, parent: QWidget | None = None) -> None:
        """Initialize the line editor for the given ops key."""
        super().__init__(parent)
        self.key = key

    def get_text(
        self, intkeys: list[str], boolkeys: list[str], stringkeys: list[str]
    ) -> int | float | bool | str | list[int]:
        """Parse the widget text into the appropriate Python type for this ops key."""
        key = self.key
        if key in {"cell_diameter", "block_size"}:
            diams = self.text().replace(" ", "").split(",")
            okey = [int(diams[0]), int(diams[1])] if len(diams) > 1 else int(diams[0])
        elif key == "ignore_flyback":
            okey = self.text().replace(" ", "").split(",")
            for i in range(len(okey)):
                okey[i] = int(okey[i])
            if len(okey) == 1 and okey[0] == -1:
                okey = []
        elif key in intkeys:
            okey = int(float(self.text()))
        elif key in boolkeys:
            okey = bool(int(self.text()))
        elif key in stringkeys:
            okey = self.text()
        else:
            okey = float(self.text())
        return okey

    def set_text(self, ops: dict[str, object]) -> None:
        """Format and display the ops value for this key in the widget."""
        key = self.key
        if key in {"cell_diameter", "block_size"}:
            if (type(ops[key]) is not int) and (len(ops[key]) > 1):
                dstr = str(int(ops[key][0])) + ", " + str(int(ops[key][1]))
            else:
                dstr = str(int(ops[key]))
        elif key == "ignore_flyback":
            if not isinstance(ops[key], (list, np.ndarray)):
                ops[key] = [ops[key]]
            if len(ops[key]) == 0:
                dstr = "-1"
            else:
                dstr = ""
                for i in ops[key]:
                    dstr += str(int(i))
                    if i < len(ops[key]) - 1:
                        dstr += ", "
        elif type(ops[key]) is not bool:
            dstr = str(ops[key])
        else:
            dstr = str(int(ops[key]))
        self.setText(dstr)


class OpsButton(QPushButton):
    """Push button that loads a preset ops configuration when clicked."""

    def __init__(self, bid: int, text: str, parent: RunWindow | None = None) -> None:
        """Initialize the ops preset button with the given label and button identifier."""
        super().__init__(parent)
        self.setText(text)
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent: RunWindow, bid: int) -> None:
        """Load a preset ops file and apply its values to the parent dialog."""
        try:
            ops_path = Path(__file__).resolve().parent.parent / "ops" / f"ops_{parent.opsname[bid]}.npy"
            ops = np.load(ops_path, allow_pickle=True).item()
            for key in ops:
                if key in parent.keylist:
                    parent.editlist[parent.keylist.index(key)].set_text(ops)
                    parent.ops[key] = ops[key]
        except Exception as error:
            console.echo(message="Could not load ops file", level=LogLevel.ERROR)
            console.echo(message=f"Error details: {error}", level=LogLevel.ERROR)


class VerticalLabel(QWidget):
    """Widget that renders text rotated 90 degrees clockwise."""

    def __init__(self, text: str | None = None) -> None:
        """Initialize the vertical label with the given text."""
        super().__init__()
        self.text = text

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Draw the label text rotated 90 degrees."""
        painter = QtGui.QPainter(self)
        painter.setPen(QtCore.Qt.GlobalColor.white)
        painter.translate(0, 0)
        painter.rotate(90)
        if self.text:
            painter.drawText(0, 0, self.text)
        painter.end()
