from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent, QFont, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from controllers.segmentation_controller import SegmentationController, SegmentationRunConfig
from controllers.thermal_camera_controller import ThermalCameraConfig, ThermalCameraController
from result_reader import AlgorithmResult, read_analysis_results


def _find_project_root(start: Path) -> Path:
    """Locate the repository root without depending on a user-specific path."""
    for candidate in (start, *start.parents):
        if (candidate / "CMakeLists.txt").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate the project root from the GUI source path.")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
DEFAULT_CAPTURE_ROOT = PROJECT_ROOT / "gui_captures"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "gui_results_live"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / ".gui_runtime" / "live_preview"
DEFAULT_ADAPTER = PROJECT_ROOT / "repo" / "src" / "tools" / "capture_thermal.py"
def _first_existing(candidates: list[Path]) -> Path | None:
    """Return the first existing candidate while preserving its written path.

    Important: virtual-environment launchers such as ``venv/bin/python`` are often
    symbolic links to ``/usr/bin/python3.x``.  Calling ``resolve()`` here would
    replace the useful venv path with the system interpreter path and would make
    the GUI appear to use ``/usr/bin/python3.x`` even though the SDK venv exists.
    """
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.exists():
            return candidate.absolute()
    return None


def _is_virtualenv_python(path: Path | None) -> bool:
    """Return True when the *path itself* points to venv/bin/python.

    Do not resolve symlinks here. On Raspberry Pi, ``venv/bin/python`` commonly
    points to ``/usr/bin/python3.11``; resolving it would incorrectly classify the
    valid pysenxor environment as the system interpreter.
    """
    if path is None:
        return False
    candidate = path.expanduser()
    return (
        candidate.is_file()
        and candidate.parent.name == "bin"
        and candidate.parent.parent.name in {"venv", ".venv"}
    )


def _discover_mi48_paths() -> tuple[Path | None, Path | None]:
    """Discover a pysenxor installation without assuming a username or fixed home path.

    Explicit environment variables have priority, followed by a small set of common
    locations below the current user's home directory. The GUI still allows manual
    selection when the SDK is installed elsewhere.
    """
    home = Path.home()
    script_env = os.environ.get("MI48_CAPTURE_SCRIPT", "").strip()
    python_env = os.environ.get("MI48_PYTHON", "").strip()
    root_env = os.environ.get("PYSENXOR_ROOT", "").strip()

    roots: list[Path] = []
    if root_env:
        roots.append(Path(root_env))
    roots.extend([
        home / "Downloads" / "thermal_demo" / "pysenxor-master",
        home / "thermal_demo" / "pysenxor-master",
        home / "pysenxor-master",
        home / "pysenxor",
    ])

    script_candidates: list[Path] = []
    python_candidates: list[Path] = []
    if script_env:
        script_candidates.append(Path(script_env))
    if python_env:
        python_candidates.append(Path(python_env))
    for root in roots:
        script_candidates.extend([
            root / "example" / "capture_stream_for_segm.py",
            root / "examples" / "capture_stream_for_segm.py",
        ])
        python_candidates.extend([
            root / "venv" / "bin" / "python",
            root / ".venv" / "bin" / "python",
        ])

    script = _first_existing(script_candidates)
    py = _first_existing(python_candidates)
    return script, py


DISCOVERED_MI48_SCRIPT, DISCOVERED_MI48_PYTHON = _discover_mi48_paths()


class ImagePreview(QLabel):
    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(250, 220)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("imagePreview")

    def set_image(self, path: Path | None) -> None:
        self._pixmap = None
        if path and path.exists():
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                self._pixmap = pixmap
        if self._pixmap is None:
            self.clear_image(self._placeholder)
        else:
            self._refresh()

    def clear_image(self, placeholder: str | None = None) -> None:
        if placeholder is not None:
            self._placeholder = placeholder
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(self._placeholder)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if not self._pixmap:
            return
        scaled = self._pixmap.scaled(
            self.contentsRect().size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setText("")
        self.setPixmap(scaled)


class MainWindow(QMainWindow):
    ALGORITHMS = [
        ("All algorithms (7)", "ALL"),
        ("DESSA", "DESSA"),
        ("DE", "DE"),
        ("SDE", "SDE"),
        ("GWO", "GWO"),
        ("PSO", "PSO"),
        ("CSA", "CSA"),
        ("FA", "FA"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Thermal Multithreshold Segmentation — GUI v0.2")
        self.resize(1380, 860)

        self.settings = QSettings("Rpi5ThermalSegmentation", "ThermalGUI")
        self.current_capture: Path | None = None
        self._analysis_results: dict[str, AlgorithmResult] = {}
        self._last_output_dir: Path | None = None
        self._analyze_after_capture = False

        self.camera_controller = ThermalCameraController(DEFAULT_RUNTIME_ROOT, self)
        self.camera_controller.output_received.connect(self._append_console)
        self.camera_controller.frame_ready.connect(self._on_live_frame)
        self.camera_controller.capture_saved.connect(self._on_capture_saved)
        self.camera_controller.camera_error.connect(self._on_camera_error)
        self.camera_controller.preview_state_changed.connect(self._on_preview_state_changed)
        self.camera_controller.busy_changed.connect(self._on_camera_busy_changed)

        self.segmentation_controller = SegmentationController(self)
        self.segmentation_controller.output_received.connect(self._append_console)
        self.segmentation_controller.run_started.connect(self._on_run_started)
        self.segmentation_controller.run_finished.connect(self._on_run_finished)
        self.segmentation_controller.run_failed.connect(self._on_run_failed)
        self.segmentation_controller.progress_changed.connect(self._on_progress_changed)

        self._build_ui()
        self._apply_style()
        self._set_defaults()
        self._restore_settings()
        self._update_camera_fields()
        self._update_command_preview()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 12, 14, 12)
        root_layout.setSpacing(10)

        title = QLabel("Thermal Image Multithreshold Segmentation")
        title_font = QFont()
        title_font.setPointSize(17)
        title_font.setBold(True)
        title.setFont(title_font)
        subtitle = QLabel(
            "GUI v0.2 · Thermal camera → live preview → single capture → analysis with one or all seven metaheuristics."
        )
        subtitle.setObjectName("subtitle")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        # Left settings panel is scrollable so the GUI remains usable at 768p.
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_scroll.setMinimumWidth(405)
        settings_scroll.setMaximumWidth(535)
        left = QWidget()
        settings_scroll.setWidget(left)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        camera_group = QGroupBox("1. Thermal camera")
        camera_form = QFormLayout(camera_group)

        self.camera_mode_combo = QComboBox()
        self.camera_mode_combo.addItem("Waveshare / MI48 (SPI-I2C)", "mi48")
        self.camera_mode_combo.addItem("USB / UVC (/dev/video*)", "uvc")
        self.camera_mode_combo.currentIndexChanged.connect(self._update_camera_fields)

        self.adapter_edit, adapter_row = self._path_row(self._browse_adapter)
        self.mi48_script_edit, mi48_script_row = self._path_row(self._browse_mi48_script)
        self.mi48_python_edit, mi48_python_row = self._path_row(self._browse_mi48_python)

        self.camera_index_spin = self._spin(0, 20, 0)
        self.preview_interval_spin = self._spin(100, 10000, 750)
        self.preview_interval_spin.setSuffix(" ms")

        camera_form.addRow("Type:", self.camera_mode_combo)
        camera_form.addRow("Adapter:", adapter_row)
        self.mi48_script_label = QLabel("Script MI48:")
        self.mi48_python_label = QLabel("Python pysenxor:")
        self.uvc_index_label = QLabel("UVC index:")
        camera_form.addRow(self.mi48_script_label, mi48_script_row)
        camera_form.addRow(self.mi48_python_label, mi48_python_row)
        camera_form.addRow(self.uvc_index_label, self.camera_index_spin)
        camera_form.addRow("Refresh interval:", self.preview_interval_spin)

        self.camera_status_label = QLabel("Camera stopped")
        self.camera_status_label.setObjectName("cameraStatus")
        camera_form.addRow("Status:", self.camera_status_label)

        camera_buttons = QWidget()
        camera_buttons_layout = QHBoxLayout(camera_buttons)
        camera_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.start_camera_button = QPushButton("Start live view")
        self.start_camera_button.clicked.connect(self._start_live_preview)
        self.stop_camera_button = QPushButton("Stop")
        self.stop_camera_button.setEnabled(False)
        self.stop_camera_button.clicked.connect(self.camera_controller.stop_preview)
        camera_buttons_layout.addWidget(self.start_camera_button, 1)
        camera_buttons_layout.addWidget(self.stop_camera_button)
        camera_form.addRow("", camera_buttons)

        capture_buttons = QWidget()
        capture_layout = QHBoxLayout(capture_buttons)
        capture_layout.setContentsMargins(0, 0, 0, 0)
        self.capture_button = QPushButton("Capture frame")
        self.capture_button.setObjectName("captureButton")
        self.capture_button.clicked.connect(self._capture_once)
        self.capture_analyze_button = QPushButton("Capture and analyze")
        self.capture_analyze_button.clicked.connect(self._capture_and_analyze)
        capture_layout.addWidget(self.capture_button)
        capture_layout.addWidget(self.capture_analyze_button, 1)
        camera_form.addRow("", capture_buttons)

        self.open_image_button = QPushButton("Use existing image…")
        self.open_image_button.setToolTip("Useful for testing the GUI without connecting the camera.")
        self.open_image_button.clicked.connect(self._browse_existing_image)
        camera_form.addRow("Offline test:", self.open_image_button)

        left_layout.addWidget(camera_group)

        analysis_group = QGroupBox("2. Multithreshold segmentation")
        analysis_form = QFormLayout(analysis_group)
        self.executable_edit, exe_row = self._path_row(self._browse_executable)
        self.output_edit, output_row = self._path_row(self._browse_output)

        self.algorithm_combo = QComboBox()
        for label, code in self.ALGORITHMS:
            self.algorithm_combo.addItem(label, code)
        self.algorithm_combo.currentIndexChanged.connect(self._update_command_preview)

        self.levels_spin = self._spin(1, 10, 3)
        self.runs_spin = self._spin(1, 10000, 40)
        self.population_spin = self._spin(2, 10000, 30)
        self.iterations_spin = self._spin(1, 100000, 50)
        for spin in (self.levels_spin, self.runs_spin, self.population_spin, self.iterations_spin):
            spin.valueChanged.connect(self._update_command_preview)

        self.fsim_check = QCheckBox("Compute FSIM")
        self.fsim_check.setChecked(True)
        self.fsim_check.toggled.connect(self._update_command_preview)

        analysis_form.addRow("segm executable:", exe_row)
        analysis_form.addRow("Results:", output_row)
        analysis_form.addRow("Algorithm:", self.algorithm_combo)
        analysis_form.addRow("Levels / thresholds:", self.levels_spin)
        analysis_form.addRow("Runs:", self.runs_spin)
        analysis_form.addRow("Population:", self.population_spin)
        analysis_form.addRow("Iterations:", self.iterations_spin)
        analysis_form.addRow("Metrics:", self.fsim_check)

        self.analyze_button = QPushButton("Analyze captured frame")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._run_analysis)
        self.stop_analysis_button = QPushButton("Stop analysis")
        self.stop_analysis_button.setEnabled(False)
        self.stop_analysis_button.clicked.connect(self.segmentation_controller.stop)
        analysis_actions = QWidget()
        analysis_actions_layout = QHBoxLayout(analysis_actions)
        analysis_actions_layout.setContentsMargins(0, 0, 0, 0)
        analysis_actions_layout.addWidget(self.analyze_button, 1)
        analysis_actions_layout.addWidget(self.stop_analysis_button)
        analysis_form.addRow("", analysis_actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Ready")
        analysis_form.addRow("Progress:", self.progress)

        left_layout.addWidget(analysis_group)

        command_group = QGroupBox("3. Equivalent analysis command")
        command_layout = QVBoxLayout(command_group)
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumHeight(110)
        self.command_preview.setObjectName("commandPreview")
        command_layout.addWidget(self.command_preview)
        left_layout.addWidget(command_group)
        left_layout.addStretch(1)

        # Right content area.
        self.tabs = QTabWidget()

        camera_tab = QWidget()
        camera_tab_layout = QHBoxLayout(camera_tab)
        camera_tab_layout.setContentsMargins(8, 8, 8, 8)
        live_col = QVBoxLayout()
        captured_col = QVBoxLayout()
        live_title = QLabel("Live thermal view")
        live_title.setObjectName("previewTitle")
        capture_title = QLabel("Frame selected for analysis")
        capture_title.setObjectName("previewTitle")
        self.live_preview = ImagePreview("Click ‘Start live view’ to acquire images")
        self.captured_preview = ImagePreview("No frame has been captured yet")
        live_col.addWidget(live_title)
        live_col.addWidget(self.live_preview, 1)
        captured_col.addWidget(capture_title)
        captured_col.addWidget(self.captured_preview, 1)
        camera_tab_layout.addLayout(live_col, 1)
        camera_tab_layout.addLayout(captured_col, 1)
        self.tabs.addTab(camera_tab, "Camera")

        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Displayed result:"))
        self.result_algorithm_combo = QComboBox()
        self.result_algorithm_combo.setEnabled(False)
        self.result_algorithm_combo.currentTextChanged.connect(self._show_selected_algorithm_result)
        selector_row.addWidget(self.result_algorithm_combo)
        selector_row.addStretch(1)
        results_layout.addLayout(selector_row)

        images_row = QHBoxLayout()
        result_original_col = QVBoxLayout()
        result_segmented_col = QVBoxLayout()
        ro_title = QLabel("Original capture")
        rs_title = QLabel("Multithresholded image")
        ro_title.setObjectName("previewTitle")
        rs_title.setObjectName("previewTitle")
        self.result_original_preview = ImagePreview("No capture")
        self.segmented_preview = ImagePreview("Run analysis")
        result_original_col.addWidget(ro_title)
        result_original_col.addWidget(self.result_original_preview, 1)
        result_segmented_col.addWidget(rs_title)
        result_segmented_col.addWidget(self.segmented_preview, 1)
        images_row.addLayout(result_original_col, 1)
        images_row.addLayout(result_segmented_col, 1)
        results_layout.addLayout(images_row, 1)
        self.tabs.addTab(results_tab, "Results")

        histogram_tab = QWidget()
        histogram_layout = QVBoxLayout(histogram_tab)
        self.histogram_preview = ImagePreview("The histogram will appear when the analysis finishes")
        histogram_layout.addWidget(self.histogram_preview)
        self.tabs.addTab(histogram_tab, "Histogram")

        convergence_tab = QWidget()
        convergence_layout = QVBoxLayout(convergence_tab)
        self.convergence_preview = ImagePreview(
            "For single-algorithm analysis, convergence data is stored in convergence.csv"
        )
        convergence_layout.addWidget(self.convergence_preview)
        self.tabs.addTab(convergence_tab, "Convergence")

        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        self.metrics_table = QTableWidget(0, 7)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Algorithm", "BestFit", "PSNR", "SSIM", "FSIM", "Time (s)", "Average thresholds"]
        )
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        metrics_layout.addWidget(self.metrics_table)
        self.results_path_label = QLabel("No results yet.")
        self.results_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.results_path_label.setWordWrap(True)
        metrics_layout.addWidget(self.results_path_label)
        self.tabs.addTab(metrics_tab, "Metrics")

        console_tab = QWidget()
        console_layout = QVBoxLayout(console_tab)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setObjectName("console")
        console_layout.addWidget(self.console)
        self.tabs.addTab(console_tab, "Console")

        splitter.addWidget(settings_scroll)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([470, 900])

        self.statusBar().showMessage("Ready")

    def _path_row(self, callback):
        edit = QLineEdit()
        edit.textChanged.connect(self._update_command_preview)
        button = QPushButton("Browse…")
        button.clicked.connect(callback)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return edit, row

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    # --------------------------------------------------------------- defaults --
    def _set_defaults(self) -> None:
        candidate = PROJECT_ROOT / "build" / "segm"
        self.executable_edit.setText(str(candidate))
        self.output_edit.setText(str(DEFAULT_OUTPUT_ROOT))
        self.adapter_edit.setText(str(DEFAULT_ADAPTER))
        self.mi48_script_edit.setText(str(DISCOVERED_MI48_SCRIPT or ""))
        self.mi48_python_edit.setText(str(DISCOVERED_MI48_PYTHON or ""))
        default_algorithm = self.algorithm_combo.findData("DESSA")
        if default_algorithm >= 0:
            self.algorithm_combo.setCurrentIndex(default_algorithm)

    @staticmethod
    def _valid_file_setting(value: object, fallback: Path | None = None) -> str:
        """Return a persisted file only when it still exists.

        This intentionally prevents QSettings from reviving paths from an older
        checkout (for example /home/<user>/.../tools/capture_thermal.py).
        """
        text = str(value or "").strip()
        if text:
            candidate = Path(text).expanduser()
            if candidate.is_file():
                return str(candidate.resolve())
        if fallback is not None and fallback.is_file():
            return str(fallback.resolve())
        return ""

    @staticmethod
    def _valid_dir_setting(value: object, fallback: Path) -> str:
        text = str(value or "").strip()
        if text:
            candidate = Path(text).expanduser()
            # Existing directories are safe to restore. For a missing directory,
            # use the checkout-local fallback instead of an obsolete absolute path.
            if candidate.is_dir():
                return str(candidate.resolve())
        return str(fallback.resolve())

    def _restore_settings(self) -> None:
        # Project-owned files must always follow the current checkout.  A previous
        # QSettings value must never override them merely because an old checkout
        # still exists on disk. This makes copied/re-extracted releases portable.
        self.adapter_edit.setText(str(DEFAULT_ADAPTER.resolve()))
        self.executable_edit.setText(str((PROJECT_ROOT / "build" / "segm").resolve()))
        self.output_edit.setText(
            self._valid_dir_setting(self.settings.value("output_root", ""), DEFAULT_OUTPUT_ROOT)
        )

        # External MI48 SDK paths may legitimately live outside this repository.
        # Restore a valid capture script, but prefer a discovered pysenxor virtual
        # environment over a previously saved system Python (for example
        # /usr/bin/python3.11).  The latter may exist while lacking the SDK modules.
        script = self._valid_file_setting(
            self.settings.value("mi48_script", ""), DISCOVERED_MI48_SCRIPT
        )
        saved_python_text = str(self.settings.value("mi48_python", "") or "").strip()
        saved_python = Path(saved_python_text).expanduser() if saved_python_text else None
        if _is_virtualenv_python(saved_python):
            python = str(saved_python.absolute())
        elif DISCOVERED_MI48_PYTHON and DISCOVERED_MI48_PYTHON.is_file():
            python = str(DISCOVERED_MI48_PYTHON.absolute())
        else:
            python = self._valid_file_setting(saved_python_text, None)
        self.mi48_script_edit.setText(script)
        self.mi48_python_edit.setText(python)

        mode = str(self.settings.value("camera_mode", "mi48"))
        idx = self.camera_mode_combo.findData(mode)
        if idx >= 0:
            self.camera_mode_combo.setCurrentIndex(idx)
        self.camera_index_spin.setValue(int(self.settings.value("camera_index", 0)))
        self.preview_interval_spin.setValue(int(self.settings.value("preview_interval", 750)))

        # Migrate stale project-owned settings immediately so the next launch also
        # contains only paths belonging to the current checkout.
        self.settings.setValue("adapter_path", str(DEFAULT_ADAPTER.resolve()))
        self.settings.setValue("segm_path", str((PROJECT_ROOT / "build" / "segm").resolve()))
        self.settings.sync()

    def _save_settings(self) -> None:
        # Do not persist project-owned paths: they are derived from PROJECT_ROOT on
        # every launch. Persist only user choices that can legitimately be external.
        self.settings.setValue("output_root", self.output_edit.text().strip())
        self.settings.setValue("mi48_script", self.mi48_script_edit.text().strip())
        self.settings.setValue("mi48_python", self.mi48_python_edit.text().strip())
        self.settings.setValue("camera_mode", self.camera_mode_combo.currentData())
        self.settings.setValue("camera_index", self.camera_index_spin.value())
        self.settings.setValue("preview_interval", self.preview_interval_spin.value())
        self.settings.sync()

    # -------------------------------------------------------------- browsing --
    def _browse_executable(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select segm executable", str(PROJECT_ROOT))
        if path:
            self.executable_edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Results folder", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _browse_adapter(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select capture_thermal.py", str(PROJECT_ROOT / "repo" / "src" / "tools"), "Python (*.py);;All files (*)")
        if path:
            self.adapter_edit.setText(path)

    def _browse_mi48_script(self) -> None:
        start = self.mi48_script_edit.text() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Select MI48 capture script", start, "Python (*.py);;All files (*)")
        if path:
            self.mi48_script_edit.setText(path)

    def _browse_mi48_python(self) -> None:
        start = self.mi48_python_edit.text() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Select pysenxor environment Python", start)
        if path:
            self.mi48_python_edit.setText(path)

    def _browse_existing_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select thermal image",
            str(PROJECT_ROOT),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        if not path:
            return
        self.current_capture = Path(path)
        self.captured_preview.set_image(self.current_capture)
        self.result_original_preview.set_image(self.current_capture)
        self.analyze_button.setEnabled(True)
        self.statusBar().showMessage(f"Image selected: {self.current_capture.name}")
        self._append_console(f"\n[GUI] Existing image selected: {self.current_capture}\n")
        self._reset_results_view()
        self._update_command_preview()

    # --------------------------------------------------------------- camera --
    def _refresh_portable_paths(self) -> None:
        """Repair project-owned paths and rediscover external MI48 paths."""
        self.adapter_edit.setText(str(DEFAULT_ADAPTER.resolve()))
        self.executable_edit.setText(str((PROJECT_ROOT / "build" / "segm").resolve()))

        current_script = Path(self.mi48_script_edit.text().strip()).expanduser() if self.mi48_script_edit.text().strip() else None
        current_python = Path(self.mi48_python_edit.text().strip()).expanduser() if self.mi48_python_edit.text().strip() else None
        script, py = _discover_mi48_paths()

        if not current_script or not current_script.is_file():
            if script:
                self.mi48_script_edit.setText(str(script))

        # A plain system Python is not accepted as an automatic MI48 choice when a
        # working pysenxor venv can be discovered. This repairs stale QSettings
        # values automatically on copied/re-extracted releases.
        if (not current_python or not current_python.is_file() or not _is_virtualenv_python(current_python)) and py:
            self.mi48_python_edit.setText(str(py))

    def _camera_config(self) -> ThermalCameraConfig:
        self._refresh_portable_paths()
        mode = str(self.camera_mode_combo.currentData())
        return ThermalCameraConfig(
            adapter_script=Path(self.adapter_edit.text().strip()).expanduser(),
            mode=mode,
            waveshare_script=Path(self.mi48_script_edit.text().strip()).expanduser() if self.mi48_script_edit.text().strip() else None,
            waveshare_python=Path(self.mi48_python_edit.text().strip()).expanduser() if self.mi48_python_edit.text().strip() else None,
            camera_index=self.camera_index_spin.value(),
            preview_interval_ms=self.preview_interval_spin.value(),
        )

    def _update_camera_fields(self) -> None:
        is_mi48 = self.camera_mode_combo.currentData() == "mi48"
        for widget in (self.mi48_script_label, self.mi48_script_edit.parentWidget(), self.mi48_python_label, self.mi48_python_edit.parentWidget()):
            widget.setVisible(is_mi48)
        self.uvc_index_label.setVisible(not is_mi48)
        self.camera_index_spin.setVisible(not is_mi48)
        self._update_command_preview()

    def _start_live_preview(self) -> None:
        self._save_settings()
        config = self._camera_config()
        error = self.camera_controller.validate_config(config)
        if error:
            QMessageBox.warning(self, "Camera configuration", error)
            return
        self.tabs.setCurrentIndex(0)
        self.camera_controller.start_preview(config)

    def _capture_once(self) -> None:
        self._analyze_after_capture = False
        self._request_capture()

    def _capture_and_analyze(self) -> None:
        self._analyze_after_capture = True
        self._request_capture()

    def _request_capture(self) -> None:
        if self.segmentation_controller.is_running:
            QMessageBox.information(self, "Analysis in progress", "Wait for the current analysis to finish before capturing another frame.")
            return
        config = self._camera_config()
        error = self.camera_controller.validate_config(config)
        if error:
            QMessageBox.warning(self, "Camera configuration", error)
            self._analyze_after_capture = False
            return
        DEFAULT_CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
        self.camera_controller.capture_once(config, DEFAULT_CAPTURE_ROOT)

    def _on_live_frame(self, path: str) -> None:
        self.live_preview.set_image(Path(path))
        self.camera_status_label.setText("Live view active · frame received")
        self.statusBar().showMessage("Thermal image received")

    def _on_capture_saved(self, path: str) -> None:
        self.current_capture = Path(path)
        self.captured_preview.set_image(self.current_capture)
        self.result_original_preview.set_image(self.current_capture)
        self.analyze_button.setEnabled(not self.segmentation_controller.is_running)
        self._reset_results_view()
        self._update_command_preview()
        self.statusBar().showMessage(f"Capture saved: {self.current_capture.name}")
        self.tabs.setCurrentIndex(0)
        if self._analyze_after_capture:
            self._analyze_after_capture = False
            self._run_analysis()

    def _on_camera_error(self, message: str) -> None:
        self._append_console(f"\n[CAMERA ERROR] {message}\n")
        self.camera_status_label.setText("Camera error")
        self._analyze_after_capture = False
        QMessageBox.warning(self, "Thermal camera", message)

    def _on_preview_state_changed(self, active: bool) -> None:
        self.start_camera_button.setEnabled(not active and not self.segmentation_controller.is_running)
        self.stop_camera_button.setEnabled(active)
        if active:
            self.camera_status_label.setText("Live view active · acquiring…")
        else:
            self.camera_status_label.setText("Camera stopped")

    def _on_camera_busy_changed(self, busy: bool) -> None:
        if busy and self.camera_controller.preview_active:
            self.camera_status_label.setText("Live view active · acquiring frame…")

    # ---------------------------------------------------------- segmentation --
    def _segmentation_config(self) -> SegmentationRunConfig:
        image = self.current_capture or Path("<current_capture.png>")
        output_text = self.output_edit.text().strip() or str(DEFAULT_OUTPUT_ROOT)
        return SegmentationRunConfig(
            executable=Path(self.executable_edit.text().strip()).expanduser(),
            image=image,
            output_root=Path(output_text).expanduser(),
            algorithm=str(self.algorithm_combo.currentData()),
            levels=self.levels_spin.value(),
            runs=self.runs_spin.value(),
            population=self.population_spin.value(),
            iterations=self.iterations_spin.value(),
            compute_fsim=self.fsim_check.isChecked(),
        )

    def _run_analysis(self) -> None:
        if not self.current_capture or not self.current_capture.exists():
            QMessageBox.warning(self, "No capture", "Start the camera and save a thermal capture first.")
            return
        if self.segmentation_controller.is_running:
            return

        config = self._segmentation_config()
        if not config.executable.exists():
            QMessageBox.warning(self, "Executable not found", f"segm was not found at:\n{config.executable}")
            return
        if not os.access(config.executable, os.X_OK):
            QMessageBox.warning(
                self,
                "Execution permission",
                f"The file does not have execution permission:\n{config.executable}\n\nRun:\nchmod +x {config.executable}",
            )
            return

        config.output_root.mkdir(parents=True, exist_ok=True)
        self.camera_controller.stop_preview()
        self._save_settings()
        self._reset_results_view()
        self.tabs.setCurrentIndex(5)  # console

        try:
            self.segmentation_controller.start(config)
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
            QMessageBox.critical(self, "Could not start", str(exc))

    def _on_run_started(self, command: str, output_dir: str) -> None:
        self._last_output_dir = Path(output_dir)
        self._append_console("\n============================================================\n")
        self._append_console("[ANALYSIS] Command executed:\n")
        self._append_console(command + "\n")
        self._append_console(f"[ANALYSIS] Results: {output_dir}\n")
        self.analyze_button.setEnabled(False)
        self.stop_analysis_button.setEnabled(True)
        self.start_camera_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.capture_analyze_button.setEnabled(False)
        self.open_image_button.setEnabled(False)
        self.statusBar().showMessage("Running multithreshold segmentation…")

    def _on_progress_changed(self, current: int, total: int, text: str) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)
            self.progress.setFormat(text)
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(current)
            self.progress.setFormat(f"{text} · %p%")

    def _on_run_finished(self, exit_code: int, output_dir: str) -> None:
        self.stop_analysis_button.setEnabled(False)
        self.capture_button.setEnabled(True)
        self.capture_analyze_button.setEnabled(True)
        self.open_image_button.setEnabled(True)
        self.start_camera_button.setEnabled(not self.camera_controller.preview_active)
        self.analyze_button.setEnabled(bool(self.current_capture and self.current_capture.exists()))

        if exit_code != 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress.setFormat(f"Error (code {exit_code})")
            self.statusBar().showMessage(f"Analysis finished with code {exit_code}")
            QMessageBox.warning(
                self,
                "Analysis not completed",
                f"segm finished with code {exit_code}. Check the Console tab for details.",
            )
            return

        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("Completed")
        self.statusBar().showMessage("Analysis completed")
        self._load_results(Path(output_dir))
        self.tabs.setCurrentIndex(1)

    def _on_run_failed(self, message: str) -> None:
        self.stop_analysis_button.setEnabled(False)
        self.analyze_button.setEnabled(bool(self.current_capture))
        self.capture_button.setEnabled(True)
        self.capture_analyze_button.setEnabled(True)
        self.open_image_button.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Could not start")
        QMessageBox.critical(self, "Error starting segm", message)

    # --------------------------------------------------------------- results --
    def _load_results(self, output_dir: Path) -> None:
        algorithm = str(self.algorithm_combo.currentData())
        self._analysis_results = read_analysis_results(output_dir, algorithm, self.levels_spin.value())
        self.results_path_label.setText(f"Results saved in: {output_dir}")

        self.result_algorithm_combo.blockSignals(True)
        self.result_algorithm_combo.clear()
        for algo in self._analysis_results:
            self.result_algorithm_combo.addItem(algo)
        self.result_algorithm_combo.setEnabled(len(self._analysis_results) > 0)
        self.result_algorithm_combo.blockSignals(False)

        self._populate_metrics_table()
        if self.result_algorithm_combo.count() > 0:
            self.result_algorithm_combo.setCurrentIndex(0)
            self._show_selected_algorithm_result(self.result_algorithm_combo.currentText())
        else:
            self.segmented_preview.clear_image("No segmented image found")
            self.histogram_preview.clear_image("No histogram found")
            self.convergence_preview.clear_image("No convergence plot found")

    def _show_selected_algorithm_result(self, algorithm: str) -> None:
        result = self._analysis_results.get(algorithm)
        if not result:
            return
        self.segmented_preview.set_image(result.segmented_image)
        self.histogram_preview.set_image(result.histogram_image)
        if result.convergence_image:
            self.convergence_preview.set_image(result.convergence_image)
        else:
            self.convergence_preview.clear_image(
                "A single-algorithm run stores numerical convergence data in convergence.csv"
            )

    def _populate_metrics_table(self) -> None:
        self.metrics_table.setRowCount(0)
        for result in self._analysis_results.values():
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)
            values = [
                result.algorithm,
                result.metrics.get("best_fit", "—"),
                result.metrics.get("psnr", "—"),
                result.metrics.get("ssim", "—"),
                result.metrics.get("fsim", "—"),
                result.metrics.get("time_s", "—"),
                result.metrics.get("avg_thresholds", "—"),
            ]
            for col, value in enumerate(values):
                self.metrics_table.setItem(row, col, QTableWidgetItem(str(value)))

    def _reset_results_view(self) -> None:
        self._analysis_results.clear()
        self.result_algorithm_combo.clear()
        self.result_algorithm_combo.setEnabled(False)
        self.segmented_preview.clear_image("Run analysis")
        self.histogram_preview.clear_image("The histogram will appear when the analysis finishes")
        self.convergence_preview.clear_image("The convergence plot will appear when the analysis finishes")
        self.metrics_table.setRowCount(0)
        self.results_path_label.setText("No results yet.")

    # --------------------------------------------------------------- helpers --
    def _update_command_preview(self) -> None:
        if not hasattr(self, "command_preview"):
            return
        try:
            config = self._segmentation_config()
            self.command_preview.setPlainText(self.segmentation_controller.build_command_preview(config))
        except Exception:
            self.command_preview.setPlainText("Configure the executable and parameters to preview the command.")

    def _append_console(self, text: str) -> None:
        if not hasattr(self, "console"):
            return
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text)
        self.console.moveCursor(QTextCursor.MoveOperation.End)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f5f7f8; }
            QLabel#subtitle { color: #5d6870; padding-bottom: 2px; }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #cbd3d7;
                border-radius: 7px;
                margin-top: 9px;
                padding-top: 10px;
                background: #ffffff;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTableWidget {
                background: white;
                border: 1px solid #c8d0d4;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton { min-height: 28px; padding: 3px 9px; }
            QPushButton#primaryButton {
                background: #207346;
                color: white;
                border: 1px solid #185b37;
                border-radius: 5px;
                font-weight: 600;
            }
            QPushButton#captureButton {
                font-weight: 600;
            }
            QPushButton#primaryButton:disabled { background: #9bb7a8; border-color: #9bb7a8; }
            QLabel#previewTitle { font-weight: 600; padding: 2px; }
            QLabel#cameraStatus { font-weight: 600; }
            QLabel#imagePreview {
                background: #101417;
                color: #c8d0d4;
                border: 1px solid #87939a;
                border-radius: 5px;
                padding: 5px;
            }
            QPlainTextEdit#console, QPlainTextEdit#commandPreview {
                font-family: monospace;
                background: #111518;
                color: #d7e0e4;
            }
            QTabWidget::pane { border: 1px solid #cbd3d7; background: white; }
            """
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        if self.segmentation_controller.is_running:
            answer = QMessageBox.question(
                self,
                "Analysis in progress",
                "An analysis is running. Do you want to stop it and close the application?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.segmentation_controller.stop()

        self._save_settings()
        self.camera_controller.shutdown()
        event.accept()
