from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


@dataclass(frozen=True)
class ThermalCameraConfig:
    """Configuration for one-frame acquisition used by the GUI live preview."""

    adapter_script: Path
    mode: str  # "mi48" or "uvc"
    waveshare_script: Optional[Path] = None
    waveshare_python: Optional[Path] = None
    camera_index: int = 0
    preview_interval_ms: int = 750


class ThermalCameraController(QObject):
    """Acquire thermal frames asynchronously using the repository capture adapter.

    The existing C++ segmentation engine is deliberately not involved in preview
    acquisition.  This controller repeatedly requests one frame from
    ``repo/src/tools/capture_thermal.py`` and sends the resulting image to the GUI.  For
    the tested MI48 workflow the adapter, in turn, calls the user's
    ``capture_stream_for_segm.py`` using the pysenxor virtual environment.
    """

    output_received = Signal(str)
    frame_ready = Signal(str)
    capture_saved = Signal(str)
    camera_error = Signal(str)
    preview_state_changed = Signal(bool)
    busy_changed = Signal(bool)

    def __init__(self, runtime_dir: Path, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.runtime_dir = runtime_dir
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._request_frame)

        self._preview_active = False
        self._active_config: Optional[ThermalCameraConfig] = None
        self._current_request_path: Optional[Path] = None
        self._last_frame: Optional[Path] = None
        self._pending_capture_root: Optional[Path] = None

    @property
    def preview_active(self) -> bool:
        return self._preview_active

    @property
    def is_busy(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def last_frame(self) -> Optional[Path]:
        return self._last_frame

    def start_preview(self, config: ThermalCameraConfig) -> None:
        error = self.validate_config(config)
        if error:
            self.camera_error.emit(error)
            return

        self._active_config = config
        if not self._preview_active:
            self._preview_active = True
            self.preview_state_changed.emit(True)
            self.output_received.emit("\n[CAMERA] Live view started.\n")

        if not self.is_busy:
            self._request_frame()

    def stop_preview(self) -> None:
        if not self._preview_active:
            return
        self._preview_active = False
        self._timer.stop()
        self.preview_state_changed.emit(False)
        self.output_received.emit(
            "[CAMERA] Live view stopped. If an acquisition was in progress, "
            "it will be allowed to finish without starting another one.\n"
        )

    def capture_once(self, config: ThermalCameraConfig, capture_root: Path) -> None:
        """Freeze/save one thermal frame.

        If preview is active and at least one valid frame is already available,
        the currently displayed frame is frozen immediately.  Otherwise a new
        one-shot acquisition is requested and saved when it completes.
        """

        error = self.validate_config(config)
        if error:
            self.camera_error.emit(error)
            return

        self._active_config = config
        if self._preview_active and self._last_frame and self._last_frame.exists():
            self._save_frame(self._last_frame, capture_root)
            return

        self._pending_capture_root = capture_root
        if not self.is_busy:
            self._request_frame()
        else:
            self.output_received.emit(
                "[CAMERA] Waiting for the current acquisition to finish before saving the capture.\n"
            )

    def shutdown(self) -> None:
        self._preview_active = False
        self._timer.stop()
        if self.is_busy:
            self._process.terminate()
            if not self._process.waitForFinished(1200):
                self._process.kill()
                self._process.waitForFinished(500)

    @staticmethod
    def validate_config(config: ThermalCameraConfig) -> str:
        if not config.adapter_script.exists():
            return f"Camera adapter not found:\n{config.adapter_script}"

        if config.mode == "mi48":
            if not config.waveshare_script or not config.waveshare_script.exists():
                return (
                    "capture_stream_for_segm.py was not found. Select the MI48 capture script "
                    "that already works with your camera."
                )
            if not config.waveshare_python or not config.waveshare_python.exists():
                return (
                    "The pysenxor environment Python executable was not found. Select, for example, "
                    ".../pysenxor-master/venv/bin/python."
                )
        elif config.mode != "uvc":
            return f"Unsupported camera mode: {config.mode}"

        return ""

    def command_preview(self, config: ThermalCameraConfig, output_placeholder: str = "<frame.png>") -> str:
        import shlex

        program, args = self._build_program_and_args(config, Path(output_placeholder))
        return " ".join(shlex.quote(str(x)) for x in [program, *args])

    def _request_frame(self) -> None:
        if self.is_busy or self._active_config is None:
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._current_request_path = self.runtime_dir / f"live_{stamp}.png"
        program, args = self._build_program_and_args(self._active_config, self._current_request_path)

        self._process.setProgram(str(program))
        self._process.setArguments([str(a) for a in args])
        self._process.setWorkingDirectory(str(self._active_config.adapter_script.parent.parent))
        self.busy_changed.emit(True)
        self._process.start()

    @staticmethod
    def _build_program_and_args(config: ThermalCameraConfig, out_path: Path) -> tuple[Path, list[str]]:
        # Run the adapter with the GUI interpreter.  In MI48 mode the adapter
        # launches the SDK script with the explicitly configured pysenxor Python.
        program = Path(sys.executable)
        args = [str(config.adapter_script), "--out", str(out_path)]

        if config.mode == "mi48":
            args.append("--waveshare-mi48")
            if config.waveshare_script:
                args.extend(["--waveshare-script", str(config.waveshare_script)])
            if config.waveshare_python:
                args.extend(["--waveshare-python", str(config.waveshare_python)])
        else:
            args.extend(["--camera-index", str(config.camera_index)])

        return program, args

    def _read_output(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.output_received.emit(data)

    def _on_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._read_output()
        self.busy_changed.emit(False)

        current = self._current_request_path
        self._current_request_path = None

        if exit_code != 0 or current is None or not current.exists():
            message = f"Thermal capture finished with code {exit_code}."
            if current is not None:
                message += f" No valid image was produced at {current}."
            self.camera_error.emit(message)
            # Do not hammer the camera in an error loop.
            if self._preview_active:
                self._preview_active = False
                self.preview_state_changed.emit(False)
            self._pending_capture_root = None
            return

        self._last_frame = current
        self.frame_ready.emit(str(current))

        if self._pending_capture_root is not None:
            root = self._pending_capture_root
            self._pending_capture_root = None
            self._save_frame(current, root)

        self._cleanup_runtime_frames(keep=6)

        if self._preview_active and self._active_config is not None:
            interval = max(100, int(self._active_config.preview_interval_ms))
            self._timer.start(interval)

    def _save_frame(self, source: Path, capture_root: Path) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        capture_dir = capture_root / f"capture_{stamp}"
        capture_dir.mkdir(parents=True, exist_ok=False)
        destination = capture_dir / f"thermal_{stamp}.png"
        shutil.copy2(source, destination)
        self.output_received.emit(f"[CAMERA] Capture saved: {destination}\n")
        self.capture_saved.emit(str(destination))

    def _cleanup_runtime_frames(self, keep: int = 6) -> None:
        try:
            frames = sorted(
                self.runtime_dir.glob("live_*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in frames[keep:]:
                try:
                    old.unlink()
                except OSError:
                    pass
        except OSError:
            pass

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.busy_changed.emit(False)
            self.camera_error.emit(
                "Could not start the capture process. Verify the Python paths, "
                "capture_thermal.py, and the camera SDK script."
            )
            if self._preview_active:
                self._preview_active = False
                self.preview_state_changed.emit(False)
