from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


ALL_ALGORITHMS = ("DE", "DESSA", "SDE", "GWO", "PSO", "CSA", "FA")


@dataclass(frozen=True)
class SegmentationRunConfig:
    executable: Path
    image: Path
    output_root: Path
    algorithm: str  # one algorithm or "ALL"
    levels: int
    runs: int
    population: int
    iterations: int
    compute_fsim: bool = True


class SegmentationController(QObject):
    """Runs the existing C++ ``segm`` executable without changing its algorithms."""

    output_received = Signal(str)
    run_started = Signal(str, str)  # human-readable command, output directory
    run_finished = Signal(int, str)  # exit code, output directory
    run_failed = Signal(str)
    progress_changed = Signal(int, int, str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        self._output_dir: Optional[Path] = None
        self._algorithm = ""
        self._completed_algorithms: set[str] = set()
        self._progress_tail = ""

    @property
    def is_running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    def build_command_preview(self, config: SegmentationRunConfig) -> str:
        # Preview uses placeholders and does not create any directories.
        if config.algorithm.upper() == "ALL":
            args = self._common_batch_args(config, Path("<folder_with_one_capture>"), Path("<results>"))
        else:
            args = self._single_args(config, Path("<results>"))
        return self._format_command(config.executable, args)

    def start(self, config: SegmentationRunConfig) -> None:
        if self.is_running:
            raise RuntimeError("A segmentation process is already running.")
        if not config.image.exists():
            raise FileNotFoundError(f"Capture not found: {config.image}")

        algorithm = config.algorithm.upper()
        if algorithm != "ALL" and algorithm not in ALL_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        run_name = f"analysis_{stamp}_{algorithm}_L{config.levels}"
        self._output_dir = config.output_root / run_name
        self._output_dir.mkdir(parents=True, exist_ok=False)

        if algorithm == "ALL":
            # --run_all works on a folder.  Isolate exactly this capture so that
            # "ALL" can never accidentally process neighboring files.
            isolated_input = self._output_dir / "input_capture"
            isolated_input.mkdir(parents=True, exist_ok=False)
            isolated_image = isolated_input / config.image.name
            shutil.copy2(config.image, isolated_image)
            args = self._common_batch_args(config, isolated_input, self._output_dir)
        else:
            args = self._single_args(config, self._output_dir)

        self._algorithm = algorithm
        self._completed_algorithms.clear()
        self._progress_tail = ""

        self._process.setProgram(str(config.executable))
        self._process.setArguments(args)
        self._process.setWorkingDirectory(str(config.executable.parent))

        command = self._format_command(config.executable, args)
        self.run_started.emit(command, str(self._output_dir))
        if algorithm == "ALL":
            self.progress_changed.emit(0, len(ALL_ALGORITHMS), "Starting 7 algorithms")
        else:
            self.progress_changed.emit(0, 0, f"Processing with {algorithm}")
        self._process.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self._process.terminate()
        QTimer.singleShot(2000, self._kill_if_needed)

    def _single_args(self, config: SegmentationRunConfig, output_dir: Path) -> list[str]:
        args = [
            "--image", str(config.image),
            "--levels", str(config.levels),
            "--algo", config.algorithm.upper(),
            "--runs", str(config.runs),
            "--pop", str(config.population),
            "--iter", str(config.iterations),
            "--outdir", str(output_dir),
        ]
        if not config.compute_fsim:
            args.append("--disable_fsim")
        return args

    def _common_batch_args(self, config: SegmentationRunConfig, input_dir: Path, output_dir: Path) -> list[str]:
        args = [
            "--run_all",
            "--batch_dir", str(input_dir),
            "--levels_list", str(config.levels),
            "--runs", str(config.runs),
            "--pop", str(config.population),
            "--iter", str(config.iterations),
            "--outdir", str(output_dir),
        ]
        if not config.compute_fsim:
            args.append("--disable_fsim")
        return args

    def _kill_if_needed(self) -> None:
        if self.is_running:
            self._process.kill()

    def _read_output(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not data:
            return
        self.output_received.emit(data)

        if self._algorithm == "ALL":
            searchable = self._progress_tail + data
            for match in re.finditer(r"\[(DE|DESSA|SDE|GWO|PSO|CSA|FA)\][^\n]*\bdone\b", searchable):
                algo = match.group(1)
                if algo not in self._completed_algorithms:
                    self._completed_algorithms.add(algo)
                    self.progress_changed.emit(
                        len(self._completed_algorithms),
                        len(ALL_ALGORITHMS),
                        f"{algo} completed ({len(self._completed_algorithms)}/{len(ALL_ALGORITHMS)})",
                    )
            self._progress_tail = searchable[-600:]

    def _on_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._read_output()
        output_dir = str(self._output_dir) if self._output_dir else ""
        if exit_code == 0:
            if self._algorithm == "ALL":
                self.progress_changed.emit(len(ALL_ALGORITHMS), len(ALL_ALGORITHMS), "Analysis completed")
            else:
                self.progress_changed.emit(1, 1, "Analysis completed")
        self.run_finished.emit(exit_code, output_dir)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.run_failed.emit(
                "Could not start 'segm'. Verify that the executable path is correct "
                "and that the file has execution permission."
            )

    @staticmethod
    def _format_command(program: Path, args: list[str]) -> str:
        import shlex

        return " ".join(shlex.quote(part) for part in [str(program), *args])
