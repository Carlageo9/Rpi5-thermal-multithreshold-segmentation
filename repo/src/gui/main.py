#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution with: python3 repo/src/gui/main.py
GUI_DIR = Path(__file__).resolve().parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

try:
    from PySide6.QtWidgets import QApplication, QMessageBox
except ImportError:
    requirements = GUI_DIR / "requirements.txt"
    print(
        "PySide6 is not installed.\n"
        "Create/activate a Python environment and install the GUI dependencies with:\n"
        f"  python3 -m pip install -r {requirements}",
        file=sys.stderr,
    )
    raise SystemExit(2)

from main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Thermal Multithreshold Segmentation")
    app.setOrganizationName("Open-source thermal segmentation project")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
