#!/usr/bin/env python3
"""
Capture adapter for the segm thermal-acquisition mode.

This script creates one PNG image at the path passed with --out.

It supports three practical capture methods:

1) OpenCV / USB UVC cameras:
   python3 repo/src/tools/capture_thermal.py --out frame.png --camera-index 0

2) Waveshare Thermal Camera HAT (B) / MI48 using the working stream script:
   python3 repo/src/tools/capture_thermal.py --out frame.png \
     --waveshare-script ~/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py \
     --waveshare-python ~/Downloads/thermal_demo/pysenxor-master/venv/bin/python

   The Waveshare script must save "thermal_capture.png" in its own folder and then exit.
   This adapter copies that file to the --out path expected by segm.

3) Environment-variable shortcut:
   export MI48_SCRIPT=~/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py
   export MI48_PYTHON=~/Downloads/thermal_demo/pysenxor-master/venv/bin/python
   python3 repo/src/tools/capture_thermal.py --out frame.png --waveshare-mi48
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def capture_with_opencv(out_path: Path, camera_index: int) -> int:
    try:
        import cv2
    except Exception as exc:
        print(f"Python OpenCV is not available: {exc}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open /dev/video camera index {camera_index}", file=sys.stderr)
        return 3
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("Camera opened, but no frame was captured", file=sys.stderr)
        return 4
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frame):
        print(f"Could not write output image: {out_path}", file=sys.stderr)
        return 5
    return 0


def find_default_mi48_script() -> Path | None:
    candidates = [
        Path.home() / "Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py",
        Path.home() / "Downloads/thermal_demo/pysenxor-master/example/stream_spi.py",
        Path.home() / "thermal_demo/pysenxor-master/example/capture_stream_for_segm.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def capture_with_waveshare_mi48(out_path: Path, script_path: str | None, python_exe: str | None) -> int:
    script = Path(script_path or os.environ.get("MI48_SCRIPT", "")).expanduser()
    if not str(script):
        found = find_default_mi48_script()
        if found is not None:
            script = found

    if not script.exists():
        print(
            "Waveshare MI48 capture script not found. Provide --waveshare-script or set MI48_SCRIPT.\n"
            "Expected example:\n"
            "  --waveshare-script ~/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py",
            file=sys.stderr,
        )
        return 20

    py = python_exe or os.environ.get("MI48_PYTHON")
    if not py:
        venv_python = script.parents[1] / "venv/bin/python"  # .../pysenxor-master/venv/bin/python
        py = str(venv_python) if venv_python.exists() else sys.executable

    workdir = script.parent
    tmp_capture = workdir / "thermal_capture.png"
    if tmp_capture.exists():
        try:
            tmp_capture.unlink()
        except Exception:
            pass

    print(f"Running MI48 capture script: {script}")
    print(f"Using Python: {py}")
    rc = subprocess.call([py, str(script)], cwd=str(workdir))
    if rc != 0:
        print(f"MI48 capture script failed with code {rc}", file=sys.stderr)
        return rc

    if not tmp_capture.exists():
        print(
            f"MI48 script finished, but it did not create {tmp_capture}.\n"
            "Make sure your capture_stream_for_segm.py saves thermal_capture.png and exits automatically.",
            file=sys.stderr,
        )
        return 21

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp_capture, out_path)
    print(f"Saved thermal image to {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output PNG path expected by segm")
    parser.add_argument("--camera-index", type=int, default=None, help="Use OpenCV/UVC capture from /dev/videoN")

    parser.add_argument("--waveshare-mi48", action="store_true", help="Use Waveshare Thermal Camera HAT (B) MI48 capture script")
    parser.add_argument("--waveshare-script", default=None, help="Path to capture_stream_for_segm.py")
    parser.add_argument("--waveshare-python", default=None, help="Python executable from the pysenxor virtual environment")

    args = parser.parse_args()
    out_path = Path(args.out)

    if args.camera_index is not None:
        return capture_with_opencv(out_path, args.camera_index)

    if args.waveshare_mi48 or args.waveshare_script or os.environ.get("MI48_SCRIPT"):
        return capture_with_waveshare_mi48(out_path, args.waveshare_script, args.waveshare_python)

    print(
        "No capture method selected.\n"
        "For your Waveshare Thermal Camera HAT (B), use:\n"
        "  --waveshare-mi48 --waveshare-script /path/to/capture_stream_for_segm.py "
        "--waveshare-python /path/to/venv/bin/python",
        file=sys.stderr,
    )
    return 10


if __name__ == "__main__":
    raise SystemExit(main())
