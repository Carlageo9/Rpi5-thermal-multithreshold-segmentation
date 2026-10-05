# Thermal Segmentation Graphical User Interface — GUI v0.2

This version integrates thermal-camera acquisition with the graphical interface while preserving the existing C++ scientific engine (`segm`). The optimization algorithms and Kapur entropy objective function are not changed by the GUI.


## Source location

To comply with the SoftwareX repository layout, the GUI source is stored under `repo/src/gui/`, alongside the C++ core (`repo/src/core/`) and the camera adapter (`repo/src/tools/`). All commands below are intended to be executed from the repository root unless stated otherwise.

## Implemented workflow

```text
Thermal camera
    ↓
Live thermal preview (successive captures)
    ↓
Capture / freeze one frame
    ↓
Select one algorithm or ALL seven
    ↓
Multilevel thresholding with segm
    ↓
Segmented images + histograms + metrics + convergence outputs
```

## GUI v0.2 features

- **Waveshare Thermal Camera HAT (B) / MI48** support through the project's existing `pysenxor` workflow.
- Optional USB/UVC camera support through `/dev/video*`.
- Live thermal preview based on repeated frame acquisition without blocking the GUI.
- **Capture frame** action to freeze and preserve one timestamped image.
- **Capture and analyze** action.
- Individual algorithm selection: `DESSA`, `DE`, `SDE`, `GWO`, `PSO`, `CSA`, or `FA`.
- **All algorithms (7)** option using exactly the same captured frame.
- Controls for threshold count, runs, population size, iterations, and FSIM.
- Visualization of the captured image, segmented image, histogram, and convergence curve.
- Algorithm selector for reviewing DE/DESSA/SDE/GWO/PSO/CSA/FA results after an ALL run.
- Comparative table for BestFit, PSNR, SSIM, FSIM, execution time, and average thresholds.
- Integrated console showing the real `segm` output.
- Portable path handling: repository-owned paths are derived from the current checkout on every launch.
- Persistent MI48 SDK settings only when the selected external files still exist; stale values are ignored automatically.
- Per-run convergence records: every optimization run generates a CSV data file and a PNG convergence plot.

> **MI48 live-view note:** the GUI uses the stable camera adapter already included in the project and requests successive frames from `capture_stream_for_segm.py`. The visible refresh rate therefore depends on the time required by that script to initialize the camera, acquire a frame, save `thermal_capture.png`, and exit. The GUI does not replace the manufacturer's `stream_spi.py` workflow and does not claim 25 FPS inside the GUI.

## 1. MI48 camera prerequisites

First verify that the camera SDK works independently:

```bash
cd ~/Downloads/thermal_demo/pysenxor-master/example
source ../venv/bin/activate
python stream_spi.py
```

The single-frame capture script used by the project must also work:

```text
~/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py
```

That script must create `thermal_capture.png` and terminate automatically.

## 2. Build `segm`

From the project root:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..
```

## 3. Prepare the GUI environment

```bash
python3 -m venv .venv-gui
source .venv-gui/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r repo/src/gui/requirements.txt
```

On Raspberry Pi OS, if Qt reports that `xcb-cursor0` is missing:

```bash
sudo apt update
sudo apt install -y libxcb-cursor0
```

## 4. Run the GUI

From a new terminal session:

```bash
cd /path/to/Rpi5-thermal-multithreshold-segmentation
source .venv-gui/bin/activate
python3 repo/src/gui/main.py
```

## 5. Portable path handling and MI48 settings

The GUI does **not** contain a hard-coded username or installation directory. At startup it locates the repository root from the location of `main_window.py`, then derives the repository-owned paths from that root:

```text
Adapter:
<current-checkout>/repo/src/tools/capture_thermal.py

Segmentation executable:
<current-checkout>/build/segm

Default GUI results:
<current-checkout>/gui_results_live
```

Old `QSettings` values for the adapter and `segm` executable are ignored and migrated automatically. Therefore, extracting a new copy of the repository in another home directory does not require deleting Qt configuration files.

For the external MI48/pysenxor SDK, the GUI first checks optional environment variables and then searches common locations relative to the **current user's home directory**. No username is embedded in the source. The following environment variables can be used when the SDK is installed elsewhere:

```bash
export PYSENXOR_ROOT="$HOME/path/to/pysenxor-master"
# or set the two files explicitly:
export MI48_CAPTURE_SCRIPT="$HOME/path/to/capture_stream_for_segm.py"
export MI48_PYTHON="$HOME/path/to/venv/bin/python"
```

The automatic search includes, among others:

```text
$HOME/Downloads/thermal_demo/pysenxor-master/
$HOME/thermal_demo/pysenxor-master/
$HOME/pysenxor-master/
$HOME/pysenxor/
```

With the tested installation, the expected pair is:

```text
Script MI48: $HOME/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py
Python pysenxor: $HOME/Downloads/thermal_demo/pysenxor-master/venv/bin/python
```

The GUI deliberately prefers this virtual-environment Python over `/usr/bin/python3` when both exist.

If the SDK is installed somewhere else and no environment variable is set, select **Script MI48** and **Python pysenxor** once with **Browse…**. For Python, the GUI expects the SDK virtual-environment interpreter (`venv/bin/python` or `.venv/bin/python`). A previously saved system interpreter such as `/usr/bin/python3.11` is not allowed to override an automatically discovered pysenxor virtual environment. This prevents a common failure in which the capture script starts with Python but cannot import the MI48 SDK dependencies.

> The GUI may display absolute paths at runtime because Qt/Python resolve files to their actual filesystem locations. These paths are discovered dynamically; they are not hard-coded in the software.

## 6. Exact execution sequence on Raspberry Pi 5

From a fresh extracted/cloned repository, use the following sequence. Replace only the first `cd` target with the directory in which you placed the project. No source file needs to be edited with a username-specific path.

```bash
# 1) Enter the repository root
cd /path/to/raspberry-pi5-thermal-segmentation-v1.0.0-SoftwareX-final

# 2) Install system build dependencies
sudo apt update
sudo apt install -y build-essential cmake pkg-config git libopencv-dev \
    python3 python3-venv python3-pip python3-opencv i2c-tools libgpiod-dev

# Optional XLSX support
sudo apt install -y libxlsxwriter-dev

# 3) Build the C++ engine
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
./segm --help
cd ..

# 4) Create the GUI environment
python3 -m venv .venv-gui
source .venv-gui/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r repo/src/gui/requirements.txt

# 5) Launch the GUI
python3 repo/src/gui/main.py
```

For an MI48 camera, the manufacturer SDK must already work independently. If it is in the tested location `$HOME/Downloads/thermal_demo/pysenxor-master`, the GUI detects it automatically. Otherwise set `PYSENXOR_ROOT`, `MI48_CAPTURE_SCRIPT`, and/or `MI48_PYTHON` as described in Section 5 or choose the two external files with **Browse…**.

After the GUI opens, verify that:

1. **Adapter** points to the `repo/src/tools/capture_thermal.py` file inside the copy you just launched.
2. **segm executable** points to the `build/segm` file inside that same copy.
3. **Script MI48** points to the working capture script and **Python pysenxor** points specifically to that SDK's `venv/bin/python` (or `.venv/bin/python`), not to `/usr/bin/python3`.
4. Click **Start live view**.

If you previously ran an older release, no manual QSettings cleanup is required: stale repository-owned paths are repaired automatically.

## 7. Recommended first test

For a short initial validation:

1. Select `DESSA`.
2. Select `3` thresholds.
3. Temporarily set `runs = 2`, `population = 10`, and `iterations = 10`.
4. Click **Start preview**.
5. Wait until at least one thermal frame is displayed.
6. Click **Capture frame**.
7. Confirm that the captured image appears in the analysis preview.
8. Click **Analyze captured frame**.
9. Review the Results, Histogram, Metrics, Convergence, and Console areas.

After validation, restore the experimental parameters required by your study.

## 8. Run all seven algorithms on one capture

Choose the ALL option in the GUI. Internally, the GUI creates an isolated input folder containing only the selected frame and executes the existing batch workflow:

```bash
segm --run_all \
  --batch_dir <folder_with_one_capture> \
  --levels_list 3 \
  --runs 40 \
  --pop 30 \
  --iter 50 \
  --outdir <results>
```

This prevents `--run_all` from accidentally processing neighboring images.

## 9. Generated folders and files

Frozen captures:

```text
gui_captures/
└── capture_YYYYMMDD_HHMMSS_mmm/
    └── thermal_YYYYMMDD_HHMMSS_mmm.png
```

Individual analysis:

```text
gui_results_live/
└── analysis_..._DESSA_L3/
    ├── segmented.png
    ├── histogram.png
    ├── convergence.csv          # backward-compatible first-run data
    ├── convergence.png          # backward-compatible first-run plot
    ├── summary.csv
    └── convergence/
        ├── DESSA_thr3_run001_convergence.csv
        ├── DESSA_thr3_run001_convergence.png
        ├── DESSA_thr3_run002_convergence.csv
        ├── DESSA_thr3_run002_convergence.png
        └── ...
```

Analysis with all algorithms:

```text
gui_results_live/
└── analysis_..._ALL_L3/
    ├── input_capture/
    ├── Results_DE/
    │   └── convergence/
    ├── Results_DESSA/
    │   └── convergence/
    ├── Results_SDE/
    │   └── convergence/
    ├── Results_GWO/
    │   └── convergence/
    ├── Results_PSO/
    │   └── convergence/
    ├── Results_CSA/
    │   └── convergence/
    └── Results_FA/
        └── convergence/
```

Each `Results_<ALGORITHM>/convergence/` directory contains one CSV file and one PNG plot for every run, image, and threshold level. The original first-run convergence plot is also retained in the algorithm folder for compatibility with existing result readers.

## 10. Camera troubleshooting

Test the MI48 script outside the GUI first:

```bash
~/Downloads/thermal_demo/pysenxor-master/venv/bin/python \
  ~/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py
```

Then verify that this file exists:

```bash
~/Downloads/thermal_demo/pysenxor-master/example/thermal_capture.png
```

Finally test the project adapter:

```bash
source .venv-gui/bin/activate
python3 repo/src/tools/capture_thermal.py \
  --out /tmp/thermal_test.png \
  --waveshare-mi48 \
  --waveshare-script ~/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py \
  --waveshare-python ~/Downloads/thermal_demo/pysenxor-master/venv/bin/python
```

If `/tmp/thermal_test.png` is created correctly, the GUI can use the same acquisition workflow.


### MI48 virtual-environment Python and symbolic links

The **Python pysenxor** field should normally show a path such as:

```text
$HOME/Downloads/thermal_demo/pysenxor-master/venv/bin/python
```

Do not replace it with `/usr/bin/python3.x` merely because `venv/bin/python`
ultimately points there. On Raspberry Pi, virtual-environment Python launchers are
often symbolic links. The GUI preserves the venv path so the correct environment
and installed pysenxor dependencies are used.
