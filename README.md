# Raspberry Pi 5 Thermal Segmentation

Open-source edge software for live thermal-image acquisition and on-device multithreshold segmentation on Raspberry Pi 5 using metaheuristic optimization.

This repository contains a C++17/OpenCV segmentation engine, a PySide6 graphical user interface, and the Python capture adapter used to acquire thermal frames from a Waveshare Thermal Camera HAT (B) / TUOPUONE MI48 thermal camera connected to a Raspberry Pi 5. The platform can process pre-existing images, batch folders, or live thermal frames captured through the Raspberry Pi GPIO interface using I2C/SPI.

## Main features

- Live thermal capture from a Waveshare Thermal Camera HAT (B) / MI48 thermal module.
- PySide6 graphical user interface for live preview, capture, analysis, result visualization, metrics, and console output.
- Camera-agnostic Python capture adapter (`repo/src/tools/capture_thermal.py`).
- C++17/OpenCV segmentation engine (`segm`).
- Kapur entropy-based multithreshold segmentation.
- Seven supported metaheuristic optimizers:
  - Differential Evolution (DE)
  - Differential Evolution-Salp Swarm hybrid (DESSA)
  - Scheduled Differential Evolution (SDE)
  - Grey Wolf Optimizer (GWO)
  - Particle Swarm Optimization (PSO)
  - Crow Search Algorithm (CSA)
  - Firefly Algorithm (FA)
- Live acquisition modes:
  - `--thermal_once`: capture and process one thermal frame.
  - `--thermal_loop`: continuously capture and process thermal frames.
- Batch and single-image modes for reproducible offline experiments.
- Timestamped output sessions that preserve previous results.
- Automatic export of captured images, segmented images, histograms, convergence curves, and metric summaries.
- Per-run convergence export: one CSV file and one PNG plot for every optimization run, stored under each analysis `convergence/` folder.
- Optional XLSX output using `libxlsxwriter`; CSV fallback when XLSX support is unavailable.

## Hardware

The tested platform used:

- Raspberry Pi 5, 8 GB RAM.
- Raspberry Pi OS 64-bit.
- Official Raspberry Pi 5 case with integrated fan.
- Waveshare Thermal Camera HAT (B) / TUOPUONE MI48 long-wave infrared thermal camera.
- 40-pin right-angle male GPIO header, 2 x 20, 2.54 mm pitch.
- Recommended Raspberry Pi 27 W USB-C power supply or equivalent regulated USB-C source.
- microSD card or SSD storage.
- Optional external monitor, keyboard, and mouse for setup.

The HAT/GPIO thermal camera uses I2C for configuration and SPI for frame transfer. It is not treated as a standard USB/UVC webcam unless a different thermal camera model is used.

## Repository structure

The source code is organized under repo/src, grouping the C++ engine, Python GUI, and camera adapter in a consistent project structure:

```text
.
├── CMakeLists.txt
├── README.md
├── LICENSE.txt
├── CITATION.cff
├── repo/
│   └── src/
│       ├── core/
│       │   ├── main.cpp
│       │   ├── batch.cpp
│       │   ├── de.cpp
│       │   ├── dessa.cpp
│       │   ├── scheduled_de.cpp
│       │   ├── gwo.cpp
│       │   ├── pso.cpp
│       │   ├── csa.cpp
│       │   ├── firefly.cpp
│       │   ├── kapur_entropy.cpp
│       │   ├── metrics.cpp
│       │   ├── histogram_plot.cpp
│       │   ├── plot.cpp
│       │   ├── xlsx.cpp
│       │   ├── thermal_capture.cpp
│       │   └── include/
│       │       └── *.hpp
│       ├── gui/
│       │   ├── main.py
│       │   ├── main_window.py
│       │   ├── result_reader.py
│       │   ├── requirements.txt
│       │   ├── controllers/
│       │   │   ├── thermal_camera_controller.py
│       │   │   └── segmentation_controller.py
│       │   └── README_GUI.md
│       └── tools/
│           └── capture_thermal.py
├── docs/
│   └── THERMAL_CAMERA_RPI5.md
├── validation/
│   ├── input/
│   ├── results/
│   ├── convergence/
│   └── summary/
└── sample_outputs/
    ├── captures/
    └── analyses/
```

The `validation/` directory contains the input image and numerical/visual artifacts used to support the illustrative validation reported in the SoftwareX manuscript. See `validation/README.md`.

## System dependencies

Install the C++ and OpenCV dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config git
sudo apt install -y libopencv-dev python3-opencv
```

Optional XLSX export support:

```bash
sudo apt install -y libxlsxwriter-dev
```

Check OpenCV:

```bash
pkg-config --modversion opencv4
```

## Enable Raspberry Pi interfaces

Enable I2C and SPI:

```bash
sudo raspi-config
```

Use:

```text
Interface Options -> I2C -> Enable
Interface Options -> SPI -> Enable
```

Reboot:

```bash
sudo reboot
```

Verify the interfaces after reboot:

```bash
i2cdetect -y 1
lsmod | grep i2c
ls /dev/spidev*
```

For the tested Waveshare Thermal Camera HAT (B), the I2C address `0x40` was observed.

## Camera SDK preparation

The tested camera workflow uses the Waveshare/Meridian MI48 Python SDK (`pysenxor`). Install it in a Python virtual environment from the SDK source directory:

```bash
cd /path/to/pysenxor-master
python3 -m venv venv
source venv/bin/activate
pip install .
pip install matplotlib scipy numpy pillow opencv-python gpiozero lgpio RPi.GPIO spidev smbus2
```

Run the manufacturer examples first to verify that the camera works independently from this segmentation repository:

```bash
cd /path/to/pysenxor-master/example
python single_capture_spi.py
python stream_spi.py
```

### Creating the capture script used by this repository

The live segmentation workflow expects a script that saves a single processed thermal frame as `thermal_capture.png` and then exits. In the tested setup this was created from the manufacturer stream example:

```bash
cd /path/to/pysenxor-master/example
cp stream_spi.py capture_stream_for_segm.py
```

Modify `capture_stream_for_segm.py` so that the OpenCV display helper returns the resized/colorized image:

```python
cv.imshow(title, cvresize)
return cvresize
```

Then, in the main `while True:` loop, replace the GUI display block with:

```python
if GUI:
    frame = cv_display(img8u)
    cv.imwrite("thermal_capture.png", frame)
    print("Saved thermal_capture.png")
    break
```

Test it:

```bash
python capture_stream_for_segm.py
ls -lh thermal_capture.png
xdg-open thermal_capture.png
```

## Build the C++ segmentation engine

From the root of this repository:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

If `libxlsxwriter` is not installed or XLSX export is not required:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_XLSXWRITER=OFF
make -j4
```

Verify the executable:

```bash
./segm --help
```

## Run the graphical interface

After building `segm`, create a dedicated Python environment from the repository root:

```bash
python3 -m venv .venv-gui
source .venv-gui/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r repo/src/gui/requirements.txt
python3 repo/src/gui/main.py
```

The GUI locates the repository root dynamically, so it does not depend on a specific username or `/home/<user>` path. Repository-owned paths are reconstructed from the current checkout on every launch: the camera adapter is `<project>/repo/src/tools/capture_thermal.py` and the scientific engine is `<project>/build/segm`. Stale Qt `QSettings` values from older copies are ignored automatically for those project-owned files.

The external MI48/pysenxor SDK is auto-discovered relative to the current user's home directory. For the tested installation, the GUI prefers `$HOME/Downloads/thermal_demo/pysenxor-master/venv/bin/python` over a system interpreter such as `/usr/bin/python3`. This is intentional: the MI48 capture script depends on packages installed inside the pysenxor virtual environment. If the SDK is installed elsewhere, use the GUI **Browse…** buttons or optionally set `PYSENXOR_ROOT`, `MI48_CAPTURE_SCRIPT`, and/or `MI48_PYTHON`. These mechanisms contain no hard-coded username. See `repo/src/gui/README_GUI.md` for the complete workflow and exact Raspberry Pi execution sequence.

## Operation modes

### 1. Single-image mode

```bash
./segm --image /path/to/image.png \
  --levels 3 \
  --algo DE \
  --runs 40 --pop 30 --iter 50 \
  --outdir results
```

### 2. Batch-folder mode with one optimizer

```bash
./segm --batch_dir /path/to/images \
  --algo CSA \
  --levels_list 2,3,4,5 \
  --runs 40 --pop 30 --iter 50 \
  --outdir results
```

### 3. Batch-folder mode with all optimizers

```bash
./segm --run_all \
  --batch_dir /path/to/images \
  --levels_list 2,3,4,5 \
  --runs 40 --pop 30 --iter 50 \
  --outdir results
```

### 4. Test the thermal capture adapter

From the `build/` directory:

```bash
python3 ../repo/src/tools/capture_thermal.py \
  --out mi48_test.png \
  --waveshare-mi48 \
  --waveshare-script /path/to/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py \
  --waveshare-python /path/to/thermal_demo/pysenxor-master/venv/bin/python
```

Verify the image:

```bash
ls -lh mi48_test.png
xdg-open mi48_test.png
```

### 5. Capture one thermal image and segment it

```bash
./segm --thermal_once \
  --capture_cmd "python3 ../repo/src/tools/capture_thermal.py --out {output} --waveshare-mi48 --waveshare-script /path/to/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py --waveshare-python /path/to/pysenxor-master/venv/bin/python" \
  --algo ALL \
  --levels_list 2,3,4,5 \
  --runs 40 --pop 30 --iter 50 \
  --outdir results_live
```

### 6. Continuous thermal acquisition and segmentation

The following command captures one frame every 60 seconds and processes it locally. `--captures 0` means infinite loop until interrupted by the user.

```bash
./segm --thermal_loop \
  --capture_cmd "python3 ../repo/src/tools/capture_thermal.py --out {output} --waveshare-mi48 --waveshare-script /path/to/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py --waveshare-python /path/to/thermal_demo/pysenxor-master/venv/bin/python" \
  --algo ALL \
  --captures 0 \
  --interval 60 \
  --levels_list 2,3,4,5 \
  --runs 40 --pop 30 --iter 50 \
  --outdir results_live
```

### 7. Continuous acquisition with one optimizer

```bash
./segm --thermal_loop \
  --capture_cmd "python3 ../repo/src/tools/capture_thermal.py --out {output} --waveshare-mi48 --waveshare-script /path/to/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py --waveshare-python /path/to/thermal_demo/pysenxor-master/venv/bin/python" \
  --algo DESSA \
  --captures 10 \
  --interval 60 \
  --levels_list 3 \
  --runs 40 --pop 30 --iter 50 \
  --outdir results_live
```

## Output structure

Each thermal execution creates a timestamped session. A typical layout is:

```text
results_live/
└── session_YYYYMMDD_HHMMSS/
    ├── captures/
    │   └── capture_YYYYMMDD_HHMMSS/
    │       └── thermal_YYYYMMDD_HHMMSS.png
    └── analyses/
        └── capture_YYYYMMDD_HHMMSS/
            ├── Results_DE/
            ├── Results_DESSA/
            ├── Results_SDE/
            ├── Results_GWO/
            ├── Results_PSO/
            ├── Results_CSA/
            └── Results_FA/
```

The algorithm-specific directories contain segmented images, histograms, convergence plots, and metric summary files.

Useful commands:

```bash
find results_live -iname "*.png"
find results_live -iname "*.csv" -o -iname "*.xlsx"
xdg-open path/to/image.png
```

## Portable GUI path behavior

The GUI is designed to survive moving, cloning, or extracting the repository to a different user account or directory. Internal paths are not read back from an older installation:

- `repo/src/tools/capture_thermal.py` is always derived from the currently running checkout.
- `build/segm` is always derived from the currently running checkout.
- Missing/obsolete output paths fall back to `<project>/gui_results_live`.
- MI48 SDK paths are external to this repository. A saved Python interpreter is reused only when it is a valid `venv/bin/python` or `.venv/bin/python`; stale system-Python selections are automatically replaced by a discovered pysenxor virtual-environment interpreter when available. Missing selections trigger auto-discovery again.

The automatic MI48 discovery checks optional environment variables first:

```bash
export PYSENXOR_ROOT="$HOME/path/to/pysenxor-master"
# alternatively:
export MI48_CAPTURE_SCRIPT="$HOME/path/to/capture_stream_for_segm.py"
export MI48_PYTHON="$HOME/path/to/venv/bin/python"
```

For the tested Raspberry Pi layout, automatic discovery resolves:

```text
Script MI48: $HOME/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py
Python pysenxor: $HOME/Downloads/thermal_demo/pysenxor-master/venv/bin/python
```

Do not select `/usr/bin/python3` for the MI48 workflow unless you intentionally installed all pysenxor dependencies into the system interpreter.

Seeing a resolved `/home/<current-user>/...` value inside the GUI is normal and does not mean the path is hard-coded; it is computed at runtime for the current system.


### Note about the MI48 Python path

For the MI48/pysenxor workflow, the GUI must use the Python launcher inside the pysenxor virtual environment, for example:

```text
$HOME/Downloads/thermal_demo/pysenxor-master/venv/bin/python
```

The file `venv/bin/python` may be a symbolic link to `/usr/bin/python3.x`. The GUI
therefore intentionally preserves the virtual-environment path instead of resolving
that symbolic link. Seeing the `.../venv/bin/python` path in the **Python pysenxor**
field is expected and ensures that the `senxor` dependencies installed in that
environment are available.


## Troubleshooting

### `Unable to locate package libopencv-dev`

Check the package name carefully. It must be `libopencv-dev`, not `libopenc-dev`.

### `/dev/i2c-1` does not exist

I2C is not enabled. Run `sudo raspi-config`, enable I2C, reboot, and test again with:

```bash
i2cdetect -y 1
```

### OpenCV cannot open `/dev/video*`

The Waveshare MI48 HAT is not a conventional USB/UVC webcam. Use the MI48 Python SDK through SPI/I2C instead of `cv2.VideoCapture()`.

### `ModuleNotFoundError: No module named 'gpiozero'`

Activate the virtual environment and install GPIO dependencies:

```bash
source /path/to/pysenxor-master/venv/bin/activate
pip install gpiozero lgpio RPi.GPIO spidev smbus2
```

### `lgpio.error: 'GPIO busy'`

Reboot the Raspberry Pi. If the error persists, verify SPI configuration and that no other process is using the camera pins.

### XLSX export is unavailable

Install `libxlsxwriter-dev` or configure the project with:

```bash
cmake .. -DUSE_XLSXWRITER=OFF
```

CSV summaries will still be generated.


## Reproducibility

This repository includes the source code, documentation, validation data, and representative outputs required to reproduce the experiments reported in the associated SoftwareX manuscript.

The `validation/` directory contains representative input data, archived numerical results, convergence information, plots, and CSV summaries used for manuscript validation.

The software version associated with the published manuscript is archived on Zenodo. See `CITATION.cff` for citation information and the corresponding DOI.

## Third-party SDK note

The Waveshare/Meridian `pysenxor` SDK is third-party software. If the SDK archive does not contain a clear redistribution license, do not redistribute it as part of this repository. Instead, document where it was obtained and include only the project-specific adapter and the minimal patch/instructions needed to create `capture_stream_for_segm.py`.

## License

This software is released under the MIT License. See `LICENSE.txt` for the full license text.

Third-party software and manufacturer SDKs remain subject to their respective licenses.

## Citation

doi: "10.5281/zenodo.23171864"
url: "https://doi.org/10.5281/zenodo.23171864"
repository-code: "https://github.com/Carlageo9/Rpi5-thermal-multithreshold-segmentation"

```text
Arturo Valdivia Gonzalez, Itzel Aranguren Navarro, Oscar Gregorio Silva Mares, Carla Georgina Sánchez Arreguín.
Raspberry Pi 5 Thermal Segmentation. Zenodo. DOI: 10.5281/zenodo.23171864.
```

