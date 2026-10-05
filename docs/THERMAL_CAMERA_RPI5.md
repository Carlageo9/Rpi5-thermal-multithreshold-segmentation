# Using an 80x62 Thermal Camera on Raspberry Pi 5 with `segm`

## 1. Hardware requirements

- Raspberry Pi 5 running Raspberry Pi OS 64-bit.
- Official or stable power supply for Raspberry Pi 5.
- 80x62 LWIR thermal camera compatible with Raspberry Pi.
- Verify the actual connection type of the purchased model:
  - HAT/GPIO: uses I2C for configuration and SPI for data transfer.
  - USB-C/USB: may require the manufacturer's Python demo or may appear as `/dev/video*` if it exposes a UVC interface.

## 2. Base packages

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config git \
    libopencv-dev python3 python3-pip python3-opencv \
    i2c-tools libgpiod-dev
sudo raspi-config
```

In `raspi-config`, enable the following interfaces if your camera is a HAT/GPIO model:

- Interface Options > I2C > Enable
- Interface Options > SPI > Enable

Reboot:

```bash
sudo reboot
```

Check the interfaces:

```bash
ls /dev/i2c-*
ls /dev/spidev*
i2cdetect -y 1
```

If the camera is USB/UVC:

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

## 3. Compile

```bash
cd /path/to/Rpi5-thermal-multithreshold-segmentation
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

If XLSX export will not be used, or if `libxlsxwriter` is not available:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_XLSXWRITER=OFF
make -j4
```

## 4. New thermal capture mode

The software now preserves previous results by creating a timestamped session folder:

```text
results_live/
  session_YYYYMMDD_HHMMSS/
    captures/
      capture_YYYYMMDD_HHMMSS/thermal_YYYYMMDD_HHMMSS.png
    analyses/
      capture_YYYYMMDD_HHMMSS/
        Results_DE/
        Results_DESSA/
        ...
```

### Single capture using all algorithms

```bash
./segm --thermal_once \
  --capture_cmd "python3 ../repo/src/tools/capture_thermal.py --out {output}" \
  --algo ALL \
  --levels_list 2,3,4,5 \
  --runs 40 --pop 30 --iter 50 \
  --outdir results_live
```

### Continuous capture every 60 seconds

```bash
./segm --thermal_loop \
  --capture_cmd "python3 ../repo/src/tools/capture_thermal.py --out {output}" \
  --algo ALL \
  --captures 0 \
  --interval 60 \
  --levels_list 2,3,4,5 \
  --runs 40 --pop 30 --iter 50 \
  --outdir results_live
```

`--captures 0` means continuous execution. To stop the process, press `Ctrl + C`.

### Single-algorithm execution

```bash
./segm --thermal_loop \
  --capture_cmd "python3 ../repo/src/tools/capture_thermal.py --out {output}" \
  --algo DESSA \
  --captures 10 \
  --interval 30 \
  --levels_list 3 \
  --runs 40 --pop 30 --iter 50 \
  --outdir results_live
```

Valid algorithm names:

```text
ALL, DE, DESSA, SDE, GWO, PSO, CSA, FA
```

## 5. Important note about the camera driver

The file `repo/src/tools/capture_thermal.py` is the integration point. If your camera includes a manufacturer's Python demo, the recommended approach is to modify that script so it saves a PNG image to the path received through `--out`.

The C++ executable is not tied to a specific camera brand. It only requires that the command passed through `--capture_cmd` produces a PNG/JPG/TIFF image that can be read by OpenCV.

---

## Portable paths in the GUI

The GUI does not embed a username or an absolute repository path. It derives the adapter and `segm` locations from the checkout that is currently being executed. If an older release stored a different project path in Qt `QSettings`, that stale project-owned path is ignored automatically.

The MI48/pysenxor SDK is an external dependency. The GUI can find the tested layout under the current user's `$HOME`, or you can set:

```bash
export PYSENXOR_ROOT="$HOME/path/to/pysenxor-master"
# alternatively
export MI48_CAPTURE_SCRIPT="$HOME/path/to/capture_stream_for_segm.py"
export MI48_PYTHON="$HOME/path/to/venv/bin/python"
```

No source-code modification is required when the repository is copied to another Raspberry Pi user account.

## 6. Tested integration with Waveshare Thermal Camera HAT (B) / MI48

For this camera, the correct workflow is:

```text
Waveshare MI48 + SPI/I2C
    ↓
capture_stream_for_segm.py
    ↓
thermal_capture.png
    ↓
repo/src/tools/capture_thermal.py
    ↓
segm --thermal_loop
```

### 6.1. Required SDK script

The following file must already exist on the Raspberry Pi:

```bash
~/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py
```

That script must save an image named:

```bash
thermal_capture.png
```

inside the same `example/` folder and then exit automatically.

### 6.2. Test capture from the `segm` adapter

From the `segm` build folder:

```bash
cd /path/to/Rpi5-thermal-multithreshold-segmentation/build

python3 ../repo/src/tools/capture_thermal.py \
  --out prueba_mi48.png \
  --waveshare-mi48 \
  --waveshare-script ~/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py \
  --waveshare-python ~/Downloads/thermal_demo/pysenxor-master/venv/bin/python
```

Verify that the image was created:

```bash
ls -lh prueba_mi48.png
xdg-open prueba_mi48.png
```

### 6.3. Run one capture and segment it with all algorithms

```bash
cd /path/to/Rpi5-thermal-multithreshold-segmentation/build

./segm --thermal_once \
  --capture_cmd "python3 ../repo/src/tools/capture_thermal.py --out {output} --waveshare-mi48 --waveshare-script ~/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py --waveshare-python ~/Downloads/thermal_demo/pysenxor-master/venv/bin/python" \
  --algo ALL \
  --levels_list 2,3,4,5 \
  --runs 40 --pop 30 --iter 50 \
  --outdir results_live
```

### 6.4. Continuous capture every 60 seconds

```bash
cd /path/to/Rpi5-thermal-multithreshold-segmentation/build

./segm --thermal_loop \
  --capture_cmd "python3 ../repo/src/tools/capture_thermal.py --out {output} --waveshare-mi48 --waveshare-script ~/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py --waveshare-python ~/Downloads/thermal_demo/pysenxor-master/venv/bin/python" \
  --algo ALL \
  --captures 0 \
  --interval 60 \
  --levels_list 2,3,4,5 \
  --runs 40 --pop 30 --iter 50 \
  --outdir results_live
```

### 6.5. Continuous capture using a single algorithm

Example using DESSA:

```bash
cd /path/to/Rpi5-thermal-multithreshold-segmentation/build

./segm --thermal_loop \
  --capture_cmd "python3 ../repo/src/tools/capture_thermal.py --out {output} --waveshare-mi48 --waveshare-script ~/Downloads/thermal_demo/pysenxor-master/example/capture_stream_for_segm.py --waveshare-python ~/Downloads/thermal_demo/pysenxor-master/venv/bin/python" \
  --algo DESSA \
  --captures 10 \
  --interval 60 \
  --levels_list 3 \
  --runs 40 --pop 30 --iter 50 \
  --outdir results_live
```

### 6.6. Where the results are saved

Each execution creates a new session:

```text
results_live/
  session_YYYYMMDD_HHMMSS/
    captures/
      capture_YYYYMMDD_HHMMSS/
        thermal_YYYYMMDD_HHMMSS.png
    analyses/
      capture_YYYYMMDD_HHMMSS/
        Results_DE/
        Results_DESSA/
        Results_SDE/
        Results_GWO/
        Results_PSO/
        Results_CSA/
        Results_FA/
```

This structure preserves all captures and previous results.

## Repository archive

The archived version of this software package is available on Zenodo:

https://doi.org/10.5281/zenodo.23171864
