#pragma once
#include <string>

struct ThermalCaptureConfig {
    std::string capture_command;   // External command. Use {output} placeholder.
    int camera_index = -1;         // OpenCV/UVC camera index. -1 disables it.
    int warmup_ms = 300;
};

// Captures one thermal image and writes it as an 8-bit grayscale image.
// Returns true if the output image was created and can be read by OpenCV.
bool capture_thermal_image(const ThermalCaptureConfig& cfg, const std::string& output_path);
