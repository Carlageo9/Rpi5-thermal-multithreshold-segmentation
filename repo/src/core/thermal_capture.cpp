#include "thermal_capture.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

static std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static std::string replace_all(std::string text, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

bool capture_thermal_image(const ThermalCaptureConfig& cfg, const std::string& output_path) {
    fs::create_directories(fs::path(output_path).parent_path());

    if (!cfg.capture_command.empty()) {
        std::string cmd = cfg.capture_command;
        if (cmd.find("{output}") != std::string::npos) {
            cmd = replace_all(cmd, "{output}", shell_quote(output_path));
        } else {
            cmd += " ";
            cmd += shell_quote(output_path);
        }
        std::cout << "Capturing thermal image with external command...\n";
        int rc = std::system(cmd.c_str());
        if (rc != 0) {
            std::cerr << "capture command failed with code: " << rc << "\n";
            return false;
        }
        cv::Mat check = cv::imread(output_path, cv::IMREAD_GRAYSCALE);
        if (check.empty()) {
            std::cerr << "capture command finished, but output image could not be read: " << output_path << "\n";
            return false;
        }
        return true;
    }

    if (cfg.camera_index >= 0) {
        cv::VideoCapture cap(cfg.camera_index);
        if (!cap.isOpened()) {
            std::cerr << "Could not open camera index: " << cfg.camera_index << "\n";
            return false;
        }
        if (cfg.warmup_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(cfg.warmup_ms));
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Camera opened, but no frame was captured.\n";
            return false;
        }
        cv::Mat gray;
        if (frame.channels() == 1) gray = frame;
        else cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        return cv::imwrite(output_path, gray);
    }

    std::cerr << "No thermal capture method selected. Use --capture_cmd or --camera_index.\n";
    return false;
}
