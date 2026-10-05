#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>

// Saves a simple histogram plot (PNG) with vertical lines at thresholds.
// thresholds are grayscale values [0..255].
void save_histogram_plot(const cv::Mat& gray, const std::vector<int>& thresholds, const std::string& outPath);
