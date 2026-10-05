#pragma once
#include <vector>
#include <opencv2/core.hpp>

// Replicates MATLAB imageGRAY(): sets each pixel to the lower bound of its threshold interval.
// thresholds are grayscale values (1..256). Output is CV_8UC1.
cv::Mat image_gray_segment(const cv::Mat& gray, const std::vector<int>& thresholds);
