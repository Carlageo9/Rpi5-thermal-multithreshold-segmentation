#pragma once

#include <opencv2/core.hpp>

// Phase congruency map (PC_2) based on Peter Kovesi's phasecong2 implementation
// as embedded in the original MATLAB FSIM reference code (FeatureSIM.m).
//
// Input:  single-channel CV_64F image (any range). Internally uses FFT.
// Output: single-channel CV_64F phase congruency map (typically in [0, 1]).
cv::Mat phasecong2_pc2(const cv::Mat& im);
