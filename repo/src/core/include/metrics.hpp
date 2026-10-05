#pragma once
#include <opencv2/core.hpp>

double mse(const cv::Mat& a, const cv::Mat& b);
double psnr(const cv::Mat& a, const cv::Mat& b);
double ssim(const cv::Mat& a, const cv::Mat& b);
double uqi(const cv::Mat& a, const cv::Mat& b, int kernelSize=3);
double qilv(const cv::Mat& a, const cv::Mat& b, int win=3); // uses square window win x win

// Additional metrics from the original MATLAB runner
double mssim_windowed(const cv::Mat& a, const cv::Mat& b, int win=3);
double haarpsi(const cv::Mat& a, const cv::Mat& b, bool preprocessWithSubsampling=true);

// FSIM (grayscale luminance only). Computationally heavy.
double fsim(const cv::Mat& a, const cv::Mat& b);
