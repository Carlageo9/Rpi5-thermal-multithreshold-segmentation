#pragma once

#include <vector>
#include <functional>
#include <opencv2/core.hpp>

struct GWOOptions {
    int population = 30;
    int max_iterations = 50;
    int lowB = 1;
    int upperB = 256;
};

struct GWOResult {
    double best_fitness = 0.0;
    std::vector<int> best_thresholds;
    std::vector<double> convergence;
    double elapsed_seconds = 0.0;
};

GWOResult grey_wolf_optimize(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const GWOOptions& opt,
    const cv::Mat& grayImage
);
