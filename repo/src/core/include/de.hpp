#pragma once
#include <vector>
#include <functional>
#include <opencv2/core.hpp>

struct DEOptions {
    int population = 30;
    int max_iterations = 50;
    double F = 0.5;
    double CR = 0.9;
    int lowB = 1;
    int upperB = 256;
};

struct DEResult {
    double best_fitness = 0.0;
    std::vector<int> best_thresholds;
    std::vector<double> convergence; // best fitness per iteration
    double elapsed_seconds = 0.0;
};

// Objective signature: (thresholds, normalized_hist_1based) -> fitness
DEResult differential_evolution(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const DEOptions& opt,
    const cv::Mat& grayImage
);
