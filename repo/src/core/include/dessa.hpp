#pragma once
#include <vector>
#include <functional>
#include <opencv2/core.hpp>

struct DESSAOptions {
    int population = 30;
    int max_iterations = 50;
    double F = 0.5;
    double CR = 0.9;
    int lowB = 1;
    int upperB = 256;
    unsigned int seed = 0; // 0 => random_device
};

struct DESSAResult {
    double best_fitness = 0.0;
    std::vector<int> best_thresholds;
    std::vector<double> convergence;
    double elapsed_seconds = 0.0;
};

// Objective signature: (thresholds, normalized_hist_1based) -> fitness
DESSAResult dessa_optimize(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const DESSAOptions& opt,
    const cv::Mat& grayImage
);
