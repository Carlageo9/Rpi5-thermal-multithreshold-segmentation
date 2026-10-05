#pragma once

#include <vector>
#include <functional>
#include <opencv2/core.hpp>

struct PSOOptions {
    int population = 30;
    int max_iterations = 50;
    int lowB = 1;
    int upperB = 256;
    // canonical-ish parameters
    double w = 0.72;
    double c1 = 1.49;
    double c2 = 1.49;
};

struct PSOResult {
    double best_fitness = 0.0;
    std::vector<int> best_thresholds;
    std::vector<double> convergence;
    double elapsed_seconds = 0.0;
};

PSOResult particle_swarm_optimize(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const PSOOptions& opt,
    const cv::Mat& grayImage
);
