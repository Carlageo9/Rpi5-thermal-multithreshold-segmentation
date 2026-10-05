#pragma once
#include <vector>
#include <functional>
#include <opencv2/core.hpp>

// Scheduled Differential Evolution (SDE)
// --------------------------------------
// This optimizer ports the hybrid idea from the user's Jupyter notebook:
//   1) "original" DE phase  -> classic differential mutation
//   2) "smooth"   DE phase  -> DE mutation blended with a temporal average
//                               of the last populations (population memory)
//
// The optimizer still MINIMIZES the objective function. In this project the
// objective is kapur_entropy(), which returns -entropy, therefore minimizing it
// is equivalent to maximizing Kapur's entropy.
struct ScheduledDEOptions {
    int population = 30;
    int max_iterations = 50;

    // Defaults copied from the notebook, not from the legacy DE module.
    double F_original = 0.8;
    double F_smooth = 0.3;
    double CR = 0.9;
    double gamma = 0.2;   // weight of the temporal smoothing term
    int history_window = 5;

    int lowB = 1;
    int upperB = 256;
    unsigned int seed = 0; // 0 => random_device
};

struct ScheduledDEResult {
    double best_fitness = 0.0;
    std::vector<int> best_thresholds;
    std::vector<double> convergence;
    double elapsed_seconds = 0.0;
};

ScheduledDEResult scheduled_differential_evolution(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const ScheduledDEOptions& opt,
    const cv::Mat& grayImage
);
