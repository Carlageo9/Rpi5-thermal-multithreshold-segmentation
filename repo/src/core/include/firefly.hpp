#pragma once
#include <vector>
#include <functional>
#include <opencv2/core.hpp>

struct FireflyOptions {
    int population = 30;          // N_particulas
    int max_iterations = 50;      // max_iteration
    double alpha = 0.5;           // randomness
    double beta0 = 1.0;           // base attractiveness
    double gamma = 1.0;           // light absorption
    int lowB = 1;
    int upperB = 256;
};

struct FireflyResult {
    double best_fitness = 0.0;
    std::vector<int> best_thresholds;
    std::vector<double> convergence;
    double elapsed_seconds = 0.0;
};

FireflyResult firefly_optimize(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const FireflyOptions& opt,
    const cv::Mat& grayImage
);
