#pragma once

#include <vector>
#include <functional>
#include <opencv2/core.hpp>

// CSA in the user's MATLAB project is **Crow Search Algorithm**.
// NOTE: Not to be confused with Cuckoo Search Algorithm.
struct CSAOptions {
    int population = 30;        // N_particulas (number of crows)
    int max_iterations = 50;    // max_iteration
    int lowB = 1;
    int upperB = 256;
    double fl = 2.0;            // flight length (learning factor)
    double AP = 0.1;            // awareness probability
};

struct CSAResult {
    double best_fitness = 0.0;
    std::vector<int> best_thresholds;
    std::vector<double> convergence;
    double elapsed_seconds = 0.0;
};

// Crow Search Algorithm optimizer (matches csa.m behavior).
CSAResult crow_search_optimize(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const CSAOptions& opt,
    const cv::Mat& grayImage
);

// Backward-compatible alias (older builds used this name).
inline CSAResult cuckoo_search_optimize(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const CSAOptions& opt,
    const cv::Mat& grayImage
){
    return crow_search_optimize(fobj, nvars, opt, grayImage);
}
