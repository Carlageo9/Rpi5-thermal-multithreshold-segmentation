#pragma once
#include <string>
#include <vector>

struct BatchConfig {
    std::string input_dir;
    std::string out_dir;
    std::string algorithm; // "DE" | "DESSA" | "SDE" | "GWO" | "PSO" | "CSA" | "FA"
    std::vector<int> levels_list {2,3,4,5};
    int runs = 40;
    int population = 30;
    int iterations = 50;
    double F = 0.5;
    double CR = 0.9;
    // PSO parameters (match pso.m)
    double pso_w = 0.72;
    double pso_c1 = 1.49;
    double pso_c2 = 1.49;
    // Firefly Algorithm (firefly.m) parameters
    double fa_alpha = 0.5;
    double fa_beta0 = 1.0;
    double fa_gamma = 1.0;
    // CSA (Crow Search Algorithm) parameters
    double csa_fl = 2.0;
    double csa_AP = 0.1;
    bool compute_fsim = true;
};

// Processes all images found in input_dir (common image extensions).
// Creates MATLAB-like folder layout under out_dir.
// Returns 0 on success, non-zero on error.
int run_batch(const BatchConfig& cfg);
