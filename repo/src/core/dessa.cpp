#include "dessa.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>

// Build a 1-based normalized histogram (size 257, index 1..256)
static std::vector<double> normalized_hist_1based(const cv::Mat& gray) {
    CV_Assert(gray.type() == CV_8UC1);
    std::vector<double> h(257, 0.0);
    const int total = gray.rows * gray.cols;
    for (int r = 0; r < gray.rows; ++r) {
        const auto* p = gray.ptr<unsigned char>(r);
        for (int c = 0; c < gray.cols; ++c) {
            int v = static_cast<int>(p[c]) + 1; // 1..256
            h[v] += 1.0;
        }
    }
    for (int i = 1; i <= 256; ++i) h[i] /= static_cast<double>(total);
    return h;
}

static inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

// Logistic map chaotic initialization, matches MATLAB seed 0.7 and r=4.
static std::vector<double> logistic_sequence(size_t n) {
    std::vector<double> seq(n);
    if (n == 0) return seq;
    seq[0] = 0.7;
    for (size_t k = 1; k < n; ++k) {
        seq[k] = 4.0 * seq[k - 1] * (1.0 - seq[k - 1]);
    }
    return seq;
}

DESSAResult dessa_optimize(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const DESSAOptions& opt,
    const cv::Mat& grayImage
) {
    CV_Assert(grayImage.type() == CV_8UC1);
    const auto t0 = std::chrono::high_resolution_clock::now();

    const int NP = std::max(4, opt.population);
    const int maxGen = std::max(1, opt.max_iterations);
    const double F = opt.F;
    const double CR = opt.CR;
    const int LowB = opt.lowB;
    const int UperB = opt.upperB;

    const std::vector<double> h = normalized_hist_1based(grayImage);

    // init population with chaotic logistic map
    const auto chaotic = logistic_sequence(static_cast<size_t>(NP) * static_cast<size_t>(nvars));
    std::vector<std::vector<int>> pop(NP, std::vector<int>(nvars, LowB));
    for (int i = 0; i < NP; ++i) {
        for (int j = 0; j < nvars; ++j) {
            const double u = chaotic[static_cast<size_t>(i) * nvars + j];
            const double x = LowB + (UperB - LowB) * u;
            pop[i][j] = clampi(static_cast<int>(std::floor(x)), LowB, UperB);
        }
        std::sort(pop[i].begin(), pop[i].end());
    }

    // RNG
    std::mt19937 rng;
    if (opt.seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(opt.seed);
    }
    std::uniform_real_distribution<double> urand(0.0, 1.0);
    std::uniform_int_distribution<int> jrand(0, nvars - 1);

    // evaluate
    std::vector<double> fitness(NP, 0.0);
    for (int i = 0; i < NP; ++i) {
        fitness[i] = fobj(pop[i], h);
    }

    int gbest_idx = static_cast<int>(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
    double gbest_f = fitness[gbest_idx];
    std::vector<int> gbest = pop[gbest_idx];

    std::vector<double> convergence(maxGen, gbest_f);

    // main loop
    for (int gen = 1; gen <= maxGen; ++gen) {
        // sort by fitness (ascending)
        std::vector<int> idx(NP);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return fitness[a] < fitness[b]; });

        std::vector<std::vector<int>> pop_sorted(NP);
        std::vector<double> fit_sorted(NP);
        for (int i = 0; i < NP; ++i) {
            pop_sorted[i] = pop[idx[i]];
            fit_sorted[i] = fitness[idx[i]];
        }
        pop.swap(pop_sorted);
        fitness.swap(fit_sorted);

        std::vector<std::vector<int>> newPop = pop;

        const double c1 = 2.0 * std::exp(-std::pow(4.0 * gen / static_cast<double>(maxGen), 2.0));

        for (int i = 0; i < NP; ++i) {
            // SSA-inspired mutation
            std::vector<double> Vd(nvars, 0.0);
            if (i == 0) {
                // leader
                for (int j = 0; j < nvars; ++j) {
                    const double c2 = urand(rng);
                    const double c3 = urand(rng);
                    const double sign = (c3 >= 0.5) ? 1.0 : -1.0;
                    const double offset = (UperB - LowB) * c2 + LowB;
                    Vd[j] = static_cast<double>(gbest[j]) + c1 * sign * offset;
                }
            } else {
                for (int j = 0; j < nvars; ++j) {
                    Vd[j] = 0.5 * (pop[i-1][j] + pop[i][j]) + F * (gbest[j] - pop[i][j]);
                }
            }

            // clamp
            std::vector<int> V(nvars, LowB);
            for (int j = 0; j < nvars; ++j) {
                V[j] = clampi(static_cast<int>(std::floor(Vd[j])), LowB, UperB);
            }

            // binomial crossover (ensure at least one dimension)
            std::vector<int> U = pop[i];
            const int jr = jrand(rng);
            for (int j = 0; j < nvars; ++j) {
                if (urand(rng) <= CR || j == jr) {
                    U[j] = V[j];
                }
            }

            std::sort(U.begin(), U.end());
            for (int j = 0; j < nvars; ++j) U[j] = clampi(U[j], LowB, UperB);

            const double trial_f = fobj(U, h);
            if (trial_f < fitness[i]) {
                newPop[i] = U;
                fitness[i] = trial_f;
            }
        }

        pop.swap(newPop);

        // update global best
        int min_idx = static_cast<int>(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
        if (fitness[min_idx] < gbest_f) {
            gbest_f = fitness[min_idx];
            gbest = pop[min_idx];
        }

        convergence[gen - 1] = gbest_f;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    DESSAResult res;
    res.best_fitness = gbest_f;
    res.best_thresholds = gbest;
    res.convergence = std::move(convergence);
    res.elapsed_seconds = elapsed;
    return res;
}
