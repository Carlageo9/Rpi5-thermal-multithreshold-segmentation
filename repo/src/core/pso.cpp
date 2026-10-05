#include "pso.hpp"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

// NOTE: This implementation is adjusted to match the user's original MATLAB pso.m
// (April 2025) which follows the GMOV.m structure and uses:
//  - pos initialized uniformly in [LowB, UperB] then fix + sort + fixer01
//  - velocity update with w, c1, c2
//  - position update then fix
//  - boundary handling via MinMaxCheck (wrap with mod)
//  - fixer01 (0 -> 1)
//  - objective evaluated on the current pos row (no forced re-sorting inside loop)

static std::vector<double> normalized_hist_1based_pso(const cv::Mat& gray) {
    CV_Assert(gray.type() == CV_8UC1);
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    double total = (double)gray.total();
    std::vector<double> h(257, 0.0);
    for (int i = 0; i < 256; i++) h[i + 1] = hist.at<float>(i) / total;
    return h;
}

// MATLAB-compatible mod for integers (always returns a value in [0, m-1] when m>0)
static int matlab_mod_int(int a, int m) {
    if (m == 0) return 0;
    int mm = std::abs(m);
    int r = a % mm;
    if (r < 0) r += mm;
    return r;
}

// Port of GMOV/MinMaxCheck.m (wrap, not clamp)
static void min_max_check_wrap(const std::vector<int>& minimum,
                               const std::vector<int>& maximum,
                               std::vector<int>& a) {
    const int n = (int)a.size();
    for (int l = 0; l < n; l++) {
        if (a[l] > maximum[l]) {
            a[l] = matlab_mod_int(a[l], maximum[l]);
        } else if (a[l] < minimum[l]) {
            a[l] = matlab_mod_int(a[l], minimum[l]);
        }
    }
}

// Port of GMOV/fixer01.m
static void fixer01(std::vector<int>& a) {
    for (int& v : a) {
        if (v == 0) v = 1;
    }
}

PSOResult particle_swarm_optimize(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const PSOOptions& opt,
    const cv::Mat& grayImage) {

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<double> h = normalized_hist_1based_pso(grayImage);

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    const int N = opt.population;

    // LowBv/UperBv like MATLAB vectors
    std::vector<int> LowBv(nvars, opt.lowB);
    std::vector<int> UperBv(nvars, opt.upperB);

    // Positions are integer after fix(); velocities are double.
    std::vector<std::vector<int>> pos(N, std::vector<int>(nvars, opt.lowB));
    std::vector<std::vector<double>> vel(N, std::vector<double>(nvars, 0.0));

    // pBest / gBest
    std::vector<std::vector<int>> pBest = pos;
    std::vector<double> pBestFitness(N, 0.0);
    std::vector<int> gBest(nvars, opt.lowB);
    double gBestFitness = std::numeric_limits<double>::infinity();

    auto eval = [&](const std::vector<int>& x) -> double {
        return fobj(x, h);
    };

    // Initialize positions exactly like MATLAB:
    // pos(:,i) = LowB + (UperB-LowB)*rand; pos=fix; pos=sort(pos,2); pos=fixer01
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < nvars; j++) {
            const double r = uni01(rng);
            const double x = (double)opt.lowB + ((double)opt.upperB - (double)opt.lowB) * r;
            pos[i][j] = (int)std::floor(x); // MATLAB fix for positives
        }
        std::sort(pos[i].begin(), pos[i].end());
        fixer01(pos[i]);

        pBest[i] = pos[i];
        pBestFitness[i] = eval(pos[i]);

        if (pBestFitness[i] < gBestFitness) {
            gBestFitness = pBestFitness[i];
            gBest = pos[i];
        }
    }

    std::vector<double> conv(opt.max_iterations);

    for (int iter = 0; iter < opt.max_iterations; iter++) {
        for (int i = 0; i < N; i++) {
            // Update velocity / position
            for (int j = 0; j < nvars; j++) {
                const double r1 = uni01(rng);
                const double r2 = uni01(rng);

                vel[i][j] = opt.w * vel[i][j]
                         + opt.c1 * r1 * ((double)pBest[i][j] - (double)pos[i][j])
                         + opt.c2 * r2 * ((double)gBest[j] - (double)pos[i][j]);

                const double xnew = (double)pos[i][j] + vel[i][j];
                pos[i][j] = (int)std::floor(xnew); // fix()
            }

            // Apply boundary and fixer01 exactly like MATLAB
            min_max_check_wrap(LowBv, UperBv, pos[i]);
            fixer01(pos[i]);
        }

        // Evaluate and update bests
        for (int i = 0; i < N; i++) {
            const double fit = eval(pos[i]);

            if (fit < pBestFitness[i]) {
                pBestFitness[i] = fit;
                pBest[i] = pos[i];
            }

            if (fit < gBestFitness) {
                gBestFitness = fit;
                gBest = pos[i];
            }
        }

        conv[iter] = gBestFitness;
    }

    // Output best thresholds.
    // MATLAB returns gBest directly; however, downstream segmentation expects sorted thresholds.
    std::vector<int> best_thr = gBest;
    std::sort(best_thr.begin(), best_thr.end());

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    PSOResult res;
    res.best_fitness = gBestFitness;
    res.best_thresholds = std::move(best_thr);
    res.convergence = std::move(conv);
    res.elapsed_seconds = dt.count();
    return res;
}
