#include "csa.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>

// Build 1-based normalized histogram vector h[1..256] like MATLAB's imhist(Im(:,:,1)) / (sz1*sz2)
static std::vector<double> normalized_hist_1based(const cv::Mat& gray){
    CV_Assert(gray.type() == CV_8UC1);
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    const double total = (double)gray.total();
    std::vector<double> h(257, 0.0);
    for(int i=0;i<256;i++){
        h[i+1] = hist.at<float>(i) / total;
    }
    return h;
}

static inline int clamp_int(int v, int lo, int hi){
    return std::max(lo, std::min(hi, v));
}

CSAResult crow_search_optimize(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const CSAOptions& opt,
    const cv::Mat& grayImage
){
    const auto t0 = std::chrono::high_resolution_clock::now();
    const std::vector<double> h = normalized_hist_1based(grayImage);

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_int_distribution<int> uniCrow(0, opt.population - 1);

    const int N = opt.population;
    const int lo = opt.lowB;
    const int hi = opt.upperB;

    // crows: current positions (integer thresholds)
    std::vector<std::vector<int>> crows(N, std::vector<int>(nvars, lo));
    std::vector<std::vector<int>> memory(N, std::vector<int>(nvars, lo));
    std::vector<double> fitness(N, std::numeric_limits<double>::infinity());
    std::vector<double> memory_fitness(N, std::numeric_limits<double>::infinity());

    // Init random population in [LowB, UperB], then sort rows and fix() (already int)
    for(int i=0;i<N;i++){
        for(int j=0;j<nvars;j++){
            // MATLAB: LowBv(i) + (UperBv-UperBv)*rand => inclusive; then fix.
            // We'll sample uniform in [lo, hi].
            int v = lo + (int)std::floor((hi - lo) * uni01(rng) + 0.0000001);
            v = clamp_int(v, lo, hi);
            crows[i][j] = v;
        }
        std::sort(crows[i].begin(), crows[i].end());
        memory[i] = crows[i];
    }

    double best_fitness = std::numeric_limits<double>::infinity();
    std::vector<int> best_solution(nvars, lo);
    std::vector<double> convergence(opt.max_iterations, best_fitness);

    for(int iter=0; iter<opt.max_iterations; ++iter){
        // Evaluate and update memory
        for(int i=0;i<N;i++){
            fitness[i] = fobj(crows[i], h);
            if(fitness[i] < memory_fitness[i]){
                memory_fitness[i] = fitness[i];
                memory[i] = crows[i];
            }
        }

        // Current best crow (based on fitness, matching csa.m)
        int idx_best = (int)std::distance(fitness.begin(), std::min_element(fitness.begin(), fitness.end()));
        if(fitness[idx_best] < best_fitness){
            best_fitness = fitness[idx_best];
            best_solution = crows[idx_best];
        }

        convergence[iter] = best_fitness;

        // Update positions
        for(int i=0;i<N;i++){
            int j = uniCrow(rng);
            while(j==i) j = uniCrow(rng);

            const double r1 = uni01(rng);
            const double r2 = uni01(rng);

            std::vector<int> new_pos(nvars);
            if(r2 > opt.AP){
                // Evasion/random move (matches: if r2 > AP then random)
                for(int k=0;k<nvars;k++){
                    int v = lo + (int)std::floor((hi - lo) * uni01(rng) + 0.0000001);
                    new_pos[k] = clamp_int(v, lo, hi);
                }
            } else {
                // Following mode: crows(i,:) + r1*fl*(memory(j,:) - crows(i,:))
                for(int k=0;k<nvars;k++){
                    const double x = (double)crows[i][k] + r1 * opt.fl * ((double)memory[j][k] - (double)crows[i][k]);
                    int v = (int)std::floor(x); // fix() in MATLAB truncates toward zero; values are within bounds so floor ok
                    new_pos[k] = clamp_int(v, lo, hi);
                }
            }

            // sort each row (matches crows = sort(crows,2))
            std::sort(new_pos.begin(), new_pos.end());
            crows[i] = std::move(new_pos);
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    CSAResult res;
    res.best_fitness = best_fitness;
    res.best_thresholds = std::move(best_solution);
    std::sort(res.best_thresholds.begin(), res.best_thresholds.end());
    res.convergence = std::move(convergence);
    res.elapsed_seconds = dt.count();
    return res;
}
