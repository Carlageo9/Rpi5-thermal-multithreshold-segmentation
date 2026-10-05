#include "firefly.hpp"
#include <opencv2/imgproc.hpp>

#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

static std::vector<double> normalized_hist_1based(const cv::Mat& gray){
    CV_Assert(gray.type() == CV_8UC1);
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    double total = (double)gray.total();
    std::vector<double> h(257, 0.0); // 1..256
    for(int i=0;i<256;i++) h[i+1] = hist.at<float>(i) / total;
    return h;
}

static inline void clamp_round_sort(std::vector<double>& x, int lowB, int upperB){
    for(double& v: x){
        if(v < (double)lowB) v = (double)lowB;
        if(v > (double)upperB) v = (double)upperB;
        v = std::round(v);
    }
    std::sort(x.begin(), x.end());
}

static inline std::vector<int> to_int_sorted(const std::vector<double>& x){
    std::vector<int> xi(x.size());
    for(size_t i=0;i<x.size();++i) xi[i] = (int)std::lround(x[i]);
    std::sort(xi.begin(), xi.end());
    return xi;
}

FireflyResult firefly_optimize(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const FireflyOptions& opt,
    const cv::Mat& grayImage
){
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<double> h = normalized_hist_1based(grayImage);

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_real_distribution<double> uniLBUB((double)opt.lowB, (double)opt.upperB);

    const int n = opt.population;
    // fireflies positions in continuous space
    std::vector<std::vector<double>> fireflies(n, std::vector<double>(nvars));
    for(int i=0;i<n;i++){
        for(int d=0; d<nvars; d++) fireflies[i][d] = uniLBUB(rng);
        clamp_round_sort(fireflies[i], opt.lowB, opt.upperB);
    }

    std::vector<double> fitness(n, 0.0);
    for(int i=0;i<n;i++){
        fitness[i] = fobj(to_int_sorted(fireflies[i]), h);
    }

    int best_idx = (int)std::distance(fitness.begin(), std::min_element(fitness.begin(), fitness.end()));
    double best_fit = fitness[best_idx];
    std::vector<double> best_sol = fireflies[best_idx];

    std::vector<double> conv(opt.max_iterations, best_fit);

    double alpha = opt.alpha;

    for(int iter=0; iter<opt.max_iterations; ++iter){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(fitness[j] < fitness[i]){
                    // distance
                    double r2 = 0.0;
                    for(int d=0; d<nvars; d++){
                        double diff = fireflies[i][d] - fireflies[j][d];
                        r2 += diff*diff;
                    }
                    double beta = opt.beta0 * std::exp(-opt.gamma * r2);

                    std::vector<double> tmp = fireflies[i];
                    for(int d=0; d<nvars; d++){
                        tmp[d] = fireflies[i][d] + beta * (fireflies[j][d] - fireflies[i][d]) + alpha * (uni01(rng) - 0.5);
                    }
                    clamp_round_sort(tmp, opt.lowB, opt.upperB);

                    double tmp_fit = fobj(to_int_sorted(tmp), h);
                    if(tmp_fit < fitness[i]){
                        fireflies[i] = std::move(tmp);
                        fitness[i] = tmp_fit;
                    }
                }
            }
        }

        // update global best
        int cur_best_idx = (int)std::distance(fitness.begin(), std::min_element(fitness.begin(), fitness.end()));
        if(fitness[cur_best_idx] < best_fit){
            best_fit = fitness[cur_best_idx];
            best_sol = fireflies[cur_best_idx];
        }

        // optional alpha reduction
        alpha *= 0.97;
        conv[iter] = best_fit;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    FireflyResult res;
    res.best_fitness = best_fit;
    res.best_thresholds = to_int_sorted(best_sol);
    res.convergence = std::move(conv);
    res.elapsed_seconds = dt.count();
    return res;
}
