#include "de.hpp"
#include <opencv2/imgproc.hpp>
#include <random>
#include <algorithm>
#include <chrono>

static std::vector<double> normalized_hist_1based(const cv::Mat& gray){
    CV_Assert(gray.type() == CV_8UC1);
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    double total = (double)gray.total();
    std::vector<double> h(257, 0.0); // 1..256 used
    for(int i=0;i<256;i++){
        h[i+1] = hist.at<float>(i) / total;
    }
    return h;
}

DEResult differential_evolution(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const DEOptions& opt,
    const cv::Mat& grayImage
){
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<double> h = normalized_hist_1based(grayImage);

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_int_distribution<int> uniInt(opt.lowB, opt.upperB);

    const int NP = opt.population;
    std::vector<std::vector<int>> pop(NP, std::vector<int>(nvars));
    for(int i=0;i<NP;i++){
        for(int j=0;j<nvars;j++) pop[i][j] = uniInt(rng);
        std::sort(pop[i].begin(), pop[i].end());
    }

    std::vector<double> fitness(NP);
    for(int i=0;i<NP;i++) fitness[i] = fobj(pop[i], h);

    int best_idx = (int)std::distance(fitness.begin(), std::min_element(fitness.begin(), fitness.end()));
    double best_fit = fitness[best_idx];
    std::vector<int> best = pop[best_idx];

    std::vector<double> conv(opt.max_iterations);

    for(int iter=0; iter<opt.max_iterations; iter++){
        for(int i=0;i<NP;i++){
            // select r1,r2,r3 distinct and !=i
            std::vector<int> idxs(NP);
            for(int k=0;k<NP;k++) idxs[k]=k;
            std::shuffle(idxs.begin(), idxs.end(), rng);
            idxs.erase(std::remove(idxs.begin(), idxs.end(), i), idxs.end());
            int r1=idxs[0], r2=idxs[1], r3=idxs[2];

            // mutant V = Xr1 + F*(Xr2 - Xr3) in continuous, then clamp+round
            std::vector<double> V(nvars);
            for(int j=0;j<nvars;j++){
                double v = (double)pop[r1][j] + opt.F * ((double)pop[r2][j] - (double)pop[r3][j]);
                v = std::max((double)opt.lowB, std::min((double)opt.upperB, v));
                V[j] = v;
            }

            // crossover
            std::vector<int> U = pop[i];
            for(int j=0;j<nvars;j++){
                if(uni01(rng) <= opt.CR){
                    int u = (int)std::lround(V[j]);
                    u = std::max(opt.lowB, std::min(opt.upperB, u));
                    U[j] = u;
                }
            }
            std::sort(U.begin(), U.end());

            double new_fit = fobj(U, h);
            if(new_fit < fitness[i]){
                pop[i] = U;
                fitness[i] = new_fit;
                if(new_fit < best_fit){
                    best_fit = new_fit;
                    best = U;
                }
            }
        }
        conv[iter] = best_fit;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    DEResult res;
    res.best_fitness = best_fit;
    res.best_thresholds = best;
    res.convergence = std::move(conv);
    res.elapsed_seconds = dt.count();
    return res;
}
