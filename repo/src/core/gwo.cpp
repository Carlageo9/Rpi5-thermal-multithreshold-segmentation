#include "gwo.hpp"
#include <opencv2/imgproc.hpp>

#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

static std::vector<double> normalized_hist_1based_gwo(const cv::Mat& gray){
    CV_Assert(gray.type() == CV_8UC1);
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    double total = (double)gray.total();
    std::vector<double> h(257, 0.0);
    for(int i=0;i<256;i++) h[i+1] = hist.at<float>(i) / total;
    return h;
}

GWOResult grey_wolf_optimize(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const GWOOptions& opt,
    const cv::Mat& grayImage
){
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<double> h = normalized_hist_1based_gwo(grayImage);

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_int_distribution<int> uniInt(opt.lowB, opt.upperB);

    const int N = opt.population;
    std::vector<std::vector<double>> X(N, std::vector<double>(nvars));
    std::vector<std::vector<int>> Xi(N, std::vector<int>(nvars));
    std::vector<double> fit(N);

    for(int i=0;i<N;i++){
        for(int j=0;j<nvars;j++) X[i][j] = (double)uniInt(rng);
        // integer copy (sorted) for objective
        for(int j=0;j<nvars;j++) Xi[i][j] = (int)std::lround(X[i][j]);
        std::sort(Xi[i].begin(), Xi[i].end());
        fit[i] = fobj(Xi[i], h);
    }

    auto argmin = [&](const std::vector<double>& v){
        return (int)std::distance(v.begin(), std::min_element(v.begin(), v.end()));
    };

    // Identify alpha, beta, delta
    int alpha_idx = argmin(fit);
    std::vector<double> Alpha = X[alpha_idx];
    double Alpha_fit = fit[alpha_idx];

    // find beta and delta by sorting indices
    std::vector<int> idx(N);
    for(int i=0;i<N;i++) idx[i]=i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return fit[a] < fit[b]; });
    std::vector<double> Beta = X[idx[1]];
    std::vector<double> Delta = X[idx[2]];

    std::vector<double> conv(opt.max_iterations);

    for(int iter=0; iter<opt.max_iterations; iter++){
        double a = 2.0 - 2.0 * ((double)iter / (double)opt.max_iterations);

        for(int i=0;i<N;i++){
            for(int j=0;j<nvars;j++){
                double r1 = uni01(rng), r2 = uni01(rng);
                double A1 = 2.0*a*r1 - a;
                double C1 = 2.0*r2;
                double D_alpha = std::fabs(C1*Alpha[j] - X[i][j]);
                double X1 = Alpha[j] - A1*D_alpha;

                r1 = uni01(rng); r2 = uni01(rng);
                double A2 = 2.0*a*r1 - a;
                double C2 = 2.0*r2;
                double D_beta = std::fabs(C2*Beta[j] - X[i][j]);
                double X2 = Beta[j] - A2*D_beta;

                r1 = uni01(rng); r2 = uni01(rng);
                double A3 = 2.0*a*r1 - a;
                double C3 = 2.0*r2;
                double D_delta = std::fabs(C3*Delta[j] - X[i][j]);
                double X3 = Delta[j] - A3*D_delta;

                double xnew = (X1 + X2 + X3) / 3.0;
                xnew = std::max((double)opt.lowB, std::min((double)opt.upperB, xnew));
                X[i][j] = xnew;
            }

            // evaluate
            for(int j=0;j<nvars;j++) Xi[i][j] = (int)std::lround(X[i][j]);
            for(int j=0;j<nvars;j++) Xi[i][j] = std::max(opt.lowB, std::min(opt.upperB, Xi[i][j]));
            std::sort(Xi[i].begin(), Xi[i].end());
            fit[i] = fobj(Xi[i], h);
        }

        // update leaders
        idx.resize(N);
        for(int i=0;i<N;i++) idx[i]=i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return fit[a] < fit[b]; });
        alpha_idx = idx[0];
        Alpha = X[alpha_idx];
        Alpha_fit = fit[alpha_idx];
        Beta = X[idx[1]];
        Delta = X[idx[2]];

        conv[iter] = Alpha_fit;
    }

    // best thresholds as ints
    std::vector<int> best_thr(nvars);
    for(int j=0;j<nvars;j++) best_thr[j] = (int)std::lround(Alpha[j]);
    for(int j=0;j<nvars;j++) best_thr[j] = std::max(opt.lowB, std::min(opt.upperB, best_thr[j]));
    std::sort(best_thr.begin(), best_thr.end());

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    GWOResult res;
    res.best_fitness = Alpha_fit;
    res.best_thresholds = std::move(best_thr);
    res.convergence = std::move(conv);
    res.elapsed_seconds = dt.count();
    return res;
}
