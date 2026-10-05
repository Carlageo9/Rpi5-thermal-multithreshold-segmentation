#include "plot.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>

void save_convergence_plot_png(const std::vector<double>& y, const std::string& title, const std::string& outPath) {
    if (y.empty()) return;

    const int W = 900;
    const int H = 500;
    const int marginL = 60, marginR = 20, marginT = 40, marginB = 50;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(255,255,255));

    // Axes
    cv::line(img, {marginL, H - marginB}, {W - marginR, H - marginB}, {0,0,0}, 2);
    cv::line(img, {marginL, marginT}, {marginL, H - marginB}, {0,0,0}, 2);

    double ymin = *std::min_element(y.begin(), y.end());
    double ymax = *std::max_element(y.begin(), y.end());
    if (ymax - ymin < 1e-12) { ymax = ymin + 1.0; }

    auto mapX = [&](int i)->int {
        const double t = (y.size()==1) ? 0.0 : (double)i / (double)(y.size()-1);
        return marginL + (int)std::round(t * (W - marginL - marginR));
    };
    auto mapY = [&](double v)->int {
        const double t = (v - ymin) / (ymax - ymin);
        return (H - marginB) - (int)std::round(t * (H - marginT - marginB));
    };

    // Plot line
    for (size_t i = 1; i < y.size(); ++i) {
        cv::line(img, {mapX((int)i-1), mapY(y[i-1])}, {mapX((int)i), mapY(y[i])}, {0,0,0}, 2);
    }

    // Title
    cv::putText(img, title, {marginL, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,0,0}, 2);
    cv::putText(img, "Iterations", {W/2 - 60, H - 15}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 2);
    cv::putText(img, "Fitness", {10, H/2}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 2);

    cv::imwrite(outPath, img);
}
