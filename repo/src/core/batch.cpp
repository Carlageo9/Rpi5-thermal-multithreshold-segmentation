#include "batch.hpp"

#include "kapur_entropy.hpp"
#include "de.hpp"
#include "gwo.hpp"
#include "pso.hpp"
#include "csa.hpp"
#include "dessa.hpp"
#include "scheduled_de.hpp"
#include "firefly.hpp"
#include "image_gray.hpp"
#include "metrics.hpp"
#include "histogram_plot.hpp"
#include "plot.hpp"
#include "xlsx.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <fstream>
#include <iomanip>

namespace fs = std::filesystem;

static bool is_image_file(const fs::path& p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext==".png" || ext==".jpg" || ext==".jpeg" || ext==".bmp" || ext==".tif" || ext==".tiff";
}

static std::string base_name_no_ext(const fs::path& p) {
    return p.stem().string();
}

static double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
}

static double stdev_sample(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (double)(v.size()-1));
}

static std::string vec_to_matlab_str(const std::vector<int>& x) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < x.size(); ++i) {
        if (i) oss << " ";
        oss << x[i];
    }
    oss << "]";
    return oss.str();
}

static std::vector<int> average_thresholds(const std::vector<std::vector<int>>& sols) {
    if (sols.empty()) return {};
    const int n = (int)sols[0].size();
    std::vector<double> acc(n, 0.0);
    for (const auto& s : sols) {
        for (int i = 0; i < n; ++i) acc[i] += s[i];
    }
    std::vector<int> out(n);
    for (int i = 0; i < n; ++i) out[i] = (int)std::round(acc[i] / (double)sols.size());
    std::sort(out.begin(), out.end());
    return out;
}

static std::vector<fs::path> list_images(const std::string& dir) {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.is_regular_file() && is_image_file(e.path())) files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

static void ensure_dir(const fs::path& p) {
    fs::create_directories(p);
}

static void save_convergence_csv(const std::vector<double>& convergence, const fs::path& outPath) {
    std::ofstream f(outPath);
    if (!f) return;
    f << "iteration,best_fitness\n";
    f << std::setprecision(17);
    for (size_t i = 0; i < convergence.size(); ++i) {
        f << (i + 1) << ',' << convergence[i] << "\n";
    }
}

static std::string run_id(int zero_based_run) {
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << (zero_based_run + 1);
    return oss.str();
}

int run_batch(const BatchConfig& cfg) {
    if (cfg.input_dir.empty()) {
        std::cerr << "batch: input_dir is empty\n";
        return 2;
    }
    if (!fs::exists(cfg.input_dir)) {
        std::cerr << "batch: input_dir does not exist: " << cfg.input_dir << "\n";
        return 3;
    }

    const auto images = list_images(cfg.input_dir);
    if (images.empty()) {
        std::cerr << "batch: no images found in: " << cfg.input_dir << "\n";
        return 4;
    }

    const std::string algo = cfg.algorithm;
    if (!(algo == "DE" || algo == "DESSA" || algo == "SDE" || algo == "GWO" || algo == "PSO" || algo == "CSA" || algo == "FA")) {
        std::cerr << "batch: unsupported algorithm: " << algo << " (use DE, DESSA, SDE, GWO, PSO, CSA or FA)\n";
        return 5;
    }

    fs::path outRoot = cfg.out_dir.empty() ? fs::path("results") : fs::path(cfg.out_dir);
    const fs::path algoDir = outRoot / ("Results_" + algo);
    const fs::path imgDir = algoDir / "imgResults";
    const fs::path convergenceDir = algoDir / "convergence";
    ensure_dir(imgDir);
    ensure_dir(convergenceDir);

    // One table per algorithm (like MATLAB per file). We'll name it Batch_<algo>.xlsx
    const fs::path xlsxPath = algoDir / ("Batch_" + algo + "_Mth.xlsx");

    std::vector<XlsxRow> rows;
    rows.reserve(images.size() * cfg.levels_list.size());

    for (const auto& p : images) {
        cv::Mat im = cv::imread(p.string(), cv::IMREAD_GRAYSCALE);
        if (im.empty()) {
            std::cerr << "batch: failed to read: " << p << "\n";
            continue;
        }
        const std::string name = base_name_no_ext(p);

        for (int levels : cfg.levels_list) {
            std::vector<double> bestFitStore;
            bestFitStore.reserve(cfg.runs);

            std::vector<std::vector<int>> bestSolStore;
            bestSolStore.reserve(cfg.runs);

            std::vector<double> psnrV, ssimV, fsimV, uqiV, mseV, qilvV, mssimV, haarV, timeV;
            psnrV.reserve(cfg.runs); ssimV.reserve(cfg.runs); fsimV.reserve(cfg.runs);
            uqiV.reserve(cfg.runs); mseV.reserve(cfg.runs); qilvV.reserve(cfg.runs);
            mssimV.reserve(cfg.runs); haarV.reserve(cfg.runs); timeV.reserve(cfg.runs);

            std::vector<double> first_convergence;

            for (int k = 0; k < cfg.runs; ++k) {
                std::vector<int> bestThr;
                double bestFit = 0.0;
                std::vector<double> convergence;
                double elapsed = 0.0;

                if (algo == "DE") {
                    DEOptions opt;
                    opt.population = cfg.population;
                    opt.max_iterations = cfg.iterations;
                    opt.F = cfg.F;
                    opt.CR = cfg.CR;
                    opt.lowB = 1;
                    opt.upperB = 256;
                    auto r = differential_evolution(kapur_entropy, levels, opt, im);
                    bestFit = r.best_fitness;
                    bestThr = r.best_thresholds;
                    convergence = r.convergence;
                    elapsed = r.elapsed_seconds;
                } else if (algo == "SDE") {
                    ScheduledDEOptions opt;
                    opt.population = cfg.population;
                    opt.max_iterations = cfg.iterations;
                    opt.lowB = 1;
                    opt.upperB = 256;
                    // Keep the notebook defaults for F_original/F_smooth/gamma.
                    // Only the population size and iteration budget come from the
                    // generic batch configuration.
                    auto r = scheduled_differential_evolution(kapur_entropy, levels, opt, im);
                    bestFit = r.best_fitness;
                    bestThr = r.best_thresholds;
                    convergence = r.convergence;
                    elapsed = r.elapsed_seconds;
                } else if (algo == "GWO") {
                    GWOOptions opt;
                    opt.population = cfg.population;
                    opt.max_iterations = cfg.iterations;
                    opt.lowB = 1;
                    opt.upperB = 256;
                    auto r = grey_wolf_optimize(kapur_entropy, levels, opt, im);
                    bestFit = r.best_fitness;
                    bestThr = r.best_thresholds;
                    convergence = r.convergence;
                    elapsed = r.elapsed_seconds;
                } else if (algo == "PSO") {
                    PSOOptions opt;
                    opt.population = cfg.population;
                    opt.max_iterations = cfg.iterations;
                    opt.lowB = 1;
                    opt.upperB = 256;
                    opt.w = cfg.pso_w;
                    opt.c1 = cfg.pso_c1;
                    opt.c2 = cfg.pso_c2;
                    auto r = particle_swarm_optimize(kapur_entropy, levels, opt, im);
                    bestFit = r.best_fitness;
                    bestThr = r.best_thresholds;
                    convergence = r.convergence;
                    elapsed = r.elapsed_seconds;
                } else if (algo == "FA") {
                    FireflyOptions opt;
                    opt.population = cfg.population;
                    opt.max_iterations = cfg.iterations;
                    opt.alpha = cfg.fa_alpha;
                    opt.beta0 = cfg.fa_beta0;
                    opt.gamma = cfg.fa_gamma;
                    opt.lowB = 1;
                    opt.upperB = 256;
                    auto r = firefly_optimize(kapur_entropy, levels, opt, im);
                    bestFit = r.best_fitness;
                    bestThr = r.best_thresholds;
                    convergence = r.convergence;
                    elapsed = r.elapsed_seconds;
                } else if (algo == "CSA") {
                    CSAOptions opt;
                    opt.population = cfg.population;
                    opt.max_iterations = cfg.iterations;
                    opt.lowB = 1;
                    opt.upperB = 256;
                    opt.fl = cfg.csa_fl;
                    opt.AP = cfg.csa_AP;
                    auto r = crow_search_optimize(kapur_entropy, levels, opt, im);
                    bestFit = r.best_fitness;
                    bestThr = r.best_thresholds;
                    convergence = r.convergence;
                    elapsed = r.elapsed_seconds;
                } else {
                    DESSAOptions opt;
                    opt.population = cfg.population;
                    opt.max_iterations = cfg.iterations;
                    opt.F = cfg.F;
                    opt.CR = cfg.CR;
                    opt.lowB = 1;
                    opt.upperB = 256;
                    auto r = dessa_optimize(kapur_entropy, levels, opt, im);
                    bestFit = r.best_fitness;
                    bestThr = r.best_thresholds;
                    convergence = r.convergence;
                    elapsed = r.elapsed_seconds;
                }

                bestFitStore.push_back(bestFit);
                bestSolStore.push_back(bestThr);

                // Segment using best thresholds of this run for metrics
                cv::Mat seg = image_gray_segment(im, bestThr);

                psnrV.push_back(psnr(im, seg));
                ssimV.push_back(ssim(im, seg));
                fsimV.push_back(cfg.compute_fsim ? fsim(im, seg) : 0.0);
                uqiV.push_back(uqi(im, seg, 3));
                mseV.push_back(mse(im, seg));
                qilvV.push_back(qilv(im, seg, 3));
                mssimV.push_back(mssim_windowed(im, seg, 3));
                haarV.push_back(haarpsi(im, seg, true));
                timeV.push_back(elapsed);

                const std::string convergenceStem = algo + "_" + name + "_thr" + std::to_string(levels)
                    + "_run" + run_id(k) + "_convergence";
                save_convergence_csv(convergence, convergenceDir / (convergenceStem + ".csv"));
                save_convergence_plot_png(
                    convergence,
                    algo + " " + name + " thr" + std::to_string(levels) + " run " + std::to_string(k + 1),
                    (convergenceDir / (convergenceStem + ".png")).string()
                );

                if (k == 0) {
                    first_convergence = convergence;
                }
            }

            const auto avgThr = average_thresholds(bestSolStore);
            cv::Mat segAvg = image_gray_segment(im, avgThr);
            cv::Mat segVis;
            segAvg.convertTo(segVis, CV_8U);
            // Save segmented image (normalized to 0..255 like MATLAB mat2gray)
            cv::Mat segNorm;
            cv::normalize(segVis, segNorm, 0, 255, cv::NORM_MINMAX);
            const std::string imgName = "Segmented_" + name + "_thr" + std::to_string(levels) + ".png";
            cv::imwrite((imgDir / imgName).string(), segNorm);

            // Histogram with threshold lines
            std::vector<int> thr255;
            thr255.reserve(avgThr.size());
            for (int t : avgThr) thr255.push_back(std::max(0, std::min(255, t-1)));
            const std::string histName = algo + "_" + name + "_thr" + std::to_string(levels) + "_hist.png";
            save_histogram_plot(im, thr255, (algoDir / histName).string());

            // Convergence plot (first run)
            const std::string convName = algo + "_" + name + "_thr" + std::to_string(levels) + "_convergence.png";
            save_convergence_plot_png(first_convergence, algo + " " + name + " thr" + std::to_string(levels), (algoDir / convName).string());

            // Row summary (match MATLAB columns)
            XlsxRow row;
            row.bstFit = *std::min_element(bestFitStore.begin(), bestFitStore.end());
            // choose thresholds corresponding to best fit
            const int bestIdx = (int)(std::min_element(bestFitStore.begin(), bestFitStore.end()) - bestFitStore.begin());
            row.bstSol = vec_to_matlab_str(bestSolStore[bestIdx]);
            row.meanBstFit = mean(bestFitStore);
            row.stdBstFit = stdev_sample(bestFitStore);
            row.psnr = mean(psnrV);
            row.ssim = mean(ssimV);
            row.fsim = mean(fsimV);
            row.uqi = mean(uqiV);
            row.mse = mean(mseV);
            row.qilv = mean(qilvV);
            row.mssim = mean(mssimV);
            row.haar = mean(haarV);
            row.bestSoluAver = vec_to_matlab_str(avgThr);
            row.time_sec = mean(timeV);
            rows.push_back(row);

            std::cout << "[" << algo << "] " << name << " thr=" << levels << " done\n";
        }
    }

    if (!write_results_table(xlsxPath.string(), rows)) {
        std::cerr << "batch: failed to write results table: " << xlsxPath << "\n";
        return 6;
    }

    return 0;
}
