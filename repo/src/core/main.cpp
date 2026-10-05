#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <thread>

#include "de.hpp"
#include "gwo.hpp"
#include "pso.hpp"
#include "csa.hpp"
#include "dessa.hpp"
#include "scheduled_de.hpp"
#include "firefly.hpp"
#include "kapur_entropy.hpp"
#include "image_gray.hpp"
#include "metrics.hpp"
#include "histogram_plot.hpp"
#include "plot.hpp"
#include "batch.hpp"
#include "xlsx.hpp"
#include "thermal_capture.hpp"

namespace fs = std::filesystem;

static void print_usage(){
    std::cout << "Usage:\n"
              << "  Interactive (choose dataset + run ALL algorithms):\n"
              << "    segm --interactive [--thermals_dir <folder>] [--benchmark_dir <folder>] [--outdir results]\n\n"
              << "  Run ALL algorithms on a folder (non-interactive):\n"
              << "    segm --run_all --batch_dir <folder> [--levels_list 2,3,4,5] [--runs 40] [--pop 30] [--iter 50] [--F 0.5] [--CR 0.9] [--w 0.72] [--c1 1.49] [--c2 1.49] [--alpha 0.5] [--beta0 1.0] [--gamma 1.0] [--fl 2.0] [--AP 0.1] [--outdir results] [--disable_fsim]\n\n"
              << "  Single image:\n"
              << "    segm --image <path> --levels <k> [--algo DE|DESSA|SDE|GWO|PSO|CSA|FA] [--runs 40] [--pop 30] [--iter 50] [--F 0.5] [--CR 0.9] [--w 0.72] [--c1 1.49] [--c2 1.49] [--alpha 0.5] [--beta0 1.0] [--gamma 1.0] [--fl 2.0] [--AP 0.1] [--outdir results] [--disable_fsim]\n\n"
              << "  Thermal camera loop / live capture:\n"
              << "    segm --thermal_loop --capture_cmd \"python3 repo/src/tools/capture_thermal.py --out {output}\" [--algo ALL|DE|DESSA|SDE|GWO|PSO|CSA|FA] [--captures 0] [--interval 5] [--levels_list 2,3,4,5] [--runs 40] [--pop 30] [--iter 50] [--outdir results_live] [--disable_fsim]\n"
              << "    segm --thermal_once --capture_cmd \"python3 repo/src/tools/capture_thermal.py --out {output}\" [same options]\n"
              << "    segm --thermal_loop --camera_index 0 [same options]   # only for USB/UVC cameras visible as /dev/video*\n\n"
              << "  Batch folder (MATLAB-like):\n"
              << "    segm --batch_dir <folder> [--algo DE|DESSA|SDE|GWO|PSO|CSA|FA] [--levels_list 2,3,4,5] [--runs 40] [--pop 30] [--iter 50] [--F 0.5] [--CR 0.9] [--w 0.72] [--c1 1.49] [--c2 1.49] [--alpha 0.5] [--beta0 1.0] [--gamma 1.0] [--fl 2.0] [--AP 0.1] [--outdir results] [--disable_fsim]\n\n"
              << "Outputs:\n"
              << "  - segmented.png\n  - histogram.png\n  - convergence.csv\n  - summary.csv\n";
}

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def=""){
    for(int i=1;i<argc;i++){
        if(std::string(argv[i])==key && i+1<argc) return argv[i+1];
    }
    return def;
}

static int get_arg_int(int argc, char** argv, const std::string& key, int def){
    std::string s = get_arg(argc, argv, key, "");
    return s.empty()? def : std::stoi(s);
}

static double get_arg_double(int argc, char** argv, const std::string& key, double def){
    std::string s = get_arg(argc, argv, key, "");
    return s.empty()? def : std::stod(s);
}

static bool has_flag(int argc, char** argv, const std::string& key){
    for(int i=1;i<argc;i++) if(std::string(argv[i])==key) return true;
    return false;
}

static int run_all_algorithms_on_folder(const BatchConfig& baseCfg){
    const std::vector<std::string> algos = {"DE", "DESSA", "SDE", "GWO", "PSO", "CSA", "FA"};
    for(const auto& a : algos){
        BatchConfig cfg = baseCfg;
        cfg.algorithm = a;
        std::cout << "\n==== Running " << a << " on folder: " << cfg.input_dir << " ====\n";
        int rc = run_batch(cfg);
        if(rc != 0){
            std::cerr << "Error while running " << a << " (code " << rc << ")\n";
            return rc;
        }
    }
    return 0;
}

static std::string prompt_line(const std::string& label, const std::string& def){
    std::cout << label;
    if(!def.empty()) std::cout << " [default: " << def << "]";
    std::cout << ": ";
    std::string s;
    std::getline(std::cin, s);
    if(s.empty()) return def;
    return s;
}

static int run_interactive(int argc, char** argv){
    // Defaults (relative to current working directory)
    std::string thermalsDefault  = get_arg(argc, argv, "--thermals_dir", "datasets/termicas");
    std::string benchDefault     = get_arg(argc, argv, "--benchmark_dir", "datasets/benchmark_classic");
    std::string outRoot          = get_arg(argc, argv, "--outdir", "results");

    std::cout << "\nSelect the image dataset to process:\n";
    std::cout << "  1) Thermal images (termicas)\n";
    std::cout << "  2) Classic benchmark (benchmark classic)\n";
    std::cout << "Option [1/2]: ";
    std::string opt;
    std::getline(std::cin, opt);
    int choice = 1;
    try{ if(!opt.empty()) choice = std::stoi(opt); } catch(...) { choice = 1; }

    std::string inputDir;
    if(choice == 2){
        inputDir = prompt_line("Benchmark image folder path", benchDefault);
    } else {
        inputDir = prompt_line("Thermal image folder path", thermalsDefault);
    }

    // Common batch config (defaults match MATLAB-like runner)
    BatchConfig cfg;
    cfg.input_dir = inputDir;
    cfg.out_dir = outRoot;
    cfg.runs = get_arg_int(argc, argv, "--runs", 40);
    cfg.population = get_arg_int(argc, argv, "--pop", 30);
    cfg.iterations = get_arg_int(argc, argv, "--iter", 50);
    cfg.F = get_arg_double(argc, argv, "--F", 0.5);
    cfg.CR = get_arg_double(argc, argv, "--CR", 0.9);
    cfg.pso_w  = get_arg_double(argc, argv, "--w", 0.72);
    cfg.pso_c1 = get_arg_double(argc, argv, "--c1", 1.49);
    cfg.pso_c2 = get_arg_double(argc, argv, "--c2", 1.49);
    cfg.fa_alpha = get_arg_double(argc, argv, "--alpha", 0.5);
    cfg.fa_beta0 = get_arg_double(argc, argv, "--beta0", 1.0);
    cfg.fa_gamma = get_arg_double(argc, argv, "--gamma", 1.0);
    cfg.csa_fl = get_arg_double(argc, argv, "--fl", 2.0);
    cfg.csa_AP = get_arg_double(argc, argv, "--AP", 0.1);
    cfg.compute_fsim = (get_arg(argc, argv, "--disable_fsim", "").empty());

    std::string ll = get_arg(argc, argv, "--levels_list", "2,3,4,5");
    cfg.levels_list.clear();
    std::stringstream ss(ll);
    std::string token;
    while(std::getline(ss, token, ',')){
        try{ cfg.levels_list.push_back(std::stoi(token)); } catch(...){ }
    }
    if(cfg.levels_list.empty()) cfg.levels_list = {2,3,4,5};

    std::cout << "\nALL images found in the following folder will be processed: " << cfg.input_dir << "\n";
    std::cout << "Algorithms: DE, DESSA, SDE (Scheduled Hybrid DE), GWO, PSO, CSA (Crow), FA (Firefly)\n";
    std::cout << "Results in: " << cfg.out_dir << "\n";

    return run_all_algorithms_on_folder(cfg);
}


static std::string now_stamp(){
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

static std::vector<int> parse_levels_list(const std::string& ll){
    std::vector<int> levels;
    std::stringstream ss(ll);
    std::string token;
    while(std::getline(ss, token, ',')){
        try { levels.push_back(std::stoi(token)); } catch(...) {}
    }
    if(levels.empty()) levels = {2,3,4,5};
    return levels;
}

static BatchConfig batch_cfg_from_args(int argc, char** argv, const std::string& inputDir, const std::string& outDir, const std::string& algo){
    BatchConfig cfg;
    cfg.input_dir = inputDir;
    cfg.out_dir = outDir;
    cfg.algorithm = algo;
    cfg.runs = get_arg_int(argc, argv, "--runs", 40);
    cfg.population = get_arg_int(argc, argv, "--pop", 30);
    cfg.iterations = get_arg_int(argc, argv, "--iter", 50);
    cfg.F = get_arg_double(argc, argv, "--F", 0.5);
    cfg.CR = get_arg_double(argc, argv, "--CR", 0.9);
    cfg.pso_w  = get_arg_double(argc, argv, "--w", 0.72);
    cfg.pso_c1 = get_arg_double(argc, argv, "--c1", 1.49);
    cfg.pso_c2 = get_arg_double(argc, argv, "--c2", 1.49);
    cfg.fa_alpha = get_arg_double(argc, argv, "--alpha", 0.5);
    cfg.fa_beta0 = get_arg_double(argc, argv, "--beta0", 1.0);
    cfg.fa_gamma = get_arg_double(argc, argv, "--gamma", 1.0);
    cfg.csa_fl = get_arg_double(argc, argv, "--fl", 2.0);
    cfg.csa_AP = get_arg_double(argc, argv, "--AP", 0.1);
    cfg.compute_fsim = (get_arg(argc, argv, "--disable_fsim", "").empty());
    cfg.levels_list = parse_levels_list(get_arg(argc, argv, "--levels_list", "2,3,4,5"));
    return cfg;
}

static int run_thermal_loop(int argc, char** argv){
    std::string outRoot = get_arg(argc, argv, "--outdir", "results_live");
    std::string algo = get_arg(argc, argv, "--algo", "ALL");
    std::transform(algo.begin(), algo.end(), algo.begin(), ::toupper);

    int captures = get_arg_int(argc, argv, "--captures", has_flag(argc, argv, "--thermal_once") ? 1 : 0); // 0 = infinite
    int intervalSec = get_arg_int(argc, argv, "--interval", 5);

    ThermalCaptureConfig capCfg;
    capCfg.capture_command = get_arg(argc, argv, "--capture_cmd", "");
    capCfg.camera_index = get_arg_int(argc, argv, "--camera_index", -1);
    capCfg.warmup_ms = get_arg_int(argc, argv, "--warmup_ms", 300);

    const fs::path sessionRoot = fs::path(outRoot) / ("session_" + now_stamp());
    const fs::path capturesRoot = sessionRoot / "captures";
    const fs::path analysesRoot = sessionRoot / "analyses";
    fs::create_directories(capturesRoot);
    fs::create_directories(analysesRoot);

    std::cout << "Thermal mode started. Session folder: " << sessionRoot << "\n";
    std::cout << "captures= " << (captures == 0 ? std::string("infinite") : std::to_string(captures))
              << ", interval= " << intervalSec << " s, algo= " << algo << "\n";

    int n = 0;
    while(captures == 0 || n < captures){
        ++n;
        const std::string stamp = now_stamp();
        const fs::path frameDir = capturesRoot / ("capture_" + stamp);
        fs::create_directories(frameDir);
        const fs::path imgPath = frameDir / ("thermal_" + stamp + ".png");

        std::cout << "\n=== Capture " << n << " -> " << imgPath << " ===\n";
        if(!capture_thermal_image(capCfg, imgPath.string())){
            std::cerr << "Capture failed. Stopping thermal loop.\n";
            return 10;
        }

        const fs::path analysisOut = analysesRoot / ("capture_" + stamp);
        fs::create_directories(analysisOut);

        if(algo == "ALL"){
            BatchConfig cfg = batch_cfg_from_args(argc, argv, frameDir.string(), analysisOut.string(), "DE");
            int rc = run_all_algorithms_on_folder(cfg);
            if(rc != 0) return rc;
        } else {
            BatchConfig cfg = batch_cfg_from_args(argc, argv, frameDir.string(), analysisOut.string(), algo);
            int rc = run_batch(cfg);
            if(rc != 0) return rc;
        }

        if(captures != 0 && n >= captures) break;
        if(intervalSec > 0) std::this_thread::sleep_for(std::chrono::seconds(intervalSec));
    }

    std::cout << "\nThermal session finished. All previous results were preserved in: " << sessionRoot << "\n";
    return 0;
}

int main(int argc, char** argv){
    if(has_flag(argc, argv, "--thermal_loop") || has_flag(argc, argv, "--thermal_once")) {
        if(has_flag(argc, argv, "--thermal_once") && get_arg(argc, argv, "--captures", "").empty()) {
            // Users can request a one-shot capture without specifying --captures.
        }
        return run_thermal_loop(argc, argv);
    }

    // Interactive mode if requested or no arguments
    if(argc == 1 || has_flag(argc, argv, "--interactive")){
        return run_interactive(argc, argv);
    }

    // Run ALL algorithms on a batch folder (non-interactive)
    if(has_flag(argc, argv, "--run_all")){
        std::string batchDirAll = get_arg(argc, argv, "--batch_dir");
        if(batchDirAll.empty()){
            std::cerr << "--run_all requires --batch_dir <folder>\n";
            print_usage();
            return 1;
        }
        BatchConfig cfg;
        cfg.input_dir = batchDirAll;
        cfg.out_dir = get_arg(argc, argv, "--outdir", "results");
        cfg.runs = get_arg_int(argc, argv, "--runs", 40);
        cfg.population = get_arg_int(argc, argv, "--pop", 30);
        cfg.iterations = get_arg_int(argc, argv, "--iter", 50);
        cfg.F = get_arg_double(argc, argv, "--F", 0.5);
        cfg.CR = get_arg_double(argc, argv, "--CR", 0.9);
        cfg.pso_w  = get_arg_double(argc, argv, "--w", 0.72);
        cfg.pso_c1 = get_arg_double(argc, argv, "--c1", 1.49);
        cfg.pso_c2 = get_arg_double(argc, argv, "--c2", 1.49);
        cfg.fa_alpha = get_arg_double(argc, argv, "--alpha", 0.5);
        cfg.fa_beta0 = get_arg_double(argc, argv, "--beta0", 1.0);
        cfg.fa_gamma = get_arg_double(argc, argv, "--gamma", 1.0);
        cfg.csa_fl = get_arg_double(argc, argv, "--fl", 2.0);
        cfg.csa_AP = get_arg_double(argc, argv, "--AP", 0.1);
        cfg.compute_fsim = (get_arg(argc, argv, "--disable_fsim", "").empty());

        std::string ll = get_arg(argc, argv, "--levels_list", "2,3,4,5");
        cfg.levels_list.clear();
        std::stringstream ss(ll);
        std::string token;
        while(std::getline(ss, token, ',')){
            try{ cfg.levels_list.push_back(std::stoi(token)); } catch(...){ }
        }
        if(cfg.levels_list.empty()) cfg.levels_list = {2,3,4,5};

        return run_all_algorithms_on_folder(cfg);
    }

    // Batch mode
    std::string batchDir = get_arg(argc, argv, "--batch_dir");
    if(!batchDir.empty()){
        BatchConfig cfg;
        cfg.input_dir = batchDir;
        cfg.out_dir = get_arg(argc, argv, "--outdir", "results");
        cfg.algorithm = get_arg(argc, argv, "--algo", "DESSA");
        cfg.runs = get_arg_int(argc, argv, "--runs", 40);
        cfg.population = get_arg_int(argc, argv, "--pop", 30);
        cfg.iterations = get_arg_int(argc, argv, "--iter", 50);
        cfg.F = get_arg_double(argc, argv, "--F", 0.5);
        cfg.CR = get_arg_double(argc, argv, "--CR", 0.9);
        // PSO parameters (match user's MATLAB pso.m)
        cfg.pso_w  = get_arg_double(argc, argv, "--w", 0.72);
        cfg.pso_c1 = get_arg_double(argc, argv, "--c1", 1.49);
        cfg.pso_c2 = get_arg_double(argc, argv, "--c2", 1.49);
        cfg.fa_alpha = get_arg_double(argc, argv, "--alpha", 0.5);
        cfg.fa_beta0 = get_arg_double(argc, argv, "--beta0", 1.0);
        cfg.fa_gamma = get_arg_double(argc, argv, "--gamma", 1.0);
        cfg.csa_fl = get_arg_double(argc, argv, "--fl", 2.0);
        cfg.csa_AP = get_arg_double(argc, argv, "--AP", 0.1);
        cfg.compute_fsim = (get_arg(argc, argv, "--disable_fsim", "").empty());

        std::string ll = get_arg(argc, argv, "--levels_list", "2,3,4,5");
        cfg.levels_list.clear();
        std::stringstream ss(ll);
        std::string token;
        while(std::getline(ss, token, ',')){
            try{ cfg.levels_list.push_back(std::stoi(token)); } catch(...){}
        }
        if(cfg.levels_list.empty()) cfg.levels_list = {2,3,4,5};

        return run_batch(cfg);
    }

    std::string imgPath = get_arg(argc, argv, "--image");
    if(imgPath.empty()) { print_usage(); return 1; }

    int levels = get_arg_int(argc, argv, "--levels", 3);
    int runs = get_arg_int(argc, argv, "--runs", 40);

    const std::string algo = get_arg(argc, argv, "--algo", "DESSA");
    const bool compute_fsim = (get_arg(argc, argv, "--disable_fsim", "").empty());

    DEOptions opt_de;
    opt_de.population = get_arg_int(argc, argv, "--pop", 30);
    opt_de.max_iterations = get_arg_int(argc, argv, "--iter", 50);
    opt_de.F = get_arg_double(argc, argv, "--F", 0.5);
    opt_de.CR = get_arg_double(argc, argv, "--CR", 0.9);

    DESSAOptions opt_dessa;
    opt_dessa.population = opt_de.population;
    opt_dessa.max_iterations = opt_de.max_iterations;
    opt_dessa.F = opt_de.F;
    opt_dessa.CR = opt_de.CR;

    ScheduledDEOptions opt_sde;
    opt_sde.population = opt_de.population;
    opt_sde.max_iterations = opt_de.max_iterations;
    opt_sde.lowB = 1;
    opt_sde.upperB = 256;
    // Deliberately preserve the notebook defaults:
    // F_original = 0.8, F_smooth = 0.3, CR = 0.9, gamma = 0.2, window = 5.

    GWOOptions opt_gwo;
    opt_gwo.population = opt_de.population;
    opt_gwo.max_iterations = opt_de.max_iterations;
    opt_gwo.lowB = 1;
    opt_gwo.upperB = 256;

    PSOOptions opt_pso;
    opt_pso.population = opt_de.population;
    opt_pso.max_iterations = opt_de.max_iterations;
    opt_pso.lowB = 1;
    opt_pso.upperB = 256;
    opt_pso.w  = get_arg_double(argc, argv, "--w", 0.72);
    opt_pso.c1 = get_arg_double(argc, argv, "--c1", 1.49);
    opt_pso.c2 = get_arg_double(argc, argv, "--c2", 1.49);

    CSAOptions opt_csa;
    opt_csa.population = opt_de.population;
    opt_csa.max_iterations = opt_de.max_iterations;
    opt_csa.lowB = 1;
    opt_csa.upperB = 256;
    opt_csa.fl = get_arg_double(argc, argv, "--fl", 2.0);
    opt_csa.AP = get_arg_double(argc, argv, "--AP", 0.1);

    FireflyOptions opt_fa;
    opt_fa.population = opt_de.population;
    opt_fa.max_iterations = opt_de.max_iterations;
    opt_fa.lowB = 1;
    opt_fa.upperB = 256;
    opt_fa.alpha = get_arg_double(argc, argv, "--alpha", 0.5);
    opt_fa.beta0 = get_arg_double(argc, argv, "--beta0", 1.0);
    opt_fa.gamma = get_arg_double(argc, argv, "--gamma", 1.0);

    std::string outdir = get_arg(argc, argv, "--outdir", "results");
    fs::create_directories(outdir);
    const fs::path convergenceDir = fs::path(outdir) / "convergence";
    fs::create_directories(convergenceDir);

    cv::Mat im = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if(im.empty()){
        std::cerr << "Could not read image: " << imgPath << "\n";
        return 2;
    }

    std::vector<double> bestFits; bestFits.reserve(runs);
    std::vector<std::vector<int>> bestSols; bestSols.reserve(runs);
    std::vector<double> psnrs, ssims, fsims, uqis, mses, qilvs, mssims, haars, times;

    std::vector<double> firstConv;

    for(int r=0;r<runs;r++){
        std::vector<int> thr;
        double fit = 0.0;
        double tsec = 0.0;
        std::vector<double> conv;

        if(algo == "DE"){
            auto res = differential_evolution(kapur_entropy, levels, opt_de, im);
            fit = res.best_fitness;
            thr = res.best_thresholds;
            tsec = res.elapsed_seconds;
            conv = res.convergence;
        } else if(algo == "SDE"){
            auto res = scheduled_differential_evolution(kapur_entropy, levels, opt_sde, im);
            fit = res.best_fitness;
            thr = res.best_thresholds;
            tsec = res.elapsed_seconds;
            conv = res.convergence;
        } else if(algo == "GWO"){
            auto res = grey_wolf_optimize(kapur_entropy, levels, opt_gwo, im);
            fit = res.best_fitness;
            thr = res.best_thresholds;
            tsec = res.elapsed_seconds;
            conv = res.convergence;
        } else if(algo == "PSO"){
            auto res = particle_swarm_optimize(kapur_entropy, levels, opt_pso, im);
            fit = res.best_fitness;
            thr = res.best_thresholds;
            tsec = res.elapsed_seconds;
            conv = res.convergence;
        } else if(algo == "CSA"){
            auto res = crow_search_optimize(kapur_entropy, levels, opt_csa, im);
            fit = res.best_fitness;
            thr = res.best_thresholds;
            tsec = res.elapsed_seconds;
            conv = res.convergence;
        } else if(algo == "FA"){
            auto res = firefly_optimize(kapur_entropy, levels, opt_fa, im);
            fit = res.best_fitness;
            thr = res.best_thresholds;
            tsec = res.elapsed_seconds;
            conv = res.convergence;
        } else {
            auto res = dessa_optimize(kapur_entropy, levels, opt_dessa, im);
            fit = res.best_fitness;
            thr = res.best_thresholds;
            tsec = res.elapsed_seconds;
            conv = res.convergence;
        }

        bestFits.push_back(fit);
        bestSols.push_back(thr);
        times.push_back(tsec);
        if(r==0) firstConv = conv;

        {
            std::ostringstream runName;
            runName << algo << "_thr" << levels << "_run"
                    << std::setw(3) << std::setfill('0') << (r + 1) << "_convergence";
            const fs::path csvPath = convergenceDir / (runName.str() + ".csv");
            std::ofstream cf(csvPath);
            if (cf) {
                cf << "iteration,best_fitness\n";
                cf << std::setprecision(17);
                for (size_t i = 0; i < conv.size(); ++i) {
                    cf << (i + 1) << ',' << conv[i] << "\n";
                }
            }
            save_convergence_plot_png(
                conv,
                algo + " thr" + std::to_string(levels) + " run " + std::to_string(r + 1),
                (convergenceDir / (runName.str() + ".png")).string()
            );
        }

        cv::Mat seg = image_gray_segment(im, thr);
        psnrs.push_back(psnr(im, seg));
        ssims.push_back(ssim(im, seg));
        fsims.push_back(compute_fsim ? fsim(im, seg) : 0.0);
        uqis.push_back(uqi(im, seg, 3));
        mses.push_back(mse(im, seg));
        qilvs.push_back(qilv(im, seg, 3));
        mssims.push_back(mssim_windowed(im, seg, 3));
        haars.push_back(haarpsi(im, seg, true));
    }

    // pick best run
    int bestIdx = (int)std::distance(bestFits.begin(), std::min_element(bestFits.begin(), bestFits.end()));
    std::vector<int> bestThr = bestSols[bestIdx];
    std::sort(bestThr.begin(), bestThr.end());

    // mean thresholds (like TempVSolu/num_runs then fix/sort)
    std::vector<double> meanThr(levels, 0.0);
    for(const auto& sol: bestSols){
        for(int j=0;j<levels;j++) meanThr[j] += sol[j];
    }
    for(double& v: meanThr) v /= (double)runs;
    std::vector<int> avgThr(levels);
    for(int j=0;j<levels;j++) avgThr[j] = (int)std::floor(meanThr[j]);
    std::sort(avgThr.begin(), avgThr.end());

    cv::Mat segmented = image_gray_segment(im, avgThr);

    cv::imwrite((fs::path(outdir)/"segmented.png").string(), segmented);
    std::vector<int> thr255; thr255.reserve(avgThr.size());
    for(int t: avgThr) thr255.push_back(std::max(0, std::min(255, t-1)));
    save_histogram_plot(im, thr255, (fs::path(outdir)/"histogram.png").string());

    // Backward-compatible first-run convergence files.
    {
        std::ofstream f((fs::path(outdir)/"convergence.csv").string());
        f << "iter,best_fitness\n";
        f << std::setprecision(17);
        for(size_t i=0;i<firstConv.size();i++) f << (i+1) << "," << firstConv[i] << "\n";
    }
    save_convergence_plot_png(
        firstConv,
        algo + " thr" + std::to_string(levels) + " run 1",
        (fs::path(outdir)/"convergence.png").string()
    );

    auto mean_of = [](const std::vector<double>& v){
        if(v.empty()) return 0.0;
        return std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
    };
    auto std_of = [](const std::vector<double>& v){
        if(v.size()<2) return 0.0;
        double m = std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
        double acc=0.0;
        for(double x: v) acc += (x-m)*(x-m);
        return std::sqrt(acc / (double)(v.size()-1));
    };

    // summary.csv
    {
        std::ofstream f((fs::path(outdir)/"summary.csv").string());
        f << "algo,best_fit,best_thresholds,mean_best_fit,std_best_fit,mean_psnr,mean_ssim,mean_fsim,mean_uqi,mean_mse,mean_qilv,mean_mssim,mean_haar,mean_time_s,avg_thresholds\n";
        // serialize vectors
        auto vec_to_str = [](const std::vector<int>& a){
            std::string s="[";
            for(size_t i=0;i<a.size();i++){
                s += std::to_string(a[i]);
                if(i+1<a.size()) s += " ";
            }
            s += "]";
            return s;
        };
        f << algo << "," << bestFits[bestIdx] << "," << '"' << vec_to_str(bestThr) << '"' << ","
          << mean_of(bestFits) << "," << std_of(bestFits) << ","
          << mean_of(psnrs) << "," << mean_of(ssims) << "," << mean_of(fsims) << "," << mean_of(uqis) << ","
          << mean_of(mses) << "," << mean_of(qilvs) << "," << mean_of(mssims) << "," << mean_of(haars) << "," << mean_of(times) << ","
          << '"' << vec_to_str(avgThr) << '"' << "\n";
    }

    std::cout << "Done. Outputs in: " << outdir << "\n";
    std::cout << "Avg thresholds: ";
    for(int t: avgThr) std::cout << t << " ";
    std::cout << "\n";

    return 0;
}
