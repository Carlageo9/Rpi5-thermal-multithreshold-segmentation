#include "scheduled_de.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

// Histogram helper identical in spirit to the other optimizers in the project:
// build a normalized histogram with MATLAB-like 1-based indexing.
std::vector<double> normalized_hist_1based(const cv::Mat& gray) {
    CV_Assert(gray.type() == CV_8UC1);

    constexpr int histSize = 256;
    float range[] = {0.0f, 256.0f};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    const double total = static_cast<double>(gray.total());
    std::vector<double> h(257, 0.0); // use slots 1..256
    for (int i = 0; i < histSize; ++i) {
        h[i + 1] = hist.at<float>(i) / total;
    }
    return h;
}

inline double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

// Convert a continuous candidate into the discrete threshold representation used
// by the segmentation objective: integers in bounds and sorted ascending.
std::vector<int> discretize_candidate(const std::vector<double>& x, int lowB, int upperB) {
    std::vector<int> out(x.size(), lowB);
    for (size_t j = 0; j < x.size(); ++j) {
        out[j] = clampi(static_cast<int>(std::lround(x[j])), lowB, upperB);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Binomial crossover with the same guarantee used in the notebook: at least one
// dimension is copied from the mutant vector.
std::vector<double> crossover_binomial(
    const std::vector<double>& parent,
    const std::vector<double>& mutant,
    double CR,
    std::mt19937& rng
) {
    const int dim = static_cast<int>(parent.size());
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_int_distribution<int> pick_dim(0, dim - 1);

    std::vector<double> trial(dim, 0.0);
    const int forced_j = pick_dim(rng);
    for (int j = 0; j < dim; ++j) {
        const bool take_mutant = (uni01(rng) < CR) || (j == forced_j);
        trial[j] = take_mutant ? mutant[j] : parent[j];
    }
    return trial;
}

// Select three distinct population indices. This mirrors the notebook, which
// samples from the whole population without excluding the current target i.
std::array<int, 3> sample_three_distinct(int population, std::mt19937& rng) {
    if (population < 3) {
        throw std::runtime_error("Scheduled DE requires at least 3 individuals.");
    }

    std::vector<int> idx(population);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    return {idx[0], idx[1], idx[2]};
}

// Build the phase schedule used by the notebook, but rescaled to the iteration
// budget requested by the C++ program. This preserves the relative importance of
// original/smooth phases even when the user asks for, e.g., only 50 iterations.
struct PhaseBlock {
    bool smooth_phase = false;
    int iterations = 0;
};

std::vector<PhaseBlock> build_scaled_schedule(int max_iterations) {
    // Reference schedule copied from the notebook: base_block * 2.
    static const std::vector<std::pair<bool, int>> reference = {
        {false,120}, {true,120},
        {false,200}, {true,100},
        {false,150}, {true,150},
        {false,120}, {true,180},
        {false, 80}, {true,120},
        {false,162}, {true,140},
        {false,180}, {true,177},
        {false,120}, {true,120},
        {false,200}, {true,100},
        {false,150}, {true,150},
        {false,120}, {true,180},
        {false, 80}, {true,120},
        {false,162}, {true,140},
        {false,180}, {true,177}
    };

    const int total_reference = 2018;
    if (max_iterations <= 0) return {};

    // Largest-remainder scaling: floor every block, then distribute the leftover
    // iterations to the blocks with the biggest fractional parts.
    std::vector<double> raw(reference.size(), 0.0);
    std::vector<int> scaled(reference.size(), 0);
    std::vector<std::pair<double, int>> remainders;
    remainders.reserve(reference.size());

    int assigned = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        raw[i] = (static_cast<double>(max_iterations) * reference[i].second) /
                 static_cast<double>(total_reference);
        scaled[i] = static_cast<int>(std::floor(raw[i]));
        assigned += scaled[i];
        remainders.push_back({raw[i] - static_cast<double>(scaled[i]), static_cast<int>(i)});
    }

    std::sort(remainders.begin(), remainders.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (int k = 0; k < max_iterations - assigned; ++k) {
        scaled[remainders[k].second] += 1;
    }

    std::vector<PhaseBlock> schedule;
    schedule.reserve(reference.size());
    for (size_t i = 0; i < reference.size(); ++i) {
        if (scaled[i] > 0) {
            schedule.push_back({reference[i].first, scaled[i]});
        }
    }
    return schedule;
}

} // namespace

ScheduledDEResult scheduled_differential_evolution(
    const std::function<double(const std::vector<int>&, const std::vector<double>&)>& fobj,
    int nvars,
    const ScheduledDEOptions& opt,
    const cv::Mat& grayImage
) {
    CV_Assert(grayImage.type() == CV_8UC1);

    const auto t0 = std::chrono::high_resolution_clock::now();
    const std::vector<double> hist = normalized_hist_1based(grayImage);

    const int NP = std::max(3, opt.population);
    const int maxGen = std::max(1, opt.max_iterations);
    const double lowB = static_cast<double>(opt.lowB);
    const double upperB = static_cast<double>(opt.upperB);

    std::mt19937 rng;
    if (opt.seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(opt.seed);
    }
    std::uniform_real_distribution<double> uniform_position(lowB, upperB);

    // Population is stored in continuous space, because the notebook mutates
    // real-valued vectors and only the objective function sees the discretized
    // threshold version.
    std::vector<std::vector<double>> pop(NP, std::vector<double>(nvars, lowB));
    for (int i = 0; i < NP; ++i) {
        for (int j = 0; j < nvars; ++j) {
            pop[i][j] = uniform_position(rng);
        }
    }

    std::vector<std::vector<int>> pop_discrete(NP);
    std::vector<double> fitness(NP, 0.0);
    for (int i = 0; i < NP; ++i) {
        pop_discrete[i] = discretize_candidate(pop[i], opt.lowB, opt.upperB);
        fitness[i] = fobj(pop_discrete[i], hist);
    }

    int best_idx = static_cast<int>(std::distance(fitness.begin(), std::min_element(fitness.begin(), fitness.end())));
    double best_fit = fitness[best_idx];
    std::vector<int> best_thr = pop_discrete[best_idx];

    std::vector<double> convergence;
    convergence.reserve(maxGen);

    // Keep the last 'history_window' continuous populations, exactly because the
    // smooth phase uses the MEAN population as a memory term.
    std::deque<std::vector<std::vector<double>>> history;
    history.push_back(pop);

    const auto schedule = build_scaled_schedule(maxGen);
    for (const auto& block : schedule) {
        for (int local_iter = 0; local_iter < block.iterations; ++local_iter) {
            // smooth_mean[idx][j] = average value of variable j for individual idx
            // across the recent populations stored in 'history'.
            std::vector<std::vector<double>> smooth_mean;
            if (block.smooth_phase) {
                smooth_mean.assign(NP, std::vector<double>(nvars, 0.0));
                for (const auto& historical_pop : history) {
                    for (int i = 0; i < NP; ++i) {
                        for (int j = 0; j < nvars; ++j) {
                            smooth_mean[i][j] += historical_pop[i][j];
                        }
                    }
                }
                const double inv_count = 1.0 / static_cast<double>(history.size());
                for (int i = 0; i < NP; ++i) {
                    for (int j = 0; j < nvars; ++j) {
                        smooth_mean[i][j] *= inv_count;
                    }
                }
            }

            for (int i = 0; i < NP; ++i) {
                const auto sel = sample_three_distinct(NP, rng);
                const int a = sel[0];
                const int b = sel[1];
                const int c = sel[2];

                std::vector<double> mutant(nvars, lowB);
                for (int j = 0; j < nvars; ++j) {
                    // Original phase: x_a + F * (x_b - x_c)
                    // Smooth   phase: (1-gamma) * original_mutant + gamma * mean_pop[a]
                    const double classical = pop[a][j] + opt.F_original * (pop[b][j] - pop[c][j]);
                    if (!block.smooth_phase) {
                        mutant[j] = clampd(classical, lowB, upperB);
                    } else {
                        const double classical_smooth = pop[a][j] + opt.F_smooth * (pop[b][j] - pop[c][j]);
                        const double blended = (1.0 - opt.gamma) * classical_smooth
                                             + opt.gamma * smooth_mean[a][j];
                        mutant[j] = clampd(blended, lowB, upperB);
                    }
                }

                const std::vector<double> trial_cont = crossover_binomial(pop[i], mutant, opt.CR, rng);
                const std::vector<int> trial_disc = discretize_candidate(trial_cont, opt.lowB, opt.upperB);
                const double trial_fit = fobj(trial_disc, hist);

                // Greedy selection: keep the new solution only if it improves.
                if (trial_fit < fitness[i]) {
                    pop[i] = trial_cont;
                    pop_discrete[i] = trial_disc;
                    fitness[i] = trial_fit;

                    if (trial_fit < best_fit) {
                        best_fit = trial_fit;
                        best_thr = trial_disc;
                    }
                }
            }

            convergence.push_back(best_fit);
            history.push_back(pop);
            while (static_cast<int>(history.size()) > std::max(1, opt.history_window)) {
                history.pop_front();
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    ScheduledDEResult res;
    res.best_fitness = best_fit;
    res.best_thresholds = best_thr;
    res.convergence = std::move(convergence);
    res.elapsed_seconds = dt.count();
    return res;
}
