#pragma once
#include <string>
#include <vector>

// Saves a simple line plot (PNG) for convergence curve.
void save_convergence_plot_png(const std::vector<double>& y, const std::string& title, const std::string& outPath);
