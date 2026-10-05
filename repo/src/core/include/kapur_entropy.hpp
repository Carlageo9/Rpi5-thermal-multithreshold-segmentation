#pragma once
#include <vector>

// Histogram is 1-based index: h[1..256] (h[0] unused)
// Thresholds x are integers in [1,256] and should be sorted ascending.
// Returns fitness = -sum(entropies) (lower is better)
double kapur_entropy(const std::vector<int>& x, const std::vector<double>& h);
