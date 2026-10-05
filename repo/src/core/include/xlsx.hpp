#pragma once
#include <string>
#include <vector>

struct XlsxRow {
    // 14 columns following MATLAB titles:
    // BstFit, Bstsol, MeanBstFitt, STD BstFit, PSNR, Ssim, Fsim, UQI, MSE, QILV, MSSIM, HAAR, BestSoluAver, Time
    double bstFit = 0.0;
    std::string bstSol;
    double meanBstFit = 0.0;
    double stdBstFit = 0.0;
    double psnr = 0.0;
    double ssim = 0.0;
    double fsim = 0.0;
    double uqi = 0.0;
    double mse = 0.0;
    double qilv = 0.0;
    double mssim = 0.0;
    double haar = 0.0;
    std::string bestSoluAver;
    double time_sec = 0.0;
};

// Writes an XLSX if libxlsxwriter is available; otherwise writes a CSV with the same base name.
// Path may be .xlsx or .csv; the implementation will decide.
bool write_results_table(const std::string& outPathXlsx, const std::vector<XlsxRow>& rows);
