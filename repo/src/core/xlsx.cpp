#include "xlsx.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>

#if defined(HAVE_XLSXWRITER)
  #include <xlsxwriter.h>
#endif

static std::vector<std::string> titles() {
    return {"BstFit","Bstsol","MeanBstFitt","STD BstFit","PSNR","Ssim","Fsim","UQI","MSE","QILV","MSSIM","HAAR","BestSoluAver","Time"};
}

static bool write_csv(const std::string& outPath, const std::vector<XlsxRow>& rows) {
    std::ofstream f(outPath);
    if (!f) return false;
    auto t = titles();
    for (size_t i = 0; i < t.size(); ++i) {
        if (i) f << ',';
        f << '"' << t[i] << '"';
    }
    f << "\n";
    f << std::setprecision(12);
    for (const auto& r : rows) {
        f << r.bstFit << ','
          << '"' << r.bstSol << '"' << ','
          << r.meanBstFit << ','
          << r.stdBstFit << ','
          << r.psnr << ','
          << r.ssim << ','
          << r.fsim << ','
          << r.uqi << ','
          << r.mse << ','
          << r.qilv << ','
          << r.mssim << ','
          << r.haar << ','
          << '"' << r.bestSoluAver << '"' << ','
          << r.time_sec
          << "\n";
    }
    return true;
}

bool write_results_table(const std::string& outPathXlsx, const std::vector<XlsxRow>& rows) {
    // If xlsxwriter is unavailable, always write CSV next to requested path.
#if !defined(HAVE_XLSXWRITER)
    std::string csv = outPathXlsx;
    if (csv.size() >= 5 && csv.substr(csv.size()-5) == ".xlsx") {
        csv = csv.substr(0, csv.size()-5) + ".csv";
    }
    return write_csv(csv, rows);
#else
    // Ensure .xlsx extension
    std::string path = outPathXlsx;
    if (!(path.size() >= 5 && path.substr(path.size()-5) == ".xlsx")) {
        path += ".xlsx";
    }

    lxw_workbook* workbook = workbook_new(path.c_str());
    if (!workbook) return false;
    lxw_worksheet* ws = workbook_add_worksheet(workbook, nullptr);
    if (!ws) {
        workbook_close(workbook);
        return false;
    }

    auto t = titles();
    for (int col = 0; col < (int)t.size(); ++col) {
        worksheet_write_string(ws, 0, col, t[col].c_str(), nullptr);
    }

    int rowi = 1;
    for (const auto& r : rows) {
        worksheet_write_number(ws, rowi, 0, r.bstFit, nullptr);
        worksheet_write_string(ws, rowi, 1, r.bstSol.c_str(), nullptr);
        worksheet_write_number(ws, rowi, 2, r.meanBstFit, nullptr);
        worksheet_write_number(ws, rowi, 3, r.stdBstFit, nullptr);
        worksheet_write_number(ws, rowi, 4, r.psnr, nullptr);
        worksheet_write_number(ws, rowi, 5, r.ssim, nullptr);
        worksheet_write_number(ws, rowi, 6, r.fsim, nullptr);
        worksheet_write_number(ws, rowi, 7, r.uqi, nullptr);
        worksheet_write_number(ws, rowi, 8, r.mse, nullptr);
        worksheet_write_number(ws, rowi, 9, r.qilv, nullptr);
        worksheet_write_number(ws, rowi, 10, r.mssim, nullptr);
        worksheet_write_number(ws, rowi, 11, r.haar, nullptr);
        worksheet_write_string(ws, rowi, 12, r.bestSoluAver.c_str(), nullptr);
        worksheet_write_number(ws, rowi, 13, r.time_sec, nullptr);
        rowi++;
    }

    const lxw_error err = workbook_close(workbook);
    return err == LXW_NO_ERROR;
#endif
}
