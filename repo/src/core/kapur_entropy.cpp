#include "kapur_entropy.hpp"
#include <cmath>
#include <algorithm>

static inline double mZero(int aa, int bb, const std::vector<double>& h){
    // sum h[aa : bb-1] (inclusive) with MATLAB-style 1-based indexing
    int bm1 = bb - 1;
    aa = std::max(1, std::min(256, aa));
    bm1 = std::max(1, std::min(256, bm1));
    if (bm1 < aa) return 0.0;
    double s = 0.0;
    for(int i=aa;i<=bm1;i++) s += h[i];
    return s;
}

double kapur_entropy(const std::vector<int>& x_in, const std::vector<double>& h){
    const int st = (int)x_in.size();
    std::vector<int> x = x_in;
    // ensure sorted
    std::sort(x.begin(), x.end());

    std::vector<double> nu(st+1, 0.0);

    for(int i=1; i<=st+1; i++){
        int ti, ti_1;
        if(i==1){
            ti = x[0];
            ti_1 = 1;
        } else if(i>st){
            ti = 256;
            ti_1 = x[st-1];
        } else {
            ti = x[i-1];
            ti_1 = x[i-2];
        }

        double prob = mZero(ti_1, ti, h);
        if(prob > 0.0){
            int start = std::max(1, std::min(256, ti_1));
            int end = std::max(1, std::min(256, ti-1));
            if(end >= start){
                double ent = 0.0;
                for(int k=start;k<=end;k++){
                    double p = h[k] / prob;
                    if(p>0.0) ent += -p * std::log(p);
                }
                nu[i-1] = ent;
            }
        }
    }

    double sum = 0.0;
    for(double v: nu) sum += v;
    return -sum;
}
