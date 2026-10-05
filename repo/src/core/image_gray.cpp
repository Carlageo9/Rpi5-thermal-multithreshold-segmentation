#include "image_gray.hpp"
#include <algorithm>

cv::Mat image_gray_segment(const cv::Mat& gray, const std::vector<int>& thresholds_in){
    CV_Assert(gray.type() == CV_8UC1);
    std::vector<int> thr = thresholds_in;
    std::sort(thr.begin(), thr.end());

    std::vector<int> limites;
    limites.reserve(thr.size()+2);
    limites.push_back(0);
    for(int t: thr){
        int tv = std::max(0, std::min(255, t)); // clamp to pixel domain
        limites.push_back(tv);
    }
    limites.push_back(255);

    cv::Mat out(gray.size(), CV_8UC1, cv::Scalar(0));

    for(int r=0;r<gray.rows;r++){
        const uint8_t* src = gray.ptr<uint8_t>(r);
        uint8_t* dst = out.ptr<uint8_t>(r);
        for(int c=0;c<gray.cols;c++){
            int pix = (int)src[c];
            // find interval [limites[k], limites[k+1]]
            for(size_t k=0; k+1<limites.size(); k++){
                if(pix >= limites[k] && pix <= limites[k+1]){
                    dst[c] = (uint8_t)limites[k];
                    break;
                }
            }
        }
    }
    return out;
}
