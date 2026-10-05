#include "histogram_plot.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>

void save_histogram_plot(const cv::Mat& gray, const std::vector<int>& thresholds_in, const std::string& outPath){
    CV_Assert(gray.type()==CV_8UC1);
    std::vector<int> thr = thresholds_in;
    for(int& t: thr) t = std::max(0, std::min(255, t));
    std::sort(thr.begin(), thr.end());

    int histSize=256;
    float range[] = {0,256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray,1,0,cv::Mat(),hist,1,&histSize,&histRange,true,false);

    double maxVal=0; cv::minMaxLoc(hist,nullptr,&maxVal);

    const int W=900, H=500;
    cv::Mat canvas(H,W,CV_8UC3,cv::Scalar(255,255,255));

    int marginL=60, marginR=20, marginT=20, marginB=60;
    cv::Rect plot(marginL, marginT, W-marginL-marginR, H-marginT-marginB);

    // axes
    cv::rectangle(canvas, plot, cv::Scalar(0,0,0), 1);

    // bars
    for(int i=0;i<256;i++){
        float v = hist.at<float>(i);
        int x0 = plot.x + (int)std::floor((double)i/256.0 * plot.width);
        int x1 = plot.x + (int)std::floor((double)(i+1)/256.0 * plot.width);
        int barW = std::max(1, x1-x0);
        int barH = (maxVal>0) ? (int)std::round((v/maxVal)*plot.height) : 0;
        cv::rectangle(canvas, cv::Rect(x0, plot.y + plot.height - barH, barW, barH), cv::Scalar(200,200,200), cv::FILLED);
    }

    // threshold lines (red-ish)
    for(int t: thr){
        int x = plot.x + (int)std::round(((double)t/255.0) * plot.width);
        cv::line(canvas, cv::Point(x, plot.y), cv::Point(x, plot.y+plot.height), cv::Scalar(0,0,255), 2, cv::LINE_AA);
    }

    // x labels
    for(int v=0; v<=255; v+=50){
        int x = plot.x + (int)std::round(((double)v/255.0) * plot.width);
        cv::line(canvas, cv::Point(x, plot.y+plot.height), cv::Point(x, plot.y+plot.height+5), cv::Scalar(0,0,0), 1);
        cv::putText(canvas, std::to_string(v), cv::Point(x-10, plot.y+plot.height+30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    }
    cv::putText(canvas, "Gray level", cv::Point(plot.x + plot.width/2 - 50, H-15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);
    cv::putText(canvas, "Frequency", cv::Point(10, plot.y + plot.height/2), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);

    cv::imwrite(outPath, canvas);
}
