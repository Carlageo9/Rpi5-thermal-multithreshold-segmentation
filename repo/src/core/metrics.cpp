#include "metrics.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <algorithm>

#include "phasecong2.hpp"

static cv::Mat toGrayDouble255(const cv::Mat& m) {
    cv::Mat gray;
    if (m.channels() == 1) {
        gray = m;
    } else {
        cv::cvtColor(m, gray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat d; gray.convertTo(d, CV_64F);
    return d;
}

static cv::Mat toDouble(const cv::Mat& m){
    cv::Mat d; m.convertTo(d, CV_64F);
    return d;
}

// Mean SSIM variant used in the original MATLAB mssim.m (windowed, not multiscale)
double mssim_windowed(const cv::Mat& a, const cv::Mat& b, int win){
    CV_Assert(a.size()==b.size() && a.type()==b.type());
    CV_Assert(win>=3 && (win%2==1));

    cv::Mat I1 = toDouble(a);
    cv::Mat I2 = toDouble(b);

    const double c1 = 0.0001;
    const double c2 = 0.0009;
    const int sz = win/2;

    double acc = 0.0;
    long long cnt = 0;
    for (int y = sz; y < a.rows - sz; ++y) {
        for (int x = sz; x < a.cols - sz; ++x) {
            cv::Rect r(x - sz, y - sz, win, win);
            cv::Mat roi1 = I1(r);
            if (!roi1.isContinuous()) roi1 = roi1.clone();
            cv::Mat p1 = roi1.reshape(1, 1);

            cv::Mat roi2 = I2(r);
            if (!roi2.isContinuous()) roi2 = roi2.clone();
            cv::Mat p2 = roi2.reshape(1, 1);
            cv::Scalar m1, s1, m2, s2;
            cv::meanStdDev(p1, m1, s1);
            cv::meanStdDev(p2, m2, s2);
            const double mu1 = m1[0];
            const double mu2 = m2[0];
            const double sig1 = s1[0];
            const double sig2 = s2[0];

            cv::Mat c1m = p1 - mu1;
            cv::Mat c2m = p2 - mu2;
            const double sig12 = cv::mean(c1m.mul(c2m))[0];

            const double num = (2*mu1*mu2 + c1) * (2*sig12 + c2);
            const double den = (mu1*mu1 + mu2*mu2 + c1) * (sig1*sig1 + sig2*sig2 + c2);
            const double v = (den != 0.0) ? (num/den) : 0.0;
            acc += v;
            cnt++;
        }
    }
    return (cnt>0) ? (acc / (double)cnt) : 0.0;
}

static inline double haar_log(double x, double alpha){
    return 1.0 / (1.0 + std::exp(-alpha * x));
}
static inline double haar_log_inv(double x, double alpha){
    // avoid division by zero
    const double eps = 1e-12;
    x = std::min(1.0 - eps, std::max(eps, x));
    return std::log(x / (1.0 - x)) / alpha;
}

static cv::Mat haar_subsample(const cv::Mat& img){
    cv::Mat blurred;
    cv::filter2D(img, blurred, -1, cv::Mat::ones(2,2,CV_64F)/4.0, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);
    cv::Mat sub( (blurred.rows+1)/2, (blurred.cols+1)/2, CV_64F );
    for(int r=0, rr=0; r<blurred.rows; r+=2, ++rr){
        const double* src = blurred.ptr<double>(r);
        double* dst = sub.ptr<double>(rr);
        for(int c=0, cc=0; c<blurred.cols; c+=2, ++cc){
            dst[cc] = src[c];
        }
    }
    return sub;
}

static cv::Mat conv_same(const cv::Mat& img, const cv::Mat& kernel){
    cv::Mat out;
    cv::filter2D(img, out, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);
    return out;
}

double haarpsi(const cv::Mat& a, const cv::Mat& b, bool preprocessWithSubsampling){
    CV_Assert(a.size()==b.size());
    CV_Assert(a.depth()==CV_8U || a.depth()==CV_16U || a.depth()==CV_32F || a.depth()==CV_64F);
    CV_Assert(b.type()==a.type());

    // Ensure double in [0,255]
    cv::Mat a8, b8;
    if (a.depth() == CV_8U) {
        a8 = a;
        b8 = b;
    } else {
        a.convertTo(a8, CV_8U);
        b.convertTo(b8, CV_8U);
    }

    const bool color = (a8.channels() == 3);
    const double C = 30.0;
    const double alpha = 4.2;

    cv::Mat Y1, Y2, I1, I2, Q1, Q2;
    if (color) {
        std::vector<cv::Mat> ch1, ch2;
        cv::split(a8, ch1);
        cv::split(b8, ch2);
        cv::Mat R1 = toDouble(ch1[2]); // OpenCV BGR
        cv::Mat G1 = toDouble(ch1[1]);
        cv::Mat B1 = toDouble(ch1[0]);
        cv::Mat R2 = toDouble(ch2[2]);
        cv::Mat G2 = toDouble(ch2[1]);
        cv::Mat B2 = toDouble(ch2[0]);
        Y1 = 0.299*R1 + 0.587*G1 + 0.114*B1;
        Y2 = 0.299*R2 + 0.587*G2 + 0.114*B2;
        I1 = 0.596*R1 - 0.274*G1 - 0.322*B1;
        I2 = 0.596*R2 - 0.274*G2 - 0.322*B2;
        Q1 = 0.211*R1 - 0.523*G1 + 0.312*B1;
        Q2 = 0.211*R2 - 0.523*G2 + 0.312*B2;
    } else {
        Y1 = toDouble(a8);
        Y2 = toDouble(b8);
    }

    if (preprocessWithSubsampling) {
        Y1 = haar_subsample(Y1);
        Y2 = haar_subsample(Y2);
        if (color) {
            I1 = haar_subsample(I1);
            I2 = haar_subsample(I2);
            Q1 = haar_subsample(Q1);
            Q2 = haar_subsample(Q2);
        }
    }

    const int nScales = 3;
    // coeffs: H1..Hn, V1..Vn
    std::vector<cv::Mat> coeffRef(2*nScales), coeffDist(2*nScales);
    for (int k = 1; k <= nScales; ++k) {
        const int sz = 1 << k;
        cv::Mat hf = cv::Mat::ones(sz, sz, CV_64F) * std::pow(2.0, -k);
        // first half rows negative
        hf.rowRange(0, sz/2) *= -1.0;
        coeffRef[k-1] = conv_same(Y1, hf);
        coeffDist[k-1] = conv_same(Y2, hf);
        coeffRef[k-1 + nScales] = conv_same(Y1, hf.t());
        coeffDist[k-1 + nScales] = conv_same(Y2, hf.t());
    }

    const int rows = Y1.rows;
    const int cols = Y1.cols;
    const int channels = color ? 3 : 2;
    std::vector<cv::Mat> localSim(channels), weights(channels);
    for (int i=0;i<channels;i++){
        localSim[i] = cv::Mat::zeros(rows, cols, CV_64F);
        weights[i] = cv::Mat::zeros(rows, cols, CV_64F);
    }

    for (int ori = 0; ori < 2; ++ori) {
        const int base = ori * nScales;
        // weights = max(|coeff(scale3)|)  (element-wise)
        // NOTE: OpenCV's cv::max() does NOT return a Mat; it writes into an output array.
        cv::Mat wRef = cv::abs(coeffRef[base + 2]);
        cv::Mat wDist = cv::abs(coeffDist[base + 2]);
        cv::Mat w;
        cv::max(wRef, wDist, w);
        weights[ori] = w;

        cv::Mat sim = cv::Mat::zeros(rows, cols, CV_64F);
        for (int s = 0; s < 2; ++s) {
            cv::Mat cr = cv::abs(coeffRef[base + s]);
            cv::Mat cd = cv::abs(coeffDist[base + s]);
            cv::Mat num = 2.0*cr.mul(cd) + C;
            cv::Mat den = cr.mul(cr) + cd.mul(cd) + C;
            cv::Mat frac;
            cv::divide(num, den, frac);
            sim += frac;
        }
        sim *= 0.5;
        localSim[ori] = sim;
    }

    if (color) {
        cv::Mat coeffRefQ = cv::abs(conv_same(Q1, cv::Mat::ones(2,2,CV_64F)/4.0));
        cv::Mat coeffDistQ = cv::abs(conv_same(Q2, cv::Mat::ones(2,2,CV_64F)/4.0));
        cv::Mat coeffRefI = cv::abs(conv_same(I1, cv::Mat::ones(2,2,CV_64F)/4.0));
        cv::Mat coeffDistI = cv::abs(conv_same(I2, cv::Mat::ones(2,2,CV_64F)/4.0));
        cv::Mat simI, simQ;
        cv::divide(2.0*coeffRefI.mul(coeffDistI) + C, coeffRefI.mul(coeffRefI) + coeffDistI.mul(coeffDistI) + C, simI);
        cv::divide(2.0*coeffRefQ.mul(coeffDistQ) + C, coeffRefQ.mul(coeffRefQ) + coeffDistQ.mul(coeffDistQ) + C, simQ);
        localSim[2] = 0.5 * (simI + simQ);
        weights[2] = 0.5 * (weights[0] + weights[1]);
    }

    // final score
    double wsum = 0.0;
    double acc = 0.0;
    for (int ch = 0; ch < channels; ++ch) {
        for (int r=0;r<rows;r++){
            const double* ls = localSim[ch].ptr<double>(r);
            const double* ww = weights[ch].ptr<double>(r);
            for (int c=0;c<cols;c++){
                const double w = ww[c];
                wsum += w;
                acc += haar_log(ls[c], alpha) * w;
            }
        }
    }
    if (wsum <= 1e-12) return 0.0;
    const double v = acc / wsum;
    const double inv = haar_log_inv(v, alpha);
    const double sim = inv * inv;
    return std::max(0.0, std::min(1.0, sim));
}

// Pragmatic FSIM approximation (gradient-only). This is NOT the full IEEE TIP FSIM implementation.
double fsim(const cv::Mat& a, const cv::Mat& b){
    // Exact FSIM (luminance only) ported from FeatureSIM.m (Zhang et al., TIP 2011).
    // Includes the original phasecong2 (PC_2) computation by Peter Kovesi.
    CV_Assert(a.size()==b.size());
    CV_Assert(a.type()==CV_8UC1 && b.type()==CV_8UC1);

    const int rows = a.rows;
    const int cols = a.cols;
    const int minDim = std::min(rows, cols);
    const int F = std::max(1, (int)std::lround((double)minDim / 256.0));

    cv::Mat Y1 = toDouble(a);
    cv::Mat Y2 = toDouble(b);

    // Downsample with average filtering, like the MATLAB reference.
    if(F > 1){
        cv::blur(Y1, Y1, cv::Size(F,F));
        cv::blur(Y2, Y2, cv::Size(F,F));
        cv::Mat d1((rows + F - 1)/F, (cols + F - 1)/F, CV_64F);
        cv::Mat d2(d1.size(), CV_64F);
        for(int y=0,yy=0;y<rows;y+=F,yy++){
            for(int x=0,xx=0;x<cols;x+=F,xx++){
                d1.at<double>(yy,xx) = Y1.at<double>(y,x);
                d2.at<double>(yy,xx) = Y2.at<double>(y,x);
            }
        }
        Y1 = d1;
        Y2 = d2;
    }

    // Phase congruency maps (PC_2).
    cv::Mat PC1 = phasecong2_pc2(Y1);
    cv::Mat PC2 = phasecong2_pc2(Y2);

    // Gradient maps (Scharr-like masks from the MATLAB reference code).
    const cv::Mat dx = (cv::Mat_<double>(3,3) << 3,0,-3, 10,0,-10, 3,0,-3) / 16.0;
    const cv::Mat dy = (cv::Mat_<double>(3,3) << 3,10,3, 0,0,0, -3,-10,-3) / 16.0;

    cv::Mat Ix1, Iy1, Ix2, Iy2;
    cv::filter2D(Y1, Ix1, -1, dx, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(Y1, Iy1, -1, dy, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(Y2, Ix2, -1, dx, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(Y2, Iy2, -1, dy, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

    cv::Mat GM1, GM2;
    cv::magnitude(Ix1, Iy1, GM1);
    cv::magnitude(Ix2, Iy2, GM2);

    const double T1 = 0.85;
    const double T2 = 160.0;

    cv::Mat PCSim = (2.0 * PC1.mul(PC2) + T1) / (PC1.mul(PC1) + PC2.mul(PC2) + T1);
    cv::Mat GSim  = (2.0 * GM1.mul(GM2) + T2) / (GM1.mul(GM1) + GM2.mul(GM2) + T2);

    cv::Mat PCm; cv::max(PC1, PC2, PCm);
    cv::Mat Sim = GSim.mul(PCSim).mul(PCm);
    const double num = cv::sum(Sim)[0];
    const double den = cv::sum(PCm)[0];
    return (den > 1e-12) ? (num/den) : 0.0;
}

double mse(const cv::Mat& a, const cv::Mat& b){
    CV_Assert(a.size()==b.size() && a.type()==b.type());
    cv::Mat diff; cv::absdiff(a,b,diff);
    cv::Mat diff2; diff.convertTo(diff2, CV_64F);
    diff2 = diff2.mul(diff2);
    return cv::mean(diff2)[0];
}

double psnr(const cv::Mat& a, const cv::Mat& b){
    double m = mse(a,b);
    if(m <= 1e-12) return 99.0;
    return 10.0 * std::log10((255.0*255.0)/m);
}

// Standard SSIM implementation for single-channel 8-bit images.
double ssim(const cv::Mat& a, const cv::Mat& b){
    CV_Assert(a.size()==b.size() && a.type()==CV_8UC1 && b.type()==CV_8UC1);

    const double L = 255.0;
    const double K1 = 0.01;
    const double K2 = 0.03;
    const double C1 = (K1*L)*(K1*L);
    const double C2 = (K2*L)*(K2*L);

    cv::Mat I1 = toDouble(a);
    cv::Mat I2 = toDouble(b);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11,11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11,11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1.mul(I1), sigma1_2, cv::Size(11,11), 1.5);
    sigma1_2 -= mu1_2;
    cv::GaussianBlur(I2.mul(I2), sigma2_2, cv::Size(11,11), 1.5);
    sigma2_2 -= mu2_2;
    cv::GaussianBlur(I1.mul(I2), sigma12, cv::Size(11,11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1 = 2*mu1_mu2 + C1;
    cv::Mat t2 = 2*sigma12 + C2;
    cv::Mat t3 = mu1_2 + mu2_2 + C1;
    cv::Mat t4 = sigma1_2 + sigma2_2 + C2;

    cv::Mat ssim_map = (t1.mul(t2)) / (t3.mul(t4));
    return cv::mean(ssim_map)[0];
}

// Universal Image Quality Index (Wang & Bovik), box window, "valid" region.
double uqi(const cv::Mat& a, const cv::Mat& b, int kernelSize){
    CV_Assert(a.size()==b.size() && a.type()==CV_8UC1 && b.type()==CV_8UC1);
    const int N = kernelSize*kernelSize;

    cv::Mat I1 = toDouble(a);
    cv::Mat I2 = toDouble(b);

    cv::Mat sumFilter = cv::Mat::ones(kernelSize, kernelSize, CV_64F);

    cv::Mat img1_sum, img2_sum, img1_sq_sum, img2_sq_sum, img12_sum;
    cv::filter2D(I1, img1_sum, -1, sumFilter, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(I2, img2_sum, -1, sumFilter, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(I1.mul(I1), img1_sq_sum, -1, sumFilter, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(I2.mul(I2), img2_sq_sum, -1, sumFilter, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(I1.mul(I2), img12_sum, -1, sumFilter, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

    // "valid" crop
    int pad = kernelSize/2;
    cv::Rect roi(pad, pad, a.cols - 2*pad, a.rows - 2*pad);
    img1_sum = img1_sum(roi);
    img2_sum = img2_sum(roi);
    img1_sq_sum = img1_sq_sum(roi);
    img2_sq_sum = img2_sq_sum(roi);
    img12_sum = img12_sum(roi);

    cv::Mat img12_sum_mul = img1_sum.mul(img2_sum);
    cv::Mat img12_sq_sum_mul = img1_sum.mul(img1_sum) + img2_sum.mul(img2_sum);

    cv::Mat top = 4.0 * (N*img12_sum - img12_sum_mul).mul(img12_sum_mul);
    cv::Mat bot = (N*(img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul).mul(img12_sq_sum_mul);

    cv::Mat divx;
    cv::divide(top, bot, divx);

    // mean excluding NaN
    double acc=0.0; int cnt=0;
    for(int r=0;r<divx.rows;r++){
        const double* p = divx.ptr<double>(r);
        for(int c=0;c<divx.cols;c++){
            double v = p[c];
            if(!std::isnan(v) && std::isfinite(v)) { acc += v; cnt++; }
        }
    }
    return (cnt>0) ? (acc/cnt) : 0.0;
}

// QILV - Quality Index based on Local Variance (Aja-Fernandez 2006)
double qilv(const cv::Mat& a, const cv::Mat& b, int win){
    CV_Assert(a.size()==b.size() && a.type()==CV_8UC1 && b.type()==CV_8UC1);

    const double L = 255.0;
    const double K1 = 0.01;
    const double K2 = 0.03;
    const double C1 = (K1*L)*(K1*L);
    const double C2 = (K2*L)*(K2*L);

    cv::Mat I1 = toDouble(a);
    cv::Mat I2 = toDouble(b);

    cv::Mat window = cv::Mat::ones(win, win, CV_64F);
    window /= cv::sum(window)[0];

    cv::Mat M1, M2;
    cv::filter2D(I1, M1, -1, window, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(I2, M2, -1, window, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

    cv::Mat V1, V2;
    cv::filter2D(I1.mul(I1), V1, -1, window, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(I2.mul(I2), V2, -1, window, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    V1 -= M1.mul(M1);
    V2 -= M2.mul(M2);

    // valid crop
    int pad = win/2;
    cv::Rect roi(pad, pad, a.cols - 2*pad, a.rows - 2*pad);
    V1 = V1(roi);
    V2 = V2(roi);

    cv::Scalar mean1, std1, mean2, std2;
    cv::meanStdDev(V1, mean1, std1);
    cv::meanStdDev(V2, mean2, std2);
    double m1 = mean1[0], m2 = mean2[0];
    double s1 = std1[0], s2 = std2[0];

    cv::Mat V1c = V1 - m1;
    cv::Mat V2c = V2 - m2;
    double s12 = cv::mean(V1c.mul(V2c))[0];

    double ind1 = (2*m1*m2 + C1) / (m1*m1 + m2*m2 + C1);
    double ind2 = (2*s1*s2 + C2) / (s1*s1 + s2*s2 + C2);
    double ind3 = (s12 + C2/2.0) / (s1*s2 + C2/2.0);
    return ind1 * ind2 * ind3;
}
