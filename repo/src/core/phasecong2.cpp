#include "phasecong2.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

static cv::Mat ifftshift(const cv::Mat& in){
    CV_Assert(in.type() == CV_64F);
    cv::Mat out(in.size(), CV_64F);
    const int r = in.rows;
    const int c = in.cols;
    const int r2 = r/2;
    const int c2 = c/2;

    for(int y=0;y<r;y++){
        int yy = (y + r2) % r;
        const double* sp = in.ptr<double>(yy);
        double* dp = out.ptr<double>(y);
        for(int x=0;x<c;x++){
            int xx = (x + c2) % c;
            dp[x] = sp[xx];
        }
    }
    return out;
}

static cv::Mat lowpassfilter(int rows, int cols, double cutoff, int n){
    CV_Assert(cutoff > 0.0 && cutoff < 0.5);
    CV_Assert(n >= 1);

    // Build radius as in MATLAB, then ifftshift.
    std::vector<double> xrange(cols);
    if(cols % 2){
        for(int i=0;i<cols;i++) xrange[i] = (i - (cols-1)/2.0) / (cols-1);
    } else {
        for(int i=0;i<cols;i++) xrange[i] = (i - cols/2.0) / cols;
    }
    std::vector<double> yrange(rows);
    if(rows % 2){
        for(int i=0;i<rows;i++) yrange[i] = (i - (rows-1)/2.0) / (rows-1);
    } else {
        for(int i=0;i<rows;i++) yrange[i] = (i - rows/2.0) / rows;
    }

    cv::Mat radius(rows, cols, CV_64F);
    for(int y=0;y<rows;y++){
        double* rp = radius.ptr<double>(y);
        for(int x=0;x<cols;x++){
            const double xx = xrange[x];
            const double yy = yrange[y];
            rp[x] = std::sqrt(xx*xx + yy*yy);
        }
    }
    radius = ifftshift(radius);

    // Butterworth low-pass.
    cv::Mat f(rows, cols, CV_64F);
    const double twoN = 2.0 * (double)n;
    for(int y=0;y<rows;y++){
        const double* rr = radius.ptr<double>(y);
        double* fp = f.ptr<double>(y);
        for(int x=0;x<cols;x++){
            const double w = rr[x];
            fp[x] = 1.0 / (1.0 + std::pow(w / cutoff, twoN));
        }
    }
    // MATLAB returns ifftshift(...) already done by above. Our radius is already shifted,
    // but lowpassfilter applies ifftshift to the final filter. Because we computed radius
    // using centered coordinates then ifftshifted, f already has origin in corners.
    return f;
}

static cv::Mat fft2_real_to_complex(const cv::Mat& im){
    CV_Assert(im.type() == CV_64F);
    cv::Mat complex;
    cv::dft(im, complex, cv::DFT_COMPLEX_OUTPUT);
    return complex;
}

static cv::Mat ifft2_complex_to_complex(const cv::Mat& freq){
    CV_Assert(freq.type() == CV_64FC2);
    cv::Mat out;
    cv::dft(freq, out, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
    return out;
}

static cv::Mat ifft2_real_to_real_scaled(const cv::Mat& freqReal, double scaleMult){
    CV_Assert(freqReal.type() == CV_64F);
    cv::Mat freqC(freqReal.size(), CV_64FC2);
    std::vector<cv::Mat> ch(2);
    ch[0] = freqReal;
    ch[1] = cv::Mat::zeros(freqReal.size(), CV_64F);
    cv::merge(ch, freqC);
    cv::Mat spatialC = ifft2_complex_to_complex(freqC);
    std::vector<cv::Mat> sch(2);
    cv::split(spatialC, sch);
    return sch[0] * scaleMult;
}

static double median_of_mat(const cv::Mat& m){
    CV_Assert(m.type() == CV_64F);
    std::vector<double> v;
    v.reserve((size_t)m.total());
    for(int y=0;y<m.rows;y++){
        const double* p = m.ptr<double>(y);
        v.insert(v.end(), p, p + m.cols);
    }
    const size_t mid = v.size()/2;
    std::nth_element(v.begin(), v.begin()+mid, v.end());
    return v[mid];
}

static cv::Mat mat_max(const cv::Mat& a, const cv::Mat& b){
    cv::Mat out;
    cv::max(a,b,out);
    return out;
}

static cv::Mat mat_max_scalar(const cv::Mat& a, double s){
    cv::Mat out;
    cv::max(a, s, out);
    return out;
}

} // namespace

cv::Mat phasecong2_pc2(const cv::Mat& im){
    CV_Assert(im.type() == CV_64F && im.channels() == 1);

    // Parameters (same as MATLAB reference code).
    const int nscale = 4;
    const int norient = 4;
    const double minWaveLength = 6.0;
    const double mult = 2.0;
    const double sigmaOnf = 0.55;
    const double dThetaOnSigma = 1.2;
    const double k = 2.0;
    const double epsilon = 1e-4;

    const int rows = im.rows;
    const int cols = im.cols;

    const double thetaSigma = CV_PI / (double)norient / dThetaOnSigma;

    // Frequency grids as in MATLAB, then ifftshift.
    std::vector<double> xrange(cols);
    if(cols % 2){
        for(int i=0;i<cols;i++) xrange[i] = (i - (cols-1)/2.0) / (cols-1);
    } else {
        for(int i=0;i<cols;i++) xrange[i] = (i - cols/2.0) / cols;
    }
    std::vector<double> yrange(rows);
    if(rows % 2){
        for(int i=0;i<rows;i++) yrange[i] = (i - (rows-1)/2.0) / (rows-1);
    } else {
        for(int i=0;i<rows;i++) yrange[i] = (i - rows/2.0) / rows;
    }

    cv::Mat radius(rows, cols, CV_64F);
    cv::Mat theta(rows, cols, CV_64F);
    for(int y=0;y<rows;y++){
        double* rp = radius.ptr<double>(y);
        double* tp = theta.ptr<double>(y);
        const double yy = yrange[y];
        for(int x=0;x<cols;x++){
            const double xx = xrange[x];
            rp[x] = std::sqrt(xx*xx + yy*yy);
            tp[x] = std::atan2(-yy, xx);
        }
    }
    radius = ifftshift(radius);
    theta  = ifftshift(theta);
    radius.at<double>(0,0) = 1.0;

    cv::Mat sintheta, costheta;
    // OpenCV does not provide cv::sin/cv::cos for cv::Mat on all builds.
    // Use polarToCart with unit magnitude to compute cos(theta) and sin(theta) element-wise.
    cv::Mat ones = cv::Mat::ones(theta.size(), theta.type());
    cv::polarToCart(ones, theta, costheta, sintheta, false);
cv::Mat lp = lowpassfilter(rows, cols, 0.45, 15);

    // log-Gabor radial filters.
    std::vector<cv::Mat> logGabor(nscale);
    const double logSigma = std::log(sigmaOnf);
    const double denom = 2.0 * logSigma * logSigma;
    for(int s=0;s<nscale;s++){
        const double wavelength = minWaveLength * std::pow(mult, (double)s);
        const double fo = 1.0 / wavelength;

        cv::Mat lg(rows, cols, CV_64F);
        for(int y=0;y<rows;y++){
            const double* rr = radius.ptr<double>(y);
            double* pp = lg.ptr<double>(y);
            for(int x=0;x<cols;x++){
                const double r = rr[x];
                const double val = std::exp( - (std::pow(std::log(r/fo), 2.0)) / denom );
                pp[x] = val;
            }
        }
        lg = lg.mul(lp);
        lg.at<double>(0,0) = 0.0;
        logGabor[s] = lg;
    }

    // Angular spreads.
    std::vector<cv::Mat> spread(norient);
    for(int o=0;o<norient;o++){
        const double angl = (double)o * CV_PI / (double)norient;
        cv::Mat ds = sintheta * std::cos(angl) - costheta * std::sin(angl);
        cv::Mat dc = costheta * std::cos(angl) + sintheta * std::sin(angl);
        cv::Mat dtheta;
        cv::phase(dc, ds, dtheta, false); // returns [0, 2pi)
        // Convert to absolute angular distance in [0, pi].
        cv::Mat twopi = cv::Mat(dtheta.size(), CV_64F, cv::Scalar(2.0*CV_PI));
        cv::Mat alt = twopi - dtheta;
        cv::min(dtheta, alt, dtheta);

        cv::Mat sp(rows, cols, CV_64F);
        for(int y=0;y<rows;y++){
            const double* dp = dtheta.ptr<double>(y);
            double* pp = sp.ptr<double>(y);
            for(int x=0;x<cols;x++){
                const double dt = dp[x];
                pp[x] = std::exp(-(dt*dt) / (2.0 * thetaSigma * thetaSigma));
            }
        }
        spread[o] = sp;
    }

    cv::Mat imagefft = fft2_real_to_complex(im);

    cv::Mat zero = cv::Mat::zeros(rows, cols, CV_64F);
    cv::Mat EnergyAll = cv::Mat::zeros(rows, cols, CV_64F);
    cv::Mat AnAll = cv::Mat::zeros(rows, cols, CV_64F);

    std::vector<double> estMeanE2n(norient, 0.0);

    const double sqrtMN = std::sqrt((double)rows * (double)cols);

    // Cache ifft of filters per scale per orientation build; per orientation we rebuild angular.
    for(int o=0;o<norient;o++){
        cv::Mat sumE = zero.clone();
        cv::Mat sumO = zero.clone();
        cv::Mat sumAn = zero.clone();
        cv::Mat Energy = zero.clone();

        std::vector<cv::Mat> EO_real(nscale), EO_imag(nscale), ifftFilt(nscale);

        double EM_n = 0.0;
        cv::Mat maxAn;

        for(int s=0;s<nscale;s++){
            cv::Mat filter = logGabor[s].mul(spread[o]);
            // ifftFilt = real(ifft2(filter))*sqrt(MN)
            ifftFilt[s] = ifft2_real_to_real_scaled(filter, sqrtMN);
            // EM_n = sum(sum(filter.^2)) at smallest scale
            if(s==0){
                cv::Mat f2 = filter.mul(filter);
                EM_n = cv::sum(f2)[0];
            }

            // EO = ifft2(imagefft .* filter)
            cv::Mat filterC(filter.size(), CV_64FC2);
            {
                std::vector<cv::Mat> ch(2);
                ch[0] = filter;
                ch[1] = cv::Mat::zeros(filter.size(), CV_64F);
                cv::merge(ch, filterC);
            }
            cv::Mat prod;
            cv::mulSpectrums(imagefft, filterC, prod, 0);
            cv::Mat EO = ifft2_complex_to_complex(prod);
            std::vector<cv::Mat> eoch(2);
            cv::split(EO, eoch);
            EO_real[s] = eoch[0];
            EO_imag[s] = eoch[1];

            cv::Mat An;
            cv::magnitude(EO_real[s], EO_imag[s], An);
            sumAn += An;
            sumE += EO_real[s];
            sumO += EO_imag[s];
            if(s==0) maxAn = An.clone();
            else cv::max(maxAn, An, maxAn);
        }

        cv::Mat XEnergy;
        cv::magnitude(sumE, sumO, XEnergy);
        XEnergy += epsilon;
        cv::Mat MeanE = sumE / XEnergy;
        cv::Mat MeanO = sumO / XEnergy;

        for(int s=0;s<nscale;s++){
            const cv::Mat& E = EO_real[s];
            const cv::Mat& O = EO_imag[s];
            Energy += E.mul(MeanE) + O.mul(MeanO) - cv::abs(E.mul(MeanO) - O.mul(MeanE));
        }

        // Noise compensation.
        // medianE2n = median(abs(EO{1,o}).^2)
        cv::Mat An1; cv::magnitude(EO_real[0], EO_imag[0], An1);
        cv::Mat E2n = An1.mul(An1);
        const double medianE2n = median_of_mat(E2n);
        const double meanE2n = -medianE2n / std::log(0.5);
        estMeanE2n[o] = meanE2n;

        const double noisePower = meanE2n / EM_n;

        cv::Mat EstSumAn2 = zero.clone();
        for(int s=0;s<nscale;s++) EstSumAn2 += ifftFilt[s].mul(ifftFilt[s]);

        cv::Mat EstSumAiAj = zero.clone();
        for(int si=0;si<nscale-1;si++){
            for(int sj=si+1;sj<nscale;sj++) EstSumAiAj += ifftFilt[si].mul(ifftFilt[sj]);
        }

        const double sumEstSumAn2 = cv::sum(EstSumAn2)[0];
        const double sumEstSumAiAj = cv::sum(EstSumAiAj)[0];

        const double EstNoiseEnergy2 = 2*noisePower*sumEstSumAn2 + 4*noisePower*sumEstSumAiAj;
        const double tau = std::sqrt(EstNoiseEnergy2/2.0);
        const double EstNoiseEnergy = tau*std::sqrt(CV_PI/2.0);
        const double EstNoiseEnergySigma = std::sqrt((2.0 - CV_PI/2.0)*tau*tau);
        double T = EstNoiseEnergy + k*EstNoiseEnergySigma;
        T = T/1.7;

        Energy = mat_max_scalar(Energy - T, 0.0);

        EnergyAll += Energy;
        AnAll += sumAn;
    }

    cv::Mat ResultPC;
    cv::divide(EnergyAll, AnAll + epsilon, ResultPC);
    // Clamp to [0,1] for numeric robustness.
    cv::min(ResultPC, 1.0, ResultPC);
    cv::max(ResultPC, 0.0, ResultPC);
    return ResultPC;
}
