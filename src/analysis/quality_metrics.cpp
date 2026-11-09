#include "mirage/analysis/quality_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace mirage::analysis {

namespace {

constexpr double kPi = std::numbers::pi;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kPow25To7 = 6103515625.0; // 25^7

[[nodiscard]] inline double deg_to_rad(double deg) noexcept {
    return deg * kDegToRad;
}

[[nodiscard]] inline double rad_to_deg(double rad) noexcept {
    return rad * kRadToDeg;
}

} // namespace

std::string QualityMetrics::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(5)
        << "MAE=" << mean_absolute_error
        << ", RMSE=" << root_mean_square_error
        << ", PSNR=" << std::setprecision(2) << peak_signal_to_noise_ratio << " dB"
        << ", Rel L2=" << std::setprecision(5) << relative_l2_error
        << ", SSIM=" << std::setprecision(4) << ssim
        << ", Mean dE00=" << std::setprecision(3) << mean_ciede2000
        << ", Max dE00=" << std::setprecision(3) << max_ciede2000;
    return oss.str();
}

double compute_ciede2000(
    double L1, double a1, double b1,
    double L2, double a2, double b2) noexcept {
    
    const double L_bar = (L1 + L2) * 0.5;
    const double C1 = std::sqrt(a1 * a1 + b1 * b1);
    const double C2 = std::sqrt(a2 * a2 + b2 * b2);
    const double C_bar = (C1 + C2) * 0.5;

    const double C_bar7 = std::pow(C_bar, 7.0);
    const double G = 0.5 * (1.0 - std::sqrt(C_bar7 / (C_bar7 + kPow25To7)));

    const double a1_prime = (1.0 + G) * a1;
    const double a2_prime = (1.0 + G) * a2;

    const double C1_prime = std::sqrt(a1_prime * a1_prime + b1 * b1);
    const double C2_prime = std::sqrt(a2_prime * a2_prime + b2 * b2);

    auto compute_hue_deg = [](double a_p, double b) noexcept -> double {
        if (std::abs(a_p) < 1e-9 && std::abs(b) < 1e-9) {
            return 0.0;
        }
        double h = rad_to_deg(std::atan2(b, a_p));
        if (h < 0.0) {
            h += 360.0;
        }
        return h;
    };

    const double h1_prime = compute_hue_deg(a1_prime, b1);
    const double h2_prime = compute_hue_deg(a2_prime, b2);

    const double delta_L_prime = L2 - L1;
    const double delta_C_prime = C2_prime - C1_prime;

    double delta_h_prime = 0.0;
    if (C1_prime * C2_prime > 1e-9) {
        const double diff = h2_prime - h1_prime;
        if (std::abs(diff) <= 180.0) {
            delta_h_prime = diff;
        } else if (diff > 180.0) {
            delta_h_prime = diff - 360.0;
        } else {
            delta_h_prime = diff + 360.0;
        }
    }

    const double delta_H_prime = 2.0 * std::sqrt(C1_prime * C2_prime) *
                                 std::sin(deg_to_rad(delta_h_prime * 0.5));

    const double L_bar_prime = L_bar;
    const double C_bar_prime = (C1_prime + C2_prime) * 0.5;

    double h_bar_prime = 0.0;
    if (C1_prime * C2_prime > 1e-9) {
        const double diff = std::abs(h1_prime - h2_prime);
        if (diff <= 180.0) {
            h_bar_prime = (h1_prime + h2_prime) * 0.5;
        } else if (h1_prime + h2_prime < 360.0) {
            h_bar_prime = (h1_prime + h2_prime + 360.0) * 0.5;
        } else {
            h_bar_prime = (h1_prime + h2_prime - 360.0) * 0.5;
        }
    } else {
        h_bar_prime = h1_prime + h2_prime;
    }

    const double T = 1.0 -
                     0.17 * std::cos(deg_to_rad(h_bar_prime - 30.0)) +
                     0.24 * std::cos(deg_to_rad(2.0 * h_bar_prime)) +
                     0.32 * std::cos(deg_to_rad(3.0 * h_bar_prime + 6.0)) -
                     0.20 * std::cos(deg_to_rad(4.0 * h_bar_prime - 63.0));

    const double delta_theta = 30.0 * std::exp(-std::pow((h_bar_prime - 275.0) / 25.0, 2.0));

    const double C_bar_prime7 = std::pow(C_bar_prime, 7.0);
    const double R_C = 2.0 * std::sqrt(C_bar_prime7 / (C_bar_prime7 + kPow25To7));

    const double L_minus_50_sq = std::pow(L_bar_prime - 50.0, 2.0);
    const double S_L = 1.0 + (0.015 * L_minus_50_sq) / std::sqrt(20.0 + L_minus_50_sq);
    const double S_C = 1.0 + 0.045 * C_bar_prime;
    const double S_H = 1.0 + 0.015 * C_bar_prime * T;

    const double R_T = -std::sin(deg_to_rad(2.0 * delta_theta)) * R_C;

    // kL = kC = kH = 1.0
    const double term_L = delta_L_prime / S_L;
    const double term_C = delta_C_prime / S_C;
    const double term_H = delta_H_prime / S_H;

    const double dE2 = term_L * term_L +
                       term_C * term_C +
                       term_H * term_H +
                       R_T * term_C * term_H;

    return std::sqrt(std::max(0.0, dE2));
}

namespace {

[[nodiscard]] double compute_ssim_single_channel(const cv::Mat& ref, const cv::Mat& target) {
    constexpr double C1 = 0.0001; // (K1 * L)^2 with K1=0.01, L=1.0
    constexpr double C2 = 0.0009; // (K2 * L)^2 with K2=0.03, L=1.0

    cv::Mat mu_ref, mu_target;
    cv::GaussianBlur(ref, mu_ref, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(target, mu_target, cv::Size(11, 11), 1.5);

    cv::Mat mu_ref_sq = mu_ref.mul(mu_ref);
    cv::Mat mu_target_sq = mu_target.mul(mu_target);
    cv::Mat mu_ref_target = mu_ref.mul(mu_target);

    cv::Mat ref_sq = ref.mul(ref);
    cv::Mat target_sq = target.mul(target);
    cv::Mat ref_target = ref.mul(target);

    cv::Mat sigma_ref_sq, sigma_target_sq, sigma_ref_target;
    cv::GaussianBlur(ref_sq, sigma_ref_sq, cv::Size(11, 11), 1.5);
    sigma_ref_sq -= mu_ref_sq;

    cv::GaussianBlur(target_sq, sigma_target_sq, cv::Size(11, 11), 1.5);
    sigma_target_sq -= mu_target_sq;

    cv::GaussianBlur(ref_target, sigma_ref_target, cv::Size(11, 11), 1.5);
    sigma_ref_target -= mu_ref_target;

    cv::Mat num = (2.0 * mu_ref_target + C1).mul(2.0 * sigma_ref_target + C2);
    cv::Mat den = (mu_ref_sq + mu_target_sq + C1).mul(sigma_ref_sq + sigma_target_sq + C2);

    cv::Mat ssim_map;
    cv::divide(num, den, ssim_map);

    cv::Scalar mean_val = cv::mean(ssim_map);
    return mean_val[0];
}

} // namespace

double compute_ssim(const cv::Mat& ref, const cv::Mat& target) {
    if (ref.size() != target.size() || ref.type() != target.type()) {
        return 0.0;
    }

    if (ref.channels() == 1) {
        return compute_ssim_single_channel(ref, target);
    }

    std::vector<cv::Mat> ref_channels;
    std::vector<cv::Mat> target_channels;
    cv::split(ref, ref_channels);
    cv::split(target, target_channels);

    double total_ssim = 0.0;
    for (size_t c = 0; c < ref_channels.size(); ++c) {
        total_ssim += compute_ssim_single_channel(ref_channels[c], target_channels[c]);
    }
    return total_ssim / static_cast<double>(ref_channels.size());
}

QualityMetrics evaluate_quality(const cv::Mat& ref_bgr, const cv::Mat& target_bgr) {
    QualityMetrics metrics{};

    if (ref_bgr.empty() || target_bgr.empty() || ref_bgr.size() != target_bgr.size()) {
        return metrics;
    }

    // 1. Numerical metrics on normalized [0, 1] BGR image
    cv::Mat diff;
    cv::absdiff(ref_bgr, target_bgr, diff);

    // Max absolute error
    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(diff.reshape(1), &min_val, &max_val);
    metrics.max_absolute_error = max_val;

    // MAE
    cv::Scalar mean_diff = cv::mean(diff);
    metrics.mean_absolute_error = (mean_diff[0] + mean_diff[1] + mean_diff[2]) / 3.0;

    // RMSE
    cv::Mat diff_sq = diff.mul(diff);
    cv::Scalar mean_sq = cv::mean(diff_sq);
    double mse = (mean_sq[0] + mean_sq[1] + mean_sq[2]) / 3.0;
    metrics.root_mean_square_error = std::sqrt(mse);

    // PSNR
    if (mse <= 1e-12) {
        metrics.peak_signal_to_noise_ratio = 100.0;
    } else {
        metrics.peak_signal_to_noise_ratio = 10.0 * std::log10(1.0 / mse);
    }

    // Relative L2
    double norm_diff = cv::norm(diff, cv::NORM_L2);
    double norm_ref = cv::norm(ref_bgr, cv::NORM_L2);
    metrics.relative_l2_error = norm_diff / (norm_ref + 1e-12);

    // 2. SSIM
    metrics.ssim = compute_ssim(ref_bgr, target_bgr);

    // 3. CIEDE2000
    cv::Mat ref_lab, target_lab;
    cv::cvtColor(ref_bgr, ref_lab, cv::COLOR_BGR2Lab);
    cv::cvtColor(target_bgr, target_lab, cv::COLOR_BGR2Lab);

    double total_dE = 0.0;
    double max_dE = 0.0;
    const int rows = ref_lab.rows;
    const int cols = ref_lab.cols;
    const size_t total_pixels = static_cast<size_t>(rows) * static_cast<size_t>(cols);

    for (int y = 0; y < rows; ++y) {
        const auto* r_ptr = ref_lab.ptr<const cv::Vec3f>(y);
        const auto* t_ptr = target_lab.ptr<const cv::Vec3f>(y);
        for (int x = 0; x < cols; ++x) {
            const double L1 = static_cast<double>(r_ptr[x][0]);
            const double a1 = static_cast<double>(r_ptr[x][1]);
            const double b1 = static_cast<double>(r_ptr[x][2]);

            const double L2 = static_cast<double>(t_ptr[x][0]);
            const double a2 = static_cast<double>(t_ptr[x][1]);
            const double b2 = static_cast<double>(t_ptr[x][2]);

            const double dE = compute_ciede2000(L1, a1, b1, L2, a2, b2);
            total_dE += dE;
            if (dE > max_dE) {
                max_dE = dE;
            }
        }
    }

    metrics.mean_ciede2000 = total_pixels > 0 ? (total_dE / static_cast<double>(total_pixels)) : 0.0;
    metrics.max_ciede2000 = max_dE;

    return metrics;
}

} // namespace mirage::analysis

