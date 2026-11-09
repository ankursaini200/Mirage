#include "mirage/enhancement/precision_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mirage::enhancement {

PrecisionMode parse_precision_mode(std::string_view str) {
    std::string lower;
    lower.reserve(str.size());
    for (char c : str) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (lower == "fp32") return PrecisionMode::FP32;
    if (lower == "fp16") return PrecisionMode::FP16;
    if (lower == "mixed") return PrecisionMode::Mixed;
    if (lower == "adaptive") return PrecisionMode::Adaptive;

    throw std::invalid_argument("Unrecognized precision mode '" + std::string(str) +
                                "' (supported: fp32, fp16, mixed, adaptive)");
}

std::string ErrorMetrics::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "Max Absolute Error:  " << max_absolute_error << "\n"
        << "Mean Absolute Error: " << mean_absolute_error << "\n"
        << "RMSE:                " << root_mean_square_error << "\n";

    if (std::isinf(peak_signal_to_noise_ratio)) {
        oss << "PSNR:                Infinity dB (Identical output)\n";
    } else {
        oss << "PSNR:                " << std::setprecision(2) << peak_signal_to_noise_ratio << " dB\n";
    }

    oss << "Relative L2 Error:   " << std::setprecision(4) << (relative_l2_error * 100.0) << " %\n"
        << "Mean Relative Error: " << std::setprecision(4) << (mean_relative_error * 100.0) << " %";
    return oss.str();
}

ErrorMetrics compute_error_metrics(const cv::Mat& reference_fp32, const cv::Mat& target) {
    if (reference_fp32.empty() || target.empty()) {
        throw std::invalid_argument("Cannot compute error metrics on empty images.");
    }
    if (reference_fp32.size() != target.size() || reference_fp32.type() != target.type()) {
        throw std::invalid_argument("Reference and target images must have identical dimensions and types.");
    }
    if (reference_fp32.type() != CV_32FC3) {
        throw std::invalid_argument("Error metrics require normalized CV_32FC3 images.");
    }

    ErrorMetrics metrics{};
    double max_err = 0.0;
    double sum_abs_err = 0.0;
    double sum_sq_err = 0.0;
    double sum_ref_sq = 0.0;
    double sum_rel_err = 0.0;
    size_t rel_count = 0;

    const size_t total_elements = static_cast<size_t>(reference_fp32.rows) *
                                  static_cast<size_t>(reference_fp32.cols) * 3;

    for (int r = 0; r < reference_fp32.rows; ++r) {
        const auto* ref_row = reference_fp32.ptr<cv::Vec3f>(r);
        const auto* tgt_row = target.ptr<cv::Vec3f>(r);

        for (int c = 0; c < reference_fp32.cols; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                const double ref_val = static_cast<double>(ref_row[c][ch]);
                const double tgt_val = static_cast<double>(tgt_row[c][ch]);
                const double diff = std::abs(tgt_val - ref_val);

                max_err = std::max(max_err, diff);
                sum_abs_err += diff;
                sum_sq_err += (diff * diff);
                sum_ref_sq += (ref_val * ref_val);

                if (std::abs(ref_val) > 1e-4) {
                    sum_rel_err += (diff / std::abs(ref_val));
                    rel_count++;
                }
            }
        }
    }

    metrics.max_absolute_error = max_err;
    metrics.mean_absolute_error = sum_abs_err / static_cast<double>(total_elements);

    const double mse = sum_sq_err / static_cast<double>(total_elements);
    metrics.root_mean_square_error = std::sqrt(mse);

    if (mse <= 1e-15) {
        metrics.peak_signal_to_noise_ratio = std::numeric_limits<double>::infinity();
    } else {
        metrics.peak_signal_to_noise_ratio = 10.0 * std::log10(1.0 / mse);
    }

    // Relative L2 error: ||target - ref||_2 / (||ref||_2 + eps)
    // eps = 1e-10 prevents division by zero if reference is completely dark/zero
    metrics.relative_l2_error = std::sqrt(sum_sq_err) / (std::sqrt(sum_ref_sq) + 1e-10);
    metrics.mean_relative_error = (rel_count > 0) ? (sum_rel_err / static_cast<double>(rel_count)) : 0.0;

    return metrics;
}

}  // namespace mirage::enhancement

