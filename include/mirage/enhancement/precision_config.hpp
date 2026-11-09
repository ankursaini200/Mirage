#pragma once

#include <string>
#include <string_view>

#include <opencv2/core.hpp>

namespace mirage::enhancement {

/**
 * @brief Numerical precision modes for image enhancement.
 */
enum class PrecisionMode {
    FP32,   ///< Uniform IEEE 754 binary32 reference implementation
    FP16,   ///< Uniform emulated IEEE 754 binary16 with intermediate rounding
    Mixed   ///< Configurable per-operation precision policy
};

[[nodiscard]] constexpr std::string_view to_string(PrecisionMode mode) noexcept {
    switch (mode) {
        case PrecisionMode::FP32: return "FP32";
        case PrecisionMode::FP16: return "FP16";
        case PrecisionMode::Mixed: return "Mixed";
    }
    return "Unknown";
}

/**
 * @brief Parses precision mode from string ("fp32", "fp16", "mixed").
 * Throws std::invalid_argument on unrecognized input.
 */
[[nodiscard]] PrecisionMode parse_precision_mode(std::string_view str);

/**
 * @brief Policy specifying the precision assigned to each operation in Mixed mode.
 */
struct MixedPrecisionPolicy {
    PrecisionMode brightness = PrecisionMode::FP32;
    PrecisionMode contrast   = PrecisionMode::FP16;
    PrecisionMode saturation = PrecisionMode::FP16;
    PrecisionMode gamma      = PrecisionMode::FP32;
};

/**
 * @brief Quantitative numerical error metrics comparing an image against the FP32 reference.
 */
struct ErrorMetrics {
    double max_absolute_error = 0.0;   ///< L-infinity error: max |I_target - I_ref|
    double mean_absolute_error = 0.0;  ///< L-1 error: mean |I_target - I_ref|
    double root_mean_square_error = 0.0; ///< RMSE: sqrt(mean((I_target - I_ref)^2))
    double peak_signal_to_noise_ratio = 0.0; ///< PSNR in dB (relative to peak value 1.0)
    double mean_relative_error = 0.0;  ///< Mean relative error for non-zero pixels

    [[nodiscard]] std::string to_string() const;
};

/**
 * @brief Computes quantitative numerical error metrics comparing a target image with an FP32 reference.
 *
 * @param reference_fp32 Ground truth reference image in CV_32FC3 format.
 * @param target Candidate image in CV_32FC3 format.
 * @throws std::invalid_argument if dimensions or types mismatch.
 */
[[nodiscard]] ErrorMetrics compute_error_metrics(const cv::Mat& reference_fp32, const cv::Mat& target);

}  // namespace mirage::enhancement

