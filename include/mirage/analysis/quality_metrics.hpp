#pragma once

#include <cmath>
#include <string>

#include <opencv2/core.hpp>

namespace mirage::analysis {

/**
 * @brief Comprehensive numerical and perceptual image quality metrics.
 */
struct QualityMetrics {
    // Numerical metrics
    double max_absolute_error = 0.0;         ///< L-infinity error: max |I_target - I_ref| in [0.0, 1.0]
    double mean_absolute_error = 0.0;        ///< L-1 error: mean |I_target - I_ref|
    double root_mean_square_error = 0.0;     ///< RMSE: sqrt(mean((I_target - I_ref)^2))
    double peak_signal_to_noise_ratio = 0.0; ///< PSNR in dB (peak signal = 1.0)
    double relative_l2_error = 0.0;          ///< ||I_target - I_ref||_2 / (||I_ref||_2 + eps)

    // Perceptual metrics
    double ssim = 0.0;                       ///< Structural Similarity Index (SSIM) in [0.0, 1.0]
    double mean_ciede2000 = 0.0;             ///< Mean CIEDE2000 color difference (Delta E 00)
    double max_ciede2000 = 0.0;              ///< Maximum CIEDE2000 color difference (Delta E 00)

    [[nodiscard]] std::string to_string() const;
};

/**
 * @brief Computes standard CIEDE2000 color difference (Delta E 00) between two Lab colors.
 *
 * Implements the CIE / Sharma et al. (2005) formulation.
 *
 * @param L1 Lightness of first color (L* in [0.0, 100.0])
 * @param a1 a* chromaticity of first color in [-127.0, +127.0]
 * @param b1 b* chromaticity of first color in [-127.0, +127.0]
 * @param L2 Lightness of second color
 * @param a2 a* chromaticity of second color
 * @param b2 b* chromaticity of second color
 * @return Delta E 00 color difference >= 0.0
 */
[[nodiscard]] double compute_ciede2000(
    double L1, double a1, double b1,
    double L2, double a2, double b2) noexcept;

/**
 * @brief Computes the Structural Similarity Index (SSIM) between two images.
 *
 * Uses standard 11x11 Gaussian window (sigma = 1.5) across normalized float channels in [0.0, 1.0].
 *
 * @param ref Reference image (CV_32FC3, [0.0, 1.0])
 * @param target Candidate image (CV_32FC3, [0.0, 1.0])
 * @return Mean SSIM in [0.0, 1.0]
 */
[[nodiscard]] double compute_ssim(const cv::Mat& ref, const cv::Mat& target);

/**
 * @brief Evaluates complete numerical and perceptual metrics comparing target with ref.
 *
 * Automatically converts BGR float images to CIELAB for exact CIEDE2000 evaluation.
 *
 * @param ref_bgr Ground truth reference image (CV_32FC3, [0.0, 1.0])
 * @param target_bgr Candidate image (CV_32FC3, [0.0, 1.0])
 * @return Complete QualityMetrics struct
 */
[[nodiscard]] QualityMetrics evaluate_quality(const cv::Mat& ref_bgr, const cv::Mat& target_bgr);

}  // namespace mirage::analysis

