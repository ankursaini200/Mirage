#pragma once

#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

namespace mirage::enhancement {

/**
 * @brief Exception thrown when enhancement operations encounter invalid inputs or parameters.
 */
class EnhancementException : public std::runtime_error {
public:
    explicit EnhancementException(const std::string& message)
        : std::runtime_error("EnhancementException: " + message) {}
};

/**
 * @brief Configuration parameters for uniform FP32 enhancement pipeline.
 */
struct EnhancementConfig {
    bool enable_brightness = false;
    float brightness = 0.0f;       ///< Additive delta in [-1.0f, 1.0f], default 0.0f

    bool enable_contrast = false;
    float contrast = 1.0f;         ///< Multiplicative factor >= 0.0f, default 1.0f
    float contrast_midpoint = 0.5f; ///< Contrast pivot in [0.0f, 1.0f], default 0.5f

    bool enable_saturation = false;
    float saturation = 1.0f;       ///< Chroma scale factor >= 0.0f, default 1.0f (0.0 = grayscale)

    bool enable_gamma = false;
    float gamma = 1.0f;            ///< Power-law exponent > 0.0f, default 1.0f

    /**
     * @brief Validates parameter ranges, throwing EnhancementException on error.
     */
    void validate() const;
};

/**
 * @brief Baseline uniform FP32 color image enhancement engine.
 *
 * Serves as the numerical reference implementation for future adaptive precision
 * and SIMD kernels. Operates exclusively on normalized CV_32FC3 BGR image data
 * with output clamped strictly to [0.0f, 1.0f].
 *
 * Mathematical Definitions:
 * 1. Brightness:  I_out = clamp(I_in + delta, 0.0f, 1.0f)
 * 2. Contrast:    I_out = clamp(midpoint + factor * (I_in - midpoint), 0.0f, 1.0f)
 * 3. Saturation:  Y = 0.114*B + 0.587*G + 0.299*R; C_out = clamp(Y + factor * (C_in - Y), 0.0f, 1.0f)
 * 4. Gamma:       I_out = clamp(I_in ^ (1.0f / gamma), 0.0f, 1.0f)
 *
 * Processing Order in Combined Pipeline:
 * Brightness -> Contrast -> Saturation -> Gamma
 */
class EnhancementEngine {
public:
    /**
     * @brief Applies additive brightness adjustment.
     * @param src Input CV_32FC3 image.
     * @param delta Additive offset in [-1.0f, 1.0f].
     */
    [[nodiscard]] static cv::Mat adjust_brightness(const cv::Mat& src, float delta);
    static void adjust_brightness(const cv::Mat& src, cv::Mat& dst, float delta);

    /**
     * @brief Applies contrast adjustment around a pivot midpoint.
     * @param src Input CV_32FC3 image.
     * @param factor Contrast multiplier >= 0.0f (1.0 = identity).
     * @param midpoint Neutral luminance anchor point (default 0.5f).
     */
    [[nodiscard]] static cv::Mat adjust_contrast(const cv::Mat& src, float factor, float midpoint = 0.5f);
    static void adjust_contrast(const cv::Mat& src, cv::Mat& dst, float factor, float midpoint = 0.5f);

    /**
     * @brief Applies saturation adjustment while preserving perceptual luminance and grayscale values.
     * @param src Input CV_32FC3 image.
     * @param factor Saturation multiplier >= 0.0f (0.0 = pure grayscale, 1.0 = identity).
     */
    [[nodiscard]] static cv::Mat adjust_saturation(const cv::Mat& src, float factor);
    static void adjust_saturation(const cv::Mat& src, cv::Mat& dst, float factor);

    /**
     * @brief Applies power-law gamma transformation (I_out = I_in ^ (1 / gamma)).
     * @param src Input CV_32FC3 image.
     * @param gamma Exponent strictly greater than 0.0f (1.0 = identity).
     */
    [[nodiscard]] static cv::Mat adjust_gamma(const cv::Mat& src, float gamma);
    static void adjust_gamma(const cv::Mat& src, cv::Mat& dst, float gamma);

    /**
     * @brief Executes the complete enhancement pipeline with enabled operations.
     */
    [[nodiscard]] static cv::Mat process(const cv::Mat& src, const EnhancementConfig& config);
    static void process(const cv::Mat& src, cv::Mat& dst, const EnhancementConfig& config);

    /**
     * @brief Validates that the input image is non-empty, has 3 channels, and is CV_32FC3.
     */
    static void validate_image(const cv::Mat& image, const std::string& operation_name);
};

}  // namespace mirage::enhancement

