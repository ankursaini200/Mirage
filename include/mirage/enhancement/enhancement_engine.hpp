#pragma once

#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

#include "mirage/enhancement/precision_config.hpp"
#include "mirage/simd/simd_backend.hpp"

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
 * @brief Configuration parameters for image enhancement.
 */
struct EnhancementConfig {
    bool enable_brightness = false;
    float brightness = 0.0f;        ///< Additive delta in [-1.0f, 1.0f], default 0.0f

    bool enable_contrast = false;
    float contrast = 1.0f;          ///< Multiplicative factor >= 0.0f, default 1.0f
    float contrast_midpoint = 0.5f; ///< Contrast pivot in [0.0f, 1.0f], default 0.5f

    bool enable_saturation = false;
    float saturation = 1.0f;        ///< Chroma scale factor >= 0.0f, default 1.0f (0.0 = grayscale)

    bool enable_gamma = false;
    float gamma = 1.0f;             ///< Power-law exponent > 0.0f, default 1.0f

    /**
     * @brief Validates parameter ranges, throwing EnhancementException on error.
     */
    void validate() const;
};

/**
 * @brief Color image enhancement engine supporting FP32 (Scalar & AVX2), emulated FP16, and Mixed precision.
 *
 * Provides:
 * - Dynamic SIMD runtime dispatch (Auto, Scalar, AVX2).
 * - Portable scalar reference implementation (fallback).
 * - AVX2 vectorized implementations for Brightness, Contrast, Saturation, and Combined B+C.
 * - FP16 Emulation with explicit IEEE 754 binary16 rounding at arithmetic boundaries.
 * - Mixed Precision execution where operations are individually assigned to FP32 or FP16.
 *
 * Mathematical Definitions:
 * 1. Brightness:  I_out = clamp(I_in + delta, 0.0f, 1.0f)
 * 2. Contrast:    I_out = clamp(midpoint + factor * (I_in - midpoint), 0.0f, 1.0f)
 * 3. Saturation:  Y = 0.114*B + 0.587*G + 0.299*R; C_out = clamp(Y + factor * (C_in - Y), 0.0f, 1.0f)
 * 4. Gamma:       I_out = clamp(I_in ^ (1.0f / gamma), 0.0f, 1.0f)
 *
 * Processing Sequence: Brightness -> Contrast -> Saturation -> Gamma
 */
class EnhancementEngine {
public:
    // --- Unified FP32 API with Dynamic Backend Dispatch ---
    [[nodiscard]] static cv::Mat adjust_brightness(
        const cv::Mat& src,
        float delta,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);
    static void adjust_brightness(
        const cv::Mat& src,
        cv::Mat& dst,
        float delta,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);

    [[nodiscard]] static cv::Mat adjust_contrast(
        const cv::Mat& src,
        float factor,
        float midpoint = 0.5f,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);
    static void adjust_contrast(
        const cv::Mat& src,
        cv::Mat& dst,
        float factor,
        float midpoint = 0.5f,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);

    [[nodiscard]] static cv::Mat adjust_saturation(
        const cv::Mat& src,
        float factor,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);
    static void adjust_saturation(
        const cv::Mat& src,
        cv::Mat& dst,
        float factor,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);

    [[nodiscard]] static cv::Mat adjust_gamma(
        const cv::Mat& src,
        float gamma,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);
    static void adjust_gamma(
        const cv::Mat& src,
        cv::Mat& dst,
        float gamma,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);

    [[nodiscard]] static cv::Mat adjust_brightness_contrast(
        const cv::Mat& src,
        float delta,
        float factor,
        float midpoint = 0.5f,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);
    static void adjust_brightness_contrast(
        const cv::Mat& src,
        cv::Mat& dst,
        float delta,
        float factor,
        float midpoint = 0.5f,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);

    [[nodiscard]] static cv::Mat process(
        const cv::Mat& src,
        const EnhancementConfig& config,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);
    static void process(
        const cv::Mat& src,
        cv::Mat& dst,
        const EnhancementConfig& config,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);

    // --- Explicit Portable Scalar Reference Kernels ---
    static void adjust_brightness_scalar(const cv::Mat& src, cv::Mat& dst, float delta);
    static void adjust_contrast_scalar(const cv::Mat& src, cv::Mat& dst, float factor, float midpoint = 0.5f);
    static void adjust_saturation_scalar(const cv::Mat& src, cv::Mat& dst, float factor);
    static void adjust_gamma_scalar(const cv::Mat& src, cv::Mat& dst, float gamma);
    static void process_scalar(const cv::Mat& src, cv::Mat& dst, const EnhancementConfig& config);

    // --- FP16 Emulated Kernels (Explicit IEEE 754 binary16 rounding) ---
    [[nodiscard]] static cv::Mat adjust_brightness_fp16(const cv::Mat& src, float delta);
    static void adjust_brightness_fp16(const cv::Mat& src, cv::Mat& dst, float delta);

    [[nodiscard]] static cv::Mat adjust_contrast_fp16(const cv::Mat& src, float factor, float midpoint = 0.5f);
    static void adjust_contrast_fp16(const cv::Mat& src, cv::Mat& dst, float factor, float midpoint = 0.5f);

    [[nodiscard]] static cv::Mat adjust_saturation_fp16(const cv::Mat& src, float factor);
    static void adjust_saturation_fp16(const cv::Mat& src, cv::Mat& dst, float factor);

    [[nodiscard]] static cv::Mat adjust_gamma_fp16(const cv::Mat& src, float gamma);
    static void adjust_gamma_fp16(const cv::Mat& src, cv::Mat& dst, float gamma);

    [[nodiscard]] static cv::Mat process_fp16(const cv::Mat& src, const EnhancementConfig& config);
    static void process_fp16(const cv::Mat& src, cv::Mat& dst, const EnhancementConfig& config);

    // --- Configurable Multi-Precision Pipeline ---
    [[nodiscard]] static cv::Mat process_with_precision(
        const cv::Mat& src,
        const EnhancementConfig& config,
        PrecisionMode mode = PrecisionMode::FP32,
        const MixedPrecisionPolicy& policy = MixedPrecisionPolicy{},
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);

    static void process_with_precision(
        const cv::Mat& src,
        cv::Mat& dst,
        const EnhancementConfig& config,
        PrecisionMode mode = PrecisionMode::FP32,
        const MixedPrecisionPolicy& policy = MixedPrecisionPolicy{},
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);

    static void validate_image(const cv::Mat& image, const std::string& operation_name);
};

}  // namespace mirage::enhancement
