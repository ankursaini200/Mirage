#pragma once

#include <opencv2/core.hpp>

#include "mirage/enhancement/enhancement_engine.hpp"

namespace mirage::simd::avx2 {

/**
 * @brief Vectorized Brightness adjustment using 256-bit AVX2 vector operations (8 floats wide).
 *
 * Mathematical definition: I_out = clamp(I_in + delta, 0.0f, 1.0f)
 * Remaining tail elements are handled with a scalar epilogue loop.
 */
void adjust_brightness_avx2(const cv::Mat& src, cv::Mat& dst, float delta);

/**
 * @brief Vectorized Contrast adjustment using 256-bit AVX2 vector operations (8 floats wide).
 *
 * Mathematical definition: I_out = clamp(midpoint + factor * (I_in - midpoint), 0.0f, 1.0f)
 * Remaining tail elements are handled with a scalar epilogue loop.
 */
void adjust_contrast_avx2(const cv::Mat& src, cv::Mat& dst, float factor, float midpoint = 0.5f);

/**
 * @brief Vectorized Rec. 709 Saturation adjustment using 256-bit AVX2 vector operations (8 pixels wide).
 *
 * De-interleaves 8 BGR pixels (24 floats) into planar vectors, calculates luminance
 * Y = 0.114*B + 0.587*G + 0.299*R, adjusts chroma, clamps to [0, 1], and re-interleaves.
 * Remaining tail pixels are handled with a scalar epilogue loop.
 */
void adjust_saturation_avx2(const cv::Mat& src, cv::Mat& dst, float factor);

/**
 * @brief Fused Vectorized Brightness and Contrast processing in a single pass.
 *
 * Computes:
 *   x1 = clamp(I_in + delta, 0.0f, 1.0f)
 *   x2 = clamp(midpoint + factor * (x1 - midpoint), 0.0f, 1.0f)
 * inside vector registers, eliminating intermediate memory write/read cycles.
 */
void adjust_brightness_contrast_avx2(
    const cv::Mat& src,
    cv::Mat& dst,
    float delta,
    float factor,
    float midpoint = 0.5f);

/**
 * @brief Processes an entire enhancement configuration using AVX2 acceleration where supported.
 *
 * Uses fused Brightness+Contrast when both are enabled, AVX2 Saturation when enabled,
 * and scalar Gamma fallback (preserving exact transcendental precision).
 */
void process_pipeline_avx2(
    const cv::Mat& src,
    cv::Mat& dst,
    const enhancement::EnhancementConfig& config);

}  // namespace mirage::simd::avx2

