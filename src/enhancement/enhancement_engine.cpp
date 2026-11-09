#include "mirage/enhancement/enhancement_engine.hpp"

#include <algorithm>
#include <cmath>

#include "mirage/core/half.hpp"
#include "mirage/cpu/cpu_features.hpp"
#include "mirage/simd/avx2_kernels.hpp"

namespace mirage::enhancement {

using mirage::core::round_to_fp16;

namespace {

simd::ExecutionBackend check_and_resolve_backend(simd::ExecutionBackend backend) {
    if (backend == simd::ExecutionBackend::AVX2) {
        if (!simd::is_backend_supported(simd::ExecutionBackend::AVX2)) {
            throw EnhancementException("AVX2 backend requested but required CPU features (AVX2/FMA) are not supported on this host CPU.");
        }
        return simd::ExecutionBackend::AVX2;
    }
    return simd::resolve_backend(backend);
}

}  // namespace

void EnhancementConfig::validate() const {
    if (enable_brightness) {
        if (!std::isfinite(brightness) || brightness < -1.0f || brightness > 1.0f) {
            throw EnhancementException("Brightness delta must be a finite value in [-1.0, 1.0].");
        }
    }
    if (enable_contrast) {
        if (!std::isfinite(contrast) || contrast < 0.0f) {
            throw EnhancementException("Contrast factor must be finite and >= 0.0.");
        }
        if (!std::isfinite(contrast_midpoint) || contrast_midpoint < 0.0f || contrast_midpoint > 1.0f) {
            throw EnhancementException("Contrast midpoint must be in [0.0, 1.0].");
        }
    }
    if (enable_saturation) {
        if (!std::isfinite(saturation) || saturation < 0.0f) {
            throw EnhancementException("Saturation factor must be finite and >= 0.0.");
        }
    }
    if (enable_gamma) {
        if (!std::isfinite(gamma) || gamma <= 0.0f) {
            throw EnhancementException("Gamma must be a finite value strictly greater than 0.0.");
        }
    }
}

void EnhancementEngine::validate_image(const cv::Mat& image, const std::string& operation_name) {
    if (image.empty()) {
        throw EnhancementException(operation_name + ": Input image matrix is empty.");
    }
    if (image.type() != CV_32FC3) {
        throw EnhancementException(operation_name + ": Expected CV_32FC3 normalized floating-point image.");
    }
    if (image.rows <= 0 || image.cols <= 0) {
        throw EnhancementException(operation_name + ": Image dimensions must be strictly positive.");
    }
}

// ============================================================================
// Portable Scalar Reference Kernels
// ============================================================================

void EnhancementEngine::adjust_brightness_scalar(const cv::Mat& src, cv::Mat& dst, float delta) {
    validate_image(src, "adjust_brightness_scalar");
    if (!std::isfinite(delta) || delta < -1.0f || delta > 1.0f) {
        throw EnhancementException("Brightness delta must be in [-1.0, 1.0].");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_ptr = src.ptr<cv::Vec3f>(r);
        auto* dst_ptr = dst.ptr<cv::Vec3f>(r);
        for (int c = 0; c < src.cols; ++c) {
            const cv::Vec3f pix = src_ptr[c];
            dst_ptr[c] = cv::Vec3f(
                std::clamp(pix[0] + delta, 0.0f, 1.0f),
                std::clamp(pix[1] + delta, 0.0f, 1.0f),
                std::clamp(pix[2] + delta, 0.0f, 1.0f)
            );
        }
    }
}

void EnhancementEngine::adjust_contrast_scalar(const cv::Mat& src, cv::Mat& dst, float factor, float midpoint) {
    validate_image(src, "adjust_contrast_scalar");
    if (!std::isfinite(factor) || factor < 0.0f) {
        throw EnhancementException("Contrast factor must be finite and >= 0.0.");
    }
    if (!std::isfinite(midpoint) || midpoint < 0.0f || midpoint > 1.0f) {
        throw EnhancementException("Contrast midpoint must be in [0.0, 1.0].");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_ptr = src.ptr<cv::Vec3f>(r);
        auto* dst_ptr = dst.ptr<cv::Vec3f>(r);
        for (int c = 0; c < src.cols; ++c) {
            const cv::Vec3f pix = src_ptr[c];
            dst_ptr[c] = cv::Vec3f(
                std::clamp(midpoint + factor * (pix[0] - midpoint), 0.0f, 1.0f),
                std::clamp(midpoint + factor * (pix[1] - midpoint), 0.0f, 1.0f),
                std::clamp(midpoint + factor * (pix[2] - midpoint), 0.0f, 1.0f)
            );
        }
    }
}

void EnhancementEngine::adjust_saturation_scalar(const cv::Mat& src, cv::Mat& dst, float factor) {
    validate_image(src, "adjust_saturation_scalar");
    if (!std::isfinite(factor) || factor < 0.0f) {
        throw EnhancementException("Saturation factor must be finite and >= 0.0.");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    // Rec. 709 luminance weights for BGR format: B: 0.114, G: 0.587, R: 0.299
    constexpr float wb = 0.114f;
    constexpr float wg = 0.587f;
    constexpr float wr = 0.299f;

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_ptr = src.ptr<cv::Vec3f>(r);
        auto* dst_ptr = dst.ptr<cv::Vec3f>(r);
        for (int c = 0; c < src.cols; ++c) {
            const cv::Vec3f pix = src_ptr[c];
            const float y = wb * pix[0] + wg * pix[1] + wr * pix[2];
            dst_ptr[c] = cv::Vec3f(
                std::clamp(y + factor * (pix[0] - y), 0.0f, 1.0f),
                std::clamp(y + factor * (pix[1] - y), 0.0f, 1.0f),
                std::clamp(y + factor * (pix[2] - y), 0.0f, 1.0f)
            );
        }
    }
}

void EnhancementEngine::adjust_gamma_scalar(const cv::Mat& src, cv::Mat& dst, float gamma) {
    validate_image(src, "adjust_gamma_scalar");
    if (!std::isfinite(gamma) || gamma <= 0.0f) {
        throw EnhancementException("Gamma must be strictly positive (> 0.0).");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    const float inv_gamma = 1.0f / gamma;

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_ptr = src.ptr<cv::Vec3f>(r);
        auto* dst_ptr = dst.ptr<cv::Vec3f>(r);
        for (int c = 0; c < src.cols; ++c) {
            const cv::Vec3f pix = src_ptr[c];
            dst_ptr[c] = cv::Vec3f(
                std::clamp(std::pow(std::max(0.0f, pix[0]), inv_gamma), 0.0f, 1.0f),
                std::clamp(std::pow(std::max(0.0f, pix[1]), inv_gamma), 0.0f, 1.0f),
                std::clamp(std::pow(std::max(0.0f, pix[2]), inv_gamma), 0.0f, 1.0f)
            );
        }
    }
}

void EnhancementEngine::process_scalar(const cv::Mat& src, cv::Mat& dst, const EnhancementConfig& config) {
    config.validate();
    validate_image(src, "process_scalar");

    src.copyTo(dst);

    if (config.enable_brightness && config.brightness != 0.0f) {
        adjust_brightness_scalar(dst, dst, config.brightness);
    }
    if (config.enable_contrast && config.contrast != 1.0f) {
        adjust_contrast_scalar(dst, dst, config.contrast, config.contrast_midpoint);
    }
    if (config.enable_saturation && config.saturation != 1.0f) {
        adjust_saturation_scalar(dst, dst, config.saturation);
    }
    if (config.enable_gamma && config.gamma != 1.0f) {
        adjust_gamma_scalar(dst, dst, config.gamma);
    }
}

// ============================================================================
// Unified API with Dynamic Dispatch
// ============================================================================

void EnhancementEngine::adjust_brightness(const cv::Mat& src, cv::Mat& dst, float delta, simd::ExecutionBackend backend) {
    const auto effective = check_and_resolve_backend(backend);
    if (effective == simd::ExecutionBackend::AVX2) {
        simd::avx2::adjust_brightness_avx2(src, dst, delta);
        return;
    }
    adjust_brightness_scalar(src, dst, delta);
}

cv::Mat EnhancementEngine::adjust_brightness(const cv::Mat& src, float delta, simd::ExecutionBackend backend) {
    cv::Mat dst;
    adjust_brightness(src, dst, delta, backend);
    return dst;
}

void EnhancementEngine::adjust_contrast(const cv::Mat& src, cv::Mat& dst, float factor, float midpoint, simd::ExecutionBackend backend) {
    const auto effective = check_and_resolve_backend(backend);
    if (effective == simd::ExecutionBackend::AVX2) {
        simd::avx2::adjust_contrast_avx2(src, dst, factor, midpoint);
        return;
    }
    adjust_contrast_scalar(src, dst, factor, midpoint);
}

cv::Mat EnhancementEngine::adjust_contrast(const cv::Mat& src, float factor, float midpoint, simd::ExecutionBackend backend) {
    cv::Mat dst;
    adjust_contrast(src, dst, factor, midpoint, backend);
    return dst;
}

void EnhancementEngine::adjust_saturation(const cv::Mat& src, cv::Mat& dst, float factor, simd::ExecutionBackend backend) {
    const auto effective = check_and_resolve_backend(backend);
    if (effective == simd::ExecutionBackend::AVX2) {
        simd::avx2::adjust_saturation_avx2(src, dst, factor);
        return;
    }
    adjust_saturation_scalar(src, dst, factor);
}

cv::Mat EnhancementEngine::adjust_saturation(const cv::Mat& src, float factor, simd::ExecutionBackend backend) {
    cv::Mat dst;
    adjust_saturation(src, dst, factor, backend);
    return dst;
}

void EnhancementEngine::adjust_gamma(const cv::Mat& src, cv::Mat& dst, float gamma, simd::ExecutionBackend /*backend*/) {
    // Gamma uses nonlinear power operation; retained scalar to prevent loss of IEEE-754 precision
    adjust_gamma_scalar(src, dst, gamma);
}

cv::Mat EnhancementEngine::adjust_gamma(const cv::Mat& src, float gamma, simd::ExecutionBackend backend) {
    cv::Mat dst;
    adjust_gamma(src, dst, gamma, backend);
    return dst;
}

void EnhancementEngine::adjust_brightness_contrast(
    const cv::Mat& src,
    cv::Mat& dst,
    float delta,
    float factor,
    float midpoint,
    simd::ExecutionBackend backend) {
    const auto effective = check_and_resolve_backend(backend);
    if (effective == simd::ExecutionBackend::AVX2) {
        simd::avx2::adjust_brightness_contrast_avx2(src, dst, delta, factor, midpoint);
        return;
    }

    validate_image(src, "adjust_brightness_contrast");
    if (!std::isfinite(delta) || delta < -1.0f || delta > 1.0f) {
        throw EnhancementException("Brightness delta must be in [-1.0, 1.0].");
    }
    if (!std::isfinite(factor) || factor < 0.0f) {
        throw EnhancementException("Contrast factor must be finite and >= 0.0.");
    }
    if (!std::isfinite(midpoint) || midpoint < 0.0f || midpoint > 1.0f) {
        throw EnhancementException("Contrast midpoint must be in [0.0, 1.0].");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_ptr = src.ptr<cv::Vec3f>(r);
        auto* dst_ptr = dst.ptr<cv::Vec3f>(r);
        for (int c = 0; c < src.cols; ++c) {
            const cv::Vec3f pix = src_ptr[c];
            cv::Vec3f out;
            for (int ch = 0; ch < 3; ++ch) {
                const float x1 = std::clamp(pix[ch] + delta, 0.0f, 1.0f);
                out[ch] = std::clamp(midpoint + factor * (x1 - midpoint), 0.0f, 1.0f);
            }
            dst_ptr[c] = out;
        }
    }
}

cv::Mat EnhancementEngine::adjust_brightness_contrast(
    const cv::Mat& src,
    float delta,
    float factor,
    float midpoint,
    simd::ExecutionBackend backend) {
    cv::Mat dst;
    adjust_brightness_contrast(src, dst, delta, factor, midpoint, backend);
    return dst;
}

void EnhancementEngine::process(
    const cv::Mat& src,
    cv::Mat& dst,
    const EnhancementConfig& config,
    simd::ExecutionBackend backend) {
    const auto effective = check_and_resolve_backend(backend);
    if (effective == simd::ExecutionBackend::AVX2) {
        simd::avx2::process_pipeline_avx2(src, dst, config);
        return;
    }
    process_scalar(src, dst, config);
}

cv::Mat EnhancementEngine::process(
    const cv::Mat& src,
    const EnhancementConfig& config,
    simd::ExecutionBackend backend) {
    cv::Mat dst;
    process(src, dst, config, backend);
    return dst;
}

// ============================================================================
// FP16 Emulated Implementation (Explicit IEEE 754 binary16 rounding)
// ============================================================================

void EnhancementEngine::adjust_brightness_fp16(const cv::Mat& src, cv::Mat& dst, float delta) {
    validate_image(src, "adjust_brightness_fp16");
    if (!std::isfinite(delta) || delta < -1.0f || delta > 1.0f) {
        throw EnhancementException("Brightness delta must be in [-1.0, 1.0].");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    const float delta_fp16 = round_to_fp16(delta);

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_ptr = src.ptr<cv::Vec3f>(r);
        auto* dst_ptr = dst.ptr<cv::Vec3f>(r);
        for (int c = 0; c < src.cols; ++c) {
            const cv::Vec3f pix = src_ptr[c];
            dst_ptr[c] = cv::Vec3f(
                round_to_fp16(std::clamp(round_to_fp16(round_to_fp16(pix[0]) + delta_fp16), 0.0f, 1.0f)),
                round_to_fp16(std::clamp(round_to_fp16(round_to_fp16(pix[1]) + delta_fp16), 0.0f, 1.0f)),
                round_to_fp16(std::clamp(round_to_fp16(round_to_fp16(pix[2]) + delta_fp16), 0.0f, 1.0f))
            );
        }
    }
}

cv::Mat EnhancementEngine::adjust_brightness_fp16(const cv::Mat& src, float delta) {
    cv::Mat dst;
    adjust_brightness_fp16(src, dst, delta);
    return dst;
}

void EnhancementEngine::adjust_contrast_fp16(const cv::Mat& src, cv::Mat& dst, float factor, float midpoint) {
    validate_image(src, "adjust_contrast_fp16");
    if (!std::isfinite(factor) || factor < 0.0f) {
        throw EnhancementException("Contrast factor must be finite and >= 0.0.");
    }
    if (!std::isfinite(midpoint) || midpoint < 0.0f || midpoint > 1.0f) {
        throw EnhancementException("Contrast midpoint must be in [0.0, 1.0].");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    const float factor_fp16 = round_to_fp16(factor);
    const float mid_fp16 = round_to_fp16(midpoint);

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_ptr = src.ptr<cv::Vec3f>(r);
        auto* dst_ptr = dst.ptr<cv::Vec3f>(r);
        for (int c = 0; c < src.cols; ++c) {
            const cv::Vec3f pix = src_ptr[c];
            cv::Vec3f out;
            for (int ch = 0; ch < 3; ++ch) {
                const float in_fp16 = round_to_fp16(pix[ch]);
                const float diff = round_to_fp16(in_fp16 - mid_fp16);
                const float scaled = round_to_fp16(factor_fp16 * diff);
                const float shifted = round_to_fp16(mid_fp16 + scaled);
                out[ch] = round_to_fp16(std::clamp(shifted, 0.0f, 1.0f));
            }
            dst_ptr[c] = out;
        }
    }
}

cv::Mat EnhancementEngine::adjust_contrast_fp16(const cv::Mat& src, float factor, float midpoint) {
    cv::Mat dst;
    adjust_contrast_fp16(src, dst, factor, midpoint);
    return dst;
}

void EnhancementEngine::adjust_saturation_fp16(const cv::Mat& src, cv::Mat& dst, float factor) {
    validate_image(src, "adjust_saturation_fp16");
    if (!std::isfinite(factor) || factor < 0.0f) {
        throw EnhancementException("Saturation factor must be finite and >= 0.0.");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    const float factor_fp16 = round_to_fp16(factor);
    const float wb_fp16 = round_to_fp16(0.114f);
    const float wg_fp16 = round_to_fp16(0.587f);
    const float wr_fp16 = round_to_fp16(0.299f);

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_ptr = src.ptr<cv::Vec3f>(r);
        auto* dst_ptr = dst.ptr<cv::Vec3f>(r);
        for (int c = 0; c < src.cols; ++c) {
            const cv::Vec3f pix = src_ptr[c];
            const float b = round_to_fp16(pix[0]);
            const float g = round_to_fp16(pix[1]);
            const float r_val = round_to_fp16(pix[2]);

            // Luminance with binary16 rounding at every step
            const float y_b = round_to_fp16(wb_fp16 * b);
            const float y_g = round_to_fp16(wg_fp16 * g);
            const float y_r = round_to_fp16(wr_fp16 * r_val);
            const float y_bg = round_to_fp16(y_b + y_g);
            const float y = round_to_fp16(y_bg + y_r);

            cv::Vec3f out;
            const float chs[3] = {b, g, r_val};
            for (int ch = 0; ch < 3; ++ch) {
                const float diff = round_to_fp16(chs[ch] - y);
                const float scaled = round_to_fp16(factor_fp16 * diff);
                const float sum = round_to_fp16(y + scaled);
                out[ch] = round_to_fp16(std::clamp(sum, 0.0f, 1.0f));
            }
            dst_ptr[c] = out;
        }
    }
}

cv::Mat EnhancementEngine::adjust_saturation_fp16(const cv::Mat& src, float factor) {
    cv::Mat dst;
    adjust_saturation_fp16(src, dst, factor);
    return dst;
}

void EnhancementEngine::adjust_gamma_fp16(const cv::Mat& src, cv::Mat& dst, float gamma) {
    validate_image(src, "adjust_gamma_fp16");
    if (!std::isfinite(gamma) || gamma <= 0.0f) {
        throw EnhancementException("Gamma must be strictly positive (> 0.0).");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    const float gamma_fp16 = round_to_fp16(gamma);
    const float inv_gamma_fp16 = round_to_fp16(1.0f / gamma_fp16);

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_ptr = src.ptr<cv::Vec3f>(r);
        auto* dst_ptr = dst.ptr<cv::Vec3f>(r);
        for (int c = 0; c < src.cols; ++c) {
            const cv::Vec3f pix = src_ptr[c];
            cv::Vec3f out;
            for (int ch = 0; ch < 3; ++ch) {
                const float in_fp16 = round_to_fp16(pix[ch]);
                const float safe_in = std::max(0.0f, in_fp16);
                const float powered = round_to_fp16(std::pow(safe_in, inv_gamma_fp16));
                out[ch] = round_to_fp16(std::clamp(powered, 0.0f, 1.0f));
            }
            dst_ptr[c] = out;
        }
    }
}

cv::Mat EnhancementEngine::adjust_gamma_fp16(const cv::Mat& src, float gamma) {
    cv::Mat dst;
    adjust_gamma_fp16(src, dst, gamma);
    return dst;
}

void EnhancementEngine::process_fp16(const cv::Mat& src, cv::Mat& dst, const EnhancementConfig& config) {
    config.validate();
    validate_image(src, "process_fp16");

    src.copyTo(dst);

    if (config.enable_brightness && config.brightness != 0.0f) {
        adjust_brightness_fp16(dst, dst, config.brightness);
    }
    if (config.enable_contrast && config.contrast != 1.0f) {
        adjust_contrast_fp16(dst, dst, config.contrast, config.contrast_midpoint);
    }
    if (config.enable_saturation && config.saturation != 1.0f) {
        adjust_saturation_fp16(dst, dst, config.saturation);
    }
    if (config.enable_gamma && config.gamma != 1.0f) {
        adjust_gamma_fp16(dst, dst, config.gamma);
    }
}

cv::Mat EnhancementEngine::process_fp16(const cv::Mat& src, const EnhancementConfig& config) {
    cv::Mat dst;
    process_fp16(src, dst, config);
    return dst;
}

// ============================================================================
// Multi-Precision Configurable Pipeline
// ============================================================================

void EnhancementEngine::process_with_precision(
    const cv::Mat& src,
    cv::Mat& dst,
    const EnhancementConfig& config,
    PrecisionMode mode,
    const MixedPrecisionPolicy& policy,
    simd::ExecutionBackend backend) {
    config.validate();
    validate_image(src, "process_with_precision");

    if (mode == PrecisionMode::FP32) {
        process(src, dst, config, backend);
        return;
    }

    if (mode == PrecisionMode::FP16) {
        process_fp16(src, dst, config);
        return;
    }

    // Mixed Precision Policy execution
    src.copyTo(dst);

    if (config.enable_brightness && config.brightness != 0.0f) {
        if (policy.brightness == PrecisionMode::FP16) {
            adjust_brightness_fp16(dst, dst, config.brightness);
        } else {
            adjust_brightness(dst, dst, config.brightness, backend);
        }
    }

    if (config.enable_contrast && config.contrast != 1.0f) {
        if (policy.contrast == PrecisionMode::FP16) {
            adjust_contrast_fp16(dst, dst, config.contrast, config.contrast_midpoint);
        } else {
            adjust_contrast(dst, dst, config.contrast, config.contrast_midpoint, backend);
        }
    }

    if (config.enable_saturation && config.saturation != 1.0f) {
        if (policy.saturation == PrecisionMode::FP16) {
            adjust_saturation_fp16(dst, dst, config.saturation);
        } else {
            adjust_saturation(dst, dst, config.saturation, backend);
        }
    }

    if (config.enable_gamma && config.gamma != 1.0f) {
        if (policy.gamma == PrecisionMode::FP16) {
            adjust_gamma_fp16(dst, dst, config.gamma);
        } else {
            adjust_gamma(dst, dst, config.gamma, backend);
        }
    }
}

cv::Mat EnhancementEngine::process_with_precision(
    const cv::Mat& src,
    const EnhancementConfig& config,
    PrecisionMode mode,
    const MixedPrecisionPolicy& policy,
    simd::ExecutionBackend backend) {
    cv::Mat dst;
    process_with_precision(src, dst, config, mode, policy, backend);
    return dst;
}

}  // namespace mirage::enhancement
