#include "mirage/enhancement/enhancement_engine.hpp"

#include <algorithm>
#include <cmath>

namespace mirage::enhancement {

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

void EnhancementEngine::adjust_brightness(const cv::Mat& src, cv::Mat& dst, float delta) {
    validate_image(src, "adjust_brightness");
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

cv::Mat EnhancementEngine::adjust_brightness(const cv::Mat& src, float delta) {
    cv::Mat dst;
    adjust_brightness(src, dst, delta);
    return dst;
}

void EnhancementEngine::adjust_contrast(const cv::Mat& src, cv::Mat& dst, float factor, float midpoint) {
    validate_image(src, "adjust_contrast");
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

cv::Mat EnhancementEngine::adjust_contrast(const cv::Mat& src, float factor, float midpoint) {
    cv::Mat dst;
    adjust_contrast(src, dst, factor, midpoint);
    return dst;
}

void EnhancementEngine::adjust_saturation(const cv::Mat& src, cv::Mat& dst, float factor) {
    validate_image(src, "adjust_saturation");
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

cv::Mat EnhancementEngine::adjust_saturation(const cv::Mat& src, float factor) {
    cv::Mat dst;
    adjust_saturation(src, dst, factor);
    return dst;
}

void EnhancementEngine::adjust_gamma(const cv::Mat& src, cv::Mat& dst, float gamma) {
    validate_image(src, "adjust_gamma");
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

cv::Mat EnhancementEngine::adjust_gamma(const cv::Mat& src, float gamma) {
    cv::Mat dst;
    adjust_gamma(src, dst, gamma);
    return dst;
}

void EnhancementEngine::process(const cv::Mat& src, cv::Mat& dst, const EnhancementConfig& config) {
    config.validate();
    validate_image(src, "process");

    // Copy source to destination buffer once; perform operations in-place
    src.copyTo(dst);

    // Fixed pipeline sequence: Brightness -> Contrast -> Saturation -> Gamma
    if (config.enable_brightness && config.brightness != 0.0f) {
        adjust_brightness(dst, dst, config.brightness);
    }
    if (config.enable_contrast && config.contrast != 1.0f) {
        adjust_contrast(dst, dst, config.contrast, config.contrast_midpoint);
    }
    if (config.enable_saturation && config.saturation != 1.0f) {
        adjust_saturation(dst, dst, config.saturation);
    }
    if (config.enable_gamma && config.gamma != 1.0f) {
        adjust_gamma(dst, dst, config.gamma);
    }
}

cv::Mat EnhancementEngine::process(const cv::Mat& src, const EnhancementConfig& config) {
    cv::Mat dst;
    process(src, dst, config);
    return dst;
}

}  // namespace mirage::enhancement

