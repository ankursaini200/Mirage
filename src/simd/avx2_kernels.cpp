#include "mirage/simd/avx2_kernels.hpp"

#include <algorithm>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__AVX2__) || defined(_M_AVX2)
#define MIRAGE_HAS_AVX2_SUPPORT 1
#include <immintrin.h>
#endif
#endif

namespace mirage::simd::avx2 {

#if defined(MIRAGE_HAS_AVX2_SUPPORT)

namespace {

inline void deinterleave_8pixels(
    __m256 v0, __m256 v1, __m256 v2,
    __m256& vb, __m256& vg, __m256& vr) noexcept {
    // 8 pixels (24 floats):
    // v0: B0 G0 R0 B1 G1 R1 B2 G2 (indices 0..7)
    // v1: R2 B3 G3 R3 B4 G4 R4 B5 (indices 0..7 in v1)
    // v2: G5 R5 B6 G6 R6 B7 G7 R7 (indices 0..7 in v2)

    // Blue channels:
    const __m256i idx_b0 = _mm256_setr_epi32(0, 3, 6, 0, 0, 0, 0, 0);
    const __m256i idx_b1 = _mm256_setr_epi32(0, 0, 0, 1, 4, 7, 0, 0);
    const __m256i idx_b2 = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 2, 5);

    __m256 b_part0 = _mm256_permutevar8x32_ps(v0, idx_b0);
    __m256 b_part1 = _mm256_permutevar8x32_ps(v1, idx_b1);
    __m256 b_part2 = _mm256_permutevar8x32_ps(v2, idx_b2);

    __m256 b_01 = _mm256_blend_ps(b_part0, b_part1, 0b00111000); // lanes 3, 4, 5
    vb = _mm256_blend_ps(b_01, b_part2, 0b11000000);             // lanes 6, 7

    // Green channels:
    const __m256i idx_g0 = _mm256_setr_epi32(1, 4, 7, 0, 0, 0, 0, 0);
    const __m256i idx_g1 = _mm256_setr_epi32(0, 0, 0, 2, 5, 0, 0, 0);
    const __m256i idx_g2 = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 3, 6);

    __m256 g_part0 = _mm256_permutevar8x32_ps(v0, idx_g0);
    __m256 g_part1 = _mm256_permutevar8x32_ps(v1, idx_g1);
    __m256 g_part2 = _mm256_permutevar8x32_ps(v2, idx_g2);

    __m256 g_01 = _mm256_blend_ps(g_part0, g_part1, 0b00011000); // lanes 3, 4
    vg = _mm256_blend_ps(g_01, g_part2, 0b11100000);             // lanes 5, 6, 7

    // Red channels:
    const __m256i idx_r0 = _mm256_setr_epi32(2, 5, 0, 0, 0, 0, 0, 0);
    const __m256i idx_r1 = _mm256_setr_epi32(0, 0, 0, 3, 6, 0, 0, 0);
    const __m256i idx_r2 = _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 4, 7);

    __m256 r_part0 = _mm256_permutevar8x32_ps(v0, idx_r0);
    __m256 r_part1 = _mm256_permutevar8x32_ps(v1, idx_r1);
    __m256 r_part2 = _mm256_permutevar8x32_ps(v2, idx_r2);

    __m256 r_01 = _mm256_blend_ps(r_part0, r_part1, 0b00011100); // lanes 2, 3, 4
    vr = _mm256_blend_ps(r_01, r_part2, 0b11100000);             // lanes 5, 6, 7
}

inline void interleave_8pixels(
    __m256 vb, __m256 vg, __m256 vr,
    __m256& v0, __m256& v1, __m256& v2) noexcept {
    // Reassemble interleaved BGR BGR BGR
    const __m256i idx_b_v0 = _mm256_setr_epi32(0, 0, 0, 1, 0, 0, 2, 0);
    const __m256i idx_g_v0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 0, 0, 2);
    const __m256i idx_r_v0 = _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 0, 0);

    __m256 b_v0 = _mm256_permutevar8x32_ps(vb, idx_b_v0);
    __m256 g_v0 = _mm256_permutevar8x32_ps(vg, idx_g_v0);
    __m256 r_v0 = _mm256_permutevar8x32_ps(vr, idx_r_v0);

    v0 = _mm256_blend_ps(b_v0, g_v0, 0b10010010);
    v0 = _mm256_blend_ps(v0,   r_v0, 0b00100100);

    const __m256i idx_b_v1 = _mm256_setr_epi32(0, 3, 0, 0, 4, 0, 0, 5);
    const __m256i idx_g_v1 = _mm256_setr_epi32(0, 0, 3, 0, 0, 4, 0, 0);
    const __m256i idx_r_v1 = _mm256_setr_epi32(2, 0, 0, 3, 0, 0, 4, 0);

    __m256 b_v1 = _mm256_permutevar8x32_ps(vb, idx_b_v1);
    __m256 g_v1 = _mm256_permutevar8x32_ps(vg, idx_g_v1);
    __m256 r_v1 = _mm256_permutevar8x32_ps(vr, idx_r_v1);

    v1 = _mm256_blend_ps(r_v1, b_v1, 0b10010010);
    v1 = _mm256_blend_ps(v1,   g_v1, 0b00100100);

    const __m256i idx_b_v2 = _mm256_setr_epi32(0, 0, 6, 0, 0, 7, 0, 0);
    const __m256i idx_g_v2 = _mm256_setr_epi32(5, 0, 0, 6, 0, 0, 7, 0);
    const __m256i idx_r_v2 = _mm256_setr_epi32(0, 5, 0, 0, 6, 0, 0, 7);

    __m256 b_v2 = _mm256_permutevar8x32_ps(vb, idx_b_v2);
    __m256 g_v2 = _mm256_permutevar8x32_ps(vg, idx_g_v2);
    __m256 r_v2 = _mm256_permutevar8x32_ps(vr, idx_r_v2);

    v2 = _mm256_blend_ps(g_v2, r_v2, 0b10010010);
    v2 = _mm256_blend_ps(v2,   b_v2, 0b00100100);
}

}  // namespace

void adjust_brightness_avx2(const cv::Mat& src, cv::Mat& dst, float delta) {
    enhancement::EnhancementEngine::validate_image(src, "adjust_brightness_avx2");
    if (!std::isfinite(delta) || delta < -1.0f || delta > 1.0f) {
        throw enhancement::EnhancementException("Brightness delta must be in [-1.0, 1.0].");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vdelta = _mm256_set1_ps(delta);

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_row = src.ptr<float>(r);
        auto* dst_row = dst.ptr<float>(r);
        const int total_floats = src.cols * 3;

        int i = 0;
        for (; i + 8 <= total_floats; i += 8) {
            const __m256 v = _mm256_loadu_ps(src_row + i);
            const __m256 added = _mm256_add_ps(v, vdelta);
            const __m256 clamped = _mm256_max_ps(vzero, _mm256_min_ps(added, vone));
            _mm256_storeu_ps(dst_row + i, clamped);
        }

        for (; i < total_floats; ++i) {
            dst_row[i] = std::clamp(src_row[i] + delta, 0.0f, 1.0f);
        }
    }
}

void adjust_contrast_avx2(const cv::Mat& src, cv::Mat& dst, float factor, float midpoint) {
    enhancement::EnhancementEngine::validate_image(src, "adjust_contrast_avx2");
    if (!std::isfinite(factor) || factor < 0.0f) {
        throw enhancement::EnhancementException("Contrast factor must be finite and >= 0.0.");
    }
    if (!std::isfinite(midpoint) || midpoint < 0.0f || midpoint > 1.0f) {
        throw enhancement::EnhancementException("Contrast midpoint must be in [0.0, 1.0].");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vmid = _mm256_set1_ps(midpoint);
    const __m256 vfactor = _mm256_set1_ps(factor);

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_row = src.ptr<float>(r);
        auto* dst_row = dst.ptr<float>(r);
        const int total_floats = src.cols * 3;

        int i = 0;
        for (; i + 8 <= total_floats; i += 8) {
            const __m256 v = _mm256_loadu_ps(src_row + i);
            const __m256 diff = _mm256_sub_ps(v, vmid);
            const __m256 scaled = _mm256_mul_ps(vfactor, diff);
            const __m256 shifted = _mm256_add_ps(vmid, scaled);
            const __m256 clamped = _mm256_max_ps(vzero, _mm256_min_ps(shifted, vone));
            _mm256_storeu_ps(dst_row + i, clamped);
        }

        for (; i < total_floats; ++i) {
            dst_row[i] = std::clamp(midpoint + factor * (src_row[i] - midpoint), 0.0f, 1.0f);
        }
    }
}

void adjust_saturation_avx2(const cv::Mat& src, cv::Mat& dst, float factor) {
    enhancement::EnhancementEngine::validate_image(src, "adjust_saturation_avx2");
    if (!std::isfinite(factor) || factor < 0.0f) {
        throw enhancement::EnhancementException("Saturation factor must be finite and >= 0.0.");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    // Rec. 709 luminance weights for BGR: B: 0.114, G: 0.587, R: 0.299
    constexpr float wb = 0.114f;
    constexpr float wg = 0.587f;
    constexpr float wr = 0.299f;

    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vwb = _mm256_set1_ps(wb);
    const __m256 vwg = _mm256_set1_ps(wg);
    const __m256 vwr = _mm256_set1_ps(wr);
    const __m256 vfactor = _mm256_set1_ps(factor);

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_row = src.ptr<float>(r);
        auto* dst_row = dst.ptr<float>(r);
        const int total_pixels = src.cols;

        int p = 0;
        // Vector loop processes 8 pixels (24 floats) per iteration
        for (; p + 8 <= total_pixels; p += 8) {
            const float* s = src_row + p * 3;
            float* d = dst_row + p * 3;

            const __m256 v0 = _mm256_loadu_ps(s);
            const __m256 v1 = _mm256_loadu_ps(s + 8);
            const __m256 v2 = _mm256_loadu_ps(s + 16);

            __m256 vb, vg, vr;
            deinterleave_8pixels(v0, v1, v2, vb, vg, vr);

            // Luminance Y = (wb*b + wg*g) + wr*r
            const __m256 y = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(vwb, vb), _mm256_mul_ps(vwg, vg)),
                _mm256_mul_ps(vwr, vr)
            );

            // C_out = clamp(y + factor * (C_in - y), 0.0f, 1.0f)
            const __m256 diff_b = _mm256_sub_ps(vb, y);
            const __m256 sat_b = _mm256_max_ps(vzero, _mm256_min_ps(_mm256_add_ps(y, _mm256_mul_ps(vfactor, diff_b)), vone));

            const __m256 diff_g = _mm256_sub_ps(vg, y);
            const __m256 sat_g = _mm256_max_ps(vzero, _mm256_min_ps(_mm256_add_ps(y, _mm256_mul_ps(vfactor, diff_g)), vone));

            const __m256 diff_r = _mm256_sub_ps(vr, y);
            const __m256 sat_r = _mm256_max_ps(vzero, _mm256_min_ps(_mm256_add_ps(y, _mm256_mul_ps(vfactor, diff_r)), vone));

            __m256 out0, out1, out2;
            interleave_8pixels(sat_b, sat_g, sat_r, out0, out1, out2);

            _mm256_storeu_ps(d, out0);
            _mm256_storeu_ps(d + 8, out1);
            _mm256_storeu_ps(d + 16, out2);
        }

        // Scalar tail loop for remaining pixels
        for (; p < total_pixels; ++p) {
            const float b = src_row[p * 3 + 0];
            const float g = src_row[p * 3 + 1];
            const float r_val = src_row[p * 3 + 2];
            const float y = wb * b + wg * g + wr * r_val;
            dst_row[p * 3 + 0] = std::clamp(y + factor * (b - y), 0.0f, 1.0f);
            dst_row[p * 3 + 1] = std::clamp(y + factor * (g - y), 0.0f, 1.0f);
            dst_row[p * 3 + 2] = std::clamp(y + factor * (r_val - y), 0.0f, 1.0f);
        }
    }
}

void adjust_brightness_contrast_avx2(
    const cv::Mat& src,
    cv::Mat& dst,
    float delta,
    float factor,
    float midpoint) {
    enhancement::EnhancementEngine::validate_image(src, "adjust_brightness_contrast_avx2");
    if (!std::isfinite(delta) || delta < -1.0f || delta > 1.0f) {
        throw enhancement::EnhancementException("Brightness delta must be in [-1.0, 1.0].");
    }
    if (!std::isfinite(factor) || factor < 0.0f) {
        throw enhancement::EnhancementException("Contrast factor must be finite and >= 0.0.");
    }
    if (!std::isfinite(midpoint) || midpoint < 0.0f || midpoint > 1.0f) {
        throw enhancement::EnhancementException("Contrast midpoint must be in [0.0, 1.0].");
    }

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(src.size(), src.type());
    }

    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vdelta = _mm256_set1_ps(delta);
    const __m256 vmid = _mm256_set1_ps(midpoint);
    const __m256 vfactor = _mm256_set1_ps(factor);

    for (int r = 0; r < src.rows; ++r) {
        const auto* src_row = src.ptr<float>(r);
        auto* dst_row = dst.ptr<float>(r);
        const int total_floats = src.cols * 3;

        int i = 0;
        for (; i + 8 <= total_floats; i += 8) {
            const __m256 v = _mm256_loadu_ps(src_row + i);
            const __m256 added = _mm256_add_ps(v, vdelta);
            const __m256 clamped1 = _mm256_max_ps(vzero, _mm256_min_ps(added, vone));
            const __m256 diff = _mm256_sub_ps(clamped1, vmid);
            const __m256 scaled = _mm256_mul_ps(vfactor, diff);
            const __m256 shifted = _mm256_add_ps(vmid, scaled);
            const __m256 clamped2 = _mm256_max_ps(vzero, _mm256_min_ps(shifted, vone));
            _mm256_storeu_ps(dst_row + i, clamped2);
        }

        for (; i < total_floats; ++i) {
            const float x1 = std::clamp(src_row[i] + delta, 0.0f, 1.0f);
            dst_row[i] = std::clamp(midpoint + factor * (x1 - midpoint), 0.0f, 1.0f);
        }
    }
}

void process_pipeline_avx2(
    const cv::Mat& src,
    cv::Mat& dst,
    const enhancement::EnhancementConfig& config) {
    config.validate();
    enhancement::EnhancementEngine::validate_image(src, "process_pipeline_avx2");

    const bool has_b = config.enable_brightness && config.brightness != 0.0f;
    const bool has_c = config.enable_contrast && config.contrast != 1.0f;
    const bool has_s = config.enable_saturation && config.saturation != 1.0f;
    const bool has_g = config.enable_gamma && config.gamma != 1.0f;

    if (!has_b && !has_c && !has_s && !has_g) {
        if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
            src.copyTo(dst);
        } else if (dst.data != src.data) {
            src.copyTo(dst);
        }
        return;
    }

    // Step 1: Brightness and Contrast (fused if both enabled)
    if (has_b && has_c) {
        adjust_brightness_contrast_avx2(src, dst, config.brightness, config.contrast, config.contrast_midpoint);
    } else if (has_b) {
        adjust_brightness_avx2(src, dst, config.brightness);
    } else if (has_c) {
        adjust_contrast_avx2(src, dst, config.contrast, config.contrast_midpoint);
    } else {
        if (dst.data != src.data) {
            src.copyTo(dst);
        }
    }

    // Step 2: Saturation (AVX2)
    if (has_s) {
        adjust_saturation_avx2(dst, dst, config.saturation);
    }

    // Step 3: Gamma (Scalar Reference Fallback to maintain exact transcendental accuracy)
    if (has_g) {
        enhancement::EnhancementEngine::adjust_gamma(dst, dst, config.gamma);
    }
}

#else

// Fallback stubs for non-x86 or non-AVX2 builds
void adjust_brightness_avx2(const cv::Mat&, cv::Mat&, float) {
    throw enhancement::EnhancementException("AVX2 kernels not compiled on this architecture.");
}

void adjust_contrast_avx2(const cv::Mat&, cv::Mat&, float, float) {
    throw enhancement::EnhancementException("AVX2 kernels not compiled on this architecture.");
}

void adjust_saturation_avx2(const cv::Mat&, cv::Mat&, float) {
    throw enhancement::EnhancementException("AVX2 kernels not compiled on this architecture.");
}

void adjust_brightness_contrast_avx2(const cv::Mat&, cv::Mat&, float, float, float) {
    throw enhancement::EnhancementException("AVX2 kernels not compiled on this architecture.");
}

void process_pipeline_avx2(const cv::Mat&, cv::Mat&, const enhancement::EnhancementConfig&) {
    throw enhancement::EnhancementException("AVX2 kernels not compiled on this architecture.");
}

#endif  // MIRAGE_HAS_AVX2_SUPPORT

}  // namespace mirage::simd::avx2

