#include "mirage/precision/adaptive_precision.hpp"

#include <chrono>
#include <cmath>

#include "mirage/image/color_converter.hpp"

namespace mirage::precision {

AdaptivePrecisionDecision AdaptivePrecisionEngine::decide_precisions(
    const analysis::SensitivityReport& report,
    double threshold) {
    if (std::isnan(threshold) || threshold < 0.0 || threshold > 1.0) {
        throw AdaptivePrecisionException("Sensitivity threshold must be a finite value in [0.0, 1.0].");
    }

    AdaptivePrecisionDecision decision{};
    decision.threshold = threshold;
    decision.total_tiles = report.tiles.size();
    decision.assignments.reserve(decision.total_tiles);

    for (const auto& t : report.tiles) {
        TilePrecisionAssignment assign{};
        assign.tile_id = t.id;
        assign.x = t.x;
        assign.y = t.y;
        assign.width = t.width;
        assign.height = t.height;
        assign.is_partial = t.is_partial;
        assign.sensitivity_score = t.sensitivity_score;

        // Default Decision Rule: score >= threshold -> FP32, score < threshold -> FP16
        if (t.sensitivity_score >= threshold) {
            assign.precision = enhancement::PrecisionMode::FP32;
            decision.fp32_count++;
        } else {
            assign.precision = enhancement::PrecisionMode::FP16;
            decision.fp16_count++;
        }

        decision.assignments.push_back(assign);
    }

    if (decision.total_tiles > 0) {
        const double total_f = static_cast<double>(decision.total_tiles);
        decision.fp32_percentage = (static_cast<double>(decision.fp32_count) * 100.0) / total_f;
        decision.fp16_percentage = (static_cast<double>(decision.fp16_count) * 100.0) / total_f;
    }

    return decision;
}

AdaptiveEnhancementResult AdaptivePrecisionEngine::enhance_adaptive(
    const cv::Mat& bgr_normalized,
    const enhancement::EnhancementConfig& config,
    int tile_width,
    int tile_height,
    double threshold,
    const analysis::SensitivityWeights& weights,
    simd::ExecutionBackend backend) {
    if (bgr_normalized.empty()) {
        throw AdaptivePrecisionException("Cannot perform adaptive enhancement on empty image.");
    }
    if (bgr_normalized.type() != CV_32FC3) {
        throw AdaptivePrecisionException("Adaptive enhancement requires normalized CV_32FC3 image.");
    }
    if (tile_width <= 0 || tile_height <= 0) {
        throw AdaptivePrecisionException("Tile dimensions must be positive integers.");
    }
    config.validate();

    // Begin strictly timed section for adaptive pipeline
    const auto start = std::chrono::high_resolution_clock::now();

    // 1. Partition image into spatial tiles
    const image::TileManager tile_mgr(bgr_normalized, tile_width, tile_height);

    // 2. Convert entire image to scientific CIELAB once
    const cv::Mat lab_img = image::ColorConverter::bgr_to_lab(bgr_normalized);

    // 3. Compute tile sensitivity scores
    analysis::SensitivityAnalyzer analyzer(weights);
    const auto sens_report = analyzer.analyze(tile_mgr, lab_img);

    // 4. Decide precision per tile based on threshold
    const auto decision = decide_precisions(sens_report, threshold);

    // 5. Pre-allocate destination matrix matching dimensions and type
    cv::Mat dst(bgr_normalized.size(), bgr_normalized.type());

    // 6. Process each tile using its assigned precision
    process_tiles(bgr_normalized, dst, tile_mgr, decision, config, backend);

    // End strictly timed section
    const auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    AdaptiveEnhancementResult result{};
    result.enhanced_image = dst;
    result.decision = decision;
    result.sensitivity_report = sens_report;
    result.adaptive_execution_time_ms = duration_ms;

    return result;
}

void AdaptivePrecisionEngine::process_tiles(
    const cv::Mat& bgr_normalized,
    cv::Mat& dst_output,
    const image::TileManager& tile_mgr,
    const AdaptivePrecisionDecision& decision,
    const enhancement::EnhancementConfig& config,
    simd::ExecutionBackend backend) {
    if (decision.assignments.size() != tile_mgr.total_tiles()) {
        throw AdaptivePrecisionException("Decision assignment count does not match tile manager tile count.");
    }

    if (dst_output.size() != bgr_normalized.size() || dst_output.type() != bgr_normalized.type()) {
        dst_output.create(bgr_normalized.size(), bgr_normalized.type());
    }

    for (const auto& t : tile_mgr.tiles()) {
        const auto& assign = decision.assignments[t.id];
        const cv::Rect rect(t.x, t.y, t.width, t.height);

        // Zero-copy ROI sub-views into input and output buffers
        const cv::Mat tile_src = bgr_normalized(rect);
        cv::Mat tile_dst = dst_output(rect);

        if (assign.precision == enhancement::PrecisionMode::FP32) {
            enhancement::EnhancementEngine::process(tile_src, tile_dst, config, backend);
        } else {
            enhancement::EnhancementEngine::process_fp16(tile_src, tile_dst, config);
        }
    }
}

}  // namespace mirage::precision

