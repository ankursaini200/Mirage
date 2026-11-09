#pragma once

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "mirage/analysis/sensitivity_analyzer.hpp"
#include "mirage/enhancement/enhancement_engine.hpp"
#include "mirage/enhancement/precision_config.hpp"
#include "mirage/image/tile_manager.hpp"
#include "mirage/simd/simd_backend.hpp"

namespace mirage::precision {

/**
 * @brief Exception thrown when adaptive precision configuration or processing encounters an error.
 */
class AdaptivePrecisionException : public std::runtime_error {
public:
    explicit AdaptivePrecisionException(const std::string& message)
        : std::runtime_error("AdaptivePrecisionException: " + message) {}
};

/**
 * @brief Default sensitivity threshold for precision decision:
 * score >= threshold -> FP32, score < threshold -> FP16.
 */
inline constexpr double DEFAULT_SENSITIVITY_THRESHOLD = 0.04;

/**
 * @brief Precision assignment record for an individual image tile.
 */
struct TilePrecisionAssignment {
    size_t tile_id = 0;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    bool is_partial = false;
    double sensitivity_score = 0.0;
    enhancement::PrecisionMode precision = enhancement::PrecisionMode::FP16;
};

/**
 * @brief Complete decision record and summary distribution across all image tiles.
 */
struct AdaptivePrecisionDecision {
    double threshold = DEFAULT_SENSITIVITY_THRESHOLD;
    size_t total_tiles = 0;
    size_t fp32_count = 0;
    size_t fp16_count = 0;
    double fp32_percentage = 0.0;   ///< In range [0.0, 100.0]
    double fp16_percentage = 0.0;   ///< In range [0.0, 100.0]
    std::vector<TilePrecisionAssignment> assignments{};
};

/**
 * @brief Result of an adaptive enhancement pipeline execution.
 */
struct AdaptiveEnhancementResult {
    cv::Mat enhanced_image;
    AdaptivePrecisionDecision decision;
    analysis::SensitivityReport sensitivity_report;
    double adaptive_execution_time_ms = 0.0; ///< Timed exclusively for adaptive processing
};

/**
 * @brief Tile-based adaptive precision decision and enhancement engine.
 */
class AdaptivePrecisionEngine {
public:
    /**
     * @brief Assigns numerical precision (FP32 or FP16) to each tile based on sensitivity score.
     *
     * Decision Rule:
     *   score >= threshold  -->  FP32
     *   score <  threshold  -->  FP16
     *
     * @param report SensitivityReport from SensitivityAnalyzer.
     * @param threshold Configurable sensitivity cutoff in [0.0, 1.0].
     * @return AdaptivePrecisionDecision with per-tile assignments and distribution metrics.
     * @throws AdaptivePrecisionException if threshold is outside [0.0, 1.0] or non-finite.
     */
    [[nodiscard]] static AdaptivePrecisionDecision decide_precisions(
        const analysis::SensitivityReport& report,
        double threshold = DEFAULT_SENSITIVITY_THRESHOLD);

    /**
     * @brief Assigns FP32 to exactly fp32_count randomly chosen tiles, and FP16 to the rest.
     *
     * Used as a deterministic null hypothesis baseline for research ablation studies.
     *
     * @param tile_mgr TileManager defining the grid and total tile count.
     * @param fp32_count Exact count of tiles to assign FP32 precision.
     * @param seed Deterministic random seed for reproducibility.
     * @return AdaptivePrecisionDecision matching the exact tile budget.
     */
    [[nodiscard]] static AdaptivePrecisionDecision decide_random(
        const image::TileManager& tile_mgr,
        size_t fp32_count,
        uint32_t seed = 42);

    /**
     * @brief Assigns random tile precisions matching the exact FP32 budget of a guided decision.
     *
     * @param tile_mgr TileManager defining the image grid.
     * @param guided_decision The sensitivity-guided decision whose fp32_count will be matched.
     * @param seed Deterministic random seed for reproducibility.
     * @return AdaptivePrecisionDecision with exact budget parity.
     */
    [[nodiscard]] static AdaptivePrecisionDecision decide_random_from_policy(
        const image::TileManager& tile_mgr,
        const AdaptivePrecisionDecision& guided_decision,
        uint32_t seed = 42);

    /**
     * @brief Executes enhancement on an image using a pre-determined precision decision.
     *
     * Timed execution strictly spans per-tile enhancement and image reconstruction.
     *
     * @param bgr_normalized Normalized input image (CV_32FC3 in [0.0, 1.0]).
     * @param config Enhancement parameter configuration.
     * @param tile_mgr TileManager defining tile partitioning.
     * @param decision Pre-computed precision decision (guided, random, or custom).
     * @param backend SIMD execution backend.
     * @return Pair of enhanced image and execution time in milliseconds.
     */
    [[nodiscard]] static std::pair<cv::Mat, double> enhance_with_decision(
        const cv::Mat& bgr_normalized,
        const enhancement::EnhancementConfig& config,
        const image::TileManager& tile_mgr,
        const AdaptivePrecisionDecision& decision,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);

    /**
     * @brief Executes tile-based adaptive precision image enhancement.
     *
     * Timed execution strictly spans:
     * 1. Sensitivity analysis
     * 2. Precision decision
     * 3. Per-tile enhancement (FP32 or software-emulated FP16)
     * 4. Full image reconstruction
     *
     * Reference FP32 computation and error analysis are strictly excluded from this timing.
     *
     * @param bgr_normalized Normalized input image (CV_32FC3 in [0.0, 1.0]).
     * @param config Enhancement parameter configuration.
     * @param tile_width Nominal tile width in pixels.
     * @param tile_height Nominal tile height in pixels.
     * @param threshold Sensitivity threshold in [0.0, 1.0].
     * @param weights Sensitivity feature weights.
     * @return Complete AdaptiveEnhancementResult.
     */
    [[nodiscard]] static AdaptiveEnhancementResult enhance_adaptive(
        const cv::Mat& bgr_normalized,
        const enhancement::EnhancementConfig& config,
        int tile_width = 32,
        int tile_height = 32,
        double threshold = DEFAULT_SENSITIVITY_THRESHOLD,
        const analysis::SensitivityWeights& weights = analysis::SensitivityWeights{},
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);

    /**
     * @brief Processes pre-tiled image using pre-computed precision assignments.
     */
    static void process_tiles(
        const cv::Mat& bgr_normalized,
        cv::Mat& dst_output,
        const image::TileManager& tile_mgr,
        const AdaptivePrecisionDecision& decision,
        const enhancement::EnhancementConfig& config,
        simd::ExecutionBackend backend = simd::ExecutionBackend::Auto);
};

}  // namespace mirage::precision

