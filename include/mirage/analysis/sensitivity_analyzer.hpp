#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "mirage/image/tile_manager.hpp"

namespace mirage::analysis {

/**
 * @brief Exception thrown when sensitivity analysis encounters invalid parameters or unsupported data.
 */
class SensitivityException : public std::runtime_error {
public:
    explicit SensitivityException(const std::string& message)
        : std::runtime_error("SensitivityException: " + message) {}
};

/**
 * @brief Configurable feature weights for the tile sensitivity score calculation.
 *
 * S = w_c * C + w_g * G + w_e * E + w_l * L
 *
 * All weights must be non-negative and sum to 1.0 (within numerical tolerance).
 */
struct SensitivityWeights {
    double color_weight = 0.25;       ///< Weight for normalized color variation (a*, b*)
    double gradient_weight = 0.25;    ///< Weight for normalized gradient magnitude (Sobel L*)
    double edge_weight = 0.25;        ///< Weight for normalized edge density (L* gradient threshold)
    double contrast_weight = 0.25;    ///< Weight for normalized local contrast (L* standard deviation)

    /**
     * @brief Validates weights ensuring non-negativity and sum == 1.0.
     * @throws SensitivityException if validation fails.
     */
    void validate() const;
};

/**
 * @brief Sensitivity metrics computed for an individual image tile.
 */
struct TileSensitivity {
    size_t id = 0;                    ///< Unique tile identifier
    int x = 0;                        ///< X-coordinate of top-left corner
    int y = 0;                        ///< Y-coordinate of top-left corner
    int width = 0;                    ///< Tile width in pixels
    int height = 0;                   ///< Tile height in pixels
    bool is_partial = false;          ///< True if clipped by image boundary

    double color_variation = 0.0;     ///< Normalized color variation C in [0.0, 1.0]
    double gradient_magnitude = 0.0;  ///< Normalized gradient magnitude G in [0.0, 1.0]
    double edge_density = 0.0;        ///< Normalized edge density E in [0.0, 1.0]
    double local_contrast = 0.0;      ///< Normalized local contrast L in [0.0, 1.0]
    double sensitivity_score = 0.0;   ///< Combined weighted sensitivity S in [0.0, 1.0]
};

/**
 * @brief Summary report of sensitivity analysis across an entire image.
 */
struct SensitivityReport {
    int image_width = 0;
    int image_height = 0;
    int nominal_tile_width = 0;
    int nominal_tile_height = 0;
    int grid_cols = 0;
    int grid_rows = 0;
    size_t total_tiles = 0;
    size_t complete_tiles = 0;
    size_t partial_tiles = 0;

    double min_sensitivity = 0.0;
    double max_sensitivity = 0.0;
    double avg_sensitivity = 0.0;

    double avg_color_variation = 0.0;
    double avg_gradient_magnitude = 0.0;
    double avg_edge_density = 0.0;
    double avg_local_contrast = 0.0;

    double execution_time_ms = 0.0;

    SensitivityWeights weights{};
    std::vector<TileSensitivity> tiles{};
};

/**
 * @brief Tile-based image sensitivity analyzer.
 *
 * Evaluates spatial variation, gradient magnitude, edge density, and local contrast per tile
 * using scientific CIELAB representations.
 */
class SensitivityAnalyzer {
public:
    /**
     * @brief Fixed threshold on Sobel gradient magnitude for edge pixel detection.
     * In CIELAB (L* in [0, 100]), Delta L* ~ 7.5 across 2 pixels yields a Sobel response ~ 30.0.
     */
    static constexpr double EDGE_THRESHOLD = 30.0;

    /**
     * @brief Theoretical maximum chromatic standard deviation in scientific CIELAB.
     * With a*, b* in [-127, 127], sqrt(127^2 + 127^2) ~= 179.605.
     */
    static constexpr double COLOR_VAR_MAX = 179.60515;

    /**
     * @brief Theoretical maximum Sobel 3x3 gradient magnitude for L* in [0, 100].
     * Max Sobel 1D is 400.0, 2D magnitude is sqrt(400^2 + 400^2) ~= 565.685.
     */
    static constexpr double GRADIENT_MAG_MAX = 565.68542;

    /**
     * @brief Theoretical maximum standard deviation of L* in [0, 100].
     * By Popoviciu's inequality, sigma_max = (100 - 0) / 2 = 50.0.
     */
    static constexpr double LOCAL_CONTRAST_MAX = 50.0;

    explicit SensitivityAnalyzer(SensitivityWeights weights = SensitivityWeights{});

    [[nodiscard]] const SensitivityWeights& weights() const noexcept { return m_weights; }
    void set_weights(const SensitivityWeights& weights);

    /**
     * @brief Analyzes a normalized BGR image (CV_32FC3, [0.0, 1.0]).
     *
     * @param bgr_normalized Normalized BGR image in [0.0, 1.0].
     * @param tile_width Nominal tile width in pixels.
     * @param tile_height Nominal tile height in pixels.
     * @return Complete SensitivityReport.
     */
    [[nodiscard]] SensitivityReport analyze(const cv::Mat& bgr_normalized,
                                           int tile_width,
                                           int tile_height) const;

    /**
     * @brief Analyzes an image given an existing TileManager and precomputed scientific CIELAB matrix.
     */
    [[nodiscard]] SensitivityReport analyze(const image::TileManager& tile_manager,
                                           const cv::Mat& lab_image) const;

    /**
     * @brief Renders a color heatmap visualization over the base image.
     *
     * @param base_image Base image (CV_8UC3 in [0, 255] or CV_32FC3 in [0.0, 1.0]).
     * @param report Sensitivity report containing tile scores.
     * @param alpha Blending opacity for the heatmap overlay (default: 0.55).
     * @param draw_labels If true, draws tile IDs where spatial dimensions permit.
     * @return Annotated BGR image (CV_8UC3) with embedded colorbar legend.
     */
    [[nodiscard]] cv::Mat render_heatmap(const cv::Mat& base_image,
                                         const SensitivityReport& report,
                                         double alpha = 0.55,
                                         bool draw_labels = true) const;

    /**
     * @brief Renders and writes the sensitivity heatmap visualization to disk.
     */
    void save_heatmap(const std::filesystem::path& path,
                      const cv::Mat& base_image,
                      const SensitivityReport& report,
                      double alpha = 0.55,
                      bool draw_labels = true) const;

    /**
     * @brief Exports individual tile sensitivity metrics to a CSV file.
     *
     * @param path Target CSV file path.
     * @param report Sensitivity report to serialize.
     */
    void export_csv(const std::filesystem::path& path,
                    const SensitivityReport& report) const;

private:
    SensitivityWeights m_weights;

    [[nodiscard]] TileSensitivity compute_tile_sensitivity(const image::Tile& tile,
                                                          const cv::Mat& lab_tile,
                                                          const cv::Mat& grad_tile,
                                                          const cv::Mat& edge_tile) const;
};

}  // namespace mirage::analysis

