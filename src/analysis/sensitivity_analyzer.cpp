#include "mirage/analysis/sensitivity_analyzer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "mirage/image/color_converter.hpp"

namespace mirage::analysis {

void SensitivityWeights::validate() const {
    if (color_weight < 0.0 || gradient_weight < 0.0 ||
        edge_weight < 0.0 || contrast_weight < 0.0) {
        throw SensitivityException("Sensitivity feature weights must be non-negative.");
    }

    const double sum = color_weight + gradient_weight + edge_weight + contrast_weight;
    if (std::abs(sum - 1.0) > 1e-5) {
        std::ostringstream ss;
        ss << "Sensitivity feature weights must sum to 1.0 (got " << std::fixed << std::setprecision(4) << sum << ").";
        throw SensitivityException(ss.str());
    }
}

SensitivityAnalyzer::SensitivityAnalyzer(SensitivityWeights weights)
    : m_weights(weights) {
    m_weights.validate();
}

void SensitivityAnalyzer::set_weights(const SensitivityWeights& weights) {
    weights.validate();
    m_weights = weights;
}

SensitivityReport SensitivityAnalyzer::analyze(const cv::Mat& bgr_normalized,
                                               int tile_width,
                                               int tile_height) const {
    if (bgr_normalized.empty()) {
        throw SensitivityException("Cannot analyze sensitivity of an empty image matrix.");
    }
    if (bgr_normalized.channels() != 3 || bgr_normalized.type() != CV_32FC3) {
        throw SensitivityException("SensitivityAnalyzer requires CV_32FC3 normalized BGR input.");
    }
    if (tile_width <= 0 || tile_height <= 0) {
        throw SensitivityException("Tile dimensions must be positive integers.");
    }

    const auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Partition image into spatial tiles
    const image::TileManager tile_manager(bgr_normalized, tile_width, tile_height);

    // 2. Convert entire image to scientific CIELAB once (L* in [0, 100], a*, b* in [-127, 127])
    const cv::Mat lab_image = image::ColorConverter::bgr_to_lab(bgr_normalized);

    // 3. Delegate to tile analyzer
    SensitivityReport report = analyze(tile_manager, lab_image);

    const auto end_time = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    report.execution_time_ms = elapsed_ms;

    return report;
}

SensitivityReport SensitivityAnalyzer::analyze(const image::TileManager& tile_manager,
                                               const cv::Mat& lab_image) const {
    if (lab_image.empty()) {
        throw SensitivityException("Cannot analyze sensitivity with empty CIELAB matrix.");
    }
    if (lab_image.channels() != 3 || lab_image.type() != CV_32FC3) {
        throw SensitivityException("CIELAB matrix must be CV_32FC3.");
    }
    if (lab_image.cols != tile_manager.image_width() || lab_image.rows != tile_manager.image_height()) {
        throw SensitivityException("TileManager dimensions do not match CIELAB image dimensions.");
    }

    SensitivityReport report{};
    report.image_width = tile_manager.image_width();
    report.image_height = tile_manager.image_height();
    report.nominal_tile_width = tile_manager.nominal_tile_width();
    report.nominal_tile_height = tile_manager.nominal_tile_height();
    report.grid_cols = tile_manager.grid_cols();
    report.grid_rows = tile_manager.grid_rows();
    report.total_tiles = tile_manager.total_tiles();
    report.complete_tiles = tile_manager.complete_tiles();
    report.partial_tiles = tile_manager.partial_tiles();
    report.weights = m_weights;

    if (report.total_tiles == 0) {
        return report;
    }

    // Extract L* channel (lightness in [0, 100])
    cv::Mat L_channel;
    cv::extractChannel(lab_image, L_channel, 0);

    // Compute Sobel gradients on full L* image (preserves boundary continuity)
    cv::Mat grad_x, grad_y;
    cv::Sobel(L_channel, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(L_channel, grad_y, CV_32F, 0, 1, 3);

    cv::Mat grad_mag;
    cv::magnitude(grad_x, grad_y, grad_mag);

    // Edge mask: pixels with gradient magnitude exceeding fixed threshold
    cv::Mat edge_mask;
    cv::compare(grad_mag, EDGE_THRESHOLD, edge_mask, cv::CMP_GE);

    report.tiles.reserve(report.total_tiles);

    double sum_s = 0.0;
    double sum_c = 0.0;
    double sum_g = 0.0;
    double sum_e = 0.0;
    double sum_l = 0.0;
    double min_s = 1.0;
    double max_s = 0.0;

    for (const auto& tile : tile_manager.tiles()) {
        const cv::Rect rect(tile.x, tile.y, tile.width, tile.height);
        const cv::Mat lab_tile = lab_image(rect);
        const cv::Mat grad_tile = grad_mag(rect);
        const cv::Mat edge_tile = edge_mask(rect);

        const TileSensitivity ts = compute_tile_sensitivity(tile, lab_tile, grad_tile, edge_tile);
        report.tiles.push_back(ts);

        sum_s += ts.sensitivity_score;
        sum_c += ts.color_variation;
        sum_g += ts.gradient_magnitude;
        sum_e += ts.edge_density;
        sum_l += ts.local_contrast;

        min_s = std::min(min_s, ts.sensitivity_score);
        max_s = std::max(max_s, ts.sensitivity_score);
    }

    const double total_f = static_cast<double>(report.total_tiles);
    report.min_sensitivity = min_s;
    report.max_sensitivity = max_s;
    report.avg_sensitivity = sum_s / total_f;
    report.avg_color_variation = sum_c / total_f;
    report.avg_gradient_magnitude = sum_g / total_f;
    report.avg_edge_density = sum_e / total_f;
    report.avg_local_contrast = sum_l / total_f;

    return report;
}

TileSensitivity SensitivityAnalyzer::compute_tile_sensitivity(const image::Tile& tile,
                                                              const cv::Mat& lab_tile,
                                                              const cv::Mat& grad_tile,
                                                              const cv::Mat& edge_tile) const {
    TileSensitivity ts{};
    ts.id = tile.id;
    ts.x = tile.x;
    ts.y = tile.y;
    ts.width = tile.width;
    ts.height = tile.height;
    ts.is_partial = tile.is_partial;

    const int total_pixels = tile.width * tile.height;
    if (total_pixels <= 0) {
        return ts;
    }

    // Feature A: Color Variation (a*, b* chromaticity channels standard deviation)
    // lab_tile channels: 0 = L*, 1 = a*, 2 = b*
    cv::Scalar mean_lab, stddev_lab;
    cv::meanStdDev(lab_tile, mean_lab, stddev_lab);

    const double sigma_a = stddev_lab[1];
    const double sigma_b = stddev_lab[2];
    const double sigma_color = std::sqrt(sigma_a * sigma_a + sigma_b * sigma_b);
    ts.color_variation = std::clamp(sigma_color / COLOR_VAR_MAX, 0.0, 1.0);

    // Feature B: Gradient Magnitude (mean Sobel magnitude on L*)
    const cv::Scalar mean_grad = cv::mean(grad_tile);
    ts.gradient_magnitude = std::clamp(mean_grad[0] / GRADIENT_MAG_MAX, 0.0, 1.0);

    // Feature C: Edge Density (proportion of pixels crossing edge gradient threshold)
    const int edge_pixels = cv::countNonZero(edge_tile);
    ts.edge_density = std::clamp(static_cast<double>(edge_pixels) / static_cast<double>(total_pixels), 0.0, 1.0);

    // Feature D: Local Contrast (L* luminance standard deviation)
    const double sigma_L = stddev_lab[0];
    ts.local_contrast = std::clamp(sigma_L / LOCAL_CONTRAST_MAX, 0.0, 1.0);

    // Combined Weighted Sensitivity Score
    const double score = m_weights.color_weight * ts.color_variation +
                         m_weights.gradient_weight * ts.gradient_magnitude +
                         m_weights.edge_weight * ts.edge_density +
                         m_weights.contrast_weight * ts.local_contrast;

    ts.sensitivity_score = std::clamp(score, 0.0, 1.0);

    return ts;
}

cv::Mat SensitivityAnalyzer::render_heatmap(const cv::Mat& base_image,
                                            const SensitivityReport& report,
                                            double alpha,
                                            bool draw_labels) const {
    if (base_image.empty()) {
        throw SensitivityException("Cannot render heatmap on empty base image.");
    }

    cv::Mat base_8u;
    if (base_image.type() == CV_32FC3) {
        base_image.convertTo(base_8u, CV_8UC3, 255.0);
    } else if (base_image.type() == CV_8UC3) {
        base_8u = base_image.clone();
    } else {
        throw SensitivityException("Base image must be CV_8UC3 or CV_32FC3.");
    }

    // 1. Build grayscale sensitivity image
    cv::Mat heatmap_gray(base_8u.size(), CV_8UC1, cv::Scalar(0));
    for (const auto& t : report.tiles) {
        const uint8_t byte_val = static_cast<uint8_t>(
            std::clamp(std::round(t.sensitivity_score * 255.0), 0.0, 255.0));
        heatmap_gray(cv::Rect(t.x, t.y, t.width, t.height)).setTo(byte_val);
    }

    // 2. Apply COLORMAP_TURBO for smooth, perceptually uniform color gradient
    cv::Mat heatmap_color;
    cv::applyColorMap(heatmap_gray, heatmap_color, cv::COLORMAP_TURBO);

    // 3. Alpha blend heatmap over base image
    const double clamped_alpha = std::clamp(alpha, 0.0, 1.0);
    cv::Mat blended;
    cv::addWeighted(base_8u, 1.0 - clamped_alpha, heatmap_color, clamped_alpha, 0.0, blended);

    // 4. Draw tile boundaries and optional labels
    for (const auto& t : report.tiles) {
        cv::rectangle(blended, cv::Rect(t.x, t.y, t.width, t.height), cv::Scalar(40, 40, 40), 1);
        if (draw_labels && t.width >= 32 && t.height >= 22) {
            const std::string label = std::to_string(t.id);
            // Shadow
            cv::putText(blended, label, cv::Point(t.x + 4, t.y + 13),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            // Foreground text
            cv::putText(blended, label, cv::Point(t.x + 3, t.y + 12),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }

    // 5. Append bottom legend bar
    const int legend_height = 36;
    cv::Mat canvas(blended.rows + legend_height, blended.cols, CV_8UC3, cv::Scalar(28, 28, 28));
    blended.copyTo(canvas(cv::Rect(0, 0, blended.cols, blended.rows)));

    // Draw horizontal colorbar gradient
    const int bar_width = std::min(240, std::max(80, blended.cols - 160));
    const int bar_height = 10;
    const int bar_x = (blended.cols - bar_width) / 2;
    const int bar_y = blended.rows + 6;

    if (bar_width > 10 && blended.cols >= 180) {
        cv::Mat colorbar_gray(bar_height, bar_width, CV_8UC1);
        for (int col = 0; col < bar_width; ++col) {
            const uint8_t val = static_cast<uint8_t>((col * 255) / (bar_width - 1));
            colorbar_gray.col(col).setTo(val);
        }
        cv::Mat colorbar_turbo;
        cv::applyColorMap(colorbar_gray, colorbar_turbo, cv::COLORMAP_TURBO);
        colorbar_turbo.copyTo(canvas(cv::Rect(bar_x, bar_y, bar_width, bar_height)));
        cv::rectangle(canvas, cv::Rect(bar_x - 1, bar_y - 1, bar_width + 2, bar_height + 2),
                     cv::Scalar(160, 160, 160), 1);

        // Labels
        cv::putText(canvas, "0.0 (Low)", cv::Point(std::max(4, bar_x - 65), bar_y + 9),
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        cv::putText(canvas, "1.0 (High)", cv::Point(bar_x + bar_width + 6, bar_y + 9),
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        cv::putText(canvas, "Tile Sensitivity Heatmap", cv::Point((blended.cols - 130) / 2, bar_y + 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(170, 170, 170), 1, cv::LINE_AA);
    }

    return canvas;
}

void SensitivityAnalyzer::save_heatmap(const std::filesystem::path& path,
                                       const cv::Mat& base_image,
                                       const SensitivityReport& report,
                                       double alpha,
                                       bool draw_labels) const {
    if (path.empty()) {
        throw SensitivityException("Cannot save heatmap to empty file path.");
    }

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    const cv::Mat rendered = render_heatmap(base_image, report, alpha, draw_labels);
    if (!cv::imwrite(path.string(), rendered)) {
        throw SensitivityException("Failed to write heatmap image to: " + path.string());
    }
}

void SensitivityAnalyzer::export_csv(const std::filesystem::path& path,
                                     const SensitivityReport& report) const {
    if (path.empty()) {
        throw SensitivityException("Cannot export CSV to empty file path.");
    }

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        throw SensitivityException("Failed to open file for CSV export: " + path.string());
    }

    // CSV Header
    file << "tile_id,x,y,width,height,is_partial,color_variation,gradient_magnitude,edge_density,local_contrast,sensitivity_score\n";

    file << std::fixed << std::setprecision(6);
    for (const auto& t : report.tiles) {
        file << t.id << ","
             << t.x << ","
             << t.y << ","
             << t.width << ","
             << t.height << ","
             << (t.is_partial ? 1 : 0) << ","
             << t.color_variation << ","
             << t.gradient_magnitude << ","
             << t.edge_density << ","
             << t.local_contrast << ","
             << t.sensitivity_score << "\n";
    }
}

}  // namespace mirage::analysis

