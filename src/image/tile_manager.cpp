#include "mirage/image/tile_manager.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace mirage::image {

TileManager::TileManager(const cv::Mat& image, int tile_width, int tile_height)
    : m_source_image(image),
      m_nominal_tile_width(tile_width),
      m_nominal_tile_height(tile_height) {
    if (image.empty()) {
        throw TileException("Cannot create TileManager from an empty image matrix.");
    }

    if (tile_width <= 0 || tile_height <= 0) {
        throw TileException("Tile width and height must be strictly positive integers (got " +
                            std::to_string(tile_width) + "x" + std::to_string(tile_height) + ").");
    }

    m_image_width = image.cols;
    m_image_height = image.rows;
    m_image_channels = image.channels();
    m_image_type = image.type();

    m_grid_cols = (m_image_width + tile_width - 1) / tile_width;
    m_grid_rows = (m_image_height + tile_height - 1) / tile_height;

    const size_t total_expected_tiles = static_cast<size_t>(m_grid_cols) * static_cast<size_t>(m_grid_rows);
    m_tiles.reserve(total_expected_tiles);

    size_t current_id = 0;
    for (int r = 0; r < m_grid_rows; ++r) {
        for (int c = 0; c < m_grid_cols; ++c) {
            const int x = c * tile_width;
            const int y = r * tile_height;
            const int w = std::min(tile_width, m_image_width - x);
            const int h = std::min(tile_height, m_image_height - y);
            const bool partial = (w < tile_width || h < tile_height);

            Tile t;
            t.id = current_id++;
            t.x = x;
            t.y = y;
            t.width = w;
            t.height = h;
            t.is_partial = partial;

            // Safe ROI view directly into source image memory without copying
            t.view = m_source_image(cv::Rect(x, y, w, h));

            if (partial) {
                m_partial_tiles++;
            } else {
                m_complete_tiles++;
            }

            m_tiles.push_back(std::move(t));
        }
    }
}

const Tile& TileManager::tile(size_t id) const {
    if (id >= m_tiles.size()) {
        throw TileException("Tile index out of range: " + std::to_string(id) +
                            " (total: " + std::to_string(m_tiles.size()) + ")");
    }
    return m_tiles[id];
}

cv::Mat TileManager::reconstruct_image() const {
    cv::Mat reconstructed(m_image_height, m_image_width, m_image_type);

    for (const auto& t : m_tiles) {
        cv::Mat target_roi = reconstructed(t.rect());
        t.view.copyTo(target_roi);
    }

    return reconstructed;
}

cv::Mat TileManager::render_tile_overlay(const cv::Scalar& border_color,
                                         int border_thickness,
                                         bool draw_labels) const {
    cv::Mat overlay;
    if (m_source_image.type() == CV_8UC3) {
        overlay = m_source_image.clone();
    } else if (m_source_image.type() == CV_32FC3) {
        m_source_image.convertTo(overlay, CV_8UC3, 255.0);
    } else {
        m_source_image.convertTo(overlay, CV_8UC3);
    }

    for (const auto& t : m_tiles) {
        // Draw tile border
        cv::rectangle(overlay, t.rect(), border_color, border_thickness);

        // Draw tile ID label if tile is large enough to render legibly
        if (draw_labels && t.width >= 24 && t.height >= 20) {
            const std::string label = std::to_string(t.id);
            const double font_scale = std::clamp(static_cast<double>(t.width) / 75.0, 0.3, 0.6);
            const int text_x = t.x + 3;
            const int text_y = t.y + std::min(14, t.height - 3);

            // Black outline / shadow for contrast
            cv::putText(overlay, label, cv::Point(text_x + 1, text_y + 1),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            // White foreground text
            cv::putText(overlay, label, cv::Point(text_x, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }

    return overlay;
}

void TileManager::save_tile_visualization(const std::filesystem::path& output_path,
                                          const cv::Scalar& border_color,
                                          int border_thickness,
                                          bool draw_labels) const {
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    const cv::Mat overlay = render_tile_overlay(border_color, border_thickness, draw_labels);
    const bool success = cv::imwrite(output_path.string(), overlay);
    if (!success) {
        throw TileException("Failed to write tile visualization to: " + output_path.string());
    }
}

}  // namespace mirage::image

