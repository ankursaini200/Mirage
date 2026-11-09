#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace mirage::image {

class TileException : public std::runtime_error {
public:
    explicit TileException(const std::string& message)
        : std::runtime_error("TileException: " + message) {}
};

/**
 * @brief Representation of an individual spatial tile within an image.
 */
struct Tile {
    size_t id = 0;              ///< Globally unique tile identifier
    int x = 0;                  ///< X-coordinate of top-left corner
    int y = 0;                  ///< Y-coordinate of top-left corner
    int width = 0;              ///< Width in pixels
    int height = 0;             ///< Height in pixels
    bool is_partial = false;    ///< True if clipped by image boundaries

    /**
     * @brief Safe, non-owning ROI view into the source image buffer.
     * References the parent image's memory directly with identical row stride.
     */
    cv::Mat view;

    [[nodiscard]] cv::Rect rect() const noexcept {
        return cv::Rect(x, y, width, height);
    }
};

/**
 * @brief Manages spatial tiling, non-divisible boundary handling, and image reconstruction.
 */
class TileManager {
public:
    /**
     * @brief Partitions an image into rectangular tiles of nominal size (tile_width x tile_height).
     *
     * @param image Source image (CV_8UC3 or CV_32FC3).
     * @param tile_width Nominal width of each tile (must be > 0).
     * @param tile_height Nominal height of each tile (must be > 0).
     * @throws TileException if image is empty or tile dimensions are non-positive.
     */
    TileManager(const cv::Mat& image, int tile_width, int tile_height);

    [[nodiscard]] size_t total_tiles() const noexcept { return m_tiles.size(); }
    [[nodiscard]] size_t complete_tiles() const noexcept { return m_complete_tiles; }
    [[nodiscard]] size_t partial_tiles() const noexcept { return m_partial_tiles; }

    [[nodiscard]] int nominal_tile_width() const noexcept { return m_nominal_tile_width; }
    [[nodiscard]] int nominal_tile_height() const noexcept { return m_nominal_tile_height; }
    [[nodiscard]] int grid_cols() const noexcept { return m_grid_cols; }
    [[nodiscard]] int grid_rows() const noexcept { return m_grid_rows; }

    [[nodiscard]] int image_width() const noexcept { return m_image_width; }
    [[nodiscard]] int image_height() const noexcept { return m_image_height; }
    [[nodiscard]] int image_channels() const noexcept { return m_image_channels; }
    [[nodiscard]] int image_type() const noexcept { return m_image_type; }

    [[nodiscard]] const std::vector<Tile>& tiles() const noexcept { return m_tiles; }
    [[nodiscard]] const Tile& tile(size_t id) const;

    /**
     * @brief Reconstructs the full image from its constituent tiles.
     * Guaranteed to match original dimensions, channels, and types without pixel loss.
     */
    [[nodiscard]] cv::Mat reconstruct_image() const;

    /**
     * @brief Generates an annotated copy of the source image displaying tile boundaries and IDs.
     */
    [[nodiscard]] cv::Mat render_tile_overlay(const cv::Scalar& border_color = cv::Scalar(0, 255, 0),
                                              int border_thickness = 1,
                                              bool draw_labels = true) const;

    /**
     * @brief Renders and writes the tile visualization overlay to disk.
     */
    void save_tile_visualization(const std::filesystem::path& output_path,
                                 const cv::Scalar& border_color = cv::Scalar(0, 255, 0),
                                 int border_thickness = 1,
                                 bool draw_labels = true) const;

private:
    cv::Mat m_source_image;
    int m_nominal_tile_width = 0;
    int m_nominal_tile_height = 0;
    int m_image_width = 0;
    int m_image_height = 0;
    int m_image_channels = 0;
    int m_image_type = 0;
    int m_grid_cols = 0;
    int m_grid_rows = 0;

    size_t m_complete_tiles = 0;
    size_t m_partial_tiles = 0;
    std::vector<Tile> m_tiles;
};

}  // namespace mirage::image

