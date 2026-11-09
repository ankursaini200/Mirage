#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <set>

#include "mirage/image/tile_manager.hpp"

namespace mirage::tests {

namespace {

cv::Mat create_test_image(int width, int height, int type = CV_8UC3) {
    cv::Mat img(height, width, type);
    if (type == CV_8UC3) {
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                img.at<cv::Vec3b>(r, c) = cv::Vec3b(
                    static_cast<uint8_t>((c * 17) % 256),
                    static_cast<uint8_t>((r * 23) % 256),
                    static_cast<uint8_t>(((c + r) * 31) % 256)
                );
            }
        }
    } else if (type == CV_32FC3) {
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                img.at<cv::Vec3f>(r, c) = cv::Vec3f(
                    static_cast<float>(c) / static_cast<float>(width),
                    static_cast<float>(r) / static_cast<float>(height),
                    static_cast<float>(c + r) / static_cast<float>(width + height)
                );
            }
        }
    }
    return img;
}

}  // namespace

TEST(TileManagerTest, DivisibleTileDivision8x8) {
    const cv::Mat img = create_test_image(64, 64);
    const image::TileManager mgr(img, 8, 8);

    EXPECT_EQ(mgr.total_tiles(), 64);
    EXPECT_EQ(mgr.complete_tiles(), 64);
    EXPECT_EQ(mgr.partial_tiles(), 0);
    EXPECT_EQ(mgr.grid_cols(), 8);
    EXPECT_EQ(mgr.grid_rows(), 8);
}

TEST(TileManagerTest, DivisibleTileDivision16x16) {
    const cv::Mat img = create_test_image(64, 64);
    const image::TileManager mgr(img, 16, 16);

    EXPECT_EQ(mgr.total_tiles(), 16);
    EXPECT_EQ(mgr.complete_tiles(), 16);
    EXPECT_EQ(mgr.partial_tiles(), 0);
    EXPECT_EQ(mgr.grid_cols(), 4);
    EXPECT_EQ(mgr.grid_rows(), 4);
}

TEST(TileManagerTest, DivisibleTileDivision32x32) {
    const cv::Mat img = create_test_image(128, 128);
    const image::TileManager mgr(img, 32, 32);

    EXPECT_EQ(mgr.total_tiles(), 16);
    EXPECT_EQ(mgr.complete_tiles(), 16);
    EXPECT_EQ(mgr.partial_tiles(), 0);
    EXPECT_EQ(mgr.grid_cols(), 4);
    EXPECT_EQ(mgr.grid_rows(), 4);
}

TEST(TileManagerTest, DivisibleTileDivision64x64) {
    const cv::Mat img = create_test_image(256, 256);
    const image::TileManager mgr(img, 64, 64);

    EXPECT_EQ(mgr.total_tiles(), 16);
    EXPECT_EQ(mgr.complete_tiles(), 16);
    EXPECT_EQ(mgr.partial_tiles(), 0);
    EXPECT_EQ(mgr.grid_cols(), 4);
    EXPECT_EQ(mgr.grid_rows(), 4);
}

TEST(TileManagerTest, NonDivisibleDimensions) {
    // 100 x 75 with 32x32 tiles
    // Grid cols = ceil(100 / 32) = 4 (widths: 32, 32, 32, 4)
    // Grid rows = ceil(75 / 32) = 3 (heights: 32, 32, 11)
    // Total tiles = 12
    // Complete tiles = 3 * 2 = 6
    // Partial boundary tiles = 12 - 6 = 6
    const cv::Mat img = create_test_image(100, 75);
    const image::TileManager mgr(img, 32, 32);

    EXPECT_EQ(mgr.total_tiles(), 12);
    EXPECT_EQ(mgr.grid_cols(), 4);
    EXPECT_EQ(mgr.grid_rows(), 3);
    EXPECT_EQ(mgr.complete_tiles(), 6);
    EXPECT_EQ(mgr.partial_tiles(), 6);

    // Verify right boundary partial tiles
    const auto& t3 = mgr.tile(3);  // col 3, row 0
    EXPECT_EQ(t3.x, 96);
    EXPECT_EQ(t3.y, 0);
    EXPECT_EQ(t3.width, 4);
    EXPECT_EQ(t3.height, 32);
    EXPECT_TRUE(t3.is_partial);

    // Verify bottom-right corner partial tile
    const auto& t11 = mgr.tile(11);  // col 3, row 2
    EXPECT_EQ(t11.x, 96);
    EXPECT_EQ(t11.y, 64);
    EXPECT_EQ(t11.width, 4);
    EXPECT_EQ(t11.height, 11);
    EXPECT_TRUE(t11.is_partial);
}

TEST(TileManagerTest, ImageSmallerThanTileSize) {
    const cv::Mat img = create_test_image(10, 15);
    const image::TileManager mgr(img, 32, 32);

    EXPECT_EQ(mgr.total_tiles(), 1);
    EXPECT_EQ(mgr.complete_tiles(), 0);
    EXPECT_EQ(mgr.partial_tiles(), 1);

    const auto& t0 = mgr.tile(0);
    EXPECT_EQ(t0.x, 0);
    EXPECT_EQ(t0.y, 0);
    EXPECT_EQ(t0.width, 10);
    EXPECT_EQ(t0.height, 15);
    EXPECT_TRUE(t0.is_partial);
}

TEST(TileManagerTest, SinglePixelImage) {
    const cv::Mat img = create_test_image(1, 1);
    const image::TileManager mgr(img, 16, 16);

    EXPECT_EQ(mgr.total_tiles(), 1);
    EXPECT_EQ(mgr.complete_tiles(), 0);
    EXPECT_EQ(mgr.partial_tiles(), 1);

    const auto& t0 = mgr.tile(0);
    EXPECT_EQ(t0.x, 0);
    EXPECT_EQ(t0.y, 0);
    EXPECT_EQ(t0.width, 1);
    EXPECT_EQ(t0.height, 1);
}

TEST(TileManagerTest, UniqueTileIdsAndCoverage) {
    const cv::Mat img = create_test_image(83, 57);
    const image::TileManager mgr(img, 16, 16);

    std::set<size_t> ids;
    size_t pixel_coverage_count = 0;

    for (size_t i = 0; i < mgr.total_tiles(); ++i) {
        const auto& t = mgr.tile(i);
        EXPECT_EQ(t.id, i);
        EXPECT_TRUE(ids.insert(t.id).second) << "Duplicate tile ID detected: " << t.id;

        // Verify tile lies strictly within image boundaries
        EXPECT_GE(t.x, 0);
        EXPECT_GE(t.y, 0);
        EXPECT_LE(t.x + t.width, img.cols);
        EXPECT_LE(t.y + t.height, img.rows);

        pixel_coverage_count += static_cast<size_t>(t.width * t.height);
    }

    // Every pixel in the image must belong to exactly one tile
    EXPECT_EQ(pixel_coverage_count, static_cast<size_t>(img.cols * img.rows));
}

TEST(TileManagerTest, CompleteImageReconstruction) {
    // Test on odd non-divisible dimensions across both 8-bit and floating-point types
    const std::vector<std::pair<int, int>> sizes = {
        {64, 64}, {100, 75}, {83, 57}, {17, 23}, {1, 1}
    };
    const std::vector<int> tile_sizes = {8, 16, 32};

    for (const auto& [w, h] : sizes) {
        for (const int ts : tile_sizes) {
            // Test CV_8UC3
            const cv::Mat original_u8 = create_test_image(w, h, CV_8UC3);
            const image::TileManager mgr_u8(original_u8, ts, ts);
            const cv::Mat reconstructed_u8 = mgr_u8.reconstruct_image();

            ASSERT_EQ(reconstructed_u8.size(), original_u8.size());
            ASSERT_EQ(reconstructed_u8.type(), original_u8.type());
            EXPECT_EQ(cv::norm(original_u8, reconstructed_u8, cv::NORM_INF), 0.0)
                << "Mismatch in reconstruction for size " << w << "x" << h << " with tile size " << ts;

            // Test CV_32FC3
            const cv::Mat original_f32 = create_test_image(w, h, CV_32FC3);
            const image::TileManager mgr_f32(original_f32, ts, ts);
            const cv::Mat reconstructed_f32 = mgr_f32.reconstruct_image();

            ASSERT_EQ(reconstructed_f32.size(), original_f32.size());
            ASSERT_EQ(reconstructed_f32.type(), original_f32.type());
            EXPECT_EQ(cv::norm(original_f32, reconstructed_f32, cv::NORM_INF), 0.0)
                << "Mismatch in float reconstruction for size " << w << "x" << h << " with tile size " << ts;
        }
    }
}

TEST(TileManagerTest, VisualizationOverlayRendering) {
    const cv::Mat img = create_test_image(100, 75);
    const image::TileManager mgr(img, 32, 32);

    const cv::Mat overlay = mgr.render_tile_overlay();
    EXPECT_EQ(overlay.size(), img.size());
    EXPECT_EQ(overlay.type(), CV_8UC3);
    EXPECT_FALSE(overlay.empty());
}

}  // namespace mirage::tests

