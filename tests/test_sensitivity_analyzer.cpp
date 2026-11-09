#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "mirage/analysis/sensitivity_analyzer.hpp"
#include "mirage/image/color_converter.hpp"

namespace mirage::tests {

namespace {

// Helper to create synthetic normalized BGR image (CV_32FC3, [0.0, 1.0])
cv::Mat create_solid_bgr(int width, int height, const cv::Scalar& bgr) {
    return cv::Mat(height, width, CV_32FC3, bgr);
}

cv::Mat create_grayscale_ramp(int width, int height) {
    cv::Mat img(height, width, CV_32FC3);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            const float val = static_cast<float>(c) / static_cast<float>(width - 1);
            img.at<cv::Vec3f>(r, c) = cv::Vec3f(val, val, val);
        }
    }
    return img;
}

cv::Mat create_color_checkerboard(int width, int height, int block_size) {
    cv::Mat img(height, width, CV_32FC3);
    const cv::Vec3f red(0.0f, 0.0f, 1.0f);
    const cv::Vec3f blue(1.0f, 0.0f, 0.0f);
    const cv::Vec3f green(0.0f, 1.0f, 0.0f);
    const cv::Vec3f yellow(0.0f, 1.0f, 1.0f);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            const int br = (r / block_size) % 2;
            const int bc = (c / block_size) % 2;
            if (br == 0 && bc == 0) img.at<cv::Vec3f>(r, c) = red;
            else if (br == 0 && bc == 1) img.at<cv::Vec3f>(r, c) = blue;
            else if (br == 1 && bc == 0) img.at<cv::Vec3f>(r, c) = green;
            else img.at<cv::Vec3f>(r, c) = yellow;
        }
    }
    return img;
}

cv::Mat create_edge_bars(int width, int height, int bar_width) {
    cv::Mat img(height, width, CV_32FC3);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            const float val = ((c / bar_width) % 2 == 0) ? 1.0f : 0.0f;
            img.at<cv::Vec3f>(r, c) = cv::Vec3f(val, val, val);
        }
    }
    return img;
}

}  // namespace

TEST(SensitivityAnalyzerTest, WeightValidation) {
    analysis::SensitivityWeights default_weights{};
    EXPECT_NO_THROW(default_weights.validate());

    // Negative weight should throw
    analysis::SensitivityWeights negative_weight{-0.1, 0.3, 0.4, 0.4};
    EXPECT_THROW(negative_weight.validate(), analysis::SensitivityException);

    // Sum not 1.0 should throw
    analysis::SensitivityWeights sum_low{0.2, 0.2, 0.2, 0.2};
    EXPECT_THROW(sum_low.validate(), analysis::SensitivityException);

    analysis::SensitivityWeights sum_high{0.3, 0.3, 0.3, 0.3};
    EXPECT_THROW(sum_high.validate(), analysis::SensitivityException);

    // Valid custom weights
    analysis::SensitivityWeights custom{0.4, 0.3, 0.2, 0.1};
    EXPECT_NO_THROW(custom.validate());
    analysis::SensitivityAnalyzer analyzer(custom);
    EXPECT_DOUBLE_EQ(analyzer.weights().color_weight, 0.4);
}

TEST(SensitivityAnalyzerTest, ConstantUniformColorZeroSensitivity) {
    const analysis::SensitivityAnalyzer analyzer{};

    // 1. Black image (0, 0, 0)
    const cv::Mat black = create_solid_bgr(64, 64, cv::Scalar(0.0f, 0.0f, 0.0f));
    const auto rep_black = analyzer.analyze(black, 32, 32);
    EXPECT_EQ(rep_black.total_tiles, 4u);
    for (const auto& t : rep_black.tiles) {
        EXPECT_NEAR(t.color_variation, 0.0, 1e-6);
        EXPECT_NEAR(t.gradient_magnitude, 0.0, 1e-6);
        EXPECT_NEAR(t.edge_density, 0.0, 1e-6);
        EXPECT_NEAR(t.local_contrast, 0.0, 1e-6);
        EXPECT_NEAR(t.sensitivity_score, 0.0, 1e-6);
    }
    EXPECT_NEAR(rep_black.avg_sensitivity, 0.0, 1e-6);

    // 2. White image (1, 1, 1)
    const cv::Mat white = create_solid_bgr(64, 64, cv::Scalar(1.0f, 1.0f, 1.0f));
    const auto rep_white = analyzer.analyze(white, 32, 32);
    EXPECT_NEAR(rep_white.avg_sensitivity, 0.0, 1e-6);

    // 3. Solid Pure Green (0, 1, 0)
    const cv::Mat green = create_solid_bgr(64, 64, cv::Scalar(0.0f, 1.0f, 0.0f));
    const auto rep_green = analyzer.analyze(green, 32, 32);
    EXPECT_NEAR(rep_green.avg_sensitivity, 0.0, 1e-6);
}

TEST(SensitivityAnalyzerTest, GrayscaleImageColorVariationZero) {
    const analysis::SensitivityAnalyzer analyzer{};

    // Horizontal ramp: B=G=R everywhere, lightness varies from 0 to 1
    const cv::Mat ramp = create_grayscale_ramp(128, 64);
    const auto report = analyzer.analyze(ramp, 32, 32);

    for (const auto& t : report.tiles) {
        // Grayscale has zero chromaticity variation in CIELAB (a*=0, b*=0) up to float conversion precision
        EXPECT_NEAR(t.color_variation, 0.0, 1e-3);
        // But spatial lightness variation should produce non-zero contrast & gradient
        EXPECT_GT(t.local_contrast, 0.0);
        EXPECT_GT(t.gradient_magnitude, 0.0);
    }
}

TEST(SensitivityAnalyzerTest, HighColorVariationVsGrayscale) {
    const analysis::SensitivityAnalyzer analyzer{};

    // Saturated checkerboard (Red, Blue, Green, Yellow)
    const cv::Mat color_img = create_color_checkerboard(64, 64, 16);
    const auto color_report = analyzer.analyze(color_img, 32, 32);

    // Grayscale checkerboard with same lightness steps
    cv::Mat gray_img(64, 64, CV_32FC3);
    for (int r = 0; r < 64; ++r) {
        for (int c = 0; c < 64; ++c) {
            const float v = ((r / 16) % 2 == (c / 16) % 2) ? 0.8f : 0.2f;
            gray_img.at<cv::Vec3f>(r, c) = cv::Vec3f(v, v, v);
        }
    }
    const auto gray_report = analyzer.analyze(gray_img, 32, 32);

    // Color image should have significantly higher color variation than grayscale
    EXPECT_GT(color_report.avg_color_variation, 0.1);
    EXPECT_NEAR(gray_report.avg_color_variation, 0.0, 1e-3);
}

TEST(SensitivityAnalyzerTest, HighGradientImage) {
    const analysis::SensitivityAnalyzer analyzer{};

    // Ramp with steep transition
    const cv::Mat ramp = create_grayscale_ramp(64, 64);
    const auto report = analyzer.analyze(ramp, 32, 32);

    EXPECT_GT(report.avg_gradient_magnitude, 0.0);
    EXPECT_GT(report.avg_local_contrast, 0.0);
    EXPECT_GT(report.avg_sensitivity, 0.0);
}

TEST(SensitivityAnalyzerTest, EdgeRichImage) {
    const analysis::SensitivityAnalyzer analyzer{};

    // Alternating black and white bars every 2 pixels -> dense edges
    const cv::Mat edges = create_edge_bars(64, 64, 2);
    const auto report = analyzer.analyze(edges, 32, 32);

    // Should detect high edge density and high local contrast
    EXPECT_GT(report.avg_edge_density, 0.1);
    EXPECT_GT(report.avg_local_contrast, 0.5);
    EXPECT_GT(report.avg_sensitivity, 0.2);
}

TEST(SensitivityAnalyzerTest, LowContrastImage) {
    const analysis::SensitivityAnalyzer analyzer{};

    // Very subtle variations: values between 0.495 and 0.505
    cv::Mat low_contrast(64, 64, CV_32FC3);
    for (int r = 0; r < 64; ++r) {
        for (int c = 0; c < 64; ++c) {
            const float v = 0.50f + 0.005f * ((r + c) % 2 == 0 ? 1.0f : -1.0f);
            low_contrast.at<cv::Vec3f>(r, c) = cv::Vec3f(v, v, v);
        }
    }
    const auto report = analyzer.analyze(low_contrast, 32, 32);

    // Contrast should be negligible (< 0.02)
    EXPECT_LT(report.avg_local_contrast, 0.02);
    EXPECT_LT(report.avg_sensitivity, 0.05);
}

TEST(SensitivityAnalyzerTest, SensitivityScoreBoundedRange) {
    const analysis::SensitivityAnalyzer analyzer{};

    const cv::Mat color_checker = create_color_checkerboard(96, 96, 12);
    const auto report = analyzer.analyze(color_checker, 32, 32);

    EXPECT_GE(report.min_sensitivity, 0.0);
    EXPECT_LE(report.max_sensitivity, 1.0);
    EXPECT_GE(report.avg_sensitivity, report.min_sensitivity);
    EXPECT_LE(report.avg_sensitivity, report.max_sensitivity);

    for (const auto& t : report.tiles) {
        EXPECT_GE(t.color_variation, 0.0);
        EXPECT_LE(t.color_variation, 1.0);
        EXPECT_GE(t.gradient_magnitude, 0.0);
        EXPECT_LE(t.gradient_magnitude, 1.0);
        EXPECT_GE(t.edge_density, 0.0);
        EXPECT_LE(t.edge_density, 1.0);
        EXPECT_GE(t.local_contrast, 0.0);
        EXPECT_LE(t.local_contrast, 1.0);
        EXPECT_GE(t.sensitivity_score, 0.0);
        EXPECT_LE(t.sensitivity_score, 1.0);
    }
}

TEST(SensitivityAnalyzerTest, PartialBoundaryTilesHandling) {
    const analysis::SensitivityAnalyzer analyzer{};

    // Non-divisible dimensions: 100 x 70 with 32x32 tiles -> 4 cols x 3 rows = 12 tiles
    const cv::Mat img = create_grayscale_ramp(100, 70);
    const auto report = analyzer.analyze(img, 32, 32);

    EXPECT_EQ(report.image_width, 100);
    EXPECT_EQ(report.image_height, 70);
    EXPECT_EQ(report.grid_cols, 4);
    EXPECT_EQ(report.grid_rows, 3);
    EXPECT_EQ(report.total_tiles, 12u);
    EXPECT_EQ(report.complete_tiles, 6u);  // 3x2 inner complete
    EXPECT_EQ(report.partial_tiles, 6u);   // boundary edge tiles

    // Corner tile (col=3, row=2) -> ID 11, width = 100 - 96 = 4, height = 70 - 64 = 6
    const auto& corner_tile = report.tiles[11];
    EXPECT_EQ(corner_tile.id, 11u);
    EXPECT_EQ(corner_tile.x, 96);
    EXPECT_EQ(corner_tile.y, 64);
    EXPECT_EQ(corner_tile.width, 4);
    EXPECT_EQ(corner_tile.height, 6);
    EXPECT_TRUE(corner_tile.is_partial);
    EXPECT_GE(corner_tile.sensitivity_score, 0.0);
    EXPECT_LE(corner_tile.sensitivity_score, 1.0);
}

TEST(SensitivityAnalyzerTest, UniqueTileIdsAndCoverage) {
    const analysis::SensitivityAnalyzer analyzer{};
    const cv::Mat img = create_solid_bgr(110, 85, cv::Scalar(0.2f, 0.4f, 0.6f));
    const auto report = analyzer.analyze(img, 32, 32);

    int total_pixel_area = 0;
    for (size_t i = 0; i < report.tiles.size(); ++i) {
        EXPECT_EQ(report.tiles[i].id, i);
        total_pixel_area += report.tiles[i].width * report.tiles[i].height;
    }
    EXPECT_EQ(total_pixel_area, 110 * 85);
}

TEST(SensitivityAnalyzerTest, DeterministicResults) {
    const analysis::SensitivityAnalyzer analyzer{};
    const cv::Mat img = create_color_checkerboard(64, 64, 8);

    const auto report1 = analyzer.analyze(img, 16, 16);
    const auto report2 = analyzer.analyze(img, 16, 16);

    EXPECT_EQ(report1.tiles.size(), report2.tiles.size());
    for (size_t i = 0; i < report1.tiles.size(); ++i) {
        EXPECT_DOUBLE_EQ(report1.tiles[i].color_variation, report2.tiles[i].color_variation);
        EXPECT_DOUBLE_EQ(report1.tiles[i].gradient_magnitude, report2.tiles[i].gradient_magnitude);
        EXPECT_DOUBLE_EQ(report1.tiles[i].edge_density, report2.tiles[i].edge_density);
        EXPECT_DOUBLE_EQ(report1.tiles[i].local_contrast, report2.tiles[i].local_contrast);
        EXPECT_DOUBLE_EQ(report1.tiles[i].sensitivity_score, report2.tiles[i].sensitivity_score);
    }
}

TEST(SensitivityAnalyzerTest, HeatmapGenerationAndFileSave) {
    const analysis::SensitivityAnalyzer analyzer{};
    const cv::Mat img = create_color_checkerboard(128, 96, 16);
    const auto report = analyzer.analyze(img, 32, 32);

    // 1. Render heatmap matrix
    const cv::Mat heatmap = analyzer.render_heatmap(img, report, 0.55, true);
    EXPECT_FALSE(heatmap.empty());
    EXPECT_EQ(heatmap.cols, 128);
    // Height should be original rows (96) + 36 (legend height)
    EXPECT_EQ(heatmap.rows, 96 + 36);
    EXPECT_EQ(heatmap.type(), CV_8UC3);

    // 2. Save to disk
    const std::filesystem::path temp_path = "output/test_heatmap_output.png";
    EXPECT_NO_THROW(analyzer.save_heatmap(temp_path, img, report));
    EXPECT_TRUE(std::filesystem::exists(temp_path));
    EXPECT_GT(std::filesystem::file_size(temp_path), 0u);
}

TEST(SensitivityAnalyzerTest, CsvReportGeneration) {
    const analysis::SensitivityAnalyzer analyzer{};
    const cv::Mat img = create_color_checkerboard(64, 64, 16);
    const auto report = analyzer.analyze(img, 32, 32);

    const std::filesystem::path csv_path = "output/test_sensitivity_output.csv";
    EXPECT_NO_THROW(analyzer.export_csv(csv_path, report));
    EXPECT_TRUE(std::filesystem::exists(csv_path));

    // Verify CSV header and row count
    std::ifstream file(csv_path);
    ASSERT_TRUE(file.is_open());
    std::string line;
    ASSERT_TRUE(std::getline(file, line));
    EXPECT_EQ(line, "tile_id,x,y,width,height,is_partial,color_variation,gradient_magnitude,edge_density,local_contrast,sensitivity_score");

    size_t row_count = 0;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            row_count++;
        }
    }
    EXPECT_EQ(row_count, report.total_tiles);
}

}  // namespace mirage::tests
