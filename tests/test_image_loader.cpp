#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "mirage/image/image_loader.hpp"

namespace mirage::tests {

class ImageLoaderTest : public ::testing::Test {
protected:
    std::filesystem::path test_dir;
    std::filesystem::path png_path;
    std::filesystem::path jpg_path;
    std::filesystem::path bmp_path;
    std::filesystem::path empty_path;
    std::filesystem::path corrupted_path;

    void SetUp() override {
        test_dir = std::filesystem::temp_directory_path() / "mirage_loader_tests";
        std::filesystem::create_directories(test_dir);

        png_path = test_dir / "test_valid.png";
        jpg_path = test_dir / "test_valid.jpg";
        bmp_path = test_dir / "test_valid.bmp";
        empty_path = test_dir / "test_empty.png";
        corrupted_path = test_dir / "test_corrupted.png";

        // Generate synthetic test image (64x48, 3-channel gradient)
        cv::Mat synthetic(48, 64, CV_8UC3);
        for (int r = 0; r < synthetic.rows; ++r) {
            for (int c = 0; c < synthetic.cols; ++c) {
                synthetic.at<cv::Vec3b>(r, c) = cv::Vec3b(
                    static_cast<uint8_t>((c * 255) / 64),
                    static_cast<uint8_t>((r * 255) / 48),
                    static_cast<uint8_t>(((c + r) * 255) / 112)
                );
            }
        }

        cv::imwrite(png_path.string(), synthetic);
        cv::imwrite(jpg_path.string(), synthetic);
        cv::imwrite(bmp_path.string(), synthetic);

        // Create empty file (0 bytes)
        {
            std::ofstream empty_file(empty_path, std::ios::binary);
        }

        // Create corrupted file (contains random bytes, not valid image header)
        {
            std::ofstream corrupted_file(corrupted_path, std::ios::binary);
            corrupted_file << "NOT_A_VALID_IMAGE_HEADER_GARBAGE_BYTES_1234567890\n";
        }
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(test_dir, ec);
    }
};

TEST_F(ImageLoaderTest, ValidPngLoading) {
    const cv::Mat img = image::ImageLoader::load(png_path);
    EXPECT_FALSE(img.empty());
    EXPECT_EQ(img.cols, 64);
    EXPECT_EQ(img.rows, 48);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_EQ(img.type(), CV_8UC3);
}

TEST_F(ImageLoaderTest, ValidJpgLoading) {
    const cv::Mat img = image::ImageLoader::load(jpg_path);
    EXPECT_FALSE(img.empty());
    EXPECT_EQ(img.cols, 64);
    EXPECT_EQ(img.rows, 48);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_EQ(img.type(), CV_8UC3);
}

TEST_F(ImageLoaderTest, ValidBmpLoading) {
    const cv::Mat img = image::ImageLoader::load(bmp_path);
    EXPECT_FALSE(img.empty());
    EXPECT_EQ(img.cols, 64);
    EXPECT_EQ(img.rows, 48);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_EQ(img.type(), CV_8UC3);
}

TEST_F(ImageLoaderTest, InvalidFilePath) {
    const std::filesystem::path nonexistent = test_dir / "does_not_exist.png";
    EXPECT_THROW({ auto _ = image::ImageLoader::load(nonexistent); (void)_; }, image::FileNotFoundException);
}

TEST_F(ImageLoaderTest, UnsupportedFormat) {
    const std::filesystem::path unsupported = test_dir / "test.xyz";
    {
        std::ofstream file(unsupported);
        file << "dummy";
    }
    EXPECT_THROW({ auto _ = image::ImageLoader::load(unsupported); (void)_; }, image::UnsupportedFormatException);
}

TEST_F(ImageLoaderTest, CorruptedImageHandling) {
    EXPECT_THROW({ auto _ = image::ImageLoader::load(corrupted_path); (void)_; }, image::CorruptedImageException);
}

TEST_F(ImageLoaderTest, EmptyImageHandling) {
    EXPECT_THROW({ auto _ = image::ImageLoader::load(empty_path); (void)_; }, image::EmptyImageException);
}

TEST_F(ImageLoaderTest, NormalizationAndDenormalization) {
    const cv::Mat img = image::ImageLoader::load(png_path);
    const cv::Mat normalized = image::ImageLoader::normalize_to_float32(img);

    EXPECT_EQ(normalized.type(), CV_32FC3);
    EXPECT_EQ(normalized.cols, img.cols);
    EXPECT_EQ(normalized.rows, img.rows);

    // Verify bounds in [0.0, 1.0]
    double min_val = 0.0, max_val = 0.0;
    cv::minMaxLoc(normalized.reshape(1), &min_val, &max_val);
    EXPECT_GE(min_val, 0.0);
    EXPECT_LE(max_val, 1.0);

    // Denormalize back to CV_8UC3
    const cv::Mat denormalized = image::ImageLoader::denormalize_to_uint8(normalized);
    EXPECT_EQ(denormalized.type(), CV_8UC3);
    EXPECT_EQ(cv::norm(img, denormalized, cv::NORM_INF), 0.0);
}

TEST_F(ImageLoaderTest, FormatInspection) {
    EXPECT_TRUE(image::ImageLoader::is_supported_format("image.png"));
    EXPECT_TRUE(image::ImageLoader::is_supported_format("IMAGE.PNG"));
    EXPECT_TRUE(image::ImageLoader::is_supported_format("photo.jpg"));
    EXPECT_TRUE(image::ImageLoader::is_supported_format("photo.jpeg"));
    EXPECT_TRUE(image::ImageLoader::is_supported_format("bitmap.bmp"));
    EXPECT_FALSE(image::ImageLoader::is_supported_format("vector.svg"));
    EXPECT_FALSE(image::ImageLoader::is_supported_format("raw.tiff"));
}

}  // namespace mirage::tests
