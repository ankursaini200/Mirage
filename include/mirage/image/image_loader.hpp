#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include <opencv2/core.hpp>

namespace mirage::image {

/**
 * @brief Base exception for all image loading and persistence errors.
 */
class ImageLoadException : public std::runtime_error {
public:
    explicit ImageLoadException(const std::string& message)
        : std::runtime_error(message) {}
};

class FileNotFoundException : public ImageLoadException {
public:
    explicit FileNotFoundException(const std::filesystem::path& path)
        : ImageLoadException("File does not exist: " + path.string()) {}
};

class UnsupportedFormatException : public ImageLoadException {
public:
    explicit UnsupportedFormatException(const std::filesystem::path& path)
        : ImageLoadException("Unsupported image format: " + path.string() +
                             " (supported: JPG, JPEG, PNG, BMP)") {}
};

class CorruptedImageException : public ImageLoadException {
public:
    explicit CorruptedImageException(const std::filesystem::path& path, const std::string& details)
        : ImageLoadException("Corrupted or unreadable image file: " + path.string() + " (" + details + ")") {}
};

class EmptyImageException : public ImageLoadException {
public:
    explicit EmptyImageException(const std::filesystem::path& path)
        : ImageLoadException("Image file contains zero dimensions or empty payload: " + path.string()) {}
};

/**
 * @brief Reliable, validating loader and normalizer for image files.
 *
 * Enforces:
 * - File existence and non-zero byte size.
 * - Format validation (.jpg, .jpeg, .png, .bmp).
 * - Exact dimension and channel preservation (CV_8UC3 in BGR ordering).
 * - Normalization to floating-point [0.0, 1.0] range (CV_32FC3).
 */
class ImageLoader {
public:
    /**
     * @brief Checks whether the given filepath has a supported extension (JPG, JPEG, PNG, BMP).
     */
    [[nodiscard]] static bool is_supported_format(const std::filesystem::path& filepath) noexcept;

    /**
     * @brief Loads a color image from disk into a CV_8UC3 BGR matrix.
     *
     * @param filepath Path to the image file.
     * @return cv::Mat Validated 8-bit unsigned 3-channel color image (BGR).
     * @throws FileNotFoundException if file does not exist.
     * @throws UnsupportedFormatException if extension is not supported.
     * @throws EmptyImageException if file is 0 bytes or loaded dimensions are <= 0.
     * @throws CorruptedImageException if decoding fails.
     */
    [[nodiscard]] static cv::Mat load(const std::filesystem::path& filepath);

    /**
     * @brief Saves an image to disk, validating format and matrix validity.
     */
    static void save(const std::filesystem::path& filepath, const cv::Mat& image);

    /**
     * @brief Normalizes a CV_8UC3 image in [0, 255] to CV_32FC3 in [0.0, 1.0].
     * Avoids unnecessary copy if already CV_32FC3.
     */
    [[nodiscard]] static cv::Mat normalize_to_float32(const cv::Mat& image);

    /**
     * @brief Denormalizes a CV_32FC3 image in [0.0, 1.0] to CV_8UC3 in [0, 255].
     */
    [[nodiscard]] static cv::Mat denormalize_to_uint8(const cv::Mat& image);

    /**
     * @brief Returns a human-readable description of an OpenCV matrix type.
     */
    [[nodiscard]] static std::string get_type_description(int type);
};

}  // namespace mirage::image

