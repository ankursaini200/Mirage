#include "mirage/image/image_loader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

namespace mirage::image {

namespace {

std::string to_lower_extension(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

}  // namespace

bool ImageLoader::is_supported_format(const std::filesystem::path& filepath) noexcept {
    const std::string ext = to_lower_extension(filepath);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

cv::Mat ImageLoader::load(const std::filesystem::path& filepath) {
    if (!std::filesystem::exists(filepath)) {
        throw FileNotFoundException(filepath);
    }

    if (!is_supported_format(filepath)) {
        throw UnsupportedFormatException(filepath);
    }

    std::error_code ec;
    const auto size = std::filesystem::file_size(filepath, ec);
    if (ec || size == 0) {
        throw EmptyImageException(filepath);
    }

    cv::Mat image = cv::imread(filepath.string(), cv::IMREAD_COLOR);

    if (image.empty()) {
        throw CorruptedImageException(filepath, "Decoding failed; invalid or corrupted image data");
    }

    if (image.rows <= 0 || image.cols <= 0) {
        throw EmptyImageException(filepath);
    }

    if (image.channels() != 3 || image.type() != CV_8UC3) {
        throw CorruptedImageException(filepath, "Unexpected channel count or bit depth");
    }

    return image;
}

void ImageLoader::save(const std::filesystem::path& filepath, const cv::Mat& image) {
    if (image.empty()) {
        throw ImageLoadException("Cannot save an empty image to " + filepath.string());
    }

    if (!is_supported_format(filepath)) {
        throw UnsupportedFormatException(filepath);
    }

    // Ensure parent directory exists
    if (filepath.has_parent_path()) {
        std::filesystem::create_directories(filepath.parent_path());
    }

    const bool success = cv::imwrite(filepath.string(), image);
    if (!success) {
        throw ImageLoadException("OpenCV failed to encode and write image to: " + filepath.string());
    }
}

cv::Mat ImageLoader::normalize_to_float32(const cv::Mat& image) {
    if (image.empty()) {
        throw ImageLoadException("Cannot normalize an empty image matrix");
    }

    if (image.type() == CV_32FC3) {
        return image;
    }

    if (image.type() != CV_8UC3) {
        throw ImageLoadException("Expected CV_8UC3 format for normalization, got " +
                                 get_type_description(image.type()));
    }

    cv::Mat normalized;
    image.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
    return normalized;
}

cv::Mat ImageLoader::denormalize_to_uint8(const cv::Mat& image) {
    if (image.empty()) {
        throw ImageLoadException("Cannot denormalize an empty image matrix");
    }

    if (image.type() == CV_8UC3) {
        return image;
    }

    if (image.type() != CV_32FC3) {
        throw ImageLoadException("Expected CV_32FC3 format for denormalization, got " +
                                 get_type_description(image.type()));
    }

    cv::Mat denormalized;
    image.convertTo(denormalized, CV_8UC3, 255.0);
    return denormalized;
}

std::string ImageLoader::get_type_description(int type) {
    switch (type) {
        case CV_8UC1:  return "CV_8UC1 (8-bit unsigned grayscale)";
        case CV_8UC3:  return "CV_8UC3 (8-bit unsigned 3-channel, BGR)";
        case CV_8UC4:  return "CV_8UC4 (8-bit unsigned 4-channel, BGRA)";
        case CV_32FC1: return "CV_32FC1 (32-bit float grayscale)";
        case CV_32FC3: return "CV_32FC3 (32-bit float 3-channel, BGR/RGB/Lab)";
        case CV_32FC4: return "CV_32FC4 (32-bit float 4-channel)";
        default:       return "Type ID " + std::to_string(type);
    }
}

}  // namespace mirage::image

