#pragma once

#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

namespace mirage::image {

/**
 * @brief Exception thrown when color space conversion encounters invalid inputs or unsupported types.
 */
class ColorConversionException : public std::runtime_error {
public:
    explicit ColorConversionException(const std::string& message)
        : std::runtime_error("ColorConversionException: " + message) {}
};

/**
 * @brief Color space transformation engine for Mirage.
 *
 * Supported Color Space Representations:
 *
 * 1. BGR (8-bit integer, CV_8UC3):
 *    - Default OpenCV format for image I/O.
 *    - Channels: Blue, Green, Red.
 *    - Range: [0, 255] per channel.
 *
 * 2. RGB (8-bit integer, CV_8UC3) or (32-bit float, CV_32FC3):
 *    - Standard color order.
 *    - Channels: Red, Green, Blue.
 *    - Range: [0, 255] (CV_8UC3) or [0.0, 1.0] (CV_32FC3).
 *
 * 3. Scientific Floating-Point CIELAB (CV_32FC3):
 *    - Perceptually uniform color space used in the Mirage research pipeline.
 *    - Converted from normalized floating-point BGR ([0.0, 1.0]).
 *    - Channels:
 *        * L* (Lightness): [0.0, 100.0] (0 = black, 100 = diffuse white).
 *        * a* (Green-to-Red chromaticity): [-127.0, +127.0] (negative = green, positive = red).
 *        * b* (Blue-to-Yellow chromaticity): [-127.0, +127.0] (negative = blue, positive = yellow).
 *
 * 4. 8-Bit Quantized Lab (CV_8UC3):
 *    - Scaled and shifted representation used by legacy/display codecs:
 *        * L -> L * 255 / 100 (in [0, 255])
 *        * a -> a + 128 (in [0, 255])
 *        * b -> b + 128 (in [0, 255])
 *    - Caution: Quantization eliminates fractional precision and introduces rounding error.
 */
class ColorConverter {
public:
    /**
     * @brief Converts BGR image to RGB format.
     * Supports both CV_8UC3 and CV_32FC3 matrices.
     */
    [[nodiscard]] static cv::Mat bgr_to_rgb(const cv::Mat& bgr);

    /**
     * @brief Converts RGB image to BGR format.
     * Supports both CV_8UC3 and CV_32FC3 matrices.
     */
    [[nodiscard]] static cv::Mat rgb_to_bgr(const cv::Mat& rgb);

    /**
     * @brief Converts normalized floating-point BGR (CV_32FC3, [0.0, 1.0]) to scientific CIELAB (CV_32FC3).
     *
     * Output ranges:
     * - Channel 0 (L*): [0.0, 100.0]
     * - Channel 1 (a*): [-127.0, 127.0]
     * - Channel 2 (b*): [-127.0, 127.0]
     *
     * @throws ColorConversionException if input is empty, not 3 channels, or not CV_32FC3.
     */
    [[nodiscard]] static cv::Mat bgr_to_lab(const cv::Mat& bgr_normalized);

    /**
     * @brief Converts scientific CIELAB (CV_32FC3, L* in [0, 100], a*, b* in [-127, 127]) to normalized BGR (CV_32FC3, [0.0, 1.0]).
     *
     * @throws ColorConversionException if input is empty, not 3 channels, or not CV_32FC3.
     */
    [[nodiscard]] static cv::Mat lab_to_bgr(const cv::Mat& lab_float);

    /**
     * @brief Converts 8-bit integer BGR (CV_8UC3) to 8-bit quantized Lab (CV_8UC3).
     * L in [0, 255], a in [0, 255], b in [0, 255].
     */
    [[nodiscard]] static cv::Mat bgr8_to_lab8(const cv::Mat& bgr8);

    /**
     * @brief Converts 8-bit quantized Lab (CV_8UC3) to 8-bit integer BGR (CV_8UC3).
     */
    [[nodiscard]] static cv::Mat lab8_to_bgr8(const cv::Mat& lab8);

private:
    static void validate_image_input(const cv::Mat& mat, const std::string& operation_name);
};

}  // namespace mirage::image

