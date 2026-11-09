#include "mirage/image/color_converter.hpp"

#include <opencv2/imgproc.hpp>

namespace mirage::image {

void ColorConverter::validate_image_input(const cv::Mat& mat, const std::string& operation_name) {
    if (mat.empty()) {
        throw ColorConversionException(operation_name + ": Input matrix is empty.");
    }
    if (mat.channels() != 3) {
        throw ColorConversionException(operation_name + ": Expected 3 channels, but input has " +
                                       std::to_string(mat.channels()) + " channels.");
    }
    if (mat.rows <= 0 || mat.cols <= 0) {
        throw ColorConversionException(operation_name + ": Matrix dimensions must be positive.");
    }
}

cv::Mat ColorConverter::bgr_to_rgb(const cv::Mat& bgr) {
    validate_image_input(bgr, "bgr_to_rgb");
    if (bgr.type() != CV_8UC3 && bgr.type() != CV_32FC3) {
        throw ColorConversionException("bgr_to_rgb: Input type must be CV_8UC3 or CV_32FC3.");
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

cv::Mat ColorConverter::rgb_to_bgr(const cv::Mat& rgb) {
    validate_image_input(rgb, "rgb_to_bgr");
    if (rgb.type() != CV_8UC3 && rgb.type() != CV_32FC3) {
        throw ColorConversionException("rgb_to_bgr: Input type must be CV_8UC3 or CV_32FC3.");
    }

    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

cv::Mat ColorConverter::bgr_to_lab(const cv::Mat& bgr_normalized) {
    validate_image_input(bgr_normalized, "bgr_to_lab");
    if (bgr_normalized.type() != CV_32FC3) {
        throw ColorConversionException(
            "bgr_to_lab: Scientific CIELAB requires normalized floating-point input (CV_32FC3, [0.0, 1.0]).");
    }

    cv::Mat lab;
    cv::cvtColor(bgr_normalized, lab, cv::COLOR_BGR2Lab);
    return lab;
}

cv::Mat ColorConverter::lab_to_bgr(const cv::Mat& lab_float) {
    validate_image_input(lab_float, "lab_to_bgr");
    if (lab_float.type() != CV_32FC3) {
        throw ColorConversionException(
            "lab_to_bgr: Scientific CIELAB requires floating-point input (CV_32FC3, L* in [0, 100], a*, b* in [-127, 127]).");
    }

    cv::Mat bgr;
    cv::cvtColor(lab_float, bgr, cv::COLOR_Lab2BGR);
    return bgr;
}

cv::Mat ColorConverter::bgr8_to_lab8(const cv::Mat& bgr8) {
    validate_image_input(bgr8, "bgr8_to_lab8");
    if (bgr8.type() != CV_8UC3) {
        throw ColorConversionException("bgr8_to_lab8: Input must be CV_8UC3.");
    }

    cv::Mat lab;
    cv::cvtColor(bgr8, lab, cv::COLOR_BGR2Lab);
    return lab;
}

cv::Mat ColorConverter::lab8_to_bgr8(const cv::Mat& lab8) {
    validate_image_input(lab8, "lab8_to_bgr8");
    if (lab8.type() != CV_8UC3) {
        throw ColorConversionException("lab8_to_bgr8: Input must be CV_8UC3.");
    }

    cv::Mat bgr;
    cv::cvtColor(lab8, bgr, cv::COLOR_Lab2BGR);
    return bgr;
}

}  // namespace mirage::image

