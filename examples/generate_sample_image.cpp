#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char* argv[]) {
    const std::string filename = (argc > 1) ? argv[1] : "sample_image.png";
    const int width = (argc > 2) ? std::stoi(argv[2]) : 190;
    const int height = (argc > 3) ? std::stoi(argv[3]) : 130;

    cv::Mat img(height, width, CV_8UC3);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            img.at<cv::Vec3b>(r, c) = cv::Vec3b(
                static_cast<uint8_t>((c * 255) / width),
                static_cast<uint8_t>((r * 255) / height),
                static_cast<uint8_t>(((c + r) * 255) / (width + height))
            );
        }
    }

    if (cv::imwrite(filename, img)) {
        std::cout << "Generated " << filename << " (" << width << "x" << height << ")\n";
        return 0;
    } else {
        std::cerr << "Failed to write " << filename << "\n";
        return 1;
    }
}

