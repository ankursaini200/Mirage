#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace {

constexpr int kSize = 512;
constexpr double kPi = std::numbers::pi;

cv::Mat generate_smooth_gradient() {
    cv::Mat img(kSize, kSize, CV_8UC3);
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            const double nx = static_cast<double>(x) / (kSize - 1);
            const double ny = static_cast<double>(y) / (kSize - 1);

            const double b = 0.5 + 0.5 * std::sin(kPi * nx);
            const double g = 0.5 + 0.5 * std::cos(kPi * ny);
            const double r = (nx + ny) * 0.5;

            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uint8_t>(std::clamp(b * 255.0, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(g * 255.0, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(r * 255.0, 0.0, 255.0))
            );
        }
    }
    return img;
}

cv::Mat generate_high_contrast_edges() {
    cv::Mat img(kSize, kSize, CV_8UC3, cv::Scalar(30, 30, 30));

    // Checkerboard regions
    constexpr int check_sz = 32;
    for (int y = 0; y < kSize / 2; ++y) {
        for (int x = 0; x < kSize / 2; ++x) {
            if (((x / check_sz) + (y / check_sz)) % 2 == 0) {
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(240, 240, 240);
            }
        }
    }

    // Concentric sharp circles in bottom-right
    const cv::Point center(384, 384);
    for (int r = 10; r < 120; r += 16) {
        cv::circle(img, center, r, cv::Scalar(0, 255, 255), 4);
    }

    // High contrast diagonal step boundaries
    for (int y = kSize / 2; y < kSize; ++y) {
        for (int x = 0; x < kSize / 2; ++x) {
            if ((x + y) % 40 < 20) {
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(220, 30, 30);
            } else {
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(30, 220, 220);
            }
        }
    }

    // Horizontal & vertical sharp stripes in top-right
    for (int y = 0; y < kSize / 2; ++y) {
        for (int x = kSize / 2; x < kSize; ++x) {
            if ((x / 16) % 2 == 0) {
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(240, 180, 20);
            } else {
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(20, 50, 180);
            }
        }
    }

    return img;
}

cv::Mat generate_detailed_texture() {
    cv::Mat img(kSize, kSize, CV_8UC3);
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            const double fx = static_cast<double>(x);
            const double fy = static_cast<double>(y);

            const double w1 = std::sin(0.15 * fx) * std::cos(0.15 * fy);
            const double w2 = std::sin(0.40 * fx + 0.30 * fy);
            const double w3 = std::cos(0.80 * fx - 0.20 * fy);
            const double noise = std::sin(fx * fy * 0.05);

            const double val = 0.5 + 0.25 * w1 + 0.15 * w2 + 0.08 * w3 + 0.02 * noise;

            const double b = val * 0.9;
            const double g = val * 1.1;
            const double r = val * 0.8;

            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uint8_t>(std::clamp(b * 255.0, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(g * 255.0, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(r * 255.0, 0.0, 255.0))
            );
        }
    }
    return img;
}

cv::Mat generate_natural_scene() {
    cv::Mat img(kSize, kSize, CV_8UC3);

    // Sky with gentle atmospheric perspective
    for (int y = 0; y < kSize; ++y) {
        const double ny = static_cast<double>(y) / kSize;
        for (int x = 0; x < kSize; ++x) {
            const double nx = static_cast<double>(x) / kSize;

            if (y < 220) {
                // Sky gradient: deep blue to haze
                const double sky_factor = static_cast<double>(y) / 220.0;
                const double b = 0.85 - 0.20 * sky_factor;
                const double g = 0.60 - 0.15 * sky_factor;
                const double r = 0.35 + 0.15 * sky_factor;
                // Add soft clouds
                const double cloud = 0.15 * (1.0 + std::sin(nx * 8.0 + std::cos(ny * 12.0)));
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uint8_t>(std::clamp((b + cloud * 0.3) * 255.0, 0.0, 255.0)),
                    static_cast<uint8_t>(std::clamp((g + cloud * 0.3) * 255.0, 0.0, 255.0)),
                    static_cast<uint8_t>(std::clamp((r + cloud * 0.4) * 255.0, 0.0, 255.0))
                );
            } else if (y < 340) {
                // Mountain range with ridgeline
                const double ridge = 220.0 + 40.0 * std::sin(nx * 7.0) + 20.0 * std::cos(nx * 15.0);
                if (y < ridge) {
                    const double b = 0.65;
                    const double g = 0.45;
                    const double r = 0.50;
                    img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                        static_cast<uint8_t>(b * 255.0),
                        static_cast<uint8_t>(g * 255.0),
                        static_cast<uint8_t>(r * 255.0)
                    );
                } else {
                    // Mountain body
                    const double m_factor = (y - ridge) / (340.0 - ridge + 1.0);
                    const double b = 0.30 + 0.10 * m_factor;
                    const double g = 0.35 + 0.10 * m_factor;
                    const double r = 0.25 + 0.05 * m_factor;
                    img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                        static_cast<uint8_t>(b * 255.0),
                        static_cast<uint8_t>(g * 255.0),
                        static_cast<uint8_t>(r * 255.0)
                    );
                }
            } else {
                // Foreground grass/foliage texture
                const double f_factor = static_cast<double>(y - 340) / (kSize - 340);
                const double detail = 0.08 * std::sin(x * 0.3) * std::cos(y * 0.3);
                const double b = 0.15 + detail;
                const double g = 0.45 + 0.20 * f_factor + detail;
                const double r = 0.20 + 0.10 * f_factor + detail;
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uint8_t>(std::clamp(b * 255.0, 0.0, 255.0)),
                    static_cast<uint8_t>(std::clamp(g * 255.0, 0.0, 255.0)),
                    static_cast<uint8_t>(std::clamp(r * 255.0, 0.0, 255.0))
                );
            }
        }
    }
    return img;
}

cv::Mat generate_color_rich() {
    cv::Mat hsv(kSize, kSize, CV_8UC3);
    const cv::Point2d center(kSize * 0.5, kSize * 0.5);
    const double max_rad = kSize * 0.45;

    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            const double dx = x - center.x;
            const double dy = y - center.y;
            const double dist = std::hypot(dx, dy);

            double angle = std::atan2(dy, dx);
            if (angle < 0.0) {
                angle += 2.0 * kPi;
            }
            // OpenCV Hue is [0, 180)
            const uint8_t h = static_cast<uint8_t>((angle / (2.0 * kPi)) * 180.0);
            const uint8_t s = static_cast<uint8_t>(std::clamp((dist / max_rad) * 255.0, 40.0, 255.0));
            const uint8_t v = 230;

            hsv.at<cv::Vec3b>(y, x) = cv::Vec3b(h, s, v);
        }
    }

    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}

cv::Mat generate_low_contrast() {
    cv::Mat img(kSize, kSize, CV_8UC3);
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            const double nx = static_cast<double>(x) / kSize;
            const double ny = static_cast<double>(y) / kSize;

            // Values compressed into narrow range [0.38, 0.52] (L* ~ 40 to 55)
            const double base = 0.45 + 0.05 * std::sin(nx * kPi * 2.0) * std::cos(ny * kPi * 2.0);
            const double b = base + 0.02 * std::sin(nx * 10.0);
            const double g = base;
            const double r = base - 0.02 * std::cos(ny * 10.0);

            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uint8_t>(std::clamp(b * 255.0, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(g * 255.0, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(r * 255.0, 0.0, 255.0))
            );
        }
    }
    return img;
}

cv::Mat generate_grayscale_portrait() {
    cv::Mat img(kSize, kSize, CV_8UC3);
    const cv::Point2d center(kSize * 0.5, kSize * 0.48);

    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            const double dx = (x - center.x) / (kSize * 0.35);
            const double dy = (y - center.y) / (kSize * 0.45);
            const double r2 = dx * dx + dy * dy;

            // Oval face luminance profile
            double lum = 0.15; // background
            if (r2 < 1.0) {
                // Smooth falloff from center highlight
                const double face = std::exp(-1.8 * r2);
                // Subtle feature modulations
                const double eye_l = std::exp(-30.0 * (std::pow(dx + 0.35, 2.0) + std::pow(dy + 0.15, 2.0)));
                const double eye_r = std::exp(-30.0 * (std::pow(dx - 0.35, 2.0) + std::pow(dy + 0.15, 2.0)));
                const double nose = 0.15 * std::exp(-15.0 * (std::pow(dx, 2.0) + std::pow(dy, 2.0)));
                const double mouth = 0.20 * std::exp(-20.0 * (std::pow(dx * 1.5, 2.0) + std::pow(dy - 0.4, 2.0)));

                lum = 0.20 + 0.65 * face - 0.25 * eye_l - 0.25 * eye_r + nose - mouth;
            }

            const uint8_t val = static_cast<uint8_t>(std::clamp(lum * 255.0, 0.0, 255.0));
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(val, val, val);
        }
    }
    return img;
}

struct ModalityEntry {
    std::string id;
    std::string filename;
    std::string modality;
    std::string description;
    cv::Mat image;
};

} // namespace

int main(int argc, char* argv[]) {
    const fs::path output_dir = (argc > 1) ? fs::path(argv[1]) : fs::path("data/experiments");
    fs::create_directories(output_dir);

    std::vector<ModalityEntry> modalities = {
        {"mod_01", "smooth_gradient.png", "Smooth Gradient",
         "Continuous sinusoidal color gradient testing false contouring and banding.",
         generate_smooth_gradient()},
        {"mod_02", "high_contrast_edges.png", "High-Contrast Edges",
         "Sharp geometric step boundaries, checkerboards, and concentric circles.",
         generate_high_contrast_edges()},
        {"mod_03", "detailed_texture.png", "Detailed Texture",
         "High-frequency multi-harmonic sinusoidal waves and subtle micro-textures.",
         generate_detailed_texture()},
        {"mod_04", "natural_scene.png", "Natural Scene",
         "Simulated landscape containing atmospheric sky, mountain ridge, and foliage.",
         generate_natural_scene()},
        {"mod_05", "color_rich.png", "Color Rich",
         "Full-gamut chromatic color wheel spanning full HSV/Lab chrominance.",
         generate_color_rich()},
        {"mod_06", "low_contrast.png", "Low Contrast",
         "Narrow dynamic range foggy scene with subtle luminance and chromatic variation.",
         generate_low_contrast()},
        {"mod_07", "grayscale_portrait.png", "Grayscale Portrait",
         "Monochromatic luminance profile with smooth face contours and facial shading.",
         generate_grayscale_portrait()}
    };

    const fs::path manifest_path = output_dir / "dataset_manifest.csv";
    std::ofstream manifest(manifest_path);
    if (!manifest.is_open()) {
        std::cerr << "Failed to open manifest file: " << manifest_path << "\n";
        return 1;
    }

    manifest << "id,filename,width,height,channels,modality,description\n";

    for (const auto& m : modalities) {
        const fs::path img_path = output_dir / m.filename;
        if (!cv::imwrite(img_path.string(), m.image)) {
            std::cerr << "Failed to write image: " << img_path << "\n";
            return 1;
        }
        std::cout << "Generated: " << img_path.string() << " (" << m.modality << ")\n";
        manifest << m.id << "," << m.filename << "," << m.image.cols << "," << m.image.rows << ","
                 << m.image.channels() << ",\"" << m.modality << "\",\"" << m.description << "\"\n";
    }

    std::cout << "Wrote dataset manifest to " << manifest_path.string() << "\n";
    return 0;
}

