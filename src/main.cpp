#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "mirage/analysis/sensitivity_analyzer.hpp"
#include "mirage/cpu/cpu_features.hpp"
#include "mirage/enhancement/enhancement_engine.hpp"
#include "mirage/enhancement/precision_config.hpp"
#include "mirage/image/image_loader.hpp"
#include "mirage/image/tile_manager.hpp"
#include "mirage/version.hpp"

namespace {

void print_help(std::string_view program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n"
              << "Mirage: " << mirage::PROJECT_DESCRIPTION << "\n\n"
              << "General Options:\n"
              << "  -v, --version                       Display Mirage version information\n"
              << "  -i, --info                          Display project, compiler, and hardware details\n"
              << "  -h, --help                          Display this help message\n\n"
              << "Image Processing & Tiling Options:\n"
              << "  --image-info <image_path>           Inspect and display image metadata\n"
              << "  --tile-info <image_path> --tile-size <size>\n"
              << "                                      Inspect tile partitioning and boundary tiles\n"
              << "  --visualize-tiles <image_path> --tile-size <size> [--output <path>]\n"
              << "                                      Render and save tile grid visualization\n\n"
              << "Enhancement Options:\n"
              << "  --enhance <input_path> --output <output_path>\n"
              << "                                      Run image enhancement pipeline\n"
              << "    [--precision <fp32|fp16|mixed>]   Numerical precision policy (default: fp32)\n"
              << "    [--brightness <val>]              Additive brightness in [-1.0, 1.0] (default: 0.0)\n"
              << "    [--contrast <val>]                Contrast factor >= 0.0 (default: 1.0)\n"
              << "    [--saturation <val>]              Saturation factor >= 0.0 (default: 1.0)\n"
              << "    [--gamma <val>]                   Gamma exponent > 0.0 (default: 1.0)\n\n"
              << "Sensitivity Analysis Options:\n"
              << "  --analyze-sensitivity <image_path> --tile-size <size>\n"
              << "                                      Analyze tile-based image sensitivity\n"
              << "    [--output <heatmap_path>]         Render and save sensitivity heatmap\n"
              << "    [--csv <csv_path>]                Export individual tile metrics to CSV\n";
}

void print_version() {
    std::cout << mirage::PROJECT_NAME << " version " << mirage::get_version_string() << "\n";
}

void print_info() {
    const auto& cpu = mirage::cpu::CpuFeatures::detect();

    std::cout << "Project name:              " << mirage::PROJECT_NAME << "\n"
              << "Project version:           " << mirage::get_version_string() << "\n"
              << "C++ standard:              " << mirage::get_cpp_standard_string() << "\n"
              << "Compiler information:      " << mirage::get_compiler_info_string() << "\n"
              << "Detected CPU architecture: " << cpu.architecture_name() << "\n"
              << "AVX2 availability:         " << cpu.avx2_status_string() << "\n";
}

int handle_image_info(const std::filesystem::path& image_path) {
    try {
        const cv::Mat img = mirage::image::ImageLoader::load(image_path);
        const cv::Mat norm = mirage::image::ImageLoader::normalize_to_float32(img);

        std::cout << "Image filename:        " << image_path.filename().string() << "\n"
                  << "Image width:           " << img.cols << "\n"
                  << "Image height:          " << img.rows << "\n"
                  << "Number of channels:    " << img.channels() << "\n"
                  << "Original data type:    " << mirage::image::ImageLoader::get_type_description(img.type()) << "\n"
                  << "Normalized data type:  " << mirage::image::ImageLoader::get_type_description(norm.type()) << "\n"
                  << "Color representation:  BGR (OpenCV standard 3-channel)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error inspecting image: " << e.what() << "\n";
        return 1;
    }
}

int parse_tile_size(const std::string& size_str) {
    try {
        const size_t x_pos = size_str.find('x');
        if (x_pos != std::string::npos) {
            return std::stoi(size_str.substr(0, x_pos));
        }
        return std::stoi(size_str);
    } catch (...) {
        return -1;
    }
}

int handle_tile_info(const std::filesystem::path& image_path, int tile_size) {
    if (tile_size <= 0) {
        std::cerr << "Error: Tile size must be a positive integer.\n";
        return 1;
    }

    try {
        const cv::Mat img = mirage::image::ImageLoader::load(image_path);
        const mirage::image::TileManager mgr(img, tile_size, tile_size);

        std::cout << "Image dimensions:            " << mgr.image_width() << " x " << mgr.image_height() << "\n"
                  << "Tile dimensions:             " << mgr.nominal_tile_width() << " x " << mgr.nominal_tile_height() << "\n"
                  << "Grid dimensions:             " << mgr.grid_cols() << " columns x " << mgr.grid_rows() << " rows\n"
                  << "Total tile count:            " << mgr.total_tiles() << "\n"
                  << "Number of complete tiles:    " << mgr.complete_tiles() << "\n"
                  << "Number of partial boundary:  " << mgr.partial_tiles() << "\n\n"
                  << "Individual Tile Coordinates:\n"
                  << std::left
                  << std::setw(6)  << "ID"
                  << std::setw(8)  << "X"
                  << std::setw(8)  << "Y"
                  << std::setw(8)  << "Width"
                  << std::setw(8)  << "Height"
                  << std::setw(10) << "Status" << "\n"
                  << std::string(48, '-') << "\n";

        for (const auto& t : mgr.tiles()) {
            std::cout << std::left
                      << std::setw(6)  << t.id
                      << std::setw(8)  << t.x
                      << std::setw(8)  << t.y
                      << std::setw(8)  << t.width
                      << std::setw(8)  << t.height
                      << std::setw(10) << (t.is_partial ? "Partial" : "Complete")
                      << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error calculating tile info: " << e.what() << "\n";
        return 1;
    }
}

int handle_visualize_tiles(const std::filesystem::path& image_path,
                           int tile_size,
                           const std::filesystem::path& output_path) {
    if (tile_size <= 0) {
        std::cerr << "Error: Tile size must be a positive integer.\n";
        return 1;
    }

    try {
        const cv::Mat img = mirage::image::ImageLoader::load(image_path);
        const mirage::image::TileManager mgr(img, tile_size, tile_size);

        std::filesystem::path dest = output_path;
        if (dest.empty()) {
            dest = "output/tile_visualization_" + std::to_string(tile_size) + "x" +
                   std::to_string(tile_size) + ".png";
        }

        mgr.save_tile_visualization(dest);

        std::cout << "Tile visualization saved successfully:\n"
                  << "  Destination:  " << dest.string() << "\n"
                  << "  Tile size:    " << tile_size << " x " << tile_size << "\n"
                  << "  Total tiles:  " << mgr.total_tiles() << " ("
                  << mgr.complete_tiles() << " complete, "
                  << mgr.partial_tiles() << " partial)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error visualizing tiles: " << e.what() << "\n";
        return 1;
    }
}

int handle_analyze_sensitivity(const std::filesystem::path& image_path,
                               int tile_size,
                               const std::filesystem::path& output_path,
                               const std::filesystem::path& csv_path) {
    if (tile_size <= 0) {
        std::cerr << "Error: Tile size must be a positive integer.\n";
        return 1;
    }

    try {
        const cv::Mat img = mirage::image::ImageLoader::load(image_path);
        const cv::Mat norm_img = mirage::image::ImageLoader::normalize_to_float32(img);

        mirage::analysis::SensitivityAnalyzer analyzer{};
        const auto report = analyzer.analyze(norm_img, tile_size, tile_size);

        std::cout << "============================================================\n"
                  << "           Mirage Tile-Based Sensitivity Analysis           \n"
                  << "============================================================\n"
                  << "Image filename:          " << image_path.filename().string() << "\n"
                  << "Image dimensions:        " << report.image_width << " x " << report.image_height << " (3 channels)\n"
                  << "Tile dimensions:         " << report.nominal_tile_width << " x " << report.nominal_tile_height << "\n"
                  << "Grid dimensions:         " << report.grid_cols << " columns x " << report.grid_rows << " rows\n"
                  << "Total tile count:        " << report.total_tiles << " ("
                  << report.complete_tiles << " complete, "
                  << report.partial_tiles << " partial)\n"
                  << "Execution time:          " << std::fixed << std::setprecision(2) << report.execution_time_ms << " ms\n"
                  << "------------------------------------------------------------\n"
                  << "Sensitivity Summary (Scores in [0.0, 1.0]):\n"
                  << "  Minimum Sensitivity:   " << std::setprecision(6) << report.min_sensitivity << "\n"
                  << "  Maximum Sensitivity:   " << report.max_sensitivity << "\n"
                  << "  Average Sensitivity:   " << report.avg_sensitivity << "\n"
                  << "------------------------------------------------------------\n"
                  << "Feature Averages (Weights: C=0.25, G=0.25, E=0.25, L=0.25):\n"
                  << "  Color Variation (C):   " << report.avg_color_variation << "\n"
                  << "  Gradient Magnitude (G):" << report.avg_gradient_magnitude << "\n"
                  << "  Edge Density (E):      " << report.avg_edge_density << "\n"
                  << "  Local Contrast (L):    " << report.avg_local_contrast << "\n"
                  << "============================================================\n";

        if (!output_path.empty()) {
            analyzer.save_heatmap(output_path, img, report);
            std::cout << "Heatmap visualization saved to: " << output_path.string() << "\n";
        }

        if (!csv_path.empty()) {
            analyzer.export_csv(csv_path, report);
            std::cout << "Tile sensitivity CSV report saved to: " << csv_path.string() << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error analyzing sensitivity: " << e.what() << "\n";
        return 1;
    }
}

int handle_enhance(const std::filesystem::path& input_path,
                   const std::filesystem::path& output_path,
                   const mirage::enhancement::EnhancementConfig& config,
                   mirage::enhancement::PrecisionMode mode,
                   const mirage::enhancement::MixedPrecisionPolicy& policy) {
    try {
        config.validate();

        const cv::Mat u8_in = mirage::image::ImageLoader::load(input_path);
        const cv::Mat f32_in = mirage::image::ImageLoader::normalize_to_float32(u8_in);

        // Compute output using selected precision mode
        const cv::Mat f32_out = mirage::enhancement::EnhancementEngine::process_with_precision(
            f32_in, config, mode, policy);
        const cv::Mat u8_out = mirage::image::ImageLoader::denormalize_to_uint8(f32_out);

        mirage::image::ImageLoader::save(output_path, u8_out);

        std::string mode_desc;
        if (mode == mirage::enhancement::PrecisionMode::FP32) {
            mode_desc = "FP32 (IEEE 754 binary32 reference)";
        } else if (mode == mirage::enhancement::PrecisionMode::FP16) {
            mode_desc = "FP16 (Emulated IEEE 754 binary16 arithmetic)";
        } else {
            mode_desc = "Mixed Precision (Brightness: " + std::string(to_string(policy.brightness)) +
                        ", Contrast: " + std::string(to_string(policy.contrast)) +
                        ", Saturation: " + std::string(to_string(policy.saturation)) +
                        ", Gamma: " + std::string(to_string(policy.gamma)) + ")";
        }

        std::cout << "Image enhancement completed successfully:\n"
                  << "  Input:               " << input_path.string() << "\n"
                  << "  Output:              " << output_path.string() << "\n"
                  << "  Dimensions:          " << u8_in.cols << " x " << u8_in.rows << "\n"
                  << "  Precision Mode:      " << mode_desc << "\n"
                  << "  Brightness:          " << (config.enable_brightness ? std::to_string(config.brightness) : "disabled") << "\n"
                  << "  Contrast:            " << (config.enable_contrast ? std::to_string(config.contrast) : "disabled") << "\n"
                  << "  Saturation:          " << (config.enable_saturation ? std::to_string(config.saturation) : "disabled") << "\n"
                  << "  Gamma:               " << (config.enable_gamma ? std::to_string(config.gamma) : "disabled") << "\n";

        // If running in FP16 or Mixed mode, compute quantitative error metrics against FP32 reference
        if (mode != mirage::enhancement::PrecisionMode::FP32) {
            const cv::Mat f32_ref = mirage::enhancement::EnhancementEngine::process(f32_in, config);
            const mirage::enhancement::ErrorMetrics metrics =
                mirage::enhancement::compute_error_metrics(f32_ref, f32_out);

            std::cout << "\nNumerical Error Analysis (vs FP32 Ground Truth Reference):\n"
                      << metrics.to_string() << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in image enhancement: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }

    const std::vector<std::string_view> args(argv + 1, argv + argc);

    if (args[0] == "--version" || args[0] == "-v") {
        print_version();
        return 0;
    }

    if (args[0] == "--info" || args[0] == "-i") {
        print_info();
        return 0;
    }

    if (args[0] == "--help" || args[0] == "-h") {
        print_help(argv[0]);
        return 0;
    }

    if (args[0] == "--image-info") {
        if (args.size() < 2) {
            std::cerr << "Error: Missing image path for --image-info\n";
            print_help(argv[0]);
            return 1;
        }
        return handle_image_info(args[1]);
    }

    if (args[0] == "--tile-info" || args[0] == "--visualize-tiles") {
        const bool visualize = (args[0] == "--visualize-tiles");
        std::filesystem::path image_path;
        int tile_size = -1;
        std::filesystem::path output_path;

        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--tile-size") {
                if (i + 1 < args.size()) {
                    tile_size = parse_tile_size(std::string(args[++i]));
                }
            } else if (args[i] == "--output" || args[i] == "-o") {
                if (i + 1 < args.size()) {
                    output_path = args[++i];
                }
            } else if (image_path.empty()) {
                image_path = args[i];
            }
        }

        if (image_path.empty()) {
            std::cerr << "Error: Missing image path.\n";
            print_help(argv[0]);
            return 1;
        }

        if (tile_size <= 0) {
            std::cerr << "Error: Please specify a positive tile size using --tile-size <size>\n";
            print_help(argv[0]);
            return 1;
        }

        if (visualize) {
            return handle_visualize_tiles(image_path, tile_size, output_path);
        } else {
            return handle_tile_info(image_path, tile_size);
        }
    }

    if (args[0] == "--analyze-sensitivity") {
        std::filesystem::path image_path;
        int tile_size = -1;
        std::filesystem::path output_path;
        std::filesystem::path csv_path;

        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--tile-size") {
                if (i + 1 < args.size()) {
                    tile_size = parse_tile_size(std::string(args[++i]));
                }
            } else if (args[i] == "--output" || args[i] == "-o") {
                if (i + 1 < args.size()) {
                    output_path = args[++i];
                }
            } else if (args[i] == "--csv") {
                if (i + 1 < args.size()) {
                    csv_path = args[++i];
                }
            } else if (image_path.empty()) {
                image_path = args[i];
            }
        }

        if (image_path.empty()) {
            std::cerr << "Error: Missing image path for --analyze-sensitivity\n";
            print_help(argv[0]);
            return 1;
        }

        if (tile_size <= 0) {
            std::cerr << "Error: Please specify a positive tile size using --tile-size <size>\n";
            print_help(argv[0]);
            return 1;
        }

        return handle_analyze_sensitivity(image_path, tile_size, output_path, csv_path);
    }

    if (args[0] == "--enhance") {
        std::filesystem::path input_path;
        std::filesystem::path output_path;
        mirage::enhancement::EnhancementConfig config{};
        mirage::enhancement::PrecisionMode precision_mode = mirage::enhancement::PrecisionMode::FP32;
        mirage::enhancement::MixedPrecisionPolicy mixed_policy{};

        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--output" || args[i] == "-o") {
                if (i + 1 < args.size()) {
                    output_path = args[++i];
                }
            } else if (args[i] == "--precision") {
                if (i + 1 < args.size()) {
                    try {
                        precision_mode = mirage::enhancement::parse_precision_mode(args[++i]);
                    } catch (const std::exception& e) {
                        std::cerr << "Error: " << e.what() << "\n";
                        return 1;
                    }
                }
            } else if (args[i] == "--brightness") {
                if (i + 1 < args.size()) {
                    try {
                        config.brightness = std::stof(std::string(args[++i]));
                        config.enable_brightness = true;
                    } catch (...) {
                        std::cerr << "Error: Invalid float value for --brightness\n";
                        return 1;
                    }
                }
            } else if (args[i] == "--contrast") {
                if (i + 1 < args.size()) {
                    try {
                        config.contrast = std::stof(std::string(args[++i]));
                        config.enable_contrast = true;
                    } catch (...) {
                        std::cerr << "Error: Invalid float value for --contrast\n";
                        return 1;
                    }
                }
            } else if (args[i] == "--saturation") {
                if (i + 1 < args.size()) {
                    try {
                        config.saturation = std::stof(std::string(args[++i]));
                        config.enable_saturation = true;
                    } catch (...) {
                        std::cerr << "Error: Invalid float value for --saturation\n";
                        return 1;
                    }
                }
            } else if (args[i] == "--gamma") {
                if (i + 1 < args.size()) {
                    try {
                        config.gamma = std::stof(std::string(args[++i]));
                        config.enable_gamma = true;
                    } catch (...) {
                        std::cerr << "Error: Invalid float value for --gamma\n";
                        return 1;
                    }
                }
            } else if (input_path.empty()) {
                input_path = args[i];
            }
        }

        if (input_path.empty()) {
            std::cerr << "Error: Missing input image path for --enhance\n";
            print_help(argv[0]);
            return 1;
        }

        if (output_path.empty()) {
            std::cerr << "Error: Missing required --output parameter for --enhance\n";
            print_help(argv[0]);
            return 1;
        }

        return handle_enhance(input_path, output_path, config, precision_mode, mixed_policy);
    }

    std::cerr << "Error: Unrecognized option '" << args[0] << "'\n\n";
    print_help(argv[0]);
    return 1;
}
