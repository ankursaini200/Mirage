#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "mirage/analysis/quality_metrics.hpp"
#include "mirage/analysis/sensitivity_analyzer.hpp"
#include "mirage/cpu/cpu_features.hpp"
#include "mirage/enhancement/enhancement_engine.hpp"
#include "mirage/enhancement/precision_config.hpp"
#include "mirage/image/image_loader.hpp"
#include "mirage/image/tile_manager.hpp"
#include "mirage/precision/adaptive_precision.hpp"
#include "mirage/simd/simd_backend.hpp"
#include "mirage/version.hpp"

namespace {

std::string json_escape(std::string_view s) {
    std::string res;
    res.reserve(s.size());
    for (char c : s) {
        if (c == '"') res += "\\\"";
        else if (c == '\\') res += "\\\\";
        else if (c == '\b') res += "\\b";
        else if (c == '\f') res += "\\f";
        else if (c == '\n') res += "\\n";
        else if (c == '\r') res += "\\r";
        else if (c == '\t') res += "\\t";
        else res += c;
    }
    return res;
}

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
              << "    [--precision <fp32|fp16|mixed|adaptive>]\n"
              << "                                      Numerical precision policy (default: fp32)\n"
              << "    [--backend <auto|scalar|avx2>]    Execution backend (default: auto)\n"
              << "    [--sensitivity-threshold <val>]   Cutoff for adaptive precision in [0.0, 1.0] (default: 0.04)\n"
              << "    [--tile-size <size>]              Nominal tile dimensions for adaptive mode (default: 32)\n"
              << "    [--brightness <val>]              Additive brightness in [-1.0, 1.0] (default: 0.0)\n"
              << "    [--contrast <val>]                Contrast factor >= 0.0 (default: 1.0)\n"
              << "    [--saturation <val>]              Saturation factor >= 0.0 (default: 1.0)\n"
              << "    [--gamma <val>]                   Gamma exponent > 0.0 (default: 1.0)\n"
              << "    [--json]                          Output structured telemetry and metrics in JSON format\n"
              << "    [--save-heatmap <path>]           Save sensitivity heatmap (adaptive mode)\n"
              << "    [--save-precision-map <path>]     Save tile precision map overlay (adaptive mode)\n\n"
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
    const auto default_backend = mirage::simd::resolve_backend(mirage::simd::ExecutionBackend::Auto);

    std::cout << "Project name:              " << mirage::PROJECT_NAME << "\n"
              << "Project version:           " << mirage::get_version_string() << "\n"
              << "C++ standard:              " << mirage::get_cpp_standard_string() << "\n"
              << "Compiler information:      " << mirage::get_compiler_info_string() << "\n"
              << "Detected CPU architecture: " << cpu.architecture_name() << "\n"
              << "AVX2 availability:         " << cpu.avx2_status_string() << "\n"
              << "F16C availability:         " << (cpu.has_f16c ? "Supported" : "Unsupported") << "\n"
              << "FMA availability:          " << (cpu.has_fma ? "Supported" : "Unsupported") << "\n"
              << "Default execution backend: auto (resolves to " << mirage::simd::to_string(default_backend) << ")\n";
}

void print_info_json() {
    const auto& cpu = mirage::cpu::CpuFeatures::detect();
    const auto default_backend = mirage::simd::resolve_backend(mirage::simd::ExecutionBackend::Auto);

    std::cout << "{\n"
              << "  \"project_name\": \"" << json_escape(mirage::PROJECT_NAME) << "\",\n"
              << "  \"project_version\": \"" << json_escape(mirage::get_version_string()) << "\",\n"
              << "  \"cpp_standard\": \"" << json_escape(mirage::get_cpp_standard_string()) << "\",\n"
              << "  \"compiler_information\": \"" << json_escape(mirage::get_compiler_info_string()) << "\",\n"
              << "  \"cpu_architecture\": \"" << json_escape(cpu.architecture_name()) << "\",\n"
              << "  \"avx2\": " << (cpu.has_avx2 ? "true" : "false") << ",\n"
              << "  \"fma\": " << (cpu.has_fma ? "true" : "false") << ",\n"
              << "  \"f16c\": " << (cpu.has_f16c ? "true" : "false") << ",\n"
              << "  \"default_backend\": \"" << mirage::simd::to_string(default_backend) << "\"\n"
              << "}\n";
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
                   const mirage::enhancement::MixedPrecisionPolicy& policy,
                   mirage::simd::ExecutionBackend backend,
                   int tile_size = 32,
                   bool json_mode = false) {
    try {
        config.validate();

        const cv::Mat u8_in = mirage::image::ImageLoader::load(input_path);
        const cv::Mat f32_in = mirage::image::ImageLoader::normalize_to_float32(u8_in);

        const auto start_time = std::chrono::high_resolution_clock::now();
        const cv::Mat f32_out = mirage::enhancement::EnhancementEngine::process_with_precision(
            f32_in, config, mode, policy, backend);
        const auto end_time = std::chrono::high_resolution_clock::now();
        const double runtime_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        const cv::Mat u8_out = mirage::image::ImageLoader::denormalize_to_uint8(f32_out);
        mirage::image::ImageLoader::save(output_path, u8_out);

        if (json_mode) {
            const int ts = tile_size > 0 ? tile_size : 32;
            const mirage::image::TileManager tm(u8_in, ts, ts);
            const size_t total_tiles = tm.total_tiles();
            const size_t fp32_cnt = (mode == mirage::enhancement::PrecisionMode::FP32) ? total_tiles : 0;
            const size_t fp16_cnt = (mode == mirage::enhancement::PrecisionMode::FP16) ? total_tiles : 0;
            const double fp32_pct = (mode == mirage::enhancement::PrecisionMode::FP32) ? 100.0 : 0.0;
            const double fp16_pct = (mode == mirage::enhancement::PrecisionMode::FP16) ? 100.0 : 0.0;
            const auto& cpu = mirage::cpu::CpuFeatures::detect();

            std::cout << "{\n"
                      << "  \"status\": \"success\",\n"
                      << "  \"command\": \"enhance\",\n"
                      << "  \"image\": {\n"
                      << "    \"input_path\": \"" << json_escape(input_path.string()) << "\",\n"
                      << "    \"output_path\": \"" << json_escape(output_path.string()) << "\",\n"
                      << "    \"heatmap_path\": \"\",\n"
                      << "    \"precision_map_path\": \"\",\n"
                      << "    \"width\": " << u8_in.cols << ",\n"
                      << "    \"height\": " << u8_in.rows << ",\n"
                      << "    \"channels\": " << u8_in.channels() << ",\n"
                      << "    \"format\": \"CV_32FC3 normalized BGR (from CV_8UC3)\"\n"
                      << "  },\n"
                      << "  \"config\": {\n"
                      << "    \"precision_mode\": \"" << mirage::enhancement::to_string(mode) << "\",\n"
                      << "    \"policy\": \"" << (mode == mirage::enhancement::PrecisionMode::Mixed ? "operation-level" : "uniform") << "\",\n"
                      << "    \"backend\": \"" << mirage::simd::to_string(backend) << "\",\n"
                      << "    \"resolved_backend\": \"" << mirage::simd::to_string(mirage::simd::resolve_backend(backend)) << "\",\n"
                      << "    \"tile_width\": " << ts << ",\n"
                      << "    \"tile_height\": " << ts << ",\n"
                      << "    \"threshold\": null,\n"
                      << "    \"brightness\": " << std::fixed << std::setprecision(4) << config.brightness << ",\n"
                      << "    \"contrast\": " << config.contrast << ",\n"
                      << "    \"saturation\": " << config.saturation << ",\n"
                      << "    \"gamma\": " << config.gamma << "\n"
                      << "  },\n"
                      << "  \"runtime\": {\n"
                      << "    \"is_measured\": true,\n"
                      << "    \"measured_pipeline_ms\": " << std::fixed << std::setprecision(4) << runtime_ms << "\n"
                      << "  },\n"
                      << "  \"tile_allocation\": {\n"
                      << "    \"total_tiles\": " << total_tiles << ",\n"
                      << "    \"fp32_count\": " << fp32_cnt << ",\n"
                      << "    \"fp16_count\": " << fp16_cnt << ",\n"
                      << "    \"fp32_percentage\": " << std::fixed << std::setprecision(2) << fp32_pct << ",\n"
                      << "    \"fp16_percentage\": " << std::fixed << std::setprecision(2) << fp16_pct << ",\n"
                      << "    \"theoretical_bandwidth_reduction_pct\": " << (fp16_pct * 0.5) << "\n"
                      << "  },\n";

            if (mode == mirage::enhancement::PrecisionMode::FP32) {
                std::cout << "  \"quality_metrics_vs_fp32_reference\": null,\n";
            } else {
                const cv::Mat f32_ref = mirage::enhancement::EnhancementEngine::process(f32_in, config);
                const auto qm = mirage::analysis::evaluate_quality(f32_ref, f32_out);
                std::cout << "  \"quality_metrics_vs_fp32_reference\": {\n"
                          << "    \"reference_description\": \"FP32 Scalar Enhancement Output (Ground Truth)\",\n"
                          << "    \"psnr_db\": " << std::fixed << std::setprecision(2) << qm.peak_signal_to_noise_ratio << ",\n"
                          << "    \"ssim\": " << std::setprecision(4) << qm.ssim << ",\n"
                          << "    \"mean_ciede2000\": " << std::setprecision(4) << qm.mean_ciede2000 << ",\n"
                          << "    \"max_ciede2000\": " << std::setprecision(4) << qm.max_ciede2000 << ",\n"
                          << "    \"mean_absolute_error\": " << std::setprecision(6) << qm.mean_absolute_error << ",\n"
                          << "    \"root_mean_square_error\": " << std::setprecision(6) << qm.root_mean_square_error << ",\n"
                          << "    \"max_absolute_error\": " << std::setprecision(6) << qm.max_absolute_error << ",\n"
                          << "    \"relative_l2_error\": " << std::setprecision(6) << qm.relative_l2_error << "\n"
                          << "  },\n";
            }

            std::cout << "  \"hardware\": {\n"
                      << "    \"cpu_architecture\": \"" << json_escape(cpu.architecture_name()) << "\",\n"
                      << "    \"has_avx2\": " << (cpu.has_avx2 ? "true" : "false") << ",\n"
                      << "    \"has_fma\": " << (cpu.has_fma ? "true" : "false") << ",\n"
                      << "    \"has_f16c\": " << (cpu.has_f16c ? "true" : "false") << "\n"
                      << "  }\n"
                      << "}\n";
            return 0;
        }

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
                  << "  Backend:             " << mirage::simd::to_string(backend)
                  << " (resolved to " << mirage::simd::to_string(mirage::simd::resolve_backend(backend)) << ")\n"
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

int handle_enhance_adaptive(const std::filesystem::path& input_path,
                            const std::filesystem::path& output_path,
                            const mirage::enhancement::EnhancementConfig& config,
                            int tile_size,
                            double threshold,
                            mirage::simd::ExecutionBackend backend,
                            bool json_mode = false,
                            const std::filesystem::path& heatmap_path = "",
                            const std::filesystem::path& precision_map_path = "") {
    try {
        config.validate();

        const cv::Mat u8_in = mirage::image::ImageLoader::load(input_path);
        const cv::Mat f32_in = mirage::image::ImageLoader::normalize_to_float32(u8_in);

        // Execute adaptive enhancement with isolated timing
        const auto result = mirage::precision::AdaptivePrecisionEngine::enhance_adaptive(
            f32_in, config, tile_size, tile_size, threshold, mirage::analysis::SensitivityWeights{}, backend);

        // Save output
        const cv::Mat u8_out = mirage::image::ImageLoader::denormalize_to_uint8(result.enhanced_image);
        mirage::image::ImageLoader::save(output_path, u8_out);

        // Save heatmap if requested
        if (!heatmap_path.empty()) {
            mirage::analysis::SensitivityAnalyzer analyzer{};
            analyzer.save_heatmap(heatmap_path, u8_in, result.sensitivity_report);
        }

        // Save precision map if requested
        if (!precision_map_path.empty()) {
            cv::Mat pmap(u8_in.size(), CV_8UC3, cv::Scalar(19, 20, 31));
            for (const auto& assign : result.decision.assignments) {
                const cv::Scalar col = (assign.precision == mirage::enhancement::PrecisionMode::FP32)
                    ? cv::Scalar(237, 128, 47)   // Blue (#2f80ed in BGR)
                    : cv::Scalar(96, 174, 39);   // Green (#27ae60 in BGR)
                const cv::Rect r(assign.x, assign.y, assign.width, assign.height);
                cv::rectangle(pmap, r, col, cv::FILLED);
                cv::rectangle(pmap, r, cv::Scalar(72, 49, 45), 1);
            }
            cv::Mat blended;
            cv::addWeighted(u8_in, 0.45, pmap, 0.55, 0.0, blended);
            mirage::image::ImageLoader::save(precision_map_path, blended);
        }

        // Generate full FP32 reference OUTSIDE the timed section
        const cv::Mat f32_ref = mirage::enhancement::EnhancementEngine::process(f32_in, config);
        const auto qm = mirage::analysis::evaluate_quality(f32_ref, result.enhanced_image);

        if (json_mode) {
            const auto& cpu = mirage::cpu::CpuFeatures::detect();
            std::cout << "{\n"
                      << "  \"status\": \"success\",\n"
                      << "  \"command\": \"enhance\",\n"
                      << "  \"image\": {\n"
                      << "    \"input_path\": \"" << json_escape(input_path.string()) << "\",\n"
                      << "    \"output_path\": \"" << json_escape(output_path.string()) << "\",\n"
                      << "    \"heatmap_path\": \"" << json_escape(heatmap_path.string()) << "\",\n"
                      << "    \"precision_map_path\": \"" << json_escape(precision_map_path.string()) << "\",\n"
                      << "    \"width\": " << u8_in.cols << ",\n"
                      << "    \"height\": " << u8_in.rows << ",\n"
                      << "    \"channels\": " << u8_in.channels() << ",\n"
                      << "    \"format\": \"CV_32FC3 normalized BGR (from CV_8UC3)\"\n"
                      << "  },\n"
                      << "  \"config\": {\n"
                      << "    \"precision_mode\": \"adaptive\",\n"
                      << "    \"policy\": \"sensitivity-guided\",\n"
                      << "    \"backend\": \"" << mirage::simd::to_string(backend) << "\",\n"
                      << "    \"resolved_backend\": \"" << mirage::simd::to_string(mirage::simd::resolve_backend(backend)) << "\",\n"
                      << "    \"tile_width\": " << tile_size << ",\n"
                      << "    \"tile_height\": " << tile_size << ",\n"
                      << "    \"threshold\": " << std::fixed << std::setprecision(4) << threshold << ",\n"
                      << "    \"brightness\": " << config.brightness << ",\n"
                      << "    \"contrast\": " << config.contrast << ",\n"
                      << "    \"saturation\": " << config.saturation << ",\n"
                      << "    \"gamma\": " << config.gamma << "\n"
                      << "  },\n"
                      << "  \"runtime\": {\n"
                      << "    \"is_measured\": true,\n"
                      << "    \"measured_pipeline_ms\": " << std::fixed << std::setprecision(4) << result.adaptive_execution_time_ms << ",\n"
                      << "    \"adaptive_stage_ms\": " << result.adaptive_execution_time_ms << ",\n"
                      << "    \"sensitivity_stage_ms\": " << result.sensitivity_report.execution_time_ms << "\n"
                      << "  },\n"
                      << "  \"tile_allocation\": {\n"
                      << "    \"total_tiles\": " << result.decision.total_tiles << ",\n"
                      << "    \"fp32_count\": " << result.decision.fp32_count << ",\n"
                      << "    \"fp16_count\": " << result.decision.fp16_count << ",\n"
                      << "    \"fp32_percentage\": " << std::fixed << std::setprecision(2) << result.decision.fp32_percentage << ",\n"
                      << "    \"fp16_percentage\": " << std::fixed << std::setprecision(2) << result.decision.fp16_percentage << ",\n"
                      << "    \"theoretical_bandwidth_reduction_pct\": " << (result.decision.fp16_percentage * 0.5) << ",\n"
                      << "    \"assignments\": [\n";

            for (size_t i = 0; i < result.decision.assignments.size(); ++i) {
                const auto& a = result.decision.assignments[i];
                std::cout << "      {\"id\": " << a.tile_id
                          << ", \"x\": " << a.x
                          << ", \"y\": " << a.y
                          << ", \"width\": " << a.width
                          << ", \"height\": " << a.height
                          << ", \"is_partial\": " << (a.is_partial ? "true" : "false")
                          << ", \"precision\": \"" << (a.precision == mirage::enhancement::PrecisionMode::FP32 ? "FP32" : "FP16") << "\""
                          << ", \"sensitivity_score\": " << std::fixed << std::setprecision(6) << a.sensitivity_score << "}"
                          << (i + 1 < result.decision.assignments.size() ? ",\n" : "\n");
            }

            std::cout << "    ]\n"
                      << "  },\n"
                      << "  \"quality_metrics_vs_fp32_reference\": {\n"
                      << "    \"reference_description\": \"FP32 Scalar Enhancement Output (Ground Truth)\",\n"
                      << "    \"psnr_db\": " << std::fixed << std::setprecision(2) << qm.peak_signal_to_noise_ratio << ",\n"
                      << "    \"ssim\": " << std::setprecision(4) << qm.ssim << ",\n"
                      << "    \"mean_ciede2000\": " << std::setprecision(4) << qm.mean_ciede2000 << ",\n"
                      << "    \"max_ciede2000\": " << std::setprecision(4) << qm.max_ciede2000 << ",\n"
                      << "    \"mean_absolute_error\": " << std::setprecision(6) << qm.mean_absolute_error << ",\n"
                      << "    \"root_mean_square_error\": " << std::setprecision(6) << qm.root_mean_square_error << ",\n"
                      << "    \"max_absolute_error\": " << std::setprecision(6) << qm.max_absolute_error << ",\n"
                      << "    \"relative_l2_error\": " << std::setprecision(6) << qm.relative_l2_error << "\n"
                      << "  },\n"
                      << "  \"hardware\": {\n"
                      << "    \"cpu_architecture\": \"" << json_escape(cpu.architecture_name()) << "\",\n"
                      << "    \"has_avx2\": " << (cpu.has_avx2 ? "true" : "false") << ",\n"
                      << "    \"has_fma\": " << (cpu.has_fma ? "true" : "false") << ",\n"
                      << "    \"has_f16c\": " << (cpu.has_f16c ? "true" : "false") << "\n"
                      << "  }\n"
                      << "}\n";
            return 0;
        }

        const mirage::enhancement::ErrorMetrics metrics =
            mirage::enhancement::compute_error_metrics(f32_ref, result.enhanced_image);

        std::cout << "Image enhancement completed successfully:\n"
                  << "  Input:                 " << input_path.string() << "\n"
                  << "  Output:                " << output_path.string() << "\n"
                  << "  Dimensions:            " << u8_in.cols << " x " << u8_in.rows << "\n"
                  << "  Precision Mode:        Adaptive (Sensitivity-Guided Precision Selection)\n"
                  << "  Backend:               " << mirage::simd::to_string(backend)
                  << " (resolved to " << mirage::simd::to_string(mirage::simd::resolve_backend(backend)) << ")\n"
                  << "  Sensitivity Threshold: " << std::fixed << std::setprecision(4) << threshold << "\n"
                  << "  Tile Dimensions:       " << tile_size << " x " << tile_size << "\n"
                  << "  Brightness:            " << (config.enable_brightness ? std::to_string(config.brightness) : "disabled") << "\n"
                  << "  Contrast:              " << (config.enable_contrast ? std::to_string(config.contrast) : "disabled") << "\n"
                  << "  Saturation:            " << (config.enable_saturation ? std::to_string(config.saturation) : "disabled") << "\n"
                  << "  Gamma:                 " << (config.enable_gamma ? std::to_string(config.gamma) : "disabled") << "\n"
                  << "  Adaptive Runtime:      " << std::fixed << std::setprecision(2) << result.adaptive_execution_time_ms << " ms\n\n"
                  << "Tile Precision Distribution:\n"
                  << "  Total Tile Count:      " << result.decision.total_tiles << "\n"
                  << "  FP32 Tiles:            " << result.decision.fp32_count << " ("
                  << std::setprecision(2) << result.decision.fp32_percentage << " %)\n"
                  << "  FP16 Tiles:            " << result.decision.fp16_count << " ("
                  << std::setprecision(2) << result.decision.fp16_percentage << " %)\n\n"
                  << "Numerical Error Analysis (vs FP32 Ground Truth Reference):\n"
                  << metrics.to_string() << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in adaptive image enhancement: " << e.what() << "\n";
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
        bool json = false;
        for (const auto& a : args) {
            if (a == "--json") json = true;
        }
        if (json) {
            print_info_json();
        } else {
            print_info();
        }
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
        std::filesystem::path heatmap_path;
        std::filesystem::path precision_map_path;
        bool json_mode = false;
        mirage::enhancement::EnhancementConfig config{};
        mirage::enhancement::PrecisionMode precision_mode = mirage::enhancement::PrecisionMode::FP32;
        mirage::simd::ExecutionBackend backend = mirage::simd::ExecutionBackend::Auto;
        mirage::enhancement::MixedPrecisionPolicy mixed_policy{};
        int tile_size = 32;
        double sensitivity_threshold = mirage::precision::DEFAULT_SENSITIVITY_THRESHOLD;

        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--output" || args[i] == "-o") {
                if (i + 1 < args.size()) {
                    output_path = args[++i];
                }
            } else if (args[i] == "--json") {
                json_mode = true;
            } else if (args[i] == "--save-heatmap") {
                if (i + 1 < args.size()) {
                    heatmap_path = args[++i];
                }
            } else if (args[i] == "--save-precision-map") {
                if (i + 1 < args.size()) {
                    precision_map_path = args[++i];
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
            } else if (args[i] == "--backend") {
                if (i + 1 < args.size()) {
                    try {
                        backend = mirage::simd::parse_execution_backend(args[++i]);
                        if (!mirage::simd::is_backend_supported(backend)) {
                            std::cerr << "Error: Execution backend '" << mirage::simd::to_string(backend)
                                      << "' requested but not supported on this host CPU.\n";
                            return 1;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error: " << e.what() << "\n";
                        return 1;
                    }
                }
            } else if (args[i] == "--tile-size") {
                if (i + 1 < args.size()) {
                    tile_size = parse_tile_size(std::string(args[++i]));
                }
            } else if (args[i] == "--sensitivity-threshold") {
                if (i + 1 < args.size()) {
                    try {
                        sensitivity_threshold = std::stod(std::string(args[++i]));
                    } catch (...) {
                        std::cerr << "Error: Invalid float value for --sensitivity-threshold\n";
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

        if (precision_mode == mirage::enhancement::PrecisionMode::Adaptive) {
            if (tile_size <= 0) {
                std::cerr << "Error: Invalid tile size for adaptive precision.\n";
                return 1;
            }
            if (sensitivity_threshold < 0.0 || sensitivity_threshold > 1.0) {
                std::cerr << "Error: Sensitivity threshold must be in [0.0, 1.0].\n";
                return 1;
            }
            return handle_enhance_adaptive(input_path, output_path, config, tile_size, sensitivity_threshold, backend, json_mode, heatmap_path, precision_map_path);
        }

        return handle_enhance(input_path, output_path, config, precision_mode, mixed_policy, backend, tile_size, json_mode);
    }

    std::cerr << "Error: Unrecognized option '" << args[0] << "'\n\n";
    print_help(argv[0]);
    return 1;
}
