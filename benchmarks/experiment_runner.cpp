#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "mirage/analysis/quality_metrics.hpp"
#include "mirage/analysis/sensitivity_analyzer.hpp"
#include "mirage/enhancement/enhancement_engine.hpp"
#include "mirage/image/color_converter.hpp"
#include "mirage/image/image_loader.hpp"
#include "mirage/image/tile_manager.hpp"
#include "mirage/precision/adaptive_precision.hpp"
#include "mirage/version.hpp"

namespace fs = std::filesystem;
using namespace mirage;
using namespace mirage::analysis;
using namespace mirage::enhancement;
using namespace mirage::image;
using namespace mirage::precision;

namespace {

struct DatasetImage {
    std::string id;
    std::string filename;
    std::string modality;
    fs::path full_path;
    cv::Mat bgr_float;
    cv::Mat ref_fp32;
};

EnhancementConfig get_experiment_enhancement_config() {
    EnhancementConfig config{};
    config.enable_brightness = true;
    config.brightness = 0.15f;
    config.enable_contrast = true;
    config.contrast = 1.25f;
    config.enable_saturation = true;
    config.saturation = 1.35f;
    config.enable_gamma = true;
    config.gamma = 1.10f;
    return config;
}

std::vector<DatasetImage> load_dataset(const fs::path& dataset_dir) {
    std::vector<DatasetImage> dataset;
    const fs::path manifest_path = dataset_dir / "dataset_manifest.csv";

    if (!fs::exists(manifest_path)) {
        throw std::runtime_error("Dataset manifest not found at " + manifest_path.string());
    }

    std::ifstream file(manifest_path);
    std::string line;
    std::getline(file, line); // Skip header

    const auto config = get_experiment_enhancement_config();

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string id, filename, w_str, h_str, c_str, modality;
        std::getline(ss, id, ',');
        std::getline(ss, filename, ',');
        std::getline(ss, w_str, ',');
        std::getline(ss, h_str, ',');
        std::getline(ss, c_str, ',');
        std::getline(ss, modality, ',');
        if (modality.size() >= 2 && modality.front() == '"' && modality.back() == '"') {
            modality = modality.substr(1, modality.size() - 2);
        }

        const fs::path img_path = dataset_dir / filename;
        if (!fs::exists(img_path)) {
            std::cerr << "Warning: image file missing: " << img_path << "\n";
            continue;
        }

        cv::Mat raw = ImageLoader::load(img_path);
        if (raw.empty()) {
            std::cerr << "Warning: failed to read image: " << img_path << "\n";
            continue;
        }

        DatasetImage entry{};
        entry.id = id;
        entry.filename = filename;
        entry.modality = modality;
        entry.full_path = img_path;
        entry.bgr_float = ImageLoader::normalize_to_float32(raw);

        // Precompute ground truth FP32 enhancement
        entry.ref_fp32 = EnhancementEngine::process_with_precision(
            entry.bgr_float, config, PrecisionMode::FP32, MixedPrecisionPolicy{}, simd::ExecutionBackend::Scalar);

        dataset.push_back(std::move(entry));
    }

    return dataset;
}

void run_experiment_a_random_vs_guided(
    const std::vector<DatasetImage>& dataset,
    const fs::path& output_file,
    int num_seeds) {
    std::cout << "\n[Experiment A] Evaluating Random vs. Guided Allocation (Hypothesis H1)...\n";
    std::ofstream out(output_file);
    out << "image_id,modality,threshold,fp32_count,total_tiles,fp32_pct,trial_type,seed,time_ms,mae,rmse,psnr,rel_l2,ssim,mean_de00,max_de00\n";

    const auto config = get_experiment_enhancement_config();
    const std::vector<double> thresholds = {0.02, 0.04, 0.08, 0.15};

    for (const auto& img : dataset) {
        std::cout << "  Processing " << img.filename << " (" << img.modality << ")...\n";
        const TileManager tile_mgr(img.bgr_float, 32, 32);

        for (double tau : thresholds) {
            // 1. Guided precision decision & execution
            const auto guided_res = AdaptivePrecisionEngine::enhance_adaptive(
                img.bgr_float, config, 32, 32, tau);
            const auto q_guided = evaluate_quality(img.ref_fp32, guided_res.enhanced_image);

            out << img.id << ",\"" << img.modality << "\","
                << tau << ","
                << guided_res.decision.fp32_count << ","
                << guided_res.decision.total_tiles << ","
                << std::fixed << std::setprecision(2) << guided_res.decision.fp32_percentage << ","
                << "guided,0,"
                << std::fixed << std::setprecision(4) << guided_res.adaptive_execution_time_ms << ","
                << std::setprecision(6) << q_guided.mean_absolute_error << ","
                << q_guided.root_mean_square_error << ","
                << std::setprecision(2) << q_guided.peak_signal_to_noise_ratio << ","
                << std::setprecision(6) << q_guided.relative_l2_error << ","
                << std::setprecision(4) << q_guided.ssim << ","
                << std::setprecision(4) << q_guided.mean_ciede2000 << ","
                << q_guided.max_ciede2000 << "\n";

            // 2. Random precision baselines (num_seeds trials with exact budget parity)
            for (int seed = 1; seed <= num_seeds; ++seed) {
                const auto rand_decision = AdaptivePrecisionEngine::decide_random_from_policy(
                    tile_mgr, guided_res.decision, static_cast<uint32_t>(seed));
                const auto [rand_img, rand_time] = AdaptivePrecisionEngine::enhance_with_decision(
                    img.bgr_float, config, tile_mgr, rand_decision);
                const auto q_rand = evaluate_quality(img.ref_fp32, rand_img);

                out << img.id << ",\"" << img.modality << "\","
                    << tau << ","
                    << rand_decision.fp32_count << ","
                    << rand_decision.total_tiles << ","
                    << std::fixed << std::setprecision(2) << rand_decision.fp32_percentage << ","
                    << "random," << seed << ","
                    << std::fixed << std::setprecision(4) << rand_time << ","
                    << std::setprecision(6) << q_rand.mean_absolute_error << ","
                    << q_rand.root_mean_square_error << ","
                    << std::setprecision(2) << q_rand.peak_signal_to_noise_ratio << ","
                    << std::setprecision(6) << q_rand.relative_l2_error << ","
                    << std::setprecision(4) << q_rand.ssim << ","
                    << std::setprecision(4) << q_rand.mean_ciede2000 << ","
                    << q_rand.max_ciede2000 << "\n";
            }
        }
    }
    std::cout << "  Saved Experiment A results to " << output_file << "\n";
}

void run_experiment_b_feature_ablation(
    const std::vector<DatasetImage>& dataset,
    const fs::path& output_file) {
    std::cout << "\n[Experiment B] Evaluating Feature Ablation Study (Hypothesis H4)...\n";
    std::ofstream out(output_file);
    out << "image_id,modality,config_name,weight_c,weight_g,weight_e,weight_l,threshold,fp32_count,total_tiles,fp32_pct,time_ms,mae,rmse,psnr,rel_l2,ssim,mean_de00,max_de00\n";

    const auto config = get_experiment_enhancement_config();

    struct AblationConfig {
        std::string name;
        SensitivityWeights weights;
    };

    constexpr double one_third = 1.0 / 3.0;
    constexpr double one_third_rem = 1.0 - 2.0 * one_third;

    const std::vector<AblationConfig> ablation_configs = {
        {"all_features",  SensitivityWeights{0.25, 0.25, 0.25, 0.25}},
        {"no_color",      SensitivityWeights{0.00, one_third, one_third, one_third_rem}},
        {"no_gradient",   SensitivityWeights{one_third, 0.00, one_third, one_third_rem}},
        {"no_edge",       SensitivityWeights{one_third, one_third, 0.00, one_third_rem}},
        {"no_contrast",   SensitivityWeights{one_third, one_third, one_third_rem, 0.00}},
        {"color_only",    SensitivityWeights{1.00, 0.00, 0.00, 0.00}},
        {"gradient_only", SensitivityWeights{0.00, 1.00, 0.00, 0.00}},
        {"edge_only",     SensitivityWeights{0.00, 0.00, 1.00, 0.00}},
        {"contrast_only", SensitivityWeights{0.00, 0.00, 0.00, 1.00}}
    };

    const std::vector<double> thresholds = {0.04, 0.08};

    for (const auto& img : dataset) {
        for (const auto& abl : ablation_configs) {
            for (double tau : thresholds) {
                const auto res = AdaptivePrecisionEngine::enhance_adaptive(
                    img.bgr_float, config, 32, 32, tau, abl.weights);
                const auto q = evaluate_quality(img.ref_fp32, res.enhanced_image);

                out << img.id << ",\"" << img.modality << "\","
                    << abl.name << ","
                    << abl.weights.color_weight << ","
                    << abl.weights.gradient_weight << ","
                    << abl.weights.edge_weight << ","
                    << abl.weights.contrast_weight << ","
                    << tau << ","
                    << res.decision.fp32_count << ","
                    << res.decision.total_tiles << ","
                    << std::fixed << std::setprecision(2) << res.decision.fp32_percentage << ","
                    << std::fixed << std::setprecision(4) << res.adaptive_execution_time_ms << ","
                    << std::setprecision(6) << q.mean_absolute_error << ","
                    << q.root_mean_square_error << ","
                    << std::setprecision(2) << q.peak_signal_to_noise_ratio << ","
                    << std::setprecision(6) << q.relative_l2_error << ","
                    << std::setprecision(4) << q.ssim << ","
                    << std::setprecision(4) << q.mean_ciede2000 << ","
                    << q.max_ciede2000 << "\n";
            }
        }
    }
    std::cout << "  Saved Experiment B results to " << output_file << "\n";
}

void run_experiment_c_tile_validation(
    const std::vector<DatasetImage>& dataset,
    const fs::path& output_file) {
    std::cout << "\n[Experiment C] Validating Tile Sensitivity Scores vs. FP16 Errors...\n";
    std::ofstream out(output_file);
    out << "image_id,modality,tile_id,x,y,width,height,is_partial,sensitivity_score,color_var,gradient_mag,edge_density,local_contrast,tile_mae,tile_rmse,tile_mean_de00\n";

    const auto config = get_experiment_enhancement_config();
    SensitivityAnalyzer analyzer;

    for (const auto& img : dataset) {
        const TileManager tile_mgr(img.bgr_float, 32, 32);
        const cv::Mat lab_img = ColorConverter::bgr_to_lab(img.bgr_float);
        const auto report = analyzer.analyze(tile_mgr, lab_img);

        const auto& tiles = tile_mgr.tiles();
        for (size_t i = 0; i < tiles.size(); ++i) {
            const auto& t = tiles[i];
            const auto& meta = report.tiles[i];

            // Evaluate tile FP32 vs FP16 error
            cv::Mat tile_fp32 = EnhancementEngine::process_with_precision(
                t.view, config, PrecisionMode::FP32, MixedPrecisionPolicy{}, simd::ExecutionBackend::Scalar);
            cv::Mat tile_fp16 = EnhancementEngine::process_with_precision(
                t.view, config, PrecisionMode::FP16, MixedPrecisionPolicy{}, simd::ExecutionBackend::Scalar);

            const auto q = evaluate_quality(tile_fp32, tile_fp16);

            out << img.id << ",\"" << img.modality << "\","
                << t.id << "," << t.x << "," << t.y << ","
                << t.width << "," << t.height << "," << (t.is_partial ? 1 : 0) << ","
                << std::fixed << std::setprecision(6) << meta.sensitivity_score << ","
                << meta.color_variation << ","
                << meta.gradient_magnitude << ","
                << meta.edge_density << ","
                << meta.local_contrast << ","
                << q.mean_absolute_error << ","
                << q.root_mean_square_error << ","
                << std::setprecision(4) << q.mean_ciede2000 << "\n";
        }
    }
    std::cout << "  Saved Experiment C results to " << output_file << "\n";
}

void run_experiment_d_pareto_sweeps(
    const std::vector<DatasetImage>& dataset,
    const fs::path& output_file) {
    std::cout << "\n[Experiment D] Sweeping Parameters for Pareto Frontier (Hypotheses H2 & H3)...\n";
    std::ofstream out(output_file);
    out << "image_id,modality,tile_size,threshold,fp32_count,total_tiles,fp32_pct,time_ms,mae,rmse,psnr,rel_l2,ssim,mean_de00,max_de00\n";

    const auto config = get_experiment_enhancement_config();
    const std::vector<int> tile_sizes = {8, 16, 32, 64};
    const std::vector<double> thresholds = {
        0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00
    };

    for (const auto& img : dataset) {
        for (int ts : tile_sizes) {
            for (double tau : thresholds) {
                const auto res = AdaptivePrecisionEngine::enhance_adaptive(
                    img.bgr_float, config, ts, ts, tau);
                const auto q = evaluate_quality(img.ref_fp32, res.enhanced_image);

                out << img.id << ",\"" << img.modality << "\","
                    << ts << ","
                    << tau << ","
                    << res.decision.fp32_count << ","
                    << res.decision.total_tiles << ","
                    << std::fixed << std::setprecision(2) << res.decision.fp32_percentage << ","
                    << std::fixed << std::setprecision(4) << res.adaptive_execution_time_ms << ","
                    << std::setprecision(6) << q.mean_absolute_error << ","
                    << q.root_mean_square_error << ","
                    << std::setprecision(2) << q.peak_signal_to_noise_ratio << ","
                    << std::setprecision(6) << q.relative_l2_error << ","
                    << std::setprecision(4) << q.ssim << ","
                    << std::setprecision(4) << q.mean_ciede2000 << ","
                    << q.max_ciede2000 << "\n";
            }
        }
    }
    std::cout << "  Saved Experiment D results to " << output_file << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    fs::path dataset_dir = "data/experiments";
    fs::path output_dir = "benchmarks/experiments";
    int num_seeds = 30;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--dataset-dir" && i + 1 < argc) {
            dataset_dir = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--seeds" && i + 1 < argc) {
            num_seeds = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: mirage_experiments [options]\n"
                      << "Options:\n"
                      << "  --dataset-dir <dir>  Path to multi-modal dataset (default: data/experiments)\n"
                      << "  --output-dir <dir>   Path to output CSV experiments (default: benchmarks/experiments)\n"
                      << "  --seeds <N>          Number of random seeds for Experiment A (default: 30)\n"
                      << "  --help, -h           Show this help message\n";
            return 0;
        }
    }

    fs::create_directories(output_dir);

    std::cout << "========================================================================================\n"
              << "       MIRAGE PHASE 9: RESEARCH EXPERIMENTS, ABLATIONS & PARETO ANALYSIS\n"
              << "========================================================================================\n"
              << "Mirage Version:      " << get_version_string() << "\n"
              << "Dataset Directory:   " << dataset_dir.string() << "\n"
              << "Output Directory:    " << output_dir.string() << "\n"
              << "Random Seeds:        " << num_seeds << "\n"
              << "========================================================================================\n";

    try {
        const auto dataset = load_dataset(dataset_dir);
        std::cout << "Loaded " << dataset.size() << " dataset modalities with ground truth FP32 references.\n";

        run_experiment_a_random_vs_guided(dataset, output_dir / "random_ablation.csv", num_seeds);
        run_experiment_b_feature_ablation(dataset, output_dir / "feature_ablation.csv");
        run_experiment_c_tile_validation(dataset, output_dir / "tile_sensitivity_errors.csv");
        run_experiment_d_pareto_sweeps(dataset, output_dir / "pareto_results.csv");

        std::cout << "\n========================================================================================\n"
                  << "All Phase 9 experiments completed successfully!\n"
                  << "========================================================================================\n";
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error during experiments: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
