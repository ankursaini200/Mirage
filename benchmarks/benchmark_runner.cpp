#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "mirage/analysis/sensitivity_analyzer.hpp"
#include "mirage/cpu/cpu_features.hpp"
#include "mirage/enhancement/enhancement_engine.hpp"
#include "mirage/image/tile_manager.hpp"
#include "mirage/precision/adaptive_precision.hpp"
#include "mirage/simd/avx2_kernels.hpp"
#include "mirage/simd/simd_backend.hpp"
#include "mirage/version.hpp"
#include "perf_sampler.hpp"

using namespace mirage;
using namespace mirage::enhancement;
using namespace mirage::simd;
using namespace mirage::precision;
using namespace mirage::analysis;
using namespace mirage::benchmark;

namespace {

struct TimingStats {
    double min_ms = 0.0;
    double max_ms = 0.0;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double stddev_ms = 0.0;
    double throughput_mpix_s = 0.0;
};

struct BenchmarkRecord {
    std::string resolution_label;
    int width = 0;
    int height = 0;
    int tile_size = 0;
    std::string operation;
    std::string backend;
    std::string precision_mode;
    double sensitivity_threshold = 0.0;
    double fp16_tile_pct = 0.0;
    double fp32_tile_pct = 0.0;

    TimingStats timing;
    PerfCounterStats perf;

    double estimated_memory_traffic_bytes_per_pix = 0.0;
    double estimated_arithmetic_intensity = 0.0;
};

template <typename Func>
std::pair<TimingStats, PerfCounterStats> measure_kernel(
    Func&& func,
    int pixels,
    HardwarePerfSampler& sampler,
    size_t warmup_runs,
    size_t measured_runs) {

    // Warmup phase (untimed, unmonitored)
    for (size_t i = 0; i < warmup_runs; ++i) {
        func();
    }

    std::vector<double> timings_ms;
    timings_ms.reserve(measured_runs);

    // Accumulated performance counters over measured runs
    PerfCounterStats total_perf{};
    total_perf.available = sampler.is_available();

    for (size_t i = 0; i < measured_runs; ++i) {
        sampler.start();
        const auto start = std::chrono::high_resolution_clock::now();
        func();
        const auto end = std::chrono::high_resolution_clock::now();
        const auto p = sampler.stop();

        const double duration = std::chrono::duration<double, std::milli>(end - start).count();
        timings_ms.push_back(duration);

        if (p.available) {
            total_perf.cycles += p.cycles;
            total_perf.instructions += p.instructions;
            total_perf.cache_references += p.cache_references;
            total_perf.cache_misses += p.cache_misses;
            total_perf.branch_instructions += p.branch_instructions;
            total_perf.branch_misses += p.branch_misses;
            total_perf.task_clock_ns += p.task_clock_ns;
        }
    }

    std::sort(timings_ms.begin(), timings_ms.end());

    TimingStats t{};
    t.min_ms = timings_ms.front();
    t.max_ms = timings_ms.back();
    t.median_ms = timings_ms[timings_ms.size() / 2];

    const double sum = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0);
    t.mean_ms = sum / static_cast<double>(timings_ms.size());

    double sq_diff_sum = 0.0;
    for (double v : timings_ms) {
        sq_diff_sum += (v - t.mean_ms) * (v - t.mean_ms);
    }
    t.stddev_ms = std::sqrt(sq_diff_sum / static_cast<double>(timings_ms.size()));
    t.throughput_mpix_s = (static_cast<double>(pixels) / 1e6) / (t.median_ms / 1e3);

    // Normalize hardware counters to average per measured run
    if (total_perf.available && measured_runs > 0) {
        const double n = static_cast<double>(measured_runs);
        total_perf.cycles = static_cast<uint64_t>(static_cast<double>(total_perf.cycles) / n);
        total_perf.instructions = static_cast<uint64_t>(static_cast<double>(total_perf.instructions) / n);
        total_perf.cache_references = static_cast<uint64_t>(static_cast<double>(total_perf.cache_references) / n);
        total_perf.cache_misses = static_cast<uint64_t>(static_cast<double>(total_perf.cache_misses) / n);
        total_perf.branch_instructions = static_cast<uint64_t>(static_cast<double>(total_perf.branch_instructions) / n);
        total_perf.branch_misses = static_cast<uint64_t>(static_cast<double>(total_perf.branch_misses) / n);
        total_perf.task_clock_ns = static_cast<uint64_t>(static_cast<double>(total_perf.task_clock_ns) / n);
    }

    return {t, total_perf};
}

cv::Mat generate_test_image(int width, int height) {
    cv::Mat img(height, width, CV_32FC3);
    for (int r = 0; r < height; ++r) {
        auto* ptr = img.ptr<cv::Vec3f>(r);
        for (int c = 0; c < width; ++c) {
            const float u = static_cast<float>(c) / static_cast<float>(width);
            const float v = static_cast<float>(r) / static_cast<float>(height);
            // Mix of gradients, high frequency ripples, and smooth regions
            const float b = 0.5f + 0.5f * std::sin(10.0f * u) * std::cos(10.0f * v);
            const float g = 0.3f + 0.6f * u;
            const float r_val = 0.2f + 0.7f * v;
            ptr[c] = cv::Vec3f(
                std::clamp(b, 0.0f, 1.0f),
                std::clamp(g, 0.0f, 1.0f),
                std::clamp(r_val, 0.0f, 1.0f)
            );
        }
    }
    return img;
}

void write_csv(const std::string& path, const std::vector<BenchmarkRecord>& records) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Warning: Could not open CSV file for writing: " << path << "\n";
        return;
    }

    out << "Resolution,Width,Height,TileSize,Operation,Backend,PrecisionMode,"
        << "SensitivityThreshold,FP16TilePct,FP32TilePct,RuntimeMedianMs,RuntimeMeanMs,"
        << "RuntimeMinMs,RuntimeMaxMs,RuntimeStdDevMs,ThroughputMPixS,CPUCycles,"
        << "Instructions,IPC,CacheReferences,CacheMisses,CacheMissRatePct,BranchMisses,"
        << "EstimatedMemoryTrafficBytesPerPix,EstimatedArithmeticIntensityFLOPsPerByte\n";

    for (const auto& r : records) {
        out << "\"" << r.resolution_label << "\","
            << r.width << ","
            << r.height << ","
            << r.tile_size << ","
            << "\"" << r.operation << "\","
            << "\"" << r.backend << "\","
            << "\"" << r.precision_mode << "\","
            << std::fixed << std::setprecision(4) << r.sensitivity_threshold << ","
            << std::fixed << std::setprecision(2) << r.fp16_tile_pct << ","
            << std::fixed << std::setprecision(2) << r.fp32_tile_pct << ","
            << std::fixed << std::setprecision(4) << r.timing.median_ms << ","
            << std::fixed << std::setprecision(4) << r.timing.mean_ms << ","
            << std::fixed << std::setprecision(4) << r.timing.min_ms << ","
            << std::fixed << std::setprecision(4) << r.timing.max_ms << ","
            << std::fixed << std::setprecision(4) << r.timing.stddev_ms << ","
            << std::fixed << std::setprecision(2) << r.timing.throughput_mpix_s << ",";

        if (r.perf.available) {
            out << r.perf.cycles << ","
                << r.perf.instructions << ","
                << std::fixed << std::setprecision(3) << r.perf.ipc() << ","
                << r.perf.cache_references << ","
                << r.perf.cache_misses << ","
                << std::fixed << std::setprecision(2) << (r.perf.cache_miss_rate() * 100.0) << ","
                << r.perf.branch_misses << ",";
        } else {
            out << "N/A,N/A,N/A,N/A,N/A,N/A,N/A,";
        }

        out << std::fixed << std::setprecision(2) << r.estimated_memory_traffic_bytes_per_pix << ","
            << std::fixed << std::setprecision(4) << r.estimated_arithmetic_intensity << "\n";
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    bool quick_mode = false;
    std::string csv_dir = "benchmarks/results";

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--quick") {
            quick_mode = true;
        } else if (arg == "--csv-dir" && i + 1 < argc) {
            csv_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: mirage_benchmarks [options]\n"
                      << "Options:\n"
                      << "  --quick          Execute fewer iterations for rapid validation\n"
                      << "  --csv-dir <dir>  Directory to save output CSV benchmark reports (default: benchmarks/results)\n"
                      << "  --help, -h       Display this help message\n";
            return 0;
        }
    }

    std::filesystem::create_directories(csv_dir);

    const auto& cpu = cpu::CpuFeatures::detect();
    HardwarePerfSampler sampler;

    std::cout << "========================================================================================\n"
              << "          MIRAGE PHASE 8: HARDWARE ARCHITECTURE & PERFORMANCE BENCHMARK SUITE\n"
              << "========================================================================================\n"
              << "Mirage Version:            " << get_version_string() << "\n"
              << "Host CPU Architecture:     " << cpu.architecture_name() << "\n"
              << "AVX2 SIMD Support:         " << cpu.avx2_status_string() << "\n"
              << "FMA Support:               " << (cpu.has_fma ? "Supported" : "Unsupported") << "\n"
              << "F16C Support:              " << (cpu.has_f16c ? "Supported" : "Unsupported") << "\n"
              << "Active Backend:            " << to_string(resolve_backend(ExecutionBackend::Auto)) << "\n"
              << "Compiler:                  " << get_compiler_info_string() << "\n"
              << "C++ Standard:              " << get_cpp_standard_string() << "\n"
              << "Hardware Perf Counters:    " << (sampler.is_available() ? "ENABLED (via perf_event_open)" : "RESTRICTED / UNAVAILABLE") << "\n"
              << "Benchmark Mode:            " << (quick_mode ? "Quick (validation)" : "Full Comprehensive (>= 10 iterations)") << "\n"
              << "========================================================================================\n\n";

    std::vector<BenchmarkRecord> all_records;
    std::vector<BenchmarkRecord> tile_records;

    // Enhancement pipeline configuration
    EnhancementConfig config{};
    config.enable_brightness = true;
    config.brightness = 0.15f;
    config.enable_contrast = true;
    config.contrast = 1.25f;
    config.enable_saturation = true;
    config.saturation = 1.35f;
    config.enable_gamma = true;
    config.gamma = 1.10f;

    struct ResolutionTest {
        std::string label;
        int width;
        int height;
    };

    const std::vector<ResolutionTest> resolutions = {
        {"Small (Cache-Resident)", 190, 130},
        {"1080p Full HD", 1920, 1080},
        {"4K UHD", 3840, 2160}
    };

    const size_t warmup = quick_mode ? 2 : 5;
    const size_t measured = quick_mode ? 5 : 12;

    for (const auto& res : resolutions) {
        const int num_pixels = res.width * res.height;
        const cv::Mat src = generate_test_image(res.width, res.height);
        cv::Mat dst(res.height, res.width, CV_32FC3);

        std::cout << "\n========================================================================================\n"
                  << "RESOLUTION: " << res.label << " (" << res.width << " x " << res.height << ", "
                  << std::fixed << std::setprecision(2) << (static_cast<double>(num_pixels) / 1e6) << " MPix)\n"
                  << "========================================================================================\n";

        std::cout << std::left
                  << std::setw(30) << "Operation / Backend"
                  << std::setw(12) << "Median(ms)"
                  << std::setw(11) << "StdDev"
                  << std::setw(13) << "Throughput"
                  << std::setw(8)  << "IPC"
                  << std::setw(12) << "CacheMiss%"
                  << std::setw(10) << "Speedup\n"
                  << std::string(96, '-') << "\n";

        auto record_and_print = [&](const std::string& op_name,
                                    const std::string& backend_name,
                                    const std::string& prec_mode,
                                    double est_traffic,
                                    double est_ai,
                                    double baseline_ms,
                                    auto&& func) -> double {
            const auto [timing, perf] = measure_kernel(func, num_pixels, sampler, warmup, measured);

            BenchmarkRecord rec{};
            rec.resolution_label = res.label;
            rec.width = res.width;
            rec.height = res.height;
            rec.tile_size = 0;
            rec.operation = op_name;
            rec.backend = backend_name;
            rec.precision_mode = prec_mode;
            rec.timing = timing;
            rec.perf = perf;
            rec.estimated_memory_traffic_bytes_per_pix = est_traffic;
            rec.estimated_arithmetic_intensity = est_ai;
            all_records.push_back(rec);

            const double speedup = (baseline_ms > 0.0 && timing.median_ms > 0.0)
                                       ? (baseline_ms / timing.median_ms)
                                       : 1.0;

            std::cout << std::left << std::setw(30) << (op_name + " (" + backend_name + ")")
                      << std::fixed << std::setprecision(3)
                      << std::setw(12) << timing.median_ms
                      << std::setw(11) << timing.stddev_ms
                      << std::setprecision(1)
                      << std::setw(13) << (std::to_string(static_cast<int>(timing.throughput_mpix_s)) + " MP/s");

            if (perf.available) {
                std::cout << std::fixed << std::setprecision(2)
                          << std::setw(8) << perf.ipc()
                          << std::setw(12) << (std::to_string(static_cast<int>(perf.cache_miss_rate() * 100.0)) + "%");
            } else {
                std::cout << std::setw(8) << "N/A" << std::setw(12) << "N/A";
            }

            if (baseline_ms > 0.0) {
                std::cout << std::fixed << std::setprecision(2) << speedup << "x\n";
            } else {
                std::cout << "1.00x (base)\n";
            }

            return timing.median_ms;
        };

        // 1. Brightness
        const double b_s = record_and_print("Brightness", "Scalar", "FP32", 24.0, 0.25, 0.0, [&]() {
            EnhancementEngine::adjust_brightness_scalar(src, dst, 0.15f);
        });
        record_and_print("Brightness", "AVX2", "FP32", 24.0, 0.25, b_s, [&]() {
            avx2::adjust_brightness_avx2(src, dst, 0.15f);
        });

        // 2. Contrast
        const double c_s = record_and_print("Contrast", "Scalar", "FP32", 24.0, 0.375, 0.0, [&]() {
            EnhancementEngine::adjust_contrast_scalar(src, dst, 1.25f, 0.5f);
        });
        record_and_print("Contrast", "AVX2", "FP32", 24.0, 0.375, c_s, [&]() {
            avx2::adjust_contrast_avx2(src, dst, 1.25f, 0.5f);
        });

        // 3. Brightness + Contrast (Unfused vs Fused)
        const double bc_s = record_and_print("Brightness+Contrast", "Scalar Unfused", "FP32", 48.0, 0.3125, 0.0, [&]() {
            EnhancementEngine::adjust_brightness_scalar(src, dst, 0.15f);
            EnhancementEngine::adjust_contrast_scalar(dst, dst, 1.25f, 0.5f);
        });
        record_and_print("Brightness+Contrast", "AVX2 Fused", "FP32", 24.0, 0.625, bc_s, [&]() {
            avx2::adjust_brightness_contrast_avx2(src, dst, 0.15f, 1.25f, 0.5f);
        });

        // 4. Saturation
        const double s_s = record_and_print("Saturation", "Scalar", "FP32", 24.0, 0.708, 0.0, [&]() {
            EnhancementEngine::adjust_saturation_scalar(src, dst, 1.35f);
        });
        record_and_print("Saturation", "AVX2", "FP32", 24.0, 0.708, s_s, [&]() {
            avx2::adjust_saturation_avx2(src, dst, 1.35f);
        });

        // 5. Gamma
        record_and_print("Gamma", "Scalar", "FP32", 24.0, 3.125, 0.0, [&]() {
            EnhancementEngine::adjust_gamma_scalar(src, dst, 1.10f);
        });

        // 6. Sensitivity Analysis (32x32 tiles)
        SensitivityAnalyzer analyzer;
        record_and_print("Sensitivity Analysis", "Tile 32x32", "FP32", 12.0, 4.167, 0.0, [&]() {
            (void)analyzer.analyze(src, 32, 32);
        });

        // 7. Full Pipeline (5 configurations)
        const double pipe_s = record_and_print("Full Pipeline", "Scalar", "FP32", 96.0, 1.115, 0.0, [&]() {
            EnhancementEngine::process_scalar(src, dst, config);
        });
        record_and_print("Full Pipeline", "AVX2", "FP32", 48.0, 2.229, pipe_s, [&]() {
            avx2::process_pipeline_avx2(src, dst, config);
        });

        // Software FP16 Emulation (run on Small & 1080p, skip 4K or use fewer runs to avoid timeout)
        if (res.width <= 1920) {
            record_and_print("Full Pipeline", "Scalar", "FP16 Emulated", 96.0, 1.115, pipe_s, [&]() {
                EnhancementEngine::process_fp16(src, dst, config);
            });
        }

        // Adaptive Precision with multiple thresholds (tau = 0.02, 0.04, 0.08)
        const double thresholds[] = {0.02, 0.04, 0.08};
        const size_t adapt_measured = (res.width <= 190) ? (quick_mode ? 5 : 10) : ((res.width <= 1920) ? (quick_mode ? 3 : 5) : 3);
        const size_t adapt_warmup = (res.width >= 3840) ? 1 : (quick_mode ? 2 : 3);

        for (double tau : thresholds) {
            // Adaptive Scalar
            {
                AdaptiveEnhancementResult res_ad{};
                const auto [timing, perf] = measure_kernel([&]() {
                    res_ad = AdaptivePrecisionEngine::enhance_adaptive(
                        src, config, 32, 32, tau, SensitivityWeights{}, ExecutionBackend::Scalar);
                }, num_pixels, sampler, adapt_warmup, adapt_measured);

                BenchmarkRecord rec{};
                rec.resolution_label = res.label;
                rec.width = res.width;
                rec.height = res.height;
                rec.tile_size = 32;
                rec.operation = "Adaptive Pipeline (tau=" + std::to_string(tau).substr(0, 4) + ")";
                rec.backend = "Scalar";
                rec.precision_mode = "Adaptive";
                rec.sensitivity_threshold = tau;
                rec.fp16_tile_pct = res_ad.decision.fp16_percentage;
                rec.fp32_tile_pct = res_ad.decision.fp32_percentage;
                rec.timing = timing;
                rec.perf = perf;
                rec.estimated_memory_traffic_bytes_per_pix = 96.0;
                rec.estimated_arithmetic_intensity = 1.115;
                all_records.push_back(rec);

                std::cout << std::left << std::setw(30) << ("Adaptive (tau=" + std::to_string(tau).substr(0, 4) + ", Scalar)")
                          << std::fixed << std::setprecision(3)
                          << std::setw(12) << timing.median_ms
                          << std::setw(11) << timing.stddev_ms
                          << std::setprecision(1)
                          << std::setw(13) << (std::to_string(static_cast<int>(timing.throughput_mpix_s)) + " MP/s");
                if (perf.available) {
                    std::cout << std::fixed << std::setprecision(2)
                              << std::setw(8) << perf.ipc()
                              << std::setw(12) << (std::to_string(static_cast<int>(perf.cache_miss_rate() * 100.0)) + "%");
                } else {
                    std::cout << std::setw(8) << "N/A" << std::setw(12) << "N/A";
                }
                const double sp = (pipe_s > 0.0) ? (pipe_s / timing.median_ms) : 1.0;
                std::cout << std::fixed << std::setprecision(2) << sp << "x\n";
            }

            // Adaptive AVX2
            {
                AdaptiveEnhancementResult res_ad{};
                const auto [timing, perf] = measure_kernel([&]() {
                    res_ad = AdaptivePrecisionEngine::enhance_adaptive(
                        src, config, 32, 32, tau, SensitivityWeights{}, ExecutionBackend::AVX2);
                }, num_pixels, sampler, adapt_warmup, adapt_measured);

                BenchmarkRecord rec{};
                rec.resolution_label = res.label;
                rec.width = res.width;
                rec.height = res.height;
                rec.tile_size = 32;
                rec.operation = "Adaptive Pipeline (tau=" + std::to_string(tau).substr(0, 4) + ")";
                rec.backend = "AVX2";
                rec.precision_mode = "Adaptive";
                rec.sensitivity_threshold = tau;
                rec.fp16_tile_pct = res_ad.decision.fp16_percentage;
                rec.fp32_tile_pct = res_ad.decision.fp32_percentage;
                rec.timing = timing;
                rec.perf = perf;
                rec.estimated_memory_traffic_bytes_per_pix = 48.0;
                rec.estimated_arithmetic_intensity = 2.229;
                all_records.push_back(rec);

                std::cout << std::left << std::setw(30) << ("Adaptive (tau=" + std::to_string(tau).substr(0, 4) + ", AVX2)")
                          << std::fixed << std::setprecision(3)
                          << std::setw(12) << timing.median_ms
                          << std::setw(11) << timing.stddev_ms
                          << std::setprecision(1)
                          << std::setw(13) << (std::to_string(static_cast<int>(timing.throughput_mpix_s)) + " MP/s");
                if (perf.available) {
                    std::cout << std::fixed << std::setprecision(2)
                              << std::setw(8) << perf.ipc()
                              << std::setw(12) << (std::to_string(static_cast<int>(perf.cache_miss_rate() * 100.0)) + "%");
                } else {
                    std::cout << std::setw(8) << "N/A" << std::setw(12) << "N/A";
                }
                const double sp = (pipe_s > 0.0) ? (pipe_s / timing.median_ms) : 1.0;
                std::cout << std::fixed << std::setprecision(2) << sp << "x\n";
            }
        }
    }

    // ========================================================================
    // TILE-SIZE ANALYSIS (8x8, 16x16, 32x32, 64x64 on 1080p)
    // ========================================================================
    std::cout << "\n========================================================================================\n"
              << "TILE-SIZE ANALYSIS ON 1080p FULL HD (8x8, 16x16, 32x32, 64x64)\n"
              << "========================================================================================\n";

    const int t_width = 1920;
    const int t_height = 1080;
    const int t_pixels = t_width * t_height;
    const cv::Mat t_src = generate_test_image(t_width, t_height);
    const int tile_sizes[] = {8, 16, 32, 64};
    constexpr double tile_tau = 0.04;

    std::cout << std::left
              << std::setw(12) << "Tile Size"
              << std::setw(14) << "Total Tiles"
              << std::setw(14) << "SensTime(ms)"
              << std::setw(14) << "Decide(ms)"
              << std::setw(14) << "AdaptAVX2(ms)"
              << std::setw(12) << "FP16 Pct"
              << std::setw(12) << "FP32 Pct"
              << "CacheMiss%\n"
              << std::string(96, '-') << "\n";

    for (int ts : tile_sizes) {
        image::TileManager tm(t_src, ts, ts);
        const auto total_tiles_count = tm.total_tiles();
        SensitivityAnalyzer analyzer;

        // 1. Sensitivity Analysis runtime
        const auto [sens_t, sens_p] = measure_kernel([&]() {
            (void)analyzer.analyze(t_src, ts, ts);
        }, t_pixels, sampler, 2, quick_mode ? 4 : 8);

        const auto report = analyzer.analyze(t_src, ts, ts);

        // 2. Decision time
        const auto [dec_t, dec_p] = measure_kernel([&]() {
            (void)AdaptivePrecisionEngine::decide_precisions(report, tile_tau);
        }, t_pixels, sampler, 2, quick_mode ? 4 : 8);

        // 3. Full Adaptive AVX2 execution time
        AdaptiveEnhancementResult a_res{};
        const auto [ad_t, ad_p] = measure_kernel([&]() {
            a_res = AdaptivePrecisionEngine::enhance_adaptive(
                t_src, config, ts, ts, tile_tau, SensitivityWeights{}, ExecutionBackend::AVX2);
        }, t_pixels, sampler, 2, quick_mode ? 4 : 8);

        BenchmarkRecord r{};
        r.resolution_label = "1080p Full HD";
        r.width = t_width;
        r.height = t_height;
        r.tile_size = ts;
        r.operation = "TileSize Sweep (tau=" + std::to_string(tile_tau).substr(0, 4) + ")";
        r.backend = "AVX2";
        r.precision_mode = "Adaptive";
        r.sensitivity_threshold = tile_tau;
        r.fp16_tile_pct = a_res.decision.fp16_percentage;
        r.fp32_tile_pct = a_res.decision.fp32_percentage;
        r.timing = ad_t;
        r.perf = ad_p;
        r.estimated_memory_traffic_bytes_per_pix = 48.0;
        r.estimated_arithmetic_intensity = 2.229;
        tile_records.push_back(r);

        std::cout << std::left
                  << std::setw(12) << (std::to_string(ts) + "x" + std::to_string(ts))
                  << std::setw(14) << total_tiles_count
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << sens_t.median_ms
                  << std::setw(14) << dec_t.median_ms
                  << std::setw(14) << ad_t.median_ms
                  << std::fixed << std::setprecision(1)
                  << std::setw(12) << (std::to_string(a_res.decision.fp16_percentage).substr(0, 4) + "%")
                  << std::setw(12) << (std::to_string(a_res.decision.fp32_percentage).substr(0, 4) + "%");
        if (ad_p.available) {
            std::cout << (std::to_string(static_cast<int>(ad_p.cache_miss_rate() * 100.0)) + "%\n");
        } else {
            std::cout << "N/A\n";
        }
    }

    // Write CSV reports
    const std::string bench_csv_path = csv_dir + "/benchmark_results.csv";
    const std::string tile_csv_path = csv_dir + "/tile_analysis.csv";

    write_csv(bench_csv_path, all_records);
    write_csv(tile_csv_path, tile_records);

    std::cout << "\n========================================================================================\n"
              << "BENCHMARK SUMMARY & ARTIFACT GENERATION\n"
              << "========================================================================================\n"
              << "Benchmark CSV written:     " << bench_csv_path << " (" << all_records.size() << " records)\n"
              << "Tile Analysis CSV written: " << tile_csv_path << " (" << tile_records.size() << " records)\n"
              << "Next step: Run Python analysis scripts to generate publication-grade performance charts.\n"
              << "========================================================================================\n";

    return 0;
}
