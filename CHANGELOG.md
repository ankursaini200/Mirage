# Changelog

All notable changes to **Mirage** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0-alpha] - 2026-11-09

### Added
- **Tile-Based Adaptive Precision Decision Engine (`mirage::precision::AdaptivePrecisionEngine`)**:
  - Precision decision policy evaluating tile sensitivity scores from Phase 5 against a configurable threshold:
    - $\text{score} \ge \text{threshold} \implies \text{FP32}$
    - $\text{score} < \text{threshold} \implies \text{FP16}$
  - Configurable sensitivity threshold in $[0.0, 1.0]$ with default $\tau = 0.04$.
  - Data structures: `TilePrecisionAssignment`, `AdaptivePrecisionDecision`, and `AdaptiveEnhancementResult`.
  - Tile-based multi-precision dispatch: processes each spatial tile ROI in its assigned numerical precision (FP32 or emulated FP16) and losslessly reconstructs the output image.
  - Full support for non-divisible dimensions and partial boundary tiles.
  - Isolated adaptive execution timing strictly measuring the adaptive pipeline (sensitivity analysis, precision decision, per-tile enhancement, image reconstruction) and excluding reference processing and error analysis.
- **Accuracy Evaluation & Numerical Error Metric Extension**:
  - Added Relative $L_2$ Error ($\|I_{\text{target}} - I_{\text{ref}}\|_2 / (\|I_{\text{ref}}\|_2 + \epsilon)$) with $\epsilon = 10^{-10}$ to prevent division by zero on zero-reference images.
- **CLI Enhancements**:
  - Extended `--enhance` with `--precision adaptive`.
  - Added `--sensitivity-threshold <val>` (default: `0.04`) and `--tile-size <size>` (default: `32`).
  - Comprehensive stdout summary reporting precision distribution breakdown (total tiles, FP32 count/percentage, FP16 count/percentage), sensitivity threshold, isolated adaptive runtime, and comparative error metrics vs. the FP32 ground truth reference.
- **Automated Test Suite Expansion**:
  - Added `tests/test_adaptive_precision.cpp` containing 17 unit tests covering precision assignment rules, threshold edge cases ($0.0$ and $1.0$), invalid thresholds, determinism, all-FP32 identity, all-FP16 match, mixed precision bounds, partial boundary tiles, small images, uniform images, dimension/type preservation, and relative $L_2$ zero-reference handling.
  - Added CLI integration tests `cli_test_enhance_adaptive_invalid_threshold` and `cli_test_enhance_adaptive_invalid_tile_size`.
  - Total automated unit tests increased to 83 across 10 test suites; 13 CTest test cases passing (100%).

## [0.5.0-alpha] - 2026-11-09

### Added
- **Tile-Based Sensitivity Analyzer Subsystem (`mirage::analysis::SensitivityAnalyzer`)**:
  - Implemented 4 standardized, normalized sensitivity features per spatial tile:
    - **Color Variation ($C$)**: Standard deviation of chromaticity channels ($a^*, b^*$) in scientific CIELAB space, normalized to theoretical maximum $\sigma_{\text{color, max}} = \sqrt{127^2 + 127^2} \approx 179.605$.
    - **Gradient Magnitude ($G$)**: Mean spatial gradient magnitude using Sobel 3x3 on $L^*$ channel, normalized to theoretical maximum $G_{\max} = \sqrt{400^2 + 400^2} \approx 565.685$.
    - **Edge Density ($E$)**: Proportion of tile pixels exceeding a fixed, documented Sobel gradient threshold ($T_{\text{edge}} = 30.0$ on $L^*$).
    - **Local Contrast ($L$)**: Luminance standard deviation of $L^*$ channel, normalized via Popoviciu's bound $\sigma_{L^*, \max} = 50.0$.
  - Combined weighted sensitivity score ($S = w_c C + w_g G + w_e E + w_l L$) with configurable weights (`SensitivityWeights`, default $0.25$ each summing to $1.0$).
  - Strict image-independent feature normalization avoiding arbitrary scaling across different image inputs.
  - Safe handling of zero-variance tiles, flat surfaces, and constant images yielding $S = 0.0$.
  - Seamless integration with `mirage::image::TileManager` supporting arbitrary nominal tile sizes (8, 16, 32, 64) and partial boundary tiles.
- **Sensitivity Heatmap Visualization Engine**:
  - Overlay rendering blending original base image with perceptually uniform `COLORMAP_TURBO` colormap.
  - Interactive grid outlining with tile ID text annotations.
  - Bottom colorbar legend detailing score mapping ($0.0$ Low to $1.0$ High).
  - Persistence via `save_heatmap(path, base_image, report)`.
- **CSV Data Export**:
  - Serialization of per-tile metrics (`tile_id, x, y, width, height, is_partial, color_variation, gradient_magnitude, edge_density, local_contrast, sensitivity_score`) via `export_csv(path, report)`.
- **CLI Extension**:
  - Added `mirage --analyze-sensitivity <image> --tile-size <size> [--output <heatmap.png>] [--csv <report.csv>]`.
  - Comprehensive stdout summary reporting dimensions, tile breakdown, min/max/avg sensitivity, and feature averages.
- **Automated Test Expansion**:
  - Added `tests/test_sensitivity_analyzer.cpp` with 13 unit tests covering constant images, grayscale zero chromaticity, high color variation, steep gradients, dense edge bars, low contrast, bounded ranges in $[0.0, 1.0]$, weight validation, partial boundary tiles, unique tile IDs, deterministic execution, heatmap generation, and CSV report export.
  - Added CLI integration tests for missing image and missing tile size parameters.
  - Total automated unit tests increased to 66 across 9 suites; 11 CTest test cases passing (100%).

## [0.4.0-alpha] - 2026-11-09

### Added
- **Fixed-Precision Computation Engine**:
  - IEEE 754 binary16 arithmetic type (`mirage::core::Half`) with exact round-to-nearest, ties-to-even (RNTE) rounding.
  - Full binary16 domain handling: normal numbers ($2^{-14}$ to 65504), subnormal numbers down to $2^{-24} \approx 5.96 \times 10^{-8}$, signed zeros ($+0.0$ and $-0.0$ preserving sign bit), overflow to $\pm\infty$, underflow, and quiet/signaling NaNs.
  - Emulated binary16 arithmetic operators (`+`, `-`, `*`, `/`) with explicit intermediate boundary rounding.
  - Multi-precision operational support in `mirage::enhancement::EnhancementEngine`:
    - `PrecisionMode`: `FP32` (uniform single-precision float reference), `FP16` (uniform binary16 emulated arithmetic), and `Mixed` (per-operation configurable precision).
    - `MixedPrecisionPolicy`: individual precision assignment across brightness, contrast, saturation, and gamma stages.
    - Specialized FP16 operational kernels (`adjust_brightness_fp16`, `adjust_contrast_fp16`, `adjust_saturation_fp16`, `adjust_gamma_fp16`).
    - Unified execution method `process_with_precision(src, dst, config, mode, policy)`.
- **Quantitative Error Analysis Facility (`mirage::enhancement::compute_error_metrics`)**:
  - Maximum Absolute Error (MAE / $L_\infty$ norm difference).
  - Mean Absolute Error (MAE / $L_1$ norm average).
  - Root Mean Square Error (RMSE / $L_2$ norm).
  - Peak Signal-to-Noise Ratio (PSNR in dB, referenced to peak signal 1.0).
  - Relative Error ($L_2$ error relative to reference $L_2$ energy).
- **CLI Enhancements**:
  - Added `--precision <fp32|fp16|mixed>` option to `--enhance`.
  - Automated quantitative error metric calculation and reporting against the FP32 reference baseline during non-FP32 CLI enhancement runs.
- **Automated Test Expansion**:
  - Added `tests/test_half.cpp` (7 unit tests for binary16 bit representation, subnormals, RNTE rounding, signed zeros, overflow/underflow, NaNs/Infs, and arithmetic operations).
  - Added `tests/test_precision_engine.cpp` (5 unit tests verifying FP32 mode exact identity, FP16 bounded error with PSNR > 50 dB, mixed precision behavior, determinism, and precision mode string parsing).
  - Added integration test `cli_test_enhance_invalid_precision` in CTest.
  - Total automated unit tests increased to 53 across 8 suites; 9 CTest test cases passing.

## [0.3.0-alpha] - 2026-11-09

### Added
- **Baseline Uniform FP32 Enhancement Engine (`mirage::enhancement::EnhancementEngine`)**:
  - Additive brightness adjustment with boundary clamping ($I_{\text{out}} = \text{clamp}(I_{\text{in}} + \delta, 0.0, 1.0)$).
  - Contrast scaling around neutral midpoint ($I_{\text{out}} = \text{clamp}(M + \alpha(I_{\text{in}} - M), 0.0, 1.0)$, $M = 0.5$).
  - Saturation adjustment using Rec. 709 luminance ($Y = 0.114B + 0.587G + 0.299R$), with proven neutral grayscale invariance and pure grayscale output when saturation is zero.
  - Power-law gamma transformation ($I_{\text{out}} = \text{clamp}(I_{\text{in}}^{1/\gamma}, 0.0, 1.0)$) with domain safety checks.
  - Strict parameter validation (`EnhancementConfig::validate()`) rejecting non-finite values, negative contrast/saturation, and non-positive gamma.
  - Fixed, deterministic processing pipeline: $\text{Brightness} \to \text{Contrast} \to \text{Saturation} \to \text{Gamma}$.
- **CLI Enhancement Command**:
  - Added `mirage --enhance <input> --output <output> [--brightness <val>] [--contrast <val>] [--saturation <val>] [--gamma <val>]`.
- **Automated Test Suite Expansion**:
  - Added 7 focused unit tests in `test_enhancement_engine.cpp` verifying brightness, contrast, saturation, gamma, parameter validation, bounds clamping, and deterministic execution, bringing total test count to 41.

## [0.2.0-alpha] - 2026-11-09

### Added
- **Image Loader Subsystem (`mirage::image::ImageLoader`)**:
  - Validated loading for JPG, JPEG, PNG, and BMP formats.
  - Rejection of corrupted and empty images with custom exception hierarchy.
  - Floating-point normalization converting `CV_8UC3` to `CV_32FC3` in $[0.0, 1.0]$.
- **Color Conversion Engine (`mirage::image::ColorConverter`)**:
  - Transformations between BGR and RGB.
  - Scientific floating-point CIELAB conversions (`bgr_to_lab`, `lab_to_bgr`).
- **Spatial Tile Manager (`mirage::image::TileManager`)**:
  - Configurable spatial tile decomposition (8x8, 16x16, 32x32, 64x64).
  - Non-divisible boundary clamping preventing out-of-bounds memory accesses.
  - Zero-copy tile views and lossless full-image reconstruction.
  - Tile grid visualization overlays.
- **CLI Commands**:
  - `--image-info`, `--tile-info`, `--visualize-tiles`.

## [0.1.0-alpha] - 2026-11-09

### Added
- **Core Architecture & CPU Detection**:
  - Reusable `mirage_core` C++20 library and `mirage` CLI client.
  - Multi-tier safe AVX2 detection via CPUID and OSXSAVE/XCR0 verification.
  - Modern CMake (>= 3.20) build configuration with presets and compiler warning flags.
  - MIT License, Open-source governance documents, and GitHub Actions CI workflow.
