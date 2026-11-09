# Changelog

All notable changes to **Mirage** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
