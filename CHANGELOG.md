# Changelog

All notable changes to **Mirage** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
