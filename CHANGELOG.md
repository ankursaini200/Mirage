# Changelog

All notable changes to **Mirage** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-alpha] - 2026-11-09

### Added
- **Image Loader Subsystem (`mirage::image::ImageLoader`)**:
  - Validated loading for JPG, JPEG, PNG, and BMP formats.
  - Strict input verification: checks file existence, non-zero file sizes, and valid magic headers.
  - Rejection of corrupted and empty images with custom exception hierarchy (`ImageLoadException`, `FileNotFoundException`, `UnsupportedFormatException`, `CorruptedImageException`, `EmptyImageException`).
  - Floating-point normalization converting `CV_8UC3` in $[0, 255]$ to `CV_32FC3` in $[0.0, 1.0]$ and inverse denormalization.
- **Color Conversion Engine (`mirage::image::ColorConverter`)**:
  - Transformations between BGR and RGB with channel integrity validation.
  - Scientific floating-point CIELAB conversions (`bgr_to_lab`, `lab_to_bgr`) operating on normalized $[0.0, 1.0]$ matrices and producing $L^* \in [0.0, 100.0]$, $a^*, b^* \in [-127.0, +127.0]$.
  - Clear architectural distinction between scientific floating-point CIELAB and 8-bit quantized Lab.
- **Spatial Tile Manager (`mirage::image::TileManager`)**:
  - Configurable spatial tile decomposition (8x8, 16x16, 32x32, 64x64, and arbitrary dimensions).
  - Robust non-divisible boundary handling preventing out-of-bounds memory accesses.
  - Zero-copy tile views referencing the parent matrix with identical row pitch.
  - Lossless full-image reconstruction (`reconstruct_image()`) verified pixel-by-pixel.
  - Tile grid visualization overlays with tile boundary rendering and ID labels.
- **CLI Commands**:
  - `--image-info <image_path>`: inspects image dimensions, channels, data types, and color space.
  - `--tile-info <image_path> --tile-size <size>`: displays grid dimensions, total/complete/partial tile counts, and coordinates.
  - `--visualize-tiles <image_path> --tile-size <size> [--output <path>]`: renders and saves annotated tile grid images.
- **Automated Test Suite**:
  - 24 new unit tests added across `test_image_loader.cpp`, `test_color_converter.cpp`, and `test_tile_manager.cpp`, bringing total unit tests to 34.

## [0.1.0-alpha] - 2026-11-09

### Added
- **Core Architecture**:
  - Reusable `mirage_core` C++20 engine library target.
  - Public interface headers under `include/mirage/` separated from implementation in `src/`.
  - Versioning and compiler identification utilities (`mirage/version.hpp`).
- **CPU Capability Detection**:
  - `mirage::cpu::CpuFeatures` module detecting host architectures (x86_64, x86_32, ARM64, ARM32, RISC-V, PowerPC).
  - Multi-tier safe AVX2 detection via x86 CPUID leaves 0, 1, and 7, with OSXSAVE and XCR0 verification.
  - Zero illegal instruction executions on non-AVX2 hardware.
- **CLI Application**:
  - `mirage` executable with `--help`, `--version`, and `--info` flags.
- **Build System**:
  - Modern CMake (>= 3.20) build configuration with `CMakePresets.json` (Debug and Release presets).
  - Strict compiler warning flags for GCC, Clang, and MSVC.
  - Target-based optional dependencies for OpenCV and Google Benchmark.
  - CTest test integration.
- **Open-Source Governance & CI**:
  - MIT License (2026 Mirage Contributors).
  - `README.md` with research motivation, architecture overview, and build guides.
  - `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md`.
  - GitHub Actions CI workflow building on Ubuntu with GCC and Clang matrix.
  - Issue templates (`bug_report.md`, `feature_request.md`) and Pull Request template.
  - Automated unit test suite using GoogleTest (`test_version.cpp`, `test_cpu_features.cpp`).
