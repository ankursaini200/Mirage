# Mirage: Adaptive Precision for Software-Based Color Image Enhancement

[![Mirage CI](https://github.com/your-username/Mirage/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/Mirage/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/Language-C%2B%2B20-blue.svg)](https://en.cppreference.com/w/cpp/20)

> **Research Disclaimer**: **Mirage is an experimental research project.** Performance improvements, energy efficiency benefits, and quality tradeoffs are active subjects of architectural investigation and **are not yet established**. Performance claims will be reported only when validated through reproducible hardware counter measurements and microbenchmarks.

---

## 1. Project Introduction

**Mirage** is an open-source, research-oriented C++20 image processing engine investigating sensitivity-guided tile-based adaptive numerical precision for color image enhancement. 

Rather than processing an entire image uniformly at single precision (FP32) or fixed-point (INT8), Mirage evaluates whether decomposing an image into spatial tiles and assigning localized precision based on sensitivity metrics (e.g., luminance gradient, chromatic complexity, and textural entropy) can optimize memory bandwidth, cache residency, and execution latency while preserving visual fidelity.

## 2. Research Motivation

In modern Computer Organization and Architecture (COA), image processing algorithms face memory walls and energy bottlenecks:
- **Uniform Single-Precision (FP32)** provides pristine numerical precision and prevents rounding errors or gamut clipping, but exerts high pressure on memory bandwidth, cache footprints, and register allocation.
- **Uniform Reduced-Precision (FP16 / INT8)** doubles or quadruples SIMD vector density and reduces memory traffic, but frequently induces visible banding, contouring artifacts, and color shifts in smooth gradients and high-dynamic-range zones.
- **Sensitivity-Guided Adaptive Precision** attempts to dynamically balance this tradeoff by assigning precision where human visual perception is most critical (smooth gradients, skin tones, edge boundaries) while utilizing high-throughput, low-precision SIMD pathways for high-frequency or masked regions.

Mirage investigates this hypothesis systematically on CPU architectures with hardware performance counter validation.

## 3. Current Development Status

- **Current Milestone**: Phase 2 — Image Input, Color Conversion, and Tile Management
- **Active Capabilities**:
  - Modular C++20 CMake build infrastructure with preset configurations.
  - Decoupled `mirage_core` engine library and `mirage` command-line executable.
  - Safe, multi-tier CPU feature detection (x86_64 CPUID, OSXSAVE, and XCR0 verification for AVX2 without executing illegal instructions).
  - Validated image loader supporting JPG, JPEG, PNG, and BMP with corrupted file and empty file rejection.
  - Color space conversion between BGR, RGB, and scientific floating-point CIELAB ($L^* \in [0, 100]$, $a^*, b^* \in [-127, 127]$).
  - Spatial tile manager with non-divisible boundary clamping, zero-copy ROI views, and lossless reconstruction.
  - Tile grid visualization overlays with configurable tile dimensions and border annotations.
  - Automated unit test suite with 34 tests executed across Google Test and CTest.
- **Future Milestones**: Sensitivity analysis (Phase 3), multi-precision kernels (Phase 4), AVX2 SIMD acceleration & runtime dispatch (Phase 5), and benchmarks with `perf` profiling (Phase 6).

## 4. Technology Stack

- **Core Engine**: C++20 (GCC 11+, Clang 14+)
- **Build System**: CMake 3.20+ (with `CMakePresets.json`)
- **Computer Vision & Image I/O**: OpenCV 4 / 5 (`core`, `imgproc`, `imgcodecs`)
- **Testing**: Google Test, CTest
- **Benchmarking**: Google Benchmark (optional dependency)
- **Research Evaluation**: Python 3.10+, NumPy (for offline analysis only)
- **CI / Automation**: GitHub Actions

## 5. Project Architecture

Mirage maintains strict separation between interface definitions, core library logic, and user-facing entry points:

```text
Mirage/
├── CMakeLists.txt              # Top-level modern CMake build definitions
├── CMakePresets.json           # Standardized build and test presets
├── include/                    # Public API headers
│   └── mirage/
│       ├── version.hpp         # Versioning and compiler identification
│       ├── cpu/
│       │   └── cpu_features.hpp # CPU architecture and feature detection
│       └── image/
│           ├── image_loader.hpp # Validating image I/O and normalization
│           ├── color_converter.hpp # BGR, RGB, and scientific CIELAB conversions
│           └── tile_manager.hpp # Spatial tiling and image reconstruction
├── src/                        # Internal implementations
│   ├── main.cpp                # CLI application entry point
│   ├── version.cpp             # Version metadata implementation
│   ├── cpu/
│   │   └── cpu_features.cpp    # CPUID and OSXSAVE feature detection
│   └── image/
│       ├── image_loader.cpp    # OpenCV I/O and float normalization
│       ├── color_converter.cpp # Color transformation algorithms
│       └── tile_manager.cpp    # Tile partitioning & reconstruction logic
├── tests/                      # Automated unit and integration test suite
│   ├── CMakeLists.txt
│   ├── test_version.cpp
│   ├── test_cpu_features.cpp
│   ├── test_image_loader.cpp
│   ├── test_color_converter.cpp
│   └── test_tile_manager.cpp
├── benchmarks/                 # Microbenchmark suite (Google Benchmark)
│   └── CMakeLists.txt
├── docs/                       # Architectural & development documentation
│   ├── architecture.md
│   └── development.md
├── analysis/                   # Offline evaluation & research notebooks
│   └── README.md
└── .github/                    # CI workflows and community templates
    ├── workflows/ci.yml
    └── ISSUE_TEMPLATE/
```

## 6. Installation Instructions

### Prerequisites (Ubuntu / Debian)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libopencv-dev libgtest-dev
```

### Prerequisites (Arch Linux)
```bash
sudo pacman -S --needed base-devel cmake ninja opencv gtest
```

## 7. Build Instructions

### Standard CMake Workflow

```bash
# Clone the repository
git clone https://github.com/your-username/Mirage.git
cd Mirage

# Configure and compile in Release mode
cmake -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j$(nproc)

# Configure and compile in Debug mode
cmake -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j$(nproc)
```

### Using CMake Presets

```bash
# Debug configuration
cmake --preset debug
cmake --build --preset debug

# Release configuration
cmake --preset release
cmake --build --preset release
```

### Running the Mirage Executable

```bash
# Display version and hardware information
./build/release/mirage --version
./build/release/mirage --info
./build/release/mirage --help

# Inspect an image
./build/release/mirage --image-info sample_image.png

# Inspect tile partitioning for non-divisible dimensions (e.g. 32x32 tiles)
./build/release/mirage --tile-info sample_image.png --tile-size 32

# Render and save tile visualization overlay
./build/release/mirage --visualize-tiles sample_image.png --tile-size 32 --output output/tile_vis.png
```

## 8. Testing Instructions

All automated tests are orchestrated via CTest:

```bash
# Run tests against Release build
ctest --test-dir build/release --output-on-failure

# Run tests against Debug build
ctest --test-dir build/debug --output-on-failure

# Run GoogleTest binary directly
./build/release/tests/mirage_tests
```

## 9. Contribution Guidelines

Contributions are welcome! Please consult [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, git branching models, and testing policies before opening a pull request. All community interactions are governed by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 10. License Information

Mirage is released under the [MIT License](LICENSE). Copyright &copy; 2026 Mirage Contributors.
Third-party dependencies and notices are documented in compliance with their respective open-source licenses.
