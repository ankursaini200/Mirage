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

- **Current Milestone**: Phase 9 — Research Experiments, Ablation Studies and Pareto Analysis (v0.9.0-alpha)
- **Active Capabilities**:
  - Modular C++20 CMake build infrastructure with preset configurations.
  - Decoupled `mirage_core` engine library and `mirage` command-line executable.
  - Safe, multi-tier CPU feature detection (x86_64 CPUID, OSXSAVE, and XCR0 verification for AVX2 and FMA without executing illegal instructions).
  - Validated image loader supporting JPG, JPEG, PNG, and BMP with corrupted and empty file rejection.
  - Color space conversion between BGR, RGB, and scientific floating-point CIELAB ($L^* \in [0, 100]$, $a^*, b^* \in [-127, 127]$).
  - Spatial tile manager with non-divisible boundary clamping, zero-copy ROI views, and lossless reconstruction.
  - Uniform FP32 reference enhancement engine implementing additive brightness, midpoint contrast, Rec. 709 luminance-preserving saturation, and power-law gamma correction.
  - Fixed-precision computation framework supporting:
    - **Uniform FP32**: Bit-exact reference implementation.
    - **Uniform FP16**: Emulated IEEE 754 binary16 arithmetic (`mirage::core::Half`) with round-to-nearest-even (RNTE), normal, subnormal, signed zero, and overflow/underflow handling.
    - **Configurable Mixed Precision**: Per-operation precision assignment (`MixedPrecisionPolicy`).
  - **Tile-Based Adaptive Precision Decision Engine (`mirage::precision::AdaptivePrecisionEngine`)**:
    - Sensitivity-guided precision decision policy: assigns FP32 for tiles with $\text{score} \ge \tau$, and FP16 for tiles with $\text{score} < \tau$.
    - Configurable sensitivity cutoff threshold $\tau \in [0.0, 1.0]$ (default $\tau = 0.04$).
    - Tile-level dispatch processing each spatial tile in its assigned numerical precision.
    - Isolated execution timing strictly measuring the adaptive pipeline.
    - Precision distribution reporting (FP32/FP16 tile counts and percentages).
  - **SIMD Optimization & Verified Feature Safety (`mirage::simd`)**:
    - Dynamic runtime dispatch: `--backend auto|scalar|avx2` with safe multi-feature validation requiring both AVX2 and FMA ISA extensions.
    - Isolated compilation: only `src/simd/avx2_kernels.cpp` compiled with architecture flags (`-mavx2 -mfma`), avoiding global vector instruction emission.
    - AVX2 FP32 Brightness and Contrast kernels (8 floats wide) with zero-cost scalar epilogue handling.
    - Fused Brightness + Contrast kernel processing in registers, cutting DRAM memory bandwidth in half (achieving **1.89x–1.93x** speedup on 1080p/4K and **1.86x** on cache-resident tiles).
    - AVX2 Saturation kernel processing 8 interleaved BGR pixels (24 floats) per iteration using `_mm256_permutevar8x32_ps` and `_mm256_blend_ps` (1.36x speedup on 1080p).
    - Exact transcendental preservation: Power-law Gamma correction deliberately retained as scalar reference to prevent IEEE 754 numerical deviation.
  - **Hardware Performance & Roofline Analysis (`docs/performance-analysis.md`)**:
    - Direct in-process Linux `perf_event_open()` counter measurement (CPU cycles, instructions, IPC, cache references, cache misses, branch mispredictions).
    - Roofline-style model and arithmetic intensity classification ($\mathcal{I} = \text{FLOPs}/\text{Byte}$), proving standalone Brightness/Contrast are memory-bound ($\mathcal{I} < 0.38$) and explaining why fusion delivers $1.9\times$ speedup while standalone vectorization yields negligible gains.
    - Empirical investigation of software FP16 emulation overhead: establishes that emulated half-precision increases instructions by **$11.4\times$** and cycles by **$17.2\times$**, demonstrating that hardware FP16 support is required for adaptive precision speedup.
    - Tile-size analysis demonstrating that $32\times 32$ pixels is optimal for high-definition imagery ($2.8\times$ faster analysis than $8\times 8$).
  - **Phase 9 Research Experiments, Ablation Studies and Pareto Analysis (`docs/research-results.md`)**:
    - Perceptual and numerical error metrics: CIEDE2000 ($\Delta E_{00}$, Sharma et al. 2005 standard), SSIM (channel-split 11x11 Gaussian window), PSNR, MAE, RMSE, and Relative $L_2$.
    - Hypothesis H1 validation: Tested against 30 random seeds at exact budget parity, showing up to **+3.08%** MAE reduction and **+7.41%** $\Delta E_{00}$ reduction over random tile assignment.
    - Hypothesis H4 ablation: Evaluated all 4 features, leave-one-out, and isolated features, uncovering high mutual information between gradient, edge, and contrast, while identifying edge density as the primary linear error predictor ($r = +0.2332, p = 1.49 \times 10^{-23}$).
    - Hypothesis H2/H3 Pareto analysis: Multi-dimensional sweeps across tile sizes ($8\times 8$ to $64\times 64$) and thresholds ($[0.00, 1.00]$), establishing Pareto non-dominated sets and determining that $32\times 32$ and $64\times 64$ provide the best tradeoff between dispatch overhead and spatial selectivity.
    - Curated 7-modality evaluation dataset and reproducible pipeline orchestrator (`benchmarks/run_experiments.sh`).
    - Publication-grade automated charting pipeline producing 6 high-resolution plots via Cairo.
  - Full automated unit test suite with 106 tests across 16 test suites and 14 CTest cases (100% pass rate).

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
│       ├── analysis/
│       │   └── sensitivity_analyzer.hpp # Tile-based sensitivity analysis & heatmap
│       ├── core/
│       │   └── half.hpp        # IEEE 754 binary16 arithmetic type & RNTE rounding
│       ├── cpu/
│       │   └── cpu_features.hpp # CPU architecture and feature detection
│       ├── enhancement/
│       │   ├── enhancement_engine.hpp # Multi-precision enhancement pipeline
│       │   └── precision_config.hpp   # Precision modes, policies & error metrics
│       ├── image/
│       │   ├── image_loader.hpp # Validating image I/O and normalization
│       │   ├── color_converter.hpp # BGR, RGB, and scientific CIELAB conversions
│       │   └── tile_manager.hpp # Spatial tiling and image reconstruction
│       ├── precision/
│       │   └── adaptive_precision.hpp # Adaptive precision decision & tile dispatch
│       └── simd/
│           ├── simd_backend.hpp # Backend enum (Auto, Scalar, AVX2) & feature resolution
│           └── avx2_kernels.hpp # AVX2 SIMD vectorized kernel declarations
├── src/                        # Internal implementations
│   ├── main.cpp                # CLI application entry point
│   ├── version.cpp             # Version metadata implementation
│   ├── analysis/
│   │   └── sensitivity_analyzer.cpp # Perceptual metrics, heatmap & CSV export
│   ├── core/
│   │   └── half.cpp            # IEEE 754 binary16 conversion and emulation logic
│   ├── cpu/
│   │   └── cpu_features.cpp    # CPUID and OSXSAVE feature detection
│   ├── enhancement/
│   │   ├── enhancement_engine.cpp # Reference FP32 & FP16 enhancement kernels
│   │   └── precision_config.cpp   # Error metrics and precision configuration
│   ├── image/
│   │   ├── image_loader.cpp    # OpenCV I/O and float normalization
│   │   ├── color_converter.cpp # Color transformation algorithms
│   │   └── tile_manager.cpp    # Tile partitioning & reconstruction logic
│   ├── precision/
│   │   └── adaptive_precision.cpp # Tile precision decision & enhancement pipeline
│   └── simd/
│       ├── simd_backend.cpp    # Backend resolution and validation
│       └── avx2_kernels.cpp    # AVX2 vectorized implementations (-mavx2 isolated)
├── tests/                      # Automated unit and integration test suite
│   ├── CMakeLists.txt
│   ├── test_version.cpp
│   ├── test_cpu_features.cpp
│   ├── test_half.cpp
│   ├── test_image_loader.cpp
│   ├── test_color_converter.cpp
│   ├── test_tile_manager.cpp
│   ├── test_enhancement_engine.cpp
│   ├── test_precision_engine.cpp
│   ├── test_sensitivity_analyzer.cpp
│   ├── test_adaptive_precision.cpp
│   └── test_simd.cpp           # SIMD equivalence, tail handling & dispatch tests
├── benchmarks/                 # Microbenchmark suite
│   ├── CMakeLists.txt
│   ├── benchmark_cpu.cpp       # Google Benchmark CPU test
│   └── benchmark_runner.cpp    # Standalone high-precision benchmarking suite
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

# Enhance an image using the uniform FP32 baseline engine (with automatic AVX2 dispatch)
./build/release/mirage --enhance sample_image.png --output output/enhanced_fp32.png \
  --precision fp32 --brightness 0.1 --contrast 1.2 --saturation 1.3 --gamma 1.0 --backend auto

# Enhance forcing portable scalar reference backend
./build/release/mirage --enhance sample_image.png --output output/enhanced_scalar.png \
  --precision fp32 --brightness 0.1 --contrast 1.2 --saturation 1.3 --gamma 1.0 --backend scalar

# Enhance requesting AVX2 acceleration explicitly
./build/release/mirage --enhance sample_image.png --output output/enhanced_avx2.png \
  --precision fp32 --brightness 0.1 --contrast 1.2 --saturation 1.3 --gamma 1.0 --backend avx2

# Enhance an image using uniform FP16 emulated arithmetic (with error reporting vs FP32)
./build/release/mirage --enhance sample_image.png --output output/enhanced_fp16.png \
  --precision fp16 --brightness 0.1 --contrast 1.2 --saturation 1.3 --gamma 1.0

# Enhance an image using mixed precision (FP16 color operations + FP32 contrast/gamma)
./build/release/mirage --enhance sample_image.png --output output/enhanced_mixed.png \
  --precision mixed --brightness 0.1 --contrast 1.2 --saturation 1.3 --gamma 1.0

# Enhance an image using sensitivity-guided tile-based adaptive precision with AVX2
./build/release/mirage --enhance sample_image.png --output output/enhanced_adaptive.png \
  --precision adaptive --tile-size 32 --sensitivity-threshold 0.04 \
  --brightness 0.1 --contrast 1.2 --saturation 1.3 --gamma 1.0 --backend avx2

# Run high-resolution performance benchmark suite (Small, 1080p, 4K)
./build/release/benchmarks/mirage_benchmarks

# Analyze tile-based image sensitivity and generate heatmap visualization and CSV report
./build/release/mirage --analyze-sensitivity sample_image.png --tile-size 32 \
  --output output/sensitivity_heatmap.png --csv output/sensitivity_report.csv
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
