# Developer Guide

This document describes build procedures, compiler flags, testing workflows, and module usage for **Mirage**.

## Prerequisites

- **C++ Compiler**: Supporting C++20 standard:
  - GCC 11+ (GCC 13+ recommended)
  - Clang 14+ (Clang 16+ recommended)
- **CMake**: Minimum version 3.20 (CMake 3.22+ recommended)
- **OpenCV**: OpenCV 4 or OpenCV 5 (`core`, `imgproc`, `imgcodecs`)
- **Testing**: Google Test (automatically detected via CMake if present)
- **Build System**: Ninja or GNU Make
- **Optional**: Google Benchmark

## Build Workflows

### Manual Configuration

#### Release Build (Default)
```bash
cmake -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j$(nproc)
```

#### Debug Build
```bash
cmake -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j$(nproc)
```

### Using CMake Presets

Mirage ships with standardized presets in `CMakePresets.json`:

```bash
# Debug preset
cmake --preset debug
cmake --build --preset debug

# Release preset
cmake --preset release
cmake --build --preset release
```

## Running the Mirage CLI

```bash
# General Information
./build/release/mirage --info
./build/release/mirage --version
./build/release/mirage --help

# Image Metadata & Tiling
./build/release/mirage --image-info sample_image.png
./build/release/mirage --tile-info sample_image.png --tile-size 32
./build/release/mirage --visualize-tiles sample_image.png --tile-size 32 --output output/tile_vis.png

# Image Enhancement with Configurable Precision
# 1. Uniform FP32 Reference Mode:
./build/release/mirage --enhance sample_image.png --output output/enhanced_fp32.png \
  --precision fp32 --brightness 0.1 --contrast 1.2 --saturation 1.3 --gamma 1.0

# 2. Uniform FP16 Emulated Mode (with automated error metrics vs FP32):
./build/release/mirage --enhance sample_image.png --output output/enhanced_fp16.png \
  --precision fp16 --brightness 0.1 --contrast 1.2 --saturation 1.3 --gamma 1.0

# 3. Mixed Precision Mode:
./build/release/mirage --enhance sample_image.png --output output/enhanced_mixed.png \
  --precision mixed --brightness 0.1 --contrast 1.2 --saturation 1.3 --gamma 1.0

# 4. Sensitivity-Guided Adaptive Precision Mode:
./build/release/mirage --enhance sample_image.png --output output/enhanced_adaptive.png \
  --precision adaptive --tile-size 32 --sensitivity-threshold 0.04 \
  --brightness 0.1 --contrast 1.2 --saturation 1.3 --gamma 1.0

# Tile Sensitivity Analysis
./build/release/mirage --analyze-sensitivity sample_image.png --tile-size 32 \
  --output output/sensitivity_heatmap.png --csv output/sensitivity_report.csv
```

### Adaptive Precision Enhancement CLI Output

```text
============================================================
              Mirage Image Enhancement
============================================================
Input file:            sample_image.png
Output file:           output/enhanced_adaptive.png
Image dimensions:      190 x 130
Precision Mode:        Adaptive (Sensitivity-Guided Precision Selection)
Sensitivity Threshold: 0.0400
Tile Dimensions:       32 x 32
Brightness:            0.100000
Contrast:              1.200000
Saturation:            disabled
Gamma:                 disabled
Adaptive Runtime:      87.60 ms

Tile Precision Distribution:
  Total Tile Count:      30
  FP32 Tiles:            20 (66.67 %)
  FP16 Tiles:            10 (33.33 %)

Numerical Error Analysis (vs FP32 Ground Truth Reference):
Max Absolute Error:  0.000751
Mean Absolute Error: 0.000026
RMSE:                0.000087
PSNR:                81.18 dB
Relative L2 Error:   0.0131 %
Mean Relative Error: 0.0071 %
```

### Sensitivity Analysis CLI Output

```text
============================================================
           Mirage Tile-Based Sensitivity Analysis           
============================================================
Image filename:          sample_image.png
Image dimensions:        190 x 130 (3 channels)
Tile dimensions:         32 x 32
Grid dimensions:         6 columns x 5 rows
Total tile count:        30 (20 complete, 10 partial)
Execution time:          1.42 ms
------------------------------------------------------------
Sensitivity Summary (Scores in [0.0, 1.0]):
  Minimum Sensitivity:   0.006665
  Maximum Sensitivity:   0.056704
  Average Sensitivity:   0.038276
------------------------------------------------------------
Feature Averages (Weights: C=0.25, G=0.25, E=0.25, L=0.25):
  Color Variation (C):   0.057488
  Gradient Magnitude (G):0.007491
  Edge Density (E):      0.000000
  Local Contrast (L):    0.088123
============================================================
Heatmap visualization saved to: output/sensitivity_heatmap.png
Tile sensitivity CSV report saved to: output/sensitivity_report.csv
```

### Numerical Error Metrics Output

When running with `--precision fp16` or `--precision mixed`, the CLI automatically computes the quantitative error metrics against the bit-exact FP32 reference:

```text
============================================================
              Mirage Image Enhancement
============================================================
Input file:         sample_image.png
Output file:        output/enhanced_fp16.png
Image dimensions:   512x512 (3 channels)
Precision mode:     FP16
Pipeline sequence:  Brightness -> Contrast -> Saturation -> Gamma
Execution status:   SUCCESS (Image saved)
------------------------------------------------------------
Numerical Error Analysis (vs FP32 Reference):
  Max Absolute Error:     0.00195312
  Mean Absolute Error:    0.00032145
  Root Mean Square Error: 0.00041230
  PSNR:                   67.69 dB
  Relative Error:         0.00078234
============================================================
```

## Running Tests

All unit tests and CLI integration tests are integrated with CTest:

```bash
# Run tests in the release build
ctest --test-dir build/release --output-on-failure

# Run tests in the debug build
ctest --test-dir build/debug --output-on-failure

# Run unit tests binary directly for detailed GoogleTest output
./build/release/tests/mirage_tests
```

## Test Suite Structure

The test suite in `tests/` covers 83 automated tests across 10 test suites:
1. `VersionTest`: Semantic versioning, C++ standard detection, and compiler string reporting.
2. `CpuFeaturesTest`: CPUID querying, AVX2 availability, OSXSAVE validation, and string formatting.
3. `HalfPrecisionTest`:
   - Exact IEEE 754 binary16 bit conversions for normal and special values.
   - Subnormal numbers down to $2^{-24} \approx 5.96 \times 10^{-8}$.
   - Round-to-nearest, ties-to-even (RNTE) rounding at exact tie thresholds.
   - Signed zeros ($+0.0$ and $-0.0$ with distinct sign bit preservation).
   - Numerical overflow to $\pm\infty$ and underflow to zero.
   - Quiet and signaling NaN preservation.
   - Emulated arithmetic operations (`+`, `-`, `*`, `/`).
4. `ImageLoaderTest`: File existence checks, supported formats (JPG, PNG, BMP), corrupted and empty file handling, and float normalization.
5. `ColorConverterTest`: BGR to RGB channel reordering, RGB to BGR symmetry, scientific CIELAB numerical range validity, known reference colors (black, white), and input validation.
6. `TileManagerTest`: 8x8, 16x16, 32x32, and 64x64 grid divisions, non-divisible image dimensions, boundary tile clamping, single-pixel handling, unique tile IDs, and lossless image reconstruction.
7. `EnhancementEngineTest`:
   - Additive brightness adjustment and upper/lower boundary clamping.
   - Contrast enhancement around neutral midpoint ($M = 0.5$).
   - Luminance-preserving saturation adjustment and grayscale neutral invariance.
   - Power-law gamma transformation.
   - Parameter range validation (rejecting negative contrast, gamma $\le 0$, etc.).
   - Output dimension and `CV_32FC3` type preservation.
   - Deterministic and bit-identical repeated execution.
8. `PrecisionEngineTest`:
   - Exact identity between FP32 mode and reference FP32 pipeline.
   - FP16 bounded numerical error (demonstrating non-zero rounding error, MAE $< 0.005$, and PSNR $> 50$ dB).
   - Mixed-precision policy verification (ensuring mixed error is bounded between FP32 and FP16).
   - Deterministic processing across repeated invocations.
   - String parsing and validation of precision modes.
9. `SensitivityAnalyzerTest`:
   - Configurable feature weight validation and sum-to-one constraint.
   - Uniform-color constant image zero sensitivity across all 4 features.
   - Grayscale zero chromaticity variation ($C = 0$) with active luminance contrast.
   - High color variation vs. grayscale checkerboards.
   - High gradient detection across sharp step edges.
   - Dense edge bars producing elevated edge density and local contrast.
   - Low-contrast image verification.
   - Bounded metric guarantee ($S, C, G, E, L \in [0.0, 1.0]$).
   - Partial boundary tile handling on non-divisible dimensions.
   - Unique sequential tile IDs and complete pixel coverage.
   - Bit-identical deterministic execution across repeated runs.
   - Heatmap matrix generation, colorbar legend rendering, and image file persistence.
   - CSV data serialization and header/row integrity verification.
10. `AdaptivePrecisionTest`:
    - Threshold cutoff verification: score $< \tau \implies \text{FP16}$, score $\ge \tau \implies \text{FP32}$.
    - Boundary thresholds: $\tau = 0.0$ forces all FP32; $\tau = 1.0$ forces all FP16.
    - Invalid threshold handling (negative, $> 1.0$, NaN throwing `AdaptivePrecisionException`).
    - Deterministic precision decisions across repeated invocations.
    - Precision distribution invariant: `fp32_count + fp16_count == total_tiles`.
    - All-FP32 adaptive execution bit-identical to reference FP32 pipeline (0 error, PSNR inf).
    - All-FP16 adaptive execution bit-identical to uniform FP16 pipeline.
    - Mixed FP16/FP32 precision behavior with bounded numerical error.
    - Boundary tile clamping and reconstruction on odd/non-divisible dimensions.
    - Single-tile handling for images smaller than nominal tile size.
    - Uniform flat image handling.
    - Deterministic end-to-end adaptive output.
    - Preservation of image dimensions and `CV_32FC3` format.
    - Relative $L_2$ error calculation and zero-reference stability.
11. `SimdBackendTest`:
    - Parsing backend strings ("auto", "scalar", "avx2") with case-insensitivity and invalid option handling.
    - Backend availability detection and host feature resolution.
12. `Avx2KernelsTest`:
    - Brightness scalar vs. AVX2 numerical bit-identity across positive, negative, and zero deltas.
    - Contrast scalar vs. AVX2 equivalence around standard and non-standard midpoints.
    - Saturation scalar vs. AVX2 numerical parity ($\le 1 \times 10^{-6}$, bounded by single-precision ULP).
    - Fused Brightness + Contrast equivalence vs. sequential scalar execution.
    - Clamping behavior at $0.0$ and $1.0$ under extreme inputs.
    - Non-vector-aligned widths (widths 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 31, 33) and odd dimensions.
    - Small images (1x1, 2x2, 3x5).
    - Partial boundary tile submatrices under zero-copy views.
    - Bit-identical deterministic execution across repeated runs.
13. `EnhancementEngineDispatchTest`:
    - Verification of `Auto`, `Scalar`, and `AVX2` dispatch routes through `EnhancementEngine::process`.
    - Exception handling when an unsupported backend is requested.
14. `AdaptiveEngineDispatchTest`:
    - Verification of adaptive precision pipeline under `backend = AVX2` and `backend = Scalar`.
15. `F16CInvestigationTest`:
    - Verification of hardware `vcvtps2ph` / `vcvtph2ps` conversion against software `Half` rounding.

## Compiler Auto-Vectorization & Assembly Findings

Compiler auto-vectorization analysis was conducted using GCC (`-fopt-info-vec-all`) and Clang (`-Rpass=loop-vectorize -Rpass-analysis=loop-vectorize`):
- **GCC Findings**:
  - Without architecture-specific flags, GCC auto-vectorizes scalar Brightness, Contrast, and Saturation using 128-bit SSE vectors (`16 byte vectors`) with unroll factor 4.
  - However, GCC inserts runtime loop versioning checks to guard against potential pointer aliasing between `src_ptr` and `dst_ptr`.
  - Gamma correction is **not vectorized** by GCC due to `std::powf` (external call without vector math runtime).
- **Clang Findings**:
  - Clang reports `loop not vectorized: Cannot vectorize uncountable loop` for inner pixel loops because matrix column counts are retrieved from class structures.
  - Clang misses auto-vectorization on complex 3-channel interleaved structures without manual unrolling.
- **AVX2 Hand-Vectorized Intrinsics Advantage**:
  - Manual AVX2 vectorization (`src/simd/avx2_kernels.cpp`) guarantees direct 256-bit YMM register utilization (`vaddps`, `vsubps`, `vmulps`, `vminps`, `vmaxps`, `vpermps`, `vblendps`).
  - Eliminates runtime aliasing versioning overhead.
  - Fuses Brightness and Contrast in registers, eliminating 50% of DRAM memory traffic.

## Multi-Compiler Validation

Mirage is developed to maintain full compatibility across GCC and Clang:

```bash
# Testing with Clang
cmake -B build/clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build/clang -j$(nproc)
ctest --test-dir build/clang --output-on-failure

# Testing with GCC
cmake -B build/gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build/gcc -j$(nproc)
ctest --test-dir build/gcc --output-on-failure

# Running Standalone Benchmark Suite with Hardware Performance Counters
./build/release/benchmarks/mirage_benchmarks

# Quick mode (fewer iterations for rapid CI/CD verification)
./build/release/benchmarks/mirage_benchmarks --quick

# Specify custom CSV destination directory
./build/release/benchmarks/mirage_benchmarks --csv-dir benchmarks/results

# Generating publication-grade performance charts
python3 scripts/plot_performance.py

# Running Linux perf CLI profiling wrapper
./scripts/run_perf.sh
```

## Hardware Performance Counter Profiling

Mirage includes native, in-process hardware performance counter sampling via the Linux `perf_event_open()` syscall (`mirage::benchmark::HardwarePerfSampler`). Counters tracked include:
- `PERF_COUNT_HW_CPU_CYCLES`
- `PERF_COUNT_HW_INSTRUCTIONS`
- `PERF_COUNT_HW_CACHE_REFERENCES`
- `PERF_COUNT_HW_CACHE_MISSES`
- `PERF_COUNT_HW_BRANCH_INSTRUCTIONS`
- `PERF_COUNT_HW_BRANCH_MISSES`
- `PERF_COUNT_SW_TASK_CLOCK`

Calculated metrics:
- $\text{IPC} = \text{Instructions} / \text{Cycles}$
- $\text{Cache Miss Rate} = (\text{Cache Misses} / \text{Cache References}) \times 100\%$

Output files generated:
- `benchmarks/results/benchmark_results.csv`
- `benchmarks/results/tile_analysis.csv`
- `benchmarks/plots/*.png` (7 publication charts)

For comprehensive analysis and findings, consult [`docs/performance-analysis.md`](performance-analysis.md).

## Research Experiments & Ablation Studies (Phase 9)

Mirage includes an automated research experiment pipeline evaluating the adaptive precision hypothesis against random tile baselines and feature ablations.

```bash
# Generate the multi-modal 7-modality test dataset
./build/release/mirage_generate_dataset data/experiments

# Run the complete C++ research experiment suite
./build/release/benchmarks/mirage_experiments \
    --dataset-dir data/experiments \
    --output-dir benchmarks/experiments \
    --seeds 30

# Run Python statistical analysis and Cairo publication plotting
python3 scripts/analyze_experiments.py

# Or execute everything in a single reproducible command:
./benchmarks/run_experiments.sh
```

Generated research artifacts:
- `benchmarks/experiments/dataset_manifest.csv`
- `benchmarks/experiments/random_ablation.csv`
- `benchmarks/experiments/feature_ablation.csv`
- `benchmarks/experiments/tile_sensitivity_errors.csv`
- `benchmarks/experiments/pareto_results.csv`
- `benchmarks/plots/*.png` (6 Phase 9 publication plots)

For full findings, statistical test results, and Pareto curves, consult [`docs/research-results.md`](research-results.md).

## Mirage Dark Creative Studio (Phase 10)

Mirage includes a desktop creative studio application implemented in PySide6 with dark theme ergonomics and interactive before/after split viewing.

```bash
# Launch the desktop creative studio using the root launcher:
./mirage-studio

# Or launch directly via Python:
python3 scripts/mirage_studio.py [image_path]

# Run automated headless verification test:
python3 scripts/test_studio_headless.py
```

### CLI JSON Integration
The studio interacts with `./build/release/mirage` via asynchronous `QProcess` invocations using structured JSON:
```bash
# General metadata in JSON:
./build/release/mirage --info --json

# Enhancement with full telemetry, heatmap, and precision map:
./build/release/mirage --enhance input.png --output output.png \
    --precision adaptive --tile-size 32 --sensitivity-threshold 0.04 \
    --json --save-heatmap heatmap.png --save-precision-map pmap.png
```
