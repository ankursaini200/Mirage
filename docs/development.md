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

The test suite in `tests/` covers 53 automated tests across 8 test suites:
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
```
