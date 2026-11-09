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
# Inspect basic system information
./build/release/mirage --info
./build/release/mirage --version
./build/release/mirage --help

# Inspect image metadata (channels, dimensions, bit depth, color space)
./build/release/mirage --image-info sample_image.png

# Inspect tile partitioning for non-divisible dimensions
./build/release/mirage --tile-info sample_image.png --tile-size 32

# Render and save tile visualization overlay
./build/release/mirage --visualize-tiles sample_image.png --tile-size 32 --output output/tile_vis.png
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

The test suite in `tests/` covers:
1. `VersionTest`: Semantic versioning, C++ standard detection, and compiler string reporting.
2. `CpuFeaturesTest`: CPUID querying, AVX2 availability, OSXSAVE validation, and string formatting.
3. `ImageLoaderTest`: File existence checks, supported formats (JPG, PNG, BMP), corrupted and empty file handling, and float normalization.
4. `ColorConverterTest`: BGR to RGB channel reordering, RGB to BGR symmetry, scientific CIELAB numerical range validity, known reference colors (black, white), and input validation.
5. `TileManagerTest`: 8x8, 16x16, 32x32, and 64x64 grid divisions, non-divisible image dimensions, boundary tile clamping, single-pixel handling, unique tile IDs, and lossless image reconstruction.

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
