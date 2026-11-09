# Developer Guide

This document describes build procedures, compiler flags, code quality standards, and testing workflows for **Mirage**.

## Prerequisites

- **C++ Compiler**: Supporting C++20 standard:
  - GCC 11+ (GCC 13+ recommended)
  - Clang 14+ (Clang 16+ recommended)
- **CMake**: Minimum version 3.20 (CMake 3.22+ recommended)
- **Build System**: Ninja or GNU Make
- **Testing**: Google Test (automatically detected via CMake if present)
- **Optional**: OpenCV 4, Google Benchmark

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

## Running Tests

All unit tests and CLI integration tests are integrated with CTest:

```bash
# Run tests in the release build
ctest --test-dir build/release --output-on-failure

# Run tests in the debug build
ctest --test-dir build/debug --output-on-failure

# Using test presets
ctest --preset release
ctest --preset debug
```

## Compiler Warnings & Strict Quality Checks

Mirage enforces rigorous compiler warning flags (`-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual`).

To treat warnings as errors during development:
```bash
cmake -B build -DMIRAGE_WARNINGS_AS_ERRORS=ON
cmake --build build
```

## Testing Compilers

Mirage is developed to be fully compatible with both GCC and Clang:

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

