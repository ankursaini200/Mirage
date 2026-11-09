# Contributing to Mirage

Thank you for your interest in contributing to **Mirage** (*Adaptive Precision for Software-Based Color Image Enhancement*).

Mirage is an open-source, research-oriented C++ project investigating whether sensitivity-guided tile-based adaptive numerical precision improves the quality-performance tradeoff in image enhancement.

## Guiding Principles

1. **C++20 Standards**: All core engine code is written in clean, modern C++20.
2. **Deterministic & Modular Architecture**: Separation of public headers (`include/mirage/`) from internal implementation (`src/`).
3. **No Premature Optimization Claims**: Performance claims must be substantiated with reproducible benchmarks using Google Benchmark or Linux `perf`.
4. **Safety and Portability**: Safe CPU feature detection and runtime dispatch; never assume target CPUs support AVX2 or native FP16.
5. **Deterministic Testing**: Automated unit tests and CTest integration for all modules.

## Getting Started

### Prerequisites

- Modern C++ compiler supporting C++20:
  - GCC 11+ (GCC 13+ recommended)
  - Clang 14+ (Clang 16+ recommended)
- CMake 3.20 or newer
- Build tool: Make or Ninja
- (Optional) Google Test for running the test suite
- (Optional) Google Benchmark for performance microbenchmarks
- (Optional) OpenCV 4 for image I/O

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/Mirage.git
cd Mirage

# Create and enter build directory
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure
```

### Using CMake Presets

```bash
# Configure and build using Debug or Release presets
cmake --preset debug
cmake --build --preset debug

cmake --preset release
cmake --build --preset release
```

## Development Workflow

1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Follow code formatting and style conventions:
   - Use `namespace mirage`.
   - Prefer standard library algorithms, RAII, and explicit ownership.
   - Avoid global mutable state.
   - Separate declarations and implementations.
3. Write unit tests for new functionality under `tests/`.
4. Verify that all tests pass:
   ```bash
   ctest --test-dir build --output-on-failure
   ```
5. Commit your changes with clear, descriptive commit messages following the Conventional Commits specification:
   - `feat: add tile partitioning logic`
   - `fix: resolve boundary check in image stride`
   - `test: add unit tests for CPU feature detection`
   - `docs: update build instructions in README`
6. Submit a Pull Request targeting `main`.

## Code of Conduct

All contributors are expected to uphold the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

