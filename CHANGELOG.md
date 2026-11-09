# Changelog

All notable changes to **Mirage** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

