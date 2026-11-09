#!/usr/bin/env bash
# scripts/check_env.sh - Verify build dependencies and toolchain for Mirage

set -euo pipefail

echo "=== Mirage Environment Verification ==="

# Check C++ Compiler
if command -v g++ >/dev/null 2>&1; then
    echo "[Found] GCC: $(g++ --version | head -n1)"
else
    echo "[Missing] GCC not found in PATH"
fi

if command -v clang++ >/dev/null 2>&1; then
    echo "[Found] Clang: $(clang++ --version | head -n1)"
else
    echo "[Missing] Clang not found in PATH"
fi

# Check CMake
if command -v cmake >/dev/null 2>&1; then
    echo "[Found] CMake: $(cmake --version | head -n1)"
else
    echo "[Missing] CMake not found in PATH"
fi

# Check CTest
if command -v ctest >/dev/null 2>&1; then
    echo "[Found] CTest: $(ctest --version | head -n1)"
else
    echo "[Missing] CTest not found in PATH"
fi

# Check GoogleTest
if pkg-config --exists gtest 2>/dev/null; then
    echo "[Found] GoogleTest via pkg-config ($(pkg-config --modversion gtest))"
else
    echo "[Notice] GoogleTest pkg-config entry not found; CMake find_package(GTest) will be used."
fi

# Check OpenCV
if pkg-config --exists opencv4 2>/dev/null; then
    echo "[Found] OpenCV: $(pkg-config --modversion opencv4)"
else
    echo "[Notice] OpenCV not found via pkg-config (optional in Phase 1)."
fi

echo "=== Environment Check Completed ==="

