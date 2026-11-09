cmake_minimum_required(VERSION 3.20)

project(Mirage
    VERSION 0.1.0
    DESCRIPTION "Adaptive Precision for Software-Based Color Image Enhancement"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if no build type is specified for single-configuration generators
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Debug, Release, RelWithDebInfo, MinSizeRel)." FORCE)
endif()

# Export compile commands for tooling (clangd, clang-tidy, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configure search paths and RPATH for dependencies
list(APPEND CMAKE_PREFIX_PATH "/tmp/opencv_install/usr" "/usr/local" "/opt/opencv")
set(CMAKE_BUILD_RPATH "/tmp/opencv_install/usr/lib")
set(CMAKE_INSTALL_RPATH "/tmp/opencv_install/usr/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# User-configurable options
option(MIRAGE_BUILD_TESTS "Build Mirage test suite" ON)
option(MIRAGE_BUILD_BENCHMARKS "Build Mirage benchmark suite" ON)
option(MIRAGE_ENABLE_OPENCV "Enable OpenCV integration" ON)
option(MIRAGE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# Compiler Warnings Configuration
add_library(mirage_warnings INTERFACE)
if(MSVC)
    target_compile_options(mirage_warnings INTERFACE
        /W4
        /permissive-
        $<$<BOOL:${MIRAGE_WARNINGS_AS_ERRORS}>:/WX>
    )
else()
    target_compile_options(mirage_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        $<$<BOOL:${MIRAGE_WARNINGS_AS_ERRORS}>:-Werror>
    )
endif()

# Find OpenCV
if(MIRAGE_ENABLE_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
    message(STATUS "OpenCV ${OpenCV_VERSION} found at ${OpenCV_DIR}")
endif()

# Core Mirage Engine Library
add_library(mirage_core
    src/version.cpp
    src/cpu/cpu_features.cpp
    src/image/image_loader.cpp
    src/image/color_converter.cpp
    src/image/tile_manager.cpp
)
add_library(mirage::mirage_core ALIAS mirage_core)

target_include_directories(mirage_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mirage_core
    PUBLIC
        mirage_warnings
        ${OpenCV_LIBS}
)

target_compile_definitions(mirage_core PUBLIC MIRAGE_HAS_OPENCV=1)
target_compile_features(mirage_core PUBLIC cxx_std_20)

# Main Application Executable
add_executable(mirage
    src/main.cpp
)

target_link_libraries(mirage
    PRIVATE
        mirage::mirage_core
)

target_compile_features(mirage PRIVATE cxx_std_20)

# Sample image generation utility
add_executable(mirage_generate_sample
    examples/generate_sample_image.cpp
)
target_link_libraries(mirage_generate_sample PRIVATE mirage::mirage_core)
target_compile_features(mirage_generate_sample PRIVATE cxx_std_20)

# Testing configuration with CTest
if(MIRAGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks configuration
if(MIRAGE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
