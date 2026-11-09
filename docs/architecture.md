# Mirage System Architecture

## 1. Architectural Overview

**Mirage** is an open-source, research-oriented C++20 engine designed to investigate whether **sensitivity-guided adaptive numerical precision** can optimize the computational efficiency of color image enhancement without sacrificing perceptual visual fidelity.

In traditional digital image processing, pipelines typically execute uniformly in either single-precision floating-point (FP32), double-precision (FP64), or fixed-point/integer representations (INT8/INT16). Uniform high precision incurs memory bandwidth and energy overheads, while uniform low precision induces visible quantization artifacts, color banding, and gamut clipping.

Mirage evaluates a tile-based approach:
1. An input image is ingested, validated, normalized, and partitioned into non-overlapping spatial tiles.
2. A fast sensitivity analysis assesses local image characteristics (e.g., luminance gradients, high-frequency texture, smooth gradients, and chroma variance).
3. Precision selection assigns numerical precision per tile (e.g., FP32 for sensitive, smooth, or high-dynamic regions; FP16 / INT8 for textured or low-sensitivity regions).
4. Enhancement kernels execute using specialized SIMD (AVX2) and scalar backends.
5. Tiles are assembled into the final output image with validated range and color fidelity.

```
+-----------------------------------------------------------------------------------+
|                                 Application Layer                                 |
|                 (CLI: --image-info, --tile-info, --visualize-tiles)               |
+------------------------------------------+----------------------------------------+
                                           |
+------------------------------------------v----------------------------------------+
|                                Mirage Core Library                                |
|                                                                                   |
|   +-----------------------+   +-----------------------+   +-------------------+   |
|   |   Version & Metadata  |   | CPU Capability Engine |   | Error Hierarchy   |   |
|   |   (mirage::version)   |   | (mirage::cpu)         |   | (mirage::image)   |   |
|   +-----------------------+   +-----------------------+   +-------------------+   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |                     Image Processing & Tiling Subsystem                   |   |
|   |                                                                           |   |
|   |   +-------------------+   +--------------------+   +------------------+   |   |
|   |   |    ImageLoader    |   |   ColorConverter   |   |   TileManager    |   |   |
|   |   |  - JPG/PNG/BMP    |   |  - BGR <-> RGB     |   |  - Configurable  |   |   |
|   |   |  - File validation|   |  - Scientific Lab  |   |  - Non-divisible |   |   |
|   |   |  - Normalization  |   |  - Range checks    |   |  - Reconstruction|   |   |
|   |   +-------------------+   +--------------------+   +------------------+   |   |
|   +---------------------------------------------------------------------------+   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |                    Hardware Abstraction & SIMD Dispatch                   |   |
|   |              (Safe CPUID, OSXSAVE, AVX2 / F16C / FMA Detection)           |   |
|   +---------------------------------------------------------------------------+   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |             [Future Phases: Sensitivity Analysis, SIMD Kernels]           |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------+
```

## 2. Phase 1 Architecture: Foundation & CPU Detection

- **Namespace Organization**: `mirage` and `mirage::cpu`.
- **Target Separation**: `mirage_core` static library and `mirage` executable client.
- **Safe Hardware Detection**:
  1. CPUID leaf 0 and 1: vendor string, features, OSXSAVE bit.
  2. `xgetbv(0)`: OS-enabled YMM state (`(XCR0 & 0x6) == 0x6`).
  3. CPUID leaf 7: AVX2 flag bit 5.
  4. Zero illegal instruction traps (`#UD`) on non-AVX2 hardware.

## 3. Phase 2 Architecture: Image Ingestion, Color Space & Tiling

### 3.1 ImageLoader (`mirage::image::ImageLoader`)
- **Format Support**: JPG, JPEG, PNG, BMP.
- **Strict Validation**: Checks file existence, non-zero file size, valid magic headers, and expected channel structure (`CV_8UC3`, BGR).
- **Graceful Error Handling**: Custom exception hierarchy (`FileNotFoundException`, `UnsupportedFormatException`, `CorruptedImageException`, `EmptyImageException`).
- **Precision Normalization**: Converts raw 8-bit unsigned images (`CV_8UC3` in $[0, 255]$) into normalized 32-bit floating-point matrices (`CV_32FC3` in $[0.0, 1.0]$) via $x / 255.0f$, with inverse denormalization.

### 3.2 ColorConverter (`mirage::image::ColorConverter`)
- **Channel Reordering**: `bgr_to_rgb` and `rgb_to_bgr` transformations.
- **Scientific CIELAB**:
  - Implements floating-point conversion from normalized BGR (`CV_32FC3` in $[0.0, 1.0]$) to scientific CIELAB (`CV_32FC3`).
  - Output ranges:
    - $L^* \in [0.0, 100.0]$ (Perceptual Lightness)
    - $a^* \in [-127.0, +127.0]$ (Green-to-Red chromaticity)
    - $b^* \in [-127.0, +127.0]$ (Blue-to-Yellow chromaticity)
  - Distinguishes floating-point CIELAB from legacy 8-bit quantized Lab ($L \in [0, 255]$, $a, b \in [0, 255]$) to eliminate quantization errors in scientific pathways.
  - Symmetrical inverse mapping (`lab_to_bgr`) with deterministic reconstruction.

### 3.3 TileManager (`mirage::image::TileManager`)
- **Configurable Spatial Partitioning**: Decomposes images into arbitrary $T_w \times T_h$ rectangular tiles (8x8, 16x16, 32x32, 64x64, or custom).
- **Non-Divisible Boundary Clamping**: When image dimensions $W, H$ are not evenly divisible by tile dimensions:
  $$W_{\text{tile}}(c) = \min(T_w, W - c \cdot T_w), \quad H_{\text{tile}}(r) = \min(T_h, H - r \cdot T_h)$$
  Boundary tiles are clamped to valid pixel boundaries, preventing any out-of-bounds memory accesses.
- **Zero-Copy Tile Representation**: Each `Tile` holds an OpenCV ROI submatrix (`cv::Mat view`) that points directly into the source image buffer with identical row pitch.
- **Lossless Reconstruction**: `reconstruct_image()` reassembles constituent tiles into a contiguous matrix, verified pixel-by-pixel with $\ell_\infty$ norm difference of $0.0$.
- **Inspection & Visualization**: `render_tile_overlay()` and `save_tile_visualization()` generate annotated overlays displaying tile boundaries and numerical IDs.

## 4. Future Architectural Roadmap

- **Phase 3**: Sensitivity Analysis Metrics (local variance, gradient energy, chromatic complexity) & Precision Selection Engine.
- **Phase 4**: Multi-Precision Kernels (FP32, FP16 storage/emulation, INT8 quantization).
- **Phase 5**: AVX2 SIMD Kernel Acceleration with Safe Runtime Dispatch.
- **Phase 6**: Hardware Counter Profiling (`perf`), Google Benchmark Microbenchmarks, and Python-based Research Evaluation.
- **Phase 7**: Interactive Desktop GUI (PySide6) and Continuous Integration.
