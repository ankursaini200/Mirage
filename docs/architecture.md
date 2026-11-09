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
|          (CLI: --enhance, --image-info, --tile-info, --visualize-tiles)           |
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
|   |   +-------------------+   +--------------------+   +------------------+   |   |
|   |   |    ImageLoader    |   |   ColorConverter   |   |   TileManager    |   |   |
|   |   |  - JPG/PNG/BMP    |   |  - BGR <-> RGB     |   |  - Configurable  |   |   |
|   |   |  - File validation|   |  - Scientific Lab  |   |  - Non-divisible |   |   |
|   |   |  - Normalization  |   |  - Range checks    |   |  - Reconstruction|   |   |
|   |   +-------------------+   +--------------------+   +------------------+   |   |
|   +---------------------------------------------------------------------------+   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |              Baseline Uniform FP32 Enhancement Engine (Phase 3)           |   |
|   |   +-------------------+   +--------------------+   +------------------+   |   |
|   |   | Brightness Shift  |   | Contrast Midpoint  |   | Saturation Rec709|   |   |
|   |   +-------------------+   +--------------------+   +------------------+   |   |
|   |   +-------------------------------------------------------------------+   |   |
|   |   | Gamma Correction (Power Law) & Deterministic Pipeline             |   |   |
|   |   +-------------------------------------------------------------------+   |   |
|   +---------------------------------------------------------------------------+   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |                    Hardware Abstraction & SIMD Dispatch                   |   |
|   |              (Safe CPUID, OSXSAVE, AVX2 / F16C / FMA Detection)           |   |
|   +---------------------------------------------------------------------------+   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |             [Future Phases: Sensitivity Analysis, FP16/SIMD Kernels]      |   |
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
- **Format Support**: Ingests JPG, JPEG, PNG, and BMP files.
- **Strict Validation**: Validates file existence, non-zero file sizes, and channel integrity (`CV_8UC3`, BGR).
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
  - Symmetrical `lab_to_bgr` reconstruction with verified numerical tolerance.

### 3.3 TileManager (`mirage::image::TileManager`)
- **Configurable Spatial Partitioning**: Decomposes images into arbitrary $T_w \times T_h$ rectangular tiles (8x8, 16x16, 32x32, 64x64, or custom).
- **Non-Divisible Boundary Clamping**: Computes clipped tile dimensions on edges:
  $$W_{\text{tile}}(c) = \min(T_w, W - c \cdot T_w), \quad H_{\text{tile}}(r) = \min(T_h, H - r \cdot T_h)$$
- **Zero-Copy Tile Representation**: Each `Tile` holds an OpenCV ROI submatrix (`cv::Mat view`) pointing directly into the parent image buffer.
- **Lossless Reconstruction**: `reconstruct_image()` reassembles constituent tiles into a contiguous matrix, verified pixel-by-pixel with $\ell_\infty$ norm difference of $0.0$.

## 4. Phase 3 Architecture: Baseline Uniform FP32 Enhancement Engine

The `EnhancementEngine` (`mirage::enhancement::EnhancementEngine`) provides the deterministic, uniform single-precision floating-point reference implementation against which future reduced-precision (FP16, INT8) and adaptive-precision kernels will be validated.

### 4.1 Mathematical Operations

All operations operate on normalized `CV_32FC3` matrices with values in $[0.0, 1.0]$.

1. **Additive Brightness Adjustment**:
   $$I_{\text{out}}(c) = \text{clamp}(I_{\text{in}}(c) + \delta, 0.0, 1.0), \quad \delta \in [-1.0, 1.0]$$
   - Parameter: `--brightness <val>` (default: `0.0`, identity).

2. **Midpoint Contrast Scaling**:
   $$I_{\text{out}}(c) = \text{clamp}(M + \alpha \cdot (I_{\text{in}}(c) - M), 0.0, 1.0), \quad \alpha \ge 0.0, \; M = 0.5$$
   - Multiplies pixel deviation from neutral mid-gray ($M = 0.5$).
   - Parameter: `--contrast <val>` (default: `1.0`, identity).

3. **Luminance-Preserving Saturation Adjustment**:
   Calculates Rec. 709 luminance in BGR channel ordering:
   $$Y = 0.114 \cdot B + 0.587 \cdot G + 0.299 \cdot R$$
   Scales color chromaticity relative to luminance:
   $$C_{\text{out}} = \text{clamp}(Y + s \cdot (C_{\text{in}} - Y), 0.0, 1.0), \quad s \ge 0.0, \; C \in \{B, G, R\}$$
   - When $s = 0.0$, produces pure Rec. 709 grayscale ($B_{\text{out}} = G_{\text{out}} = R_{\text{out}} = Y$).
   - When $B = G = R$, $(C - Y) = 0$, so saturation scaling preserves grayscale neutrals without color casts.
   - Parameter: `--saturation <val>` (default: `1.0`, identity).

4. **Power-Law Gamma Transformation**:
   $$I_{\text{out}}(c) = \text{clamp}\left(I_{\text{in}}(c)^{\frac{1}{\gamma}}, 0.0, 1.0\right), \quad \gamma > 0.0$$
   - Applies exponential companding; $\gamma > 1.0$ brightens midtones/shadows, $\gamma < 1.0$ deepens contrast.
   - Parameter: `--gamma <val>` (default: `1.0`, identity).

### 4.2 Pipeline Processing Sequence

To guarantee numerical reproducibility across future precision variants, operations execute in a strict, fixed order:
$$\text{Input} \longrightarrow \text{Brightness} \longrightarrow \text{Contrast} \longrightarrow \text{Saturation} \longrightarrow \text{Gamma} \longrightarrow \text{Output}$$

## 5. Future Architectural Roadmap

- **Phase 4**: Multi-Precision Kernels (FP16 storage & emulation, INT8 quantization) & Adaptive Tiling Engine.
- **Phase 5**: AVX2 SIMD Kernel Acceleration with Safe Runtime Dispatch.
- **Phase 6**: Hardware Counter Profiling (`perf`), Google Benchmark Microbenchmarks, and Python-based Research Evaluation.
- **Phase 7**: Interactive Desktop GUI (PySide6) and Continuous Integration.
