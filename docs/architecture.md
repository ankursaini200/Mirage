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
|          (CLI: --enhance, --analyze-sensitivity, --image-info, --tile-info)       |
|            Supports --precision <fp32|fp16|mixed>, Heatmaps & CSV Reports         |
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
|   |                   Numerical Emulation Core (mirage::core)                 |   |
|   |   +-------------------------------------------------------------------+   |   |
|   |   | Half (IEEE 754 binary16: 1s 5e 10m, RNTE, Subnormals, NaNs, Inf)  |   |   |
|   |   +-------------------------------------------------------------------+   |   |
|   +---------------------------------------------------------------------------+   |
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
|   |             Tile-Based Sensitivity Analyzer (mirage::analysis)            |   |
|   |   +-------------------------------------------------------------------+   |   |
|   |   | Features: Color Variation (C), Gradient (G), Edge (E), Contrast(L)|   |   |
|   |   | Weighted Score S = w_c*C + w_g*G + w_e*E + w_l*L in [0.0, 1.0]    |   |   |
|   |   | Heatmap Visualization (COLORMAP_TURBO) & CSV Report Export        |   |   |
|   |   +-------------------------------------------------------------------+   |   |
|   +---------------------------------------------------------------------------+   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |              Fixed-Precision Enhancement Engine (mirage::enhancement)     |   |
|   |   +------------------------------------+   +--------------------------+   |   |
|   |   | Precision Modes: FP32, FP16, Mixed |   | Error Analysis Facility  |   |   |
|   |   | (MixedPrecisionPolicy assignment)  |   | (MAE, RMSE, PSNR, RelErr)|   |   |
|   |   +------------------------------------+   +--------------------------+   |   |
|   |   +-------------------+   +--------------------+   +------------------+   |   |
|   |   | Brightness Shift  |   | Contrast Midpoint  |   | Saturation Rec709|   |   |
|   |   +-------------------+   +--------------------+   +------------------+   |   |
|   |   +-------------------------------------------------------------------+   |   |
|   |   | Gamma Correction (Power Law) & Deterministic Execution Pipeline   |   |   |
|   |   +-------------------------------------------------------------------+   |   |
|   +---------------------------------------------------------------------------+   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |                    Hardware Abstraction & SIMD Dispatch                   |   |
|   |              (Safe CPUID, OSXSAVE, AVX2 / F16C / FMA Detection)           |   |
|   +---------------------------------------------------------------------------+   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |             [Future Phases: Adaptive Precision Selection & AVX2]          |   |
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

## 5. Phase 4 Architecture: Fixed-Precision Computation Framework

Phase 4 establishes a controlled, deterministic numerical evaluation framework by implementing IEEE 754 binary16 arithmetic emulation alongside the baseline FP32 engine, complete with quantitative numerical error metrics.

### 5.1 IEEE 754 Binary16 Emulation (`mirage::core::Half`)

Modern commodity x86_64 processors with AVX2/F16C support 16-bit float conversions (`vcvtps2ph`, `vcvtph2ps`), but **lack native 16-bit floating-point vector arithmetic** (which requires AVX-512 FP16 or AVX10). To provide cross-platform, deterministic evaluation of reduced numerical precision without architectural bias:
- **Numerical Representation**: Standard IEEE 754-2008 binary16 format:
  - 1 sign bit ($s$)
  - 5 exponent bits ($e$, bias $B = 15$)
  - 10 fraction/mantissa bits ($m$)
- **Rounding Semantics**: Exact **Round-to-Nearest, Ties-to-Even (RNTE)** rounding mode. When a 32-bit single-precision float falls exactly halfway between two representable binary16 values, it rounds to the value with an even least significant bit ($m_0 = 0$).
- **Numerical Domain**:
  - **Normal numbers**: Exponents $e \in [1, 30]$, magnitude $2^{-14} \approx 6.10 \times 10^{-5}$ up to $65504.0$.
  - **Subnormal numbers**: Exponent $e = 0$, gradual underflow down to $2^{-24} \approx 5.96 \times 10^{-8}$.
  - **Signed zeros**: Fully preserves distinct $+0.0$ (`0x0000`) and $-0.0$ (`0x8000`) according to IEEE 754 signbit rules.
  - **Infinities & Overflow**: Values whose magnitude exceeds $65519.5$ saturate to $+ \infty$ (`0x7C00`) or $-\infty$ (`0xFC00`).
  - **Underflow**: Magnitudes strictly below the minimum subnormal tie threshold ($2^{-25}$) flush to signed zero.
  - **NaN Handling**: Preserves quiet and signaling NaNs (`0x7E00` payload preservation).
- **Emulated Arithmetic Semantics**: Binary operators (`+`, `-`, `*`, `/`) promote operands to IEEE 754 single-precision `float`, evaluate the arithmetic expression, and explicitly round the result back to `Half` at the operational boundary.
- **Architectural Clarification**: This framework explicitly models the *numerical precision characteristics* of binary16 computation and is documented as **software-emulated FP16**, not native hardware FP16 execution.

### 5.2 Precision Configurations (`mirage::enhancement`)

Mirage supports three fixed-precision configurations:
1. **Uniform FP32 (`PrecisionMode::FP32`)**:
   - The ground-truth reference baseline executing all operations in 32-bit single precision.
2. **Uniform FP16 (`PrecisionMode::FP16`)**:
   - All four operations (Brightness, Contrast, Saturation, Gamma) execute through `Half` arithmetic with binary16 quantization applied at each intermediate boundary.
3. **Mixed Precision (`PrecisionMode::Mixed`)**:
   - Configurable via `MixedPrecisionPolicy`, assigning individual operational stages to either `FP32` or `FP16`.
   - Default mixed policy executes light color operations (Brightness, Saturation) in `FP16` while reserving `FP32` for non-linear and contrast-sensitive operations (Contrast, Gamma).

### 5.3 Quantitative Error Analysis Framework

To quantitatively evaluate numerical error propagation in reduced precision compared against the FP32 ground truth reference, `mirage::enhancement::compute_error_metrics(ref, target)` calculates:

1. **Maximum Absolute Error (MAE / $L_\infty$ Norm)**:
   $$\text{MaxAbsErr} = \max_{c, y, x} |I_{\text{target}}(x, y, c) - I_{\text{ref}}(x, y, c)|$$
2. **Mean Absolute Error (MAE / $L_1$ Norm)**:
   $$\text{MeanAbsErr} = \frac{1}{3 \cdot W \cdot H} \sum_{c} \sum_{y} \sum_{x} |I_{\text{target}}(x, y, c) - I_{\text{ref}}(x, y, c)|$$
3. **Root Mean Square Error (RMSE / $L_2$ Norm)**:
   $$\text{RMSE} = \sqrt{\frac{1}{3 \cdot W \cdot H} \sum_{c} \sum_{y} \sum_{x} \left(I_{\text{target}}(x, y, c) - I_{\text{ref}}(x, y, c)\right)^2}$$
4. **Peak Signal-to-Noise Ratio (PSNR in dB)**:
   $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right), \quad \text{MAX} = 1.0$$
   *(Defined as $+\infty$ when $\text{MSE} = 0.0$)*
5. **Relative Error**:
   $$\text{RelErr} = \frac{\|I_{\text{target}} - I_{\text{ref}}\|_2}{\|I_{\text{ref}}\|_2 + \epsilon}$$

## 6. Phase 5 Architecture: Tile-Based Sensitivity Analyzer

Phase 5 introduces the image feature extraction and spatial sensitivity analyzer (`mirage::analysis::SensitivityAnalyzer`) to quantitatively characterize local image dynamics before adaptive precision selection.

> [!NOTE]
> The sensitivity analyzer calculates and reports local image variation metrics. It **does not assign numerical precision (FP16/FP32)**, which is reserved for subsequent adaptive pipeline phases. Feature weights represent initial experimental parameters rather than established perceptual ground truth.

### 6.1 Sensitivity Feature Formulations

For each spatial tile $T$ containing $N = W \cdot H$ pixels, four normalized features are computed from scientific CIELAB representations ($L^* \in [0, 100], a^*, b^* \in [-127, 127]$):

1. **Color Variation ($C$)**:
   Measures chromatic dispersion across the $a^*$ and $b^*$ channels:
   $$\sigma_{a^*} = \sqrt{\frac{1}{N}\sum (a_i^* - \bar{a}^*)^2}, \quad \sigma_{b^*} = \sqrt{\frac{1}{N}\sum (b_i^* - \bar{b}^*)^2}$$
   $$\sigma_{\text{color}} = \sqrt{\sigma_{a^*}^2 + \sigma_{b^*}^2}$$
   Normalized against the theoretical maximum chromatic deviation $\sigma_{\text{color, max}} = \sqrt{127^2 + 127^2} \approx 179.605$:
   $$C = \text{clamp}\left(\frac{\sigma_{\text{color}}}{\sigma_{\text{color, max}}}, 0.0, 1.0\right)$$
   For neutral or grayscale regions ($a^* = b^* = 0$), $C = 0.0$.

2. **Gradient Magnitude ($G$)**:
   Measures spatial intensity changes on the $L^*$ channel using the Sobel operator ($3 \times 3$):
   $$G_{x} = \text{Sobel}_x(L^*), \quad G_{y} = \text{Sobel}_y(L^*)$$
   $$|\nabla L^*|_i = \sqrt{G_{x, i}^2 + G_{y, i}^2}$$
   $$\bar{G} = \frac{1}{N} \sum_{i=1}^N |\nabla L^*|_i$$
   Normalized against the theoretical maximum Sobel 3x3 magnitude for $L^* \in [0, 100]$ ($G_{\max} = \sqrt{400^2 + 400^2} \approx 565.685$):
   $$G = \text{clamp}\left(\frac{\bar{G}}{G_{\max}}, 0.0, 1.0\right)$$

3. **Edge Density ($E$)**:
   Computes the proportion of tile pixels crossing a fixed, documented Sobel gradient magnitude threshold ($T_{\text{edge}} = 30.0$ on $L^*$):
   $$E = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\left(|\nabla L^*|_i \ge T_{\text{edge}}\right), \quad E \in [0.0, 1.0]$$

4. **Local Contrast ($L$)**:
   Measures luminance spread across the $L^*$ channel:
   $$\sigma_{L^*} = \sqrt{\frac{1}{N}\sum_{i=1}^N (L_i^* - \bar{L}^*)^2}$$
   Normalized via Popoviciu's inequality for bounded intervals $[0, 100]$ ($\sigma_{L^*, \max} = \frac{100 - 0}{2} = 50.0$):
   $$L = \text{clamp}\left(\frac{\sigma_{L^*}}{50.0}, 0.0, 1.0\right)$$

### 6.2 Weighted Sensitivity Combination

The tile sensitivity score $S$ is computed as a convex combination:
$$S = w_c \cdot C + w_g \cdot G + w_e \cdot E + w_l \cdot L, \quad S \in [0.0, 1.0]$$
- Constraints: $w_c, w_g, w_e, w_l \ge 0$ and $w_c + w_g + w_e + w_l = 1.0$.
- Default experimental weights: $w_c = 0.25, w_g = 0.25, w_e = 0.25, w_l = 0.25$.
- Guaranteed bounded: $S \in [0.0, 1.0]$. Flat or constant images yield $S = 0.0$.

### 6.3 Heatmap Visualization & CSV Export

- **Heatmap Rendering**: Maps sensitivity scores $S \to [0, 255]$ with `COLORMAP_TURBO`, alpha-blended over the original image ($\alpha = 0.55$) with tile boundaries and tile IDs.
- **Embedded Legend**: Bottom banner includes a colorbar with calibration marks ($0.0$ Low to $1.0$ High).
- **CSV Serialization**: Dumps `tile_id, x, y, width, height, is_partial, color_variation, gradient_magnitude, edge_density, local_contrast, sensitivity_score`.

## 7. Future Architectural Roadmap

- **Phase 6**: Tile-Based Adaptive Precision Selection (Thresholding, Budget Policies, Tile Dispatch).
- **Phase 7**: AVX2 SIMD Kernel Acceleration with Safe Runtime Dispatch.
- **Phase 8**: Hardware Counter Profiling (`perf`), Google Benchmark Microbenchmarks, and Python-based Research Evaluation.
- **Phase 9**: Interactive Desktop GUI (PySide6) and Continuous Integration.
