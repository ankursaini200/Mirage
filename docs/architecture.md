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
|   |         Tile-Based Adaptive Precision Engine (mirage::precision)          |   |
|   |   +-------------------------------------------------------------------+   |   |
|   |   | Threshold Policy: score >= tau -> FP32, score < tau -> FP16       |   |   |
|   |   | Precision Distribution: Tile counts & percentages (FP32/FP16)     |   |   |
|   |   | Tile-Level Multi-Precision Dispatch & Lossless Reconstruction     |   |   |
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

## 7. Phase 6 Architecture: Tile-Based Adaptive Precision Decision Engine

Phase 6 implements the closed-loop adaptive precision decision pipeline (`mirage::precision::AdaptivePrecisionEngine`), connecting the perceptual sensitivity analysis from Phase 5 to the multi-precision enhancement kernels from Phase 4.

### 7.1 Decision Policy & Thresholding

Each spatial tile $T_i$ is assigned a numerical execution precision based on its composite sensitivity score $S_i \in [0.0, 1.0]$ evaluated against a configurable threshold $\tau \in [0.0, 1.0]$:

$$\text{Precision}(T_i) = \begin{cases} \text{FP32} & \text{if } S_i \ge \tau \\ \text{FP16} & \text{if } S_i < \tau \end{cases}$$

- **Default Sensitivity Threshold**: $\tau = 0.04$ (configurable via `--sensitivity-threshold <val>`).
- **Heuristic Interpretation**: High spatial variation or contrast-rich regions receive single-precision (FP32) computation to prevent gamut distortion or cumulative rounding errors, while low-sensitivity regions utilize half-precision (FP16).
- **Threshold Extremes**:
  - $\tau = 0.0$: Forces all tiles to FP32, yielding bit-identical output to the uniform FP32 reference baseline with 0.0 error ($PSNR = \infty\text{ dB}$).
  - $\tau = 1.0$: Forces all tiles to FP16, producing identical results to the uniform FP16 pipeline.
  - $\tau \in (0.0, 1.0)$: Selects mixed spatial precision, maintaining visual fidelity while evaluating accuracy tradeoffs.

### 7.2 Tile-Level Multi-Precision Dispatch & Reconstruction

The adaptive pipeline executes in five coordinated steps:
1. **Spatial Decomposition**: `mirage::image::TileManager` partitions the normalized BGR image into $W_{\text{tile}} \times H_{\text{tile}}$ tiles with non-divisible boundary clamping.
2. **Perceptual Analysis**: Full-image CIELAB conversion and Sobel filtering are computed once, yielding tile-level scores ($C, G, E, L, S$).
3. **Precision Decision**: `AdaptivePrecisionEngine::decide_precisions` determines tile precision assignments and compiles distribution statistics (`fp32_count`, `fp16_count`, percentages).
4. **Tile Dispatch**: Each tile's ROI submatrix is processed directly:
   - FP32 tiles invoke `EnhancementEngine::process(tile_src, tile_dst, config)`
   - FP16 tiles invoke `EnhancementEngine::process_fp16(tile_src, tile_dst, config)`
5. **Lossless Image Assembly**: Output ROI submatrices directly map into the pre-allocated destination buffer, guaranteeing zero tile-stitching seam artifacts or pixel duplication.

### 7.3 Performance Timing & Software FP16 Semantics

- **Strict Timing Separation**: The adaptive execution duration (`adaptive_execution_time_ms`) strictly times steps 1 through 5. Baseline FP32 reference generation and error metric computation are evaluated outside the timed block to prevent architectural profiling distortion.
- **Software-Emulated FP16 Limitation**: Because x86_64 CPUs lack native FP16 vector arithmetic instructions (prior to AVX-512 FP16 / AVX10), half-precision is emulated in software via IEEE 754 binary16 rounding. Consequently, running FP16 in software introduces rounding overhead and does not yield speedup over scalar FP32. Actual runtime gains will be measured when hardware-accelerated SIMD kernels are introduced in Phase 7.

### 7.4 Relative $L_2$ Error Formulation

To evaluate numerical deviation relative to signal energy:
$$\text{Relative } L_2 \text{ Error} = \frac{\|I_{\text{target}} - I_{\text{ref}}\|_2}{\|I_{\text{ref}}\|_2 + \epsilon} = \frac{\sqrt{\sum (I_{\text{target}}(x, y, c) - I_{\text{ref}}(x, y, c))^2}}{\sqrt{\sum I_{\text{ref}}(x, y, c)^2} + \epsilon}$$
where $\epsilon = 10^{-10}$ ensures numerical stability without division-by-zero on completely dark images.

## 8. Phase 7 Architecture: SIMD Optimization with AVX2 & Runtime Dispatch

### 8.1 Minimal SIMD Dispatch Architecture

To preserve strict cross-platform portability on CPUs without AVX2 (or non-x86 architectures) while enabling hardware-accelerated vectorization on supported processors:
- **Runtime Dispatch Model**: Execution backend is selected dynamically via `mirage::simd::ExecutionBackend` (`Auto`, `Scalar`, `AVX2`).
  - `Auto`: Queries `CpuFeatures::detect().has_avx2` at runtime. If supported, routes to AVX2 kernels; otherwise falls back gracefully to portable scalar implementations.
  - `Scalar`: Forces the portable reference implementation.
  - `AVX2`: Explicitly requests AVX2 execution; throws an `EnhancementException` if the host CPU or OS lacks AVX2 state management.
- **Isolated Architecture Compilation**:
  - Global project compilation options remain strictly standard (`-O3` on Release, no global `-mavx2`).
  - Architecture-specific flags (`-mavx2 -mfma -mf16c` on GCC/Clang, `/arch:AVX2` on MSVC) are scoped **exclusively** to `src/simd/avx2_kernels.cpp`.
  - Non-AVX2 hosts never execute `src/simd/avx2_kernels.cpp` functions, preventing invalid opcode (`#UD`) hardware exceptions.

### 8.2 Hardware Feature Distinctions: AVX2 vs. F16C vs. Native FP16

A fundamental Computer Organization and Architecture requirement in Mirage is distinguishing:
1. **AVX2 (Advanced Vector Extensions 2)**: 256-bit vector integer and single/double-precision floating-point arithmetic. Does NOT provide native 16-bit floating-point arithmetic.
2. **F16C (Half-Precision Floating-Point Conversion)**: Hardware instruction set (`vcvtps2ph`, `vcvtph2ps`) providing single-instruction conversion between FP32 vectors and 16-bit IEEE 754 binary16 format. Does NOT perform arithmetic operations in FP16.
3. **Native FP16 Arithmetic**: Vector hardware execution of arithmetic directly on 16-bit floats (e.g. AVX-512 FP16, ARM NEON `fp16`). Not supported by standard AVX2.

### 8.3 AVX2 FP32 Kernel Implementations

- **Brightness Adjustment**:
  - Processes 8 single-precision floats (`__m256`) per vector iteration using `_mm256_add_ps` and clamps with `_mm256_max_ps(0.0f, _mm256_min_ps(x, 1.0f))`.
  - Unaligned memory loads and stores (`_mm256_loadu_ps` / `_mm256_storeu_ps`) ensure safe processing across arbitrary strides and partial tiles.
  - Tail loop processes remaining $N \pmod 8$ elements scalar-wise.
- **Contrast Adjustment**:
  - Implements $(I_{\text{in}} - \text{midpoint}) \times \text{factor} + \text{midpoint}$ using `_mm256_sub_ps`, `_mm256_mul_ps`, and `_mm256_add_ps`.
  - Maintains bit-identical parity with the scalar reference implementation.
- **Fused Brightness + Contrast**:
  - Evaluates Brightness clamping followed immediately by Contrast adjustment in vector registers.
  - Eliminates intermediate memory store and reload passes, reducing DRAM memory traffic by 50%.
  - Yields **1.81x–1.90x** speedup on 1080p and 4K images and **3.31x** on cache-resident tiles.
- **Rec. 709 Saturation**:
  - Vectorized across 8 interleaved BGR pixels (24 floats) per iteration using `_mm256_permutevar8x32_ps` and `_mm256_blend_ps` cross-lane permutations.
  - Computes luminance $Y = 0.114 B + 0.587 G + 0.299 R$ across 8 pixels simultaneously, applies chroma scaling, clamps to $[0, 1]$, and reassembles interleaved BGR output.
  - Achieves bit-identical or $\le 1 \times 10^{-6}$ numerical parity.
- **Gamma Correction**:
  - Nonlinear power operation ($I^{1/\gamma}$) retained as scalar reference to maintain exact IEEE 754 transcendental accuracy without introducing polynomial approximations.

### 8.4 Measured Performance Analysis

| Operation / Benchmark | Small (190x130) | 1080p (1920x1080) | 4K UHD (3840x2160) | Memory / Compute Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Brightness (FP32 AVX2)** | 0.011 ms (1.52x) | 2.657 ms (1.01x) | 10.909 ms (1.00x) | Memory-bandwidth bound at 1080p/4K; compute-bound on Small |
| **Contrast (FP32 AVX2)** | 0.015 ms (1.61x) | 2.631 ms (1.02x) | 10.957 ms (1.01x) | Memory-bandwidth bound at 1080p/4K; compute-bound on Small |
| **Combined B+C (Fused AVX2)**| **0.008 ms (3.31x)** | **2.875 ms (1.81x)** | **10.618 ms (1.90x)** | **DRAM traffic cut by 50%; substantial speedup across all sizes** |
| **Saturation (FP32 AVX2)** | 0.019 ms (1.75x) | 2.627 ms (1.22x) | 11.329 ms (1.13x) | Compute-intensive 24-float vector permutation & lerp |
| **Full Pipeline (FP32 AVX2)**| 0.426 ms (1.09x) | 37.991 ms (1.11x) | 151.951 ms (1.12x) | AVX2 B+C+Sat speedup with scalar Gamma fallback |

## 9. Phase 8 Architecture: Hardware Performance Counters, Roofline Model & Profiling

### 9.1 Multi-Feature Safety Architecture
Phase 8 enforces strict multi-feature validation:
- Disassembly of `avx2_kernels.cpp.o` identified compiler-synthesized FMA instructions (`vfmadd132ps`, `vfmadd231ps`, `vfmadd132ss`) alongside 256-bit AVX2 vectors.
- Dispatch validation was hardened:
  $$\text{is\_backend\_supported}(\text{ExecutionBackend::AVX2}) \iff \text{features.has\_avx2} \land \text{features.has\_fma}$$
- `resolve_backend(ExecutionBackend::Auto)` evaluates this condition dynamically, guaranteeing that CPUs lacking FMA will never execute code containing FMA instructions.
- `-mf16c` was decoupled from `avx2_kernels.cpp` in `CMakeLists.txt` since F16C instructions are not emitted in the core kernel object.

### 9.2 In-Process Hardware Performance Counter Profiling
Mirage integrates direct, native hardware performance counter measurement (`mirage::benchmark::HardwarePerfSampler`) utilizing the Linux `perf_event_open()` system call (`exclude_kernel = 1`, `exclude_hv = 1`).
Counters tracked:
- CPU Cycles (`PERF_COUNT_HW_CPU_CYCLES`)
- Instructions Retired (`PERF_COUNT_HW_INSTRUCTIONS`)
- Cache References & Misses (`PERF_COUNT_HW_CACHE_REFERENCES`, `PERF_COUNT_HW_CACHE_MISSES`)
- Branch Instructions & Mispredictions (`PERF_COUNT_HW_BRANCH_INSTRUCTIONS`, `PERF_COUNT_HW_BRANCH_MISSES`)
- Task Clock Time (`PERF_COUNT_SW_TASK_CLOCK`)

### 9.3 Roofline-Style Modeling & Arithmetic Intensity
Operations are categorized against the machine ridge point ($\mathcal{I}_{\text{ridge}} = 4.27 \text{ FLOP/Byte}$ on Intel Core i7-13650HX with dual-channel DDR5):
- **Memory-Bound ($\mathcal{I} < 4.27 \text{ FLOP/Byte}$)**:
  - Brightness ($\mathcal{I} = 0.250$), Contrast ($\mathcal{I} = 0.375$), Unfused B+C ($\mathcal{I} = 0.313$), Fused B+C ($\mathcal{I} = 0.625$), Saturation ($\mathcal{I} = 0.708$).
  - **Empirical Proof**: Vectorized Brightness reduces retired instructions by $3.07\times$ on 1080p, but IPC drops from 2.46 to 0.84 due to 66% cache misses and DRAM memory bus saturation.
  - **Fusion Advantage**: Fused Brightness+Contrast cuts memory bus traffic from 48 bytes/pixel to 24 bytes/pixel, achieving a **$1.89\times–1.93\times$ speedup**.
- **Compute-Bound ($\mathcal{I} \ge 3.0 \text{ FLOP/Byte}$)**:
  - Gamma Correction ($\mathcal{I} = 3.125$), Sensitivity Analysis ($\mathcal{I} = 4.167$).

### 9.4 Software FP16 Emulation Overhead Breakdown
Hardware measurements on 1080p Full HD demonstrate that software IEEE 754 half-precision rounding, bit manipulation, and tie-breaking increases instructions retired by **$11.4\times$** (from 575M to 6.58B instructions) and execution cycles by **$17.2\times$**, proving that adaptive precision cannot yield execution time savings without hardware FP16 execution units.

For complete performance tables, charts, and reproduction commands, refer to [`docs/performance-analysis.md`](performance-analysis.md).
| **Full Pipeline (FP16 Emulated)**| 8.849 ms (0.05x) | 680.547 ms (0.06x) | — | Software emulation overhead (rounding at every arithmetic boundary) |

## 9. Future Architectural Roadmap

- **Phase 8**: Hardware Counter Profiling (`perf`), Google Benchmark Microbenchmarks, and Python-based Research Evaluation.
- **Phase 9**: Interactive Desktop GUI (PySide6) and Continuous Integration.
