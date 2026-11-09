# Mirage System Architecture

## 1. Architectural Overview

**Mirage** is an open-source, research-oriented C++20 engine designed to investigate whether **sensitivity-guided adaptive numerical precision** can optimize the computational efficiency of color image enhancement without sacrificing perceptual visual fidelity.

In traditional digital image processing, pipelines typically execute uniformly in either single-precision floating-point (FP32), double-precision (FP64), or fixed-point/integer representations (INT8/INT16). Uniform high precision incurs memory bandwidth and energy overheads, while uniform low precision induces visible quantization artifacts, color banding, and gamut clipping.

Mirage evaluates a tile-based approach:
1. An input image is partitioned into non-overlapping spatial tiles.
2. A fast sensitivity analysis assesses local image characteristics (e.g., luminance gradients, high-frequency texture, smooth gradients, and chroma variance).
3. Precision selection assigns numerical precision per tile (e.g., FP32 for sensitive, smooth, or high-dynamic regions; FP16 / INT8 for textured or low-sensitivity regions).
4. Enhancement kernels execute using specialized SIMD (AVX2) and scalar backends.
5. Tiles are assembled into the final output image with validated range and color fidelity.

```
+-------------------------------------------------------------------+
|                        Application Layer                          |
|                  (CLI Entry Point / Future GUI)                   |
+---------------------------------+---------------------------------+
                                  |
+---------------------------------v---------------------------------+
|                        Mirage Core Library                        |
|                                                                   |
|   +-----------------------+           +-----------------------+   |
|   |   Version & Metadata  |           | CPU Capability Engine |   |
|   +-----------------------+           +-----------------------+   |
|                                                                   |
|   +-----------------------------------------------------------+   |
|   |         Hardware Abstraction & SIMD Runtime Dispatch       |   |
|   |    (Safe CPUID, OSXSAVE, AVX2 / F16C / FMA Detection)     |   |
|   +-----------------------------------------------------------+   |
|                                                                   |
|   +-----------------------------------------------------------+   |
|   |         [Future Phases: Tiling, Sensitivity, Kernels]     |   |
|   +-----------------------------------------------------------+   |
+-------------------------------------------------------------------+
```

## 2. Phase 1 Architecture: Foundation & CPU Detection

In Phase 1, the architecture establishes:
- **Clean Namespace Boundaries**: All symbols reside in `mirage` and sub-namespaces such as `mirage::cpu`.
- **Decoupled Library vs. Executable**: `mirage_core` is built as an independent, reusable library target, decoupled from `mirage` CLI.
- **Safe CPUID Runtime Feature Detection**:
  - Processor identification across x86, x86_64, ARM, RISC-V, and PowerPC.
  - Safe multi-tier AVX2 detection:
    1. Query CPUID leaf 1 for processor capability.
    2. Verify operating system state save support via OSXSAVE (`ECX` bit 27).
    3. Verify OS has enabled YMM register state via `xgetbv(0)` (`XCR0[2:1] == 11b`).
    4. Query CPUID leaf 7, subleaf 0 for AVX2 capability (`EBX` bit 5).
  - Guarantee: zero illegal instruction traps (`#UD`) on non-AVX2 hardware.

## 3. Future Architectural Roadmap

- **Phase 2**: Core Image and Tile Abstractions, Color Spaces (sRGB, linear RGB, Lab).
- **Phase 3**: Sensitivity Metrics & Tile Partitioning Engine.
- **Phase 4**: Multi-Precision Kernels (FP32, FP16 storage/emulation, INT8 quantization).
- **Phase 5**: AVX2 SIMD Kernel Acceleration with Dynamic Runtime Dispatch.
- **Phase 6**: Hardware Counter Profiling (`perf`), Google Benchmark Microbenchmarks, and Python-based Research Evaluation.
- **Phase 7**: Interactive Desktop GUI (PySide6) and Continuous Integration.

