#!/usr/bin/env python3
"""
Mirage Dark Creative Studio — Desktop Application
==================================================
Adaptive Precision for Software-Based Color Image Enhancement
Phase 10: Complete Desktop GUI Implementation

Integrates with the C++20 Mirage executable via asynchronous QProcess
with a structured JSON communication contract.
"""

import csv
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # Graceful fallback if numpy is absent

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QPointF, QRectF, QSize, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Project Paths & Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIN_PATH_CANDIDATES = [
    PROJECT_ROOT / "build" / "release" / "mirage",
    PROJECT_ROOT / "build" / "mirage",
    PROJECT_ROOT / "mirage",
]
EXP_BIN_CANDIDATES = [
    PROJECT_ROOT / "build" / "release" / "benchmarks" / "mirage_experiments",
    PROJECT_ROOT / "build" / "benchmarks" / "mirage_experiments",
    PROJECT_ROOT / "build" / "debug" / "benchmarks" / "mirage_experiments",
    PROJECT_ROOT / "build" / "clang" / "benchmarks" / "mirage_experiments",
]

DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "experiments" / "natural_scene.png"
if not DEFAULT_IMAGE_PATH.exists():
    DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "experiments" / "smooth_gradient.png"

# Color Palette Constants — Premium Graphite Creative Studio
BG_MAIN = "#090C11"             # Main application background
BG_BOTTOM_DARK = "#080B10"      # Bottom background gradient stop
BG_WORKSPACE = "#151B23"        # Main workspace / gradient start
BG_SECONDARY = "#28333D"        # Secondary panels
BG_ADJUSTMENT = "#394651"       # Adjustment panels
BG_CONTROL = "#53616C"          # Control surfaces
BORDER_COLOR = "#35414B"        # Subtle blue-grey borders
BORDER_ACCENT = "#477FA8"       # Focused / active border accent

# Semantic Graphite Layers & Compatibility Aliases
BG_PRIMARY = BG_MAIN
BG_SURFACE = BG_SECONDARY
BG_INPUT = BG_WORKSPACE
BG_SUBCARD = "#1B232D"          # Recessed analytics sub-card surface
BG_HOVER = "#303D48"

SLATE_BG = BG_MAIN
SLATE_PANEL = BG_SECONDARY
SLATE_SURFACE = BG_ADJUSTMENT
SLATE_ACCENT = "#477FA8"
SLATE_ACCENT_LIGHT = "#5C96C2"

ACCENT_BLUE = "#477FA8"         # Selection accent & FP32 indicator
ACCENT_BLUE_LIGHT = "#5C96C2"   # Restrained hover / highlight state
ACCENT_BLUE_DARK = "#366588"    # Restrained pressed state
ACCENT_GREEN = "#35A879"        # FP16 indicator & Ready state
ACCENT_ORANGE = "#D9944A"       # Running / active progress state
ACCENT_RED = "#D15B5B"          # Error / unavailable indicator

TEXT_PRIMARY = "#E5E9ED"        # Primary text
TEXT_SECONDARY = "#A6B0B9"      # Secondary text
TEXT_MUTED = "#82909B"          # Muted text


def find_mirage_binary() -> Optional[Path]:
    for path in BIN_PATH_CANDIDATES:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def find_experiments_binary() -> Optional[Path]:
    for path in EXP_BIN_CANDIDATES:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


# ---------------------------------------------------------------------------
# Structured Research Catalog (Mapped Strictly to Verified Assets)
# ---------------------------------------------------------------------------
RESEARCH_CATEGORIES: List[Dict] = [
    {
        "id": "adaptive_precision",
        "name": "Adaptive Precision",
        "description": "Evaluates sensitivity-guided FP32/FP16 tile allocation against 30-seed random baselines, feature weights, and FP16 error.",
        "preview_plot": "random_vs_guided.png",
        "experiments": [
            {
                "id": "random_vs_guided",
                "short_title": "Guided vs. Random",
                "title": "Sensitivity-Guided vs. Random Tile Allocation",
                "question": "At identical FP32 tile budget parity, does sensitivity-guided assignment produce lower error than random selection?",
                "plot_file": "random_vs_guided.png",
                "csv_file": "benchmarks/experiments/random_ablation.csv",
                "explanation": (
                    "Compares reconstruction Mean Absolute Error (MAE) across 7 image modalities between sensitivity-guided "
                    "allocation and 30 randomized seeds enforcing the exact same FP32 tile count."
                ),
                "findings": [
                    "MAE reduced up to +3.08% on High-Contrast Edges, +1.45% on Natural Scene, and +1.35% on Color Rich",
                    "Mean CIEDE2000 color error reduced up to +7.41% (Grayscale Portrait) and +2.97% (Color Rich)",
                    "Overall guided mean error improved (+0.57% MAE, +0.54% ΔE00); homogeneous modalities tie at 100%/0% FP32",
                ],
                "dataset_info": "7 modalities (512×512 CV_32FC3) × 4 thresholds (τ ∈ {0.02, 0.04, 0.08, 0.15}) × 30 random seeds.",
                "filter_cols": ["modality", "trial_type", "threshold"],
                "default_metric": "mae",
            },
            {
                "id": "feature_ablation",
                "short_title": "Feature Ablation",
                "title": "Sensitivity Feature Weighting & Ablation Analysis",
                "question": "How do color variation, gradient magnitude, edge density, and local contrast contribute individually and under leave-one-out ablation?",
                "plot_file": "feature_ablation.png",
                "csv_file": "benchmarks/experiments/feature_ablation.csv",
                "explanation": (
                    "Measures mean perceptual color difference (CIEDE2000 ΔE00) and FP32 allocation percentage across equal baseline weighting, "
                    "leave-one-out ablation, and isolated single-feature scoring."
                ),
                "findings": [
                    "Equal weighting (all_features) achieves 79.26 dB PSNR (ΔE00 = 0.03016) at 27.6% FP32 allocation",
                    "Edge density and gradient magnitude share mutual information (<0.3 dB PSNR variance when either is omitted)",
                    "Isolated color_only fails severely: under-allocates FP32 to 12.1% tiles and loses 4.41 dB PSNR",
                ],
                "dataset_info": "7 modalities × 9 weighting policies × 2 thresholds (τ ∈ {0.04, 0.08}).",
                "filter_cols": ["modality", "config_name", "threshold"],
                "default_metric": "mean_de00",
            },
            {
                "id": "sensitivity_vs_error",
                "short_title": "Sensitivity vs. Error",
                "title": "Tile Sensitivity Score vs. Actual FP16 Quantization Error",
                "question": "Does the tile sensitivity score correlate with measured FP16-vs-FP32 rounding error across individual spatial tiles?",
                "plot_file": "sensitivity_vs_error.png",
                "csv_file": "benchmarks/experiments/tile_sensitivity_errors.csv",
                "explanation": (
                    "Evaluates 1,792 non-overlapping 32×32 tiles across 7 modalities, plotting calculated sensitivity S ∈ [0, 1] "
                    "directly against tile-level FP16 quantization MAE."
                ),
                "findings": [
                    "Edge density provides strongest positive correlation: Pearson r = +0.2332 (p = 1.49e-23)",
                    "Composite score linear fit r = +0.0244 (MAE); Q4 high-sensitivity tiles average MAE = 0.000158 vs Q2 = 0.000133",
                    "Smooth low-luminance regions amplify CIELAB ΔE00 shifts despite small BGR rounding deltas",
                ],
                "dataset_info": "N = 1,792 individual 32×32 tiles across all 7 benchmark modalities.",
                "filter_cols": ["modality"],
                "default_metric": "tile_mae",
            },
        ],
    },
    {
        "id": "performance_analysis",
        "name": "Performance Analysis",
        "description": "Analyzes AVX2 SIMD speedup, kernel fusion memory savings, hardware IPC, cache misses, and roofline intensity.",
        "preview_plot": "scalar_vs_avx2_speedup.png",
        "experiments": [
            {
                "id": "scalar_vs_avx2",
                "short_title": "AVX2 Speedup",
                "title": "AVX2 SIMD Vectorization & Kernel Fusion Speedup",
                "question": "Where do 256-bit AVX2 vectorization and kernel fusion accelerate color image enhancement over scalar code?",
                "plot_file": "scalar_vs_avx2_speedup.png",
                "csv_file": "benchmarks/results/benchmark_results.csv",
                "explanation": (
                    "Compares 1080p Full HD speedup factors over scalar execution for standalone pixel passes, fused Brightness+Contrast, "
                    "Saturation, and the full enhancement pipeline."
                ),
                "findings": [
                    "Fused Brightness+Contrast achieves 1.89× speedup (3.913 ms → 2.072 ms) by cutting DRAM passes in half",
                    "Saturation improves 1.36× (2.890 ms → 2.128 ms); full FP32 pipeline improves 1.11×",
                    "Standalone linear passes (Brightness 1.02×, Contrast 0.98×) saturate DDR5 bandwidth at 1080p",
                ],
                "dataset_info": "Intel Core i7-13650HX hardware runs across Small, 1080p Full HD, and 4K UHD buffers.",
                "filter_cols": ["Resolution", "Backend", "PrecisionMode"],
                "default_filter": {"Resolution": "1080p Full HD"},
                "default_metric": "RuntimeMedianMs",
            },
            {
                "id": "runtime_vs_resolution",
                "short_title": "Resolution Scaling",
                "title": "Pipeline Latency Scaling Across Image Resolutions",
                "question": "How do scalar, AVX2, FP16-emulated, and adaptive pipelines scale from L2-cache-resident to 4K frames?",
                "plot_file": "runtime_vs_resolution.png",
                "csv_file": "benchmarks/results/benchmark_results.csv",
                "explanation": (
                    "Traces log-scale pipeline latency across Small (190×130), 1080p Full HD (2.07 MP), and 4K UHD (8.29 MP), "
                    "contrasting native hardware FP32 execution against software binary16 emulation."
                ),
                "findings": [
                    "Small (L2-resident): Scalar FP32 0.458 ms vs AVX2 0.418 ms; Fused B+C speeds up 1.86×",
                    "4K UHD: Scalar 163.45 ms vs AVX2 146.52 ms (1.12×); Adaptive (τ=0.04) takes 2,759.1 ms",
                    "Software binary16 emulation incurs ~17.2× slowdown relative to native x86_64 FP32 arithmetic",
                ],
                "dataset_info": "Median execution latency across 3 resolutions (0.02 MP, 2.07 MP, 8.29 MP).",
                "filter_cols": ["Resolution", "Backend", "PrecisionMode"],
                "default_metric": "RuntimeMedianMs",
            },
            {
                "id": "ipc_by_configuration",
                "short_title": "IPC & Memory Stalls",
                "title": "Hardware Instructions Per Cycle (IPC) by Operation",
                "question": "Why does aggressive AVX2 instruction reduction lead to lower retired IPC on streaming kernels?",
                "plot_file": "ipc_by_configuration.png",
                "csv_file": "benchmarks/results/benchmark_results.csv",
                "explanation": (
                    "Displays retired Instructions Per Cycle sampled directly via Linux perf_event_open across scalar and AVX2 "
                    "implementations at 1080p resolution."
                ),
                "findings": [
                    "Brightness: AVX2 reduces retired instructions 3.07× (19.2M → 6.25M), while IPC drops 2.46 → 0.84",
                    "Lower AVX2 IPC reveals DRAM store/load buffer stalls rather than compute bottlenecks",
                    "Full compute-heavy pipeline maintains ~3.76 IPC for both scalar and AVX2 due to powf() gamma evaluation",
                ],
                "dataset_info": "Linux perf_event_open hardware counters (exclude_kernel=1) on 1080p workloads.",
                "filter_cols": ["Resolution", "Backend", "PrecisionMode"],
                "default_filter": {"Resolution": "1080p Full HD"},
                "default_metric": "IPC",
            },
            {
                "id": "cache_miss_rate",
                "short_title": "Cache Miss Rate",
                "title": "Cache Miss Rate by Image Resolution and Workload",
                "question": "At what buffer size do enhancement passes spill beyond L2/L3 caches into DRAM-bound traffic?",
                "plot_file": "cache_miss_rate_by_configuration.png",
                "csv_file": "benchmarks/results/benchmark_results.csv",
                "explanation": (
                    "Compares measured cache miss percentages across cache-resident 190×130 frames, 1080p frames (24.9 MB per CV_32FC3 frame), "
                    "and 4K UHD frames (99.5 MB per frame exceeding 24 MiB L3 cache)."
                ),
                "findings": [
                    "Small cache-resident images show 0.0%–3.5% LLC misses across pipeline passes",
                    "1080p Full HD streaming reads/writes trigger 59.9%–66.2% cache miss rates",
                    "4K UHD frames experience severe L3 spilling with 70.2%–84.1% DRAM miss rates",
                ],
                "dataset_info": "Hardware cache references and misses collected across all 3 benchmark resolutions.",
                "filter_cols": ["Resolution", "Backend", "PrecisionMode"],
                "default_metric": "CacheMissRatePct",
            },
            {
                "id": "roofline_ai",
                "short_title": "Roofline Intensity",
                "title": "Arithmetic Intensity & Architectural Roofline Classification",
                "question": "Which kernels sit in the memory-bound regime versus approaching the processor's compute ridge point?",
                "plot_file": "arithmetic_intensity_by_operation.png",
                "csv_file": "benchmarks/results/benchmark_results.csv",
                "explanation": (
                    "Plots estimated arithmetic intensity (FLOPs/Byte transferred) against the DDR5 roofline ridge threshold (~4.27 FLOP/Byte), "
                    "identifying why fusion improves linear passes while compute passes depend on FLOPs."
                ),
                "findings": [
                    "Linear kernels are deeply memory-bound: Brightness 0.25, Contrast 0.38, Saturation 0.71 FLOP/Byte",
                    "Fused Brightness+Contrast doubles arithmetic intensity from 0.312 → 0.625 FLOP/Byte",
                    "Gamma (3.125 FLOP/B) and Sensitivity Analysis (4.167 FLOP/B) approach compute-bound execution",
                ],
                "dataset_info": "Memory traffic estimation (24–96 Bytes/pixel) vs arithmetic complexity per kernel.",
                "filter_cols": ["Resolution", "Backend"],
                "default_filter": {"Resolution": "1080p Full HD"},
                "default_metric": "EstimatedArithmeticIntensityFLOPsPerByte",
            },
        ],
    },
    {
        "id": "pareto_analysis",
        "name": "Pareto Analysis",
        "description": "Examines Pareto frontiers balancing MAE, CIEDE2000 perceptual quality, FP32 budget allocation, and latency.",
        "preview_plot": "pareto_error_runtime.png",
        "experiments": [
            {
                "id": "pareto_mae_runtime",
                "short_title": "MAE vs. Runtime",
                "title": "Pareto Frontier: Numerical Error (MAE) vs. Execution Time",
                "question": "Which tile sizes and sensitivity thresholds form the Pareto frontier balancing numerical accuracy against latency?",
                "plot_file": "pareto_error_runtime.png",
                "csv_file": "benchmarks/experiments/pareto_results.csv",
                "explanation": (
                    "Plots Mean Absolute Error against pipeline latency across 4 tile granularities and 13 sensitivity thresholds "
                    "τ ∈ [0.00, 1.00], revealing non-dominated operating configurations."
                ),
                "findings": [
                    "At 27.6% FP32 allocation, guided adaptive execution preserves 79.26 dB PSNR and reduces theoretical traffic by 36.2%",
                    "64×64 configurations dominate lower-latency Pareto points across all non-zero error budgets",
                    "Software binary16 emulation reverses CPU speed-accuracy slope relative to native hardware FP16 targets",
                ],
                "dataset_info": "7 modalities × 4 tile sizes (8..64) × 13 thresholds (N=364 parameter points).",
                "filter_cols": ["modality", "tile_size", "threshold"],
                "default_metric": "mae",
            },
            {
                "id": "pareto_deltae_runtime",
                "short_title": "CIEDE2000 vs. Runtime",
                "title": "Perceptual Pareto Frontier: CIEDE2000 (ΔE00) vs. Runtime",
                "question": "How well does sensitivity-guided adaptation contain human-visible CIEDE2000 color error across FP32 budgets?",
                "plot_file": "pareto_deltae_runtime.png",
                "csv_file": "benchmarks/experiments/pareto_results.csv",
                "explanation": (
                    "Evaluates Sharma et al. (2005) ΔE00 color difference on exact CIELAB matrices relative to the FP32 scalar reference "
                    "as sensitivity thresholds sweep from all-FP32 (τ=0.0) to all-FP16 (τ=1.0)."
                ),
                "findings": [
                    "64×64 τ=0.02 (54.2% FP32): mean ΔE00 = 0.01620 at 38.1 ms (non-dominated)",
                    "64×64 τ=0.04 (32.8% FP32): mean ΔE00 = 0.02290 at 49.2 ms (non-dominated)",
                    "Even τ=1.0 (0% FP32) limits mean ΔE00 to 0.04410, far below the ~1.0 human JND visibility threshold",
                ],
                "dataset_info": "Perceptual CIEDE2000 measurements paired with runtime and FP32 budget in pareto_results.csv.",
                "filter_cols": ["modality", "tile_size", "threshold"],
                "default_metric": "mean_de00",
            },
            {
                "id": "allocation_vs_runtime",
                "short_title": "Quality vs. FP32 Budget",
                "title": "Adaptive Precision Tile Distribution vs. Emulation Runtime",
                "question": "How does reducing the FP32 tile ratio across sensitivity thresholds impact software execution latency?",
                "plot_file": "adaptive_distribution_vs_runtime.png",
                "csv_file": "benchmarks/results/benchmark_results.csv",
                "explanation": (
                    "Compares cache-resident pipeline latency across Uniform FP32 and adaptive runs at τ=0.02 (0% FP16), τ=0.04 (37% FP16), "
                    "and τ=0.08 (100% software FP16)."
                ),
                "findings": [
                    "Uniform native FP32: 0.42 ms; τ=0.02 (100% FP32 tiles): 0.69 ms (sensitivity + dispatch cost)",
                    "τ=0.04 (36.7% FP16 / 63.3% FP32): 2.81 ms; τ=0.08 (100% emulated FP16): 9.30 ms",
                    "Confirms software FP16 memory reduction (50% traffic) incurs CPU emulation cost without native FP16 ISA",
                ],
                "dataset_info": "Cache-resident AVX2 adaptive sweeps vs Uniform FP32 baseline (benchmark_results.csv).",
                "filter_cols": ["Resolution", "PrecisionMode", "Backend"],
                "default_filter": {"Resolution": "Small (Cache-Resident)"},
                "default_metric": "RuntimeMedianMs",
            },
        ],
    },
    {
        "id": "tile_architecture",
        "name": "Tile Architecture",
        "description": "Studies spatial tile granularity from 8×8 to 64×64 pixels, dispatch overhead, and locality trade-offs.",
        "preview_plot": "tile_size_comparison.png",
        "experiments": [
            {
                "id": "tile_size_comparison",
                "short_title": "Tile-Size Comparison",
                "title": "Architectural Tile Size Trade-Offs (8×8 to 64×64)",
                "question": "How does nominal tile dimension affect runtime overhead and numerical error across 512×512 images?",
                "plot_file": "tile_size_comparison.png",
                "csv_file": "benchmarks/experiments/pareto_results.csv",
                "explanation": (
                    "Aggregates runtime means and variance across all thresholds for 8×8 (4,096 tiles), 16×16 (1,024 tiles), "
                    "32×32 (256 tiles), and 64×64 (64 tiles) per 512×512 image."
                ),
                "findings": [
                    "64×64 achieves lowest mean runtime (49.1 ms) and lowest MAE (0.000100) through longer vector loops",
                    "8×8 suffers +27.9% runtime overhead (62.7 ms) dispatching 4,096 per-tile regions",
                    "32×32 (53.7 ms, +9.5%) and 64×64 represent the optimal architectural granularity window",
                ],
                "dataset_info": "Tile size sweep across 8, 16, 32, and 64 px tiles in pareto_results.csv.",
                "filter_cols": ["tile_size", "modality", "threshold"],
                "default_metric": "time_ms",
            },
            {
                "id": "runtime_vs_tile_size",
                "short_title": "Dispatch Overhead",
                "title": "Sensitivity Stage Overhead & Execution Latency vs. Tile Size",
                "question": "How fast does sensitivity analysis overhead shrink relative to threshold decision latency on 1080p frames?",
                "plot_file": "runtime_vs_tile_size.png",
                "csv_file": "benchmarks/results/tile_analysis.csv",
                "explanation": (
                    "Tracks sensitivity extraction time and lightweight decision overhead (<0.11 ms) across 8×8 through 64×64 tiles "
                    "on 1080p Full HD imagery."
                ),
                "findings": [
                    "1080p sensitivity analysis time drops 2.59× from 30.8 ms (8×8) to 11.9 ms (64×64)",
                    "End-to-end 1080p adaptive runtime improves from 764.8 ms (8×8) down to 708.7 ms (64×64)",
                    "Decision engine thresholding itself takes <0.11 ms across all evaluated tile counts",
                ],
                "dataset_info": "1080p Full HD AVX2 TileSize sweep with hardware perf counters (tile_analysis.csv).",
                "filter_cols": ["TileSize"],
                "default_metric": "RuntimeMedianMs",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Typography Setup
# ---------------------------------------------------------------------------
def setup_typography(app: QApplication) -> None:
    """
    Checks for approved fonts CODE Bold and CODE Regular.
    If unavailable, explicitly reports this to stderr and configures
    fallback system fonts without silently obscuring the missing fonts.
    """
    font_dir = PROJECT_ROOT / "assets" / "fonts"
    code_candidates = [
        font_dir / "Code-Bold.otf",
        font_dir / "Code-Regular.otf",
        font_dir / "CODE-Bold.otf",
        font_dir / "CODE-Regular.otf",
        font_dir / "CODE Bold.otf",
        font_dir / "CODE Regular.otf",
        font_dir / "Code-Bold.ttf",
        font_dir / "Code-Regular.ttf",
        font_dir / "CODE-Bold.ttf",
        font_dir / "CODE-Regular.ttf",
        font_dir / "CODE Bold.ttf",
        font_dir / "CODE Regular.ttf",
    ]

    loaded_code_font = False
    for p in code_candidates:
        if p.exists():
            font_id = QFontDatabase.addApplicationFont(str(p))
            if font_id >= 0:
                loaded_code_font = True

    # Check font database strictly for actual CODE families (never match 'Fira Code' / 'Source Code Pro')
    system_families = QFontDatabase.families()
    has_system_code = any(
        f.strip().upper() in {"CODE", "CODE REGULAR", "CODE BOLD", "CODE LIGHT"}
        for f in system_families
    )

    if not loaded_code_font and not has_system_code:
        sys.stderr.write(
            "[Mirage Studio Typography] Notice: Approved fonts 'CODE Bold' and 'CODE Regular' "
            "were not found in 'assets/fonts/' or system paths.\n"
            "Preserving typography policy: Falling back to system fonts "
            "('Fira Code', 'JetBrains Mono', 'DejaVu Sans Mono', 'sans-serif').\n"
        )
        app_font = QFont()
        app_font.setFamilies(["Fira Code", "JetBrains Mono", "DejaVu Sans Mono", "Segoe UI", "sans-serif"])
        app_font.setPointSize(10)
        app.setFont(app_font)
    else:
        app_font = QFont()
        app_font.setFamilies(["CODE Regular", "CODE Bold", "Fira Code", "sans-serif"])
        app_font.setPointSize(10)
        app.setFont(app_font)


def studio_font(base_widget: Optional[QWidget] = None, point_size: int = 10, bold: bool = False) -> QFont:
    """Return a QFont preserving CODE Bold (headings) / CODE Regular (supporting) with system fallback."""
    f = QFont(base_widget.font()) if base_widget else QFont()
    f.setFamilies(
        ["CODE Bold", "Fira Code", "JetBrains Mono", "Segoe UI", "sans-serif"]
        if bold
        else ["CODE Regular", "Fira Code", "JetBrains Mono", "Segoe UI", "sans-serif"]
    )
    f.setPointSize(point_size)
    f.setBold(bold)
    return f


def draw_turbo_colorbar(painter: QPainter, x: float, y: float, width: float, height: float) -> None:
    """Draw a calibrated vertical TURBO colormap legend from 1.0 (top) to 0.0 (bottom)."""
    grad = QLinearGradient(x, y, x, y + height)
    grad.setColorAt(0.00, QColor("#e72929"))
    grad.setColorAt(0.25, QColor("#f4ab22"))
    grad.setColorAt(0.50, QColor("#54e346"))
    grad.setColorAt(0.75, QColor("#29b6f6"))
    grad.setColorAt(1.00, QColor("#303f9f"))

    painter.setBrush(grad)
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(QRectF(x, y, width, height), 2, 2)

    painter.setPen(QColor(TEXT_SECONDARY))
    painter.setFont(studio_font(point_size=7, bold=False))
    painter.drawText(int(x - 22), int(y + 8), "1.0")
    painter.drawText(int(x - 22), int(y + height / 2 + 4), "0.5")
    painter.drawText(int(x - 22), int(y + height), "0.0")


# ---------------------------------------------------------------------------
# Dark Creative Studio Theme Stylesheet
# ---------------------------------------------------------------------------
DARK_STUDIO_QSS = f"""
QMainWindow, QWidget#centralWidget {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: "CODE Regular", "Fira Code", "JetBrains Mono", "Segoe UI", sans-serif;
}}

/* Top Navigation Bar */
QFrame#topNavBar {{
    background-color: {BG_PRIMARY};
    border-bottom: 1px solid {BORDER_COLOR};
    padding: 4px 12px;
}}

QLabel#brandTitle {{
    color: {TEXT_PRIMARY};
    font-family: "CODE Bold", "Fira Code", "JetBrains Mono", "Segoe UI", sans-serif;
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 2.0px;
}}

QLabel#brandSubtitle {{
    color: {TEXT_SECONDARY};
    font-family: "CODE Regular", "Fira Code", "JetBrains Mono", "Segoe UI", sans-serif;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.8px;
}}

QPushButton.navTab {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
}}

QPushButton.navTab:hover {{
    color: {TEXT_PRIMARY};
    background-color: rgba(138, 79, 255, 0.15);
}}

QPushButton.navTab:checked {{
    color: #ffffff;
    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_PURPLE}, stop:1 {ACCENT_PURPLE_GRADIENT_END});
    font-weight: bold;
}}

/* Sidebars and Panels */
QFrame.panelCard {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 10px;
}}

QFrame.subCard {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 8px;
}}

QLabel.sectionHeader {{
    color: {TEXT_PRIMARY};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}}

QLabel.subHeader {{
    color: {TEXT_SECONDARY};
    font-size: 10px;
    font-weight: 600;
}}

/* Sliders */
QSlider::groove:horizontal {{
    height: 5px;
    background-color: {BG_INPUT};
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_PURPLE}, stop:1 {ACCENT_PURPLE_LIGHT});
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background-color: #ffffff;
    border: 2px solid {ACCENT_PURPLE};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {ACCENT_PURPLE_LIGHT};
    border: 2px solid #ffffff;
}}

/* Combo Boxes */
QComboBox {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 11px;
}}

QComboBox:hover {{
    border: 1px solid {BORDER_ACCENT};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_PURPLE};
    selection-color: #ffffff;
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    outline: none;
}}

/* Push Buttons */
QPushButton {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {BG_HOVER};
    border: 1px solid {BORDER_ACCENT};
}}

QPushButton:pressed {{
    background-color: {BG_PRIMARY};
}}

QPushButton.subTab {{
    background-color: {BG_INPUT};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 5px;
    padding: 4px 8px;
    font-size: 10px;
    font-weight: 600;
}}

QPushButton.subTab:hover {{
    color: {TEXT_PRIMARY};
    background-color: {BG_HOVER};
    border: 1px solid {BORDER_ACCENT};
}}

QPushButton.subTab:checked {{
    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_PURPLE}, stop:1 {ACCENT_PURPLE_GRADIENT_END});
    color: #ffffff;
    font-weight: bold;
    border: 1px solid {ACCENT_PURPLE};
}}

QPushButton#btnPrimary {{
    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_PURPLE}, stop:1 {ACCENT_PURPLE_GRADIENT_END});
    color: #ffffff;
    border: none;
    border-radius: 7px;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 700;
}}

QPushButton#btnPrimary:hover {{
    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #9d65ff, stop:1 #7d44fb);
}}

QPushButton#btnPrimary:pressed {{
    background-color: #5b28d0;
}}

QPushButton#btnPrimary:disabled {{
    background-color: #38344e;
    color: #726f86;
}}

/* Badges & Value Labels */
QLabel.valBadge {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 10px;
    font-weight: 600;
    min-width: 42px;
    qproperty-alignment: AlignCenter;
}}

/* Status Bar */
QStatusBar {{
    background-color: {BG_PRIMARY};
    border-top: 1px solid {BORDER_COLOR};
    color: {TEXT_SECONDARY};
    font-size: 10px;
    padding: 2px 8px;
}}

/* Scroll Area */
QScrollArea {{
    background-color: transparent;
    border: none;
}}

QScrollBar:vertical {{
    background-color: {BG_PRIMARY};
    width: 6px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {BORDER_ACCENT};
    border-radius: 3px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {ACCENT_PURPLE};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* Research & Experiments Workspace (Graphite / Slate Design) */
QWidget#experimentsView {{
    background-color: {SLATE_BG};
    color: {TEXT_PRIMARY};
    font-family: "CODE Regular", "Fira Code", "JetBrains Mono", "Segoe UI", sans-serif;
}}
QWidget#experimentsView QFrame.expPanel {{
    background-color: {SLATE_PANEL};
    border: 1px solid {SLATE_SURFACE};
    border-radius: 8px;
}}
QWidget#experimentsView QFrame.expSurface {{
    background-color: {SLATE_SURFACE};
    border: 1px solid #4a5a68;
    border-radius: 6px;
}}
QWidget#experimentsView QFrame.categoryCard {{
    background-color: {SLATE_PANEL};
    border: 1px solid {SLATE_SURFACE};
    border-radius: 10px;
}}
QWidget#experimentsView QFrame.categoryCard:hover {{
    border: 1px solid {SLATE_ACCENT};
    background-color: #2f3b46;
}}
QWidget#experimentsView QPushButton.expBtn {{
    background-color: {SLATE_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid #4c5c6b;
    border-radius: 6px;
    padding: 5px 12px;
    font-family: "CODE Bold", "Fira Code", "JetBrains Mono", sans-serif;
    font-size: 10px;
    font-weight: 700;
}}
QWidget#experimentsView QPushButton.expBtn:hover {{
    border: 1px solid {SLATE_ACCENT};
    background-color: #435260;
}}
QWidget#experimentsView QPushButton.expBtn:checked,
QWidget#experimentsView QPushButton.expPrimaryBtn {{
    background-color: {SLATE_ACCENT};
    color: #ffffff;
    border: 1px solid {SLATE_ACCENT_LIGHT};
    font-family: "CODE Bold", "Fira Code", "JetBrains Mono", sans-serif;
    font-size: 10px;
    font-weight: 700;
    border-radius: 6px;
    padding: 5px 12px;
}}
QWidget#experimentsView QPushButton.expPrimaryBtn:hover {{
    background-color: #548eba;
}}
QWidget#experimentsView QPushButton.expSegBtn {{
    background-color: {SLATE_PANEL};
    color: #cbd5e1;
    border: 1px solid {SLATE_SURFACE};
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 10px;
    font-weight: 600;
}}
QWidget#experimentsView QPushButton.expSegBtn:hover {{
    border: 1px solid {SLATE_ACCENT};
    color: #ffffff;
}}
QWidget#experimentsView QPushButton.expSegBtn:checked {{
    background-color: {SLATE_ACCENT};
    color: #ffffff;
    border: 1px solid {SLATE_ACCENT_LIGHT};
}}
QWidget#experimentsView QComboBox,
QWidget#experimentsView QSpinBox,
QWidget#experimentsView QLineEdit {{
    background-color: {SLATE_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid #4c5c6b;
    border-radius: 5px;
    padding: 4px 8px;
}}
QWidget#experimentsView QTableWidget {{
    background-color: {SLATE_BG};
    alternate-background-color: #161d24;
    color: #e2e8f0;
    gridline-color: {SLATE_PANEL};
    border: 1px solid {SLATE_SURFACE};
    border-radius: 6px;
}}
QWidget#experimentsView QHeaderView::section {{
    background-color: {SLATE_PANEL};
    color: {TEXT_PRIMARY};
    border: 1px solid {SLATE_SURFACE};
    padding: 5px 8px;
    font-family: "CODE Bold", "Fira Code", "JetBrains Mono", sans-serif;
    font-size: 9px;
    font-weight: 700;
}}
QWidget#experimentsView QPlainTextEdit {{
    background-color: {SLATE_BG};
    color: #cbd5e1;
    border: 1px solid {SLATE_SURFACE};
    border-radius: 6px;
    font-family: "CODE Regular", "Fira Code", "JetBrains Mono", monospace;
    font-size: 9px;
}}
"""


# ---------------------------------------------------------------------------
# Custom Widget 1: Interactive Split Image Viewer with Zoom & Pan
# ---------------------------------------------------------------------------
class SplitImageViewer(QWidget):
    """
    Renders Before/After comparison with interactive split slider,
    smooth zoom/pan, dimension badge, and original vs enhanced badges.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._original_pixmap: Optional[QPixmap] = None
        self._enhanced_pixmap: Optional[QPixmap] = None
        self._original_path: str = ""
        self._enhanced_path: str = ""
        self._mode_label: str = "Adaptive"

        self._split_ratio: float = 0.5  # 0.0 to 1.0
        self._zoom: float = 1.0
        self._pan: QPointF = QPointF(0, 0)

        self._dragging_split: bool = False
        self._dragging_pan: bool = False
        self._last_mouse_pos: QPointF = QPointF(0, 0)

        self._split_mode: str = "split"  # "split", "original", "enhanced"
        self._img_width: int = 0
        self._img_height: int = 0
        self._format_label: str = "RGB (BGR)"

    def set_images(self, original_path: str, enhanced_path: str, mode_label: str = "Adaptive", format_label: str = "RGB (BGR)") -> None:
        self._original_path = original_path
        self._enhanced_path = enhanced_path
        self._mode_label = mode_label
        self._format_label = format_label

        if original_path and os.path.exists(original_path):
            self._original_pixmap = QPixmap(original_path)
            self._img_width = self._original_pixmap.width()
            self._img_height = self._original_pixmap.height()
        else:
            self._original_pixmap = None

        if enhanced_path and os.path.exists(enhanced_path):
            self._enhanced_pixmap = QPixmap(enhanced_path)
        else:
            self._enhanced_pixmap = self._original_pixmap

        self.update()

    def zoom_in(self) -> None:
        self._zoom = min(self._zoom * 1.25, 8.0)
        self.update()

    def zoom_out(self) -> None:
        self._zoom = max(self._zoom / 1.25, 0.2)
        self.update()

    def reset_zoom(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._split_ratio = 0.5
        self.update()

    def fit_to_screen(self) -> None:
        if not self._original_pixmap or self._original_pixmap.isNull():
            return
        if self.width() <= 50 or self.height() <= 50:
            return
        available_w = max(100.0, float(self.width() - 20))
        available_h = max(100.0, float(self.height() - 20))
        w_ratio = available_w / max(1, self._original_pixmap.width())
        h_ratio = available_h / max(1, self._original_pixmap.height())
        self._zoom = max(0.1, min(w_ratio, h_ratio))
        self._pan = QPointF(0, 0)
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.fit_to_screen()

    def set_split_mode(self, mode: str) -> None:
        self._split_mode = mode
        self.update()

    def _get_image_rect(self) -> QRectF:
        if not self._original_pixmap or self._original_pixmap.isNull():
            return QRectF(0, 0, 0, 0)
        target_w = self._original_pixmap.width() * self._zoom
        target_h = self._original_pixmap.height() * self._zoom
        center_x = self.width() / 2.0 + self._pan.x()
        center_y = self.height() / 2.0 + self._pan.y()
        return QRectF(center_x - target_w / 2.0, center_y - target_h / 2.0, target_w, target_h)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Background canvas
        painter.fillRect(self.rect(), QColor(BG_PRIMARY))

        if not self._original_pixmap or self._original_pixmap.isNull():
            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(QFont("sans-serif", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded.\nClick 'Open Image...' to begin.")
            return

        img_rect = self._get_image_rect()

        # Handle split viewing mode
        if self._split_mode == "original" or not self._enhanced_pixmap:
            painter.drawPixmap(img_rect.toRect(), self._original_pixmap)
        elif self._split_mode == "enhanced":
            painter.drawPixmap(img_rect.toRect(), self._enhanced_pixmap)
        else:
            # Interactive split mode
            split_x = img_rect.left() + img_rect.width() * self._split_ratio

            # 1. Draw Original on left side
            painter.save()
            left_clip = QRectF(img_rect.left(), img_rect.top(), max(0.0, split_x - img_rect.left()), img_rect.height())
            painter.setClipRect(left_clip)
            painter.drawPixmap(img_rect.toRect(), self._original_pixmap)
            painter.restore()

            # 2. Draw Enhanced on right side
            painter.save()
            right_clip = QRectF(split_x, img_rect.top(), max(0.0, img_rect.right() - split_x), img_rect.height())
            painter.setClipRect(right_clip)
            painter.drawPixmap(img_rect.toRect(), self._enhanced_pixmap)
            painter.restore()

            # 3. Draw Split Line
            line_pen = QPen(QColor("#ffffff"), 2)
            painter.setPen(line_pen)
            painter.drawLine(int(split_x), int(img_rect.top()), int(split_x), int(img_rect.bottom()))

            # 4. Draw Draggable Handle circle < | >
            handle_y = img_rect.center().y()
            handle_r = 18
            handle_rect = QRectF(split_x - handle_r, handle_y - handle_r, handle_r * 2, handle_r * 2)

            painter.setBrush(QColor(BG_PRIMARY))
            handle_border_pen = QPen(QColor(ACCENT_PURPLE), 2)
            painter.setPen(handle_border_pen)
            painter.drawEllipse(handle_rect)

            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("sans-serif", 9, QFont.Bold))
            painter.drawText(handle_rect, Qt.AlignCenter, "⟨ | ⟩")

        # 5. Badges Overlay
        badge_font = QFont("sans-serif", 9, QFont.DemiBold)
        painter.setFont(badge_font)

        badge_y = max(14.0, img_rect.top() + 14) if img_rect.height() >= 120 else 14.0
        badge_x_left = max(14.0, img_rect.left() + 14) if img_rect.width() >= 260 else 14.0

        # Original badge (Left)
        badge_orig_rect = QRectF(badge_x_left, badge_y, 118, 24)
        painter.setBrush(QColor(20, 22, 35, 210))
        painter.setPen(QPen(QColor(BORDER_COLOR), 1))
        painter.drawRoundedRect(badge_orig_rect, 6, 6)
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.drawText(badge_orig_rect, Qt.AlignCenter, "Original (Input)")

        # Enhanced badge (Right)
        enhanced_title = f"Enhanced ({self._mode_label})"
        enh_w = max(140.0, float(len(enhanced_title) * 7.5 + 24))
        badge_x_right = min(self.width() - enh_w - 14, img_rect.right() - enh_w - 14) if img_rect.width() >= 260 else self.width() - enh_w - 14
        badge_enh_rect = QRectF(badge_x_right, badge_y, enh_w, 24)
        painter.setBrush(QColor(138, 79, 255, 60))
        painter.setPen(QPen(QColor(ACCENT_PURPLE), 1))
        painter.drawRoundedRect(badge_enh_rect, 6, 6)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(badge_enh_rect, Qt.AlignCenter, enhanced_title)

        # Dimension Badge (Bottom Right)
        dim_text = f"{self._img_width} × {self._img_height} | {self._format_label}"
        dim_x = min(self.width() - 174, img_rect.right() - 174) if img_rect.width() >= 260 else self.width() - 174
        dim_y = min(self.height() - 34, img_rect.bottom() - 32) if img_rect.height() >= 120 else self.height() - 34
        dim_rect = QRectF(dim_x, dim_y, 160, 22)
        painter.setBrush(QColor(20, 22, 35, 190))
        painter.setPen(QPen(QColor(BORDER_COLOR), 1))
        painter.drawRoundedRect(dim_rect, 4, 4)
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setFont(QFont("sans-serif", 8))
        painter.drawText(dim_rect, Qt.AlignCenter, dim_text)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            img_rect = self._get_image_rect()
            if img_rect.width() > 0:
                split_x = img_rect.left() + img_rect.width() * self._split_ratio
                if abs(event.position().x() - split_x) <= 24:
                    self._dragging_split = True
                    return
            self._dragging_pan = True
            self._last_mouse_pos = event.position()
        elif event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            self._dragging_pan = True
            self._last_mouse_pos = event.position()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging_split:
            img_rect = self._get_image_rect()
            if img_rect.width() > 0:
                rel = (event.position().x() - img_rect.left()) / img_rect.width()
                self._split_ratio = max(0.01, min(0.99, rel))
                self.update()
        elif self._dragging_pan:
            delta = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()
            self._pan += delta
            self.update()
        else:
            # Check cursor hover
            img_rect = self._get_image_rect()
            if img_rect.width() > 0:
                split_x = img_rect.left() + img_rect.width() * self._split_ratio
                if abs(event.position().x() - split_x) <= 20:
                    self.setCursor(Qt.SplitHCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._dragging_split = False
        self._dragging_pan = False
        self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        degrees = event.angleDelta().y() / 8.0
        steps = degrees / 15.0
        factor = 1.15 ** steps
        new_zoom = max(0.1, min(10.0, self._zoom * factor))

        # Zoom centering towards mouse cursor
        mouse_pos = event.position()
        center = QPointF(self.width() / 2.0 + self._pan.x(), self.height() / 2.0 + self._pan.y())
        diff = mouse_pos - center
        self._pan += diff * (1.0 - new_zoom / self._zoom)
        self._zoom = new_zoom
        self.update()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self.fit_to_screen()


# ---------------------------------------------------------------------------
# Custom Widget 2: Donut Chart for Tile Allocation
# ---------------------------------------------------------------------------
class DonutChartWidget(QWidget):
    """
    Vector-painted donut chart displaying FP32 and FP16 tile allocation.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 110)
        self._total_tiles: int = 0
        self._fp32_count: int = 0
        self._fp16_count: int = 0
        self._fp32_pct: float = 0.0
        self._fp16_pct: float = 0.0

    def set_allocation(self, total: int, fp32: int, fp16: int, fp32_pct: float, fp16_pct: float) -> None:
        self._total_tiles = total
        self._fp32_count = fp32
        self._fp16_count = fp16
        self._fp32_pct = fp32_pct
        self._fp16_pct = fp16_pct
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        donut_center_x = 36
        donut_center_y = h / 2
        radius_outer = 32
        radius_inner = 20

        rect_outer = QRectF(donut_center_x - radius_outer, donut_center_y - radius_outer, radius_outer * 2, radius_outer * 2)

        if self._total_tiles == 0:
            # Draw neutral placeholder
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(BORDER_COLOR))
            painter.drawEllipse(rect_outer)
            painter.setBrush(QColor(BG_SURFACE))
            painter.drawEllipse(QRectF(donut_center_x - radius_inner, donut_center_y - radius_inner, radius_inner * 2, radius_inner * 2))

            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(QFont("sans-serif", 8))
            painter.drawText(rect_outer, Qt.AlignCenter, "--\nTiles")
            return

        # Calculate angles (in 16ths of a degree)
        start_angle = 90 * 16
        fp32_span = int(-self._fp32_pct / 100.0 * 360 * 16)
        fp16_span = int(-self._fp16_pct / 100.0 * 360 * 16)

        # Draw FP32 slice (Blue)
        if self._fp32_count > 0:
            painter.setBrush(QColor(ACCENT_BLUE))
            painter.setPen(Qt.NoPen)
            painter.drawPie(rect_outer, start_angle, fp32_span)

        # Draw FP16 slice (Green)
        if self._fp16_count > 0:
            painter.setBrush(QColor(ACCENT_GREEN))
            painter.setPen(Qt.NoPen)
            painter.drawPie(rect_outer, start_angle + fp32_span, fp16_span)

        # Draw Inner Hole to make donut
        painter.setBrush(QColor(BG_SURFACE))
        painter.setPen(Qt.NoPen)
        rect_inner = QRectF(donut_center_x - radius_inner, donut_center_y - radius_inner, radius_inner * 2, radius_inner * 2)
        painter.drawEllipse(rect_inner)

        # Center Text
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.setFont(QFont("sans-serif", 10, QFont.Bold))
        text_rect = QRectF(donut_center_x - radius_inner, donut_center_y - 12, radius_inner * 2, 13)
        painter.drawText(text_rect, Qt.AlignCenter, str(self._total_tiles))

        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setFont(QFont("sans-serif", 6, QFont.Medium))
        sub_rect = QRectF(donut_center_x - radius_inner, donut_center_y + 1, radius_inner * 2, 10)
        painter.drawText(sub_rect, Qt.AlignCenter, "Tiles")

        # Right Legend
        legend_x = 76
        box_size = 7

        # FP32 Legend
        painter.fillRect(legend_x, int(donut_center_y - 16), box_size, box_size, QColor(ACCENT_BLUE))
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.setFont(QFont("sans-serif", 7, QFont.DemiBold))
        painter.drawText(legend_x + 11, int(donut_center_y - 9), f"FP32: {self._fp32_count} ({self._fp32_pct:.1f}%)")

        # FP16 Legend
        painter.fillRect(legend_x, int(donut_center_y + 4), box_size, box_size, QColor(ACCENT_GREEN))
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.drawText(legend_x + 11, int(donut_center_y + 11), f"FP16: {self._fp16_count} ({self._fp16_pct:.1f}%)")


# ---------------------------------------------------------------------------
# Custom Widget 3: Performance Bar Chart (Measured Hardware Runtimes)
# ---------------------------------------------------------------------------
class PerformanceBarChart(QWidget):
    """
    Renders horizontal bars for measured hardware runtimes.
    Guarantees ZERO FABRICATION: displays 'Not measured' if a mode has not run.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 110)
        self._fp32_ms: Optional[float] = None
        self._fp16_ms: Optional[float] = None
        self._adaptive_ms: Optional[float] = None
        self._adaptive_label: str = "Adaptive (τ = 0.04)"

    def update_runtimes(self, fp32: Optional[float], fp16: Optional[float], adaptive: Optional[float], adaptive_label: str = "Adaptive (τ = 0.04)") -> None:
        self._fp32_ms = fp32
        self._fp16_ms = fp16
        self._adaptive_ms = adaptive
        self._adaptive_label = adaptive_label
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        left_margin = 56
        right_margin = 10
        chart_w = w - left_margin - right_margin
        top_margin = 12

        # Max scale
        valid_vals = [v for v in [self._fp32_ms, self._fp16_ms, self._adaptive_ms] if v is not None and v > 0]
        max_val = max(valid_vals) if valid_vals else 800.0
        max_scale = max(800.0, math.ceil(max_val / 200.0) * 200.0)

        bars = [
            ("FP32", self._fp32_ms),
            ("FP16", self._fp16_ms),
            ("Adaptive", self._adaptive_ms),
        ]

        bar_h = 14
        row_gap = 24

        for i, (label, val) in enumerate(bars):
            y = top_margin + i * row_gap

            # Label
            painter.setPen(QColor(TEXT_SECONDARY))
            painter.setFont(QFont("sans-serif", 8, QFont.Medium))
            painter.drawText(QRectF(4, y - 1, left_margin - 8, bar_h + 2), Qt.AlignRight | Qt.AlignVCenter, label)

            # Bar track
            track_rect = QRectF(left_margin, y, chart_w, bar_h)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(BG_INPUT))
            painter.drawRoundedRect(track_rect, 3, 3)

            # Bar fill
            if val is not None and val > 0:
                bar_len = max(4.0, min(chart_w, (val / max_scale) * chart_w))
                bar_rect = QRectF(left_margin, y, bar_len, bar_h)

                grad = QLinearGradient(left_margin, y, left_margin + bar_len, y)
                grad.setColorAt(0, QColor(ACCENT_PURPLE))
                grad.setColorAt(1, QColor(ACCENT_PURPLE_LIGHT))
                painter.setBrush(grad)
                painter.drawRoundedRect(bar_rect, 3, 3)

                # Value text
                painter.setPen(QColor(TEXT_PRIMARY))
                painter.setFont(QFont("sans-serif", 7, QFont.Bold))
                val_text = f"{val:.2f} ms"
                if bar_len > 60:
                    painter.drawText(bar_rect.adjusted(0, 0, -6, 0), Qt.AlignRight | Qt.AlignVCenter, val_text)
                else:
                    painter.drawText(QRectF(left_margin + bar_len + 4, y, 60, bar_h), Qt.AlignLeft | Qt.AlignVCenter, val_text)
            else:
                painter.setPen(QColor(TEXT_MUTED))
                painter.setFont(QFont("sans-serif", 7))
                painter.drawText(track_rect, Qt.AlignCenter, "Not measured")

        # X-Axis ticks
        axis_y = top_margin + len(bars) * row_gap + 4
        painter.setPen(QPen(QColor(BORDER_COLOR), 1))
        painter.drawLine(left_margin, int(axis_y), left_margin + int(chart_w), int(axis_y))

        tick_steps = [0, int(max_scale * 0.25), int(max_scale * 0.5), int(max_scale * 0.75), int(max_scale)]
        for t in tick_steps:
            tx = left_margin + (t / max_scale) * chart_w
            painter.drawLine(int(tx), int(axis_y), int(tx), int(axis_y + 3))
            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(QFont("sans-serif", 6))
            painter.drawText(QRectF(tx - 15, axis_y + 4, 30, 10), Qt.AlignCenter, str(t))


# ---------------------------------------------------------------------------
# Custom Widget 4: Heatmap and Analysis Overlay Viewer
# ---------------------------------------------------------------------------
class HeatmapOverlayViewer(QWidget):
    """
    Displays sensitivity heatmap (TURBO colormap), tile precision map,
    difference map, and RGB histogram.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 140)

        self._current_tab: str = "Sensitivity"
        self._heatmap_pixmap: Optional[QPixmap] = None
        self._precision_pixmap: Optional[QPixmap] = None
        self._difference_pixmap: Optional[QPixmap] = None

    def set_overlay_images(self, heatmap_path: str, precision_path: str, diff_path: str = "") -> None:
        if heatmap_path and os.path.exists(heatmap_path):
            self._heatmap_pixmap = QPixmap(heatmap_path)
        if precision_path and os.path.exists(precision_path):
            self._precision_pixmap = QPixmap(precision_path)
        if diff_path and os.path.exists(diff_path):
            self._difference_pixmap = QPixmap(diff_path)
        self.update()

    def set_active_tab(self, tab_name: str) -> None:
        self._current_tab = tab_name
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        w = self.width()
        h = self.height()

        active_pixmap = None
        if self._current_tab == "Sensitivity":
            active_pixmap = self._heatmap_pixmap
        elif self._current_tab == "Precision Map":
            active_pixmap = self._precision_pixmap
        elif self._current_tab == "Difference":
            active_pixmap = self._difference_pixmap

        if active_pixmap and not active_pixmap.isNull():
            # Reserve 28px on right for TURBO colorbar if Sensitivity
            img_w = w - (32 if self._current_tab == "Sensitivity" else 0)
            target_rect = QRectF(0, 0, img_w, h)
            scaled = active_pixmap.scaled(int(img_w), int(h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            draw_x = (img_w - scaled.width()) / 2
            draw_y = (h - scaled.height()) / 2
            painter.drawPixmap(int(draw_x), int(draw_y), scaled)

            # Draw TURBO color bar on the right
            if self._current_tab == "Sensitivity":
                draw_turbo_colorbar(painter, float(w - 24), 10.0, 8.0, float(max(20, h - 20)))
        else:
            painter.fillRect(self.rect(), QColor(BG_INPUT))
            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(QFont("sans-serif", 8))
            painter.drawText(self.rect(), Qt.AlignCenter, f"{self._current_tab}\n(Run enhancement to display)")


# ---------------------------------------------------------------------------
# Custom Widget 5: Filmstrip Carousel Ribbon
# ---------------------------------------------------------------------------
class FilmstripCarousel(QWidget):
    """
    Horizontal ribbon displaying dataset images for quick selection.
    """
    image_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(74)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(6)

        self._btn_prev = QToolButton(self)
        self._btn_prev.setText("⟨")
        self._btn_prev.setFixedSize(24, 60)
        self._btn_prev.clicked.connect(self._scroll_left)
        main_layout.addWidget(self._btn_prev)

        self._scroll_area = QScrollArea(self)
        self._scroll_area.setFixedHeight(66)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._container_layout = QHBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(8)
        self._scroll_area.setWidget(self._container)
        main_layout.addWidget(self._scroll_area)

        self._btn_next = QToolButton(self)
        self._btn_next.setText("⟩")
        self._btn_next.setFixedSize(24, 60)
        self._btn_next.clicked.connect(self._scroll_right)
        main_layout.addWidget(self._btn_next)

        self._cards: List[QPushButton] = []
        self._current_path: str = ""
        self._load_dataset_thumbnails()

    def _load_dataset_thumbnails(self) -> None:
        dataset_dir = PROJECT_ROOT / "data" / "experiments"
        if not dataset_dir.exists():
            return

        image_files = sorted(list(dataset_dir.glob("*.png")))
        for img_path in image_files:
            btn = QPushButton(self._container)
            btn.setFixedSize(72, 56)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("image_path", str(img_path))

            pix = QPixmap(str(img_path))
            if not pix.isNull():
                thumb = pix.scaled(68, 52, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                btn.setIcon(QIcon(thumb))
                btn.setIconSize(QSize(66, 50))

            btn.clicked.connect(lambda checked=False, p=str(img_path): self.select_image(p))
            self._container_layout.addWidget(btn)
            self._cards.append(btn)

    def select_image(self, path: str) -> None:
        self._current_path = path
        for card in self._cards:
            if card.property("image_path") == path:
                card.setStyleSheet(f"border: 2px solid {ACCENT_PURPLE}; border-radius: 6px; background: {BG_SURFACE};")
            else:
                card.setStyleSheet(f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; background: {BG_INPUT};")
        self.image_selected.emit(path)

    def _scroll_left(self) -> None:
        sb = self._scroll_area.horizontalScrollBar()
        sb.setValue(sb.value() - 160)

    def _scroll_right(self) -> None:
        sb = self._scroll_area.horizontalScrollBar()
        sb.setValue(sb.value() + 160)


# ---------------------------------------------------------------------------
# Custom Widget 6: Responsive High-Resolution Figure Viewer
# ---------------------------------------------------------------------------
class FigureViewerWidget(QWidget):
    """
    Dedicated responsive high-resolution figure viewer for publication plots.
    Automatically scales the active figure to fill the available display area
    using anti-aliased smooth transformation while preserving aspect ratio.
    Caches both the decoded QPixmap and the scaled viewport QPixmap to avoid redundant I/O and scaling.
    Displays clear, formatted error messages if an image file is missing or invalid.
    """
    def __init__(self, parent=None, canvas_bg: str = BG_INPUT, border_color: str = BORDER_COLOR):
        super().__init__(parent)
        self._canvas_bg: str = canvas_bg
        self._border_color: str = border_color
        self._current_path: str = ""
        self._resolved_path: str = ""
        self._current_pixmap: Optional[QPixmap] = None
        self._scaled_pixmap: Optional[QPixmap] = None
        self._scaled_cache_key: Optional[Tuple[str, int, int, float]] = None
        self._error_message: str = ""
        self._cache: Dict[str, QPixmap] = {}
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(360, 260)

    def load_figure(self, file_path: str) -> None:
        """
        Loads the figure from an absolute or resolved file path.
        Caches the original unscaled QPixmap in memory.
        """
        self._current_path = file_path
        self._resolved_path = ""
        self._scaled_pixmap = None
        self._scaled_cache_key = None
        self._error_message = ""

        if not file_path:
            self._current_pixmap = None
            self._error_message = "No publication figure selected."
            self.update()
            return

        p = Path(file_path).resolve()
        resolved_str = str(p)
        if not p.is_file():
            self._current_pixmap = None
            self._cache.pop(resolved_str, None)
            self._error_message = (
                f"Figure File Not Found:\n\n{p.name}\n\n"
                f"Resolved path: {p}\n\n"
                "Please verify that benchmarks/plots/ contains the generated figures."
            )
            self.update()
            return

        self._resolved_path = resolved_str
        if resolved_str in self._cache:
            self._current_pixmap = self._cache[resolved_str]
        else:
            pix = QPixmap(resolved_str)
            if pix.isNull():
                self._current_pixmap = None
                self._error_message = (
                    f"Failed to Decode Image File:\n\n{p.name}\n\n"
                    f"File path: {p}\n\n"
                    "The image file may be corrupted or in an unsupported format."
                )
                self.update()
                return
            self._cache[resolved_str] = pix
            self._current_pixmap = pix

        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Draw canvas container
        painter.fillRect(self.rect(), QColor(self._canvas_bg))

        # Inherit approved typography (CODE Bold / CODE Regular with configured fallback)
        hdr_font = QFont(self.font())
        hdr_font.setPointSize(12)
        hdr_font.setBold(True)
        body_font = QFont(self.font())
        body_font.setPointSize(10)
        body_font.setBold(False)

        # Error State Display
        if self._error_message:
            body_w = max(120, self.width() - 80)
            fm_body = QtGui.QFontMetrics(body_font)
            body_bounds = fm_body.boundingRect(
                QtCore.QRect(0, 0, int(body_w), 2000),
                Qt.AlignHCenter | Qt.TextWordWrap,
                self._error_message,
            )
            total_h = 34 + body_bounds.height()
            top_y = max(20.0, (self.height() - total_h) / 2.0)

            painter.setPen(QColor(ACCENT_RED))
            painter.setFont(hdr_font)
            hdr_rect = QRectF(20.0, top_y, max(100.0, self.width() - 40.0), 28.0)
            painter.drawText(hdr_rect, Qt.AlignCenter, "⚠  FIGURE UNAVAILABLE")

            painter.setPen(QColor(TEXT_SECONDARY))
            painter.setFont(body_font)
            body_rect = QRectF(40.0, top_y + 34.0, float(body_w), float(body_bounds.height() + 8))
            painter.drawText(body_rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self._error_message)
            return

        if not self._current_pixmap or self._current_pixmap.isNull():
            painter.setPen(QColor(TEXT_MUTED))
            empty_font = QFont(body_font)
            empty_font.setPointSize(11)
            painter.setFont(empty_font)
            painter.drawText(self.rect(), Qt.AlignCenter, "No publication figure loaded.")
            return

        # Determine available area with clean margins
        margin = 16
        avail_w = max(10, self.width() - margin * 2)
        avail_h = max(10, self.height() - margin * 2)
        dpr = max(1.0, float(self.devicePixelRatioF()))
        target_w = max(1, int(round(avail_w * dpr)))
        target_h = max(1, int(round(avail_h * dpr)))

        cache_key = (self._resolved_path, target_w, target_h, dpr)
        if self._scaled_cache_key != cache_key or self._scaled_pixmap is None:
            scaled = self._current_pixmap.scaled(
                target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            scaled.setDevicePixelRatio(dpr)
            self._scaled_pixmap = scaled
            self._scaled_cache_key = cache_key

        disp_size = self._scaled_pixmap.deviceIndependentSize()
        draw_w = disp_size.width()
        draw_h = disp_size.height()
        draw_x = (self.width() - draw_w) / 2.0
        draw_y = (self.height() - draw_h) / 2.0

        # Draw crisp subtle border around figure without extra decorative clutter
        painter.setPen(QPen(QColor(self._border_color), 1))
        painter.drawRect(QRectF(draw_x - 1.0, draw_y - 1.0, draw_w + 1.0, draw_h + 1.0))
        painter.drawPixmap(QtCore.QPointF(draw_x, draw_y), self._scaled_pixmap)


# ---------------------------------------------------------------------------
# Custom Widget 7: Scaled Result Visualization Preview (Heatmap / Precision Map)
# ---------------------------------------------------------------------------
class ResultImagePreview(QWidget):
    """
    Aspect-ratio-preserving preview canvas for actual generated backend
    sensitivity heatmaps and tile precision allocation maps in the Results tab.
    """
    def __init__(
        self,
        empty_text: str = "Not available",
        show_turbo_bar: bool = False,
        parent=None,
        canvas_bg: str = BG_INPUT,
        border_color: str = BORDER_COLOR,
    ):
        super().__init__(parent)
        self._canvas_bg: str = canvas_bg
        self._border_color: str = border_color
        self._pixmap: Optional[QPixmap] = None
        self._empty_text: str = empty_text
        self._show_turbo_bar: bool = show_turbo_bar
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(140, 110)

    def set_image_path(self, file_path: str, empty_text: Optional[str] = None) -> None:
        if empty_text is not None:
            self._empty_text = empty_text
        if file_path and os.path.isfile(file_path):
            pix = QPixmap(file_path)
            self._pixmap = pix if not pix.isNull() else None
        else:
            self._pixmap = None
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.fillRect(self.rect(), QColor(self._canvas_bg))
        painter.setPen(QPen(QColor(self._border_color), 1))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 4, 4)

        if not self._pixmap or self._pixmap.isNull():
            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(studio_font(self, point_size=8, bold=False))
            painter.drawText(self.rect().adjusted(8, 8, -8, -8), Qt.AlignCenter | Qt.TextWordWrap, self._empty_text)
            return

        pad = 6
        bar_reserve = 34 if self._show_turbo_bar else 0
        avail_w = max(10, self.width() - pad * 2 - bar_reserve)
        avail_h = max(10, self.height() - pad * 2)
        scaled = self._pixmap.scaled(avail_w, avail_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        draw_x = pad + (avail_w - scaled.width()) / 2.0
        draw_y = pad + (avail_h - scaled.height()) / 2.0
        painter.drawPixmap(int(draw_x), int(draw_y), scaled)

        if self._show_turbo_bar:
            draw_turbo_colorbar(painter, float(self.width() - 18), 8.0, 8.0, float(max(24, self.height() - 16)))


# ---------------------------------------------------------------------------
# Custom Widget 8: Research Category Landing Card
# ---------------------------------------------------------------------------
class CategoryCardWidget(QFrame):
    """Clickable research category card displaying name, summary, chart count, and preview thumbnail."""
    clicked = Signal(str)

    def __init__(self, category: Dict, fig_count: int, preview_path: str, parent=None):
        super().__init__(parent)
        self._cat_id: str = category["id"]
        self.setProperty("class", "categoryCard")
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(14)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        # Count chip
        badge = QLabel(f"{fig_count} FIGURES", self)
        badge.setFont(studio_font(badge, point_size=8, bold=True))
        badge.setStyleSheet(
            f"color: #ffffff; background-color: {SLATE_ACCENT}; border-radius: 4px; padding: 2px 8px;"
        )
        badge.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        left_col.addWidget(badge)

        title = QLabel(category["name"], self)
        title.setFont(studio_font(title, point_size=14, bold=True))
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        left_col.addWidget(title)

        desc = QLabel(category["description"], self)
        desc.setWordWrap(True)
        desc.setFont(studio_font(desc, point_size=9, bold=False))
        desc.setStyleSheet("color: #cbd5e1;")
        left_col.addWidget(desc, 1)

        btn = QPushButton("Open Category  →", self)
        btn.setProperty("class", "expPrimaryBtn")
        btn.setFixedWidth(160)
        btn.clicked.connect(lambda: self.clicked.emit(self._cat_id))
        left_col.addWidget(btn)

        layout.addLayout(left_col, 1)

        # Small representative preview thumbnail
        preview = ResultImagePreview(
            empty_text="Preview unavailable",
            show_turbo_bar=False,
            parent=self,
            canvas_bg=SLATE_BG,
            border_color=SLATE_SURFACE,
        )
        preview.setFixedSize(210, 146)
        preview.set_image_path(preview_path)
        layout.addWidget(preview, 0, Qt.AlignVCenter)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._cat_id)
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# Main Application Window: Mirage Dark Creative Studio
# ---------------------------------------------------------------------------
class MirageStudioMainWindow(QMainWindow):
    """
    Main desktop GUI window for Mirage.
    """
    def __init__(self, initial_image: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Mirage")
        self.resize(1340, 840)
        self.setMinimumSize(1080, 700)

        self._mirage_bin = find_mirage_binary()
        self._exp_bin = find_experiments_binary()
        self._active_image_path = initial_image or str(DEFAULT_IMAGE_PATH)
        self._temp_dir = tempfile.mkdtemp(prefix="mirage_studio_")
        self._process: Optional[QtCore.QProcess] = None
        self._exp_process: Optional[QtCore.QProcess] = None
        self._run_counter: int = 0

        # Experiments workspace state (lazy viewing — never reruns seeds on open)
        self._exp_csv_cache: Dict[str, Tuple[List[str], List[List[str]]]] = {}
        self._active_exp_cat_id: str = RESEARCH_CATEGORIES[0]["id"]
        self._active_exp_id: str = RESEARCH_CATEGORIES[0]["experiments"][0]["id"]
        self._exp_filter_combos: Dict[str, QComboBox] = {}
        self._exp_syncing_filters: bool = False
        self._exp_sort_col: Optional[int] = None
        self._exp_sort_order: Qt.SortOrder = Qt.AscendingOrder

        # State storage (committed strictly after successful execution)
        self._measured_fp32_ms: Optional[float] = None
        self._measured_fp16_ms: Optional[float] = None
        self._measured_adaptive_ms: Optional[float] = None
        self._latest_telemetry: Dict = {}
        self._last_result_telemetry: Dict = {}
        self._last_result_input_path: str = ""
        self._last_result_output_path: str = ""
        self._last_result_heatmap_path: str = ""
        self._last_result_precision_path: str = ""

        self._init_ui()
        self._load_system_info()

        # Load initial image
        if self._active_image_path and os.path.exists(self._active_image_path):
            self._filmstrip.select_image(self._active_image_path)
            self._on_image_selected(self._active_image_path)

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Top Navigation Bar
        top_bar = self._create_top_bar()
        root_layout.addWidget(top_bar)

        # 2. Main Stacked Views
        self._main_stack = QStackedWidget(self)
        root_layout.addWidget(self._main_stack, 1)

        # View 0: Enhance Studio
        enhance_view = self._create_enhance_view()
        self._main_stack.addWidget(enhance_view)

        # View 1: Analyze View
        analyze_view = self._create_analyze_view()
        self._main_stack.addWidget(analyze_view)

        # View 2: Benchmark View
        benchmark_view = self._create_benchmark_view()
        self._main_stack.addWidget(benchmark_view)

        # View 3: Experiments & Ablation View
        experiments_view = self._create_experiments_view()
        self._main_stack.addWidget(experiments_view)

        # View 4: Publication Results View
        results_view = self._create_results_view()
        self._main_stack.addWidget(results_view)

        # 3. Bottom Status Bar
        self._status_bar = self._create_status_bar()
        self.setStatusBar(self._status_bar)

    def _create_top_bar(self) -> QFrame:
        top_bar = QFrame(self)
        top_bar.setObjectName("topNavBar")
        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(12)

        # Brand / Logo
        brand_layout = QVBoxLayout()
        brand_layout.setSpacing(2)
        brand_layout.setContentsMargins(2, 0, 12, 0)
        lbl_title = QLabel("MIRAGE", self)
        lbl_title.setObjectName("brandTitle")
        lbl_subtitle = QLabel("Adaptive Precision Imaging", self)
        lbl_subtitle.setObjectName("brandSubtitle")
        brand_layout.addWidget(lbl_title)
        brand_layout.addWidget(lbl_subtitle)
        layout.addLayout(brand_layout)

        layout.addSpacing(56)

        # Nav Tabs
        self._nav_group = QtWidgets.QButtonGroup(self)
        self._nav_group.setExclusive(True)

        tabs = [
            ("Enhance", 0),
            ("Analyze", 1),
            ("Benchmark", 2),
            ("Experiments", 3),
            ("Results", 4),
        ]

        for title, idx in tabs:
            btn = QPushButton(title, self)
            btn.setProperty("class", "navTab")
            btn.setCheckable(True)
            if idx == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked=False, i=idx: self._switch_view(i))
            self._nav_group.addButton(btn, idx)
            layout.addWidget(btn)

        layout.addStretch(1)

        # Quick action icons
        btn_open = QToolButton(self)
        btn_open.setText("📁")
        btn_open.setToolTip("Open Image")
        btn_open.clicked.connect(self._on_open_image_dialog)
        layout.addWidget(btn_open)

        btn_save = QToolButton(self)
        btn_save.setText("💾")
        btn_save.setToolTip("Save Result")
        btn_save.clicked.connect(self._on_save_result_dialog)
        layout.addWidget(btn_save)

        return top_bar

    def _create_enhance_view(self) -> QWidget:
        view = QWidget(self)
        layout = QHBoxLayout(view)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        # Left Column: Controls (Width ~300)
        left_col = self._create_left_sidebar()
        layout.addWidget(left_col)

        # Center Column: Viewer + Toolbar + Carousel
        center_col = self._create_center_viewport()
        layout.addWidget(center_col, 1)

        # Right Column: Analytics & Hardware Cards (Width ~320)
        right_col = self._create_right_sidebar()
        layout.addWidget(right_col)

        return view

    def _create_left_sidebar(self) -> QWidget:
        panel = QFrame(self)
        panel.setProperty("class", "panelCard")
        panel.setFixedWidth(290)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # PROJECT Group
        lbl_proj = QLabel("PROJECT", self)
        lbl_proj.setProperty("class", "sectionHeader")
        layout.addWidget(lbl_proj)

        btn_open_img = QPushButton("📁  Open Image...", self)
        btn_open_img.clicked.connect(self._on_open_image_dialog)
        layout.addWidget(btn_open_img)

        btn_save_res = QPushButton("💾  Save Result...", self)
        btn_save_res.clicked.connect(self._on_save_result_dialog)
        layout.addWidget(btn_save_res)

        btn_export_rep = QPushButton("📄  Export Report...", self)
        btn_export_rep.clicked.connect(self._on_export_report_dialog)
        layout.addWidget(btn_export_rep)

        layout.addSpacing(6)

        # ENHANCEMENT Group
        lbl_enh = QLabel("ENHANCEMENT", self)
        lbl_enh.setProperty("class", "sectionHeader")
        layout.addWidget(lbl_enh)

        # Sliders: Brightness, Contrast, Saturation, Gamma
        self._slider_bright, self._lbl_bright = self._create_slider_row("Brightness", -1.0, 1.0, 0.10, 100, layout)
        self._slider_contrast, self._lbl_contrast = self._create_slider_row("Contrast", 0.0, 3.0, 1.20, 100, layout)
        self._slider_sat, self._lbl_sat = self._create_slider_row("Saturation", 0.0, 3.0, 1.30, 100, layout)
        self._slider_gamma, self._lbl_gamma = self._create_slider_row("Gamma", 0.1, 3.0, 1.10, 100, layout)

        layout.addSpacing(6)

        # PRECISION Group
        lbl_prec = QLabel("PRECISION", self)
        lbl_prec.setProperty("class", "sectionHeader")
        layout.addWidget(lbl_prec)

        # Mode
        row_mode = QHBoxLayout()
        lbl_m = QLabel("Mode", self)
        lbl_m.setProperty("class", "subHeader")
        row_mode.addWidget(lbl_m)
        self._combo_mode = QComboBox(self)
        self._combo_mode.addItems(["Adaptive", "FP32 (Reference)", "FP16 (Emulated)", "Mixed Precision"])
        self._combo_mode.currentIndexChanged.connect(self._on_precision_mode_changed)
        row_mode.addWidget(self._combo_mode, 1)
        layout.addLayout(row_mode)

        # Policy
        row_pol = QHBoxLayout()
        lbl_p = QLabel("Policy", self)
        lbl_p.setProperty("class", "subHeader")
        row_pol.addWidget(lbl_p)
        self._combo_policy = QComboBox(self)
        self._combo_policy.addItems(["Sensitivity-Guided", "30-Seed Random Selection (Research Ablation)"])
        self._combo_policy.currentIndexChanged.connect(self._on_policy_changed)
        row_pol.addWidget(self._combo_policy, 1)
        layout.addLayout(row_pol)

        # Tile Size
        row_ts = QHBoxLayout()
        lbl_ts = QLabel("Tile Size", self)
        lbl_ts.setProperty("class", "subHeader")
        row_ts.addWidget(lbl_ts)
        self._combo_tilesize = QComboBox(self)
        self._combo_tilesize.addItems(["16 × 16", "32 × 32", "64 × 64", "128 × 128"])
        self._combo_tilesize.setCurrentIndex(1)  # 32x32
        row_ts.addWidget(self._combo_tilesize, 1)
        layout.addLayout(row_ts)

        # Threshold (tau)
        self._slider_tau, self._lbl_tau = self._create_slider_row("Threshold (τ)", 0.00, 0.20, 0.040, 1000, layout)

        layout.addStretch(1)

        # Primary Action Button: Run Enhancement
        self._btn_run = QPushButton("▶   Run Enhancement", self)
        self._btn_run.setObjectName("btnPrimary")
        self._btn_run.clicked.connect(self._on_run_enhancement)
        layout.addWidget(self._btn_run)

        return panel

    def _create_slider_row(self, name: str, min_val: float, max_val: float, init_val: float, scale: int, parent_layout: QVBoxLayout) -> Tuple[QSlider, QLabel]:
        row_header = QHBoxLayout()
        lbl_name = QLabel(name, self)
        lbl_name.setProperty("class", "subHeader")
        row_header.addWidget(lbl_name)

        fmt = "+.2f" if (name == "Brightness" and init_val >= 0) else (".3f" if "Threshold" in name else ".2f")
        val_str = f"{init_val:{fmt}}" if not (name == "Brightness" and init_val > 0) else f"+{init_val:.2f}"
        lbl_val = QLabel(val_str, self)
        lbl_val.setProperty("class", "valBadge")
        row_header.addWidget(lbl_val, 0, Qt.AlignRight)
        parent_layout.addLayout(row_header)

        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(int(min_val * scale), int(max_val * scale))
        slider.setValue(int(init_val * scale))

        def _on_val_change(v):
            real_val = v / scale
            if name == "Brightness":
                s = f"+{real_val:.2f}" if real_val > 0 else f"{real_val:.2f}"
            elif "Threshold" in name:
                s = f"{real_val:.3f}"
            else:
                s = f"{real_val:.2f}"
            lbl_val.setText(s)

        slider.valueChanged.connect(_on_val_change)
        parent_layout.addWidget(slider)

        return slider, lbl_val

    def _create_center_viewport(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Viewport Toolbar
        toolbar = QFrame(container)
        toolbar.setProperty("class", "panelCard")
        toolbar.setFixedHeight(40)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 2, 8, 2)
        tb_layout.setSpacing(8)

        btn_hand = QToolButton(toolbar)
        btn_hand.setText("✋")
        btn_hand.setToolTip("Pan Tool")
        tb_layout.addWidget(btn_hand)

        btn_zin = QToolButton(toolbar)
        btn_zin.setText("🔍+")
        btn_zin.setToolTip("Zoom In")
        btn_zin.clicked.connect(lambda: self._viewer.zoom_in())
        tb_layout.addWidget(btn_zin)

        btn_zout = QToolButton(toolbar)
        btn_zout.setText("🔍-")
        btn_zout.setToolTip("Zoom Out")
        btn_zout.clicked.connect(lambda: self._viewer.zoom_out())
        tb_layout.addWidget(btn_zout)

        btn_11 = QPushButton("1:1", toolbar)
        btn_11.setFixedWidth(36)
        btn_11.setToolTip("Actual Size (100%)")
        btn_11.clicked.connect(lambda: self._viewer.reset_zoom())
        tb_layout.addWidget(btn_11)

        btn_fit = QToolButton(toolbar)
        btn_fit.setText("⛶")
        btn_fit.setToolTip("Fit to Screen")
        btn_fit.clicked.connect(lambda: self._viewer.fit_to_screen())
        tb_layout.addWidget(btn_fit)

        self._lbl_zoom_pct = QLabel("100%", toolbar)
        self._lbl_zoom_pct.setProperty("class", "valBadge")
        tb_layout.addWidget(self._lbl_zoom_pct)

        tb_layout.addStretch(1)

        # Mode toggles: Split, Original, Enhanced
        btn_mode_split = QToolButton(toolbar)
        btn_mode_split.setText("◫ Split")
        btn_mode_split.setToolTip("Interactive Split View")
        btn_mode_split.clicked.connect(lambda: self._viewer.set_split_mode("split"))
        tb_layout.addWidget(btn_mode_split)

        btn_mode_enh = QToolButton(toolbar)
        btn_mode_enh.setText("⬛ Enh.")
        btn_mode_enh.setToolTip("Show Enhanced Only")
        btn_mode_enh.clicked.connect(lambda: self._viewer.set_split_mode("enhanced"))
        tb_layout.addWidget(btn_mode_enh)

        btn_mode_orig = QToolButton(toolbar)
        btn_mode_orig.setText("⬜ Orig.")
        btn_mode_orig.setToolTip("Show Original Only")
        btn_mode_orig.clicked.connect(lambda: self._viewer.set_split_mode("original"))
        tb_layout.addWidget(btn_mode_orig)

        layout.addWidget(toolbar)

        # Central Image Viewer
        self._viewer = SplitImageViewer(container)
        layout.addWidget(self._viewer, 1)

        # Bottom Filmstrip Carousel
        self._filmstrip = FilmstripCarousel(container)
        self._filmstrip.image_selected.connect(self._on_image_selected)
        layout.addWidget(self._filmstrip)

        return container

    def _create_right_sidebar(self) -> QWidget:
        panel = QFrame(self)
        panel.setProperty("class", "panelCard")
        panel.setFixedWidth(410)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Top Sub-Tabs: Sensitivity, Precision Map, Difference, Histogram
        subtab_layout = QHBoxLayout()
        subtab_layout.setSpacing(4)
        self._subtab_group = QtWidgets.QButtonGroup(self)

        subtabs = ["Sensitivity", "Precision Map", "Difference", "Histogram"]
        for i, st in enumerate(subtabs):
            btn = QPushButton(st, panel)
            btn.setProperty("class", "subTab")
            btn.setCheckable(True)
            if i == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked=False, t=st: self._overlay_viewer.set_active_tab(t))
            self._subtab_group.addButton(btn, i)
            subtab_layout.addWidget(btn)

        layout.addLayout(subtab_layout)

        # Heatmap / Overlay Viewport
        self._overlay_viewer = HeatmapOverlayViewer(panel)
        self._overlay_viewer.setFixedHeight(140)
        layout.addWidget(self._overlay_viewer)

        # Row 1: Tile Allocation & Key Results
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        # Card: Tile Allocation
        card_alloc = QFrame(panel)
        card_alloc.setProperty("class", "subCard")
        alloc_layout = QVBoxLayout(card_alloc)
        alloc_layout.setContentsMargins(6, 6, 6, 6)
        alloc_layout.setSpacing(4)
        lbl_alloc = QLabel("TILE ALLOCATION", card_alloc)
        lbl_alloc.setProperty("class", "sectionHeader")
        alloc_layout.addWidget(lbl_alloc)
        self._donut_chart = DonutChartWidget(card_alloc)
        alloc_layout.addWidget(self._donut_chart)
        row1.addWidget(card_alloc, 1)

        # Card: Key Results
        card_results = QFrame(panel)
        card_results.setProperty("class", "subCard")
        res_layout = QVBoxLayout(card_results)
        res_layout.setContentsMargins(6, 6, 6, 6)
        res_layout.setSpacing(4)
        lbl_res = QLabel("KEY RESULTS (vs FP32)", card_results)
        lbl_res.setProperty("class", "sectionHeader")
        res_layout.addWidget(lbl_res)

        self._lbl_psnr = self._create_metric_row("PSNR", "-- dB", res_layout)
        self._lbl_ssim = self._create_metric_row("SSIM", "--", res_layout)
        self._lbl_ciede = self._create_metric_row("ΔE00 (mean)", "--", res_layout)
        self._lbl_l2 = self._create_metric_row("Rel. L2 Error", "-- %", res_layout)
        row1.addWidget(card_results, 1)

        layout.addLayout(row1)

        # Row 2: Runtime & Hardware + System Info
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        # Card: Runtime & Hardware Bar Chart
        card_perf = QFrame(panel)
        card_perf.setProperty("class", "subCard")
        perf_layout = QVBoxLayout(card_perf)
        perf_layout.setContentsMargins(6, 6, 6, 6)
        perf_layout.setSpacing(4)
        lbl_perf = QLabel("RUNTIME & HARDWARE", card_perf)
        lbl_perf.setProperty("class", "sectionHeader")
        perf_layout.addWidget(lbl_perf)
        self._perf_chart = PerformanceBarChart(card_perf)
        perf_layout.addWidget(self._perf_chart)
        row2.addWidget(card_perf, 1)

        # Card: System Info
        card_sys = QFrame(panel)
        card_sys.setProperty("class", "subCard")
        sys_layout = QVBoxLayout(card_sys)
        sys_layout.setContentsMargins(6, 6, 6, 6)
        sys_layout.setSpacing(4)
        lbl_sys = QLabel("SYSTEM INFO", card_sys)
        lbl_sys.setProperty("class", "sectionHeader")
        sys_layout.addWidget(lbl_sys)

        self._lbl_cpu = QLabel("CPU: Detecting...", card_sys)
        self._lbl_cpu.setFont(QFont("sans-serif", 8, QFont.Bold))
        sys_layout.addWidget(self._lbl_cpu)

        self._lbl_avx2 = self._create_sys_row("AVX2", False, sys_layout)
        self._lbl_fma = self._create_sys_row("FMA", False, sys_layout)
        self._lbl_f16c = self._create_sys_row("F16C", False, sys_layout)
        row2.addWidget(card_sys, 1)

        layout.addLayout(row2)

        return panel

    def _create_metric_row(self, label: str, init_val: str, parent_layout: QVBoxLayout) -> QLabel:
        row = QHBoxLayout()
        lbl = QLabel(label, self)
        lbl.setProperty("class", "subHeader")
        row.addWidget(lbl)
        val = QLabel(init_val, self)
        val.setFont(QFont("sans-serif", 9, QFont.Bold))
        val.setAlignment(Qt.AlignRight)
        row.addWidget(val)
        parent_layout.addLayout(row)
        return val

    def _create_sys_row(self, feat: str, enabled: bool, parent_layout: QVBoxLayout) -> QLabel:
        row = QHBoxLayout()
        lbl = QLabel(feat, self)
        lbl.setProperty("class", "subHeader")
        row.addWidget(lbl)
        val = QLabel("✓" if enabled else "✗", self)
        val.setStyleSheet(f"color: {ACCENT_GREEN if enabled else ACCENT_RED}; font-weight: bold;")
        val.setAlignment(Qt.AlignRight)
        row.addWidget(val)
        parent_layout.addLayout(row)
        return val

    def _create_status_bar(self) -> QStatusBar:
        sb = QStatusBar(self)
        self._status_icon = QLabel("● READY", self)
        self._status_icon.setStyleSheet(f"color: {ACCENT_GREEN}; font-weight: bold; font-size: 10px;")
        sb.addWidget(self._status_icon)

        self._status_text = QLabel("Ready for adaptive enhancement | Select image or parameters to execute", self)
        sb.addWidget(self._status_text, 1)

        lbl_version = QLabel("MIRAGE v0.9.0-ALPHA", self)
        lbl_version.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 600; font-size: 10px;")
        sb.addPermanentWidget(lbl_version)
        return sb

    # -----------------------------------------------------------------------
    # Secondary Views: Analyze, Benchmark, Experiments, Results
    # -----------------------------------------------------------------------
    def _create_analyze_view(self) -> QWidget:
        view = QWidget(self)
        layout = QVBoxLayout(view)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        lbl_header = QLabel("Sensitivity Analyzer & Tile Feature Breakdown", view)
        lbl_header.setFont(QFont("sans-serif", 14, QFont.Bold))
        layout.addWidget(lbl_header)

        desc = QLabel(
            "Inspect tile-based sensitivity scores computed across color variation, gradient energy, "
            "luminance variance, and spatial frequency. Guidance threshold determines FP32/FP16 boundary.",
            view,
        )
        desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(desc)

        self._analyze_table = QTableWidget(view)
        self._analyze_table.setColumnCount(8)
        self._analyze_table.setHorizontalHeaderLabels([
            "Tile ID", "X", "Y", "Width", "Height", "Partial", "Precision", "Sensitivity Score"
        ])
        self._analyze_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._analyze_table, 1)

        return view

    def _create_benchmark_view(self) -> QWidget:
        view = QWidget(self)
        layout = QVBoxLayout(view)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        lbl_header = QLabel("Hardware Microbenchmarks & Performance Evaluation", view)
        lbl_header.setFont(QFont("sans-serif", 14, QFont.Bold))
        layout.addWidget(lbl_header)

        desc = QLabel(
            "Evaluation of AVX2 vectorization, scalar fallbacks, and emulated FP16 arithmetic. "
            "Data loaded directly from verified hardware performance runs in benchmarks/results/benchmark_results.csv.",
            view,
        )
        desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(desc)

        self._bench_table = QTableWidget(view)
        self._bench_table.setColumnCount(6)
        self._bench_table.setHorizontalHeaderLabels([
            "Benchmark Name", "Resolution", "Backend", "Precision", "Time (ms)", "Throughput (MPix/s)"
        ])
        self._bench_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._bench_table, 1)

        self._load_benchmark_csv()
        return view

    def _create_experiments_view(self) -> QWidget:
        view = QWidget(self)
        view.setObjectName("experimentsView")
        layout = QVBoxLayout(view)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        # Persistent Workspace Header (No phase numbers exposed)
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(12)

        hdr_text_col = QVBoxLayout()
        hdr_text_col.setSpacing(2)
        self._lbl_exp_page_title = QLabel("Research & Experiments", view)
        self._lbl_exp_page_title.setFont(studio_font(self._lbl_exp_page_title, point_size=15, bold=True))
        self._lbl_exp_page_title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        hdr_text_col.addWidget(self._lbl_exp_page_title)

        self._lbl_exp_page_subtitle = QLabel(
            "Explore adaptive precision, performance characteristics and experimental findings.", view
        )
        self._lbl_exp_page_subtitle.setFont(studio_font(self._lbl_exp_page_subtitle, point_size=9, bold=False))
        self._lbl_exp_page_subtitle.setStyleSheet("color: #9fb0c0;")
        hdr_text_col.addWidget(self._lbl_exp_page_subtitle)
        hdr_row.addLayout(hdr_text_col, 1)

        self._btn_exp_back = QPushButton("←  All Categories", view)
        self._btn_exp_back.setProperty("class", "expBtn")
        self._btn_exp_back.clicked.connect(self._show_experiments_landing)
        self._btn_exp_back.setVisible(False)
        hdr_row.addWidget(self._btn_exp_back)

        self._btn_exp_open_runner = QPushButton("⚙  Run Experiments...", view)
        self._btn_exp_open_runner.setProperty("class", "expBtn")
        self._btn_exp_open_runner.setToolTip("Execute benchmark or experiment workflows separately from viewing saved results")
        self._btn_exp_open_runner.clicked.connect(self._show_experiment_runner_view)
        hdr_row.addWidget(self._btn_exp_open_runner)

        layout.addLayout(hdr_row)

        # Stack: 0 = Category Landing Page | 1 = Category/Experiment Explorer | 2 = Separate Experiment Runner
        self._exp_stack = QStackedWidget(view)
        layout.addWidget(self._exp_stack, 1)

        # -------------------------------------------------------------------
        # Page 0: Category Landing Page (4 clean cards, no auto-opened chart)
        # -------------------------------------------------------------------
        landing_page = QWidget(view)
        landing_layout = QVBoxLayout(landing_page)
        landing_layout.setContentsMargins(0, 6, 0, 0)
        landing_layout.setSpacing(14)

        grid = QGridLayout()
        grid.setSpacing(16)
        plots_dir = (PROJECT_ROOT / "benchmarks" / "plots").resolve()
        self._exp_category_cards: Dict[str, CategoryCardWidget] = {}

        for idx, cat in enumerate(RESEARCH_CATEGORIES):
            existing_figs = [
                exp for exp in cat["experiments"]
                if (plots_dir / exp["plot_file"]).is_file()
            ]
            fig_count = len(existing_figs)
            preview_path = str((plots_dir / cat["preview_plot"]).resolve())
            card = CategoryCardWidget(cat, fig_count, preview_path, landing_page)
            card.clicked.connect(self._open_experiment_category)
            self._exp_category_cards[cat["id"]] = card
            grid.addWidget(card, idx // 2, idx % 2)

        landing_layout.addLayout(grid, 1)
        self._exp_stack.addWidget(landing_page)

        # -------------------------------------------------------------------
        # Page 1: Individual Experiment Workspace
        # -------------------------------------------------------------------
        detail_page = QWidget(view)
        detail_layout = QVBoxLayout(detail_page)
        detail_layout.setContentsMargins(0, 2, 0, 0)
        detail_layout.setSpacing(8)

        # Top navigation strip: Category tabs + Experiment pill switcher
        nav_strip = QHBoxLayout()
        nav_strip.setSpacing(8)

        self._exp_cat_btn_group = QtWidgets.QButtonGroup(detail_page)
        self._exp_cat_btn_group.setExclusive(True)
        for cat in RESEARCH_CATEGORIES:
            btn = QPushButton(cat["name"], detail_page)
            btn.setProperty("class", "expSegBtn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked=False, cid=cat["id"]: self._open_experiment_category(cid))
            btn.setProperty("categoryId", cat["id"])
            self._exp_cat_btn_group.addButton(btn)
            nav_strip.addWidget(btn)

        detail_layout.addLayout(nav_strip)

        # Dedicated Experiment Selector Strip inside Left Panel (zero crowding, zero clipped ampersands)
        exp_pill_row = QHBoxLayout()
        exp_pill_row.setSpacing(6)
        lbl_exp_pick = QLabel("Experiments:", detail_page)
        lbl_exp_pick.setFont(studio_font(lbl_exp_pick, point_size=9, bold=True))
        lbl_exp_pick.setStyleSheet("color: #9fb0c0;")
        exp_pill_row.addWidget(lbl_exp_pick)
        self._exp_pills_host = QWidget(detail_page)
        self._exp_pills_layout = QHBoxLayout(self._exp_pills_host)
        self._exp_pills_layout.setContentsMargins(0, 0, 0, 0)
        self._exp_pills_layout.setSpacing(6)
        self._exp_pill_group = QtWidgets.QButtonGroup(detail_page)
        self._exp_pill_group.setExclusive(True)
        exp_pill_row.addWidget(self._exp_pills_host)
        exp_pill_row.addStretch(1)
        detail_layout.addLayout(exp_pill_row)

        # Main 2-column content: Left = Figure + Research Summary | Right = Interactive CSV Explorer
        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        # Left Panel (#28333D)
        fig_panel = QFrame(detail_page)
        fig_panel.setProperty("class", "expPanel")
        fig_panel_layout = QVBoxLayout(fig_panel)
        fig_panel_layout.setContentsMargins(12, 10, 12, 10)
        fig_panel_layout.setSpacing(6)

        self._lbl_exp_title = QLabel("", fig_panel)
        self._lbl_exp_title.setFont(studio_font(self._lbl_exp_title, point_size=12, bold=True))
        self._lbl_exp_title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        self._lbl_exp_title.setWordWrap(True)
        fig_panel_layout.addWidget(self._lbl_exp_title)

        q_row = QHBoxLayout()
        q_row.setSpacing(8)
        self._lbl_exp_question = QLabel("", fig_panel)
        self._lbl_exp_question.setFont(studio_font(self._lbl_exp_question, point_size=9, bold=True))
        self._lbl_exp_question.setStyleSheet(f"color: {SLATE_ACCENT_LIGHT};")
        self._lbl_exp_question.setWordWrap(True)
        q_row.addWidget(self._lbl_exp_question, 1)

        self._lbl_exp_tech_meta = QLabel("", fig_panel)
        self._lbl_exp_tech_meta.setFont(studio_font(self._lbl_exp_tech_meta, point_size=7, bold=False))
        self._lbl_exp_tech_meta.setStyleSheet(
            f"color: #9fb0c0; background-color: {SLATE_BG}; border: 1px solid {SLATE_SURFACE}; border-radius: 4px; padding: 2px 6px;"
        )
        self._lbl_exp_tech_meta.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        q_row.addWidget(self._lbl_exp_tech_meta, 0, Qt.AlignTop)
        fig_panel_layout.addLayout(q_row)

        self._figure_viewer = FigureViewerWidget(fig_panel, canvas_bg=SLATE_BG, border_color=SLATE_SURFACE)
        fig_panel_layout.addWidget(self._figure_viewer, 1)

        # Explanation, findings, and dataset metadata adjustment surface (#394651)
        info_surface = QFrame(fig_panel)
        info_surface.setProperty("class", "expSurface")
        info_layout = QVBoxLayout(info_surface)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.setSpacing(4)

        self._lbl_exp_explanation = QLabel("", info_surface)
        self._lbl_exp_explanation.setWordWrap(True)
        self._lbl_exp_explanation.setFont(studio_font(self._lbl_exp_explanation, point_size=8, bold=False))
        self._lbl_exp_explanation.setStyleSheet(f"color: {TEXT_PRIMARY};")
        info_layout.addWidget(self._lbl_exp_explanation)

        self._lbl_exp_findings = QLabel("", info_surface)
        self._lbl_exp_findings.setWordWrap(True)
        self._lbl_exp_findings.setFont(studio_font(self._lbl_exp_findings, point_size=8, bold=False))
        self._lbl_exp_findings.setStyleSheet("color: #dbeafe;")
        info_layout.addWidget(self._lbl_exp_findings)

        self._lbl_exp_dataset = QLabel("", info_surface)
        self._lbl_exp_dataset.setWordWrap(True)
        self._lbl_exp_dataset.setFont(studio_font(self._lbl_exp_dataset, point_size=7, bold=False))
        self._lbl_exp_dataset.setStyleSheet("color: #cbd5e1;")
        info_layout.addWidget(self._lbl_exp_dataset)

        fig_panel_layout.addWidget(info_surface)
        content_row.addWidget(fig_panel, 13)

        # Right Panel: Interactive CSV Explorer & Metric Summary (#28333D)
        data_panel = QFrame(detail_page)
        data_panel.setProperty("class", "expPanel")
        data_layout = QVBoxLayout(data_panel)
        data_layout.setContentsMargins(10, 10, 10, 10)
        data_layout.setSpacing(7)

        data_hdr = QHBoxLayout()
        lbl_data_title = QLabel("EXPERIMENT DATA EXPLORER", data_panel)
        lbl_data_title.setFont(studio_font(lbl_data_title, point_size=9, bold=True))
        lbl_data_title.setStyleSheet(f"color: {TEXT_PRIMARY}; letter-spacing: 0.8px;")
        data_hdr.addWidget(lbl_data_title)
        data_hdr.addStretch(1)

        self._lbl_exp_csv_badge = QLabel("0 rows", data_panel)
        self._lbl_exp_csv_badge.setFont(studio_font(self._lbl_exp_csv_badge, point_size=8, bold=True))
        self._lbl_exp_csv_badge.setStyleSheet(
            f"color: #ffffff; background-color: {SLATE_SURFACE}; border-radius: 4px; padding: 2px 8px;"
        )
        data_hdr.addWidget(self._lbl_exp_csv_badge)
        data_layout.addLayout(data_hdr)

        # Interactive filter controls (#394651)
        filter_box = QFrame(data_panel)
        filter_box.setProperty("class", "expSurface")
        filter_grid = QGridLayout(filter_box)
        filter_grid.setContentsMargins(8, 6, 8, 6)
        filter_grid.setSpacing(6)

        self._exp_filter_labels: List[QLabel] = []
        self._exp_filter_widgets: List[QComboBox] = []
        for idx in range(2):
            lbl_f = QLabel("Filter:", filter_box)
            lbl_f.setFont(studio_font(lbl_f, point_size=8, bold=True))
            cmb_f = QComboBox(filter_box)
            cmb_f.currentIndexChanged.connect(self._on_exp_filter_changed)
            self._exp_filter_labels.append(lbl_f)
            self._exp_filter_widgets.append(cmb_f)
            filter_grid.addWidget(lbl_f, idx, 0)
            filter_grid.addWidget(cmb_f, idx, 1)

        lbl_m = QLabel("Metric:", filter_box)
        lbl_m.setFont(studio_font(lbl_m, point_size=8, bold=True))
        self._combo_exp_metric = QComboBox(filter_box)
        self._combo_exp_metric.currentIndexChanged.connect(self._on_exp_filter_changed)
        filter_grid.addWidget(lbl_m, 2, 0)
        filter_grid.addWidget(self._combo_exp_metric, 2, 1)

        btn_reset_filters = QPushButton("↺ Reset", filter_box)
        btn_reset_filters.setProperty("class", "expBtn")
        btn_reset_filters.clicked.connect(self._reset_exp_filters)
        filter_grid.addWidget(btn_reset_filters, 2, 2)
        filter_grid.setColumnStretch(1, 1)
        data_layout.addWidget(filter_box)

        # Live summary computed cleanly from the filtered CSV rows
        self._lbl_exp_stats_bar = QLabel("Metric summary: --", data_panel)
        self._lbl_exp_stats_bar.setFont(studio_font(self._lbl_exp_stats_bar, point_size=8, bold=False))
        self._lbl_exp_stats_bar.setStyleSheet(
            f"color: #cbd5e1; background-color: {SLATE_BG}; border: 1px solid {SLATE_SURFACE}; border-radius: 4px; padding: 4px 8px;"
        )
        data_layout.addWidget(self._lbl_exp_stats_bar)

        self._exp_csv_table = QTableWidget(data_panel)
        self._exp_csv_table.setAlternatingRowColors(True)
        self._exp_csv_table.setSortingEnabled(False)
        exp_header = self._exp_csv_table.horizontalHeader()
        exp_header.setStretchLastSection(False)
        exp_header.setSectionsClickable(True)
        exp_header.setSortIndicatorShown(True)
        exp_header.sectionClicked.connect(self._on_exp_table_sort_requested)
        data_layout.addWidget(self._exp_csv_table, 1)

        content_row.addWidget(data_panel, 10)
        detail_layout.addLayout(content_row, 1)
        self._exp_stack.addWidget(detail_page)

        # -------------------------------------------------------------------
        # Page 2: Separated Experiment Runner Workflow
        # -------------------------------------------------------------------
        runner_page = QWidget(view)
        runner_layout = QVBoxLayout(runner_page)
        runner_layout.setContentsMargins(0, 4, 0, 0)
        runner_layout.setSpacing(10)

        runner_card = QFrame(runner_page)
        runner_card.setProperty("class", "expPanel")
        rc_layout = QVBoxLayout(runner_card)
        rc_layout.setContentsMargins(16, 14, 16, 14)
        rc_layout.setSpacing(10)

        rc_title = QLabel("Execute C++ Research Experiments", runner_card)
        rc_title.setFont(studio_font(rc_title, point_size=12, bold=True))
        rc_layout.addWidget(rc_title)

        rc_desc = QLabel(
            "Runs the compiled C++ experiment suite (random ablation, feature ablation, tile sensitivity validation, and Pareto sweeps) "
            "asynchronously in the background without blocking existing result inspection.",
            runner_card,
        )
        rc_desc.setWordWrap(True)
        rc_desc.setFont(studio_font(rc_desc, point_size=9, bold=False))
        rc_desc.setStyleSheet("color: #cbd5e1;")
        rc_layout.addWidget(rc_desc)

        cfg_box = QFrame(runner_card)
        cfg_box.setProperty("class", "expSurface")
        cfg_grid = QGridLayout(cfg_box)
        cfg_grid.setContentsMargins(10, 10, 10, 10)
        cfg_grid.setSpacing(8)

        cfg_grid.addWidget(QLabel("Dataset Directory:", cfg_box), 0, 0)
        self._edit_exp_dataset = QLineEdit(str((PROJECT_ROOT / "data" / "experiments").resolve()), cfg_box)
        cfg_grid.addWidget(self._edit_exp_dataset, 0, 1)

        cfg_grid.addWidget(QLabel("Output Directory:", cfg_box), 1, 0)
        self._edit_exp_output = QLineEdit(str((PROJECT_ROOT / "benchmarks" / "experiments").resolve()), cfg_box)
        cfg_grid.addWidget(self._edit_exp_output, 1, 1)

        btn_browse_out = QPushButton("Browse...", cfg_box)
        btn_browse_out.setProperty("class", "expBtn")
        btn_browse_out.clicked.connect(self._on_browse_exp_output)
        cfg_grid.addWidget(btn_browse_out, 1, 2)

        cfg_grid.addWidget(QLabel("Random Seeds (N):", cfg_box), 2, 0)
        self._spin_exp_seeds = QSpinBox(cfg_box)
        self._spin_exp_seeds.setRange(1, 50)
        self._spin_exp_seeds.setValue(30)
        self._spin_exp_seeds.setFixedWidth(90)
        cfg_grid.addWidget(self._spin_exp_seeds, 2, 1, Qt.AlignLeft)

        self._chk_exp_replot = QtWidgets.QCheckBox("Regenerate statistical charts via analyze_experiments.py after completion", cfg_box)
        self._chk_exp_replot.setStyleSheet(f"color: {TEXT_PRIMARY};")
        cfg_grid.addWidget(self._chk_exp_replot, 3, 1, 1, 2)

        rc_layout.addWidget(cfg_box)

        ctrl_row = QHBoxLayout()
        self._btn_exp_run_exec = QPushButton("▶  Run Experiment Suite", runner_card)
        self._btn_exp_run_exec.setProperty("class", "expPrimaryBtn")
        self._btn_exp_run_exec.clicked.connect(self._on_start_experiment_runner)
        ctrl_row.addWidget(self._btn_exp_run_exec)

        self._btn_exp_cancel_exec = QPushButton("■  Cancel", runner_card)
        self._btn_exp_cancel_exec.setProperty("class", "expBtn")
        self._btn_exp_cancel_exec.setEnabled(False)
        self._btn_exp_cancel_exec.clicked.connect(self._on_cancel_experiment_runner)
        ctrl_row.addWidget(self._btn_exp_cancel_exec)

        self._lbl_exp_runner_status = QLabel("Idle — Ready to run", runner_card)
        self._lbl_exp_runner_status.setFont(studio_font(self._lbl_exp_runner_status, point_size=9, bold=True))
        self._lbl_exp_runner_status.setStyleSheet("color: #9fb0c0;")
        ctrl_row.addWidget(self._lbl_exp_runner_status, 1)
        rc_layout.addLayout(ctrl_row)

        self._txt_exp_log = QtWidgets.QPlainTextEdit(runner_card)
        self._txt_exp_log.setReadOnly(True)
        self._txt_exp_log.setPlaceholderText("Experiment execution logs will stream here...")
        rc_layout.addWidget(self._txt_exp_log, 1)

        runner_layout.addWidget(runner_card, 1)
        self._exp_stack.addWidget(runner_page)

        # Always start on the Category Landing Page (never opens a chart or reruns seeds automatically)
        self._show_experiments_landing()
        return view

    def _create_results_view(self) -> QWidget:
        view = QWidget(self)
        outer = QVBoxLayout(view)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(0)

        self._results_stack = QStackedWidget(view)
        outer.addWidget(self._results_stack, 1)

        # Page 0: Empty State before any successful backend execution
        empty_page = QWidget(view)
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.setAlignment(Qt.AlignCenter)

        empty_card = QFrame(empty_page)
        empty_card.setProperty("class", "panelCard")
        empty_card.setFixedWidth(560)
        empty_card.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Maximum)
        card_layout = QVBoxLayout(empty_card)
        card_layout.setContentsMargins(36, 32, 36, 32)
        card_layout.setSpacing(12)

        lbl_empty_badge = QLabel("RESULTS DASHBOARD", empty_card)
        lbl_empty_badge.setProperty("class", "sectionHeader")
        lbl_empty_badge.setAlignment(Qt.AlignCenter)
        lbl_empty_badge.setFont(studio_font(lbl_empty_badge, point_size=9, bold=True))
        card_layout.addWidget(lbl_empty_badge)

        self._lbl_results_empty_title = QLabel("No enhancement results yet", empty_card)
        self._lbl_results_empty_title.setAlignment(Qt.AlignCenter)
        self._lbl_results_empty_title.setFont(studio_font(self._lbl_results_empty_title, point_size=15, bold=True))
        self._lbl_results_empty_title.setStyleSheet(f"color: {TEXT_PRIMARY}; padding: 2px 0;")
        card_layout.addWidget(self._lbl_results_empty_title)

        self._lbl_results_empty_body = QLabel(
            "Run an enhancement to view image comparisons, precision allocation and quality metrics.",
            empty_card,
        )
        self._lbl_results_empty_body.setAlignment(Qt.AlignCenter)
        self._lbl_results_empty_body.setWordWrap(True)
        self._lbl_results_empty_body.setMinimumHeight(44)
        self._lbl_results_empty_body.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self._lbl_results_empty_body.setFont(studio_font(self._lbl_results_empty_body, point_size=10, bold=False))
        self._lbl_results_empty_body.setStyleSheet(f"color: {TEXT_SECONDARY}; line-height: 1.4;")
        card_layout.addWidget(self._lbl_results_empty_body)

        card_layout.addSpacing(6)
        btn_go_enhance = QPushButton("▶  Go to Enhance", empty_card)
        btn_go_enhance.setObjectName("btnPrimary")
        btn_go_enhance.setFixedWidth(200)
        btn_go_enhance.clicked.connect(lambda: self._switch_view(0))
        card_layout.addWidget(btn_go_enhance, 0, Qt.AlignCenter)

        empty_layout.addWidget(empty_card)
        self._results_stack.addWidget(empty_page)

        # Page 1: Dynamic Latest-Execution Results Dashboard
        dash_page = QWidget(view)
        dash_layout = QVBoxLayout(dash_page)
        dash_layout.setContentsMargins(0, 0, 0, 0)
        dash_layout.setSpacing(8)

        # Dashboard Header + Export Actions (Section E)
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)

        hdr_text_col = QVBoxLayout()
        hdr_text_col.setSpacing(2)
        lbl_dash_hdr = QLabel("Latest Enhancement Execution Results", dash_page)
        lbl_dash_hdr.setFont(studio_font(lbl_dash_hdr, point_size=13, bold=True))
        hdr_text_col.addWidget(lbl_dash_hdr)

        self._res_header_meta = QLabel("", dash_page)
        self._res_header_meta.setFont(studio_font(self._res_header_meta, point_size=8, bold=False))
        self._res_header_meta.setStyleSheet(f"color: {TEXT_SECONDARY};")
        hdr_text_col.addWidget(self._res_header_meta)
        hdr_row.addLayout(hdr_text_col, 1)

        btn_save_img = QPushButton("💾  Save Enhanced Image...", dash_page)
        btn_save_img.setFont(studio_font(btn_save_img, point_size=8, bold=False))
        btn_save_img.clicked.connect(self._on_save_result_dialog)
        hdr_row.addWidget(btn_save_img)

        btn_export_json = QPushButton("📄  Export Report...", dash_page)
        btn_export_json.setFont(studio_font(btn_export_json, point_size=8, bold=False))
        btn_export_json.clicked.connect(self._on_export_report_dialog)
        hdr_row.addWidget(btn_export_json)

        btn_save_viz = QPushButton("🖼  Save Visualizations...", dash_page)
        btn_save_viz.setFont(studio_font(btn_save_viz, point_size=8, bold=False))
        btn_save_viz.clicked.connect(self._on_save_visualizations_dialog)
        hdr_row.addWidget(btn_save_viz)

        dash_layout.addLayout(hdr_row)

        # Split Main Area: Left (Comparison Viewer) | Right (Summary + Metrics + Precision Visualizations)
        body_row = QHBoxLayout()
        body_row.setSpacing(10)

        # Section A: Main Image Comparison Card
        comp_card = QFrame(dash_page)
        comp_card.setProperty("class", "panelCard")
        comp_layout = QVBoxLayout(comp_card)
        comp_layout.setContentsMargins(8, 8, 8, 8)
        comp_layout.setSpacing(6)

        comp_tb = QHBoxLayout()
        comp_tb.setSpacing(5)
        lbl_comp = QLabel("IMAGE COMPARISON", comp_card)
        lbl_comp.setProperty("class", "sectionHeader")
        lbl_comp.setFont(studio_font(lbl_comp, point_size=9, bold=True))
        comp_tb.addWidget(lbl_comp)
        comp_tb.addStretch(1)

        for label, cb in [
            ("🔍+", lambda: self._res_viewer.zoom_in()),
            ("🔍-", lambda: self._res_viewer.zoom_out()),
            ("1:1", lambda: self._res_viewer.reset_zoom()),
            ("⛶ Fit", lambda: self._res_viewer.fit_to_screen()),
            ("◫ Split", lambda: self._res_viewer.set_split_mode("split")),
            ("⬛ Enh.", lambda: self._res_viewer.set_split_mode("enhanced")),
            ("⬜ Orig.", lambda: self._res_viewer.set_split_mode("original")),
        ]:
            tb_btn = QToolButton(comp_card)
            tb_btn.setText(label)
            tb_btn.clicked.connect(cb)
            comp_tb.addWidget(tb_btn)

        comp_layout.addLayout(comp_tb)

        self._res_viewer = SplitImageViewer(comp_card)
        comp_layout.addWidget(self._res_viewer, 1)

        meta_bar = QFrame(comp_card)
        meta_bar.setProperty("class", "subCard")
        meta_layout = QHBoxLayout(meta_bar)
        meta_layout.setContentsMargins(8, 4, 8, 4)
        meta_layout.setSpacing(10)

        self._res_lbl_img_dims = QLabel("Dimensions: --", meta_bar)
        self._res_lbl_img_dims.setFont(studio_font(self._res_lbl_img_dims, point_size=8, bold=False))
        self._res_lbl_img_dims.setStyleSheet(f"color: {TEXT_PRIMARY};")
        meta_layout.addWidget(self._res_lbl_img_dims)

        self._res_lbl_img_format = QLabel("Format: --", meta_bar)
        self._res_lbl_img_format.setFont(studio_font(self._res_lbl_img_format, point_size=8, bold=False))
        self._res_lbl_img_format.setStyleSheet(f"color: {TEXT_SECONDARY};")
        meta_layout.addWidget(self._res_lbl_img_format, 1)

        comp_layout.addWidget(meta_bar)
        body_row.addWidget(comp_card, 11)

        # Right Analytics Area (Compact cards so Sections B, C, D all appear cleanly without scrolling at 1340x840)
        right_scroll = QScrollArea(dash_page)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(490)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 2, 0)
        right_layout.setSpacing(8)

        # Top row of right column: Section B (Execution Summary) | Section C (Quality Metrics) side-by-side
        cards_top_row = QHBoxLayout()
        cards_top_row.setSpacing(8)

        # Section B: Execution Summary
        card_exec = QFrame(right_container)
        card_exec.setProperty("class", "panelCard")
        exec_layout = QVBoxLayout(card_exec)
        exec_layout.setContentsMargins(8, 7, 8, 7)
        exec_layout.setSpacing(3)
        lbl_exec = QLabel("EXECUTION SUMMARY", card_exec)
        lbl_exec.setProperty("class", "sectionHeader")
        lbl_exec.setFont(studio_font(lbl_exec, point_size=9, bold=True))
        exec_layout.addWidget(lbl_exec)

        self._res_lbl_mode = self._create_metric_row("Mode", "--", exec_layout)
        self._res_lbl_backend = self._create_metric_row("Backend", "--", exec_layout)
        self._res_lbl_tilesize = self._create_metric_row("Tile Size", "--", exec_layout)
        self._res_lbl_threshold = self._create_metric_row("Threshold (τ)", "--", exec_layout)
        self._res_lbl_total_tiles = self._create_metric_row("Total Tiles", "--", exec_layout)
        self._res_lbl_fp32_tiles = self._create_metric_row("FP32 Tiles", "--", exec_layout)
        self._res_lbl_fp16_tiles = self._create_metric_row("FP16 Tiles", "--", exec_layout)
        self._res_lbl_runtime = self._create_metric_row("Runtime", "--", exec_layout)
        exec_layout.addStretch(1)
        cards_top_row.addWidget(card_exec, 1)

        # Section C: Quality Metrics (vs FP32 Reference)
        card_qm = QFrame(right_container)
        card_qm.setProperty("class", "panelCard")
        qm_layout = QVBoxLayout(card_qm)
        qm_layout.setContentsMargins(8, 7, 8, 7)
        qm_layout.setSpacing(3)

        lbl_qm = QLabel("QUALITY METRICS", card_qm)
        lbl_qm.setProperty("class", "sectionHeader")
        lbl_qm.setFont(studio_font(lbl_qm, point_size=9, bold=True))
        qm_layout.addWidget(lbl_qm)

        self._res_lbl_ref_note = QLabel("vs FP32 Reference Output", card_qm)
        self._res_lbl_ref_note.setFont(studio_font(self._res_lbl_ref_note, point_size=7, bold=False))
        self._res_lbl_ref_note.setStyleSheet(f"color: {ACCENT_PURPLE_LIGHT}; margin-bottom: 2px;")
        self._res_lbl_ref_note.setWordWrap(True)
        qm_layout.addWidget(self._res_lbl_ref_note)

        self._res_lbl_psnr = self._create_metric_row("PSNR", "--", qm_layout)
        self._res_lbl_ssim = self._create_metric_row("SSIM", "--", qm_layout)
        self._res_lbl_ciede = self._create_metric_row("Mean ΔE00", "--", qm_layout)
        self._res_lbl_mae = self._create_metric_row("MAE", "--", qm_layout)
        self._res_lbl_rmse = self._create_metric_row("RMSE", "--", qm_layout)
        self._res_lbl_max_err = self._create_metric_row("Max Abs Err", "--", qm_layout)
        self._res_lbl_l2 = self._create_metric_row("Rel. L2 Err", "--", qm_layout)
        qm_layout.addStretch(1)
        cards_top_row.addWidget(card_qm, 1)

        right_layout.addLayout(cards_top_row)

        # Section D: Precision Visualization Card (Donut + Sensitivity Heatmap + Tile Precision Map)
        card_vis = QFrame(right_container)
        card_vis.setProperty("class", "panelCard")
        vis_layout = QVBoxLayout(card_vis)
        vis_layout.setContentsMargins(8, 7, 8, 7)
        vis_layout.setSpacing(6)

        lbl_vis = QLabel("PRECISION VISUALIZATION", card_vis)
        lbl_vis.setProperty("class", "sectionHeader")
        lbl_vis.setFont(studio_font(lbl_vis, point_size=9, bold=True))
        vis_layout.addWidget(lbl_vis)

        self._res_donut = DonutChartWidget(card_vis)
        self._res_donut.setFixedHeight(86)
        vis_layout.addWidget(self._res_donut)

        maps_row = QHBoxLayout()
        maps_row.setSpacing(8)

        # Left Map: Actual Sensitivity Heatmap + TURBO Legend
        hm_col = QVBoxLayout()
        hm_col.setSpacing(3)
        lbl_hm = QLabel("Sensitivity Heatmap", card_vis)
        lbl_hm.setProperty("class", "subHeader")
        lbl_hm.setFont(studio_font(lbl_hm, point_size=8, bold=True))
        hm_col.addWidget(lbl_hm)
        self._res_heatmap_preview = ResultImagePreview("No heatmap generated", show_turbo_bar=True, parent=card_vis)
        self._res_heatmap_preview.setMinimumHeight(170)
        hm_col.addWidget(self._res_heatmap_preview, 1)
        lbl_hm_leg = QLabel("TURBO: 0.0 Low → 1.0 High", card_vis)
        lbl_hm_leg.setFont(studio_font(lbl_hm_leg, point_size=7, bold=False))
        lbl_hm_leg.setStyleSheet(f"color: {TEXT_SECONDARY};")
        hm_col.addWidget(lbl_hm_leg)
        maps_row.addLayout(hm_col, 1)

        # Right Map: Actual Tile Precision Allocation Map + Legend
        pm_col = QVBoxLayout()
        pm_col.setSpacing(3)
        lbl_pm = QLabel("Tile Precision Map", card_vis)
        lbl_pm.setProperty("class", "subHeader")
        lbl_pm.setFont(studio_font(lbl_pm, point_size=8, bold=True))
        pm_col.addWidget(lbl_pm)
        self._res_precision_preview = ResultImagePreview("No precision map generated", show_turbo_bar=False, parent=card_vis)
        self._res_precision_preview.setMinimumHeight(170)
        pm_col.addWidget(self._res_precision_preview, 1)
        lbl_pm_leg = QLabel(
            f"<span style='color:{ACCENT_BLUE};'>■</span> FP32 &nbsp;&nbsp; "
            f"<span style='color:{ACCENT_GREEN};'>■</span> FP16",
            card_vis,
        )
        lbl_pm_leg.setFont(studio_font(lbl_pm_leg, point_size=7, bold=False))
        pm_col.addWidget(lbl_pm_leg)
        maps_row.addLayout(pm_col, 1)

        vis_layout.addLayout(maps_row, 1)
        right_layout.addWidget(card_vis, 1)

        right_scroll.setWidget(right_container)
        body_row.addWidget(right_scroll, 11)

        dash_layout.addLayout(body_row, 1)
        self._results_stack.addWidget(dash_page)
        self._results_stack.setCurrentIndex(0)
        return view

    # -----------------------------------------------------------------------
    # Logic & Event Handlers
    # -----------------------------------------------------------------------
    def _switch_view(self, index: int) -> None:
        self._main_stack.setCurrentIndex(index)
        btn = self._nav_group.button(index)
        if btn and not btn.isChecked():
            btn.setChecked(True)
        if index == 3 and hasattr(self, "_exp_stack") and self._exp_stack.currentIndex() == 1:
            self._figure_viewer.update()
        elif index == 4 and self._last_result_telemetry:
            self._res_viewer.fit_to_screen()

    def _on_precision_mode_changed(self) -> None:
        mode = self._combo_mode.currentText()
        is_adaptive = "Adaptive" in mode
        self._slider_tau.setEnabled(is_adaptive)
        self._combo_tilesize.setEnabled(is_adaptive or "Mixed" in mode)
        self._combo_policy.setEnabled(is_adaptive)

    def _on_policy_changed(self) -> None:
        pol = self._combo_policy.currentText()
        if "30-Seed Random" in pol:
            self._switch_view(3)
            self._open_experiment_category("adaptive_precision", "random_vs_guided")

    def _on_open_image_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Input Image", str(PROJECT_ROOT / "data" / "experiments"), "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self._on_image_selected(path)

    def _on_save_result_dialog(self) -> None:
        if not self._last_result_output_path or not os.path.exists(self._last_result_output_path):
            QtWidgets.QMessageBox.information(self, "Save Result", "Please execute an enhancement run first.")
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save Enhanced Image", "enhanced_output.png", "PNG (*.png);;JPEG (*.jpg)")
        if dest:
            pix = QPixmap(self._last_result_output_path)
            if pix.save(dest):
                self._status_text.setText(f"Result saved to {dest}")
            else:
                QtWidgets.QMessageBox.warning(self, "Save Failed", f"Could not write enhanced image to {dest}.")

    def _on_export_report_dialog(self) -> None:
        if not self._last_result_telemetry:
            QtWidgets.QMessageBox.information(self, "Export Report", "No enhancement telemetry available to export.")
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Export Execution Report JSON", "mirage_telemetry_report.json", "JSON (*.json)")
        if dest:
            with open(dest, "w") as f:
                json.dump(self._last_result_telemetry, f, indent=2)
            self._status_text.setText(f"Execution report exported to {dest}")

    def _on_save_visualizations_dialog(self) -> None:
        hm = self._last_result_heatmap_path
        pm = self._last_result_precision_path
        has_hm = bool(hm and os.path.isfile(hm))
        has_pm = bool(pm and os.path.isfile(pm))
        if not has_hm and not has_pm:
            QtWidgets.QMessageBox.information(
                self,
                "Save Analysis Visualizations",
                "No sensitivity heatmap or tile precision map was generated for the latest run "
                "(spatial overlays are generated in Adaptive mode).",
            )
            return
        dest_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Analysis Visualizations", str(PROJECT_ROOT / "output")
        )
        if dest_dir:
            saved_files: List[str] = []
            stem = Path(self._last_result_input_path).stem if self._last_result_input_path else "mirage"
            if has_hm:
                hm_dest = os.path.join(dest_dir, f"{stem}_sensitivity_heatmap.png")
                shutil.copy2(hm, hm_dest)
                saved_files.append(Path(hm_dest).name)
            if has_pm:
                pm_dest = os.path.join(dest_dir, f"{stem}_precision_map.png")
                shutil.copy2(pm, pm_dest)
                saved_files.append(Path(pm_dest).name)
            self._status_text.setText(f"Saved visualizations to {dest_dir}: {', '.join(saved_files)}")

    def _on_image_selected(self, image_path: str) -> None:
        self._active_image_path = image_path
        self._viewer.set_images(image_path, image_path, "Input")
        self._viewer.fit_to_screen()
        self._status_text.setText(f"Loaded {Path(image_path).name} | Ready to enhance")

    # -----------------------------------------------------------------------
    # Research & Experiments Workspace Logic
    # -----------------------------------------------------------------------
    def _get_category_by_id(self, cat_id: str) -> Dict:
        for cat in RESEARCH_CATEGORIES:
            if cat["id"] == cat_id:
                return cat
        return RESEARCH_CATEGORIES[0]

    def _get_experiment_by_id(self, cat: Dict, exp_id: str) -> Dict:
        for exp in cat["experiments"]:
            if exp["id"] == exp_id:
                return exp
        return cat["experiments"][0]

    def _show_experiments_landing(self) -> None:
        self._exp_stack.setCurrentIndex(0)
        self._btn_exp_back.setVisible(False)
        self._lbl_exp_page_title.setText("Research & Experiments")
        self._lbl_exp_page_subtitle.setText(
            "Explore adaptive precision, performance characteristics and experimental findings."
        )

    def _show_experiment_runner_view(self) -> None:
        self._exp_stack.setCurrentIndex(2)
        self._btn_exp_back.setVisible(True)
        self._lbl_exp_page_title.setText("Research & Experiments — Execute Suite")
        self._lbl_exp_page_subtitle.setText(
            "Run the C++ experiment runner separately from viewing existing research outputs."
        )

    def _open_experiment_category(self, cat_id: str, exp_id: Optional[str] = None) -> None:
        cat = self._get_category_by_id(cat_id)
        self._active_exp_cat_id = cat["id"]
        target_exp = exp_id or cat["experiments"][0]["id"]

        for btn in self._exp_cat_btn_group.buttons():
            btn.setChecked(btn.property("categoryId") == cat["id"])

        # Rebuild human-readable experiment pill buttons for this category
        while self._exp_pills_layout.count():
            item = self._exp_pills_layout.takeAt(0)
            w = item.widget()
            if w:
                self._exp_pill_group.removeButton(w)
                w.hide()
                w.setParent(None)
                w.deleteLater()

        for exp in cat["experiments"]:
            btn = QPushButton(exp["short_title"].replace("&", "&&"), self._exp_pills_host)
            btn.setProperty("class", "expBtn")
            btn.setCheckable(True)
            btn.setProperty("expId", exp["id"])
            btn.clicked.connect(lambda checked=False, eid=exp["id"]: self._select_experiment(eid))
            self._exp_pill_group.addButton(btn)
            self._exp_pills_layout.addWidget(btn)

        self._exp_stack.setCurrentIndex(1)
        self._btn_exp_back.setVisible(True)
        self._lbl_exp_page_title.setText(f"Research & Experiments — {cat['name']}")
        self._lbl_exp_page_subtitle.setText(cat["description"])
        self._select_experiment(target_exp)

    def _select_experiment(self, exp_id: str) -> None:
        cat = self._get_category_by_id(self._active_exp_cat_id)
        exp = self._get_experiment_by_id(cat, exp_id)
        self._active_exp_id = exp["id"]
        self._exp_sort_col = None
        self._exp_sort_order = Qt.AscendingOrder

        for btn in self._exp_pill_group.buttons():
            btn.setChecked(btn.property("expId") == exp["id"])

        plot_path = (PROJECT_ROOT / "benchmarks" / "plots" / exp["plot_file"]).resolve()
        csv_rel = exp["csv_file"]

        self._lbl_exp_title.setText(exp["title"])
        self._lbl_exp_tech_meta.setText(f"Plot: {exp['plot_file']}  ·  Data: {Path(csv_rel).name}")
        self._lbl_exp_question.setText(f"Research Question: {exp['question']}")
        self._lbl_exp_explanation.setText(exp["explanation"])
        self._lbl_exp_findings.setText("Key Findings:  • " + "   • ".join(exp["findings"]))
        self._lbl_exp_dataset.setText(f"Configuration & Dataset: {exp['dataset_info']}")

        self._figure_viewer.load_figure(str(plot_path))
        self._populate_exp_csv_filters(exp)
        self._refresh_exp_csv_table(exp)

    def _read_csv_cached(self, rel_path: str) -> Tuple[List[str], List[List[str]]]:
        csv_path = (PROJECT_ROOT / rel_path).resolve()
        key = str(csv_path)
        if key in self._exp_csv_cache:
            return self._exp_csv_cache[key]
        if not csv_path.is_file():
            return ([], [])
        with open(key, "r", newline="") as f:
            rows = [r for r in csv.reader(f) if r]
        if not rows:
            return ([], [])
        result = (rows[0], rows[1:])
        self._exp_csv_cache[key] = result
        return result

    def _populate_exp_csv_filters(self, exp: Dict) -> None:
        headers, data_rows = self._read_csv_cached(exp["csv_file"])
        self._exp_syncing_filters = True
        try:
            filter_cols = exp.get("filter_cols", [])
            defaults = exp.get("default_filter", {})

            for slot_idx in range(2):
                lbl = self._exp_filter_labels[slot_idx]
                cmb = self._exp_filter_widgets[slot_idx]
                cmb.clear()
                if slot_idx < len(filter_cols) and filter_cols[slot_idx] in headers:
                    col_name = filter_cols[slot_idx]
                    col_i = headers.index(col_name)
                    lbl.setText(f"{col_name}:")
                    cmb.setProperty("filterCol", col_name)
                    cmb.addItem("All", "")
                    distinct = sorted({r[col_i] for r in data_rows if col_i < len(r)}, key=lambda s: (0, float(s)) if self._is_float(s) else (1, s))
                    for v in distinct:
                        cmb.addItem(v, v)
                    if col_name in defaults:
                        d_idx = cmb.findData(defaults[col_name])
                        if d_idx >= 0:
                            cmb.setCurrentIndex(d_idx)
                    lbl.setVisible(True)
                    cmb.setVisible(True)
                else:
                    cmb.setProperty("filterCol", "")
                    lbl.setVisible(False)
                    cmb.setVisible(False)

            # Populate metric selector from numeric columns in this CSV
            self._combo_exp_metric.clear()
            exclude_meta = {"image_id", "tile_id", "x", "y", "width", "height", "is_partial", "seed", "Width", "Height"}
            numeric_cols: List[str] = []
            if data_rows:
                sample = data_rows[0]
                for c_idx, h in enumerate(headers):
                    if h not in exclude_meta and c_idx < len(sample) and self._is_float(sample[c_idx]):
                        numeric_cols.append(h)
            for mc in numeric_cols:
                self._combo_exp_metric.addItem(mc, mc)
            pref = exp.get("default_metric", "")
            if pref:
                m_idx = self._combo_exp_metric.findData(pref)
                if m_idx >= 0:
                    self._combo_exp_metric.setCurrentIndex(m_idx)
        finally:
            self._exp_syncing_filters = False

    @staticmethod
    def _is_float(val: str) -> bool:
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    def _on_exp_filter_changed(self) -> None:
        if self._exp_syncing_filters:
            return
        cat = self._get_category_by_id(self._active_exp_cat_id)
        exp = self._get_experiment_by_id(cat, self._active_exp_id)
        self._refresh_exp_csv_table(exp)

    def _reset_exp_filters(self) -> None:
        self._exp_syncing_filters = True
        try:
            for cmb in self._exp_filter_widgets:
                if cmb.count() > 0:
                    cmb.setCurrentIndex(0)
        finally:
            self._exp_syncing_filters = False
        self._on_exp_filter_changed()

    def _on_exp_table_sort_requested(self, col_idx: int) -> None:
        if self._exp_sort_col == col_idx:
            self._exp_sort_order = (
                Qt.DescendingOrder if self._exp_sort_order == Qt.AscendingOrder else Qt.AscendingOrder
            )
        else:
            self._exp_sort_col = col_idx
            self._exp_sort_order = Qt.AscendingOrder
        cat = self._get_category_by_id(self._active_exp_cat_id)
        exp = self._get_experiment_by_id(cat, self._active_exp_id)
        self._refresh_exp_csv_table(exp)

    def _refresh_exp_csv_table(self, exp: Dict) -> None:
        headers, data_rows = self._read_csv_cached(exp["csv_file"])
        if not headers:
            self._exp_csv_table.setRowCount(0)
            self._exp_csv_table.setColumnCount(0)
            self._lbl_exp_csv_badge.setText("0 rows")
            self._lbl_exp_stats_bar.setText("Dataset file unavailable")
            return

        # Apply active filters
        active_filters: List[Tuple[int, str]] = []
        for cmb in self._exp_filter_widgets:
            col_name = cmb.property("filterCol")
            target_val = cmb.currentData()
            if col_name and target_val:
                if col_name in headers:
                    active_filters.append((headers.index(col_name), str(target_val)))

        filtered = [
            row for row in data_rows
            if all(ci < len(row) and row[ci] == val for ci, val in active_filters)
        ]

        # Highlight metric column and compute real summary stats
        metric_col = self._combo_exp_metric.currentData() or ""
        metric_idx = headers.index(metric_col) if metric_col in headers else -1
        if metric_idx >= 0 and filtered:
            vals = [float(r[metric_idx]) for r in filtered if metric_idx < len(r) and self._is_float(r[metric_idx])]
            if vals:
                self._lbl_exp_stats_bar.setText(
                    f"{metric_col}  |  Count: {len(vals)}  |  Min: {min(vals):.6g}  |  Mean: {sum(vals)/len(vals):.6g}  |  Max: {max(vals):.6g}"
                )
            else:
                self._lbl_exp_stats_bar.setText(f"{metric_col}  |  Count: 0")
        else:
            self._lbl_exp_stats_bar.setText(f"Filtered records: {len(filtered)} rows")

        # Global numeric-aware Python sort over filtered rows before display slicing
        if self._exp_sort_col is not None and 0 <= self._exp_sort_col < len(headers):
            sc = self._exp_sort_col
            rev = self._exp_sort_order == Qt.DescendingOrder

            def _sort_key(row: List[str]) -> Tuple[int, float, str]:
                v = row[sc] if sc < len(row) else ""
                if self._is_float(v):
                    return (0, float(v), "")
                return (1, 0.0, v.lower())

            filtered = sorted(filtered, key=_sort_key, reverse=rev)

        max_display = 400
        shown = filtered[:max_display]
        suffix = f" (showing {max_display})" if len(filtered) > max_display else ""
        self._lbl_exp_csv_badge.setText(f"{len(filtered)} rows{suffix}")

        self._exp_csv_table.setColumnCount(len(headers))
        self._exp_csv_table.setHorizontalHeaderLabels(headers)
        self._exp_csv_table.setRowCount(len(shown))

        exp_header = self._exp_csv_table.horizontalHeader()
        if self._exp_sort_col is not None and 0 <= self._exp_sort_col < len(headers):
            exp_header.setSortIndicator(self._exp_sort_col, self._exp_sort_order)
        else:
            exp_header.setSortIndicator(-1, Qt.AscendingOrder)

        for r_idx, row in enumerate(shown):
            for c_idx, col_name in enumerate(headers):
                val = row[c_idx] if c_idx < len(row) else ""
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)

                # Highlight selected metric and preserve FP32 (blue) / FP16 (green) precision semantics
                if c_idx == metric_idx:
                    item.setForeground(QColor(SLATE_ACCENT_LIGHT))
                    item.setFont(studio_font(self._exp_csv_table, point_size=9, bold=True))
                elif "fp32" in col_name.lower():
                    item.setForeground(QColor(ACCENT_BLUE))
                elif "fp16" in col_name.lower():
                    item.setForeground(QColor(ACCENT_GREEN))
                elif col_name == "trial_type":
                    item.setForeground(QColor(ACCENT_GREEN if val == "guided" else ACCENT_ORANGE))

                self._exp_csv_table.setItem(r_idx, c_idx, item)

    def _on_browse_exp_output(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select Experiment Output Directory", self._edit_exp_output.text()
        )
        if d:
            self._edit_exp_output.setText(d)

    def _on_start_experiment_runner(self) -> None:
        if not self._exp_bin:
            self._lbl_exp_runner_status.setText("Error: mirage_experiments binary not found")
            return
        ds_dir = self._edit_exp_dataset.text().strip()
        out_dir = self._edit_exp_output.text().strip()
        seeds = self._spin_exp_seeds.value()

        self._txt_exp_log.clear()
        self._txt_exp_log.appendPlainText(
            f"$ {self._exp_bin} --dataset-dir \"{ds_dir}\" --output-dir \"{out_dir}\" --seeds {seeds}\n"
        )
        self._btn_exp_run_exec.setEnabled(False)
        self._btn_exp_cancel_exec.setEnabled(True)
        self._lbl_exp_runner_status.setText("Running C++ experiment suite...")
        self._lbl_exp_runner_status.setStyleSheet(f"color: {ACCENT_ORANGE};")

        self._exp_process = QtCore.QProcess(self)
        self._exp_process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self._exp_process.readyReadStandardOutput.connect(self._on_exp_runner_output)
        self._exp_process.finished.connect(self._on_exp_runner_finished)
        self._exp_process.start(
            str(self._exp_bin),
            ["--dataset-dir", ds_dir, "--output-dir", out_dir, "--seeds", str(seeds)],
        )

    def _on_cancel_experiment_runner(self) -> None:
        if self._exp_process and self._exp_process.state() != QtCore.QProcess.NotRunning:
            self._exp_process.kill()
            self._txt_exp_log.appendPlainText("\n[Cancelled by user]\n")

    def _on_exp_runner_output(self) -> None:
        if self._exp_process:
            chunk = bytes(self._exp_process.readAllStandardOutput()).decode("utf-8", errors="replace")
            if chunk:
                self._txt_exp_log.insertPlainText(chunk)

    def _on_exp_runner_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        self._on_exp_runner_output()
        if exit_code == 0 and exit_status == QtCore.QProcess.NormalExit:
            if self._chk_exp_replot.isChecked():
                self._lbl_exp_runner_status.setText("Generating publication plots (analyze_experiments.py)...")
                replot_proc = QtCore.QProcess(self)
                replot_proc.setWorkingDirectory(str(PROJECT_ROOT))
                replot_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
                replot_proc.start("python3", [str(PROJECT_ROOT / "scripts" / "analyze_experiments.py")])
                if replot_proc.waitForFinished(60000):
                    out = bytes(replot_proc.readAllStandardOutput()).decode("utf-8", errors="replace")
                    if out:
                        self._txt_exp_log.insertPlainText("\n" + out)
            self._exp_csv_cache.clear()
            self._figure_viewer._cache.clear()
            self._lbl_exp_runner_status.setText("Completed successfully")
            self._lbl_exp_runner_status.setStyleSheet(f"color: {ACCENT_GREEN};")
        else:
            self._lbl_exp_runner_status.setText(f"Failed (exit code {exit_code})")
            self._lbl_exp_runner_status.setStyleSheet(f"color: {ACCENT_RED};")

        self._btn_exp_run_exec.setEnabled(True)
        self._btn_exp_cancel_exec.setEnabled(False)

    def _load_system_info(self) -> None:
        if not self._mirage_bin:
            self._lbl_cpu.setText("Binary: Not found")
            return

        proc = QtCore.QProcess(self)
        proc.start(str(self._mirage_bin), ["--info", "--json"])
        if proc.waitForFinished(3000):
            out = bytes(proc.readAllStandardOutput()).decode("utf-8")
            try:
                data = json.loads(out)
                self._lbl_cpu.setText(f"CPU: {data.get('cpu_architecture', 'x86_64')}")
                has_avx2 = data.get("avx2", False)
                has_fma = data.get("fma", False)
                has_f16c = data.get("f16c", False)

                self._lbl_avx2.setText("✓" if has_avx2 else "✗")
                self._lbl_avx2.setStyleSheet(f"color: {ACCENT_GREEN if has_avx2 else ACCENT_RED}; font-weight: bold;")

                self._lbl_fma.setText("✓" if has_fma else "✗")
                self._lbl_fma.setStyleSheet(f"color: {ACCENT_GREEN if has_fma else ACCENT_RED}; font-weight: bold;")

                self._lbl_f16c.setText("✓" if has_f16c else "✗")
                self._lbl_f16c.setStyleSheet(f"color: {ACCENT_GREEN if has_f16c else ACCENT_RED}; font-weight: bold;")
            except Exception as e:
                self._lbl_cpu.setText("CPU: x86_64")

    def _load_benchmark_csv(self) -> None:
        csv_path = PROJECT_ROOT / "benchmarks" / "results" / "benchmark_results.csv"
        if not csv_path.exists():
            return
        try:
            with open(csv_path, "r") as f:
                lines = [line.strip().split(",") for line in f if line.strip()]
            if not lines:
                return
            headers = lines[0]
            rows = lines[1:]
            self._bench_table.setRowCount(len(rows))
            for r_idx, row in enumerate(rows):
                for c_idx, val in enumerate(row[:6]):
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignCenter)
                    self._bench_table.setItem(r_idx, c_idx, item)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Asynchronous Execution via QProcess
    # -----------------------------------------------------------------------
    def _on_run_enhancement(self) -> None:
        if not self._mirage_bin:
            QtWidgets.QMessageBox.critical(self, "Error", "Mirage binary not found. Please build the project with CMake.")
            return

        if not self._active_image_path or not os.path.exists(self._active_image_path):
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a valid input image.")
            return

        self._btn_run.setEnabled(False)
        self._btn_run.setText("Processing...")
        self._status_icon.setText("● RUNNING")
        self._status_icon.setStyleSheet(f"color: {ACCENT_ORANGE}; font-weight: bold;")
        self._status_text.setText("Executing adaptive enhancement pipeline via C++ Mirage CLI...")

        self._run_counter += 1
        run_tag = f"run_{self._run_counter:04d}"
        input_path = self._active_image_path
        out_path = os.path.join(self._temp_dir, f"{run_tag}_enhanced.png")
        heatmap_path = os.path.join(self._temp_dir, f"{run_tag}_heatmap.png")
        precision_path = os.path.join(self._temp_dir, f"{run_tag}_precision_map.png")

        mode_text = self._combo_mode.currentText()
        mode_arg = "adaptive"
        if "FP32" in mode_text:
            mode_arg = "fp32"
        elif "FP16" in mode_text:
            mode_arg = "fp16"
        elif "Mixed" in mode_text:
            mode_arg = "mixed"

        ts_str = self._combo_tilesize.currentText().split()[0]
        tau_val = self._slider_tau.value() / 1000.0

        bright_val = self._slider_bright.value() / 100.0
        contrast_val = self._slider_contrast.value() / 100.0
        sat_val = self._slider_sat.value() / 100.0
        gamma_val = self._slider_gamma.value() / 100.0

        args = [
            "--enhance", input_path,
            "--output", out_path,
            "--precision", mode_arg,
            "--tile-size", ts_str,
            "--sensitivity-threshold", f"{tau_val:.4f}",
            "--brightness", f"{bright_val:.2f}",
            "--contrast", f"{contrast_val:.2f}",
            "--saturation", f"{sat_val:.2f}",
            "--gamma", f"{gamma_val:.2f}",
            "--json",
        ]
        if mode_arg == "adaptive":
            args.extend(["--save-heatmap", heatmap_path, "--save-precision-map", precision_path])
        else:
            heatmap_path = ""
            precision_path = ""

        self._last_process_error: str = ""
        self._process = QtCore.QProcess(self)
        self._process.errorOccurred.connect(self._on_process_error)
        self._process.finished.connect(
            lambda exit_code, exit_status: self._on_process_finished(
                exit_code, input_path, out_path, heatmap_path, precision_path, mode_arg, tau_val, exit_status
            )
        )
        self._process.start(str(self._mirage_bin), args)

    def _on_process_error(self, error: QtCore.QProcess.ProcessError) -> None:
        # Handle startup / spawn failures that never emit QProcess.finished
        if error == QtCore.QProcess.FailedToStart:
            self._btn_run.setEnabled(True)
            self._btn_run.setText("▶   Run Enhancement")
            diag = (
                f"Failed to launch Mirage binary ({self._mirage_bin}): "
                f"{self._process.errorString() if self._process else 'Unknown spawn error'}. "
                "Verify executable permissions and runtime library paths."
            )
            self._last_process_error = diag
            self._status_icon.setText("● ERROR")
            self._status_icon.setStyleSheet(f"color: {ACCENT_RED}; font-weight: bold;")
            self._status_text.setText(diag)
            sys.stderr.write(f"[Mirage Studio Error] {diag}\n")
            if QtWidgets.QApplication.platformName() != "offscreen":
                QtWidgets.QMessageBox.critical(self, "Enhancement Launch Failure", diag)

    def _on_process_finished(
        self,
        exit_code: int,
        input_path: str,
        out_path: str,
        heatmap_path: str,
        precision_path: str,
        mode_arg: str,
        tau_val: float,
        exit_status: QtCore.QProcess.ExitStatus = QtCore.QProcess.NormalExit,
    ) -> None:
        self._btn_run.setEnabled(True)
        self._btn_run.setText("▶   Run Enhancement")

        if exit_code != 0 or exit_status != QtCore.QProcess.NormalExit:
            err = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace").strip()
            out = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            detail = err or out or f"No diagnostic output (exit status: {exit_status})"

            hint = ""
            if exit_code == 127 or "cannot open shared object file" in detail or "error while loading shared libraries" in detail:
                hint = (
                    "\n\nDiagnosis: Dynamic linker failed to resolve required OpenCV shared libraries. "
                    f"Run `ldd \"{self._mirage_bin}\"` to inspect unresolved dependencies or rebuild with CMake."
                )

            summary = f"Enhancement failed (exit {exit_code}): {detail}"
            full_diag = f"Binary: {self._mirage_bin}\nExit Code: {exit_code}\n\nDetails:\n{detail}{hint}"
            self._last_process_error = full_diag

            self._status_icon.setText("● ERROR")
            self._status_icon.setStyleSheet(f"color: {ACCENT_RED}; font-weight: bold;")
            self._status_text.setText(summary)
            sys.stderr.write(f"[Mirage Studio Error] {full_diag}\n")
            if QtWidgets.QApplication.platformName() != "offscreen":
                QtWidgets.QMessageBox.critical(self, "Enhancement Execution Error", full_diag)
            return

        stdout_raw = bytes(self._process.readAllStandardOutput()).decode("utf-8")
        try:
            telemetry = json.loads(stdout_raw)
            if telemetry.get("status") != "success":
                raise ValueError(f"Unexpected backend status: {telemetry.get('status')}")
            self._latest_telemetry = telemetry
            self._apply_telemetry(telemetry, input_path, out_path, heatmap_path, precision_path, mode_arg, tau_val)
        except (json.JSONDecodeError, ValueError) as e:
            diag = f"Enhancement JSON telemetry error (previous Results preserved): {e}"
            self._last_process_error = diag
            self._status_icon.setText("● ERROR")
            self._status_icon.setStyleSheet(f"color: {ACCENT_RED}; font-weight: bold;")
            self._status_text.setText(diag)
            sys.stderr.write(f"[Mirage Studio Error] {diag}\n")

    def _apply_telemetry(
        self,
        data: Dict,
        input_path: str,
        out_path: str,
        heatmap_path: str,
        precision_path: str,
        mode_arg: str,
        tau_val: float,
    ) -> None:
        img_info = data.get("image", {})
        resolved_input = img_info.get("input_path") or input_path
        resolved_output = img_info.get("output_path") or out_path
        resolved_hm = img_info.get("heatmap_path") or heatmap_path
        resolved_pm = img_info.get("precision_map_path") or precision_path

        # Commit latest valid results only after backend succeeded cleanly
        self._last_result_telemetry = data
        self._last_result_input_path = resolved_input
        self._last_result_output_path = resolved_output
        self._last_result_heatmap_path = resolved_hm
        self._last_result_precision_path = resolved_pm

        # Update Enhance Viewer & Overlays
        mode_display = f"Adaptive (τ = {tau_val:.2f})" if mode_arg == "adaptive" else mode_arg.upper()
        self._viewer.set_images(resolved_input, resolved_output, mode_display)

        self._overlay_viewer.set_overlay_images(resolved_hm, resolved_pm)

        # Update Tile Allocation Donut
        alloc = data.get("tile_allocation", {})
        total_tiles = alloc.get("total_tiles", 0)
        fp32_cnt = alloc.get("fp32_count", 0)
        fp16_cnt = alloc.get("fp16_count", 0)
        fp32_pct = alloc.get("fp32_percentage", 0.0)
        fp16_pct = alloc.get("fp16_percentage", 0.0)
        self._donut_chart.set_allocation(total_tiles, fp32_cnt, fp16_cnt, fp32_pct, fp16_pct)

        # Populate Analyze Table with Tile Assignments
        assignments = alloc.get("assignments", [])
        if assignments:
            self._analyze_table.setRowCount(len(assignments))
            for r_idx, a in enumerate(assignments):
                self._analyze_table.setItem(r_idx, 0, QTableWidgetItem(str(a.get("id"))))
                self._analyze_table.setItem(r_idx, 1, QTableWidgetItem(str(a.get("x"))))
                self._analyze_table.setItem(r_idx, 2, QTableWidgetItem(str(a.get("y"))))
                self._analyze_table.setItem(r_idx, 3, QTableWidgetItem(str(a.get("width"))))
                self._analyze_table.setItem(r_idx, 4, QTableWidgetItem(str(a.get("height"))))
                self._analyze_table.setItem(r_idx, 5, QTableWidgetItem(str(a.get("is_partial"))))
                prec_item = QTableWidgetItem(str(a.get("precision")))
                if a.get("precision") == "FP32":
                    prec_item.setForeground(QColor(ACCENT_BLUE))
                else:
                    prec_item.setForeground(QColor(ACCENT_GREEN))
                self._analyze_table.setItem(r_idx, 6, prec_item)
                self._analyze_table.setItem(r_idx, 7, QTableWidgetItem(f"{a.get('sensitivity_score', 0.0):.6f}"))
        else:
            self._analyze_table.setRowCount(0)

        # Update Enhance Sidebar Key Results
        qm = data.get("quality_metrics_vs_fp32_reference")
        if qm:
            psnr_val = qm.get("psnr_db", 0.0)
            ssim_val = qm.get("ssim", 0.0)
            ciede_val = qm.get("mean_ciede2000", 0.0)
            l2_val = qm.get("relative_l2_error", 0.0) * 100.0

            self._lbl_psnr.setText(f"{psnr_val:.2f} dB" if psnr_val < 99 else "> 100 dB")
            self._lbl_ssim.setText(f"{ssim_val:.4f}")
            self._lbl_ciede.setText(f"{ciede_val:.4f}")
            self._lbl_l2.setText(f"{l2_val:.4f} %")
        else:
            self._lbl_psnr.setText("∞ (Reference)")
            self._lbl_ssim.setText("1.0000")
            self._lbl_ciede.setText("0.0000")
            self._lbl_l2.setText("0.0000 %")

        # Update Hardware Runtime Bar Chart
        rt = data.get("runtime", {})
        measured_ms = rt.get("measured_pipeline_ms", 0.0)

        if mode_arg == "adaptive":
            self._measured_adaptive_ms = measured_ms
            label = f"Adaptive (τ = {tau_val:.2f})"
        elif mode_arg == "fp32":
            self._measured_fp32_ms = measured_ms
            label = f"Adaptive (τ = {tau_val:.2f})"
        elif mode_arg == "fp16":
            self._measured_fp16_ms = measured_ms
            label = f"Adaptive (τ = {tau_val:.2f})"
        else:
            label = f"Adaptive (τ = {tau_val:.2f})"

        self._perf_chart.update_runtimes(self._measured_fp32_ms, self._measured_fp16_ms, self._measured_adaptive_ms, label)

        # Update Results Tab Dashboard automatically from actual backend JSON
        self._update_results_dashboard(
            data, resolved_input, resolved_output, resolved_hm, resolved_pm, mode_display
        )

        # Update Status Bar
        self._status_icon.setText("● READY")
        self._status_icon.setStyleSheet(f"color: {ACCENT_GREEN}; font-weight: bold;")
        psnr_str = f"PSNR: {qm.get('psnr_db', 0.0):.2f} dB" if qm else "Reference FP32"
        ssim_str = f"SSIM: {qm.get('ssim', 1.0):.4f}" if qm else "SSIM: 1.0000"
        self._status_text.setText(f"Last Run: {mode_display} | {measured_ms:.2f} ms | {psnr_str} | {ssim_str}")

    def _update_results_dashboard(
        self,
        data: Dict,
        input_path: str,
        output_path: str,
        heatmap_path: str,
        precision_path: str,
        mode_display: str,
    ) -> None:
        img = data.get("image", {})
        cfg = data.get("config", {})
        rt = data.get("runtime", {})
        alloc = data.get("tile_allocation", {})
        qm = data.get("quality_metrics_vs_fp32_reference")

        width = img.get("width", 0)
        height = img.get("height", 0)
        channels = img.get("channels", 3)
        fmt = img.get("format", "CV_32FC3 normalized BGR (from CV_8UC3)")
        in_name = Path(input_path).name if input_path else "--"
        out_name = Path(output_path).name if output_path else "--"

        # Header meta & comparison viewer (Section A)
        self._res_header_meta.setText(
            f"Input: {in_name}  →  Output: {out_name}  |  {width} × {height} × {channels}  |  {mode_display}"
        )
        self._res_viewer.set_images(input_path, output_path, mode_display, format_label=f"{channels}ch ({fmt.split()[0]})")
        self._res_viewer.fit_to_screen()
        self._res_lbl_img_dims.setText(f"Dimensions: {width} × {height} × {channels}")
        self._res_lbl_img_format.setText(f"Format: {fmt}  |  Input: {in_name}")

        # Execution Summary (Section B)
        pmode = str(cfg.get("precision_mode", "--"))
        policy = str(cfg.get("policy", "--"))
        self._res_lbl_mode.setText(f"{pmode} ({policy})")

        req_be = str(cfg.get("backend", "--"))
        res_be = str(cfg.get("resolved_backend", req_be))
        self._res_lbl_backend.setText(f"{res_be} ({req_be})")

        tw = cfg.get("tile_width", 0)
        th = cfg.get("tile_height", 0)
        self._res_lbl_tilesize.setText(f"{tw} × {th}" if pmode.lower() == "adaptive" else f"{tw} × {th} (grid)")

        thresh = cfg.get("threshold")
        self._res_lbl_threshold.setText(f"{thresh:.4f}" if isinstance(thresh, (int, float)) else "N/A (Uniform/Mixed)")

        total_t = alloc.get("total_tiles", 0)
        fp32_c = alloc.get("fp32_count", 0)
        fp16_c = alloc.get("fp16_count", 0)
        fp32_p = alloc.get("fp32_percentage", 0.0)
        fp16_p = alloc.get("fp16_percentage", 0.0)
        self._res_lbl_total_tiles.setText(str(total_t))

        if pmode.lower() == "mixed":
            self._res_lbl_fp32_tiles.setText("Operation-level (Mixed)")
            self._res_lbl_fp16_tiles.setText("Operation-level (Mixed)")
        else:
            self._res_lbl_fp32_tiles.setText(f"{fp32_c} ({fp32_p:.2f} %)")
            self._res_lbl_fp16_tiles.setText(f"{fp16_c} ({fp16_p:.2f} %)")
        self._res_lbl_fp32_tiles.setStyleSheet(f"color: {ACCENT_BLUE}; font-weight: bold;")
        self._res_lbl_fp16_tiles.setStyleSheet(f"color: {ACCENT_GREEN}; font-weight: bold;")

        exec_ms = rt.get("measured_pipeline_ms", 0.0)
        self._res_lbl_runtime.setText(f"{exec_ms:.2f} ms")
        self._res_lbl_runtime.setStyleSheet(f"color: {ACCENT_PURPLE_LIGHT}; font-weight: bold;")

        # Quality Metrics vs FP32 Reference (Section C)
        if qm:
            ref_desc = qm.get("reference_description", "FP32 Scalar Enhancement Output (Ground Truth)")
            self._res_lbl_ref_note.setText(f"Reference: {ref_desc} — never input image")
            psnr_v = qm.get("psnr_db", 0.0)
            self._res_lbl_psnr.setText(f"{psnr_v:.2f} dB" if psnr_v < 100.0 else f"{psnr_v:.2f} dB (Exact)")
            self._res_lbl_ssim.setText(f"{qm.get('ssim', 0.0):.4f}")
            self._res_lbl_ciede.setText(f"{qm.get('mean_ciede2000', 0.0):.4f}")
            self._res_lbl_mae.setText(f"{qm.get('mean_absolute_error', 0.0):.6f}")
            self._res_lbl_rmse.setText(f"{qm.get('root_mean_square_error', 0.0):.6f}")
            self._res_lbl_max_err.setText(f"{qm.get('max_absolute_error', 0.0):.6f}")
            self._res_lbl_l2.setText(f"{qm.get('relative_l2_error', 0.0) * 100.0:.4f} %")
        else:
            self._res_lbl_ref_note.setText("Execution IS the FP32 Reference Baseline (0 error vs self)")
            self._res_lbl_psnr.setText("∞ (FP32 Reference)")
            self._res_lbl_ssim.setText("1.0000")
            self._res_lbl_ciede.setText("0.0000")
            self._res_lbl_mae.setText("0.000000")
            self._res_lbl_rmse.setText("0.000000")
            self._res_lbl_max_err.setText("0.000000")
            self._res_lbl_l2.setText("0.0000 %")

        # Precision Visualizations (Section D)
        self._res_donut.set_allocation(total_t, fp32_c, fp16_c, fp32_p, fp16_p)
        empty_overlay_msg = "Not generated\n(Adaptive mode only)" if pmode.lower() != "adaptive" else "Overlay file missing"
        self._res_heatmap_preview.set_image_path(heatmap_path, empty_text=empty_overlay_msg)
        self._res_precision_preview.set_image_path(precision_path, empty_text=empty_overlay_msg)

        # Reveal populated dashboard (Page 1)
        self._results_stack.setCurrentIndex(1)


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Mirage")
    app.setApplicationVersion("0.9.0-alpha")
    app.setStyleSheet(DARK_STUDIO_QSS)

    setup_typography(app)

    initial_image = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    window = MirageStudioMainWindow(initial_image=initial_image)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
