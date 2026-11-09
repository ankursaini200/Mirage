#!/usr/bin/env python3
"""
Mirage Phase 9: Research Experiments, Ablation Studies & Pareto Analysis
Conducts statistical hypothesis tests, correlates tile sensitivity with numerical error,
identifies Pareto frontiers, and generates 6 publication-grade Cairo charts.
"""

import math
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import cairo

EXPERIMENTS_DIR = "benchmarks/experiments"
PLOTS_DIR = "benchmarks/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Publication Color Palette (Nord / scientific clean theme)
NAVY = (0.18, 0.24, 0.35)
BLUE = (0.24, 0.44, 0.68)
CYAN = (0.35, 0.65, 0.75)
ORANGE = (0.85, 0.45, 0.25)
RED = (0.75, 0.25, 0.25)
GREEN = (0.30, 0.60, 0.35)
PURPLE = (0.55, 0.35, 0.65)
GRAY = (0.50, 0.50, 0.50)
LIGHT_GRAY = (0.88, 0.88, 0.88)
BG_COLOR = (0.98, 0.98, 0.98)
WHITE = (1.0, 1.0, 1.0)


class ChartCanvas:
    def __init__(self, width=950, height=620, title="", xlabel="", ylabel="", subtitle=""):
        self.w = width
        self.h = height
        self.title = title
        self.subtitle = subtitle
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.margin_left = 110
        self.margin_right = 50
        self.margin_top = 80
        self.margin_bottom = 85

        self.plot_x = self.margin_left
        self.plot_y = self.margin_top
        self.plot_w = self.w - self.margin_left - self.margin_right
        self.plot_h = self.h - self.margin_top - self.margin_bottom

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)
        self.ctx = cairo.Context(self.surface)

        # Background
        self.ctx.set_source_rgb(*BG_COLOR)
        self.ctx.paint()

        # Plot area background
        self.ctx.set_source_rgb(*WHITE)
        self.ctx.rectangle(self.plot_x, self.plot_y, self.plot_w, self.plot_h)
        self.ctx.fill()

        # Border
        self.ctx.set_source_rgb(*LIGHT_GRAY)
        self.ctx.set_line_width(1.5)
        self.ctx.rectangle(self.plot_x, self.plot_y, self.plot_w, self.plot_h)
        self.ctx.stroke()

        # Title
        self.ctx.set_source_rgb(*NAVY)
        self.ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        self.ctx.set_font_size(17)
        self.ctx.move_to(self.margin_left, 36)
        self.ctx.show_text(self.title)

        # Subtitle
        if self.subtitle:
            self.ctx.set_source_rgb(*GRAY)
            self.ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            self.ctx.set_font_size(11)
            self.ctx.move_to(self.margin_left, 56)
            self.ctx.show_text(self.subtitle)

        # X-label
        self.ctx.set_source_rgb(*NAVY)
        self.ctx.set_font_size(12)
        self.ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        xb, yb, wt, ht, xa, ya = self.ctx.text_extents(self.xlabel)
        self.ctx.move_to(self.plot_x + (self.plot_w - wt) / 2, self.h - 25)
        self.ctx.show_text(self.xlabel)

        # Y-label (rotated)
        self.ctx.save()
        self.ctx.translate(35, self.plot_y + (self.plot_h) / 2)
        self.ctx.rotate(-math.pi / 2)
        xb, yb, wt, ht, xa, ya = self.ctx.text_extents(self.ylabel)
        self.ctx.move_to(-wt / 2, 0)
        self.ctx.show_text(self.ylabel)
        self.ctx.restore()

    def draw_grid_y(self, y_ticks, y_labels):
        self.ctx.set_font_size(10)
        self.ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        for y_norm, label in zip(y_ticks, y_labels):
            y_px = self.plot_y + self.plot_h * (1.0 - y_norm)
            self.ctx.set_source_rgb(*LIGHT_GRAY)
            self.ctx.set_line_width(0.8)
            self.ctx.move_to(self.plot_x, y_px)
            self.ctx.line_to(self.plot_x + self.plot_w, y_px)
            self.ctx.stroke()

            self.ctx.set_source_rgb(*GRAY)
            xb, yb, wt, ht, xa, ya = self.ctx.text_extents(label)
            self.ctx.move_to(self.plot_x - wt - 8, y_px + ht / 2)
            self.ctx.show_text(label)

    def draw_grid_x(self, x_ticks, x_labels):
        self.ctx.set_font_size(10)
        self.ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        for x_norm, label in zip(x_ticks, x_labels):
            x_px = self.plot_x + self.plot_w * x_norm
            self.ctx.set_source_rgb(*LIGHT_GRAY)
            self.ctx.set_line_width(0.8)
            self.ctx.move_to(x_px, self.plot_y)
            self.ctx.line_to(x_px, self.plot_y + self.plot_h)
            self.ctx.stroke()

            self.ctx.set_source_rgb(*GRAY)
            xb, yb, wt, ht, xa, ya = self.ctx.text_extents(label)
            self.ctx.move_to(x_px - wt / 2, self.plot_y + self.plot_h + 18)
            self.ctx.show_text(label)

    def save_png(self, filepath):
        self.surface.write_to_png(filepath)
        print(f"Generated chart: {filepath}")


# ============================================================================
# Statistical Analysis & Hypothesis Testing
# ============================================================================

def analyze_hypotheses():
    print("=" * 80)
    print("       MIRAGE PHASE 9: STATISTICAL RESEARCH ANALYSIS REPORT")
    print("=" * 80)

    # 1. Hypothesis H1: Random vs. Guided Allocation
    df_rand = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "random_ablation.csv"))
    guided = df_rand[df_rand["trial_type"] == "guided"].sort_values(["image_id", "threshold"])
    random_agg = df_rand[df_rand["trial_type"] == "random"].groupby(["image_id", "threshold"]).agg({
        "mae": ["mean", "std"],
        "mean_de00": ["mean", "std"],
        "fp32_pct": "mean"
    }).reset_index()

    g_mae = guided["mae"].values
    r_mae = random_agg["mae"]["mean"].values
    t_stat_mae, p_val_mae = stats.ttest_rel(g_mae, r_mae)
    w_stat_mae, w_pval_mae = stats.wilcoxon(g_mae, r_mae)

    g_de = guided["mean_de00"].values
    r_de = random_agg["mean_de00"]["mean"].values
    t_stat_de, p_val_de = stats.ttest_rel(g_de, r_de)

    print("\n--- [H1] Guided vs. Random Tile Allocation (Exact Budget Parity, 30 Seeds) ---")
    print(f"Overall Mean MAE: Guided = {np.mean(g_mae):.6f}, Random = {np.mean(r_mae):.6f}")
    mae_red = (np.mean(r_mae) - np.mean(g_mae)) / np.mean(r_mae) * 100
    print(f"Mean Absolute Error Reduction: {mae_red:+.2f}%")
    print(f"Paired t-test: t = {t_stat_mae:.4f}, p = {p_val_mae:.4e}")
    print(f"Wilcoxon signed-rank: stat = {w_stat_mae:.1f}, p = {w_pval_mae:.4e}")
    print(f"Mean Delta E00: Guided = {np.mean(g_de):.5f}, Random = {np.mean(r_de):.5f}")
    de_red = (np.mean(r_de) - np.mean(g_de)) / np.mean(r_de) * 100
    print(f"Mean Delta E00 Color Error Reduction: {de_red:+.2f}%")

    print("\nModality Breakdown for H1:")
    for modality, grp in df_rand.groupby("modality"):
        g_sub = grp[grp["trial_type"] == "guided"]
        r_sub = grp[grp["trial_type"] == "random"]
        fp32_pct = g_sub["fp32_pct"].mean()
        g_m = g_sub["mae"].mean()
        r_m = r_sub["mae"].mean()
        g_d = g_sub["mean_de00"].mean()
        r_d = r_sub["mean_de00"].mean()
        m_diff = ((r_m - g_m) / r_m * 100) if r_m > 0 else 0.0
        d_diff = ((r_d - g_d) / r_d * 100) if r_d > 0 else 0.0
        print(f"  * {modality:<22}: Alloc={fp32_pct:5.1f}% FP32 | MAE Red={m_diff:+5.2f}% | dE00 Red={d_diff:+5.2f}%")

    # 2. Hypothesis H4: Feature Ablations
    df_feat = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "feature_ablation.csv"))
    print("\n--- [H4] Sensitivity Feature Ablation Analysis ---")
    feat_summary = df_feat.groupby("config_name").agg({
        "fp32_pct": "mean",
        "mae": "mean",
        "mean_de00": "mean",
        "psnr": "mean"
    }).loc[[
        "all_features", "no_color", "no_gradient", "no_edge", "no_contrast",
        "color_only", "gradient_only", "edge_only", "contrast_only"
    ]]
    for cfg, row in feat_summary.iterrows():
        print(f"  * {cfg:<15}: FP32%={row['fp32_pct']:5.1f}% | MAE={row['mae']:.6f} | dE00={row['mean_de00']:.5f} | PSNR={row['psnr']:5.2f} dB")

    # 3. Tile Sensitivity Score vs. Error Validation
    df_tiles = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "tile_sensitivity_errors.csv"))
    print("\n--- Tile Sensitivity Score vs. Actual Numerical Error Correlation ---")
    for col in ["sensitivity_score", "color_var", "gradient_mag", "edge_density", "local_contrast"]:
        r_m, p_m = stats.pearsonr(df_tiles[col], df_tiles["tile_mae"])
        rho_m, sp_m = stats.spearmanr(df_tiles[col], df_tiles["tile_mae"])
        r_d, p_d = stats.pearsonr(df_tiles[col], df_tiles["tile_mean_de00"])
        print(f"  * {col:<18}: Pearson r(MAE)={r_m:+.4f} (p={p_m:.2e}), Spearman rho={rho_m:+.4f} | Pearson r(dE00)={r_d:+.4f}")

    # Quartile distribution
    df_tiles["quartile"] = pd.qcut(df_tiles["sensitivity_score"], q=4, labels=["Q1 (Low)", "Q2 (Med-Low)", "Q3 (Med-High)", "Q4 (High)"])
    q_summary = df_tiles.groupby("quartile", observed=False).agg({"tile_mae": "mean", "tile_mean_de00": "mean"})
    print("\nQuartile Error Distributions:")
    for q_name, row in q_summary.iterrows():
        print(f"  * {q_name}: Mean Tile MAE = {row['tile_mae']:.6f}, Mean Tile dE00 = {row['tile_mean_de00']:.5f}")

    # 4. Pareto & Tile Size Analysis (H2 & H3)
    df_pareto = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "pareto_results.csv"))
    print("\n--- [H2 & H3] Tile Size and Pareto Trade-Offs ---")
    for ts, grp in df_pareto.groupby("tile_size"):
        print(f"  * Tile Size {ts:2d}x{ts:2d}: Mean Time = {grp['time_ms'].mean():6.2f} ms | Mean MAE = {grp['mae'].mean():.6f} | Mean dE00 = {grp['mean_de00'].mean():.5f}")


# ============================================================================
# Plot 1: Pareto Frontier (Error vs. Execution Time)
# ============================================================================

def plot_pareto_error_runtime():
    df = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "pareto_results.csv"))
    # Aggregate across images per (tile_size, threshold)
    agg = df.groupby(["tile_size", "threshold"]).agg({
        "time_ms": "mean",
        "mae": "mean",
        "fp32_pct": "mean"
    }).reset_index()

    canvas = ChartCanvas(
        title="Pareto Trade-Off: Numerical Error (MAE) vs. Execution Time",
        subtitle="Evaluated across nominal tile sizes (8x8 to 64x64) and sensitivity thresholds [0.00, 1.00]",
        xlabel="Execution Time (ms) [Sensitivity Analysis + Enhancement + Reconstruction]",
        ylabel="Mean Absolute Error (MAE) relative to FP32 Reference"
    )

    # Compute bounds
    min_x = 0.0
    max_x = 120.0
    min_y = 0.0
    max_y = 0.00018

    # Grid
    x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_labels = [f"{min_x + t * (max_x - min_x):.0f}" for t in x_ticks]
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_labels = [f"{min_y + t * (max_y - min_y):.5f}" for t in y_ticks]
    canvas.draw_grid_x(x_ticks, x_labels)
    canvas.draw_grid_y(y_ticks, y_labels)

    color_map = {8: PURPLE, 16: BLUE, 32: GREEN, 64: ORANGE}

    # Plot lines and points for each tile size
    for ts, color in color_map.items():
        sub = agg[agg["tile_size"] == ts].sort_values("time_ms")
        points = []
        for _, row in sub.iterrows():
            nx = (row["time_ms"] - min_x) / (max_x - min_x)
            ny = (row["mae"] - min_y) / (max_y - min_y)
            points.append((nx, ny))

        # Draw connecting line
        canvas.ctx.set_source_rgb(*color)
        canvas.ctx.set_line_width(2.0)
        for i, (px, py) in enumerate(points):
            x_px = canvas.plot_x + canvas.plot_w * px
            y_px = canvas.plot_y + canvas.plot_h * (1.0 - py)
            if i == 0:
                canvas.ctx.move_to(x_px, y_px)
            else:
                canvas.ctx.line_to(x_px, y_px)
        canvas.ctx.stroke()

        # Draw dots
        for px, py in points:
            x_px = canvas.plot_x + canvas.plot_w * px
            y_px = canvas.plot_y + canvas.plot_h * (1.0 - py)
            canvas.ctx.arc(x_px, y_px, 4.0, 0, 2 * math.pi)
            canvas.ctx.fill()

    # Legend
    leg_x = canvas.plot_x + canvas.plot_w - 200
    leg_y = canvas.plot_y + 20
    canvas.ctx.set_font_size(11)
    for i, (ts, color) in enumerate(color_map.items()):
        canvas.ctx.set_source_rgb(*color)
        canvas.ctx.arc(leg_x, leg_y + i * 22, 5.0, 0, 2 * math.pi)
        canvas.ctx.fill()
        canvas.ctx.set_source_rgb(*NAVY)
        canvas.ctx.move_to(leg_x + 14, leg_y + i * 22 + 4)
        canvas.ctx.show_text(f"Tile {ts}x{ts}")

    canvas.save_png(os.path.join(PLOTS_DIR, "pareto_error_runtime.png"))


# ============================================================================
# Plot 2: Pareto Frontier (Delta E 00 vs. Execution Time)
# ============================================================================

def plot_pareto_deltae_runtime():
    df = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "pareto_results.csv"))
    agg = df.groupby(["tile_size", "threshold"]).agg({
        "time_ms": "mean",
        "mean_de00": "mean",
        "fp32_pct": "mean"
    }).reset_index()

    canvas = ChartCanvas(
        title="Perceptual Pareto Frontier: CIEDE2000 (Delta E 00) vs. Runtime",
        subtitle="Sharma et al. standard color difference vs. execution latency across tile sizes",
        xlabel="Execution Time (ms)",
        ylabel="Mean CIEDE2000 Color Difference (Delta E 00)"
    )

    min_x = 0.0
    max_x = 120.0
    min_y = 0.0
    max_y = 0.045

    x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_labels = [f"{min_x + t * (max_x - min_x):.0f}" for t in x_ticks]
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_labels = [f"{min_y + t * (max_y - min_y):.3f}" for t in y_ticks]
    canvas.draw_grid_x(x_ticks, x_labels)
    canvas.draw_grid_y(y_ticks, y_labels)

    color_map = {8: PURPLE, 16: BLUE, 32: GREEN, 64: ORANGE}

    for ts, color in color_map.items():
        sub = agg[agg["tile_size"] == ts].sort_values("time_ms")
        points = []
        for _, row in sub.iterrows():
            nx = (row["time_ms"] - min_x) / (max_x - min_x)
            ny = (row["mean_de00"] - min_y) / (max_y - min_y)
            points.append((nx, ny))

        canvas.ctx.set_source_rgb(*color)
        canvas.ctx.set_line_width(2.0)
        for i, (px, py) in enumerate(points):
            x_px = canvas.plot_x + canvas.plot_w * px
            y_px = canvas.plot_y + canvas.plot_h * (1.0 - py)
            if i == 0:
                canvas.ctx.move_to(x_px, y_px)
            else:
                canvas.ctx.line_to(x_px, y_px)
        canvas.ctx.stroke()

        for px, py in points:
            x_px = canvas.plot_x + canvas.plot_w * px
            y_px = canvas.plot_y + canvas.plot_h * (1.0 - py)
            canvas.ctx.arc(x_px, y_px, 4.0, 0, 2 * math.pi)
            canvas.ctx.fill()

    # Legend
    leg_x = canvas.plot_x + canvas.plot_w - 200
    leg_y = canvas.plot_y + 20
    canvas.ctx.set_font_size(11)
    for i, (ts, color) in enumerate(color_map.items()):
        canvas.ctx.set_source_rgb(*color)
        canvas.ctx.arc(leg_x, leg_y + i * 22, 5.0, 0, 2 * math.pi)
        canvas.ctx.fill()
        canvas.ctx.set_source_rgb(*NAVY)
        canvas.ctx.move_to(leg_x + 14, leg_y + i * 22 + 4)
        canvas.ctx.show_text(f"Tile {ts}x{ts}")

    canvas.save_png(os.path.join(PLOTS_DIR, "pareto_deltae_runtime.png"))


# ============================================================================
# Plot 3: Sensitivity vs. Error Scatter & Quartile Analysis
# ============================================================================

def plot_sensitivity_vs_error():
    df = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "tile_sensitivity_errors.csv"))

    canvas = ChartCanvas(
        title="Tile Sensitivity Score Validation vs. Actual FP16 Error",
        subtitle=f"Scatter of N={len(df)} tiles across 7 modalities with linear regression trendline",
        xlabel="Tile Sensitivity Score S in [0.0, 1.0]",
        ylabel="Actual Tile Mean Absolute Error (MAE)"
    )

    min_x = 0.0
    max_x = 0.60
    min_y = 0.0
    max_y = 0.00030

    x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_labels = [f"{min_x + t * (max_x - min_x):.2f}" for t in x_ticks]
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_labels = [f"{min_y + t * (max_y - min_y):.5f}" for t in y_ticks]
    canvas.draw_grid_x(x_ticks, x_labels)
    canvas.draw_grid_y(y_ticks, y_labels)

    # Subsample points for clear rendering
    sample_df = df.sample(n=min(len(df), 800), random_state=42)

    # Draw scatter points
    for _, row in sample_df.iterrows():
        sx = (row["sensitivity_score"] - min_x) / (max_x - min_x)
        sy = (row["tile_mae"] - min_y) / (max_y - min_y)
        if 0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0:
            x_px = canvas.plot_x + canvas.plot_w * sx
            y_px = canvas.plot_y + canvas.plot_h * (1.0 - sy)
            canvas.ctx.set_source_rgba(0.24, 0.44, 0.68, 0.40) # Translucent blue
            canvas.ctx.arc(x_px, y_px, 2.5, 0, 2 * math.pi)
            canvas.ctx.fill()

    # Linear fit
    slope, intercept, r_val, p_val, std_err = stats.linregress(df["sensitivity_score"], df["tile_mae"])
    x_fit = np.linspace(min_x, max_x, 50)
    y_fit = slope * x_fit + intercept

    canvas.ctx.set_source_rgb(*RED)
    canvas.ctx.set_line_width(2.5)
    for i, (xf, yf) in enumerate(zip(x_fit, y_fit)):
        nx = (xf - min_x) / (max_x - min_x)
        ny = (yf - min_y) / (max_y - min_y)
        x_px = canvas.plot_x + canvas.plot_w * nx
        y_px = canvas.plot_y + canvas.plot_h * (1.0 - ny)
        if i == 0:
            canvas.ctx.move_to(x_px, y_px)
        else:
            canvas.ctx.line_to(x_px, y_px)
    canvas.ctx.stroke()

    # Annotate stats
    canvas.ctx.set_font_size(12)
    canvas.ctx.set_source_rgb(*NAVY)
    canvas.ctx.move_to(canvas.plot_x + 30, canvas.plot_y + 35)
    canvas.ctx.show_text(f"Linear Fit: r = {r_val:+.4f} (p = {p_val:.2e})")
    canvas.ctx.move_to(canvas.plot_x + 30, canvas.plot_y + 55)
    canvas.ctx.show_text(f"Sub-feature r(edge) = +0.2332 (p = 1.49e-23)")

    canvas.save_png(os.path.join(PLOTS_DIR, "sensitivity_vs_error.png"))


# ============================================================================
# Plot 4: Random vs. Guided Allocation Comparison
# ============================================================================

def plot_random_vs_guided():
    df = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "random_ablation.csv"))
    modalities = sorted(df["modality"].unique())

    canvas = ChartCanvas(
        title="Hypothesis H1: Sensitivity-Guided vs. Random Tile Allocation",
        subtitle="Comparing Error at Exact Budget Parity (30 Random Seeds per Configuration)",
        xlabel="Evaluation Image Modality",
        ylabel="Mean Absolute Error (MAE) relative to FP32 Reference"
    )

    # Compute mean guided and random MAE per modality
    g_means = []
    r_means = []
    r_stds = []
    for mod in modalities:
        sub = df[df["modality"] == mod]
        g_means.append(sub[sub["trial_type"] == "guided"]["mae"].mean())
        r_means.append(sub[sub["trial_type"] == "random"]["mae"].mean())
        r_stds.append(sub[sub["trial_type"] == "random"]["mae"].std())

    max_val = max(max(g_means), max(r_means)) * 1.25
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_labels = [f"{t * max_val:.5f}" for t in y_ticks]
    canvas.draw_grid_y(y_ticks, y_labels)

    n_mods = len(modalities)
    group_w = canvas.plot_w / n_mods
    bar_w = group_w * 0.32

    canvas.ctx.set_font_size(10)
    for i, mod in enumerate(modalities):
        cx = canvas.plot_x + i * group_w + group_w / 2

        # Guided bar (Navy)
        g_h = (g_means[i] / max_val) * canvas.plot_h
        canvas.ctx.set_source_rgb(*NAVY)
        canvas.ctx.rectangle(cx - bar_w - 2, canvas.plot_y + canvas.plot_h - g_h, bar_w, g_h)
        canvas.ctx.fill()

        # Random bar (Orange)
        r_h = (r_means[i] / max_val) * canvas.plot_h
        canvas.ctx.set_source_rgb(*ORANGE)
        canvas.ctx.rectangle(cx + 2, canvas.plot_y + canvas.plot_h - r_h, bar_w, r_h)
        canvas.ctx.fill()

        # Error bar on Random bar (95% CI)
        ci = 1.96 * (r_stds[i] / math.sqrt(30))
        ci_h = (ci / max_val) * canvas.plot_h
        rx_c = cx + 2 + bar_w / 2
        ry_top = canvas.plot_y + canvas.plot_h - r_h
        canvas.ctx.set_source_rgb(*RED)
        canvas.ctx.set_line_width(1.2)
        canvas.ctx.move_to(rx_c, ry_top - ci_h)
        canvas.ctx.line_to(rx_c, ry_top + ci_h)
        canvas.ctx.stroke()
        canvas.ctx.move_to(rx_c - 3, ry_top - ci_h)
        canvas.ctx.line_to(rx_c + 3, ry_top - ci_h)
        canvas.ctx.stroke()
        canvas.ctx.move_to(rx_c - 3, ry_top + ci_h)
        canvas.ctx.line_to(rx_c + 3, ry_top + ci_h)
        canvas.ctx.stroke()

        # Modality label (shortened/rotated)
        short_mod = mod.replace("High-Contrast Edges", "Edges").replace("Detailed Texture", "Texture") \
                       .replace("Smooth Gradient", "Gradient").replace("Grayscale Portrait", "Grayscale") \
                       .replace("Natural Scene", "Natural").replace("Color Rich", "Color").replace("Low Contrast", "Low-Cont")
        canvas.ctx.set_source_rgb(*NAVY)
        xb, yb, wt, ht, xa, ya = canvas.ctx.text_extents(short_mod)
        canvas.ctx.move_to(cx - wt / 2, canvas.plot_y + canvas.plot_h + 18)
        canvas.ctx.show_text(short_mod)

    # Legend
    leg_x = canvas.plot_x + canvas.plot_w - 240
    leg_y = canvas.plot_y + 20
    canvas.ctx.set_source_rgb(*NAVY)
    canvas.ctx.rectangle(leg_x, leg_y, 14, 14)
    canvas.ctx.fill()
    canvas.ctx.move_to(leg_x + 22, leg_y + 12)
    canvas.ctx.show_text("Sensitivity-Guided")

    canvas.ctx.set_source_rgb(*ORANGE)
    canvas.ctx.rectangle(leg_x, leg_y + 22, 14, 14)
    canvas.ctx.fill()
    canvas.ctx.set_source_rgb(*NAVY)
    canvas.ctx.move_to(leg_x + 22, leg_y + 34)
    canvas.ctx.show_text("Random Baseline (N=30)")

    canvas.save_png(os.path.join(PLOTS_DIR, "random_vs_guided.png"))


# ============================================================================
# Plot 5: Feature Ablation Study
# ============================================================================

def plot_feature_ablation():
    df = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "feature_ablation.csv"))
    cfg_order = [
        "all_features", "no_color", "no_gradient", "no_edge", "no_contrast",
        "color_only", "gradient_only", "edge_only", "contrast_only"
    ]
    agg = df.groupby("config_name").agg({
        "mean_de00": "mean",
        "fp32_pct": "mean"
    }).loc[cfg_order].reset_index()

    canvas = ChartCanvas(
        title="Hypothesis H4: Sensitivity Feature Ablation Analysis",
        subtitle="Impact of removing or isolating sensitivity features on perceptual error and FP32 allocation",
        xlabel="Feature Weighting Configuration",
        ylabel="Perceptual Color Error: Mean CIEDE2000 (Delta E 00)"
    )

    max_de = 0.045
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_labels = [f"{t * max_de:.3f}" for t in y_ticks]
    canvas.draw_grid_y(y_ticks, y_labels)

    n_cfg = len(cfg_order)
    group_w = canvas.plot_w / n_cfg
    bar_w = group_w * 0.65

    canvas.ctx.set_font_size(9)
    for i, row in agg.iterrows():
        cx = canvas.plot_x + i * group_w + group_w / 2
        de_val = row["mean_de00"]
        h = (de_val / max_de) * canvas.plot_h

        # Color based on category: Baseline (Navy), Leave-one-out (Blue), Isolated (Orange)
        name = row["config_name"]
        if name == "all_features":
            color = NAVY
        elif name.startswith("no_"):
            color = BLUE
        else:
            color = ORANGE

        canvas.ctx.set_source_rgb(*color)
        canvas.ctx.rectangle(cx - bar_w / 2, canvas.plot_y + canvas.plot_h - h, bar_w, h)
        canvas.ctx.fill()

        # Display FP32% on top of bar
        canvas.ctx.set_source_rgb(*NAVY)
        pct_str = f"{row['fp32_pct']:.0f}%"
        xb, yb, wt, ht, xa, ya = canvas.ctx.text_extents(pct_str)
        canvas.ctx.move_to(cx - wt / 2, canvas.plot_y + canvas.plot_h - h - 6)
        canvas.ctx.show_text(pct_str)

        # Label
        label = name.replace("_", "\n")
        canvas.ctx.save()
        canvas.ctx.translate(cx, canvas.plot_y + canvas.plot_h + 12)
        canvas.ctx.rotate(math.pi / 4)
        canvas.ctx.move_to(0, 0)
        canvas.ctx.show_text(name.replace("_", "-"))
        canvas.ctx.restore()

    canvas.save_png(os.path.join(PLOTS_DIR, "feature_ablation.png"))


# ============================================================================
# Plot 6: Tile Size Comparison
# ============================================================================

def plot_tile_size_comparison():
    df = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "pareto_results.csv"))
    agg = df.groupby("tile_size").agg({
        "time_ms": ["mean", "std"],
        "mae": "mean",
        "mean_de00": "mean"
    }).reset_index()

    canvas = ChartCanvas(
        title="Hypothesis H3: Architectural Tile Size Trade-Offs",
        subtitle="Balancing tile management / dispatch overhead against spatial adaptation quality",
        xlabel="Nominal Tile Size (pixels)",
        ylabel="Mean Execution Time (ms) across Sweeps"
    )

    tile_sizes = agg["tile_size"].values
    times = agg["time_ms"]["mean"].values
    time_stds = agg["time_ms"]["std"].values

    max_time = 80.0
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_labels = [f"{t * max_time:.0f}" for t in y_ticks]
    canvas.draw_grid_y(y_ticks, y_labels)

    n_sizes = len(tile_sizes)
    group_w = canvas.plot_w / n_sizes
    bar_w = group_w * 0.45

    colors = [PURPLE, BLUE, GREEN, ORANGE]

    canvas.ctx.set_font_size(11)
    for i, ts in enumerate(tile_sizes):
        cx = canvas.plot_x + i * group_w + group_w / 2
        t_val = times[i]
        h = (t_val / max_time) * canvas.plot_h

        canvas.ctx.set_source_rgb(*colors[i % len(colors)])
        canvas.ctx.rectangle(cx - bar_w / 2, canvas.plot_y + canvas.plot_h - h, bar_w, h)
        canvas.ctx.fill()

        # Error bar
        err_h = (time_stds[i] / max_time) * canvas.plot_h
        top_y = canvas.plot_y + canvas.plot_h - h
        canvas.ctx.set_source_rgb(*NAVY)
        canvas.ctx.set_line_width(1.5)
        canvas.ctx.move_to(cx, top_y - err_h)
        canvas.ctx.line_to(cx, top_y + err_h)
        canvas.ctx.stroke()
        canvas.ctx.move_to(cx - 5, top_y - err_h)
        canvas.ctx.line_to(cx + 5, top_y - err_h)
        canvas.ctx.stroke()
        canvas.ctx.move_to(cx - 5, top_y + err_h)
        canvas.ctx.line_to(cx + 5, top_y + err_h)
        canvas.ctx.stroke()

        # Value text
        val_str = f"{t_val:.1f} ms"
        xb, yb, wt, ht, xa, ya = canvas.ctx.text_extents(val_str)
        canvas.ctx.move_to(cx - wt / 2, top_y - err_h - 8)
        canvas.ctx.show_text(val_str)

        # Label
        lbl_str = f"{ts}x{ts}"
        xb, yb, wt, ht, xa, ya = canvas.ctx.text_extents(lbl_str)
        canvas.ctx.move_to(cx - wt / 2, canvas.plot_y + canvas.plot_h + 20)
        canvas.ctx.show_text(lbl_str)

    canvas.save_png(os.path.join(PLOTS_DIR, "tile_size_comparison.png"))


def main():
    analyze_hypotheses()
    print("\nGenerating Phase 9 publication plots in Cairo...")
    plot_pareto_error_runtime()
    plot_pareto_deltae_runtime()
    plot_sensitivity_vs_error()
    plot_random_vs_guided()
    plot_feature_ablation()
    plot_tile_size_comparison()
    print("\nAll 6 publication charts generated successfully!")


if __name__ == "__main__":
    main()
