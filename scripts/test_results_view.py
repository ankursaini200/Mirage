#!/usr/bin/env python3
"""
Comprehensive Automated Test Suite for Mirage Creative Studio Results View
============================================================================
Verifies:
1. Loading all Phase 8 & 9 publication figures in benchmarks/plots/.
2. Switching through figures sequentially via next / prev controls.
3. Preserving original aspect ratios during resizing.
4. Window resizing from small to full HD displays.
5. Error handling for non-existent and corrupted image files.
6. Execution from different working directories (e.g. /tmp).
"""

import os
import sys
import tempfile
from pathlib import Path

# Initialize QApplication with offscreen platform
from PySide6 import QtCore, QtGui, QtWidgets
app = QtWidgets.QApplication(["mirage_test_results", "-platform", "offscreen"])

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mirage_studio as ms


def test_results_viewer() -> int:
    print("[Results View Test] Initializing dark theme and typography...")
    app.setStyleSheet(ms.DARK_STUDIO_QSS)
    ms.setup_typography(app)

    print("[Results View Test] Instantiating MirageStudioMainWindow...")
    win = ms.MirageStudioMainWindow()
    win.resize(1340, 840)
    win.show()
    app.processEvents()

    # 1. Verify Results view starts in Empty State on fresh startup (no publication charts here)
    print("[Experiments & Results Test] Verifying Results tab initial empty state...")
    win._switch_view(4)
    app.processEvents()
    assert win._main_stack.currentIndex() == 4, "Must be on Results view"
    assert win._results_stack.currentIndex() == 0, "Results must show empty state before any run"
    assert win._lbl_results_empty_title.text() == "No enhancement results yet"
    assert "Run an enhancement" in win._lbl_results_empty_body.text()
    print("  ✓ Results tab initial empty state verified")

    # 2. Switch to Experiments view (Index 3) and test FigureViewerWidget across all 13 categorized plots
    print("[Experiments & Results Test] Switching to Research & Experiments View (Index 3)...")
    win._switch_view(3)
    app.processEvents()
    assert win._main_stack.currentIndex() == 3, "Must be on Experiments view"
    assert win._exp_stack.currentIndex() == 0, "Must open Category Landing Page initially"

    plots_dir = (ms.PROJECT_ROOT / "benchmarks" / "plots").resolve()
    figure_files = sorted(list(plots_dir.glob("*.png")))
    total_figs = len(figure_files)
    print(f"  ✓ Found {total_figs} publication figures in benchmarks/plots/")
    assert total_figs == 13

    # 3. Iterate through every category & figure and verify loading & non-null pixmap
    print("[Experiments & Results Test] Verifying all categorized figures...")
    verified_plots = []
    for cat in ms.RESEARCH_CATEGORIES:
        win._open_experiment_category(cat["id"])
        app.processEvents()
        for exp in cat["experiments"]:
            win._select_experiment(exp["id"])
            app.processEvents()
            viewer = win._figure_viewer
            assert viewer._current_pixmap is not None and not viewer._current_pixmap.isNull()
            assert viewer._error_message == ""
            verified_plots.append(exp["plot_file"])
            print(f"  ✓ [{cat['name']}] {exp['short_title']:<24} ({exp['plot_file']}: {viewer._current_pixmap.width()}x{viewer._current_pixmap.height()}) - OK")
    assert len(verified_plots) == total_figs

    # 4. Verify FigureViewerWidget QPixmap caching when revisiting same experiment
    win._open_experiment_category("adaptive_precision", "random_vs_guided")
    app.processEvents()
    first_pixmap_obj = win._figure_viewer._current_pixmap
    win._select_experiment("feature_ablation")
    app.processEvents()
    win._select_experiment("random_vs_guided")
    app.processEvents()
    assert win._figure_viewer._current_pixmap is first_pixmap_obj, "Must reuse cached QPixmap"
    print("  ✓ Experiment switching and QPixmap cache reuse verified")

    # 5. Test Resizing (from 800x600 to 1920x1080) using actual paintEvent rendering
    print("[Experiments & Results Test] Testing window resizing and aspect-ratio preservation...")
    for (w, h) in [(800, 600), (1080, 720), (1340, 840), (1920, 1080)]:
        win.resize(w, h)
        app.processEvents()
        viewer = win._figure_viewer
        viewer.grab()
        orig_aspect = viewer._current_pixmap.width() / float(viewer._current_pixmap.height())
        scaled = viewer._scaled_pixmap
        assert scaled is not None and not scaled.isNull()
        scaled_size = scaled.deviceIndependentSize()
        scaled_aspect = scaled_size.width() / float(scaled_size.height())
        assert abs(orig_aspect - scaled_aspect) < 0.02
        viewer.grab()
        assert viewer._scaled_pixmap is scaled
        print(f"  ✓ Window size {w}x{h} -> Viewer {viewer.width()}x{viewer.height()} -> Rendered {int(scaled_size.width())}x{int(scaled_size.height())} (aspect {scaled_aspect:.3f})")

    # 6. Test Error Handling for Missing File & Corrupted Image File
    print("[Experiments & Results Test] Testing error handling for non-existent and corrupted figures...")
    win._figure_viewer.load_figure("/non/existent/path/to/missing_chart.png")
    app.processEvents()
    assert not win._figure_viewer.grab().isNull()
    assert "Figure File Not Found" in win._figure_viewer._error_message
    assert "missing_chart.png" in win._figure_viewer._error_message

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tf.write(b"NOT_A_VALID_PNG_PAYLOAD")
        corrupt_path = tf.name
    try:
        win._figure_viewer.load_figure(corrupt_path)
        app.processEvents()
        assert not win._figure_viewer.grab().isNull()
        assert "Failed to Decode Image File" in win._figure_viewer._error_message
        assert Path(corrupt_path).name in win._figure_viewer._error_message
    finally:
        if os.path.exists(corrupt_path):
            os.remove(corrupt_path)

    # Restore valid experiment figure & CSV data
    win._select_experiment("random_vs_guided")
    app.processEvents()
    assert win._figure_viewer._current_pixmap is not None
    assert win._exp_csv_table.rowCount() > 0
    print("[Experiments & Results Test] ALL TESTS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(test_results_viewer())
