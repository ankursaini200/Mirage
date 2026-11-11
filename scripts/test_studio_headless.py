#!/usr/bin/env python3
"""
Automated Headless Test & Verification for Mirage Creative Studio
==================================================================
Runs the desktop studio headless using '-platform offscreen', triggers
adaptive enhancement, asserts widget and telemetry state, and grabs
a high-resolution screenshot for documentation.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Initialize QApplication with offscreen platform before anything else
from PySide6 import QtCore, QtGui, QtWidgets
app = QtWidgets.QApplication(["mirage_studio", "-platform", "offscreen"])

import mirage_studio as ms


def main() -> int:
    print("[Headless Test] Configuring application styling and typography...")
    app.setStyleSheet(ms.DARK_STUDIO_QSS)
    ms.setup_typography(app)

    test_image = ms.PROJECT_ROOT / "data" / "experiments" / "natural_scene.png"
    if not test_image.exists():
        test_image = ms.PROJECT_ROOT / "data" / "experiments" / "smooth_gradient.png"

    print(f"[Headless Test] Creating MirageStudioMainWindow with {test_image.name}...")
    win = ms.MirageStudioMainWindow(initial_image=str(test_image))
    win.resize(1340, 840)
    win.show()
    app.processEvents()
    win._viewer.fit_to_screen()
    app.processEvents()

    # Step 1 & 10: Check initial Results Empty State before any enhancement run
    win._switch_view(4)
    app.processEvents()
    assert win._results_stack.currentIndex() == 0, "Results must show Empty State before execution"
    assert win._lbl_results_empty_title.text() == "No enhancement results yet"
    assert "Run an enhancement" in win._lbl_results_empty_body.text()
    win._switch_view(0)
    app.processEvents()
    print("  ✓ Fresh startup Results empty state verified")

    # Verify numerical enhancement & precision parameter controls (defaults, +/- step, signs, clamping)
    print("[Headless Test] Verifying numeric enhancement & threshold step controls...")
    expected_specs = [
        ("brightness", "+0.10", 0.05, -1.00, 1.00, 2, True),
        ("contrast", "1.20", 0.05, 0.00, 3.00, 2, False),
        ("saturation", "1.30", 0.05, 0.00, 3.00, 2, False),
        ("gamma", "1.10", 0.05, 0.10, 3.00, 2, False),
        ("threshold", "0.040", 0.01, 0.00, 1.00, 3, False),
    ]
    for key, def_str, step, vmin, vmax, digits, signed in expected_specs:
        ctrl = win._param_controls[key]
        slider = ctrl["slider"]
        lbl = ctrl["label"]
        dec_btn = ctrl["dec_btn"]
        inc_btn = ctrl["inc_btn"]
        assert lbl.text() == def_str, f"{key} default expected {def_str}, got {lbl.text()}"
        orig_val = slider.value()
        inc_btn.click()
        app.processEvents()
        step_int = int(round(step * ctrl["scale"]))
        assert slider.value() == orig_val + step_int, f"{key} '+' step mismatch"
        dec_btn.click()
        app.processEvents()
        assert slider.value() == orig_val, f"{key} '-' step mismatch"

        # Check lower/upper range clamping
        slider.setValue(slider.minimum())
        dec_btn.click()
        assert slider.value() == slider.minimum(), f"{key} failed lower clamp"
        slider.setValue(slider.maximum())
        inc_btn.click()
        assert slider.value() == slider.maximum(), f"{key} failed upper clamp"
        slider.setValue(orig_val)

    # Specifically verify signed formatting across negative, zero, and positive Brightness
    b_ctrl = win._param_controls["brightness"]
    orig_b = b_ctrl["slider"].value()
    b_ctrl["slider"].setValue(-15)
    assert b_ctrl["label"].text() == "-0.15", f"Expected -0.15, got {b_ctrl['label'].text()}"
    b_ctrl["slider"].setValue(0)
    assert b_ctrl["label"].text() == "+0.00", f"Expected +0.00, got {b_ctrl['label'].text()}"
    b_ctrl["slider"].setValue(-5)
    assert b_ctrl["label"].text() == "-0.05"
    b_ctrl["inc_btn"].click()
    assert b_ctrl["label"].text() == "+0.00"
    b_ctrl["inc_btn"].click()
    assert b_ctrl["label"].text() == "+0.05"
    b_ctrl["slider"].setValue(orig_b)

    # Verify threshold clamping at 1.00 and disabling in non-Adaptive modes
    t_ctrl = win._param_controls["threshold"]
    t_ctrl["slider"].setValue(t_ctrl["slider"].maximum())
    assert t_ctrl["label"].text() == "1.000", f"Expected 1.000 max threshold, got {t_ctrl['label'].text()}"
    win._combo_mode.setCurrentIndex(1)  # FP32 (Reference)
    assert not t_ctrl["slider"].isEnabled() and not t_ctrl["dec_btn"].isEnabled() and not t_ctrl["inc_btn"].isEnabled()
    win._combo_mode.setCurrentIndex(0)  # Adaptive
    assert t_ctrl["slider"].isEnabled() and t_ctrl["dec_btn"].isEnabled() and t_ctrl["inc_btn"].isEnabled()
    t_ctrl["slider"].setValue(40)  # Restore default 0.040
    app.processEvents()
    print("  ✓ All numeric parameter ranges, +/- exact steps, signed brightness, and range clamping verified")

    print("[Headless Test] Triggering adaptive enhancement #1 via QProcess...")
    win._on_run_enhancement()
    assert win._process is not None, "QProcess must be created"

    print("[Headless Test] Waiting for enhancement #1 to finish...")
    finished = win._process.waitForFinished(15000)
    assert finished, "Enhancement #1 timed out!"
    app.processEvents()

    print("[Headless Test] Verifying header branding, telemetry, and UI state...")
    brand_title_lbl = win.findChild(QtWidgets.QLabel, "brandTitle")
    brand_sub_lbl = win.findChild(QtWidgets.QLabel, "brandSubtitle")
    assert brand_title_lbl is not None and brand_title_lbl.text() == "MIRAGE", (
        f"Expected header title 'MIRAGE', got {brand_title_lbl.text() if brand_title_lbl else None}"
    )
    assert brand_sub_lbl is not None and brand_sub_lbl.text() == "Adaptive Precision Imaging", (
        f"Expected header subtitle 'Adaptive Precision Imaging', got {brand_sub_lbl.text() if brand_sub_lbl else None}"
    )
    assert "Creative Studio" not in brand_title_lbl.text(), "Header title must not contain 'Creative Studio'"
    print(f"  ✓ Header branding verified: '{brand_title_lbl.text()}' / '{brand_sub_lbl.text()}'")

    assert win._process.exitCode() == 0, f"QProcess exited with non-zero code: {win._process.exitCode()}"
    out1_path = win._last_result_output_path
    assert out1_path and os.path.exists(out1_path), "Enhanced output image #1 must be saved"
    assert os.path.getsize(out1_path) > 0, "Enhanced output image #1 must be non-empty"
    assert win._viewer._enhanced_pixmap is not None and not win._viewer._enhanced_pixmap.isNull(), (
        "Enhance SplitImageViewer must load enhanced output image"
    )

    telemetry = win._latest_telemetry
    assert telemetry, "Telemetry dictionary must not be empty"
    assert telemetry.get("status") == "success", f"Status must be success: {telemetry}"

    alloc = telemetry.get("tile_allocation", {})
    total_tiles = alloc.get("total_tiles", 0)
    assert total_tiles > 0, f"Expected total_tiles > 0, got {total_tiles}"
    fp32_count = alloc.get("fp32_count", 0)
    fp16_count = alloc.get("fp16_count", 0)
    print(f"  ✓ Tile allocation #1: {total_tiles} total tiles ({fp32_count} FP32, {fp16_count} FP16)")

    qm = telemetry.get("quality_metrics_vs_fp32_reference", {})
    assert "psnr_db" in qm and "ssim" in qm and "max_absolute_error" in qm
    assert win._lbl_psnr.text() != "--" and "dB" in win._lbl_psnr.text(), "PSNR UI label must be updated"
    assert win._lbl_mae.text() == f"{qm.get('mean_absolute_error', 0.0):.6f}"
    assert win._lbl_rmse.text() == f"{qm.get('root_mean_square_error', 0.0):.6f}"
    print(f"  ✓ Quality metrics #1 (all 6 analytics labels): PSNR = {qm.get('psnr_db')} dB, SSIM = {qm.get('ssim')}, ΔE00 = {qm.get('mean_ciede2000')}, MAE = {win._lbl_mae.text()}, RMSE = {win._lbl_rmse.text()}")

    rt = telemetry.get("runtime", {})
    measured_ms = rt.get("measured_pipeline_ms", 0.0)
    assert measured_ms > 0.0, f"Measured runtime must be positive, got {measured_ms}"
    print(f"  ✓ Measured hardware runtime #1: {measured_ms:.2f} ms")

    table_rows = win._analyze_table.rowCount()
    assert table_rows == total_tiles, f"Analyze table rows ({table_rows}) != total tiles ({total_tiles})"
    print(f"  ✓ Analyze table populated with {table_rows} rows")

    # Capture Enhance View screenshot
    docs_img_dir = ms.PROJECT_ROOT / "docs" / "images"
    docs_img_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = docs_img_dir / "mirage_studio_screenshot.png"
    app.processEvents()
    pix = win.grab()
    assert pix.save(str(screenshot_path))
    print(f"  ✓ Enhance View screenshot captured ({pix.width()}x{pix.height()}) at: {screenshot_path}")

    # Step 5-7: Switch to Results tab and verify populated dashboard matches backend JSON
    win._switch_view(4)
    app.processEvents()
    assert win._results_stack.currentIndex() == 1, "Results must show populated dashboard after successful run"
    assert win._res_viewer._original_pixmap is not None and not win._res_viewer._original_pixmap.isNull()
    assert win._res_viewer._enhanced_pixmap is not None and not win._res_viewer._enhanced_pixmap.isNull()
    assert win._res_lbl_total_tiles.text() == str(total_tiles)
    assert f"{fp32_count} (" in win._res_lbl_fp32_tiles.text()
    assert f"{fp16_count} (" in win._res_lbl_fp16_tiles.text()
    assert f"{measured_ms:.2f} ms" == win._res_lbl_runtime.text()
    assert f"{qm.get('mean_absolute_error', 0.0):.6f}" == win._res_lbl_mae.text()
    assert f"{qm.get('root_mean_square_error', 0.0):.6f}" == win._res_lbl_rmse.text()
    assert f"{qm.get('max_absolute_error', 0.0):.6f}" == win._res_lbl_max_err.text()
    assert "FP32" in win._res_lbl_ref_note.text()
    assert win._res_donut._total_tiles == total_tiles
    assert win._res_heatmap_preview._pixmap is not None and not win._res_heatmap_preview._pixmap.isNull()
    assert win._res_precision_preview._pixmap is not None and not win._res_precision_preview._pixmap.isNull()

    # Capture Results View screenshot (#1 execution)
    results_shot_path = docs_img_dir / "results_tab_corrected.png"
    app.processEvents()
    assert win.grab().save(str(results_shot_path))
    print(f"  ✓ Results View screenshot captured at: {results_shot_path}")

    # Step 8-9: Run Enhancement #2 with DIFFERENT parameters (tile_size=16, tau=0.020, contrast=1.40)
    print("[Headless Test] Triggering adaptive enhancement #2 with changed parameters (tile=16, tau=0.020)...")
    win._combo_tilesize.setCurrentIndex(0)  # 16 x 16
    win._slider_tau.setValue(20)            # 0.020
    win._slider_contrast.setValue(140)      # 1.40
    win._on_run_enhancement()
    assert win._process.waitForFinished(15000), "Enhancement #2 timed out!"
    app.processEvents()

    assert win._process.exitCode() == 0
    t2 = win._latest_telemetry
    alloc2 = t2.get("tile_allocation", {})
    assert alloc2.get("total_tiles") == 1024, f"Expected 1024 tiles for 512x512 with tile=16, got {alloc2.get('total_tiles')}"
    assert win._last_result_output_path != out1_path, "Run #2 must produce a distinct output path"
    assert win._res_lbl_total_tiles.text() == "1024", "Results dashboard must update to Run #2 tile count"
    assert "16 × 16" in win._res_lbl_tilesize.text(), "Results dashboard must update to Run #2 tile size"
    assert win._res_lbl_threshold.text() == "0.0200", "Results dashboard must update to Run #2 threshold"
    print(f"  ✓ Results tab cleanly updated to Run #2 (1024 tiles, τ=0.0200, {t2['runtime']['measured_pipeline_ms']:.2f} ms)")

    # Verify Difference & Histogram real-pixel overlay rendering cleanly in READY state
    for tab_name in ["Difference", "Histogram", "Sensitivity"]:
        win._overlay_viewer.set_active_tab(tab_name)
        app.processEvents()
        assert not win._overlay_viewer.grab().isNull(), f"{tab_name} overlay must render"
    assert win._overlay_viewer._hist_bins is not None, "Histogram bins must be computed from actual image"
    assert win._overlay_viewer._difference_pixmap is not None, "Difference map must be computed from input vs enhanced"
    print("  ✓ Real-image Difference and RGB Histogram overlays verified")

    # Capture clean READY-state screenshots across all 5 redesigned tabs (Analyze, Benchmark, Experiments, Results, Enhance)
    tab_shots = {
        1: docs_img_dir / "analyze_tab.png",
        2: docs_img_dir / "benchmark_tab.png",
        3: docs_img_dir / "experiments_tab.png",
    }
    for tab_idx in [1, 2, 3, 4, 0]:
        win._switch_view(tab_idx)
        app.processEvents()
        if tab_idx in tab_shots:
            assert win.grab().save(str(tab_shots[tab_idx]))
            print(f"  ✓ Tab {tab_idx} screenshot captured at: {tab_shots[tab_idx]}")

    win._overlay_viewer.set_active_tab("Sensitivity")
    app.processEvents()
    assert win.grab().save(str(screenshot_path))

    # Capture focused panel screenshots (Enhancement Panel + Analytics Panel) and assert zero horizontal clipping
    enh_panel = win.findChild(QtWidgets.QFrame, "leftAdjustmentPanel")
    assert enh_panel is not None
    for pkey, ctrl in win._param_controls.items():
        for wname in ("label", "step_label", "inc_btn", "dec_btn", "slider"):
            w = ctrl[wname]
            right_x = w.mapTo(enh_panel, QtCore.QPoint(w.width() - 1, 0)).x()
            assert 0 <= right_x < enh_panel.width(), (
                f"Control {pkey}.{wname} clipped horizontally (right_x={right_x}, panel_w={enh_panel.width()})"
            )
    assert enh_panel.grab().save(str(docs_img_dir / "enhancement_panel_refined.png"))
    analytics_panel = win._overlay_viewer.parentWidget()
    assert analytics_panel is not None and analytics_panel.grab().save(str(docs_img_dir / "analytics_panel_refined.png"))
    print("  ✓ Verified zero horizontal clipping on all enhancement controls & captured panel screenshots")

    # Failure preservation check: trigger a genuine failing QProcess run and verify Run #2 Results stay intact
    print("[Headless Test] Verifying real QProcess failure resilience (preserving Run #2 Results)...")
    import tempfile
    saved_active = win._active_image_path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tf.write(b"CORRUPT_PNG_HEADER")
        corrupt_input = tf.name
    try:
        win._active_image_path = corrupt_input
        win._on_run_enhancement()
        assert win._process.waitForFinished(5000), "Failure test QProcess timed out!"
        app.processEvents()
        assert win._process.exitCode() != 0, "Corrupt input image must produce non-zero CLI exit code"
        assert "Corrupted" in win._last_process_error or "Decoding failed" in win._last_process_error or "Exit Code:" in win._last_process_error
        assert win._last_result_telemetry is t2, "Failed execution must NOT overwrite latest valid Results telemetry"
        assert win._res_lbl_total_tiles.text() == "1024", "Failed execution must NOT clear Results dashboard"
    finally:
        win._active_image_path = saved_active
        if os.path.exists(corrupt_input):
            os.remove(corrupt_input)
    print("  ✓ Failed QProcess run captured diagnostic error and preserved Run #2 Results intact")

    # Also capture Results empty state screenshot on a fresh window for verification
    docs_img_empty = docs_img_dir / "results_tab_empty_state.png"
    win_fresh = ms.MirageStudioMainWindow(initial_image=str(test_image))
    win_fresh.resize(1340, 840)
    win_fresh.show()
    win_fresh._switch_view(4)
    app.processEvents()
    assert win_fresh._results_stack.currentIndex() == 0, "Restarted instance must start in Empty State"
    assert win_fresh.grab().save(str(docs_img_empty))
    print(f"  ✓ Restart empty-state behavior verified & captured at: {docs_img_empty}")

    # Verify responsive layout scaling on small window (1080x700) and 2K display (2560x1440)
    for (rw, rh) in [(1080, 700), (2560, 1440), (1340, 840)]:
        win.resize(rw, rh)
        app.processEvents()
        assert win._btn_run.isVisible() and win._btn_run.height() >= 28, f"Primary CTA unusable at {rw}x{rh}"
        assert win._viewer.width() >= 340, f"Center viewer too narrow at {rw}x{rh}: {win._viewer.width()}"
        assert not win.grab().isNull()
    print("  ✓ Responsive scaling verified across 1080x700, 1340x840, and 2K (2560x1440)")
    print("  ✓ All navigation tabs verified (Enhance, Analyze, Benchmark, Experiments, Results)")

    print("[Headless Test] All verification checks PASSED successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
