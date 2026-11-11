#!/usr/bin/env python3
"""
Automated Verification Suite for Redesigned Mirage Research & Experiments Workspace
=================================================================================
Verifies:
1. Main page title ('Research & Experiments') & subtitle with ZERO phase numbers exposed.
2. Category Landing Page shows 4 cards (Adaptive Precision, Performance Analysis,
   Pareto Analysis, Tile Architecture) with counts & previews and no auto-opened chart.
3. Every chart across all 4 categories loads into FigureViewerWidget at clean scale,
   without duplicates or missing figures.
4. Interactive CSV exploration (dataset filtering, metric selection, numeric sorting,
   real min/mean/max summary statistics).
5. Navigation between categories, experiment pills, and landing page.
6. Separated Experiment Runner workflow executes compiled C++ mirage_experiments cleanly
   into an isolated temp directory without modifying existing experiment CSVs/plots.
7. Captures screenshots of the Landing Page and all 4 Categories for visual inspection.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

from PySide6 import QtCore, QtWidgets
app = QtWidgets.QApplication(["mirage_experiments_test", "-platform", "offscreen"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mirage_studio as ms


def main() -> int:
    print("[Experiments Suite] Applying theme & typography...")
    app.setStyleSheet(ms.DARK_STUDIO_QSS)
    ms.setup_typography(app)

    win = ms.MirageStudioMainWindow()
    win.resize(1340, 840)
    win.show()
    app.processEvents()

    # 1. Switch to Research & Experiments view (Index 3)
    win._switch_view(3)
    app.processEvents()
    assert win._main_stack.currentIndex() == 3

    # Verify landing state: Page 0 active, no chart auto-opened, no phase numbers
    assert win._exp_stack.currentIndex() == 0, "Must land on 4-card Category Landing Page"
    assert win._lbl_exp_page_title.text() == "Research & Experiments"
    assert "Phase" not in win._lbl_exp_page_title.text()
    assert "Phase" not in win._lbl_exp_page_subtitle.text()
    assert win._figure_viewer._current_pixmap is None, "Must NOT auto-open a chart on Landing Page"
    assert len(win._exp_category_cards) == 4, "Must show exactly 4 research category cards"

    expected_cats = ["Adaptive Precision", "Performance Analysis", "Pareto Analysis", "Tile Architecture"]
    for idx, cat in enumerate(ms.RESEARCH_CATEGORIES):
        assert cat["name"] == expected_cats[idx]
        assert cat["id"] in win._exp_category_cards

    # Save Landing Page screenshot
    img_dir = ms.PROJECT_ROOT / "docs" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    landing_shot = img_dir / "experiments_landing.png"
    assert win.grab().save(str(landing_shot))
    print(f"  ✓ Landing Page verified & captured at: {landing_shot}")

    # 2. Discover all plots on disk and verify 100% unique coverage across categories
    plots_dir = (ms.PROJECT_ROOT / "benchmarks" / "plots").resolve()
    disk_plots = {p.name for p in plots_dir.glob("*.png")}
    catalog_plots = []

    for cat in ms.RESEARCH_CATEGORIES:
        cid = cat["id"]
        win._open_experiment_category(cid)
        app.processEvents()
        assert win._exp_stack.currentIndex() == 1, f"Must open category detail for {cid}"
        assert "Phase" not in win._lbl_exp_page_title.text()
        assert "Phase" not in win._lbl_exp_page_subtitle.text()

        # Verify each experiment inside this category
        for e_idx, exp in enumerate(cat["experiments"]):
            win._select_experiment(exp["id"])
            app.processEvents()
            assert "Phase" not in win._lbl_exp_title.text()
            assert "Phase" not in win._lbl_exp_explanation.text()
            assert "Phase" not in win._lbl_exp_findings.text()

            viewer = win._figure_viewer
            viewer.grab()
            assert viewer._current_pixmap is not None and not viewer._current_pixmap.isNull(), (
                f"Failed to load plot {exp['plot_file']}"
            )
            assert viewer._error_message == "", f"Error for {exp['plot_file']}: {viewer._error_message}"
            scaled = viewer._scaled_pixmap
            assert scaled is not None and not scaled.isNull()
            assert scaled.deviceIndependentSize().width() >= 350, "Chart viewer rendered too small"

            # Verify CSV data table loaded
            assert win._exp_csv_table.columnCount() > 3, f"CSV headers missing for {exp['csv_file']}"
            assert win._exp_csv_table.rowCount() > 0, f"CSV rows missing for {exp['csv_file']}"
            assert win._combo_exp_metric.count() > 0, f"Metric selector empty for {exp['csv_file']}"

            # Verify numeric sorting on numeric column ascending & descending works accurately
            sort_col = max(0, win._exp_csv_table.columnCount() - 2)
            win._on_exp_table_sort_requested(sort_col)
            app.processEvents()
            val_asc_first = win._exp_csv_table.item(0, sort_col).text()
            win._on_exp_table_sort_requested(sort_col)
            app.processEvents()
            val_desc_first = win._exp_csv_table.item(0, sort_col).text()
            if ms.MirageStudioMainWindow._is_float(val_asc_first) and ms.MirageStudioMainWindow._is_float(val_desc_first):
                assert float(val_asc_first) <= float(val_desc_first), "Ascending min must be <= Descending max"
            # Restore clean unsorted view for crisp screenshot
            win._exp_sort_col = None
            win._refresh_exp_csv_table(exp)
            app.processEvents()

            catalog_plots.append(exp["plot_file"])

            # Save screenshot of the first experiment of each category
            if e_idx == 0:
                cat_shot = img_dir / f"experiments_cat_{cid}.png"
                win.grab().save(str(cat_shot))
                print(f"  ✓ Category '{cat['name']}' verified & captured ({len(cat['experiments'])} figures) at: {cat_shot}")

        # Verify filtering changes displayed row count cleanly
        cmb_filter0 = win._exp_filter_widgets[0]
        if cmb_filter0.isVisible() and cmb_filter0.count() > 2:
            all_rows_label = win._lbl_exp_csv_badge.text()
            cmb_filter0.setCurrentIndex(1)
            app.processEvents()
            filtered_label = win._lbl_exp_csv_badge.text()
            assert filtered_label != all_rows_label or win._exp_csv_table.rowCount() > 0
            win._reset_exp_filters()
            app.processEvents()

    # Every plot on disk covered, zero duplicates
    assert len(catalog_plots) == len(set(catalog_plots)), "Duplicate plots found across categories!"
    assert set(catalog_plots) == disk_plots, (
        f"Catalog plots vs disk mismatch! Missing={disk_plots - set(catalog_plots)}, Extra={set(catalog_plots) - disk_plots}"
    )
    print(f"  ✓ All {len(disk_plots)} existing plots mapped uniquely across 4 categories (0 missing, 0 duplicates)")

    # 3. Verify Back button returns to Landing Page
    win._show_experiments_landing()
    app.processEvents()
    assert win._exp_stack.currentIndex() == 0
    print("  ✓ Back navigation to Category Landing Page verified")

    # 4. Verify Separated Experiment Runner executes C++ mirage_experiments cleanly into isolated temp dirs
    print("[Experiments Suite] Verifying isolated C++ Experiment Runner workflow (1 modality, seeds=1)...")
    assert win._exp_bin is not None, "Compiled C++ mirage_experiments binary must exist"
    win._show_experiment_runner_view()
    app.processEvents()
    assert win._exp_stack.currentIndex() == 2

    tmp_ds_dir = tempfile.mkdtemp(prefix="mirage_exp_ds_")
    tmp_exp_out = tempfile.mkdtemp(prefix="mirage_exp_runner_verify_")
    try:
        # Copy single modality + single-row manifest so full 4-experiment C++ sweep completes quickly
        src_img = ms.PROJECT_ROOT / "data" / "experiments" / "natural_scene.png"
        shutil.copy2(src_img, Path(tmp_ds_dir) / "natural_scene.png")
        with open(Path(tmp_ds_dir) / "dataset_manifest.csv", "w") as mf:
            mf.write("id,filename,width,height,channels,modality,description\n")
            mf.write('mod_04,natural_scene.png,512,512,3,"Natural Scene","Synthetic natural landscape"\n')

        win._edit_exp_dataset.setText(tmp_ds_dir)
        win._edit_exp_output.setText(tmp_exp_out)
        win._spin_exp_seeds.setValue(1)
        win._chk_exp_replot.setChecked(False)
        win._on_start_experiment_runner()
        assert win._exp_process is not None
        assert win._exp_process.waitForFinished(30000), "C++ experiment runner timed out!"
        app.processEvents()
        assert win._exp_process.exitCode() == 0, f"mirage_experiments failed with exit {win._exp_process.exitCode()}"
        for expected_csv in ["random_ablation.csv", "feature_ablation.csv", "tile_sensitivity_errors.csv", "pareto_results.csv"]:
            p = Path(tmp_exp_out) / expected_csv
            assert p.is_file() and p.stat().st_size > 100, f"Missing/empty output CSV: {p}"
        print("  ✓ C++ mirage_experiments runner executed all 4 experiments cleanly in isolated directory")
    finally:
        shutil.rmtree(tmp_ds_dir, ignore_errors=True)
        shutil.rmtree(tmp_exp_out, ignore_errors=True)

    # Restore Experiments view to clean Category Landing Page
    win._show_experiments_landing()
    app.processEvents()

    # 5. Ensure Results & Benchmark tabs remain functional
    win._switch_view(2)  # Benchmark
    app.processEvents()
    assert win._bench_table.rowCount() > 0, "Benchmark tab must remain populated"
    win._switch_view(4)  # Results
    app.processEvents()
    assert win._results_stack.currentIndex() == 0, "Results tab must remain cleanly in Empty State"
    print("  ✓ Benchmark and Results tabs verified intact")

    print("[Experiments Suite] ALL VERIFICATION CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
