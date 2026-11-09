# Offline Research Analysis & Evaluation

This directory contains offline research evaluation scripts, statistical notebooks, and analysis tooling for the **Mirage** project.

## Purpose

The core image enhancement, sensitivity analysis, and precision selection engine is implemented strictly in C++20. The Python scripts in this directory are reserved exclusively for:
- Offline statistical evaluation of enhanced images (PSNR, SSIM, Delta E 2000, CIE Lab color distance).
- Parsing and plotting hardware performance counter data (`perf`) and Google Benchmark output.
- Generating publication-quality Pareto curves comparing image quality (dB PSNR / SSIM) versus execution latency (ms) and energy estimates.

## Dependencies

Future analysis phases will utilize:
- Python 3.10+
- NumPy
- Matplotlib
- scikit-image
- colour-science

## Structure (Future Phases)

- `metrics/`: Python implementations of color difference and perceptual fidelity metrics.
- `plots/`: Automated plotting scripts for Pareto frontiers and error histograms.
- `notebooks/`: Jupyter notebooks for exploratory data analysis of experimental datasets.

