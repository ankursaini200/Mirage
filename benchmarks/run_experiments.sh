#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================================"
echo "          MIRAGE PHASE 9: RESEARCH EXPERIMENTS & REPRODUCIBILITY PIPELINE"
echo "================================================================================"

BUILD_DIR="${ROOT_DIR}/build/release"
DATASET_DIR="${ROOT_DIR}/data/experiments"
EXPERIMENTS_DIR="${ROOT_DIR}/benchmarks/experiments"
PLOTS_DIR="${ROOT_DIR}/benchmarks/plots"

mkdir -p "${DATASET_DIR}" "${EXPERIMENTS_DIR}" "${PLOTS_DIR}"

# 1. Build targets
echo "[1/4] Building Mirage Release targets..."
cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target mirage_generate_dataset mirage_experiments -j"$(nproc)"

# 2. Generate multi-modal dataset if missing or forced
if [[ ! -f "${DATASET_DIR}/dataset_manifest.csv" ]]; then
    echo "[2/4] Generating 7-modality test dataset in ${DATASET_DIR}..."
    "${BUILD_DIR}/mirage_generate_dataset" "${DATASET_DIR}"
else
    echo "[2/4] Test dataset already exists in ${DATASET_DIR}. Skipping generation."
fi

# 3. Execute core C++ research experiments
echo "[3/4] Running Mirage C++ experiments (Random ablation, Feature ablation, Tile validation, Pareto sweeps)..."
"${BUILD_DIR}/benchmarks/mirage_experiments" \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${EXPERIMENTS_DIR}" \
    --seeds 30

# 4. Run Python statistical analysis and Cairo publication plotting
echo "[4/4] Executing Python statistical hypothesis tests and rendering Cairo publication plots..."
python3 "${ROOT_DIR}/scripts/analyze_experiments.py"

echo "================================================================================"
echo "Phase 9 experiment pipeline completed successfully!"
echo "CSVs saved to:   ${EXPERIMENTS_DIR}"
echo "Plots saved to:  ${PLOTS_DIR}"
echo "================================================================================"

