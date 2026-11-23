#!/usr/bin/env bash
# Run the whole pipeline (data -> features -> all models -> tests) on generated data,
# so the project can be tried without the 53 GB REES46 dataset. Takes a few minutes.
#
# It writes to its own folders, so the shipped results and the dashboard are untouched:
#   data/synthetic_try/   results/synthetic_try/
# Usage:  bash scripts/try_synthetic.sh
set -euo pipefail
cd "$(dirname "$0")/.."
W=synthetic_try
PY=${PYTHON:-.venv/bin/python}
[ -x "$PY" ] || PY=python
SETS=(
  --mode synthetic
  --set "paths.interim_dir=data/$W/interim"
  --set "paths.processed_dir=data/$W/processed"
  --set "paths.tables_dir=results/$W/tables"
  --set "paths.runs_dir=results/$W/runs"
  --set "paths.figures_dir=results/$W/figures"
  --set "paths.all_runs_csv=results/$W/tables/all_runs.csv"
)
mkdir -p "data/$W/interim" "data/$W/processed" "results/$W/tables" "results/$W/runs" "results/$W/figures"
"$PY" scripts/01_prepare_data.py "${SETS[@]}"
"$PY" scripts/02_build_features.py "${SETS[@]}"
"$PY" scripts/03_run_experiments.py "${SETS[@]}"
echo
echo "Done. Result tables: results/$W/tables/ (synthetic data, so the numbers are not meaningful)."
