#!/usr/bin/env bash
# Full-scale validation: rerun the main experiment on 100 % of sessions (no sampling) at K=5.
#
# The headline results use a 10 % session sample (data.session_sample_fraction=0.1) to fit in
# memory. This script checks that the conclusions hold on all ~89.7 M sessions. It writes to a
# separate workspace so the 10 % results and the dashboard are untouched:
#
#   data/full100/{interim,processed}   results/full100/{tables,runs,figures}
#
# The typed monthly Parquet files from step 1 are reused (symlinked), not converted again.
# Usage:  bash scripts/run_full_scale.sh          (log: results/full100/pipeline.log)
set -euo pipefail
cd "$(dirname "$0")/.."

W=full100
mkdir -p "data/$W/interim" "data/$W/processed" "results/$W/tables" "results/$W/runs" "results/$W/figures"
[ -e "data/$W/interim/raw" ] || ln -s ../../interim/raw "data/$W/interim/raw"

SETS=(
  --mode full
  --set data.session_sample_fraction=null
  --set "sessionize.prefix_k=[5]"
  --set "paths.interim_dir=data/$W/interim"
  --set "paths.processed_dir=data/$W/processed"
  --set "paths.tables_dir=results/$W/tables"
  --set "paths.runs_dir=results/$W/runs"
  --set "paths.figures_dir=results/$W/figures"
  --set "paths.all_runs_csv=results/$W/tables/all_runs.csv"
)
PY=.venv/bin/python
TIME="/usr/bin/time -l"     # records peak memory of each step

echo "start $(date '+%F %H:%M')"
$TIME $PY scripts/01_prepare_data.py "${SETS[@]}"
$TIME $PY scripts/02_build_features.py "${SETS[@]}"
$TIME $PY scripts/03_run_experiments.py "${SETS[@]}" \
  --models logistic_regression lightgbm random_forest gru --ks 5 --seeds 42 --n-resamples 300
echo "end $(date '+%F %H:%M')"
