# Real-Time Session-Based E-Commerce Purchase Intention Prediction

**A comparative analysis of tabular machine learning and deep neural architectures on REES46.**

Joint coursework for the NCI MSc in AI, Machine Learning module, by Karan Ray and Merve Kara.

From only the **first K events** of a shopping session (K ∈ {3, 5, 10}), we predict
whether the session will **end in a purchase**. We compare a rule baseline, logistic
regression, random forest, LightGBM, MLP, TabNet and a GRU over the raw event sequence.

This is Phase 1, the research and analytics version. It has deliberately **no MLOps**:
experiments are logged to plain files under `results/`. The production blueprint is
kept unchanged for Phase 2 in [docs/archive/blueprint_with_mlops.md](docs/archive/blueprint_with_mlops.md).

---

## Research questions

The main question is split into five sub-questions. The full design is in
[docs/research_blueprint.md](docs/research_blueprint.md).

| RQ | Question | Hypotheses | Script → notebook |
|---|---|---|---|
| RQ1 | Which model family predicts purchase best? | H1a trees > LR · H1b MLP/TabNet > LR · H1c GRU ≈ best aggregated model at K=10 | `03` → `03_results_comparison` |
| RQ2 | How early can we predict, i.e. what is the effect of K? | H2 performance rises with K, with diminishing returns | `03` → `03_results_comparison` |
| RQ3 | What does each feature group add? | H3 behavioural ratios add PR-AUC on top of counts (+ exploratory variant) | `04` → `03_results_comparison` |
| RQ4 | Is it fast enough for real time, and at what cost? | H4 trees dominate the accuracy–latency Pareto front | `05` → `03_results_comparison` |
| RQ5 | Which behaviours drive predictions, and what do the errors look like? | H5 cart signals rank first in SHAP | `06` → `04_explainability_errors` |

**Leakage rules**, enforced in `src/data/sessionize.py` and covered by `tests/test_leakage.py`:

- A session is used only if it has **more than K** events.
- Sessions with a purchase **inside** the first K events are dropped.
- The target is a purchase **after** event K.
- Features are computed from the first K events only.
- The split is **chronological** (70/15/15 by session start). The cut-offs are shared across all K, and sessions that cross a cut-off are removed.
- Every preprocessing statistic is fitted on train only.

**Evaluation:**

- PR-AUC is the primary metric, read alongside lift (= PR-AUC ÷ base rate), ROC-AUC, and F1 at a threshold tuned on validation.
- Each configuration runs with three seeds.
- Paired bootstrap and McNemar tests compare models, Holm-corrected within the confirmatory family.

---

## Project layout

```
config/config.yaml        every setting (data, K, features, models, evaluation, efficiency)
src/
  config.py               load / override / validate the config
  data/                   loader (CSV → cleaned Parquet), sessionize (prefixes + target),
                          split (chronological), synthetic (REES46-shaped test data)
  features/               build_features (G1–G5), preprocessing (fit on train), sequences (GRU)
  models/                 BaseModel + rule, LR, RF, LightGBM, MLP, TabNet, GRU
  evaluation/             metrics, statistics (bootstrap, McNemar, Holm), efficiency (latency)
  explainability/         SHAP / TabNet attention, group aggregation, error groups
  utils/                  results_logger (replaces MLflow), notebook helpers
  serving/                predictor (score a live session with the saved models), results reader
scripts/                  01–06 run the study, in order (see below)
notebooks/                01–04 present the results
app/                      Streamlit dashboard: results pages + live predictor
tests/                    pytest suite (synthetic data, temporary folders)
data/                     raw/ (you add the CSVs), interim/, processed/k{3,5,10}/   — git-ignored
results/                  runs/ (one folder per run, git-ignored), tables/, figures/
docs/                     research blueprint, proposals analysis, literature review, archive
```

### Feature groups (RQ3 ablation ladder)

| Group | Features |
|---|---|
| G1 activity | unique_products_count, session_duration_sec, max_gap_sec |
| G2 cart signals | view_count, cart_count, has_cart, first_cart_position, last_event_type |
| G3 behavioural ratios | cart_to_view_ratio |
| G4 category hierarchy | unique_categories_count, unique_brands_count, main_category |
| G5 price & time | price_mean, price_max, price_sum, price_std, hour_of_day, day_of_week |

`view_count` belongs to G2: with K fixed, view = K − cart − remove, so it carries
the cart signal.

The REES46 multi-category data has **no `remove_from_cart` events** (only view, cart and
purchase), so the proposal's `remove_count` and `remove_to_cart_ratio` would be 0 for every
session. They are left out, giving 18 features. `build_features.py` still computes them, so
listing them in `config.yaml → features.groups` (and `preprocessing.log1p_features`) switches
them back on for a dataset with removals.

---

## Setup

You need Python **3.12**. Everything runs on CPU; no GPU is needed.

```bash
python3.12 -m venv .venv
```
```bash
.venv/bin/pip install -r requirements.txt
```

**macOS only.** LightGBM needs `libomp`, which macOS doesn't ship. Without Homebrew, point
LightGBM at the copy bundled with PyTorch. Run this once, and again after reinstalling
`lightgbm` or `torch`:

```bash
bash scripts/fix_macos_lightgbm.sh
```

To check the install, run the tests. They use synthetic data only and write to
temporary folders, so they don't need the dataset:

```bash
.venv/bin/python -m pytest -q
```

### Data

REES46 "eCommerce behavior data from a multi-category store", Oct 2019 – Apr 2020. It's
on Kaggle (2019-Oct and 2019-Nov); the dataset page links to the remaining months. Put
the monthly CSVs, unzipped and with their original names, in `data/raw/`:

```
data/raw/2019-Oct.csv      ← enough for --mode dev
data/raw/2019-Nov.csv  …  data/raw/2020-Apr.csv   ← needed for --mode full
```

Raw and generated data are git-ignored.

### Data modes

Every script and notebook takes a mode. The files each mode uses are listed in
`config.yaml → data`.

| Mode | Data | Use |
|---|---|---|
| `synthetic` | generated in memory, REES46-shaped | trying the pipeline end to end in minutes |
| `dev` | `2019-Oct.csv` | development and first results |
| `full` | Oct 2019 – Apr 2020 | final results for the report |

---

## Running the study

Run the scripts in order from the project root. Each one reads the previous one's output.

```bash
.venv/bin/python scripts/01_prepare_data.py --mode dev
```
```bash
.venv/bin/python scripts/02_build_features.py --mode dev
```
```bash
.venv/bin/python scripts/03_run_experiments.py --mode dev
```
```bash
.venv/bin/python scripts/04_run_ablation.py --mode dev
```
```bash
.venv/bin/python scripts/05_benchmark_latency.py --mode dev
```
```bash
.venv/bin/python scripts/06_explain.py --mode dev --ks 3 5 10
```

| Step | What it does | Main outputs (`results/tables`, `results/figures`) |
|---|---|---|
| 01 | Clean events, build K-event prefixes and targets, chronological split | `data_summary_<mode>.json` |
| 02 | G1–G5 feature tables and integrity checks, GRU sequence alignment, buyer profile | `feature_summary_<mode>.json`, `feature_profile_<mode>_k*.csv` |
| 03 | **RQ1/RQ2**: 7 models × 3 K × 3 seeds, threshold tuning, significance tests | `rq1_model_comparison`, `rq1_pairwise_tests`, `rq2_by_k`, `rq1_rq2_hypotheses` |
| 04 | **RQ3**: ablation ladder S1–S5 + variant on LightGBM and MLP | `rq3_ablation`, `rq3_step_tests`, `rq3_hypothesis` |
| 05 | **RQ4**: single-thread latency (p50/p95/p99), throughput, memory, size | `rq4_efficiency`, `rq4_hardware`, `rq4_hypothesis` |
| 06 | **RQ5**: SHAP / attention, group importance, agreement, error groups | `rq5_importance_*`, `rq5_error_profile_*`, `rq5_hypothesis` |

Scripts 05 and 06 retrain nothing: they load the primary-seed models saved by step 03.

**Common options.** Every script accepts `--mode`, `--set` and `--config`. Scripts 03–06 also take `--models` and `--ks`, and scripts 03–04 take `--seeds` and `--n-resamples`:

```bash
.venv/bin/python scripts/03_run_experiments.py --mode dev --models logistic_regression lightgbm --ks 5 --seeds 42
```
```bash
.venv/bin/python scripts/03_run_experiments.py --mode dev --analyse-only
```
```bash
.venv/bin/python scripts/03_run_experiments.py --mode dev --set models.lightgbm.n_estimators=500
```

- `--set key.path=value` overrides any config value and can be repeated. A typo in the key raises an error instead of being ignored.
- `--analyse-only` (scripts 03 and 04) rebuilds the tables and tests from runs that are already logged, without training.
- `--no-report` (script 01) skips the per-rule cleaning counts, which saves time on `full`.
- `--overwrite` (script 01) rebuilds cached Parquet files after you change a cleaning rule.
- `--skip-sequences` (script 02) skips building and checking the GRU sequences.

**Switching modes.** `data/processed/` holds one mode at a time. After switching
(e.g. `dev` → `full`), rerun scripts 01 and 02 before the experiments. The notebooks
detect a mismatch and tell you which command to run.

### Notebooks

```bash
NB_MODE=dev .venv/bin/jupyter lab
```

| Notebook | Needs | Contents |
|---|---|---|
| `01_eda` | step 01 | cleaning, event mix, sessions, first-purchase position vs K, eligibility per K, drift, cart signal |
| `02_feature_analysis` | steps 01–02 | integrity checks, buyer vs non-buyer SMD, single-feature AUC, redundancy, log transforms, drift, GRU view |
| `03_results_comparison` | steps 03–05 | RQ1–RQ4 tables and figures, calibration, Pareto fronts, **hypothesis scoreboard** |
| `04_explainability_errors` | steps 03, 06 | RQ5 group/feature importance, model agreement, error analysis, buyers every model misses |

Run notebook 03 before 04: notebook 04 fills the H5 row of the scoreboard that notebook 03 writes
(`results/tables/hypothesis_scoreboard_<mode>.csv`).

**Configuring the notebooks:**

- Set `NB_MODE` to pick the data mode. Set `NB_OVERRIDES` to a `;`-separated list of
  `key=value` overrides, the same as `--set`.
- A notebook whose inputs are missing stops, or skips that section, and prints the
  command to run. With `NB_MODE=synthetic`, notebooks 01 and 02 build their data on the fly.
- Notebooks are committed **without outputs**, which a test enforces. Clear outputs
  before committing.

---

## Dashboard

An interactive front end for presenting the study. It reads what scripts 01–06 wrote and
loads the saved models, so run the study first (at least steps 01–03 for one mode).

```bash
streamlit run app/main.py
```

Then open <http://localhost:8501>. The sidebar picks the data mode (every mode with results
in `results/tables/` is listed) and the prefix length K. It has these pages:

| Page | Shows |
|---|---|
| Overview | Headline numbers, the hypothesis scoreboard, the pipeline |
| Data & pipeline | Cleaning, eligibility per K, the chronological split, EDA figures |
| RQ1–2 · Models & K | PR-AUC with CIs, lift vs K, pairwise significance heatmap |
| RQ3 · Features | Ablation ladder, step gains, ratios-instead-of-counts variant |
| RQ4 · Real-time cost | Accuracy vs latency (Pareto front), throughput, cost sheet |
| RQ5 · Explainability | Attribution by feature group, model agreement, missed buyers |
| Live predictor | Build a session (or replay a real test session) and score it with every model |

The live predictor uses `src/serving/predictor.py`: raw events → the same feature code as
training → each model's saved preprocessor → score and validation-tuned threshold, plus
LightGBM TreeSHAP contributions. It does not depend on Streamlit, so a later phase can put
the same `SessionPredictor` behind an HTTP API, or score larger (`full`) runs, without
changes. Replaying real sessions needs `data/processed/` to hold the selected mode.

## Where results go

- **`results/runs/<run_id>/`**: one folder per trained or benchmarked model. It holds
  `config_snapshot.yaml`, `params.json`, `metrics.json`, per-session `predictions.parquet`,
  `model.pkl`, artifacts (preprocessor, encoder, training history), the git commit, and
  whether the tree had uncommitted changes. This folder is git-ignored.
- **`results/tables/all_runs.csv`**: one row per run, with flattened params and metrics.
  It replaces the MLflow UI; filter it with Polars or pandas.
- **`results/tables/*_<mode>.csv|json`** and **`results/figures/*_<mode>.png`**: the
  tables and figures for the report. File names carry the mode, so `dev` and `full`
  results sit side by side.

## Reproducibility

- **One config:** `config/config.yaml` is the single source of truth, and every run saves a copy of it.
- **Seeds:** `project.seeds` = [42, 7, 2026]. Statistical tests use the first (primary) seed; mean ± std is reported over all three.
- **Determinism:** LightGBM runs in deterministic mode. Sessions sampled for SHAP and bootstrap resamples are seeded.
- **Fair latency:** benchmarks use one CPU thread (`efficiency.n_threads`), and `rq4_hardware_<mode>.json` records the machine and library versions.
- **Real-time budget:** `efficiency.realtime_budget_ms` (10 ms, p95, one session) is an assumption, to be stated in the report.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite covers config validation, cleaning, the leakage rules, the split, features,
preprocessing, every model, metrics against scikit-learn, statistics, efficiency, SHAP
aggregation and the results logger. It also runs scripts 01–06 and all four notebooks
end to end on synthetic data in temporary folders.

## Documents

- [docs/research_blueprint.md](docs/research_blueprint.md): research questions, hypotheses, models, leakage rules, evaluation.
- [docs/proposals_analysis.md](docs/proposals_analysis.md): how the two individual proposals were merged.
- [docs/literature_review.md](docs/literature_review.md): literature review (in progress).
- [docs/archive/blueprint_with_mlops.md](docs/archive/blueprint_with_mlops.md): the Phase 2 production / MLOps plan, kept for later.
