# Real-Time Session-Based E-Commerce Purchase Intention Prediction

[![CI](https://github.com/KaranRay1/rees46-purchase-intention/actions/workflows/ci.yml/badge.svg)](https://github.com/KaranRay1/rees46-purchase-intention/actions/workflows/ci.yml)

**A comparative analysis of tabular machine learning and deep neural architectures on REES46.**

Joint coursework for the NCI MSc in AI, Machine Learning module, by Karan Ray and Merve Kara.

From only the **first K events** of a shopping session (K ∈ {3, 5, 10}), we predict
whether the session will **end in a purchase**. We compare a rule baseline, logistic
regression, random forest, LightGBM, MLP, TabNet and a GRU over the raw event sequence.

The study is packaged as a product, the **Purchase Intent Engine**:

- a dashboard with shop analytics, the two teams' models, live scoring, a campaign planner and monitoring;
- a REST API (`POST /score`) and a Docker image;
- model and data cards generated from the results;
- CI on every push.

Experiments are logged to plain files under `results/` (no MLflow). The fuller MLOps
blueprint is kept in [docs/archive/blueprint_with_mlops.md](docs/archive/blueprint_with_mlops.md).

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
  data/                   loader (CSV → cleaned Parquet), categories (Dec 2019 tree fix),
                          sessionize (prefixes + target), split (chronological), synthetic
  features/               build_features (G1–G5), preprocessing (fit on train), sequences (GRU)
  models/                 BaseModel + rule, LR, RF, LightGBM, MLP, TabNet, GRU
  evaluation/             metrics, statistics (bootstrap, McNemar, Holm), efficiency (latency),
                          monitoring (weekly accuracy, PSI drift)
  explainability/         SHAP / TabNet attention, group aggregation, error groups
  utils/                  results_logger (replaces MLflow), notebook helpers
  serving/                predictor (score a live session), recommend (engine model, intent levels),
                          campaign (planner maths), catalog (shop aggregates), results reader
  api/                    FastAPI service: /health, /models, /score, /score/all
scripts/                  01–06 run the study; 07 monitoring, 08 model/data cards, 09 demo bundle
notebooks/                01–04 present the results
app/                      Streamlit dashboard (business, models, engine pages)
tests/                    pytest suite (synthetic data, temporary folders)
data/                     raw/ (you add the CSVs), interim/, processed/k{3,5,10}/   — git-ignored
results/                  runs/ (one folder per run, git-ignored), tables/, figures/, demo/ (bundle)
docs/                     model and data cards, research blueprint, literature review, archive
Dockerfile, docker-compose.yml, requirements-serve.txt   serving image (API + dashboard)
.github/workflows/ci.yml  lint, tests and image build on every push
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

**Sharing or trying the project without the dataset:** see [HOW_TO_RUN.md](HOW_TO_RUN.md).
`bash scripts/try_synthetic.sh` runs the whole pipeline on generated data in about a minute, and
`bash scripts/make_share_zip.sh` builds a zip of the project without the data (`dist/`).

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

REES46 "eCommerce behavior data from a multi-category store", Oct 2019 – Apr 2020
([Kaggle page](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store);
the monthly files are also served without login from `https://data.rees46.com/datasets/marketplace/`).
Download them with:

```bash
.venv/bin/python scripts/00_download_data.py --mode dev
```

(`--mode full` for all seven months, ~53 GB on disk.) The script puts the unzipped CSVs, with their
original names, in `data/raw/`:

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

The multi-month results in this repository were produced with a 10 % sample of whole
sessions (about 40 M events, the size of one month) and all three seeds, about five hours
on a laptop:

```bash
python scripts/01_prepare_data.py --mode full --set data.session_sample_fraction=0.1
```

Pass the same `--set` to scripts 02–09.

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
```bash
.venv/bin/python scripts/07_monitor.py --mode dev
```
```bash
.venv/bin/python scripts/08_model_cards.py --mode dev
```
```bash
.venv/bin/python scripts/10_calibrate.py --mode dev
```
```bash
.venv/bin/python scripts/09_build_demo.py --mode dev
```

| Step | What it does | Main outputs (`results/tables`, `results/figures`) |
|---|---|---|
| 01 | Clean events, build K-event prefixes and targets, chronological split | `data_summary_<mode>.json` |
| 02 | G1–G5 feature tables and integrity checks, GRU sequence alignment, buyer profile | `feature_summary_<mode>.json`, `feature_profile_<mode>_k*.csv` |
| 03 | **RQ1/RQ2**: 7 models × 3 K × 3 seeds, threshold tuning, significance tests | `rq1_model_comparison`, `rq1_pairwise_tests`, `rq2_by_k`, `rq1_rq2_hypotheses` |
| 04 | **RQ3**: ablation ladder S1–S5 + variant on LightGBM and MLP | `rq3_ablation`, `rq3_step_tests`, `rq3_hypothesis` |
| 05 | **RQ4**: single-thread latency (p50/p95/p99), throughput, memory, size | `rq4_efficiency`, `rq4_hardware`, `rq4_hypothesis` |
| 06 | **RQ5**: SHAP / attention, group importance, agreement, error groups | `rq5_importance_*`, `rq5_error_profile_*`, `rq5_hypothesis` |
| 07 | Monitoring: weekly PR-AUC (95 % CI) on the weeks after training, PSI drift per feature and month | `monitoring_weekly`, `monitoring_drift`, `monitoring_summary` |
| 08 | Model card and data card from the tables | `docs/model_card_<mode>.md`, `docs/data_card_<mode>.md` |
| 09 | Demo bundle for hosting (models, test scores, catalogue, calibration); run after 10 | `results/demo/<mode>/` |
| 10 | Calibration: Platt scaling fitted on the validation weeks turns scores into purchase probabilities (ranking unchanged) | `calibration.json` per run, `calibration`, `calibration_reliability` |

Scripts 05–10 retrain nothing: they load the primary-seed models saved by step 03.

**Monitoring another mode.** Script 07 needs `data/processed` to hold the same mode and
stops with an explanation otherwise. To monitor `dev` while `data/processed` holds `full`,
prepare `dev` into its own folder and point script 07 at it:

```bash
.venv/bin/python scripts/01_prepare_data.py --mode dev --set paths.processed_dir=data/interim/processed_dev
```
```bash
.venv/bin/python scripts/02_build_features.py --mode dev --set paths.processed_dir=data/interim/processed_dev --skip-sequences
```
```bash
.venv/bin/python scripts/07_monitor.py --mode dev --set paths.processed_dir=data/interim/processed_dev
```

**Category fix.** From 1 December 2019 REES46 uses a new category tree whose labels do
not describe the products (iPhones appear as `construction.tools.light`). Cleaning maps
every product back to its Oct/Nov 2019 label (`data.cleaning.harmonise_categories`,
`src/data/categories.py`); it has no effect on data inside one tree, such as `dev`.

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

An application-style front end: shop analytics, the two teams' models, and the live
scoring engine. It reads what scripts 01–06 wrote and loads the saved models, so run the
study first (at least steps 01–03 for one mode).

```bash
streamlit run app/main.py
```

Then open <http://localhost:8501>. The sidebar picks the dataset (every mode with results
in `results/tables/` is listed, multi-month first) and the decision point K, the number of
clicks seen before scoring. The model pages show only results that hold up statistically.

| Section | Page | Shows |
|---|---|---|
| Business | Overview | Shop KPIs, monthly performance, the engine in production, top departments and products, model insights |
| | Sales & traffic | All months or any single month: KPIs with month-on-month change, daily trend, funnel, revenue by department, month-by-month table (CSV), the train/test window |
| | Product catalogue | Search by product ID, brand or category; filter by month, department, brand, price, views; product and brand lists (CSV); product detail with orders and price per month and category best sellers |
| Models | Machine learning · Merve | Rule, LR, random forest, LightGBM: accuracy, what they look at, what each feature group adds, speed |
| | Deep learning · Karan | MLP, TabNet, GRU: the same four views |
| | Model comparison | Head-to-head with significance, accuracy vs speed, the model to ship, the shared limit |
| Engine | Live scoring | Score a built or replayed visit: probability, intent level, suggested action, explanation, every model's opinion |
| | Campaign planner | Who should get a discount: skip the visitors who buy anyway, net value vs offering to everyone and vs random, with your discount, offer effect and costs |
| | Monitoring | Weekly PR-AUC with 95 % CI on the weeks after training, what drives the dips, input drift (PSI) per feature and month, alert rules |

The Business pages use `src/serving/catalog.py`, which summarises the cleaned events once
per mode (per day into `results/tables/daily_profile_<mode>.csv`, per product and month
into `data/interim/product_month_<mode>.parquet`; about 15 s for the seven months).
REES46 switched to a new category tree in December 2019 whose labels do not match the
products (iPhones appear as `construction.tools.light`), so each product keeps its
Oct/Nov 2019 category, and products first seen later are mapped through their new
category id.

The live predictor uses `src/serving/predictor.py`: raw events → the same feature code as
training → each model's saved preprocessor → score and validation-tuned threshold, plus
LightGBM TreeSHAP contributions. Replaying real sessions needs `data/processed/` to hold
the selected mode.

**Campaign planner.** Under a uniform offer effect, discounting the highest-scoring
visitors is worse than random: they buy anyway. The planner therefore uses the engine to
*leave out* the top-scoring share and finds the share that maximises net value
(`src/serving/campaign.py`). The offer effect is an assumption to measure with an A/B test.

## Advanced ML (steps 12-16)

Ten extensions beyond the main study. Each one is judged on the same test weeks as RQ1, with
paired bootstrap tests and Holm correction, and shown on the dashboard page **Advanced ML**.

| Step | Question | Method | Main outputs |
|---|---|---|---|
| 12 | Can the engine say "unsure" with a guarantee? | Class-conditional split conformal prediction, calibrated on validation | `conformal`, `conformal_weekly` |
| 12 | Do ML and DL models make different mistakes? | Stacked generalisation: logistic meta-model on validation scores | `stacking`, `stacking_tests`, `error_overlap` |
| 13 | Does the model decay, and does retraining fix it? | Monthly deployment replay: frozen vs expanding vs sliding windows | `retraining`, `retraining_tests` |
| 14 | Do learned representations help? | item2vec (PPMI-SVD of co-views) and LLM text embeddings as features | `embeddings_ablation`, `embeddings_tests` |
| 15 | How much is tuning worth? | Optuna TPE on the validation weeks, confirmed with 3 seeds on test | `tuning`, `tuning_trials`, `tuning_importance` |
| 16 | Does attention beat recurrence? | Transformer encoder vs GRU on the same click sequences | `transformer`, `transformer_tests`, `transformer_attention` |
| 17 | Does knowing the user's past help? | G6 user history (earlier sessions in a 30-day window, from the full event log) added to LightGBM | `user_history_ablation`, `user_history_tests`, `user_history_segments` |
| 18 | Which model earns the most? | Every model drives the same discount campaign; cutoff chosen on half the test visits, profit measured on the other half, paired bootstrap + Holm, discount x uplift sensitivity (shown on **Campaign planner**) | `business_value`, `business_value_tests`, `business_value_sensitivity` |
| 19 | Does the model work, and treat visits, alike across groups? | Segment (fairness) analysis by category, price, cart, time of day, weekday/weekend and new/returning user: PR-AUC lift, calibration gap, flag rate, TPR/FPR at one global cutoff; GRU vs RF per segment (Holm) | `segment_metrics`, `segment_disparity`, `segment_tests` |
| 20 | Does a tool-using agent beat plain RAG? | Local LLM agent (Ollama function calling) with six tools: docs search, table reading, model comparison, live session scoring, campaign valuation; scored against RAG on standard and tool-needing questions (shown on **Analyst assistant**) | `agent_eval_summary`, `agent_eval_answers` |

```bash
.venv/bin/python scripts/12_conformal_stacking.py --mode full --set data.session_sample_fraction=0.1
```
```bash
.venv/bin/python scripts/13_drift_retraining.py --mode full --set data.session_sample_fraction=0.1
```
```bash
.venv/bin/python scripts/14_embeddings.py --mode full --set data.session_sample_fraction=0.1
```
```bash
.venv/bin/python scripts/15_tune.py --mode full --set data.session_sample_fraction=0.1
```
```bash
.venv/bin/python scripts/16_transformer.py --mode full --set data.session_sample_fraction=0.1
```
```bash
.venv/bin/python scripts/17_user_history.py --mode full --set data.session_sample_fraction=0.1
```
```bash
.venv/bin/python scripts/18_business_value.py --mode full --set data.session_sample_fraction=0.1
```
```bash
.venv/bin/python scripts/19_segments.py --mode full --set data.session_sample_fraction=0.1
```
```bash
.venv/bin/python scripts/20_agent_eval.py --mode full --set data.session_sample_fraction=0.1 --models llama3.1:8b qwen3:14b
```

Step 14 adds the text embeddings only when `ollama pull nomic-embed-text` has been run. Step 17 reads user history from the full
event log written by `scripts/run_full_scale.sh` when it exists. Steps 12
and 16 add no models to the RQ1 tables: the Transformer is logged as experiment
`rq6_transformer` and the tuned models as `rq7_tuned`.

## AI analyst assistant (LLM + RAG)

The **Analyst assistant** page (dashboard section *AI*) answers questions about the models,
results, data and monitoring with retrieval-augmented generation:

- **Knowledge base** (`src/llm/corpus.py`): the report, model and data cards, report updates and
  this README split into passages, plus fact sheets generated from every result table.
- **Retrieval** (`src/llm/retrieval.py`): BM25, embeddings (`nomic-embed-text` via Ollama) and a
  hybrid using reciprocal rank fusion.
- **Generation** (`src/llm/providers.py`): an open model running locally in Ollama
  (`llama3.1:8b` by default; any pulled model such as `qwen3:14b` can be picked), told to
  answer only from the sources and cite them as [S1]. Nothing leaves the computer and no
  API key is needed.
- **Checks** (`src/llm/rag.py`): every citation must exist and every number must appear in the
  sources at the precision quoted. The Live scoring page uses the same checks for its
  "Explain in plain English" button.
- **Evaluation** (`scripts/11_rag_eval.py`): hand-written and table-generated questions with
  known answers; recall@k and MRR per retriever, and correctness, abstention, grounding and
  latency per model.

Setup (all optional; without a model the assistant still shows the best sources):

```bash
cp .env.example .env
```
```bash
ollama pull llama3.1:8b
```
```bash
ollama pull nomic-embed-text
```

Then evaluate (compares the local models on the same questions):

```bash
.venv/bin/python scripts/11_rag_eval.py --mode full --set data.session_sample_fraction=0.1 --models llama3.1:8b qwen3:14b
```

## REST API and Docker

```bash
uvicorn src.api.app:app --port 8000
```

Interactive docs: <http://localhost:8000/docs>. `PIE_MODE` picks the dataset (default `full`).

| Endpoint | Does |
|---|---|
| `GET /health` | Status, mode and the K values with trained models |
| `GET /models?k=5` | Models, thresholds and the recommended engine model |
| `POST /score` | One visit's first K events → calibrated purchase probability, raw score, buyer yes/no, intent, action, latency; `explain: true` adds LightGBM contributions |
| `POST /score/all` | The same from every model |

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{"k": 3, "events": [
  {"event_time": "2020-04-10T14:00:00Z", "event_type": "view", "product_id": 1005115, "category_code": "electronics.smartphone", "brand": "apple", "price": 900.9},
  {"event_time": "2020-04-10T14:00:40Z", "event_type": "view", "product_id": 1005115, "category_code": "electronics.smartphone", "brand": "apple", "price": 900.9},
  {"event_time": "2020-04-10T14:01:30Z", "event_type": "cart", "product_id": 1005115, "category_code": "electronics.smartphone", "brand": "apple", "price": 900.9}]}'
```

Both services in containers (results and data are mounted, not copied into the image):

```bash
docker compose up --build
```

API on <http://localhost:8000>, dashboard on <http://localhost:8501>.

## Online demo

`results/demo/<mode>/` (built by `scripts/09_build_demo.py`, about 35 MB) holds what a
hosted copy needs without the raw data or run folders. It contains:

- the primary-seed models, except the 100 MB random forest;
- every model's test scores;
- the product catalogue.

When the run folders are missing (a fresh clone, or a cloud host), the dashboard and API
switch to the bundle automatically. `PIE_DEMO=1` forces it; `PIE_DEMO=0` turns it off.

To host the dashboard on Streamlit Community Cloud:

1. Push the repository to GitHub.
2. On share.streamlit.io, create an app with main file `app/main.py` and Python 3.12.
3. Streamlit reads `app/requirements.txt`, which installs CPU-only PyTorch.

Visit replay is not available in the demo.

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
- **`docs/model_card_<mode>.md`, `docs/data_card_<mode>.md`**: generated by script 08
  from the tables; regenerate after every run.

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

The suite covers config validation, cleaning, category harmonisation, the leakage rules,
the split, features, preprocessing, every model, metrics against scikit-learn,
statistics, efficiency, SHAP aggregation, the results logger, monitoring, the campaign
maths, the REST API and the demo switch-over. It also runs scripts 01–07, every
dashboard page and all four notebooks end to end on synthetic data in temporary folders.

`ruff check .` lints the code (settings in `ruff.toml`). GitHub Actions runs lint, the
tests and a Docker build on every push (`.github/workflows/ci.yml`).

## Documents

- [docs/model_card_full.md](docs/model_card_full.md) and [docs/data_card_full.md](docs/data_card_full.md): the engine's model card and the data card (generated).
- [docs/research_blueprint.md](docs/research_blueprint.md): research questions, hypotheses, models, leakage rules, evaluation.
- [docs/proposals_analysis.md](docs/proposals_analysis.md): how the two individual proposals were merged.
- [docs/literature_review.md](docs/literature_review.md): literature review (in progress).
- [docs/archive/blueprint_with_mlops.md](docs/archive/blueprint_with_mlops.md): the Phase 2 production / MLOps plan, kept for later.
