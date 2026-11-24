# How to run this project (without the dataset)

**Real-Time Session-Based E-Commerce Purchase Intention Prediction** (REES46)
Karan Ray and Merve Kara, NCI MSc in Artificial Intelligence, Machine Learning.

This folder contains the full code, the result tables and figures, the trained demo models
and the documents. **The REES46 dataset is not included** (53 GB). It is public, and
`scripts/00_download_data.py` fetches it straight from the source when you need it (section 5).
You don't need it for anything in sections 1 to 4.

**Dataset:** REES46 *eCommerce behavior data from a multi-category store*, Oct 2019 – Apr 2020
- dataset page: <https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store>
- direct monthly files (no login): `https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz` …
  `2020-Apr.csv.gz`

| You want to… | Needs the dataset? | Time |
|---|---|---|
| 1. Open the dashboard with all our results and live predictions | No | 5 min |
| 2. Run the test suite (300+ tests) | No | 2 min |
| 3. Run the whole pipeline end to end on generated data | No | 1 min |
| 4. Open the notebooks | No | – |
| 5. Reproduce the real results | Yes | ~5 h |

---

## 0. Install (once)

You need **Python 3.12** (3.11 also works). Everything runs on CPU; no GPU is needed.
Open a terminal **in this folder**.

macOS / Linux:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The commands below use the macOS/Linux form `.venv/bin/python`. On Windows use
`.venv\Scripts\python` instead.

**macOS only:** LightGBM needs `libomp`. Run once: `bash scripts/fix_macos_lightgbm.sh`
(or `brew install libomp`).

## 1. Dashboard

```bash
.venv/bin/python -m streamlit run app/main.py
```

It opens at <http://localhost:8501>. Because the training run folders are not shipped, the
dashboard uses the **demo bundle** in `results/demo/full/` automatically: the primary-seed
models for every K, their test-period scores and the product catalogue. Every page works:
model comparison, live prediction, campaign planner, monitoring, explanations and Advanced ML.

The **Analyst assistant** page (local LLM + RAG) also needs [Ollama](https://ollama.com) with
the models we used. Without Ollama, the page still answers by retrieval only. With a model, *Agent* mode lets it call tools: read result tables, compare models, score a visit you describe and value a campaign.

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## 2. Tests

```bash
.venv/bin/python -m pytest -q
```

The tests generate small REES46-shaped data in temporary folders. They cover data cleaning,
leakage rules, features, every model, the statistics, the API, the dashboard pages and the
LLM/RAG code.

## 3. Whole pipeline on generated data

```bash
bash scripts/try_synthetic.sh
```

This runs data preparation, sessionising, features, all eight models and the statistical
tests on generated sessions. It writes to `data/synthetic_try/` and `results/synthetic_try/`,
so the shipped results are not changed. The numbers mean nothing (the data is random); the
point is to see every step run. Without bash (Windows), run the three commands inside that
script by hand, with `--mode synthetic`.

## 4. Notebooks

`notebooks/` holds the four analysis notebooks with their outputs saved. Open them with
`.venv/bin/python -m jupyter lab`.

## 5. Reproduce the real results (needs the dataset)

1. Download the data with one command. It streams each month from REES46's public server,
   unzips it on the fly into `data/raw/`, skips files already there, and checks free disk space first:

```bash
.venv/bin/python scripts/00_download_data.py --mode dev
```

   `--mode dev` fetches October 2019 only (1.7 GB download, ~5.7 GB on disk): enough to run every
   step with `--mode dev`. `--mode full` fetches all seven months (17.7 GB download, ~53 GB on
   disk). `--check` only tests the links. If the server is ever unavailable, download the CSVs
   from the Kaggle page above and put them in `data/raw/` with their original names.

2. Run the pipeline; see **Running the study** in `README.md`. The headline results use a 10 %
   session sample:

```bash
.venv/bin/python scripts/01_prepare_data.py --mode full --set data.session_sample_fraction=0.1
```

   followed by scripts 02 to 20 with the same `--set`. A machine with 24 GB of RAM is
   enough. `scripts/run_full_scale.sh` repeats the main study on 100 % of sessions
   (~32 GB peak memory).

---

## Where things are

| Folder | What it holds |
|---|---|
| `src/` | the library: data, features, models, evaluation, serving, LLM/RAG |
| `scripts/` | the numbered pipeline steps 01 to 20 |
| `app/` | the Streamlit dashboard |
| `results/tables/`, `results/figures/` | every result in the report (`*_full` = REES46 run) |
| `results/full100/` | the 100 % full-data validation |
| `results/demo/` | the small model bundle the dashboard uses |
| `docs/` | report, process guide, learning guide, model and data cards |
| `tests/` | pytest suite |
| `config/config.yaml` | every setting (data, K values, features, models, seeds) |
