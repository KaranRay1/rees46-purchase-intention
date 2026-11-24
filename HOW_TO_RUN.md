# How to run this project (without the dataset)

**Real-Time Session-Based E-Commerce Purchase Intention Prediction** (REES46)
Karan Ray and Merve Kara, NCI MSc in Artificial Intelligence, Machine Learning.

This folder contains the full code, the result tables and figures, the trained demo models
and the documents. **The REES46 dataset is not included**: it is 53 GB, and it is published
on Kaggle under its own licence. You don't need it to run anything in sections 1 to 4.

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
the models we used. Without Ollama, the page still answers by retrieval only.

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

1. Download *eCommerce behavior data from a multi-category store* (REES46) from Kaggle,
   Oct 2019 to Apr 2020, and put the seven monthly CSVs in `data/raw/` with their original
   names (`2019-Oct.csv` … `2020-Apr.csv`).
2. Run the pipeline; see **Running the study** in `README.md`. The headline results use a 10 %
   session sample:

```bash
.venv/bin/python scripts/01_prepare_data.py --mode full --set data.session_sample_fraction=0.1
```

   followed by scripts 02 to 19 with the same `--set`. A machine with 24 GB of RAM is
   enough. `scripts/run_full_scale.sh` repeats the main study on 100 % of sessions
   (~32 GB peak memory).

---

## Where things are

| Folder | What it holds |
|---|---|
| `src/` | the library: data, features, models, evaluation, serving, LLM/RAG |
| `scripts/` | the numbered pipeline steps 01 to 19 |
| `app/` | the Streamlit dashboard |
| `results/tables/`, `results/figures/` | every result in the report (`*_full` = REES46 run) |
| `results/full100/` | the 100 % full-data validation |
| `results/demo/` | the small model bundle the dashboard uses |
| `docs/` | report, process guide, learning guide, model and data cards |
| `tests/` | pytest suite |
| `config/config.yaml` | every setting (data, K values, features, models, seeds) |
