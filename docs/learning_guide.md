# Learning guide: the advanced steps (10–16) and the local AI assistant

This guide explains, step by step, what each new part of the project does, **why** it is
there, **where** the code lives, **how** to run it yourself and **how to read** its results.
Steps 01–09 (data, features, the seven models, statistics, dashboard) are described in the
README; this guide covers what was added on top to make the project master's level.

Every command is run from the project folder with the project's own Python:

```bash
cd "/Users/karan/Documents/NCI/Machine Learning/Project/E Commerce ML"
```

All full-data commands below use the same data setting as the main results:
`--mode full --set data.session_sample_fraction=0.1` (seven months, 10 % of sessions).

---

## 0. The tools you have installed, and what each one is for

| Tool | What it is | Used for |
|---|---|---|
| Python virtual env (`.venv`) | An isolated copy of Python with the project's packages | Everything; always use `.venv/bin/python` |
| PyTorch | Deep-learning library | MLP, GRU, Transformer |
| LightGBM, scikit-learn | Classical ML libraries | Trees, logistic regression, stacking, PCA/SVD |
| Optuna | Hyperparameter search library | Step 15 (tuning) |
| Streamlit | Web-app framework | The dashboard |
| **Ollama** | A program that runs open language models on your Mac | The AI assistant (steps 11) and text embeddings (step 14) |
| `llama3.1:8b` (4.9 GB) | Meta's open 8-billion-parameter chat model | Writes the assistant's answers (default) |
| `qwen3:14b` (9.3 GB) | Alibaba's open 14-billion-parameter chat model | Second model, compared against Llama |
| `nomic-embed-text` (274 MB) | An embedding model: turns text into a 768-number vector | Search in the assistant; product-text features |

Useful Ollama commands:

```bash
ollama list
```
```bash
ollama run llama3.1:8b
```
(type a question to chat; `/bye` to leave)

Nothing is sent to the internet: every model runs on your computer.

---

## Step 10: Calibration — turning scores into real probabilities

**Problem.** Models are trained with class weights (buyers are only ~10 % of visits), so a
model may say "0.62" for a visit whose true chance of purchase is only 0.20. The ranking is
fine (PR-AUC is unaffected) but the number is not a probability.

**Idea: Platt scaling.** Fit a tiny logistic regression on the *validation* set that maps
the model's score `s` to `p = 1 / (1 + exp(-(a·logit(s) + b)))`. Two numbers per model.

**Code.** `src/serving/calibration.py` (`fit_platt`, `brier`, `ece`, `reliability`),
`scripts/10_calibrate.py`.

**Run.**
```bash
.venv/bin/python scripts/10_calibrate.py --mode full --set data.session_sample_fraction=0.1
```

**Read the results** (`results/tables/calibration_full.csv`):
* **ECE** (expected calibration error): average gap between predicted and actual buyer
  rate across 10 bins. Ours fell from ~0.30 to ~0.01: "30 % off" became "1 % off".
* **Brier score**: mean squared error of the probability (lower is better), ~0.18 → ~0.08.
* The dashboard's **Live scoring** page now shows "Purchase probability".

**Try yourself.** Compare Platt with *isotonic regression* (`sklearn.isotonic.IsotonicRegression`).

---

## Step 11: The AI analyst assistant — LLM + RAG (local only)

**Problem.** A language model on its own invents numbers ("hallucinates"). We want it to
answer questions about *our* project using *our* results only.

**Idea: Retrieval-Augmented Generation (RAG).**
1. **Knowledge base** (`src/llm/corpus.py`): the report, README, model/data cards are cut at
   headings into passages; every result table is turned into short "fact sheets". ~124 chunks.
2. **Retrieval** (`src/llm/retrieval.py`): for a question, find the 5 most relevant passages.
   * **BM25**: classic keyword search (counts matching words, rewards rare words).
   * **Dense**: embed question and passages with `nomic-embed-text`; rank by cosine similarity
     (finds passages with the same *meaning* even with different words).
   * **Hybrid**: merge both rankings with *reciprocal rank fusion*: score = Σ 1/(60 + rank).
3. **Generation** (`src/llm/providers.py`, `src/llm/rag.py`): the passages are pasted into the
   prompt as [S1]…[S5] and the local model (Llama 3.1 8B) is told to answer only from them
   and cite every claim.
4. **Checks** (`src/llm/rag.py`): every [Sn] must exist, and every number in the answer must
   appear in a source (allowing rounding and % forms). Failures are flagged in the app.

**Run the evaluation** (Llama vs Qwen on the same questions):
```bash
.venv/bin/python scripts/11_rag_eval.py --mode full --set data.session_sample_fraction=0.1 --models llama3.1:8b qwen3:14b
```

**Read the results** (`rag_retrieval_full.csv`, `rag_answer_summary_full.csv`):
* **recall@5**: share of questions where a correct passage is in the top 5 (BM25: 0.95).
* **MRR**: 1 / rank of the first correct passage, averaged (1.0 = always first).
* **accuracy**: answers containing the known correct value (Llama: 85 %).
* **abstention**: for unanswerable questions, did it say "I don't know"? (100 %).
* **numeric grounding**: share of numbers in answers found in the sources (100 %).
* **seconds**: time per answer.

**Try yourself.** Edit `config/rag_eval_questions.yaml` to add questions; change the number
of sources with `--k 8`; pull another model (`ollama pull mistral`) and add it to `--models`.

---

## Step 12: Conformal prediction and stacking

### 12a. Conformal prediction — "how sure is the model?"

**Problem.** A single probability does not say how reliable it is.

**Idea: Mondrian (class-conditional) split conformal prediction.** On the validation set,
for each class separately, find the score cut-off so that 90 % of that class's visits would
be covered. On new visits the output is a *set*: {buyer}, {non-buyer}, or {buyer, non-buyer}
("unsure"). Guarantee: the true class is in the set ≥ 90 % of the time, **with no assumptions
about the model**.

**Code.** `src/evaluation/conformal.py`; script `scripts/12_conformal_stacking.py`.

**Read** (`conformal_full.csv`): *coverage* should match the target (95/90/80 %);
*share_confident_buyer* etc. tell how many visits get a single-label decision. Ours: at 90 %,
43 % of visits get a confident answer, and "confident buyers" convert at 34 % vs 10 % overall.

### 12b. Stacking — combining models

**Idea.** Feed each model's score (as a logit) into a small logistic regression that learns
how much to trust each model. Trained on validation with blocked time-ordered CV to choose its
regularisation `C`. Compared with simple averaging and with the best single model.

**Read** (`stacking_full.csv`, `stacking_tests_full.csv`): the deep-learning stack
(MLP+TabNet+GRU) gains +0.002 PR-AUC, significant after Holm correction; the ML stack gains
nothing. `error_overlap_full.csv` shows how often two models make the *same* mistakes —
stacking helps only when errors differ.

**Run.**
```bash
.venv/bin/python scripts/12_conformal_stacking.py --mode full --set data.session_sample_fraction=0.1
```

---

## Step 13: Concept drift and retraining

**Problem.** Shopping behaviour changes (Black Friday, Christmas, January sales). A model
trained once slowly gets worse.

**Idea.** Replay the seven months as a real deployment: for each month from the 4th on, score
it with a model trained under four policies — `frozen` (first 3 months only), `expanding`
(all earlier months), `window_3` (last 3 months), `window_1` (last month). Compare each with
`frozen` by paired bootstrap.

**Code.** `scripts/13_drift_retraining.py` (`training_months` decides the months).

**Read** (`retraining_full.csv`): every retraining policy beats `frozen` from February;
`window_3` gains ~+0.014 PR-AUC in March and April. Conclusion for the report: retrain
monthly on a rolling 3-month window.

**Run.**
```bash
.venv/bin/python scripts/13_drift_retraining.py --mode full --set data.session_sample_fraction=0.1
```

---

## Step 14: Learned embeddings as features

**Problem.** The model sees a product only as an ID or category; it does not know that two
phones are similar.

**Ideas.**
* **item2vec** (`src/features/embeddings.py`): count how often two products are viewed one
  after another in a session (before the validation period only, so no leakage), turn the
  counts into PPMI (how much more often than chance), and compress with SVD to 16 numbers per
  product. Similar products get similar vectors — the same idea as word2vec for words.
* **Text embeddings**: the category path and brand ("electronics smartphone · apple") are
  embedded with `nomic-embed-text` via Ollama and reduced with PCA to 8 numbers.
* **Session features**: the mean vector of the visited products, the share with a vector,
  and *coherence* (cosine similarity between consecutive products: focused or wandering?).

**Run.**
```bash
.venv/bin/python scripts/14_embeddings.py --mode full --set data.session_sample_fraction=0.1
```

**Read** (`embeddings_ablation_full.csv`, `embeddings_tests_full.csv`): LightGBM with base
features vs +item2vec vs +text vs +both, 3 seeds, paired bootstrap against base.
Earlier run: item2vec +0.0037 at K=3 and +0.0019 at K=5, both significant.

---

## Step 15: Hyperparameter tuning with Optuna

**Problem.** The models used hand-picked settings (learning rate, tree depth…). Were they good?

**Idea.** Optuna's **TPE** (Tree-structured Parzen Estimator) is a Bayesian search: it tries
settings, models which regions give high validation PR-AUC, and samples more there. Trial 0
is always the original settings, so tuning can never do worse on validation. Afterwards
**fANOVA** estimates which hyperparameters mattered most.

**Code.** `scripts/15_tune.py` (`space()` defines the search ranges for LightGBM, MLP, GRU).

**Run** (long: 40 + 15 + 10 trials):
```bash
.venv/bin/python scripts/15_tune.py --mode full --set data.session_sample_fraction=0.1
```

**Read** (`tuning_full.csv`, `tuning_trials_full.csv`, `tuning_importance_full.csv`): default
vs tuned test PR-AUC with a significance test; which parameters drove the gain. Important:
tuning uses **validation only**; the test weeks are touched once at the end.

---

## Step 16: Transformer sequence model

**Problem.** The GRU reads clicks one by one; a Transformer lets every click look at every
other click at once (self-attention), the architecture behind modern LLMs.

**Idea** (`src/models/sequence_models.py`, `TransformerModel`): each click becomes a vector
(event type, category, price…) plus a learned position vector; a special **[CLS]** token is
added in front; 2 attention layers × 4 heads; the [CLS] output predicts purchase. The
attention weights of [CLS] show **which click positions the model looks at**.

**Run.**
```bash
.venv/bin/python scripts/16_transformer.py --mode full --set data.session_sample_fraction=0.1
```

**Read** (`transformer_full.csv`, `transformer_tests_full.csv`, `transformer_attention_full.csv`):
Transformer vs GRU PR-AUC over 3 seeds with a paired test, p95 latency per visit, and the
average attention per click position (usually the last clicks get the most attention).

Settings are in `config/config.yaml` under `models.transformer` (d_model, n_heads, n_layers…).
Try `--set models.transformer.n_layers=1` and compare.

---

## Checking progress of the long runs

```bash
.venv/bin/python scripts/progress.py --watch 60
```

Shows each long step (16, 15, 14, 11) as running / queued / done with a progress bar,
refreshed every 60 s (Ctrl+C to stop; it only reads logs, so stopping it stops nothing).
The full output of each step is in `results/runs/<step>.log`, e.g.
`tail -f results/runs/tuning.log` to watch Optuna trial by trial.

## Where to see everything

```bash
.venv/bin/streamlit run app/main.py
```

Then open http://localhost:8501:
* **Models → Advanced ML**: tabs for conformal, stacking, drift, embeddings, tuning, Transformer.
* **AI → Analyst assistant**: ask questions; pick Llama or Qwen; see sources and checks.
* **Engine → Live scoring**: purchase probability and "Explain in plain English".

## How everything is tested

```bash
.venv/bin/python -m pytest -q
```
```bash
.venv/bin/ruff check src app scripts tests
```

Tests run every script on a small synthetic dataset, so they finish in minutes and need no
real data or Ollama (the language model is replaced by a fake one in `tests/test_llm.py`).

## Suggested order to study the code

1. `src/serving/calibration.py` (short, pure maths)
2. `src/llm/retrieval.py` → `src/llm/rag.py` → `src/llm/providers.py`
3. `src/evaluation/conformal.py` and `src/evaluation/stacking.py`
4. `scripts/13_drift_retraining.py`
5. `src/features/embeddings.py`
6. `scripts/15_tune.py`
7. `TransformerModel` in `src/models/sequence_models.py`
