# Research Blueprint (active) — coursework version, no MLOps

**Title (as submitted):** Real-Time Session-Based E-Commerce Purchase Intention Prediction:
A Comparative Analysis of Tabular Machine Learning and Deep Neural Architectures
**Authors:** Karan Ray, Merve Kara — NCI MSc AI, Machine Learning module.

> Scope: the **whole joint project** in one codebase (Merve's tabular half + Karan's deep half).
> MLOps version archived at `docs/archive/blueprint_with_mlops.md` (product phase, later).
> Proposal review: `docs/proposals_analysis.md`.

---

## 1. Why no MLOps here

Karan's original RQ2 measured an MLflow + FastAPI + Docker pipeline. The supervisor asked why MLOps
is needed. It isn't, for the research questions: it answers "how do we serve this?", not
"what works, how early, and why?". RQ2 is therefore replaced by an **offline efficiency study** (RQ4 below).

Reproducibility without MLflow:
- one `config/config.yaml`, fixed seeds (3 per configuration);
- each run writes `results/runs/<run_id>/` → `config_snapshot.yaml`, `metrics.json`, `predictions.parquet`;
- all runs appended to `results/tables/all_runs.csv`.

"Real-time" = a prediction is available right after the K-th event; we report per-session latency (p50/p95).

---

## 2. Research questions

**Main RQ:** How accurately, how early, how efficiently and on what behavioural evidence can a
purchase be predicted from the first K events of a session — and do deep neural architectures
improve on tabular machine learning for this task?

| ID | Sub-question | Source | Experiment |
|----|--------------|--------|------------|
| RQ1 | **Model families.** Do tree ensembles (RF, LightGBM) and deep models (MLP, TabNet, GRU) achieve significantly higher positive-class F1 and PR-AUC than rule-based and logistic-regression baselines — and how do trees and deep models compare with each other? | H1 of both proposals | All models × K × 3 seeds; paired bootstrap / McNemar |
| RQ2 | **How early?** How does performance change with prefix length K ∈ {3, 5, 10}? | both (ablation section) | Same grid, plotted vs K |
| RQ3 | **Which features?** What does each feature group add — especially behavioural ratios? | Merve (ratios) + Karan (feature ladder) | Ablation ladder G1→G5 on LightGBM + MLP |
| RQ4 | **Efficiency / real-time feasibility.** What is the accuracy vs latency, throughput, memory and size trade-off? | replaces Karan's MLOps RQ2 | In-process benchmarks, fixed hardware |
| RQ5 | **Explainability.** Which behaviours drive predictions, do model families rely on the same signals, and what characterises false positives/negatives? | Merve RQ2 | TreeSHAP (RF, LGBM), KernelSHAP sample (MLP), TabNet masks; SHAP on error groups |

**Hypotheses**
- H1a: RF / LightGBM with behavioural ratios > LR baseline on F1 and PR-AUC (Merve H1).
- H1b: MLP / TabNet > LR baseline on F1 and PR-AUC (Karan H1).
- H1c (exploratory): GRU over the raw event sequence ≥ best aggregated-feature model at K = 10.
- H2: performance rises with K, with diminishing gains from 5 → 10.
- H3: adding behavioural ratios gives a significant PR-AUC gain over counts only.
- H4: tree models dominate the accuracy–latency Pareto front on CPU.
- H5: cart-related features rank highest in SHAP across all model families.

---

## 3. Models

| Family | Model | Input | Notes |
|--------|-------|-------|-------|
| Baseline | Rule: `cart_count > 0` | aggregated | no training |
| Baseline | Logistic Regression | aggregated, scaled + one-hot top categories | class weights |
| Tree | Random Forest | aggregated | balanced weights, tuned depth / min_samples_split |
| Tree | LightGBM | aggregated | early stopping on val |
| Deep tabular | MLP | aggregated, scaled | Dense → ReLU → BatchNorm → Dropout ×2–3, PyTorch |
| Deep tabular | TabNet | aggregated | `pytorch-tabnet`, sequential attention masks |
| Deep sequence | GRU (extension) | raw K-event sequence | embeddings(event_type, category) + scaled price & time-gap → 1-layer GRU(64) → sigmoid |

Fairness: same K, same split, same seeds, same CPU for every model.
Imbalance: class weights in training + decision threshold tuned on validation (max F1), applied unchanged to test.

---

## 4. Data & leakage rules

- REES46 multi-category logs, Polars lazy scans → Parquet. Develop on one month; final runs on several months.
- Per `user_session`, sort by `event_time`; keep sessions with > K events; features from events 1..K only.
- `target = 1` if a purchase occurs **after** event K; drop sessions with a purchase inside 1..K.
  → purchase/view and purchase/cart ratios are always 0 and are **not used**.
- Chronological 70/15/15 split by session start time. Scalers / encoders / vocabularies fitted on train only.
- `product_id` used only through derived counts (and GRU embedding if needed), never one-hot.

## 5. Feature groups (ablation ladder, RQ3)

| Group | Adds |
|-------|------|
| G1 activity | view_count, unique products, duration, longest inter-event gap |
| G2 + cart signals | cart_count, remove_count, position of first cart, last event type |
| G3 + behavioural ratios | cart/view, remove/cart (ε = 1e-6) |
| G4 + category hierarchy | top-level / sub category of most-viewed item, unique categories & brands |
| G5 + price & time context | price mean/max/sum/std, hour, weekday |

## 6. Evaluation

- Primary: positive-class Precision, Recall, F1, PR-AUC. Secondary: Macro-F1, ROC-AUC.
- Uncertainty: 3 seeds; bootstrap 95% CIs; paired bootstrap on PR-AUC, McNemar on labels.
- Robustness: per-month test sets; cohorts by session length.
- Error analysis: FP/FN by session length, early cart appearance, item variety, time window, + SHAP.
- Efficiency: training time, latency per session (single, batch; p50/p95), throughput, peak RAM, model size.

## 7. Literature

See `docs/proposals_analysis.md §4` (refs [1]–[10] from the proposals) plus TabNet (Arik & Pfister 2021),
Grinsztajn et al. 2022, Shwartz-Ziv & Armon 2022, Saito & Rehmsmeier 2015.

## 8. Phases

1. Ingestion, sessionisation, leakage tests, temporal split
2. EDA
3. Feature groups + sequence tensors
4. Baselines → RF / LightGBM → MLP / TabNet → GRU
5. Evaluation, statistics, efficiency
6. SHAP + error analysis
7. Report: tables and figures generated from `results/`
