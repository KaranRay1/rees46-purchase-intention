# Proposal Analysis — Karan Ray & Merve Kara (submitted 12/10/2026)

Shared title: *Real-Time Session-Based E-Commerce Purchase Intention Prediction:
A Comparative Analysis of Tabular Machine Learning and Deep Neural Architectures*
Module: Machine Learning (NCI, lecturer Prashanth Nayak). Dataset: REES46.

## 1. What each proposal commits to

| Item | Karan (deep-learning half) | Merve (tabular half) |
|------|----------------------------|----------------------|
| Core models | **MLP** (dense + ReLU + BatchNorm + Dropout), **TabNet** | **Random Forest**, **LightGBM** |
| Baselines | Rule-based (cart presence), Logistic Regression | same |
| Features | Session interaction features, normalised | **Behavioural ratios** (cart/view, remove/cart) |
| RQ1 / H1 | DNN beats linear baselines on positive-class F1 + PR-AUC (seeds + stats tests) | Trees + ratios beat linear baselines on F1 + PR-AUC |
| RQ2 / H2 | Latency/throughput of **MLflow + FastAPI + Docker** pipeline | **SHAP**: global importance + local decision paths |
| Explainability | — | TreeSHAP: summary, dependence, force plots; SHAP on FP/FN |
| Ablation | Feature groups: past activity → + cart signals → + category hierarchy, at K = 3, 5, 10 | Baseline vs + behavioural ratios at K = 3, 5, 10 |
| Robustness | Monthly periods, session cohorts, CIs | same |
| Error analysis | FP/FN by session length, early cart, item variety, time window | FP/FN via SHAP |
| Efficiency | Train time, latency (p95), req/s, memory, storage | same (copied) |
| Imbalance | Class weights + validation-tuned threshold (preferred) | same |

Shared, identical sections: dataset/leakage rules, temporal split, train-only scalers,
metrics (P/R/F1/PR-AUC primary; Macro-F1/ROC-AUC secondary), limitations.

## 2. What changes for the coursework (research-only version)

- **Karan's RQ2 (MLOps) is replaced** — the supervisor asked why MLOps is needed.
  It becomes an *offline* efficiency question: latency (mean, p95), throughput (sessions/s),
  memory, model size, measured in-process on fixed hardware. No FastAPI / Docker / MLflow.
  MLOps goes to the later product phase (`docs/archive/blueprint_with_mlops.md`).
- **The two halves merge naturally into one comparison**: baselines → trees → deep tabular models,
  on the same prefixes, splits and seeds. That is exactly what the shared title promises.
- **SHAP (Merve) can also explain Karan's models** for a cross-family comparison
  (TreeSHAP for trees; DeepSHAP/KernelSHAP on a sample for MLP; TabNet's own attention masks).
- **Both ablation designs are kept**, as one feature-group ladder:
  G1 activity counts → G2 + cart signals → G3 + behavioural ratios → G4 + category hierarchy → G5 + price/time.

## 3. Issues found in the proposals (fix in the final report)

1. **Copy-paste leftovers in Merve's paper**: evaluation section talks about "neural models" and
   "FastAPI endpoints"; limitations mention "deep learning and MLOps"; abstract says "clinical" level.
2. **Both papers have two `Index Terms` lines** (one for each half).
3. **Karan's methodology repeats the class-weighting sentence twice** and contradicts itself on which is "standard" vs "preferred".
4. **Purchase-based ratios are always 0** under the leakage rule (purchases inside the prefix are removed),
   so only cart/view and remove/cart ratios are valid. Merve's proposal already uses only these two. ✔
5. **"Real-time" is never defined.** Define it as: prediction available after the K-th event,
   with per-session inference latency reported (p50/p95).
6. **Neither proposal mentions a sequence model.** The GRU added earlier is *not* in either proposal
   (see decision below).
7. Hypotheses say "significantly higher" — name the test: paired bootstrap on PR-AUC / McNemar on predictions, 3–5 seeds.
8. Table I lists `product_id` as a feature — high-cardinality ID; use it only via derived counts
   (unique products) or embeddings, not one-hot.

## 4. Literature already cited (reuse in `docs/literature_review.md`)

[1] Sakar et al. 2019 (MLP + LSTM) · [2] Satu & Islam 2023 · [3] Esmeli & Gokce 2025 (post-cart XAI) ·
[4] Hidasi et al. 2016 (GRU4Rec) · [5] Han et al. 2024 (END4Rec) · [6] Setiawan et al. 2026 (RF + MLOps) ·
[7] Breiman 2001 · [8] Ke et al. 2017 (LightGBM) · [9] Lundberg & Lee 2017 (SHAP) · [10] REES46 dataset.
To add: Arik & Pfister 2021 (TabNet), Grinsztajn et al. 2022, Shwartz-Ziv & Armon 2022, Saito & Rehmsmeier 2015.

## 5. Ideas parked for the product phase (not coursework)

FastAPI scoring endpoint + Docker, MLflow tracking/registry, CI/CD, streaming
(score after each event, not just at K), intervention layer (discount / recommendation triggers),
drift monitoring across months.
