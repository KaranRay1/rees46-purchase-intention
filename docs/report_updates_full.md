# Report updates: seven-month results (three seeds)

Generated for the report rewrite. It compares the numbers in `docs/report.md` (single seed, before the category fix; tables at commit `10042ff`) with the current tables in `results/tables/*_full.*`. **Bold** = changed at the precision the report uses. Nothing here edits the report.

## Why the numbers moved

1. **Three seeds.** Every learned model is now trained with seeds 42, 7 and 2026 and the tables report the mean (and standard deviation). The report quoted one seed.
2. **Category fix.** From 1 Dec 2019 REES46 switched to a new category tree with wrong labels (e.g. iPhones as `construction.tools.light`). Products now keep their Oct/Nov label, so the category features of Dec–Apr visits changed for every model.
3. The data, sessions, splits and the rule baseline are unchanged (the rule does not use categories), so differences are model effects, not data effects.

## Hypothesis verdicts

| Id | Hypothesis | Report said | Now | Evidence now |
|---|---|---|---|---|
| H1a | Tree ensembles beat logistic regression | partly supported | partly supported | significant wins 4/6 |
| H1b | Deep tabular models beat logistic regression | partly supported | partly supported | significant wins 3/6 |
| H1c | GRU matches the best aggregated model at K=10 | not supported | **supported** | **vs random_forest: diff -0.0010, CI [-0.0041, +0.0020], p_holm 1** |
| H2 | Performance rises with K, with diminishing returns | not supported | not supported | rising 0/7, diminishing 0/7 (PR-AUC lift) |
| H3 | Behavioural ratios (G3) add PR-AUC on top of G1+G2 | not supported | not supported | significant gains 0/6 |
| H3-variant | Ratios can replace the cart block (G1+G3 vs G1+G2) | exploratory | exploratory | variant significantly worse in 6/6 comparisons |
| H4 | Tree models dominate the accuracy–latency Pareto front | not supported | not supported | trees dominate at 0/3 K |
| H5 | Cart signals (G2) rank first in SHAP importance | partly supported | partly supported | **G2 top in 13/15 (model, K)** |

**H1c flips to supported.** At K=10 the GRU is now 0.0010 below the random forest with a CI that includes zero (Holm p = 1), so §4's H1c paragraph and conclusion ("the random forest leads narrowly") need rewriting: at K=10 the GRU and random forest are statistically tied.

## §4 RQ1 table: test PR-AUC (lift)

| Team | Model | K=3 | K=5 | K=10 |
|---|---|---|---|---|
| ML | Rule baseline | 0.246 (2.32×) | 0.214 (2.11×) | 0.176 (1.80×) |
| ML | Logistic regression | 0.336 → **0.340** (3.17× → **3.21×**) | 0.323 → **0.325** (3.18× → **3.20×**) | 0.290 → **0.291** (2.97× → **2.98×**) |
| ML | Random forest | 0.354 → **0.356** (3.34× → **3.35×**) ± 0.0001 | 0.336 → **0.337** (3.31×) ± 0.0002 | 0.294 → **0.296** (3.01× → **3.03×**) ± 0.0004 |
| ML | LightGBM | 0.351 → **0.353** (3.31× → **3.33×**) ± 0.0000 | 0.333 (3.27× → **3.28×**) ± 0.0002 | 0.292 (2.99×) ± 0.0020 |
| DL | MLP | 0.352 → **0.353** (3.32× → **3.33×**) ± 0.0001 | 0.332 → **0.333** (3.27×) ± 0.0001 | 0.289 → **0.291** (2.96× → **2.98×**) ± 0.0003 |
| DL | TabNet | 0.343 → **0.346** (3.23× → **3.27×**) ± 0.0052 | 0.326 (3.21×) ± 0.0023 | 0.289 → **0.288** (2.96× → **2.95×**) ± 0.0031 |
| DL | GRU (sequence) | 0.349 → **0.353** (3.29× → **3.33×**) ± 0.0016 | 0.334 → **0.336** (3.28× → **3.30×**) ± 0.0006 | 0.289 → **0.296** (2.96× → **3.03×**) ± 0.0002 |

- ROC-AUC of the learned models: 0.750–0.762 (report: 0.745–0.761); rule: 0.655–0.686.
- Seed-to-seed standard deviation of PR-AUC is at most 0.0052 (TabNet), far smaller than the bootstrap CIs.

## §4 Machine learning vs deep learning (best of each team)

| K | Best ML | Best DL | Difference | Significant? | Report said |
|---|---|---|---:|---|---|
| 3 | Random forest 0.356 | MLP 0.353 | +0.002 | yes (p = 0.00) | Random forest vs MLP, yes |
| 5 | Random forest 0.337 | GRU (sequence) 0.336 | +0.001 | no (p = 0.36) | Random forest vs GRU (sequence), no |
| 10 | Random forest 0.296 | GRU (sequence) 0.296 | +0.000 | no (p = 1.00) | Random forest vs GRU (sequence), yes |

## §10 Engine model (fastest of the models tied with the best)

| K | Report said | Now |
|---|---|---|
| 3 | Random forest | **GRU (sequence)** |
| 5 | GRU (sequence) | GRU (sequence) |
| 10 | Logistic regression | Logistic regression |

## §5 RQ2: PR-AUC lift by K

| Model | K=3 | K=5 | K=10 |
|---|---|---|---|
| Rule baseline | 2.32× | 2.11× | 1.80× |
| Logistic regression | 3.17× → **3.21×** | 3.18× → **3.20×** | 2.97× → **2.98×** |
| Random forest | 3.34× → **3.35×** | 3.31× | 3.01× → **3.03×** |
| LightGBM | 3.31× → **3.33×** | 3.27× → **3.28×** | 2.99× |
| MLP | 3.32× → **3.33×** | 3.27× | 2.96× → **2.98×** |
| TabNet | 3.23× → **3.27×** | 3.21× | 2.96× → **2.95×** |
| GRU (sequence) | 3.29× → **3.33×** | 3.28× → **3.30×** | 2.96× → **3.03×** |

## §6 RQ3: feature-group ladder (PR-AUC)

| K | Model | Step | Report said | Now |
|---|---|---|---|---|
| 3 | LightGBM | S1:G1 | 0.218 | **0.219** |
| 3 | LightGBM | S4:G1+G2+G3+G4 | 0.343 | **0.346** |
| 3 | LightGBM | S5:G1+G2+G3+G4+G5 | 0.351 | **0.353** |
| 3 | MLP | S1:G1 | 0.219 | **0.220** |
| 3 | MLP | S3:G1+G2+G3 | 0.335 | **0.334** |
| 3 | MLP | S4:G1+G2+G3+G4 | 0.343 | **0.346** |
| 3 | MLP | S5:G1+G2+G3+G4+G5 | 0.352 | **0.353** |
| 5 | LightGBM | S1:G1 | 0.226 | **0.225** |
| 5 | LightGBM | V:ratios_instead_of_counts | 0.291 | **0.292** |
| 5 | LightGBM | S4:G1+G2+G3+G4 | 0.325 | **0.329** |
| 5 | MLP | S2:G1+G2 | 0.319 | **0.318** |
| 5 | MLP | S3:G1+G2+G3 | 0.315 | **0.316** |
| 5 | MLP | S4:G1+G2+G3+G4 | 0.325 | **0.328** |
| 5 | MLP | S5:G1+G2+G3+G4+G5 | 0.332 | **0.333** |
| 10 | LightGBM | S2:G1+G2 | 0.276 | **0.277** |
| 10 | LightGBM | S3:G1+G2+G3 | 0.274 | **0.275** |
| 10 | LightGBM | S4:G1+G2+G3+G4 | 0.284 | **0.288** |
| 10 | MLP | S1:G1 | 0.198 | **0.197** |
| 10 | MLP | S2:G1+G2 | 0.273 | **0.274** |
| 10 | MLP | S3:G1+G2+G3 | 0.272 | **0.273** |
| 10 | MLP | S4:G1+G2+G3+G4 | 0.284 | **0.286** |
| 10 | MLP | S5:G1+G2+G3+G4+G5 | 0.289 | **0.291** |

Rows not listed are unchanged at three decimals.

## §7 RQ4: latency per visit (p95, batch of 1, ms)

| Model | K=3 | K=5 | K=10 |
|---|---|---|---|
| Rule baseline | 0.001 | 0.001 | 0.001 |
| Logistic regression | 0.032 → **0.033** | 0.032 | 0.030 |
| Random forest | 4.995 → **6.804** | 5.040 → **5.412** | 5.010 → **5.216** |
| LightGBM | 0.145 → **0.157** | 0.155 → **0.158** | 0.131 → **0.144** |
| MLP | 0.059 → **0.065** | 0.060 → **0.080** | 0.058 → **0.060** |
| TabNet | 0.950 → **1.098** | 0.964 → **0.944** | 0.965 → **1.085** |
| GRU (sequence) | 0.074 → **0.076** | 0.086 → **0.087** | 0.125 → **0.130** |

Latency is re-measured on every run, so small changes are timing noise; the conclusion (every model well inside a 10 ms budget) is what matters.

## §8 RQ5: which feature group each model relies on most

H5 (cart signals rank first): 12/15 → **13/15** (model, K) pairs. Only these changed:

| K | Model | Top group: report | Now | Cart share now |
|---|---|---|---|---|
| 3 | TabNet | G5_price_time | G5_price_time | 5% → **33%** |
| 5 | TabNet | G5_price_time | **G2_cart_signals** | 28% → **39%** |
| 10 | TabNet | G5_price_time | **G1_activity** | 35% → **16%** |

## New material the report does not cover yet

- **Calibration** (`calibration_full.csv`, `scripts/10_calibrate.py`): Platt scaling fitted on the validation weeks. On the test period the learned models' expected calibration error falls from 0.30–0.34 to 0.007–0.015 and the Brier score from 0.175–0.203 to 0.079–0.082. PR-AUC is unchanged (the map is monotone).
- **Monitoring** (`monitoring_*_full.*`, `scripts/07_monitor.py`): weekly PR-AUC of every model on the 10 validation and test weeks; K=3: GRU 0.27–0.47, K=5: GRU 0.25–0.43, K=10: GRU 0.22–0.38. No input drifts beyond PSI 0.10 after training.
- **Category fix**, **campaign planner** (suppression of high-intent visitors), **REST API**, **Docker** and **model/data cards** (`docs/model_card_full.md`, `docs/data_card_full.md`) for §10.
