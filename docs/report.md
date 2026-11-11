# Real-Time Session-Based E-Commerce Purchase Intention Prediction

**A Comparative Analysis of Tabular Machine Learning and Deep Neural Architectures**

Karan Ray and Merve Kara, MSc in Artificial Intelligence, National College of Ireland, Machine Learning module.

> **Draft results report: October 2019 development run (`--mode dev`).**
> Every number below comes from the files in `results/tables/` (suffix `_dev`) produced by
> `scripts/01`–`06`. The figures are in `results/figures/`, and the executed notebooks are in
> `results/notebooks/`. Please check, rewrite and extend this text before you submit it, and
> declare any AI assistance as NCI's academic-integrity policy requires.

---

## 1. Data

**Source.** The REES46 multi-category store event log (eCommerce behavior data from a
multi-category store). This run uses **October 2019** only (`data/raw/2019-Oct.csv`).
November 2019 to April 2020 are downloaded but not used yet; see §8.

**Cleaning** (`data_summary_dev.json`):

| Step | Rows removed |
|---|---:|
| Over-long sessions | 104,147 |
| Price ≤ 0 or missing | 68,673 |
| Exact duplicate rows | 30,217 |
| Missing session id | 2 |
| **Rows in → rows out** | **42,448,764 → 42,245,725** |

This leaves 9,222,333 sessions, of which 629,343 contain a purchase. The log holds only
`view`, `cart` and `purchase` events. **No `remove_from_cart` events occur**, so the two
removal features in the original design were dropped (see §2.2).

**Prediction task.** At each prefix length K ∈ {3, 5, 10} we see only the first K events of a
session and predict whether a purchase happens *later in the same session*. To prevent
leakage, a session is eligible only if it has more than K events and no purchase within its
first K events.

| K | Eligible sessions | Purchase after event K | Base rate (all) | Base rate (test) |
|---|---:|---:|---:|---:|
| 3 | 3,239,990 | 310,417 | 9.58 % | 8.85 % |
| 5 | 2,059,825 | 177,715 | 8.63 % | 7.89 % |
| 10 | 873,245 | 68,438 | 7.84 % | 7.21 % |

**Split.** The split is chronological 70 / 15 / 15 by session start, with cut-offs shared by
all K: validation starts 2019-10-22 08:26:45 and test starts 2019-10-26 19:59:30. Sessions
that cross a cut-off are purged (1,760 at K=3, 1,419 at K=5 and 881 at K=10). The test month-end
base rate is about 0.8 points below train at every K, which is mild drift.

## 2. Method

### 2.1 Models

The comparison covers seven models:

- **Baselines:** a rule ("buyer if the prefix contains a cart event", which gives a binary
  score) and logistic regression (LR).
- **Tree ensembles:** random forest (RF) and LightGBM.
- **Deep tabular models:** an MLP and TabNet.
- **Sequence model:** a GRU over the raw event sequence of the prefix.

Each learned model is trained with 3 seeds (42, 7, 2026), with early stopping on the
validation split where the model supports it.

### 2.2 Features

There are 18 session-prefix features in five groups. They are built from the prefix only.

| Group | Features |
|---|---|
| G1 activity | unique_products_count, session_duration_sec, max_gap_sec |
| G2 cart signals | view_count, cart_count, has_cart, first_cart_position, last_event_type |
| G3 behavioural ratio | cart_to_view_ratio |
| G4 category hierarchy | unique_categories_count, unique_brands_count, main_category |
| G5 price and time | price_mean, price_max, price_sum, price_std, hour_of_day, day_of_week |

Because the data has no removals and K is fixed, view_count = K − cart_count, and
cart_to_view_ratio is a deterministic function of cart_count. This matters for H3 (§5).

### 2.3 Evaluation

- **Primary metric:** test PR-AUC (average precision). We also report it as **lift** over the
  test base rate, because the base rate differs by K. ROC-AUC, positive-class F1, precision and
  recall are secondary metrics.
- **Uncertainty:** 1,000-resample bootstrap 95 % CIs on the primary seed.
- **Significance:** paired bootstrap on PR-AUC and McNemar's test at the decision threshold. The
  pre-registered hypotheses form a confirmatory family (13 tests for RQ1/RQ2) that is
  Holm-corrected at α = 0.05. All other comparisons are exploratory.
- **Efficiency:** in-process latency (batch 1, 256 and 4,096), throughput, peak memory and
  model size. These are measured single-threaded on one machine (Apple arm64, 15 cores,
  25.8 GB RAM, Python 3.12). The real-time budget is p95 single-session latency ≤ 10 ms, which
  is an assumption.
- **Explainability:** run on 2,000 test sessions per K. Methods are TreeSHAP (RF, LightGBM),
  LinearSHAP (LR), DeepSHAP (MLP) and TabNet attention masks. Attributions are summed into
  feature-group shares.

## 3. RQ1: Model families

Test PR-AUC (mean of 3 seeds) with its bootstrap 95 % CI, and lift over the base rate
(`rq1_model_comparison_dev.csv`):

| Model | K=3 | K=5 | K=10 |
|---|---|---|---|
| Rule baseline | 0.198 (2.23×) | 0.163 (2.06×) | 0.124 (1.72×) |
| Logistic regression | 0.290 (3.28×) | 0.260 (3.29×) | 0.223 (3.09×) |
| Random forest | 0.313 (3.54×) | 0.280 (3.54×) | 0.232 (3.22×) |
| **LightGBM** | **0.315 (3.56×)** | **0.280 (3.55×)** | 0.232 (3.22×) |
| MLP | 0.313 (3.53×) | 0.278 (3.53×) | **0.233 (3.23×)** |
| TabNet | 0.304 (3.43×) | 0.273 (3.46×) | 0.228 (3.16×) |
| GRU (sequence) | 0.302 (3.42×) | 0.270 (3.41×) | 0.226 (3.14×) |

- CIs are narrow: about ±0.005 at K=3 and ±0.009 at K=10. Seed-to-seed spread is below 0.001,
  except TabNet (0.003–0.008).
- ROC-AUC is between 0.70 and 0.72 for every learned model, and 0.58–0.61 for the rule.
- At the default threshold, positive-class F1 differs little between models (K=3: 0.334–0.336),
  so PR-AUC is the more informative comparison.

**Hypotheses** (`rq1_rq2_hypotheses_dev.json`):

- **H1a, trees beat LR: supported.** The difference is significant in 6/6 (model, K) pairs:
  +0.023 to +0.025 at K=3, about +0.020 at K=5 and about +0.009 at K=10.
- **H1b, deep tabular models beat LR: supported.** The difference is significant in 6/6 pairs:
  MLP +0.023 and TabNet +0.021 at K=3, and smaller gains at larger K.
- **H1c, the GRU matches the best aggregated model at K=10: not supported.** The GRU is 0.0077
  PR-AUC below the MLP, with Holm p < 0.001.

**Trees vs deep models** (exploratory, `rq1_pairwise_tests_dev.csv`):

- LightGBM, RF and the MLP are practically tied. The differences are at most 0.002, and most are
  not significant after Holm correction.
- LightGBM's small lead over the MLP is significant at K=3 (+0.0019) and K=5 (+0.0020), but not
  at K=10, where the MLP is 0.0007 ahead (p = 1.0).
- TabNet and the GRU trail the leaders slightly but consistently.

**Conclusion.** Every learned model clearly beats the rule baseline. Non-linear models beat LR
by 1–2.5 PR-AUC points. Among the non-linear models, the family matters far less than the
features.

Figures: `rq1_pr_curves_k{3,5,10}_dev.png`, `res_*_dev.png`.

## 4. RQ2: How early can we predict?

The table shows PR-AUC lift by K (`rq2_by_k_dev.csv`):

| Model | K=3 | K=5 | K=10 |
|---|---:|---:|---:|
| LightGBM | 3.56 | 3.55 | 3.22 |
| MLP | 3.53 | 3.53 | 3.23 |
| GRU | 3.42 | 3.41 | 3.14 |
| LR | 3.28 | 3.29 | 3.09 |

**H2, performance rises with K with diminishing returns: not supported** (0/7 models).

- Lift is flat from K=3 to K=5 and *falls* at K=10. Absolute PR-AUC falls at every step.
- K is not the only thing that changes. Each K has a different eligible population: longer
  sessions, and only those without a purchase in the first K events. So a later prediction is
  a prediction on a harder subset.
- In practical terms, three events are enough to rank sessions about as well as ten, and the
  prediction is available much earlier.

Figure: `rq2_metrics_vs_k_dev.png`.

## 5. RQ3: Which features matter?

The ablation ladder adds feature groups G1→G5 in turn, for LightGBM and the MLP, with 3 seeds
per K (`rq3_ablation_dev.csv`, `rq3_step_tests_dev.csv`). The table shows the PR-AUC gain from
each step for LightGBM; the MLP is within ±0.002.

| Step | K=3 | K=5 | K=10 |
|---|---:|---:|---:|
| S1: G1 activity only | 0.156 | 0.159 | 0.141 |
| + G2 cart signals | **+0.134** | **+0.099** | **+0.066** |
| + G3 cart/view ratio | +0.000 | −0.000 | −0.001 |
| + G4 category | +0.010 | +0.009 | +0.012 |
| + G5 price and time | +0.015 | +0.014 | +0.015 |
| Full model (S5) | 0.315 | 0.280 | 0.232 |

**H3, behavioural ratios add PR-AUC: partly supported** (1/6 significant).

- Only the MLP at K=3 gains significantly, and only by +0.0007 (Holm p = 0.036).
- At K=5 the MLP *loses* 0.0009 (p = 0.04).
- The other four steps are not significant.

This result was expected: with no removals and fixed K, the ratio is a function of cart_count,
which G2 already contains.

The variant "ratios instead of counts" (G1+G3 vs G1+G2) tests whether the ratio can *replace*
the cart block:

- It recovers most of the cart signal. LightGBM gains +0.129 over S1 at K=3.
- It is still significantly below the full G2 block: −0.005 at K=3, −0.016 at K=5 and −0.018
  at K=10.
- The reason is that G2 also carries has_cart, first_cart_position and last_event_type.

**Conclusion.** Cart signals are by far the most valuable group. Category and price/time
features each add a smaller but significant 1–1.5 points.

Figure: `rq3_ablation_dev.png`.

## 6. RQ4: Efficiency and real-time feasibility

Single-thread benchmarks (`rq4_efficiency_dev.csv`), at K=5:

| Model | PR-AUC | p95 latency, batch 1 (ms) | Throughput, batch 4,096 (sessions/s) | Size (MB) |
|---|---:|---:|---:|---:|
| Rule | 0.163 | 0.001 | 1.6 × 10⁹ | < 0.01 |
| LR | 0.260 | 0.03 | 3.8 × 10⁷ | < 0.01 |
| MLP | 0.278 | 0.06 | 3.7 × 10⁶ | 0.09 |
| GRU | 0.270 | 0.09 | 4.2 × 10⁵ | 0.07 |
| LightGBM | 0.280 | 0.16 | 6.5 × 10⁴ | 1.9 |
| TabNet | 0.276 | 1.04 | 1.9 × 10⁵ | 0.83 |
| Random forest | 0.279 | 5.19 | 5.4 × 10⁴ | 102 |

- **Every model meets the 10 ms budget at every K.**
- Random forest is the slowest and largest model by about two orders of magnitude, and it buys
  no accuracy over LightGBM.

**H4, trees dominate the accuracy–latency Pareto front: not supported** (0/3 K;
`rq4_hypothesis_dev.json`).

- The non-baseline Pareto front is {LightGBM, MLP} at K=3 and K=5, and {MLP} alone at K=10.
- The MLP is about 2.7× faster than LightGBM per session (p95 of 0.06 vs 0.16 ms at K=5), at
  nearly the same PR-AUC.

**Caveat.** Training times in the RQ1 table are indicative only. Some step-03 runs shared the
machine with other jobs, and runs were resumed after an interruption. The latency benchmark ran
alone.

Figure: `rq4_pareto_dev.png`.

## 7. RQ5: Explainability and errors

**Group attribution shares** (`rq5_importance_groups_k*_dev.csv`). Mean |attribution| is summed
per group and normalised to 1. The table shows the mean share across LR, RF, LightGBM, MLP and
TabNet:

| Group | K=3 | K=5 | K=10 |
|---|---:|---:|---:|
| G5 price and time | 0.40 | 0.39 | 0.39 |
| G4 category | 0.24 | 0.23 | 0.22 |
| G2 cart signals | 0.20 | 0.23 | 0.21 |
| G1 activity | 0.08 | 0.08 | 0.12 |
| G3 ratio | 0.08 | 0.06 | 0.06 |

**H5, cart signals rank highest for every model family: partly supported** (2/15
`rq5_hypothesis_dev.json`). G2 ranks first only for the random forest, at K=3 and K=5. For every
other model and K, G5 ranks first.

**Reading this together with RQ3:**

- G5 receives the largest attribution share, yet adding it gains only about 0.015 PR-AUC.
  Adding G2 gains 0.07–0.13.
- Attribution measures how much a model *uses* a group, not how much the group *adds*. G5 has 6
  features, and 37 one-hot inputs for the MLP, so its shares accumulate.
- Price features are also partly substitutes for the cart signals they co-vary with. The
  ablation is the better guide to value; SHAP describes reliance.

**Rank agreement.** Group rankings agree almost perfectly among LR, LightGBM, the MLP and the RF
(Spearman ρ = 0.9–1.0 at K=10). TabNet's attention ranking disagrees (ρ = 0.3–0.4): it puts
much more weight on the ratio (0.19–0.29) and almost none on G2 at K=3 and K=5. TabNet masks are
not SHAP values, so this comparison is qualitative.

**Errors** (`rq5_error_profile_*`, `rq5_metrics_by_session_length_*`):

- False negatives are buyers *without a cart event in the prefix*. For TabNet at K=10, only
  0.9 % have a cart event, against 78 % of true positives. They view more distinct products.
- False positives look like true positives, carted sessions that did not convert. With prefix
  features alone they are hard to separate.
- PR-AUC is highest for short sessions and falls for sessions of 20–50 events. The ≥ 50-event
  cohorts are too small in the 2,000-session sample to interpret (n = 20–56).
- At K=5, **17,570 of 24,440 test buyers (72 %) are missed by every model** at its decision
  threshold, including the GRU and the rule (`rq5_explain_summary_dev.json`).
  - None of these sessions has a cart event in its first five events.
  - Their profile is almost the same as non-buyers' (3.4 vs 3.6 distinct products; median
    duration 191 vs 194 s).
  - The 28 % of buyers that are caught had carted in 83 % of cases.

  This is a limit of the information in a short prefix, not of any model family. It is also why
  recall stays low and PR-AUC is capped near 0.3.

Figures: `rq5_*_dev.png` and `xai_*_dev.png`.

## 8. Hypothesis scoreboard

| Hypothesis | Verdict (October dev run) |
|---|---|
| H1a: trees > LR | Supported (6/6) |
| H1b: deep tabular > LR | Supported (6/6) |
| H1c: GRU ≈ best aggregated at K=10 | Not supported (−0.008, p < 0.001) |
| H2: performance rises with K, diminishing | Not supported (0/7) |
| H3: ratios add PR-AUC | Partly supported (1/6, +0.0007) |
| H4: trees dominate the Pareto front | Not supported (0/3; the MLP is on the front) |
| H5: cart signals rank highest everywhere | Partly supported (2/15) |
| H3 variant: ratios replace the cart block | Exploratory: significantly worse in 6/6 |

The full table is in `results/tables/hypothesis_scoreboard_dev.csv`, written by notebooks 03 and
04.

## 9. Threats to validity and limitations

1. **One month.** All results are from October 2019, so they reflect one month of one store.
   The test week is the last week of October, and its base rate is 0.8 points lower than
   training.
2. **Later months differ.** Spot checks of November 2019 to April 2020 show category codes that
   look broken from December 2019 (for example, an Apple product filed under
   `construction.tools.light`). They also show a sharp rise in cart rates. Results for G4 and for
   any multi-month run must be re-checked before they are generalised. A full run will also need
   `data.session_sample_fraction` because of memory (step 01 on October alone peaked at 14.3 GB).
3. **No removal events.** The original removal features could not be used, and G3 shrinks to
   one ratio that is determined by cart_count. H3 is therefore a weak test here.
4. **K changes the population.** Comparisons across K (RQ2) mix the effect of more events with
   the effect of a different, harder subset of sessions.
5. **Threshold metrics.** F1, precision and recall depend on the decision threshold. PR-AUC is
   the primary basis for the conclusions.
6. **Explainability sample.** The explanations use 2,000 test sessions per K. Some
   session-length cohorts are very small. Attribution shares depend on group size and on the
   method, and TabNet attention is not comparable to SHAP.
7. **Hardware.** Latency was measured on one machine, single-threaded, in-process. It excludes
   feature computation and network or serving overhead. The 10 ms budget is an assumption.
8. **Tuning.** Hyper-parameters are fixed in `config/config.yaml` rather than searched per
   model, which may favour models whose defaults suit this data.

## 10. Reproducing

See `README.md` for the run order and settings. In short:

1. `scripts/01_prepare_data.py --mode dev`
2. `scripts/02_build_features.py --mode dev`
3. `scripts/03`–`06`
4. The notebooks, executed with `NB_MODE=dev`.
