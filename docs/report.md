# Real-Time Session-Based E-Commerce Purchase Intention Prediction

**A Comparative Analysis of Tabular Machine Learning and Deep Neural Architectures**

Karan Ray and Merve Kara, MSc in Artificial Intelligence, National College of Ireland, Machine Learning module.

> **Draft results report: seven-month run (`--mode full`, October 2019 – April 2020),
> with the October 2019 run (`--mode dev`) as a replication check.**
> Every number below comes from the files in `results/tables/` (suffix `_full`, or `_dev`
> where stated) produced by `scripts/01`–`06`. The figures are in `results/figures/`, and the
> executed notebooks are in `results/notebooks/`. Please check, rewrite and extend this text
> before you submit it, and declare any AI assistance as NCI's academic-integrity policy
> requires.

---

## 0. Who did what

The project joins two proposals into one codebase and one comparison
(`docs/research_blueprint.md`, `docs/proposals_analysis.md`):

| | Merve Kara: machine learning | Karan Ray: deep learning |
|---|---|---|
| Models | Rule baseline, logistic regression, random forest, LightGBM | MLP, TabNet, GRU over the raw event sequence |
| Research questions led | RQ3 behavioural ratios (H3), RQ5 explainability (SHAP) | RQ3 feature ladder, RQ4 efficiency (replaces the original MLOps question) |
| Shared | Data pipeline, RQ1/RQ2 comparison, statistics, dashboard | |

The dashboard keeps this split: one page per team, followed by a page that compares the two
teams' models directly (§10).

## 1. Research question

Can the first few clicks of an online-shop visit tell us whether the visitor will buy later
in the same visit, quickly enough to act while the visit is still live? And does classic
machine learning or deep learning answer that better?

- **RQ1:** Which model family predicts best?
- **RQ2:** How early can we predict, i.e. after how many events K?
- **RQ3:** Which feature groups matter?
- **RQ4:** Is scoring fast enough for real time, and what does each model cost?
- **RQ5:** What drives the predictions, and which buyers are missed?

## 2. Data

**Source.** The REES46 multi-category store event log ("eCommerce behavior data from a
multi-category store"), all seven months from **October 2019 to April 2020**
(`data/raw/2019-Oct.csv` … `2020-Apr.csv`).

**Sampling.** The full log holds 411.7 M events in 89.7 M sessions. To run on one laptop, step
01 keeps a deterministic, hash-based **10 % sample of whole sessions**
(`data.session_sample_fraction=0.1`). Whole sessions are kept or dropped together, so no
session is cut. The sample (8.87 M sessions) is about the size of October alone.

**Cleaning** (`data_summary_full.json`):

| Step | Rows removed |
|---|---:|
| Session sampling (90 % of sessions) | 370,524,468 |
| Over-long sessions | 867,105 |
| Exact duplicate rows | 135,438 |
| Price ≤ 0 or missing | 70,943 |
| Missing session id | 226 |
| **Rows in → rows out** | **411,709,736 → 40,111,556** |

This leaves 8,871,684 sessions, of which 527,121 (5.9 %) contain a purchase. The log holds
only `view`, `cart` and `purchase` events. **No `remove_from_cart` events occur**, so the two
removal features in the original design were dropped (see §3.2).

**Months** (`monthly_profile_full.csv`). Sessions per month in the sample range from 0.92 M
(October) to 1.54 M (December). The share of sessions with a purchase moves between 4.8 %
(January 2020) and 6.8 % (March and April 2020), so the months differ noticeably.

**Prediction task.** At each prefix length K ∈ {3, 5, 10} we see only the first K events of a
session and predict whether a purchase happens *later in the same session*. To prevent
leakage, a session is eligible only if it has more than K events and no purchase within its
first K events.

| K | Eligible sessions | Purchase after event K | Base rate (all) | Base rate (test) |
|---|---:|---:|---:|---:|
| 3 | 2,920,763 | 306,389 | 10.49 % | 10.61 % |
| 5 | 1,895,715 | 182,700 | 9.64 % | 10.16 % |
| 10 | 838,235 | 73,134 | 8.72 % | 9.77 % |

**Split.** The split is chronological 70 / 15 / 15 by session start, with cut-offs shared by
all K:

- **Train:** 1 October 2019 – 1 March 2020.
- **Validation:** from 2020-03-01 02:58 (March 2020).
- **Test:** from 2020-04-04 07:44 to the end of April 2020.

Sessions that cross a cut-off are purged (257 at K=3, 211 at K=5, 132 at K=10). The base rate
drifts upwards over time: at K=5 it is 9.0 % in train, 11.9 % in validation and 10.2 % in
test. The models are therefore tested on a later, different period, which is the realistic
use case.

## 3. Method

### 3.1 Models

| Team | Model | Input |
|---|---|---|
| Machine learning (Merve) | Rule: "buyer if the prefix contains a cart event" (binary score) | prefix |
| | Logistic regression (LR) | 18 features |
| | Random forest (RF) | 18 features |
| | LightGBM | 18 features |
| Deep learning (Karan) | MLP | 18 features |
| | TabNet | 18 features |
| | GRU | raw event sequence of the prefix |

The seven-month run uses the primary seed (42). In the October run, three seeds (42, 7, 2026)
moved PR-AUC by less than 0.001 for every model except TabNet (0.003–0.008), so one seed is
enough for the comparisons below. Early stopping on the validation split is used where the
model supports it. Hyper-parameters are fixed in `config/config.yaml`.

### 3.2 Features

There are 18 session-prefix features in five groups. They are built from the prefix only.

| Group | Features |
|---|---|
| G1 activity | unique_products_count, session_duration_sec, max_gap_sec |
| G2 cart signals | view_count, cart_count, has_cart, first_cart_position, last_event_type |
| G3 behavioural ratio | cart_to_view_ratio |
| G4 category hierarchy | unique_categories_count, unique_brands_count, main_category |
| G5 price and time | price_mean, price_max, price_sum, price_std, hour_of_day, day_of_week |

Because the data has no removals and K is fixed, view_count = K − cart_count, and
cart_to_view_ratio is a deterministic function of cart_count. This matters for H3 (§6).

### 3.3 Evaluation

- **Primary metric:** test PR-AUC (average precision). We also report it as **lift** over the
  test base rate, because the base rate differs by K. ROC-AUC, positive-class F1, precision and
  recall are secondary metrics.
- **Uncertainty:** 1,000-resample bootstrap 95 % CIs.
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

## 4. RQ1: Model families

Test PR-AUC with lift over the base rate (`rq1_model_comparison_full.csv`). Bootstrap 95 % CIs
are about ±0.005 at K=3 and ±0.008 at K=10.

| Team | Model | K=3 | K=5 | K=10 |
|---|---|---|---|---|
| ML | Rule baseline | 0.246 (2.32×) | 0.214 (2.11×) | 0.176 (1.80×) |
| ML | Logistic regression | 0.336 (3.17×) | 0.323 (3.18×) | 0.290 (2.97×) |
| ML | **Random forest** | **0.354 (3.34×)** | **0.336 (3.31×)** | **0.294 (3.01×)** |
| ML | LightGBM | 0.351 (3.31×) | 0.333 (3.27×) | 0.292 (2.99×) |
| DL | MLP | 0.352 (3.32×) | 0.332 (3.27×) | 0.289 (2.96×) |
| DL | TabNet | 0.343 (3.23×) | 0.326 (3.21×) | 0.289 (2.96×) |
| DL | GRU (sequence) | 0.349 (3.29×) | 0.334 (3.28×) | 0.289 (2.96×) |

- ROC-AUC is between 0.745 and 0.761 for every learned model, and 0.655–0.686 for the rule.
- At the tuned threshold, positive-class F1 differs little between models (K=3: 0.432–0.433),
  so PR-AUC is the more informative comparison.

**Hypotheses** (`rq1_rq2_hypotheses_full.json`):

- **H1a, trees beat LR: partly supported (4/6).** At K=3, RF is +0.018 and LightGBM +0.015
  ahead. At K=5, RF is +0.013 and LightGBM +0.010 ahead. All four are significant. At K=10 the
  gains shrink to +0.004 (RF, Holm p = 0.12) and +0.001 (LightGBM, p = 1.0).
- **H1b, deep tabular models beat LR: partly supported (3/6).** At K=3, the MLP is +0.015 and
  TabNet +0.007 ahead. At K=5, the MLP is +0.010 ahead. All three are significant. TabNet's
  +0.003 at K=5 is not (p = 0.06). At K=10 both are 0.001 *below* LR (p = 1.0).
- **H1c, the GRU matches the best aggregated model at K=10: not supported.** The GRU is 0.005
  PR-AUC below the random forest (CI −0.008 to −0.003, Holm p = 0.012).

**Machine learning vs deep learning** (exploratory, `rq1_pairwise_tests_full.csv`). This
compares each team's best model:

| K | Best ML | Best DL | Difference | Significant? |
|---|---|---|---:|---|
| 3 | RF 0.354 | MLP 0.352 | +0.002 | yes |
| 5 | RF 0.336 | GRU 0.334 | +0.002 | no (p = 0.052) |
| 10 | RF 0.294 | GRU 0.289 | +0.005 | yes |

- LightGBM and the MLP are tied at every K (p ≥ 0.07).
- TabNet trails RF, LightGBM, the MLP and the GRU at K=3 and K=5 (by 0.006–0.011), and ties
  with them at K=10.

**Conclusion.** Every learned model clearly beats the rule baseline, by 0.09–0.12 PR-AUC.
Non-linear models beat LR when the prefix is short (K=3 and 5), but at K=10 LR catches up.
Between the two teams the gap is at most 0.005 PR-AUC. The random forest leads narrowly, but
LightGBM, the MLP and the GRU are practically equivalent. **The model family matters far
less than the features.**

Figures: `rq1_pr_curves_k{3,5,10}_full.png`, `res_rq1_*_full.png`.

## 5. RQ2: How early can we predict?

The table shows PR-AUC lift by K (`rq2_by_k_full.csv`):

| Model | K=3 | K=5 | K=10 |
|---|---:|---:|---:|
| Random forest | 3.34 | 3.31 | 3.01 |
| LightGBM | 3.31 | 3.27 | 2.99 |
| MLP | 3.32 | 3.27 | 2.96 |
| GRU | 3.29 | 3.28 | 2.96 |
| LR | 3.17 | 3.18 | 2.97 |

**H2, performance rises with K with diminishing returns: not supported** (0/7 models).

- Lift is flat from K=3 to K=5 and *falls* at K=10, and absolute PR-AUC falls at every step.
  This is the same pattern as in October.
- K is not the only thing that changes. Each K has a different eligible population: longer
  sessions, and only those without a purchase in the first K events. So a later prediction is
  a prediction on a harder subset.
- In practical terms, three events are enough to rank sessions about as well as ten, and the
  prediction is available much earlier.

Figure: `rq2_metrics_vs_k_full.png`.

## 6. RQ3: Which features matter?

The ablation ladder adds feature groups G1→G5 in turn, for LightGBM and the MLP
(`rq3_ablation_full.csv`, `rq3_step_tests_full.csv`). The table shows the PR-AUC gain from
each step for LightGBM; the MLP is within ±0.005.

| Step | K=3 | K=5 | K=10 |
|---|---:|---:|---:|
| S1: G1 activity only | 0.219 | 0.226 | 0.196 |
| + G2 cart signals | **+0.116** | **+0.092** | **+0.080** |
| + G3 cart/view ratio | +0.000 | −0.000 | −0.002 |
| + G4 category | +0.008 | +0.007 | +0.010 |
| + G5 price and time | +0.009 | +0.008 | +0.007 |
| Full model (S5) | 0.351 | 0.333 | 0.292 |

All G2, G4 and G5 steps are significant.

**H3, behavioural ratios add PR-AUC: not supported** (0/6 significant gains).

- Adding the ratio never helps.
- Twice it *hurts* significantly: the MLP at K=5 (−0.004) and LightGBM at K=10 (−0.002).

This result was expected: with no removals and fixed K, the ratio is a function of cart_count,
which G2 already contains.

The variant "ratios instead of counts" (G1+G3 vs G1+G2) tests whether the ratio can *replace*
the cart block:

- It recovers much of the cart signal. LightGBM gains +0.107 over S1 at K=3.
- It is still significantly below the full G2 block: −0.009 at K=3, −0.027 at K=5 and −0.035
  at K=10.
- The reason is that G2 also carries has_cart, first_cart_position and last_event_type.

**Conclusion.** Cart signals are by far the most valuable group. Category and price/time
features each add a smaller but significant 0.7–1.0 points.

Figures: `rq3_ablation_full.png`, `res_rq3_*_full.png`.

## 7. RQ4: Efficiency and real-time feasibility

Single-thread benchmarks (`rq4_efficiency_full.csv`), at K=5:

| Team | Model | PR-AUC | p95 latency, batch 1 (ms) | Throughput, batch 4,096 (sessions/s) | Size (MB) |
|---|---|---:|---:|---:|---:|
| ML | Rule | 0.214 | 0.001 | 1.6 × 10⁹ | < 0.01 |
| ML | LR | 0.323 | 0.03 | 3.7 × 10⁷ | < 0.01 |
| DL | MLP | 0.332 | 0.06 | 2.8 × 10⁶ | 0.09 |
| DL | GRU | 0.334 | 0.09 | 4.1 × 10⁵ | 0.07 |
| ML | LightGBM | 0.333 | 0.16 | 9.6 × 10⁴ | 1.2 |
| DL | TabNet | 0.326 | 0.96 | 1.7 × 10⁵ | 0.83 |
| ML | Random forest | 0.336 | 5.04 | 5.2 × 10⁴ | 106 |

- **Every model meets the 10 ms budget at every K.**
- The random forest is the most accurate model, but it is also by far the slowest and largest.
  It takes about 5 ms per session and 84–112 MB, about 30× slower than LightGBM.

**H4, trees dominate the accuracy–latency Pareto front: not supported** (0/3 K;
`rq4_hypothesis_full.json`). The non-baseline Pareto front mixes both teams' models:

| K | Pareto front (non-baseline) |
|---|---|
| 3 | RF, MLP |
| 5 | RF, MLP, GRU |
| 10 | RF, LightGBM, MLP, GRU |

**Which model to deploy.** One practical rule is to take the fastest of the models that are
statistically tied with the most accurate one. That gives:

| K | Deployed model | Latency | Why |
|---|---|---:|---|
| 3 | Random forest | 5.0 ms | Significantly ahead of everything else |
| 5 | GRU | 0.09 ms | Tied with the RF (p = 0.052) |
| 10 | Logistic regression | 0.03 ms | Tied with the RF and LightGBM |

The dashboard uses this rule for its recommended engine.

**Caveat.** Latency covers model inference only, not feature computation or network overhead.

Figures: `rq4_pareto_full.png`, `res_rq4_*_full.png`.

## 8. RQ5: Explainability and errors

**Group attribution shares** (`rq5_importance_groups_k*_full.csv`). Mean |attribution| is
summed per group and normalised to 1. The table shows the mean share across LR, RF, LightGBM,
MLP and TabNet:

| Group | K=3 | K=5 | K=10 |
|---|---:|---:|---:|
| G2 cart signals | 0.37 | 0.42 | 0.39 |
| G5 price and time | 0.30 | 0.32 | 0.35 |
| G4 category | 0.19 | 0.16 | 0.14 |
| G1 activity | 0.08 | 0.04 | 0.06 |
| G3 ratio | 0.06 | 0.07 | 0.06 |

**H5, cart signals rank highest for every model family: partly supported** (12/15;
`rq5_hypothesis_full.json`).

- G2 ranks first for LR, RF, LightGBM and the MLP at every K.
- The three exceptions are TabNet, whose attention masks put price and time first at every K.

Unlike the October run, attribution and ablation now agree: the group the models lean on
most (G2) is also the group that adds the most PR-AUC (§6). On October alone, G5 received the
largest share.

**Rank agreement.** Group rankings agree almost perfectly among LR, LightGBM and the MLP
(Spearman ρ = 1.0 at K=10), and closely with the RF (ρ = 0.9). TabNet's attention ranking
agrees less (ρ = 0.5–0.6). TabNet masks are not SHAP values, so this comparison is
qualitative.

**Errors** (`rq5_error_profile_*_full.csv`, `rq5_explain_summary_full.json`):

- False negatives are buyers *without a cart event in the prefix*. At K=5, 0.8 % (LR) to 6 %
  (RF) of false negatives have one, against 100 % of true positives.
- False positives look like true positives: carted sessions that did not convert. With prefix
  features alone they are hard to separate.
- At K=5, **16,431 of 30,046 test buyers (55 %) are missed by every model** at its decision
  threshold, including the GRU and the rule.
  - None of these sessions has a cart event in its first five events.
  - Their profile is almost the same as non-buyers' (3.5 vs 3.6 distinct products; median
    duration 211 vs 212 s).
  - The 45 % of buyers who are caught had carted in 99.8 % of cases.

  This is a limit of the information in a short prefix, not of any model family. It is also
  why recall stays near 0.4 and PR-AUC is capped near 0.35.

Figures: `rq5_*_full.png`.

## 9. Replication: October only vs seven months

| Hypothesis | October 2019 (3 seeds) | Oct 2019 – Apr 2020 (1 seed) |
|---|---|---|
| H1a: trees > LR | Supported (6/6) | Partly supported (4/6; not at K=10) |
| H1b: deep tabular > LR | Supported (6/6) | Partly supported (3/6; not at K=10) |
| H1c: GRU ≈ best aggregated at K=10 | Not supported (−0.008 vs MLP) | Not supported (−0.005 vs RF) |
| H2: performance rises with K, diminishing | Not supported (0/7) | Not supported (0/7) |
| H3: ratios add PR-AUC | Partly supported (1/6, +0.0007) | Not supported (0/6) |
| H4: trees dominate the Pareto front | Not supported (0/3) | Not supported (0/3) |
| H5: cart signals rank highest everywhere | Partly supported (2/15) | Partly supported (12/15) |
| H3 variant: ratios replace the cart block | Exploratory: worse in 6/6 | Exploratory: worse in 6/6 |
| Best model, K=5 | LightGBM 0.280 | Random forest 0.336 |
| Buyers missed by every model, K=5 | 72 % | 55 % |

**Results that hold on both data sets:**

- Learned models beat the rule by a wide margin.
- Machine learning and deep learning end within about 0.005 PR-AUC of each other.
- Cart behaviour is the key signal, and the ratio adds nothing beyond it.
- Earlier prediction (small K) loses nothing.
- Every model is fast enough for real time.

**Results that change with more data:**

- The advantage of non-linear models over LR at K=10 disappears.
- Explanations shift towards cart signals.
- The test-period base rate is higher (10.2 % vs 7.9 % at K=5), so absolute PR-AUC values are
  higher. Lift is the comparable measure.

The full scoreboards are in `hypothesis_scoreboard_{dev,full}.csv`.

## 10. From analysis to product

The trained models are packaged behind a framework-free serving layer
(`src/serving/predictor.py`).

- **Input:** raw events for one visit.
- **Processing:** the same feature code as training (`build_features`, or the GRU's sequence
  encoder), then each model's saved preprocessor and its validation-tuned threshold.
- **Output:** a purchase probability and a decision per model, plus LightGBM TreeSHAP
  contributions.

A test checks that a test-set session scored live gets the same score the training run logged
for it (within 10⁻⁵).

The Streamlit dashboard (`app/`) presents the work in three steps:

1. **Research:** an overview, and the data with its monthly profile and time-ordered split.
2. **Analysis:**
   - one page for Merve's machine-learning models and one for Karan's deep-learning models,
     each answering the same four questions (models, accuracy, attribution, speed);
   - an ML vs DL page with the significance-tested winner per K and the recommended engine.
3. **Product:** a live demo. It scores a built or replayed visit and returns:
   - the probability;
   - an intent level (high intent, warm, or just browsing);
   - a suggested action;
   - the explanation;
   - every model's opinion.

The dashboard states only findings that are statistically supported. Because `SessionPredictor`
does not depend on Streamlit, the same scoring function can sit behind an HTTP API.

## 11. Threats to validity and limitations

1. **Sample and seeds.** The seven-month run uses 10 % of sessions and one seed. The October
   run suggests that seed variation is negligible except for TabNet. The sampling is random by
   session, but small differences (≤ 0.002) between the top models should not be read as
   rankings.
2. **A special test period.** The test period is April 2020, during the first COVID-19
   lockdowns. The validation month (March 2020) has the highest buyer rate in the data (11.9 %
   at K=5), so thresholds tuned on it may be miscalibrated for other periods.
3. **Category codes.** Spot checks from December 2019 onwards show category codes that look
   broken (for example, an Apple product filed under `construction.tools.light`). G4 results
   should be treated with care.
4. **No removal events.** The original removal features could not be used, and G3 shrinks to
   one ratio that is determined by cart_count. H3 is therefore a weak test here.
5. **K changes the population.** Comparisons across K (RQ2) mix the effect of more events with
   the effect of a different, harder subset of sessions.
6. **Threshold metrics.** F1, precision and recall depend on the decision threshold. PR-AUC is
   the primary basis for the conclusions.
7. **Explainability sample.** The explanations use 2,000 test sessions per K. Attribution
   shares depend on group size and on the method, and TabNet attention is not comparable to
   SHAP.
8. **Hardware.** Latency was measured on one machine, single-threaded, in-process. It excludes
   feature computation and network or serving overhead. The 10 ms budget is an assumption.
9. **Tuning.** Hyper-parameters are fixed in `config/config.yaml` rather than searched per
   model, which may favour models whose defaults suit this data.

## 12. Reproducing

See `README.md` for the run order and settings. In short, for the seven-month run:

1. `scripts/01_prepare_data.py --mode full --set data.session_sample_fraction=0.1`
2. `scripts/02_build_features.py`, `05_benchmark_latency.py` and `06_explain.py --ks 3 5 10`,
   each with the same `--mode full --set …`.
3. `scripts/03_run_experiments.py` and `04_run_ablation.py`, each with the same settings plus
   `--seeds 42`.
4. The notebooks, executed with `NB_MODE=full`.
5. `streamlit run app/main.py` for the dashboard.

The October run is the same with `--mode dev`, no sampling and all three seeds.
