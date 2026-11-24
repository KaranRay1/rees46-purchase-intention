# Report outline: evidence map (notes, not text)

> **Read first.** The NCI brief forbids AI-written submission content. This file is a **map of
> evidence**: for each required section, what the brief asks for, the points the evidence
> supports, and where each number lives. Write every sentence yourself. Take numbers from
> `facts.md` or `results/tables/`, never from memory or from `docs/report.md` (an older,
> AI-drafted results summary that must not be submitted).

**Format (brief, section 4):** IEEE conference template, **5–7 double-column pages** including
figures and tables. References and the cover sheet are excluded. More than 7 pages: 5-point
penalty. Name and student number on the front page. Distribution of work inside the report.
AI-use declaration on the cover sheet (see `ai_use_log.md`).

**Suggested page budget (7 pages):** Abstract + Intro 0.7 · Related work 0.8 · Methods 0.6 ·
Methodology 1.0 · Results 2.8 · Limitations/ethics 0.5 · Conclusion 0.3 · Reproducibility +
work split 0.3. Space allows about **4 figures + 2 tables**: choose from the shortlist below.

| Asset | Shows | Suggested use |
|---|---|---|
| `figures/fig1_rq1_model_comparison` (full width) | PR-AUC with CI, all models, K = 3/5/10 | **main result (keep)** |
| `tables/tab_main_k5` | accuracy + efficiency at K=5 | **main table (keep)** |
| `figures/fig5_extensions_forest` | what each extension adds | **keep**: compresses 5 experiments into one figure |
| `figures/fig4_rq4_accuracy_vs_latency` | accuracy vs latency trade-off | keep, or replace with numbers in tab_main |
| `figures/fig3_rq3_feature_ladder` | feature-group ablation | ablation requirement |
| `figures/fig6_segments_lift` | segment robustness/fairness | robustness/bias requirement |
| `figures/fig2_rq2_lift_by_k` | lift vs K | optional: can be one sentence + fig1 |
| `figures/fig7_agent_vs_rag` | LLM agent vs RAG | optional (secondary contribution) |
| `tables/tab_data_splits` | split sizes, dates, buyer rates | data requirement (could be text) |
| `tables/tab_extensions_k5` | numbers behind fig5 | appendix-style; pick fig5 **or** this |
| `tables/tab_full_scale` | 10% vs 100% sessions | generalisation requirement |
| `tables/tab_hypotheses` | hypothesis verdicts | compact way to answer H1–H5 |

---

## Abstract (100–150 words)
Brief asks for: problem, research question, data, methods, principal findings, most important limitation.
- Problem: purchase intent from the first K clicks of an anonymous session (real-time).
- Data: REES46 multi-category store, 7 months (Oct 2019–Apr 2020); 10% session sample; K ∈ {3, 5, 10}.
- Methods: rule baseline; LR, RF, LightGBM; MLP, TabNet, GRU; 3 seeds; paired bootstrap + Holm.
- Findings to pick from: ML ≈ DL (top models within ~0.005 PR-AUC; tie RF/GRU at 10%, GRU slightly
  ahead on 100%); cart signals dominate; user history largest gain (+0.055 at K=5); GRU ~60× faster
  than RF at p95 → chosen engine.
- Limitation: single dataset/shop; 10% sample (checked on 100%); profit figures rest on assumptions.
- Numbers: facts.md → *RQ1*, *Extensions*, *RQ4*, *Robustness*.

## I. Introduction
Brief: context and importance, research question, hypotheses/objectives, contribution.
- Context points: most sessions anonymous and short; decision must be made while the visitor is still
  on site; ~10% of eligible sessions buy (class imbalance → PR-AUC).
- Research question: from `docs/research_blueprint.md` (main RQ + RQ1–RQ5). ≤ 50 words in the proposal;
  keep the same RQ, or explain the change.
- Hypotheses: H1a–H5 → `tables/tab_hypotheses` (verdicts already computed, Holm-corrected).
- Contribution candidates (choose 2–4): controlled ML vs DL comparison under one leakage-safe pipeline;
  early-prediction curve over K; efficiency study replacing MLOps (supervisor feedback); feature/user-
  history ablation; robustness (100% data, drift replay, segments); local LLM agent as an analyst
  interface (optional).
- **Changes since the proposal must be stated** (brief, last paragraph): MLOps RQ replaced by an offline
  efficiency study (blueprint §1); proposals of both team members merged; extensions added (steps 10–20);
  LLM part runs locally (no paid API).

## II. Related work
Brief: critical discussion, gap, relation to previous work. Rubric: 10% of the mark.
Papers already in the proposal ([1]–[6] in `references.bib`) + candidates to read and verify:
- Session purchase prediction: Sakar et al. 2019 [1] (UCI dataset, MLP/LSTM, *session-level aggregate*
  data, not early prefixes); Requena et al. 2020 (clickstream intent with minimal browsing; Coveo data);
  Esmeli & Gokce 2025 [3] (post-cart XAI on REES46-type data); Satu & Islam 2023 [2] (feature engineering).
- Sequence models: Hidasi et al. GRU4Rec [4] (next-item, not purchase); END4Rec [5]; Transformer (Vaswani).
- Trees vs DL on tabular data: Grinsztajn et al. 2022; Shwartz-Ziv & Armon 2022; TabNet (Arik & Pfister).
- Evaluation on imbalance: Saito & Rehmsmeier 2015 (PR vs ROC); Dietterich 1998 (paired tests); Holm 1979.
- Pipeline/MLOps: Setiawan et al. 2026 [6] (RF + MLOps → why the study instead measures latency offline).
- Gap points (to argue in your own words): earlier work scores whole sessions (after the fact), rarely
  tests *how early*; ML vs DL comparisons often lack seeds/significance tests/leakage control; efficiency
  rarely measured alongside accuracy.

## III. Methods and baselines
Brief: justify methods; strengths, limitations, computational needs; simple + conventional ML + neural.
Rubric "choice of methods" rewards complex methods with a well-thought-out rationale.
- Simple baseline: rule (cart in prefix) → `src/models/` (rule_based). PR-AUC K=5 in facts.md.
- Conventional ML: LR (linear reference), RF, LightGBM (Merve Kara).
- Neural: MLP, TabNet (tabular DL), GRU over the raw event sequence (Karan Ray).
- "Additional method / intervention" required by the brief: user history (G6), embeddings, tuning,
  stacking, calibration, conformal prediction, Transformer, LLM agent: pick what you discuss.
- Compute needs: train time, size, latency → `tables/tab_main_k5`, facts.md *RQ4*.

## IV. Methodology and experimental setup
Brief: data, splits, preprocessing, features, configs, hyperparameters, seeds, procedure, leakage.
Rubric "Methodology" = 30% (largest weight): every stage of CRISP-DM/KDD, every decision justified.
- Data + cleaning counts → facts.md *Data*; `docs/data_card_full.md`; `config/config.yaml`.
- Sessionising + eligibility per K (too short / bought within prefix / eligible) → facts.md *Data*.
- Leakage rules: prefix-only features; sessions with a purchase inside the prefix dropped; target =
  purchase after event K; chronological split with boundary purge → `src/data/sessionize.py` docstring,
  `src/data/split.py`; tests in `tests/test_leakage.py`.
- Split: chronological 70/15/15, dates and buyer rates → `tables/tab_data_splits`.
- Features G1–G5 (18 features) → `config/config.yaml → features.groups`; G6 user history → `src/features/user_history.py`.
- Preprocessing fitted on train only (scaling, top-N categories) → `src/features/preprocessing.py`.
- Hyperparameters: defaults in `config/config.yaml`; tuning on validation only (Optuna, step 15);
  decision threshold tuned on validation. **Test set never used for selection** (brief requirement).
- Seeds 42 / 7 / 2026; 10% session sample (memory) and the 100% check.
- Statistics: paired bootstrap (1000 resamples), McNemar, Holm correction; PR-AUC primary (imbalance).

## V. Results and evaluation
Rubric "Evaluation" = 20%: diverse situations, parametrisations, sampling.
- **A. Main comparison (RQ1):** fig1 + tab_main_k5. Key facts: ranking per K; which pairs tie after
  Holm (`rq1_pairwise_tests`); rule baseline far below.
- **B. How early (RQ2):** lift falls as K rises (H2 not supported) → fig2 or one sentence. Note that
  the eligible population changes with K (facts.md *Data*), so this is not the same sessions.
- **C. Efficiency (RQ4):** latency/size/train time/throughput → tab_main_k5 / fig4; GRU vs RF p95.
- **D. Tuning procedure:** Optuna trials per model, val-only selection, test gain → facts.md *Extensions*.
- **E. Ablation/sensitivity (required):** feature ladder fig3 (G2 cart block is the big step; G3 ratios
  add nothing: H3); SHAP group shares (`rq5_importance_groups_k5`); user history G6 + 1 h/24 h gap check;
  business sensitivity grid (optional).
- **F. Robustness/generalisation (required):** 100% vs 10% (tab_full_scale); monthly drift replay
  (`retraining_tests`: few significant differences); segments (fig6, `segment_disparity`); 3 seeds.
- **G. Error analysis (required: "structured manual"):** see *Manual error analysis* below;
  plus automated: `rq5_error_profile_k5`, `rq5_metrics_by_session_length_k5`, `error_overlap`.
- **H. (Optional) LLM agent vs RAG:** fig7, facts.md *LLM agent*.

### Manual error analysis (to do by hand)
`error_analysis_sheet.csv` holds 48 real test sessions (K=5), 12 per group: both models miss a buyer
(FN), both flag a non-buyer (FP), only GRU right, only RF right. `group_size` shows how common each
group is in the whole test set. For each row, read the clicks and fill in:
- `error_type`: your own category (e.g. "browse-only buyer", "cart abandoner", "price comparison",
  "single-product revisit", "unknown category", …). Define the categories yourself after reading ~10 rows.
- `likely_cause` and `notes`.
Then count categories per group and compare GRU vs RF errors. That table is your structured manual
error analysis. Keep the sheet as evidence (and mention it in the reproducibility statement).

## VI. Limitations, ethics and responsible use
Brief: threats to validity, dataset limits, bias, privacy/licensing, compute limits, deployment risks.
- Validity: single shop/dataset; 10% sample (mitigated by 100% check); one seed on 100%; K changes the
  population; category labels changed in Dec 2019 (`docs/data_card_full.md`).
- Bias/fairness: no protected attributes → segment analysis only (fig6): price and category gaps;
  new vs returning users flagged at the same rate (facts.md *Segments*).
- Privacy: pseudonymous user ids; G6 user history needs tracking/consent; licence of REES46 (see dataset page).
- Responsible use: suppression campaign = some visitors get fewer discounts; profit numbers rest on
  assumptions (facts.md *Business value*); calibration and conformal sets for honest uncertainty.
- LLM: local models only; numeric grounding checks; agent still errs on document questions (fig7).
- Compute: laptop-scale (24 GB RAM); 100% run peak memory (`results/full100/pipeline.log`).

## VII. Conclusion and future work
Brief: answer the RQ, contribution, limitations, realistic future work. Rubric: anchor in literature.
- Answer per RQ from `tables/tab_hypotheses`.
- Future-work candidates (realistic): second dataset; per-event (streaming) scoring; A/B test of the
  campaign uplift; consent-aware user history; hybrid router (RAG vs agent).

## Reproducibility statement
Brief: code, data access, environment, configs, seeds, procedure.
- Code: zip from `scripts/make_share_zip.sh`; GitHub repository (private).
- Data access: `scripts/00_download_data.py` (public REES46 files) + Kaggle page (footnote).
- Environment: Python 3.12, `requirements.txt`; CPU only; macOS note for LightGBM.
- Config: `config/config.yaml`; seeds 42/7/2026; 10% sample flag.
- Procedure: `HOW_TO_RUN.md` (steps 00–21); tests (`pytest`, 320 tests); synthetic quick run.

## Distribution of work (required)
- Starting point: `docs/report.md` §0 table and the dashboard team pages (Merve: rule, LR, RF, LightGBM,
  RQ3 ratios, RQ5 SHAP; Karan: MLP, TabNet, GRU, RQ3 ladder, RQ4 efficiency, extensions). Check with
  Merve and state it in your own words.
