# Q&A preparation: likely questions and where the evidence is

Prepare your own answers; this lists the questions examiners tend to ask about a project like
this and points to the evidence. Practise answering each in under 30 seconds.

## Method and validity
| Likely question | Evidence to look at |
|---|---|
| How did you prevent data leakage? | `src/data/sessionize.py` docstring (prefix-only features, purchase-in-prefix dropped, target after K); `src/data/split.py` (chronological, boundary purge); `tests/test_leakage.py` |
| Why PR-AUC and not accuracy/ROC-AUC? | buyer rate ≈ 10% (facts.md *Data*); Saito & Rehmsmeier 2015 in `references.bib` |
| Was the test set used for tuning or thresholds? | No: Optuna, thresholds and calibration all use validation (`scripts/15_tune.py`, `scripts/10_calibrate.py`) |
| Why only a 10% sample? Do results hold on all data? | memory (24 GB); `results/full100/tables/scale_comparison_k5.csv`, facts.md *Robustness* |
| How do you know differences are real? | 3 seeds, paired bootstrap (1000), McNemar, Holm (`src/evaluation/statistics.py`); `rq1_pairwise_tests` |
| Why does performance fall as K grows (H2 not supported)? | the eligible population changes with K (facts.md *Data*: eligible counts and buyer rates per K); harder sessions remain |
| Why did you drop the MLOps question from the proposal? | `docs/research_blueprint.md` §1 (supervisor feedback → offline efficiency study) |

## Results
| Likely question | Evidence |
|---|---|
| So is deep learning better or not? | fig1; RF/GRU tie at 10%, GRU slightly ahead at 100%; Grinsztajn 2022 / Shwartz-Ziv 2022 for context |
| Why choose the GRU as the engine? | p95 latency and size (`rq4_efficiency`); tied accuracy; `src/serving/recommend.py` rule |
| What matters most for prediction? | ablation ladder (`rq3_ablation`), SHAP group shares (`rq5_importance_groups_k5`) |
| Isn't the user-history gain leakage? | only sessions ended before the visit started; 1 h / 24 h gap check (`user_history_gap_check_full.csv`); `tests/test_user_history.py` |
| Why did TabNet do worse? | `rq1_model_comparison`, training time in `rq4_efficiency`; discuss tuning budget |
| What errors do the models make? | your labelled `error_analysis_sheet.csv`; `rq5_error_profile_k5`; `error_overlap` (models miss the same buyers) |

## Ethics and deployment
| Likely question | Evidence |
|---|---|
| Is the model fair? | no protected attributes in the data; segment analysis (`segment_disparity`, fig6) |
| Privacy of user history? | pseudonymous ids; needs consent/tracking in practice (outline §VI) |
| Would you deploy it? What could go wrong? | drift replay (`retraining_tests`), calibration, conformal sets; suppression means some visitors get fewer discounts; profit rests on assumed uplift |
| Did you use AI tools? | cover-sheet declaration (`../report_pack/ai_use_log.md`): be ready to explain what you did yourself |

## Know your own code
Examiners may ask either of you to explain any part of the code. Be ready to walk through, in
your own words: the sessionising/leakage logic, one ML model and one DL model of your own, the
paired bootstrap, and how a live prediction is made (`src/serving/predictor.py`).
`docs/learning_guide.md` explains each step for revision.
