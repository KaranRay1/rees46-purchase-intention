# Facts sheet (auto-generated)

Every number below is read from `results/tables/*_full.*` by `scripts/21_report_pack.py`. Cite figures from here or from the tables, never from memory. Rebuild after any rerun.


## Data

- Raw events / sessions (7 months): **411,709,736 / 89,693,595**  ·  `data_summary`
- After 10% session sample + cleaning: events / sessions / buying sessions: **40,111,556 / 8,871,684 / 527,121**  ·  `data_summary`
- Removed by cleaning rule 'unparseable_event_time': **0**  ·  `data_summary`
- Removed by cleaning rule 'null_session': **226**  ·  `data_summary`
- Removed by cleaning rule 'session_sampling': **370,524,468**  ·  `data_summary`
- Removed by cleaning rule 'unknown_event_type': **0**  ·  `data_summary`
- Removed by cleaning rule 'non_positive_or_null_price': **70,943**  ·  `data_summary`
- Removed by cleaning rule 'duplicate_rows': **135,438**  ·  `data_summary`
- Removed by cleaning rule 'over_long_sessions': **867,105**  ·  `data_summary`
- K=3: too short / bought within prefix / eligible / buyer rate: **5,829,395 / 121,526 / 2,920,763 / 0.1049**  ·  `data_summary`
- K=5: too short / bought within prefix / eligible / buyer rate: **6,886,493 / 89,476 / 1,895,715 / 0.0964**  ·  `data_summary`
- K=10: too short / bought within prefix / eligible / buyer rate: **7,986,929 / 46,520 / 838,235 / 0.0872**  ·  `data_summary`

## Robustness: 10% sample vs 100% of sessions (K=5)

- GRU PR-AUC 10% -> 100%: **0.3358 -> 0.3371**  ·  `full100/scale_comparison_k5`
- Random forest PR-AUC 10% -> 100%: **0.3369 -> 0.3358**  ·  `full100/scale_comparison_k5`
- LightGBM PR-AUC 10% -> 100%: **0.3331 -> 0.3321**  ·  `full100/scale_comparison_k5`
- Logistic reg. PR-AUC 10% -> 100%: **0.3250 -> 0.3258**  ·  `full100/scale_comparison_k5`
- 100%: random forest minus GRU [CI], Holm p: **-0.0013 [-0.0019, -0.0006], 0.000**  ·  `full100/rq1_pairwise_tests`

## Robustness: drift and retraining (K=5, monthly replay)

- 2020-01 window_1 vs frozen: **+0.0098 [+0.0069, +0.0126]**  ·  `retraining_tests`
- 2020-02 expanding vs frozen: **+0.0056 [+0.0047, +0.0065]**  ·  `retraining_tests`
- 2020-02 window_3 vs frozen: **+0.0036 [+0.0024, +0.0049]**  ·  `retraining_tests`
- 2020-02 window_1 vs frozen: **+0.0050 [+0.0024, +0.0079]**  ·  `retraining_tests`
- 2020-03 expanding vs frozen: **+0.0081 [+0.0070, +0.0090]**  ·  `retraining_tests`
- 2020-03 window_3 vs frozen: **+0.0140 [+0.0118, +0.0159]**  ·  `retraining_tests`
- 2020-03 window_1 vs frozen: **+0.0109 [+0.0085, +0.0130]**  ·  `retraining_tests`
- 2020-04 expanding vs frozen: **+0.0091 [+0.0082, +0.0102]**  ·  `retraining_tests`
- 2020-04 window_3 vs frozen: **+0.0145 [+0.0126, +0.0165]**  ·  `retraining_tests`
- 2020-04 window_1 vs frozen: **+0.0124 [+0.0106, +0.0146]**  ·  `retraining_tests`
- Significant month x policy differences: **10 of 12**  ·  `retraining_tests`

## Calibration (Platt, fitted on validation)

- GRU K=3 ECE raw -> calibrated: **0.3004 -> 0.0142**  ·  `calibration`
- GRU K=5 ECE raw -> calibrated: **0.3132 -> 0.0112**  ·  `calibration`
- GRU K=10 ECE raw -> calibrated: **0.3250 -> 0.0103**  ·  `calibration`

## Conformal prediction (engine model)

- K=5 target 95%: coverage / buyers' coverage / unsure share: **0.950 / 0.951 / 0.743**  ·  `conformal`
- K=5 target 90%: coverage / buyers' coverage / unsure share: **0.899 / 0.903 / 0.565**  ·  `conformal`
- K=5 target 80%: coverage / buyers' coverage / unsure share: **0.797 / 0.803 / 0.271**  ·  `conformal`

## Hypotheses

- H1a (RQ1) Tree ensembles beat logistic regression: **partly supported: significant wins 4/6**  ·  `hypothesis_scoreboard`
- H1b (RQ1) Deep tabular models beat logistic regression: **partly supported: significant wins 3/6**  ·  `hypothesis_scoreboard`
- H1c (RQ1) GRU matches the best aggregated model at K=10: **supported: vs random_forest: diff -0.0010, CI [-0.0041, +0.0020], p_holm 1**  ·  `hypothesis_scoreboard`
- H2 (RQ2) Performance rises with K, with diminishing returns: **not supported: rising 0/7, diminishing 0/7 (PR-AUC lift)**  ·  `hypothesis_scoreboard`
- H3 (RQ3) Behavioural ratios (G3) add PR-AUC on top of G1+G2: **not supported: significant gains 0/6**  ·  `hypothesis_scoreboard`
- H3-variant (RQ3) Ratios can replace the cart block (G1+G3 vs G1+G2): **exploratory: variant significantly worse in 6/6 comparisons**  ·  `hypothesis_scoreboard`
- H4 (RQ4) Tree models dominate the accuracy–latency Pareto front: **not supported: trees dominate at 0/3 K**  ·  `hypothesis_scoreboard`
- H5 (RQ5) Cart signals (G2) rank first in SHAP importance: **partly supported: G2 top in 13/15 (model, K)**  ·  `hypothesis_scoreboard`

## RQ1: model comparison (test PR-AUC, mean of 3 seeds)

- K=3 Random forest PR-AUC (sd): **0.3555 (0.0001)**  ·  `rq1_model_comparison`
- K=3 MLP PR-AUC (sd): **0.3534 (0.0001)**  ·  `rq1_model_comparison`
- K=3 GRU PR-AUC (sd): **0.3530 (0.0016)**  ·  `rq1_model_comparison`
- K=3 LightGBM PR-AUC (sd): **0.3530 (0.0000)**  ·  `rq1_model_comparison`
- K=3 TabNet PR-AUC (sd): **0.3465 (0.0052)**  ·  `rq1_model_comparison`
- K=3 Logistic reg. PR-AUC (sd): **0.3401 (0.0000)**  ·  `rq1_model_comparison`
- K=3 Rule baseline PR-AUC (sd): **0.2459 (0.0000)**  ·  `rq1_model_comparison`
- K=3 buyer rate (test): **0.1061**  ·  `rq1_model_comparison`
- K=5 Random forest PR-AUC (sd): **0.3369 (0.0002)**  ·  `rq1_model_comparison`
- K=5 GRU PR-AUC (sd): **0.3358 (0.0006)**  ·  `rq1_model_comparison`
- K=5 LightGBM PR-AUC (sd): **0.3331 (0.0002)**  ·  `rq1_model_comparison`
- K=5 MLP PR-AUC (sd): **0.3327 (0.0001)**  ·  `rq1_model_comparison`
- K=5 TabNet PR-AUC (sd): **0.3265 (0.0023)**  ·  `rq1_model_comparison`
- K=5 Logistic reg. PR-AUC (sd): **0.3250 (0.0000)**  ·  `rq1_model_comparison`
- K=5 Rule baseline PR-AUC (sd): **0.2144 (0.0000)**  ·  `rq1_model_comparison`
- K=5 buyer rate (test): **0.1016**  ·  `rq1_model_comparison`
- K=10 Random forest PR-AUC (sd): **0.2961 (0.0004)**  ·  `rq1_model_comparison`
- K=10 GRU PR-AUC (sd): **0.2957 (0.0002)**  ·  `rq1_model_comparison`
- K=10 LightGBM PR-AUC (sd): **0.2923 (0.0020)**  ·  `rq1_model_comparison`
- K=10 Logistic reg. PR-AUC (sd): **0.2913 (0.0000)**  ·  `rq1_model_comparison`
- K=10 MLP PR-AUC (sd): **0.2908 (0.0003)**  ·  `rq1_model_comparison`
- K=10 TabNet PR-AUC (sd): **0.2880 (0.0031)**  ·  `rq1_model_comparison`
- K=10 Rule baseline PR-AUC (sd): **0.1762 (0.0000)**  ·  `rq1_model_comparison`
- K=10 buyer rate (test): **0.0977**  ·  `rq1_model_comparison`

## RQ2: how early (PR-AUC lift = PR-AUC / buyer rate)

- GRU K=3 PR-AUC / lift: **0.3530 / 3.33**  ·  `rq2_by_k`
- GRU K=5 PR-AUC / lift: **0.3358 / 3.30**  ·  `rq2_by_k`
- GRU K=10 PR-AUC / lift: **0.2957 / 3.03**  ·  `rq2_by_k`
- LightGBM K=3 PR-AUC / lift: **0.3530 / 3.33**  ·  `rq2_by_k`
- LightGBM K=5 PR-AUC / lift: **0.3331 / 3.28**  ·  `rq2_by_k`
- LightGBM K=10 PR-AUC / lift: **0.2923 / 2.99**  ·  `rq2_by_k`
- Random forest K=3 PR-AUC / lift: **0.3555 / 3.35**  ·  `rq2_by_k`
- Random forest K=5 PR-AUC / lift: **0.3369 / 3.31**  ·  `rq2_by_k`
- Random forest K=10 PR-AUC / lift: **0.2961 / 3.03**  ·  `rq2_by_k`

## RQ3: feature ablation (cumulative ladder)

- K=5 LightGBM S1:G1 PR-AUC: **0.2254**  ·  `rq3_ablation`
- K=5 LightGBM S2:G1+G2 PR-AUC: **0.3180**  ·  `rq3_ablation`
- K=5 LightGBM S3:G1+G2+G3 PR-AUC: **0.3178**  ·  `rq3_ablation`
- K=5 LightGBM S4:G1+G2+G3+G4 PR-AUC: **0.3287**  ·  `rq3_ablation`
- K=5 LightGBM S5:G1+G2+G3+G4+G5 PR-AUC: **0.3331**  ·  `rq3_ablation`
- K=5 MLP S1:G1 PR-AUC: **0.2277**  ·  `rq3_ablation`
- K=5 MLP S2:G1+G2 PR-AUC: **0.3179**  ·  `rq3_ablation`
- K=5 MLP S3:G1+G2+G3 PR-AUC: **0.3160**  ·  `rq3_ablation`
- K=5 MLP S4:G1+G2+G3+G4 PR-AUC: **0.3285**  ·  `rq3_ablation`
- K=5 MLP S5:G1+G2+G3+G4+G5 PR-AUC: **0.3327**  ·  `rq3_ablation`
- K=5 LightGBM add G2 gain [CI] Holm p: **+0.0915 [+0.0875, +0.0953] p=0.000**  ·  `rq3_step_tests`
- K=5 LightGBM add G3 gain [CI] Holm p: **-0.0001 [-0.0006, +0.0003] p=1.000**  ·  `rq3_step_tests`
- K=5 LightGBM add G4 gain [CI] Holm p: **+0.0108 [+0.0087, +0.0130] p=0.000**  ·  `rq3_step_tests`
- K=5 LightGBM add G5 gain [CI] Holm p: **+0.0045 [+0.0028, +0.0063] p=0.000**  ·  `rq3_step_tests`
- K=5 MLP add G2 gain [CI] Holm p: **+0.0912 [+0.0871, +0.0950] p=0.000**  ·  `rq3_step_tests`
- K=5 MLP add G3 gain [CI] Holm p: **-0.0037 [-0.0047, -0.0027] p=0.000**  ·  `rq3_step_tests`
- K=5 MLP add G4 gain [CI] Holm p: **+0.0134 [+0.0113, +0.0153] p=0.000**  ·  `rq3_step_tests`
- K=5 MLP add G5 gain [CI] Holm p: **+0.0040 [+0.0022, +0.0060] p=0.000**  ·  `rq3_step_tests`
- K=5 mean SHAP share G2_cart_signals: **0.429**  ·  `rq5_importance_groups_k5`
- K=5 mean SHAP share G5_price_time: **0.297**  ·  `rq5_importance_groups_k5`
- K=5 mean SHAP share G4_category_hierarchy: **0.191**  ·  `rq5_importance_groups_k5`
- K=5 mean SHAP share G3_behavioural_ratios: **0.047**  ·  `rq5_importance_groups_k5`
- K=5 mean SHAP share G1_activity: **0.036**  ·  `rq5_importance_groups_k5`

## RQ4: efficiency (K=5)

- Rule baseline p95 latency / size / train time / throughput: **0.0008 ms / 0.000 MB / 0.0 s / 1,280,680 per s**  ·  `rq4_efficiency`
- Logistic reg. p95 latency / size / train time / throughput: **0.0316 ms / 0.002 MB / 1.3 s / 33,881 per s**  ·  `rq4_efficiency`
- MLP p95 latency / size / train time / throughput: **0.0796 ms / 0.090 MB / 54.6 s / 16,683 per s**  ·  `rq4_efficiency`
- GRU p95 latency / size / train time / throughput: **0.0875 ms / 0.068 MB / 249.0 s / 12,007 per s**  ·  `rq4_efficiency`
- LightGBM p95 latency / size / train time / throughput: **0.1579 ms / 0.929 MB / 5.8 s / 7,069 per s**  ·  `rq4_efficiency`
- TabNet p95 latency / size / train time / throughput: **0.9437 ms / 0.825 MB / 326.6 s / 1,101 per s**  ·  `rq4_efficiency`
- Random forest p95 latency / size / train time / throughput: **5.4123 ms / 105.128 MB / 33.6 s / 195 per s**  ·  `rq4_efficiency`

## Extensions at K=5 (paired bootstrap; green in fig5 = significant after Holm)

- + user history (G6), LightGBM: **+0.0555 [+0.0520, +0.0591] significant**  ·  `see fig5 sources`
- + item2vec & text embeddings: **+0.0019 [+0.0007, +0.0031] significant**  ·  `see fig5 sources`
- Optuna tuning, LightGBM: **+0.0005 [-0.0001, +0.0011] n.s.**  ·  `see fig5 sources`
- Optuna tuning, MLP: **+0.0011 [+0.0004, +0.0017] significant**  ·  `see fig5 sources`
- Optuna tuning, GRU: **+0.0036 [+0.0024, +0.0047] significant**  ·  `see fig5 sources`
- Stacking all models vs best single: **+0.0012 [+0.0002, +0.0023] n.s.**  ·  `see fig5 sources`
- Transformer vs GRU: **+0.0012 [-0.0001, +0.0027] n.s.**  ·  `see fig5 sources`
- User history gain K=3: **+0.0742 [+0.0712, +0.0774]**  ·  `user_history_tests`
- User history gain K=5: **+0.0555 [+0.0520, +0.0591]**  ·  `user_history_tests`
- User history gain K=10: **+0.0387 [+0.0343, +0.0432]**  ·  `user_history_tests`
- User history K=5, new users (share of test visits n=64,968): **0.3456 -> 0.3508**  ·  `user_history_segments`
- User history K=5, returning users (share of test visits n=230,677): **0.3308 -> 0.3955**  ·  `user_history_segments`
- User history gain K=5 counting only sessions ended >= 0 h before: **+0.0555**  ·  `user_history_gap_check`
- User history gain K=5 counting only sessions ended >= 1 h before: **+0.0479**  ·  `user_history_gap_check`
- User history gain K=5 counting only sessions ended >= 24 h before: **+0.0364**  ·  `user_history_gap_check`

## Segments / fairness (GRU)

- K=3 cart: worst/best segment, lift ratio, TPR gap, flag-rate gap: **carted / no cart, 0.86, 0.959, 0.979**  ·  `segment_disparity`
- K=3 category: worst/best segment, lift ratio, TPR gap, flag-rate gap: **appliances / computers, 0.72, 0.222, 0.119**  ·  `segment_disparity`
- K=3 day_type: worst/best segment, lift ratio, TPR gap, flag-rate gap: **weekday / weekend, 0.99, 0.013, 0.008**  ·  `segment_disparity`
- K=3 price: worst/best segment, lift ratio, TPR gap, flag-rate gap: **1 low / 4 high, 0.58, 0.099, 0.058**  ·  `segment_disparity`
- K=3 time_of_day: worst/best segment, lift ratio, TPR gap, flag-rate gap: **morning (6-12 h) / evening (18-24 h), 0.66, 0.045, 0.065**  ·  `segment_disparity`
- K=3 user_type: worst/best segment, lift ratio, TPR gap, flag-rate gap: **returning / new, 0.83, 0.043, 0.009**  ·  `segment_disparity`
- K=5 cart: worst/best segment, lift ratio, TPR gap, flag-rate gap: **carted / no cart, 0.92, 0.985, 0.968**  ·  `segment_disparity`
- K=5 category: worst/best segment, lift ratio, TPR gap, flag-rate gap: **appliances / computers, 0.76, 0.224, 0.099**  ·  `segment_disparity`
- K=5 day_type: worst/best segment, lift ratio, TPR gap, flag-rate gap: **weekday / weekend, 0.98, 0.012, 0.010**  ·  `segment_disparity`
- K=5 price: worst/best segment, lift ratio, TPR gap, flag-rate gap: **1 low / 4 high, 0.54, 0.102, 0.050**  ·  `segment_disparity`
- K=5 time_of_day: worst/best segment, lift ratio, TPR gap, flag-rate gap: **morning (6-12 h) / evening (18-24 h), 0.71, 0.074, 0.071**  ·  `segment_disparity`
- K=5 user_type: worst/best segment, lift ratio, TPR gap, flag-rate gap: **returning / new, 0.79, 0.077, 0.004**  ·  `segment_disparity`
- K=10 cart: worst/best segment, lift ratio, TPR gap, flag-rate gap: **carted / no cart, 0.87, 0.932, 0.839**  ·  `segment_disparity`
- K=10 category: worst/best segment, lift ratio, TPR gap, flag-rate gap: **appliances / other, 0.70, 0.202, 0.089**  ·  `segment_disparity`
- K=10 day_type: worst/best segment, lift ratio, TPR gap, flag-rate gap: **weekend / weekday, 1.00, 0.016, 0.010**  ·  `segment_disparity`
- K=10 price: worst/best segment, lift ratio, TPR gap, flag-rate gap: **1 low / 4 high, 0.57, 0.052, 0.085**  ·  `segment_disparity`
- K=10 time_of_day: worst/best segment, lift ratio, TPR gap, flag-rate gap: **morning (6-12 h) / evening (18-24 h), 0.70, 0.076, 0.073**  ·  `segment_disparity`
- K=10 user_type: worst/best segment, lift ratio, TPR gap, flag-rate gap: **returning / new, 0.82, 0.092, 0.008**  ·  `segment_disparity`
- GRU vs random forest: segments with a significant difference (Holm), all K: **3 of 72**  ·  `segment_tests`
- K=5 GRU new users: flag rate / lift: **0.133 / 3.97**  ·  `segment_metrics`
- K=5 GRU returning users: flag rate / lift: **0.129 / 3.15**  ·  `segment_metrics`

## Business value (illustrative; assumptions in business_value_summary)

- Assumptions: **discount 0.1, uplift 0.02, order value $365**  ·  `business_value_summary`
- K=3 GRU net/month vs offer-to-all gain: **$16,291,211 (+$6,415,955)**  ·  `business_value`
- K=5 GRU net/month vs offer-to-all gain: **$11,137,844 (+$3,863,425)**  ·  `business_value`
- K=10 GRU net/month vs offer-to-all gain: **$5,075,377 (+$1,387,293)**  ·  `business_value`
- K=3 spread between ML/DL models: **$90,149**  ·  `business_value`
- K=5 spread between ML/DL models: **$44,694**  ·  `business_value`
- K=10 spread between ML/DL models: **$65,842**  ·  `business_value`
- K=3 Spearman PR-AUC rank vs profit rank: **0.93**  ·  `business_value_summary`
- K=5 Spearman PR-AUC rank vs profit rank: **0.61**  ·  `business_value_summary`
- K=10 Spearman PR-AUC rank vs profit rank: **0.46**  ·  `business_value_summary`

## LLM agent vs RAG (local Ollama models)

- llama3.1:8b agent on standard questions (n=30): accuracy / grounding / s: **0.815 / 0.933 / 2.6**  ·  `agent_eval_summary`
- llama3.1:8b rag on standard questions (n=30): accuracy / grounding / s: **0.926 / 0.967 / 2.3**  ·  `agent_eval_summary`
- llama3.1:8b agent on tool questions (n=18): accuracy / grounding / s: **1.000 / 1.000 / 2.7**  ·  `agent_eval_summary`
- llama3.1:8b rag on tool questions (n=18): accuracy / grounding / s: **0.333 / 1.000 / 2.9**  ·  `agent_eval_summary`
- qwen3:14b agent on standard questions (n=30): accuracy / grounding / s: **0.704 / 1.000 / 4.2**  ·  `agent_eval_summary`
- qwen3:14b rag on standard questions (n=30): accuracy / grounding / s: **0.926 / 0.967 / 5.0**  ·  `agent_eval_summary`
- qwen3:14b agent on tool questions (n=18): accuracy / grounding / s: **0.778 / 0.833 / 5.7**  ·  `agent_eval_summary`
- qwen3:14b rag on tool questions (n=18): accuracy / grounding / s: **0.222 / 1.000 / 3.7**  ·  `agent_eval_summary`
- Retrieval bm25 recall@5 / MRR@10: **0.950 / 0.871**  ·  `rag_retrieval`
- Retrieval dense recall@5 / MRR@10: **0.700 / 0.429**  ·  `rag_retrieval`
- Retrieval hybrid recall@5 / MRR@10: **0.917 / 0.744**  ·  `rag_retrieval`
- RAG step 11 ollama:llama3.1:8b: accuracy / abstention / grounding: **0.900 / 1.000 / 0.968**  ·  `rag_answer_summary`
- RAG step 11 ollama:qwen3:14b: accuracy / abstention / grounding: **0.933 / 1.000 / 0.984**  ·  `rag_answer_summary`
