# ARCHIVED — Original blueprint (with MLOps / MLflow)

> Kept aside on purpose. Not used in the current research build.
> The active blueprint is `docs/research_blueprint.md`.
> Reuse this file later when adding MLflow tracking / deployment for large scale.

---

# MSc AI Project Master Blueprint

## Project Title
Real-Time Session-Based E-Commerce Purchase Intention Prediction: A Comparative Analysis of Tabular Machine Learning and Deep Neural Architectures

## Dataset Specifications
- Primary Dataset: REES46 E-Commerce Behavior Dataset (multi-category store clickstream event logs).
- Raw Fields: event_time (UTC timestamp), event_type (view, cart, remove_from_cart, purchase), product_id, category_id, category_code, brand, price, user_id, user_session.
- Dataset Scale: ~285M rows across Oct 2019 – Apr 2020.

## System Architecture & Workflow
```
Data Layer (REES46 Logs) -> Feature Engineering (Behavioral Ratios) -> Machine Learning (RF / GBM)
   -> Explainability (SHAP) -> Experiment Tracking (MLflow) -> Evaluation & Metrics (F1, PR-AUC, Latency)
```

## Phased Blueprint Details

### Phase 1: Project Architecture & Data Ingestion
- Modular folder layout separating data loader, feature engine, baseline/ensemble models, explainability, and evaluation pipelines.
- Polars/Pandas loader for .csv logs. Session prefixing (K ∈ {3,5,10}) and filtering to avoid target leakage.
- Chronological train/val/test splits based on session start timestamp.

### Phase 2: Behavioral Ratio Feature Engineering
- Session aggregations: view_count, cart_count, remove_count, session duration, price avg/max/sum, unique categories/brands.
- Ratios: Cart/View, Purchase/View, Purchase/Cart, Remove/Cart (each with + ε in denominator).
- Fit StandardScaler and encoders strictly on training split.

### Phase 3: Model Pipeline & MLflow Tracking
- Rule-based baseline: predict purchase if N_cart > 0.
- Logistic Regression on scaled continuous features + one-hot top categories.
- RandomForestClassifier (balanced class weights, tuned depth, min_samples_split).
- LGBMClassifier with early stopping on validation PR-AUC/loss.
- MLflow: log_params, log_metrics, sklearn.log_model, lightgbm.log_model.

### Phase 4: SHAP Explainability
- shap.TreeExplainer; global beeswarm + mean |SHAP|; local waterfall/force plots.

### Phase 5: Evaluation & Diagnostics
- Precision, Recall, F1, Macro-F1, PR-AUC, ROC-AUC.
- Training time, batch/single inference latency, memory, artifact size.
- Ablation of ratios across K; FP/FN inspection via SHAP.

## Original Cursor prompt — key module specs
- config/config.yaml: paths, prefix_k: 5, split 0.7/0.15/0.15,
  random_forest {n_estimators 100, max_depth 12, class_weight balanced, random_state 42},
  lightgbm {n_estimators 200, learning_rate 0.05, max_depth 8, num_leaves 31, class_weight balanced, random_state 42}.
- src/data_loader.py: polars loader, process_session_prefixes(df, prefix_k), chronological split, synthetic fallback (1,000 sessions).
- src/feature_engineering.py: BehavioralFeatureEngineer (aggregates, ratios with ε=1e-6, train-only StandardScaler).
- src/models.py: RuleBasedBaseline, LogisticRegressionBaseline, RandomForestPipeline, LightGBMPipeline.
- src/explainability.py: SHAPExplainer, plot_global_importance, plot_local_waterfall.
- src/evaluation.py: evaluate_model -> metrics dict incl. latency (perf_counter) and memory (psutil).
- experiments/mlflow_tracker.py: run_mlflow_experiment(...) under experiment "REES46_Purchase_Intention".
- main.py CLI + notebooks/EDA_and_Pipeline.ipynb.

Original requirements: polars, pandas, numpy, scikit-learn, lightgbm, shap, mlflow, matplotlib, seaborn, pyyaml, psutil.
