"""Step 15: Bayesian hyperparameter tuning (Optuna TPE) with time-ordered validation.

Needs steps 1-3. The RQ1 models use hand-set starting values; this asks how much tuning is
worth. For each model in ``--models`` at K = ``--k``:

1. **Search**: Optuna's TPE sampler (Bergstra et al., 2011) proposes settings; each trial
   trains on the training weeks and is scored by PR-AUC on the **validation weeks**, which
   come after them in time (forward-chaining holdout; the test weeks are never seen).
2. **Confirm**: the best setting is retrained with every project seed (experiment
   ``rq7_tuned``) and judged on the test weeks against the default setting from RQ1:
   mean +- sd over seeds and a paired bootstrap of the primary seeds.
3. **Explain**: which hyperparameters mattered (fANOVA importance over the trials).

Writes ``tuning_trials``, ``tuning`` and ``tuning_importance`` (CSV) to ``results/tables``.

Examples::

    python scripts/15_tune.py --mode full --set data.session_sample_fraction=0.1
    python scripts/15_tune.py --mode full --models lightgbm --trials 60
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402
from _experiment import InputCache, train_and_log, xy  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.config import resolve_path, set_seed  # noqa: E402
from src.evaluation.statistics import holm, paired_bootstrap  # noqa: E402
from src.models.base import build_model  # noqa: E402
from src.serving.predictor import available_runs  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402
from src.utils.results_logger import load_predictions  # noqa: E402

EXPERIMENT = "rq7_tuned"
HIDDEN = {"256-128-64": [256, 128, 64], "128-64-32": [128, 64, 32], "128-64": [128, 64], "64-32": [64, 32]}


def space(trial: optuna.Trial, model: str) -> dict:
    if model == "lightgbm":
        return {"num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}
    if model == "mlp":
        return {"hidden_layers": HIDDEN[trial.suggest_categorical("hidden_layers", list(HIDDEN))],
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]), "max_epochs": 20}
    if model in ("gru", "transformer"):
        p = {"dropout": trial.suggest_float("dropout", 0.0, 0.4),
             "learning_rate": trial.suggest_float("learning_rate", 2e-4, 3e-3, log=True), "max_epochs": 15}
        if model == "gru":
            p |= {"hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
                  "num_layers": trial.suggest_int("num_layers", 1, 2),
                  "bidirectional": trial.suggest_categorical("bidirectional", [False, True])}
        else:
            p |= {"d_model": trial.suggest_categorical("d_model", [32, 64, 96]),
                  "n_layers": trial.suggest_int("n_layers", 1, 3)}
        return p
    raise ValueError(f"No search space for {model!r}")


def default_trial(cfg, model: str) -> dict:
    """The RQ1 setting expressed in the search space (enqueued as trial 0 for reference)."""
    m = cfg["models"][model]
    if model == "lightgbm":
        return {"num_leaves": m["num_leaves"], "learning_rate": m["learning_rate"],
                "min_child_samples": m["min_child_samples"], "subsample": m["subsample"],
                "colsample_bytree": m["colsample_bytree"], "reg_lambda": 1e-3}
    if model == "mlp":
        hidden = next((n for n, v in HIDDEN.items() if v == list(m["hidden_layers"])), "128-64-32")
        return {"hidden_layers": hidden, "dropout": m["dropout"], "learning_rate": m["learning_rate"],
                "weight_decay": m["weight_decay"], "batch_size": m["batch_size"]}
    if model == "gru":
        return {"dropout": m["dropout"], "learning_rate": m["learning_rate"], "hidden_size": m["hidden_size"],
                "num_layers": m["num_layers"], "bidirectional": m["bidirectional"]}
    return {"dropout": m["dropout"], "learning_rate": m["learning_rate"], "d_model": m["d_model"], "n_layers": m["n_layers"]}


def make_objective(cfg, model: str, seed: int, columns, extra: dict, X_tr, y_tr, X_va, y_va):
    """Optuna objective: train on the training weeks, return PR-AUC on the validation weeks."""
    def objective(trial: optuna.Trial) -> float:
        params = space(trial, model)
        set_seed(seed)
        m = build_model(model, with_params(cfg, model, params), seed=seed, columns=columns, **extra)
        m.fit(X_tr, y_tr, X_va, y_va)
        return float(average_precision_score(y_va, m.predict_proba(X_va)))
    return objective


def with_params(cfg, model: str, params: dict):
    c = copy.deepcopy(cfg)
    c["models"][model] = {**c["models"][model], **params}
    return c


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=["lightgbm", "mlp", "gru"])
    p.add_argument("--k", type=int, help="prefix length (default: sessionize.default_k)")
    p.add_argument("--trials", type=int, nargs="+", default=[40, 15, 10], help="trials per model (one value or one per model)")
    p.add_argument("--n-resamples", type=int, default=1000)
    args = p.parse_args(argv)
    cfg = setup(args)
    store = ResultStore(cfg)
    k = args.k or int(cfg["sessionize"]["default_k"])
    seeds = [int(s) for s in cfg["project"]["seeds"]]
    trials = args.trials if len(args.trials) == len(args.models) else [args.trials[0]] * len(args.models)
    cache = InputCache(cfg)
    rq1 = store.csv("rq1_model_comparison")
    base = available_runs(cfg)
    runs_dir = resolve_path(cfg, "runs_dir")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial_rows, summary, imp_rows, raw = [], [], [], []

    for model, n_trials in zip(args.models, trials):
        data, _, extra, columns = cache.get(k, model)
        X_tr, y_tr = xy(data["train"], model)
        X_va, y_va = xy(data["val"], model)

        objective = make_objective(cfg, model, seeds[0], columns, extra, X_tr, y_tr, X_va, y_va)

        with stage(f"K={k} {model}: {n_trials} Optuna trials (TPE) on the validation weeks"):
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seeds[0]),
                                        study_name=f"{model}_k{k}")
            default = default_trial(cfg, model)
            study.enqueue_trial(default)                  # trial 0 = the RQ1 setting, as a reference point
            def report(st, t, n=n_trials):            # one line per trial, so progress can be followed in the log
                print(f"   trial {t.number + 1}/{n}: val PR-AUC {t.value:.4f} (best {st.best_value:.4f})", flush=True)

            study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[report])
            for t in study.trials:
                trial_rows.append({"k": k, "model": model, "trial": t.number, "val_pr_auc": t.value,
                                   "seconds": (t.datetime_complete - t.datetime_start).total_seconds(),
                                   "params": str(t.params)})
            try:
                for name, v in optuna.importance.get_param_importances(study).items():
                    imp_rows.append({"k": k, "model": model, "param": name, "importance": float(v)})
            except (RuntimeError, ValueError):
                pass
            best = space(optuna.trial.FixedTrial(study.best_params), model)
            best.pop("max_epochs", None)
            print(f"   best val PR-AUC {study.best_value:.4f} (default {study.trials[0].value:.4f}): {study.best_params}")

        with stage(f"K={k} {model}: retrain the best setting with {len(seeds)} seeds"):
            tuned_cfg = with_params(cfg, model, best)
            tuned_cache = InputCache(tuned_cfg)
            tuned_cache._tab, tuned_cache._seq = cache._tab, cache._seq          # same inputs, no re-preparation
            runs = [train_and_log(tuned_cfg, tuned_cache, EXPERIMENT, model, k, s) for s in seeds]
            vals = [float(r["test_pr_auc"]) for r in runs]
            d = rq1.filter((pl.col("k") == k) & (pl.col("model") == model)).row(0, named=True)
            summary.append({"k": k, "model": model, "trials": n_trials, "best_val_pr_auc": study.best_value,
                            "default_val_pr_auc": study.trials[0].value, "tuned_test_mean": float(np.mean(vals)),
                            "tuned_test_sd": float(np.std(vals, ddof=1)), "default_test_mean": d["pr_auc_mean"],
                            "default_test_sd": d["pr_auc_std"], "best_params": str(study.best_params)})
            tp = load_predictions(runs_dir / runs[0]["run_id"], "test").sort("user_session")
            dr = base.filter((pl.col("k") == k) & (pl.col("model") == model))["run_id"][0]
            dp = load_predictions(runs_dir / dr, "test").sort("user_session")
            j = tp.join(dp.select("user_session", pl.col("score").alias("default")), on="user_session")
            t = paired_bootstrap(j["y_true"].to_numpy(), j["score"].to_numpy(), j["default"].to_numpy(), "pr_auc",
                                 n_resamples=args.n_resamples, seed=seeds[0])
            raw.append({"k": k, "model": model, "diff": t["diff"], "lower": t["lower"], "upper": t["upper"],
                        "p_value": t["p_value"]})
    for r, ph in zip(raw, holm([r["p_value"] for r in raw])):
        r["p_holm"], r["significant"] = ph, ph < float(cfg["evaluation"]["alpha"])
    res = pl.DataFrame(summary).join(pl.DataFrame(raw), on=["k", "model"], how="left")
    store.tables_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(trial_rows).write_csv(store.path("tuning_trials", "csv"))
    res.write_csv(store.path("tuning", "csv"))
    if imp_rows:
        pl.DataFrame(imp_rows).write_csv(store.path("tuning_importance", "csv"))
    print("\nTuned vs default (test PR-AUC):")
    print_table(res.select("k", "model", pl.col("default_test_mean").round(4), pl.col("tuned_test_mean").round(4),
                           pl.col("diff").round(4), pl.col("p_holm").round(3), "significant").to_dicts())


if __name__ == "__main__":
    main()
