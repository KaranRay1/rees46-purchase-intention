"""Step 12: conformal prediction sets and stacked ML + DL ensembles (no retraining).

Needs steps 1-3 (and ``data/processed`` for the same mode, for session start times).
Uses the primary-seed validation and test scores of every learned model.

* **Conformal** (``src/evaluation/conformal.py``): class-conditional split conformal sets
  calibrated on the validation weeks for alpha = 0.05, 0.10, 0.20; coverage, set sizes and
  the share of visits the engine can decide confidently, on the test weeks and week by week.
* **Stacking** (``src/evaluation/stacking.py``): meta-models over ML-only, DL-only and all
  base models, fitted on the validation weeks, judged on the test weeks with a paired
  bootstrap against the best single model (chosen on validation), Holm-corrected per K;
  plus how often pairs of models miss the same buyers.

Writes ``conformal``, ``conformal_weekly``, ``stacking``, ``stacking_tests``,
``error_overlap`` (CSV) and ``advanced_summary`` (JSON) to ``results/tables``.

Examples::

    python scripts/12_conformal_stacking.py --mode synthetic
    python scripts/12_conformal_stacking.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.config import processed_dir_for_k, resolve_path  # noqa: E402
from src.evaluation.conformal import MondrianConformal, weekly_coverage  # noqa: E402
from src.evaluation.stacking import ENSEMBLES, error_overlap, fit_meta, logit, mean_logit, mean_rank  # noqa: E402
from src.evaluation.statistics import bootstrap_settings, holm, paired_bootstrap  # noqa: E402
from src.serving.predictor import available_runs  # noqa: E402
from src.serving.recommend import recommended  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402
from src.utils.notebook import prepared_data_status  # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)


def wide_scores(cfg, k: int) -> tuple[pl.DataFrame, list[str]]:
    """One row per val/test session: split, y, session_start, and score_<model> / pred_<model> per model."""
    runs = available_runs(cfg).filter((pl.col("k") == k) & (pl.col("model") != "rule_based"))
    runs_dir = resolve_path(cfg, "runs_dir")
    wide, models = None, []
    for r in runs.iter_rows(named=True):
        f = runs_dir / r["run_id"] / "predictions.parquet"
        if not f.exists():
            continue
        p = pl.read_parquet(f).select("user_session", "split", "y_true",
                                      pl.col("score").alias(f"score_{r['model']}"),
                                      pl.col("y_pred").alias(f"pred_{r['model']}"))
        wide = p if wide is None else wide.join(p.drop("split", "y_true"), on="user_session", how="inner")
        models.append(r["model"])
    starts = pl.read_parquet(processed_dir_for_k(cfg, k) / "sessions.parquet", columns=["user_session", "session_start"])
    wide = wide.join(starts, on="user_session", how="inner").sort("session_start")
    return wide, models


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--ks", type=int, nargs="+", help="prefix lengths (default: all with trained runs)")
    p.add_argument("--n-resamples", type=int, help="bootstrap resamples (default: evaluation.bootstrap)")
    args = p.parse_args(argv)
    cfg = setup(args)
    mode = cfg["data"]["mode"]
    store = ResultStore(cfg)
    store.tables_dir.mkdir(parents=True, exist_ok=True)
    runs = available_runs(cfg)
    if not runs.height:
        raise SystemExit(f"No trained runs for mode {mode}; run scripts/03_run_experiments.py first.")
    ks = args.ks or sorted(set(runs["k"].to_list()))
    status = prepared_data_status(cfg)
    wrong = {k: status.get(k) for k in ks if status.get(k) != mode}
    if wrong:
        raise SystemExit(f"data/processed holds {wrong}, not {mode!r}; prepare it or pass --set paths.processed_dir=...")
    boot = bootstrap_settings(cfg)
    if args.n_resamples:
        boot["n_resamples"] = args.n_resamples
    seed = cfg["project"]["seeds"][0]
    rq1, pairs, eff = store.csv("rq1_model_comparison"), store.csv("rq1_pairwise_tests"), store.csv("rq4_efficiency")

    conf_rows, weekly_rows, stack_rows, test_rows, overlap_rows = [], [], [], [], []
    summary: dict = {"mode": mode, "alphas": ALPHAS, "ks": ks}
    for k in ks:
        with stage(f"K={k}: conformal sets and stacking"):
            w, models = wide_scores(cfg, k)
            val, test = w.filter(pl.col("split") == "val"), w.filter(pl.col("split") == "test")
            yv, yt = val["y_true"].to_numpy(), test["y_true"].to_numpy()
            ship = recommended(rq1, pairs, eff, k)[0] if rq1 is not None and eff is not None else models[0]
            week = test["session_start"].dt.truncate("1w").dt.date().to_numpy()

            # ---------------- conformal
            for m in models:
                for a in ALPHAS:
                    cp = MondrianConformal.fit(yv, val[f"score_{m}"].to_numpy(), a)
                    conf_rows.append({"k": k, "model": m, "engine": m == ship,
                                      **cp.evaluate(yt, test[f"score_{m}"].to_numpy())})
            cp = MondrianConformal.fit(yv, val[f"score_{ship}"].to_numpy(), 0.10)
            weekly_rows.append(weekly_coverage(cp, yt, test[f"score_{ship}"].to_numpy(), week)
                               .with_columns(k=pl.lit(k), model=pl.lit(ship), alpha=pl.lit(0.10)))

            # ---------------- stacking
            Lv = {m: logit(val[f"score_{m}"].to_numpy()) for m in models}
            Lt = {m: logit(test[f"score_{m}"].to_numpy()) for m in models}
            base_val = {m: average_precision_score(yv, val[f"score_{m}"]) for m in models}
            best = max(base_val, key=base_val.get)                         # chosen on validation, not test
            for m in models:
                stack_rows.append({"k": k, "method": m, "kind": "single", "members": m,
                                   "val_pr_auc": base_val[m], "test_pr_auc": average_precision_score(yt, test[f"score_{m}"]),
                                   "best_single": m == best})
            scores: dict[str, np.ndarray] = {}
            for name, members in ENSEMBLES.items():
                members = [m for m in members if m in models]
                if len(members) < 2:
                    continue
                Xv = np.column_stack([Lv[m] for m in members])
                Xt = np.column_stack([Lt[m] for m in members])
                meta, C, cv = fit_meta(Xv, yv, seed=seed)
                scores[name] = meta.predict_proba(Xt)[:, 1]
                stack_rows.append({"k": k, "method": name, "kind": "stacked (logistic meta-model)",
                                   "members": ", ".join(members), "val_pr_auc": max(cv.values()),
                                   "test_pr_auc": average_precision_score(yt, scores[name]), "C": C,
                                   "weights": json.dumps({m: round(float(c), 4) for m, c in zip(members, meta.coef_[0])})})
            members = [m for m in ENSEMBLES["stack_all"] if m in models]
            St = np.column_stack([test[f"score_{m}"].to_numpy() for m in members])
            for name, s in (("mean_logit_all", mean_logit(np.column_stack([Lt[m] for m in members]))),
                            ("mean_rank_all", mean_rank(St))):
                scores[name] = s
                stack_rows.append({"k": k, "method": name, "kind": "average", "members": ", ".join(members),
                                   "test_pr_auc": average_precision_score(yt, s)})
            raw = []
            for name, s in scores.items():
                t = paired_bootstrap(yt, s, test[f"score_{best}"].to_numpy(), "pr_auc", seed=seed, **boot)
                raw.append({"k": k, "ensemble": name, "vs": best, **{x: t[x] for x in
                            ("a", "b", "diff", "lower", "upper", "p_value", "share_a_better")}})
            for r, ph in zip(raw, holm([r["p_value"] for r in raw])):
                r["p_holm"] = ph
                r["significant"] = ph < float(cfg["evaluation"]["alpha"])
            test_rows += raw
            overlap_rows += [{"k": k, **o} for o in error_overlap(yt, {m: test[f"pred_{m}"].to_numpy() for m in models})]
            top = max(raw, key=lambda r: r["diff"])
            summary[f"k{k}"] = {"engine": ship, "best_single_on_val": best, "best_ensemble": top["ensemble"],
                                "gain": top["diff"], "p_holm": top["p_holm"]}

    conf = pl.DataFrame(conf_rows)
    conf.write_csv(store.path("conformal", "csv"))
    pl.concat(weekly_rows).with_columns(pl.col("week").cast(pl.String)).write_csv(store.path("conformal_weekly", "csv"))
    pl.DataFrame(stack_rows).write_csv(store.path("stacking", "csv"))
    tests = pl.DataFrame(test_rows)
    tests.write_csv(store.path("stacking_tests", "csv"))
    pl.DataFrame(overlap_rows).write_csv(store.path("error_overlap", "csv"))
    store.path("advanced_summary", "json").write_text(json.dumps(summary, indent=2, default=str))

    print("\nConformal sets for the engine model (test period):")
    print_table(conf.filter(pl.col("engine")).select(
        "k", "model", "alpha", pl.col("coverage").round(3), pl.col("coverage_buyers").round(3),
        pl.col("coverage_non_buyers").round(3), pl.col("share_unsure").round(3),
        pl.col("share_confident_buyer").round(3), pl.col("precision_confident_buyer").round(3)).to_dicts())
    print("\nEnsembles vs the best single model (test PR-AUC):")
    print_table(tests.select("k", "ensemble", "vs", pl.col("a").round(4), pl.col("b").round(4),
                             pl.col("diff").round(4), pl.col("p_holm").round(3), "significant").to_dicts())


if __name__ == "__main__":
    main()
