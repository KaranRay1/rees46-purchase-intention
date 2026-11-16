"""Step 7: monitoring. Would the engine keep working after launch?

Needs steps 1-3. Nothing is retrained. For each K:

* **weekly accuracy**: every model's primary-seed predictions on the validation and
  test weeks (both after the training period), scored week by week by session start:
  PR-AUC with a bootstrap 95 % CI, lift over the week's buyer rate, precision and
  recall at the tuned threshold, and mean score vs buyer rate (calibration);
* **input drift**: the Population Stability Index of each of the 18 features per
  calendar month against the training period (per week when the data covers fewer
  than three months, e.g. ``dev``).

``data/processed`` must hold the same mode (it holds one mode at a time); point
``paths.processed_dir`` elsewhere to monitor another mode without rebuilding it.

Writes ``monitoring_weekly_<mode>.csv``, ``monitoring_drift_<mode>.csv`` and
``monitoring_summary_<mode>.json`` to ``results/tables``.

Examples::

    python scripts/07_monitor.py --mode synthetic
    python scripts/07_monitor.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import polars as pl  # noqa: E402

from src.config import processed_dir_for_k, resolve_path  # noqa: E402
from src.data.sessionize import TARGET  # noqa: E402
from src.evaluation.monitoring import PSI_MAJOR, PSI_MODERATE, drift_table, weekly_performance  # noqa: E402
from src.serving.predictor import available_runs  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402
from src.utils.notebook import prepared_data_status  # noqa: E402

NOT_FEATURES = {"user_session", "split", TARGET, "prediction_time"}


def monitor_k(cfg, k: int, models: list[str] | None, n_resamples: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    pdir = processed_dir_for_k(cfg, k)
    starts = pl.read_parquet(pdir / "sessions.parquet", columns=["user_session", "session_start"])
    runs = available_runs(cfg).filter(pl.col("k") == k)
    if models:
        runs = runs.filter(pl.col("model").is_in(models))
    runs = runs.filter(pl.col("model") != "rule_based")
    weekly = []
    for r in runs.iter_rows(named=True):
        p = (pl.scan_parquet(resolve_path(cfg, "runs_dir") / r["run_id"] / "predictions.parquet")
             .filter(pl.col("split").is_in(["val", "test"])).collect()
             .join(starts, on="user_session", how="inner"))
        w = weekly_performance(p, float(r["threshold"]), n_resamples=n_resamples, seed=cfg["project"]["seeds"][0])
        if w.height:
            weekly.append(w.with_columns(k=pl.lit(k), model=pl.lit(r["model"])))
    feats = pl.read_parquet(pdir / "features.parquet").join(starts, on="user_session")
    n_months = feats.select(pl.col("session_start").dt.strftime("%Y-%m").n_unique()).item()
    weekly_periods = n_months < 3
    period = (pl.col("session_start").dt.truncate("1w").dt.strftime("%Y-%m-%d") if weekly_periods
              else pl.col("session_start").dt.strftime("%Y-%m"))
    feats = feats.with_columns(period=period)
    cols = [c for c in feats.columns if c not in NOT_FEATURES | {"session_start", "period"}]
    drift = drift_table(feats, cols, feats.filter(pl.col("split") == "train"), by="period").with_columns(
        k=pl.lit(k), granularity=pl.lit("week" if weekly_periods else "month"))
    split_of = feats.group_by("period").agg(pl.col("split").cast(pl.String).mode().first().alias("split"),
                                            days=pl.col("session_start").dt.date().n_unique())
    drift = drift.join(split_of, on="period", how="left")
    if weekly_periods:   # a part-week (start or end of the data) cannot match a full week's weekday mix
        part = (pl.col("days") < 7) & (pl.col("feature") == "day_of_week")
        drift = drift.with_columns(psi=pl.when(part).then(None).otherwise(pl.col("psi")),
                                   status=pl.when(part).then(pl.lit("n/a")).otherwise(pl.col("status")))
    return (pl.concat(weekly, how="diagonal_relaxed") if weekly else pl.DataFrame()), drift


def summarise(weekly: pl.DataFrame, drift: pl.DataFrame) -> dict:
    out: dict = {"psi_thresholds": {"moderate": PSI_MODERATE, "major": PSI_MAJOR}, "models": {}, "drift": {}}
    if weekly.height:
        for (k, m), g in weekly.group_by(["k", "model"], maintain_order=True):
            out["models"].setdefault(str(k), {})[m] = {
                "weeks": g.height, "pr_auc_min": float(g["pr_auc"].min()), "pr_auc_max": float(g["pr_auc"].max()),
                "lift_min": float(g["lift"].min()), "lift_max": float(g["lift"].max()),
            }
    after = drift.filter(pl.col("split") != "train")
    for (k,), g in after.group_by(["k"], maintain_order=True):
        worst = g.sort("psi", descending=True).head(5)
        out["drift"][str(k)] = {
            "major": sorted(set(g.filter(pl.col("status") == "major")["feature"].to_list())),
            "moderate": sorted(set(g.filter(pl.col("status") == "moderate")["feature"].to_list())),
            "worst": worst.select("period", "feature", "psi").to_dicts(),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--ks", type=int, nargs="+", help="prefix lengths (default: all with trained runs)")
    p.add_argument("--models", nargs="+", help="models to monitor (default: all except the rule baseline)")
    p.add_argument("--n-resamples", type=int, default=200, help="bootstrap resamples per week")
    args = p.parse_args(argv)
    cfg = setup(args)
    store = ResultStore(cfg)
    mode = cfg["data"]["mode"]
    status = prepared_data_status(cfg)
    ks = args.ks or [int(k) for k in cfg["sessionize"]["prefix_k"]]
    wrong = {k: status.get(k) for k in ks if status.get(k) != mode}
    if wrong:
        held = ", ".join(f"K={k}: {v or 'nothing'}" for k, v in wrong.items())
        raise SystemExit(
            f"data/processed holds {held}, not {mode!r}. "
            f"Rebuild it with `python scripts/01_prepare_data.py --mode {mode}` and `02_build_features.py`, or pass "
            f"--set paths.processed_dir=<folder prepared for {mode}>.")
    runs = available_runs(cfg)
    if not runs.height:
        raise SystemExit(f"No trained runs for mode {mode}; run scripts/03_run_experiments.py first.")
    ks = [k for k in ks if k in set(runs["k"].to_list())]
    weekly, drift = [], []
    for k in ks:
        with stage(f"K={k}: weekly accuracy and drift"):
            w, d = monitor_k(cfg, k, args.models, args.n_resamples)
            weekly.append(w)
            drift.append(d)
    weekly_df = pl.concat([w for w in weekly if w.height], how="diagonal_relaxed") if any(w.height for w in weekly) \
        else pl.DataFrame()
    drift_df = pl.concat(drift, how="diagonal_relaxed")
    store.tables_dir.mkdir(parents=True, exist_ok=True)
    if weekly_df.height:
        weekly_df.write_csv(store.path("monitoring_weekly", "csv"))
    drift_df.write_csv(store.path("monitoring_drift", "csv"))
    summary = summarise(weekly_df, drift_df)
    store.path("monitoring_summary", "json").write_text(json.dumps(summary, indent=2, default=str))

    print("\nWeekly PR-AUC range (validation + test weeks):")
    print_table([{"k": k, "model": m, **{x: v[x] for x in ("weeks", "pr_auc_min", "pr_auc_max")}}
                 for k, ms in summary["models"].items() for m, v in ms.items()])
    print("\nDrift after training (PSI > 0.25 = major):")
    print_table([{"k": k, "major": ", ".join(v["major"]) or "none", "moderate": ", ".join(v["moderate"]) or "none"}
                 for k, v in summary["drift"].items()])
    print(f"\nWrote {store.path('monitoring_weekly', 'csv').name}, {store.path('monitoring_drift', 'csv').name}, "
          f"{store.path('monitoring_summary', 'json').name}")


if __name__ == "__main__":
    main()
