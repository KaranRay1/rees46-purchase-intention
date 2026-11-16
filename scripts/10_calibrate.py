"""Step 10: calibrate scores into purchase probabilities (Platt scaling on the validation weeks).

Needs step 3. Nothing is retrained. For every primary-seed model and K, fits
``p = sigmoid(a * logit(score) + b)`` on the validation predictions, saves it as
``calibration.json`` in the run folder (the live predictor, API and dashboard apply it
from then on) and checks it on the untouched test period: Brier score, expected
calibration error (ECE) and mean predicted probability vs the actual buyer rate.

Writes ``calibration_<mode>.csv`` (one row per K and model) and
``calibration_reliability_<mode>.csv`` (predicted vs observed buyer rate per
probability bin, test period) to ``results/tables``.

Examples::

    python scripts/10_calibrate.py --mode synthetic
    python scripts/10_calibrate.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import polars as pl  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.serving.calibration import brier, ece, fit_platt, reliability  # noqa: E402
from src.serving.predictor import available_runs  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--bins", type=int, default=10, help="probability bins for ECE and the reliability table")
    args = p.parse_args(argv)
    cfg = setup(args)
    runs = available_runs(cfg)
    if not runs.height:
        raise SystemExit(f"No trained runs for mode {cfg['data']['mode']}; run scripts/03_run_experiments.py first.")
    runs_dir = resolve_path(cfg, "runs_dir")
    rows, rel = [], []
    with stage("Fit on validation, check on test"):
        for r in runs.sort("k").iter_rows(named=True):
            run_dir = runs_dir / r["run_id"]
            f = run_dir / "predictions.parquet"
            if not f.exists():
                continue
            preds = pl.read_parquet(f, columns=["split", "y_true", "score"])
            val, test = preds.filter(pl.col("split") == "val"), preds.filter(pl.col("split") == "test")
            if not val.height or not test.height:
                continue
            cal = fit_platt(val["y_true"].to_numpy(), val["score"].to_numpy())
            y, raw = test["y_true"].to_numpy(), test["score"].to_numpy()
            prob = cal(raw)
            stats = {
                "buyer_rate": float(y.mean()), "mean_score_raw": float(raw.mean()), "mean_prob": float(prob.mean()),
                "brier_raw": brier(y, raw), "brier": brier(y, prob),
                "ece_raw": ece(y, raw, args.bins), "ece": ece(y, prob, args.bins),
                "threshold_prob": float(cal(float(r["threshold"]))),
            }
            cal.save(run_dir, fitted_on="val", **{f"test_{k}": round(v, 6) for k, v in stats.items()})
            rows.append({"k": r["k"], "model": r["model"], "a": cal.a, "b": cal.b, **stats})
            rel += [{"k": r["k"], "model": r["model"], "kind": kind, **b}
                    for kind, s in (("raw", raw), ("calibrated", prob)) for b in reliability(y, s, args.bins)]
    table = pl.DataFrame(rows)
    store = ResultStore(cfg)
    store.tables_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(store.path("calibration", "csv"))
    pl.DataFrame(rel).write_csv(store.path("calibration_reliability", "csv"))
    print("\nTest period (fitted on validation only):")
    print_table(table.select("k", "model", pl.col("buyer_rate").round(3), pl.col("mean_score_raw").round(3),
                             pl.col("mean_prob").round(3), pl.col("ece_raw").round(3), pl.col("ece").round(3),
                             pl.col("brier_raw").round(4), pl.col("brier").round(4)).to_dicts())
    print(f"\nWrote calibration.json to {table.height} run folders, {store.path('calibration', 'csv').name} and "
          f"{store.path('calibration_reliability', 'csv').name}")


if __name__ == "__main__":
    main()
