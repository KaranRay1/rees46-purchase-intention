"""Step 9: build the lightweight demo bundle for hosting the dashboard and API online.

Needs steps 1-3 (and the dashboard's product cache, built here if missing). Copies into
``results/demo/<mode>/``, which is small enough for git:

* ``runs/<run_id>/``: the primary-seed models with their preprocessors, for every K and
  model whose ``model.pkl`` is at most ``--max-model-mb`` (the random forest is ~110 MB);
* ``runs.csv``: those runs' rows of ``all_runs.csv``;
* ``test_predictions_k<K>.parquet``: every model's test-period outcome and score, for the
  Campaign planner (no session ids);
* ``product_month.parquet``: the product catalogue (one row per product and month).

The app and API switch to the bundle automatically when the full run folders are not on
the machine (or when ``PIE_DEMO=1``); see ``src.serving.results.demo_overrides``.

Examples::

    python scripts/09_build_demo.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import polars as pl  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.serving.catalog import product_month  # noqa: E402
from src.serving.predictor import available_runs  # noqa: E402
from src.serving.results import demo_dir  # noqa: E402

RUN_FILES = ("model.pkl", "metrics.json", "params.json", "run.json", "calibration.json")


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--max-model-mb", type=float, default=20.0, help="skip saved models larger than this")
    args = p.parse_args(argv)
    cfg = setup(args)
    runs = available_runs(cfg)
    if not runs.height:
        raise SystemExit(f"No trained runs for mode {cfg['data']['mode']}; run scripts/03_run_experiments.py first.")
    out = demo_dir(cfg)
    if out.exists():
        shutil.rmtree(out)
    (out / "runs").mkdir(parents=True)
    runs_dir = resolve_path(cfg, "runs_dir")

    kept, report = [], []
    with stage("Models"):
        for r in runs.iter_rows(named=True):
            src = runs_dir / r["run_id"]
            size = (src / "model.pkl").stat().st_size / 1e6 if (src / "model.pkl").exists() else float("inf")
            ok = size <= args.max_model_mb
            report.append({"k": r["k"], "model": r["model"], "model_mb": round(size, 2), "bundled": ok})
            if not ok:
                continue
            dst = out / "runs" / r["run_id"]
            dst.mkdir()
            for f in RUN_FILES:
                if (src / f).exists():
                    shutil.copy2(src / f, dst / f)
            shutil.copytree(src / "artifacts", dst / "artifacts")
            kept.append(r["run_id"])
        all_runs = pl.read_csv(resolve_path(cfg, "all_runs_csv"), infer_schema=False)
        all_runs.filter(pl.col("run_id").is_in(kept)).write_csv(out / "runs.csv")
        print_table(report)

    with stage("Test predictions (all models)"):
        for k in sorted(set(runs["k"].to_list())):
            parts = []
            for r in runs.filter(pl.col("k") == k).iter_rows(named=True):
                f = runs_dir / r["run_id"] / "predictions.parquet"
                if f.exists():
                    parts.append(pl.scan_parquet(f).filter(pl.col("split") == "test")
                                 .select(pl.lit(r["model"]).alias("model"), pl.col("y_true").cast(pl.Int8),
                                         pl.col("score").cast(pl.Float32)).collect())
            if parts:
                pl.concat(parts).write_parquet(out / f"test_predictions_k{k}.parquet", compression="zstd",
                                               compression_level=10)

    with stage("Product catalogue"):
        pm = product_month(cfg)
        if pm is not None:
            pm.write_parquet(out / "product_month.parquet", compression="zstd", compression_level=10)

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    (out / "manifest.json").write_text(json.dumps({"mode": cfg["data"]["mode"], "runs": kept, "size_mb": round(total, 1),
                                                   "skipped": [x for x in report if not x["bundled"]]}, indent=2))
    print(f"\nWrote {out.relative_to(resolve_path(cfg, 'tables_dir').parent.parent)} ({total:.1f} MB)")


if __name__ == "__main__":
    main()
