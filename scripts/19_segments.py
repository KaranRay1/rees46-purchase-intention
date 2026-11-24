"""Step 19: segment (fairness) analysis. Do the models work, and treat visits, alike across groups?

Needs step 3 (saved test predictions) and step 2 (features) for the mode; step 10 (calibration)
for the calibration columns and step 17's session cache for new vs returning users (both optional).
See ``src/evaluation/segments.py`` for the segments and metrics.

For every K and model: PR-AUC, PR-AUC lift, ROC-AUC, calibration gap, flag rate, true- and
false-positive rate per segment, and the best-to-worst spread per dimension. For the engine (GRU)
each segment's PR-AUC gets a bootstrap CI, and in every segment the GRU is compared with the
random forest (paired bootstrap, Holm over segments) to see whether the ML ~ DL tie holds everywhere.

Writes ``segment_metrics``, ``segment_disparity``, ``segment_tests`` (CSV) and ``segment_summary``
(JSON) to ``results/tables``.

Examples::

    python scripts/19_segments.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import polars as pl  # noqa: E402

from src.config import processed_dir_for_k, resolve_path  # noqa: E402
from src.evaluation.segments import assign_segments, disparity, price_edges, segment_table  # noqa: E402
from src.evaluation.statistics import bootstrap_ci, holm, paired_bootstrap  # noqa: E402
from src.features.build_features import load_feature_table  # noqa: E402
from src.features.user_history import history_features  # noqa: E402
from src.serving.calibration import Calibrator  # noqa: E402
from src.serving.predictor import available_runs  # noqa: E402
from src.serving.results import ResultStore, test_predictions  # noqa: E402

MODELS = ["rule_based", "logistic_regression", "random_forest", "lightgbm", "mlp", "tabnet", "gru"]
ENGINE, RIVAL = "gru", "random_forest"


def calibrator(cfg, runs: pl.DataFrame, k: int, model: str) -> Calibrator | None:
    r = runs.filter((pl.col("k") == k) & (pl.col("model") == model)) if runs.height else runs
    return Calibrator.load(resolve_path(cfg, "runs_dir") / r["run_id"][0]) if r.height else None


def returning_flags(cfg, k: int, test_ids: pl.Series) -> pl.DataFrame | None:
    """G6 new/returning flag for the test visits, from step 17's cached session table (if any)."""
    cache = sorted((Path(resolve_path(cfg, "interim_dir")) / "user_history").glob("sessions_*.parquet"),
                   key=lambda p: p.stat().st_size)
    if not cache:
        return None
    targets = pl.read_parquet(processed_dir_for_k(cfg, k) / "sessions.parquet",
                              columns=["user_session", "user_id", "session_start"]
                              ).filter(pl.col("user_session").is_in(test_ids.implode()))
    return history_features(targets, pl.read_parquet(cache[-1]))


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--ks", type=int, nargs="+", help="prefix lengths (default: config)")
    p.add_argument("--flag-share", type=float, default=0.13,
                   help="share of visits flagged 'likely buyer' (the campaign's best skip share at K=5)")
    p.add_argument("--min-size", type=int, default=500, help="smallest segment reported")
    p.add_argument("--n-resamples", type=int, default=500)
    args = p.parse_args(argv)
    cfg = setup(args)
    store = ResultStore(cfg)
    ks = args.ks or [int(k) for k in cfg["sessionize"]["prefix_k"]]
    seed = int(cfg["project"]["seeds"][0])
    alpha = float(cfg["evaluation"]["alpha"])
    runs = available_runs(cfg)

    tables, tests, used_user_type = [], [], {}
    for k in ks:
        with stage(f"K={k}: metrics per segment"):
            feats = load_feature_table(cfg, k)
            edges = price_edges(feats.filter(pl.col("split") == "train")["price_mean"])
            test = feats.filter(pl.col("split") == "test")
            scores, probs, ids, y = {}, {}, None, None
            for m in MODELS:
                pr = test_predictions(cfg, k, m)
                if pr is None or "user_session" not in pr.columns:
                    continue
                if ids is None:
                    ids, y = pr["user_session"], pr["y_true"].to_numpy()
                elif not (pr["user_session"] == ids).all():
                    raise SystemExit(f"K={k}: {m} predictions are not in the same visit order")
                scores[m] = pr["score"].to_numpy()
                cal = calibrator(cfg, runs, k, m)
                if cal is not None:
                    probs[m] = cal(scores[m])
            if not scores:
                print("   no saved test predictions with visit ids; skipped")
                continue
            ret = returning_flags(cfg, k, ids)
            used_user_type[k] = ret is not None
            segs = assign_segments(test, edges, returning=ret)
            segs = pl.DataFrame({"user_session": ids}).join(segs, on="user_session", how="left", maintain_order="left")
            t = segment_table(y, scores, segs, probs, flag_share=args.flag_share, min_size=args.min_size)
            tables.append(t.with_columns(k=pl.lit(k)))
            print(f"   {len(scores)} models, {t.filter(pl.col('model') == t['model'][0]).height} segments")

        if ENGINE in scores:
            with stage(f"K={k}: engine CIs and {ENGINE} vs {RIVAL} per segment"):
                raw = []
                for dim in [c.removeprefix("seg_") for c in segs.columns if c.startswith("seg_")]:
                    lab = segs[f"seg_{dim}"].to_numpy()
                    for seg in sorted({x for x in lab if x is not None}):
                        mk = lab == seg
                        if mk.sum() < args.min_size or y[mk].sum() == 0:
                            continue
                        ci = bootstrap_ci(y[mk], scores[ENGINE][mk], "pr_auc", n_resamples=args.n_resamples, seed=seed)
                        row = {"k": k, "dimension": dim, "segment": seg, "engine": ENGINE,
                               "engine_pr_auc": ci["estimate"], "engine_low": ci["lower"], "engine_high": ci["upper"]}
                        if RIVAL in scores:
                            pt = paired_bootstrap(y[mk], scores[ENGINE][mk], scores[RIVAL][mk], "pr_auc",
                                                  n_resamples=args.n_resamples, seed=seed)
                            row |= {"rival": RIVAL, "diff": pt["diff"], "lower": pt["lower"], "upper": pt["upper"],
                                    "p_value": pt["p_value"]}
                        raw.append(row)
                if raw and "p_value" in raw[0]:
                    for r, ph in zip(raw, holm([r["p_value"] for r in raw])):
                        r["p_holm"], r["significant"] = ph, ph < alpha
                tests += raw

    if not tables:
        raise SystemExit("No saved test predictions with visit ids; run 03_run_experiments.py first.")
    met = pl.concat(tables, how="diagonal")
    disp = pl.concat([disparity(met.filter(pl.col("k") == k)).with_columns(k=pl.lit(k))
                      for k in met["k"].unique().sort()])
    met.write_csv(store.path("segment_metrics", "csv"))
    disp.write_csv(store.path("segment_disparity", "csv"))
    pl.DataFrame(tests).write_csv(store.path("segment_tests", "csv"))
    store.path("segment_summary", "json").write_text(json.dumps(
        {"mode": cfg["data"]["mode"], "flag_share": args.flag_share, "min_size": args.min_size, "engine": ENGINE,
         "rival": RIVAL, "user_type_available": used_user_type, "n_resamples": args.n_resamples}, indent=2))

    k0 = 5 if 5 in met["k"].to_list() else int(met["k"][0])
    print(f"\nEngine ({ENGINE}) per segment at K={k0}:")
    print_table(met.filter((pl.col("k") == k0) & (pl.col("model") == ENGINE)).select(
        "dimension", "segment", "n", pl.col("base_rate").round(3), pl.col("pr_auc").round(3),
        pl.col("pr_auc_lift").round(2), pl.col("calibration_gap").round(3), pl.col("flag_rate").round(3),
        pl.col("tpr").round(3), pl.col("fpr").round(3)).to_dicts())
    print(f"\nSpread per dimension ({ENGINE}, all K):")
    print_table(disp.filter(pl.col("model") == ENGINE).sort("k", "dimension").select(
        "k", "dimension", "worst_segment", "best_segment", pl.col("lift_ratio").round(2), pl.col("tpr_gap").round(3),
        pl.col("flag_rate_gap").round(3), pl.col("max_abs_calibration_gap").round(3)).to_dicts())
    if tests and "p_holm" in tests[0]:
        tt = pl.DataFrame(tests)
        print(f"\n{ENGINE} vs {RIVAL} per segment: significant after Holm in "
              f"{int(tt['significant'].sum())}/{tt.height} segment tests")
        print_table(tt.filter(pl.col("significant")).select("k", "dimension", "segment",
                                                            pl.col("diff").round(4), pl.col("p_holm").round(3)).to_dicts())


if __name__ == "__main__":
    main()
