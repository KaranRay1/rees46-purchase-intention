"""Step 6 — RQ5: which behaviours drive predictions, do model families agree,
and what characterises their errors?

Needs step 3 (``03_run_experiments.py``). Nothing is retrained: for each
model in ``explainability.models`` the primary-seed RQ1 model is loaded and
explained on a random sample of ``n_explain_samples`` test sessions, with
``n_background`` training sessions as background where the method needs it
(``src.explainability.shap_analysis``: TreeSHAP, LinearSHAP, DeepSHAP, TabNet
attention).

Per K (default: ``sessionize.default_k``; ``--ks`` for more):

* **global importance** — feature level (one-hot columns summed back) and
  group level (G1–G5), share of mean |attribution| per model, and Spearman
  rank agreement between models;
* **H5** — is the cart-signal group (G2) the most important group for every
  model family?
* **error analysis** — sessions split into TP / FP / FN / TN at each model's
  tuned threshold: mean attribution per feature in each group, a profile of
  observable session traits per group (cart presence, first-cart position,
  item variety, hour, total session length), and metrics by total
  session-length cohort (``evaluation.cohorts.session_length_bins``);
* **figures** — beeswarm and waterfalls (most confident TP / FP / FN / TN) per
  model, and importance comparisons across models (features and groups).

Total session length is analysis-only information (it is not known at
prediction time) and is used only to describe errors, never as an input.

Examples::

    python scripts/06_explain.py --mode synthetic
    python scripts/06_explain.py --mode dev --ks 3 5 10 --set explainability.n_explain_samples=5000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402
from _experiment import InputCache, latest_runs, primary_predictions, primary_run  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.config import feature_list, resolve_path  # noqa: E402
from src.data.split import load_split_sessions  # noqa: E402
from src.evaluation.metrics import metrics_by_group  # noqa: E402
from src.explainability import shap_analysis as sa  # noqa: E402
from src.models.base import BaseModel  # noqa: E402
from src.utils.results_logger import start_run  # noqa: E402

SOURCE = "rq1_models"
EXPERIMENT = "rq5_explain"
CART_GROUP = "G2_cart_signals"
PROFILE_FEATURES = ["has_cart", "first_cart_position", "cart_count",
                    "unique_products_count", "unique_categories_count", "hour_of_day"]


# =============================================================================
# Per-model explanation
# =============================================================================
def explain_one(cfg: Mapping[str, Any], cache: InputCache, runs: pl.DataFrame, k: int, model_name: str,
                sessions_test: pl.DataFrame, out_fig: Path) -> dict[str, Any]:
    part = runs.filter((pl.col("k") == k) & (pl.col("model") == model_name))
    prim = primary_run(cfg, part)
    model = BaseModel.load(resolve_path(cfg, "runs_dir") / prim["run_id"][0] / "model.pkl")
    data, pre, _, columns = cache.get(k, model_name)
    test, train = data["test"], data["train"]
    if sorted(test.session_ids) != primary_predictions(cfg, part)["user_session"].to_list():
        raise RuntimeError(f"K={k} {model_name}: rebuilt test inputs do not match the RQ1 run's sessions")

    ex = cfg["explainability"]
    seed = cfg["project"]["seeds"][0]
    rows = sa.sample_rows(len(test.y), int(ex["n_explain_samples"]), seed=seed)
    bg = train.X[sa.sample_rows(len(train.y), int(ex["n_background"]), seed=seed + 1)]
    X = test.X[rows]
    y = test.y[rows]
    ids = [test.session_ids[i] for i in rows]

    raw = sa.explain_model(model, X, columns, bg)
    feats = sa.aggregate_onehot(raw, feature_list(cfg))
    groups = sa.aggregate_groups(feats, cfg["features"]["groups"])

    scores = model.predict_proba(X)
    pred = (scores >= model.threshold).astype(int)
    err = sa.error_groups(y, pred)

    # figures
    tag = f"{model_name}_k{k}_{cfg['data']['mode']}"
    figs = [sa.plot_beeswarm(feats, out_fig / f"rq5_beeswarm_{tag}.png"),
            sa.plot_importance_bar(feats, out_fig / f"rq5_importance_{tag}.png")]
    if np.isfinite(feats.base_value):
        for g, idxs in sa.pick_examples(scores, err, n_per_group=int(ex["n_local_examples"])).items():
            for j, i in enumerate(idxs):
                figs.append(sa.plot_waterfall(
                    feats, i, out_fig / f"rq5_waterfall_{tag}_{g}_{j + 1}.png",
                    title=f"{model_name}, K={k}: {g.replace('_', ' ')} (score {scores[i]:.3f})"))

    # error analysis
    by_err = sa.shap_by_error_group(feats, err).with_columns(pl.lit(model_name).alias("model"), pl.lit(k).alias("k"))
    sample = pl.DataFrame({"user_session": ids, "y_true": y, "score": scores, "group": err}).join(
        sessions_test.select("user_session", "n_events_total"), on="user_session", how="left")
    return {"model": model_name, "k": k, "feature_attr": feats, "group_attr": groups, "method": raw.method,
            "scale": raw.scale, "errors_by_attr": by_err, "sample": sample,
            "threshold": model.threshold, "source_run_id": prim["run_id"][0], "figures": figs}


def length_bins(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Label each value with its ``[low, high)`` bin, e.g. ``"[10, 20)"`` or ``"[50, inf)"``."""
    edges = list(edges)
    idx = np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)
    names = [f"[{edges[i]:g}, {edges[i + 1]:g})" for i in range(len(edges) - 1)]
    return np.array([names[i] for i in idx])


def error_profile(cfg: Mapping[str, Any], k: int, sample: pl.DataFrame) -> pl.DataFrame:
    """Mean observable traits per error group, from the unscaled feature table."""
    from src.features.build_features import load_feature_table

    table = load_feature_table(cfg, k, "test").select("user_session", *PROFILE_FEATURES)
    joined = sample.join(table, on="user_session", how="left")
    return (joined.group_by("group")
            .agg(pl.len().alias("n"), pl.col("score").mean().alias("mean_score"),
                 pl.col("n_events_total").mean().alias("mean_total_events"),
                 *[pl.col(f).cast(pl.Float64).mean().alias(f"mean_{f}") for f in PROFILE_FEATURES])
            .sort("group"))


# =============================================================================
# Run
# =============================================================================
def explain_k(cfg: Mapping[str, Any], models: Sequence[str], k: int) -> dict[str, Any]:
    mode = cfg["data"]["mode"]
    tables_dir = resolve_path(cfg, "tables_dir")
    fig_dir = resolve_path(cfg, "figures_dir")
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs = latest_runs(cfg, SOURCE)
    cache = InputCache(cfg)
    sessions_test = load_split_sessions(cfg, k, "test").select("user_session", "n_events_total").collect()

    results = {}
    for m in models:
        if runs.filter((pl.col("k") == k) & (pl.col("model") == m)).height == 0:
            print(f"   skip K={k} {m}: no RQ1 run")
            continue
        r = explain_one(cfg, cache, runs, k, m, sessions_test, fig_dir)
        results[m] = r
        print(f"   K={k:<2} {m:<20} {r['method']:<16} explained {len(r['sample'])} sessions", flush=True)
    if not results:
        raise RuntimeError(f"Nothing to explain for K={k}.")

    wide_f, corr_f = sa.compare_importance({m: r["feature_attr"] for m, r in results.items()})
    wide_g, corr_g = sa.compare_importance({m: r["group_attr"] for m, r in results.items()})
    sa.plot_importance_comparison(wide_f, fig_dir / f"rq5_compare_features_k{k}_{mode}.png", top=12)
    sa.plot_importance_comparison(wide_g, fig_dir / f"rq5_compare_groups_k{k}_{mode}.png", top=len(wide_g))

    by_err = pl.concat([r["errors_by_attr"] for r in results.values()])
    profiles = pl.concat([error_profile(cfg, k, r["sample"]).with_columns(pl.lit(m).alias("model"))
                          for m, r in results.items()])
    bins = cfg["evaluation"]["cohorts"]["session_length_bins"]
    cohort_rows = []
    for m, r in results.items():
        s = r["sample"].filter(pl.col("n_events_total").is_not_null())
        labels = length_bins(s["n_events_total"].to_numpy(), bins)
        coh = metrics_by_group(s["y_true"].to_numpy(), s["score"].to_numpy(), labels, r["threshold"])
        cohort_rows.append(coh.with_columns(pl.lit(m).alias("model")))
    cohorts = pl.concat(cohort_rows)

    wide_f.write_csv(tables_dir / f"rq5_importance_features_k{k}_{mode}.csv")
    wide_g.write_csv(tables_dir / f"rq5_importance_groups_k{k}_{mode}.csv")
    pl.concat([corr_f.with_columns(level=pl.lit("feature")), corr_g.with_columns(level=pl.lit("group"))]) \
        .write_csv(tables_dir / f"rq5_rank_agreement_k{k}_{mode}.csv")
    by_err.write_csv(tables_dir / f"rq5_attribution_by_error_group_k{k}_{mode}.csv")
    profiles.write_csv(tables_dir / f"rq5_error_profile_k{k}_{mode}.csv")
    cohorts.write_csv(tables_dir / f"rq5_metrics_by_session_length_k{k}_{mode}.csv")

    for m, r in results.items():
        with start_run(cfg, EXPERIMENT, m, k=k, seed=cfg["project"]["seeds"][0], feature_set="all") as run:
            run.log_params({"source_run_id": r["source_run_id"], "method": r["method"], "scale": r["scale"],
                            "n_explained": len(r["sample"]), "n_background": cfg["explainability"]["n_background"]})
            imp_g = sa.global_importance(r["group_attr"])
            run.log_metrics({"group_share": dict(zip(imp_g["feature"], imp_g["share"])),
                             "top_group": imp_g["feature"][0],
                             "top_feature": sa.global_importance(r["feature_attr"])["feature"][0]})
            run.log_artifact("feature_importance", sa.global_importance(r["feature_attr"]))
            run.log_artifact("group_importance", imp_g)

    return {"k": k, "results": results, "features": wide_f, "groups": wide_g,
            "agreement_features": corr_f, "agreement_groups": corr_g,
            "by_error": by_err, "error_profile": profiles, "cohorts": cohorts}


def hypothesis_h5(per_k: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    ev = []
    for k, res in per_k.items():
        wide = res["groups"]
        for m in [c for c in wide.columns if c not in ("feature", "mean_share")]:
            top = wide.sort(m, descending=True)
            ev.append({"k": k, "model": m, "top_group": top["feature"][0], "top_share": float(top[m][0]),
                       "cart_group_share": float(wide.filter(pl.col("feature") == CART_GROUP)[m][0])})
    n_ok = sum(e["top_group"] == CART_GROUP for e in ev)
    return {"H5_cart_signals_rank_highest_for_every_family": {
        "verdict": "not tested" if not ev else "supported" if n_ok == len(ev) else
                   "partly supported" if n_ok else "not supported",
        "cart_group_top": f"{n_ok}/{len(ev)}",
        "measure": "share of mean |attribution| summed over feature group G2 (exact for SHAP); "
                   "TabNet uses attention shares",
        "evidence": ev}}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="subset of explainability.models")
    parser.add_argument("--ks", nargs="+", type=int, help="prefix lengths (default: sessionize.default_k)")
    args = parser.parse_args(argv)
    cfg = setup(args)
    models = args.models or cfg["explainability"]["models"]
    bad = [m for m in models if m not in cfg["explainability"]["models"]]
    if bad:
        parser.error(f"models not in explainability.models: {bad}")
    ks = args.ks or [cfg["sessionize"]["default_k"]]
    mode = cfg["data"]["mode"]
    print(f"RQ5 explainability — mode={mode}, models={models}, K={ks}, "
          f"{cfg['explainability']['n_explain_samples']} test sessions per model")

    per_k: dict[int, dict[str, Any]] = {}
    for k in ks:
        with stage(f"Explain K={k}"):
            res = explain_k(cfg, models, k)
            per_k[k] = res
            print("   importance by feature group (share of mean |attribution|):")
            print_table(res["groups"].to_dicts())
            print("   rank agreement between models (groups):")
            print_table(res["agreement_groups"].to_dicts())
            prof = res["error_profile"].filter(pl.col("model") == models[-1] if models[-1] in res["results"]
                                               else pl.lit(True))
            print(f"   error profile ({prof['model'][0]}):")
            print_table(prof.select("group", "n", "mean_score", "mean_has_cart", "mean_first_cart_position",
                                    "mean_unique_products_count", "mean_total_events").to_dicts())

    hyp = hypothesis_h5(per_k)
    out = resolve_path(cfg, "tables_dir") / f"rq5_hypothesis_{mode}.json"
    out.write_text(json.dumps(hyp, indent=2, default=float))
    h = hyp["H5_cart_signals_rank_highest_for_every_family"]
    print(f"\nH5: {h['verdict']} (cart group ranked first in {h['cart_group_top']})")
    print(f"tables -> {resolve_path(cfg, 'tables_dir')}\nfigures -> {resolve_path(cfg, 'figures_dir')}")
    return {"per_k": per_k, "hypothesis": hyp}


if __name__ == "__main__":
    main()
