"""Step 18: which model earns the most? Model comparison in money, not just PR-AUC.

Needs step 3 (saved test predictions) for the mode. Each model's scores drive the same discount
campaign as the dashboard's Campaign planner: offer a discount to everyone except visits scored
at or above a cutoff (they buy anyway). See ``src/evaluation/business_value.py``.

* The cutoff is chosen on a random half of the test visits and the profit measured on the other.
* Profit is scaled to **a month of the whole shop** (test days -> 30 days, sample -> all visits).
* For every K, each model is compared with the most profitable one (paired bootstrap, Holm), and
  with offering the discount to everyone.
* A sensitivity grid (discount x uplift) shows when the engine's ranking pays off at all.

The marketing inputs are assumptions (defaults: 10 % discount, 2 % of non-buyers converted by the
offer, order value = revenue per buying visit in the data, no contact cost).

Writes ``business_value``, ``business_value_tests``, ``business_value_sensitivity`` (CSV) and
``business_value_summary`` (JSON) to ``results/tables``.

Examples::

    python scripts/18_business_value.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import polars as pl  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.evaluation.business_value import compare_models, paired_diff  # noqa: E402
from src.evaluation.statistics import holm  # noqa: E402
from src.serving.campaign import Offer  # noqa: E402
from src.serving.catalog import daily_profile  # noqa: E402
from src.serving.results import ResultStore, test_predictions  # noqa: E402

MODELS = ["rule_based", "logistic_regression", "random_forest", "lightgbm", "mlp", "tabnet", "gru"]
ENGINE = "gru"   # the deployed model (fastest of the top tier)


def monthly_scale(summary: dict, k: int) -> tuple[float, float, float]:
    """(scale, test days, sample share): multiply a test-period value to get a month of the whole shop."""
    sp = (summary.get("splits") or {}).get(str(k), {}).get("test")
    days = 1.0
    if sp:
        days = max((datetime.fromisoformat(sp["last_start"]) - datetime.fromisoformat(sp["first_start"])
                    ).total_seconds() / 86400, 1.0)
    clean = summary.get("cleaning") or {}
    sample = clean["sessions_out"] / clean["sessions_in"] if clean.get("sessions_in") else 1.0
    return (30 / days) / sample, days, sample


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--ks", type=int, nargs="+", help="prefix lengths (default: config)")
    p.add_argument("--discount", type=float, default=0.10)
    p.add_argument("--uplift", type=float, default=0.02, help="share of offered non-buyers the offer converts")
    p.add_argument("--aov", type=float, help="average order value (default: revenue per buying visit)")
    p.add_argument("--contact-cost", type=float, default=0.0)
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--sensitivity-k", type=int, default=5)
    args = p.parse_args(argv)
    cfg = setup(args)
    store = ResultStore(cfg)
    ks = args.ks or [int(k) for k in cfg["sessionize"]["prefix_k"]]
    seed = int(cfg["project"]["seeds"][0])
    alpha = float(cfg["evaluation"]["alpha"])
    summary = store.json("data_summary") or {}
    rq1 = store.csv("rq1_model_comparison")
    aov = args.aov
    if aov is None:
        daily = daily_profile(cfg)
        aov = float(daily["revenue"].sum() / max(daily["buying_sessions"].sum(), 1)) if daily is not None else 300.0
    offer = Offer(aov=aov, discount=args.discount, uplift=args.uplift, contact_cost=args.contact_cost)
    print(f"offer: {offer.discount:.0%} discount, {offer.uplift:.1%} uplift, order value ${offer.aov:,.0f}, "
          f"contact ${offer.contact_cost:.2f}")

    rows, tests, spear, scales, preds = [], [], {}, {}, {}
    for k in ks:
        with stage(f"K={k}: profit of every model"):
            scores, y = {}, None
            for m in MODELS:
                pr = test_predictions(cfg, k, m)
                if pr is None or not pr.height:
                    continue
                if y is None:
                    y, ref = pr["y_true"].to_numpy(), pr["user_session"] if "user_session" in pr.columns else None
                elif ref is not None and "user_session" in pr.columns and not (pr["user_session"] == ref).all():
                    raise SystemExit(f"K={k}: {m} predictions are not in the same visit order")
                scores[m] = pr["score"].to_numpy()
            if not scores:
                print("   no saved test predictions; skipped")
                continue
            preds[k] = (y, scores)
            scale, days, sample = monthly_scale(summary, k)
            # the evaluation half is ~half the test visits, so x2 before the monthly scale
            scale_half = 2 * scale
            scales[k] = {"scale": scale, "test_days": days, "sample": sample}
            res = compare_models(y, scores, offer, n_resamples=args.n_resamples, seed=seed)
            pra = {}
            if rq1 is not None:
                pra = {r["model"]: r["pr_auc_mean"] for r in rq1.filter(pl.col("k") == k).iter_rows(named=True)}
            best = max(res["rows"], key=lambda r: r["net"])["model"]
            for r in res["rows"]:
                rows.append({"k": k, **r, "net": r["net"] * scale_half, "net_low": r["net_low"] * scale_half,
                             "net_high": r["net_high"] * scale_half, "blanket": res["blanket"] * scale_half,
                             "gain_vs_blanket": (r["net"] - res["blanket"]) * scale_half,
                             "pr_auc": pra.get(r["model"]), "best": r["model"] == best})
            raw = []
            for r in res["rows"]:
                m = r["model"]
                if m != best:
                    t = paired_diff(res["draws"][m], res["draws"][best], r["net"],
                                    next(x["net"] for x in res["rows"] if x["model"] == best))
                    raw.append({"k": k, "model": m, "vs": best, **t})
            for r, ph in zip(raw, holm([r["p_value"] for r in raw])):
                r["p_holm"], r["significant"] = ph, ph < alpha
            if ENGINE in res["draws"]:
                eng = next(x for x in res["rows"] if x["model"] == ENGINE)
                t = paired_diff(res["draws"][ENGINE], res["blanket_draws"], eng["net"], res["blanket"])
                raw.append({"k": k, "model": ENGINE, "vs": "offer_to_everyone", **t, "p_holm": t["p_value"],
                            "significant": t["p_value"] < alpha})
            for r in raw:
                for c in ("diff", "lower", "upper"):
                    r[c] *= scale_half
            tests += raw
            both = [(pra[r["model"]], r["net"]) for r in res["rows"] if r["model"] in pra]
            if len(both) >= 3:
                spear[k] = float(spearmanr(*zip(*both)).statistic)

    sens = []
    sk = args.sensitivity_k if args.sensitivity_k in preds else (next(iter(preds)) if preds else None)
    if sk is not None:
        y, scores = preds[sk]
        model = ENGINE if ENGINE in scores else next(iter(scores))
        with stage(f"Sensitivity at K={sk} ({model}): discount x uplift"):
            scale_half = 2 * scales[sk]["scale"]
            for d in (0.05, 0.10, 0.15, 0.20, 0.30):
                for u in (0.005, 0.01, 0.02, 0.03, 0.05):
                    o = Offer(aov=aov, discount=d, uplift=u, contact_cost=args.contact_cost)
                    r = compare_models(y, {model: scores[model]}, o, n_resamples=1, seed=seed)
                    row = r["rows"][0]
                    sens.append({"k": sk, "model": model, "discount": d, "uplift": u,
                                 "share_skipped": row["share_skipped"], "net": row["net"] * scale_half,
                                 "blanket": r["blanket"] * scale_half,
                                 "gain_vs_blanket": (row["net"] - r["blanket"]) * scale_half})

    if not rows:
        raise SystemExit("No saved test predictions for any K; run 03_run_experiments.py first.")
    res_df = pl.DataFrame(rows)
    res_df.write_csv(store.path("business_value", "csv"))
    pl.DataFrame(tests).write_csv(store.path("business_value_tests", "csv"))
    if sens:
        pl.DataFrame(sens).write_csv(store.path("business_value_sensitivity", "csv"))
    store.path("business_value_summary", "json").write_text(json.dumps(
        {"mode": cfg["data"]["mode"], "offer": {"aov": aov, "discount": args.discount, "uplift": args.uplift,
                                                "contact_cost": args.contact_cost},
         "engine": ENGINE, "units": "net value per month for the whole shop", "scales": scales,
         "spearman_pr_auc_vs_profit": spear, "n_resamples": args.n_resamples,
         "method": "cutoff chosen on a random half of the test visits, profit measured on the other half"}, indent=2))

    print("\nNet value per month, whole shop (cutoff chosen on half A, measured on half B):")
    print_table(res_df.sort("k", "net", descending=[False, True]).select(
        "k", "model", pl.col("pr_auc").round(4), pl.col("share_skipped").round(3),
        (pl.col("net") / 1e3).round(0).alias("net_k$"), (pl.col("net_low") / 1e3).round(0).alias("low_k$"),
        (pl.col("net_high") / 1e3).round(0).alias("high_k$"), (pl.col("gain_vs_blanket") / 1e3).round(0).alias("vs_all_k$"),
    ).to_dicts())
    print("\nVs the most profitable model (and the engine vs offering to everyone):")
    print_table(pl.DataFrame(tests).select("k", "model", "vs", (pl.col("diff") / 1e3).round(1).alias("diff_k$"),
                                           pl.col("p_holm").round(3), "significant").to_dicts())
    print(f"\nSpearman (PR-AUC rank vs profit rank): {json.dumps({k: round(v, 2) for k, v in spear.items()})}")
    if sens:
        print("\nSensitivity: gain over offering to everyone (k$/month):")
        sd = pl.DataFrame(sens).with_columns((pl.col("gain_vs_blanket") / 1e3).round(0))
        print_table(sd.pivot("uplift", index="discount", values="gain_vs_blanket").to_dicts())


if __name__ == "__main__":
    main()
