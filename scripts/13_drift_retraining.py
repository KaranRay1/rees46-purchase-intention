"""Step 13: does the model decay over time, and does retraining fix it? (concept drift)

Needs steps 1-2 for the mode (``data/processed``). Uses LightGBM (fast, and statistically
tied with the best models) on the K-click features, and replays the seven months as a
deployment would: for each month from the 4th on, the model scores that month's visits
after being trained under one of four policies:

* ``frozen``: trained once on the first three months, never updated;
* ``expanding``: retrained every month on all earlier months;
* ``window_3``: retrained every month on the previous three months;
* ``window_1``: retrained every month on the previous month only.

Within each training set the last 15 % of visits (by time) are the validation set for early
stopping and the threshold, exactly like the main experiments. Every policy is run with
every project seed; the primary seed's scores are compared with ``frozen`` by a paired
bootstrap on the same visits (Holm-corrected per month).

Writes ``retraining`` and ``retraining_tests`` (CSV) to ``results/tables``.

Examples::

    python scripts/13_drift_retraining.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.config import processed_dir_for_k, set_seed  # noqa: E402
from src.evaluation.statistics import holm, paired_bootstrap  # noqa: E402
from src.features.preprocessing import load_feature_table, prepare_splits  # noqa: E402
from src.models.base import build_model  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402
from src.utils.notebook import prepared_data_status  # noqa: E402

POLICIES = ("frozen", "expanding", "window_3", "window_1")


def training_months(policy: str, months: list[str], i: int, first: int = 3) -> list[str]:
    """Months used to train the model that scores ``months[i]``."""
    if policy == "frozen":
        return months[:first]
    if policy == "expanding":
        return months[:i]
    if policy.startswith("window_"):
        w = int(policy.split("_")[1])
        return months[max(0, i - w):i]
    raise ValueError(policy)


def label_splits(table: pl.DataFrame, train_months: list[str], test_month: str, val_share: float = 0.15) -> pl.DataFrame:
    tr = table.filter(pl.col("month").is_in(train_months)).sort("session_start")
    cut = int(tr.height * (1 - val_share))
    tr = tr.with_columns(split=pl.when(pl.int_range(pl.len()) < cut).then(pl.lit("train")).otherwise(pl.lit("val")))
    te = table.filter(pl.col("month") == test_month).with_columns(split=pl.lit("test"))
    return pl.concat([tr, te])


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--ks", type=int, nargs="+", help="prefix lengths (default: sessionize.default_k)")
    p.add_argument("--model", default="lightgbm", help="model to retrain (a tabular model)")
    p.add_argument("--first", type=int, default=3, help="months in the frozen model's training set")
    p.add_argument("--n-resamples", type=int, default=500, help="bootstrap resamples for the paired tests")
    args = p.parse_args(argv)
    cfg = setup(args)
    mode = cfg["data"]["mode"]
    ks = args.ks or [int(cfg["sessionize"]["default_k"])]
    status = prepared_data_status(cfg)
    if any(status.get(k) != mode for k in ks):
        raise SystemExit(f"data/processed holds {status}, not {mode!r}; prepare it first.")
    store = ResultStore(cfg)
    store.tables_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in cfg["project"]["seeds"]]
    rows, tests = [], []
    for k in ks:
        starts = pl.read_parquet(processed_dir_for_k(cfg, k) / "sessions.parquet", columns=["user_session", "session_start"])
        table = (load_feature_table(cfg, k).join(starts, on="user_session", how="inner")
                 .with_columns(month=pl.col("session_start").dt.strftime("%Y-%m")))
        months = sorted(table["month"].unique().to_list())
        if len(months) <= args.first:
            raise SystemExit(f"Need more than {args.first} months of data; mode {mode} has {len(months)}.")
        for i in range(args.first, len(months)):
            m = months[i]
            with stage(f"K={k}, scoring {m}"):
                primary: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for policy in POLICIES:
                    tm = training_months(policy, months, i, args.first)
                    data, _ = prepare_splits(cfg, k, args.model, table=label_splits(table, tm, m).drop("month", "session_start"))
                    for seed in seeds:
                        set_seed(seed)
                        model = build_model(args.model, cfg, seed=seed, columns=data["train"].columns)
                        model.fit(data["train"].X, data["train"].y, data["val"].X, data["val"].y)
                        s = model.predict_proba(data["test"].X)
                        y = data["test"].y
                        if seed == seeds[0]:
                            primary[policy] = (y, s)
                        rows.append({"k": k, "month": m, "policy": policy, "seed": seed,
                                     "train_months": f"{tm[0]}..{tm[-1]}", "n_train": int(len(data["train"].y)),
                                     "n_test": int(len(y)), "buyer_rate": float(y.mean()),
                                     "pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
                                     "lift": float(average_precision_score(y, s) / y.mean())})
                y, s0 = primary["frozen"]
                raw = []
                for policy in POLICIES[1:]:
                    t = paired_bootstrap(y, primary[policy][1], s0, "pr_auc", n_resamples=args.n_resamples, seed=seeds[0])
                    raw.append({"k": k, "month": m, "policy": policy, "vs": "frozen", "diff": t["diff"],
                                "lower": t["lower"], "upper": t["upper"], "p_value": t["p_value"]})
                for r, ph in zip(raw, holm([r["p_value"] for r in raw])):
                    r["p_holm"], r["significant"] = ph, ph < float(cfg["evaluation"]["alpha"])
                tests += raw
    res = pl.DataFrame(rows)
    res.write_csv(store.path("retraining", "csv"))
    pl.DataFrame(tests).write_csv(store.path("retraining_tests", "csv"))
    summ = res.group_by("k", "month", "policy").agg(pl.col("pr_auc").mean().round(4), pl.col("pr_auc").std().round(4).alias("sd"),
                                                    pl.col("buyer_rate").first().round(4)).sort("k", "month", "policy")
    print("\nPR-AUC by month and policy (mean over seeds):")
    print_table(summ.to_dicts())
    print("\nRetraining vs frozen (primary seed, paired bootstrap):")
    print_table(pl.DataFrame(tests).select("k", "month", "policy", pl.col("diff").round(4), pl.col("p_holm").round(3),
                                           "significant").to_dicts())


if __name__ == "__main__":
    main()
