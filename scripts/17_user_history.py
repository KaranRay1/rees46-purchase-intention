"""Step 17: do user-level history features (G6) improve the models?

Needs steps 1-2 for the mode. G6 (``src/features/user_history.py``) summarises each user's
earlier sessions in a trailing window: how many, how many bought or carted, the smoothed
purchase rate, days since the last visit / purchase and a returning-user flag. Only sessions
that ended before the current one started count.

History is read from the full cleaned event log when it exists
(``data/full100/interim/events_full.parquet`` from ``run_full_scale.sh``), because the 10 %
modelling sample keeps only one session in ten and would miss most earlier visits. Otherwise
it falls back to ``data/interim/events_<mode>.parquet`` (pass ``--history-events`` to choose).

For every K, LightGBM is trained with every project seed on G1-G5 and on G1-G6 and compared on
the test weeks (paired bootstrap on the primary seed, Holm-corrected over K). The gain is also
split by new vs returning users.

Writes ``user_history_ablation``, ``user_history_tests``, ``user_history_segments`` (CSV) and
``user_history_summary`` (JSON) to ``results/tables``.

Examples::

    python scripts/17_user_history.py --mode full --set data.session_sample_fraction=0.1
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

from src.config import PROJECT_ROOT as ROOT, feature_list, processed_dir_for_k, resolve_path, set_seed  # noqa: E402
from src.data.sessionize import TARGET  # noqa: E402
from src.evaluation.statistics import holm, paired_bootstrap  # noqa: E402
from src.features.build_features import load_feature_table  # noqa: E402
from src.features.preprocessing import TabularPreprocessor  # noqa: E402
from src.features.user_history import HISTORY_FEATURES, history_features, session_table  # noqa: E402
from src.models.base import build_model  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402
from src.utils.notebook import prepared_data_status  # noqa: E402


def lgbm_inputs(cfg, table: pl.DataFrame, extra: list[str]):
    """Train/val/test matrices for LightGBM (raw profile) on G1-G5 plus ``extra`` numeric columns."""
    pre = TabularPreprocessor(features=feature_list(cfg) + extra, categorical=cfg["features"]["categorical"],
                              log1p=False, scale=False, categorical_encoding="ordinal",
                              top_n=cfg["features"]["top_n_categories"])
    parts = {s: table.filter(pl.col("split") == s) for s in ("train", "val", "test")}
    pre.fit(parts["train"])
    out = {s: (pre.transform(d).to_numpy().astype(np.float32), d[TARGET].to_numpy().astype(np.int8))
           for s, d in parts.items()}
    return out, pre.output_columns, parts["test"]


def history_source(cfg, given: str | None) -> Path:
    if given:
        return Path(given)
    full = ROOT / "data" / "full100" / "interim" / "events_full.parquet"
    if cfg["data"]["mode"] == "full" and full.exists():
        return full
    return Path(resolve_path(cfg, "interim_dir")) / f"events_{cfg['data']['mode']}.parquet"


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--ks", type=int, nargs="+", help="prefix lengths (default: all prepared)")
    p.add_argument("--window-days", type=int, default=30, help="trailing history window")
    p.add_argument("--history-events", help="cleaned events Parquet to read history from")
    p.add_argument("--n-resamples", type=int, default=1000)
    args = p.parse_args(argv)
    cfg = setup(args)
    mode = cfg["data"]["mode"]
    ks = args.ks or [int(k) for k in cfg["sessionize"]["prefix_k"]]
    status = prepared_data_status(cfg)
    if any(status.get(k) != mode for k in ks):
        raise SystemExit(f"data/processed holds {status}, not {mode!r}; prepare it first.")
    store = ResultStore(cfg)
    seeds = [int(s) for s in cfg["project"]["seeds"]]
    alpha = float(cfg["evaluation"]["alpha"])
    src = history_source(cfg, args.history_events)

    targets = pl.concat([
        pl.read_parquet(processed_dir_for_k(cfg, k) / "sessions.parquet",
                        columns=["user_session", "user_id", "session_start"]) for k in ks
    ]).unique("user_session")
    with stage(f"Session table for history ({src.relative_to(ROOT) if src.is_relative_to(ROOT) else src})"):
        cache = Path(resolve_path(cfg, "interim_dir")) / "user_history"
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / f"sessions_{src.stem}_{src.stat().st_size}.parquet"
        if not f.exists():
            users = targets.select("user_id").unique()
            ev = pl.scan_parquet(src).join(users.lazy(), on="user_id", how="semi")
            session_table(ev).sink_parquet(f)
        sessions = pl.read_parquet(f)
        print(f"   {sessions.height:,} sessions of {targets['user_id'].n_unique():,} users")
    with stage(f"G6 features ({args.window_days}-day window) for {targets.height:,} sessions"):
        hist = history_features(targets, sessions, window_days=args.window_days)
        del sessions

    rows, raw, seg = [], [], []
    returning_share = {}
    for k in ks:
        with stage(f"K={k}: LightGBM on G1-G5 vs G1-G6"):
            table = load_feature_table(cfg, k).join(hist, on="user_session", how="left")
            primary: dict[str, np.ndarray] = {}
            for name, extra in {"base": [], "history": HISTORY_FEATURES}.items():
                data, cols, test = lgbm_inputs(cfg, table, extra)
                for seed in seeds:
                    set_seed(seed)
                    m = build_model("lightgbm", cfg, seed=seed, columns=cols)
                    m.fit(data["train"][0], data["train"][1], data["val"][0], data["val"][1])
                    s = m.predict_proba(data["test"][0])
                    if seed == seeds[0]:
                        primary[name] = s
                    rows.append({"k": k, "feature_set": name, "seed": seed, "n_inputs": len(cols),
                                 "val_pr_auc": float(average_precision_score(data["val"][1], m.predict_proba(data["val"][0]))),
                                 "test_pr_auc": float(average_precision_score(data["test"][1], s))})
            y = data["test"][1]
            t = paired_bootstrap(y, primary["history"], primary["base"], "pr_auc",
                                 n_resamples=args.n_resamples, seed=seeds[0])
            raw.append({"k": k, "feature_set": "history", "vs": "base", "diff": t["diff"], "lower": t["lower"],
                        "upper": t["upper"], "p_value": t["p_value"]})
            ret = test["uh_is_returning"].to_numpy() == 1
            returning_share[k] = float(ret.mean())
            for label, mask in (("new", ~ret), ("returning", ret)):
                yy = y[mask]
                seg.append({"k": k, "segment": label, "sessions": int(mask.sum()),
                            "positive_rate": float(yy.mean()) if yy.size else None,
                            "pr_auc_base": float(average_precision_score(yy, primary["base"][mask])) if yy.sum() else None,
                            "pr_auc_history": float(average_precision_score(yy, primary["history"][mask])) if yy.sum() else None})
    for r, ph in zip(raw, holm([r["p_value"] for r in raw])):
        r["p_holm"], r["significant"] = ph, ph < alpha

    res = pl.DataFrame(rows)
    res.write_csv(store.path("user_history_ablation", "csv"))
    pl.DataFrame(raw).write_csv(store.path("user_history_tests", "csv"))
    segs = pl.DataFrame(seg).with_columns(gain=pl.col("pr_auc_history") - pl.col("pr_auc_base"))
    segs.write_csv(store.path("user_history_segments", "csv"))
    store.path("user_history_summary", "json").write_text(json.dumps(
        {"mode": mode, "window_days": args.window_days, "history_source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
         "features": HISTORY_FEATURES, "returning_share_test": returning_share}, indent=2))

    print("\nTest PR-AUC (mean over seeds):")
    print_table(res.group_by("k", "feature_set").agg(pl.col("test_pr_auc").mean().round(4),
                                                     pl.col("test_pr_auc").std().round(4).alias("sd"))
                .sort("k", "feature_set").to_dicts())
    print("\nG1-G6 vs G1-G5 (primary seed):")
    print_table(pl.DataFrame(raw).select("k", pl.col("diff").round(4), pl.col("lower").round(4), pl.col("upper").round(4),
                                         pl.col("p_holm").round(3), "significant").to_dicts())
    print("\nBy user type (primary seed):")
    print_table(segs.select("k", "segment", "sessions", pl.col("positive_rate").round(3), pl.col("pr_auc_base").round(4),
                            pl.col("pr_auc_history").round(4), pl.col("gain").round(4)).to_dicts())


if __name__ == "__main__":
    main()
