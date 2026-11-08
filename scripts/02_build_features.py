"""Step 2 — features: G1–G5 tabular features and GRU sequences for every K.

Needs step 1 (``01_prepare_data.py``). For each K in ``sessionize.prefix_k``:

1. builds ``data/processed/k{K}/features.parquet`` (``src.features.build_features``);
2. checks the table: one row per session, no missing values, no
   analysis-only columns, and view + cart + remove = K;
3. builds the GRU sequences in memory (``src.features.sequences``) and checks
   they cover exactly the same sessions, in the same order, as the tabular
   data (required for paired model comparisons);
4. computes a **buyer vs non-buyer profile** on the training split:
   mean of every numeric feature for each class and the standardised mean
   difference (SMD = difference ÷ pooled standard deviation) — a quick,
   model-free view of which behaviours separate buyers, for the EDA section.

Writes ``results/tables/feature_summary_<mode>.json`` and
``results/tables/feature_profile_<mode>_k{K}.csv``.

Examples::

    python scripts/02_build_features.py --mode synthetic
    python scripts/02_build_features.py --mode dev --skip-sequences
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.config import feature_list, resolve_path  # noqa: E402
from src.data.sessionize import ANALYSIS_ONLY_COLUMNS, TARGET  # noqa: E402
from src.features import build_features  # noqa: E402
from src.features.preprocessing import prepare_splits  # noqa: E402
from src.features.sequences import prepare_sequence_splits  # noqa: E402


def check_table(table: pl.DataFrame, cfg: dict, k: int) -> list[str]:
    """Return a list of problems (empty = all good)."""
    problems = []
    feats = feature_list(cfg)
    if not table["user_session"].is_unique().all():
        problems.append("duplicate sessions")
    nulls = {c: n for c, n in zip(feats, table.select(feats).null_count().row(0)) if n}
    if nulls:
        problems.append(f"missing values: {nulls}")
    leaked = set(table.columns) & (set(ANALYSIS_ONLY_COLUMNS) - {TARGET})
    if leaked:
        problems.append(f"analysis-only columns present: {sorted(leaked)}")
    if not (table["view_count"] + table["cart_count"] + table["remove_count"] == k).all():
        problems.append("view + cart + remove != K for some sessions")
    return problems


def buyer_profile(train: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Per numeric feature: mean for buyers / non-buyers and standardised mean difference."""
    numeric = [f for f in feature_list(cfg) if f not in cfg["features"]["categorical"]]
    rows = []
    for f in numeric:
        x = train[f].cast(pl.Float64)
        if f in cfg["preprocessing"]["log1p_features"]:
            x = x.clip(lower_bound=0).log1p()      # same scale the linear/neural models see
        y = train[TARGET].to_numpy() == 1
        a, b = x.to_numpy()[y], x.to_numpy()[~y]
        pooled = math.sqrt(((a.var(ddof=1) if len(a) > 1 else 0) + (b.var(ddof=1) if len(b) > 1 else 0)) / 2)
        smd = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
        rows.append({"feature": f, "log1p": f in cfg["preprocessing"]["log1p_features"],
                     "mean_buyers": float(a.mean()) if len(a) else math.nan,
                     "mean_non_buyers": float(b.mean()) if len(b) else math.nan,
                     "smd": float(smd)})
    return pl.DataFrame(rows).sort(pl.col("smd").abs(), descending=True)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--skip-sequences", action="store_true", help="do not build/check GRU sequences")
    args = parser.parse_args(argv)
    cfg = setup(args)
    mode = cfg["data"]["mode"]
    ks = cfg["sessionize"]["prefix_k"]
    tables_dir = resolve_path(cfg, "tables_dir")
    tables_dir.mkdir(parents=True, exist_ok=True)
    print(f"Feature building — mode={mode}, K={ks}, {len(feature_list(cfg))} features")

    summary: dict[str, Any] = {"mode": mode, "features": feature_list(cfg), "k": {}}

    with stage("1/3 Build tabular features"):
        paths = build_features.run(cfg)
        for k, p in paths.items():
            print(f"   K={k}: {p}")

    with stage("2/3 Check tables and sequences"):
        rows = []
        for k in ks:
            table = build_features.load_feature_table(cfg, k)
            problems = check_table(table, cfg, k)
            if problems:
                raise RuntimeError(f"Feature table for K={k} failed checks: {problems}")
            counts = {s: table.filter(pl.col("split") == s).height for s in ("train", "val", "test")}
            entry: dict[str, Any] = {"sessions": counts, "checks": "passed"}

            if not args.skip_sequences:
                seq, enc = prepare_sequence_splits(cfg, k)
                tab, _ = prepare_splits(cfg, k, "lightgbm", table=table)
                for s in ("train", "val", "test"):
                    if seq[s].session_ids != tab[s].session_ids or not np.array_equal(seq[s].y, tab[s].y):
                        raise RuntimeError(f"K={k} {s}: GRU sequences and tabular rows are not aligned")
                entry["sequence_shape_train"] = list(seq["train"].cat.shape[:2])
                entry["embedding_spec"] = {f: list(v) for f, v in enc.embedding_spec().items()}

            summary["k"][str(k)] = entry
            rows.append({"K": k, **counts, "sequences": "aligned" if not args.skip_sequences else "skipped",
                         "checks": "passed"})
        print_table(rows)

    with stage("3/3 Buyer vs non-buyer profile (training split)"):
        for k in ks:
            train = build_features.load_feature_table(cfg, k, "train")
            prof = buyer_profile(train, cfg)
            prof.write_csv(tables_dir / f"feature_profile_{mode}_k{k}.csv")
            summary["k"][str(k)]["top_smd"] = prof.head(5).select("feature", "smd").to_dicts()
        k0 = cfg["sessionize"]["default_k"]
        print(f"   K={k0}, strongest differences (|SMD| ≥ 0.5 is a medium effect, ≥ 0.8 large):")
        prof = pl.read_csv(tables_dir / f"feature_profile_{mode}_k{k0}.csv")
        print_table(prof.head(8).to_dicts(), ["feature", "mean_buyers", "mean_non_buyers", "smd"])

    out = tables_dir / f"feature_summary_{mode}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary -> {out}")
    return summary


if __name__ == "__main__":
    main()
