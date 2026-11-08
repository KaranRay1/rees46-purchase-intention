"""Step 1 — data preparation: raw CSV → cleaned events → session prefixes → splits.

Runs the whole data stage for every K in ``sessionize.prefix_k``:

1. ``src.data.loader``     REES46 CSV → typed Parquet (cached per month) → cleaned events
2. ``src.data.sessionize`` first-K-event prefixes and leakage-safe targets
3. ``src.data.split``      chronological 70/15/15 split with shared cut-offs

and writes one summary for the report's data section to
``results/tables/data_summary_<mode>.json`` (cleaning counts, eligibility
funnel per K, split sizes and positive rates, cut-off dates).

Examples::

    python scripts/01_prepare_data.py --mode synthetic
    python scripts/01_prepare_data.py --mode dev                 # 2019-Oct.csv in data/raw/
    python scripts/01_prepare_data.py --mode full --no-report    # all months, skip per-step counts
    python scripts/01_prepare_data.py --mode dev --overwrite     # after changing a cleaning rule
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.data import sessionize, split  # noqa: E402
from src.data.loader import ingest  # noqa: E402


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true", help="rebuild cached Parquet files")
    parser.add_argument("--no-report", action="store_true", help="skip per-step cleaning counts (faster on full data)")
    args = parser.parse_args(argv)
    cfg = setup(args)
    mode = cfg["data"]["mode"]
    print(f"Data preparation — mode={mode}, K={cfg['sessionize']['prefix_k']}")

    summary: dict[str, Any] = {"mode": mode}

    with stage("1/3 Load and clean events"):
        events_path = ingest(cfg, overwrite=args.overwrite, with_report=not args.no_report)
        print(f"   cleaned events -> {events_path}")
        report_path = events_path.parent / f"cleaning_report_{mode}.json"
        if report_path.exists() and not args.no_report:
            cleaning = json.loads(report_path.read_text())
            summary["cleaning"] = cleaning
            print(f"   rows {cleaning['rows_in']:,} -> {cleaning['rows_out']:,}; "
                  f"sessions {cleaning['sessions_in']:,} -> {cleaning['sessions_out']:,}")
            print_table([{"rule": k, "rows_removed": v} for k, v in cleaning["removed"].items()])

    with stage("2/3 Session prefixes and targets"):
        funnel = sessionize.run(cfg)
        summary["sessionize"] = {str(k): v for k, v in funnel.items()}
        print_table(
            [{"K": k, "sessions": r["sessions_total"], "too_short": r["too_short"],
              "bought_in_prefix": r["purchase_within_prefix"], "eligible": r["eligible"],
              "positive_rate": r["positive_rate"]} for k, r in funnel.items()]
        )

    with stage("3/3 Chronological split"):
        splits = split.run(cfg)
        cutoffs = json.loads(split.cutoffs_path(cfg).read_text())
        summary["split_cutoffs"] = cutoffs
        summary["splits"] = {str(k): v for k, v in splits.items()}
        print(f"   validation starts {cutoffs['val_start']}, test starts {cutoffs['test_start']}")
        rows = []
        for k, r in splits.items():
            for s in ("train", "val", "test"):
                if s in r:
                    rows.append({"K": k, "split": s, "sessions": r[s]["sessions"],
                                 "share": r[s]["fraction"], "positive_rate": r[s]["positive_rate"]})
            rows.append({"K": k, "split": "purged", "sessions": r["purged_boundary_sessions"]})
        print_table(rows, ["K", "split", "sessions", "share", "positive_rate"])

    out = resolve_path(cfg, "tables_dir") / f"data_summary_{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary for the report -> {out}")
    return summary


if __name__ == "__main__":
    main()
