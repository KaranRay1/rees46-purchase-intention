"""Step 16: a Transformer encoder over the clicks, against the GRU (advanced extension).

Needs steps 1-3. Trains ``models.transformer`` on the same per-event sequences, splits and
recipe as the GRU (``torch_base.py``) for every K and project seed, logged as experiment
``rq6_transformer`` (it does not change the RQ1 tables). Then:

* test PR-AUC (mean and sd over seeds) next to the GRU and the best RQ1 model;
* paired bootstrap of the primary-seed Transformer against the GRU and the best model,
  Holm-corrected per K;
* where the Transformer looks: mean [CLS] attention per click position (test visits);
* single-visit latency (p95 over 300 calls, one thread) of the Transformer and the GRU.

Writes ``transformer``, ``transformer_tests`` and ``transformer_attention`` (CSV).

Examples::

    python scripts/16_transformer.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402
from _experiment import InputCache, train_and_log  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.evaluation.statistics import holm, paired_bootstrap  # noqa: E402
from src.models.base import BaseModel  # noqa: E402
from src.serving.predictor import available_runs  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402
from src.utils.results_logger import load_predictions  # noqa: E402

EXPERIMENT = "rq6_transformer"


def p95_latency(model, data, n: int = 300) -> float:
    torch.set_num_threads(1)
    cat, num = data.cat, data.num
    times = []
    for i in range(n + 20):
        j = i % len(cat)
        t0 = time.perf_counter()
        model.predict_proba((cat[j:j + 1], num[j:j + 1]))
        if i >= 20:
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 95))


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--ks", type=int, nargs="+", help="prefix lengths (default: sessionize.prefix_k)")
    p.add_argument("--seeds", type=int, nargs="+", help="seeds (default: project.seeds)")
    p.add_argument("--n-resamples", type=int, default=1000)
    args = p.parse_args(argv)
    cfg = setup(args)
    store = ResultStore(cfg)
    ks = args.ks or [int(k) for k in cfg["sessionize"]["prefix_k"]]
    seeds = args.seeds or [int(s) for s in cfg["project"]["seeds"]]
    rq1 = store.csv("rq1_model_comparison")
    base = available_runs(cfg)
    runs_dir = resolve_path(cfg, "runs_dir")
    cache = InputCache(cfg)
    rows, tests, attn = [], [], []
    for k in ks:
        with stage(f"K={k}: Transformer x {len(seeds)} seeds"):
            trained = [train_and_log(cfg, cache, EXPERIMENT, "transformer", k, s) for s in seeds]
            vals = [float(r["test_pr_auc"]) for r in trained]
            prim = trained[0]
            data, _, _, _ = cache.get(k, "transformer")
            tmodel = BaseModel.load(runs_dir / prim["run_id"] / "model.pkl")
            attn += [{"k": k, "position": i + 1, "attention": float(a)} for i, a in enumerate(tmodel.attention(data["test"]))]
            t_pred = load_predictions(runs_dir / prim["run_id"], "test").sort("user_session")
            row = {"k": k, "model": "transformer", "n_seeds": len(vals), "pr_auc_mean": float(np.mean(vals)),
                   "pr_auc_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                   "n_parameters": int(tmodel.n_parameters()), "train_time_sec": float(np.mean(
                       [float(r.get("training_train_time_sec") or 0) for r in trained])),
                   "latency_p95_ms": p95_latency(tmodel, data["test"])}
            rows.append(row)
            others = rq1.filter(pl.col("k") == k) if rq1 is not None else pl.DataFrame()
            best = others.filter(pl.col("model") != "rule_based").sort("pr_auc_mean", descending=True)["model"][0] \
                if others.height else None
            raw = []
            for rival in dict.fromkeys(["gru", best]):
                if rival is None:
                    continue
                rr = base.filter((pl.col("k") == k) & (pl.col("model") == rival))
                if not rr.height:
                    continue
                r_pred = load_predictions(runs_dir / rr["run_id"][0], "test").sort("user_session")
                j = t_pred.join(r_pred.select("user_session", pl.col("score").alias("rival")), on="user_session")
                t = paired_bootstrap(j["y_true"].to_numpy(), j["score"].to_numpy(), j["rival"].to_numpy(), "pr_auc",
                                     n_resamples=args.n_resamples, seed=seeds[0])
                raw.append({"k": k, "model": "transformer", "vs": rival, "a": t["a"], "b": t["b"], "diff": t["diff"],
                            "lower": t["lower"], "upper": t["upper"], "p_value": t["p_value"]})
                if rival == "gru":
                    gm = BaseModel.load(runs_dir / rr["run_id"][0] / "model.pkl")
                    g = others.filter(pl.col("model") == "gru").row(0, named=True)
                    rows.append({"k": k, "model": "gru", "n_seeds": g["n_seeds"], "pr_auc_mean": g["pr_auc_mean"],
                                 "pr_auc_std": g["pr_auc_std"], "n_parameters": int(gm.n_parameters()),
                                 "train_time_sec": g["train_time_sec_mean"], "latency_p95_ms": p95_latency(gm, data["test"])})
            for r, ph in zip(raw, holm([r["p_value"] for r in raw])):
                r["p_holm"], r["significant"] = ph, ph < float(cfg["evaluation"]["alpha"])
            tests += raw
    store.tables_dir.mkdir(parents=True, exist_ok=True)
    res = pl.DataFrame(rows)
    res.write_csv(store.path("transformer", "csv"))
    pl.DataFrame(tests).write_csv(store.path("transformer_tests", "csv"))
    pl.DataFrame(attn).write_csv(store.path("transformer_attention", "csv"))
    print("\nTransformer vs GRU:")
    print_table(res.with_columns(pl.col(pl.Float64).round(4)).to_dicts())
    print_table(pl.DataFrame(tests).select("k", "vs", pl.col("diff").round(4), pl.col("p_holm").round(3), "significant").to_dicts())


if __name__ == "__main__":
    main()
