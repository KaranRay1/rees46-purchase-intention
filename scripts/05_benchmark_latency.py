"""Step 5 — RQ4: is real-time purchase prediction feasible, and at what cost?

Needs step 3 (``03_run_experiments.py``). Nothing is retrained: for every
(K, model) the **primary-seed model saved by RQ1** is loaded and benchmarked
on its own test inputs (rebuilt with the same preprocessing; session order is
checked against the run's saved predictions).

Measured with ``src.evaluation.efficiency`` on **one CPU thread**:
latency p50 / p95 / p99 for batches of 1 (the real-time case), 256 and 4096,
per-session latency, throughput, inference peak memory, model size and
parameter count. Training time and memory come from the RQ1 runs (mean over
seeds). Each benchmark is logged as experiment ``rq4_efficiency``.

Outputs::

    results/tables/rq4_efficiency_<mode>.csv      one row per (K, model)
    results/tables/rq4_hardware_<mode>.json       CPU, RAM, library versions
    results/tables/rq4_hypothesis_<mode>.json     real-time budget check + H4
    results/figures/rq4_pareto_<mode>.png         PR-AUC vs single-session p95 latency

H4 ("tree models dominate the accuracy–latency Pareto front on CPU") is
judged on the non-baseline models (RF, LightGBM, MLP, TabNet, GRU): it holds
at a K when every Pareto-optimal one of them is a tree model. Baselines are
still shown, because a near-free rule or linear model on the front is itself
a finding. The real-time budget is ``efficiency.realtime_budget_ms`` (p95,
batch size 1) — an assumption to state in the report.

Examples::

    python scripts/05_benchmark_latency.py --mode synthetic
    python scripts/05_benchmark_latency.py --mode dev --ks 5 --set efficiency.n_repeats=500
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402
from _experiment import InputCache, latest_runs, primary_predictions, primary_run  # noqa: E402

import polars as pl  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.evaluation.efficiency import benchmark_model, hardware_info  # noqa: E402
from src.evaluation.statistics import seed_summary  # noqa: E402
from src.models.base import BaseModel  # noqa: E402
from src.utils.results_logger import start_run  # noqa: E402

SOURCE = "rq1_models"
EXPERIMENT = "rq4_efficiency"
BASELINES = ("rule_based", "logistic_regression")
TREES = ("random_forest", "lightgbm")


# =============================================================================
# Benchmark
# =============================================================================
def benchmark_one(cfg: Mapping[str, Any], cache: InputCache, runs: pl.DataFrame, k: int, model_name: str) -> dict:
    part = runs.filter((pl.col("k") == k) & (pl.col("model") == model_name))
    prim = primary_run(cfg, part)
    run_dir = resolve_path(cfg, "runs_dir") / prim["run_id"][0]
    model_path = run_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} missing — RQ1 must run with logging.save_models=true.")
    model = BaseModel.load(model_path)

    data, _, _, _ = cache.get(k, model_name)
    test = data["test"]
    expected = primary_predictions(cfg, part)["user_session"].to_list()
    if sorted(test.session_ids) != expected:
        raise RuntimeError(f"K={k} {model_name}: rebuilt test inputs do not match the RQ1 run's sessions")

    eff = benchmark_model(model, test, cfg)
    row: dict[str, Any] = {
        "k": k, "model": model_name, "source_run_id": prim["run_id"][0],
        "test_pr_auc_mean": seed_summary(part["test_pr_auc"].to_list())["mean"],
        "test_pr_auc_primary": float(prim["test_pr_auc"][0]),
        "train_time_sec_mean": seed_summary(part["training_train_time_sec"].to_list())["mean"],
        "train_peak_memory_mb_mean": seed_summary(part["training_train_peak_memory_mb"].to_list())["mean"],
        "n_test": len(test.y),
        **{key: v for key, v in eff.items() if key not in ("model", "train_time_sec")},
    }

    with start_run(cfg, EXPERIMENT, model_name, k=k, seed=int(prim["seed"][0]), feature_set="all") as run:
        run.log_params({"source_run_id": row["source_run_id"], "n_threads": eff["n_threads"],
                        "batch_sizes": cfg["efficiency"]["batch_sizes"],
                        "n_repeats": cfg["efficiency"]["n_repeats"], "n_warmup": cfg["efficiency"]["n_warmup"]})
        run.log_metrics({"efficiency": {k_: v for k_, v in row.items()
                                        if k_ not in ("k", "model", "source_run_id")}})
    return row


def benchmark_all(cfg: Mapping[str, Any], models: Sequence[str], ks: Sequence[int]) -> pl.DataFrame:
    runs = latest_runs(cfg, SOURCE)
    cache = InputCache(cfg)
    rows = []
    for k in ks:
        for m in models:
            if runs.filter((pl.col("k") == k) & (pl.col("model") == m)).height == 0:
                print(f"   skip K={k} {m}: no RQ1 run")
                continue
            r = benchmark_one(cfg, cache, runs, k, m)
            print(f"   K={k:<2} {m:<20} p50={r['latency_b1_p50_ms']:.3f} ms  p95={r['latency_b1_p95_ms']:.3f} ms  "
                  f"size={r['model_size_mb']:.2f} MB", flush=True)
            rows.append(r)
    return pl.DataFrame(rows)


# =============================================================================
# Analyse
# =============================================================================
def pareto_front(points: pl.DataFrame, x: str = "latency_b1_p95_ms", y: str = "test_pr_auc_mean") -> list[str]:
    """Models not dominated by any other (lower-or-equal latency AND higher-or-equal PR-AUC, one strictly)."""
    front = []
    rows = points.select("model", x, y).to_dicts()
    for a in rows:
        dominated = any(
            b["model"] != a["model"] and b[x] <= a[x] and b[y] >= a[y] and (b[x] < a[x] or b[y] > a[y])
            for b in rows
        )
        if not dominated:
            front.append(a["model"])
    return front


def assess(cfg: Mapping[str, Any], table: pl.DataFrame) -> dict[str, Any]:
    budget = float(cfg["efficiency"]["realtime_budget_ms"])
    out: dict[str, Any] = {"realtime_budget_ms_p95_single_session": budget, "n_threads": cfg["efficiency"]["n_threads"],
                           "per_k": {}}
    h4_ok = []
    for k in sorted(table["k"].unique().to_list()):
        t = table.filter(pl.col("k") == k)
        within = t.filter(pl.col("latency_b1_p95_ms") <= budget)["model"].to_list()
        front_all = pareto_front(t)
        contenders = t.filter(~pl.col("model").is_in(BASELINES))
        front_models = pareto_front(contenders) if contenders.height else []
        trees_only = bool(front_models) and all(m in TREES for m in front_models)
        h4_ok.append(trees_only)
        out["per_k"][str(k)] = {
            "within_budget": within,
            "over_budget": [m for m in t["model"].to_list() if m not in within],
            "pareto_front_all_models": front_all,
            "pareto_front_non_baseline": front_models,
            "trees_dominate": trees_only,
        }
    n = sum(h4_ok)
    out["H4_trees_dominate_accuracy_latency_front"] = {
        "verdict": "not tested" if not h4_ok else "supported" if n == len(h4_ok) else
                   "partly supported" if n else "not supported",
        "k_where_trees_dominate": f"{n}/{len(h4_ok)}",
        "rule": "at each K, every Pareto-optimal non-baseline model (PR-AUC vs p95 single-session latency) is a tree model",
    }
    return out


def plot_pareto(cfg: Mapping[str, Any], table: pl.DataFrame, assessment: Mapping[str, Any], path: Path) -> Path:
    ks = sorted(table["k"].unique().to_list())
    budget = float(cfg["efficiency"]["realtime_budget_ms"])
    models = [m for m in cfg["models"]["enabled"] if m in table["model"].to_list()]
    fig, axes = plt.subplots(1, len(ks), figsize=(4.8 * len(ks), 4.2), squeeze=False)
    for ax, k in zip(axes[0], ks):
        t = table.filter(pl.col("k") == k)
        front = set(assessment["per_k"][str(k)]["pareto_front_all_models"])
        for j, m in enumerate(models):
            r = t.filter(pl.col("model") == m)
            if not r.height:
                continue
            x, y = r["latency_b1_p95_ms"][0], r["test_pr_auc_mean"][0]
            ax.scatter([x], [y], s=60, color=f"C{j}", edgecolor="black" if m in front else "none",
                       linewidth=1.5, label=m, zorder=3)
        fr = t.filter(pl.col("model").is_in(list(front))).sort("latency_b1_p95_ms")
        ax.step(fr["latency_b1_p95_ms"].to_list(), fr["test_pr_auc_mean"].to_list(), where="post",
                color="grey", linewidth=1, zorder=1)
        ax.axvline(budget, color="red", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("p95 latency, 1 session, 1 thread (ms, log)")
        ax.set_title(f"K={k}", fontsize=10)
    axes[0][0].set_ylabel("test PR-AUC (mean over seeds)")
    axes[0][0].legend(fontsize=7, title="outlined = Pareto-optimal", title_fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg["logging"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# CLI
# =============================================================================
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="subset of models.enabled (default: all)")
    parser.add_argument("--ks", nargs="+", type=int, help="subset of sessionize.prefix_k (default: all)")
    args = parser.parse_args(argv)
    cfg = setup(args)
    models = args.models or cfg["models"]["enabled"]
    ks = args.ks or cfg["sessionize"]["prefix_k"]
    mode = cfg["data"]["mode"]
    tables_dir = resolve_path(cfg, "tables_dir")
    tables_dir.mkdir(parents=True, exist_ok=True)

    hw = hardware_info()
    (tables_dir / f"rq4_hardware_{mode}.json").write_text(json.dumps(hw, indent=2))
    print(f"RQ4 efficiency — mode={mode}, models={models}, K={ks}, threads={cfg['efficiency']['n_threads']}")
    print(f"   machine: {hw['processor']}, {hw['cpu_logical']} logical CPUs, {hw['ram_gb']} GB RAM, "
          f"Python {hw['python']}")

    with stage("1/2 Benchmark saved RQ1 models"):
        table = benchmark_all(cfg, models, ks)
        table.write_csv(tables_dir / f"rq4_efficiency_{mode}.csv")

    with stage("2/2 Real-time budget and Pareto front"):
        assessment = assess(cfg, table)
        (tables_dir / f"rq4_hypothesis_{mode}.json").write_text(json.dumps(assessment, indent=2))
        fig = plot_pareto(cfg, table, assessment, resolve_path(cfg, "figures_dir") / f"rq4_pareto_{mode}.png")
        k0 = cfg["sessionize"]["default_k"] if cfg["sessionize"]["default_k"] in ks else ks[0]
        b_max = int(max(cfg["efficiency"]["batch_sizes"]))   # batch sizes come from config
        print(f"   K={k0}:")
        print_table(table.filter(pl.col("k") == k0).select(
            "model", "test_pr_auc_mean", "latency_b1_p50_ms", "latency_b1_p95_ms",
            f"latency_b{b_max}_per_session_ms", f"throughput_b{b_max}_per_sec", "model_size_mb",
            "train_time_sec_mean").to_dicts())
        for k, info in assessment["per_k"].items():
            print(f"   K={k}: within {assessment['realtime_budget_ms_p95_single_session']:g} ms budget: "
                  f"{len(info['within_budget'])}/{len(info['within_budget']) + len(info['over_budget'])}; "
                  f"Pareto front: {', '.join(info['pareto_front_all_models'])}")
        print(f"   H4: {assessment['H4_trees_dominate_accuracy_latency_front']['verdict']}")
        print(f"   figure -> {fig}")
    return {"table": table, "assessment": assessment, "hardware": hw, "figure": fig}


if __name__ == "__main__":
    main()
