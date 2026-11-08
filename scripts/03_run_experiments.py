"""Step 3 — main experiment for RQ1 (model families) and RQ2 (how early, K).

Needs steps 1–2. Two phases:

**Train** — every enabled model × every K × every seed:
  prepare inputs (tabular profile per model, or GRU sequences) → fit (time and
  memory recorded) → tune the threshold on validation F1 → score val and test →
  log everything with ``src.utils.results_logger`` under experiment
  ``rq1_models`` (params, metrics, per-session predictions, model,
  preprocessor/encoder, training history). The rule baseline is deterministic,
  so it runs once (first seed).

**Analyse** — from the logged runs (so it can be re-run with ``--analyse-only``):
  * ``rq1_model_comparison_<mode>.csv``  mean ± std over seeds of every test metric,
    plus a 95% bootstrap CI of PR-AUC on the primary seed's predictions;
  * ``rq1_pairwise_tests_<mode>.csv``    every model pair at each K: paired-bootstrap
    PR-AUC difference (CI, p) and McNemar on labels. Holm's correction is applied
    within two families: **confirmatory** = the comparisons the hypotheses make
    (each tree / deep-tabular model vs logistic regression at every K, and the
    GRU vs the best aggregated model at the largest K) and **exploratory** =
    all other pairs. Verdicts use only the confirmatory family;
  * ``rq2_by_k_<mode>.csv``             PR-AUC, PR-AUC lift (= PR-AUC ÷ base rate) and
    ROC-AUC per model and K;
  * figures ``rq1_pr_curves_k{K}_<mode>.png`` and ``rq2_metrics_vs_k_<mode>.png``;
  * ``rq1_rq2_hypotheses_<mode>.json``   H1a / H1b / H1c / H2 verdicts with the numbers behind them.

Statistical tests use the **primary seed** (first in ``project.seeds``) so
every comparison is between two fixed models on the same test sessions;
variation across seeds is reported separately as mean ± std.

Why lift and ROC-AUC for RQ2: each K has a different set of eligible
sessions with a different purchase rate, and PR-AUC scales with the base
rate. Comparing raw PR-AUC across K would partly measure the base rate, so
RQ2 is read on lift and ROC-AUC (base-rate free) next to raw PR-AUC.

Examples::

    python scripts/03_run_experiments.py --mode synthetic
    python scripts/03_run_experiments.py --mode dev --models logistic_regression lightgbm --ks 5
    python scripts/03_run_experiments.py --mode dev --analyse-only
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from _experiment import InputCache, latest_runs, primary_predictions, primary_run, train_and_log  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.evaluation.statistics import bootstrap_ci, holm, mcnemar, paired_bootstrap, seed_summary  # noqa: E402

EXPERIMENT = "rq1_models"
METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall", "macro_f1"]
TREE = ("random_forest", "lightgbm")
DEEP_TABULAR = ("mlp", "tabnet")


# =============================================================================
# Train
# =============================================================================
def train_all(cfg: Mapping[str, Any], models: Sequence[str], ks: Sequence[int], seeds: Sequence[int]) -> list[dict]:
    cache = InputCache(cfg)
    done = []
    total = sum(1 if m == "rule_based" else len(seeds) for m in models) * len(ks)
    i = 0
    for k in ks:
        for m in models:
            for seed in (seeds[:1] if m == "rule_based" else seeds):   # the rule is deterministic
                i += 1
                row = train_and_log(cfg, cache, EXPERIMENT, m, k, seed)
                print(f"   [{i}/{total}] K={k:<2} {m:<20} seed={seed:<5} "
                      f"test PR-AUC={row['test_pr_auc']:.4f}  F1={row['test_f1']:.4f}  "
                      f"train={row['training_train_time_sec']:.1f}s", flush=True)
                done.append({"run_id": row["run_id"], "k": k, "model": m, "seed": seed})
    return done


# =============================================================================
# Analyse
# =============================================================================
def _model_order(cfg: Mapping[str, Any], present: Sequence[str]) -> list[str]:
    return [m for m in cfg["models"]["enabled"] if m in present]


def comparison_table(cfg: Mapping[str, Any], runs: pl.DataFrame, n_resamples: int) -> pl.DataFrame:
    """Mean ± std over seeds per (K, model) + PR-AUC bootstrap CI on the primary seed."""
    conf = cfg["evaluation"]["bootstrap"]["confidence"]
    rows = []
    for k in sorted(runs["k"].unique().to_list()):
        for m in _model_order(cfg, runs["model"].unique().to_list()):
            part = runs.filter((pl.col("k") == k) & (pl.col("model") == m))
            if part.height == 0:
                continue
            row: dict[str, Any] = {"k": k, "model": m, "n_seeds": part.height,
                                   "base_rate": float(part["test_base_rate"][0])}
            for met in METRICS:
                s = seed_summary(part[f"test_{met}"].to_list())
                row[f"{met}_mean"], row[f"{met}_std"] = s["mean"], s["std"]
            row["train_time_sec_mean"] = seed_summary(part["training_train_time_sec"].to_list())["mean"]
            preds = primary_predictions(cfg, part)
            ci = bootstrap_ci(preds["y_true"].to_numpy(), preds["score"].to_numpy(), "pr_auc",
                              n_resamples=n_resamples, confidence=conf, seed=0)
            row["pr_auc_primary"], row["pr_auc_ci_low"], row["pr_auc_ci_high"] = ci["estimate"], ci["lower"], ci["upper"]
            row["pr_auc_lift_mean"] = row["pr_auc_mean"] / row["base_rate"] if row["base_rate"] else float("nan")
            rows.append(row)
    return pl.DataFrame(rows)


def best_aggregated(table: pl.DataFrame, k: int) -> str | None:
    """Best non-sequence, non-rule model by mean test PR-AUC at K (the GRU's H1c opponent)."""
    agg = table.filter((pl.col("k") == k) & ~pl.col("model").is_in(["gru", "rule_based"]))
    return agg.sort("pr_auc_mean", descending=True)["model"][0] if agg.height else None


def confirmatory_pairs(table: pl.DataFrame) -> set[tuple[int, frozenset]]:
    """The comparisons the hypotheses actually make (one Holm family).

    H1a / H1b: each tree and deep-tabular model vs logistic regression, at every K.
    H1c:       GRU vs the best aggregated-feature model, at the largest K.
    """
    present = set(table["model"].to_list())
    ks = sorted(table["k"].unique().to_list())
    pairs = {(k, frozenset((m, "logistic_regression")))
             for k in ks for m in (*TREE, *DEEP_TABULAR) if m in present and "logistic_regression" in present}
    best = best_aggregated(table, max(ks)) if ks else None
    if "gru" in present and best:
        pairs.add((max(ks), frozenset(("gru", best))))
    return pairs


def pairwise_tests(
    cfg: Mapping[str, Any], runs: pl.DataFrame, n_resamples: int, confirmatory: set | None = None
) -> pl.DataFrame:
    """All model pairs per K on the primary seed: paired bootstrap PR-AUC + McNemar.

    Holm's correction is applied **within each family**: ``confirmatory`` (the
    hypothesis comparisons, see ``confirmatory_pairs``) and ``exploratory``
    (every other pair). Hypothesis verdicts use only the confirmatory family.
    """
    confirmatory = confirmatory or set()
    conf = cfg["evaluation"]["bootstrap"]["confidence"]
    rows = []
    for k in sorted(runs["k"].unique().to_list()):
        preds = {}
        for m in _model_order(cfg, runs["model"].unique().to_list()):
            part = runs.filter((pl.col("k") == k) & (pl.col("model") == m))
            if part.height:
                preds[m] = primary_predictions(cfg, part)
        for a, b in combinations(list(preds), 2):
            pa, pb = preds[a], preds[b]
            if not pa["user_session"].equals(pb["user_session"]):
                raise RuntimeError(f"K={k}: {a} and {b} were tested on different sessions")
            y = pa["y_true"].to_numpy()
            boot = paired_bootstrap(y, pa["score"].to_numpy(), pb["score"].to_numpy(), "pr_auc",
                                    n_resamples=n_resamples, confidence=conf, seed=0)
            mc = mcnemar(y, pa["y_pred"].to_numpy(), pb["y_pred"].to_numpy())
            family = "confirmatory" if (k, frozenset((a, b))) in confirmatory else "exploratory"
            rows.append({"k": k, "model_a": a, "model_b": b, "family": family,
                         "pr_auc_a": boot["a"], "pr_auc_b": boot["b"],
                         "pr_auc_diff": boot["diff"], "diff_ci_low": boot["lower"], "diff_ci_high": boot["upper"],
                         "p_bootstrap": boot["p_value"], "mcnemar_b": mc["b"], "mcnemar_c": mc["c"],
                         "p_mcnemar": mc["p_value"]})
    df = pl.DataFrame(rows)
    if df.height == 0:
        return df
    parts = []
    for fam in ("confirmatory", "exploratory"):
        part = df.filter(pl.col("family") == fam)
        if part.height:
            parts.append(part.with_columns(
                p_bootstrap_holm=pl.Series(holm(part["p_bootstrap"].to_list())),
                p_mcnemar_holm=pl.Series(holm(part["p_mcnemar"].to_list())),
            ))
    return pl.concat(parts).sort(["k", "model_a", "model_b"])


def by_k_table(table: pl.DataFrame) -> pl.DataFrame:
    """RQ2: PR-AUC, lift and ROC-AUC per model across K."""
    return table.select("model", "k", "base_rate", "pr_auc_mean", "pr_auc_lift_mean", "roc_auc_mean", "f1_mean") \
        .sort(["model", "k"])


def _pair(tests: pl.DataFrame, k: int, a: str, b: str) -> dict | None:
    """Test row for a vs b at K, oriented so that diff = a − b."""
    hit = tests.filter((pl.col("k") == k) & (pl.col("model_a") == a) & (pl.col("model_b") == b))
    if hit.height:
        return hit.row(0, named=True)
    hit = tests.filter((pl.col("k") == k) & (pl.col("model_a") == b) & (pl.col("model_b") == a))
    if not hit.height:
        return None
    r = hit.row(0, named=True)
    return {**r, "model_a": a, "model_b": b, "pr_auc_diff": -r["pr_auc_diff"],
            "diff_ci_low": -r["diff_ci_high"], "diff_ci_high": -r["diff_ci_low"]}


def hypotheses(cfg: Mapping[str, Any], table: pl.DataFrame, tests: pl.DataFrame) -> dict[str, Any]:
    """Verdicts for H1a, H1b, H1c and H2 with the evidence behind each."""
    alpha = cfg["evaluation"]["alpha"]
    ks = sorted(table["k"].unique().to_list())
    out: dict[str, Any] = {
        "alpha": alpha,
        "test": "paired bootstrap on test PR-AUC, Holm-adjusted within the confirmatory family",
        "confirmatory_tests": int((tests["family"] == "confirmatory").sum()) if tests.height else 0,
    }

    def versus_lr(models: Sequence[str]) -> dict[str, Any]:
        ev = []
        for k in ks:
            for m in models:
                r = _pair(tests, k, m, "logistic_regression")
                if r:
                    ev.append({"k": k, "model": m, "diff": r["pr_auc_diff"], "p_holm": r["p_bootstrap_holm"],
                               "significant_better": r["pr_auc_diff"] > 0 and r["p_bootstrap_holm"] < alpha})
        n_ok = sum(e["significant_better"] for e in ev)
        verdict = "not tested" if not ev else "supported" if n_ok == len(ev) else \
            "partly supported" if n_ok else "not supported"
        return {"verdict": verdict, "significant_wins": f"{n_ok}/{len(ev)}", "evidence": ev}

    out["H1a_trees_beat_logistic_regression"] = versus_lr(TREE)
    out["H1b_deep_tabular_beat_logistic_regression"] = versus_lr(DEEP_TABULAR)

    k_max = max(ks)
    best = best_aggregated(table, k_max)
    if "gru" in table["model"].to_list() and best:
        r = _pair(tests, k_max, "gru", best)
        if r:
            worse = r["pr_auc_diff"] < 0 and r["p_bootstrap_holm"] < alpha
            out["H1c_gru_matches_best_aggregated_at_max_k"] = {
                "verdict": "not supported" if worse else "supported",
                "k": k_max, "best_aggregated_model": best, "diff": r["pr_auc_diff"],
                "ci": [r["diff_ci_low"], r["diff_ci_high"]], "p_holm": r["p_bootstrap_holm"]}

    if len(ks) >= 3:
        h2 = []
        for m in table["model"].unique().to_list():
            lift = table.filter(pl.col("model") == m).sort("k")["pr_auc_lift_mean"].to_list()
            if len(lift) == len(ks):
                rising = all(b >= a for a, b in zip(lift, lift[1:]))
                gains = [b - a for a, b in zip(lift, lift[1:])]
                h2.append({"model": m, "lift_by_k": dict(zip(map(str, ks), lift)), "rising": rising,
                           "diminishing": rising and gains[-1] < gains[0]})
        n_rise = sum(e["rising"] for e in h2)
        n_dim = sum(e["diminishing"] for e in h2)
        out["H2_performance_rises_with_k_diminishing"] = {
            "verdict": "supported" if n_dim == len(h2) else "partly supported" if n_rise else "not supported",
            "measure": "PR-AUC lift over the base rate (base rate differs by K)",
            "models_rising": f"{n_rise}/{len(h2)}", "models_diminishing": f"{n_dim}/{len(h2)}", "evidence": h2}
    return out


def plot_pr_curves(cfg: Mapping[str, Any], runs: pl.DataFrame, k: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    base = None
    for m in _model_order(cfg, runs["model"].unique().to_list()):
        part = runs.filter((pl.col("k") == k) & (pl.col("model") == m))
        if not part.height:
            continue
        prim = primary_run(cfg, part)
        p = primary_predictions(cfg, part)
        y, s = p["y_true"].to_numpy(), p["score"].to_numpy()
        prec, rec, _ = precision_recall_curve(y, s)
        ax.step(rec, prec, where="post", label=f"{m} (AP {prim['test_pr_auc'][0]:.3f})", linewidth=1.2)
        base = y.mean()
    if base is not None:
        ax.axhline(base, color="grey", linestyle="--", linewidth=1, label=f"base rate {base:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Precision–recall on test, K={k}")
    ax.legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg["logging"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    return path


def plot_vs_k(cfg: Mapping[str, Any], table: pl.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for m in _model_order(cfg, table["model"].unique().to_list()):
        t = table.filter(pl.col("model") == m).sort("k")
        ks = t["k"].to_list()
        yerr = np.nan_to_num(t["pr_auc_std"].cast(pl.Float64).fill_null(0).to_numpy())   # 1 seed -> no bar
        axes[0].errorbar(ks, t["pr_auc_mean"].to_list(), yerr=yerr,
                         marker="o", capsize=3, label=m)
        axes[1].plot(ks, t["pr_auc_lift_mean"].to_list(), marker="o", label=m)
        axes[2].plot(ks, t["roc_auc_mean"].to_list(), marker="o", label=m)
    base = table.group_by("k").agg(pl.col("base_rate").first()).sort("k")
    axes[0].plot(base["k"].to_list(), base["base_rate"].to_list(), "k--", linewidth=1, label="base rate")
    for ax, title in zip(axes, ["test PR-AUC (mean ± std over seeds)", "PR-AUC lift (÷ base rate)", "test ROC-AUC"]):
        ax.set_xlabel("prefix length K")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(sorted(table["k"].unique().to_list()))
    axes[0].legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg["logging"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    return path


def analyse(cfg: Mapping[str, Any], n_resamples: int) -> dict[str, Any]:
    mode = cfg["data"]["mode"]
    tables_dir = resolve_path(cfg, "tables_dir")
    figures_dir = resolve_path(cfg, "figures_dir")
    tables_dir.mkdir(parents=True, exist_ok=True)

    runs = latest_runs(cfg, EXPERIMENT)
    table = comparison_table(cfg, runs, n_resamples)
    tests = pairwise_tests(cfg, runs, n_resamples, confirmatory_pairs(table))
    by_k = by_k_table(table)
    table.write_csv(tables_dir / f"rq1_model_comparison_{mode}.csv")
    tests.write_csv(tables_dir / f"rq1_pairwise_tests_{mode}.csv")
    by_k.write_csv(tables_dir / f"rq2_by_k_{mode}.csv")

    figs = [plot_pr_curves(cfg, runs, k, figures_dir / f"rq1_pr_curves_k{k}_{mode}.png")
            for k in sorted(table["k"].unique().to_list())]
    if table["k"].n_unique() > 1:
        figs.append(plot_vs_k(cfg, table, figures_dir / f"rq2_metrics_vs_k_{mode}.png"))

    hyp = hypotheses(cfg, table, tests)
    (tables_dir / f"rq1_rq2_hypotheses_{mode}.json").write_text(json.dumps(hyp, indent=2, default=float))
    return {"table": table, "tests": tests, "by_k": by_k, "hypotheses": hyp, "figures": figs}


# =============================================================================
# CLI
# =============================================================================
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="subset of models.enabled (default: all)")
    parser.add_argument("--ks", nargs="+", type=int, help="subset of sessionize.prefix_k (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, help="subset of project.seeds (default: all)")
    parser.add_argument("--analyse-only", action="store_true", help="skip training; analyse logged runs")
    parser.add_argument("--n-resamples", type=int, help="bootstrap resamples (default: config)")
    args = parser.parse_args(argv)
    cfg = setup(args)

    models = args.models or cfg["models"]["enabled"]
    unknown = [m for m in models if m not in cfg["models"]["enabled"]]
    if unknown:
        parser.error(f"models not in models.enabled: {unknown}")
    ks = args.ks or cfg["sessionize"]["prefix_k"]
    seeds = args.seeds or cfg["project"]["seeds"]
    n_resamples = args.n_resamples or cfg["evaluation"]["bootstrap"]["n_resamples"]
    print(f"RQ1/RQ2 experiment — mode={cfg['data']['mode']}, models={models}, K={ks}, seeds={seeds}")

    if not args.analyse_only:
        with stage("1/2 Train and evaluate"):
            train_all(cfg, models, ks, seeds)

    with stage("2/2 Analyse"):
        res = analyse(cfg, n_resamples)
        t = res["table"]
        print_table(t.select("k", "model", "pr_auc_mean", "pr_auc_std", "pr_auc_ci_low", "pr_auc_ci_high",
                             "roc_auc_mean", "f1_mean", "train_time_sec_mean").to_dicts())
        if res["tests"].height:
            for fam in ("confirmatory", "exploratory"):
                part = res["tests"].filter(pl.col("family") == fam)
                sig = part.filter(pl.col("p_bootstrap_holm") < cfg["evaluation"]["alpha"])
                print(f"   {fam}: {sig.height}/{part.height} PR-AUC differences significant after Holm")
        for name, h in res["hypotheses"].items():
            if isinstance(h, dict) and "verdict" in h:
                print(f"   {name}: {h['verdict']}")
        print(f"   tables -> {resolve_path(cfg, 'tables_dir')}")
        print(f"   figures -> {resolve_path(cfg, 'figures_dir')}")
    return res


if __name__ == "__main__":
    main()
