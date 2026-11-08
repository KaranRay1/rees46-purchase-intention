"""Step 4 — RQ3 ablation: what does each feature group add?

Needs steps 1–2. Trains each model in ``features.ablation_models``
(LightGBM and MLP by default) on the cumulative ladder in
``features.ablation_ladder``::

    S1 = G1 activity
    S2 = G1 + G2 cart signals
    S3 = G1 + G2 + G3 behavioural ratios
    S4 = … + G4 category hierarchy
    S5 = … + G5 price & time            (= the full feature set of RQ1)

for every K and seed, with the same recipe as RQ1 (logged as experiment
``rq3_ablation``). Then, on the primary seed, each step is compared with the
step before it (paired bootstrap on test PR-AUC + McNemar):

* **confirmatory** (H3): S3 vs S2 — do the behavioural ratios help? — for every
  model and K; Holm-corrected as one family;
* **exploratory**: every other consecutive step; Holm-corrected separately;
* **variant** (``features.ablation_variants``): extra feature sets trained next to
  the ladder and compared with their reference step and with S1, e.g.
  ``ratios_instead_of_counts`` = G1+G3 vs S2 = G1+G2. With K fixed the ratios
  are functions of the G2 counts, so H3 ("ratios on top of counts") can add
  little by construction; the variant asks instead whether the ratios can
  *replace* the cart block. Holm-corrected as its own family.

Outputs (``results/tables`` and ``results/figures``)::

    rq3_ablation_<mode>.csv        mean ± std over seeds per (K, model, step)
    rq3_step_tests_<mode>.csv      consecutive-step differences with CI and p-values
    rq3_hypothesis_<mode>.json     H3 verdict and evidence
    rq3_ablation_<mode>.png        PR-AUC lift along the ladder, one panel per K

Examples::

    python scripts/04_run_ablation.py --mode synthetic
    python scripts/04_run_ablation.py --mode dev --ks 5 --models lightgbm
    python scripts/04_run_ablation.py --mode dev --analyse-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402
from _experiment import InputCache, latest_runs, primary_predictions, train_and_log  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.evaluation.statistics import holm, mcnemar, paired_bootstrap, seed_summary  # noqa: E402

EXPERIMENT = "rq3_ablation"
METRICS = ["pr_auc", "roc_auc", "f1"]
RATIO_GROUP = "G3_behavioural_ratios"


def step_names(cfg: Mapping[str, Any]) -> list[str]:
    """Short labels for the ladder steps, e.g. ``S3:G1+G2+G3``."""
    out = []
    for i, step in enumerate(cfg["features"]["ablation_ladder"], start=1):
        out.append(f"S{i}:" + "+".join(g.split("_")[0] for g in step))
    return out


def variants(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extra feature sets from ``features.ablation_variants`` with their run labels."""
    out = []
    for v in cfg["features"].get("ablation_variants", []) or []:
        short = "+".join(g.split("_")[0] for g in v["groups"])
        out.append({"name": v["name"], "label": f"V:{v['name']}", "short": short,
                    "groups": list(v["groups"]), "compare_to": int(v["compare_to"])})
    return out


def ratio_step_index(cfg: Mapping[str, Any]) -> int | None:
    """Index of the first ladder step that adds the behavioural-ratio group (H3)."""
    for i, step in enumerate(cfg["features"]["ablation_ladder"]):
        if RATIO_GROUP in step and i > 0 and RATIO_GROUP not in cfg["features"]["ablation_ladder"][i - 1]:
            return i
    return None


# =============================================================================
# Train
# =============================================================================
def train_all(cfg: Mapping[str, Any], models: Sequence[str], ks: Sequence[int], seeds: Sequence[int]) -> None:
    ladder = cfg["features"]["ablation_ladder"]
    names = step_names(cfg)
    sets = list(zip(ladder, names)) + [(v["groups"], v["label"]) for v in variants(cfg)]
    cache = InputCache(cfg)
    total = len(ks) * len(models) * len(sets) * len(seeds)
    i = 0
    for k in ks:
        for m in models:
            for groups, name in sets:
                for seed in seeds:
                    i += 1
                    row = train_and_log(cfg, cache, EXPERIMENT, m, k, seed, groups=groups, feature_set=name)
                    print(f"   [{i}/{total}] K={k:<2} {m:<10} {name:<18} seed={seed:<5} "
                          f"inputs={int(row['training_n_inputs']):<3} test PR-AUC={row['test_pr_auc']:.4f}",
                          flush=True)


# =============================================================================
# Analyse
# =============================================================================
def ablation_table(cfg: Mapping[str, Any], runs: pl.DataFrame) -> pl.DataFrame:
    """Mean ± std per (K, model, feature set). Variant rows carry ``is_variant`` and
    the ladder step they are compared with in ``step``; ``pr_auc_gain`` is the
    change from the previous ladder step (ladder rows only)."""
    order = {n: (i + 1, False) for i, n in enumerate(step_names(cfg))}
    order.update({v["label"]: (v["compare_to"], True) for v in variants(cfg)})
    rows = []
    for (k, m, fs), part in runs.group_by(["k", "model", "feature_set"]):
        if fs not in order:
            continue  # a step from an older ladder definition
        step, is_variant = order[fs]
        row: dict[str, Any] = {"k": k, "model": m, "step": step, "is_variant": is_variant, "feature_set": fs,
                               "n_inputs": int(part["training_n_inputs"][0]), "n_seeds": part.height,
                               "base_rate": float(part["test_base_rate"][0])}
        for met in METRICS:
            s = seed_summary(part[f"test_{met}"].to_list())
            row[f"{met}_mean"], row[f"{met}_std"] = s["mean"], s["std"]
        row["pr_auc_lift_mean"] = row["pr_auc_mean"] / row["base_rate"] if row["base_rate"] else float("nan")
        rows.append(row)
    table = pl.DataFrame(rows).sort(["k", "model", "is_variant", "step"])
    ladder = table.filter(~pl.col("is_variant")).with_columns(
        pr_auc_gain=pl.col("pr_auc_mean").diff().over(["k", "model"]),
    )
    var = table.filter(pl.col("is_variant")).with_columns(pr_auc_gain=pl.lit(None, dtype=pl.Float64))
    return pl.concat([ladder, var]).sort(["k", "model", "is_variant", "step"])


def step_tests(cfg: Mapping[str, Any], runs: pl.DataFrame, n_resamples: int) -> pl.DataFrame:
    """Paired comparisons on the primary seed (bootstrap PR-AUC + McNemar).

    * each ladder step vs the previous one — ``confirmatory`` for the step adding
      G3 (H3), ``exploratory`` otherwise;
    * each variant vs its ``compare_to`` step and vs step 1 — family ``variant``.
    Holm's correction is applied within each family.
    """
    names = step_names(cfg)
    h3 = ratio_step_index(cfg)
    conf = cfg["evaluation"]["bootstrap"]["confidence"]
    rows = []

    def compare(k, m, preds, cur, prev, step, added, family):
        a, b = preds[cur], preds[prev]
        if not a["user_session"].equals(b["user_session"]):
            raise RuntimeError(f"K={k} {m}: {cur} and {prev} scored different sessions")
        y = a["y_true"].to_numpy()
        boot = paired_bootstrap(y, a["score"].to_numpy(), b["score"].to_numpy(), "pr_auc",
                                n_resamples=n_resamples, confidence=conf, seed=0)
        mc = mcnemar(y, a["y_pred"].to_numpy(), b["y_pred"].to_numpy())
        rows.append({"k": k, "model": m, "step": step, "added": added, "from": prev, "to": cur, "family": family,
                     "pr_auc_before": boot["b"], "pr_auc_after": boot["a"], "pr_auc_gain": boot["diff"],
                     "gain_ci_low": boot["lower"], "gain_ci_high": boot["upper"],
                     "p_bootstrap": boot["p_value"], "p_mcnemar": mc["p_value"]})

    for k in sorted(runs["k"].unique().to_list()):
        for m in sorted(runs["model"].unique().to_list()):
            preds = {}
            for name in names + [v["label"] for v in variants(cfg)]:
                part = runs.filter((pl.col("k") == k) & (pl.col("model") == m) & (pl.col("feature_set") == name))
                if part.height:
                    preds[name] = primary_predictions(cfg, part)
            for i in range(1, len(names)):
                prev, cur = names[i - 1], names[i]
                if prev in preds and cur in preds:
                    compare(k, m, preds, cur, prev, i + 1, cur.split("+")[-1],
                            "confirmatory" if i == h3 else "exploratory")
            for v in variants(cfg):
                if v["label"] not in preds:
                    continue
                for ref in sorted({v["compare_to"], 1}):
                    if names[ref - 1] in preds:
                        compare(k, m, preds, v["label"], names[ref - 1], ref, v["short"], "variant")
    df = pl.DataFrame(rows)
    if df.height == 0:
        return df
    parts = []
    for fam in ("confirmatory", "exploratory", "variant"):
        part = df.filter(pl.col("family") == fam)
        if part.height:
            parts.append(part.with_columns(
                p_bootstrap_holm=pl.Series(holm(part["p_bootstrap"].to_list())),
                p_mcnemar_holm=pl.Series(holm(part["p_mcnemar"].to_list())),
            ))
    return pl.concat(parts).sort(["k", "model", "step"])


def hypothesis_h3(cfg: Mapping[str, Any], tests: pl.DataFrame) -> dict[str, Any]:
    alpha = cfg["evaluation"]["alpha"]
    conf = tests.filter(pl.col("family") == "confirmatory") if tests.height else tests
    ev = [{"k": r["k"], "model": r["model"], "gain": r["pr_auc_gain"],
           "ci": [r["gain_ci_low"], r["gain_ci_high"]], "p_holm": r["p_bootstrap_holm"],
           "significant_gain": r["pr_auc_gain"] > 0 and r["p_bootstrap_holm"] < alpha}
          for r in conf.iter_rows(named=True)]
    n_ok = sum(e["significant_gain"] for e in ev)
    verdict = "not tested" if not ev else "supported" if n_ok == len(ev) else \
        "partly supported" if n_ok else "not supported"
    out: dict[str, Any] = {"H3_behavioural_ratios_add_pr_auc": {
        "verdict": verdict, "significant_gains": f"{n_ok}/{len(ev)}", "alpha": alpha,
        "comparison": "ladder step adding G3 vs the step before it; paired bootstrap on test PR-AUC, "
                      "Holm-adjusted over the confirmatory family",
        "evidence": ev}}

    names = step_names(cfg)
    var_tests = tests.filter(pl.col("family") == "variant") if tests.height else tests
    for v in variants(cfg):
        ref = names[v["compare_to"] - 1]
        rows = var_tests.filter((pl.col("to") == v["label"]) & (pl.col("from") == ref)) if var_tests.height else var_tests
        evv = []
        for r in rows.iter_rows(named=True):
            sig = r["p_bootstrap_holm"] < alpha
            evv.append({"k": r["k"], "model": r["model"], "diff_vs_reference": r["pr_auc_gain"],
                        "ci": [r["gain_ci_low"], r["gain_ci_high"]], "p_holm": r["p_bootstrap_holm"],
                        "outcome": "better" if sig and r["pr_auc_gain"] > 0 else
                                   "worse" if sig and r["pr_auc_gain"] < 0 else "no significant difference"})
        n_worse = sum(e["outcome"] == "worse" for e in evv)
        out[f"exploratory_{v['name']}"] = {
            "question": f"Can {v['short']} replace {ref.split(':')[1]}? (variant vs ladder step {v['compare_to']})",
            "summary": "not tested" if not evv else
                       f"variant significantly worse in {n_worse}/{len(evv)} comparisons",
            "evidence": evv}
    return out


def plot_ladder(cfg: Mapping[str, Any], table: pl.DataFrame, path: Path) -> Path:
    ks = sorted(table["k"].unique().to_list())
    names = step_names(cfg)
    fig, axes = plt.subplots(1, len(ks), figsize=(4.5 * len(ks), 4), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], ks):
        for j, m in enumerate(sorted(table["model"].unique().to_list())):
            colour = f"C{j}"
            t = table.filter((pl.col("k") == k) & (pl.col("model") == m) & ~pl.col("is_variant")).sort("step")
            std = np.nan_to_num(t["pr_auc_std"].cast(pl.Float64).fill_null(0).to_numpy())
            lift_std = std / t["base_rate"].to_numpy()
            ax.errorbar(t["step"].to_list(), t["pr_auc_lift_mean"].to_list(), yerr=lift_std,
                        marker="o", capsize=3, color=colour, label=m)
            v = table.filter((pl.col("k") == k) & (pl.col("model") == m) & pl.col("is_variant"))
            for r in v.iter_rows(named=True):     # hollow marker at the step it replaces
                ax.plot([r["step"] + 0.12], [r["pr_auc_lift_mean"]], marker="D", markerfacecolor="none",
                        color=colour, linestyle="none",
                        label=f"{m}: {r['feature_set'][2:]}" if k == ks[0] else None)
        ax.set_xticks(range(1, len(names) + 1), [n.split(":")[1] for n in names], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"K={k}", fontsize=10)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    axes[0][0].set_ylabel("test PR-AUC lift (÷ base rate)")
    axes[0][0].legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg["logging"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    return path


def analyse(cfg: Mapping[str, Any], n_resamples: int) -> dict[str, Any]:
    mode = cfg["data"]["mode"]
    tables_dir = resolve_path(cfg, "tables_dir")
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs = latest_runs(cfg, EXPERIMENT, keys=("k", "model", "feature_set", "seed"))
    table = ablation_table(cfg, runs)
    tests = step_tests(cfg, runs, n_resamples)
    hyp = hypothesis_h3(cfg, tests)
    table.write_csv(tables_dir / f"rq3_ablation_{mode}.csv")
    tests.write_csv(tables_dir / f"rq3_step_tests_{mode}.csv")
    (tables_dir / f"rq3_hypothesis_{mode}.json").write_text(json.dumps(hyp, indent=2, default=float))
    fig = plot_ladder(cfg, table, resolve_path(cfg, "figures_dir") / f"rq3_ablation_{mode}.png")
    return {"table": table, "tests": tests, "hypothesis": hyp, "figure": fig}


# =============================================================================
# CLI
# =============================================================================
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="subset of features.ablation_models")
    parser.add_argument("--ks", nargs="+", type=int, help="subset of sessionize.prefix_k (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, help="subset of project.seeds (default: all)")
    parser.add_argument("--analyse-only", action="store_true", help="skip training; analyse logged runs")
    parser.add_argument("--n-resamples", type=int, help="bootstrap resamples (default: config)")
    args = parser.parse_args(argv)
    cfg = setup(args)

    models = args.models or cfg["features"]["ablation_models"]
    bad = [m for m in models if m not in cfg["features"]["ablation_models"]]
    if bad:
        parser.error(f"models not in features.ablation_models: {bad}")
    ks = args.ks or cfg["sessionize"]["prefix_k"]
    seeds = args.seeds or cfg["project"]["seeds"]
    n_resamples = args.n_resamples or cfg["evaluation"]["bootstrap"]["n_resamples"]
    if ratio_step_index(cfg) is None:
        print("   note: no ladder step adds G3 on its own, so H3 will not be tested")
    print(f"RQ3 ablation — mode={cfg['data']['mode']}, models={models}, K={ks}, seeds={seeds}")
    print("   ladder: " + ", ".join(step_names(cfg)))

    if not args.analyse_only:
        with stage("1/2 Train along the feature ladder"):
            train_all(cfg, models, ks, seeds)

    with stage("2/2 Analyse"):
        res = analyse(cfg, n_resamples)
        print_table(res["table"].select("k", "model", "feature_set", "n_inputs", "pr_auc_mean", "pr_auc_std",
                                        "pr_auc_gain", "pr_auc_lift_mean").to_dicts())
        conf = res["tests"].filter(pl.col("family") == "confirmatory") if res["tests"].height else res["tests"]
        if conf.height:
            print("   H3 tests (adding behavioural ratios):")
            print_table(conf.select("k", "model", "pr_auc_gain", "gain_ci_low", "gain_ci_high",
                                    "p_bootstrap_holm").to_dicts())
        h = res["hypothesis"]["H3_behavioural_ratios_add_pr_auc"]
        print(f"   H3: {h['verdict']} ({h['significant_gains']} significant gains)")
        for v in variants(cfg):
            vt = res["tests"].filter((pl.col("family") == "variant") & (pl.col("to") == v["label"]))
            if vt.height:
                print(f"   Variant {v['name']} ({v['short']}) vs reference steps:")
                print_table(vt.select("k", "model", "from", "pr_auc_before", "pr_auc_after", "pr_auc_gain",
                                      "gain_ci_low", "gain_ci_high", "p_bootstrap_holm").to_dicts())
                summary = res["hypothesis"]["exploratory_" + v["name"]]["summary"]
                print(f"   {summary}")
        print(f"   figure -> {res['figure']}")
    return res


if __name__ == "__main__":
    main()
